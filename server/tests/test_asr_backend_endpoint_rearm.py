"""Regression: backend-endpoint final in ``_asr_stream_backend`` must re-arm.

Root cause (fix under test, server/main.py backend-endpoint branch)
===================================================================
When ``stream.get_partial()`` reports ``is_endpoint=True``, the handler sends
a final — but previously kept polling the SAME stream instance. Backends with
*sticky* endpoint results (RK chunk-confirm: after ``_finalize_utterance`` its
``get_partial()`` keeps returning ``is_final=True`` with the composed text on
every subsequent poll) then re-emit the same stale final on EVERY audio chunk
(~400ms) — a "final storm" that also feeds one stale embedding per poll into
the online diarizer.

The fix mirrors the frontend-VAD endpoint path: after sending the final,
close the old stream → ``asr_be.create_stream(language=...)`` →
``vad_session.reset()``.

These tests drive ``server.main._asr_stream_backend`` directly with fake
WS / backend / stream objects (no real server, no real models), in the style
of ``test_v2v_asr_desync_selfheal.py``.
"""
from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from fastapi import WebSocketDisconnect

import server.main as main_mod
from server.core import diarization as diar_mod
from server.core import speaker_embedding as spk_mod


def _asynctest(fn):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(fn(*args, **kwargs))
        finally:
            loop.close()
    wrapper.__name__ = fn.__name__
    return wrapper


# 100 ms of silence @16 kHz mono int16 — one "chunk" as the client would send.
_CHUNK = np.zeros(1600, dtype=np.int16).tobytes()
_SR = 16000


# ── fakes ──────────────────────────────────────────────────────────────

class FakeStickyStream:
    """Endpoint after ``endpoint_after`` accepted chunks, then STICKY:
    ``get_partial()`` returns the same ``(text, True)`` forever — the RK
    chunk-confirm post-finalize behavior. ``endpoint_after=None`` → never
    endpoints (fresh stream that just accumulates)."""

    def __init__(self, ident: int, endpoint_after):
        self.ident = ident
        self.endpoint_after = endpoint_after
        self.chunks = 0
        self.sample_lengths = []
        self.finalized = False
        self.closed = False
        self.prepare_calls = 0
        self.finalize_calls = 0

    def accept_waveform(self, sr: int, samples) -> None:  # noqa: ANN001
        self.chunks += 1
        self.sample_lengths.append(len(samples))

    def get_partial(self):
        if (self.endpoint_after is not None
                and self.chunks >= self.endpoint_after):
            self.finalized = True
        if self.finalized:
            # Sticky: identical stale final on every poll after finalize.
            return f"第一句文本-s{self.ident}", True
        return "", False

    def prepare_finalize(self) -> None:
        self.prepare_calls += 1

    def finalize(self):
        self.finalize_calls += 1
        return f"第一句文本-s{self.ident}", None

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    """``create_stream()`` returns a fresh counted stream each call.

    ``schedule`` gives each successive stream its ``endpoint_after``; once
    exhausted, the last entry repeats.
    """

    name = "fake-sticky-endpoint"

    def __init__(self, schedule=(2,)):
        self.schedule = list(schedule)
        self.streams: list[FakeStickyStream] = []
        self.create_calls = 0

    def create_stream(self, language: str = "auto"):
        idx = min(self.create_calls, len(self.schedule) - 1)
        self.create_calls += 1
        s = FakeStickyStream(self.create_calls, self.schedule[idx])
        self.streams.append(s)
        return s


class FakeWS:
    """Scripted WebSocket: yields queued messages then raises disconnect."""

    def __init__(self, messages):
        self._msgs = list(messages)
        self.sent: list[dict] = []
        self.close_calls: list[tuple] = []

    async def receive(self):
        if self._msgs:
            return self._msgs.pop(0)
        raise WebSocketDisconnect(1000)

    async def send_json(self, payload) -> None:
        self.sent.append(payload)

    async def close(self, code=None, reason=None) -> None:
        self.close_calls.append((code, reason))


class FailFinalWS(FakeWS):
    """Simulate a client disappearing while the final is sent."""

    async def send_json(self, payload) -> None:
        if payload.get("type") == "final":
            raise RuntimeError("client gone")
        await super().send_json(payload)


class FakeVADSession:
    """Never fires SPEECH_END — isolates the backend-endpoint branch."""

    def __init__(self):
        self.reset_calls = 0

    def process(self, samples):  # noqa: ANN001
        return None  # != VADSession.SPEECH_END

    def reset(self) -> None:
        self.reset_calls += 1


class FakeMultiEventVAD:
    """Return an end→start pair from one PCM block with explicit offsets."""

    def __init__(self):
        self.reset_calls = 0

    def process_events(self, samples):  # noqa: ANN001
        from server.core.vad import VADSession
        return [
            (VADSession.SPEECH_END, 400),
            (VADSession.SPEECH_START, 800),
        ]

    def reset(self) -> None:
        self.reset_calls += 1


class FakeDiarizer:
    def __init__(self):
        self.assign_calls: list[tuple] = []

    def assign(self, emb, start: float, end: float):  # noqa: ANN001
        self.assign_calls.append((start, end))
        return SimpleNamespace(speaker="S1", confidence=0.9)


def _finals(ws: FakeWS) -> list[dict]:
    return [p for p in ws.sent if p.get("type") == "final"]


async def _run(ws, backend, **kwargs):
    await main_mod._asr_stream_backend(
        ws, backend, language="auto", sample_rate=_SR, **kwargs
    )


# ── tests ──────────────────────────────────────────────────────────────

@_asynctest
async def test_sticky_endpoint_emits_one_final_not_a_storm():
    """One endpoint → ONE final, independent of how many more chunks arrive.

    Stream #1 endpoints after 2 chunks and stays sticky; the re-armed stream
    #2 never endpoints. Pre-fix, every chunk after the endpoint re-polled the
    sticky stream #1 → one duplicate stale final per chunk (N-1 finals for N
    chunks). Post-fix the count must NOT grow with the chunk count.
    """
    for n_chunks in (4, 8):
        backend = FakeBackend(schedule=(2, None))
        ws = FakeWS([{"bytes": _CHUNK}] * n_chunks)
        await _run(ws, backend)
        finals = _finals(ws)
        assert len(finals) == 1, (
            f"final storm: {len(finals)} finals for {n_chunks} chunks "
            f"(sticky endpoint was re-emitted instead of re-arming)"
        )
        assert finals[0]["text"] == "第一句文本-s1"


@_asynctest
async def test_endpoint_final_rearms_stream():
    """After the backend-endpoint final: old stream closed, create_stream
    called again (re-arm), and subsequent audio goes to the NEW stream."""
    backend = FakeBackend(schedule=(2, None))
    ws = FakeWS([{"bytes": _CHUNK}] * 5)
    await _run(ws, backend)
    # initial open + one re-arm after the single endpoint
    assert backend.create_calls == 2
    s1, s2 = backend.streams
    assert s1.closed is True          # old stream released on re-arm
    assert s1.chunks == 2             # fed only until its endpoint
    assert s2.chunks == 3             # remaining chunks go to the fresh stream
    assert s2.closed is True          # finally-block cleanup on disconnect


@_asynctest
async def test_finals_track_create_stream_not_chunk_count():
    """Every stream endpoints after 2 chunks (all sticky). 6 chunks →
    exactly 3 finals, one per stream generation; create_stream == finals + 1
    (the initial open plus one re-arm per final). Each final carries the text
    of its OWN stream — no stale repeats."""
    backend = FakeBackend(schedule=(2,))
    ws = FakeWS([{"bytes": _CHUNK}] * 6)
    await _run(ws, backend)
    finals = _finals(ws)
    assert len(finals) == 3
    assert backend.create_calls == len(finals) + 1
    texts = [p["text"] for p in finals]
    assert texts == ["第一句文本-s1", "第一句文本-s2", "第一句文本-s3"]
    assert len(set(texts)) == len(texts)  # no duplicated stale final


@_asynctest
async def test_endpoint_rearm_resets_vad_session():
    """When a server-side VAD session rides along (?vad=), the backend-endpoint
    re-arm must reset it, exactly like the frontend-VAD endpoint path."""
    backend = FakeBackend(schedule=(2, None))
    vad = FakeVADSession()
    ws = FakeWS([{"bytes": _CHUNK}] * 4)
    await _run(ws, backend, vad_session=vad)
    assert len(_finals(ws)) == 1
    assert vad.reset_calls == 1


@_asynctest
async def test_disconnect_message_abandons_without_prepare_or_finalize():
    """ASGI disconnect messages are teardown, not an empty-audio EOS frame."""
    backend = FakeBackend(schedule=(None,))
    ws = FakeWS([{"bytes": _CHUNK}, {"type": "websocket.disconnect"}])
    await _run(ws, backend)
    assert _finals(ws) == []
    stream = backend.streams[0]
    assert stream.prepare_calls == 0
    assert stream.finalize_calls == 0
    assert stream.closed is True


@_asynctest
async def test_vad_end_start_in_one_block_rearms_and_preserves_offsets():
    """A same-block end→start keeps each segment on its own stream."""
    backend = FakeBackend(schedule=(None, None))
    vad = FakeMultiEventVAD()
    ws = FakeWS([{"bytes": _CHUNK}, {"type": "websocket.disconnect"}])

    await _run(ws, backend, vad_session=vad)

    assert [p["type"] for p in ws.sent] == ["vad_endpoint", "final"]
    assert backend.create_calls == 2
    first, second = backend.streams
    assert first.sample_lengths == [400]
    assert second.sample_lengths == [400, 800]
    assert first.closed is True
    assert second.closed is True
    # process_events already advanced VAD state for the later start; resetting
    # here would erase that state before the next block arrives.
    assert vad.reset_calls == 0


@_asynctest
async def test_vad_final_send_failure_exits_connection_without_rearm():
    """A failed final send stops the outer receive loop and closes the stream."""
    backend = FakeBackend(schedule=(None, None))
    vad = FakeMultiEventVAD()
    ws = FailFinalWS([{"bytes": _CHUNK}, {"type": "websocket.disconnect"}])

    await _run(ws, backend, vad_session=vad)

    assert backend.create_calls == 1
    assert backend.streams[0].sample_lengths == [400]
    assert backend.streams[0].closed is True


@_asynctest
async def test_empty_bytes_is_eos_and_finalizes_once():
    """A real zero-length bytes frame remains the explicit EOS protocol."""
    backend = FakeBackend(schedule=(None,))
    ws = FakeWS([{"bytes": _CHUNK}, {"bytes": b""}])
    await _run(ws, backend)
    assert len(_finals(ws)) == 1
    stream = backend.streams[0]
    assert stream.prepare_calls == 1
    assert stream.finalize_calls == 1
    assert stream.closed is True


@_asynctest
async def test_diarizer_assign_once_per_utterance():
    """The storm's downstream damage: each stale final fed one stale embedding
    into the online diarizer. Post-fix, the diarizer's assign() runs exactly
    once per real utterance.

    The heavy pieces (speaker-embedding model, real OnlineDiarizer) are
    monkeypatched at their module seams — ``_augment_final_payload`` itself
    runs for real, so the assign-per-final wiring is exercised end to end.
    """
    fake_diar = FakeDiarizer()

    saved = (
        diar_mod.make_session_diarizer,
        spk_mod.compute_embedding,
        spk_mod.embedding_payload,
    )
    diar_mod.make_session_diarizer = lambda: fake_diar
    spk_mod.compute_embedding = lambda samples, sr: np.ones(4, dtype=np.float32)
    spk_mod.embedding_payload = lambda emb: {
        "speaker_embedding": [1.0, 1.0, 1.0, 1.0], "dim": 4, "normalized": True,
    }
    try:
        backend = FakeBackend(schedule=(2, None))
        ws = FakeWS([{"bytes": _CHUNK}] * 8)
        await _run(ws, backend, spk_on=True, diarize_on=True)
    finally:
        (diar_mod.make_session_diarizer,
         spk_mod.compute_embedding,
         spk_mod.embedding_payload) = saved

    finals = _finals(ws)
    assert len(finals) == 1
    assert len(fake_diar.assign_calls) == 1, (
        f"diarizer fed {len(fake_diar.assign_calls)} embeddings for 1 utterance"
    )
    assert finals[0]["speaker"] == "S1"
    assert "start" in finals[0] and "end" in finals[0]


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok {_name}")
