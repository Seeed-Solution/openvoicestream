"""Frontend-VAD speech-onset preroll back-fill (/v2v/stream).

Regression test for the systematic *first-word drop* observed on the reBot
demo (real-machine 2026-06-15): silero only emits SPEECH_START after the word
onset has crossed its threshold, so the leading audio frames were consumed by
``vad.process()`` while no ASR turn was open and never reached the decoder.
Short single-word commands ("wave") lost their only content word; multi-word
commands lost the leading word ("grab the box" → "the box").

The fix keeps a short rolling preroll ring of pre-speech frames and replays it
into the fresh ASR stream on SPEECH_START, before the trigger chunk. This test
drives the real /v2v dispatch loop with a *scripted* VAD (so it does not need
the silero ONNX model) and a *recording* ASR stream, then asserts the onset
frames buffered before SPEECH_START were delivered to the stream.

Mirrors the doubles in ``test_v2v_vad_event.py``; kept separate so the recording
stream does not perturb that test's assertions.
"""
from __future__ import annotations

import os
import sys
import time
from typing import List

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from server.core.asr_backend import ASRBackend, ASRCapability
from server.core import vad as vad_mod


# ──────────────────────────────────────────────────────────────────────
# Test doubles — a stream that records every waveform it is handed.
# ──────────────────────────────────────────────────────────────────────
class _RecordingStream:
    def __init__(self) -> None:
        self.chunks: List[np.ndarray] = []
        self.finalized = False

    def accept_waveform(self, sr, samples):  # noqa: ANN001
        self.chunks.append(np.asarray(samples, dtype=np.float32).copy())

    def get_partial(self):
        return "", False

    def finalize(self):
        self.finalized = True
        return "ok", None

    def cancel(self):
        pass

    def cancel_and_finalize(self):
        return ""

    @property
    def total_samples(self) -> int:
        return int(sum(c.size for c in self.chunks))


class _RecordingASRBackend(ASRBackend):
    def __init__(self) -> None:
        self.streams_created: List[_RecordingStream] = []

    @property
    def name(self):
        return "recording-preroll"

    @property
    def capabilities(self):
        return {ASRCapability.STREAMING}

    @property
    def sample_rate(self):
        return 16000

    def is_ready(self):
        return True

    def preload(self):
        return None

    def transcribe(self, audio_bytes, language="auto"):
        from server.core.asr_backend import TranscriptionResult
        return TranscriptionResult(text="", duration=0.0, inference_time=0.0,
                                   rtf=0.0, n_tokens=0, per_token_ms=0.0,
                                   backend=self.name)

    def transcribe_audio(self, audio, language="auto"):
        return self.transcribe(b"", language)

    def create_stream(self, language="auto"):
        s = _RecordingStream()
        self.streams_created.append(s)
        return s


class _ScriptedVAD:
    """Returns a scripted sequence of VAD events; None once exhausted."""

    def __init__(self, events):
        self._events = list(events)
        self._i = 0

    def process(self, samples):
        if self._i < len(self._events):
            ev = self._events[self._i]
            self._i += 1
            return ev
        return None


def _const_pcm16(value: int, ms: int = 50, sr: int = 16000) -> bytes:
    n = (sr * ms) // 1000
    return np.full(n, int(value), dtype=np.int16).tobytes()


@pytest.fixture
def recording_backend(monkeypatch):
    import server.main as main_mod
    from server.core.coordinator import init_coordinator
    from server.core import session_limiter

    init_coordinator({"mode": "concurrent"})
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({})

    be = _RecordingASRBackend()
    monkeypatch.setattr(main_mod, "_asr_backend", be, raising=False)
    monkeypatch.setattr(main_mod, "_get_asr_backend", lambda: be)
    return be


def _open_v2v(client, *, multi_utterance=False, vad="silero"):
    cfg = {
        "type": "config",
        "asr_language": "en",
        "vad": vad,
        "sample_rate": 16000,
        "multi_utterance": multi_utterance,
    }
    ws = client.websocket_connect("/v2v/stream")
    ws.__enter__()
    ws.send_json(cfg)
    return ws


def _drain_until(ws, want_type, timeout_s=5.0, max_msgs=80):
    deadline = time.monotonic() + timeout_s
    seen = []
    while time.monotonic() < deadline and len(seen) < max_msgs:
        try:
            payload = ws.receive_json()
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"WS recv error: {e}; seen={seen}")
        seen.append(payload)
        if payload.get("type") == want_type:
            return seen
    raise AssertionError(f"timed out waiting for {want_type}; seen={seen}")


def test_preroll_backfills_onset_frames(recording_backend, monkeypatch):
    """Three idle frames precede SPEECH_START; all three must be replayed into
    the stream ahead of the trigger chunk (onset back-fill), so the decoder
    sees the full utterance rather than only the post-trigger tail."""
    from fastapi.testclient import TestClient
    from server.main import app

    # None,None,None → buffered as preroll; then SPEECH_START, then SPEECH_END.
    fake_vad = _ScriptedVAD(events=[
        None, None, None,
        vad_mod.VADSession.SPEECH_START,
        vad_mod.VADSession.SPEECH_END,
    ])
    monkeypatch.setattr(vad_mod, "create_vad", lambda *a, **kw: fake_vad)

    client = TestClient(app)
    ws = _open_v2v(client, multi_utterance=False, vad="silero")
    try:
        # 3 pre-speech frames (the onset silero would clip) marked with a
        # distinct amplitude, then the SPEECH_START + SPEECH_END frames.
        for _ in range(3):
            ws.send_bytes(_const_pcm16(1000, ms=50))   # onset / preroll
        ws.send_bytes(_const_pcm16(2000, ms=50))       # SPEECH_START trigger
        ws.send_bytes(_const_pcm16(3000, ms=50))       # SPEECH_END tail

        _drain_until(ws, "asr_final")
    finally:
        ws.__exit__(None, None, None)

    assert recording_backend.streams_created, "no ASR stream was created"
    stream = recording_backend.streams_created[0]
    audio = np.concatenate(stream.chunks) if stream.chunks else np.empty(0)

    # 5 frames × 800 samples = 4000 if the onset was back-filled; without the
    # preroll fix only the trigger + tail (1600) would survive.
    assert stream.total_samples == 4000, (
        f"expected 5 frames (4000 samples) reached the decoder, got "
        f"{stream.total_samples}; preroll onset back-fill missing"
    )
    # The onset marker amplitude (1000/32768) must be present — i.e. the
    # pre-SPEECH_START frames really were delivered, not just counted.
    onset_val = 1000 / 32768.0
    assert np.any(np.isclose(audio, onset_val, atol=1e-3)), (
        "onset-marked frames absent from the decoder input (first word dropped)"
    )


def test_preroll_disabled_drops_onset(recording_backend, monkeypatch):
    """With OVS_VAD_PREROLL_MS=0 the back-fill is off — the pre-SPEECH_START
    frames are NOT replayed (pins the knob + documents the old behaviour)."""
    monkeypatch.setenv("OVS_VAD_PREROLL_MS", "0")
    from fastapi.testclient import TestClient
    from server.main import app

    fake_vad = _ScriptedVAD(events=[
        None, None, None,
        vad_mod.VADSession.SPEECH_START,
        vad_mod.VADSession.SPEECH_END,
    ])
    monkeypatch.setattr(vad_mod, "create_vad", lambda *a, **kw: fake_vad)

    client = TestClient(app)
    ws = _open_v2v(client, multi_utterance=False, vad="silero")
    try:
        for _ in range(3):
            ws.send_bytes(_const_pcm16(1000, ms=50))
        ws.send_bytes(_const_pcm16(2000, ms=50))
        ws.send_bytes(_const_pcm16(3000, ms=50))
        _drain_until(ws, "asr_final")
    finally:
        ws.__exit__(None, None, None)

    stream = recording_backend.streams_created[0]
    # Only the SPEECH_START trigger + SPEECH_END tail (2 × 800) reach the stream.
    assert stream.total_samples == 1600, (
        f"preroll disabled should deliver only trigger+tail (1600), got "
        f"{stream.total_samples}"
    )


def test_backend_owned_stream_does_not_drop_later_vad_start_chunk(
    recording_backend, monkeypatch
):
    """A backend-owned stream may open before frontend VAD emits start.

    Once that lazy-open path has set ``asr_started_once``, a later VAD start is
    a barge-in notification only; its audio still belongs to the active stream
    and must be accepted exactly once.
    """
    from fastapi.testclient import TestClient
    from server.main import app

    recording_backend.prefer_backend_endpoint_vad = True
    monkeypatch.setattr(_RecordingStream, "prefer_backend_endpoint_vad", True, raising=False)
    fake_vad = _ScriptedVAD(events=[
        None,                         # backend-owned lazy open + accept
        vad_mod.VADSession.SPEECH_START,  # must still be accepted
        vad_mod.VADSession.SPEECH_END,
    ])
    monkeypatch.setattr(vad_mod, "create_vad", lambda *a, **kw: fake_vad)

    client = TestClient(app)
    ws = _open_v2v(client, multi_utterance=False, vad="silero")
    try:
        ws.send_bytes(_const_pcm16(1000, ms=50))
        ws.send_bytes(_const_pcm16(2000, ms=50))
        ws.send_bytes(_const_pcm16(3000, ms=50))
        ws.send_json({"type": "asr_eos"})
        _drain_until(ws, "asr_final")
    finally:
        ws.__exit__(None, None, None)

    stream = recording_backend.streams_created[0]
    assert stream.total_samples == 2400, (
        "backend-owned active stream must accept all three chunks exactly once"
    )
    audio = np.concatenate(stream.chunks)
    assert np.allclose(audio[:800], 1000 / 32768.0, atol=1e-3)
    assert np.allclose(audio[800:1600], 2000 / 32768.0, atol=1e-3)
    assert np.allclose(audio[1600:], 3000 / 32768.0, atol=1e-3)
