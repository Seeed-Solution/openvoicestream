"""v011 regression: native ASR empty-binary EOS must finalize like text EOU.

The native receiver used to queue a bare ``b""`` frame as ordinary bytes.
With ``vad_session=None`` the processor then called ``_process_audio(b"")``,
whose ``np.frombuffer(b"", int16)`` split yields no segment and therefore
NEVER finalized. The legacy empty-PCM end-of-utterance semantics (and the
performance client, which stamps EOS by sending ``b""``) were lost.

This test AST-extracts the ACTUAL ``_asr_stream_backend`` coroutine, the
actual ``_AsrSlotJobs`` class and the actual ``_split_vad_block`` /
``_unpack_finalize_result`` / ``_augment_final_payload`` helpers from
``server/main.py`` (no mirrored pipeline, no fabricated handler). It execs
them against a controlled fake native backend/stream whose ``accept_waveform``
and ``finalize`` are observable and return KNOWN text, plus the real numpy
available in the test environment. The controlled jobs are finite, every
gate is opened in ``finally`` and every task is joined within the original
deadlines, so a failure cannot hang the executor.

Cases:
  1. normal PCM + empty-binary EOS -> EXACTLY one final, ``finalize`` called
     exactly once (empty EOS routed to the retained stream).
  2. empty-binary EOS while a cancel is armed -> discarded, NO finalize; the
     matching cancel receipt then resolves and a reset re-arms cleanly.
  3. the existing text ``end_utterance`` / ``type=eou`` path still works.
  4. disconnect without EOS -> NO finalize (no offline finalize on
     disconnect).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import pathlib
import threading
import time

import numpy as np
import pytest

# The root pytest configuration has no pytest-asyncio auto mode, so async
# test functions in this module must be explicitly marked.
pytestmark = pytest.mark.asyncio

# Existing accepted fixture helpers: FakeWS (scripted websocket) and
# _RecorderSlot (observable slot). Imported via a valid pytest import path
# (server/tests is a package with __init__.py) — no AST-lookalike handler.
from server.tests.test_asr_native_control_pipeline import (
    FakeWS,
    _RecorderSlot,
    _extract_source as _extract_source_ref,
)

logging.basicConfig(level=logging.CRITICAL)

_MAIN = pathlib.Path(__file__).resolve().parents[1] / "main.py"


# ---------------------------------------------------------------------------
# Controlled module build: REAL extracted pipeline + REAL numpy/helpers
# ---------------------------------------------------------------------------

def _extract_source(names: set[str]) -> str:
    return _extract_source_ref(names)


def _build_module(pool) -> tuple:
    """Exec the ACTUAL pipeline with controlled globals and REAL helpers.

    Unlike the plain lifecycle fixture, ``_split_vad_block`` /
    ``_unpack_finalize_result`` / ``_augment_final_payload`` are the REAL
    server helpers, because the empty-EOS path runs ``_handle_end_utterance``
    which calls ``prepare_finalize``/``finalize`` and then unpacks + augments
    the final payload.
    """
    recorder = _RecorderSlot()
    env = {
        "asyncio": asyncio,
        "json": json,
        "logging": logging,
        "threading": threading,
        "concurrent": concurrent.futures,
        "contextlib": contextlib,
        "logger": logging.getLogger("test.asr.native.empty_eos"),
        "np": np,
        "_AsrSlotJobs": None,
        "_get_asr_executor": lambda: pool,
        "_asr_no_slot": recorder.make(),
        "_asr_utterance_slot": recorder.make(),
        "_diar_mod_be": _FakeDiarMod,
        "_send_asr_busy": None,
        "_is_pool_saturated": lambda e: (False, None),
        "_asr_stream_backend": None,
        "WebSocket": object,
        "asynccontextmanager": contextlib.asynccontextmanager,
        "ThreadPoolExecutor": concurrent.futures.ThreadPoolExecutor,
    }
    helpers = _extract_source(
        {"_split_vad_block", "_unpack_finalize_result", "_augment_final_payload"}
    )
    pipeline = _extract_source({"_AsrSlotJobs", "_asr_stream_backend"})
    exec(compile(helpers + "\n" + pipeline, str(_MAIN), "exec"), env)
    return env["_asr_stream_backend"], recorder


class _FakeDiarMod:
    @staticmethod
    def make_session_diarizer():
        return None


# ---------------------------------------------------------------------------
# Observable native backend / stream (known text, counted finalize)
# ---------------------------------------------------------------------------

FINAL_TEXT = "hello world"
FINAL_LANG = "en"


class ObservableStream:
    """Native-faithful stream with observable accept/finalize.

    ``accept_waveform`` counts the actual PCM it receives; ``finalize`` counts
    its calls and returns the KNOWN finalize contract tuple ``(text, lang)``.
    ``close`` marks closed unconditionally (an unarmed stream is releasable),
    which is what the reset/teardown proof checks.
    """

    def __init__(self, backend, sid: str):
        self._backend = backend
        self._session_id = sid
        self._closed = False
        self._cancel_intent = None
        self._cancel_confirm = None
        self._cancel_exit = False
        self.close_calls = 0
        self.accept_calls = 0
        self.bytes_accepted = 0
        self.finalize_calls = 0
        self.prepare_calls = 0
        self.partial_calls = 0
        self.arm_cancel_calls = 0
        self.request_cancel_calls = 0

    # -- native cancel surface (canonical shape) --
    def arm_cancel(self):
        self.arm_cancel_calls += 1
        self._cancel_intent = {"sid": self._session_id}
        return self._session_id

    def request_cancel(self, timeout_s: float):
        self.request_cancel_calls += 1
        ev = self._backend.cancel_gate.wait(timeout=5)
        if not ev:
            raise TimeoutError("fake cancel helper timeout")
        receipt = self._backend.programmed_receipt
        if receipt is None:
            raise TimeoutError("fake cancel helper timeout")
        if isinstance(receipt, BaseException):
            raise receipt
        if (
            isinstance(receipt, dict)
            and receipt.get("event") == "cancelled"
            and receipt.get("id") == self._session_id
            and receipt.get("ok") is False
        ):
            self._cancel_confirm = {"sid": self._session_id, "receipt": receipt}
        return receipt

    # -- ASR stream surface --
    def accept_waveform(self, sample_rate, samples):
        self.accept_calls += 1
        self.bytes_accepted += int(getattr(samples, "size", len(samples)))

    def prepare_finalize(self):
        self.prepare_calls += 1

    def finalize(self):
        self.finalize_calls += 1
        # Mirror the ACTUAL native backend contract (see V8 audit: worker
        # finalize closes the stream in its finally block): after a terminal
        # finalize the SID is proven closed. Test-protocol scope only — this
        # mock close is NOT canonical capacity evidence.
        self._closed = True
        self._backend.events.append(("finalize_close", self._session_id))
        return FINAL_TEXT, FINAL_LANG

    def get_partial(self):
        self.partial_calls += 1
        return ("", False)

    def close(self):
        self.close_calls += 1
        self._backend.events.append(("close", self._session_id))
        self._closed = True


class ObservableBackend:
    name = "trt_edgellm"

    def __init__(self, create_gate: threading.Event | None = None):
        self._worker = None
        self.streams: list[ObservableStream] = []
        self.events: list[tuple[str, str | None]] = []
        self.create_gate = create_gate
        self.cancel_gate = threading.Event()
        self.programmed_receipt = None
        self._n = 0
        self._lock = threading.Lock()

    def _use_streaming_worker(self):
        return True

    def create_stream(self, language: str = "auto"):
        if self.create_gate is not None:
            self.create_gate.wait(timeout=5)
        with self._lock:
            self._n += 1
            s = ObservableStream(self, f"sid-{self._n}")
            self.streams.append(s)
            self.events.append(("create", s._session_id))
        return s

    def release(self):
        self.cancel_gate.set()


def _matching_receipt(sid: str) -> dict:
    return {"event": "cancelled", "id": sid, "ok": False}


# ---------------------------------------------------------------------------
# Harness (per-test ordinary pool; all gates/tasks closed in finally)
# ---------------------------------------------------------------------------

@pytest.fixture()
def harness():
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-asr-empty-eos"
    )
    factory, recorder = _build_module(pool)

    def run(ws, backend, **kw):
        return factory(
            ws=ws,
            asr_be=backend,
            language="auto",
            sample_rate=16000,
            vad_session=None,
            punct_on=False,
            spk_on=False,
            diarize_on=False,
            per_utterance_slot=False,
            **kw,
        )

    yield run, recorder
    pool.shutdown(wait=True)


async def _wait(predicate, timeout=5.0, msg="condition"):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timeout waiting for {msg}")
        await asyncio.sleep(0.01)
    return True


async def _cleanup(task, backend, *, expect_done=True, timeout=5.0):
    """Finite teardown: release gates, then bounded-join the handler task."""
    backend.cancel_gate.set()
    if expect_done:
        await asyncio.wait_for(task, timeout=timeout)
    else:
        if not task.done():
            task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, timeout=timeout)


# ---------------------------------------------------------------------------
# 1. normal PCM + empty-binary EOS -> exactly one final / one finalize
# ---------------------------------------------------------------------------

async def test_normal_pcm_empty_eos_finalizes_once(harness):
    run, recorder = harness
    be = ObservableBackend()
    ws = FakeWS([])
    task = asyncio.create_task(run(ws, be))
    try:
        await _wait(lambda: len(be.streams) >= 1)
        stream = be.streams[0]
        # Normal nonempty PCM: accepted, but with no VAD it does not finalize.
        pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
        ws.binary(pcm)
        await _wait(lambda: stream.accept_calls >= 1)
        assert stream.finalize_calls == 0
        # Empty-binary EOS must finalize the retained stream exactly once.
        ws.binary(b"")
        await _wait(lambda: stream.finalize_calls >= 1, msg="empty-EOS finalize")
        finals = [m for m in ws.sent if isinstance(m, dict) and m.get("type") == "final"]
        assert len(finals) == 1, ws.sent
        assert finals[0]["text"] == FINAL_TEXT
        assert finals[0]["is_final"] is True
        assert stream.finalize_calls == 1
        assert stream.prepare_calls == 1
        # Disconnect must NOT trigger another offline finalize.
        ws.disconnect()
        await _cleanup(task, be)
        await asyncio.sleep(0.1)
        assert stream.finalize_calls == 1
    finally:
        await _cleanup(task, be, expect_done=task.done())


# ---------------------------------------------------------------------------
# 2. empty EOS while cancel armed -> discarded (no finalize); reset re-arms
# ---------------------------------------------------------------------------

async def test_empty_eos_while_cancel_armed_discarded_then_reset(harness):
    run, recorder = harness
    be = ObservableBackend()
    # Cancel helper returns a MATCHING receipt immediately (gate open) so the
    # cancel resolves without an artificial thread wait, and reset can re-arm.
    be.cancel_gate.set()
    ws = FakeWS([])
    task = asyncio.create_task(run(ws, be))
    try:
        await _wait(lambda: len(be.streams) >= 1)
        stream = be.streams[0]
        be.programmed_receipt = _matching_receipt(stream._session_id)
        # Arm the cancel via the text control command.
        ws.text({"command": "cancel"})
        await _wait(lambda: stream._cancel_confirm is not None, msg="cancel receipt")
        # Empty-binary EOS while armed: discarded, NO finalize.
        ws.binary(b"")
        await asyncio.sleep(0.2)
        assert stream.finalize_calls == 0, "empty EOS must be discarded while armed"
        assert not any(
            isinstance(m, dict) and m.get("type") == "final" for m in ws.sent
        )
        # Matching cancel resolved; an explicit reset re-arms a new generation.
        ws.text({"command": "reset"})
        await _wait(lambda: len(be.streams) >= 2, msg="reset re-arm")
        assert stream.close_calls >= 1 and stream._closed
        assert stream.finalize_calls == 0
        # Clean teardown.
        ws.disconnect()
        await _cleanup(task, be)
    finally:
        await _cleanup(task, be, expect_done=task.done())


# ---------------------------------------------------------------------------
# 3. existing text EOU still finalizes (regression guard)
# ---------------------------------------------------------------------------

async def test_text_end_utterance_still_finalizes(harness):
    run, recorder = harness
    be = ObservableBackend()
    ws = FakeWS([])
    task = asyncio.create_task(run(ws, be))
    try:
        await _wait(lambda: len(be.streams) >= 1)
        stream = be.streams[0]
        pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
        ws.binary(pcm)
        await _wait(lambda: stream.accept_calls >= 1)
        ws.text({"command": "end_utterance"})
        await _wait(lambda: stream.finalize_calls >= 1, msg="text EOU finalize")
        finals = [m for m in ws.sent if isinstance(m, dict) and m.get("type") == "final"]
        assert len(finals) == 1
        assert finals[0]["text"] == FINAL_TEXT
        assert stream.finalize_calls == 1
        ws.disconnect()
        await _cleanup(task, be)
    finally:
        await _cleanup(task, be, expect_done=task.done())


# ---------------------------------------------------------------------------
# 5. disconnect AFTER proven-closed SID -> NO arm_cancel / request_cancel
# ---------------------------------------------------------------------------

async def test_disconnect_after_proven_closed_no_cancel(harness):
    run, recorder = harness
    be = ObservableBackend()
    ws = FakeWS([])
    task = asyncio.create_task(run(ws, be))
    try:
        await _wait(lambda: len(be.streams) >= 1)
        stream = be.streams[0]
        assert stream._closed is False, "stream must start open"
        pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
        ws.binary(pcm)
        await _wait(lambda: stream.accept_calls >= 1)
        # Normal PCM + empty-binary EOS: exactly one final, and the mock
        # stream is proven closed by the terminal finalize (backend contract).
        ws.binary(b"")
        await _wait(
            lambda: stream.finalize_calls >= 1 and stream._closed,
            msg="final + proven closed",
        )
        finals = [m for m in ws.sent if isinstance(m, dict) and m.get("type") == "final"]
        assert len(finals) == 1
        # Disconnect against an ALREADY CLOSED SID: the cancel must NOT be
        # armed (no arm_cancel, no blocking request_cancel for a dead stream);
        # teardown cleanup/drain still runs. No fabricated extra final.
        ws.disconnect()
        await _cleanup(task, be)
        await asyncio.sleep(0.1)
        assert stream.arm_cancel_calls == 0, "closed SID must skip arm_cancel"
        assert stream.request_cancel_calls == 0, "closed SID must skip request_cancel"
        assert stream._cancel_intent is None
        finals = [m for m in ws.sent if isinstance(m, dict) and m.get("type") == "final"]
        assert len(finals) == 1, "exactly the original final, nothing fabricated"
    finally:
        await _cleanup(task, be, expect_done=task.done())


# ---------------------------------------------------------------------------
# 6. live PCM disconnect (no EOS) -> cancel IS armed, no finalize
# ---------------------------------------------------------------------------

async def test_live_disconnect_still_arms_cancel(harness):
    run, recorder = harness
    be = ObservableBackend()
    be.cancel_gate.set()
    ws = FakeWS([])
    task = asyncio.create_task(run(ws, be))
    try:
        await _wait(lambda: len(be.streams) >= 1)
        stream = be.streams[0]
        be.programmed_receipt = _matching_receipt(stream._session_id)
        pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
        ws.binary(pcm)
        await _wait(lambda: stream.accept_calls >= 1)
        assert stream._closed is False, "live stream is open before disconnect"
        assert stream.finalize_calls == 0
        # Live (unconfirmed-closed) stream: disconnect MUST still arm and run
        # the native cancel for the retained SID.
        ws.disconnect()
        await _wait(lambda: stream.request_cancel_calls >= 1, msg="native cancel")
        await _cleanup(task, be)
        await asyncio.sleep(0.1)
        assert stream.arm_cancel_calls >= 1, "live disconnect must arm the cancel"
        assert stream.request_cancel_calls >= 1
        assert stream._cancel_confirm is not None, "matching receipt confirms"
        # NO finalize and NO fabricated final on disconnect.
        assert stream.finalize_calls == 0
        assert not any(
            isinstance(m, dict) and m.get("type") == "final" for m in ws.sent
        )
    finally:
        await _cleanup(task, be, expect_done=task.done())


# ---------------------------------------------------------------------------
# 4. disconnect without EOS -> NO finalize
# ---------------------------------------------------------------------------

async def test_disconnect_no_eos_no_finalize(harness):
    run, recorder = harness
    be = ObservableBackend()
    ws = FakeWS([])
    task = asyncio.create_task(run(ws, be))
    try:
        await _wait(lambda: len(be.streams) >= 1)
        stream = be.streams[0]
        pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
        ws.binary(pcm)
        await _wait(lambda: stream.accept_calls >= 1)
        ws.disconnect()
        await _cleanup(task, be)
        await asyncio.sleep(0.1)
        assert stream.finalize_calls == 0, "disconnect must never finalize offline"
        assert not any(
            isinstance(m, dict) and m.get("type") == "final" for m in ws.sent
        )
    finally:
        await _cleanup(task, be, expect_done=task.done())
