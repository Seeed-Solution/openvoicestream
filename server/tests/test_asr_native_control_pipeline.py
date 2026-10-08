"""U3-final lifecycle tests: retained native stream lifetime across
reset / cancel / disconnect.

These tests AST-extract the ACTUAL ``_asr_stream_backend`` coroutine and the
ACTUAL ``_AsrSlotJobs`` class from ``server/main.py`` (no mirrored copies of
the pipeline) and exec them against controlled globals: a fake WS, a fake
native backend/stream factory with a per-test ordinary pool of one worker plus
the pipeline's own dedicated control pool, and an observable no-op slot so
admission release can be observed.

Scenarios:
  1. normal reset closes the old stream BEFORE creating the new one (real
     event journal, not a vacuous index check), then a disconnect arms the
     native cancel which resolves with an ACTUAL matching receipt produced by
     the REAL request_cancel path, ending the connection cleanly.
  2. cancel helper returns TimeoutError (gate opened explicitly so the
     programmed Timeout path starts at once), client disconnects: the slot /
     admission is NOT released while the native proof is UNKNOWN; teardown
     only after a SIMULATED matching receipt is injected into the fake stream
     EXACTLY as the canonical backend stores it (fixture-only teardown proof,
     NOT real device qualification).
  3. blocked create_stream + handler cancellation: while the actual thread CF
     must still drain, the task is NOT done and the slot is held; only after
     the gate opens is the actually-created stream (recorded inside the
     creation callable) CLOSED before the slot is released (real close event
     journal ordered before the observed slot exit).
  4. invalid "matching" receipt with wrong sid / ok=True: NO cancel_ack and
     NO fake free — the slot stays held (quarantine) until a SIMULATED
     matching receipt (documented fixture injection, same canonical shape)
     provides the teardown proof.

Everything is bounded by absolute-budget waits (no ``asyncio.timeout`` — NX
3.10) and ``asyncio.wait_for``; all threads are joined via the pipeline's own
drains and per-test pool shutdowns (never killed). The ordinary pool is
PER-TEST (no module-global scope leak).
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import contextlib
import json
import logging
import pathlib
import threading
import time

import pytest

logging.basicConfig(level=logging.CRITICAL)

_MAIN = pathlib.Path(__file__).resolve().parents[1] / "main.py"


# ---------------------------------------------------------------------------
# AST extraction of the ACTUAL server code
# ---------------------------------------------------------------------------

def _extract_source(names: set[str]) -> str:
    tree = ast.parse(_MAIN.read_text())
    chunks = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in names:
            chunks.append(ast.get_source_segment(_MAIN.read_text(), node))
    assert len(chunks) == len(names), (names,)
    return "\n".join(chunks)


class _RecorderSlot:
    """Controlled global slot CM: records enter/exit so tests can observe
    whether the execution slot / admission was actually released.

    When ``log`` is set (typically the backend's event journal), enter/exit
    markers are appended there too, so ordering across backend and slot
    events is observable in ONE list.
    """

    def __init__(self):
        self.entered = 0
        self.exited = 0
        self.log: list | None = None

    def make(self):
        rec = self

        @contextlib.asynccontextmanager
        async def _slot(jobs):
            rec.entered += 1
            if rec.log is not None:
                rec.log.append(("slot_enter", None))
            try:
                yield 0.0
            finally:
                await jobs.drain()
                rec.exited += 1
                if rec.log is not None:
                    rec.log.append(("slot_exit", None))

        return _slot


def _build_module(pool) -> tuple:
    """Exec the ACTUAL _AsrSlotJobs + _asr_stream_backend with controlled
    globals (PER-TEST ordinary pool) and return (coro_factory, recorder)."""
    recorder = _RecorderSlot()
    env = {
        "asyncio": asyncio,
        "json": json,
        "logging": logging,
        "threading": threading,
        "concurrent": concurrent.futures,
        "contextlib": contextlib,
        "logger": logging.getLogger("test.asr.native"),
        "_AsrSlotJobs": None,  # filled below (class defined by extracted code)
        "_get_asr_executor": lambda: pool,
        "_asr_no_slot": recorder.make(),
        "_asr_utterance_slot": recorder.make(),
        "_diar_mod_be": _FakeDiarMod,
        "_send_asr_busy": None,
        "_split_vad_block": None,
        "_augment_final_payload": None,
        "_unpack_finalize_result": None,
        "_is_pool_saturated": lambda e: (False, None),
        "_asr_stream_backend": None,
        "WebSocket": object,
        "asynccontextmanager": contextlib.asynccontextmanager,
        "ThreadPoolExecutor": concurrent.futures.ThreadPoolExecutor,
    }
    src = _extract_source({"_AsrSlotJobs", "_asr_stream_backend"})
    exec(compile(src, str(_MAIN), "exec"), env)
    return env["_asr_stream_backend"], recorder


class _FakeDiarMod:
    @staticmethod
    def make_session_diarizer():
        return None


class FakeWS:
    """Controlled websocket: scripted inbound frames, recorded sends."""

    def __init__(self, script):
        self._q: asyncio.Queue = asyncio.Queue()
        for item in script:
            self._q.put_nowait(item)
        self.sent: list = []
        self._closed = False

    async def receive(self):
        return await self._q.get()

    async def send_json(self, payload):
        self.sent.append(payload)

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    def text(self, obj):
        self._q.put_nowait({"type": "websocket.receive", "text": json.dumps(obj)})

    def binary(self, data: bytes):
        self._q.put_nowait({"type": "websocket.receive", "bytes": data})

    def disconnect(self):
        self._q.put_nowait({"type": "websocket.disconnect"})


class FakeStream:
    """Controlled native stream: _closed / _cancel_intent / _cancel_confirm /
    _cancel_exit exactly as the canonical streaming stream exposes them.

    ``_cancel_confirm`` is stored EXACTLY as the canonical backend stores it:
    ``{"sid": <intent sid>, "receipt": <returned receipt dict>}``.
    """

    def __init__(self, backend, sid: str):
        self._backend = backend
        self._session_id = sid
        self._closed = False
        self._cancelled = False
        self._cancel_intent = None
        self._cancel_confirm = None
        self._cancel_exit = False
        self.close_calls = 0
        # set by the test: close() refuses (no _closed) while this is set
        self.refuse_close_until_released = False

    def arm_cancel(self):
        self._cancelled = True
        self._cancel_intent = {"sid": self._session_id}
        return self._session_id

    def request_cancel(self, timeout_s: float):
        ev = self._backend.cancel_gate.wait(timeout=5)
        if not ev:
            raise TimeoutError("fake cancel helper timeout")
        receipt = self._backend.programmed_receipt
        if receipt is None:
            raise TimeoutError("fake cancel helper timeout")
        if isinstance(receipt, BaseException):
            raise receipt
        self._cancel_confirm = {"sid": self._session_id, "receipt": receipt}
        return receipt

    def accept_waveform(self, sample_rate, samples):
        pass

    def prepare_finalize(self):
        pass

    def finalize(self):
        self.close()
        return "", None

    def get_partial(self):
        return ("", False)

    def close(self):
        self.close_calls += 1
        # REAL close-event journal: every actual close attempt is recorded,
        # even a refused one, so ordering (close before next create, close
        # before slot release) is asserted against evidence, not vacuously.
        self._backend.events.append(("close", self._session_id))
        if self.refuse_close_until_released and not self._backend.released:
            return  # refuses: backend cannot release the native session
        self._closed = True


class FakeBackend:
    """Controlled trt_edgellm-like backend: observable create/close order,
    worker snapshot source, programmed cancel receipts."""

    name = "trt_edgellm"

    def __init__(self, create_gate: threading.Event | None = None):
        self._worker = None  # "Popen-like"; None = UNKNOWN reference
        self.streams: list[FakeStream] = []
        self.events: list[tuple[str, str | None]] = []
        self.create_gate = create_gate
        self.cancel_gate = threading.Event()
        self.programmed_receipt = None
        self.released = False
        self._n = 0
        self._lock = threading.Lock()

    def _use_streaming_worker(self):
        return True

    def create_stream(self, language: str = "auto"):
        if self.create_gate is not None:
            self.create_gate.wait(timeout=5)
        with self._lock:
            self._n += 1
            s = FakeStream(self, f"sid-{self._n}")
            self.streams.append(s)
            self.events.append(("create", s._session_id))
        return s

    def release(self):
        self.released = True
        self.cancel_gate.set()


def _matching_receipt(sid: str) -> dict:
    """The canonical MATCHING native cancel receipt shape (event cancelled /
    id == captured sid / ok False) — the ONLY shape the server validates."""
    return {"event": "cancelled", "id": sid, "ok": False}


def _inject_simulated_matching_receipt(stream: FakeStream) -> None:
    """DOCUMENTED FIXTURE-ONLY SIMULATION.

    Deliberately inject a matching native receipt into the fake stream in the
    EXACT shape the canonical backend stores it (``{"sid": intent_sid,
    "receipt": {event cancelled / id == sid / ok False}}``) so the server's
    UNKNOWN-proof quarantine can be observed to end when a REAL matching
    proof arrives. This simulates the receipt a real device would produce;
    it is NOT real device qualification and proves nothing about hardware.
    """
    stream._cancel_confirm = {
        "sid": stream._session_id,
        "receipt": _matching_receipt(stream._session_id),
    }


@pytest.fixture()
def harness():
    # PER-TEST ordinary pool: no module-global executor leaking scope.
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-asr-ordinary"
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
    # All gates are opened and all tasks joined by each test BEFORE this
    # shutdown runs; shutdown therefore waits only for bookkeeping.
    pool.shutdown(wait=True)


async def _wait(predicate, timeout=5.0, msg="condition"):
    # Absolute-budget poll: no asyncio.timeout (not available on NX 3.10).
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timeout waiting for {msg}")
        await asyncio.sleep(0.01)
    return True


def _idx(events, marker) -> int:
    return events.index(marker)


# ---------------------------------------------------------------------------
# 1. normal reset closes before create ( journaled ), then disconnect ends
#    cleanly via an ACTUAL matching receipt from the real request_cancel path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normal_reset_closes_old_before_create_then_disconnect(harness):
    run, recorder = harness
    be = FakeBackend()
    recorder.log = be.events  # one combined observable journal
    ws = FakeWS([])
    task = asyncio.create_task(run(ws, be))
    await _wait(lambda: len(be.streams) >= 1)
    old = be.streams[0]
    ws.text({"command": "reset"})
    await _wait(
        lambda: any(isinstance(m, dict) and m.get("type") == "reset" for m in ws.sent)
    )
    assert old.close_calls >= 1 and old._closed
    assert len(be.streams) >= 2
    # REAL journal ordering: the old stream's close is recorded BEFORE the
    # second create (previously vacuous — the journal did not exist).
    assert _idx(be.events, ("close", old._session_id)) < _idx(
        be.events, ("create", be.streams[1]._session_id)
    )
    assert _idx(be.events, ("slot_enter", None)) < _idx(
        be.events, ("create", be.streams[0]._session_id)
    )
    # Disconnect arms a native cancel on the CURRENT stream; program the
    # receipt on the REAL request_cancel path (gate opened explicitly) so the
    # teardown proof is an actual matching receipt, not a fake hard release.
    cur = be.streams[1]
    be.cancel_gate.set()
    be.programmed_receipt = _matching_receipt(cur._session_id)
    ws.disconnect()
    await asyncio.wait_for(task, timeout=5)
    await _wait(lambda: recorder.exited >= 1)
    assert recorder.entered >= 1
    assert cur._cancel_confirm == {
        "sid": cur._session_id,
        "receipt": _matching_receipt(cur._session_id),
    }


# ---------------------------------------------------------------------------
# 2. cancel timeout + disconnect: slot held while proof UNKNOWN until a
#    SIMULATED matching receipt (documented injection) provides teardown proof
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_timeout_disconnect_holds_slot_until_matching_receipt(harness):
    run, recorder = harness
    be = FakeBackend()
    # Open the cancel gate BEFORE arming so the programmed Timeout path starts
    # immediately (no artificial 5 s thread wait inside request_cancel).
    be.cancel_gate.set()
    be.programmed_receipt = TimeoutError("fake helper timeout")
    ws = FakeWS([])
    task = asyncio.create_task(run(ws, be))
    await _wait(lambda: len(be.streams) >= 1)
    stream = be.streams[0]
    stream.refuse_close_until_released = True
    ws.text({"command": "cancel"})
    await _wait(
        lambda: any(
            isinstance(m, dict) and m.get("error") == "cancel_timeout" for m in ws.sent
        )
    )
    assert not any(
        isinstance(m, dict) and m.get("type") == "cancel_ack" for m in ws.sent
    )
    ws.disconnect()
    # Quarantine: UNKNOWN native proof (timeout receipt is not a resolution,
    # _closed refused, worker reference UNKNOWN) → slot must stay held.
    await asyncio.sleep(0.4)
    assert recorder.exited == 0, "slot released before any matching receipt!"
    assert not task.done()
    # SIMULATED (documented injection, canonical shape) matching receipt, then
    # release the refusing close and observe actual completion. This is a
    # fixture-only teardown proof, NOT real device qualification.
    _inject_simulated_matching_receipt(stream)
    be.release()
    await asyncio.wait_for(task, timeout=5)
    await _wait(lambda: recorder.exited >= 1)
    assert stream.close_calls >= 1


# ---------------------------------------------------------------------------
# 3. blocked create + handler cancel: created stream retained, closed BEFORE
#    slot release (real close journal vs observed slot exit)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancelled_create_retains_and_closes_created_stream(harness):
    run, recorder = harness
    create_gate = threading.Event()
    be = FakeBackend(create_gate=create_gate)
    recorder.log = be.events
    ws = FakeWS([])  # handler ends via cancellation, not a disconnect frame
    task = asyncio.create_task(run(ws, be))
    await asyncio.sleep(0.2)  # handler parked inside the blocked create
    task.cancel()
    # While the actual create CF must still drain (gate closed), the task is
    # NOT done and the slot is held — no premature release is possible.
    await asyncio.sleep(0.2)
    assert not task.done(), "cancelled handler finished before its CF drained!"
    assert recorder.exited == 0, "slot released while create CF still draining!"
    # NOW open the actual thread gate: the CF completes, the created stream is
    # retained (recorded INSIDE the creation callable), closed, proven, and
    # only then is the slot released.
    create_gate.set()
    await asyncio.wait_for(task, timeout=5)
    await _wait(lambda: recorder.exited >= 1)
    assert len(be.streams) >= 1
    stream = be.streams[0]
    assert stream.close_calls >= 1 and stream._closed, (
        "created stream must be closed BEFORE the slot is released"
    )
    sid = stream._session_id
    # Real journal ordering: create < close < slot_exit.
    assert _idx(be.events, ("create", sid)) < _idx(be.events, ("close", sid))
    assert _idx(be.events, ("close", sid)) < _idx(be.events, ("slot_exit", None))


# ---------------------------------------------------------------------------
# 4. invalid matching receipt (wrong sid, ok=True): no ACK, no fake free
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_matching_receipt_ok_true_no_ack_no_release(harness):
    run, recorder = harness
    be = FakeBackend()
    # Gate opened explicitly: the invalid receipt is delivered at once (no
    # artificial wait) and must be REJECTED by the server's validation.
    be.cancel_gate.set()
    be.programmed_receipt = {"event": "cancelled", "id": "whatever", "ok": True}
    ws = FakeWS([])
    task = asyncio.create_task(run(ws, be))
    await _wait(lambda: len(be.streams) >= 1)
    stream = be.streams[0]
    stream.refuse_close_until_released = True
    ws.text({"command": "cancel"})
    await _wait(
        lambda: any(
            isinstance(m, dict) and m.get("error") == "cancel_receipt_invalid"
            for m in ws.sent
        )
    )
    assert not any(
        isinstance(m, dict) and m.get("type") == "cancel_ack" for m in ws.sent
    ), "invalid receipt must never ACK"
    ws.disconnect()
    # Quarantine: the stored receipt does NOT match (ok True), close refused,
    # worker UNKNOWN → NO fake free; the slot stays held.
    await asyncio.sleep(0.4)
    assert recorder.exited == 0, "slot faked free on ok=True receipt!"
    assert not task.done()
    # Teardown only on an actual matching proof: the documented SIMULATED
    # matching receipt (canonical shape; fixture-only, NOT device
    # qualification) plus release of the refusing close.
    _inject_simulated_matching_receipt(stream)
    be.release()
    await asyncio.wait_for(task, timeout=5)
    await _wait(lambda: recorder.exited >= 1)
