"""v011 regression: disconnect must RETAIN (shield + await), never CANCEL,
the native control task so a matching receipt is observed and the slot is
released WITHOUT any injected receipt.

This test AST-extracts the ACTUAL ``_asr_stream_backend`` coroutine and the
ACTUAL ``_AsrSlotJobs`` class from ``server/main.py`` (no mirrored pipeline),
execs them against a per-test ordinary pool and an INSTRUMENTED dedicated
control pool (the actual control executor), and reproduces the exact
disconnect sequence the corrected awaited probe observed:

  create sid-1 -> reset -> create sid-2 -> disconnect while the control
  request_cancel is submitted on the dedicated one-thread control pool.

Before the fix, teardown ``_ct.cancel()``-ed the control wrapper before the
helper's ``request_cancel`` ever reached the pool, so no ``cf_submit`` /
``rc_start`` was observed, ``_cancel_confirm`` stayed None, the native-faithful
``close`` REFUSED (armed but unconfirmed) and the slot was held until a
fixture-only injected receipt. After the fix the SAME retained control task is
shielded + awaited, the real matching receipt is stored, close succeeds and the
slot exits with NO injection.

Assertions (all from the real runtime journal / instrumented pool, no sleeps
used to force a pass):
  * the control CF is actually submitted AND started on the dedicated pool;
  * the real ``request_cancel`` (matching receipt) stores ``_cancel_confirm``;
  * the retained slot exits with NO injected receipt;
  * NO ``cancel_ack`` is sent after the disconnect (client_gone);
  * NO live control CF and no leaked pool threads at the end;
  * the native-faithful fake close REFUSES while armed-but-unconfirmed
    (fidelity guard, checked independently).
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import pathlib
import threading
import time

import pytest

logging.basicConfig(level=logging.CRITICAL)

_MAIN = pathlib.Path(__file__).resolve().parents[1] / "main.py"

# ---------------------------------------------------------------------------
# AST extraction of the ACTUAL server code (same approach as the accepted
# native-control pipeline harness + the corrected awaited probe).
# ---------------------------------------------------------------------------


def _extract_source(names: set[str]) -> str:
    text = _MAIN.read_text()
    tree = ast.parse(text)
    chunks = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in names:
            chunks.append(ast.get_source_segment(text, node))
    assert len(chunks) == len(names), (names,)
    return "\n".join(chunks)


class _RecorderSlot:
    """Observable no-op slot: records enter/exit (and mirrors into the backend
    journal so ordering across backend and slot events is one list)."""

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


class _FakeDiarMod:
    @staticmethod
    def make_session_diarizer():
        return None


def _build_module(pool) -> tuple:
    recorder = _RecorderSlot()
    env = {
        "asyncio": asyncio,
        "json": json,
        "logging": logging,
        "threading": threading,
        "concurrent": concurrent.futures,
        "contextlib": contextlib,
        "logger": logging.getLogger("test.asr.native.disconnect"),
        "_AsrSlotJobs": None,
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
        "ThreadPoolExecutor": InstrumentedControlPool,
    }
    # ACTUAL helper functions from server/main.py, AST-extracted + exec'd into
    # the same namespace (no mirrored copies).
    helpers = _extract_source(
        {"_split_vad_block", "_unpack_finalize_result", "_augment_final_payload"}
    )
    src = helpers + "\n" + _extract_source(
        {"_AsrSlotJobs", "_asr_stream_backend"}
    )
    exec(compile(src, str(_MAIN), "exec"), env)
    return env["_asr_stream_backend"], recorder


# ---------------------------------------------------------------------------
# Instrumented dedicated control pool + backend/stream fakes
# ---------------------------------------------------------------------------

_JOURNAL: list = []
_JLOCK = threading.Lock()
_T0 = time.monotonic()
_CF_OPEN = 0
_CF_LOCK = threading.Lock()


def _j(ev, detail=None):
    with _JLOCK:
        _JOURNAL.append(
            (round(time.monotonic() - _T0, 4), threading.current_thread().name, ev, detail)
        )


def _cf_open() -> int:
    with _CF_LOCK:
        return _CF_OPEN


class InstrumentedControlPool(concurrent.futures.ThreadPoolExecutor):
    """The pipeline's OWN dedicated control pool, instrumented to prove the
    control CF is submitted and actually STARTS (not cancelled before run)."""

    def submit(self, fn, *a, **kw):
        global _CF_OPEN
        _j("cf_submit", {"fn": getattr(fn, "__name__", repr(fn))})
        fut = super().submit(fn, *a, **kw)
        with _CF_LOCK:
            _CF_OPEN += 1

        def _cb(f):
            global _CF_OPEN
            with _CF_LOCK:
                _CF_OPEN -= 1
            if f.cancelled():
                _j("cf_CANCELLED_before_run", {})
            else:
                _j("cf_done", {"error": repr(f.exception()) if f.exception() else None})

        fut.add_done_callback(_cb)
        return fut


class FakeWS:
    def __init__(self, script):
        self._q: asyncio.Queue = asyncio.Queue()
        for item in script:
            self._q.put_nowait(item)
        self.sent: list = []

    async def receive(self):
        return await self._q.get()

    async def send_json(self, payload):
        self.sent.append(payload)

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    def text(self, obj):
        self._q.put_nowait({"type": "websocket.receive", "text": json.dumps(obj)})

    def disconnect(self):
        self._q.put_nowait({"type": "websocket.disconnect"})


class FakeStream:
    """Canonical-shape native stream. ``close`` is native-FAITHFUL: it REFUSES
    (does not mark closed) while a cancel intent is armed but unconfirmed."""

    def __init__(self, backend, sid: str):
        self._backend = backend
        self._session_id = sid
        self._closed = False
        self._cancel_intent = None
        self._cancel_confirm = None
        self._cancel_exit = False
        self.close_calls = 0
        self.waveform_gate = None  # optional Event: blocks the ordinary lane

    def arm_cancel(self):
        self._cancel_intent = {"sid": self._session_id}
        return self._session_id

    def request_cancel(self, timeout_s: float):
        _j("rc_start", {"sid": self._session_id, "timeout": timeout_s})
        ev = self._backend.cancel_gate.wait(timeout=5)
        if not ev:
            raise TimeoutError("fake cancel helper timeout")
        receipt = self._backend.programmed_receipt
        if receipt is None:
            raise TimeoutError("fake cancel helper timeout")
        if isinstance(receipt, BaseException):
            raise receipt
        # Native fidelity: a confirmation is stored ONLY for a receipt that
        # actually matches THIS stream (event=cancelled, id=this sid,
        # ok=False). A mismatched receipt must NOT fake-confirm the stream.
        if (
            isinstance(receipt, dict)
            and receipt.get("event") == "cancelled"
            and receipt.get("id") == self._session_id
            and receipt.get("ok") is False
        ):
            self._cancel_confirm = {"sid": self._session_id, "receipt": receipt}
        else:
            _j("rc_receipt_ignored_mismatch", {"sid": self._session_id})
        _j("rc_end", {"sid": self._session_id, "confirm": self._cancel_confirm})
        return receipt

    def accept_waveform(self, sample_rate, samples):
        if self.waveform_gate is not None:
            _j("waveform_blocked_enter", {"sid": self._session_id})
            self.waveform_gate.wait(timeout=5)
            _j("waveform_blocked_exit", {"sid": self._session_id})

    def prepare_finalize(self):
        pass

    def finalize(self):
        self.close()
        return "", None

    def get_partial(self):
        return ("", False)

    def close(self):
        self.close_calls += 1
        self._backend.events.append(("close", self._session_id))
        sid = self._session_id
        if self._cancel_confirm is not None and self._cancel_confirm.get("sid") == sid:
            self._closed = True
            _j("close_ok_confirmed", {"sid": sid})
            return
        if self._cancel_intent is not None and not self._cancel_exit:
            _j("close_refused_armed_unconfirmed", {"sid": sid})
            return
        self._closed = True
        _j("close_ok_unarmed", {"sid": sid})


class FakeBackend:
    name = "trt_edgellm"

    def __init__(self, create_gate=None):
        self._worker = None  # UNKNOWN reference (never a fake exit)
        self.streams: list = []
        self.events: list = []
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
            s = FakeStream(self, f"sid-{self._n}")
            self.streams.append(s)
            self.events.append(("create", s._session_id))
        _j("create", {"sid": s._session_id})
        return s

    def release(self):
        self.cancel_gate.set()


def _matching_receipt(sid: str) -> dict:
    return {"event": "cancelled", "id": sid, "ok": False}


# ---------------------------------------------------------------------------
# Bounded teardown / journal helpers (fixture-only; never touch server code)
# ---------------------------------------------------------------------------

_JOURNAL_DIR = pathlib.Path("/tmp/slv-v011-native-control-pool-scope-fix")


def _save_journal(name: str, original_input_sha: str, failed: bool,
                  error: str | None = None) -> None:
    """Persist the raw journal even when the test failed."""
    _JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    with _JLOCK:
        entries = list(_JOURNAL)
    path = _JOURNAL_DIR / f"journal.{name}.json"
    path.write_text(json.dumps({
        "test": name,
        "original_input_sha256": original_input_sha,
        "failed": failed,
        "error": error,
        "cf_open_at_end": _cf_open(),
        "entries": entries,
    }, indent=2) + "\n")


async def _await_task_bounded(task: asyncio.Task, timeout: float) -> bool:
    """Observe task done WITHOUT cancel-and-indefinite-wait.

    Returns True iff the task completed within the deadline. Never cancels the
    task and never awaits it unboundedly: ``asyncio.wait`` with an absolute
    timeout observes completion and returns whatever is still pending.
    """
    done, pending = await asyncio.wait({task}, timeout=timeout)
    return bool(done) and task in done


def _release_fake_gates(be: "FakeBackend") -> None:
    """Release the ACTUAL fake gates so any worker thread can return."""
    be.cancel_gate.set()
    for s in be.streams:
        if s.waveform_gate is not None:
            s.waveform_gate.set()


async def _bounded_cleanup(task: asyncio.Task, be: "FakeBackend",
                           ordinary_pool: concurrent.futures.ThreadPoolExecutor,
                           timeout: float = 2.0) -> None:
    """Fixture-only bounded cleanup, preserving the original failure.

    Always releases the real fake gates. Waits a BOUNDED time for the task to
    finish before cancelling it, then waits a further bounded time for the
    cancellation. Actual pool threads are joined only after we know all owned
    futures are done; otherwise shutdown is non-blocking (no unbounded
    failure shutdown).
    """
    _release_fake_gates(be)
    if not task.done():
        done = await _await_task_bounded(task, timeout)
        if not done and not task.done():
            task.cancel()
            await _await_task_bounded(task, timeout)
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass
    # Join pool threads only when nothing is owned/live; otherwise do NOT
    # block (threads stay owned by the retained futures, not by this fixture).
    if _cf_open() == 0:
        ordinary_pool.shutdown(wait=True)
    else:
        ordinary_pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Fidelity guard: the native-faithful close REFUSES armed-but-unconfirmed
# ---------------------------------------------------------------------------

def test_native_faithful_close_refuses_armed_unconfirmed():
    be = FakeBackend()
    be.cancel_gate.set()
    s = be.create_stream()
    # Matching receipt for the SID this stream actually created (NOT sid-x).
    be.programmed_receipt = _matching_receipt(s._session_id)
    s.arm_cancel()
    s.close()  # armed, never confirmed
    assert s.close_calls == 1 and s._closed is False, (
        "fake close must REFUSE while armed-but-unconfirmed (native fidelity)"
    )
    # Once a REAL matching request_cancel confirms, the same close succeeds.
    s.request_cancel(1.0)
    s.close()
    assert s._closed is True


# ---------------------------------------------------------------------------
# MAIN regression: disconnect retains control task -> real receipt, slot
# exits, no ACK after client_gone, no live CF/threads/pending tasks.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disconnect_retains_control_task_no_injection():
    global _JOURNAL
    _JOURNAL = []
    main_sha = hashlib.sha256(_MAIN.read_bytes()).hexdigest()
    _j("input_hash", {"main": main_sha})

    ordinary_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-asr-ordinary"
    )
    factory, recorder = _build_module(ordinary_pool)
    be = FakeBackend()
    recorder.log = be.events
    ws = FakeWS([])

    task = asyncio.create_task(
        factory(
            ws=ws, asr_be=be, language="auto", sample_rate=16000,
            vad_session=None, punct_on=False, spk_on=False, diarize_on=False,
            per_utterance_slot=False,
        )
    )

    async def _wait(pred, timeout=5.0, msg="condition"):
        deadline = time.monotonic() + timeout
        while not pred():
            if time.monotonic() > deadline:
                raise AssertionError(f"timeout waiting for {msg}")
            await asyncio.sleep(0.01)

    failed = False
    error: str | None = None
    try:
        await _wait(lambda: len(be.streams) >= 1, msg="first create")
        old = be.streams[0]
        ws.text({"command": "reset"})
        await _wait(
            lambda: any(isinstance(m, dict) and m.get("type") == "reset" for m in ws.sent),
            msg="reset ack",
        )
        assert old.close_calls >= 1 and old._closed
        assert len(be.streams) >= 2
        cur = be.streams[1]

        # Program the MATCHING receipt and open the cancel gate BEFORE disconnect,
        # so the ONLY way the control CF can fail to resolve is teardown cancelling
        # the wrapper before the CF starts (the exact pre-fix bug).
        be.cancel_gate.set()
        be.programmed_receipt = _matching_receipt(cur._session_id)
        _j("gate_opened_receipt_programmed", {"sid": cur._session_id})

        ws.disconnect()
        # Bounded observation: observe done WITHOUT cancelling the task and
        # WITHOUT an unbounded await.
        done = await _await_task_bounded(task, 5.0)
        assert done, "pipeline task did not finish (bounded observation)"
        await _wait(lambda: recorder.exited >= 1, msg="slot exit")

        # ---- assertions from real runtime evidence ----
        # 1) control CF actually submitted AND started on the dedicated pool.
        assert any(e[2] == "cf_submit" for e in _JOURNAL), "control CF never submitted!"
        assert any(e[2] == "rc_start" for e in _JOURNAL), "control request_cancel never started!"
        assert not any(e[2] == "cf_CANCELLED_before_run" for e in _JOURNAL), (
            "control CF was cancelled before it could run (pre-fix bug)"
        )
        # 2) the REAL request_cancel stored the matching confirmation (no injection).
        assert cur._cancel_confirm == {
            "sid": cur._session_id,
            "receipt": _matching_receipt(cur._session_id),
        }, f"real matching receipt not stored: {cur._cancel_confirm!r}"
        assert not any(e[2] == "injected_simulated_matching_receipt" for e in _JOURNAL)
        # 3) native-faithful close SUCCEEDED because the receipt was confirmed.
        assert cur._closed is True and cur.close_calls >= 1
        assert not any(e[2] == "close_refused_armed_unconfirmed" for e in _JOURNAL), (
            "close refused armed-unconfirmed: control task was not retained/awaited"
        )
        # 4) NO cancel_ack after disconnect (client_gone).
        assert not any(
            isinstance(m, dict) and m.get("type") == "cancel_ack" for m in ws.sent
        ), "cancel_ack must never be sent after disconnect"
        # 5) slot released and no live control CF.
        assert recorder.entered >= 1 and recorder.exited >= 1
        assert _cf_open() == 0, f"live control CF(s) remain: {_cf_open()}"
        # ordering: slot_enter before first create, close before slot_exit.
        assert be.events.index(("slot_enter", None)) < be.events.index(
            ("create", old._session_id)
        )
        assert be.events.index(("close", cur._session_id)) < be.events.index(
            ("slot_exit", None)
        )
    except BaseException as exc:  # noqa: BLE001 - record then re-raise
        failed = True
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        await _bounded_cleanup(task, be, ordinary_pool)
        _save_journal(
            "disconnect_retains_control_task_no_injection", main_sha, failed, error
        )


# ---------------------------------------------------------------------------
# Blocked ordinary lane during disconnect: the DEDICATED control CF must
# submit/start and confirm while the ordinary lane is still blocked inside the
# backend, and the slot must remain held until the actual ordinary CF (the
# blocked accept_waveform) finally completes.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blocked_ordinary_lane_control_starts_before_release():
    global _JOURNAL
    _JOURNAL = []
    ordinary_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-asr-ordinary-blocked"
    )
    factory, recorder = _build_module(ordinary_pool)
    be = FakeBackend()
    recorder.log = be.events
    ws = FakeWS([])

    task = asyncio.create_task(
        factory(
            ws=ws, asr_be=be, language="auto", sample_rate=16000,
            vad_session=None, punct_on=False, spk_on=False, diarize_on=False,
            per_utterance_slot=False,
        )
    )

    async def _wait(pred, timeout=5.0, msg="condition"):
        deadline = time.monotonic() + timeout
        while not pred():
            if time.monotonic() > deadline:
                raise AssertionError(f"timeout waiting for {msg}")
            await asyncio.sleep(0.01)

    await _wait(lambda: len(be.streams) >= 1, msg="create")
    cur = be.streams[0]
    # Block the ONE ordinary worker inside the backend; any further ordinary
    # job (drain) cannot proceed until this is released.
    gate = threading.Event()
    cur.waveform_gate = gate
    be.cancel_gate.set()
    be.programmed_receipt = _matching_receipt(cur._session_id)

    failed = False
    error: str | None = None
    main_sha = hashlib.sha256(_MAIN.read_bytes()).hexdigest()
    try:
        # Enqueue audio so the ordinary lane enters accept_waveform and blocks.
        ws._q.put_nowait({"type": "websocket.receive", "bytes": b"\x00\x00" * 64})
        await _wait(
            lambda: any(e[2] == "waveform_blocked_enter" for e in _JOURNAL),
            msg="ordinary lane blocked inside accept_waveform",
        )

        ws.disconnect()
        # The dedicated control CF must start and confirm WHILE the ordinary lane
        # is still blocked, and the slot must NOT be released yet.
        await _wait(
            lambda: any(e[2] == "rc_end" for e in _JOURNAL),
            msg="control confirm on dedicated pool while ordinary blocked",
        )
        # Bounded observation (NOT a pass-forcing sleep): confirm the slot did
        # not exit while the ordinary CF is still blocked.
        assert not recorder.exited, (
            "slot released while the actual ordinary CF was still blocked"
        )
        assert any(e[2] == "waveform_blocked_enter" for e in _JOURNAL)
        assert not any(e[2] == "waveform_blocked_exit" for e in _JOURNAL)

        # Release the ordinary lane: the real CF completes, then the slot exits.
        gate.set()
        done = await _await_task_bounded(task, 5.0)
        assert done, "pipeline task did not finish (bounded observation)"
        await _wait(lambda: recorder.exited >= 1, msg="slot exit after release")
        assert any(e[2] == "waveform_blocked_exit" for e in _JOURNAL)
        assert not any(
            isinstance(m, dict) and m.get("type") == "cancel_ack" for m in ws.sent
        ), "cancel_ack must never be sent after disconnect"
        assert _cf_open() == 0, f"live control CF(s) remain: {_cf_open()}"
    except BaseException as exc:  # noqa: BLE001 - record then re-raise
        failed = True
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not gate.is_set():
            gate.set()
        await _bounded_cleanup(task, be, ordinary_pool)
        _save_journal(
            "blocked_ordinary_lane_control_starts_before_release",
            main_sha, failed, error,
        )
