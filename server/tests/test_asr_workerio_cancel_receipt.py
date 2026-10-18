"""Focused tests for WorkerIO.cancel_and_wait (native ASR cancel receipt).

Imports the CANONICAL module ``voxedge.backends.jetson.worker_io`` directly —
never ``server.core.worker_io`` (a different copy).

Status: AUTHORED, UNRUN (static author pass only; runtime execution follows
separately).
"""

from __future__ import annotations

import io
import json
import queue
import threading
import time

import pytest

from voxedge.backends.jetson.worker_io import WorkerExitError, WorkerIO


class _FakeStdin:
    """Records every JSON line written; lets tests assert 'nothing written'."""

    def __init__(self) -> None:
        self._buf = io.StringIO()
        self.lines: list[str] = []

    def write(self, s: str) -> int:
        self._buf.write(s)
        return len(s)

    def flush(self) -> None:
        data = self._buf.getvalue()
        self._buf.seek(0)
        self._buf.truncate(0)
        for line in data.splitlines():
            if line:
                self.lines.append(line)

    @property
    def payload(self) -> str:
        return "".join(self.lines)


class _FakeStdout:
    """Iterator-of-lines backed by a queue, mimicking Popen stdout."""

    def __init__(self) -> None:
        self._q: "queue.Queue[bytes | None]" = queue.Queue()

    def feed(self, obj: dict) -> None:
        self._q.put((json.dumps(obj) + "\n").encode("utf-8"))

    def close(self) -> None:
        self._q.put(None)

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item


class FakeProc:
    """Lightweight Popen stand-in (read-only reuse of existing patterns)."""

    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout()


@pytest.fixture(autouse=True)
def _close_tracked_wio_instances(monkeypatch):
    """Capture every WorkerIO constructed during the test and tear it down.

    Wraps the ORIGINAL ``WorkerIO.__init__`` (constructor behavior unchanged,
    receipt behavior never faked). On teardown each instance gets:
    ``wio.close()``, then this file's own ``_FakeStdout.close()`` (idempotent:
    a second ``None`` sentinel in the queue is harmless), then the ACTUAL
    ``wio._reader_thread`` joined with a finite 2s budget. A reader that is
    still alive after the join FAILS the fixture — never silently suppressed.
    """
    instances: list[WorkerIO] = []
    orig_init = WorkerIO.__init__

    def _tracking_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        instances.append(self)

    monkeypatch.setattr(WorkerIO, "__init__", _tracking_init, raising=True)
    yield
    failures: list[str] = []
    for idx, wio in enumerate(instances):
        try:
            wio.close()
        except Exception as exc:  # noqa: BLE001 - recorded, not suppressed
            failures.append(f"wio[{idx}].close() raised {exc!r}")
        stdout = getattr(getattr(wio, "_proc", None), "stdout", None)
        if stdout is not None:
            try:
                stdout.close()  # double-EOF guard: extra sentinel is harmless
            except Exception as exc:  # noqa: BLE001
                failures.append(f"wio[{idx}] FakeStdout.close() raised {exc!r}")
        thread = getattr(wio, "_reader_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                failures.append(
                    f"wio[{idx}] _reader_thread still alive after 2s join"
                )
    if failures:
        raise AssertionError("reader teardown failures: " + "; ".join(failures))


def _make_wio(concurrency: int = 4) -> tuple[WorkerIO, FakeProc]:
    proc = FakeProc()
    return WorkerIO(proc, concurrency), proc


def _cancelled(sid: str, epoch: int = 7) -> dict:
    return {"event": "cancelled", "id": sid, "ok": False, "epoch": epoch}


def _waiter_thread(wio: WorkerIO, rid: str, timeout_s: float):
    """Run cancel_and_wait in a thread; return (thread, result box)."""
    box: dict = {}

    def _run() -> None:
        try:
            box["result"] = wio.cancel_and_wait(rid, timeout_s)
        except BaseException as exc:  # noqa: BLE001 - test capture
            box["error"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, box


# --------------------------------------------------------------------------
# idle cancel: receipt observed with NO inflight queue registered
# --------------------------------------------------------------------------

def test_idle_cancel_receipt_observed_without_inflight_queue():
    wio, proc = _make_wio()
    t, box = _waiter_thread(wio, "sess-idle", 5.0)

    # Wait until the waiter has registered + written its cancel line.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not proc.stdin.lines:
        time.sleep(0.005)
    assert proc.stdin.lines == [json.dumps({"type": "cancel", "id": "sess-idle"})]
    assert wio._inflight == {}, "idle session must have no inflight queue"

    proc.stdout.feed(_cancelled("sess-idle", epoch=3))
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert box.get("error") is None
    receipt = box["result"]
    assert receipt["event"] == "cancelled"
    assert receipt["id"] == "sess-idle"
    assert receipt["ok"] is False  # ok:false is EXPECTED, not an error
    assert receipt["epoch"] == 3  # actual payload preserved verbatim
    assert wio._cancel_receipts == {}
    assert wio._cancel_waiter_count == 0


# --------------------------------------------------------------------------
# active cancel: same event reaches the original consumer AND the receipt,
# without replacing the inflight queue or touching the semaphore
# --------------------------------------------------------------------------

def test_active_cancel_fans_out_without_replacing_queue_or_sem():
    wio, proc = _make_wio(concurrency=4)
    # Simulate an active consumer exactly as request() would have left it.
    consumer_q: "queue.Queue" = queue.Queue()
    with wio._inflight_lock:
        wio._inflight["sess-active"] = consumer_q
    wio._sem.acquire()  # consumer holds one slot

    t, box = _waiter_thread(wio, "sess-active", 5.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not proc.stdin.lines:
        time.sleep(0.005)

    event = _cancelled("sess-active", epoch=9)
    proc.stdout.feed(event)

    t.join(timeout=2.0)
    assert not t.is_alive()
    assert box.get("error") is None
    assert box["result"] == event

    # The ORIGINAL consumer queue got the same terminal (fan-out, not steal).
    got = consumer_q.get(timeout=1.0)
    assert got == event
    # Queue object itself was never replaced.
    with wio._inflight_lock:
        assert wio._inflight.get("sess-active") is consumer_q
    # Semaphore accounting untouched by the cancellation observer.
    assert wio._sem._value == 3  # 4 - 1 held slot
    with wio._inflight_lock:
        assert wio._cancel_receipts == {}
        assert wio._cancel_waiter_count == 0


# --------------------------------------------------------------------------
# immediate ACK race: terminal already queued before the waiter get()s
# --------------------------------------------------------------------------

def test_immediate_cancelled_event_race():
    wio, proc = _make_wio()
    t, box = _waiter_thread(wio, "sess-race", 5.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not proc.stdin.lines:
        time.sleep(0.005)
    # Feed instantly — the reader may enqueue before the waiter's get().
    proc.stdout.feed(_cancelled("sess-race"))
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert box.get("error") is None
    assert box["result"]["event"] == "cancelled"


# --------------------------------------------------------------------------
# mismatched / unsolicited terminals never satisfy the receipt
# --------------------------------------------------------------------------

def test_mismatched_terminal_is_ignored():
    wio, proc = _make_wio()
    t, box = _waiter_thread(wio, "sess-mine", 0.8)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not proc.stdin.lines:
        time.sleep(0.005)
    # Wrong id and a non-cancelled dict: neither may satisfy the receipt.
    proc.stdout.feed(_cancelled("sess-other"))
    proc.stdout.feed({"event": "done", "id": "sess-mine", "ok": True})
    time.sleep(0.05)
    assert box.get("error") is None or isinstance(box.get("error"), TimeoutError)
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert isinstance(box.get("error"), TimeoutError)
    # The waiter unregistered itself even on timeout.
    assert wio._cancel_receipts == {}
    assert wio._cancel_waiter_count == 0


# --------------------------------------------------------------------------
# timeout clears the observer and leaves other requests intact
# --------------------------------------------------------------------------

def test_timeout_clears_observer_and_leaves_other_requests_intact():
    wio, proc = _make_wio(concurrency=4)

    # A normal in-flight request that must survive the failed cancel.
    survivor_q: "queue.Queue" = queue.Queue()
    with wio._inflight_lock:
        wio._inflight["sess-keep"] = survivor_q

    t, box = _waiter_thread(wio, "sess-gone", 0.3)
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert isinstance(box.get("error"), TimeoutError)
    assert wio._cancel_receipts == {}
    assert wio._cancel_waiter_count == 0

    # The surviving request still receives events and can still be cancelled
    # via the legacy best-effort cancel() (unchanged behavior).
    proc.stdout.feed({"event": "partial", "id": "sess-keep"})
    assert survivor_q.get(timeout=1.0)["event"] == "partial"
    wio.cancel("sess-keep")
    assert proc.stdin.lines[-1] == json.dumps({"type": "cancel", "id": "sess-keep"})


# --------------------------------------------------------------------------
# worker exit / close wakes pending receipts with WorkerExitError
# --------------------------------------------------------------------------

def test_worker_exit_wakes_pending_receipt():
    wio, proc = _make_wio()
    t, box = _waiter_thread(wio, "sess-exit", 30.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not proc.stdin.lines:
        time.sleep(0.005)
    wio.close()
    t.join(timeout=2.0)
    assert not t.is_alive()
    err = box.get("error")
    assert isinstance(err, WorkerExitError)
    assert wio._cancel_receipts == {}
    assert wio._cancel_waiter_count == 0


def test_reader_thread_eof_wakes_pending_receipt():
    wio, proc = _make_wio()
    t, box = _waiter_thread(wio, "sess-eof", 30.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not proc.stdin.lines:
        time.sleep(0.005)
    proc.stdout.close()  # reader loop ends -> finally wakes receipts
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert isinstance(box.get("error"), WorkerExitError)


# --------------------------------------------------------------------------
# duplicate / bounded admission, all rejected BEFORE any write
# --------------------------------------------------------------------------

def test_duplicate_waiter_rejected_before_write():
    wio, proc = _make_wio()
    t, box = _waiter_thread(wio, "sess-dup", 5.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and "sess-dup" not in wio._cancel_receipts:
        time.sleep(0.005)

    with pytest.raises(RuntimeError):
        wio.cancel_and_wait("sess-dup", 5.0)
    # Only the FIRST cancel line was written.
    assert proc.stdin.lines == [json.dumps({"type": "cancel", "id": "sess-dup"})]

    proc.stdout.feed(_cancelled("sess-dup"))
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_waiter_count_bounded_by_configured_concurrency():
    wio, proc = _make_wio(concurrency=2)
    threads = []
    boxes = []
    for i in range(2):
        t, box = _waiter_thread(wio, f"sess-cap-{i}", 5.0)
        threads.append(t)
        boxes.append(box)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and wio._cancel_waiter_count < 2:
        time.sleep(0.005)
    assert wio._cancel_waiter_count == 2

    with pytest.raises(RuntimeError):
        wio.cancel_and_wait("sess-cap-overflow", 5.0)
    assert "sess-cap-overflow" not in wio._cancel_receipts

    for i, (t, _box) in enumerate(zip(threads, boxes)):
        proc.stdout.feed(_cancelled(f"sess-cap-{i}"))
        t.join(timeout=2.0)
        assert not t.is_alive()


# --------------------------------------------------------------------------
# invalid / expired inputs: ValueError before ANY write
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rid,timeout_s",
    [
        ("", 1.0),
        (None, 1.0),
        (123, 1.0),
        ("sess-x", 0),
        ("sess-x", -1.0),
        ("sess-x", True),  # bool must be refused
        ("sess-x", float("nan")),
        ("sess-x", float("inf")),
        ("sess-x", "2"),  # not a number
    ],
)
def test_invalid_inputs_raise_before_write(rid, timeout_s):
    wio, proc = _make_wio()
    with pytest.raises(ValueError):
        wio.cancel_and_wait(rid, timeout_s)
    assert proc.stdin.lines == [], "nothing may be written for invalid inputs"
    assert wio._cancel_receipts == {}
    assert wio._cancel_waiter_count == 0


# --------------------------------------------------------------------------
# no budget renewal / no floors: late receipt fails
# --------------------------------------------------------------------------

def test_late_receipt_does_not_satisfy_wait():
    wio, proc = _make_wio()
    t, box = _waiter_thread(wio, "sess-late", 0.25)
    # Let the timeout fire, THEN deliver the receipt: it must arrive too late
    # and must not retroactively satisfy anyone.
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert isinstance(box.get("error"), TimeoutError)
    proc.stdout.feed(_cancelled("sess-late"))
    time.sleep(0.05)
    assert box.get("error") is not None
    # Waiter still unregistered despite the late event.
    assert wio._cancel_receipts == {}
    assert wio._cancel_waiter_count == 0


def test_absolutely_no_budget_renewal_or_floor():
    """A single tiny budget must not be stretched: deadline is absolute."""
    wio, proc = _make_wio()
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        wio.cancel_and_wait("sess-nofloor", 0.05)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, "wait must honor the absolute deadline, not a renewed one"
    assert wio._cancel_receipts == {}
    assert wio._cancel_waiter_count == 0


# --------------------------------------------------------------------------
# telemetry non-interference
# --------------------------------------------------------------------------

def test_telemetry_never_touches_receipt_queues():
    wio, proc = _make_wio()
    t, box = _waiter_thread(wio, "sess-tel", 0.5)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not proc.stdin.lines:
        time.sleep(0.005)
    # Canonical telemetry line carrying an id: must NOT land in the receipt.
    proc.stdout.feed({"type": "asr_ifb_health", "id": "sess-tel", "healthy": True})
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert isinstance(box.get("error"), TimeoutError)
    assert wio._cancel_receipts == {}
    assert wio._cancel_waiter_count == 0


# --------------------------------------------------------------------------
# legacy cancel() behavior/counters unchanged
# --------------------------------------------------------------------------

def test_legacy_cancel_unchanged_and_counter_increments():
    wio, proc = _make_wio()
    before = WorkerIO._cancel_count
    wio.cancel("sess-legacy")
    assert proc.stdin.lines == [json.dumps({"type": "cancel", "id": "sess-legacy"})]
    assert WorkerIO._cancel_count == before + 1
    # cancel_and_wait must NOT bump the legacy counter (isolated method).
    t, box = _waiter_thread(wio, "sess-nocount", 0.2)
    t.join(timeout=3.0)
    count_after_waiter = WorkerIO._cancel_count
    assert count_after_waiter == before + 1
