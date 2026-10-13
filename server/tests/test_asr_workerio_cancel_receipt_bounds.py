"""Focused contracts for WorkerIO.cancel_and_wait late-lock + bounded receipts.

These tests are AUTHORED BUT NOT RUN by the corrective task that added them
(``UNRUN`` in the evidence). They pin three concrete properties of the
native-ASR cancel receipt path in ``voxedge.backends.jetson.worker_io``:

  1. LATE STDIN LOCK (correction 1): if ``_stdin_lock.acquire(timeout=...)``
     returns after the absolute ``deadline``, ``cancel_and_wait`` must raise
     ``TimeoutError`` with ZERO bytes written to stdin, must unregister the
     per-id receipt queue, and must leave the stdin lock released.

  2. BOUNDED RECEIPT QUEUE (correction 2): a burst of repeated matching
     ``cancelled`` terminals must NEVER block the reader thread and the
     receipt queue must retain at most ONE (the first) terminal; an unrelated
     real request must still make progress afterwards.

  3. CLOSE / EOF ON A FULL RECEIPT QUEUE (correction 2): ``close()`` and the
     reader's EOF finalizer must not block when a receipt queue is already
     full (maxsize=1) and must preserve the already-retained terminal.

The fake clock is injected only into the ``worker_io`` module namespace
(``_wio.time``); the shared stdlib ``time`` module object is NEVER mutated,
so no other test/thread observes the fake. Every helper thread is joined
with a finite timeout inside ``try/finally`` cleanup.
"""

from __future__ import annotations

import importlib
import json
import queue
import threading
import time

import pytest

_wio = importlib.import_module("voxedge.backends.jetson.worker_io")
WorkerIO = _wio.WorkerIO
TimeoutError_ = TimeoutError  # module may shadow nothing; explicit alias


# ── fake subprocess plumbing ────────────────────────────────────────────────


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            self.writes.append(s)
        return len(s)

    def flush(self) -> None:
        pass


class _FakeStdoutQueue:
    def __init__(self) -> None:
        self._q: "queue.Queue[str | None]" = queue.Queue()

    def feed(self, line: str) -> None:
        self._q.put(line if line.endswith("\n") else line + "\n")

    def eof(self) -> None:
        self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdoutQueue()


@pytest.fixture(autouse=True)
def _close_tracked_wio_instances(monkeypatch):
    """Capture every WorkerIO constructed during the test and tear it down.

    Wraps the ORIGINAL ``WorkerIO.__init__`` (constructor behavior unchanged,
    receipt behavior never faked, stdlib ``time`` never globally mutated).
    On teardown each instance gets ``wio.close()``, then this file's own
    ``_FakeStdoutQueue.eof()`` (idempotent: a second ``None`` sentinel in the
    queue is harmless), then the ACTUAL ``wio._reader_thread`` joined with a
    finite 2s budget. A reader still alive after the join FAILS the fixture
    — never silently suppressed.
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
                stdout.eof()  # double-EOF guard: extra sentinel is harmless
            except Exception as exc:  # noqa: BLE001
                failures.append(f"wio[{idx}] FakeStdout.eof() raised {exc!r}")
        thread = getattr(wio, "_reader_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                failures.append(
                    f"wio[{idx}] _reader_thread still alive after 2s join"
                )
    if failures:
        raise AssertionError("reader teardown failures: " + "; ".join(failures))


def _make_wio(proc=None):
    proc = proc if proc is not None else _FakeProc()
    wio = WorkerIO(proc, concurrency=2)
    return wio, proc


# ── deterministic source-specific clock + lock ──────────────────────────────


class _FakeClock:
    """A source-specific monotonic clock. Never touches the stdlib module."""

    def __init__(self, start: float = 1000.0) -> None:
        self._t = float(start)
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self._t

    def advance(self, delta: float) -> None:
        with self._lock:
            self._t += float(delta)


class _AdvancingLock:
    """Real Lock semantics; advances a fake clock on each acquire().

    ``seconds`` is consumed from the fake clock the first time ``acquire`` is
    called, simulating an acquire that returned only after the wall-clock
    deadline had passed. The underlying lock itself is a genuine
    ``threading.Lock`` so release/re-acquire semantics are real.
    """

    def __init__(self, clock: _FakeClock, seconds: float) -> None:
        self._clock = clock
        self._seconds = float(seconds)
        self._advance_once = True
        self._lock = threading.Lock()

    def _advance(self) -> None:
        if self._advance_once:
            self._advance_once = False
            self._clock.advance(self._seconds)

    def acquire(self, timeout=None):
        self._advance()
        if timeout is None:
            return self._lock.acquire()
        return self._lock.acquire(timeout=timeout)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


class _NoAdvanceLock:
    """Plain non-advancing wrapper used when the clock is faked globally."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def acquire(self, timeout=None):
        if timeout is None:
            return self._lock.acquire()
        return self._lock.acquire(timeout=timeout)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


# ── correction 1: late stdin-lock acquisition fails closed ──────────────────


def test_late_lock_acquisition_raises_timeout_with_zero_writes(monkeypatch):
    clock = _FakeClock(start=1000.0)
    monkeypatch.setattr(_wio, "time", clock, raising=True)
    wio, proc = _make_wio()

    try:
        # The lock acquisition consumes 0.5s of fake time; the deadline is
        # only 0.1s away, so the in-lock re-check must trip BEFORE any write.
        wio._stdin_lock = _AdvancingLock(clock, seconds=0.5)

        with pytest.raises(TimeoutError_) as excinfo:
            wio.cancel_and_wait("sid-late", 0.1)

        assert "deadline expired after acquiring" in str(excinfo.value)
        # ZERO bytes written to stdin.
        assert proc.stdin.writes == []
        # Receipt waiter unregistered (finally ran).
        assert wio._cancel_receipts == {}
        assert wio._cancel_waiter_count == 0
        # Stdin lock was released by the inner finally.
        assert wio._stdin_lock.locked() is False
        # The lock is genuinely re-acquirable after the failure.
        assert wio._stdin_lock.acquire(timeout=1.0) is True
        wio._stdin_lock.release()
    finally:
        wio.close()


# ── correction 2: repeated matching burst never blocks the reader ────────────


def test_repeated_matching_cancelled_burst_keeps_one_and_progresses(monkeypatch):
    clock = _FakeClock(start=2000.0)
    monkeypatch.setattr(_wio, "time", clock, raising=True)
    wio, proc = _make_wio()
    wio._stdin_lock = _NoAdvanceLock()

    # Register a receipt waiter directly (we exercise the reader fan-out, not
    # the full cancel_and_wait wait loop).
    receipt_q: "queue.Queue" = queue.Queue(maxsize=1)
    with wio._inflight_lock:
        wio._cancel_receipts["sid-burst"] = receipt_q
        wio._cancel_waiter_count += 1

    try:
        # Burst of repeated matching terminals. The reader must consume all of
        # them without blocking, retaining only the FIRST.
        for i in range(50):
            proc.stdout.feed(
                json.dumps(
                    {"event": "cancelled", "id": "sid-burst", "ok": False, "epoch": i}
                )
            )
        # A concurrent unrelated real request must still make progress: its
        # matching terminal must be fanned out to its inflight queue.
        real_q: "queue.Queue" = queue.Queue()
        with wio._inflight_lock:
            wio._inflight["sid-real"] = real_q
        proc.stdout.feed(json.dumps({"event": "done", "id": "sid-real"}))

        # The receipt queue holds at most one item.
        first = receipt_q.get(timeout=2.0)
        assert first["event"] == "cancelled"
        assert first["id"] == "sid-burst"
        assert receipt_q.empty() is True
        assert receipt_q.maxsize == 1

        # The reader kept going and delivered the real request's terminal.
        done = real_q.get(timeout=2.0)
        assert done == {"event": "done", "id": "sid-real"}

        # The reader thread is still alive (burst did not kill or wedge it).
        assert wio._reader_thread.is_alive() is True
    finally:
        with wio._inflight_lock:
            wio._inflight.pop("sid-real", None)
            wio._cancel_receipts.pop("sid-burst", None)
            wio._cancel_waiter_count = max(0, wio._cancel_waiter_count - 1)
        wio.close()


# ── correction 2: close() on a full receipt queue never blocks ──────────────


def test_close_on_full_receipt_queue_does_not_block(monkeypatch):
    clock = _FakeClock(start=3000.0)
    monkeypatch.setattr(_wio, "time", clock, raising=True)
    wio, proc = _make_wio()

    receipt_q: "queue.Queue" = queue.Queue(maxsize=1)
    # Full: one retained real terminal already present.
    receipt_q.put_nowait({"event": "cancelled", "id": "sid-close", "ok": False})
    with wio._inflight_lock:
        wio._cancel_receipts["sid-close"] = receipt_q
        wio._cancel_waiter_count += 1

    closer = threading.Thread(target=wio.close, daemon=True)
    try:
        closer.start()
        closer.join(timeout=2.0)
        assert closer.is_alive() is False, "close() blocked on a full receipt queue"
        # The retained first terminal survived; the dropped wake is acceptable.
        retained = receipt_q.get_nowait()
        assert retained["event"] == "cancelled"
        assert retained["id"] == "sid-close"
    finally:
        if closer.is_alive():
            closer.join(timeout=1.0)
        # close() already cleared the registry; guard the counter.
        with wio._inflight_lock:
            wio._cancel_waiter_count = max(0, wio._cancel_waiter_count - 1)


# ── correction 2: reader EOF on a full receipt queue never blocks ────────────


def test_readereof_on_full_receipt_queue_does_not_block(monkeypatch):
    clock = _FakeClock(start=4000.0)
    monkeypatch.setattr(_wio, "time", clock, raising=True)
    wio, proc = _make_wio()

    receipt_q: "queue.Queue" = queue.Queue(maxsize=1)
    with wio._inflight_lock:
        wio._cancel_receipts["sid-eof"] = receipt_q
        wio._cancel_waiter_count += 1

    try:
        # Fill the receipt queue via the reader's own fan-out (matching
        # terminal), then EOF the stdout iterator. If the EOF finalizer used a
        # blocking put on the full queue, the reader would wedge and never
        # terminate; we join the reader with a finite timeout to prove it does.
        proc.stdout.feed(json.dumps({"event": "cancelled", "id": "sid-eof", "ok": False}))
        # Wait (bounded) for the reader to fan the terminal out.
        deadline = time.monotonic() + 2.0
        while receipt_q.empty() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert receipt_q.full() is True

        proc.stdout.eof()
        wio._reader_thread.join(timeout=2.0)
        assert wio._reader_thread.is_alive() is False, (
            "reader blocked on a full receipt queue at EOF"
        )
        # Retained first terminal preserved.
        retained = receipt_q.get_nowait()
        assert retained["event"] == "cancelled"
        assert retained["id"] == "sid-eof"
    finally:
        with wio._inflight_lock:
            wio._cancel_receipts.pop("sid-eof", None)
            wio._cancel_waiter_count = max(0, wio._cancel_waiter_count - 1)
        wio.close()


# ── correction 2: bounded helper itself drops on full without raising ────────


def test_offer_receipt_is_nonblocking_and_drops_when_full():
    rq: "queue.Queue" = queue.Queue(maxsize=1)
    WorkerIO._offer_receipt(rq, {"event": "cancelled", "id": "x"})
    # Second offer must not raise and must not block; first is retained.
    WorkerIO._offer_receipt(rq, {"event": "cancelled", "id": "x"})
    assert rq.qsize() == 1
    assert rq.get_nowait() == {"event": "cancelled", "id": "x"}
