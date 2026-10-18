"""Transport-foundation tests for ``TRTEdgeLLMASRBackend._worker_cancel_and_wait``.

Status: AUTHORED, UNRUN (static author pass only — a root reviewer must run these
before any CPU run; see EVIDENCE.md).

What this file pins (transport ONLY — NOT stream state, NOT the legacy
``cancel_and_finalize`` path, NOT a consumer cancellation carve-out):

  1. The helper returns the ACTUAL matching native receipt dict verbatim
     (``id`` / ``epoch`` / ``ok=false``) received from a real WorkerIO.
  2. An ordinary ``_worker_request`` consumer blocked on another thread keeps
     its own semaphore/queue and the helper's cancel still receives its
     receipt — the helper does NOT hold ``self._worker_lock`` across the
     blocking WIO wait. (The backend's normal ``ok=false`` request handling is
     deliberately left untouched; this pins the lock/transport property, not a
     backend protocol change.)
  3. A ``TimeoutError`` from WorkerIO leaves the worker healthy: no
     ``_clear_worker_if_current``, no respawn (``_ensure_worker`` is booby
     trapped), and the captured pair is retained.
  4. Missing / failed / live-without-WorkerIO pairs reject with the existing
     ``WorkerExitError`` and write NOTHING (no respawn / no stdin bytes).
  5. Invalid ``sid`` / ``timeout_s`` reject with ``ValueError`` BEFORE any
     write or worker lookup.
  6. A genuine worker EOF maps to the backend ``WorkerExitError`` and clears
     ONLY the current captured worker identity (compare-and-handle).

The canonical ``voxedge.backends.jetson.worker_io.WorkerIO`` is imported via
``importlib`` (same convention as the sibling cancel-receipt suites) because
this repo is not the package root at collection time. Real stdlib Python child
processes are used as the scripted worker over genuine OS pipes — the cancel
receipt is NEVER faked at the WorkerIO layer. Every spawned child is a
recognized short-lived Python process and is cleaned up by closing its stdin
(graceful) and a finite ``wait`` — NEVER ``SIGKILL`` — with a finite reader
join.
"""

from __future__ import annotations

import importlib
import json
import select
import subprocess
import sys
import threading
import time

import pytest

_wio_mod = importlib.import_module("voxedge.backends.jetson.worker_io")
WorkerIO = _wio_mod.WorkerIO
WorkerExitError = _wio_mod.WorkerExitError

_asr_mod = importlib.import_module("voxedge.backends.jetson.trt_edge_llm_asr")
TRTEdgeLLMASRBackend = _asr_mod.TRTEdgeLLMASRBackend
BackendWorkerExitError = _asr_mod.WorkerExitError


# ---------------------------------------------------------------------------
# Scripted stdlib Python child — a REAL subprocess over REAL OS pipes.
# ---------------------------------------------------------------------------

# Reads one JSON object per line from stdin and replies on stdout:
#   * {"type": "cancel", "id": <sid>}  -> {"event":"cancelled","id":sid,
#                                          "ok":false,"epoch":<n>}  (the native
#                                          cancelled terminal shape)
#   * anything carrying an "id"       -> {"event":"echo_ack","id":sid,
#                                          "ok":true} (ordinary request reply)
# It exits at EOF and NEVER loops unbounded: a fixed readline loop only.
_CHILD_SOURCE = r"""
import json
import sys

EPOCH = 7
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        obj = json.loads(raw)
    except Exception:
        continue
    if not isinstance(obj, dict):
        continue
    if obj.get("type") == "cancel":
        sid = obj.get("id")
        sys.stdout.write(json.dumps(
            {"event": "cancelled", "id": sid, "ok": False, "epoch": EPOCH}
        ) + "\n")
        sys.stdout.flush()
        continue
    sid = obj.get("id")
    if sid is not None:
        sys.stdout.write(json.dumps(
            {"event": "echo_ack", "id": sid, "ok": True}
        ) + "\n")
        sys.stdout.flush()
"""


# Opt-in "hold_ordinary" scripted mode: the child records the ordinary request
# ``ordinary-1`` as the SINGLE bounded pending id and does NOT reply to it until
# a ``cancel`` input is received. Only then does it emit the real native
# ``cancelled`` payload followed by the pending ordinary ``echo_ack`` — so the
# ordinary response is causally AFTER the cancel input. This guarantees genuine
# overlap at the WorkerIO layer (no loose poll/sleep assumption). All other
# fixture default modes are unchanged.
_HOLD_ORDINARY_CHILD_SOURCE = r"""
import json
import sys

EPOCH = 7
pending = None
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        obj = json.loads(raw)
    except Exception:
        continue
    if not isinstance(obj, dict):
        continue
    if obj.get("type") == "cancel":
        sid = obj.get("id")
        sys.stdout.write(json.dumps(
            {"event": "cancelled", "id": sid, "ok": False, "epoch": EPOCH}
        ) + "\n")
        sys.stdout.flush()
        if pending is not None:
            sys.stdout.write(json.dumps(
                {"event": "echo_ack", "id": pending, "ok": True}
            ) + "\n")
            sys.stdout.flush()
            pending = None
        continue
    sid = obj.get("id")
    if sid == "ordinary-1":
        # hold_ordinary: bounded single pending request, no reply yet.
        pending = sid
        # Wire barrier: prove this child actually PROCESSED (read) and HELD
        # the ordinary request. stderr is a side channel only — no receipt,
        # no terminal, no stdout traffic.
        sys.stderr.write("ORDINARY_HELD\n")
        sys.stderr.flush()
        continue
    if sid is not None:
        sys.stdout.write(json.dumps(
            {"event": "echo_ack", "id": sid, "ok": True}
        ) + "\n")
        sys.stdout.flush()
"""


class _Child:
    """Owns one scripted Python child + its canonical WorkerIO.

    Teardown is GRACEFUL ONLY: close stdin (child observes EOF and returns),
    bounded ``wait``, then ``terminate`` only if the graceful close did not
    reap it in budget — never ``SIGKILL``/``killpg``. The WorkerIO reader
    thread is joined with a finite budget and a still-alive thread FAILS the
    fixture (never silently suppressed).
    """

    def __init__(self, concurrency: int = 4, source: str = _CHILD_SOURCE) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.wio = WorkerIO(self.proc, concurrency)

    def close(self) -> list[str]:
        problems: list[str] = []
        try:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception as exc:  # noqa: BLE001 - recorded, not suppressed
            problems.append(f"stdin.close() raised {exc!r}")
        try:
            self.wio.close()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"wio.close() raised {exc!r}")
        # Graceful first: EOF should let the child return on its own.
        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            # Bounded escalation: polite TERM (never SIGKILL).
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                problems.append("child did not exit after terminate()")
        for stream in (self.proc.stdout, self.proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"stream.close() raised {exc!r}")
        thread = getattr(self.wio, "_reader_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
            if thread.is_alive():
                problems.append("WorkerIO _reader_thread still alive after 3s join")
        return problems


@pytest.fixture
def make_child():
    children: list[_Child] = []

    def _factory(concurrency: int = 4, source: str = _CHILD_SOURCE) -> _Child:
        c = _Child(concurrency=concurrency, source=source)
        children.append(c)
        return c

    yield _factory

    problems: list[str] = []
    for idx, c in enumerate(children):
        for p in c.close():
            problems.append(f"child[{idx}]: {p}")
    if problems:
        raise AssertionError("child teardown failures: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# Minimal backend construction — ``__new__`` + only the fields the helper
# touches, so no model preload / worker launch is triggered.
# ---------------------------------------------------------------------------

def _make_backend(child: _Child):
    backend = TRTEdgeLLMASRBackend.__new__(TRTEdgeLLMASRBackend)
    backend._worker_lock = threading.Lock()
    backend._worker = child.proc
    backend._wio = child.wio
    backend._worker_failed = False
    backend._worker_failed_reason = None
    backend._worker_stderr_tail = _deque80()
    # BOOBY TRAP: the helper must NEVER launch/preload. Any call that reaches
    # the launch site is a hard failure, not a silent spawn.
    def _forbidden(*_a, **_k):
        raise AssertionError("_ensure_worker must NOT be called by the helper")

    backend._ensure_worker = _forbidden  # type: ignore[method-assign]
    return backend


def _deque80():
    from collections import deque

    return deque(maxlen=80)


# ---------------------------------------------------------------------------
# 1. ACTUAL matching receipt returned verbatim.
# ---------------------------------------------------------------------------

def test_cancel_returns_actual_matching_receipt_verbatim(make_child):
    child = make_child()
    backend = _make_backend(child)

    receipt = backend._worker_cancel_and_wait("sess-abc", 5.0)

    # Verbatim native shape from the CHILD (never a fabricated ACK).
    assert receipt == {"event": "cancelled", "id": "sess-abc", "ok": False, "epoch": 7}
    assert receipt["id"] == "sess-abc"
    assert receipt["ok"] is False


# ---------------------------------------------------------------------------
# 2. Ordinary request consumer + helper cancel concurrency (backend lock).
# ---------------------------------------------------------------------------

def test_cancel_receives_while_ordinary_request_consumer_blocked(make_child):
    """An ACTUAL WorkerIO-level ordinary consumer on another thread stays
    blocked mid-request and the helper's cancel still receives its receipt.

    NOTE (doc accuracy): this exercises the ordinary consumer at the REAL
    ``WorkerIO.request`` level (request generator + ``_inflight`` registration
    under the real ``_inflight_lock``); the backend's ordinary
    ``_worker_request`` path itself is deliberately NOT called here. The
    transport property pinned: the helper does not hold ``self._worker_lock``
    across the blocking WIO wait, so an ordinary in-flight request and the
    cancel genuinely overlap at the WIO layer.

    Causal overlap is guaranteed by the child's opt-in ``hold_ordinary`` mode:
    the child withholds the ``ordinary-1`` reply until AFTER it has emitted the
    real native ``cancelled`` payload for ``sess-concurrent``. No poll/sleep
    assumption, no mocked ACK/reader/semaphore stubs.
    """
    child = make_child(concurrency=4, source=_HOLD_ORDINARY_CHILD_SOURCE)
    backend = _make_backend(child)

    started = threading.Event()
    box: dict = {}

    def _ordinary_blocked():
        gen = child.wio.request({"event": "begin", "id": "ordinary-1"})
        started.set()
        try:
            # First (and only) event — the withheld echo_ack. The generator
            # would normally continue until done/cancelled, but the backend's
            # real path breaks after the first event, so we observe exactly one
            # event and keep the request "in flight" at the WIO layer until
            # then.
            for ev in gen:
                box["ordinary"] = ev
                break
        finally:
            gen.close()

    t = threading.Thread(target=_ordinary_blocked, daemon=True)
    t.start()
    assert started.wait(timeout=5.0), "ordinary consumer thread did not start"

    # Finite MONOTONIC-deadline wait for the REAL ``ordinary-1`` registration
    # in WorkerIO's ``_inflight`` map (under the real ``_inflight_lock``) —
    # NOT a mere thread-started proof. NOTE: WorkerIO registers the inflight
    # queue BEFORE taking ``_stdin_lock`` and writing stdin, so registration
    # alone does NOT prove the child received the request.
    registered = False
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with child.wio._inflight_lock:
            if "ordinary-1" in child.wio._inflight:
                registered = True
                break
        time.sleep(0.01)
    assert registered, "ordinary-1 never registered in WorkerIO _inflight"

    # WIRE BARRIER: wait on the ACTUAL child stderr (select.select, finite
    # remaining from the SAME monotonic deadline) for the child's
    # ``ORDINARY_HELD`` marker. This proves the scripted child PROCESSED and
    # HELD the ordinary request before the cancel is invoked — closing the
    # register-before-write race (cancel can never overtake the ordinary
    # request write). No new threads, no mocked receipt, no stdout terminal.
    remaining = deadline - time.monotonic()
    assert remaining > 0.0, "no budget left for the stderr wire barrier"
    readable, _, _ = select.select([child.proc.stderr], [], [], remaining)
    assert readable, "child never signaled ORDINARY_HELD before deadline"
    assert child.proc.stderr.readline() == "ORDINARY_HELD\n"

    # The child has the ordinary request HELD and has NOT replied: the partial
    # (echo_ack) must not have been returned before the cancel.
    assert t.is_alive(), "ordinary consumer finished before cancel (hold broken)"
    assert "ordinary" not in box, (
        "ordinary-1 partial response arrived before the cancel input"
    )

    # Now the helper cancels a DIFFERENT native session id. The child emits the
    # real cancelled receipt through actual stdout, THEN the pending ordinary
    # echo_ack — so the ordinary response is causally after the cancel input.
    receipt = backend._worker_cancel_and_wait("sess-concurrent", 5.0)
    assert receipt == {
        "event": "cancelled",
        "id": "sess-concurrent",
        "ok": False,
        "epoch": 7,
    }

    t.join(timeout=5.0)
    assert not t.is_alive(), "ordinary consumer thread did not finish"
    assert box["ordinary"]["event"] == "echo_ack"
    assert box["ordinary"]["id"] == "ordinary-1"


# ---------------------------------------------------------------------------
# 3. Timeout leaves the worker healthy (no clear, no respawn).
# ---------------------------------------------------------------------------

def test_timeout_leaves_healthy_worker_uncleared(make_child):
    """A ``TimeoutError`` from WorkerIO propagates UNCHANGED and the captured
    pair is retained: no ``_clear_worker_if_current``, no failed marker, no
    respawn (``_ensure_worker`` booby trap still armed)."""
    # A child that NEVER emits a receipt (reads and discards): the wait must
    # time out within budget.
    silent = r"""
import sys
for _raw in sys.stdin:
    pass
"""
    child = make_child(concurrency=4, source=silent)
    backend = _make_backend(child)

    # REGRESSION PIN (blocker): builtin TimeoutError subclasses OSError, so a
    # helper whose OSError arm precedes its TimeoutError arm would catch the
    # receipt timeout as "broken stdin". Assert the class relationship so the
    # regression stays expressible if CPython ever changes it.
    assert issubclass(TimeoutError, OSError)

    with pytest.raises(TimeoutError) as ei:
        backend._worker_cancel_and_wait("sess-timeout", 0.3)
    # It must be the ORIGINAL TimeoutError, NOT remapped to WorkerExitError
    # (which the broken-stdin OSError arm would raise).
    assert type(ei.value) is TimeoutError, type(ei.value)
    assert not isinstance(ei.value, BackendWorkerExitError)

    # Pair retained; worker NOT marked failed, NOT cleared, NOT restarted.
    assert backend._worker is child.proc
    assert backend._wio is child.wio
    assert backend._worker_failed is False
    assert backend._worker_failed_reason is None
    # Booby trap still armed: reaching _ensure_worker would have failed the test.
    assert child.proc.poll() is None


# ---------------------------------------------------------------------------
# 4. Missing / failed / live-without-WorkerIO reject; no spawn, no write.
# ---------------------------------------------------------------------------

def test_missing_worker_rejects_without_spawn(make_child):
    child = make_child()
    backend = _make_backend(child)
    backend._worker = None
    backend._wio = None

    with pytest.raises(BackendWorkerExitError):
        backend._worker_cancel_and_wait("sess-nope", 1.0)

    assert backend._worker is None
    assert backend._wio is None


def test_failed_worker_rejects_without_spawn(make_child):
    child = make_child()
    backend = _make_backend(child)
    backend._worker_failed = True
    backend._worker_failed_reason = "prior broken pipe"

    with pytest.raises(BackendWorkerExitError) as ei:
        backend._worker_cancel_and_wait("sess-failed", 1.0)
    assert "failed" in str(ei.value)
    # Pair unchanged; still no respawn.
    assert backend._worker is child.proc
    assert backend._wio is child.wio


def test_live_without_wio_rejects_without_spawn(make_child):
    child = make_child()
    backend = _make_backend(child)
    backend._wio = None  # live process, no usable WorkerIO

    with pytest.raises(BackendWorkerExitError) as ei:
        backend._worker_cancel_and_wait("sess-no-wio", 1.0)
    assert "live process without WorkerIO" in str(ei.value)
    assert backend._worker is child.proc


# ---------------------------------------------------------------------------
# 5. Invalid sid / timeout reject BEFORE any write or worker lookup.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_sid",
    ["", None, 123, b"bytes", ["x"]],
)
def test_invalid_sid_rejected_before_lookup(make_child, bad_sid):
    child = make_child()
    backend = _make_backend(child)
    # Remove the worker so a lookup-first implementation would raise
    # WorkerExitError instead of ValueError.
    backend._worker = None
    backend._wio = None
    with pytest.raises(ValueError):
        backend._worker_cancel_and_wait(bad_sid, 1.0)


@pytest.mark.parametrize(
    "bad_timeout",
    [0, -1.0, float("inf"), float("nan"), True, False, "1.0", None],
)
def test_invalid_timeout_rejected_before_lookup(make_child, bad_timeout):
    child = make_child()
    backend = _make_backend(child)
    backend._worker = None
    backend._wio = None
    with pytest.raises(ValueError):
        backend._worker_cancel_and_wait("sess-ok", bad_timeout)


# ---------------------------------------------------------------------------
# 6. Genuine worker EOF maps to WorkerExitError and clears only current pair.
# ---------------------------------------------------------------------------

def test_genuine_eof_raises_and_clears_only_current_worker(make_child):
    """The child is ALIVE for the cancel write, then exits (EOF) while the
    helper waits for a receipt that will never come. The canonical WIO exit
    sentinel must map to the backend ``WorkerExitError`` and clear the captured
    pair ONLY when it is still the current PROVEN-exited pair."""
    # Child reads lines but NEVER replies: the cancel writes fine, then the
    # child is made to exit while the helper waits on the receipt.
    silent = r"""
import sys
for _raw in sys.stdin:
    pass
"""
    child = make_child(concurrency=4, source=silent)
    backend = _make_backend(child)

    box: dict = {}

    def _run():
        try:
            box["result"] = backend._worker_cancel_and_wait("sess-eof", 10.0)
        except BaseException as exc:  # noqa: BLE001 - test capture
            box["error"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Let the cancel line land, then cause a GRACEFUL child exit (stdin EOF).
    time.sleep(0.2)
    child.proc.stdin.close()
    child.proc.wait(timeout=5.0)

    t.join(timeout=5.0)
    assert not t.is_alive(), "helper thread did not finish after genuine EOF"
    assert isinstance(box.get("error"), BackendWorkerExitError), box

    # Proven-exited current pair is cleared; NO respawn.
    assert backend._worker is None
    assert backend._wio is None
    assert backend._worker_failed is False


def test_clear_only_if_current_does_not_touch_newer_worker(make_child):
    """A captured worker that is NO LONGER current is not cleared by the
    genuine-exit path (older failure must never discard a newer worker)."""
    silent = r"""
import sys
for _raw in sys.stdin:
    pass
"""
    eof_child = make_child(concurrency=4, source=silent)
    eof_child.proc.stdin.close()
    eof_child.proc.wait(timeout=5.0)

    other = make_child(concurrency=4)
    backend = _make_backend(eof_child)
    # Swap in a NEWER (still-live) pair after the helper captured nothing yet;
    # simulate an older captured failure by calling the compare-and-handle with
    # the OLD pair while the backend now owns the NEW one.
    backend._worker = other.proc
    backend._wio = other.wio

    backend._clear_worker_if_current(
        eof_child.proc, eof_child.wio, reason="older failure"
    )

    # Newer live pair untouched.
    assert backend._worker is other.proc
    assert backend._wio is other.wio
    assert backend._worker_failed is False
