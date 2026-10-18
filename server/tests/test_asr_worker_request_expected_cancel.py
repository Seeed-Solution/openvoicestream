"""Focused tests for the opt-in ``_worker_request(expected_cancel=...)`` arm.

Status: AUTHORED, UNRUN (AST/compile author pass only — fixture correction
revision; see /tmp/slv-v011-asr-expected-cancel-fixture-correction/).

What this file pins (expected-cancel carve-out ONLY):

  1. GENUINE OVERLAP, same SID: a real-stdio scripted child HOLDS the ordinary
     request ``sess-1`` and writes ``HELD`` to STDERR; the barrier waits on
     the ACTUAL stderr wire via ``select`` with one 5s monotonic deadline
     (NOT a mere ``_inflight`` registration poll). The canonical
     ``_worker_cancel_and_wait`` (same SID) triggers the native ``cancelled``
     terminal which WorkerIO fans out to BOTH the receipt queue and the
     ordinary inflight queue. The ordinary consumer must receive the EXACT
     SAME actual payload (ok=false, epoch=7) as the helper, its generator is
     closed, the inflight entry is released, and the marker event REMAINS SET
     (no clear anywhere). Unexpected thread exceptions are captured and MUST
     be absent.
  2. NO opt-in (``expected_cancel=None``): the same cancelled terminal keeps
     the existing generic behavior — ``WorkerProtocolError``.
  3. UNARMED event (never set): same — ``WorkerProtocolError``, no widening.
  4. WRONG ID: a ``cancelled`` terminal with ``request_id="sess-1"`` (so the
     canonical WorkerIO reader routes it to the ordinary ``sess-1`` queue)
     but retaining raw ``id="sess-OTHER"`` must NOT route to success even
     with an armed event — the backend gate fails on the raw id mismatch.
  5. ARMED + cancelled ``ok=True`` (same id): existing ordinary behavior
     preserved — returned UNCHANGED as a normal (non-error) response; the
     arm (which requires ``ok is False``) does not participate.
  6. ARMED + ``event="error"`` ``ok=False`` same id: STILL
     ``WorkerProtocolError`` — the arm is exclusively for ``cancelled``.

``_ensure_worker`` is stubbed with the exact ``lambda: None`` fixture (the
fake pair is already owned); no production changes. No GPU import, no
server/core involvement, no mocked WIO queues/reader/semaphore: real
``WorkerIO`` over real OS pipes with stdlib Python children. Every child
teardown is graceful and finite (stdin EOF → bounded wait → polite TERM),
never SIGKILL; the reader thread is joined with a finite budget.
"""

from __future__ import annotations

import importlib
import json
import os
import select
import subprocess
import sys
import threading
import time
from collections import deque

import pytest

_wio_mod = importlib.import_module("voxedge.backends.jetson.worker_io")
WorkerIO = _wio_mod.WorkerIO

_asr_mod = importlib.import_module("voxedge.backends.jetson.trt_edge_llm_asr")
TRTEdgeLLMASRBackend = _asr_mod.TRTEdgeLLMASRBackend
WorkerProtocolError = _asr_mod.WorkerProtocolError

EPOCH = 7
_BARRIER_BUDGET_S = 5.0

# Holds ordinary request ``sess-1`` until a cancel input arrives, then emits
# the native cancelled terminal (epoch 7) for the cancel's own id — which in
# the overlap tests IS the same SID as the held ordinary request. Before
# holding it writes ``HELD`` to STDERR (never stdout) as the actual wire
# barrier. Any other id request gets an immediate echo_ack.
_HOLD_CHILD_SOURCE = r"""
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
            pending = None
        continue
    sid = obj.get("id")
    if sid == "sess-1":
        pending = sid
        sys.stderr.write("HELD\n")
        sys.stderr.flush()
        continue
    if sid is not None:
        sys.stdout.write(json.dumps(
            {"event": "echo_ack", "id": sid, "ok": True}
        ) + "\n")
        sys.stdout.flush()
"""

# Replies to the ordinary request ``sess-1`` with a cancelled terminal whose
# RAW ``id`` does NOT match, while ``request_id`` targets the ordinary
# ``sess-1`` queue so the canonical WorkerIO reader actually routes it there
# (otherwise the terminal is unsolicited and the consumer would hang). The
# backend gate must fail on the raw id mismatch.
_WRONG_ID_CHILD_SOURCE = r"""
import json
import sys

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
    sid = obj.get("id")
    if sid == "sess-1":
        sys.stdout.write(json.dumps(
            {"event": "cancelled", "id": "sess-OTHER", "ok": False,
             "epoch": 7, "request_id": "sess-1"}
        ) + "\n")
        sys.stdout.flush()
    elif sid is not None:
        sys.stdout.write(json.dumps(
            {"event": "echo_ack", "id": sid, "ok": True}
        ) + "\n")
        sys.stdout.flush()
"""

# Replies with a cancelled terminal whose ok is TRUE (non-native shape): the
# arm must not fire (it requires ok is False); the existing ordinary
# non-error return path returns it UNCHANGED.
_OK_TRUE_CHILD_SOURCE = r"""
import json
import sys

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
    sid = obj.get("id")
    if sid == "sess-1":
        sys.stdout.write(json.dumps(
            {"event": "cancelled", "id": "sess-1", "ok": True,
             "request_id": "sess-1"}
        ) + "\n")
        sys.stdout.flush()
    elif sid is not None:
        sys.stdout.write(json.dumps(
            {"event": "echo_ack", "id": sid, "ok": True}
        ) + "\n")
        sys.stdout.flush()
"""

# Replies with an ERROR terminal (ok=False) for the SAME id: the arm is
# exclusively for ``event=="cancelled"``; an error must still raise the
# existing WorkerProtocolError even with an armed event.
_ERROR_CHILD_SOURCE = r"""
import json
import sys

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
    sid = obj.get("id")
    if sid == "sess-1":
        sys.stdout.write(json.dumps(
            {"event": "error", "id": "sess-1", "ok": False,
             "error": "boom", "request_id": "sess-1"}
        ) + "\n")
        sys.stdout.flush()
    elif sid is not None:
        sys.stdout.write(json.dumps(
            {"event": "echo_ack", "id": sid, "ok": True}
        ) + "\n")
        sys.stdout.flush()
"""


class _Child:
    """One scripted real-stdio child + its canonical WorkerIO.

    Finite graceful teardown ONLY: stdin EOF, bounded wait, polite TERM as
    bounded escalation — never SIGKILL; reader joined with finite budget.
    """

    def __init__(self, source: str, concurrency: int = 4) -> None:
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
        except Exception as exc:  # noqa: BLE001
            problems.append(f"stdin.close() raised {exc!r}")
        try:
            self.wio.close()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"wio.close() raised {exc!r}")
        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
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
                problems.append("WorkerIO _reader_thread still alive")
        return problems


@pytest.fixture
def make_child():
    children: list[_Child] = []

    def _factory(source: str = _HOLD_CHILD_SOURCE, concurrency: int = 4) -> _Child:
        c = _Child(source=source, concurrency=concurrency)
        children.append(c)
        return c

    yield _factory

    problems: list[str] = []
    for idx, c in enumerate(children):
        for p in c.close():
            problems.append(f"child[{idx}]: {p}")
    if problems:
        raise AssertionError("child teardown failures: " + "; ".join(problems))


def _make_backend(child: _Child):
    """``__new__`` backend over an ALREADY-OWNED fake pair.

    ``_ensure_worker`` is the exact ``lambda: None`` fixture: the pair is
    already owned, so the real ``_worker_request`` must never spawn/preload.
    No production change — only the test fixture.
    """
    backend = TRTEdgeLLMASRBackend.__new__(TRTEdgeLLMASRBackend)
    backend._worker_lock = threading.Lock()
    backend._worker = child.proc
    backend._wio = child.wio
    backend._worker_failed = False
    backend._worker_failed_reason = None
    backend._worker_stderr_tail = deque(maxlen=80)
    backend._ensure_worker = lambda: None  # type: ignore[method-assign]
    return backend


def _wait_held_on_stderr(child: _Child) -> None:
    """Barrier on the ACTUAL wire: bounded ``select`` on the child's real
    stderr fd until the ``HELD`` marker arrives, with ONE 5s monotonic
    deadline. This proves the child has READ the ordinary request line
    (it only writes HELD after holding it) — no poll/sleep assumption.
    Buffer must be consumed ONLY via raw ``os.read`` on the fd (a TextIO
    ``read1``/``read(1)`` would leave the remainder of the line inside the
    TextIOWrapper's internal buffer, starving the next ``select`` while the
    child holds). No other stderr TextIO reads happen until teardown.
    """
    assert child.proc.stderr is not None
    fd = child.proc.stderr.fileno()
    deadline = time.monotonic() + _BARRIER_BUDGET_S
    buf = b""
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        readable, _, _ = select.select([fd], [], [], max(0.0, remaining))
        if not readable:
            break
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        buf += chunk
        if b"HELD\n" in buf:
            return
    raise AssertionError(
        f"child never wrote HELD to stderr within {_BARRIER_BUDGET_S}s "
        f"(got {buf!r})"
    )


def _run_ordinary(backend, payload: dict, expected_cancel, box: dict):
    """Run the ACTUAL ``_worker_request`` on a thread, capturing ANY
    unexpected exception into ``box`` (never silently suppressed)."""

    def _ordinary():
        started.set()
        try:
            box["result"] = backend._worker_request(
                payload, expected_cancel=expected_cancel
            )
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            box["error"] = exc

    started = threading.Event()
    t = threading.Thread(target=_ordinary, daemon=True)
    t.start()
    assert started.wait(timeout=_BARRIER_BUDGET_S)
    return t


# ---------------------------------------------------------------------------
# 1. Genuine overlap: same-SID cancel, armed event, actual dict to consumer.
# ---------------------------------------------------------------------------

def test_armed_expected_cancel_returns_actual_dict_to_ordinary_consumer(make_child):
    child = make_child()
    backend = _make_backend(child)
    marker = threading.Event()
    marker.set()  # armed BEFORE the request is sent
    box: dict = {}

    t = _run_ordinary(
        backend, {"event": "begin", "id": "sess-1"}, marker, box
    )
    # ACTUAL wire barrier: the child has held sess-1 and wrote HELD to
    # stderr over the real pipe.
    _wait_held_on_stderr(child)
    assert t.is_alive(), "ordinary request finished before cancel (hold broken)"

    # Canonical helper, SAME SID: native cancelled ok=false epoch=7 receipt.
    receipt = backend._worker_cancel_and_wait("sess-1", _BARRIER_BUDGET_S)
    assert receipt == {
        "event": "cancelled",
        "id": "sess-1",
        "ok": False,
        "epoch": 7,
    }

    t.join(timeout=_BARRIER_BUDGET_S)
    assert not t.is_alive(), "ordinary _worker_request thread did not finish"
    assert "error" not in box, f"unexpected thread exception: {box['error']!r}"
    # EXACT SAME actual payload — fanned out by WorkerIO, returned verbatim.
    assert box["result"] == receipt
    # The marker is NEVER cleared by the backend.
    assert marker.is_set(), "expected_cancel marker was cleared"
    # The ordinary generator closed and the inflight slot is released.
    with child.wio._inflight_lock:
        assert "sess-1" not in child.wio._inflight
    # The worker stays healthy: no failure marker, no respawn.
    assert backend._worker_failed is False
    assert backend._worker is child.proc


# ---------------------------------------------------------------------------
# 2. No opt-in: existing generic ok=false behavior preserved.
# ---------------------------------------------------------------------------

def test_no_optin_cancelled_terminal_raises_worker_protocol_error(make_child):
    child = make_child()
    backend = _make_backend(child)
    box: dict = {}

    t = _run_ordinary(
        backend, {"event": "begin", "id": "sess-1"}, None, box
    )
    _wait_held_on_stderr(child)

    receipt = backend._worker_cancel_and_wait("sess-1", _BARRIER_BUDGET_S)
    assert receipt["event"] == "cancelled"

    t.join(timeout=_BARRIER_BUDGET_S)
    assert not t.is_alive()
    assert isinstance(box.get("error"), WorkerProtocolError), (
        f"expected generic WorkerProtocolError, got {box.get('error')!r}"
    )


# ---------------------------------------------------------------------------
# 3. Unarmed event: no widening to success.
# ---------------------------------------------------------------------------

def test_unarmed_expected_cancel_raises_worker_protocol_error(make_child):
    child = make_child()
    backend = _make_backend(child)
    marker = threading.Event()  # NOT set
    box: dict = {}

    t = _run_ordinary(
        backend, {"event": "begin", "id": "sess-1"}, marker, box
    )
    _wait_held_on_stderr(child)

    receipt = backend._worker_cancel_and_wait("sess-1", _BARRIER_BUDGET_S)
    assert receipt["event"] == "cancelled"

    t.join(timeout=_BARRIER_BUDGET_S)
    assert not t.is_alive()
    assert isinstance(box.get("error"), WorkerProtocolError), (
        f"unarmed event must NOT route to success, got {box.get('error')!r}"
    )
    assert not marker.is_set()


# ---------------------------------------------------------------------------
# 4. Wrong id (routed via request_id): armed event still refuses.
# ---------------------------------------------------------------------------

def test_wrong_id_cancelled_terminal_raises_even_when_armed(make_child):
    child = make_child(source=_WRONG_ID_CHILD_SOURCE)
    backend = _make_backend(child)
    marker = threading.Event()
    marker.set()
    with pytest.raises(WorkerProtocolError):
        backend._worker_request(
            {"event": "begin", "id": "sess-1"}, expected_cancel=marker
        )
    # Gate failure never clears the marker.
    assert marker.is_set()


# ---------------------------------------------------------------------------
# 5. Armed + cancelled ok=True: existing ordinary return, unchanged.
# ---------------------------------------------------------------------------

def test_armed_cancelled_ok_true_preserves_ordinary_return(make_child):
    child = make_child(source=_OK_TRUE_CHILD_SOURCE)
    backend = _make_backend(child)
    marker = threading.Event()
    marker.set()
    result = backend._worker_request(
        {"event": "begin", "id": "sess-1"}, expected_cancel=marker
    )
    # Returned UNCHANGED through the existing non-error path (the arm requires
    # ok is False and did NOT participate).
    assert result == {
        "event": "cancelled",
        "id": "sess-1",
        "ok": True,
        "request_id": "sess-1",
    }
    assert marker.is_set()


# ---------------------------------------------------------------------------
# 6. Armed + event=error ok=False same id: STILL WorkerProtocolError.
# ---------------------------------------------------------------------------

def test_armed_error_terminal_still_raises_worker_protocol_error(make_child):
    child = make_child(source=_ERROR_CHILD_SOURCE)
    backend = _make_backend(child)
    marker = threading.Event()
    marker.set()
    with pytest.raises(WorkerProtocolError):
        backend._worker_request(
            {"event": "begin", "id": "sess-1"}, expected_cancel=marker
        )
    assert marker.is_set()
