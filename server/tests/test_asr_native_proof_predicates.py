"""Proof-predicate tests for server/main.py ASR native cancel paths.

Extracts the ACTUAL nested functions ``_arm_cancel``, ``_control_cancel``,
``_old_native_cancel_resolved`` and ``_retained_entry_released`` from the real
websocket handler source via AST (no duplicated predicate logic), compiles and
executes each against controlled fakes in isolated globals.
"""

import ast
import asyncio
import os
import types

import pytest

# Parse the ACTUAL server/main.py source (no import: module import pulls the
# full FastAPI app). All predicates below are extracted via AST.
MAIN_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"
)

HANDLER_NAME = "_asr_stream_backend"


def _find_module_handler(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == HANDLER_NAME:
            return node
    raise AssertionError(f"handler {HANDLER_NAME} not found in server/main.py")


def _find_nested(parent, name):
    for node in ast.walk(parent):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"nested function {name} not found in {HANDLER_NAME}")


def _extract(name):
    src = open(MAIN_SRC).read()
    tree = ast.parse(src)
    handler = _find_module_handler(tree)
    node = _find_nested(handler, name)
    mod = ast.Module(body=[node], type_ignores=[])
    code = compile(mod, f"<{name}>", "exec")
    return code


def _make_g(extra=None):
    g = {"__name__": "controlled"}
    if extra:
        g.update(extra)
    return g


class FakeStream:
    def __init__(self, receipt=None):
        self.arm_calls = 0
        self._receipt = receipt

    def arm_cancel(self):
        self.arm_calls += 1
        return "SID-1"

    def request_cancel(self, timeout):
        r = self._receipt
        if isinstance(r, Exception):
            raise r
        return r


class FakeJobs:
    def __init__(self, receipt="OK"):
        self.receipt = receipt
        self.calls = 0

    async def run(self, fn, timeout, executor=None):
        self.calls += 1
        r = self.receipt
        if isinstance(r, Exception):
            raise r
        return r


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, obj):
        self.sent.append(obj)


class FakePool:
    pass


class FakeWorker:
    def __init__(self, code=None):
        self._code = code

    def poll(self):
        return self._code


def _new_st():
    return {
        "cancel_armed": False,
        "cancel_sid": None,
        "epoch": 0,
        "run_epoch": 0,
        "control_task": None,
        "control_outcome": None,
        "client_gone": False,
        "done": False,
    }


def _valid_receipt(sid="SID-1"):
    return {"event": "cancelled", "id": sid, "ok": False}


def _build_control(receipt, st=None):
    st = st or _new_st()
    # simulate an ARMED current cancel generation (as _arm_cancel leaves it)
    st["epoch"] = 1
    st["run_epoch"] = 1
    st["cancel_armed"] = True
    st["cancel_sid"] = "SID-1"
    ws = FakeWS()

    async def _send_now(obj):
        ws.sent.append(obj)

    cap_stream = types.SimpleNamespace(request_cancel=(lambda: receipt))
    g = _make_g({
        "st": st,
        "asyncio": asyncio,
        "conn_jobs": FakeJobs(receipt),
        "control_pool": FakePool(),
        "_send_now": _send_now,
        "send_lock": asyncio.Lock(),
        "ws": ws,
        "logger": __import__("logging").getLogger("test"),
        "_is_worker_exit": lambda e: False,
    })
    g["cap_stream"] = cap_stream
    exec(_extract("_control_cancel"), g)
    return g["_control_cancel"], st, g["ws"], g["conn_jobs"], g["cap_stream"]


def _build_arm(st, conn_jobs):
    ctrl_fn, _, ctrl_ws, _, _ = _build_control(_valid_receipt(), st)
    g = _make_g({
        "st": st,
        "asyncio": asyncio,
        "logger": __import__("logging").getLogger("test"),
        "asr_be": types.SimpleNamespace(name="fake"),
        "_control_cancel": ctrl_fn,
    })
    exec(_extract("_arm_cancel"), g)
    # _build_control pre-arms st for the ACK guard; reset so the arm path
    # itself is exercised from a clean state.
    st["cancel_armed"] = False
    st["control_task"] = None
    st["epoch"] = 0
    st["run_epoch"] = 0
    return lambda stream: g["_arm_cancel"](stream, conn_jobs), ctrl_ws


@pytest.mark.asyncio
async def test_valid_matching_receipt_emits_exactly_one_ack():
    st = _new_st()
    ctrl, st, ws, jobs, cap = _build_control(_valid_receipt(), st)
    await ctrl("SID-1", cap, jobs)
    acks = [m for m in ws.sent if m.get("type") == "cancel_ack"]
    assert len(acks) == 1
    assert acks[0]["id"] == "SID-1"
    assert acks[0]["epoch"] == st["epoch"]
    assert st["control_outcome"] == "confirmed"


@pytest.mark.asyncio
async def test_duplicate_arm_no_extra_control_or_ack():
    st = _new_st()
    arm, ws = _build_arm(st, FakeJobs(_valid_receipt()))
    stream = FakeStream(_valid_receipt())
    arm(stream)
    task = st["control_task"]
    assert task is not None
    arm(stream)  # duplicate while armed: must be a no-op
    assert stream.arm_calls == 1
    assert st["control_task"] is task  # no second control task scheduled
    await asyncio.wait_for(task, timeout=2.0)
    assert st["control_outcome"] == "confirmed"
    # BUG 1 proof: with the arm-time run_epoch sync, a valid matching cancel
    # on the CURRENT generation emits EXACTLY ONE ack with current SID+epoch.
    acks = [m for m in ws.sent if m.get("type") == "cancel_ack"]
    assert len(acks) == 1
    assert acks[0]["id"] == "SID-1" and acks[0]["epoch"] == st["epoch"]


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [
    {"event": "cancelled", "id": "SID-1", "ok": True},
    {"event": "cancelled", "id": "OTHER", "ok": False},
    {"event": "other", "id": "SID-1", "ok": False},
    "not-a-dict",
])
async def test_invalid_receipts_no_ack_distinct_error(receipt):
    ctrl, st, ws, jobs, cap = _build_control(receipt)
    await ctrl("SID-1", cap, jobs)
    assert [m for m in ws.sent if m.get("type") == "cancel_ack"] == []
    errs = [m for m in ws.sent if m.get("type") == "error"]
    assert errs and errs[-1]["error"] == "cancel_receipt_invalid"
    assert st["control_outcome"] == "error"


@pytest.mark.asyncio
async def test_old_generation_no_ack():
    st = _new_st()
    ctrl, st, ws, jobs, cap = _build_control(_valid_receipt(), st)
    st["run_epoch"] = st["epoch"] - 1  # stale generation
    await ctrl("SID-1", cap, jobs)
    assert [m for m in ws.sent if m.get("type") == "cancel_ack"] == []
    assert st["control_outcome"] == "confirmed"  # confirmed but no ACK sent


def _build_release_fns():
    g = _make_g({"getattr": getattr})
    exec(_extract("_old_native_cancel_resolved"), g)
    resolved = g["_old_native_cancel_resolved"]
    g2 = _make_g({"getattr": getattr, "_old_native_cancel_resolved": resolved})
    exec(_extract("_retained_entry_released"), g2)
    return resolved, g2["_retained_entry_released"]


def _stream(**kw):
    s = types.SimpleNamespace(_closed=False, _cancel_exit=False,
                              _cancel_intent=None, _cancel_confirm=None)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _confirm(sid="S1"):
    return {"sid": sid, "receipt": {"event": "cancelled", "id": sid, "ok": False}}


def test_release_predicates():
    _, released = _build_release_fns()
    worker = FakeWorker(None)  # alive; same object, controlled poll

    def entry(s, w=worker):
        return {"stream": s, "worker": w}

    # actual backend-closed -> release
    assert released(entry(_stream(_closed=True))) is True
    # proven cancel exit -> release
    assert released(entry(_stream(_cancel_exit=True))) is True
    # exact matching ok=False confirm with intent -> release
    assert released(entry(_stream(_cancel_intent={"sid": "S1"},
                                  _cancel_confirm=_confirm()))) is True
    # wrong receipt shape / ok True with intent -> NOT release (alive worker)
    bad = {"sid": "S1", "receipt": {"event": "cancelled", "id": "S1", "ok": True}}
    assert released(entry(_stream(_cancel_intent={"sid": "S1"},
                                  _cancel_confirm=bad))) is False
    wrong = {"sid": "S1", "receipt": {"event": "cancelled", "id": "X", "ok": False}}
    assert released(entry(_stream(_cancel_intent={"sid": "S1"},
                                  _cancel_confirm=wrong))) is False
    # uncancelled stream close (no intent, _closed False, worker alive) -> FALSE
    assert released(entry(_stream())) is False
    # missing worker (None), no intent, not closed -> UNKNOWN -> FALSE
    assert released(entry(_stream(), None)) is False
    # armed-but-unresolved: worker alive, poll None -> FALSE
    assert released(entry(_stream(_cancel_intent={"sid": "S1"}))) is False
    # same worker object observed exited -> release
    worker._code = 0
    assert released(entry(_stream())) is True
