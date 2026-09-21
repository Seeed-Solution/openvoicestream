"""Deterministic server-loop tool-call scenario harness + P0 batch.

Closes the "tool-call flow" coverage gap from
``agent/docs/voice-scenario-test-catalog.md``: every existing e2e is a
client-loop run with no tools, so the production *server-loop* tool path
(SLV's LLM proxies tool execution back to the agent via SERVER_TOOL_CALL)
had no scenario coverage beyond protocol units.

This module is fully deterministic (no real SLV / LLM / device):
  * a ``FakeSLV`` spy captures every ``send_tool_result`` (ok / name / error),
  * stub tools with controllable outcomes are registered per test,
  * ``ServerToolCall`` events are injected straight through the production
    dispatch entry point ``BaseApp._dispatch_one`` (same code path SLV's
    reader drives in prod).

Harness design (how it injects + captures):
  inject:  ``await app._dispatch_one(ServerToolCall(id, name, arguments))``
           → real ``_spawn_tool_task`` → ``_handle_server_tool_call`` →
             ``registry.dispatch`` → ``slv.send_tool_result``.
  capture: ``FakeSLV.tool_results`` — a list of
           ``{"id","name","ok","result","error"}`` dicts in send order.
  drain:   tool handlers run as background tasks (#F4), so tests await
           ``_drain_tool_tasks(app)`` before asserting the captured results.

Scenarios implemented (catalog IDs): TC-006, TC-008, TC-009, TC-010,
TC-010b, TC-012, TC-014, ER-001, ER-009. See module-level test docstrings
for per-scenario notes.
"""
from __future__ import annotations

import asyncio

import pytest

from ovs_agent import Config, Session
from ovs_agent.app_base import BaseApp
from ovs_agent.slv_client import SLVClient, ServerToolCall
from ovs_agent.state import ConvState
from ovs_agent.tools import ToolRegistry


# ── harness ──────────────────────────────────────────────────────────


async def _drain_tool_tasks(app: BaseApp) -> None:
    """SERVER_TOOL_CALL dispatch is backgrounded (#F4): ``_dispatch_one``
    returns before the CLIENT_TOOL_RESULT is sent. Await the tracked tasks so
    assertions observe completed results."""
    while getattr(app, "_pending_tool_tasks", None):
        await asyncio.gather(
            *list(app._pending_tool_tasks), return_exceptions=True
        )


class FakeSLV:
    """Spy that captures every server-loop frame the agent sends.

    ``tool_results`` is the assertion surface for the tool scenarios: one
    dict per ``send_tool_result`` in send order. ``_ws`` is exposed so a test
    can simulate a dead WS (TC-014) — the *real* drop logic lives in
    ``SLVClient._send_json(connect_if_dead=False)``, so TC-014 uses a real
    SLVClient instead of this spy (see ``_dead_ws_client``).
    """

    def __init__(self) -> None:
        self.advertised: list[dict] = []
        self.tool_results: list[dict] = []
        self.text_frames: list[str] = []
        self.flushed = 0
        self.aborted = 0
        self._ws = object()  # truthy: "connected"

    async def advertise_tools(self, tools, *, system_prompt=None, llm_params=None):
        self.advertised.append(
            {"tools": tools, "system_prompt": system_prompt, "llm_params": llm_params}
        )

    async def send_tool_result(self, call_id, name, *, ok, result=None, error=None):
        self.tool_results.append(
            {"id": call_id, "name": name, "ok": ok, "result": result, "error": error}
        )

    async def send_text(self, text: str) -> None:
        self.text_frames.append(text)

    async def flush_tts(self) -> None:
        self.flushed += 1

    async def abort(self, *, keep_asr: bool = False) -> None:
        self.aborted += 1

    async def reconnect(self) -> None:
        return None

    def session_gen(self) -> int:
        return 0


class _SpyPlugin:
    """Records hook fan-out so deterministic-sim tests can assert that an
    event did/did not produce a user utterance, state change, etc."""

    name = "spy"

    def __init__(self) -> None:
        self.user_utterances: list = []
        self.state_changes: list = []

    async def on_user_utterance(self, text):
        self.user_utterances.append(text)

    async def on_state_change(self, payload):
        self.state_changes.append(payload)


def _make_app(
    *,
    server_loop: bool = True,
    registry: ToolRegistry | None = None,
    slv: FakeSLV | None = None,
    plugins: list | None = None,
) -> BaseApp:
    """Build a BaseApp via ``__new__`` (no audio/WS setup), matching the
    existing server-loop unit pattern (``tests/test_server_loop_client.py``)."""
    cfg = Config(
        system_prompt="SYS",
        server_loop=server_loop,
        tools_max_iterations=4,
        llm_model="qwen-test",
        pipeline_mode="always_on",  # no auto-sleep timer in these tests
    )
    app = BaseApp.__new__(BaseApp)
    app.config = cfg
    app.tool_registry = registry if registry is not None else ToolRegistry()
    app.session = Session()
    app.events = type("E", (), {"emit": lambda *a, **k: None})()
    app.modes = None
    app.slv = slv if slv is not None else FakeSLV()
    app.plugins = plugins if plugins is not None else []
    app._pending_tool_tasks = set()
    app._advertised_gen = 0
    app._state = ConvState.IDLE
    app._last_user_utterance_text = ""
    app._eos_sent_this_turn = False
    return app


async def _inject(app: BaseApp, *calls: ServerToolCall) -> None:
    """Inject one or more ServerToolCall events through the real dispatch
    entry point and drain the backgrounded handlers."""
    for c in calls:
        await app._dispatch_one(c)
    await _drain_tool_tasks(app)


# ── TC-006: tool failure → ok=False → second attempt succeeds ─────────


@pytest.mark.asyncio
async def test_tc006_tool_failure_then_retry_succeeds():
    """TC-006: first dispatch returns ``started:False`` → ok=False; the
    server-loop LLM retries → second dispatch returns ``started:True`` →
    ok=True. Asserts the exact ok sequence ``[False, True]`` (no stranding,
    same call resolved cleanly twice)."""
    reg = ToolRegistry()
    attempts = {"n": 0}

    @reg.tool(name="grasp_object", description="grasp")
    def grasp_object(label: str = "box") -> dict:
        attempts["n"] += 1
        if attempts["n"] == 1:
            # parallel-mode refusal shape: started:False + error, no success key
            return {"started": False, "error": "perception: object not found"}
        return {"started": True, "action": "grasp_object", "label": label}

    app = _make_app(registry=reg)

    await _inject(app, ServerToolCall(id="g1", name="grasp_object", arguments={"label": "box"}))
    await _inject(app, ServerToolCall(id="g2", name="grasp_object", arguments={"label": "box"}))

    oks = [r["ok"] for r in app.slv.tool_results]
    assert oks == [False, True], app.slv.tool_results
    first, second = app.slv.tool_results
    assert first["id"] == "g1" and first["error"] == "perception: object not found"
    assert second["id"] == "g2" and second["result"]["started"] is True
    assert attempts["n"] == 2


# ── TC-008: sequential tool calls grasp→put_down→grasp→wave ───────────


@pytest.mark.asyncio
async def test_tc008_sequential_tool_calls_no_crosstalk():
    """TC-008: four motions issued back-to-back (the server awaits each
    result before the next). Asserts all four resolve ok=True, in order, with
    no id/name cross-talk and each handler ran exactly once."""
    reg = ToolRegistry()
    ran: list[str] = []

    def _register(tool_name: str):
        @reg.tool(name=tool_name, description=f"{tool_name} motion")
        def _h() -> dict:
            ran.append(tool_name)
            return {"started": True, "action": tool_name}

    for n in ("grasp_object", "put_down", "wave"):
        _register(n)

    app = _make_app(registry=reg)

    seq = [
        ServerToolCall(id="s1", name="grasp_object", arguments={}),
        ServerToolCall(id="s2", name="put_down", arguments={}),
        ServerToolCall(id="s3", name="grasp_object", arguments={}),
        ServerToolCall(id="s4", name="wave", arguments={}),
    ]
    # Sequential semantics: await each result before issuing the next.
    for c in seq:
        await _inject(app, c)

    results = app.slv.tool_results
    assert [r["id"] for r in results] == ["s1", "s2", "s3", "s4"]
    assert [r["name"] for r in results] == [
        "grasp_object", "put_down", "grasp_object", "wave",
    ]
    assert all(r["ok"] is True for r in results)
    assert ran == ["grasp_object", "put_down", "grasp_object", "wave"]


# ── TC-012: concurrent tool_call (second arrives before first returns) ─


@pytest.mark.asyncio
async def test_tc012_concurrent_tool_calls_run_as_background_tasks():
    """TC-012: a second SERVER_TOOL_CALL is dispatched while the first is
    still running (slow handler not yet returned).

    Documents the *actual* implementation: ``_spawn_tool_task`` schedules each
    call as an independent background task with NO app-level motion mutex (see
    PRODUCT-OBSERVATION in the report). So both tasks are in-flight
    concurrently; once released, both resolve ok=True. The dispatch loop is
    never blocked by a running tool (the #F4 contract)."""
    reg = ToolRegistry()
    started_a = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    @reg.tool(name="slow_move", description="slow")
    async def slow_move() -> dict:
        started_a.set()
        await release.wait()
        order.append("slow_move")
        return {"started": True, "action": "slow_move"}

    @reg.tool(name="quick_move", description="quick")
    async def quick_move() -> dict:
        order.append("quick_move")
        return {"started": True, "action": "quick_move"}

    app = _make_app(registry=reg)

    # 1st call (slow) — dispatch returns immediately, handler is in-flight.
    await app._dispatch_one(ServerToolCall(id="c1", name="slow_move", arguments={}))
    await asyncio.wait_for(started_a.wait(), timeout=1.0)
    assert app.slv.tool_results == []  # slow handler hasn't returned yet
    assert len(app._pending_tool_tasks) == 1

    # 2nd call arrives WHILE the 1st is still running → second background task.
    await app._dispatch_one(ServerToolCall(id="c2", name="quick_move", arguments={}))
    assert len(app._pending_tool_tasks) == 2, "both tool tasks in-flight concurrently"

    # quick_move can complete before slow_move is released (no mutex serializes).
    release.set()
    await _drain_tool_tasks(app)

    by_id = {r["id"]: r for r in app.slv.tool_results}
    assert by_id["c1"]["ok"] is True and by_id["c2"]["ok"] is True
    assert set(order) == {"slow_move", "quick_move"}
    assert not app._pending_tool_tasks


# ── TC-010: tool timeout → error result, dispatch loop never blocks ───


@pytest.mark.asyncio
async def test_tc010_slow_tool_does_not_block_dispatch_loop():
    """TC-010 (#F4 non-blocking contract).

    There ARE two timeout layers (see TC-010b for the agent-side one):
    ``ToolRegistry.dispatch`` wraps *async* handlers in
    ``asyncio.wait_for(..., timeout_s)`` (default 10s), and the SLV owns a
    separate ~15s tool_result budget. This test asserts the orthogonal #F4
    contract that holds regardless of either timeout: a slow/hung tool does
    NOT block ``_dispatch_one`` — subsequent events keep flowing — and once it
    finishes the result is delivered. (Released before its 10s default, so no
    timeout fires here; TC-010b covers the timeout path.)"""
    reg = ToolRegistry()
    release = asyncio.Event()
    other_ran: list[str] = []

    @reg.tool(name="hang", description="hangs until released")
    async def hang() -> dict:
        await release.wait()
        return {"started": True, "action": "hang"}

    @reg.tool(name="other", description="fast")
    async def other() -> dict:
        other_ran.append("other")
        return {"started": True, "action": "other"}

    app = _make_app(registry=reg)

    # Dispatch the hung tool — returns immediately despite the handler blocking.
    await asyncio.wait_for(
        app._dispatch_one(ServerToolCall(id="h1", name="hang", arguments={})),
        timeout=1.0,
    )
    assert app.slv.tool_results == []  # hung tool produced no result yet

    # The dispatch loop is free: a subsequent event is handled while hang runs.
    # (Don't use _inject here — its drain would gather the still-hung task too;
    # the 'other' task is awaited explicitly instead.)
    await app._dispatch_one(ServerToolCall(id="o1", name="other", arguments={}))
    # Await only the fast task by polling until 'other' has produced its result.
    for _ in range(100):
        if any(r["id"] == "o1" for r in app.slv.tool_results):
            break
        await asyncio.sleep(0.01)
    assert other_ran == ["other"]
    assert [r["id"] for r in app.slv.tool_results] == ["o1"]  # hang still pending

    # Release the hung tool → its result is delivered (never lost / deadlocked).
    release.set()
    await _drain_tool_tasks(app)
    assert {r["id"] for r in app.slv.tool_results} == {"o1", "h1"}
    assert all(r["ok"] is True for r in app.slv.tool_results)


# ── TC-014: send_tool_result on a dead WS must not misdeliver ─────────


def _dead_ws_client() -> SLVClient:
    """A real SLVClient with no live WS — exercises the production drop logic
    in ``_send_json(connect_if_dead=False)`` rather than a spy."""
    c = SLVClient.__new__(SLVClient)
    c._ws = None
    c._closed = False
    c._send_lock = asyncio.Lock()
    c.config = {}
    c.url = "ws://unused"
    return c


@pytest.mark.asyncio
async def test_tc014_tool_result_dropped_on_dead_ws_no_reconnect():
    """TC-014: when the WS that issued the SERVER_TOOL_CALL has died, the
    agent must DROP the tool_result (its call_id is meaningless to a fresh
    session) and must NOT auto-connect — else it misdelivers to a new session
    and stalls the server turn (#6).

    Uses the *real* ``SLVClient.send_tool_result`` → ``_send_json`` path with
    ``_ws=None``. Patches ``connect`` to a tripwire that fails the test if the
    drop logic ever auto-connects."""
    c = _dead_ws_client()
    connect_calls = {"n": 0}

    async def _tripwire_connect():
        connect_calls["n"] += 1

    c.connect = _tripwire_connect  # type: ignore[method-assign]

    # Dead WS → result is dropped, connect() is never called.
    await c.send_tool_result("call_dead", "wave", ok=True, result={"started": True})
    assert connect_calls["n"] == 0, "must NOT auto-connect for a stale tool_result"
    assert c._ws is None

    # The next session works normally: give it a live fake WS and send again.
    sent: list[str] = []

    class _LiveWS:
        async def send(self, data):
            sent.append(data)

    c._ws = _LiveWS()
    await c.send_tool_result("call_live", "wave", ok=True, result={"started": True})
    assert len(sent) == 1
    import json as _json

    frame = _json.loads(sent[0])
    assert frame["call_id"] == "call_live" and frame["ok"] is True


# ── TC-009: wrong tool dispatched, then corrected — both resolve clean ─


@pytest.mark.asyncio
async def test_tc009_wrong_tool_then_corrected_resolves_cleanly():
    """TC-009: the server-loop LLM first selects the WRONG tool (``wave`` when
    the user asked to grasp), the user corrects, and the LLM then selects the
    RIGHT tool (``grasp_object``). Both calls are independent SERVER_TOOL_CALLs.

    Faithful agent-side assertion: each call resolves ok=True on its own id,
    in order, each handler runs exactly once, and the wrong first call leaves
    NO residue that strands or mis-routes the corrected call. (The agent can't
    know ``wave`` was 'wrong' — selection is the SLV LLM's job; the agent's
    contract is that every dispatched call resolves cleanly and independently.)
    """
    reg = ToolRegistry()
    ran: list[str] = []

    @reg.tool(name="wave", description="wave hello")
    def wave() -> dict:
        ran.append("wave")
        return {"started": True, "action": "wave"}

    @reg.tool(name="grasp_object", description="grasp")
    def grasp_object(label: str = "box") -> dict:
        ran.append("grasp_object")
        return {"started": True, "action": "grasp_object", "label": label}

    app = _make_app(registry=reg)

    # Wrong tool first, then the corrected tool (sequential: await each).
    await _inject(app, ServerToolCall(id="w1", name="wave", arguments={}))
    await _inject(
        app, ServerToolCall(id="g1", name="grasp_object", arguments={"label": "box"})
    )

    results = app.slv.tool_results
    assert [r["id"] for r in results] == ["w1", "g1"]
    assert [r["name"] for r in results] == ["wave", "grasp_object"]
    assert all(r["ok"] is True for r in results)
    assert ran == ["wave", "grasp_object"]  # each ran once, in order
    assert results[1]["result"]["label"] == "box"  # corrected call carried its args


# ── TC-010b: async tool exceeding its timeout_s → ok=False (real path) ─


@pytest.mark.asyncio
async def test_tc010b_async_tool_timeout_returns_ok_false_error():
    """TC-010b: an *async* tool that runs past its registered ``timeout_s``
    is cancelled by ``ToolRegistry.dispatch`` (``asyncio.wait_for``) and the
    registry returns ``{"success": False, "error": "tool ... timed out ..."}``
    → ``_handle_server_tool_call`` reports ok=False with that error.

    This is the agent-side per-tool timeout the catalog asked for — it DOES
    exist at the registry layer (``registry.py:299``), contrary to the older
    'no per-tool timeout' note. It applies to coroutine handlers only; a
    blocking *sync* handler is not wrapped and would not time out here. Audit
    2026-06-14: every production arm tool is already ``async def`` + wraps the
    blocking serial motion in ``asyncio.to_thread``
    (``plugins/actuator_actions.py``), so they all inherit this timeout; the
    sync caveat only governs future tools. A hung tool therefore surfaces as a
    clean failure the server-loop LLM can react to, not an indefinite stall."""
    reg = ToolRegistry()

    @reg.tool(name="slow_grasp", description="overruns", timeout_s=0.05)
    async def slow_grasp() -> dict:
        await asyncio.sleep(5.0)  # far past the 0.05s budget
        return {"started": True, "action": "slow_grasp"}  # never reached

    app = _make_app(registry=reg)

    await _inject(app, ServerToolCall(id="t1", name="slow_grasp", arguments={}))

    assert len(app.slv.tool_results) == 1
    r = app.slv.tool_results[0]
    assert r["id"] == "t1" and r["name"] == "slow_grasp"
    assert r["ok"] is False
    assert "timed out" in (r["error"] or ""), r
    assert not app._pending_tool_tasks  # the timed-out task was drained, not leaked


# ── ER-009: arm unavailable / perception-fail → clean ok=False, no loop ─


@pytest.mark.asyncio
async def test_er009_arm_unavailable_surfaces_ok_false_once():
    """ER-009: the arm/perception is unavailable, so the motion handler
    refuses with ``{"started": False, "error": ...}`` (the parallel-mode
    refusal shape). The agent must report ok=False with that error EXACTLY
    ONCE — never ok=True (which made the real arm loop the same motion every
    round until the iteration cap, 2026-06-12) — and the dispatch must not
    strand. Covers both the explicit refusal dict and a handler that *raises*
    (``ToolRegistry.dispatch`` catches → ``success:False``)."""
    reg = ToolRegistry()

    @reg.tool(name="grasp_object", description="grasp")
    def grasp_object(label: str = "box") -> dict:
        return {"started": False, "error": "arm offline: motor bridge not connected"}

    @reg.tool(name="put_down", description="put down")
    def put_down() -> dict:
        raise RuntimeError("perception: no plane detected")

    app = _make_app(registry=reg)

    await _inject(app, ServerToolCall(id="a1", name="grasp_object", arguments={}))
    await _inject(app, ServerToolCall(id="a2", name="put_down", arguments={}))

    refusal, crash = app.slv.tool_results
    assert refusal["id"] == "a1" and refusal["ok"] is False
    assert refusal["error"] == "arm offline: motor bridge not connected"
    assert refusal["result"] is None  # ok=False frames carry error, not result
    # A raising handler is caught by the registry and also surfaces ok=False
    # (never crashes the dispatch task or strands the server turn).
    assert crash["id"] == "a2" and crash["ok"] is False
    assert "perception: no plane detected" in (crash["error"] or "")
    assert not app._pending_tool_tasks


# ── ER-001: empty asr_final ignored (deterministic-sim layer) ─────────


@pytest.mark.asyncio
async def test_er001_empty_final_from_thinking_ignored_resets_to_idle():
    """ER-001: an empty ASRFinal arriving while THINKING (after a client EOS)
    must NOT produce a user utterance / LLM turn, and the FSM must recover to
    IDLE (race #2 guard) rather than stranding in THINKING.

    Deterministic-sim layer: drives the real ``_dispatch_one(ASRFinal(...))``
    with ``server_loop=False`` (no tools involved). A ``_SpyPlugin`` records
    the hook fan-out so we can assert *no* ``on_user_utterance`` fired."""
    from ovs_agent.app_base import ASRFinal

    spy = _SpyPlugin()
    app = _make_app(server_loop=False, plugins=[spy])
    app._state = ConvState.THINKING
    app._eos_sent_this_turn = True  # we sent client EOS, expected a real final

    # Empty-text final (server VAD coalesced silence) with session_complete.
    evt = ASRFinal(text="", language=None, duplicate_of_streamed=False, session_complete=True)
    await app._dispatch_one(evt)
    # Let _set_state's fire-and-forget on_state_change broadcast task run.
    await asyncio.sleep(0)

    # No user utterance was broadcast — empty final never routed to the LLM.
    assert spy.user_utterances == [], "empty final must not produce a user utterance"
    # FSM recovered to IDLE (race #2), not stuck in THINKING.
    assert app._state == ConvState.IDLE
