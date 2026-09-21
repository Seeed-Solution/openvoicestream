"""Barge-in / stop during an in-flight server-loop tool (TC-007 / TC-011).

Catalog gap: "interrupt-during-tool". The subtle, demo-relevant truth is WHICH
interruption stops WHAT:

  * ``_interrupt_current_turn_for_barge_in()`` (barge-in: the user talks over the
    assistant) cancels the LLM streaming task, stops local playback, and sends
    SLV's in-band ``abort`` — i.e. it interrupts SPEECH. It does NOT touch
    ``_pending_tool_tasks`` and does NOT broadcast ``on_sleep``, so an in-flight
    server tool (e.g. a running grasp) keeps going. **Talking over the robot
    does not stop the arm.**
  * ``sleep()`` (explicit "停/睡觉" / stop-intent) cancels the LLM turn AND
    broadcasts ``on_sleep`` — GraspPlugin aborts the physical motion on that
    hook. But even ``sleep()`` does not force-cancel the ``_pending_tool_tasks``
    set itself; the physical abort is the plugin's job via ``on_sleep``, and the
    tool-result wrapper task is only force-cancelled at ``shutdown()``.

Locking this prevents two regressions: (a) a barge-in silently killing a
legitimately-running motion, or (b) a "stop!" failing to reach the arm because
someone wired the abort to the tool-task cancel instead of the ``on_sleep`` hook.

PRODUCT-DECISION (confirmed 2026-06-14): barge-in deliberately does NOT stop the
arm — talking over the robot interrupts only its speech, never its motion. To
halt a motion the user must say 停/睡觉 (sleep → on_sleep → GraspPlugin abort).
No barge→arm-abort hook will be added; these tests pin that intended boundary.
"""
from __future__ import annotations

import asyncio

import pytest

from ovs_agent import Config, Session
from ovs_agent.app_base import BaseApp
from ovs_agent.slv_client import ServerToolCall
from ovs_agent.state import ConvState
from ovs_agent.tools import ToolRegistry


class _SpySLV:
    def __init__(self) -> None:
        self.aborted = 0
        self.tool_results: list[dict] = []
        self._ws = object()

    async def abort(self, *, keep_asr: bool = False) -> None:
        self.aborted += 1

    async def send_tool_result(self, call_id, name, *, ok, result=None, error=None):
        self.tool_results.append({"id": call_id, "name": name, "ok": ok})

    def session_gen(self) -> int:
        return 0


class _SpyAudio:
    def __init__(self) -> None:
        self.stop_count = 0
        self.is_playing = True

    async def stop_playback(self) -> None:
        self.stop_count += 1


class _GraspSpyPlugin:
    """Stands in for GraspPlugin: aborts the physical motion when it sees the
    ``on_sleep`` hook (the real abort path). Records what hooks fired."""

    name = "grasp-spy"

    def __init__(self) -> None:
        self.aborted = 0
        self.hooks: list[str] = []

    async def on_sleep(self, _data=None):
        self.hooks.append("on_sleep")
        self.aborted += 1


def _make_app(*, registry: ToolRegistry, plugins=None) -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.config = Config(system_prompt="SYS", server_loop=True, tools_max_iterations=4)
    app.tool_registry = registry
    app.session = Session()
    app.events = type("E", (), {"emit": lambda *a, **k: None})()
    app.modes = None
    app.slv = _SpySLV()
    app.audio = _SpyAudio()
    app.plugins = plugins if plugins is not None else []
    app._pending_tool_tasks = set()
    app._advertised_gen = 0
    app._state = ConvState.SPEAKING
    app._llm_turn_task = None
    app._sleep_task = None
    app._playback_drain_task = None
    app._wake_command_timeout_task = None
    app._asr_watchdog_task = None
    app._first_tts_seen = True
    app._eos_sent_this_turn = True
    app._last_user_utterance_text = ""
    app._broadcasts: list[tuple[str, object]] = []

    async def _record_broadcast(name, data=None):
        app._broadcasts.append((name, data))
        # fan out to plugins so the GraspSpy on_sleep hook actually fires.
        for p in app.plugins:
            hook = getattr(p, f"on_{name[3:]}" if name.startswith("on_") else name, None)
            if callable(hook):
                await hook(data)

    app._broadcast = _record_broadcast  # type: ignore[assignment]
    app._set_state = lambda s: setattr(app, "_state", s)  # type: ignore[assignment]
    return app


async def _dispatch_slow_tool(app: BaseApp, release: asyncio.Event) -> None:
    """Inject a SERVER_TOOL_CALL whose handler blocks on ``release`` — leaves a
    real in-flight task in ``_pending_tool_tasks`` (the running arm motion)."""
    await app._dispatch_one(ServerToolCall(id="g1", name="grasp_object", arguments={}))
    # _dispatch_one returns immediately (#F4): the handler is a background task.
    for _ in range(100):
        if app._pending_tool_tasks:
            break
        await asyncio.sleep(0.005)
    assert len(app._pending_tool_tasks) == 1, "slow tool should be in-flight"


# ── TC-011: barge-in interrupts speech but leaves the running tool alone ─


@pytest.mark.asyncio
async def test_tc011_barge_in_interrupts_speech_but_not_inflight_tool():
    reg = ToolRegistry()
    release = asyncio.Event()
    grasp = _GraspSpyPlugin()

    @reg.tool(name="grasp_object", description="grasp")
    async def grasp_object() -> dict:
        await release.wait()  # motion still running…
        return {"started": True, "action": "grasp_object"}

    app = _make_app(registry=reg, plugins=[grasp])
    await _dispatch_slow_tool(app, release)

    # An LLM/TTS turn is also in flight (the assistant is speaking).
    turn_started = asyncio.Event()

    async def _llm_turn():
        turn_started.set()
        await asyncio.sleep(100)

    app._llm_turn_task = asyncio.create_task(_llm_turn())
    await asyncio.wait_for(turn_started.wait(), timeout=1.0)

    # User barges in.
    await app._interrupt_current_turn_for_barge_in()

    # Speech IS interrupted:
    assert app._llm_turn_task.cancelled()
    assert app.slv.aborted == 1
    assert app.audio.stop_count == 1
    # …but the in-flight tool (arm motion) is NOT cancelled, and no on_sleep
    # fired, so GraspPlugin never aborted the motion.
    assert len(app._pending_tool_tasks) == 1, "barge-in must not cancel the tool task"
    assert all(not t.done() for t in app._pending_tool_tasks)
    assert grasp.aborted == 0, "talking over the robot must not abort the arm"
    assert ("on_sleep", None) not in app._broadcasts

    # Release → the tool completes normally and reports its result.
    release.set()
    while app._pending_tool_tasks:
        await asyncio.gather(*list(app._pending_tool_tasks), return_exceptions=True)
    assert [r["id"] for r in app.slv.tool_results] == ["g1"]
    assert app.slv.tool_results[0]["ok"] is True


# ── TC-007: explicit sleep() reaches the arm via on_sleep, not task-cancel ─


@pytest.mark.asyncio
async def test_tc007_sleep_aborts_arm_via_on_sleep_hook_not_tool_task_cancel():
    reg = ToolRegistry()
    release = asyncio.Event()
    grasp = _GraspSpyPlugin()

    @reg.tool(name="grasp_object", description="grasp")
    async def grasp_object() -> dict:
        await release.wait()
        return {"started": True, "action": "grasp_object"}

    app = _make_app(registry=reg, plugins=[grasp])
    await _dispatch_slow_tool(app, release)

    turn_started = asyncio.Event()

    async def _llm_turn():
        turn_started.set()
        await asyncio.sleep(100)

    app._llm_turn_task = asyncio.create_task(_llm_turn())
    await asyncio.wait_for(turn_started.wait(), timeout=1.0)

    # User says "停/睡觉" → sleep().
    await app.sleep()

    # sleep() IS a cancel: LLM turn cancelled, on_sleep broadcast, SLV aborted.
    assert app._state == ConvState.SLEEPING
    assert app._llm_turn_task.cancelled()
    assert ("on_sleep", None) in app._broadcasts
    # The arm is halted by the GraspPlugin reacting to on_sleep — NOT by
    # sleep() force-cancelling the tool-result task (which it never does).
    assert grasp.aborted == 1, "on_sleep must reach GraspPlugin to abort the motion"
    assert len(app._pending_tool_tasks) == 1, (
        "sleep() does not force-cancel the tool-result task; the physical abort "
        "is the on_sleep plugin hook's job (task teardown only happens at shutdown)"
    )

    # cleanup: release the still-pending tool task.
    release.set()
    while app._pending_tool_tasks:
        await asyncio.gather(*list(app._pending_tool_tasks), return_exceptions=True)
