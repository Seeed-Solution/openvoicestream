"""Lifecycle: idle→sleep→wake + sleep-vs-command-return distinction.

Closes the LC (lifecycle) coverage gap from
``agent/docs/voice-scenario-test-catalog.md``. The wake-time *reconnect
policy* is already locked by ``test_wake_reconnect_policy.py`` (LC-003 long-idle
reconnect, LC-006 unhealthy/failed-reconnect-stays-SLEEPING). What had NO
coverage is the rest of the lifecycle FSM:

  * LC-002 — the auto-sleep timer (``_sleep_after``) firing IDLE→SLEEPING after
    ``sleep_timeout_s``, the "in-flight turn delays sleep" guard, and a re-wake
    re-arming the timer.
  * LC-008 — explicit ``sleep()`` is a CANCEL action (broadcasts ``on_sleep`` →
    GraspPlugin aborts an in-flight grasp, cancels the LLM/tool turn, aborts
    SLV), whereas ``_return_to_sleep_after_command_turn()`` closes the
    wake-command window WITHOUT cancelling the in-flight tool. Mixing these up
    would make "say sleep to stop the arm" silently not stop it (or, inversely,
    make a normal command-window close kill a running grasp).

These drive the REAL ``sleep()`` / ``wake()`` / ``_sleep_after`` /
``_reset_sleep_timer`` methods; only leaf I/O (slv / audio / broadcast / tones /
``_set_state``) is stubbed, matching ``test_wake_reconnect_policy.py``.
"""
from __future__ import annotations

import asyncio

import pytest

from ovs_agent import Config
from ovs_agent.app_base import BaseApp
from ovs_agent.state import ConvState


class _SpySLV:
    def __init__(self) -> None:
        self.abort_count = 0
        self.reconnect_count = 0

    def is_healthy(self) -> bool:
        return True

    def seconds_since_activity(self) -> float:
        return 5.0  # hot turn → wake() won't force a reconnect

    def is_reconnecting(self) -> bool:
        return False

    async def abort(self, *, keep_asr: bool = False) -> None:
        self.abort_count += 1

    async def reconnect(self) -> None:
        self.reconnect_count += 1


class _SpyAudio:
    def __init__(self) -> None:
        self.stop_count = 0

    async def stop_playback(self) -> None:
        self.stop_count += 1

    def arm_for_next_turn(self) -> None:
        return None


def _make_lifecycle_app(
    *,
    state: ConvState,
    pipeline_mode: str = "wake_word",
    sleep_timeout_s: float = 30.0,
) -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.config = Config(
        system_prompt="SYS",
        pipeline_mode=pipeline_mode,
        sleep_timeout_s=sleep_timeout_s,
    )
    app._state = state
    app.slv = _SpySLV()
    app.audio = _SpyAudio()
    app._llm_turn_task = None
    app._sleep_task = None
    app._playback_drain_task = None
    app._wake_command_timeout_task = None
    app._first_tts_seen = False
    app._eos_sent_this_turn = False
    app._wake_command_retry_after_no_final = False
    app._slv_reconnect_count = 0
    app._broadcasts: list[tuple[str, object]] = []

    async def _record_broadcast(name, data=None):
        app._broadcasts.append((name, data))

    async def _noop_async(*a, **k):
        return None

    app._broadcast = _record_broadcast  # type: ignore[assignment]
    app._readvertise_after_reconnect = _noop_async  # type: ignore[assignment]
    # Stub _set_state to the bare transition (matches test_wake_reconnect_policy);
    # the lifecycle assertions are at the FSM level, not _set_state internals.
    app._set_state = lambda s: setattr(app, "_state", s)  # type: ignore[assignment]
    app._play_wake_tone = lambda: None  # type: ignore[assignment]
    app._play_sleep_tone = lambda: None  # type: ignore[assignment]
    return app


async def _spawn_inflight_turn(app: BaseApp) -> asyncio.Task:
    """Attach a never-finishing task to ``_llm_turn_task`` (an in-flight
    LLM/tool turn, e.g. a running grasp) and wait until it's actually running."""
    started = asyncio.Event()

    async def _long_turn():
        started.set()
        await asyncio.sleep(100)

    task = asyncio.create_task(_long_turn())
    app._llm_turn_task = task
    await asyncio.wait_for(started.wait(), timeout=1.0)
    return task


# ── LC-008: explicit sleep() cancels the in-flight tool; command-return does not


@pytest.mark.asyncio
async def test_lc008_explicit_sleep_cancels_inflight_turn_and_broadcasts():
    """LC-008: ``sleep()`` (user/admin "睡觉/停") is a CANCEL: it broadcasts
    ``on_sleep`` (GraspPlugin aborts the grasp on this hook), cancels the
    in-flight LLM/tool turn, aborts the SLV stream, and stops playback."""
    app = _make_lifecycle_app(state=ConvState.SPEAKING)
    task = await _spawn_inflight_turn(app)

    await app.sleep()

    assert app._state == ConvState.SLEEPING
    assert ("on_sleep", None) in app._broadcasts, app._broadcasts
    assert task.cancelled(), "the in-flight tool/LLM turn must be cancelled"
    assert app.slv.abort_count == 1
    assert app.audio.stop_count == 1


@pytest.mark.asyncio
async def test_lc008_command_return_does_not_cancel_tool_or_broadcast_sleep():
    """LC-008 (contrast): ``_return_to_sleep_after_command_turn()`` closes the
    post-wake command window but is intentionally NOT a cancel — it must NOT
    broadcast ``on_sleep`` (so an in-flight grasp keeps running) and must NOT
    cancel the LLM/tool turn or abort SLV. This is the exact distinction the
    catalog calls out; conflating the two would either strand the arm or kill a
    legitimately-running motion."""
    app = _make_lifecycle_app(state=ConvState.IDLE)
    task = await _spawn_inflight_turn(app)

    await app._return_to_sleep_after_command_turn()

    assert app._state == ConvState.SLEEPING
    assert ("on_sleep", None) not in app._broadcasts, app._broadcasts
    assert not task.done(), "command-return must NOT cancel the in-flight tool"
    assert app.slv.abort_count == 0
    assert app.audio.stop_count == 0

    task.cancel()  # cleanup


@pytest.mark.asyncio
async def test_lc_sleep_is_idempotent_when_already_sleeping():
    """``sleep()`` is a no-op (no second on_sleep, no abort) when already
    SLEEPING — a re-issued sleep must not re-cancel / re-abort."""
    app = _make_lifecycle_app(state=ConvState.SLEEPING)
    await app.sleep()
    assert app._broadcasts == []
    assert app.slv.abort_count == 0


# ── LC-002: auto-sleep timer fires IDLE→SLEEPING; re-wake re-arms it ──


@pytest.mark.asyncio
async def test_lc002_idle_auto_sleeps_after_timeout_then_wake_rearms():
    """LC-002: after ``sleep_timeout_s`` of IDLE the auto-sleep timer fires
    (IDLE→SLEEPING, ``on_sleep`` broadcast), then a wake re-arms a fresh
    timer."""
    app = _make_lifecycle_app(
        state=ConvState.IDLE, pipeline_mode="wake_word", sleep_timeout_s=0.05
    )
    app._reset_sleep_timer()
    assert app._sleep_task is not None and not app._sleep_task.done()

    await asyncio.sleep(0.15)  # past the 0.05s budget → timer fires sleep()

    assert app._state == ConvState.SLEEPING
    assert ("on_sleep", None) in app._broadcasts

    # Re-wake (hot, healthy → no reconnect) returns to IDLE and re-arms the timer.
    await app.wake(source="external")
    assert app._state == ConvState.IDLE
    assert app.slv.reconnect_count == 0  # hot/healthy wake doesn't churn the WS
    assert app._sleep_task is not None and not app._sleep_task.done()

    app._sleep_task.cancel()  # cleanup


@pytest.mark.asyncio
async def test_lc002_inflight_turn_delays_auto_sleep():
    """LC-002 guard: ``_sleep_after`` only sleeps if still IDLE — if a turn is
    in flight (THINKING) when the timer expires, the agent stays awake (the
    turn delays sleep) rather than cutting off mid-response."""
    app = _make_lifecycle_app(
        state=ConvState.THINKING, pipeline_mode="wake_word", sleep_timeout_s=0.05
    )
    app._reset_sleep_timer()

    await asyncio.sleep(0.15)

    assert app._state == ConvState.THINKING, "in-flight turn must not be slept on"
    assert ("on_sleep", None) not in app._broadcasts


@pytest.mark.asyncio
async def test_lc002_always_on_never_arms_sleep_timer():
    """always_on mode has no auto-sleep: ``_reset_sleep_timer`` is a no-op so
    the agent never drops to SLEEPING on its own (legacy continuous mode)."""
    app = _make_lifecycle_app(
        state=ConvState.IDLE, pipeline_mode="always_on", sleep_timeout_s=0.05
    )
    app._reset_sleep_timer()
    assert app._sleep_task is None

    await asyncio.sleep(0.1)
    assert app._state == ConvState.IDLE
    assert app._broadcasts == []
