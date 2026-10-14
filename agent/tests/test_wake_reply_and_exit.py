"""Tests for wake reply text, exit intent, and stop-to-sleep behaviour."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ovs_agent.app_base import BaseApp
from ovs_agent.config import Config
from ovs_agent.event_bus import EventBus
from ovs_agent.state import ConvState
from ovs_agent.slv_client import ASRFinal


def _stub_app(config: Config | None = None) -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.config = config if config is not None else Config()
    app.events = EventBus()
    app.plugins = []
    app._state = ConvState.IDLE
    app._slv_reconnect_count = 0
    app._llm_turn_task = None
    app._first_tts_seen = False
    app._vad_state = "idle"
    app._vad_speech_ms = 0
    app._vad_silence_ms = 0
    app._vad_eos_sent = False
    app._client_vad = None
    app._eos_sent_this_turn = False
    app._asr_watchdog_task = None
    app._thinking_watchdog_task = None
    app._sleep_task = None
    app._ptt_explicit_eos_pending = False
    app._mic_rms_broadcast_task = None
    app._stop_words_cache = None
    app._sleep_after_assistant_turn = False
    app.audio = SimpleNamespace(
        output_sr=16000,
        is_playing=False,
        stop_playback=AsyncMock(),
        play=AsyncMock(),
        play_notification=MagicMock(),
        arm_for_next_turn=MagicMock(),
        playback_position_ms=MagicMock(return_value=0),
        set_output_sample_rate=MagicMock(),
    )

    app.slv = SimpleNamespace(
        abort=AsyncMock(),
        reconnect=AsyncMock(),
        speak=AsyncMock(),
        send_text=AsyncMock(),
        flush_tts=AsyncMock(),
        is_healthy=MagicMock(return_value=True),
        seconds_since_activity=MagicMock(return_value=0.5),
        truncate_active_response=AsyncMock(),
    )
    app.session = SimpleNamespace(history=[])
    return app


def test_is_exit_intent_matches_chinese():
    app = _stub_app()
    assert app._is_exit_intent("退下") is True
    assert app._is_exit_intent("退下吧") is True
    assert app._is_exit_intent("退下。") is True
    assert app._is_exit_intent("退出") is True
    assert app._is_exit_intent("请退下") is True
    assert app._is_exit_intent("休眠") is True
    assert app._is_exit_intent("结束对话") is True
    assert app._is_exit_intent("再见") is True
    assert app._is_exit_intent("再见！") is True
    assert app._is_exit_intent("拜拜") is True
    assert app._is_exit_intent("拜") is True
    assert app._is_exit_intent("下次聊") is True
    assert app._is_exit_intent("回头见") is True


def test_is_exit_intent_matches_english():
    app = _stub_app()
    assert app._is_exit_intent("exit") is True
    assert app._is_exit_intent("Exit.") is True
    assert app._is_exit_intent("quit") is True
    assert app._is_exit_intent("goodbye") is True
    assert app._is_exit_intent("bye") is True
    assert app._is_exit_intent("bye now") is True
    assert app._is_exit_intent("exit please") is True
    assert app._is_exit_intent("sleep") is True


def test_is_exit_intent_matches_wake_prefix():
    app = _stub_app()
    app.config.wake_phrases = ["hey jarvis", "你好小智"]
    assert app._is_exit_intent("Hey Jarvis 退下") is True
    assert app._is_exit_intent("hey jarvis, bye") is True
    assert app._is_exit_intent("你好小智 再见") is True


def test_is_exit_intent_rejects_negatives():
    app = _stub_app()
    assert app._is_exit_intent("不要退下") is False
    assert app._is_exit_intent("退出系统可以吗") is False
    assert app._is_exit_intent("exiting") is False
    assert app._is_exit_intent("") is False
    assert app._is_exit_intent("   ") is False


@pytest.mark.asyncio
async def test_wake_with_reply_text():
    cfg = Config(pipeline_mode="wake_word", wake_reply_text="在呢")
    app = _stub_app(cfg)
    app._state = ConvState.SLEEPING

    received_events = []
    plugin = SimpleNamespace(name="test_plugin", on_wake_reply=lambda d: received_events.append(d))
    app.plugins.append(plugin)

    await app.wake(source="openwakeword")

    assert app._state == ConvState.IDLE
    app.slv.speak.assert_awaited_once_with("在呢", conversation="none")
    assert received_events == ["在呢"]


@pytest.mark.asyncio
async def test_wake_with_reply_text_list():
    cfg = Config(pipeline_mode="wake_word", wake_reply_text=["在呢", "我在"])
    app = _stub_app(cfg)
    app._state = ConvState.SLEEPING

    await app.wake(source="openwakeword")

    assert app._state == ConvState.IDLE
    app.slv.speak.assert_awaited_once()
    called_text = app.slv.speak.await_args.args[0]
    assert called_text in ("在呢", "我在")


@pytest.mark.asyncio
async def test_wake_without_reply_text():
    cfg = Config(pipeline_mode="wake_word", wake_reply_text="")
    app = _stub_app(cfg)
    app._state = ConvState.SLEEPING

    await app.wake(source="openwakeword")

    assert app._state == ConvState.IDLE
    app.slv.speak.assert_not_called()


@pytest.mark.asyncio
async def test_exit_intent_transitions_to_sleeping():
    app = _stub_app()
    app._set_state(ConvState.IDLE)

    exit_events = []
    plugin = SimpleNamespace(name="test_plugin", on_user_exit_intent=lambda d: exit_events.append(d))
    app.plugins.append(plugin)


    on_utt_called = False
    async def on_user_utterance(text):
        nonlocal on_utt_called
        on_utt_called = True
    app.on_user_utterance = on_user_utterance

    evt = ASRFinal(text="退下", session_complete=True)
    await app._dispatch_one(evt)

    assert on_utt_called is False
    assert app.session.history == []
    app.slv.abort.assert_awaited()
    app.audio.stop_playback.assert_awaited()
    assert app._state == ConvState.SLEEPING
    assert exit_events == ["退下"]


@pytest.mark.asyncio
async def test_exit_intent_with_exit_reply():
    cfg = Config(exit_reply_text="那我先退下了")
    app = _stub_app(cfg)
    app._set_state(ConvState.IDLE)

    evt = ASRFinal(text="退下", session_complete=True)
    await app._dispatch_one(evt)

    # Spoke farewell
    app.slv.speak.assert_awaited_once_with("那我先退下了", conversation="none")
    assert app._sleep_after_assistant_turn is True

    # Assistant finishes speaking farewell -> completes turn -> enters SLEEPING
    await app._complete_assistant_turn()
    assert app._state == ConvState.SLEEPING
    assert app._sleep_after_assistant_turn is False


@pytest.mark.asyncio
async def test_stop_to_sleep_enabled():
    cfg = Config(stop_to_sleep=True)
    app = _stub_app(cfg)
    app._set_state(ConvState.IDLE)

    evt = ASRFinal(text="停下", session_complete=True)
    await app._dispatch_one(evt)

    assert app._state == ConvState.SLEEPING


@pytest.mark.asyncio
async def test_stop_to_sleep_disabled():
    cfg = Config(stop_to_sleep=False)
    app = _stub_app(cfg)
    app._set_state(ConvState.IDLE)

    evt = ASRFinal(text="停下", session_complete=True)
    await app._dispatch_one(evt)

    assert app._state == ConvState.IDLE


@pytest.mark.asyncio
async def test_traditional_chinese_stop_and_exit():
    """Verify that traditional Chinese variants like '別說了', '再見' match."""
    cfg = Config(exit_reply_text="那我先退下了")
    app = _stub_app(cfg)
    app._set_state(ConvState.IDLE)

    # Traditional "別說了。" -> stop intent
    evt = ASRFinal(text="別說了。", session_complete=True)
    await app._dispatch_one(evt)
    assert app._state == ConvState.IDLE

    # Traditional "再見！" -> exit intent
    evt2 = ASRFinal(text="再見！", session_complete=True)
    await app._dispatch_one(evt2)
    app.slv.speak.assert_awaited_with("那我先退下了", conversation="none")
    assert app._sleep_after_assistant_turn is True



@pytest.mark.asyncio
async def test_single_char_stop_and_exit_not_dropped_as_low_signal():
    """'停' and '拜' are 1 character, which would normally be dropped as low-signal,
    but must be preserved and processed as stop/exit intents."""
    app = _stub_app()
    app._set_state(ConvState.IDLE)

    # Test "停"
    evt_stop = ASRFinal(text="停", session_complete=True)
    await app._dispatch_one(evt_stop)
    app.slv.abort.assert_awaited()
    assert app._state == ConvState.IDLE

    # Test "拜"
    evt_exit = ASRFinal(text="拜", session_complete=True)
    await app._dispatch_one(evt_exit)
    assert app._state == ConvState.SLEEPING


@pytest.mark.asyncio
async def test_wake_command_single_turn_greeting_does_not_prematurely_sleep():
    """In wake_command_single_turn mode, the completion of wake greeting
    must NOT transition to SLEEPING. It must keep listening for the command."""
    from ovs_agent.slv_client import TTSDone

    cfg = Config(
        pipeline_mode="wake_word",
        wake_command_single_turn=True,
        wake_reply_text="在呢",
        wake_command_timeout_s=5.0,
    )
    app = _stub_app(cfg)
    app._state = ConvState.SLEEPING

    await app.wake(source="openwakeword")
    assert app._state == ConvState.IDLE
    assert app._in_wake_greeting is True

    # Greeting audio plays, flipping state to SPEAKING
    from ovs_agent.slv_client import TTSAudio
    await app._dispatch_one(TTSAudio(pcm=b"\x00" * 320, sample_rate=16000))
    assert app._state == ConvState.SPEAKING

    # SLV finishes playing the greeting
    await app._dispatch_one(TTSDone(session_complete=False))

    # Must be IDLE listening for the user's command, NOT SLEEPING and NOT stuck in SPEAKING!
    assert app._state == ConvState.IDLE
    assert app._in_wake_greeting is False



@pytest.mark.asyncio
async def test_exit_reply_rearms_playback_and_clears_barged_in():
    """Exit reply must rearm playback and clear BARGED_IN so farewell is audible."""
    cfg = Config(exit_reply_text="那我先退下了")
    app = _stub_app(cfg)
    app._set_state(ConvState.BARGED_IN)

    evt = ASRFinal(text="退下", session_complete=True)
    await app._dispatch_one(evt)

    # Audio playback was re-armed
    assert app.audio.arm_for_next_turn.call_count >= 2
    # State reset from BARGED_IN to IDLE so TTS audio isn't discarded
    assert app._state == ConvState.IDLE

    assert app._sleep_after_assistant_turn is True
    app.slv.speak.assert_awaited_once_with("那我先退下了", conversation="none")

