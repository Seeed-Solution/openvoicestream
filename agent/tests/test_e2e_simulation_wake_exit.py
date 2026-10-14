"""End-to-end simulation test for complete wake-greeting, barge-in, and exit-to-sleep lifecycle."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ovs_agent.app_base import BaseApp
from ovs_agent.config import Config
from ovs_agent.event_bus import EventBus
from ovs_agent.state import ConvState
from ovs_agent.slv_client import ASRFinal, TTSAudio, TTSDone, ASRPartial


@pytest.mark.asyncio
async def test_full_lifecycle_simulation():
    """Simulate complete user journey:
    1. Agent boots in SLEEPING state (wake_word pipeline mode).
    2. Wake occurs with wake_reply_text="在呢":
       - state flips to IDLE
       - speaks "在呢"
       - greeting finishes (TTSDone), state remains IDLE (listening for user command)
    3. User speaks a command -> agent enters THINKING -> starts SPEAKING long reply
    4. User barges in with "退下" while agent is SPEAKING:
       - barge-in interrupts playback immediately
       - user utterance "退下" decoded
       - exit intent matched -> speaks exit_reply_text="那我先退下了"
       - exit reply finishes -> agent enters SLEEPING state!
    5. While SLEEPING, user speech/noise is dropped and does not trigger LLM.
    """
    cfg = Config(
        pipeline_mode="wake_word",
        wake_reply_text="在呢",
        exit_reply_text="那我先退下了",
        wake_command_single_turn=False,
    )

    app = BaseApp.__new__(BaseApp)
    app.config = cfg
    app.events = EventBus()
    app.plugins = []
    app._state = ConvState.SLEEPING
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
    app._in_wake_greeting = False
    app._wake_command_retry_after_no_final = False

    spoken_lines = []
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
    async def mock_speak(text, conversation="none"):
        spoken_lines.append(text)
    app.slv = SimpleNamespace(
        abort=AsyncMock(),
        reconnect=AsyncMock(),
        speak=AsyncMock(side_effect=mock_speak),
        send_text=AsyncMock(),
        flush_tts=AsyncMock(),
        is_healthy=MagicMock(return_value=True),
        seconds_since_activity=MagicMock(return_value=0.5),
        truncate_active_response=AsyncMock(),
    )
    app.session = SimpleNamespace(history=[])

    user_utterances_handled = []
    async def mock_run_user_utterance(text, language=None):
        user_utterances_handled.append(text)
    app._run_user_utterance = mock_run_user_utterance

    # ── Step 1: Booted in SLEEPING ──
    assert app._state == ConvState.SLEEPING

    # ── Step 2: Wake up ──
    await app.wake(source="openwakeword")
    assert app._state == ConvState.IDLE
    assert spoken_lines == ["在呢"]

    # Greeting plays audio then completes
    await app._dispatch_one(TTSAudio(pcm=b"\x00" * 320, sample_rate=16000))
    assert app._state == ConvState.SPEAKING
    await app._dispatch_one(TTSDone(session_complete=False))
    assert app._state == ConvState.IDLE
    assert app._in_wake_greeting is False

    # ── Step 3: User speaks a regular command ──
    await app._dispatch_one(ASRFinal(text="今天天气怎么样", session_complete=False))
    await asyncio.sleep(0.01)
    assert app._state == ConvState.THINKING
    assert user_utterances_handled == ["今天天气怎么样"]


    # Assistant starts speaking the weather report
    await app._dispatch_one(TTSAudio(pcm=b"\x00" * 320, sample_rate=16000))
    assert app._state == ConvState.SPEAKING
    app.audio.is_playing = True

    # ── Step 4: User barges in and says "退下" ──
    # User begins speaking (ASRPartial) -> barge-in interrupts
    app._speaking_since_ts = 0.0  # bypass AEC delay guard for test
    await app._dispatch_one(ASRPartial(text="退下"))
    assert app._state == ConvState.BARGED_IN

    app.audio.stop_playback.assert_awaited()
    app.slv.abort.assert_awaited()

    # User finishes utterance -> ASRFinal arrives
    await app._dispatch_one(ASRFinal(text="退下", session_complete=True))
    # Handled as exit intent: speaks farewell "那我先退下了"
    assert "那我先退下了" in spoken_lines
    assert app._sleep_after_assistant_turn is True

    # Farewell completes playback
    await app._dispatch_one(TTSDone(session_complete=True))
    # Agent transitions to SLEEPING!
    assert app._state == ConvState.SLEEPING
    assert app._sleep_after_assistant_turn is False

    # ── Step 5: Background speech while SLEEPING is dropped ──
    # User/room speech while SLEEPING must NOT trigger _run_user_utterance
    curr_turns = len(user_utterances_handled)
    await app._dispatch_one(ASRFinal(text="今天星期几", session_complete=False))
    assert len(user_utterances_handled) == curr_turns
    assert app._state == ConvState.SLEEPING
