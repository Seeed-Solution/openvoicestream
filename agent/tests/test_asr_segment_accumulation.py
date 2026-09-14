"""Tests for mid-utterance ASR segment accumulation.

The RK Qwen3-ASR backend ends an ASR turn on its own internal webrtcvad
endpoint (~400 ms of silence) even when the session declares client-owned
endpointing (``vad: none`` / ``client_vad_drive_eos``). Each such final
must be ACCUMULATED into the pending utterance; only the final that
answers the agent's own ``asr_eos`` is the authoritative utterance end
and dispatches the concatenated text to the LLM.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ovs_agent.app_base import BaseApp, _join_asr_segment_text
from ovs_agent.slv_client import ASRFinal
from ovs_agent.state import ConvState
from ovs_agent.event_bus import EventBus


def _fresh_app(*, drive_eos: bool = True) -> BaseApp:
    """BaseApp built via __new__ with a mock SLV + recorded LLM turns."""
    app = BaseApp.__new__(BaseApp)
    app.events = EventBus()
    app.plugins = []
    app.config = SimpleNamespace(
        asr_final_timeout_s=5.0,
        client_vad_drive_eos=drive_eos,
        pipeline_mode="always_on",
        stop_words=[],
        thinking_timeout_s=60.0,
    )
    app._state = ConvState.IDLE
    app._slv_reconnect_count = 0
    app._eos_sent_this_turn = False
    app._asr_watchdog_task = None
    app._stall_watchdog_task = None
    app._thinking_watchdog_task = None
    app._llm_turn_task = None
    app._first_tts_seen = False
    app._ptt_explicit_eos_pending = False
    app._last_user_utterance_text = ""
    app._pending_asr_utterance_text = ""
    app._wake_command_timeout_task = None
    app._sleep_task = None
    app._vad_state = "idle"
    app._vad_speech_ms = 0
    app._vad_silence_ms = 0
    app._vad_eos_sent = False
    app._client_vad = None
    app._stop_words_cache = None
    app.slv = MagicMock()
    app.slv.reconnect = AsyncMock()
    app.slv.abort = AsyncMock()
    app.audio = SimpleNamespace(
        is_playing=False,
        arm_for_next_turn=lambda: None,
        stop_playback=AsyncMock(),
    )
    app.llm_calls: list[str] = []

    async def _record(text, detected_language=None):
        app.llm_calls.append(text)

    app._run_user_utterance = _record
    return app


async def _settle(app: BaseApp) -> None:
    """Let the spawned llm-turn task run, then stand down watchdogs."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    app._cancel_thinking_watchdog()


def test_join_asr_segment_text_spacing_rules():
    # CJK stays unspaced on both sides of the seam.
    assert _join_asr_segment_text("帮我查一下", "N64的价格") == "帮我查一下N64的价格"
    assert _join_asr_segment_text("你好", "世界") == "你好世界"
    # Two ASCII word characters at the seam get exactly one space.
    assert _join_asr_segment_text("hello", "world") == "hello world"
    # Segments are stripped; empty-after-strip segments are ignored.
    assert _join_asr_segment_text("  hello  ", "") == "hello"
    assert _join_asr_segment_text("", " world ") == "world"


@pytest.mark.asyncio
async def test_two_mid_utterance_finals_concatenate_into_one_llm_turn():
    """Backend endpointing splits one sentence; the post-EOS final must
    dispatch ONE LLM turn with the concatenated text."""
    app = _fresh_app(drive_eos=True)
    app._set_state(ConvState.LISTENING)

    # Backend's internal 400ms endpoint fires mid-sentence (agent has
    # NOT sent asr_eos yet): accumulate, do not dispatch.
    await app._dispatch_one(ASRFinal(text="帮我查一下", session_complete=False))
    assert app.llm_calls == []
    assert app._pending_asr_utterance_text == "帮我查一下"
    assert app._state == ConvState.LISTENING

    # Client VAD silence -> asr_eos -> post-EOS final ends the utterance.
    app._eos_sent_this_turn = True
    app._set_state(ConvState.THINKING)
    await app._dispatch_one(ASRFinal(text="N64的价格", session_complete=False))
    await _settle(app)

    assert app.llm_calls == ["帮我查一下N64的价格"]
    assert app._last_user_utterance_text == "帮我查一下N64的价格"


@pytest.mark.asyncio
async def test_low_signal_final_between_segments_does_not_pollute_pending():
    app = _fresh_app(drive_eos=True)
    app._set_state(ConvState.LISTENING)
    await app._dispatch_one(ASRFinal(text="帮我查一下", session_complete=False))

    # Noise finals while the utterance is still open: dropped, pending
    # untouched, FSM left in LISTENING.
    await app._dispatch_one(ASRFinal(text="嗯", session_complete=False))
    assert app._pending_asr_utterance_text == "帮我查一下"
    assert app._state == ConvState.LISTENING
    await app._dispatch_one(ASRFinal(text="", session_complete=False))
    assert app._pending_asr_utterance_text == "帮我查一下"
    assert app.llm_calls == []

    app._eos_sent_this_turn = True
    app._set_state(ConvState.THINKING)
    await app._dispatch_one(ASRFinal(text="N64", session_complete=False))
    await _settle(app)
    assert app.llm_calls == ["帮我查一下N64"]


@pytest.mark.asyncio
async def test_single_final_regression_byte_identical():
    """One final per utterance (common case) must behave exactly as
    before: same text reaches the LLM, exactly one LLM call."""
    app = _fresh_app(drive_eos=True)
    app._set_state(ConvState.LISTENING)
    app._eos_sent_this_turn = True
    app._set_state(ConvState.THINKING)
    await app._dispatch_one(ASRFinal(text="你好世界", session_complete=False))
    await _settle(app)
    assert app.llm_calls == ["你好世界"]
    assert app._last_user_utterance_text == "你好世界"


@pytest.mark.asyncio
async def test_server_vad_mode_still_dispatches_every_final_immediately():
    """client_vad_drive_eos=False: no client EOS exists, so every final
    is a turn end (unchanged legacy behaviour)."""
    app = _fresh_app(drive_eos=False)
    app._set_state(ConvState.LISTENING)
    await app._dispatch_one(ASRFinal(text="第一段", session_complete=False))
    await _settle(app)
    app._set_state(ConvState.LISTENING)
    await app._dispatch_one(ASRFinal(text="第二段", session_complete=False))
    await _settle(app)
    assert app.llm_calls == ["第一段", "第二段"]


@pytest.mark.asyncio
async def test_pending_cleared_after_dispatch_no_glue_onto_next_turn():
    app = _fresh_app(drive_eos=True)
    app._set_state(ConvState.LISTENING)
    await app._dispatch_one(ASRFinal(text="前半句", session_complete=False))
    app._eos_sent_this_turn = True
    app._set_state(ConvState.THINKING)
    await app._dispatch_one(ASRFinal(text="后半句", session_complete=False))
    await _settle(app)
    assert app.llm_calls == ["前半句后半句"]
    assert app._pending_asr_utterance_text == ""

    # A later standalone final is a fresh utterance — no gluing.
    app._set_state(ConvState.LISTENING)
    app._eos_sent_this_turn = True
    app._set_state(ConvState.THINKING)
    await app._dispatch_one(ASRFinal(text="下一句", session_complete=False))
    await _settle(app)
    assert app.llm_calls == ["前半句后半句", "下一句"]


@pytest.mark.asyncio
async def test_empty_post_eos_final_dispatches_accumulated_pending():
    """RK commonly returns empty text for the post-EOS final when it
    already finalized the audio on its own endpoint: the accumulated
    segments must still be dispatched, not dropped."""
    app = _fresh_app(drive_eos=True)
    app._set_state(ConvState.LISTENING)
    await app._dispatch_one(ASRFinal(text="第一部分", session_complete=False))
    app._eos_sent_this_turn = True
    app._set_state(ConvState.THINKING)
    await app._dispatch_one(ASRFinal(text="", session_complete=False))
    await _settle(app)
    assert app.llm_calls == ["第一部分"]
    assert app._pending_asr_utterance_text == ""


@pytest.mark.asyncio
async def test_duplicate_close_final_does_not_repeat_accumulated_text():
    """Server multi_utterance close-out finals repeat the last streamed final
    (`duplicate_of_streamed = final_text == last_streamed_final`, server/main.py).
    When a segment was already accumulated, that close final carries NO new
    text — the LLM must see the utterance exactly once, not twice.
    """
    app = _fresh_app(drive_eos=True)
    app._set_state(ConvState.LISTENING)

    # Backend endpoint final: accumulated, not dispatched.
    await app._dispatch_one(ASRFinal(text="帮我查一下", session_complete=False))
    assert app._pending_asr_utterance_text == "帮我查一下"

    # Client EOS, then the close-out final duplicating that same text.
    app._eos_sent_this_turn = True
    app._set_state(ConvState.THINKING)
    await app._dispatch_one(
        ASRFinal(
            text="帮我查一下",
            session_complete=True,
            duplicate_of_streamed=True,
        )
    )
    await _settle(app)

    assert app.llm_calls == ["帮我查一下"], (
        "a duplicate close final must not be joined onto the pending segments"
    )


@pytest.mark.asyncio
async def test_watchdog_drops_pending_segments_when_turn_is_abandoned():
    """No post-EOS final arrives (empty/dropped final): the turn is abandoned,
    so the accumulated mid-utterance segments must NOT survive into the next
    utterance (they would be glued onto its text)."""
    app = _fresh_app(drive_eos=True)
    app.config.asr_final_timeout_s = 0.01
    app._set_state(ConvState.LISTENING)

    await app._dispatch_one(ASRFinal(text="帮我查一下", session_complete=False))
    assert app._pending_asr_utterance_text == "帮我查一下"

    # EOS sent, no final ever arrives -> watchdog fires.
    app._eos_sent_this_turn = True
    app._set_state(ConvState.THINKING)
    app._asr_watchdog_task = asyncio.create_task(app._asr_final_watchdog())
    await asyncio.sleep(0.05)

    assert app._pending_asr_utterance_text == ""
    assert app._state == ConvState.IDLE

    # The next utterance must start clean.
    app._set_state(ConvState.LISTENING)
    await app._dispatch_one(ASRFinal(text="新问题", session_complete=False))
    assert app._pending_asr_utterance_text == "新问题"
    assert app.llm_calls == []
