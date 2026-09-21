"""P0-A regression: dispatch loop must not block during an LLM turn.

While `on_user_utterance` streams tokens, the dispatch loop must still
process queued TTSAudio (so the speaker plays) and ASRPartial (so
barge-in cancels the in-flight LLM turn, aborts SLV TTS, AND silences the speaker —
per HIGH-2 + commit fa13846; otherwise streaming tokens immediately
re-start TTS and undo the barge-in stop_playback).

NB: barge-in deliberately does NOT reconnect SLV. Use the in-band
`abort` control so SLV cancels TTS / drains queued sentences while the
WebSocket remains alive for ASR continuity.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ovs_agent.app_base import BaseApp
from ovs_agent.slv_client import ASRFinal, ASRPartial, TTSAudio
from ovs_agent.state import ConvState


class _FakeAudio:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.is_playing = False
        self.stop_calls = 0
        self.armed = 0

    async def play(self, pcm: bytes) -> None:
        self.is_playing = True
        self.played.append(pcm)

    async def stop_playback(self) -> None:
        self.is_playing = False
        self.stop_calls += 1

    def arm_for_next_turn(self) -> None:
        self.armed += 1

    def set_output_sample_rate(self, sr: int) -> None:
        pass


class _FakeSLV:
    def __init__(self) -> None:
        self.aborted = 0
        self.reconnects = 0

    async def abort(self, *, keep_asr: bool = False) -> None:
        self.aborted += 1

    async def reconnect(self) -> None:
        self.reconnects += 1


@pytest.mark.asyncio
async def test_dispatch_does_not_block_on_llm_turn():
    app = BaseApp.__new__(BaseApp)
    app.audio = _FakeAudio()
    app.slv = _FakeSLV()
    app.plugins = []
    app._first_tts_seen = False
    app._llm_turn_task = None
    app._state = ConvState.IDLE
    app._slv_reconnect_count = 0
    app._eos_sent_this_turn = False
    app._asr_watchdog_task = None
    app._thinking_watchdog_task = None
    app._sleep_task = None
    app._ptt_explicit_eos_pending = False
    app._vad_state = "idle"
    app._vad_speech_ms = 0
    app._vad_silence_ms = 0
    app._vad_eos_sent = False
    app._client_vad = None
    app._mic_rms_broadcast_task = None
    app._stop_words_cache = None
    app.config = SimpleNamespace(
        pipeline_mode="always_on",
        sleep_timeout_s=30.0,
        thinking_timeout_s=20.0,
        barge_in_min_chars=1,
        barge_in_min_speaking_ms=0,
    )

    llm_started = asyncio.Event()
    llm_release = asyncio.Event()
    seen_during_llm: dict[str, bool] = {"audio": False, "stopped": False}

    async def slow_on_user_utterance(text: str, detected_language: str | None = None) -> None:
        llm_started.set()
        try:
            await llm_release.wait()  # block until test releases
        except asyncio.CancelledError:
            raise

    app.on_user_utterance = slow_on_user_utterance  # type: ignore[assignment]

    # 1. ASRFinal -> spawns LLM turn task (non-blocking).
    await app._dispatch_one(ASRFinal(text="hi", duplicate_of_streamed=False))
    await asyncio.wait_for(llm_started.wait(), timeout=1.0)

    # 2. While LLM is "running", dispatch a TTSAudio -- must reach speaker.
    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=24000))
    seen_during_llm["audio"] = len(app.audio.played) == 1
    reconnects_before_barge = app.slv.reconnects

    # 3. While LLM is "running", dispatch ASRPartial -- must stop playback,
    #    abort SLV TTS without reconnecting, and cancel the LLM turn.
    app.audio.is_playing = True
    await app._dispatch_one(ASRPartial(text="wait"))
    seen_during_llm["stopped"] = app.audio.stop_calls >= 1

    # HIGH-2 fix: barge-in MUST cancel the in-flight LLM turn so streaming
    # tokens don't immediately restart TTS and undo stop_playback.
    assert app._llm_turn_task.done(), "LLM turn should be cancelled by barge-in"
    assert app._llm_turn_task.cancelled() or isinstance(
        app._llm_turn_task.exception(), asyncio.CancelledError
    ), "LLM turn should be cancelled, not completed normally"
    assert seen_during_llm["audio"], "TTSAudio not played while LLM streaming"
    assert seen_during_llm["stopped"], "ASRPartial did not stop playback during LLM stream"
    assert app.slv.aborted == 1
    assert app.slv.reconnects == reconnects_before_barge, (
        "barge-in should not reconnect SLV in the hot path"
    )

    llm_release.set()  # no-op now; task already cancelled
