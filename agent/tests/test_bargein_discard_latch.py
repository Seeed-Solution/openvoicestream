"""Regression: after barge-in, SLV's tail-end TTS frames must be dropped,
not re-queued by `audio.play()` and not re-flip state to SPEAKING.

Before the fix, `stop_playback()` only drained the queue + cleared
`_is_playing`. The very next `TTSAudio` from SLV's already-buffered tail
called `audio.play()` -> `_is_playing=True` + re-queue -> playback
resumed within ~100ms. The TTSAudio dispatch handler also flipped the
state machine back from BARGED_IN -> SPEAKING because `_first_tts_seen`
had been reset.
"""
from __future__ import annotations

import asyncio

import pytest

from types import SimpleNamespace

from ovs_agent.app_base import BaseApp
from ovs_agent.audio_io import AudioIO
from ovs_agent.slv_client import (
    ASRFinal,
    ASRPartial,
    TTSAudio,
    TTSDone,
    TTSSentenceDone,
    TTSStarted,
)
from ovs_agent.state import ConvState


@pytest.mark.asyncio
async def test_stop_playback_latches_discard_for_pcm():
    audio = AudioIO()
    # No _ensure_output() / sounddevice involved: stop_playback just toggles
    # the latch and play() short-circuits before touching the output stream.
    await audio.stop_playback()
    assert audio._discard_playback is True
    # play() must drop the chunk + must NOT re-flip is_playing.
    await audio.play(b"\x01\x00" * 8)
    assert audio.is_playing is False
    audio.arm_for_next_turn()
    assert audio._discard_playback is False


class _FakeAudio:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.is_playing = False
        self.armed = 0
        self.discard = False

    async def play(self, pcm: bytes) -> None:
        if self.discard:
            return
        self.is_playing = True
        self.played.append(pcm)

    async def stop_playback(self) -> None:
        self.is_playing = False
        self.discard = True

    def arm_for_next_turn(self) -> None:
        self.armed += 1
        self.discard = False

    def set_output_sample_rate(self, sr: int) -> None:
        pass

    def mark_playback_done(self) -> None:
        # Remote TTS is done sending, but local buffered audio may still be
        # draining through the speaker.
        pass

    def playback_position_ms(self, response_id=None) -> int:
        return 456


class _FakeSLV:
    def __init__(self) -> None:
        self.aborted = 0
        self.reconnects = 0
        self.truncations = []

    async def abort(self, *, keep_asr: bool = False) -> None:
        self.aborted += 1

    async def reconnect(self) -> None:
        self.reconnects += 1

    async def truncate_active_response(self, audio_end_ms: int) -> None:
        self.truncations.append(audio_end_ms)


def _make_app() -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.audio = _FakeAudio()
    app.slv = _FakeSLV()
    app.plugins = []
    app._first_tts_seen = False
    app._llm_turn_task = None
    app._state = ConvState.IDLE
    app._slv_reconnect_count = 0
    app.config = SimpleNamespace(
        pipeline_mode="always_on",
        sleep_timeout_s=30.0,
        barge_in_min_chars=1,
        barge_in_min_speaking_ms=0,
        # Opt-in TTS-output hardening under test (gated off by default).
        tts_drop_duplicate_window_s=2.0,
        playback_drain_enabled=True,
        playback_drain_timeout_s=10.0,
    )
    app._sleep_task = None
    app._eos_sent_this_turn = False
    app._asr_watchdog_task = None
    app._thinking_watchdog_task = None
    app._ptt_explicit_eos_pending = False
    app._last_tts_started_sentence = ""
    app._last_tts_started_ts = 0.0
    app._drop_current_tts_sentence = False
    app._vad_state = "idle"
    app._vad_speech_ms = 0
    app._vad_silence_ms = 0
    app._vad_eos_sent = False
    app._client_vad = None
    app._mic_rms_broadcast_task = None
    app._stop_words_cache = None
    return app


@pytest.mark.asyncio
async def test_bargein_does_not_resume_on_tail_tts():
    """After ASRPartial barge-in, subsequent TTSAudio frames from SLV's
    buffered tail must NOT reach the speaker and must NOT flip state
    back from BARGED_IN to SPEAKING."""
    app = _make_app()
    # Simulate: first TTS frame played, we're SPEAKING.
    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=24000))
    assert app._state == ConvState.SPEAKING
    assert len(app.audio.played) == 1

    # User barge-in via SLV partial -> cancel TTS and locally discard any
    # tail frames that were already in flight.
    await app._dispatch_one(ASRPartial(text="等等"))
    assert app._state == ConvState.BARGED_IN
    assert app.audio.discard is True
    assert app.slv.aborted == 1
    assert app.slv.truncations == [456]
    assert app.slv.reconnects == 0

    # Tail-end TTS chunks SLV keeps streaming: dropped, no state flip.
    await app._dispatch_one(TTSAudio(pcm=b"\x02\x00" * 8, sample_rate=24000))
    await app._dispatch_one(TTSAudio(pcm=b"\x03\x00" * 8, sample_rate=24000))
    assert app._state == ConvState.BARGED_IN, "state must stay BARGED_IN"
    assert len(app.audio.played) == 1, "tail TTS must not reach speaker"


@pytest.mark.asyncio
async def test_asr_final_rearms_playback_for_next_turn():
    """ASRFinal for a fresh user utterance must clear the discard latch
    so the next turn's TTS is audible."""
    app = _make_app()
    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=24000))
    await app._dispatch_one(ASRPartial(text="等等"))
    assert app.audio.discard is True

    async def _noop(text: str) -> None:
        return None

    app.on_user_utterance = _noop  # type: ignore[assignment]
    # New ASRFinal carrying the user's barge-in utterance.
    await app._dispatch_one(ASRFinal(text="重来", duplicate_of_streamed=False))
    assert app.audio.armed == 1
    assert app.audio.discard is False
    # Wait briefly so the spawned LLM turn task settles.
    if app._llm_turn_task is not None:
        await asyncio.wait_for(app._llm_turn_task, timeout=1.0)


@pytest.mark.asyncio
async def test_empty_asr_final_also_rearms_playback():
    """Empty ASRFinal (noise-triggered) must also clear the discard latch
    so a follow-up typed-text turn (no ASRFinal at all) isn't silently
    swallowed."""
    app = _make_app()
    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=24000))
    await app._dispatch_one(ASRPartial(text="hm"))
    assert app.audio.discard is True
    # Empty final -- IDLE early-return path.
    await app._dispatch_one(ASRFinal(text="", duplicate_of_streamed=False))
    assert app._state == ConvState.IDLE
    assert app.audio.discard is False
    assert app.audio.armed >= 1


@pytest.mark.asyncio
async def test_tts_done_does_not_demote_barged_in():
    """If SLV's tail TTS finishes (TTSDone) while we are BARGED_IN, the
    state must stay BARGED_IN — VAD silence-end / ASRFinal owns the
    next transition. Demoting to IDLE here would also start the auto-
    sleep timer mid-utterance under push_to_talk mode."""
    app = _make_app()
    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=24000))
    await app._dispatch_one(ASRPartial(text="等等"))
    assert app._state == ConvState.BARGED_IN
    # Add a mark_playback_done shim so the BaseApp branch hits it.
    app.audio.mark_playback_done = lambda: setattr(app.audio, "is_playing", False)
    await app._dispatch_one(TTSDone())
    assert app._state == ConvState.BARGED_IN
    assert app.audio.is_playing is False


@pytest.mark.asyncio
async def test_duplicate_tts_sentence_audio_is_dropped():
    """SLV can echo a direct send_text+tts_flush fallback twice.

    The second identical TTSStarted in a short window should not replay the
    same PCM or flip the FSM back into SPEAKING.
    """
    app = _make_app()

    await app._dispatch_one(TTSStarted(sentence="没听清，请再说一遍。"))
    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=16000))
    await app._dispatch_one(TTSSentenceDone(sentence="没听清，请再说一遍。"))
    await app._dispatch_one(TTSDone(session_complete=False))

    first_count = len(app.audio.played)
    assert first_count == 1

    await app._dispatch_one(TTSStarted(sentence="没听清，请再说一遍。"))
    await app._dispatch_one(TTSAudio(pcm=b"\x02\x00" * 8, sample_rate=16000))
    await app._dispatch_one(TTSSentenceDone(sentence="没听清，请再说一遍。"))
    await app._dispatch_one(TTSDone(session_complete=False))

    assert len(app.audio.played) == first_count


@pytest.mark.asyncio
async def test_tts_done_waits_for_local_playback_drain_before_idle():
    """TTSDone means SLV stopped streaming, not that local audio is silent."""
    app = _make_app()
    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=24000))
    assert app._state == ConvState.SPEAKING
    assert app.audio.is_playing is True

    await app._dispatch_one(TTSDone())
    assert app._state == ConvState.SPEAKING

    app.audio.is_playing = False
    task = getattr(app, "_playback_drain_task")
    await asyncio.wait_for(task, timeout=1.0)
    assert app._state == ConvState.IDLE


@pytest.mark.asyncio
async def test_response_done_waits_for_local_playback_drain_before_idle():
    """Realtime V2 response.done is terminal remotely, not locally audible."""
    from ovs_agent.slv_client import ResponseDone, ResponseOutputAudioDone

    app = _make_app()
    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=24000))
    assert app._state == ConvState.SPEAKING

    await app._dispatch_one(ResponseOutputAudioDone(response_id="resp_1"))
    await app._dispatch_one(ResponseDone(
        response_id="resp_1",
        status="completed",
        response={"id": "resp_1", "status": "completed"},
    ))
    assert app._state == ConvState.SPEAKING

    app.audio.is_playing = False
    await asyncio.wait_for(app._playback_drain_task, timeout=1.0)
    assert app._state == ConvState.IDLE


@pytest.mark.asyncio
async def test_cancelled_response_done_preserves_barged_in():
    from ovs_agent.slv_client import ResponseDone

    app = _make_app()
    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=24000))
    await app._dispatch_one(ASRPartial(text="等等"))
    assert app._state == ConvState.BARGED_IN

    await app._dispatch_one(ResponseDone(
        response_id="resp_old",
        status="cancelled",
        response={"id": "resp_old", "status": "cancelled"},
    ))
    assert app._state == ConvState.BARGED_IN
