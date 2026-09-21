"""Tests for ConvState transitions in BaseApp."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from ovs_agent.app_base import BaseApp
from ovs_agent.state import ConvState
from ovs_agent.event_bus import EventBus
from ovs_agent.slv_client import TTSAudio


def _fresh_app() -> BaseApp:
    """Build a BaseApp without touching audio/llm/slv (via __new__)."""
    from ovs_agent.config import Config

    app = BaseApp.__new__(BaseApp)
    app.config = Config()
    app.events = EventBus()
    app.plugins = []
    app._state = ConvState.IDLE
    app._slv_reconnect_count = 0
    app._llm_turn_task = None
    app._first_tts_seen = False
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
    return app


@pytest.mark.asyncio
async def test_set_state_only_fires_on_change():
    app = _fresh_app()
    seen: list[dict] = []

    class P:
        name = "p"
        async def on_state_change(self, data):  # noqa: ANN001
            seen.append(data)

    app.plugins.append(P())

    app._set_state(ConvState.IDLE)  # no-op
    await asyncio.sleep(0.01)
    assert seen == []

    app._set_state(ConvState.LISTENING)
    await asyncio.sleep(0.01)
    assert len(seen) == 1
    assert seen[0]["state"] == "listening"
    assert seen[0]["prev"] == "idle"


@pytest.mark.asyncio
async def test_golden_path_transitions():
    app = _fresh_app()
    transitions: list[tuple[str, str]] = []

    def cb(d):
        transitions.append((d["prev"], d["state"]))

    app.events.subscribe("state_change", cb)

    for s in (
        ConvState.LISTENING,
        ConvState.THINKING,
        ConvState.SPEAKING,
        ConvState.IDLE,
    ):
        app._set_state(s)

    assert transitions == [
        ("idle", "listening"),
        ("listening", "thinking"),
        ("thinking", "speaking"),
        ("speaking", "idle"),
    ]


@pytest.mark.asyncio
async def test_barge_in_transition():
    app = _fresh_app()
    app._set_state(ConvState.SPEAKING)
    app._set_state(ConvState.BARGED_IN)
    app._set_state(ConvState.THINKING)
    assert app._state == ConvState.THINKING


@pytest.mark.asyncio
async def test_barge_in_cancels_llm_turn():
    """ASRPartial during SPEAKING must cancel the in-flight LLM task,
    stop playback, abort SLV TTS without reconnecting, and re-arm
    _first_tts_seen.
    """
    from ovs_agent.slv_client import ASRPartial

    app = _fresh_app()
    # Minimal stubs for the barge-in branch.
    abort_calls: list[int] = []
    stop_calls: list[int] = []

    class _Audio:
        is_playing = True
        async def stop_playback(self):
            stop_calls.append(1)

    class _SLV:
        reconnects = 0

        async def abort(self, *, keep_asr: bool = False):
            abort_calls.append(1)

        async def reconnect(self):
            self.reconnects += 1

    app.audio = _Audio()
    app.slv = _SLV()
    app._first_tts_seen = True
    app._state = ConvState.SPEAKING

    # Long-running fake LLM turn that should be cancelled.
    cancelled = asyncio.Event()
    async def _llm_turn():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    app._llm_turn_task = asyncio.create_task(_llm_turn(), name="llm-turn")
    # Give it one loop tick to actually start.
    await asyncio.sleep(0)

    await app._dispatch_one(ASRPartial(text="打断一下"))

    assert cancelled.is_set(), "LLM turn was not cancelled by barge-in"
    assert abort_calls == [1]
    assert app.slv.reconnects == 0
    assert stop_calls == [1]
    assert app._first_tts_seen is False
    assert app._state == ConvState.BARGED_IN


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [ConvState.IDLE, ConvState.THINKING])
async def test_client_vad_does_not_barge_from_stale_playback(state):
    """Buffered tail audio is not proof that an active assistant turn exists."""
    app = _fresh_app()
    stop_calls: list[int] = []
    abort_calls: list[int] = []

    class _Vad:
        name = "test"
        threshold = 0.5

        def is_speech(self, _chunk):
            return True

    class _Audio:
        is_playing = True

        async def stop_playback(self):
            stop_calls.append(1)

    class _SLV:
        async def abort(self, *, keep_asr: bool = False):
            abort_calls.append(1)

    app._client_vad = _Vad()
    app.audio = _Audio()
    app.slv = _SLV()
    app.config.client_vad_speech_min_ms = 100
    app._state = state

    await app._update_vad(b"\x00\x00" * 1600, 100)

    assert stop_calls == []
    assert abort_calls == []
    assert app._state == state


@pytest.mark.asyncio
async def test_client_vad_barges_only_from_active_speaking_state():
    app = _fresh_app()
    stop_calls: list[int] = []
    abort_calls: list[int] = []

    class _Vad:
        name = "test"
        threshold = 0.5

        def is_speech(self, _chunk):
            return True

    class _Audio:
        is_playing = True

        async def stop_playback(self):
            self.is_playing = False
            stop_calls.append(1)

    class _SLV:
        async def abort(self, *, keep_asr: bool = False):
            abort_calls.append(1)

    app._client_vad = _Vad()
    app.audio = _Audio()
    app.slv = _SLV()
    app.config.client_vad_speech_min_ms = 100
    app._state = ConvState.SPEAKING

    await app._update_vad(b"\x00\x00" * 1600, 100)

    assert stop_calls == [1]
    assert abort_calls == [1]
    assert app._state == ConvState.BARGED_IN


@pytest.mark.asyncio
async def test_client_vad_does_not_self_barge_during_initial_echo_guard():
    app = _fresh_app()
    stop_calls: list[int] = []
    abort_calls: list[int] = []

    class _Vad:
        def is_speech(self, _chunk):
            return True

    class _Audio:
        is_playing = True

        async def stop_playback(self):
            stop_calls.append(1)

    class _SLV:
        async def abort(self, *, keep_asr: bool = False):
            abort_calls.append(1)

    app._client_vad = _Vad()
    app.audio = _Audio()
    app.slv = _SLV()
    app.config.client_vad_speech_min_ms = 100
    app.config.barge_in_min_speaking_ms = 500
    app._state = ConvState.SPEAKING
    app._speaking_since_ts = time.monotonic()

    await app._update_vad(b"\x00\x00" * 1600, 100)

    assert stop_calls == []
    assert abort_calls == []
    assert app._state == ConvState.SPEAKING


@pytest.mark.asyncio
async def test_client_vad_honors_barge_in_disabled():
    app = _fresh_app()
    stop_calls: list[int] = []
    abort_calls: list[int] = []

    class _Vad:
        name = "test"
        threshold = 0.5

        def is_speech(self, _chunk):
            return True

    class _Audio:
        is_playing = True

        async def stop_playback(self):
            stop_calls.append(1)

    class _SLV:
        async def abort(self, *, keep_asr: bool = False):
            abort_calls.append(1)

    app._client_vad = _Vad()
    app.audio = _Audio()
    app.slv = _SLV()
    app.config.client_vad_speech_min_ms = 100
    app.config.barge_in_enabled = False
    app._state = ConvState.SPEAKING

    await app._update_vad(b"\x00\x00" * 1600, 100)

    assert stop_calls == []
    assert abort_calls == []
    assert app._state == ConvState.SPEAKING


@pytest.mark.asyncio
async def test_tts_audio_broadcasts_every_frame_but_speaks_once():
    app = _fresh_app()
    app._first_tts_seen = False
    broadcasts: list[tuple[str, dict]] = []

    class _Audio:
        def __init__(self):
            self.played = []
            self.sample_rates = []

        def set_output_sample_rate(self, sample_rate):
            self.sample_rates.append(sample_rate)

        async def play(self, pcm):
            self.played.append(pcm)

    async def _br(name, data=None):
        broadcasts.append((name, data))

    app.audio = _Audio()
    app._broadcast = _br

    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=24000))
    await app._dispatch_one(TTSAudio(pcm=b"\x02\x00" * 4, sample_rate=24000))

    frames = [data for name, data in broadcasts if name == "on_tts_audio_frame"]
    assert frames == [
        {"sample_rate": 24000, "frame_len": 16, "first": True},
        {"sample_rate": 24000, "frame_len": 8, "first": False},
    ]
    assert app.audio.sample_rates == [24000]
    assert len(app.audio.played) == 2


@pytest.mark.asyncio
async def test_slv_error_in_sleeping_stays_sleeping():
    """Regression: a transport error while SLEEPING must NOT
    force the FSM back to IDLE — that would hot-mic the agent in
    wake_word / push_to_talk mode."""
    from ovs_agent.slv_client import SLVError

    app = _fresh_app()
    app._llm_turn_task = None
    app._state = ConvState.SLEEPING

    await app._dispatch_one(SLVError(message="ws ping timeout"))
    assert app._state == ConvState.SLEEPING


@pytest.mark.asyncio
async def test_slv_error_in_thinking_drops_to_idle():
    """Non-sleeping SLVError still resets FSM to IDLE (status quo)."""
    from ovs_agent.slv_client import SLVError

    app = _fresh_app()
    app._llm_turn_task = None
    app._state = ConvState.THINKING

    await app._dispatch_one(SLVError(message="boom"))
    assert app._state == ConvState.IDLE


@pytest.mark.asyncio
async def test_sleeping_drops_asr_partial_and_endpoint():
    """Regression: late-arriving ASR events after /sleep must
    not be processed — would otherwise leak through to on_user_utterance
    and silently wake the agent."""
    from ovs_agent.slv_client import ASRPartial, ASREndpoint, ASRFinal

    app = _fresh_app()
    app._state = ConvState.SLEEPING
    app._llm_turn_task = None
    app._first_tts_seen = False
    app._slv_reconnect_count = 0

    broadcasts: list[tuple] = []
    async def _br(name, *args):
        broadcasts.append((name, args))
    app.broadcast = _br
    app._broadcast = _br

    class _Audio:
        is_playing = False
        async def stop_playback(self):
            pass

    class _SLV:
        reconnected = 0
        async def reconnect(self):
            type(self).reconnected += 1
        async def abort(self, *, keep_asr: bool = False):
            pass

    app.audio = _Audio()
    app.slv = _SLV()

    # ASRPartial dropped silently.
    await app._dispatch_one(ASRPartial(text="hello"))
    assert all(n != "on_user_partial" for n, _ in broadcasts)
    # ASREndpoint dropped silently.
    await app._dispatch_one(ASREndpoint())
    assert all(n != "on_user_speech_start" for n, _ in broadcasts)
    # ASRFinal with session_complete=True triggers reconnect but does
    # NOT broadcast on_user_utterance and does NOT spawn LLM turn.
    await app._dispatch_one(ASRFinal(
        text="real text", duplicate_of_streamed=False, session_complete=True
    ))
    assert _SLV.reconnected == 1
    assert all(n != "on_user_utterance" for n, _ in broadcasts), (
        f"on_user_utterance leaked through SLEEPING gate: {broadcasts}"
    )
    assert app._llm_turn_task is None
    assert app._state == ConvState.SLEEPING


def test_conv_state_values():
    assert ConvState.IDLE.value == "idle"
    assert ConvState.SPEAKING.value == "speaking"
    assert ConvState.BARGED_IN.value == "barged_in"
    assert list(ConvState) == [
        ConvState.IDLE,
        ConvState.LISTENING,
        ConvState.THINKING,
        ConvState.SPEAKING,
        ConvState.BARGED_IN,
        ConvState.SLEEPING,
    ]
    assert ConvState.SLEEPING.value == "sleeping"


@pytest.mark.asyncio
@pytest.mark.parametrize("drive_eos", [True, False])
async def test_echo_segment_ending_does_not_end_the_reply(drive_eos):
    """Regression (rk3588, 2026-09-21): a client-VAD segment that began during
    playback and was rejected by the echo guard used to send asr_eos and flip
    SPEAKING → THINKING when it ended. Barge-in then stayed disabled for the
    rest of the reply (both paths require SPEAKING) and a real interruption
    1.2 s later was ignored."""
    import time as _time

    app = _fresh_app()
    speech = {"on": True}
    eos_calls: list[int] = []

    class _Vad:
        name = "test"
        threshold = 0.5

        def is_speech(self, _chunk):
            return speech["on"]

        def reset(self):
            pass

    class _Audio:
        is_playing = True

        async def stop_playback(self):
            raise AssertionError("echo must not stop playback")

    class _SLV:
        async def abort(self, *, keep_asr: bool = False):
            raise AssertionError("echo must not abort")

        async def asr_eos(self):
            eos_calls.append(1)

    app._client_vad = _Vad()
    app.audio = _Audio()
    app.slv = _SLV()
    app.config.client_vad_speech_min_ms = 100
    app.config.client_vad_silence_ms = 200
    app.config.client_vad_drive_eos = drive_eos
    app.config.barge_in_min_speaking_ms = 500
    app._state = ConvState.SPEAKING
    app._speaking_since_ts = _time.monotonic()  # playback just started

    await app._update_vad(b"\x00\x00" * 1600, 100)  # echo onset at ~0 ms
    assert app._vad_state == "speech"
    assert app._state == ConvState.SPEAKING

    speech["on"] = False
    for _ in range(3):  # 300 ms of silence ≥ client_vad_silence_ms
        await app._update_vad(b"\x00\x00" * 1600, 100)

    assert app._state == ConvState.SPEAKING, "echo segment must not end the reply"
    assert eos_calls == []
    assert app._vad_state == "idle"

    # The next real segment, past the echo guard, still barges in.
    app._speaking_since_ts = _time.monotonic() - 1.0
    aborts: list[bool] = []

    async def _abort(*, keep_asr: bool = False):
        aborts.append(keep_asr)

    async def _stop():
        app.audio.is_playing = False

    app.slv.abort = _abort
    app.audio.stop_playback = _stop
    speech["on"] = True
    await app._update_vad(b"\x00\x00" * 1600, 100)
    assert app._state == ConvState.BARGED_IN
    assert aborts == [True]
