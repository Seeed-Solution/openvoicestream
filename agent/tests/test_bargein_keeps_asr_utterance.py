"""Regression: a speech-driven barge-in must not throw away the audio the
user is interrupting WITH.

Measured on rk3588 (2026-09-21, cloud LLM, injected clips): the user said
"Stop. Please answer in one sentence." over the assistant's reply. The agent
fired barge-in on the partial 'Stop, please.', sent a plain `abort`, and the
server cancelled the in-flight ASR utterance 0.3 s later (speech log:
``Qwen3 streaming VAD backend`` re-init at 01:55:05.208, then
``finalize: 2 chunks, 0.80s audio text='One sentence.'``). The LLM therefore
received "One sentence." — everything said before the barge-in threshold was
gone.

Two halves of the same loss, both covered here:
  * the frame: barge-in sends ``abort(keep_asr=True)`` so the server keeps
    the utterance (see server/core/v2v.py, and
    server/tests/test_v2v_abort_keep_asr.py for the server half);
  * the local buffer: mid-utterance ASR segments already accumulated for
    THIS utterance must survive the barge-in instead of being cleared.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from ovs_agent.app_base import BaseApp
from ovs_agent.slv_client import SLVClient
from ovs_agent.state import ConvState

class _FakeAudio:
    def __init__(self) -> None:
        self.stopped = 0

    async def stop_playback(self) -> None:
        self.stopped += 1

    def playback_position_ms(self, response_id=None) -> int:
        return 0


class _RecordingSLV:
    def __init__(self) -> None:
        self.aborts: list[bool] = []
        self.truncations: list[int] = []

    async def abort(self, *, keep_asr: bool = False) -> None:
        self.aborts.append(keep_asr)

    async def truncate_active_response(self, audio_end_ms: int) -> None:
        self.truncations.append(audio_end_ms)


def _make_app() -> BaseApp:
    """Only what ``_interrupt_current_turn_for_barge_in`` touches."""
    app = BaseApp.__new__(BaseApp)
    app.audio = _FakeAudio()
    app.slv = _RecordingSLV()
    app._llm_turn_task = None
    app._asr_watchdog_task = None
    app._eos_sent_this_turn = True
    app._first_tts_seen = True
    app._state = ConvState.SPEAKING
    return app


@pytest.mark.asyncio
async def test_bargein_sends_keep_asr_and_keeps_pending_segments():
    app = _make_app()
    # A mid-utterance segment of the barge-in utterance: the RK Qwen3-ASR
    # backend emits an internal final after ~400 ms of silence, well before
    # the agent's asr_eos, so this buffer holds the HEAD of what the user is
    # saying right now.
    app._pending_asr_utterance_text = "Stop, please"

    await app._interrupt_current_turn_for_barge_in()

    assert app.slv.aborts == [True], "barge-in must ask the server to keep ASR"
    assert app._pending_asr_utterance_text == "Stop, please", (
        "the head of the barge-in utterance must survive the interrupt"
    )


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _client_with_ws() -> tuple[SLVClient, _FakeWS]:
    client = SLVClient.__new__(SLVClient)
    client.protocol_version = 1
    client._send_lock = asyncio.Lock()
    ws = _FakeWS()
    client._ws = ws
    return client, ws


@pytest.mark.asyncio
async def test_abort_frame_carries_keep_asr_only_when_asked():
    client, ws = _client_with_ws()

    await client.abort()
    await client.abort(keep_asr=True)

    assert ws.sent[0] == {"type": "abort"}, (
        "a plain abort (stop / sleep / dashboard) keeps the old semantics"
    )
    assert ws.sent[1] == {"type": "abort", "keep_asr": True}


@pytest.mark.asyncio
async def test_v2_response_cancel_carries_keep_asr():
    """Realtime V2 is the default protocol (config.realtime_protocol_version=2);
    the flag must survive there too, not only on the legacy v1 frame."""
    client, ws = _client_with_ws()
    client.protocol_version = 2
    client._active_response_id = "resp_1"

    await client.abort()
    await client.abort(keep_asr=True)

    assert ws.sent[0] == {"type": "response.cancel", "response_id": "resp_1"}
    assert ws.sent[1] == {
        "type": "response.cancel", "response_id": "resp_1", "keep_asr": True,
    }



@pytest.mark.asyncio
async def test_reply_boundary_hook_fires_on_every_reply_start():
    """BaseApp ends a superseded reply's tail on this hook; every way a new
    reply starts on the wire must fire it (Codex review of #111)."""
    client, ws = _client_with_ws()
    client.protocol_version = 2
    client._active_response_id = None
    fired: list[int] = []
    client.on_reply_text = lambda: fired.append(1)

    await client.send_text("")            # empty chunk: not a reply start
    assert fired == []
    await client.send_text("Sure.")
    await client.flush_tts()
    await client.speak("Done.")
    await client.create_response()
    assert len(fired) == 4


class _EosSLV(_RecordingSLV):
    def __init__(self) -> None:
        super().__init__()
        self.eos = 0

    async def asr_eos(self) -> None:
        self.eos += 1


class _Cfg:
    client_vad_drive_eos = True
    client_vad_silence_ms = 20
    thinking_timeout_s = 60.0


def _make_eos_app(vad_state: str) -> BaseApp:
    app = _make_app()
    app.slv = _EosSLV()
    app.config = _Cfg()
    app._vad_state = vad_state
    app._state = ConvState.BARGED_IN
    app._thinking_watchdog_task = None
    return app


async def _drain(app: BaseApp) -> None:
    for name in ("_bargein_eos_task", "_asr_watchdog_task", "_thinking_watchdog_task"):
        task = getattr(app, name, None)
        if task is not None and not task.done():
            task.cancel()


@pytest.mark.asyncio
async def test_partial_bargein_after_vad_segment_ended_closes_kept_utterance():
    """rk3588 2026-09-21: echo VAD segment ended (eos suppressed), the ASR
    partial of that audio fired barge-in 170 ms later, keep_asr kept the
    utterance, and nothing ever sent asr_eos — the turn hit the server's 45 s
    per-turn deadline. With VAD outside a segment the agent must close it."""
    app = _make_eos_app(vad_state="idle")

    await app._interrupt_current_turn_for_barge_in()
    await asyncio.sleep(0.1)

    assert app.slv.eos == 1
    assert app._state == ConvState.THINKING
    await _drain(app)


@pytest.mark.asyncio
async def test_bargein_eos_fallback_yields_to_a_new_vad_segment():
    app = _make_eos_app(vad_state="idle")

    await app._interrupt_current_turn_for_barge_in()
    # The user keeps talking: the VAD speech-start path cancels the fallback
    # and the segment ends through the normal speech→silence edge.
    app._vad_state = "speech"
    app._cancel_bargein_eos_fallback()
    await asyncio.sleep(0.1)

    assert app.slv.eos == 0
    assert app._state == ConvState.BARGED_IN
    await _drain(app)


@pytest.mark.asyncio
async def test_bargein_inside_a_vad_segment_leaves_eos_to_vad():
    app = _make_eos_app(vad_state="speech")

    await app._interrupt_current_turn_for_barge_in()
    await asyncio.sleep(0.1)

    assert app.slv.eos == 0
    assert getattr(app, "_bargein_eos_task", None) is None
    await _drain(app)
