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
