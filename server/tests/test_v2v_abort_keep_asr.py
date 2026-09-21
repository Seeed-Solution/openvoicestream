"""``abort`` + ``keep_asr`` (server half of the barge-in head-loss fix).

Measured on rk3588 (2026-09-21): the agent fired barge-in on the partial
'Stop, please.', sent ``abort``, and this handler cancelled the in-flight ASR
utterance — the one the user was interrupting WITH. The LLM got "One
sentence." instead of "Stop, please answer in one sentence.".

A client now marks a speech-driven barge-in with ``keep_asr: true``; the
handler then cancels TTS only. Structural assertions against live source,
matching test_v2v_ping_keepalive.py — the full handler needs ASR/TTS/VAD
wiring to drive end to end. Agent half: agent/tests/test_bargein_keeps_asr_utterance.py.
"""
from __future__ import annotations

import inspect

import pytest


def _abort_branch() -> str:
    from server import main as appmod

    src = inspect.getsource(appmod.v2v_stream)
    start = src.find("elif typ == v2v_proto.CLIENT_ABORT:")
    assert start != -1, "dispatcher has no CLIENT_ABORT branch"
    end = src.find("except WebSocketDisconnect:", start)
    assert end != -1
    return src[start:end]


def test_abort_still_cancels_tts_unconditionally():
    body = _abort_branch()
    tts_cancel = body.find('state["current_tts_task"]')
    keep = body.find('payload.get("keep_asr")')
    assert tts_cancel != -1 and keep != -1
    assert tts_cancel < keep, "TTS must be cancelled regardless of keep_asr"


def test_asr_cancel_is_gated_on_keep_asr():
    body = _abort_branch()
    cancel_at = body.find('asr_manager.cancel("bargein")')
    assert cancel_at != -1, "plain abort must still cancel ASR"
    gate = body.rfind("not keep_asr", 0, cancel_at)
    assert gate != -1, "ASR cancel must sit behind `not keep_asr`"


def test_keep_asr_is_bounded_by_utterance_length():
    """Keeping the stream also keeps what it heard during playback (echo on
    boards without AEC). Past the bound the abort must fall back to cancel."""
    body = _abort_branch()
    assert "_keep_asr_max_s()" in body
    assert "keep_asr = False" in body


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", 5.0),
        ("3.5", 3.5),
        ("0", 0.0),        # 0 = never keep, a legitimate opt-out
        ("nan", 5.0),      # nan <= 0 is False; must not disable the bound
        ("inf", 5.0),
        ("-1", 5.0),
        ("garbage", 5.0),
    ],
)
def test_keep_asr_max_s_parsing(monkeypatch, raw, expected):
    from server import main as appmod

    monkeypatch.setenv("OVS_V2V_ABORT_KEEP_ASR_MAX_S", raw)
    assert appmod._keep_asr_max_s() == expected


class _FakeWS:
    def __init__(self, frames):
        self._frames = list(frames)

    async def receive(self):
        return {"text": self._frames.pop(0)}


class _FakeAdapter:
    def __init__(self):
        self.cancelled = []

    def mark_cancelled(self, reason):
        self.cancelled.append(reason)


def test_realtime_v2_proxy_carries_keep_asr_on_response_cancel():
    """Server-loop sessions go through _RealtimeV2WebSocketProxy, which
    rebuilds the frame; it must not drop keep_asr."""
    import json

    from server import main as appmod
    from server.core import v2v as v2v_proto

    frames = [
        json.dumps({"type": "response.cancel", "response_id": "r", "keep_asr": True}),
        json.dumps({"type": "response.cancel", "response_id": "r"}),
        json.dumps({"type": "input_audio_buffer.clear", "keep_asr": True}),
    ]
    import asyncio

    proxy = appmod._RealtimeV2WebSocketProxy(_FakeWS(frames), _FakeAdapter())

    async def _drain():
        return [json.loads((await proxy.receive())["text"]) for _ in frames]

    got = asyncio.run(_drain())

    assert got[0] == {"type": v2v_proto.CLIENT_ABORT, "keep_asr": True}
    assert got[1] == {"type": v2v_proto.CLIENT_ABORT}
    assert got[2] == {"type": v2v_proto.CLIENT_ABORT}, (
        "clearing the input buffer is destructive; keep_asr must not apply"
    )


def test_keep_asr_bound_uses_the_connection_rate():
    """The accepted-sample counter is at the connection's input rate; dividing
    by the backend rate turns 2 s of 48 kHz audio into 6 s."""
    body = _abort_branch()
    at = body.find("kept_s = ")
    assert at != -1
    stmt = body[at: body.find("max_keep_s", at)]
    assert "sample_rate" in stmt and "asr_be" not in stmt
