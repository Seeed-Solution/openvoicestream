"""Test vad_event frame emission from /v2v/stream.

Verifies that the server emits ``{"type":"vad_event","event":"speech_start"}``
on VAD-detected speech onset and ``{"type":"vad_event","event":"speech_end"}``
on VAD-detected endpoint. These frames let clients (e.g. clawd-reachy-mini)
update their playback / state machine in lockstep with server-side VAD,
without inferring it from asr_partial timing.

Scenarios
---------

1. SPEECH_START → server sends vad_event BEFORE any asr_partial /
   asr_final could arrive. (The client_eos path proves nothing about
   VAD; we instead inject a scripted VAD via monkeypatched
   ``vad_mod.create_vad``.)

2. SPEECH_END → server sends vad_event in the same audio-chunk handling
   step that latches ``endpoint_pending``. We assert the event arrives,
   and that it arrives in the right order relative to ``asr_final``.

3. Constant present in protocol module — pin the symbolic name so
   downstream clients can `import SERVER_VAD_EVENT` without a typo
   surfacing only at runtime.

The fake VAD is fully scripted: its ``process()`` returns SPEECH_START
on the first chunk and SPEECH_END on the second, so we don't depend on
silero ONNX models being present (CI / Mac dev).
"""

from __future__ import annotations

import os
import sys
import time
from typing import List

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from server.core.asr_backend import ASRBackend, ASRCapability
from server.core import v2v as v2v_proto


# ──────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────


class _FakeStream:
    def __init__(self, backend: "_FakeASRBackend"):
        self._backend = backend
        self.finalized = False

    def accept_waveform(self, sr, samples):
        pass

    def get_partial(self):
        return "", False

    def finalize(self):
        self.finalized = True
        # New ASRStream ABC contract: ``(text, detected_language)``.
        return self._backend._next_final_text(), None

    def cancel(self):
        pass

    def cancel_and_finalize(self):
        return ""


class _FakeASRBackend(ASRBackend):
    def __init__(self, finals: List[str]):
        self._finals = list(finals)
        self._idx = 0
        self.streams_created: List[_FakeStream] = []

    @property
    def name(self):
        return "fake-vad-event"

    @property
    def capabilities(self):
        return {ASRCapability.STREAMING}

    @property
    def sample_rate(self):
        return 16000

    def is_ready(self):
        return True

    def preload(self):
        return None

    def transcribe(self, audio_bytes, language="auto"):
        from server.core.asr_backend import TranscriptionResult
        return TranscriptionResult(text="", duration=0.0, inference_time=0.0,
                                   rtf=0.0, n_tokens=0, per_token_ms=0.0,
                                   backend=self.name)

    def transcribe_audio(self, audio, language="auto"):
        return self.transcribe(b"", language)

    def create_stream(self, language="auto"):
        s = _FakeStream(self)
        self.streams_created.append(s)
        return s

    def _next_final_text(self):
        if self._idx < len(self._finals):
            t = self._finals[self._idx]
            self._idx += 1
            return t
        return ""


class _ScriptedVAD:
    """Returns a scripted sequence of VAD events.

    Each call to ``process()`` returns the next event in ``events`` (or
    ``None`` once the script is exhausted). Mirrors the
    ``VADSession.process()`` contract from server.core.vad.
    """

    def __init__(self, events):
        self._events = list(events)
        self._i = 0

    def process(self, samples):
        if self._i < len(self._events):
            ev = self._events[self._i]
            self._i += 1
            return ev
        return None


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _silence_pcm16(ms=50, sr=16000):
    return np.zeros((sr * ms) // 1000, dtype=np.int16).tobytes()


@pytest.fixture
def fake_asr_backend(monkeypatch):
    import server.main as main_mod
    from server.core.coordinator import init_coordinator
    from server.core import session_limiter
    init_coordinator({"mode": "concurrent"})
    # The /v2v/stream handler admits the WS through the global session limiter;
    # TestClient(app) doesn't run the startup lifespan that calls init_limiter,
    # so without this the connection is rejected with 1011
    # session_limiter_unavailable (seen=[]). Mirror the other limiter tests.
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({})

    be = _FakeASRBackend(finals=["hello world", "next utterance", "more"])
    monkeypatch.setattr(main_mod, "_asr_backend", be, raising=False)
    monkeypatch.setattr(main_mod, "_get_asr_backend", lambda: be)
    return be


@pytest.fixture
def scripted_vad(monkeypatch):
    """Patch vad_mod.create_vad to return our scripted fake.

    The events list is mutable so tests can pre-load it. Returns the
    fake instance so tests can assert on it if needed.
    """
    from server.core import vad as vad_mod

    fake = _ScriptedVAD(events=[
        vad_mod.VADSession.SPEECH_START,
        vad_mod.VADSession.SPEECH_END,
    ])

    monkeypatch.setattr(vad_mod, "create_vad", lambda *a, **kw: fake)
    return fake


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


def test_vad_event_constants_defined():
    """Pin the protocol-level symbol names."""
    assert v2v_proto.SERVER_VAD_EVENT == "vad_event"
    assert v2v_proto.VAD_EVENT_SPEECH_START == "speech_start"
    assert v2v_proto.VAD_EVENT_SPEECH_END == "speech_end"


def test_v2v_rejects_explicit_vad_when_initialization_fails(
    fake_asr_backend, monkeypatch
):
    """An unavailable requested VAD must not silently become forced-EOS."""
    from fastapi.testclient import TestClient
    from server.core import vad as vad_mod
    from server.main import app

    def _fail(*_args, **_kwargs):
        raise RuntimeError("onnxruntime missing")

    monkeypatch.setattr(vad_mod, "create_vad", _fail)
    client = TestClient(app)
    with client.websocket_connect("/v2v/stream") as ws:
        ws.send_json({
            "type": "config", "asr_language": "en", "vad": "silero",
            "sample_rate": 16000,
        })
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "VAD init failed" in error["error"]


def test_vad_none_remains_explicitly_disabled():
    from server.core import vad as vad_mod

    assert vad_mod.create_vad("none", sample_rate=16000) is None


def _open_v2v(client, *, multi_utterance=False, vad="silero"):
    cfg = {
        "type": "config",
        "asr_language": "en",
        "vad": vad,
        "sample_rate": 16000,
        "multi_utterance": multi_utterance,
    }
    ws = client.websocket_connect("/v2v/stream")
    ws.__enter__()
    ws.send_json(cfg)
    return ws


def _drain_for(ws, types_wanted, timeout_s=5.0, max_msgs=50):
    """Receive JSON until all `types_wanted` (set of type strings) have
    been seen at least once, or timeout. Returns ordered list of payloads.
    """
    deadline = time.monotonic() + timeout_s
    seen = []
    remaining = set(types_wanted)
    while time.monotonic() < deadline and len(seen) < max_msgs:
        try:
            payload = ws.receive_json()
        except Exception as e:
            raise AssertionError(f"WS recv error: {e}; seen={seen}")
        seen.append(payload)
        remaining.discard(payload.get("type"))
        if not remaining:
            return seen
    raise AssertionError(
        f"timed out waiting for {types_wanted}; missing={remaining}; seen={seen}"
    )


def test_vad_event_speech_start_and_end_emitted(fake_asr_backend, scripted_vad):
    """Driving the WS with two binary chunks triggers speech_start
    then speech_end vad_event frames, in that order, with speech_end
    arriving no later than the asr_final it precedes."""
    from fastapi.testclient import TestClient
    from server.main import app

    client = TestClient(app)
    ws = _open_v2v(client, multi_utterance=False, vad="silero")
    try:
        # 1st audio chunk → scripted VAD returns SPEECH_START
        ws.send_bytes(_silence_pcm16(50))
        # 2nd audio chunk → scripted VAD returns SPEECH_END → endpoint
        ws.send_bytes(_silence_pcm16(50))

        seen = _drain_for(ws, {"vad_event", "asr_final"})

        vad_events = [p for p in seen if p.get("type") == "vad_event"]
        events_only = [p.get("event") for p in vad_events]
        assert "speech_start" in events_only, f"missing speech_start in {seen}"
        assert "speech_end" in events_only, f"missing speech_end in {seen}"

        # Ordering: speech_start before speech_end before asr_final
        types = [p.get("type") for p in seen]
        # locate first speech_start, first speech_end, first asr_final
        def first_idx(pred):
            for i, p in enumerate(seen):
                if pred(p):
                    return i
            return -1

        i_start = first_idx(lambda p: p.get("type") == "vad_event"
                                       and p.get("event") == "speech_start")
        i_end   = first_idx(lambda p: p.get("type") == "vad_event"
                                       and p.get("event") == "speech_end")
        i_final = first_idx(lambda p: p.get("type") == "asr_final")
        assert 0 <= i_start < i_end < i_final, (
            f"ordering violated: start={i_start} end={i_end} final={i_final}; seen={seen}"
        )
    finally:
        ws.__exit__(None, None, None)


def test_vad_event_speech_start_payload_shape(fake_asr_backend, scripted_vad):
    """Speech_start frame must be exactly {"type":"vad_event","event":"speech_start"}
    — no extra required fields that could break minimalist clients."""
    from fastapi.testclient import TestClient
    from server.main import app

    client = TestClient(app)
    ws = _open_v2v(client, multi_utterance=False, vad="silero")
    try:
        ws.send_bytes(_silence_pcm16(50))   # triggers SPEECH_START
        # Read until we see the vad_event speech_start
        deadline = time.monotonic() + 3.0
        found = None
        while time.monotonic() < deadline:
            p = ws.receive_json()
            if p.get("type") == "vad_event" and p.get("event") == "speech_start":
                found = p
                break
        assert found is not None, "speech_start vad_event never arrived"
        # Must use the documented field names
        assert set(found.keys()) >= {"type", "event"}
        assert found["type"] == v2v_proto.SERVER_VAD_EVENT
        assert found["event"] == v2v_proto.VAD_EVENT_SPEECH_START
    finally:
        ws.__exit__(None, None, None)


def test_realtime_v2_handshake_and_canonical_vad_transcription(
    fake_asr_backend, scripted_vad
):
    from fastapi.testclient import TestClient
    from server.main import app

    client = TestClient(app)
    ws = client.websocket_connect(
        "/v2v/stream", subprotocols=["seeed.realtime.v2"]
    )
    ws.__enter__()
    try:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["session"]["protocol_version"] == 2

        ws.send_json({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "sample_rate": 16000,
                            "channels": 1,
                            "endianness": "little",
                        },
                        "transcription": {"language": "en"},
                        "turn_detection": {
                            "type": "server_vad",
                            "backend": "silero",
                            "silence_duration_ms": 400,
                            "create_response": False,
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "sample_rate": 16000,
                            "channels": 1,
                            "endianness": "little",
                        }
                    },
                },
            },
        })
        updated = ws.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["session"]["id"] == created["session"]["id"]

        ws.send_bytes(_silence_pcm16(50))
        ws.send_bytes(_silence_pcm16(50))
        seen = _drain_for(ws, {
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "conversation.item.input_audio_transcription.completed",
        })
        types = [event["type"] for event in seen]
        assert types.index("input_audio_buffer.speech_started") < types.index(
            "input_audio_buffer.speech_stopped"
        ) < types.index("conversation.item.input_audio_transcription.completed")
        final = next(
            event for event in seen
            if event["type"] == "conversation.item.input_audio_transcription.completed"
        )
        assert final["transcript"] == "hello world"
    finally:
        ws.__exit__(None, None, None)


def test_realtime_v2_rejects_unavailable_server_managed_response(
    fake_asr_backend, monkeypatch
):
    from fastapi.testclient import TestClient
    from server.main import app

    monkeypatch.delenv("OVS_V2V_ENGINE", raising=False)
    monkeypatch.delenv("OVS_V2V_SERVER_LOOP", raising=False)
    client = TestClient(app)
    ws = client.websocket_connect(
        "/v2v/stream", subprotocols=["seeed.realtime.v2"]
    )
    ws.__enter__()
    try:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "audio": {
                    "input": {
                        "format": {"sample_rate": 16000, "channels": 1},
                        "transcription": {"language": "en"},
                        "turn_detection": {
                            "type": "server_vad",
                            "create_response": True,
                        },
                    },
                    "output": {},
                },
            },
        })
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["error"]["code"] == "unsupported_create_response"
        assert event["error"]["param"].endswith("create_response")
    finally:
        ws.__exit__(None, None, None)


def test_realtime_v2_cloud_provider_fails_with_structured_credential_error(
    fake_asr_backend, monkeypatch
):
    from fastapi.testclient import TestClient
    from server.main import app

    monkeypatch.setenv("OVS_REALTIME_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    ws = client.websocket_connect(
        "/v2v/stream", subprotocols=["seeed.realtime.v2"]
    )
    ws.__enter__()
    try:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["session"]["provider"] == "openai"
        ws.send_json({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 16000},
                        "transcription": {"language": "en"},
                        "turn_detection": {
                            "type": "server_vad", "create_response": False,
                        },
                    },
                    "output": {"language": "en"},
                },
            },
        })
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == "provider_connection_error"
        assert "OPENAI_API_KEY" in error["error"]["message"]
    finally:
        ws.__exit__(None, None, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
