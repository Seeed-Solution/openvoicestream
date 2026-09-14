"""Per-session ASR stream options from the v2v config.

`vad_endpoint_silence_ms` is the one option today: the backend endpoint
threshold for ASR backends that own their own endpoint VAD (RK Qwen3 — see
docs/CONFIGURATION.md "Streaming ASR endpointing"). Before this channel
existed the only lever was the profile, and the profile stamped the env value
back at startup, so a client could not tune it per session at all.

Rules under test: absent → backend defaults; valid int → forwarded as int;
0 → backend defaults; unparseable / out-of-range → ignored with a warning, and
the session keeps working.
"""
from __future__ import annotations

import logging

from server.main import _v2v_asr_stream_options


def test_absent_key_keeps_backend_defaults():
    assert _v2v_asr_stream_options({}) == {}


def test_valid_value_is_forwarded_as_int():
    assert _v2v_asr_stream_options({"vad_endpoint_silence_ms": 1500}) == {
        "vad_endpoint_silence_ms": 1500
    }
    # Numeric strings come in through --env-file / form-ish clients.
    assert _v2v_asr_stream_options({"vad_endpoint_silence_ms": "1200"}) == {
        "vad_endpoint_silence_ms": 1200
    }


def test_zero_means_backend_default():
    assert _v2v_asr_stream_options({"vad_endpoint_silence_ms": 0}) == {}


def test_invalid_values_are_ignored_not_fatal(caplog):
    with caplog.at_level(logging.WARNING):
        assert _v2v_asr_stream_options({"vad_endpoint_silence_ms": "soon"}) == {}
        assert _v2v_asr_stream_options({"vad_endpoint_silence_ms": -1}) == {}
        assert _v2v_asr_stream_options({"vad_endpoint_silence_ms": 60001}) == {}
        assert _v2v_asr_stream_options({"vad_endpoint_silence_ms": None}) == {}
    assert "ignoring invalid vad_endpoint_silence_ms" in caplog.text
    assert "ignoring out-of-range vad_endpoint_silence_ms" in caplog.text


def test_realtime_v2_turn_detection_carries_the_option():
    """V2 clients put the knob next to the other turn-boundary fields."""
    from server.core.v2v import session_update_to_legacy_config

    payload = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "transcription": {"language": "zh"},
                    "turn_detection": {
                        "type": "none",
                        "vad_endpoint_silence_ms": 1500,
                    },
                },
                "output": {"format": {"type": "audio/pcm", "rate": 16000},
                           "language": "zh"},
            },
        },
    }
    cfg = session_update_to_legacy_config(payload)
    assert _v2v_asr_stream_options(cfg) == {"vad_endpoint_silence_ms": 1500}


def test_realtime_v2_without_the_option_keeps_profile_value():
    from server.core.v2v import session_update_to_legacy_config

    payload = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "turn_detection": {"type": "none"},
                },
                "output": {"format": {"type": "audio/pcm", "rate": 16000}},
            },
        },
    }
    cfg = session_update_to_legacy_config(payload)
    assert _v2v_asr_stream_options(cfg) == {}


def test_operator_env_used_when_session_is_silent(monkeypatch):
    """A deployment tunes the threshold without editing the image profile.

    Named OVS_V2V_* on purpose: the profile loader re-stamps its own
    VAD_ENDPOINT_SILENCE_MS at startup, so that key never reaches the backend.
    """
    monkeypatch.setenv("OVS_V2V_VAD_ENDPOINT_SILENCE_MS", "2200")
    assert _v2v_asr_stream_options({}) == {"vad_endpoint_silence_ms": 2200}


def test_session_value_wins_over_operator_default(monkeypatch):
    monkeypatch.setenv("OVS_V2V_VAD_ENDPOINT_SILENCE_MS", "2200")
    assert _v2v_asr_stream_options({"vad_endpoint_silence_ms": 900}) == {
        "vad_endpoint_silence_ms": 900
    }


def test_invalid_or_zero_operator_default_falls_back_to_backend(monkeypatch):
    for raw in ("soon", "-5", "60001", "0"):
        monkeypatch.setenv("OVS_V2V_VAD_ENDPOINT_SILENCE_MS", raw)
        assert _v2v_asr_stream_options({}) == {}, raw


def test_no_operator_default_means_backend_default(monkeypatch):
    monkeypatch.delenv("OVS_V2V_VAD_ENDPOINT_SILENCE_MS", raising=False)
    assert _v2v_asr_stream_options({}) == {}


def test_health_exposes_the_endpointing_capability(monkeypatch):
    """Clients pick exactly one detector; /health must tell them which.

    Added with the 2026-09-13 endpoint work: the demo page reads
    ``asr_owns_endpointing`` instead of guessing from the backend name.
    """
    import asyncio
    import json as _json

    from server import main as server_main

    class _Be:
        name = "rk:qwen3_asr_rk"
        capabilities = []
        prefer_backend_endpoint_vad = True

        def is_ready(self):
            return True

    monkeypatch.setattr(server_main, "_get_asr_backend", lambda: _Be())
    resp = asyncio.run(server_main.health())
    payload = _json.loads(resp.body)
    assert payload["asr_owns_endpointing"] is True


def test_health_reports_false_for_backends_that_do_not_endpoint():
    import asyncio
    import json as _json

    from server import main as server_main

    class _Be:
        name = "sherpa:sensevoice"
        capabilities = []
        prefer_backend_endpoint_vad = False

        def is_ready(self):
            return True

    original = server_main._get_asr_backend
    server_main._get_asr_backend = lambda: _Be()
    try:
        payload = _json.loads(asyncio.run(server_main.health()).body)
    finally:
        server_main._get_asr_backend = original
    assert payload["asr_owns_endpointing"] is False
