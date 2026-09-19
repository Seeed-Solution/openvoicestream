from __future__ import annotations

import importlib.util
import struct
import time
from pathlib import Path


# Moved out of the repo root by 9017136 ("slim repo root — relocate evidence"),
# which renamed validation/ to docs/validation/ and added validation/ to
# .gitignore. This path was not updated with it, so every test in this file has
# raised FileNotFoundError from the loader ever since — a red that looked like
# three failing assertions but was one stale path.
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "validation"
    / "v091-runtime-qualification-20260725"
    / "tts-http-stream-gate.py"
)


def _load_gate():
    spec = importlib.util.spec_from_file_location("v091_tts_http_stream_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, status: int, payload: bytes):
        self.status_code = status
        self._payload = payload

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self._payload

    def close(self):
        return None


def test_cancel_json_error_cannot_masquerade_as_pcm(monkeypatch):
    gate = _load_gate()
    monkeypatch.setattr(
        gate.requests,
        "post",
        lambda *args, **kwargs: _Response(503, b'{"error":"startup failed"}'),
    )
    monkeypatch.setattr(
        gate,
        "stream_once",
        lambda *args, **kwargs: {"status": 200, "passed": True},
    )

    result = gate.cancel_recovery(
        "http://device",
        timeout=1,
        expected_sample_rate=24000,
        recovery_timeout=1,
    )

    assert result["cancel_status"] == 503
    assert result["cancel_stream_valid"] is False
    assert result["passed"] is False


def test_cancel_recovery_allows_temporary_429_then_valid_pcm(monkeypatch):
    gate = _load_gate()
    payload = struct.pack("<I", 24000) + b"\x01\x00" * 8
    monkeypatch.setattr(
        gate.requests,
        "post",
        lambda *args, **kwargs: _Response(200, payload),
    )
    attempts = iter(
        [
            {"status": 429, "retry_after": "10", "passed": False},
            {
                "status": 200,
                "retry_after": None,
                "sample_rate": 24000,
                "pcm_bytes": 16,
                "passed": True,
            },
        ]
    )
    monkeypatch.setattr(gate, "stream_once", lambda *args, **kwargs: next(attempts))
    sleep_delays = []
    monkeypatch.setattr(gate.time, "sleep", sleep_delays.append)

    result = gate.cancel_recovery(
        "http://device",
        timeout=1,
        expected_sample_rate=24000,
        recovery_timeout=0.02,
    )

    assert result["cancel_stream_valid"] is True
    assert result["recovery_429_count"] == 1
    assert len(sleep_delays) == 1
    assert 0 < sleep_delays[0] <= 0.02
    assert result["recovery"]["passed"] is True
    assert result["passed"] is True


def test_cancel_recovery_rejects_success_after_wall_deadline(monkeypatch):
    gate = _load_gate()
    payload = struct.pack("<I", 24000) + b"\x01\x00" * 8
    monkeypatch.setattr(
        gate.requests,
        "post",
        lambda *args, **kwargs: _Response(200, payload),
    )

    def _late_success(*args, **kwargs):
        del args, kwargs
        time.sleep(0.06)
        return {"status": 200, "passed": True}

    monkeypatch.setattr(gate, "stream_once", _late_success)

    result = gate.cancel_recovery(
        "http://device",
        timeout=1,
        expected_sample_rate=24000,
        recovery_timeout=0.01,
    )

    assert result["recovery"]["status"] == 200
    assert result["recovery"]["passed"] is False
    assert result["recovery_deadline_met"] is False
    assert result["passed"] is False
