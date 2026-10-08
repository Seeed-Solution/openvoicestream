from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "configs" / "profiles"
COMPOSE = ROOT / "deploy" / "docker-compose.kokoro-long32-canary.yml"


def _profile(platform: str) -> dict:
    path = PROFILE_DIR / f"{platform}-kokoro-long32-canary.json"
    return json.loads(path.read_text())


def test_each_candidate_is_target_specific_and_has_no_unresolved_artifact_set():
    for platform in ("rk3588", "rk3576"):
        profile = _profile(platform)
        env = profile["env"]
        assert profile["name"] == f"{platform}-kokoro-long32-canary"
        assert env["RK_PLATFORM"] == platform
        assert env["KOKORO_LONG32_PLATFORM"] == platform
        assert env["KOKORO_RKNN_PLATFORM"] == platform
        assert env["KOKORO_RKNN_MODE"] == "long32"
        assert env["RK_ENSURE_MATCHA_RESOURCES"] == "0"
        assert env["ASR_BACKEND"] == "disabled"
        assert env["RK_ARTIFACT_AUTO_DOWNLOAD"] == "0"
        assert "RK_ARTIFACT_SET" not in env
        assert env["KOKORO_LONG32_ROOT"].endswith(f"/{platform}")
        assert env["KOKORO_LONG32_FRONTEND_ROOT"] == "/opt/kokoro-long32-frontend"
        assert env["KOKORO_LONG32_ROUTE_TS"] == ("400,640" if platform == "rk3588" else "416,640")
        assert platform in env["KOKORO_FRONT_RKNN"]
        assert platform in env["KOKORO_RKNN_VOCODER_FRONT_PATH"]


def test_compose_isolated_canary_and_explicit_rollback_mounts():
    text = COMPOSE.read_text()
    assert "speech-kokoro-long32-canary:" in text
    assert "container_name: ${KOKORO_CANARY_CONTAINER:-openvoicestream-kokoro-long32-canary}" in text
    assert "network_mode: host" in text
    assert "--port" in text and "8622" in text
    assert "8621" not in text
    assert "/opt/kokoro-hybrid:ro" in text
    assert "/opt/kokoro-long32:ro" in text
    assert "/opt/kokoro-long32-frontend:ro" in text
    assert "ASR_BACKEND=disabled" in text
    assert "TTS_BACKEND=kokoro_rknn" in text
    assert "KOKORO_RKNN_MODE=long32" in text
    assert "RK_ENSURE_MATCHA_RESOURCES=0" in text
    assert "KOKORO_LONG32_ROUTE_TS=${KOKORO_LONG32_ROUTE_TS:-}" in text
    assert "RK_ARTIFACT_AUTO_DOWNLOAD=0" in text
    assert "http://127.0.0.1:8622/health" in text
    assert "KOKORO_CANARY_PORT" not in text


def test_compose_requires_one_target_and_reviewed_image():
    text = COMPOSE.read_text()
    assert "KOKORO_LONG32_VOICE_IMAGE:?" in text
    assert "KOKORO_CANARY_TARGET:?" in text
    assert "OVS_PROFILE=${KOKORO_CANARY_TARGET:" in text
    assert "KOKORO_RKNN_PLATFORM=${KOKORO_CANARY_TARGET:" in text
    assert "KOKORO_LONG32_ROOT=/opt/kokoro-long32/${KOKORO_CANARY_TARGET:" in text
    assert "KOKORO_LONG32_FRONTEND_ROOT=/opt/kokoro-long32-frontend" in text
    assert "KOKORO_HYBRID_HOST_MODEL_DIR:?" in text
    assert "KOKORO_LONG32_HOST_MODEL_DIR:?" in text
    assert "KOKORO_LONG32_FRONTEND_HOST_DIR:?" in text
    assert "/home/radxa/" not in text
    assert "KOKORO_FRONT_RKNN=${KOKORO_FRONT_RKNN:-${KOKORO_CANARY_TARGET:" in text
    assert "KOKORO_LONG32_FALLBACK=${KOKORO_LONG32_FALLBACK:-hybrid}" in text
    assert "KOKORO_LONG32_ENFORCE_RTF=${KOKORO_LONG32_ENFORCE_RTF:-0}" in text
