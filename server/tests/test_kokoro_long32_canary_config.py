from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


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


# ── compose -> profile_loader integration ────────────────────────────────
#
# The profile owns the reviewed two-bucket route. Compose must not inject an
# empty KOKORO_LONG32_ROUTE_TS when the operator did not set one; an explicit
# operator value is an override. These tests render the compose environment
# for each target, start a fresh interpreter with exactly that environment,
# apply the profile and read back the effective route.

_HOST_ENV = {
    "KOKORO_LONG32_VOICE_IMAGE": "reviewed/candidate:test",
    "KOKORO_HYBRID_HOST_MODEL_DIR": "/srv/kokoro-hybrid",
    "KOKORO_LONG32_HOST_MODEL_DIR": "/srv/kokoro-long32",
    "KOKORO_LONG32_FRONTEND_HOST_DIR": "/srv/kokoro-long32-frontend",
}
_PROFILE_ROUTE = {"rk3588": "400,640", "rk3576": "416,640"}
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:-|:\?)([^${}]*))?\}")


def _interpolate(value: str, host: dict) -> str:
    """Compose-style ${VAR}, ${VAR:-default}, ${VAR:?err}, innermost first."""
    while True:
        m = _VAR.search(value)
        if m is None:
            return value
        name, op, arg = m.group(1), m.group(2), m.group(3) or ""
        cur = host.get(name, "")
        if op == ":?" and not cur:
            raise ValueError(f"{name} required: {arg}")
        if op == ":-" and not cur:
            cur = arg
        value = value[: m.start()] + cur + value[m.end():]


def _compose_environment_entries() -> list[str]:
    lines = COMPOSE.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "environment:")
    indent = len(lines[start]) - len(lines[start].lstrip())
    entries = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        assert stripped.startswith("- "), line
        entries.append(stripped[2:])
    return entries


def _container_env(host: dict) -> dict:
    """Environment the container gets, as compose builds it from ``host``."""
    env = {}
    for entry in _compose_environment_entries():
        if "=" in entry:
            key, raw = entry.split("=", 1)
            env[key] = _interpolate(raw, host)
        elif entry in host:
            # Bare key: forwarded only when set on the host.
            env[entry] = host[entry]
    return env


def _effective_route(container_env: dict) -> str | None:
    code = (
        "import os\n"
        "from server.core import profile_loader\n"
        "profile_loader.apply_profile_from_env()\n"
        "print(os.environ.get('KOKORO_LONG32_ROUTE_TS', '<unset>'))\n"
    )
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT), **container_env}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return out[-1]


@pytest.mark.parametrize("target", ["rk3588", "rk3576"])
def test_unset_route_leaves_profile_two_bucket_route_effective(target):
    host = {**_HOST_ENV, "KOKORO_CANARY_TARGET": target}
    container = _container_env(host)
    assert "KOKORO_LONG32_ROUTE_TS" not in container
    assert container["OVS_PROFILE"] == f"{target}-kokoro-long32-canary"
    assert _effective_route(container) == _PROFILE_ROUTE[target]


@pytest.mark.parametrize("target", ["rk3588", "rk3576"])
def test_explicit_operator_route_overrides_profile(target):
    host = {**_HOST_ENV, "KOKORO_CANARY_TARGET": target,
            "KOKORO_LONG32_ROUTE_TS": "320,640"}
    container = _container_env(host)
    assert container["KOKORO_LONG32_ROUTE_TS"] == "320,640"
    assert _effective_route(container) == "320,640"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker compose not available")
@pytest.mark.parametrize("target", ["rk3588", "rk3576"])
def test_interpolation_helper_matches_docker_compose(target):
    """Pins the helper above to real compose rendering where docker exists."""
    host = {**_HOST_ENV, "KOKORO_CANARY_TARGET": target}
    proc = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", "--format", "json"],
        env={"PATH": os.environ.get("PATH", ""), **host},
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"docker compose config unavailable: {proc.stderr.strip()[:200]}")
    rendered = json.loads(proc.stdout)["services"]["speech-kokoro-long32-canary"]["environment"]
    rendered = {k: v for k, v in rendered.items() if v is not None}
    assert rendered == _container_env(host)
