"""Static deployment invariants for persistent user-created voice data."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILES = (
    ROOT / "deploy/docker-compose.yml",
    ROOT / "deploy/docker-compose.edgellm-v091-voice.yml",
    ROOT / "deploy/docker-compose.rpi.yml",
    ROOT / "deploy/docker-compose.rk.yml",
    ROOT / "deploy/docker-compose.rk3588.yml",
    ROOT / "deploy/spark/docker-compose.spark.yml",
)
DATA_VOLUME = "seeed-local-voice-data"
DATA_MOUNT = "/opt/seeed-local-voice/data"
SPEAKERS_FILE = f"{DATA_MOUNT}/speakers.json"
SPARK_DIR = f"{DATA_MOUNT}/sparktts_voices"


def _environment(service: dict) -> dict[str, str]:
    env = service.get("environment", {})
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    out: dict[str, str] = {}
    for item in env:
        key, value = str(item).split("=", 1)
        out[key] = value
    return out


def _mounts(service: dict) -> list[str]:
    mounts: list[str] = []
    for item in service.get("volumes", []):
        if isinstance(item, str):
            mounts.append(item)
        else:
            mounts.append(f"{item.get('source')}:{item.get('target')}")
    return mounts


def test_every_voice_service_has_external_data_volume_and_explicit_paths():
    for path in COMPOSE_FILES:
        document = yaml.safe_load(path.read_text())
        services = document["services"]
        # Every listed compose file has one service that hosts the voice API.
        service = services.get("speech") or services.get("spark-voice-stack")
        assert service is not None, path
        env = _environment(service)
        mounts = _mounts(service)
        assert f"{DATA_VOLUME}:{DATA_MOUNT}" in mounts, path
        assert env["OVS_TTS_SPEAKERS_FILE"] == SPEAKERS_FILE, path
        assert env["SPARKTTS_VOICES_DIR"] == SPARK_DIR, path
        # Models stay on a separate volume/path and are never redirected into
        # the user-data root.
        if path.name != "docker-compose.spark.yml":
            assert any("models" in mount.split(":", 1)[0] for mount in mounts), path
        assert any(DATA_MOUNT in mount for mount in mounts), path

        volume = document["volumes"][DATA_VOLUME]
        assert volume["name"] == DATA_VOLUME, path
        assert volume.get("external") is not True, path
