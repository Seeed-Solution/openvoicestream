import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]

_RELEASE_PROFILES = (
    "rk3576-default",
    "rk3576-multilang",
    "rk3588-default",
    "rk3588-multilang",
)

_PROFILE_DEFAULT_KEYS = (
    "ASR_MAX_NEW_TOKENS",
    "ASR_FINAL_STOP_ON_PUNCT",
    "ASR_FINAL_STOP_MIN_CHARS",
    "ASR_FINAL_STOP_MIN_CHUNKS",
    "ASR_NPU_CORE_MASK",
    "QWEN3_ASR_CHUNK_CONFIRM",
    "QWEN3_ASR_STREAM_MODE",
    "QWEN3_ASR_STREAM_TRUE",
    "QWEN3_ASR_TRUE_ROLL_SEC",
    "QWEN3_ASR_TRUE_PARTIAL_TOKENS",
    "QWEN3_ASR_TRUE_PARTIAL_INTERVAL_MS",
    "QWEN3_ASR_TRUE_PARTIAL_WARMUP",
    "QWEN3_ASR_VAD_FINAL_ASYNC",
    "QWEN3_ASR_FRONTEND_EOU_MIN_AUDIO_S",
    "VAD_ENDPOINT_SILENCE_MS",
    "MATCHA_USE_ORT",
    "MATCHA_MODEL_SEQ_LEN",
    "MATCHA_MIN_MEL_FRAMES",
    "MATCHA_STREAM_CHUNK_MS",
)


def _profile_env(name: str) -> dict[str, str]:
    data = json.loads((ROOT / "configs" / "profiles" / f"{name}.json").read_text())
    return data["env"]


def test_rk_profiles_do_not_shadow_true_streaming():
    profiles = [
        "rk3576-default",
        "rk3576-multilang",
        "rk3588-default",
        "rk3588-multilang",
    ]
    for name in profiles:
        env = _profile_env(name)
        assert env["QWEN3_ASR_STREAM_TRUE"] == "1"
        assert env["QWEN3_ASR_CHUNK_CONFIRM"] == "0"


def test_rk_qwen3_asr_release_profiles_use_optimized_w8a8_defaults():
    for name in _RELEASE_PROFILES:
        env = _profile_env(name)
        assert env["ASR_DECODER_QUANT"] == "w8a8"
        assert env["ASR_ENABLED_CPUS"] == "4"
        assert env["ASR_MAX_NEW_TOKENS"] == "64"
        assert env["ASR_FINAL_STOP_ON_PUNCT"] == "1"
        assert env["QWEN3_ASR_TRUE_ROLL_SEC"] == "5"
        assert env["QWEN3_ASR_TRUE_PARTIAL_TOKENS"] == "8"


def test_rk_release_profiles_contain_the_complete_latency_contract():
    for name in _RELEASE_PROFILES:
        env = _profile_env(name)
        for key in _PROFILE_DEFAULT_KEYS:
            assert key in env, (name, key)
        assert env["ASR_NPU_CORE_MASK"] == "NPU_CORE_1"
        # Must stay above the client VAD silence of the shipped agent (600 ms)
        # plus the EOS->finalize budget, or the backend endpoint race truncates
        # mid-sentence. See server/core/rk_profile_contract.py for the measured
        # 400/1200/5000 ms comparison.
        assert int(env["VAD_ENDPOINT_SILENCE_MS"]) >= 1200
        expected_min_mel = "96" if env["RK_PLATFORM"] == "rk3576" else "72"
        assert env["MATCHA_MIN_MEL_FRAMES"] == expected_min_mel
        assert env["MATCHA_STREAM_CHUNK_MS"] == "40"
        expected_frames = "600" if name.startswith("rk3576-") else "256"
        assert env["VOCOS_FRAMES"] == expected_frames
        expected_async = "0" if name.startswith("rk3576-") else "1"
        assert env["QWEN3_ASR_VAD_FINAL_ASYNC"] == expected_async


def test_rk_artifact_contract_matches_platform_async_final_policy():
    manifest = json.loads(
        (ROOT / "deploy" / "artifacts" / "rk_manifest.json").read_text()
    )
    artifact_sets = manifest["artifact_sets"]
    for platform, expected in (("rk3576", "0"), ("rk3588", "1")):
        matching = [
            spec for name, spec in artifact_sets.items()
            if name.startswith(f"{platform}-")
            and "qwen3" in str(spec.get("runtime_contract", {}).get("asr_path", ""))
        ]
        assert matching, platform
        for spec in matching:
            contract = spec["runtime_contract"]["env"]
            assert contract["QWEN3_ASR_VAD_FINAL_ASYNC"] == expected


def _compose_environment(path: Path) -> dict[str, str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = document["services"]["speech"]["environment"]
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    return {
        str(item).split("=", 1)[0]: str(item).split("=", 1)[1]
        for item in raw
        if "=" in str(item)
    }


def test_rk_compose_does_not_shadow_profile_defaults():
    """Compose may expose an override, but must not turn profile defaults into
    import-time operator values.  Empty ``${KEY:-}`` expressions are passed to
    the container as empty and are intentionally ignored by profile_loader.
    """
    compose_files = (
        ROOT / "deploy" / "docker-compose.rk.yml",
        ROOT / "deploy" / "docker-compose.radxa.yml",
        ROOT / "deploy" / "docker-compose.rk3588-ha.yml",
    )
    for path in compose_files:
        env = _compose_environment(path)
        for key in _PROFILE_DEFAULT_KEYS:
            if key in env:
                assert env[key] == f"${{{key}:-}}", (path.name, key, env[key])
        assert env.get("ASR_NPU_CORE_MASK") != "NPU_CORE_AUTO", path.name
        assert env.get("VAD_ENDPOINT_SILENCE_MS") != "800", path.name
