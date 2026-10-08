from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.core.rk_profile_contract import runtime_status


_PIPER_PROFILE = Path(__file__).parents[2] / "configs" / "profiles" / "rk3576-piper.json"


def _profile(device: str = "rk3576") -> dict:
    name = f"{device}-default"
    env = {
        "RK_PLATFORM": device,
        "ASR_MAX_NEW_TOKENS": "64",
        "ASR_FINAL_STOP_ON_PUNCT": "0",
        "ASR_FINAL_STOP_MIN_CHARS": "8",
        "ASR_FINAL_STOP_MIN_CHUNKS": "2",
        "ASR_NPU_CORE_MASK": "NPU_CORE_1",
        "QWEN3_ASR_CHUNK_CONFIRM": "0",
        "QWEN3_ASR_STREAM_MODE": "true_streaming",
        "QWEN3_ASR_STREAM_TRUE": "1",
        "QWEN3_ASR_TRUE_ROLL_SEC": "5",
        "QWEN3_ASR_TRUE_PARTIAL_TOKENS": "8",
        "QWEN3_ASR_TRUE_PARTIAL_INTERVAL_MS": "1500",
        "QWEN3_ASR_TRUE_PARTIAL_WARMUP": "2",
        "QWEN3_ASR_VAD_FINAL_ASYNC": "0" if device == "rk3576" else "1",
        "QWEN3_ASR_FRONTEND_EOU_MIN_AUDIO_S": "2.5",
        "VAD_ENDPOINT_SILENCE_MS": "1500",
        "MATCHA_USE_ORT": "1",
        "MATCHA_MODEL_SEQ_LEN": "80",
        "MATCHA_MIN_MEL_FRAMES": "96" if device == "rk3576" else "72",
        "MATCHA_STREAM_CHUNK_MS": "40",
        "VOCOS_FRAMES": "600" if device == "rk3576" else "256",
    }
    return {"name": name, "env": env}


def test_release_profile_contract_reports_verified_effective_runtime():
    profile = _profile()
    status = runtime_status(profile, profile["env"])
    assert status["required"] is True
    assert status["verified"] is True
    assert status["missing_profile"] == []
    assert status["missing_runtime"] == []
    assert status["mismatches"] == {}


def test_release_profile_contract_catches_batch_mode_and_auto_npu():
    profile = _profile("rk3588")
    runtime = dict(profile["env"])
    runtime.update(
        QWEN3_ASR_STREAM_MODE="window",
        QWEN3_ASR_STREAM_TRUE="0",
        ASR_NPU_CORE_MASK="NPU_CORE_AUTO",
        VAD_ENDPOINT_SILENCE_MS="800",
    )
    status = runtime_status(profile, runtime)
    assert status["verified"] is False
    assert set(status["mismatches"]) >= {
        "QWEN3_ASR_STREAM_MODE",
        "QWEN3_ASR_STREAM_TRUE",
        "ASR_NPU_CORE_MASK",
        "VAD_ENDPOINT_SILENCE_MS",
    }


def test_release_profile_contract_rejects_cross_platform_async_final_policy():
    for device, wrong in (("rk3576", "1"), ("rk3588", "0")):
        profile = _profile(device)
        runtime = dict(profile["env"])
        runtime["QWEN3_ASR_VAD_FINAL_ASYNC"] = wrong
        status = runtime_status(profile, runtime)
        assert status["verified"] is False
        assert status["mismatches"]["QWEN3_ASR_VAD_FINAL_ASYNC"]["expected"] != wrong


def test_release_profile_contract_enforces_platform_matcha_context():
    for device, wrong in (("rk3576", "72"), ("rk3588", "96")):
        profile = _profile(device)
        runtime = dict(profile["env"])
        runtime["MATCHA_MIN_MEL_FRAMES"] = wrong
        status = runtime_status(profile, runtime)
        assert status["verified"] is False
        assert status["mismatches"]["MATCHA_MIN_MEL_FRAMES"]["expected"] != wrong


def test_non_release_profile_is_not_subject_to_contract():
    status = runtime_status({"name": "rk3576-sensevoice", "env": {}}, {})
    assert status["required"] is False
    assert status["verified"] is True


def test_rk3576_piper_profile_keeps_artifact_seq_len_and_matcha_fallback():
    profile = json.loads(_PIPER_PROFILE.read_text(encoding="utf-8"))
    status = runtime_status(profile, profile["env"])
    assert status["required"] is True
    assert status["verified"] is True
    assert status["settings"]["PIPER_SEQ_LEN"] == "256"
    assert status["settings"]["PIPER_CHINESE_FALLBACK"] == "matcha_rknn"
    assert status["settings"]["PIPER_ENABLE_FRONTEND_NPU"] == "1"

    drifted = runtime_status(profile, dict(profile["env"], PIPER_SEQ_LEN="128"))
    assert drifted["verified"] is False
    assert drifted["mismatches"]["PIPER_SEQ_LEN"]["expected"] == "256"

    missing = dict(profile, env=dict(profile["env"]))
    missing["env"].pop("PIPER_CHINESE_FALLBACK")
    status = runtime_status(missing, missing["env"])
    assert status["verified"] is False
    assert "PIPER_CHINESE_FALLBACK" in status["missing_profile"]


@pytest.mark.parametrize("toggle", ["0", "1"])
def test_rk3576_piper_frontend_toggle_accepts_matching_cpu_or_npu_mode(toggle):
    profile = json.loads(_PIPER_PROFILE.read_text(encoding="utf-8"))
    profile["env"]["PIPER_ENABLE_FRONTEND_NPU"] = toggle
    status = runtime_status(profile, dict(profile["env"]))
    assert status["verified"] is True
    assert status["settings"]["PIPER_ENABLE_FRONTEND_NPU"] == toggle


def test_rk3576_piper_frontend_toggle_rejects_drift_missing_and_invalid_values():
    profile = json.loads(_PIPER_PROFILE.read_text(encoding="utf-8"))

    drifted = runtime_status(profile, dict(profile["env"], PIPER_ENABLE_FRONTEND_NPU="0"))
    assert drifted["verified"] is False
    assert drifted["mismatches"]["PIPER_ENABLE_FRONTEND_NPU"]["expected"] == "1"

    missing_profile = dict(profile, env=dict(profile["env"]))
    missing_profile["env"].pop("PIPER_ENABLE_FRONTEND_NPU")
    status = runtime_status(missing_profile, dict(profile["env"]))
    assert status["verified"] is False
    assert "PIPER_ENABLE_FRONTEND_NPU" in status["missing_profile"]

    invalid_profile = dict(profile, env=dict(profile["env"], PIPER_ENABLE_FRONTEND_NPU="2"))
    status = runtime_status(invalid_profile, dict(invalid_profile["env"]))
    assert status["verified"] is False
    assert status["mismatches"]["PIPER_ENABLE_FRONTEND_NPU"]["expected"] == "0 or 1"

    invalid_runtime = runtime_status(profile, dict(profile["env"], PIPER_ENABLE_FRONTEND_NPU="yes"))
    assert invalid_runtime["verified"] is False
    assert invalid_runtime["mismatches"]["PIPER_ENABLE_FRONTEND_NPU"]["expected"] == "1"

    whitespace_profile = dict(
        profile, env=dict(profile["env"], PIPER_ENABLE_FRONTEND_NPU=" 1")
    )
    status = runtime_status(whitespace_profile, dict(whitespace_profile["env"]))
    assert status["verified"] is False
    assert status["mismatches"]["PIPER_ENABLE_FRONTEND_NPU"]["expected"] == "0 or 1"

    whitespace_runtime = runtime_status(
        profile, dict(profile["env"], PIPER_ENABLE_FRONTEND_NPU="1 ")
    )
    assert whitespace_runtime["verified"] is False
    assert whitespace_runtime["mismatches"]["PIPER_ENABLE_FRONTEND_NPU"]["expected"] == "1"


def test_final_punctuation_stop_is_off_on_both_platforms():
    # It cut a two-sentence utterance to its first sentence and saved one EOS
    # token; measured on RK3588 and RK3576 (see rk_profile_contract.py).
    for device in ("rk3588", "rk3576"):
        profile = _profile(device)
        status = runtime_status(profile, profile["env"])
        assert status["verified"] is True
        assert status["settings"]["ASR_FINAL_STOP_ON_PUNCT"] == "0"

        drifted = runtime_status(profile, dict(profile["env"], ASR_FINAL_STOP_ON_PUNCT="1"))
        assert drifted["verified"] is False
        assert "ASR_FINAL_STOP_ON_PUNCT" in drifted["mismatches"]
