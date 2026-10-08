"""Tests for server.core.leaf_composition (TRACK 1 SLICE 1, standalone).

Covers:
  * union-pull ASR-invariance under TTS choice (+ shared sub-leaf de-dup);
  * precision default resolution + flip (model default re-resolves leaves);
  * validator rejects nano FP16-TTS-n2 (memory), unknown leaf id, double-TTS;
  * env precedence (leaf < overrides) + ${VAR} expansion.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from server.core import leaf_composition as lc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env():
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)

@pytest.fixture
def registry() -> lc.Registry:
    """The real on-disk registry under configs/leaves/."""
    return lc.load_registry()


ASR_N1 = "asr.qwen3_asr.orin-nx.n1"
ASR_N2 = "asr.qwen3_asr.orin-nx.n2"
TTS_N1 = "tts.qwen3_tts.orin-nx.n1"
TTS_N2 = "tts.qwen3_tts.orin-nx.n2"
TTS_SHARED = "tts.qwen3_tts.shared"


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def test_registry_loads_expected_leaves(registry):
    for lid in (ASR_N1, ASR_N2, TTS_N1, TTS_N2, TTS_SHARED):
        assert lid in registry.leaves, lid
    assert "orin-nx" in registry.devices
    assert "orin-nano" in registry.devices
    assert "qwen3-tts-customvoice" in registry.models


# ---------------------------------------------------------------------------
# Union-pull ASR invariance under TTS choice
# ---------------------------------------------------------------------------

def test_asr_pull_invariant_under_tts_choice(registry):
    asr_only = set(resolve_asr_files(registry))
    with_n1 = set(resolve_asr_files(registry, TTS_N1))
    with_n2 = set(resolve_asr_files(registry, TTS_N2))
    assert asr_only == with_n1 == with_n2
    assert asr_only  # non-empty


def resolve_asr_files(registry, tts_leaf=None):
    selected = [ASR_N2] + ([tts_leaf] if tts_leaf else [])
    all_files = lc.resolve_pull(selected, registry)
    asr_leaf = registry.get_leaf(ASR_N2)
    asr_files = set(asr_leaf.artifacts.files)
    return [f for f in all_files if f in asr_files]


def test_asr_n1_n2_same_engine_files(registry):
    assert lc.resolve_pull([ASR_N1], registry) == lc.resolve_pull([ASR_N2], registry)


def test_pull_dedups_shared_subleaf(registry):
    # Both TTS leaves require the same shared sub-leaf; its files appear once.
    files = lc.resolve_pull([TTS_N1, TTS_N2], registry)
    assert len(files) == len(set(files))
    shared = registry.get_leaf(TTS_SHARED)
    for f in shared.artifacts.files:
        assert files.count(f) == 1


def test_pull_is_deterministic(registry):
    a = lc.resolve_pull([ASR_N2, TTS_N1], registry)
    b = lc.resolve_pull([ASR_N2, TTS_N1], registry)
    assert a == b


# ---------------------------------------------------------------------------
# Precision resolution + flip
# ---------------------------------------------------------------------------

def test_precision_resolves_from_model_default(registry):
    leaf = registry.get_leaf(TTS_N2)
    assert leaf.precision is None  # unset on the leaf
    assert registry.resolve_precision(leaf, "orin-nx") == "fp16"


def test_precision_flip_reresolves_all_leaves(registry):
    # Flip the model default jetson fp16 -> w8a16; every leaf re-resolves.
    flipped = lc.ModelSpec(
        id="qwen3-tts-customvoice",
        default_precision={"jetson": "w8a16"},
    )
    registry.models["qwen3-tts-customvoice"] = flipped
    for lid in (TTS_N1, TTS_N2, TTS_SHARED):
        leaf = registry.get_leaf(lid)
        assert registry.resolve_precision(leaf, "orin-nx") == "w8a16"


def test_explicit_leaf_precision_wins(registry):
    leaf = lc.Leaf(
        id="tts.x.test", capability="tts", model="qwen3-tts-customvoice",
        precision="w4a16",
    )
    assert registry.resolve_precision(leaf, "orin-nx") == "w4a16"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_validate_ok_on_nx(registry):
    # DELTA semantics: base_reservation(4400) + ASR-n2 delta(1925)
    # + TTS-n1 kv1024 delta(3039) + shared sub-leaf(0) = 9364 <= 15656*0.85=13307.
    plan = lc.validate_composition("orin-nx", [ASR_N2, TTS_N1], registry)
    assert plan.device == "orin-nx"
    assert set(plan.leaf_ids) == {ASR_N2, TTS_N1}
    # peak = device base reservation + leaf deltas (shared sub-leaf 0 MB once)
    assert plan.peak_unified_mb == 4400 + 1925 + 3039
    # precision resolved in the plan
    by_id = {r.leaf.id: r.precision for r in plan.resolved}
    assert by_id[TTS_N1] == "fp16"


def test_validate_ok_on_nx_asr_n2_tts_n2(registry):
    # The bug-motivating combo: absolute-sum (6335+9057=15392) falsely rejected
    # this; DELTA math accepts it.
    # base(4400) + ASR-n2 delta(1925) + TTS-n2 delta(4636) = 10961
    # <= 15656*0.85 = 13307 → PASS.
    plan = lc.validate_composition("orin-nx", [ASR_N2, TTS_N2], registry)
    assert plan.peak_unified_mb == 4400 + 1925 + 4636
    assert plan.peak_unified_mb <= plan.headroom_mb


def test_validate_ok_on_nx_asr_n2_tts_n1(registry):
    # base(4400) + ASR-n2 delta(1925) + TTS-n1 kv1024 delta(3039) = 9364 → PASS.
    plan = lc.validate_composition("orin-nx", [ASR_N2, TTS_N1], registry)
    assert plan.peak_unified_mb == 9364
    assert plan.peak_unified_mb <= plan.headroom_mb


def test_validate_rejects_nano_fp16_tts_n1_memory(registry):
    # Conservative product composition still rejects Nano: a standalone
    # 2026-06-12 kv1024 smoke can dual-open ASR-ready + TTS on a cleaned host,
    # but it used swap and is not a wide-margin default.
    # base(4400) + TTS-n1 delta(3039) = 7439 > 7864*0.85 = 6684 → REJECT.
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("orin-nano", [TTS_N1], registry)
    assert "memory budget" in str(exc.value)
    assert "orin-nano" in str(exc.value)


def test_validate_rejects_nano_fp16_tts_n2_memory(registry):
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("orin-nano", [ASR_N2, TTS_N2], registry)
    assert "memory budget" in str(exc.value)
    assert "orin-nano" in str(exc.value)


def test_validate_rejects_unknown_leaf_id(registry):
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("orin-nx", ["tts.qwen3_tts.orin-nano.n2"], registry)
    assert "not built" in str(exc.value)


def test_validate_rejects_double_tts(registry):
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("orin-nx", [TTS_N1, TTS_N2], registry)
    assert "illegal pairing" in str(exc.value)
    assert "tts" in str(exc.value)


def test_validate_rejects_unknown_device(registry):
    # NOTE: rk3588 was added to the device registry when the adapted RK/RPi/Mac
    # leaves landed, so use a genuinely-absent device name here.
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("nonexistent-device", [ASR_N2], registry)
    assert "unknown device" in str(exc.value)


# ---------------------------------------------------------------------------
# Env precedence + ${VAR} expansion
# ---------------------------------------------------------------------------

def test_env_merges_leaf_and_shared(registry):
    env = lc.resolve_env([TTS_N1], registry)
    # from the concrete leaf
    assert "EDGE_LLM_TTS_TALKER_ENGINE" in env
    # from the shared sub-leaf (pulled via requires)
    assert env["EDGE_LLM_TTS_TALKER_BACKEND"] == "qwen3_tts_explicit_kv"


def test_env_override_wins(registry):
    env = lc.resolve_env(
        [TTS_N1], registry,
        overrides={"EDGE_LLM_TTS_TALKER_BACKEND": "custom_override"},
    )
    assert env["EDGE_LLM_TTS_TALKER_BACKEND"] == "custom_override"


def test_env_var_expansion(registry):
    env = lc.resolve_env(
        [ASR_N2], registry,
        overrides={"QWEN3_ARTIFACT_ROOT": "/opt/models/qwen3-edgellm"},
    )
    assert env["EDGE_LLM_ASR_ENGINE_DIR"] == (
        "/opt/models/qwen3-edgellm/engines/orin-nx/highperf-v080/"
        "asr_thinker_full_fp8embed"
    )


def test_env_unknown_var_preserved(registry):
    # No QWEN3_ARTIFACT_ROOT override → the ${VAR} is preserved verbatim.
    env = lc.resolve_env([ASR_N2], registry)
    assert "${QWEN3_ARTIFACT_ROOT}" in env["EDGE_LLM_ASR_ENGINE_DIR"]


def test_resolve_env_standalone_module_level(registry):
    # Standalone: resolve_env does NOT read os.environ (env layer is later).
    import os
    os.environ["QWEN3_ARTIFACT_ROOT"] = "/should/not/leak"
    try:
        env = lc.resolve_env([ASR_N2], registry)
        assert "/should/not/leak" not in env["EDGE_LLM_ASR_ENGINE_DIR"]
        assert "${QWEN3_ARTIFACT_ROOT}" in env["EDGE_LLM_ASR_ENGINE_DIR"]
    finally:
        os.environ.pop("QWEN3_ARTIFACT_ROOT", None)


# ---------------------------------------------------------------------------
# SLICE 3: composition profile end-to-end + downloader required_files override
# ---------------------------------------------------------------------------

_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs" / "profiles" / "jetson-qwen3-composition-nx.json"
)


def _load_composition_profile() -> dict:
    return json.loads(_PROFILE_PATH.read_text())


def test_composition_profile_parses_and_resolves_union(registry):
    """The new composition profile parses and apply_composition() returns the
    de-duped union-pull list = exactly resolve_pull(asr, tts) over the leaves.

    Asserts against the real registry so a leaf-id typo in the profile is caught.
    """
    from server.core import composition_boot as cb

    profile = _load_composition_profile()
    assert "composition" in profile
    comp = profile["composition"]
    assert comp["device"] == "orin-nx"

    # The selected leaf ids must exist in the registry (no silent fallback).
    asr_id = comp["asr"]
    tts_id = comp["tts"]
    assert asr_id in registry.leaves, asr_id
    assert tts_id in registry.leaves, tts_id

    expected = lc.resolve_pull([asr_id, tts_id], registry)
    assert expected  # non-empty
    assert len(expected) == len(set(expected))  # de-duped

    pulls = cb.apply_composition(profile)
    assert pulls == expected


def test_flat_profile_no_composition_is_noop():
    """A profile WITHOUT a composition block yields an empty pull list."""
    from server.core import composition_boot as cb

    assert cb.apply_composition({"name": "flat-profile", "env": {}}) == []
    assert cb.apply_composition(None) == []


# --- model_downloader required_files override --------------------------------

def _write_qwen3_manifest(tmp_path: Path, required_files: list[str]) -> Path:
    manifest = {
        "artifact_sets": {
            "test-set": {
                "root": str(tmp_path / "artifacts"),
                "required_files": required_files,
            }
        }
    }
    mp = tmp_path / "qwen3_manifest.json"
    mp.write_text(json.dumps(manifest))
    return mp


def test_via_hf_uses_manifest_files_when_override_none(tmp_path, monkeypatch):
    """Flat path (override None): the artifact_set's own required_files drive
    the download — the override is ignored when empty."""
    from server.core import model_downloader as md

    manifest_files = ["engines/a", "engines/b"]
    mp = _write_qwen3_manifest(tmp_path, manifest_files)

    captured: dict = {}

    def _fake_ensure_artifacts(missing_paths):
        captured["paths"] = list(missing_paths)

    # Patch the symbol imported INSIDE the function (it does a fresh
    # `from server.core.qwen3_artifact_downloader import ensure_artifacts`).
    import server.core.qwen3_artifact_downloader as qad
    monkeypatch.setattr(qad, "ensure_artifacts", _fake_ensure_artifacts)
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")

    root = tmp_path / "artifacts"
    md._ensure_qwen3_artifacts_via_hf(str(mp), "test-set", None)

    expected = {str(root / rf) for rf in manifest_files}
    assert set(captured["paths"]) == expected


def test_via_hf_override_replaces_manifest_required_files(tmp_path, monkeypatch):
    """Composition path (non-empty override): the override REPLACES the
    artifact_set's required_files for the download set."""
    from server.core import model_downloader as md

    # Manifest declares DIFFERENT files than the override; the override wins.
    mp = _write_qwen3_manifest(tmp_path, ["engines/manifest-only"])
    override = ["engines/from-composition-1", "engines/from-composition-2"]

    captured: dict = {}

    def _fake_ensure_artifacts(missing_paths):
        captured["paths"] = list(missing_paths)

    import server.core.qwen3_artifact_downloader as qad
    monkeypatch.setattr(qad, "ensure_artifacts", _fake_ensure_artifacts)
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")

    root = tmp_path / "artifacts"
    md._ensure_qwen3_artifacts_via_hf(str(mp), "test-set", override)

    expected = {str(root / rf) for rf in override}
    assert set(captured["paths"]) == expected
    # The manifest-only file must NOT have been requested.
    assert str(root / "engines/manifest-only") not in set(captured["paths"])


# ---------------------------------------------------------------------------
# NEW adapted backend leaves (paraformer / sensevoice / rk-qwen3 / matcha /
# kokoro / moss) + new validator checks (family coupling, NPU exclusivity).
# All 25 tests above stay green: the new leaves declare new ids/devices and the
# new validator checks are opt-in (no-op for the qwen3 jetson leaves).
# ---------------------------------------------------------------------------

# ASR
PARAFORMER_TRT = "asr.paraformer_trt.orin.n2"
PARAFORMER_SHARED = "asr.paraformer.shared"
SENSEVOICE_TRT = "asr.sensevoice_trt.orin.n1"
SENSEVOICE_RKNN_3576 = "asr.sensevoice_rknn.rk3576.n1"
SENSEVOICE_RKNN_3588 = "asr.sensevoice_rknn.rk3588.n1"
SENSEVOICE_SHERPA_RPI5 = "asr.sensevoice_sherpa.rpi5.n4"
SENSEVOICE_SHARED = "asr.sensevoice.shared"
QWEN3_ASR_RK_3576_W8A8 = "asr.qwen3_asr_rk.rk3576.w8a8"
QWEN3_ASR_RK_3576_W4A16 = "asr.qwen3_asr_rk.rk3576.w4a16g128"
QWEN3_ASR_RK_3588_W8A8 = "asr.qwen3_asr_rk.rk3588.w8a8"
PARAFORMER_RKNN_3576 = "asr.paraformer_rknn.rk3576.hybrid"
PARAFORMER_RKNN_3588 = "asr.paraformer_rknn.rk3588.hybrid"
PARAFORMER_RKNN_3576_FULL = "asr.paraformer_rknn.rk3576.full"
PARAFORMER_SHERPA_RPI5 = "asr.paraformer_sherpa.rpi5.n4"
PARAFORMER_SHERPA_RPI4 = "asr.paraformer_sherpa.rpi4.n4"
PARAFORMER_SHERPA_MAC = "asr.paraformer_sherpa.mac.n4"

# TTS
MATCHA_TRT = "tts.matcha_trt.orin.n2"
MATCHA_SHERPA_RPI5 = "tts.matcha_sherpa.rpi5.n4"
MATCHA_RKNN_3576 = "tts.matcha_rknn.rk3576.n1"
MATCHA_RKNN_3588 = "tts.matcha_rknn.rk3588.n1"
MATCHA_SHARED = "tts.matcha.shared"
KOKORO_TRT_PERF = "tts.kokoro_trt.orin.perf.n2"
KOKORO_TRT_QUALITY = "tts.kokoro_trt.orin.quality.n2"
KOKORO_TRT_LONG = "tts.kokoro_trt.orin.long.n2"
KOKORO_RKNN_3588 = "tts.kokoro_rknn.rk3588.3stage"
MOSS_TRT = "tts.moss_tts_nano.orin.trt.n2"
MOSS_ORT = "tts.moss_tts_nano.orin.ort.n1"

_ALL_NEW_LEAVES = [
    PARAFORMER_TRT, PARAFORMER_SHARED,
    SENSEVOICE_TRT, SENSEVOICE_RKNN_3576, SENSEVOICE_RKNN_3588,
    SENSEVOICE_SHERPA_RPI5, SENSEVOICE_SHARED,
    QWEN3_ASR_RK_3576_W8A8, QWEN3_ASR_RK_3576_W4A16, QWEN3_ASR_RK_3588_W8A8,
    PARAFORMER_RKNN_3576, PARAFORMER_RKNN_3588, PARAFORMER_RKNN_3576_FULL,
    PARAFORMER_SHERPA_RPI4, PARAFORMER_SHERPA_RPI5, PARAFORMER_SHERPA_MAC,
    MATCHA_TRT, MATCHA_SHERPA_RPI5, MATCHA_RKNN_3576, MATCHA_RKNN_3588,
    MATCHA_SHARED,
    KOKORO_TRT_PERF, KOKORO_TRT_QUALITY, KOKORO_TRT_LONG, KOKORO_RKNN_3588,
    MOSS_TRT, MOSS_ORT,
]


# --- registry loading / new devices --------------------------------------

def test_new_leaves_all_load(registry):
    for lid in _ALL_NEW_LEAVES:
        assert lid in registry.leaves, lid


def test_new_devices_present(registry):
    for dev in ("rk3576", "rk3588", "rpi4", "rpi5", "mac-cpu", "agx-orin"):
        assert dev in registry.devices, dev
    assert registry.devices["rk3576"].device_class == "rk"
    assert registry.devices["rpi5"].device_class == "rpi"
    assert registry.devices["mac-cpu"].device_class == "mac"


def test_new_models_present(registry):
    for m in ("paraformer-streaming", "sensevoice", "matcha-icefall-zh-en",
              "kokoro-multi-lang-v1_0", "moss-tts-nano-v1", "qwen3-asr-rk"):
        assert m in registry.models, m


# --- resolve_pull: each new leaf resolves to a non-empty, expected file set ---

@pytest.mark.parametrize("leaf_id,expected_substr", [
    (PARAFORMER_TRT, "paraformer_encoder_dp4_400.plan"),
    (SENSEVOICE_TRT, "sense-voice-encoder.scaled.fixed.onnx"),
    (SENSEVOICE_RKNN_3588, "sense-voice-encoder.rk3588.fp16-scaled.t172.rknn"),
    (SENSEVOICE_SHERPA_RPI5, "sensevoice/model.int8.onnx"),
    (QWEN3_ASR_RK_3576_W8A8, "decoder_qwen3.w8a8.rk3576.rkllm"),
    (QWEN3_ASR_RK_3576_W4A16, "decoder_qwen3.w4a16_g128.rk3576.rkllm"),
    (QWEN3_ASR_RK_3588_W8A8, "qwen3_asr_encoder_merged.fp16.15s.rk3588.rknn"),
    (PARAFORMER_RKNN_3576, "encoder_prefix_to_block30.400.fp16.rknn"),
    (PARAFORMER_RKNN_3576_FULL, "rknn/rk3576/encoder.400.fp16.rknn"),
    (PARAFORMER_SHERPA_RPI5, "paraformer-streaming/encoder.onnx"),
    (MATCHA_TRT, "vocos_fp16.engine"),
    (MATCHA_RKNN_3588, "matcha-s64.rknn"),
    (KOKORO_TRT_PERF, "kokoro_prefix_encoder_dyn4_128_fp16.engine"),
    (KOKORO_RKNN_3588, "style.npy"),
    (MOSS_TRT, "moss_tts_prefill.plan"),
])
def test_new_leaf_resolves_pull(registry, leaf_id, expected_substr):
    files = lc.resolve_pull([leaf_id], registry)
    assert files, leaf_id
    assert any(expected_substr in f for f in files), (leaf_id, expected_substr)


def test_paraformer_pulls_shared_subleaf(registry):
    files = lc.resolve_pull([PARAFORMER_TRT], registry)
    assert "paraformer-streaming/encoder.onnx" in files
    assert "paraformer-streaming/tokens.txt" in files


def test_kokoro_rknn_voice_pack_present(registry):
    # gotcha guard: missing style.npy -> silent output, so both must be pulled.
    files = lc.resolve_pull([KOKORO_RKNN_3588], registry)
    assert "opt/kokoro-rknn/default.npy" in files
    assert "opt/kokoro-rknn/style.npy" in files


def test_paraformer_rknn_all_buckets_listed(registry):
    files = lc.resolve_pull([PARAFORMER_RKNN_3588], registry)
    for bucket in ("40", "80", "160", "240", "400"):
        needle = f"encoder_prefix_to_block30.{bucket}.fp16.rknn"
        assert any(needle in f for f in files), bucket
    assert any("decoder.400x40.fp16.rknn" in f for f in files)


def test_paraformer_rknn_full_buckets_and_no_hybrid_suffix(registry):
    # FULL encoder mode: full-encoder buckets {40,80,160,400} + RKNN decoder,
    # and NONE of the hybrid-only artifacts (prefix buckets / encoder suffix ONNX
    # / decoder-rknn.onnx).
    files = lc.resolve_pull([PARAFORMER_RKNN_3576_FULL], registry)
    for bucket in ("40", "80", "160", "400"):
        needle = f"rknn/rk3576/encoder.{bucket}.fp16.rknn"
        assert any(needle in f for f in files), bucket
    assert any("rknn/rk3576/decoder.400x40.fp16.rknn" in f for f in files)
    assert any("opt/asr/paraformer/tokens.txt" in f for f in files)
    # full mode must NOT carry the hybrid prefix/suffix artifacts
    assert not any("encoder_prefix_to_block30" in f for f in files)
    assert not any("encoder_suffix_from_block30.onnx" in f for f in files)
    assert not any("decoder-rknn.onnx" in f for f in files)


def test_paraformer_sherpa_pulls_shared_cdn_model(registry):
    # CPU sherpa reuses the SAME paraformer-streaming CDN model (encoder.onnx +
    # tokens.txt) via the asr.paraformer.shared sub-leaf; NO TRT .plan engines.
    for leaf_id in (PARAFORMER_SHERPA_RPI4, PARAFORMER_SHERPA_RPI5,
                    PARAFORMER_SHERPA_MAC):
        files = lc.resolve_pull([leaf_id], registry)
        assert files, leaf_id
        assert "paraformer-streaming/encoder.onnx" in files, leaf_id
        assert "paraformer-streaming/tokens.txt" in files, leaf_id
        assert not any(".plan" in f for f in files), leaf_id


# --- precision resolution for the new models ------------------------------

def test_sensevoice_precision_by_class(registry):
    assert registry.resolve_precision(
        registry.get_leaf(SENSEVOICE_RKNN_3588), "rk3588") == "fp16-scaled"
    assert registry.resolve_precision(
        registry.get_leaf(SENSEVOICE_SHERPA_RPI5), "rpi5") == "int8"
    assert registry.resolve_precision(
        registry.get_leaf(SENSEVOICE_TRT), "orin-nx") == "fp16"


def test_qwen3_rk_w4a16_explicit_precision_wins(registry):
    # The leaf pins w4a16_g128 explicitly (overrides the rk w8a8 default).
    assert registry.resolve_precision(
        registry.get_leaf(QWEN3_ASR_RK_3576_W4A16), "rk3576") == "w4a16_g128"
    assert registry.resolve_precision(
        registry.get_leaf(QWEN3_ASR_RK_3576_W8A8), "rk3576") == "w8a8"


# --- representative compositions validate ---------------------------------

def test_compose_paraformer_matcha_on_orin(registry):
    plan = lc.validate_composition("orin-nx", [PARAFORMER_TRT, MATCHA_TRT], registry)
    assert set(plan.leaf_ids) == {PARAFORMER_TRT, MATCHA_TRT}
    assert plan.peak_unified_mb <= plan.headroom_mb


def test_compose_sensevoice_matcha_sherpa_on_rpi5(registry):
    plan = lc.validate_composition(
        "rpi5", [SENSEVOICE_SHERPA_RPI5, MATCHA_SHERPA_RPI5], registry)
    assert set(plan.leaf_ids) == {SENSEVOICE_SHERPA_RPI5, MATCHA_SHERPA_RPI5}
    assert plan.peak_unified_mb <= plan.headroom_mb


def test_compose_paraformer_sherpa_matcha_sherpa_on_rpi5(registry):
    # The DEFAULT streaming-ASR + TTS pairing on RPi5: both cpu.* (same family),
    # both small CPU footprints → composes within the rpi5 budget.
    plan = lc.validate_composition(
        "rpi5", [PARAFORMER_SHERPA_RPI5, MATCHA_SHERPA_RPI5], registry)
    assert set(plan.leaf_ids) == {PARAFORMER_SHERPA_RPI5, MATCHA_SHERPA_RPI5}
    assert plan.peak_unified_mb <= plan.headroom_mb


def test_compose_paraformer_rknn_full_matcha_rknn_on_rk3576(registry):
    # FULL-encoder RKNN ASR + Matcha RKNN TTS: both rk.* (LANGUAGE_MODE=rk) and
    # both exclusive:npu → needs admission=serial (mirrors the hybrid case).
    plan = lc.validate_composition(
        "rk3576", [PARAFORMER_RKNN_3576_FULL, MATCHA_RKNN_3576], registry,
        admission="serial")
    assert set(plan.leaf_ids) == {PARAFORMER_RKNN_3576_FULL, MATCHA_RKNN_3576}
    assert plan.peak_unified_mb <= plan.headroom_mb


def test_compose_kokoro_perf_alone_on_orin(registry):
    plan = lc.validate_composition("orin-nx", [KOKORO_TRT_PERF], registry)
    assert plan.peak_unified_mb <= plan.headroom_mb


# --- new validator check: backend-family coupling -------------------------

def test_reject_jetson_x_rk_family_mix(registry):
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("orin-nx", [PARAFORMER_TRT, MATCHA_RKNN_3588], registry)
    assert "backend families" in str(exc.value)


def test_reject_rk_x_jetson_via_language_mode(registry):
    # qwen3 RK ASR (LANGUAGE_MODE=rk) paired with a jetson Matcha TTS leaf.
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("rk3588", [QWEN3_ASR_RK_3588_W8A8, MATCHA_TRT], registry)
    assert "backend families" in str(exc.value)


def test_same_family_rk_ok(registry):
    # Two rk.* leaves (same family) compose fine when admission=serial handles NPU.
    plan = lc.validate_composition(
        "rk3588", [QWEN3_ASR_RK_3588_W8A8, MATCHA_RKNN_3588], registry,
        admission="serial")
    assert set(plan.leaf_ids) == {QWEN3_ASR_RK_3588_W8A8, MATCHA_RKNN_3588}


def test_existing_qwen3_jetson_pair_unaffected_by_family_check(registry):
    # Both are jetson.trt_edge_llm -> same family -> no rejection (no-op guard).
    plan = lc.validate_composition("orin-nx", [ASR_N2, TTS_N1], registry)
    assert set(plan.leaf_ids) == {ASR_N2, TTS_N1}


# --- new validator check: NPU exclusivity ---------------------------------

def test_reject_double_npu_without_serial(registry):
    # qwen3 RK ASR + matcha RKNN TTS both declare resources.exclusive: npu.
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("rk3588", [QWEN3_ASR_RK_3588_W8A8, MATCHA_RKNN_3588], registry)
    assert "exclusive-resource" in str(exc.value)
    assert "npu" in str(exc.value)


def test_double_npu_ok_with_serial(registry):
    plan = lc.validate_composition(
        "rk3588", [QWEN3_ASR_RK_3588_W8A8, MATCHA_RKNN_3588], registry,
        admission="serial")
    assert plan.peak_unified_mb <= plan.headroom_mb


def test_single_npu_leaf_no_exclusivity_error(registry):
    # One NPU leaf alone never trips the exclusivity check.
    plan = lc.validate_composition("rk3576", [SENSEVOICE_RKNN_3576], registry)
    assert set(plan.leaf_ids) == {SENSEVOICE_RKNN_3576}


# ---------------------------------------------------------------------------
# v0.9.0 leaves (edgellm-v090 engine set). The v080 leaves above are kept
# untouched for rollback — these cases only cover the NEW *-v090.yaml files.
# ---------------------------------------------------------------------------

ASR_V090_N1 = "asr.qwen3_asr_v090.orin-nx.n1"
ASR_V090_N2 = "asr.qwen3_asr_v090.orin-nx.n2"
TTS_V090_N1 = "tts.qwen3_tts_v090.orin-nx.n1"
TTS_V090_N2 = "tts.qwen3_tts_v090.orin-nx.n2"
TTS_V090_SHARED = "tts.qwen3_tts_v090.shared"
TTS_BASE_V090_N1 = "tts.qwen3_tts_base_v090.orin-nx.n1"
TTS_BASE_V090_N2 = "tts.qwen3_tts_base_v090.orin-nx.n2"
TTS_BASE_V090_FILES = "tts.qwen3_tts_base_v090.model_files"

_V090_PLUGIN = "/opt/edgellm-v090/libNvInfer_edgellm_plugin.so"


def test_registry_loads_v090_leaves(registry):
    for lid in (ASR_V090_N1, ASR_V090_N2, TTS_V090_N1, TTS_V090_N2,
                TTS_V090_SHARED, TTS_BASE_V090_N1, TTS_BASE_V090_N2,
                TTS_BASE_V090_FILES):
        assert lid in registry.leaves, lid


def test_v090_asr_env_points_at_highperf_v090(registry):
    env = lc.resolve_env(
        [ASR_V090_N2], registry,
        overrides={"QWEN3_ARTIFACT_ROOT": "/opt/models/qwen3-edgellm"},
    )
    assert env["EDGE_LLM_ASR_ENGINE_DIR"] == (
        "/opt/models/qwen3-edgellm/engines/orin-nx/highperf-v090/"
        "asr_thinker_full_int4_b2"
    )
    assert env["EDGE_LLM_ASR_AUDIO_ENC_DIR"] == (
        "/opt/models/qwen3-edgellm/engines/orin-nx/highperf-v090/"
        "asr_audio_encoder"
    )
    # v0.9.0: plugin path is required and must be absolute (cwd-resolution fix).
    assert env["EDGELLM_PLUGIN_PATH"] == _V090_PLUGIN
    # v0.9.0 WAV-ingest mode: the leaf sets EDGELLM_REQUEST_AUDIO_WAV=1 (which
    # the voxedge ASR preload guard reads to skip the mel-asset check), and the
    # host-side mel front-end env keys are retired on the v090 path.
    assert env["EDGELLM_REQUEST_AUDIO_WAV"] == "1"
    assert "EDGE_LLM_ASR_MEL_SETTINGS" not in env
    assert "EDGE_LLM_ASR_MEL_FILTERS" not in env


def test_v090_asr_precision_pinned_int4(registry):
    # models.yaml qwen3-asr jetson default stays fp16 (v080 leaves); the v090
    # thinker is the int4 build so the leaf pins precision explicitly.
    leaf = registry.get_leaf(ASR_V090_N2)
    assert leaf.precision == "int4"
    assert registry.resolve_precision(leaf, "orin-nx") == "int4"


def test_v090_tts_single_b2_talker_serves_n1_and_n2(registry):
    # v0.9.0: ONE talker engine (b2 batch lane) serves both concurrency leaves
    # (batched decode inside one runtime) → identical pull sets, like ASR.
    assert (lc.resolve_pull([TTS_V090_N1], registry)
            == lc.resolve_pull([TTS_V090_N2], registry))
    files = lc.resolve_pull([TTS_V090_N1], registry)
    assert "engines/orin-nx/highperf-v090/tts_talker_int4_b2" in files


def test_v090_tts_env_lean_nonstateful_worker(registry):
    env = lc.resolve_env(
        [TTS_V090_N2], registry,
        overrides={"QWEN3_ARTIFACT_ROOT": "/opt/models/qwen3-edgellm"},
    )
    # Lean non-stateful code2wav path (v080 stateful surgery retired).
    assert env["EDGE_LLM_TTS_STATEFUL_CODE2WAV"] == "0"
    # v0.9.0 lean worker is streaming-native (no output_file mode) → the leaf
    # must also flag STREAMING_ONLY so non-streaming /tts aggregates chunks,
    # matching the profile. (Without it, composition-mode /tts hits KeyError.)
    assert env["EDGE_LLM_TTS_STREAMING_ONLY"] == "1"
    assert env["EDGE_LLM_TTS_CODE2WAV_DIR"] == (
        "/opt/models/qwen3-edgellm/engines/orin-nx/highperf-v090/"
        "tts_code2wav_lean"
    )
    assert env["EDGE_LLM_TTS_TALKER_DIR"].endswith("tts_talker_int4_b2")
    assert env["EDGELLM_PLUGIN_PATH"] == _V090_PLUGIN
    assert env["OVS_TTS_WORKER_CONCURRENCY"] == "2"
    # The v0.7-style explicit_kv path is gone on v090 leaves.
    assert "EDGE_LLM_TTS_TALKER_ENGINE" not in env
    assert "EDGE_LLM_TTS_TALKER_BACKEND" not in env


def test_v090_base_pull_includes_engines_and_model_files(registry):
    files = lc.resolve_pull([TTS_BASE_V090_N1], registry)
    # Engines from the unified highperf-v090 tree (incl. NEW spk encoder)...
    for f in (
        "engines/orin-nx/highperf-v090/tts_base_talker",
        "engines/orin-nx/highperf-v090/tts_base_code_predictor",
        "engines/orin-nx/highperf-v090/tts_base_code2wav_lean",
        "engines/orin-nx/highperf-v090/tts_base_spk_encoder",
    ):
        assert f in files, f
    # ...plus the version-independent model-side files via the sub-leaf.
    assert "models/qwen3-tts-base/tokenizer/tokenizer.json" in files
    assert "models/qwen3-tts-base/ref_embedding.b64.txt" in files
    env = lc.resolve_env([TTS_BASE_V090_N1], registry)
    assert env["EDGE_LLM_TTS_STATEFUL_CODE2WAV"] == "0"
    assert env["EDGELLM_PLUGIN_PATH"] == _V090_PLUGIN
    assert env["EDGE_LLM_TTS_SPK_ENCODER_DIR"].endswith("tts_base_spk_encoder")


def test_v090_compositions_validate_on_nx(registry):
    for tts in (TTS_V090_N2, TTS_BASE_V090_N2):
        plan = lc.validate_composition("orin-nx", [ASR_V090_N2, tts], registry)
        assert set(plan.leaf_ids) == {ASR_V090_N2, tts}
        assert plan.peak_unified_mb <= plan.headroom_mb
