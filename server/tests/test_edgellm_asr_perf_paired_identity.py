"""Paired/variant identity contract tests for the ASR WS perf gate.

Drives the ACTUAL functions of bench/perf/edgellm_asr_ws_perf_gate.py
(paired_threshold_document, compare_runs, load_identity_file, qualification)
against in-memory fixture documents. All SHA pins used here are the
HISTORICAL accepted pins used as test fixtures ONLY — they are NOT current
app-loaded proof and never flip the live physical gate.

Finite CPU only: no network, no suppressors, no device.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.perf.edgellm_asr_ws_perf_gate import (
    IdentityError,
    IDENTITY_STRICT_SHARED_FIELDS,
    IDENTITY_VARIANT_FIELDS,
    compare_runs,
    load_identity_file,
    paired_threshold_document,
    qualification,
)

# Historical accepted pins — test fixtures ONLY, not app-loaded proof.
SHARED_BASE_PROFILE_SHA256 = (
    "14ea5e0080f25bbcc85ef8f403f7adb38d557f6f5306cd08b5c90ed20e478877"
)
B1_PROFILE_SHA256 = (
    "389d32a310694264c7bdab43b9ae6924d291cd8ba82783e95878918f44687672"
)
B2_PROFILE_SHA256 = (
    "33d5524df15fe8e0c9a05c1feff875ed4ea56f2b7d6f77ae5f99c7dc6b3a827b"
)
B1_ENGINE_SHA256 = (
    "e48fa187e4620128f626ba76eaa58e6aedd41180687a4895951f20b4994ccd43"
)
B2_ENGINE_SHA256 = (
    "f90f2db141d65976e43bd5fde93302b2d266f369dc47bd8cc9a2ae894594062c"
)

MANIFEST_SHA256 = "4a20ee1cecb1da7264cb8d5cd59419de656dc02ff259eceab1325296905984a6"
ORDERED_HASHES = [f"{i:064x}" for i in range(100)]


def _identity(slot_variant: str) -> dict[str, str]:
    """Variant identity: shared contract identical, variant fields distinct."""
    b2 = slot_variant == "candidate"
    return {
        "target_device": "spark",
        "sdk_version": "2.3.0",
        "upstream_commit": "a" * 40,
        "worker_sha256": "w" * 64,
        "plugin_sha256": "p" * 64,
        "profile_family": "qwen3-asr",
        "base_profile_sha256": SHARED_BASE_PROFILE_SHA256,
        "profile_sha256": B2_PROFILE_SHA256 if b2 else B1_PROFILE_SHA256,
        "engine_sha256": B2_ENGINE_SHA256 if b2 else B1_ENGINE_SHA256,
        "config_sha256": "c" * 64 if b2 else "b" * 64,
        "slot_variant": slot_variant,
    }


def _phase_metrics(phase: str) -> dict:
    if phase == "b1":
        return {
            "rows_ok": 100,
            "request_wall_s": {"p50": 1.0, "p95": 1.5, "count": 100},
            "rtf": {"p95": 0.4, "count": 100},
            "throughput_audio_s_per_s": 1.0,
            "realized_width": 1,
            "requested_width": 1,
        }
    return {
        "rows_ok": 100,
        "request_wall_s": {"p50": 1.2, "p95": 1.8, "count": 100},
        "rtf": {"p95": 0.5, "count": 100},
        "aggregate_throughput_audio_s_per_s": 1.4,
        "aggregate_valid": True,
        "realized_width": 2,
        "requested_width": 2,
    }


def _run_doc(phase: str) -> dict:
    """A run document shaped like the driver's ACTUAL output schema.

    ``mode`` is a real ``--mode`` choice (``ws``/``http``, never the
    unsupported ``chunked``), ``pace`` is the actual boolean parser flag,
    ``chunk_ms`` is a positive int, and ``service_identity`` carries the
    actual nonempty capabilities model/backend provenance.
    """
    conc = 1 if phase == "b1" else 2
    return {
        "schema_version": 3,
        "mode": "ws",
        "measurement_basis": "app",
        "concurrency": conc,
        "phases_executed": [phase],
        "config": {"chunk_ms": 100, "pace": True},
        "service_identity": {
            "asr_model_id": "qwen3-asr-0.6b",
            "asr_backend": "rkllm",
            "identity_present": True,
        },
        "corpus": {
            "manifest_sha256": MANIFEST_SHA256,
            "items": [{"sha256": h} for h in ORDERED_HASHES],
        },
        "metrics": {
            phase: _phase_metrics(phase),
            "quality": {"wer": 0.03, "ref_words": 787},
        },
        "artifact_identity": {
            "status": "PINNED",
            "identity": _identity("baseline" if phase == "b1" else "candidate"),
            "live_artifact_verified": False,
            "live_artifact_status": "UNPROVEN",
        },
    }


# ── Service identity / protocol fields missing on BOTH sides ─────────


def test_missing_service_identity_on_both_sides_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        del doc["service_identity"]
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    joined = " ".join(result["problems"])
    assert "asr_model_id missing/unresolved" in joined
    assert "asr_backend missing/unresolved" in joined
    cmp_doc = compare_runs(b1, b2)
    assert cmp_doc["guard_status"] == "FAIL"


def test_wrong_model_on_one_side_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    b2["service_identity"]["asr_model_id"] = "whisper-tiny"
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("asr_model_id mismatch" in p for p in result["problems"])
    cmp_doc = compare_runs(b1, b2)
    assert any("asr_model_id mismatch" in p for p in cmp_doc["problems"])


def test_wrong_backend_on_one_side_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    b1["service_identity"]["asr_backend"] = "sherpa-onnx"
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("asr_backend mismatch" in p for p in result["problems"])


def test_missing_mode_on_both_sides_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        del doc["mode"]
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("not an actual supported driver mode" in p for p in result["problems"])


def test_unsupported_chunked_mode_on_both_sides_rejected():
    # 'chunked' is NOT an actual --mode choice (ws|http); both sides agreeing
    # on it is still not protocol identity.
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        doc["mode"] = "chunked"
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("chunked" in p for p in result["problems"])


def test_missing_manifest_on_both_sides_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        doc["corpus"]["manifest_sha256"] = ""
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("manifest sha256 missing/unresolved" in p for p in result["problems"])


def test_nonfrozen_manifest_on_both_sides_rejected_in_paired_path():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        doc["corpus"]["manifest_sha256"] = "e" * 64
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("frozen English100" in p for p in result["problems"])


def test_empty_corpus_on_both_sides_rejected():
    # [] == [] is NOT a shared corpus.
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        doc["corpus"]["items"] = []
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    joined = " ".join(result["problems"])
    assert "ordered corpus hashes missing/empty" in joined
    assert "corpus item count 0 != 100" in joined


def test_reordered_corpus_on_one_side_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    b2["corpus"]["items"] = list(reversed(b2["corpus"]["items"]))
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("ordered corpus hashes differ" in p for p in result["problems"])


def test_missing_pace_on_both_sides_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        del doc["config"]["pace"]
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("config.pace missing/invalid" in p for p in result["problems"])


def test_missing_chunk_ms_on_both_sides_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        del doc["config"]["chunk_ms"]
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("config.chunk_ms missing/invalid" in p for p in result["problems"])


def test_bool_chunk_ms_rejected():
    # True == 1 must not stand in for chunk_ms 1.
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        doc["config"]["chunk_ms"] = True
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("chunk_ms missing/invalid" in p for p in result["problems"])


# ── Compare guard: native baseline never allowed ─────────────────────


def test_compare_runs_rejects_native_baseline():
    b1 = _run_doc("b1")
    base = _run_doc("b2")
    base["measurement_basis"] = "native"
    cmp_doc = compare_runs(b1, base)
    assert cmp_doc["guard_status"] == "FAIL"
    assert any("native baselines are rejected" in p for p in cmp_doc["problems"])


# ── Typed phase provenance ───────────────────────────────────────────


def test_bool_width_provenance_rejected():
    # realized_width=True must not stand in for width 1 (bool is not int).
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    b1["metrics"]["b1"]["realized_width"] = True
    b1["metrics"]["b1"]["requested_width"] = True
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("width provenance" in p for p in result["problems"])


def test_bool_concurrency_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    b1["concurrency"] = True
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("requested concurrency" in p for p in result["problems"])


# ── Permitted variant differences ────────────────────────────────────


def test_valid_shared_base_with_distinct_variants_permits_pairing():
    result = paired_threshold_document(_run_doc("b1"), _run_doc("b2"))
    assert result["status"] == "PASS", result
    vp = result["variant_provenance"]
    for name in IDENTITY_VARIANT_FIELDS:
        assert vp[name]["b1"] != vp[name]["b2"]
    assert vp["profile_sha256"]["b1"] == B1_PROFILE_SHA256
    assert vp["profile_sha256"]["b2"] == B2_PROFILE_SHA256
    # Actual phase provenance recorded, not invented.
    assert result["phase_provenance"]["b1"]["realized_width"] == 1
    assert result["phase_provenance"]["b2"]["realized_width"] == 2


def test_compare_runs_permits_variant_differences_and_records_both_sides():
    cmp_doc = compare_runs(_run_doc("b1"), _run_doc("b2"))
    assert cmp_doc["problems"] == []
    assert cmp_doc["identical_protocol"] is True
    assert cmp_doc["guard_status"] != "FAIL"
    assert cmp_doc["permitted_differences"]["profile_sha256"]["app"] == (
        B1_PROFILE_SHA256
    )
    assert cmp_doc["permitted_differences"]["profile_sha256"]["baseline"] == (
        B2_PROFILE_SHA256
    )
    assert cmp_doc["permitted_differences"]["engine_sha256"]["baseline"] == (
        B2_ENGINE_SHA256
    )
    assert cmp_doc["variant_provenance"]["slot_variant"] == {
        "app": "baseline",
        "baseline": "candidate",
    }


# ── Shared contract violations ───────────────────────────────────────


@pytest.mark.parametrize("field", IDENTITY_STRICT_SHARED_FIELDS)
def test_wrong_shared_field_on_one_side_rejects(field):
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    b2["artifact_identity"]["identity"][field] = "0" * 64
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any(field in p for p in result["problems"])
    cmp_doc = compare_runs(b1, b2)
    assert cmp_doc["guard_status"] == "FAIL"


def test_missing_strict_field_on_both_sides_is_not_none_none_match():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        del doc["artifact_identity"]["identity"]["base_profile_sha256"]
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any(
        "base_profile_sha256 missing/unresolved" in p for p in result["problems"]
    )
    cmp_doc = compare_runs(b1, b2)
    assert cmp_doc["guard_status"] == "FAIL"


def test_absent_artifact_identity_is_unproven_never_pass():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    for doc in (b1, b2):
        doc["artifact_identity"] = {
            "status": "UNPROVEN",
            "live_artifact_status": "UNPROVEN",
        }
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"


def test_missing_variant_provenance_is_unproven():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    del b2["artifact_identity"]["identity"]["engine_sha256"]
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("engine_sha256 missing/unresolved" in p for p in result["problems"])


# ── Protocol / corpus / config equality still enforced ───────────────


def test_ordered_corpus_drift_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    b2["corpus"]["items"][7]["sha256"] = "f" * 64
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("ordered corpus hashes differ" in p for p in result["problems"])


def test_pace_chunk_mode_mismatches_rejected():
    b1 = _run_doc("b1")
    b2 = _run_doc("b2")
    b2["config"]["pace"] = "as-fast-as-possible"
    b2["config"]["chunk_ms"] = 200
    b2["mode"] = "whole"
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    joined = " ".join(result["problems"])
    assert "pace" in joined and "chunk_ms" in joined and "mode" in joined


# ── Actual phase provenance ──────────────────────────────────────────


def test_swapped_phase_provenance_rejected():
    b1 = _run_doc("b1")
    b2 = _run_doc("b2")
    # B2 document that actually holds a second copy of the one-lane B1 phase.
    swapped = _run_doc("b1")
    swapped["artifact_identity"] = b2["artifact_identity"]
    result = paired_threshold_document(b1, swapped)
    assert result["status"] == "UNPROVEN"
    joined = " ".join(result["problems"])
    assert "does not prove phase b2" in joined
    assert "requested concurrency 1" in joined and "(expected 2)" in joined


def test_missing_phase_width_provenance_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    del b2["metrics"]["b2"]["realized_width"]
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("width provenance" in p for p in result["problems"])


def test_missing_phases_executed_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    del b1["phases_executed"]
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"


def test_synthetic_one_lane_b2_rejected():
    b1, b2 = _run_doc("b1"), _run_doc("b2")
    b2["metrics"]["b2"]["realized_width"] = 1
    result = paired_threshold_document(b1, b2)
    assert result["status"] == "UNPROVEN"
    assert any("concurrency 2" in p for p in result["problems"])


# ── Physical gate trust boundary ─────────────────────────────────────


def test_paired_pass_never_flips_live_artifact_physical_gate():
    result = paired_threshold_document(_run_doc("b1"), _run_doc("b2"))
    assert result["status"] == "PASS"
    # Structural assertion: the actual result dict carries no
    # live_artifact_verified proof key. The explanatory physical_proof_note
    # names the literal field for prose only; that mention must never be
    # mistaken for an actual proof value on a paired PASS.
    assert "live_artifact_verified" not in result
    assert "physical_proof_note" in result
    assert "never becomes physical verification" in result["physical_proof_note"]


def test_qualification_stays_unproven_without_physical_proof():
    outcome = qualification(
        b1=[], b2=[], corpus=_fixture_corpus(),
        identity_present=True,
        artifact_identity_present=True,
        width_ok=True,
        readiness_ok=True,
        runtime_ifb_ok=True,
        live_artifact_verified=False,  # self-attested JSON equal is NOT proof
        resources_proven=True,
        controls_complete=True,
        aggregate_valid=True,
        thresholds_status="PASS",
    )
    assert outcome["status"] == "UNPROVEN"
    assert any("live artifact identity UNPROVEN" in r for r in outcome["reasons"])


def _fixture_corpus():
    """Minimal frozen-shape corpus object for qualification() only."""
    from types import SimpleNamespace

    # EXACT frozen shape: 100 items / 787 reference words, duration 1.0.
    # One 589-word item + 99 two-word items = 589 + 198 = 787 words, so the
    # fixture cannot trigger an unrelated reference-word-count downgrade.
    long_item = SimpleNamespace(transcript="word " * 588 + "word", duration_s=1.0)
    short_item = SimpleNamespace(transcript="hello world", duration_s=1.0)
    return SimpleNamespace(items=[long_item] + [short_item] * 99)


# ── Identity file load validation ────────────────────────────────────


def test_load_identity_file_requires_base_profile_sha256(tmp_path: Path):
    payload = _identity("baseline")
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_identity_file(path, load_identity_file_sha(path))
    assert loaded["identity"]["base_profile_sha256"] == SHARED_BASE_PROFILE_SHA256
    assert loaded["identity"]["profile_sha256"] == B1_PROFILE_SHA256

    del payload["base_profile_sha256"]
    bad = tmp_path / "identity-missing-base.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    sha = load_identity_file_sha(bad)
    with pytest.raises(IdentityError, match="base_profile_sha256"):
        load_identity_file(bad, sha)


def load_identity_file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
