"""Strict v0.11 TTS worker provenance semantics."""

import copy
import hashlib
import json
from types import SimpleNamespace


_PAYLOAD = (
    "worker",
    "plugin.so",
    "talker/config.json",
    "talker/llm.engine",
    "cp/llm.engine",
    "cp/config.json",
    "cp/codec_embeddings.safetensors",
    "cp/lm_heads.safetensors",
    "tok/tokenizer.json",
    "c2w/config.json",
    "c2w/code2wav.engine",
    "clone/speaker_encoder.engine",
    "clone/speech_tokenizer_encoder.engine",
    "checkpoint/model.safetensors",
)


def _fixture(tmp_path):
    from server.core.qwen3_artifact_downloader import _strict_cache_model, _strict_cache_repo

    model = "qwen3-tts-0.6b-base"
    repo = "org/base"
    revision = "a" * 40
    root = tmp_path / "runtime"
    cache = tmp_path / "cache" / _strict_cache_repo(repo) / _strict_cache_model(model) / revision
    files = {}
    for rel in _PAYLOAD:
        content = rel.encode("utf-8")
        for base in (root, cache):
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        files[rel] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
    manifest = {
        "model_id": model,
        "_source": {
            "repo": repo,
            "revision": revision,
            "model_id": model,
            "canonical_model_id": model,
        },
        "files": files,
        "provenance": {
            "source_sdk_version_evidence": "0.11",
            "worker_build_definitions": {
                "EDGELLM_QWEN3_TTS_V011": 1,
            },
        },
    }
    env = {}
    config = SimpleNamespace(
        reference_artifact_verified=False,
        reference_tmp_dir=str(root / "ref-tmp"),
        worker_binary=str(root / "worker"),
        plugin_path=str(root / "plugin.so"),
        talker_dir=str(root / "talker"),
        code_predictor_dir=str(root / "cp"),
        tokenizer_dir=str(root / "tok"),
        code2wav_dir=str(root / "c2w"),
        clone_encoder_dir=str(root / "clone"),
        checkpoint_dir=str(root / "checkpoint"),
    )
    (root / "ref-tmp").mkdir()
    profile = {
        "model_artifacts": [{
            "model_id": model,
            "canonical_model_id": model,
            "repo": repo,
            "revision": revision,
            "root": str(root),
            "cache_root": str(tmp_path / "cache"),
            "files": list(_PAYLOAD),
            "strict": True,
        }],
    }
    manifest_path = cache / "manifest.json"

    def write_manifest(value):
        manifest_path.write_text(json.dumps(value), encoding="utf-8")

    return config, profile, env, manifest, write_manifest, files["worker"]["sha256"]


def test_v011_worker_binary_hash_is_separate_from_source_hash(tmp_path):
    from server.core.voxedge_backend_config import _verify_base_reference_artifacts

    config, profile, env, manifest, write_manifest, worker_sha = _fixture(tmp_path)
    source_sha = "9" * 64
    defs = manifest["provenance"]["worker_build_definitions"]

    # Current manifests carry independent C++ source and installed-binary
    # provenance.  The real verifier must accept the differing hashes.
    explicit = copy.deepcopy(manifest)
    explicit["provenance"]["worker_build_definitions"] = {
        **defs,
        "native_worker_source_sha256": source_sha,
        "native_worker_binary_sha256": worker_sha,
    }
    write_manifest(explicit)
    assert _verify_base_reference_artifacts(profile, config, env) is True

    # An explicit bad/empty/wrong-type binary lock is fail-closed and cannot
    # fall back to a matching legacy field.
    for bad_binary in ("b" * 64, "", [], None):
        bad = copy.deepcopy(explicit)
        bad["provenance"]["worker_build_definitions"]["native_worker_binary_sha256"] = bad_binary
        write_manifest(bad)
        assert _verify_base_reference_artifacts(profile, config, env) is False

    # Old manifests without the explicit key retain compatibility: the old
    # source-named field was the worker binary lock in that format.
    legacy = copy.deepcopy(manifest)
    legacy["provenance"]["worker_build_definitions"]["native_worker_source_sha256"] = worker_sha
    write_manifest(legacy)
    assert _verify_base_reference_artifacts(profile, config, env) is True

    missing = copy.deepcopy(manifest)
    write_manifest(missing)
    assert _verify_base_reference_artifacts(profile, config, env) is False
