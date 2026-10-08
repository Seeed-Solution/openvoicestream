"""Parity tests: OVS builder wrappers vs direct voxedge factory functions.

Verifies that ``server.core.voxedge_backend_config.build_trt_edge_llm_tts_config``
and ``build_trt_edge_llm_asr_config`` (which now delegate to the canonical voxedge
factories) produce config objects with identical fields when called with the same
env dict as the factories directly.
"""

import pytest
import hashlib
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _all_fields(cfg) -> dict:
    """Return all dataclass fields as a dict (comparable across instances)."""
    import dataclasses
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


# ---------------------------------------------------------------------------
# TTS parity
# ---------------------------------------------------------------------------

class TestTTSBuilderParity:
    BASE_ENV = {
        # paths
        "EDGE_LLM_TTS_BIN": "/opt/edge/tts_inference",
        "EDGE_LLM_TTS_WORKER_BIN": "/opt/edge/tts_worker",
        "EDGELLM_PLUGIN_PATH": "/opt/edge/plugin.so",
        "EDGE_LLM_TTS_TALKER_DIR": "/opt/models/talker",
        "EDGE_LLM_TTS_CP_DIR": "/opt/models/cp",
        "EDGE_LLM_TTS_TOKENIZER_DIR": "/opt/models/tok",
        "EDGE_LLM_TTS_CODE2WAV_DIR": "/opt/models/c2w",
        "QWEN3_SPEAKER_ENCODER": "/opt/models/speaker.onnx",
        # identity
        "OVS_TTS_MODEL_ID": "my_tts",
        "OVS_TTS_BACKEND": "edgellm_worker",
        # concurrency
        "OVS_TTS_WORKER_CONCURRENCY": "2",
        # runtime
        "EDGE_LLM_QWEN3_PROFILE": "highperf",
        "EDGE_LLM_TTS_PERF_PROFILE": "quality",
        "EDGE_LLM_TTS_STATEFUL_CODE2WAV": "1",
        # sampling
        "OVS_TTS_SEED": "99",
        "OVS_TTS_TALKER_TEMPERATURE": "0.7",
        "OVS_TTS_TALKER_TOP_K": "30",
        "OVS_TTS_TOP_P": "0.95",
        "OVS_TTS_PREDICTOR_TEMPERATURE": "0.8",
        "OVS_TTS_PREDICTOR_TOP_K": "25",
        "OVS_TTS_PREDICTOR_TOP_P": "0.9",
        "TTS_MAX_AUDIO_LENGTH": "800",
        "TTS_MIN_AUDIO_LENGTH": "20",
        "TTS_REPETITION_PENALTY": "1.02",
        "TTS_CODEC_EOS_LOGIT_OFFSET": "0.5",
        # segmentation
        "EDGE_LLM_TTS_SEGMENT_TEXT": "1",
        "EDGE_LLM_TTS_SEGMENT_MAX_CHARS": "100",
        "EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS": "40",
        "EDGE_LLM_TTS_SEGMENT_PAUSE_MS": "70",
        "EDGE_LLM_TTS_HARD_SEGMENT_PAUSE_MS": "110",
        # streaming
        "EDGE_LLM_TTS_STREAMING_PROFILE": "continuous_playback",
        "EDGE_LLM_TTS_FIRST_CHUNK_FRAMES": "32",
        "EDGE_LLM_TTS_CHUNK_FRAMES": "64",
    }

    def test_parity_with_direct_factory(self):
        from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config
        from voxedge.backends.jetson.trt_edge_llm_tts import build_config_from_env

        env = dict(self.BASE_ENV)
        ovs_cfg = build_trt_edge_llm_tts_config(profile=None, env=env)
        vox_cfg = build_config_from_env(env=env)

        ovs_fields = _all_fields(ovs_cfg)
        vox_fields = _all_fields(vox_cfg)

        # extra_worker_env and artifact_ref can differ; compare all other fields.
        skip = {"extra_worker_env", "artifact_ref"}
        for k in ovs_fields:
            if k in skip:
                continue
            assert ovs_fields[k] == vox_fields[k], (
                f"Field '{k}' differs: OVS={ovs_fields[k]!r}, voxedge={vox_fields[k]!r}"
            )

    def test_parity_minimal_env(self):
        """With minimal env, both builders return identical config."""
        from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config
        from voxedge.backends.jetson.trt_edge_llm_tts import build_config_from_env

        env = {}
        ovs_cfg = build_trt_edge_llm_tts_config(profile=None, env=env)
        vox_cfg = build_config_from_env(env=env)

        skip = {"extra_worker_env", "artifact_ref"}
        for k in _all_fields(ovs_cfg):
            if k in skip:
                continue
            assert _all_fields(ovs_cfg)[k] == _all_fields(vox_cfg)[k], (
                f"Field '{k}' differs for minimal env"
            )

    def test_profile_worker_concurrency_injection(self):
        """Profile tts_worker_concurrency is injected when OVS_TTS_WORKER_CONCURRENCY absent."""
        from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config

        profile = {"tts_worker_concurrency": 3}
        env = {}  # no OVS_TTS_WORKER_CONCURRENCY
        cfg = build_trt_edge_llm_tts_config(profile=profile, env=env)
        assert cfg.worker_concurrency == 3

    def test_profile_concurrency_overridden_by_env(self):
        """Explicit env takes priority over profile concurrency."""
        from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config

        profile = {"tts_worker_concurrency": 3}
        env = {"OVS_TTS_WORKER_CONCURRENCY": "5"}
        cfg = build_trt_edge_llm_tts_config(profile=profile, env=env)
        assert cfg.worker_concurrency == 5

    def test_profile_env_maps_optional_base_clone_paths(self):
        """Profile env values reach the canonical voxedge config unchanged."""
        from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config

        cfg = build_trt_edge_llm_tts_config(profile={
            "env": {
                "EDGE_LLM_TTS_CLONE_ENCODER_DIR": "/profile/clone",
                "EDGE_LLM_TTS_CHECKPOINT_DIR": "/profile/checkpoint",
            }
        }, env={})
        assert cfg.clone_encoder_dir == "/profile/clone"
        assert cfg.checkpoint_dir == "/profile/checkpoint"

    def test_strict_base_reference_proof_reads_locked_cache(self, tmp_path):
        from server.core.qwen3_artifact_downloader import _strict_cache_model, _strict_cache_repo
        from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config

        root = tmp_path / "runtime"
        for rel in ("talker", "cp", "tok", "c2w", "clone", "checkpoint", "ref-tmp"):
            (root / rel).mkdir(parents=True)
        files = {}
        for rel in ("worker", "plugin.so", "talker/config.json", "talker/llm.engine", "cp/llm.engine", "cp/config.json", "cp/codec_embeddings.safetensors", "cp/lm_heads.safetensors", "tok/tokenizer.json", "c2w/config.json", "c2w/code2wav.engine", "c2w/code2wav_stateful.engine", "clone/speaker_encoder.engine", "clone/speech_tokenizer_encoder.engine", "checkpoint/model.safetensors"):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(rel.encode())
            files[rel] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
        repo, model, revision = "org/base", "qwen3-tts-0.6b-base", "a" * 40
        cache = tmp_path / "cache" / _strict_cache_repo(repo) / _strict_cache_model(model) / revision
        for rel in files:
            target = cache / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / rel).read_bytes())
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
                    "native_worker_source_sha256": files["worker"]["sha256"],
                },
            },
        }
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        env = {
            "OVS_TTS_MODEL_ID": model,
            "EDGE_LLM_TTS_BIN": str(root / "worker"),
            "EDGE_LLM_TTS_WORKER_BIN": str(root / "worker"),
            "EDGELLM_PLUGIN_PATH": str(root / "plugin.so"),
            "EDGE_LLM_TTS_TALKER_DIR": str(root / "talker"),
            "EDGE_LLM_TTS_CP_DIR": str(root / "cp"),
            "EDGE_LLM_TTS_TOKENIZER_DIR": str(root / "tok"),
            "EDGE_LLM_TTS_CODE2WAV_DIR": str(root / "c2w"),
            "EDGE_LLM_TTS_CLONE_ENCODER_DIR": str(root / "clone"),
            "EDGE_LLM_TTS_CHECKPOINT_DIR": str(root / "checkpoint"),
            "EDGE_LLM_TTS_REFERENCE_TMP_DIR": str(root / "ref-tmp"),
            "EDGE_LLM_TTS_TEXT_PROJECTION": "host_fp32",
            "EDGE_LLM_TTS_PROMPT_KV_CACHE": "0",
        }
        profile = {"model_artifacts": [{
            "model_id": model, "canonical_model_id": model,
            "repo": repo, "revision": revision,
            "root": str(root), "cache_root": str(tmp_path / "cache"),
            "files": list(files), "strict": True,
        }]}
        cfg = build_trt_edge_llm_tts_config(profile=profile, env=env)
        assert cfg.reference_artifact_verified is True

        # A provisioned entry carries the remote, repo-relative manifest path
        # the downloader fetched from. The downloader persists the manifest at
        # <cache>/manifest.json, so the proof must still read the local copy.
        prefixed_profile = {"model_artifacts": [{
            **profile["model_artifacts"][0],
            "manifest": f"models/{model}/manifest.json",
        }]}
        assert not (cache / "models").exists()
        prefixed_cfg = build_trt_edge_llm_tts_config(profile=prefixed_profile, env=env)
        assert prefixed_cfg.reference_artifact_verified is True

        from voxedge.backends.jetson.trt_edge_llm_tts import TRTEdgeLLMTTSBackend
        backend = TRTEdgeLLMTTSBackend(cfg)
        assert backend.supports_reference_audio_cloning is True

        for bad_source in (
            {"repo": repo, "revision": revision, "model_id": model},
            {"repo": repo, "revision": revision, "model_id": model, "canonical_model_id": "qwen3-tts-WRONG"},
        ):
            (cache / "manifest.json").write_text(
                json.dumps({**manifest, "_source": bad_source}), encoding="utf-8"
            )
            bad_cfg = build_trt_edge_llm_tts_config(profile=profile, env=env)
            assert bad_cfg.reference_artifact_verified is False

        stateful_only_files = {key: value for key, value in files.items() if key != "c2w/code2wav.engine"}
        (cache / "manifest.json").write_text(
            json.dumps({**manifest, "files": stateful_only_files}), encoding="utf-8"
        )
        stateful_cfg = build_trt_edge_llm_tts_config(profile=profile, env=env)
        assert stateful_cfg.reference_artifact_verified is False

        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        override_env = dict(env)
        override_env["EDGE_LLM_TTS_TALKER_ENGINE"] = str(root / "outside-talk.engine")
        Path(override_env["EDGE_LLM_TTS_TALKER_ENGINE"]).write_bytes(b"outside")
        override_cfg = build_trt_edge_llm_tts_config(profile=profile, env=override_env)
        assert override_cfg.reference_artifact_verified is False

        missing_env = dict(env)
        missing_env["EDGE_LLM_TTS_REFERENCE_TMP_DIR"] = str(root / "missing-ref-tmp")
        missing_cfg = build_trt_edge_llm_tts_config(profile=profile, env=missing_env)
        assert missing_cfg.reference_artifact_verified is False
        assert TRTEdgeLLMTTSBackend(missing_cfg).supports_reference_audio_cloning is False

        # The worker's indexed checkpoint reader resolves every weight_map
        # value below checkpoint_dir; directory-prefix presence is not proof.
        index_rel = "checkpoint/model.safetensors.index.json"
        shard_rel = "checkpoint/model-00001-of-00001.safetensors"
        index_path = root / index_rel
        shard_path = root / shard_rel
        shard_path.write_bytes(b"indexed-shard")
        index_path.write_text(json.dumps({"weight_map": {"weight": shard_path.name}}), encoding="utf-8")
        for rel in (index_rel, shard_rel):
            target = cache / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / rel).read_bytes())
            files[rel] = {
                "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest(),
                "size": (root / rel).stat().st_size,
            }
        (cache / "manifest.json").write_text(json.dumps({**manifest, "files": files}), encoding="utf-8")
        indexed_cfg = build_trt_edge_llm_tts_config(profile=profile, env=env)
        assert indexed_cfg.reference_artifact_verified is True

        # A declared index without its referenced shard is rejected.
        shard_path.unlink()
        missing_shard_cfg = build_trt_edge_llm_tts_config(profile=profile, env=env)
        assert missing_shard_cfg.reference_artifact_verified is False
        shard_path.write_bytes(b"indexed-shard")
        (cache / shard_rel).write_bytes(shard_path.read_bytes())

        # A present shard without an immutable manifest lock is rejected.
        unlocked_files = {key: value for key, value in files.items() if key != shard_rel}
        (cache / "manifest.json").write_text(json.dumps({**manifest, "files": unlocked_files}), encoding="utf-8")
        unlocked_cfg = build_trt_edge_llm_tts_config(profile=profile, env=env)
        assert unlocked_cfg.reference_artifact_verified is False

        (cache / "manifest.json").write_text(json.dumps({**manifest, "files": files}), encoding="utf-8")
        # Empty and malformed maps, plus traversal references, are fail-closed.
        for weight_map in ({}, {"weight": "../outside.safetensors"}, None):
            index_path.write_text(
                json.dumps({} if weight_map is None else {"weight_map": weight_map}),
                encoding="utf-8",
            )
            (cache / index_rel).write_bytes(index_path.read_bytes())
            malformed_cfg = build_trt_edge_llm_tts_config(profile=profile, env=env)
            assert malformed_cfg.reference_artifact_verified is False
        index_path.write_text(json.dumps({"weight_map": {"weight": shard_path.name}}), encoding="utf-8")
        (cache / index_rel).write_bytes(index_path.read_bytes())

    def test_process_env_overrides_profile_clone_paths(self):
        """Explicit process env keeps precedence over profile env values."""
        from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config

        cfg = build_trt_edge_llm_tts_config(profile={
            "env": {
                "EDGE_LLM_TTS_CLONE_ENCODER_DIR": "/profile/clone",
                "EDGE_LLM_TTS_CHECKPOINT_DIR": "/profile/checkpoint",
            }
        }, env={
            "EDGE_LLM_TTS_CLONE_ENCODER_DIR": "/process/clone",
            "EDGE_LLM_TTS_CHECKPOINT_DIR": "/process/checkpoint",
        })
        assert cfg.clone_encoder_dir == "/process/clone"
        assert cfg.checkpoint_dir == "/process/checkpoint"

    def test_clone_paths_fail_closed_when_old_factory_ignores_env(self, monkeypatch):
        """An old helper that drops the new env must not report a usable config."""
        import voxedge.backends.jetson.trt_edge_llm_tts as canonical
        from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config

        monkeypatch.setattr(canonical, "build_config_from_env", lambda env: object())
        with pytest.raises(RuntimeError, match="require a voxedge build"):
            build_trt_edge_llm_tts_config(env={
                "EDGE_LLM_TTS_CLONE_ENCODER_DIR": "/clone",
                "EDGE_LLM_TTS_CHECKPOINT_DIR": "/checkpoint",
            })

    def test_clone_paths_fail_closed_when_factory_maps_wrong_value(self, monkeypatch):
        """A helper exposing fields but returning a different path is rejected."""
        import voxedge.backends.jetson.trt_edge_llm_tts as canonical
        from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config

        monkeypatch.setattr(
            canonical,
            "build_config_from_env",
            lambda env: type("Config", (), {
                "clone_encoder_dir": "/wrong",
                "checkpoint_dir": "/checkpoint",
            })(),
        )
        with pytest.raises(RuntimeError, match="did not map"):
            build_trt_edge_llm_tts_config(env={
                "EDGE_LLM_TTS_CLONE_ENCODER_DIR": "/clone",
                "EDGE_LLM_TTS_CHECKPOINT_DIR": "/checkpoint",
            })

    def test_old_factory_without_clone_paths_still_passes_when_unconfigured(self, monkeypatch):
        """Legacy default profiles retain the old helper result unchanged."""
        import voxedge.backends.jetson.trt_edge_llm_tts as canonical
        from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config

        sentinel = object()
        monkeypatch.setattr(canonical, "build_config_from_env", lambda env: sentinel)
        assert build_trt_edge_llm_tts_config(env={}) is sentinel


# ---------------------------------------------------------------------------
# ASR parity
# ---------------------------------------------------------------------------

class TestASRBuilderParity:
    BASE_ENV = {
        # paths
        "EDGE_LLM_ASR_BIN": "/opt/edge/asr_bin",
        "EDGE_LLM_ASR_WORKER_BIN": "/opt/edge/asr_worker",
        "EDGE_LLM_ASR_PLUGIN_PATH": "/opt/edge/asr_plugin.so",
        "EDGE_LLM_ASR_ENGINE_DIR": "/opt/models/asr_engine",
        "EDGE_LLM_ASR_AUDIO_ENC_DIR": "/opt/models/audio_enc",
        # flags
        "EDGE_LLM_ASR_MAX_CONCURRENT": "2",
        "EDGE_LLM_ASR_STREAM_MODE": "accumulate",
        "EDGE_LLM_ASR_STREAM_CHUNK_SEC": "0.4",
        "EDGE_LLM_ASR_STREAM_UNFIXED_CHUNKS": "3",
        "EDGE_LLM_ASR_STREAM_UNFIXED_TOKENS": "7",
        "EDGE_LLM_ASR_MEL_SETTINGS": "/opt/mel_settings.json",
        "EDGE_LLM_ASR_MEL_FILTERS": "/opt/mel_filters.npy",
        # sampling
        "ASR_TEMPERATURE": "0.9",
        "ASR_TOP_P": "0.95",
        "ASR_TOP_K": "3",
        "ASR_MAX_GENERATE_LENGTH": "150",
        # offline segmentation
        "EDGE_LLM_ASR_OFFLINE_SEGMENT": "1",
        "EDGE_LLM_ASR_OFFLINE_SEGMENT_SEC": "5.0",
        "EDGE_LLM_ASR_OFFLINE_MIN_SEGMENT_SEC": "0.3",
        # warmup
        "EDGE_LLM_ASR_PREWARM_MAX": "4",
        "EDGE_LLM_ASR_CUDA_GRAPH": "0",
    }

    def test_parity_with_direct_factory(self):
        from server.core.voxedge_backend_config import build_trt_edge_llm_asr_config
        from voxedge.backends.jetson.trt_edge_llm_asr import build_config_from_env

        env = dict(self.BASE_ENV)
        ovs_cfg = build_trt_edge_llm_asr_config(profile=None, env=env)
        vox_cfg = build_config_from_env(env=env)

        skip = {"extra_worker_env", "artifact_ref"}
        for k in _all_fields(ovs_cfg):
            if k in skip:
                continue
            assert _all_fields(ovs_cfg)[k] == _all_fields(vox_cfg)[k], (
                f"Field '{k}' differs: OVS={_all_fields(ovs_cfg)[k]!r}, "
                f"voxedge={_all_fields(vox_cfg)[k]!r}"
            )

    def test_parity_minimal_env(self):
        from server.core.voxedge_backend_config import build_trt_edge_llm_asr_config
        from voxedge.backends.jetson.trt_edge_llm_asr import build_config_from_env

        env = {}
        ovs_cfg = build_trt_edge_llm_asr_config(profile=None, env=env)
        vox_cfg = build_config_from_env(env=env)

        skip = {"extra_worker_env", "artifact_ref"}
        for k in _all_fields(ovs_cfg):
            if k in skip:
                continue
            assert _all_fields(ovs_cfg)[k] == _all_fields(vox_cfg)[k], (
                f"Field '{k}' differs for minimal env"
            )

    def test_profile_max_slots_injection(self):
        """Profile asr_max_slots is injected when EDGE_LLM_ASR_MAX_CONCURRENT absent."""
        from server.core.voxedge_backend_config import build_trt_edge_llm_asr_config

        profile = {"asr_max_slots": 4}
        env = {}
        cfg = build_trt_edge_llm_asr_config(profile=profile, env=env)
        assert cfg.max_slots == 4

    def test_profile_max_slots_overridden_by_env(self):
        """Explicit env takes priority over profile asr_max_slots."""
        from server.core.voxedge_backend_config import build_trt_edge_llm_asr_config

        profile = {"asr_max_slots": 4}
        env = {"EDGE_LLM_ASR_MAX_CONCURRENT": "2"}
        cfg = build_trt_edge_llm_asr_config(profile=profile, env=env)
        assert cfg.max_slots == 2


# ---------------------------------------------------------------------------
# env=None path: OVS wrappers read os.environ when env not passed
# ---------------------------------------------------------------------------

def test_tts_wrapper_env_none_reads_os_environ(monkeypatch):
    """OVS wrapper 默认 env=None 时应透传 os.environ 给 voxedge factory。"""
    monkeypatch.setenv("EDGE_LLM_TTS_WORKER_BIN", "/tmp/fake_tts_worker")
    from server.core.voxedge_backend_config import build_trt_edge_llm_tts_config
    cfg = build_trt_edge_llm_tts_config()
    assert cfg.worker_binary == "/tmp/fake_tts_worker"


def test_asr_wrapper_env_none_reads_os_environ(monkeypatch):
    monkeypatch.setenv("EDGE_LLM_ASR_WORKER_BIN", "/tmp/fake_asr_worker")
    from server.core.voxedge_backend_config import build_trt_edge_llm_asr_config
    cfg = build_trt_edge_llm_asr_config()
    assert cfg.worker_binary == "/tmp/fake_asr_worker"
