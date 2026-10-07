"""Parity tests: OVS builder wrappers vs direct voxedge factory functions.

Verifies that ``server.core.voxedge_backend_config.build_trt_edge_llm_tts_config``
and ``build_trt_edge_llm_asr_config`` (which now delegate to the canonical voxedge
factories) produce config objects with identical fields when called with the same
env dict as the factories directly.
"""

import pytest


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
