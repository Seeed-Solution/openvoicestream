"""env → WhisperASRConfig, and the capability probe that reads it."""
from __future__ import annotations

import dataclasses

import pytest

from server.core import voxedge_backend_config as vbc

_ENV = {"MODEL_DIR": "/opt/m"}
# The variant is per-accelerator: it names the artifact, and the window and
# boundary guard are read off it.
_VARIANT = {"hailo": "tiny", "rknn": "base10", "tensorrt": "base"}


def test_a_variant_is_required():
    # It selects the decoder family and the compiled window, so there is no
    # safe default: guessing pairs a tiny encoder with a base decoder, which
    # yields fluent nonsense rather than an error.
    with pytest.raises(ValueError, match="WHISPER_VARIANT"):
        vbc.build_whisper_asr_config("rknn", env={})


@pytest.mark.parametrize("kind,variant,window,cutoff", [
    # The window belongs to the ARTIFACT, not the accelerator: Hailo ships tiny
    # at 10 s and base at 5 s, RK ships 10 s and 20 s. A default keyed only by
    # accelerator handed the 5 s HEF a 10 s window.
    ("hailo", "tiny", 10.0, 1.0),
    ("hailo", "base", 5.0, 1.0),
    ("rknn", "base10", 10.0, 0.0),
    ("rknn", "base20", 20.0, 0.0),
    ("tensorrt", "base", 30.0, 0.0),
])
def test_per_artifact_window_and_cutoff_defaults(kind, variant, window, cutoff):
    cfg = vbc.build_whisper_asr_config(kind, env={**_ENV, "WHISPER_VARIANT": variant})
    assert (cfg.window_s, cfg.padding_cutoff_s) == (window, cutoff)
    assert cfg.encoder_kind == kind


def test_directories_default_under_model_dir():
    # MODEL_DIR still works as a root; the layout underneath is the one the
    # downloader writes, and the decoder family follows WHISPER_VARIANT.
    cfg = vbc.build_whisper_asr_config("rknn", env={**_ENV, "WHISPER_VARIANT": "base10"})
    assert cfg.vocab_dir == "/opt/m/whisper"
    assert cfg.decoder_dir == "/opt/m/whisper/decoder/base"


def _whisper_config_has_decoder_kind() -> bool:
    from voxedge.backends.whisper import WhisperASRConfig

    return any(f.name == "decoder_kind" for f in dataclasses.fields(WhisperASRConfig))


_HAS_DECODER_KIND = _whisper_config_has_decoder_kind()


def test_trt_decoder_request_fails_loudly_on_a_voxedge_without_the_field():
    if _HAS_DECODER_KIND:
        pytest.skip("installed voxedge declares decoder_kind")
    cfg = vbc.build_whisper_asr_config("tensorrt", env={**_ENV, "WHISPER_VARIANT": "base"})
    assert not hasattr(cfg, "decoder_kind")
    with pytest.raises(ValueError, match="WHISPER_DECODER_KIND"):
        vbc.build_whisper_asr_config(
            "tensorrt",
            env={**_ENV, "WHISPER_VARIANT": "base", "WHISPER_DECODER_KIND": "tensorrt"},
        )


@pytest.mark.skipif(not _HAS_DECODER_KIND, reason="installed voxedge has no decoder_kind")
def test_decoder_kind_defaults_to_legacy_cpu_and_trt_paths_are_explicit():
    cfg = vbc.build_whisper_asr_config(
        "tensorrt", env={**_ENV, "WHISPER_VARIANT": "base"}
    )
    assert cfg.decoder_kind == "onnx_cpu"
    cfg = vbc.build_whisper_asr_config(
        "tensorrt",
        env={
            **_ENV,
            "WHISPER_VARIANT": "base",
            "WHISPER_DECODER_KIND": "tensorrt",
            "WHISPER_DECODER_PREFILL_PLAN": "/plans/prefill.plan",
            "WHISPER_DECODER_STEP_PLAN": "/plans/step.plan",
        },
    )
    assert cfg.decoder_kind == "tensorrt"
    assert cfg.decoder_prefill_path == "/plans/prefill.plan"
    assert cfg.decoder_step_path == "/plans/step.plan"


def test_env_overrides_win():
    cfg = vbc.build_whisper_asr_config("rknn", env={
        **_ENV,
        "WHISPER_VARIANT": "base20",
        "WHISPER_WINDOW_S": "20",
        "WHISPER_LANGUAGE": "zh",
        "WHISPER_DECODER_THREADS": "4",
        "WHISPER_ALL_CORES": "1",
        "WHISPER_VOCAB_DIR": "/data/vocab",
    })
    assert cfg.window_s == 20.0
    assert cfg.language == "zh"
    assert cfg.decoder_threads == 4
    assert cfg.all_cores is True
    assert cfg.vocab_dir == "/data/vocab"


def test_an_unparseable_window_raises_rather_than_defaulting():
    # The window selects the encoder's compiled shape, and rknn-lite does not
    # validate it — a silent fallback returns plausible nonsense instead.
    with pytest.raises(ValueError, match="WHISPER_WINDOW_S"):
        vbc.build_whisper_asr_config(
            "rknn", env={**_ENV, "WHISPER_VARIANT": "base10", "WHISPER_WINDOW_S": "wide"}
        )


@pytest.mark.parametrize("spec,kind", [
    ("hailo.whisper", "hailo"),
    ("rk.whisper", "rknn"),
    ("jetson.whisper_trt", "tensorrt"),
])
def test_spec_dispatch(monkeypatch, spec, kind):
    monkeypatch.setenv("WHISPER_VARIANT", _VARIANT[kind])
    assert vbc.build_config_for_spec(spec, "asr", {}).encoder_kind == kind


def test_capability_probe_reads_the_encoder_kind(monkeypatch):
    """The probe builds a stub without __init__, so it must not need hardware.

    Hailo hands /dev/hailo0 to one process, so its path needs a cross-backend
    device mutex; a TRT engine shares the GPU with the TTS stack and does not.
    """
    monkeypatch.setenv("WHISPER_VARIANT", "base")
    from voxedge.backends.whisper import WhisperASR

    hailo = vbc.concurrency_capability_for_spec("hailo.whisper", WhisperASR, "asr", {})
    trt = vbc.concurrency_capability_for_spec("jetson.whisper_trt", WhisperASR, "asr", {})
    assert hailo.requires_exclusive_device is True
    assert trt.requires_exclusive_device is False
    assert hailo.max_concurrent == 1 and hailo.is_stateful is False
