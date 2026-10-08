"""Tests for server.core.model_downloader profile-driven routing.

Regression: orin-nano 2026-05-25 silent Qwen3-skip bug. When OVS_PROFILE
selected a Qwen3 ASR profile but the environment had LANGUAGE_MODE=zh_en
pre-set, ensure_models() routed by language_mode and skipped
_ensure_qwen3_artifacts(). After fix, routing is profile-driven first
(asr_backend/tts_backend) and language_mode-driven second; both UNION.
"""

from __future__ import annotations

import hashlib
import os
from unittest.mock import patch

import pytest

from server.core import model_downloader


def _no_op_download(url, dest_dir, **kwargs):  # pragma: no cover - safety net
    raise AssertionError(
        f"unexpected real download attempt: {url} -> {dest_dir}"
    )


def test_profile_with_trt_edge_llm_triggers_qwen3_even_when_lang_mode_zh_en(
    tmp_path, monkeypatch,
):
    """Profile asr_backend=jetson.trt_edge_llm must call _ensure_qwen3_artifacts
    even when language_mode='zh_en' (the legacy zh_en path would otherwise
    never call it). This is the core orin-nano regression fix."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "asr_backend": "jetson.trt_edge_llm",
            "tts_backend": "jetson.matcha_trt",
        },
    )
    monkeypatch.setattr(model_downloader, "_matcha_model_files_valid", lambda _: True)
    # Patch the symbol the function actually calls (module-level lookup).
    with patch.object(
        model_downloader, "_ensure_qwen3_artifacts"
    ) as mock_qwen3, patch.object(
        model_downloader, "_download_and_extract", side_effect=_no_op_download,
    ):
        # Pretend zh_en assets (matcha + paraformer) are already present
        # so no actual download fires. language_mode='zh_en' unions in
        # the legacy zh_en requirements alongside profile-driven matcha.
        for sub, files in (
            ("matcha-icefall-zh-en", ("model-steps-3.onnx", "tokens.txt", "lexicon.txt")),
            ("paraformer-streaming", ("encoder.onnx", "tokens.txt")),
        ):
            d = tmp_path / sub
            d.mkdir()
            for f in files:
                (d / f).write_text("x")

        model_downloader.ensure_models(
            language_mode="zh_en", model_dir=str(tmp_path),
        )

    mock_qwen3.assert_called_once()


def test_matcha_profile_selects_only_qwen_asr_artifact_directories(monkeypatch):
    root = "/opt/models/edgellm-v091"
    monkeypatch.setenv("QWEN3_ARTIFACT_ROOT", root)
    profile = {
        "asr_backend": "jetson.trt_edge_llm",
        "tts_backend": "jetson.matcha_trt",
        "env": {
            "EDGE_LLM_ASR_AUDIO_ENC_DIR": f"{root}/engines/asr_audio_encoder",
            "MATCHA_MODEL_BASE": "/opt/models/matcha-icefall-zh-en",
        },
        "required_engines": [
            {
                "engine_path": (
                    f"{root}/engines/asr_thinker_full_int4_b1/llm.engine"
                )
            },
            {
                "engine_path": (
                    "/opt/models/matcha-icefall-zh-en/engines/vocos_fp16.engine"
                )
            },
        ],
    }

    assert model_downloader._profile_qwen_required_files(profile) == [
        "engines/asr_audio_encoder/audio/audio_encoder.engine",
        "engines/asr_thinker_full_int4_b1/llm.engine",
    ]


def test_no_profile_zh_en_legacy_path_does_not_call_qwen3(tmp_path, monkeypatch):
    """Backward-compat: no profile + LANGUAGE_MODE=zh_en must NOT trigger
    Qwen3 (legacy behaviour for users who never opted into profiles)."""
    from server.core import profile_loader
    monkeypatch.setattr(profile_loader, "current_profile", lambda: {})
    monkeypatch.setattr(model_downloader, "_matcha_model_files_valid", lambda _: True)
    with patch.object(
        model_downloader, "_ensure_qwen3_artifacts"
    ) as mock_qwen3, patch.object(
        model_downloader, "_download_and_extract", side_effect=_no_op_download,
    ):
        # Pretend both zh_en models exist so we don't trip the downloader.
        for sub, files in (
            ("matcha-icefall-zh-en", ("model-steps-3.onnx", "tokens.txt", "lexicon.txt")),
            ("paraformer-streaming", ("encoder.onnx", "tokens.txt")),
        ):
            d = tmp_path / sub
            d.mkdir()
            for f in files:
                (d / f).write_text("x")

        model_downloader.ensure_models(
            language_mode="zh_en", model_dir=str(tmp_path),
        )

    mock_qwen3.assert_not_called()


@pytest.mark.parametrize("disabled", ["0", "false", "no", "off"])
def test_whisper_profile_filters_zipformer_and_does_not_download(
    tmp_path, monkeypatch, disabled,
):
    """An explicit Whisper ASR profile must not inherit the English Zipformer."""
    from server.core import profile_loader

    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {"asr_backend": "jetson.whisper_trt", "tts_backend": None},
    )
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", disabled)
    with patch.object(model_downloader, "_ensure_whisper_artifacts") as mock_whisper, \
            patch.object(model_downloader, "_download_and_extract") as mock_download, \
            patch.object(model_downloader.os, "makedirs") as mock_makedirs:
        model_downloader.ensure_models("en", str(tmp_path))

    mock_whisper.assert_called_once_with("jetson.whisper_trt")
    mock_download.assert_not_called()
    mock_makedirs.assert_not_called()


def test_cpu_sherpa_profile_retains_zipformer(tmp_path, monkeypatch):
    """Selecting the CPU Sherpa ASR backend keeps its legacy Zipformer bundle."""
    from server.core import profile_loader

    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {"asr_backend": "cpu.sherpa_asr", "tts_backend": None},
    )
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "0")
    model = tmp_path / "zipformer-en"
    model.mkdir()
    (model / "encoder.int8.onnx").write_text("cached")
    (model / "tokens.txt").write_text("cached")
    with patch.object(model_downloader, "_download_and_extract") as mock_download:
        model_downloader.ensure_models("en", str(tmp_path))

    mock_download.assert_not_called()


def test_no_profile_en_keeps_legacy_bundle(tmp_path, monkeypatch):
    """No profile backend preserves both English legacy model requirements."""
    from server.core import profile_loader

    monkeypatch.setattr(profile_loader, "current_profile", lambda: {})
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "0")
    for name, files in {
        "zipformer-en": ("encoder.int8.onnx", "tokens.txt"),
        "kokoro-multi-lang-v1_0": (
            "model.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt",
        ),
    }.items():
        model = tmp_path / name
        model.mkdir()
        for filename in files:
            (model / filename).write_text("cached")
    with patch.object(model_downloader, "_download_and_extract") as mock_download:
        model_downloader.ensure_models("en", str(tmp_path))

    mock_download.assert_not_called()


@pytest.mark.parametrize("disabled", ["0", "false", "no", "off"])
def test_legacy_offline_missing_fails_before_download_or_mkdir(
    tmp_path, monkeypatch, disabled,
):
    """Disabled auto-download must fail closed for missing legacy artifacts."""
    from server.core import profile_loader

    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {"asr_backend": "cpu.sherpa_asr", "tts_backend": None},
    )
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", disabled)
    with patch.object(model_downloader, "_download_and_extract") as mock_download, \
            patch.object(model_downloader.os, "makedirs") as mock_makedirs:
        with pytest.raises(RuntimeError, match="zipformer-en"):
            model_downloader.ensure_models("en", str(tmp_path))

    mock_download.assert_not_called()
    mock_makedirs.assert_not_called()


@pytest.mark.parametrize("enabled", ["1", None])
def test_legacy_missing_keeps_download_behavior_when_enabled_or_unset(
    tmp_path, monkeypatch, enabled,
):
    """Enabled and unset flags retain the existing legacy downloader path."""
    from server.core import profile_loader

    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {"asr_backend": "cpu.sherpa_asr", "tts_backend": None},
    )
    if enabled is None:
        monkeypatch.delenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", raising=False)
    else:
        monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", enabled)
    with patch.object(model_downloader, "_download_and_extract") as mock_download:
        model_downloader.ensure_models("en", str(tmp_path))

    mock_download.assert_called_once()


def test_explicit_qwen_model_source_skips_legacy_multilanguage_aggregate(
    tmp_path, monkeypatch,
):
    """v091 model-level Qwen sources must not trigger the legacy 26-file set."""
    from server.core import profile_loader, qwen3_artifact_downloader

    profile = {
        "asr_backend": "jetson.trt_edge_llm",
        "model_artifacts": [
            {
                "model_id": "qwen3-asr",
                "canonical_model_id": "qwen3-asr-0.6b",
                "repo": "harvestsu/qwen3-asr-0.6b-jetson-artifacts",
            },
        ],
    }
    monkeypatch.setattr(profile_loader, "current_profile", lambda: profile)
    with patch.object(
        qwen3_artifact_downloader, "ensure_model_requests", return_value=True,
    ) as mock_profile, patch.object(
        model_downloader, "_ensure_qwen3_artifacts"
    ) as mock_qwen3:
        model_downloader.ensure_models(
            language_mode="multilanguage", model_dir=str(tmp_path),
        )

    mock_profile.assert_called_once()
    assert mock_profile.call_args.args[0][0]["canonical_model_id"] == "qwen3-asr-0.6b"
    mock_qwen3.assert_not_called()


def test_no_profile_multilanguage_still_uses_legacy_qwen3_path(tmp_path, monkeypatch):
    """Without a profile/model source, multilanguage keeps its legacy Qwen3 path."""
    from server.core import profile_loader

    monkeypatch.setattr(profile_loader, "current_profile", lambda: {})
    with patch.object(model_downloader, "_ensure_qwen3_artifacts") as mock_qwen3:
        model_downloader.ensure_models(
            language_mode="multilanguage", model_dir=str(tmp_path),
        )

    mock_qwen3.assert_called_once_with(None)


def test_moss_profile_triggers_moss_provision(tmp_path, monkeypatch):
    """A profile with tts_backend=jetson.moss_tts_nano must fire the MOSS
    provisioner (#47 unified-entry dispatch), even though MOSS does not go
    through the MODELS/CDN tarball mechanism."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {"asr_backend": None, "tts_backend": "jetson.moss_tts_nano"},
    )
    with patch.object(model_downloader, "_ensure_moss_artifacts") as mock_moss, \
            patch.object(model_downloader, "_ensure_qwen3_artifacts"), \
            patch.object(
                model_downloader, "_download_and_extract", side_effect=_no_op_download
            ):
        # multilanguage mode + moss-only TTS: no matcha/paraformer required.
        model_downloader.ensure_models(
            language_mode="multilanguage", model_dir=str(tmp_path),
        )

    mock_moss.assert_called_once()


def test_asr_artifact_manifest_routes_to_edgellm_and_not_qwen3(tmp_path, monkeypatch):
    """A trt_edge_llm profile that declares `asr_artifact_manifest` must go to
    the flat-manifest v090 provisioner and must NOT also fire the qwen3 path —
    including via the language_mode='multilanguage' branch, which is exactly
    what the v090 profile sets (LANGUAGE_MODE=multilanguage)."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "asr_backend": "jetson.trt_edge_llm",
            "tts_backend": "jetson.moss_tts_nano",
            "asr_artifact_manifest": "deploy/artifacts/edgellm_v090_manifest.json",
        },
    )
    with patch.object(model_downloader, "_ensure_edgellm_v090_artifacts") as mock_v090, \
            patch.object(model_downloader, "_ensure_qwen3_artifacts") as mock_qwen3, \
            patch.object(model_downloader, "_ensure_moss_artifacts"), \
            patch.object(
                model_downloader, "_download_and_extract", side_effect=_no_op_download
            ):
        model_downloader.ensure_models(
            language_mode="multilanguage", model_dir=str(tmp_path),
        )

    mock_v090.assert_called_once_with("deploy/artifacts/edgellm_v090_manifest.json")
    mock_qwen3.assert_not_called()


def test_trt_edge_llm_without_manifest_field_keeps_qwen3_path(tmp_path, monkeypatch):
    """No `asr_artifact_manifest` → unchanged legacy behaviour: qwen3 fires and
    the v090 provisioner stays out of it."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "asr_backend": "jetson.trt_edge_llm",
            "tts_backend": "jetson.moss_tts_nano",
        },
    )
    with patch.object(model_downloader, "_ensure_edgellm_v090_artifacts") as mock_v090, \
            patch.object(model_downloader, "_ensure_qwen3_artifacts") as mock_qwen3, \
            patch.object(model_downloader, "_ensure_moss_artifacts"), \
            patch.object(
                model_downloader, "_download_and_extract", side_effect=_no_op_download
            ):
        model_downloader.ensure_models(
            language_mode="multilanguage", model_dir=str(tmp_path),
        )

    mock_v090.assert_not_called()
    assert mock_qwen3.call_count >= 1


def test_v090_profile_json_dispatches_to_edgellm(tmp_path, monkeypatch):
    """End-to-end on the real profile file: loading the shipped
    jetson-edgellm-v090-moss.json routes ASR provisioning to the v090
    manifest, never to qwen3."""
    import json
    from pathlib import Path

    from server.core import profile_loader

    profile = json.loads(
        (Path(__file__).resolve().parents[2]
         / "configs" / "profiles" / "jetson-edgellm-v090-moss.json").read_text()
    )
    monkeypatch.setattr(profile_loader, "current_profile", lambda: profile)
    with patch.object(model_downloader, "_ensure_edgellm_v090_artifacts") as mock_v090, \
            patch.object(model_downloader, "_ensure_qwen3_artifacts") as mock_qwen3, \
            patch.object(model_downloader, "_ensure_moss_artifacts") as mock_moss, \
            patch.object(
                model_downloader, "_download_and_extract", side_effect=_no_op_download
            ):
        model_downloader.ensure_models(
            language_mode=profile["env"]["LANGUAGE_MODE"], model_dir=str(tmp_path),
        )

    mock_v090.assert_called_once_with("deploy/artifacts/edgellm_v090_manifest.json")
    mock_qwen3.assert_not_called()
    mock_moss.assert_called_once()


def test_non_moss_profile_does_not_trigger_moss(tmp_path, monkeypatch):
    """A non-MOSS profile (Qwen3 ASR + Matcha TTS) must NOT fire the MOSS
    provisioner."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "asr_backend": "jetson.trt_edge_llm",
            "tts_backend": "jetson.matcha_trt",
        },
    )
    monkeypatch.setattr(model_downloader, "_matcha_model_files_valid", lambda _: True)
    with patch.object(model_downloader, "_ensure_moss_artifacts") as mock_moss, \
            patch.object(model_downloader, "_ensure_qwen3_artifacts"), \
            patch.object(
                model_downloader, "_download_and_extract", side_effect=_no_op_download
            ):
        # matcha already present so no real download fires.
        d = tmp_path / "matcha-icefall-zh-en"
        d.mkdir()
        for f in ("model-steps-3.onnx", "tokens.txt", "lexicon.txt"):
            (d / f).write_text("x")
        model_downloader.ensure_models(
            language_mode="multilanguage", model_dir=str(tmp_path),
        )

    mock_moss.assert_not_called()


def test_matcha_release_manifest_contains_real_sha256_and_sizes():
    manifest = model_downloader._load_matcha_manifest()
    bundle = manifest["model_bundle"]
    assert bundle["sha256"] == (
        "a181221508eee0f4388465fea17f81f3dd0fb2f1528279d657c72033591fa9cc"
    )
    assert bundle["size"] == 129182339
    assert set(bundle["required_files"]) == {
        "model-steps-3.onnx", "tokens.txt", "lexicon.txt",
    }
    for group in (bundle["required_files"], manifest["split_onnx"]):
        for lock in group.values():
            assert len(lock["sha256"]) == 64
            assert lock["size"] > 0


def test_matcha_cache_requires_locked_size_and_sha(tmp_path, monkeypatch):
    model = tmp_path / "matcha-icefall-zh-en"
    model.mkdir()
    payloads = {
        "model-steps-3.onnx": b"model",
        "tokens.txt": b"tokens",
        "lexicon.txt": b"lexicon",
    }
    locks = {
        name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in payloads.items()
    }
    manifest = {
        "model_id": "matcha-icefall-zh-en",
        "model_bundle": {"required_files": locks},
    }
    monkeypatch.setattr(model_downloader, "_load_matcha_manifest", lambda: manifest)
    for name, data in payloads.items():
        (model / name).write_bytes(data)
    assert model_downloader._matcha_model_files_valid(model)

    (model / "tokens.txt").write_bytes(b"tampered")
    assert not model_downloader._matcha_model_files_valid(model)


def test_matcha_split_onnx_redownloads_invalid_cache_with_lock(tmp_path, monkeypatch):
    from server.core import hf_artifacts

    payloads = {"encoder.onnx": b"locked-encoder", "estimator.onnx": b"locked-est"}
    locks = {
        name: {
            "repo_path": f"models/matcha/{name}",
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in payloads.items()
    }
    manifest = {"model_id": "matcha-icefall-zh-en", "split_onnx": locks}
    monkeypatch.setattr(model_downloader, "_load_matcha_manifest", lambda: manifest)
    monkeypatch.setattr(model_downloader, "_MATCHA_SPLIT_ONNX_FILES", tuple(locks))
    monkeypatch.setenv("MATCHA_ACOUSTIC_EP", "SPLIT_TRT")
    monkeypatch.setenv("MATCHA_SPLIT_ENCODER_ONNX", str(tmp_path / "encoder.onnx"))
    (tmp_path / "encoder.onnx").write_bytes(b"bad")

    calls = []
    def fake_download(rel, dest, expected_sha256=None):
        name = dest.name
        calls.append((rel, name, expected_sha256))
        dest.write_bytes(payloads[name])
        return dest

    monkeypatch.setattr(hf_artifacts, "download_file", fake_download)
    model_downloader._ensure_matcha_split_onnx(str(tmp_path))

    assert {name for _, name, _ in calls} == set(payloads)
    for rel, name, sha in calls:
        assert rel == locks[name]["repo_path"]
        assert sha == locks[name]["sha256"]


def test_matcha_split_onnx_download_failure_is_fail_closed(tmp_path, monkeypatch):
    from server.core import hf_artifacts

    data = b"locked"
    locks = {"encoder.onnx": {
        "repo_path": "models/matcha/encoder.onnx",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }}
    monkeypatch.setattr(
        model_downloader,
        "_load_matcha_manifest",
        lambda: {"model_id": "matcha-icefall-zh-en", "split_onnx": locks},
    )
    monkeypatch.setattr(model_downloader, "_MATCHA_SPLIT_ONNX_FILES", tuple(locks))
    monkeypatch.setenv("MATCHA_ACOUSTIC_EP", "SPLIT_TRT")
    monkeypatch.setenv("MATCHA_SPLIT_ENCODER_ONNX", str(tmp_path / "encoder.onnx"))
    monkeypatch.setattr(
        hf_artifacts,
        "download_file",
        lambda *a, **k: (_ for _ in ()).throw(hf_artifacts.ArtifactError("offline")),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        model_downloader._ensure_matcha_split_onnx(str(tmp_path))


def test_symlinked_model_dir_counts_as_present(tmp_path, monkeypatch):
    """A model dir reached through a symlink must not read as missing.

    Regression: harvest-pi 2026-08-09. /opt/models/sensevoice/ held a symlink
    to the extracted tarball rather than the files themselves. os.walk does not
    descend into symlinked directories by default, so the readiness scan found
    no model.int8.onnx, ensure_models() decided SenseVoice was missing, tried to
    re-download it from huggingface.co (unreachable on the device), and killed
    startup with sys.exit(1) -- while the model sat right there on disk.
    """
    from server.core import profile_loader
    monkeypatch.setattr(profile_loader, "current_profile", lambda: {})
    # SenseVoice only enters the requirement set under ENSURE_OFFLINE_ASR,
    # which is exactly how the rpi/jetson images are configured.
    monkeypatch.setenv("ENSURE_OFFLINE_ASR", "1")
    monkeypatch.delenv("ASR_BACKEND", raising=False)

    model_dir = tmp_path / "models"
    # The real extraction lives beside the expected dir...
    extracted = model_dir / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    extracted.mkdir(parents=True)
    (extracted / "model.int8.onnx").write_bytes(b"\x00")
    # ...and the dir the downloader checks only points at it.
    sensevoice = model_dir / "sensevoice"
    sensevoice.mkdir()
    (sensevoice / extracted.name).symlink_to(extracted, target_is_directory=True)

    paraformer = model_dir / "paraformer-streaming"
    paraformer.mkdir()
    (paraformer / "encoder.onnx").write_bytes(b"\x00")
    (paraformer / "tokens.txt").write_bytes(b"\x00")

    monkeypatch.setattr(model_downloader, "_matcha_model_files_valid", lambda p: True)
    (model_dir / "matcha-icefall-zh-en").mkdir()
    monkeypatch.setattr(model_downloader, "_patch_kokoro_voices", lambda d: None)

    with patch.object(model_downloader, "_download_and_extract", _no_op_download):
        # Must return cleanly: nothing missing, so nothing downloaded.
        model_downloader.ensure_models("zh_en", str(model_dir))


def test_symlink_cycle_is_walked_once_not_repeatedly(tmp_path, monkeypatch):
    """A directory link back to an ancestor must not be re-walked.

    followlinks=True (needed so a symlinked model dir is seen at all) also lets
    os.walk revisit directories through a cycle. Measured, the cycle does NOT
    hang: both macOS and Linux stop it at the symlink-resolution depth limit
    (82 iterations on the Pi 5, ELOOP, silently swallowed by os.walk). So this
    is wasted traversal, not a hang — but it is unbounded-looking work on every
    readiness scan, and pruning by realpath costs nothing.

    Asserts the walk count directly: without the guard this structure yields
    dozens of directories, with it exactly the distinct ones.
    """
    from server.core import profile_loader
    monkeypatch.setattr(profile_loader, "current_profile", lambda: {})
    monkeypatch.setenv("ENSURE_OFFLINE_ASR", "1")
    monkeypatch.delenv("ASR_BACKEND", raising=False)

    model_dir = tmp_path / "models"
    sensevoice = model_dir / "sensevoice"
    sensevoice.mkdir(parents=True)
    (sensevoice / "model.int8.onnx").write_bytes(b"\x00")
    nested = sensevoice / "nested"
    nested.mkdir()
    (nested / "loop").symlink_to(sensevoice, target_is_directory=True)

    paraformer = model_dir / "paraformer-streaming"
    paraformer.mkdir()
    (paraformer / "encoder.onnx").write_bytes(b"\x00")
    (paraformer / "tokens.txt").write_bytes(b"\x00")

    monkeypatch.setattr(model_downloader, "_matcha_model_files_valid", lambda p: True)
    (model_dir / "matcha-icefall-zh-en").mkdir()
    monkeypatch.setattr(model_downloader, "_patch_kokoro_voices", lambda d: None)

    real_walk = os.walk
    counts: dict[str, int] = {}

    def counting_walk(top, *a, **kw):
        for item in real_walk(top, *a, **kw):
            counts[str(top)] = counts.get(str(top), 0) + 1
            yield item

    monkeypatch.setattr(model_downloader.os, "walk", counting_walk)

    with patch.object(model_downloader, "_download_and_extract", _no_op_download):
        model_downloader.ensure_models("zh_en", str(model_dir))

    # sensevoice/, sensevoice/nested/, and the link resolved once — never the
    # nested/loop/nested/loop/... chain.
    walked = counts[str(sensevoice)]
    assert walked <= 3, f"cycle re-walked: {walked} directories visited"
