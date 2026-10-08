from __future__ import annotations

from pathlib import Path

import pytest

from server.core import model_downloader as md


def _profile(monkeypatch, spec="rk.whisper"):
    from server.core import profile_loader

    monkeypatch.setattr(
        profile_loader, "current_profile", lambda: {"asr_backend": spec}
    )


def _write(path: Path, data: bytes = b"cached"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _cache(root: Path, spec="rk.whisper", variant="base10", *, env=None):
    env = env or {}
    encoder, decoders = md._WHISPER_ENCODER_FILES[spec][variant]
    encoder_path = Path(env.get("WHISPER_ENCODER_PATH", root / encoder))
    if spec == "jetson.whisper_trt" and "WHISPER_ENCODER_PATH" not in env:
        encoder_path = root / md._WHISPER_TRT_PLAN
    _write(encoder_path)
    decoder_dir = Path(env.get(
        "WHISPER_DECODER_DIR", root / "decoder" / ("tiny" if "tiny" in variant else "base")
    ))
    if env.get("WHISPER_DECODER_KIND", "onnx_cpu") == "onnx_cpu":
        for name in decoders:
            _write(decoder_dir / Path(name).name)
    else:
        _write(Path(env["WHISPER_DECODER_PREFILL_PLAN"]))
        _write(Path(env["WHISPER_DECODER_STEP_PLAN"]))
    vocab_dir = Path(env.get("WHISPER_VOCAB_DIR", root))
    _write(vocab_dir / "mel_80_filters.txt")
    _write(vocab_dir / f"vocab_{env.get('WHISPER_LANGUAGE', 'en')}.txt")


def test_offline_ensure_models_accepts_complete_cached_rk_route(monkeypatch, tmp_path):
    _profile(monkeypatch)
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "0")
    monkeypatch.setenv("WHISPER_MODEL_DIR", str(tmp_path))
    _cache(tmp_path)
    calls = []
    monkeypatch.setattr(md, "_build_whisper_trt_engine", lambda *a: calls.append(a))
    monkeypatch.setattr(md.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(md.os, "makedirs", lambda *a, **k: calls.append((a, k)))

    md.ensure_models("custom", str(tmp_path))
    assert calls == []


@pytest.mark.parametrize("missing", [
    "encoder", "decoder", "mel", "vocab",
])
def test_offline_missing_resource_fails_before_download_or_build(monkeypatch, tmp_path, missing):
    _profile(monkeypatch)
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "false")
    monkeypatch.setenv("WHISPER_MODEL_DIR", str(tmp_path))
    _cache(tmp_path)
    encoder, decoders = md._WHISPER_ENCODER_FILES["rk.whisper"]["base10"]
    targets = {
        "encoder": tmp_path / encoder,
        "decoder": tmp_path / "decoder" / "base" / Path(decoders[0]).name,
        "mel": tmp_path / "mel_80_filters.txt",
        "vocab": tmp_path / "vocab_en.txt",
    }
    targets[missing].unlink()
    calls = []
    monkeypatch.setattr(md, "_build_whisper_trt_engine", lambda *a: calls.append("build"))
    monkeypatch.setattr(md.subprocess, "run", lambda *a, **k: calls.append("download"))
    monkeypatch.setattr(md.os, "makedirs", lambda *a, **k: calls.append("mkdir"))

    with pytest.raises(RuntimeError, match="incomplete"):
        md.ensure_models("custom", str(tmp_path))
    assert calls == []


def test_offline_rejects_empty_and_directory_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "0")
    monkeypatch.setenv("WHISPER_MODEL_DIR", str(tmp_path))
    _cache(tmp_path)
    (tmp_path / "vocab_en.txt").write_bytes(b"")
    with pytest.raises(RuntimeError, match="vocabulary"):
        md._ensure_whisper_artifacts("rk.whisper")
    _write(tmp_path / "vocab_en.txt")
    (tmp_path / "mel_80_filters.txt").unlink()
    (tmp_path / "mel_80_filters.txt").mkdir()
    with pytest.raises(RuntimeError, match="mel filters"):
        md._ensure_whisper_artifacts("rk.whisper")


def test_offline_rejects_unknown_decoder_kind_before_filesystem_work(monkeypatch, tmp_path):
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "no")
    monkeypatch.setenv("WHISPER_MODEL_DIR", str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("WHISPER_DECODER_KIND", "mystery")
    monkeypatch.setattr(md.os, "makedirs", lambda *a, **k: pytest.fail("mkdir"))
    with pytest.raises(RuntimeError, match="WHISPER_DECODER_KIND"):
        md._ensure_whisper_artifacts("rk.whisper")


def test_offline_explicit_paths_and_tensorrt_decoder_do_not_require_cpu_onnx(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "off")
    monkeypatch.setenv("WHISPER_MODEL_DIR", str(tmp_path / "unused"))
    env = {
        "WHISPER_ENCODER_PATH": str(tmp_path / "enc.rknn"),
        "WHISPER_DECODER_KIND": "tensorrt",
        "WHISPER_DECODER_PREFILL_PLAN": str(tmp_path / "prefill.plan"),
        "WHISPER_DECODER_STEP_PLAN": str(tmp_path / "step.plan"),
        "WHISPER_VOCAB_DIR": str(tmp_path / "vocab"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _write(Path(env["WHISPER_ENCODER_PATH"]))
    _write(Path(env["WHISPER_DECODER_PREFILL_PLAN"]))
    _write(Path(env["WHISPER_DECODER_STEP_PLAN"]))
    _write(Path(env["WHISPER_VOCAB_DIR"]) / "mel_80_filters.txt")
    _write(Path(env["WHISPER_VOCAB_DIR"]) / "vocab_en.txt")

    md._ensure_whisper_artifacts("rk.whisper")


def test_offline_jetson_requires_cached_plan_but_not_source_onnx(monkeypatch, tmp_path):
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "0")
    monkeypatch.setenv("WHISPER_MODEL_DIR", str(tmp_path))
    _cache(tmp_path, spec="jetson.whisper_trt", variant="base")
    md._ensure_whisper_artifacts("jetson.whisper_trt")
    (tmp_path / md._WHISPER_TRT_PLAN).unlink()
    with pytest.raises(RuntimeError, match="encoder"):
        md._ensure_whisper_artifacts("jetson.whisper_trt")


def test_default_auto_mode_still_downloads_and_builds(monkeypatch, tmp_path):
    monkeypatch.delenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", raising=False)
    monkeypatch.setenv("WHISPER_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("WHISPER_VARIANT", "base")
    monkeypatch.setattr(md.shutil, "which", lambda _: None)

    class _Response:
        headers = {"Content-Length": "6"}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, _size):
            if hasattr(self, "done"): return b""
            self.done = True
            return b"cached"

    monkeypatch.setattr(md.urllib.request, "urlopen", lambda *a, **k: _Response())
    built = []
    monkeypatch.setattr(md, "_build_whisper_trt_engine", lambda *a: built.append(a))
    md._ensure_whisper_artifacts("jetson.whisper_trt")
    assert len(built) == 1
    assert Path(built[0][0]).is_file()
