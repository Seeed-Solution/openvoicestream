from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.core import model_downloader as md


def _write(path: Path, data: bytes = b"cached") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _cache(root: Path) -> Path:
    _write(root / md._SENSEVOICE_TRT_ONNX)
    for name in md._SENSEVOICE_RKNN_SHARED:
        _write(root / name)
    plan = root / "sensevoice.plan"
    _write(plan)
    return plan


@pytest.mark.parametrize("disabled", ["0", "false", "no", "off"])
def test_offline_missing_fails_before_network_mkdir_or_build(
    monkeypatch, tmp_path, disabled
):
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", disabled)
    monkeypatch.setenv("SENSEVOICE_TRT_MODEL_DIR", str(tmp_path / "missing"))
    calls = []
    monkeypatch.setattr(md.os, "makedirs", lambda *a, **k: calls.append("mkdir"))
    monkeypatch.setattr(md.subprocess, "run", lambda *a, **k: calls.append("curl"))
    monkeypatch.setattr(md, "_build_sensevoice_trt_engine", lambda *a: calls.append("build"))

    with pytest.raises(RuntimeError, match="incomplete") as exc:
        md._ensure_sensevoice_trt_artifacts()

    assert calls == []


def test_offline_complete_cached_plan_passes_without_side_effects(monkeypatch, tmp_path):
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "off")
    monkeypatch.setenv("SENSEVOICE_TRT_MODEL_DIR", str(tmp_path))
    _cache(tmp_path)
    calls = []
    monkeypatch.setattr(md.os, "makedirs", lambda *a, **k: calls.append("mkdir"))
    monkeypatch.setattr(md.subprocess, "run", lambda *a, **k: calls.append("curl"))
    monkeypatch.setattr(md, "_build_sensevoice_trt_engine", lambda *a: calls.append("build"))

    md._ensure_sensevoice_trt_artifacts()

    assert calls == []


def test_offline_stale_plan_fails_before_build(monkeypatch, tmp_path):
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "false")
    monkeypatch.setenv("SENSEVOICE_TRT_MODEL_DIR", str(tmp_path))
    plan = _cache(tmp_path)
    spec = md._sensevoice_build_spec(str(tmp_path / md._SENSEVOICE_TRT_ONNX))
    old_spec = {**spec, "argmax": not spec["argmax"]}
    with plan.with_name(plan.name + ".buildinfo.json").open("w", encoding="utf-8") as fh:
        json.dump({"trt": "unknown", "spec": old_spec}, fh)
    calls = []
    monkeypatch.setattr(md, "_build_sensevoice_trt_engine", lambda *a: calls.append("build"))
    monkeypatch.setattr(md.subprocess, "run", lambda *a, **k: calls.append("curl"))

    with pytest.raises(RuntimeError, match="stale"):
        md._ensure_sensevoice_trt_artifacts()

    assert calls == []


def test_offline_valid_bpe_override_does_not_require_default_bpe(monkeypatch, tmp_path):
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "0")
    monkeypatch.setenv("SENSEVOICE_TRT_MODEL_DIR", str(tmp_path))
    plan = _cache(tmp_path)
    (tmp_path / "chn_jpn_yue_eng_ko_spectok.bpe.model").unlink()
    override = tmp_path / "custom" / "voice.bpe.model"
    _write(override)
    monkeypatch.setenv("SENSEVOICE_TRT_BPE", str(override))
    calls = []
    monkeypatch.setattr(md.os, "makedirs", lambda *a, **k: calls.append("mkdir"))
    monkeypatch.setattr(md.subprocess, "run", lambda *a, **k: calls.append("curl"))
    monkeypatch.setattr(md, "_build_sensevoice_trt_engine", lambda *a: calls.append("build"))

    md._ensure_sensevoice_trt_artifacts()

    assert calls == []


@pytest.mark.parametrize("empty", [False, True])
def test_offline_missing_or_empty_bpe_override_fails_before_side_effects(
    monkeypatch, tmp_path, empty
):
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "false")
    monkeypatch.setenv("SENSEVOICE_TRT_MODEL_DIR", str(tmp_path))
    _cache(tmp_path)
    override = tmp_path / "custom" / "voice.bpe.model"
    if empty:
        _write(override, b"")
    monkeypatch.setenv("SENSEVOICE_TRT_BPE", str(override))
    calls = []
    monkeypatch.setattr(md.os, "makedirs", lambda *a, **k: calls.append("mkdir"))
    monkeypatch.setattr(md.subprocess, "run", lambda *a, **k: calls.append("curl"))
    monkeypatch.setattr(md, "_build_sensevoice_trt_engine", lambda *a: calls.append("build"))

    with pytest.raises(RuntimeError, match=str(override)):
        md._ensure_sensevoice_trt_artifacts()

    assert calls == []


@pytest.mark.parametrize("enabled", [None, "true", "1"])
def test_enabled_or_unset_keeps_download_path(monkeypatch, tmp_path, enabled):
    if enabled is None:
        monkeypatch.delenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", raising=False)
    else:
        monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", enabled)
    monkeypatch.setenv("SENSEVOICE_TRT_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(md.shutil, "which", lambda _: "curl")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args[0][0])
        output = Path(args[0][args[0].index("-o") + 1])
        _write(output)

    monkeypatch.setattr(md.subprocess, "run", fake_run)
    monkeypatch.setattr(md, "_build_sensevoice_trt_engine", lambda *a: calls.append("build"))
    monkeypatch.setattr(md, "_sensevoice_build_spec", lambda _: {})
    monkeypatch.setattr(md, "_sensevoice_engine_staleness", lambda *_: None)

    md._ensure_sensevoice_trt_artifacts()

    assert calls[:4] == ["curl", "curl", "curl", "curl"]
