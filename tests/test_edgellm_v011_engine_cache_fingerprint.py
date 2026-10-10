"""CPU contract tests for the v0.11 producer-bound engine-cache patch."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/edgellm-v011-engine-cache-producer-fingerprint.patch"


def _source_root() -> Path:
    override = os.environ.get("EDGELLM_V011_SOURCE_ROOT")
    root = Path(override).expanduser().resolve() if override else Path(
        "/home/harvest/edgellm-v011-audit-20261002/upstream")
    required = root / "experimental/server/runtime/engine_build.py"
    if override and not required.is_file():
        raise RuntimeError(
            f"EDGELLM_V011_SOURCE_ROOT lacks required SDK file: {required}")
    return root


UPSTREAM = _source_root()


def _module(tmp_path: Path):
    if not (UPSTREAM / "experimental/server/runtime/engine_build.py").is_file():
        pytest.skip("fixed upstream audit tree not present")
    sdk = tmp_path / "sdk"
    for relative in (
            "experimental/server/runtime/engine_build.py",
            "experimental/server/runtime/engine_layout.py",
            "experimental/builder/core/builder.py",
            "experimental/builder/core/artifacts/runtime_artifacts.py"):
        target = sdk / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(UPSTREAM / relative, target)
    checked = subprocess.run(
        ["git", "apply", "--check", str(PATCH)], cwd=sdk,
        capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    applied = subprocess.run(
        ["git", "apply", str(PATCH)], cwd=sdk,
        capture_output=True, text=True)
    assert applied.returncode == 0, applied.stderr

    experimental = types.ModuleType("experimental")
    experimental.__path__ = [str(sdk / "experimental")]
    server = types.ModuleType("experimental.server")
    server.__path__ = [str(sdk / "experimental/server")]
    runtime = types.ModuleType("experimental.server.runtime")
    runtime.__path__ = [str(sdk / "experimental/server/runtime")]
    sys.modules.update({"experimental": experimental,
                        "experimental.server": server,
                        "experimental.server.runtime": runtime})
    spec = importlib.util.spec_from_file_location(
        "experimental.server.runtime.engine_build",
        sdk / "experimental/server/runtime/engine_build.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _builder_stubs(monkeypatch):
    class Component:
        LLM = "llm"
        DLLM = "dllm"
        TALKER = "talker"
        CODE_PREDICTOR = "code_predictor"
        CODE2WAV = "code2wav"

    class Spec:
        def output_path(self, root, role=None, tp_size=None):
            return str(Path(root) / "llm.engine")

        def config_path(self, root, role=None, tp_size=None):
            return str(Path(root) / "config.json")

    contracts = types.ModuleType("experimental.builder.core.contracts")
    contracts.Component = Component
    contracts.SpecRole = types.SimpleNamespace(BASE="base", DRAFT="draft")
    contracts.resolve_components = lambda *args, **kwargs: (Component.LLM,)
    contracts.component_spec = lambda component: Spec()
    bundle = types.ModuleType("experimental.builder.core.bundle")

    class BundleConfig:
        root_model_type = "stub"
        components = (Component.LLM,)

        @classmethod
        def from_pretrained(cls, model_dir):
            return cls()

    bundle.BundleConfig = BundleConfig
    core = types.ModuleType("experimental.builder.core")
    core.contracts = contracts
    monkeypatch.setitem(sys.modules, "experimental.builder.core", core)
    monkeypatch.setitem(sys.modules, "experimental.builder.core.contracts",
                        contracts)
    monkeypatch.setitem(sys.modules, "experimental.builder.core.bundle", bundle)


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"stub"}', encoding="utf-8")
    return path


def test_same_path_binary_change_changes_key_and_ready_requires_receipt(
        tmp_path, monkeypatch):
    module = _module(tmp_path)
    plugin = tmp_path / "plugin.so"
    plugin.write_bytes(b"one")
    options = module.BuildOptions(plugin_path=str(plugin), max_batch_size=2,
                                  max_input_len=8192,
                                  max_kv_cache_capacity=8192)
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    first = module.bundle_cache_path(str(checkpoint), str(tmp_path / "cache"), options)
    producer = module._producer_fingerprint(options)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "llm.engine").write_bytes(b"engine")
    (bundle / "config.json").write_text(json.dumps({
        "engine_role": "llm", "builder_config": {
            "max_input_len": 8192, "max_kv_cache_capacity": 8192,
            "max_batch_size": 2}, "checkpoint_identity": "x"}), encoding="utf-8")
    _builder_stubs(monkeypatch)
    assert module._is_ready(str(checkpoint), str(bundle), options, producer) is False
    module._write_producer_receipt(str(bundle), options, producer)
    assert module._is_ready(str(checkpoint), str(bundle), options, producer) is True
    receipt_path = bundle / module._PRODUCER_RECEIPT
    for malformed in ([], None, "receipt"):
        receipt_path.write_text(json.dumps(malformed), encoding="utf-8")
        assert module._is_ready(str(checkpoint), str(bundle), options,
                                 producer) is False
    module._write_producer_receipt(str(bundle), options, producer)
    plugin.write_bytes(b"two")
    assert module.bundle_cache_path(str(checkpoint), str(tmp_path / "cache"), options) != first
    assert module._is_ready(str(checkpoint), str(bundle), options) is False


def test_profile_and_receipt_producer_abort_during_build_and_cache_hits(
        tmp_path, monkeypatch):
    module = _module(tmp_path)
    _builder_stubs(monkeypatch)
    plugin = tmp_path / "plugin.so"
    plugin.write_bytes(b"stable")
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    cache = tmp_path / "cache"
    options = module.BuildOptions(plugin_path=str(plugin), max_batch_size=2,
                                  max_input_len=8192,
                                  max_kv_cache_capacity=8192)
    calls = []
    builder = types.ModuleType("experimental.builder.cli")

    def build(argv):
        calls.append(argv)
        out = Path(argv[argv.index("--engine-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "llm.engine").write_bytes(b"engine")
        (out / "config.json").write_text(json.dumps({
            "engine_role": "llm", "builder_config": {
                "max_input_len": 8192, "max_kv_cache_capacity": 8192,
                "max_batch_size": 2}, "checkpoint_identity": "x"}), encoding="utf-8")

    builder.main = build
    monkeypatch.setitem(sys.modules, "experimental.builder.cli", builder)
    monkeypatch.setattr(module, "resolve_model_dir", lambda model, cache="": str(checkpoint))
    prepared = module.prepare_model(str(checkpoint), str(cache), options)
    assert prepared.built is True
    assert len(calls) == 1
    prepared_again = module.prepare_model(str(checkpoint), str(cache), options)
    assert prepared_again.built is False
    assert len(calls) == 1

    plugin.write_bytes(b"changed")
    assert module.bundle_cache_path(str(checkpoint), str(cache), options) != prepared.bundle_dir

    plugin.write_bytes(b"stable-again")
    def changing_build(argv):
        build(argv)
        plugin.write_bytes(b"changed-during-build")
    builder.main = changing_build
    with pytest.raises(RuntimeError, match="changed during engine build"):
        module.prepare_model(str(checkpoint), str(cache), options)


def test_missing_plugin_fails_before_build_and_b1_b2_keys_differ(tmp_path):
    module = _module(tmp_path)
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    missing = tmp_path / "missing.so"
    with pytest.raises(FileNotFoundError, match="plugin library"):
        module.bundle_cache_path(str(checkpoint), str(tmp_path / "cache"),
                                 module.BuildOptions(plugin_path=str(missing)))
    plugin = tmp_path / "plugin.so"
    plugin.write_bytes(b"same")
    b1 = module.BuildOptions(plugin_path=str(plugin), max_input_len=8192,
                             max_kv_cache_capacity=8192, max_batch_size=1)
    b2 = module.BuildOptions(plugin_path=str(plugin), max_input_len=8192,
                             max_kv_cache_capacity=8192, max_batch_size=2)
    assert module.bundle_cache_path(str(checkpoint), str(tmp_path / "cache"), b1) != \
        module.bundle_cache_path(str(checkpoint), str(tmp_path / "cache"), b2)


def test_bad_source_root_is_a_collection_error(tmp_path):
    env = dict(os.environ)
    env["EDGELLM_V011_SOURCE_ROOT"] = str(tmp_path / "missing-sdk")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", __file__],
        env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "EDGELLM_V011_SOURCE_ROOT lacks required SDK file" in (
        result.stdout + result.stderr)
