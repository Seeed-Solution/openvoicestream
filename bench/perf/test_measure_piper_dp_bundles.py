import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("measure_piper_dp_bundles.py")
SPEC = importlib.util.spec_from_file_location("measure_piper_dp_bundles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _bundle(tmp_path, names, name="en_US"):
    root = tmp_path / name
    root.mkdir()
    for name in names:
        (root / name).write_bytes(name.encode())
    source = tmp_path / "piper.py"
    source.write_bytes(b"source")
    return root, source


def test_bundle_contracts_are_mode_specific(tmp_path):
    cpu, source = _bundle(tmp_path, ("encoder.onnx", "model.onnx.json", "flow_decoder.rknn"))
    result = MODULE.bundle_files(cpu, source, "cpu")
    assert set(result) == {"encoder.onnx", "model.onnx.json", "flow_decoder.rknn", "piper_source.py"}
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        MODULE.bundle_files(cpu, source, "npu")

    npu, source = _bundle(tmp_path, (
        "manifest.json", "model.onnx.json", "text_encoder.rknn",
        "remainder.onnx", "flow_decoder.rknn",
    ), name="npu")
    result = MODULE.bundle_files(npu, source, "npu")
    assert "text_encoder.rknn" in result
    with pytest.raises(FileNotFoundError, match="encoder.onnx"):
        MODULE.bundle_files(npu, source, "cpu")


def test_resolve_actual_mode_uses_loaded_production_flags():
    assert MODULE.resolve_actual_mode(type("Model", (), {"_frontend_npu": True, "_hybrid": False})()) == "frontend_npu"
    assert MODULE.resolve_actual_mode(type("Model", (), {"_frontend_npu": False, "_hybrid": True})()) == "hybrid"
    assert MODULE.resolve_actual_mode(type("Model", (), {"_frontend_npu": False, "_hybrid": False})()) == "legacy"


def test_parser_defaults_and_opt_ins(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runner", "--model-dir", "bundle", "--output", "out.json"])
    args = MODULE.parse_args()
    assert args.frontend_mode == "npu"
    assert args.save_all_wavs is False
    assert args.seed is None

    monkeypatch.setattr("sys.argv", [
        "runner", "--model-dir", "bundle", "--output", "out.json",
        "--frontend-mode", "cpu", "--save-all-wavs", "--seed", "123",
    ])
    args = MODULE.parse_args()
    assert (args.frontend_mode, args.save_all_wavs, args.seed) == ("cpu", True, 123)
