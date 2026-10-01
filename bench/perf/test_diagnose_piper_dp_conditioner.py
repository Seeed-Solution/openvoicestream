import json
import sys
import types
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from diagnose_piper_dp_conditioner import diagnose


HIDDEN = "/enc_p/encoder/Mul_2_output_0"
MASK = "/enc_p/Cast_1_output_0"
COND = "/dp/proj_output_0"


def _save_models(root: Path):
    hidden = helper.make_tensor_value_info(HIDDEN, TensorProto.FLOAT, [1, 1, 4])
    mask = helper.make_tensor_value_info(MASK, TensorProto.FLOAT, [1, 1, 4])
    cond = helper.make_tensor_value_info(COND, TensorProto.FLOAT, [1, 1, 4])
    raw = helper.make_tensor_value_info("/dp/logw_output_0", TensorProto.FLOAT, [1, 1, 4])
    ceil = helper.make_tensor_value_info("/Ceil_output_0", TensorProto.FLOAT, [1, 1, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 1, 4])
    y_mask = helper.make_tensor_value_info("y_mask", TensorProto.FLOAT, [1, 1, 4])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.25])

    full = helper.make_graph([
        helper.make_node("Add", [HIDDEN, MASK], [COND], name="/dp/proj/add"),
        helper.make_node("Add", [COND, "one"], [raw.name], name="/dp/logw"),
        helper.make_node("Ceil", [raw.name], [ceil.name], name="/dp/Ceil"),
        helper.make_node("Identity", [COND], [z.name], name="/dp/z"),
        helper.make_node("Identity", [MASK], [y_mask.name], name="/dp/y_mask"),
    ], "full", [hidden, mask], [z, y_mask], [one])
    model = helper.make_model(full, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, root / "source.onnx")

    prefix = helper.make_graph([
        helper.make_node("Add", [HIDDEN, MASK], [COND], name="/dp/proj/add"),
    ], "prefix", [hidden, mask], [hidden, mask, cond])
    model = helper.make_model(prefix, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, root / "prefix.onnx")

    conditioner = helper.make_graph([
        helper.make_node("Add", [HIDDEN, MASK], [COND], name="/dp/proj/add"),
    ], "conditioner", [hidden, mask], [cond])
    model = helper.make_model(conditioner, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, root / "conditioner.onnx")

    remainder = helper.make_graph([
        helper.make_node("Identity", [COND], [z.name], name="/dp/z"),
        helper.make_node("Identity", [MASK], [y_mask.name], name="/dp/y_mask"),
    ], "remainder", [hidden, mask, cond], [z, y_mask])
    model = helper.make_model(remainder, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, root / "remainder.onnx")


def _manifest(root: Path):
    data = {
        "bucket": {"input": [1, 4]},
        "prefix": {
            "inputs": [{"name": HIDDEN}, {"name": MASK}],
            "outputs": [
                {"name": HIDDEN, "dtype": "float", "shape": [1, 1, 4]},
                {"name": MASK, "dtype": "float", "shape": [1, 1, 4]},
                {"name": COND, "dtype": "float", "shape": [1, 1, 4]},
            ],
        },
        "remainder": {"outputs": ["z", "y_mask"]},
        "dp_conditioner": {
            "enabled": True,
            "tensor": {"name": COND},
            "io": {"inputs": [{"name": HIDDEN}, {"name": MASK}]},
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(data))
    return path


def test_three_way_diagnosis_records_conditioner_and_ceil(tmp_path):
    _save_models(tmp_path)
    manifest = _manifest(tmp_path)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "case3", "whole_tokens": 3}]))
    fused = tmp_path / "fused.npz"
    hidden = np.zeros((1, 1, 4), np.float32)
    mask = np.ones((1, 1, 4), np.float32)
    cond = np.full((1, 1, 4), 1.25, np.float32)  # deliberate fused error
    np.savez(fused, **{
        f"case3|{HIDDEN}": hidden,
        f"case3|{MASK}": mask,
        f"case3|{COND}": cond,
    })
    Args = type("Args", (), {
        "source_encoder": tmp_path / "source.onnx",
        "prefix": tmp_path / "prefix.onnx",
        "remainder": tmp_path / "remainder.onnx",
        "conditioner": tmp_path / "conditioner.onnx",
        "manifest": manifest,
        "cases": cases,
        "tokens_json": None,
        "allow_synthetic": True,
        "case_ids": None,
        "fused_npz": fused,
        "rknn_prefix": None,
        "capture_output": None,
        "ceil_name": None,
        "noise_scale": 0.0,
        "noise_w": 0.0,
        "length_scale": 1.0,
    })

    result = diagnose(Args)
    row = result["rows"][0]
    assert row["token_source"] == "synthetic_arange"
    assert row["conditioner"]["max_abs"] == 0.0
    assert row["cpu_tail"]["z"]["max_abs"] == 0.0
    assert row["fused_prefix"][COND]["max_abs"] == 1.25
    assert row["fused_tail"]["z"]["max_abs"] == 1.25
    assert row["duration"]["ceil_output"] == [[[2.0, 2.0, 2.0, 2.0]]]
    assert row["duration"]["cpu_conditioner"]["ceil_output"] == row["duration"]["ceil_output"]
    assert row["duration"]["fused_prefix"]["ceil_output"] != row["duration"]["ceil_output"]


def test_actual_token_ids_are_consumed_and_recorded(tmp_path):
    _save_models(tmp_path)
    manifest = _manifest(tmp_path)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "case3", "whole_tokens": 3}]))
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"case3": [17, 23, 41]}))
    Args = type("Args", (), {
        "source_encoder": tmp_path / "source.onnx",
        "prefix": tmp_path / "prefix.onnx",
        "remainder": tmp_path / "remainder.onnx",
        "conditioner": tmp_path / "conditioner.onnx",
        "manifest": manifest,
        "cases": cases,
        "tokens_json": tokens,
        "allow_synthetic": False,
        "case_ids": None,
        "fused_npz": None,
        "rknn_prefix": None,
        "capture_output": None,
        "ceil_name": None,
        "noise_scale": 0.0,
        "noise_w": 0.0,
        "length_scale": 1.0,
    })
    row = diagnose(Args)["rows"][0]
    assert row["token_source"] == "json"
    assert row["tokens"] == 3


def test_missing_ceil_metadata_fails_closed(tmp_path):
    _save_models(tmp_path)
    manifest = _manifest(tmp_path)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "case3", "whole_tokens": 3}]))
    source = onnx.load(tmp_path / "source.onnx")
    nodes = [n for n in source.graph.node if n.op_type != "Ceil"]
    del source.graph.node[:]
    source.graph.node.extend(nodes)
    onnx.save(source, tmp_path / "no_ceil.onnx")
    Args = type("Args", (), {
        "source_encoder": tmp_path / "no_ceil.onnx",
        "prefix": tmp_path / "prefix.onnx",
        "remainder": tmp_path / "remainder.onnx",
        "conditioner": tmp_path / "conditioner.onnx",
        "manifest": manifest,
        "cases": cases,
        "tokens_json": None,
        "allow_synthetic": True,
        "case_ids": None,
        "fused_npz": None,
        "rknn_prefix": None,
        "capture_output": None,
        "ceil_name": None,
        "noise_scale": 0.0,
        "noise_w": 0.0,
        "length_scale": 1.0,
    })
    import pytest
    with pytest.raises(ValueError, match="Ceil"):
        diagnose(Args)


def test_rknn_capture_uses_manifest_order_and_releases(tmp_path, monkeypatch):
    _save_models(tmp_path)
    manifest = _manifest(tmp_path)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "case3", "whole_tokens": 3}]))
    (tmp_path / "prefix.rknn").write_bytes(b"fixture-rknn")
    state = {"released": False, "inputs": None}

    class FakeRKNN:
        def __init__(self, **_kwargs): pass
        def load_rknn(self, path):
            assert path.endswith("prefix.rknn")
            return 0
        def init_runtime(self): return 0
        def inference(self, inputs):
            state["inputs"] = inputs
            return [inputs[0], inputs[1], inputs[0] + inputs[1]]
        def release(self): state["released"] = True

    api = types.ModuleType("rknnlite.api")
    api.RKNNLite = FakeRKNN
    pkg = types.ModuleType("rknnlite")
    pkg.api = api
    monkeypatch.setitem(sys.modules, "rknnlite", pkg)
    monkeypatch.setitem(sys.modules, "rknnlite.api", api)
    Args = type("Args", (), {
        "source_encoder": tmp_path / "source.onnx",
        "prefix": tmp_path / "prefix.onnx",
        "remainder": tmp_path / "remainder.onnx",
        "conditioner": tmp_path / "conditioner.onnx",
        "manifest": manifest,
        "cases": cases,
        "tokens_json": None,
        "allow_synthetic": True,
        "case_ids": None,
        "fused_npz": None,
        "rknn_prefix": tmp_path / "prefix.rknn",
        "capture_output": tmp_path / "capture.npz",
        "ceil_name": None,
        "noise_scale": 0.0,
        "noise_w": 0.0,
        "length_scale": 1.0,
    })
    result = diagnose(Args)
    assert result["rows"][0]["fused_prefix"][COND]["max_abs"] == 0.0
    assert state["released"] is True
    assert (tmp_path / "capture.json").exists()
