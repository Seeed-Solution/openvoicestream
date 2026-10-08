#!/usr/bin/env python3
"""Deterministic ORT/CPU-conditioner/fused-prefix Piper DP diagnosis.

The fused-prefix path is supplied as an offline ``.npz`` capture.  The runner
never invents tensor names: conditioner metadata comes from the frontend
manifest and the duration probe is selected from an actual ``Ceil`` node (or
an explicit ``--ceil-name``).  Missing graph metadata is a hard error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _value_info(model: onnx.ModelProto, name: str):
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        if value.name == name:
            return value
    raise ValueError(f"tensor metadata is missing for {name!r}")


def _dtype_numpy(type_name: str):
    return {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
    }.get(type_name, np.float32)


def _shape_from_ort(inp, seq_len: int):
    out = []
    for dim in inp.shape:
        if isinstance(dim, int) and dim > 0:
            out.append(dim)
        elif isinstance(dim, str):
            out.append(seq_len)
        else:
            raise ValueError(f"input {inp.name!r} has an unknown symbolic dimension {inp.shape!r}")
    return tuple(out)


def _feeds(session: ort.InferenceSession, seq_len: int, token_count: int,
           noise_scale: float, noise_w: float, length_scale: float,
           token_values=None, boundary_names=()):
    feeds = {}
    for inp in session.get_inputs():
        name = inp.name
        shape = _shape_from_ort(inp, seq_len)
        dtype = _dtype_numpy(inp.type)
        lname = name.lower()
        if name in boundary_names:
            value = np.zeros(shape, dtype=dtype)
            if "mask" in lname:
                value[..., :token_count] = 1
        elif name == "input" or lname.endswith("tokens"):
            shape = (1, seq_len)
            value = np.zeros(shape, dtype=dtype)
            if token_values is None:
                token_values = np.arange(1, token_count + 1, dtype=dtype)
            values = np.asarray(token_values, dtype=dtype)
            if values.ndim != 1 or len(values) != token_count:
                raise ValueError("token_values must be a one-dimensional sequence matching token_count")
            value[0, :token_count] = values
        elif name == "input_lengths" or "length" in lname:
            value = np.asarray([token_count], dtype=dtype)
        elif name == "scales":
            value = np.asarray([noise_scale, length_scale, noise_w], dtype=dtype)
        elif name == "x_mask" or "mask" in lname:
            value = np.zeros(shape, dtype=dtype)
            value[..., :token_count] = 1
        elif name in {"sid", "speaker_id"} or "speaker" in lname:
            value = np.zeros(shape, dtype=dtype)
        else:
            raise ValueError(f"unsupported graph input {name!r}; provide an explicit fixture adapter")
        feeds[name] = value
    return feeds


def _metrics(expected, actual):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        return {"shape_equal": False, "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape)}
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    denom = np.linalg.norm(expected.astype(np.float64))
    return {
        "shape_equal": True,
        "finite": bool(np.isfinite(actual).all()),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff) / denom) if denom else float(np.linalg.norm(diff)),
    }


def _tensor_record(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def _remainder_feeds(session, values, seq_len):
    feeds = {}
    for inp in session.get_inputs():
        if inp.name in values:
            feeds[inp.name] = values[inp.name]
            continue
        lname = inp.name.lower()
        if lname in {"audio_length", "cumulative_durations"}:
            feeds[inp.name] = np.zeros(_shape_from_ort(inp, seq_len), dtype=_dtype_numpy(inp.type))
        elif lname in {"sid", "speaker_id"}:
            feeds[inp.name] = np.zeros(_shape_from_ort(inp, seq_len), dtype=_dtype_numpy(inp.type))
        else:
            raise ValueError(f"remainder input {inp.name!r} has no manifest/source feed")
    return feeds


def _ceil_probe(model: onnx.ModelProto, ceil_name: str | None,
                conditioner_name: str):
    ceils = [n for n in model.graph.node if n.op_type == "Ceil"]
    if ceil_name:
        nodes = [n for n in ceils if n.name == ceil_name or n.output[0] == ceil_name]
    else:
        nodes = ceils
    if len(nodes) != 1:
        names = [n.name or n.output[0] for n in ceils]
        raise ValueError(f"expected exactly one duration Ceil node; found {names}")
    node = nodes[0]
    producer = {out: n for n in model.graph.node for out in n.output}
    seen = set()
    stack = [node.input[0]]
    depends_on_conditioner = False
    while stack:
        value = stack.pop()
        if value in seen:
            continue
        seen.add(value)
        if value == conditioner_name:
            depends_on_conditioner = True
            break
        if value in producer:
            stack.extend(producer[value].input)
    if not depends_on_conditioner:
        raise ValueError(f"duration Ceil {node.name or node.output[0]!r} does not depend on conditioner {conditioner_name!r}")
    return node.name or node.output[0], node.input[0], node.output[0]


def _probe_model(model_path: Path, extra_outputs: list[str], override: str | None = None):
    model = onnx.load(str(model_path))
    model = onnx.shape_inference.infer_shapes(model)
    original_outputs = [x.name for x in model.graph.output]
    if override:
        vi = _value_info(model, override)
        for node in model.graph.node:
            for i, out in enumerate(node.output):
                if out == override:
                    node.output[i] = override + "__original"
            for i, inp in enumerate(node.input):
                if inp == override:
                    node.input[i] = override
        model.graph.input.append(vi)
    names = list(dict.fromkeys(original_outputs + extra_outputs))
    existing = {x.name for x in model.graph.output}
    for name in names:
        if name in existing:
            continue
        model.graph.output.append(_value_info(model, name))
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        onnx.save(model, f.name)
        return ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])


def _npz_capture(path: Path | None, case_id: str, names: list[str]):
    if path is None:
        return None
    data = np.load(path, allow_pickle=False)
    values = {}
    for name in names:
        key = f"{case_id}|{name}"
        if key not in data:
            raise ValueError(f"fused capture is missing NPZ key {key!r}")
        values[name] = np.asarray(data[key])
    return values


def _declared_shape(meta, seq_len):
    shape = []
    for index, dim in enumerate(meta.get("shape", [])):
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        elif isinstance(dim, str):
            # Exported manifests use opaque symbolic names for dynamic batch
            # dimensions; the fixed frontend bucket supplies batch=1 and the
            # trailing sequence dimension.
            shape.append(1 if index == 0 else seq_len)
        else:
            raise ValueError(f"manifest has unsupported symbolic output shape {meta.get('name')!r}: {meta.get('shape')!r}")
    return tuple(shape)


def _capture_rknn(prefix_path: Path, prefix_onnx_path: Path, output_path: Path, manifest: dict,
                  prefix_session, cases, wanted, token_map, hidden_names,
                  seq_len: int, noise_scale: float, noise_w: float,
                  length_scale: float, source_path: Path):
    """Capture prefix outputs using the runtime's RKNNLite API and manifest order."""
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError("--rknn-prefix requires the board rknnlite.api package") from exc
    prefix_meta = manifest.get("prefix", {})
    input_names = [x["name"] if isinstance(x, dict) else x
                   for x in prefix_meta.get("inputs", [])]
    output_meta = prefix_meta.get("outputs", [])
    output_names = [x["name"] if isinstance(x, dict) else x for x in output_meta]
    if not input_names or not output_names:
        raise ValueError("manifest prefix.inputs and prefix.outputs are required for RKNN capture")
    prefix_onnx = onnx.shape_inference.infer_shapes(onnx.load(str(prefix_onnx_path)))
    onnx_values = {v.name: v for v in list(prefix_onnx.graph.output) + list(prefix_onnx.graph.value_info)}
    resolved_meta = []
    for item in output_meta:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("manifest prefix.outputs entries must declare names")
        meta = dict(item)
        value = onnx_values.get(meta["name"])
        if value is None:
            raise ValueError(f"prefix ONNX is missing output metadata for {meta['name']!r}")
        if "shape" not in meta:
            meta["shape"] = [d.dim_value if d.HasField("dim_value") else d.dim_param
                              for d in value.type.tensor_type.shape.dim]
        if "dtype" not in meta:
            meta["dtype"] = {1: "float32", 10: "float16", 6: "int32", 7: "int64"}.get(
                value.type.tensor_type.elem_type, "unknown")
        resolved_meta.append(meta)
    output_meta = resolved_meta
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = {}
    evidence = {"model": {"path": str(prefix_path), "sha256": sha256(prefix_path)},
                "source": {"path": str(source_path), "sha256": sha256(source_path)},
                "inputs": input_names, "outputs": output_names, "cases": []}
    rknn = RKNNLite(verbose=False)
    try:
        if rknn.load_rknn(str(prefix_path)) != 0:
            raise RuntimeError("RKNNLite.load_rknn failed")
        if rknn.init_runtime() != 0:
            raise RuntimeError("RKNNLite.init_runtime failed")
        for case in cases:
            case_id = str(case.get("id"))
            if wanted and case_id not in wanted:
                continue
            token_values = token_map.get(case_id)
            if token_values is None and isinstance(case.get("tokens"), list):
                token_values = case["tokens"]
            raw_count = case.get("whole_tokens", case.get("tokens", 0))
            token_count = len(token_values) if token_values is not None else int(raw_count)
            if token_count <= 0 or token_count > seq_len:
                continue
            feeds = _feeds(prefix_session, seq_len, token_count, noise_scale, noise_w,
                           length_scale, token_values, hidden_names)
            missing = [name for name in input_names if name not in feeds]
            if missing:
                raise ValueError(f"manifest prefix input is not available: {missing}")
            out = rknn.inference(inputs=[feeds[name] for name in input_names])
            if out is None or len(out) != len(output_names):
                raise RuntimeError(f"RKNN prefix output count mismatch for {case_id}")
            case_evidence = {"id": case_id, "tokens": token_count,
                             "token_source": "json" if token_values is not None else "synthetic_arange",
                             "inputs": {name: _tensor_record(feeds[name]) for name in input_names},
                             "outputs": {}}
            for name, meta, array in zip(output_names, output_meta, out):
                array = np.asarray(array)
                expected_shape = _declared_shape(meta, seq_len)
                if tuple(array.shape) != expected_shape:
                    raise ValueError(f"RKNN output {name!r} shape {array.shape} != manifest {expected_shape}")
                declared_dtype = str(meta["dtype"]).lower()
                declared_dtype = {"float": "float32", "double": "float64",
                                  "tensor(float)": "float32", "tensor(float16)": "float16"}.get(
                                      declared_dtype, declared_dtype)
                actual_dtype = str(array.dtype).lower()
                compatible = (declared_dtype == actual_dtype or
                              {declared_dtype, actual_dtype} <= {"float16", "float32"})
                if not compatible:
                    raise ValueError(f"RKNN output {name!r} dtype {actual_dtype} != manifest {declared_dtype}")
                values[f"{case_id}|{name}"] = array
                case_evidence["outputs"][name] = {"shape": list(array.shape),
                                                   "dtype": actual_dtype,
                                                   "declared_dtype": declared_dtype,
                                                   "dtype_compatible": compatible}
            evidence["cases"].append(case_evidence)
    finally:
        release = getattr(rknn, "release", None)
        if release is not None:
            release()
    np.savez(output_path, **values)
    output_path.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")
    return output_path


def diagnose(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prefix_meta = manifest.get("prefix", {})
    prefix_names = [x["name"] if isinstance(x, dict) else x
                    for x in prefix_meta.get("outputs", [])]
    if not prefix_names:
        raise ValueError("manifest prefix.outputs is missing")
    dp_meta = manifest.get("dp_conditioner", {}).get("tensor") or {}
    conditioner_name = dp_meta.get("name")
    if not conditioner_name:
        raise ValueError("manifest dp_conditioner.tensor.name is missing")
    if conditioner_name not in prefix_names:
        raise ValueError("conditioner tensor is not declared as a prefix output")
    remainder_outputs = manifest.get("remainder", {}).get("outputs", [])
    if len(remainder_outputs) != 2:
        raise ValueError("manifest remainder.outputs must contain z and y_mask")
    seq_len = int(manifest.get("bucket", {}).get("input", [0, 0])[1])
    if seq_len <= 0:
        raise ValueError("manifest bucket.input does not declare a positive sequence length")
    source = onnx.shape_inference.infer_shapes(onnx.load(str(args.source_encoder)))
    _value_info(source, conditioner_name)
    ceil_node, ceil_input, ceil_output = _ceil_probe(source, args.ceil_name, conditioner_name)
    hidden_names = [x["name"] for x in manifest.get("dp_conditioner", {}).get("io", {}).get("inputs", [])]
    if len(hidden_names) != 2:
        raise ValueError("manifest dp_conditioner.io.inputs must declare hidden and mask")
    extra = list(dict.fromkeys(prefix_names + hidden_names + [conditioner_name, ceil_input, ceil_output]))
    full_session = _probe_model(args.source_encoder, extra)
    duration_session = _probe_model(
        args.source_encoder, [ceil_input, ceil_output], override=conditioner_name
    )
    prefix_session = ort.InferenceSession(str(args.prefix), providers=["CPUExecutionProvider"])
    conditioner_session = ort.InferenceSession(str(args.conditioner), providers=["CPUExecutionProvider"])
    remainder_session = ort.InferenceSession(str(args.remainder), providers=["CPUExecutionProvider"])
    full_output_names = [x.name for x in full_session.get_outputs()]
    if conditioner_name not in full_output_names:
        raise ValueError(f"source graph probe did not expose conditioner {conditioner_name!r}")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = cases.get("prompts", cases) if isinstance(cases, dict) else cases
    token_map = {}
    if args.tokens_json:
        loaded_tokens = json.loads(args.tokens_json.read_text(encoding="utf-8"))
        if isinstance(loaded_tokens, dict) and isinstance(loaded_tokens.get("rows"), list):
            token_map = {str(row["id"]): row["tokens"] for row in loaded_tokens["rows"]
                         if row.get("status") == "ok" and isinstance(row.get("tokens"), list)}
        else:
            token_map = loaded_tokens.get("tokens", loaded_tokens) if isinstance(loaded_tokens, dict) else {}
    wanted = set(args.case_ids.split(",")) if args.case_ids else None
    if args.rknn_prefix and args.fused_npz:
        raise ValueError("--rknn-prefix and --fused-npz are mutually exclusive")
    fused_npz = args.fused_npz
    if args.rknn_prefix:
        if not args.capture_output:
            raise ValueError("--capture-output is required with --rknn-prefix")
        fused_npz = _capture_rknn(
            args.rknn_prefix, args.prefix, args.capture_output, manifest, prefix_session,
            cases, wanted, token_map, hidden_names, seq_len,
            args.noise_scale, args.noise_w, args.length_scale, args.source_encoder,
        )
    rows = []
    for case in cases:
        case_id = str(case.get("id"))
        if wanted and case_id not in wanted:
            continue
        token_values = token_map.get(case_id)
        if token_values is None and isinstance(case.get("tokens"), list):
            token_values = case["tokens"]
        raw_count = case.get("whole_tokens", case.get("tokens", 0))
        if token_values is None and not args.allow_synthetic:
            raise ValueError(f"case {case_id!r} has no actual token ids; provide --tokens-json")
        token_count = len(token_values) if token_values is not None else int(raw_count)
        if token_count <= 0 or token_count > seq_len:
            continue
        feeds = _feeds(full_session, seq_len, token_count,
                       args.noise_scale, args.noise_w, args.length_scale,
                       token_values, hidden_names)
        full_values = full_session.run(None, feeds)
        full = dict(zip(full_output_names, full_values))
        prefix_inputs = _feeds(prefix_session, seq_len, token_count,
                               args.noise_scale, args.noise_w, args.length_scale,
                               token_values, hidden_names)
        prefix_values = prefix_session.run(None, prefix_inputs)
        prefix_output_names = [x.name for x in prefix_session.get_outputs()]
        prefix = dict(zip(prefix_output_names, prefix_values))
        missing_prefix = [name for name in hidden_names if name not in prefix]
        if missing_prefix:
            raise ValueError(f"prefix capture is missing conditioner inputs {missing_prefix}")
        cond_cpu_inputs = {name: prefix[name] for name in hidden_names}
        cond_cpu = conditioner_session.run(None, cond_cpu_inputs)
        cond_outputs = [x.name for x in conditioner_session.get_outputs()]
        if len(cond_outputs) != 1:
            raise ValueError("conditioner graph must have exactly one output")
        cpu_cond = cond_cpu[0]
        fused = _npz_capture(fused_npz, case_id, prefix_names) if fused_npz else None
        base = dict(feeds)
        base.update(prefix_inputs)
        base.update({name: prefix[name] for name in prefix_names if name in prefix})
        base[conditioner_name] = cpu_cond
        cpu_tail = dict(zip([x.name for x in remainder_session.get_outputs()],
                            remainder_session.run(None, _remainder_feeds(remainder_session, base, seq_len))))
        row = {
            "id": case_id, "tokens": token_count,
            "token_source": "json" if token_values is not None else "synthetic_arange",
            "inputs": {name: _tensor_record(value) for name, value in feeds.items()},
            "conditioner": _metrics(full[conditioner_name], cpu_cond),
            "cpu_tail": {name: _metrics(full[name], cpu_tail[name]) for name in remainder_outputs},
            "duration": {
                "ceil_node": ceil_node, "ceil_input": np.asarray(full[ceil_input]).tolist(),
                "ceil_output": np.asarray(full[ceil_output]).tolist(),
            },
        }
        duration_names = [x.name for x in duration_session.get_outputs()]
        duration_inputs = {x.name for x in duration_session.get_inputs()}
        cpu_duration_feeds = {
            name: feeds[name] for name in duration_inputs
            if name in duration_inputs and name in feeds
        }
        cpu_duration_feeds[conditioner_name] = cpu_cond
        cpu_duration = dict(zip(duration_names, duration_session.run(None, cpu_duration_feeds)))
        row["duration"]["cpu_conditioner"] = {
            "ceil_input": np.asarray(cpu_duration[ceil_input]).tolist(),
            "ceil_output": np.asarray(cpu_duration[ceil_output]).tolist(),
        }
        if fused is not None:
            fused_base = dict(feeds)
            fused_base.update(prefix_inputs)
            fused_base.update({name: fused[name] for name in prefix_names})
            fused_tail = dict(zip([x.name for x in remainder_session.get_outputs()],
                                  remainder_session.run(None, _remainder_feeds(remainder_session, fused_base, seq_len))))
            row["fused_prefix"] = {name: _metrics(full[name], fused[name]) for name in prefix_names}
            row["fused_tail"] = {name: _metrics(full[name], fused_tail[name]) for name in remainder_outputs}
            fused_cond_from_fused_hidden = conditioner_session.run(
                None, {name: fused[name] for name in hidden_names}
            )[0]
            npu5_cpu_base = dict(feeds)
            npu5_cpu_base.update(prefix_inputs)
            npu5_cpu_base.update({name: fused[name] for name in prefix_names})
            npu5_cpu_base[conditioner_name] = fused_cond_from_fused_hidden
            npu5_cpu_tail = dict(zip(
                [x.name for x in remainder_session.get_outputs()],
                remainder_session.run(None, _remainder_feeds(remainder_session, npu5_cpu_base, seq_len)),
            ))
            row["npu5_cpu_conditioner"] = _metrics(full[conditioner_name], fused_cond_from_fused_hidden)
            row["npu5_cpu_conditioner_vs_ort_prefix"] = _metrics(cpu_cond, fused_cond_from_fused_hidden)
            row["npu_conditioner_vs_cpu_conditioner"] = _metrics(
                fused_cond_from_fused_hidden, fused[conditioner_name]
            )
            row["npu5_cpu_tail"] = {
                name: _metrics(full[name], npu5_cpu_tail[name]) for name in remainder_outputs
            }
            source_cpu_base = dict(feeds)
            source_cpu_base.update(prefix_inputs)
            source_cpu_base.update({name: fused[name] for name in prefix_names})
            source_cpu_base[conditioner_name] = full[conditioner_name]
            source_cpu_tail = dict(zip(
                [x.name for x in remainder_session.get_outputs()],
                remainder_session.run(None, _remainder_feeds(remainder_session, source_cpu_base, seq_len)),
            ))
            row["source_cpu_conditioner"] = {"shape_equal": True, "finite": True,
                                             "max_abs": 0.0, "rel_l2": 0.0}
            row["source_cpu_tail"] = {
                name: _metrics(full[name], source_cpu_tail[name]) for name in remainder_outputs
            }
            fused_duration_feeds = {
                name: feeds[name] for name in duration_inputs
                if name in duration_inputs and name in feeds
            }
            fused_duration_feeds[conditioner_name] = fused[conditioner_name]
            fused_duration = dict(zip(duration_names, duration_session.run(None, fused_duration_feeds)))
            row["duration"]["fused_prefix"] = {
                "ceil_input": np.asarray(fused_duration[ceil_input]).tolist(),
                "ceil_output": np.asarray(fused_duration[ceil_output]).tolist(),
            }
            npu5_cpu_duration_feeds = dict(fused_duration_feeds)
            npu5_cpu_duration_feeds[conditioner_name] = fused_cond_from_fused_hidden
            npu5_cpu_duration = dict(zip(
                duration_names, duration_session.run(None, npu5_cpu_duration_feeds)
            ))
            row["duration"]["npu5_cpu_conditioner"] = {
                "ceil_input": np.asarray(npu5_cpu_duration[ceil_input]).tolist(),
                "ceil_output": np.asarray(npu5_cpu_duration[ceil_output]).tolist(),
            }
            source_cpu_duration_feeds = dict(fused_duration_feeds)
            source_cpu_duration_feeds[conditioner_name] = full[conditioner_name]
            source_cpu_duration = dict(zip(
                duration_names, duration_session.run(None, source_cpu_duration_feeds)
            ))
            row["duration"]["source_cpu_conditioner"] = {
                "ceil_input": np.asarray(source_cpu_duration[ceil_input]).tolist(),
                "ceil_output": np.asarray(source_cpu_duration[ceil_output]).tolist(),
            }
        rows.append(row)
    if not rows:
        raise ValueError("no cases selected within the declared bucket")
    return {
        "source": {"path": str(args.source_encoder), "sha256": sha256(args.source_encoder)},
        "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest), "seq_len": seq_len},
        "artifacts": {k: {"path": str(v), "sha256": sha256(v)}
                      for k, v in (("prefix", args.prefix), ("remainder", args.remainder), ("conditioner", args.conditioner))},
        "conditioner": conditioner_name, "prefix_outputs": prefix_names,
        "duration_probe": {"ceil_node": ceil_node, "ceil_input": ceil_input, "ceil_output": ceil_output},
        "noise_scale": args.noise_scale, "noise_w": args.noise_w, "length_scale": args.length_scale,
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-encoder", type=Path, required=True)
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--remainder", type=Path, required=True)
    p.add_argument("--conditioner", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--tokens-json", type=Path,
                   help="JSON object mapping case id to the actual token-id list")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="test-only opt-in for arange token ids when --tokens-json is absent")
    p.add_argument("--case-ids")
    p.add_argument("--fused-npz", type=Path)
    p.add_argument("--rknn-prefix", type=Path,
                   help="board RKNNLite prefix model; outputs are captured to --capture-output")
    p.add_argument("--capture-output", type=Path,
                   help="NPZ path for --rknn-prefix capture (JSON sidecar is also written)")
    p.add_argument("--ceil-name")
    p.add_argument("--noise-scale", type=float, default=0.0)
    p.add_argument("--noise-w", type=float, default=0.0)
    p.add_argument("--length-scale", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(diagnose(args), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
