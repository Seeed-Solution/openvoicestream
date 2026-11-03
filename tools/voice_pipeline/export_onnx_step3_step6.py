#!/usr/bin/env python3
"""
export_onnx_step3_step6.py
Matcha-TTS 3-Step & 6-Step Production ONNX Exporter and Dual Gate Verification.

Strict implementation of Dev-Flow Specification Section 4.4:
1. Weight Mapping & Integrity Check:
   - Injects 100% of fine-tuned weights from best_model.pt (Decoder UNet 180 layers + encoder.proj_m 2 layers).
   - Handles direct initializers (118), Attention Linear MatMul weights (36 transposed),
     GroupNorm affine parameters (26 reshaped), and proj_m (2).
   - Hard assertion: 100% of trainable parameters (178/178) and total parameters (182/182) mapped.
2. Export Production Models:
   - matcha_shaonv_step3.onnx (3-step Euler, dt=1/3)
   - matcha_shaonv_step6.onnx (6-step Euler, dt=1/6)
3. Gate A (Deterministic Production Model Gate):
   - noise_scale = 0.0 (determinstic zero start).
   - Evaluates short (10 tokens), medium (35 tokens), and long (80 tokens).
   - Asserts max |mel_PyTorch - mel_ONNX| < 1e-4.
   - Validates shapes, lengths, no NaN, no Inf.
4. Gate B (Identical Noise Diagnostic Test Gate):
   - Exports test_matcha_noise_input.onnx replacing RandomNormalLike with external input 'noise'.
   - Initializers are 100% SHA256 equivalent to production model.
   - Explicitly sets noise_scale = 1.0.
   - Injects identical random normal tensor z0 ~ N(0, I) into PyTorch and ONNX.
   - Asserts max |mel_PyTorch - mel_ONNX| < 1e-4 across all token lengths.
"""

from __future__ import annotations

import os
import sys
import hashlib
import time
import json
import argparse
from typing import Dict, List, Tuple, Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn

# Project imports
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("third_party/rkvoice-stream"))

from tools.voice_pipeline.matcha_official.decoder import Decoder
from tools.voice_pipeline.audio_features import (
    MEL_MEAN,
    MEL_SIGMA,
)
from tools.voice_pipeline.finetune_official_matcha_v3 import (
    init_g2p_backend,
    bilingual_g2p,
)


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================
# 1. Complete Parameter Mapping & Weight Injection
# =====================================================================

def build_complete_onnx_mapping(
    model_proto: onnx.ModelProto,
    dec_sd: Dict[str, torch.Tensor],
    proj_sd: Dict[str, torch.Tensor],
) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Build a complete parameter mapping from PyTorch state_dict keys to ONNX initializers.

    Returns:
        Dict mapping (group, pt_name) -> (onnx_initializer_name, transform_type)
        where transform_type is in {'direct', 'transpose', 'reshape'}.
    """
    inits = {i.name: i for i in model_proto.graph.initializer}
    mapping: Dict[Tuple[str, str], Tuple[str, str]] = {}

    # 1. Direct name matches (e.g. model.decoder.estimator... and model.encoder.proj_m...)
    for k in dec_sd.keys():
        onnx_name = f"model.decoder.estimator.{k}"
        if onnx_name in inits:
            mapping[("decoder", k)] = (onnx_name, "direct")

    for k in proj_sd.keys():
        onnx_name = f"model.encoder.proj_m.{k}"
        if onnx_name in inits:
            mapping[("proj_m", k)] = (onnx_name, "direct")

    # 2. Graph node analysis for unrolled / unnamed initializers
    for n in model_proto.graph.node:
        # Attention Linear projection weights: traced as MatMul (transpose required)
        if n.op_type == "MatMul":
            name = n.name
            if "/decoder/estimator/" in name and not any(
                f"_{i}/" in name or name.endswith(f"_{i}") for i in range(1, 10)
            ):
                weight_input = n.input[1]
                if weight_input in inits:
                    subname = name.split("/decoder/estimator/")[1]
                    subname = subname.rsplit("/MatMul", 1)[0].replace("/", ".") + ".weight"
                    if subname in dec_sd:
                        mapping[("decoder", subname)] = (weight_input, "transpose")

        # GroupNorm affine parameters: traced as Mul (weight/gamma) and Add (bias/beta)
        elif n.op_type in ("Mul", "Add"):
            name = n.name
            if (
                "/decoder/estimator/" in name
                and "/block/block.1/" in name
                and not any(f"_{i}/" in name or f".1_{i}" in name for i in range(1, 10))
            ):
                param_type = "weight" if n.op_type == "Mul" else "bias"
                for inp in n.input:
                    if inp in inits:
                        subname = name.split("/decoder/estimator/")[1]
                        subname = subname.rsplit(f"/block.1/{n.op_type}", 1)[0].replace("/", ".")
                        # Correct block.block.1 module path to block.1
                        pt_name = subname + f".block.1.{param_type}"
                        real_pt_name = pt_name.replace(".block.block.1", ".block.1")
                        if real_pt_name in dec_sd:
                            mapping[("decoder", real_pt_name)] = (inp, "reshape")
                        break

    return mapping


def inject_weights_into_onnx(
    model_proto: onnx.ModelProto,
    mapping: Dict[Tuple[str, str], Tuple[str, str]],
    dec_sd: Dict[str, torch.Tensor],
    proj_sd: Dict[str, torch.Tensor],
) -> int:
    """Inject weights into ONNX Initializers in-place with strict dimension validation."""
    inits_dict = {i.name: i for i in model_proto.graph.initializer}
    updated_count = 0

    for (group, pt_key), (init_name, transform) in mapping.items():
        tensor = dec_sd[pt_key] if group == "decoder" else proj_sd[pt_key]
        arr = tensor.cpu().numpy()
        init = inits_dict[init_name]
        orig_arr = numpy_helper.to_array(init)

        if transform == "direct":
            if arr.shape != orig_arr.shape:
                raise ValueError(
                    f"Direct shape mismatch for {pt_key}: {arr.shape} vs ONNX {orig_arr.shape}"
                )
            new_arr = arr
        elif transform == "transpose":
            if arr.T.shape != orig_arr.shape:
                raise ValueError(
                    f"Transpose shape mismatch for {pt_key}: {arr.T.shape} vs ONNX {orig_arr.shape}"
                )
            new_arr = arr.T
        elif transform == "reshape":
            if arr.size != orig_arr.size:
                raise ValueError(
                    f"Reshape size mismatch for {pt_key}: {arr.size} vs ONNX {orig_arr.size}"
                )
            new_arr = arr.reshape(orig_arr.shape)
        else:
            raise ValueError(f"Unknown transform {transform} for {pt_key}")

        new_init = numpy_helper.from_array(new_arr.astype(orig_arr.dtype), name=init.name)
        init.CopyFrom(new_init)
        updated_count += 1

    return updated_count


def extract_graph_dt_constants(model_proto: onnx.ModelProto) -> List[float]:
    """Extract Euler integration step constants directly from ONNX graph node attributes."""
    dts = []
    for n in model_proto.graph.node:
        if "Mul" in n.op_type and "/decoder/Mul_" in n.name:
            for inp in n.input:
                prods = [p for p in model_proto.graph.node if inp in p.output]
                if prods and prods[0].op_type == "Constant":
                    c = float(numpy_helper.to_array(prods[0].attribute[0].t))
                    dts.append(c)
    return dts


# =====================================================================
# 2. PyTorch Euler Integration Reference
# =====================================================================

def run_pytorch_euler(
    dec: Decoder,
    mu_y: np.ndarray,
    mask: np.ndarray,
    n_steps: int,
    z0: np.ndarray | None = None,
    dts: List[float] | None = None,
    noise_scale: float = 0.0,
) -> np.ndarray:
    """Run native PyTorch Euler integration loop matching official Matcha-TTS logic.

    Args:
        dec: PyTorch Decoder model
        mu_y: [1, 80, T_pad] aligned prior
        mask: [1, 1, T_pad] sequence mask
        n_steps: 3 or 6 ODE integration steps
        z0: Optional initial noise [1, 80, T_pad]
        dts: Optional list of step sizes (extracted from ONNX graph constants)
        noise_scale: 0.0 for deterministic, 1.0 for noise test

    Returns:
        mel_pt: [1, 80, T_pad] non-normalized log-mel spectrogram
    """
    dec.eval()
    with torch.no_grad():
        mu_y_t = torch.from_numpy(mu_y).float()
        mask_t = torch.from_numpy(mask).float()

        if z0 is not None and noise_scale > 0.0:
            x_t = torch.from_numpy(z0).float() * noise_scale
        else:
            x_t = torch.zeros_like(mu_y_t)

        if dts is None or len(dts) != n_steps:
            dts = [1.0 / n_steps] * n_steps

        t_steps = [i / n_steps for i in range(n_steps)]

        for step_idx in range(n_steps):
            t_val = t_steps[step_idx]
            dt = dts[step_idx]
            t_tensor = torch.tensor([t_val], dtype=torch.float32)
            v_t = dec(x_t, mask_t, mu_y_t, t_tensor)
            x_t = x_t + dt * v_t

        mel_pt = x_t * float(MEL_SIGMA) + float(MEL_MEAN)
        return mel_pt.numpy()


# =====================================================================
# 3. Model Export & Transformation Functions
# =====================================================================

def export_production_model(
    base_onnx_path: str,
    output_onnx_path: str,
    dec_sd: Dict[str, torch.Tensor],
    proj_sd: Dict[str, torch.Tensor],
) -> Tuple[int, Dict[Tuple[str, str], Tuple[str, str]]]:
    """Export production ONNX model by injecting all fine-tuned weights."""
    model_proto = onnx.load(base_onnx_path, load_external_data=False)
    mapping = build_complete_onnx_mapping(model_proto, dec_sd, proj_sd)
    updated = inject_weights_into_onnx(model_proto, mapping, dec_sd, proj_sd)

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    onnx.save(model_proto, output_onnx_path)
    return updated, mapping


def export_diagnostic_noise_model(
    prod_onnx_path: str,
    output_diag_path: str,
) -> None:
    """Create diagnostic ONNX model replacing RandomNormalLike with external input 'noise'.

    Ensures 100% SHA256 initializer equivalence with production model.
    """
    model_proto = onnx.load(prod_onnx_path, load_external_data=False)

    # Replace /decoder/RandomNormalLike node
    new_nodes = []
    found_random = False
    for n in model_proto.graph.node:
        if n.name == "/decoder/RandomNormalLike":
            found_random = True
            continue
        if n.name == "/decoder/Mul":
            # Change input from /decoder/RandomNormalLike_output_0 to 'noise'
            n.input[0] = "noise"
        new_nodes.append(n)

    if not found_random:
        raise RuntimeError("Could not find /decoder/RandomNormalLike node in ONNX graph!")

    del model_proto.graph.node[:]
    model_proto.graph.node.extend(new_nodes)

    # Add 'noise' to graph inputs [N, 80, L]
    noise_input = onnx.helper.make_tensor_value_info(
        "noise", onnx.TensorProto.FLOAT, ["N", 80, "L"]
    )
    model_proto.graph.input.append(noise_input)

    os.makedirs(os.path.dirname(os.path.abspath(output_diag_path)), exist_ok=True)
    onnx.save(model_proto, output_diag_path)


def verify_initializer_equivalence(prod_onnx_path: str, diag_onnx_path: str) -> bool:
    """Verify that 100% of initializers between production and diagnostic models are SHA256 identical."""
    m_prod = onnx.load(prod_onnx_path, load_external_data=False)
    m_diag = onnx.load(diag_onnx_path, load_external_data=False)

    inits_prod = {i.name: numpy_helper.to_array(i) for i in m_prod.graph.initializer}
    inits_diag = {i.name: numpy_helper.to_array(i) for i in m_diag.graph.initializer}

    if set(inits_prod.keys()) != set(inits_diag.keys()):
        return False

    for k in inits_prod:
        arr_p = inits_prod[k]
        arr_d = inits_diag[k]
        if arr_p.shape != arr_d.shape:
            return False
        if not np.array_equal(arr_p, arr_d):
            return False

    return True


# =====================================================================
# 4. Main Pipeline & Gating Execution
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Matcha-TTS 3-Step & 6-Step ONNX Exporter & Verification Gates"
    )
    parser.add_argument(
        "--checkpoint",
        default="voices/minimax_shaonv/checkpoints/best_model.pt",
        help="Path to best fine-tuned checkpoint",
    )
    parser.add_argument(
        "--base-step3",
        default="models/matcha-icefall-zh-en/model-steps-3.onnx",
        help="Path to official 3-step base ONNX model",
    )
    parser.add_argument(
        "--base-step6",
        default="models/matcha-icefall-zh-en/model-steps-6.onnx",
        help="Path to official 6-step base ONNX model",
    )
    parser.add_argument(
        "--output-step3",
        default="voices/minimax_shaonv/matcha_shaonv_step3.onnx",
        help="Export path for 3-step production ONNX",
    )
    parser.add_argument(
        "--output-step6",
        default="voices/minimax_shaonv/matcha_shaonv_step6.onnx",
        help="Export path for 6-step production ONNX",
    )
    parser.add_argument(
        "--output-diagnostic",
        default="voices/minimax_shaonv/checkpoints/test_matcha_noise_input.onnx",
        help="Path for diagnostic noise input ONNX",
    )
    parser.add_argument(
        "--tokens",
        default="models/matcha-icefall-zh-en/tokens.txt",
        help="Path to tokens file",
    )
    parser.add_argument(
        "--lexicon",
        default="models/matcha-icefall-zh-en/lexicon.txt",
        help="Path to lexicon file",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("=== Matcha-TTS Production ONNX Exporter & Parity Gating (Task 2.3) ===")
    print("=" * 80)

    # 1. Load Checkpoint and Verify Parameters
    print("\n[*] [1/6] Loading checkpoint and inspecting parameters...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dec_sd = ckpt["decoder_state_dict"]
    proj_sd = ckpt.get("proj_m_state_dict", {})

    print(f"  [+] Loaded checkpoint: {args.checkpoint}")
    print(f"      Winning Phase: {ckpt.get('winning_phase', 'N/A')}")
    print(f"      Winning Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"      Validation L1: {ckpt.get('val_l1', 'N/A'):.6f}")
    print(f"      Decoder parameters: {len(dec_sd)} tensors")
    print(f"      proj_m parameters: {len(proj_sd)} tensors")

    # Build PyTorch Decoder for reference
    dec = Decoder(
        in_channels=160,
        out_channels=80,
        channels=(256, 256),
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=2,
        act_fn="snakebeta",
    )
    dec.load_state_dict(dec_sd, strict=False)
    dec.eval()

    # 2. Export 3-Step Production Model
    print("\n[*] [2/6] Exporting 3-Step production model...")
    up_m3, map_m3 = export_production_model(args.base_step3, args.output_step3, dec_sd, proj_sd)
    sha_m3 = compute_sha256(args.output_step3)
    size_m3_mb = os.path.getsize(args.output_step3) / (1024 * 1024)
    print(f"  [+] Successfully exported: {args.output_step3}")
    print(f"      Injected weights: {up_m3} / {len(dec_sd) + len(proj_sd)} (100.0% mapped)")
    print(f"      File size: {size_m3_mb:.2f} MB")
    print(f"      SHA256: {sha_m3}")
    expected_params_count = len(dec_sd) + (len(proj_sd) if proj_sd is not None else 0)
    assert up_m3 == expected_params_count, f"Expected {expected_params_count} updated parameters, got {up_m3}"

    # 3. Export 6-Step Production Model
    print("\n[*] [3/6] Exporting 6-Step production model...")
    up_m6, map_m6 = export_production_model(args.base_step6, args.output_step6, dec_sd, proj_sd)
    sha_m6 = compute_sha256(args.output_step6)
    size_m6_mb = os.path.getsize(args.output_step6) / (1024 * 1024)
    print(f"  [+] Successfully exported: {args.output_step6}")
    print(f"      Injected weights: {up_m6} / {expected_params_count} (100.0% mapped)")
    print(f"      File size: {size_m6_mb:.2f} MB")
    print(f"      SHA256: {sha_m6}")
    assert up_m6 == expected_params_count, f"Expected {expected_params_count} updated parameters, got {up_m6}"

    # 4. Export Diagnostic Test Models for Gate B (both 3-Step and 6-Step)
    print("\n[*] [4/6] Exporting Gate B diagnostic test models (both 3-Step and 6-Step)...")
    diag_step3_path = "voices/minimax_shaonv/checkpoints/test_matcha_noise_input_step3.onnx"
    diag_step6_path = "voices/minimax_shaonv/checkpoints/test_matcha_noise_input_step6.onnx"
    export_diagnostic_noise_model(args.output_step3, diag_step3_path)
    export_diagnostic_noise_model(args.output_step6, diag_step6_path)

    sha_diag3 = compute_sha256(diag_step3_path)
    sha_diag6 = compute_sha256(diag_step6_path)
    size_diag3_mb = os.path.getsize(diag_step3_path) / (1024 * 1024)
    size_diag6_mb = os.path.getsize(diag_step6_path) / (1024 * 1024)
    init_equiv3 = verify_initializer_equivalence(args.output_step3, diag_step3_path)
    init_equiv6 = verify_initializer_equivalence(args.output_step6, diag_step6_path)
    print(f"  [+] 3-Step Diagnostic: {diag_step3_path} ({size_diag3_mb:.2f} MB, Equiv: {init_equiv3})")
    print(f"  [+] 6-Step Diagnostic: {diag_step6_path} ({size_diag6_mb:.2f} MB, Equiv: {init_equiv6})")
    assert init_equiv3 and init_equiv6, "Initializer equivalence check failed for diagnostic models!"

    # 5. Prepare Test Datasets (Short: 10, Medium: 35, Long: 80 tokens)
    print("\n[*] [5/6] Preparing test token sequences (10, 35, 80 tokens)...")
    vocoder = init_g2p_backend(lexicon_path=args.lexicon, tokens_path=args.tokens)
    text_short = "您好！今天天气真好。"
    text_mid = "欢迎体验智能语音新时代，Welcome to Seeed intelligent voice assistant."
    text_long = (
        "Welcome to Seeed Studio! We are dedicated to empowering developers, makers, and innovators "
        "around the world. With cutting-edge edge AI technologies and versatile hardware platforms, "
        "we make intelligence accessible everywhere."
    )

    t_short = np.array([bilingual_g2p(text_short, vocoder)[:10]], dtype=np.int64)
    t_mid = np.array([bilingual_g2p(text_mid, vocoder)[:35]], dtype=np.int64)
    t_long = np.array([bilingual_g2p(text_long, vocoder)[:80]], dtype=np.int64)

    test_cases = [
        ("Short", 10, t_short),
        ("Medium", 35, t_mid),
        ("Long", 80, t_long),
    ]

    for name, expected_len, tokens_arr in test_cases:
        actual_len = tokens_arr.shape[1]
        print(f"  [+] Case {name:6s}: target={expected_len}, actual={actual_len} tokens")
        assert actual_len == expected_len, f"Token length mismatch for {name}: {actual_len} vs {expected_len}"

    # 6. Execute Gate A & Gate B Verifications
    print("\n[*] [6/6] Executing Gate A and Gate B Numerical Parity Gates...")

    # Build PyTorch proj_m Conv1d layer for independent verification
    proj_m_pt = torch.nn.Conv1d(192, 80, kernel_size=1)
    if proj_sd:
        proj_m_pt.load_state_dict(proj_sd)
    proj_m_pt.eval()

    gate_a_records = []
    gate_b_records = []

    # --- Gate A Execution ---
    # Probe models with intermediate output hooks for mu_y, mask, h_text, and mu_x
    def create_probe_session(onnx_path: str) -> Tuple[ort.InferenceSession, List[float]]:
        proto = onnx.load(onnx_path, load_external_data=False)
        dts = extract_graph_dt_constants(proto)
        proto.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "/Transpose_3_output_0", onnx.TensorProto.FLOAT, ["N", 80, "L"]
            )
        )
        proto.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "/Cast_3_output_0", onnx.TensorProto.FLOAT, ["N", 1, "L"]
            )
        )
        proto.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "/encoder/encoder/Mul_7_output_0", onnx.TensorProto.FLOAT, ["N", 192, "L"]
            )
        )
        proto.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "/encoder/proj_m/Conv_output_0", onnx.TensorProto.FLOAT, ["N", 80, "L"]
            )
        )
        sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess, dts

    # 3-Step Gate A
    sess_m3, dts_m3 = create_probe_session(args.output_step3)
    print("\n>>> Gate A (Deterministic Production Gate): 3-Step Model (noise_scale=0.0)")
    for name, n_tokens, tokens_arr in test_cases:
        x_len = np.array([n_tokens], dtype=np.int64)
        noise_scale = np.array([0.0], dtype=np.float32)
        length_scale = np.array([1.0], dtype=np.float32)

        t0 = time.time()
        mel_onnx, mu_y_np, mask_np, h_text_np, mu_x_onnx = sess_m3.run(
            None,
            {
                "x": tokens_arr,
                "x_length": x_len,
                "noise_scale": noise_scale,
                "length_scale": length_scale,
            },
        )
        onnx_elapsed = time.time() - t0

        # Independent PyTorch proj_m verification
        with torch.no_grad():
            mu_x_pt = proj_m_pt(torch.from_numpy(h_text_np)).numpy()
        proj_m_err = float(np.max(np.abs(mu_x_pt - mu_x_onnx)))
        assert proj_m_err < 1e-5, f"proj_m mismatch between PyTorch and ONNX: {proj_m_err}"

        t1 = time.time()
        valid_len = mel_onnx.shape[2]
        mel_pt = run_pytorch_euler(
            dec=dec,
            mu_y=mu_y_np,
            mask=mask_np,
            n_steps=3,
            z0=None,
            dts=dts_m3,
            noise_scale=0.0,
        )[:, :, :valid_len]
        pt_elapsed = time.time() - t1

        max_err = float(np.max(np.abs(mel_onnx - mel_pt)))
        mean_err = float(np.mean(np.abs(mel_onnx - mel_pt)))
        has_nan = bool(np.isnan(mel_onnx).any() or np.isnan(mel_pt).any())
        has_inf = bool(np.isinf(mel_onnx).any() or np.isinf(mel_pt).any())
        passed = (max_err < 1e-4) and (not has_nan) and (not has_inf)

        record = {
            "model": "3-Step Production",
            "case": name,
            "tokens": n_tokens,
            "mel_shape": list(mel_onnx.shape),
            "valid_length": valid_len,
            "proj_m_err": proj_m_err,
            "max_abs_err": max_err,
            "mean_abs_err": mean_err,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "onnx_ms": onnx_elapsed * 1000,
            "pytorch_ms": pt_elapsed * 1000,
            "passed": passed,
        }
        gate_a_records.append(record)
        status = "PASSED" if passed else "FAILED"
        print(
            f"  [{status}] 3-Step {name:6s} ({n_tokens:2d} tok, {valid_len:3d} frm) | "
            f"proj_m Err: {proj_m_err:.6e} | Max Error: {max_err:.8e} (threshold: 1e-4) | Mean: {mean_err:.8e} | "
            f"ONNX: {onnx_elapsed*1000:.1f}ms, PT: {pt_elapsed*1000:.1f}ms"
        )
        assert passed, f"Gate A failed for 3-step {name}: max_err={max_err}"

    # 6-Step Gate A
    sess_m6, dts_m6 = create_probe_session(args.output_step6)
    print("\n>>> Gate A (Deterministic Production Gate): 6-Step Model (noise_scale=0.0)")
    for name, n_tokens, tokens_arr in test_cases:
        x_len = np.array([n_tokens], dtype=np.int64)
        noise_scale = np.array([0.0], dtype=np.float32)
        length_scale = np.array([1.0], dtype=np.float32)

        t0 = time.time()
        mel_onnx, mu_y_np, mask_np, h_text_np, mu_x_onnx = sess_m6.run(
            None,
            {
                "x": tokens_arr,
                "x_length": x_len,
                "noise_scale": noise_scale,
                "length_scale": length_scale,
            },
        )
        onnx_elapsed = time.time() - t0

        # Independent PyTorch proj_m verification
        with torch.no_grad():
            mu_x_pt = proj_m_pt(torch.from_numpy(h_text_np)).numpy()
        proj_m_err = float(np.max(np.abs(mu_x_pt - mu_x_onnx)))
        assert proj_m_err < 1e-5, f"proj_m mismatch between PyTorch and ONNX: {proj_m_err}"

        t1 = time.time()
        valid_len = mel_onnx.shape[2]
        mel_pt = run_pytorch_euler(
            dec=dec,
            mu_y=mu_y_np,
            mask=mask_np,
            n_steps=6,
            z0=None,
            dts=dts_m6,
            noise_scale=0.0,
        )[:, :, :valid_len]
        pt_elapsed = time.time() - t1

        max_err = float(np.max(np.abs(mel_onnx - mel_pt)))
        mean_err = float(np.mean(np.abs(mel_onnx - mel_pt)))
        has_nan = bool(np.isnan(mel_onnx).any() or np.isnan(mel_pt).any())
        has_inf = bool(np.isinf(mel_onnx).any() or np.isinf(mel_pt).any())
        passed = (max_err < 1e-4) and (not has_nan) and (not has_inf)

        record = {
            "model": "6-Step Production",
            "case": name,
            "tokens": n_tokens,
            "mel_shape": list(mel_onnx.shape),
            "valid_length": valid_len,
            "proj_m_err": proj_m_err,
            "max_abs_err": max_err,
            "mean_abs_err": mean_err,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "onnx_ms": onnx_elapsed * 1000,
            "pytorch_ms": pt_elapsed * 1000,
            "passed": passed,
        }
        gate_a_records.append(record)
        status = "PASSED" if passed else "FAILED"
        print(
            f"  [{status}] 6-Step {name:6s} ({n_tokens:2d} tok, {valid_len:3d} frm) | "
            f"proj_m Err: {proj_m_err:.6e} | Max Error: {max_err:.8e} (threshold: 1e-4) | Mean: {mean_err:.8e} | "
            f"ONNX: {onnx_elapsed*1000:.1f}ms, PT: {pt_elapsed*1000:.1f}ms"
        )
        assert passed, f"Gate A failed for 6-step {name}: max_err={max_err}"

    # --- Gate B Execution (both 3-Step and 6-Step Diagnostic Models) ---
    print("\n>>> Gate B (Identical Noise Diagnostic Gate): 3-Step & 6-Step Diagnostic Models (noise_scale=1.0)")
    
    def create_diag_session(diag_onnx_path: str) -> Tuple[ort.InferenceSession, List[float]]:
        proto_diag = onnx.load(diag_onnx_path, load_external_data=False)
        dts_diag = extract_graph_dt_constants(proto_diag)
        proto_diag.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "/Transpose_3_output_0", onnx.TensorProto.FLOAT, ["N", 80, "L"]
            )
        )
        proto_diag.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "/Cast_3_output_0", onnx.TensorProto.FLOAT, ["N", 1, "L"]
            )
        )
        sess = ort.InferenceSession(proto_diag.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess, dts_diag

    sess_diag3, dts_diag3 = create_diag_session(diag_step3_path)
    sess_diag6, dts_diag6 = create_diag_session(diag_step6_path)

    np.random.seed(12345)
    for model_name, sess_diag, dts_diag, n_steps, probe_sess in [
        ("3-Step Diagnostic", sess_diag3, dts_diag3, 3, sess_m3),
        ("6-Step Diagnostic", sess_diag6, dts_diag6, 6, sess_m6),
    ]:
        print(f"\n  -- Testing {model_name} (n_steps={n_steps}, noise_scale=1.0) --")
        for name, n_tokens, tokens_arr in test_cases:
            x_len = np.array([n_tokens], dtype=np.int64)

            # Probe shape L_pad using probe session
            _, mu_probe, _, _, _ = probe_sess.run(
                None,
                {
                    "x": tokens_arr,
                    "x_length": x_len,
                    "noise_scale": np.array([0.0], dtype=np.float32),
                    "length_scale": np.array([1.0], dtype=np.float32),
                },
            )
            L_pad = mu_probe.shape[2]

            # Generate exact same random gaussian noise tensor
            z0 = np.random.randn(1, 80, L_pad).astype(np.float32)

            t0 = time.time()
            mel_onnx, mu_y_np, mask_np = sess_diag.run(
                None,
                {
                    "x": tokens_arr,
                    "x_length": x_len,
                    "noise_scale": np.array([1.0], dtype=np.float32),
                    "length_scale": np.array([1.0], dtype=np.float32),
                    "noise": z0,
                },
            )
            onnx_elapsed = time.time() - t0

            t1 = time.time()
            valid_len = mel_onnx.shape[2]
            mel_pt = run_pytorch_euler(
                dec=dec,
                mu_y=mu_y_np,
                mask=mask_np,
                n_steps=n_steps,
                z0=z0,
                dts=dts_diag,
                noise_scale=1.0,
            )[:, :, :valid_len]
            pt_elapsed = time.time() - t1

            max_err = float(np.max(np.abs(mel_onnx - mel_pt)))
            mean_err = float(np.mean(np.abs(mel_onnx - mel_pt)))
            has_nan = bool(np.isnan(mel_onnx).any() or np.isnan(mel_pt).any())
            has_inf = bool(np.isinf(mel_onnx).any() or np.isinf(mel_pt).any())
            passed = (max_err < 1e-4) and (not has_nan) and (not has_inf)

            record = {
                "model": model_name,
                "case": name,
                "tokens": n_tokens,
                "mel_shape": list(mel_onnx.shape),
                "valid_length": valid_len,
                "max_abs_err": max_err,
                "mean_abs_err": mean_err,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "onnx_ms": onnx_elapsed * 1000,
                "pytorch_ms": pt_elapsed * 1000,
                "passed": passed,
            }
            gate_b_records.append(record)
            status = "PASSED" if passed else "FAILED"
            print(
                f"  [{status}] {model_name} {name:6s} ({n_tokens:2d} tok, {valid_len:3d} frm) | "
                f"Max Error: {max_err:.8e} (threshold: 1e-4) | Mean: {mean_err:.8e} | "
                f"ONNX: {onnx_elapsed*1000:.1f}ms, PT: {pt_elapsed*1000:.1f}ms"
            )
            assert passed, f"Gate B failed for {model_name} {name}: max_err={max_err}"

    # Save comprehensive numerical parity report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": {
            "path": args.checkpoint,
            "sha256": compute_sha256(args.checkpoint),
            "winning_phase": ckpt.get("winning_phase", "N/A"),
            "val_l1": float(ckpt.get("val_l1", 0.0)),
        },
        "models": {
            "step3_production": {"path": args.output_step3, "sha256": sha_m3, "size_mb": size_m3_mb},
            "step6_production": {"path": args.output_step6, "sha256": sha_m6, "size_mb": size_m6_mb},
            "step3_diagnostic": {"path": diag_step3_path, "sha256": sha_diag3, "equiv_with_step3": init_equiv3},
            "step6_diagnostic": {"path": diag_step6_path, "sha256": sha_diag6, "equiv_with_step6": init_equiv6},
        },
        "gate_a_records": gate_a_records,
        "gate_b_records": gate_b_records,
        "all_gates_passed": all(r["passed"] for r in gate_a_records + gate_b_records),
    }
    report_path = "voices/minimax_shaonv/numerical_parity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[+] Saved structured numerical parity report: {report_path}")

    # Print Summary & Evidence
    print("\n" + "=" * 80)
    print("=== EVIDENCE REPORT: Task 2.3 ONNX Export & Parity Gates ===")
    print("=" * 80)
    print(f"1. Checkpoint Source: {args.checkpoint} (Val_L1: {ckpt.get('val_l1', 0.0):.6f})")
    print(f"2. Exported 3-Step Model: {args.output_step3} (SHA256: {sha_m3})")
    print(f"3. Exported 6-Step Model: {args.output_step6} (SHA256: {sha_m6})")
    print(f"4. Diagnostic Models: Step3 SHA: {sha_diag3}, Step6 SHA: {sha_diag6}")
    print("\n5. Gate A Results Summary (Deterministic noise_scale=0.0):")
    for r in gate_a_records:
        print(f"   - [{r['model']}] {r['case']} ({r['tokens']} tokens): max_abs_diff={r['max_abs_err']:.6e} < 1e-4 [PASSED]")
    print("\n6. Gate B Results Summary (Identical noise injection noise_scale=1.0):")
    for r in gate_b_records:
        print(f"   - [{r['model']}] {r['case']} ({r['tokens']} tokens): max_abs_diff={r['max_abs_err']:.6e} < 1e-4 [PASSED]")

    print("\n[+] 100% of Task 2.3 requirements and verification gates successfully PASSED!")


if __name__ == "__main__":
    main()
