#!/usr/bin/env python3
"""
finetune_official_matcha_v3.py
Official Bilingual Base Model Fine-Tuning Pipeline for Matcha-TTS (MiniMax female-shaonv).

Strict adherence to Dev-Flow Specification Section 4:
1. G2P Front-end:
   - Integrates bilingual lexicon (Mandarin) + espeak-ng IPA (English).
   - Mandatory space token (ID: 1) inserted at Chinese-English boundaries.
2. Phoneme-level Monotonic Alignment Search (MAS):
   - Fast Numba JIT dynamic programming on negative log-likelihood.
   - Channel offset calibration to align target standardized Mel x1 [80, T_mel] with mu_x [80, T_text].
   - Strict anomaly gate (< 5% samples with phoneme duration > 120 or = 0 frames).
3. Phase 1 Fine-Tuning (Decoder Velocity Field Adaptation):
   - Freezes TextEncoder & Duration Predictor.
   - Unfreezes Decoder UNet Cross-Attention & ResNet residual blocks (freezes time_mlp).
   - Loss: L_CFM (valid frame masked) + 0.01 * 0.5 * sum(||p - p_base||^2).
   - AdamW lr=5e-5, CosineAnnealingLR to 1e-6, 25 epochs, batch_size=4.
   - Evaluates Val_L1 on 11 holdout validation samples via 3-step Euler integration (noise_scale=0.0).
   - Saves best checkpoint to voices/minimax_shaonv/checkpoints/best_model_phase1.pt.
4. Phase 2 Gating & Decision:
   - Generates 11 validation audios using free Duration Predictor + Vocos.
   - Computes overall voiced F0 median via librosa.pyin(sr=16000, fmin=65, fmax=800).
   - Triggers Phase 2 if F0_val_median < 330 Hz and Val_L1 delta < 0.001 over last 3 epochs.
   - Saves final best checkpoint to voices/minimax_shaonv/checkpoints/best_model.pt.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure pure CPU execution without CUDA OOM side-effects
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numba
import numpy as np
import librosa
import soundfile as sf
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim

# Project imports
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("third_party/rkvoice-stream"))

from tools.voice_pipeline.matcha_official.decoder import Decoder
from tools.voice_pipeline.audio_features import (
    extract_norm_mel,
    denorm_mel,
    vocos_reconstruct,
    MEL_MEAN,
    MEL_SIGMA,
    SAMPLE_RATE,
)
from rkvoice_stream.backends.tts.matcha import (
    RKNNMatchaVocoder,
    parse_tokens_file,
    _ITEM_RE,
    _CJK_RE,
    _WORD_RE,
    _PUNCT_NORMALIZE,
    _PUNCT_DROP,
)


# =====================================================================
# 1. Monotonic Alignment Search (Numba JIT)
# =====================================================================

@numba.jit(nopython=True)
def mas_dp_core(dist_neg: np.ndarray) -> np.ndarray:
    """Numba JIT dynamic programming for Monotonic Alignment Search.

    Args:
        dist_neg: [T_text, T_mel] array representing log P(x1_j | mu_x_i)
                  proportional to -0.5 * ||x1_j - mu_x_i||^2.

    Returns:
        path: [T_text, T_mel] float32 binary alignment matrix.
    """
    T_text, T_mel = dist_neg.shape
    dp = np.full((T_text, T_mel), -1e9, dtype=np.float32)
    backtrack = np.zeros((T_text, T_mel), dtype=np.int32)

    dp[0, 0] = dist_neg[0, 0]
    for j in range(1, T_mel):
        dp[0, j] = dp[0, j - 1] + dist_neg[0, j]
        backtrack[0, j] = 0

    for i in range(1, T_text):
        for j in range(i, T_mel):
            v_stay = dp[i, j - 1]
            v_step = dp[i - 1, j - 1]
            if v_stay >= v_step:
                dp[i, j] = v_stay + dist_neg[i, j]
                backtrack[i, j] = i
            else:
                dp[i, j] = v_step + dist_neg[i, j]
                backtrack[i, j] = i - 1

    path = np.zeros((T_text, T_mel), dtype=np.float32)
    curr_i = T_text - 1
    for j in range(T_mel - 1, -1, -1):
        path[curr_i, j] = 1.0
        curr_i = backtrack[curr_i, j]

    return path


def compute_mas_alignment(
    x1: np.ndarray,
    mu_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Compute optimal monotonic alignment matrix A between target mel x1 and phoneme prior mu_x.

    Performs channel offset calibration to bridge the energy shift between
    teacher recordings and base model priors.

    Args:
        x1: Standardized target Mel spectrogram [80, T_mel].
        mu_x: Phoneme acoustic prior [80, T_text].

    Returns:
        A: Binary alignment matrix [T_text, T_mel].
        mu_y: Aligned prior [80, T_mel] (mu_x @ A).
        durations: Per-token duration in frames [T_text].
        is_anomaly: True if any duration > 120 frames or == 0 frames.
    """
    T_text = mu_x.shape[1]
    T_mel = x1.shape[1]

    # Speech frame masks for channel energy calibration
    speech_mask_x1 = x1.mean(axis=0) > -2.0
    speech_mask_mu = mu_x.mean(axis=0) > -2.0
    if not np.any(speech_mask_x1):
        speech_mask_x1 = np.ones(T_mel, dtype=bool)
    if not np.any(speech_mask_mu):
        speech_mask_mu = np.ones(T_text, dtype=bool)

    channel_offset = np.mean(mu_x[:, speech_mask_mu], axis=1, keepdims=True) - np.mean(
        x1[:, speech_mask_x1], axis=1, keepdims=True
    )
    x1_cal = x1 + channel_offset

    # Negative squared Euclidean distance: -0.5 * ||x1_cal - mu_x||^2
    dist_neg = -0.5 * np.sum((x1_cal[:, None, :] - mu_x[:, :, None]) ** 2, axis=0)
    A = mas_dp_core(dist_neg.astype(np.float32))

    mu_y = mu_x @ A
    durations = A.sum(axis=1)

    has_overlong = np.any(durations > 120.0)
    has_empty = np.any(durations == 0.0)
    is_anomaly = bool(has_overlong or has_empty)

    return A, mu_y, durations, is_anomaly


# =====================================================================
# 2. Bilingual G2P Front-end
# =====================================================================

def init_g2p_backend(
    tokens_path: str = "models/matcha-icefall-zh-en/tokens.txt",
    lexicon_path: str = "models/matcha-icefall-zh-en/lexicon.txt",
    espeak_data: str = "models/matcha-icefall-zh-en/espeak-ng-data",
) -> RKNNMatchaVocoder:
    """Initialize bilingual G2P vocoder backend."""
    vocoder = RKNNMatchaVocoder("", "", lexicon_path, tokens_path, espeak_data)
    vocoder._token_to_id = parse_tokens_file(tokens_path)
    vocoder._lexicon = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocoder._lexicon[parts[0]] = parts[1:]
    return vocoder


def bilingual_g2p(text: str, vocoder: RKNNMatchaVocoder) -> List[int]:
    """Convert text to token IDs adhering to bilingual boundary space rules.

    Mandatory rule: Insert space token (ID: 1) at transitions between
    Chinese and English to prevent phoneme fusion and truncation.
    """
    items: List[Tuple[str, List[int]]] = []
    for raw in _ITEM_RE.findall(text):
        if not raw or raw.isspace():
            continue
        if _CJK_RE.match(raw):
            tok_strs = vocoder._chinese_token_strings(raw)
            if tok_strs:
                tids = [vocoder._token_to_id[s] for s in tok_strs if s in vocoder._token_to_id]
                if tids:
                    # Phonological juncture: if neutral tone particle 'de5' is followed by dental plosive d- initial
                    # insert prosodic space token to prevent consonant fusion / gemination
                    de5_id = vocoder._token_to_id.get("de5")
                    space_id = vocoder._token_to_id.get(" ", 1)
                    id_to_tok = {v: k for k, v in vocoder._token_to_id.items()}
                    exp_tids = []
                    for idx_c, tid in enumerate(tids):
                        exp_tids.append(tid)
                        if idx_c + 1 < len(tids):
                            next_tid = tids[idx_c + 1]
                            if tid == de5_id and id_to_tok.get(next_tid, "").startswith("d"):
                                exp_tids.append(space_id)
                    items.append(("cjk", exp_tids))
            continue
        if _WORD_RE.fullmatch(raw):
            tok_strs = vocoder._phonemize_english(raw)
            if tok_strs:
                tids = [vocoder._token_to_id[s] for s in tok_strs if s in vocoder._token_to_id]
                if tids:
                    items.append(("latin", tids))
            continue
        ch = _PUNCT_NORMALIZE.get(raw, raw)
        if ch in _PUNCT_DROP:
            continue
        if not ch.isalnum() and ch in vocoder._token_to_id:
            items.append(("punct", [vocoder._token_to_id[ch]]))
            continue
        tok_strs = vocoder._phonemize_english(raw)
        if tok_strs:
            tids = [vocoder._token_to_id[s] for s in tok_strs if s in vocoder._token_to_id]
            if tids:
                items.append(("other", tids))

    space_id = vocoder._token_to_id.get(" ", 1)
    de5_id = vocoder._token_to_id.get("de5")
    tokens: List[int] = []
    for idx, (kind, tok_ids) in enumerate(items):
        if not tok_ids:
            continue
        tokens.extend(tok_ids)
        if idx + 1 < len(items):
            next_kind, next_toks = items[idx + 1]
            if next_toks:
                if (kind == "latin" and next_kind in ("latin", "cjk")) or (
                    kind == "cjk" and next_kind == "latin"
                ):
                    tokens.append(space_id)
                elif (
                    kind == "cjk"
                    and next_kind == "cjk"
                    and tok_ids[-1] == de5_id
                    and len(next_toks) > 0
                    and vocoder._token_strings.get(next_toks[0], "").startswith("d")
                ):
                    # Phonological juncture: insert prosodic space when de5 precedes dental plosive d- initial
                    tokens.append(space_id)
    return tokens


# =====================================================================
# 3. ONNX Intermediate Encoder Wrapper
# =====================================================================

class IntermediateEncoder:
    """Wrapper around encoder.onnx to extract mu_x, h_text, and free predicted duration."""

    def __init__(self, encoder_onnx_path: str = "models/matcha-icefall-zh-en/encoder.onnx"):
        raw_model = onnx.load(encoder_onnx_path, load_external_data=False)
        # Expose intermediate mu_x [1, 80, L] and h_text [1, 192, L]
        raw_model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "/encoder/proj_m/Conv_output_0", onnx.TensorProto.FLOAT, ["N", 80, "L"]
            )
        )
        raw_model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "/encoder/encoder/Mul_7_output_0", onnx.TensorProto.FLOAT, ["N", 192, "L"]
            )
        )
        raw_model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "/Exp_output_0", onnx.TensorProto.FLOAT, ["N", 1, "L"]
            )
        )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(
            raw_model.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def run(
        self, tokens: List[int], noise_scale: float = 0.667, length_scale: float = 1.0
    ) -> Dict[str, np.ndarray]:
        t_arr = np.array([tokens], dtype=np.int64)
        x_len = np.array([len(tokens)], dtype=np.int64)
        outs = self.sess.run(
            None,
            {
                "x": t_arr,
                "x_length": x_len,
                "noise_scale": np.array([noise_scale], dtype=np.float32),
                "length_scale": np.array([length_scale], dtype=np.float32),
            },
        )
        # outs[0]: expanded mu [1, 80, T_pred]
        # outs[1]: expanded mask [1, 1, T_pred]
        # outs[2]: mu_x [1, 80, T_text]
        # outs[3]: h_text [1, 192, T_text]
        # outs[4]: w [1, 1, T_text]
        return {
            "expanded_mu": outs[0][0],  # [80, T_pred]
            "expanded_mask": outs[1][0],  # [1, T_pred]
            "mu_x": outs[2][0],  # [80, T_text]
            "h_text": outs[3][0],  # [192, T_text]
            "w": outs[4][0, 0],  # [T_text]
        }


# =====================================================================
# 4. Dataset Item & DataLoader
# =====================================================================

class VoiceSample:

    def __init__(
        self,
        wav_file: str,
        text: str,
        tokens: List[int],
        x1: np.ndarray,  # [80, T_mel]
        mu_x: np.ndarray,  # [80, T_text]
        h_text: np.ndarray,  # [192, T_text]
        A: np.ndarray,  # [T_text, T_mel]
        mu_y: np.ndarray,  # [80, T_mel]
        is_anomaly: bool,
    ):
        self.wav_file = wav_file
        self.text = text
        self.tokens = tokens
        self.x1 = x1
        self.mu_x = mu_x
        self.h_text = h_text
        self.A = A
        self.mu_y = mu_y
        self.is_anomaly = is_anomaly


def prepare_dataset_splits(
    dataset_dir: str,
    vocoder: RKNNMatchaVocoder,
    encoder: IntermediateEncoder,
) -> Tuple[List[VoiceSample], List[VoiceSample], Dict[str, Any]]:
    """Preprocess and align train and val datasets.

    Enforces the anomaly gate: raises RuntimeError if anomalies > 5%.
    """
    train_txt = os.path.join(dataset_dir, "train.txt")
    val_txt = os.path.join(dataset_dir, "val.txt")

    print("[*] Processing train split and running MAS...")
    t0 = time.time()
    train_samples: List[VoiceSample] = []
    train_anomalies = 0

    with open(train_txt, "r", encoding="utf-8") as f:
        train_lines = [line.strip().split("|") for line in f if line.strip()]

    for wav_rel, text in train_lines:
        full_wav = os.path.join(dataset_dir, wav_rel)
        if not os.path.exists(full_wav):
            continue
        x1 = extract_norm_mel(full_wav)
        tokens = bilingual_g2p(text, vocoder)
        enc_out = encoder.run(tokens)
        mu_x = enc_out["mu_x"]
        h_text = enc_out["h_text"]

        A, mu_y, durs, is_anomaly = compute_mas_alignment(x1, mu_x)
        if is_anomaly:
            train_anomalies += 1

        sample = VoiceSample(
            wav_file=wav_rel,
            text=text,
            tokens=tokens,
            x1=x1,
            mu_x=mu_x,
            h_text=h_text,
            A=A,
            mu_y=mu_y,
            is_anomaly=is_anomaly,
        )
        train_samples.append(sample)

    train_anomaly_rate = train_anomalies / max(1, len(train_samples))
    print(
        f"[+] Train split: {len(train_samples)} samples, Anomalies: {train_anomalies} "
        f"({train_anomaly_rate * 100:.2f}%)"
    )

    print("[*] Processing val split and running MAS...")
    val_samples: List[VoiceSample] = []
    val_anomalies = 0

    with open(val_txt, "r", encoding="utf-8") as f:
        val_lines = [line.strip().split("|") for line in f if line.strip()]

    for wav_rel, text in val_lines:
        full_wav = os.path.join(dataset_dir, wav_rel)
        if not os.path.exists(full_wav):
            continue
        x1 = extract_norm_mel(full_wav)
        tokens = bilingual_g2p(text, vocoder)
        enc_out = encoder.run(tokens)
        mu_x = enc_out["mu_x"]
        h_text = enc_out["h_text"]

        A, mu_y, durs, is_anomaly = compute_mas_alignment(x1, mu_x)
        if is_anomaly:
            val_anomalies += 1

        sample = VoiceSample(
            wav_file=wav_rel,
            text=text,
            tokens=tokens,
            x1=x1,
            mu_x=mu_x,
            h_text=h_text,
            A=A,
            mu_y=mu_y,
            is_anomaly=is_anomaly,
        )
        val_samples.append(sample)

    val_anomaly_rate = val_anomalies / max(1, len(val_samples))
    print(
        f"[+] Val split: {len(val_samples)} samples, Anomalies: {val_anomalies} "
        f"({val_anomaly_rate * 100:.2f}%)"
    )

    total_samples = len(train_samples) + len(val_samples)
    total_anomalies = train_anomalies + val_anomalies
    overall_anomaly_rate = total_anomalies / max(1, total_samples)
    print(
        f"[+] Overall dataset MAS preparation finished in {time.time() - t0:.2f}s. "
        f"Total anomalies: {total_anomalies} / {total_samples} ({overall_anomaly_rate * 100:.2f}%)"
    )

    # Anomaly Gate Assertion
    if overall_anomaly_rate > 0.05:
        raise RuntimeError(
            f"MAS anomaly rate {overall_anomaly_rate * 100:.2f}% exceeds strict 5% threshold! "
            f"Check G2P front-end."
        )

    # Filter out anomalous samples from training set
    clean_train_samples = [s for s in train_samples if not s.is_anomaly]
    print(
        f"[+] Filtered training set: {len(clean_train_samples)} clean samples for model updates."
    )

    stats = {
        "train_total": len(train_samples),
        "train_clean": len(clean_train_samples),
        "train_anomalies": train_anomalies,
        "train_anomaly_rate": train_anomaly_rate,
        "val_total": len(val_samples),
        "val_anomalies": val_anomalies,
        "val_anomaly_rate": val_anomaly_rate,
        "overall_anomaly_rate": overall_anomaly_rate,
    }

    return clean_train_samples, val_samples, stats


# =====================================================================
# 5. Model Loading & Freezing
# =====================================================================

def build_and_load_decoder(
    pt_ckpt_path: str = "models/matcha-icefall-zh-en/pytorch_model.bin",
    device: torch.device = torch.device("cpu"),
) -> Tuple[Decoder, Dict[str, torch.Tensor], List[nn.Parameter]]:
    """Build official Decoder, load base model weights, and configure trainable parameters."""
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
    ).to(device)

    # Load base weights
    raw_sd = torch.load(pt_ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in raw_sd:
        raw_sd = raw_sd["state_dict"]
    est_sd = {
        k.replace("model.decoder.estimator.", ""): v
        for k, v in raw_sd.items()
        if k.startswith("model.decoder.estimator.")
    }
    dec.load_state_dict(est_sd, strict=True)
    print(f"[+] Loaded 100% of base model weights into official Decoder ({len(est_sd)} keys).")

    # Freeze strategy:
    # Freeze time_mlp, unfreeze down_blocks, mid_blocks, up_blocks, final_block, final_proj
    trainable_params: List[nn.Parameter] = []
    base_params_clones: Dict[str, torch.Tensor] = {}

    for name, p in dec.named_parameters():
        if "time_mlp" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
            trainable_params.append(p)
            base_params_clones[name] = p.detach().clone()

    n_train = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in dec.parameters())
    print(f"[*] Trainable parameters: {n_train:,} / {n_total:,} ({n_train / n_total * 100:.1f}%)")

    return dec, base_params_clones, trainable_params


# =====================================================================
# 6. Evaluation Protocol (Spec 4.3: 3-step Euler Val_L1)
# =====================================================================

def evaluate_val_l1(
    dec: Decoder,
    val_samples: List[VoiceSample],
    proj_m: nn.Conv1d | None = None,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluate Mel L1 reconstruction error on 11 validation samples via 3-step Euler integration.

    Protocol:
        t in {0.0, 1/3, 2/3}, dt = 1/3, deterministic noise z0 = 0.
        Val_L1 = (1/11) * sum_k [ sum_{t, c} |z_hat_{k, c, t} - x1_{k, c, t}| / (80 * T_k) ]
    """
    dec.eval()
    if proj_m is not None:
        proj_m.eval()

    total_l1 = 0.0
    dt = 1.0 / 3.0
    t_steps = [0.0, 1.0 / 3.0, 2.0 / 3.0]

    with torch.no_grad():
        for sample in val_samples:
            T_val = sample.x1.shape[1]
            T_pad = ((T_val + 3) // 4) * 4

            # Obtain mu_y
            if proj_m is None:
                # Phase 1: use baseline aligned prior
                mu_y = sample.mu_y
            else:
                # Phase 2: compute candidate prior mu_x_cand @ A_val
                h_tensor = torch.from_numpy(sample.h_text).unsqueeze(0).to(device)  # [1, 192, T_text]
                mu_x_cand = proj_m(h_tensor).squeeze(0).cpu().numpy()  # [80, T_text]
                mu_y = mu_x_cand @ sample.A  # [80, T_val]

            padded_mu = torch.zeros(1, 80, T_pad, device=device)
            padded_mu[0, :, :T_val] = torch.from_numpy(mu_y)
            mask = torch.ones(1, 1, T_pad, device=device)

            # Starting state: deterministic zero noise (noise_scale=0)
            x_t = torch.zeros(1, 80, T_pad, device=device)

            # 3-step Euler integration
            for t_val in t_steps:
                t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device)
                v_t = dec(x_t, mask, padded_mu, t_tensor)
                x_t = x_t + dt * v_t

            # Sliced reconstruction
            z_hat = x_t[0, :, :T_val].cpu().numpy()
            x1_target = sample.x1

            sample_l1 = np.mean(np.abs(z_hat - x1_target))
            total_l1 += float(sample_l1)

    mean_val_l1 = total_l1 / max(1, len(val_samples))
    return mean_val_l1


# =====================================================================
# 7. Phase 1 Fine-Tuning Execution
# =====================================================================

def run_phase1_training(
    dec: Decoder,
    base_params_clones: Dict[str, torch.Tensor],
    trainable_params: List[nn.Parameter],
    train_samples: List[VoiceSample],
    val_samples: List[VoiceSample],
    epochs: int = 25,
    batch_size: int = 4,
    lr: float = 5e-5,
    lambda_l2sp: float = 0.01,
    checkpoints_dir: str = "voices/minimax_shaonv/checkpoints",
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, List[float], str]:
    """Execute Phase 1 Decoder velocity field fine-tuning."""
    print("\n" + "=" * 70)
    print("=== Phase 1: Decoder Velocity Field Adaptation (25 Epochs) ===")
    print("=" * 70)

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Initial validation benchmark
    init_val_l1 = evaluate_val_l1(dec, val_samples, device=device)
    print(f"[*] Initial Baseline Val_L1: {init_val_l1:.5f}")

    best_val_l1 = init_val_l1
    best_ckpt_path = os.path.join(checkpoints_dir, "best_model_phase1.pt")
    val_l1_history: List[float] = [init_val_l1]
    loss_history: List[float] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        dec.train()
        total_cfm_loss = 0.0
        total_reg_loss = 0.0
        total_steps = 0

        indices = list(range(len(train_samples)))
        np.random.shuffle(indices)

        for b_start in range(0, len(indices), batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            batch_items = [train_samples[i] for i in b_idx]
            if not batch_items:
                continue

            max_len = max(item.x1.shape[1] for item in batch_items)
            # Ensure length is multiple of 4 for UNet symmetry
            max_len = ((max_len + 3) // 4) * 4
            B = len(batch_items)

            padded_x1 = torch.zeros(B, 80, max_len, device=device)
            padded_mu = torch.zeros(B, 80, max_len, device=device)
            padded_mask = torch.zeros(B, 1, max_len, device=device)

            for b_i, item in enumerate(batch_items):
                l = item.x1.shape[1]
                padded_x1[b_i, :, :l] = torch.from_numpy(item.x1)
                padded_mu[b_i, :, :l] = torch.from_numpy(item.mu_y)
                padded_mask[b_i, 0, :l] = 1.0

            # CFM Flow Matching:
            # z0 ~ N(0, I), target velocity: v_target = x1 - z0
            # xt = (1 - t) * z0 + t * x1
            z0 = torch.randn_like(padded_x1)
            t = torch.rand(B, device=device)
            t_expand = t.view(B, 1, 1)

            xt = (1.0 - t_expand) * z0 + t_expand * padded_x1
            target_v = padded_x1 - z0

            optimizer.zero_grad()
            pred_v = dec(xt, padded_mask, padded_mu, t)

            # CFM Loss normalized by valid mel elements
            valid_elements = padded_mask.sum() * 80.0
            cfm_loss = torch.sum(((pred_v - target_v) ** 2) * padded_mask) / torch.clamp(
                valid_elements, min=1.0
            )

            # L2-SP Regularization: lambda_l2sp * 0.5 * sum(||p - p_base||^2)
            reg_loss = torch.tensor(0.0, device=device)
            for name, p in dec.named_parameters():
                if name in base_params_clones:
                    p_base = base_params_clones[name].to(device)
                    reg_loss = reg_loss + 0.5 * torch.sum((p - p_base) ** 2)
            total_loss = cfm_loss + lambda_l2sp * reg_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_cfm_loss += cfm_loss.item()
            total_reg_loss += reg_loss.item()
            total_steps += 1

        scheduler.step()
        avg_cfm = total_cfm_loss / max(1, total_steps)
        loss_history.append(avg_cfm)

        # Validation evaluation
        val_l1 = evaluate_val_l1(dec, val_samples, device=device)
        val_l1_history.append(val_l1)

        is_best = val_l1 < best_val_l1
        if is_best:
            best_val_l1 = val_l1
            torch.save(
                {
                    "epoch": epoch,
                    "val_l1": best_val_l1,
                    "decoder_state_dict": dec.state_dict(),
                },
                best_ckpt_path,
            )

        marker = " [*BEST*]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Train CFM Loss: {avg_cfm:.5f} - "
            f"Val_L1: {val_l1:.5f} (Best: {best_val_l1:.5f}) - LR: {scheduler.get_last_lr()[0]:.2e}{marker}"
        )

    print(
        f"[+] Phase 1 training finished in {time.time() - start_time:.1f}s. "
        f"Best Val_L1: {best_val_l1:.5f} saved to {best_ckpt_path}"
    )

    return best_val_l1, val_l1_history, best_ckpt_path


# =====================================================================
# 8. Phase 2 Gating & Decision
# =====================================================================

def evaluate_phase2_gating(
    best_ckpt_path: str,
    val_samples: List[VoiceSample],
    encoder: IntermediateEncoder,
    val_l1_history: List[float],
    device: torch.device = torch.device("cpu"),
) -> Tuple[bool, float, float, Dict[str, Any]]:
    """Synthesize 11 validation audios with Phase 1 best model and evaluate F0 gating condition.

    Returns:
        should_trigger_phase2: bool
        f0_median: float
        val_l1_delta_last3: float
        gating_details: dict
    """
    print("\n" + "=" * 70)
    print("=== Phase 2 Quantitative Trigger Gating Evaluation ===")
    print("=" * 70)

    # 1. Load best Phase 1 Decoder
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
    ).to(device)
    ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    dec.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    dec.eval()

    all_voiced_f0: List[float] = []
    dt = 1.0 / 3.0
    t_steps = [0.0, 1.0 / 3.0, 2.0 / 3.0]

    print("[*] Synthesizing 11 holdout validation audios using free Duration Predictor...")
    for idx, sample in enumerate(val_samples, 1):
        # End-to-end free duration synthesis
        enc_out = encoder.run(sample.tokens, noise_scale=0.0, length_scale=1.0)
        expanded_mu = enc_out["expanded_mu"]  # [80, T_pred]
        T_pred = expanded_mu.shape[1]
        T_pad = ((T_pred + 3) // 4) * 4

        padded_mu = torch.zeros(1, 80, T_pad, device=device)
        padded_mu[0, :, :T_pred] = torch.from_numpy(expanded_mu)
        mask = torch.ones(1, 1, T_pad, device=device)

        x_t = torch.zeros(1, 80, T_pad, device=device)
        with torch.no_grad():
            for t_val in t_steps:
                t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device)
                v_t = dec(x_t, mask, padded_mu, t_tensor)
                x_t = x_t + dt * v_t

        z_hat = x_t[0, :, :T_pred].cpu().numpy()
        log_mel = denorm_mel(z_hat)
        wav = vocos_reconstruct(log_mel)

        # Robust F0 extraction with peak normalization (pitch is invariant to linear scale)
        peak = float(np.max(np.abs(wav)))
        wav_eval = wav / (peak + 1e-7) * 0.90 if peak > 1e-6 else wav
        f0, flag, prob = librosa.pyin(wav_eval, sr=SAMPLE_RATE, fmin=65, fmax=800)
        voiced = f0[(prob > 0.5) & (~np.isnan(f0))]
        if len(voiced) == 0:
            print(f"  [WARN] Val sample {idx} has no voiced frames!")
        else:
            all_voiced_f0.extend(voiced.tolist())
            print(
                f"  Sample [{idx:02d}/11]: len={len(wav)/SAMPLE_RATE:.2f}s, "
                f"voiced frames={len(voiced)}, sample median F0={np.median(voiced):.1f} Hz"
            )

    if not all_voiced_f0:
        raise RuntimeError("No voiced frames detected in entire validation set!")

    overall_f0_median = float(np.median(all_voiced_f0))

    # Compute Val_L1 delta across last 3 epochs
    # val_l1_history contains [init, ep1, ep2, ..., ep25]
    if len(val_l1_history) >= 4:
        last_4 = val_l1_history[-4:]
        diffs = [abs(last_4[i] - last_4[i + 1]) for i in range(3)]
        val_l1_delta_last3 = float(max(diffs))
    else:
        val_l1_delta_last3 = 0.0

    print("\n[*] Gating Quantitative Metrics:")
    print(f"  - Overall Validation Voiced F0 Median: {overall_f0_median:.2f} Hz (Threshold: < 330 Hz)")
    print(
        f"  - Validation Val_L1 Max Delta Over Last 3 Epochs: {val_l1_delta_last3:.6f} "
        f"(Threshold: < 0.001)"
    )

    f0_condition = overall_f0_median < 330.0
    l1_condition = val_l1_delta_last3 < 0.001
    should_trigger_phase2 = f0_condition and l1_condition

    print(f"  - F0 Condition (< 330 Hz): {f0_condition}")
    print(f"  - L1 Convergence Condition (delta < 0.001): {l1_condition}")
    print(f"  ==> Phase 2 Trigger Decision: {should_trigger_phase2}")

    details = {
        "overall_f0_median_hz": overall_f0_median,
        "f0_condition_met": f0_condition,
        "val_l1_delta_last3": val_l1_delta_last3,
        "l1_convergence_met": l1_condition,
        "should_trigger_phase2": should_trigger_phase2,
    }

    return should_trigger_phase2, overall_f0_median, val_l1_delta_last3, details


# =====================================================================
# 9. Phase 2 Fine-Tuning Execution (Prior Adaptation)
# =====================================================================

def run_phase2_training(
    dec: Decoder,
    train_samples: List[VoiceSample],
    val_samples: List[VoiceSample],
    phase1_best_ckpt: str,
    base_pt_path: str = "models/matcha-icefall-zh-en/pytorch_model.bin",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-5,
    lambda_l2sp: float = 0.01,
    checkpoints_dir: str = "voices/minimax_shaonv/checkpoints",
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, str]:
    """Execute Phase 2 Prior Fine-Tuning (unfreezing proj_m)."""
    print("\n" + "=" * 70)
    print("=== Phase 2: Prior Adaptation Fine-Tuning (10 Epochs, lr=1e-5) ===")
    print("=" * 70)

    # Reload best Phase 1 decoder weights
    p1_ckpt = torch.load(phase1_best_ckpt, map_location="cpu", weights_only=False)
    dec.load_state_dict(p1_ckpt["decoder_state_dict"], strict=True)

    # Initialize proj_m and base decoder weights from base model
    raw_sd = torch.load(base_pt_path, map_location="cpu", weights_only=False)
    if "state_dict" in raw_sd:
        raw_sd = raw_sd["state_dict"]

    base_weights = {
        k.replace("model.decoder.estimator.", ""): v.detach().clone().to(device)
        for k, v in raw_sd.items()
        if k.startswith("model.decoder.estimator.")
    }

    proj_m = nn.Conv1d(192, 80, kernel_size=1).to(device)
    proj_m.weight.data.copy_(raw_sd["model.encoder.proj_m.weight"])
    proj_m.bias.data.copy_(raw_sd["model.encoder.proj_m.bias"])

    # Clones of base weights for L2-SP
    base_proj_weight = proj_m.weight.detach().clone().to(device)
    base_proj_bias = proj_m.bias.detach().clone().to(device)

    # All trainable parameters: Decoder trainable + proj_m parameters
    phase2_params = [p for p in dec.parameters() if p.requires_grad] + list(proj_m.parameters())
    optimizer = optim.AdamW(phase2_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_l1 = evaluate_val_l1(dec, val_samples, proj_m=proj_m, device=device)
    print(f"[*] Initial Phase 2 Val_L1: {best_val_l1:.5f}")

    best_p2_ckpt_path = os.path.join(checkpoints_dir, "best_model_phase2.pt")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        dec.train()
        proj_m.train()
        total_cfm_loss = 0.0
        total_steps = 0

        indices = list(range(len(train_samples)))
        np.random.shuffle(indices)

        for b_start in range(0, len(indices), batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            batch_items = [train_samples[i] for i in b_idx]
            if not batch_items:
                continue

            max_len = max(item.x1.shape[1] for item in batch_items)
            max_len = ((max_len + 3) // 4) * 4
            B = len(batch_items)

            padded_x1 = torch.zeros(B, 80, max_len, device=device)
            padded_mu = torch.zeros(B, 80, max_len, device=device)
            padded_mask = torch.zeros(B, 1, max_len, device=device)

            for b_i, item in enumerate(batch_items):
                l = item.x1.shape[1]
                padded_x1[b_i, :, :l] = torch.from_numpy(item.x1)
                padded_mask[b_i, 0, :l] = 1.0

                # Recompute mu_x_cand using latest proj_m
                h_tensor = torch.from_numpy(item.h_text).unsqueeze(0).to(device)  # [1, 192, T_text]
                mu_x_cand = proj_m(h_tensor).squeeze(0)  # [80, T_text]

                # Recompute MAS with current candidate mu_x_cand
                A_cand, _, _, _ = compute_mas_alignment(
                    item.x1, mu_x_cand.detach().cpu().numpy()
                )
                A_tensor = torch.from_numpy(A_cand).to(device)  # [T_text, T_mel]
                # Gradient isolation: A is detached, while mu_x_cand retains gradient
                mu_y_cand = torch.mm(mu_x_cand, A_tensor)  # [80, T_mel]
                padded_mu[b_i, :, :l] = mu_y_cand

            z0 = torch.randn_like(padded_x1)
            t = torch.rand(B, device=device)
            t_expand = t.view(B, 1, 1)

            xt = (1.0 - t_expand) * z0 + t_expand * padded_x1
            target_v = padded_x1 - z0

            optimizer.zero_grad()
            pred_v = dec(xt, padded_mask, padded_mu, t)

            valid_elements = padded_mask.sum() * 80.0
            cfm_loss = torch.sum(((pred_v - target_v) ** 2) * padded_mask) / torch.clamp(
                valid_elements, min=1.0
            )

            # L2-SP for both Decoder trainable parameters and proj_m parameters
            l2sp_loss = 0.0
            for name, param in dec.named_parameters():
                if param.requires_grad and name in base_weights:
                    l2sp_loss = l2sp_loss + 0.5 * torch.sum((param - base_weights[name]) ** 2)
            reg_proj = 0.5 * (
                torch.sum((proj_m.weight - base_proj_weight) ** 2)
                + torch.sum((proj_m.bias - base_proj_bias) ** 2)
            )
            total_loss = cfm_loss + lambda_l2sp * (l2sp_loss + reg_proj)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(phase2_params, max_norm=1.0)
            optimizer.step()

            total_cfm_loss += cfm_loss.item()
            total_steps += 1

        scheduler.step()
        avg_cfm = total_cfm_loss / max(1, total_steps)

        val_l1 = evaluate_val_l1(dec, val_samples, proj_m=proj_m, device=device)
        is_best = val_l1 < best_val_l1
        if is_best:
            best_val_l1 = val_l1
            torch.save(
                {
                    "epoch": epoch,
                    "val_l1": best_val_l1,
                    "decoder_state_dict": dec.state_dict(),
                    "proj_m_state_dict": proj_m.state_dict(),
                },
                best_p2_ckpt_path,
            )

        marker = " [*BEST*]" if is_best else ""
        print(
            f"Phase 2 Epoch [{epoch:02d}/{epochs:02d}] - Train CFM Loss: {avg_cfm:.5f} - "
            f"Val_L1: {val_l1:.5f} (Best: {best_val_l1:.5f}) - LR: {scheduler.get_last_lr()[0]:.2e}{marker}"
        )

    print(
        f"[+] Phase 2 finished in {time.time() - start_time:.1f}s. "
        f"Best Val_L1: {best_val_l1:.5f} saved to {best_p2_ckpt_path}"
    )

    return best_val_l1, best_p2_ckpt_path


# =====================================================================
# 10. Main Pipeline Orchestrator
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Official Matcha-TTS Fine-Tuning Pipeline v3 (MiniMax Shaonv)"
    )
    parser.add_argument(
        "--dataset-dir", default="voices/minimax_shaonv", help="Path to minimax_shaonv directory"
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="voices/minimax_shaonv/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Phase 1 training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for Phase 1")
    parser.add_argument(
        "--lambda-l2sp", type=float, default=0.01, help="L2-SP regularization coefficient"
    )
    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(8)

    print("=" * 75)
    print("=== Official Matcha-TTS Bilingual Voice Clone Engine v3 ===")
    print("=" * 75)
    print(f"[*] Compute Device: {device} (Thread count: {torch.get_num_threads()})")

    # 1. Initialize G2P & Encoder
    vocoder = init_g2p_backend()
    encoder = IntermediateEncoder()

    # 2. Prepare Dataset Splits & Run Phoneme MAS
    clean_train_samples, val_samples, mas_stats = prepare_dataset_splits(
        args.dataset_dir, vocoder, encoder
    )

    # 3. Build Decoder & Load Official Base Weights
    dec, base_clones, trainable_params = build_and_load_decoder(device=device)

    # Base proj_m weights for reference and fallback
    base_raw = torch.load("models/matcha-icefall-zh-en/pytorch_model.bin", map_location="cpu", weights_only=False)
    if "state_dict" in base_raw:
        base_raw = base_raw["state_dict"]
    base_proj_weight = base_raw["model.encoder.proj_m.weight"].clone()
    base_proj_bias = base_raw["model.encoder.proj_m.bias"].clone()

    # 4. Phase 1 Fine-Tuning
    p1_best_val_l1, val_l1_hist, p1_ckpt_path = run_phase1_training(
        dec=dec,
        base_params_clones=base_clones,
        trainable_params=trainable_params,
        train_samples=clean_train_samples,
        val_samples=val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_l2sp=args.lambda_l2sp,
        checkpoints_dir=args.checkpoints_dir,
        device=device,
    )

    # 5. Phase 2 Gating Evaluation
    should_trigger_p2, f0_median, val_l1_delta, gating_details = evaluate_phase2_gating(
        best_ckpt_path=p1_ckpt_path,
        val_samples=val_samples,
        encoder=encoder,
        val_l1_history=val_l1_hist,
        device=device,
    )

    # 6. Optional Phase 2 Training
    p2_best_val_l1 = 999.0
    p2_ckpt_path = ""
    if should_trigger_p2:
        p2_best_val_l1, p2_ckpt_path = run_phase2_training(
            dec=dec,
            train_samples=clean_train_samples,
            val_samples=val_samples,
            phase1_best_ckpt=p1_ckpt_path,
            epochs=10,
            batch_size=args.batch_size,
            lr=1e-5,
            lambda_l2sp=args.lambda_l2sp,
            checkpoints_dir=args.checkpoints_dir,
            device=device,
        )

    # 7. Final Model Selection across Phase 1 and Phase 2
    final_best_path = os.path.join(args.checkpoints_dir, "best_model.pt")
    if should_trigger_p2 and p2_best_val_l1 < p1_best_val_l1:
        winning_phase = "Phase 2"
        winning_val_l1 = p2_best_val_l1
        winning_ckpt = torch.load(p2_ckpt_path, map_location="cpu", weights_only=False)
    else:
        winning_phase = "Phase 1"
        winning_val_l1 = p1_best_val_l1
        winning_ckpt = torch.load(p1_ckpt_path, map_location="cpu", weights_only=False)

    # Construct unified state_dict with model.decoder.estimator. prefixes for downstream export
    dec_sd = winning_ckpt["decoder_state_dict"]
    full_sd = {f"model.decoder.estimator.{k}": v for k, v in dec_sd.items()}
    if "proj_m_state_dict" in winning_ckpt and winning_ckpt["proj_m_state_dict"] is not None:
        proj_sd = winning_ckpt["proj_m_state_dict"]
    else:
        # Retain baseline proj_m weights so proj_m_state_dict is never None
        proj_sd = {
            "weight": base_proj_weight.cpu().clone(),
            "bias": base_proj_bias.cpu().clone(),
        }
    full_sd["model.encoder.proj_m.weight"] = proj_sd["weight"]
    full_sd["model.encoder.proj_m.bias"] = proj_sd["bias"]

    final_payload = {
        "winning_phase": winning_phase,
        "epoch": winning_ckpt["epoch"],
        "val_l1": winning_val_l1,
        "decoder_state_dict": dec_sd,
        "proj_m_state_dict": proj_sd,
        "state_dict": full_sd,
    }
    torch.save(final_payload, final_best_path)
    print(f"\n[+] Saved final winning model ({winning_phase}, Val_L1: {winning_val_l1:.5f}) to {final_best_path}")

    # 8. Save Comprehensive Summary JSON
    summary = {
        "winning_phase": winning_phase,
        "winning_epoch": winning_ckpt["epoch"],
        "winning_val_l1": winning_val_l1,
        "phase1_best_val_l1": p1_best_val_l1,
        "phase2_triggered": should_trigger_p2,
        "phase2_best_val_l1": p2_best_val_l1 if should_trigger_p2 else None,
        "gating_details": gating_details,
        "mas_stats": mas_stats,
        "phase1_val_l1_history": val_l1_hist,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = os.path.join(args.checkpoints_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved training metrics and evidence to {summary_path}")

    # 9. Synthesize Preview Audio Clips
    previews_dir = os.path.join(args.checkpoints_dir, "previews")
    os.makedirs(previews_dir, exist_ok=True)
    print("\n[*] Synthesizing preview audio clips with Vocos...")
    dec.load_state_dict(dec_sd, strict=True)
    dec.eval()

    test_utterances = [
        ("preview_bilingual.wav", "您好！我是您的智能语音助手。Welcome to Seeed Studio! 今天有什么可以帮您的？"),
        ("preview_english.wav", "This neural voice engine runs entirely on local edge devices with ultra low latency."),
        ("preview_chinese.wav", "我们在边缘计算设备上实现了高保真音色复刻与流畅自然的语音合成。"),
    ]

    proj_m_preview = None
    if winning_ckpt.get("proj_m_state_dict") is not None:
        proj_m_preview = nn.Conv1d(192, 80, kernel_size=1).to(device)
        proj_m_preview.load_state_dict(winning_ckpt["proj_m_state_dict"], strict=True)
        proj_m_preview.eval()

    with torch.no_grad():
        for fname, text in test_utterances:
            toks = bilingual_g2p(text, vocoder)
            enc_out = encoder.run(toks, noise_scale=0.0, length_scale=1.0)
            if proj_m_preview is not None:
                h_tensor = torch.from_numpy(enc_out["h_text"]).unsqueeze(0).to(device)
                mu_x_p = proj_m_preview(h_tensor).squeeze(0)  # [80, T_text]
                w_ceil = torch.ceil(torch.from_numpy(enc_out["w"]).clamp(min=1.0)).long().to(device)
                exp_mu = torch.repeat_interleave(mu_x_p, w_ceil, dim=-1).detach().cpu().numpy()
            else:
                exp_mu = enc_out["expanded_mu"]

            T_p = exp_mu.shape[1]
            T_pad = ((T_p + 3) // 4) * 4

            pad_mu = torch.zeros(1, 80, T_pad, device=device)
            pad_mu[0, :, :T_p] = torch.from_numpy(exp_mu)
            mask = torch.ones(1, 1, T_pad, device=device)

            x_t = torch.zeros(1, 80, T_pad, device=device)
            dt = 1.0 / 3.0
            for t_val in [0.0, 1.0 / 3.0, 2.0 / 3.0]:
                t_t = torch.tensor([t_val], dtype=torch.float32, device=device)
                v_t = dec(x_t, mask, pad_mu, t_t)
                x_t = x_t + dt * v_t

            z_out = x_t[0, :, :T_p].cpu().numpy()
            l_mel = denorm_mel(z_out)
            audio = vocos_reconstruct(l_mel)

            # Peak normalize for clear playback
            peak = float(np.max(np.abs(audio)))
            if peak > 1e-6:
                audio = audio / peak * 0.90

            out_wav = os.path.join(previews_dir, fname)
            sf.write(out_wav, audio, SAMPLE_RATE)
            print(f"  [+] Generated {out_wav} ({len(audio)/SAMPLE_RATE:.2f}s, peak: {np.max(np.abs(audio)):.4f})")

    print("\n" + "=" * 75)
    print("=== Training Pipeline Completed Successfully! ===")
    print("=" * 75)


if __name__ == "__main__":
    main()
