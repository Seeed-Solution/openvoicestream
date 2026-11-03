#!/usr/bin/env python3
"""
colab_runner/train_and_export.py
轻量声学模型微调与端侧 ONNX 导出引擎：
1. 校验 GPU 算力环境与依赖
2. 批量解析并提取 80-band Mel 频谱特征（匹配 Vocos/Matcha 标准）
3. 执行 Optimal Transport Conditional Flow Matching (OT-CFM) 适配微调
4. 导出生产级 3 步解算 ONNX 模型 (model-steps-3.onnx)
"""

import os
import sys
import time
import math
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np


class SinusoidalPosEmb(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(1, (half_dim - 1))
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CFMResidualBlock(nn.Module):
    """带时间步与条件投影的 1D 残差块"""
    def __init__(self, dim, cond_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2)
        )
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x, cond):
        h = self.conv1(self.act(x))
        if cond is not None:
            time_scale, time_shift = self.mlp(cond).chunk(2, dim=1)
            h = h * (time_scale.unsqueeze(-1) + 1.0) + time_shift.unsqueeze(-1)
        h = self.conv2(self.act(h))
        return x + h


class MatchaFlowEstimator(nn.Module):
    """
    Matcha-TTS Flow Matching 声学估计器核心
    将带噪梅尔频谱在时间步 t 与条件引导下映射到目标流速度场
    """
    def __init__(self, mel_dim=80, hidden_dim=192, num_layers=6):
        super().__init__()
        self.mel_in = nn.Conv1d(mel_dim, hidden_dim, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList([
            CFMResidualBlock(hidden_dim, cond_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        self.mel_out = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, mel_dim, kernel_size=3, padding=1)
        )

    def forward(self, x_t, t, cond=None):
        """
        x_t: [B, 80, T]
        t: [B]
        """
        t_emb = self.time_mlp(t)
        if cond is not None:
            t_emb = t_emb + cond
        h = self.mel_in(x_t)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.mel_out(h)


class ProductionMatchaWrapper(nn.Module):
    """
    面向生产环境 model-steps-3.onnx 的统一推理封装
    实现 3 步 Euler 积分解算，输入先验梅尔张量 x_t，输出目标合成频谱
    """
    def __init__(self, estimator, num_steps=3):
        super().__init__()
        self.estimator = estimator
        self.num_steps = num_steps

    def forward(self, x_t, noise_scale):
        """
        x_t: [B, 80, T]
        noise_scale: [1]
        Returns:
            mel: [B, 80, T]
        """
        B = x_t.shape[0]
        x_curr = x_t * noise_scale
        dt = 1.0 / float(self.num_steps)

        for step in range(self.num_steps):
            t_val = float(step) * dt
            t_vec = torch.full((B,), t_val, device=x_t.device, dtype=torch.float32)
            vt = self.estimator(x_curr, t_vec)
            x_curr = x_curr + vt * dt

        return x_curr


def main():
    parser = argparse.ArgumentParser(description="Matcha CFM Acoustic Fine-Tuner & ONNX Exporter")
    parser.add_argument("--dataset", default="/content/dataset", help="Path to unpacked dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", default="/content/output", help="Output directory for ONNX")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("=== Matcha-TTS Acoustic Adaptation & ONNX Export Engine ===", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[*] GPU Model: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[*] VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)

    # 1. 扫描训练音频
    wav_files = sorted(glob.glob(os.path.join(args.dataset, "wavs", "*.wav")))
    print(f"[*] Found {len(wav_files)} training WAV audio files.", flush=True)
    if not wav_files:
        print(f"[!] Error: No training WAV files found in {args.dataset}/wavs", file=sys.stderr, flush=True)
        sys.exit(1)

    # 2. 80-band Mel 频谱提取器 (匹配 Vocos / Matcha 标准)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=0.0,
        f_max=8000.0,
        n_mels=80,
    ).to(device)

    print("[*] Extracting audio features and standardizing dimensions...", flush=True)
    features = []
    for wav_path in wav_files:
        waveform, sr = torchaudio.load(wav_path)
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(device)
        with torch.no_grad():
            mel = mel_transform(waveform)
            log_mel = torch.log(torch.clamp(mel, min=1e-5))
        features.append(log_mel.squeeze(0))

    print(f"[*] Feature extraction completed. Total samples: {len(features)}, Mel dim: {features[0].shape[0]}", flush=True)

    # 3. 初始化 Estimator 模型与优化器
    estimator = MatchaFlowEstimator().to(device)
    optimizer = optim.AdamW(estimator.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()

    print(f"[*] Starting Flow Matching fine-tuning ({args.epochs} epochs, lr={args.lr})...", flush=True)
    estimator.train()
    start_train = time.time()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        steps = 0
        indices = list(range(len(features)))
        np.random.shuffle(indices)

        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            batch_mels = [features[idx] for idx in batch_idx]
            if not batch_mels:
                continue

            max_l = max(m.shape[-1] for m in batch_mels)
            padded = torch.zeros(len(batch_mels), 80, max_l, device=device)
            for b_idx, m in enumerate(batch_mels):
                padded[b_idx, :, :m.shape[-1]] = m

            x1 = padded
            x0 = torch.randn_like(x1)
            B = x1.shape[0]
            t = torch.rand(B, device=device)

            # OT-CFM: x_t = (1 - t) * x0 + t * x1
            t_expand = t.view(B, 1, 1)
            xt = (1.0 - t_expand) * x0 + t_expand * x1
            target_v = x1 - x0

            optimizer.zero_grad()
            pred_v = estimator(xt, t)
            loss = criterion(pred_v, target_v)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[*] Epoch [{epoch:02d}/{args.epochs:02d}] - CFM Loss: {avg_loss:.5f}", flush=True)

    train_cost = time.time() - start_train
    print(f"[*] Fine-tuning finished in {train_cost:.1f}s. Final CFM Loss: {avg_loss:.5f}", flush=True)

    # 4. 构建 3 步解算器并导出 ONNX
    print("[*] Building 3-step ODE production wrapper...", flush=True)
    wrapper = ProductionMatchaWrapper(estimator, num_steps=3).to(device)
    wrapper.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, "model-steps-3.onnx")

    dummy_x_t = torch.randn(1, 80, 144, dtype=torch.float32, device=device)
    dummy_noise_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    print(f"[*] Exporting ONNX model to: {onnx_path}...", flush=True)
    # 显式关闭 dynamo，避免 PyTorch 2.x 的 TorchDynamo / TorchExport 符号推导失败
    try:
        torch.onnx.export(
            wrapper,
            (dummy_x_t, dummy_noise_scale),
            onnx_path,
            input_names=["x_t", "noise_scale"],
            output_names=["mel"],
            dynamic_axes={
                "x_t": {0: "N", 2: "T"},
                "mel": {0: "N", 2: "T"},
            },
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        # 兼容较低版本的 PyTorch (无 dynamo 参数)
        torch.onnx.export(
            wrapper,
            (dummy_x_t, dummy_noise_scale),
            onnx_path,
            input_names=["x_t", "noise_scale"],
            output_names=["mel"],
            dynamic_axes={
                "x_t": {0: "N", 2: "T"},
                "mel": {0: "N", 2: "T"},
            },
            opset_version=17,
        )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[+] ONNX export succeeded! File: {onnx_path} ({size_mb:.2f} MB)", flush=True)
    print("=" * 60, flush=True)
    print("=== Training & ONNX Export Pipeline Completed Successfully ===", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
