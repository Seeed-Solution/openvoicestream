#!/usr/bin/env python3
"""
finetune_official_matcha.py
基于 Matcha-TTS 官方原生 1D UNet Decoder 的声音克隆微调引擎

关键技术实现：
1. 双语前端：集成 12.9 万词本地 CMUDict (ARPAbet -> IPA) 与汉字拼音，支持无缝中英混读；
2. 架构无损：采用官方原生 Decoder (包含完整下采样、上采样、跨层 Skip Connection、Mish 与 SnakeBeta 激活)；
3. 精准微调策略：锁定高层时序注意力机制（保留音素清晰度与抗噪能力），专注微调 ResNet 声学残差块与 Mel 投影层；
4. OT-CFM 条件流匹配：在文本先验 mu 引导下，将向量场速度流平滑推向 MiniMax 少女亲和音色分布；
5. 端到端试听验证：采用 3 步 Euler 积分解算，结合 Vocos 神经声码器输出 16kHz 无损波形。
"""

import os
import sys
import re
import csv
import time
import pickle
import argparse
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort
import kaldi_native_fbank as knf
import torch
import torch.nn as nn
import torch.optim as optim
from pypinyin import pinyin, Style

# 导入官方原生 Decoder
from tools.voice_pipeline.matcha_official.decoder import Decoder

# 常量
MEL_MEAN = -2.9521978
MEL_SIGMA = 5.446792
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80

ARPA_TO_IPA = {
    "EH": "ɛ", "AH": "ə", "UW": "u", "IY": "i", "OW": "O",
    "AY": "I", "EY": "A", "AW": "W", "OY": "Y", "ER": "ɜɹ",
    "AA": "ɑ", "AE": "æ", "AO": "ɔ", "IH": "ɪ", "UH": "ʊ",
    "SH": "ʃ", "CH": "ʧ", "JH": "ʤ", "TH": "θ", "DH": "ð",
    "NG": "ŋ", "ZH": "ʒ", "Y": "j", "R": "ɹ", "W": "w",
    "B": "b", "D": "d", "G": "ɡ", "P": "p", "T": "t", "K": "k",
    "M": "m", "N": "n", "F": "f", "V": "v", "S": "s", "Z": "z",
    "L": "l", "HH": "h"
}


def load_cmudict(cmu_path: str):
    with open(cmu_path, "rb") as f:
        return pickle.load(f)


def load_token_mapping(tokens_path: str):
    token_to_id = {}
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 1:
                token_to_id[" "] = int(parts[0])
            elif len(parts) >= 2:
                token_to_id[parts[0]] = int(parts[1])
    return token_to_id


def text_to_tokens_bilingual(text: str, token_to_id: dict, cmu: dict) -> list[int]:
    result = []
    tokens_raw = re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|[^\u4e00-\u9fffA-Za-z\s]", text)
    for seg in tokens_raw:
        if "\u4e00" <= seg <= "\u9fff":
            py = pinyin(seg, style=Style.TONE3, neutral_tone_with_five=True)[0][0]
            if py[-1] not in "12345":
                py += "1"
            if py in token_to_id:
                result.append(token_to_id[py])
        elif seg.isalpha():
            w = seg.upper()
            if w in cmu:
                syls = cmu[w]
                for syl in syls:
                    for ph in syl:
                        p_core = "".join(c for c in ph if not c.isdigit())
                        ipa = ARPA_TO_IPA.get(p_core, "")
                        for ch in ipa:
                            if ch in token_to_id:
                                result.append(token_to_id[ch])
            else:
                for ch in seg.lower():
                    if ch in token_to_id:
                        result.append(token_to_id[ch])
        else:
            p_map = {"，": ",", "。": ".", "！": "!", "？": "?"}
            p = p_map.get(seg, seg)
            if p in token_to_id:
                result.append(token_to_id[p])
    return result


def extract_mel(wav_path: str) -> np.ndarray:
    wav, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        n_mels=N_MELS,
        fmin=0.0,
        fmax=8000.0,
        power=1.0,
        center=True,
        pad_mode="reflect",
    )
    log_mel = np.log(np.maximum(mel, 1e-5))
    norm_mel = (log_mel - MEL_MEAN) / MEL_SIGMA
    return norm_mel.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Official Matcha-TTS Fine-Tuner")
    parser.add_argument("--dataset-dir", default="voices/minimax_shaonv", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output-dir", default="voices/minimax_shaonv", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("=== Official Matcha-TTS Architectural Fine-Tuning Engine ===")
    print("=" * 65)
    print(f"[*] Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. 资源加载
    cmu_path = "/home/harvest/project/wakeword-pipeline/downloads/MeloTTS/melo/text/cmudict_cache.pickle"
    tokens_path = "models/matcha-icefall-zh-en/tokens.txt"
    encoder_path = "/tmp/test_matcha_encoder.onnx"
    vocos_path = "models/matcha-icefall-zh-en/vocos-16khz-univ.onnx"
    pt_ckpt_path = "models/matcha-icefall-zh-en/pytorch_model.bin"

    print("[*] Loading dictionaries and Text Encoder...")
    cmu = load_cmudict(cmu_path)
    token_to_id = load_token_mapping(tokens_path)
    enc_sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    print(f"[+] Loaded CMUDict ({len(cmu)} words) and tokens ({len(token_to_id)} IDs)")

    # 2. 初始化官方 Decoder 并严密加载底座权重
    print("\n[*] Initializing official Decoder and loading base weights...")
    dec = Decoder(
        in_channels=160, out_channels=80, channels=(256, 256),
        dropout=0.0, attention_head_dim=64, n_blocks=1, num_mid_blocks=2, num_heads=2,
        act_fn="snakebeta",
    ).to(device)

    raw_sd = torch.load(pt_ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in raw_sd:
        raw_sd = raw_sd["state_dict"]
    est_sd = {k.replace("model.decoder.estimator.", ""): v for k, v in raw_sd.items() if k.startswith("model.decoder.estimator.")}
    dec.load_state_dict(est_sd, strict=True)
    print("[+] Successfully loaded 100% of base model weights into official Decoder!")

    # 3. 冻结高维注意力与时间编码，仅微调声学残差卷积与投影层
    trainable_params = []
    for name, p in dec.named_parameters():
        if "transformer_blocks" in name or "time_" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
            trainable_params.append(p)

    trainable_num = sum(p.numel() for p in trainable_params)
    total_num = sum(p.numel() for p in dec.parameters())
    print(f"[*] Trainable parameters: {trainable_num:,} / {total_num:,} ({trainable_num/total_num*100:.1f}%)")

    # 4. 特征提取与对齐
    metadata_path = os.path.join(args.dataset_dir, "metadata.csv")
    wav_dir = os.path.join(args.dataset_dir, "wavs")
    pairs = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                pairs.append((row[0].strip(), row[1].strip()))

    print(f"\n[*] Processing and aligning {len(pairs)} training pairs...")
    dataset = []
    t0 = time.time()
    for wav_file, text in pairs:
        full_wav = os.path.join(wav_dir, wav_file)
        if not os.path.exists(full_wav):
            continue
        real_mel = extract_mel(full_wav) # [80, T_real]

        t_ids = text_to_tokens_bilingual(text, token_to_id, cmu)
        if len(t_ids) == 0:
            continue

        tokens_np = np.array([t_ids], dtype=np.int64)
        x_len = np.array([len(t_ids)], dtype=np.int64)
        mu, mask = enc_sess.run(None, {
            "x": tokens_np,
            "x_length": x_len,
            "noise_scale": np.array([0.667], dtype=np.float32),
            "length_scale": np.array([1.0], dtype=np.float32),
        })

        mu = mu[0] # [80, T_mu]
        t_real = real_mel.shape[1]
        t_mu = mu.shape[1]
        if t_mu != t_real:
            mu_aligned = np.zeros((80, t_real), dtype=np.float32)
            x_old = np.linspace(0, 1, t_mu)
            x_new = np.linspace(0, 1, t_real)
            for c in range(80):
                mu_aligned[c] = np.interp(x_new, x_old, mu[c])
        else:
            mu_aligned = mu

        dataset.append((
            torch.from_numpy(real_mel).to(device),
            torch.from_numpy(mu_aligned).to(device)
        ))

    print(f"[+] Dataset prepped in {time.time() - t0:.1f}s. Valid samples: {len(dataset)}")

    # 5. OT-CFM 微调训练
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"\n[*] Starting OT-CFM training ({args.epochs} epochs, lr={args.lr})...")
    dec.train()
    start_train = time.time()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        steps = 0
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            batch_pairs = [dataset[idx] for idx in batch_idx]
            if not batch_pairs:
                continue

            max_len = max(m.shape[1] for m, _ in batch_pairs)
            # 对齐时间帧数为 4 的整数倍，避免 UNet 下采样和上采样奇数帧差 1
            max_len = ((max_len + 3) // 4) * 4
            B = len(batch_pairs)

            padded_x1 = torch.zeros(B, 80, max_len, device=device)
            padded_mu = torch.zeros(B, 80, max_len, device=device)
            padded_mask = torch.zeros(B, 1, max_len, device=device)

            for b_i, (m, mu) in enumerate(batch_pairs):
                l = m.shape[1]
                padded_x1[b_i, :, :l] = m
                padded_mu[b_i, :, :l] = mu
                padded_mask[b_i, 0, :l] = 1.0

            z0 = torch.randn_like(padded_x1) * 0.667
            x0 = padded_mu + z0
            x1 = padded_x1

            t = torch.rand(B, device=device)
            t_expand = t.view(B, 1, 1)

            xt = (1.0 - t_expand) * x0 + t_expand * x1
            target_v = x1 - x0

            optimizer.zero_grad()
            pred_v = dec(xt, padded_mask, padded_mu, t)
            # 仅在有效 mask 区域计算损失
            loss = criterion(pred_v * padded_mask, target_v * padded_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        avg_loss = total_loss / max(1, steps)
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"[*] Epoch [{epoch:02d}/{args.epochs:02d}] - CFM Loss: {avg_loss:.5f} - LR: {scheduler.get_last_lr()[0]:.2e}")

    train_duration = time.time() - start_train
    print(f"\n[+] Fine-tuning completed in {train_duration:.1f}s. Final CFM Loss: {avg_loss:.5f}")

    # 6. 保存微调模型权重
    save_pt = os.path.join(args.output_dir, "estimator_finetuned.pt")
    torch.save(dec.state_dict(), save_pt)
    print(f"[+] Saved fine-tuned checkpoint: {save_pt}")

    # 7. 端到端 3-step Euler 试听音频合成
    print("\n[*] Synthesizing test audio using 3-step Euler solver...")
    dec.eval()
    vocos_sess = ort.InferenceSession(vocos_path, providers=["CPUExecutionProvider"])

    test_cases = [
        ("finetuned_real01_bilingual.wav", "您好！我是您的智能语音助手。Welcome to Seeed Studio! 今天有什么可以帮您的？"),
        ("finetuned_real02_tech.wav", "我们在边缘计算设备上部署了 Matcha-TTS 与 MiniMax 声音克隆模型，推理延迟低于一百毫秒。"),
        ("finetuned_real03_morning.wav", "Good morning! 今天的气温适宜，阳光明媚，适合户外活动。Have a wonderful day!"),
    ]

    samples_dir = "voices/samples"
    os.makedirs(samples_dir, exist_ok=True)

    for fname, text in test_cases:
        t_ids = text_to_tokens_bilingual(text, token_to_id, cmu)
        tokens_np = np.array([t_ids], dtype=np.int64)
        x_len = np.array([len(t_ids)], dtype=np.int64)

        mu, mask = enc_sess.run(None, {
            "x": tokens_np,
            "x_length": x_len,
            "noise_scale": np.array([0.667], dtype=np.float32),
            "length_scale": np.array([1.0], dtype=np.float32),
        })

        # 对齐推理长度为 4 的整数倍
        cur_t = mu.shape[2]
        if cur_t % 4 != 0:
            target_t = ((cur_t + 3) // 4) * 4
            pad_w = target_t - cur_t
            mu = np.pad(mu, ((0, 0), (0, 0), (0, pad_w)), mode="edge")
            mask = np.pad(mask, ((0, 0), (0, 0), (0, pad_w)), mode="constant", constant_values=1.0)

        mu_t = torch.from_numpy(mu).to(device)
        mask_t = torch.from_numpy(mask).to(device)

        # 3-step Euler
        torch.manual_seed(42)
        z = torch.randn_like(mu_t) * 0.667
        x = mu_t + z
        steps = 3
        dt = 1.0 / steps

        with torch.no_grad():
            for s_idx in range(steps):
                t_val = float(s_idx) / steps
                t_vec = torch.full((1,), t_val, device=device)
                v = dec(x, mask_t, mu_t, t_vec)
                x = x + v * dt

        mel_pt = (x * MEL_SIGMA + MEL_MEAN).cpu().numpy()

        mag, vx, vy = vocos_sess.run(["mag", "x", "y"], {"mels": mel_pt})
        stft_result = knf.StftResult(
            real=(mag * vx)[0].transpose().reshape(-1).tolist(),
            imag=(mag * vy)[0].transpose().reshape(-1).tolist(),
            num_frames=mag.shape[2],
        )
        stft_cfg = knf.StftConfig(
            n_fft=1024, hop_length=256, win_length=1024, window_type="hann",
            center=True, pad_mode="reflect", normalized=False,
        )
        istft = knf.IStft(stft_cfg)
        audio = np.array(istft(stft_result))

        out_path = os.path.join(samples_dir, fname)
        sf.write(out_path, audio, SAMPLE_RATE)
        rms = np.sqrt(np.mean(audio**2))
        print(f"  [+] Generated: {out_path} ({len(audio)/SAMPLE_RATE:.2f}s, RMS: {rms:.4f})")

    print("=" * 65)
    print("=== All Pipeline Tasks Completed Successfully! ===")
    print("=" * 65)


if __name__ == "__main__":
    main()
