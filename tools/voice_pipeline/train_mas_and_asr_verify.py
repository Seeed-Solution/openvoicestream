#!/usr/bin/env python3
"""
train_mas_and_asr_verify.py
使用标准 MAS 单调对齐 + 官方 G2P 前端 + 纯 OT-CFM 流匹配进行 MiniMax 音色微调，
并以本地运行的 ASR 服务进行闭环转录验证。
"""

import os
import sys
import re
import csv
import time
import requests
import numba
import numpy as np
import librosa
import soundfile as sf
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
import kaldi_native_fbank as knf

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("third_party/rkvoice-stream"))

from tools.voice_pipeline.matcha_official.decoder import Decoder
from rkvoice_stream.backends.tts.matcha import RKNNMatchaVocoder, parse_tokens_file

MEL_MEAN = -2.9521978
MEL_SIGMA = 5.446792
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80

@numba.jit(nopython=True)
def mas_numba(dist):
    """Numba JIT accelerated Monotonic Alignment Search"""
    T_text, T_mel = dist.shape
    dp = np.full((T_text, T_mel), -1e9, dtype=np.float32)
    backtrack = np.zeros((T_text, T_mel), dtype=np.int32)

    dp[0, 0] = dist[0, 0]
    for j in range(1, T_mel):
        dp[0, j] = dp[0, j-1] + dist[0, j]
        backtrack[0, j] = 0

    for i in range(1, T_text):
        for j in range(i, T_mel):
            v1 = dp[i, j-1]
            v2 = dp[i-1, j-1]
            if v1 >= v2:
                dp[i, j] = v1 + dist[i, j]
                backtrack[i, j] = i
            else:
                dp[i, j] = v2 + dist[i, j]
                backtrack[i, j] = i - 1

    path = np.zeros((T_text, T_mel), dtype=np.float32)
    curr_i = T_text - 1
    for j in range(T_mel - 1, -1, -1):
        path[curr_i, j] = 1.0
        curr_i = backtrack[curr_i, j]

    return path

def extract_mel(wav_path: str) -> np.ndarray:
    wav, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window="hann", n_mels=N_MELS,
        fmin=0.0, fmax=8000.0, power=1.0, center=True, pad_mode="reflect",
    )
    log_mel = np.log(np.maximum(mel, 1e-5))
    norm_mel = (log_mel - MEL_MEAN) / MEL_SIGMA
    return norm_mel.astype(np.float32)

def main():
    device = torch.device("cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0)})")

    tokens_path = "models/matcha-icefall-zh-en/tokens.txt"
    lexicon_path = "models/matcha-icefall-zh-en/lexicon.txt"
    espeak_data = "models/matcha-icefall-zh-en/espeak-ng-data"
    encoder_path = "models/matcha-icefall-zh-en/encoder.onnx"
    base_m6_path = "models/matcha-icefall-zh-en/model-steps-6.onnx"
    vocos_path = "models/matcha-icefall-zh-en/vocos-16khz-univ.onnx"
    pt_ckpt_path = "models/matcha-icefall-zh-en/pytorch_model.bin"

    vocoder = RKNNMatchaVocoder("", "", lexicon_path, tokens_path, espeak_data)
    vocoder._token_to_id = parse_tokens_file(tokens_path)
    vocoder._lexicon = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2: vocoder._lexicon[parts[0]] = parts[1:]

    enc_sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])

    # 初始化 Decoder 并加载官方底座
    dec = Decoder(
        in_channels=160, out_channels=80, channels=(256, 256),
        dropout=0.0, attention_head_dim=64, n_blocks=1,
        num_mid_blocks=2, num_heads=2, act_fn="snakebeta",
    ).to(device)

    raw_sd = torch.load(pt_ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in raw_sd: raw_sd = raw_sd["state_dict"]
    est_sd = {k.replace("model.decoder.estimator.", ""): v for k, v in raw_sd.items() if k.startswith("model.decoder.estimator.")}
    dec.load_state_dict(est_sd, strict=True)

    m6_proto = onnx.load(base_m6_path, load_external_data=False)
    m6_inits = {i.name: numpy_helper.to_array(i) for i in m6_proto.graph.initializer}

    for k, v in dec.named_parameters():
        onnx_k = f"model.decoder.estimator.{k}"
        if onnx_k in m6_inits:
            arr = m6_inits[onnx_k]
            if tuple(v.shape) == arr.shape:
                v.data.copy_(torch.from_numpy(arr.copy()))

    # 参数冻结策略：
    # 锁定所有骨干特征提取层，微调 final_block、final_proj 和各 block 的 mlp 条件映射层
    # 这样能极大保全先验音素结构，避免过平滑塌陷
    trainable_params = []
    trainable_names = []
    for name, p in dec.named_parameters():
        if "final_" in name or "mlp" in name or "res_conv" in name:
            p.requires_grad = True
            trainable_params.append(p)
            trainable_names.append(name)
        else:
            p.requires_grad = False

    print(f"[*] Trainable parameters: {sum(p.numel() for p in trainable_params):,} / {sum(p.numel() for p in dec.parameters()):,} ({len(trainable_params)} tensors)")

    # 提取数据集并使用 MAS 精准对齐
    dataset_dir = "voices/minimax_shaonv"
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    wav_dir = os.path.join(dataset_dir, "wavs")
    pairs = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2: pairs.append((row[0].strip(), row[1].strip()))

    print(f"\n[*] Processing and aligning {len(pairs)} pairs with Monotonic Alignment Search (MAS)...")
    dataset = []
    t0 = time.time()
    for wav_file, text in pairs:
        full_wav = os.path.join(wav_dir, wav_file)
        if not os.path.exists(full_wav): continue
        real_mel = extract_mel(full_wav)
        t_ids = vocoder.text_to_tokens(text)
        if len(t_ids) == 0: continue

        tokens_np = np.array([t_ids], dtype=np.int64)
        x_len = np.array([len(t_ids)], dtype=np.int64)
        mu, _ = enc_sess.run(None, {
            "x": tokens_np, "x_length": x_len,
            "noise_scale": np.array([0.667], dtype=np.float32),
            "length_scale": np.array([1.0], dtype=np.float32),
        })
        mu = mu[0] # [80, T_text]

        # 计算真实 MAS 单调对齐路径
        dist = -0.5 * np.sum((real_mel[:, None, :] - mu[:, :, None])**2, axis=0) # [T_text, T_mel]
        path = mas_numba(dist.astype(np.float32)) # [T_text, T_mel]
        mu_aligned = mu @ path # [80, T_mel]

        dataset.append((
            torch.from_numpy(real_mel).to(device),
            torch.from_numpy(mu_aligned.astype(np.float32)).to(device),
        ))

    print(f"[+] Successfully aligned {len(dataset)} samples using MAS in {time.time() - t0:.2f}s")

    # 训练循环
    optimizer = optim.AdamW(trainable_params, lr=2e-5, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    dec.train()
    epochs = 30
    batch_size = 4
    print(f"\n[*] Starting Voice Adaptation ({epochs} epochs, lr=2e-5)...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_pairs = [dataset[idx] for idx in batch_idx]
            if not batch_pairs: continue

            max_len = ((max(m.shape[1] for m, _ in batch_pairs) + 3) // 4) * 4
            B = len(batch_pairs)

            padded_x1 = torch.zeros(B, 80, max_len, device=device)
            padded_mu = torch.zeros(B, 80, max_len, device=device)
            padded_mask = torch.zeros(B, 1, max_len, device=device)

            for b_i, (m, mu) in enumerate(batch_pairs):
                l = m.shape[1]
                padded_x1[b_i, :, :l] = m
                padded_mu[b_i, :, :l] = mu
                padded_mask[b_i, 0, :l] = 1.0

            z0 = torch.randn_like(padded_x1)
            x0 = z0
            x1 = padded_x1

            t = torch.rand(B, device=device).view(B, 1, 1)
            xt = (1.0 - t) * x0 + t * x1
            target_v = x1 - x0

            optimizer.zero_grad()
            pred_v = dec(xt, padded_mask, padded_mu, t.squeeze(-1).squeeze(-1))
            loss = criterion(pred_v * padded_mask, target_v * padded_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[*] Epoch [{epoch:02d}/{epochs:02d}] CFM Loss: {total_loss / steps:.5f}")

    # 导出微调 ONNX
    output_onnx_path = "voices/minimax_shaonv/matcha_shaonv_step6.onnx"
    dec_state = dec.state_dict()
    injected_count = 0
    for init in m6_proto.graph.initializer:
        if init.name.startswith("model.decoder.estimator."):
            pt_param_name = init.name.replace("model.decoder.estimator.", "")
            if pt_param_name in dec_state and pt_param_name in trainable_names:
                arr = dec_state[pt_param_name].cpu().numpy()
                orig_arr = numpy_helper.to_array(init)
                if arr.shape == orig_arr.shape:
                    init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
                    injected_count += 1

    onnx.save(m6_proto, output_onnx_path)
    print(f"\n[+] Exported ONNX: {output_onnx_path} ({injected_count} weights updated)")

    # 端到端测试与本地 ASR 闭环转录验证
    test_sess = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    vocos_sess = ort.InferenceSession(vocos_path, providers=["CPUExecutionProvider"])

    stft_cfg = knf.StftConfig(
        n_fft=1024, hop_length=256, win_length=1024,
        window_type="hann", center=True, pad_mode="reflect", normalized=False,
    )
    istft = knf.IStft(stft_cfg)

    def mel_to_audio(mel):
        mag, x, y = vocos_sess.run(["mag", "x", "y"], {"mels": mel})
        stft_result = knf.StftResult(
            real=(mag * x)[0].transpose().reshape(-1).tolist(),
            imag=(mag * y)[0].transpose().reshape(-1).tolist(),
            num_frames=mag.shape[2],
        )
        return np.array(istft(stft_result))

    test_cases = [
        ("minimax_shaonv_test01_bilingual.wav", "你好，我是 Seeed Studio 智能语音助手。Hello, welcome to the future.", "zh"),
        ("minimax_shaonv_test02_english.wav", "Welcome to Seeed Studio! We make open-source hardware accessible everywhere.", "en"),
    ]

    print("\n" + "=" * 60)
    print("=== LIVE ASR TRANSCRIPTION VERIFICATION ===")
    print("=" * 60)

    fleet_bin = os.path.expanduser("~/.rpty/bin/fleet")
    for fname, text, lang in test_cases:
        t_ids = vocoder.text_to_tokens(text)
        # 边界修复：在中英交界处插入空格 token 1
        fixed_ids = []
        for i in range(len(t_ids)):
            fixed_ids.append(t_ids[i])
            if t_ids[i] == 1624 and i + 1 < len(t_ids) and t_ids[i+1] == 48:
                fixed_ids.append(1)

        tokens_np = np.array([fixed_ids], dtype=np.int64)
        x_len = np.array([len(fixed_ids)], dtype=np.int64)

        mel = test_sess.run(["mel"], {
            "x": tokens_np, "x_length": x_len,
            "noise_scale": np.array([0.33], dtype=np.float32),
            "length_scale": np.array([1.0], dtype=np.float32),
        })[0]

        audio = mel_to_audio(mel)
        # 归一化到 0.9 峰值
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.90 / peak)

        out_path = f"voices/samples/{fname}"
        sf.write(out_path, audio, SAMPLE_RATE)
        rms = np.sqrt(np.mean(audio**2))
        dur = len(audio) / SAMPLE_RATE

        # 调取本地 8621 的 ASR 进行转录
        with open(out_path, "rb") as f:
            resp = requests.post(f"http://127.0.0.1:8621/asr?language={lang}", files={"file": (fname, f, "audio/wav")})
            asr_res = resp.json().get("text", "")

        print(f"\n[Case] {fname} ({dur:.2f}s, RMS={rms:.4f}, Peak={np.max(np.abs(audio)):.4f})")
        print(f"  Target Text: {text}")
        print(f"  ASR Result:  {asr_res}")

        # 推送到 MacBook
        os.system(f"{fleet_bin} push macbook {out_path} /Users/harvest/Downloads/{fname}")

if __name__ == "__main__":
    main()
