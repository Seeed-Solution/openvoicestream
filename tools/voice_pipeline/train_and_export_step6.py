#!/usr/bin/env python3
"""
train_and_export_step6.py
Matcha-TTS 中英双语底座针对 MiniMax 少女亲和音色端到端微调与 6-Step ONNX 导出引擎

流程：
1. 数据加载与前处理：
   - 使用 rkvoice-stream 官方逆向前端完成精准中英双语分词 (pinyin + espeak-ng -> tokens.txt)；
   - 提取 16kHz 80-mel 语谱图，并按照 Matcha-TTS 规范归一化；
   - 利用官方 Encoder ONNX 提取文本音素先验 mu 并线性插值对齐。
2. 架构无损微调 (OT-CFM)：
   - 官方原生 Decoder (1D UNet, SnakeBeta) 初始化并严格对齐底座权重；
   - 锁定 Transformer 注意力层与 TimeMLP，微调 ResNet 声学残差块与 Mel 投影层；
   - 在 GPU (cuda:0) 上以 30 epochs 迭代训练，记录 loss 下降。
3. 6 步 Euler 解算 ONNX 注入导出：
   - 将微调后的参数精确回写至官方 model-steps-6.onnx；
   - 验证导出模型签名 [x, x_length, noise_scale, length_scale] -> [mel]。
4. 验证音频合成与 MacBook 推送：
   - 合成中英混合句与长英文句；
   - 通过 Vocos 神经声码器输出无损波形；
   - 调用 fleet push 推送至 MacBook 试听。
"""

import os
import sys
import re
import csv
import time
import argparse
import numpy as np
import librosa
import soundfile as sf
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim

# 导入工程内部模块
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("third_party/rkvoice-stream"))

from tools.voice_pipeline.matcha_official.decoder import Decoder
from rkvoice_stream.backends.tts.matcha import RKNNMatchaVocoder, parse_tokens_file
import kaldi_native_fbank as knf

# 声学模型常量
MEL_MEAN = -2.9521978
MEL_SIGMA = 5.446792
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80


def extract_mel(wav_path: str) -> np.ndarray:
    """提取 16kHz 80-band Log-Mel 语谱图并标准化"""
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
    parser = argparse.ArgumentParser(description="Matcha-TTS MiniMax Fine-Tuning and 6-Step ONNX Exporter")
    parser.add_argument("--dataset-dir", default="voices/minimax_shaonv", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output-dir", default="voices/minimax_shaonv", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("=== Matcha-TTS MiniMax Voice Adaptation Engine (6-Step Euler) ===")
    print("=" * 70)
    print(f"[*] Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"[*] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 1. 资产与前端初始化
    tokens_path = "models/matcha-icefall-zh-en/tokens.txt"
    lexicon_path = "models/matcha-icefall-zh-en/lexicon.txt"
    espeak_data = "models/matcha-icefall-zh-en/espeak-ng-data"
    encoder_path = "models/matcha-icefall-zh-en/encoder.onnx"
    base_m6_path = "models/matcha-icefall-zh-en/model-steps-6.onnx"
    vocos_path = "models/matcha-icefall-zh-en/vocos-16khz-univ.onnx"
    pt_ckpt_path = "models/matcha-icefall-zh-en/pytorch_model.bin"

    print("\n[*] Initializing frontend and Text Encoder session...")
    vocoder = RKNNMatchaVocoder("", "", lexicon_path, tokens_path, espeak_data)
    vocoder._token_to_id = parse_tokens_file(tokens_path)
    vocoder._lexicon = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocoder._lexicon[parts[0]] = parts[1:]

    enc_sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    print(f"[+] Loaded tokens ({len(vocoder._token_to_id)} IDs), lexicon ({len(vocoder._lexicon)} entries)")

    # 2. 官方 Decoder 模型初始化与精准底座加载
    print("\n[*] Initializing official Decoder and loading base weights...")
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

    # 从 pytorch_model.bin 完整加载 180 个层
    raw_sd = torch.load(pt_ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in raw_sd:
        raw_sd = raw_sd["state_dict"]
    est_sd = {
        k.replace("model.decoder.estimator.", ""): v
        for k, v in raw_sd.items()
        if k.startswith("model.decoder.estimator.")
    }
    dec.load_state_dict(est_sd, strict=True)
    print(f"[+] Successfully loaded {len(est_sd)}/180 parameters from {pt_ckpt_path}")

    # 用 model-steps-6.onnx 中的对应权重精确对齐底座
    m6_proto = onnx.load(base_m6_path, load_external_data=False)
    m6_inits = {i.name: numpy_helper.to_array(i) for i in m6_proto.graph.initializer}
    overridden_count = 0
    for k, v in dec.named_parameters():
        onnx_k = f"model.decoder.estimator.{k}"
        if onnx_k in m6_inits:
            arr = m6_inits[onnx_k]
            if tuple(v.shape) == arr.shape:
                v.data.copy_(torch.from_numpy(arr.copy()))
                overridden_count += 1
    print(f"[+] Aligned {overridden_count} parameters directly with {base_m6_path} base initializers")

    # 3. 参数冻结与优化器配置
    # 锁定 Transformer 注意力块 (down/mid/up 的 .1. 层)、TimeMLP 和 GroupNorm
    trainable_params = []
    trainable_names = []
    for name, p in dec.named_parameters():
        if re.search(r"(down|mid|up)_blocks\.\d+\.1\.", name):
            p.requires_grad = False
        elif "time_" in name or "block.1." in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
            trainable_params.append(p)
            trainable_names.append(name)

    trainable_num = sum(p.numel() for p in trainable_params)
    total_num = sum(p.numel() for p in dec.parameters())
    print(f"[*] Trainable parameters: {trainable_num:,} / {total_num:,} ({trainable_num/total_num*100:.1f}%) across {len(trainable_params)} tensors")

    # 4. 数据特征提取与音素先验对齐
    metadata_path = os.path.join(args.dataset_dir, "metadata.csv")
    wav_dir = os.path.join(args.dataset_dir, "wavs")
    pairs = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                pairs.append((row[0].strip(), row[1].strip()))

    print(f"\n[*] Processing and aligning {len(pairs)} MiniMax audio-text pairs...")
    dataset = []
    t0 = time.time()
    for wav_file, text in pairs:
        full_wav = os.path.join(wav_dir, wav_file)
        if not os.path.exists(full_wav):
            continue
        real_mel = extract_mel(full_wav)  # [80, T_real]

        t_ids = vocoder.text_to_tokens(text)
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

        mu = mu[0]  # [80, T_mu]
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
            torch.from_numpy(mu_aligned).to(device),
        ))

    print(f"[+] Preprocessed {len(dataset)} valid samples in {time.time() - t0:.1f}s")

    # 5. OT-CFM 条件流匹配训练
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"\n[*] Starting OT-CFM training ({args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size})...")
    dec.train()
    start_train = time.time()

    loss_history = []
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
            # 对齐为 4 的整数倍
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

            # OT-CFM: x0 = mu + z0, x1 = target_mel
            z0 = torch.randn_like(padded_x1) * 0.667
            x0 = padded_mu + z0
            x1 = padded_x1

            t = torch.rand(B, device=device)
            t_expand = t.view(B, 1, 1)

            xt = (1.0 - t_expand) * x0 + t_expand * x1
            target_v = x1 - x0

            optimizer.zero_grad()
            pred_v = dec(xt, padded_mask, padded_mu, t)
            loss = criterion(pred_v * padded_mask, target_v * padded_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        avg_loss = total_loss / max(1, steps)
        loss_history.append((epoch, avg_loss, scheduler.get_last_lr()[0]))

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"[*] Epoch [{epoch:02d}/{args.epochs:02d}] - CFM Loss: {avg_loss:.5f} - LR: {scheduler.get_last_lr()[0]:.2e}")

    train_duration = time.time() - start_train
    print(f"\n[+] Fine-tuning completed in {train_duration:.1f}s. Initial Loss: {loss_history[0][1]:.5f} -> Final Loss: {loss_history[-1][1]:.5f}")

    # 6. 保存微调权重 Checkpoint
    save_pt = os.path.join(args.output_dir, "estimator_finetuned.pt")
    torch.save(dec.state_dict(), save_pt)
    print(f"[+] Saved fine-tuned checkpoint: {save_pt} ({os.path.getsize(save_pt)/(1024*1024):.1f} MB)")

    # 7. 回写微调权重至 6-step ONNX 模型
    output_onnx_path = os.path.join(args.output_dir, "matcha_shaonv_step6.onnx")
    print(f"\n[*] Injecting fine-tuned weights into 6-step ONNX model: {output_onnx_path}...")
    dec_state = dec.state_dict()

    updated_weights_count = 0
    for init in m6_proto.graph.initializer:
        if init.name.startswith("model.decoder.estimator."):
            pt_param_name = init.name.replace("model.decoder.estimator.", "")
            if pt_param_name in dec_state and pt_param_name in trainable_names:
                tensor = dec_state[pt_param_name]
                arr = tensor.cpu().numpy()
                orig_arr = numpy_helper.to_array(init)
                if arr.shape == orig_arr.shape:
                    new_init = numpy_helper.from_array(arr, name=init.name)
                    init.CopyFrom(new_init)
                    updated_weights_count += 1

    onnx.save(m6_proto, output_onnx_path)
    out_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"[+] Successfully exported {output_onnx_path} ({out_size_mb:.2f} MB, {updated_weights_count} fine-tuned weight tensors injected)")

    # 8. 校验导出的 ONNX 结构与签名
    print("\n[*] Validating exported ONNX model signature and runtime...")
    test_sess = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    inputs = [(i.name, i.type, i.shape) for i in test_sess.get_inputs()]
    outputs = [(o.name, o.type, o.shape) for o in test_sess.get_outputs()]
    print(f"[*] Exported ONNX Inputs: {inputs}")
    print(f"[*] Exported ONNX Outputs: {outputs}")

    # 9. 端到端测试音频合成
    print("\n[*] Synthesizing evaluation test sentences...")
    vocos_sess = ort.InferenceSession(vocos_path, providers=["CPUExecutionProvider"])

    test_sentences = [
        (
            "minimax_shaonv_test01_bilingual.wav",
            "你好，我是 Seeed Studio 智能语音助手。Hello, welcome to the future."
        ),
        (
            "minimax_shaonv_test02_english.wav",
            "Welcome to Seeed Studio! We are dedicated to empowering developers, makers, and innovators around the world. With cutting-edge edge AI technologies and versatile hardware platforms, we make open-source intelligence accessible everywhere."
        )
    ]

    samples_dir = "voices/samples"
    os.makedirs(samples_dir, exist_ok=True)
    generated_wavs = []

    stft_cfg = knf.StftConfig(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window_type="hann",
        center=True,
        pad_mode="reflect",
        normalized=False,
    )
    istft = knf.IStft(stft_cfg)

    for fname, text in test_sentences:
        t_ids = vocoder.text_to_tokens(text)
        tokens_np = np.array([t_ids], dtype=np.int64)
        x_len = np.array([len(t_ids)], dtype=np.int64)
        noise_scale = np.array([0.667], dtype=np.float32)
        length_scale = np.array([1.0], dtype=np.float32)

        t_start = time.time()
        mel = test_sess.run(
            ["mel"],
            {
                "x": tokens_np,
                "x_length": x_len,
                "noise_scale": noise_scale,
                "length_scale": length_scale,
            }
        )[0]
        matcha_cost = time.time() - t_start

        t_voc = time.time()
        mag, x, y = vocos_sess.run(["mag", "x", "y"], {"mels": mel})
        stft_result = knf.StftResult(
            real=(mag * x)[0].transpose().reshape(-1).tolist(),
            imag=(mag * y)[0].transpose().reshape(-1).tolist(),
            num_frames=mag.shape[2],
        )
        audio = np.array(istft(stft_result))
        vocos_cost = time.time() - t_voc

        out_path = os.path.join(samples_dir, fname)
        sf.write(out_path, audio, SAMPLE_RATE)
        duration_s = len(audio) / SAMPLE_RATE
        rms = np.sqrt(np.mean(audio ** 2))
        print(f"  [+] Synthesized: {out_path} ({duration_s:.2f}s, RTF: {(matcha_cost + vocos_cost)/duration_s:.3f}, RMS: {rms:.4f})")
        generated_wavs.append((out_path, fname, duration_s, rms))

    # 10. 推送到 MacBook
    fleet_bin = os.path.expanduser("~/.rpty/bin/fleet")
    print(f"\n[*] Pushing generated WAVs to MacBook via fleet...")
    import subprocess
    fleet_logs = []
    for full_path, fname, dur, _ in generated_wavs:
        target_remote = f"/Users/harvest/Downloads/{fname}"
        cmd = [fleet_bin, "push", "macbook", full_path, target_remote]
        print(f"  [>] Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"      Exit: {res.returncode}, Stdout: {res.stdout.strip()}, Stderr: {res.stderr.strip()}")
        fleet_logs.append({
            "file": fname,
            "remote_path": target_remote,
            "code": res.returncode,
            "duration": dur,
            "output": res.stdout.strip()
        })

    print("=" * 70)
    print("=== All Matcha-TTS MiniMax Fine-Tuning Tasks Completed! ===")
    print("=" * 70)

    # 打印最终总结
    print("\n[TRAINING SUMMARY]")
    print(f"- Dataset samples: {len(dataset)}")
    print(f"- Total epochs: {args.epochs}")
    print(f"- Loss decline: {loss_history[0][1]:.5f} -> {loss_history[-1][1]:.5f}")
    print(f"- Exported ONNX: {output_onnx_path} ({out_size_mb:.2f} MB)")
    print(f"- Injected weight tensors: {updated_weights_count}")
    for item in fleet_logs:
        print(f"- Pushed to MacBook: {item['remote_path']} (Exit {item['code']}, duration: {item['duration']:.2f}s)")


if __name__ == "__main__":
    main()
