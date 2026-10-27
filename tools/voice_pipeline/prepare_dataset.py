#!/usr/bin/env python3
"""
tools/voice_pipeline/prepare_dataset.py
数据清洗与训练集划分脚本：
1. 校验每个生成的 WAV 完整性与音频时长
2. 拆分 train.txt 与 val.txt
3. 打包生成用于远程训练的 dataset.tar.gz
"""

import os
import sys
import tarfile
import random
import wave

def get_wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0

def main():
    root = "/home/harvest/project/seeed-local-voice/voices/zgl_matcha_pipeline"
    meta_path = os.path.join(root, "metadata.csv")
    wavs_dir = os.path.join(root, "wavs")

    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    valid_entries = []
    total_duration = 0.0
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 2:
            fname = parts[0]
            text = parts[1]
            wav_file = os.path.join(wavs_dir, fname)
            dur = get_wav_duration(wav_file)
            if dur > 0.5:
                valid_entries.append((fname, text, dur))
                total_duration += dur

    print(f"Total valid audio samples: {len(valid_entries)}")
    print(f"Total effective audio duration: {total_duration:.1f}s ({total_duration/60.0:.2f} mins)")

    # 划分训练集与验证集 (90% train, 10% val)
    random.seed(42)
    random.shuffle(valid_entries)
    split_idx = int(len(valid_entries) * 0.9)
    train_set = valid_entries[:split_idx]
    val_set = valid_entries[split_idx:]

    train_file = os.path.join(root, "train.txt")
    val_file = os.path.join(root, "val.txt")

    with open(train_file, "w", encoding="utf-8") as f:
        for fname, text, dur in train_set:
            f.write(f"wavs/{fname}|{text}\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for fname, text, dur in val_set:
            f.write(f"wavs/{fname}|{text}\n")

    print(f"Train samples: {len(train_set)} -> {train_file}")
    print(f"Val samples: {len(val_set)} -> {val_file}")

    # 打包为 zgl_dataset.tar.gz
    archive_path = os.path.join(root, "zgl_dataset.tar.gz")
    print(f"Archiving dataset to {archive_path}...")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(train_file, arcname="train.txt")
        tar.add(val_file, arcname="val.txt")
        tar.add(meta_path, arcname="metadata.csv")
        for fname, _, _ in valid_entries:
            wav_path = os.path.join(wavs_dir, fname)
            tar.add(wav_path, arcname=f"wavs/{fname}")

    sz_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"Dataset package ready! Size: {sz_mb:.2f} MB")

if __name__ == "__main__":
    main()
