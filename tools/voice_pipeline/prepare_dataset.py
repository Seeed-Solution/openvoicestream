#!/usr/bin/env python3
"""
tools/voice_pipeline/prepare_dataset.py
数据清洗与训练集划分脚本：
1. 校验每个生成的 WAV 完整性、采样率与实际解码时长
2. 拆分互斥的 train.txt 与 val.txt
3. 打包生成用于远程训练的 dataset.tar.gz
"""

import os
import sys
import tarfile
import random
import wave
import argparse


def get_wav_info(path: str):
    """读取 WAV 文件头部并校验实际可读数据帧数"""
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            dur = frames / float(rate) if rate > 0 else 0.0
            
            # 基础数据体大小校验 (防止截断伪装)
            expected_data_size = frames * channels * sampwidth
            file_size = os.path.getsize(path)
            if file_size < expected_data_size:
                return 0.0, 0, 0
                
            return dur, rate, channels
    except Exception:
        return 0.0, 0, 0


def main():
    parser = argparse.ArgumentParser(description="Clean, split, and package speech training dataset")
    parser.add_argument("--data-dir", required=True, help="Target voice directory (containing metadata.csv and wavs/)")
    parser.add_argument("--split-ratio", type=float, default=0.9, help="Train set ratio (default: 0.9)")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Minimum duration in seconds")
    parser.add_argument("--archive-name", default="dataset.tar.gz", help="Archive package filename")
    args = parser.parse_args()

    root = os.path.abspath(args.data_dir)
    meta_path = os.path.join(root, "metadata.csv")
    wavs_dir = os.path.join(root, "wavs")

    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Run synthesis first.", file=sys.stderr)
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    valid_entries = []
    total_duration = 0.0
    for line in lines:
        # 只切分第一个管道符，避免文本内部含有竖线时被截断
        parts = line.split("|", 1)
        if len(parts) == 2:
            fname = parts[0].strip()
            text = parts[1].strip()
            wav_file = os.path.join(wavs_dir, fname)
            if not os.path.exists(wav_file):
                continue
            dur, rate, channels = get_wav_info(wav_file)
            if dur >= args.min_duration and rate == 24000:
                valid_entries.append((fname, text, dur))
                total_duration += dur

    print(f"[*] Total valid audio samples: {len(valid_entries)}")
    print(f"[*] Total effective audio duration: {total_duration:.1f}s ({total_duration/60.0:.2f} mins)")

    if not valid_entries:
        print("Error: No valid audio samples found to package!", file=sys.stderr)
        sys.exit(1)

    # 严格互斥划分训练集与验证集
    random.seed(42)
    random.shuffle(valid_entries)
    
    # 限制 ratio 范围
    ratio = max(0.5, min(0.95, args.split_ratio))
    split_idx = int(len(valid_entries) * ratio)
    if split_idx >= len(valid_entries) and len(valid_entries) > 1:
        split_idx = len(valid_entries) - 1

    train_set = valid_entries[:split_idx]
    val_set = valid_entries[split_idx:]

    train_file = os.path.join(root, "train.txt")
    val_file = os.path.join(root, "val.txt")

    with open(train_file, "w", encoding="utf-8") as f:
        for fname, text, _ in train_set:
            f.write(f"wavs/{fname}|{text}\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for fname, text, _ in val_set:
            f.write(f"wavs/{fname}|{text}\n")

    print(f"[*] Train set: {len(train_set)} -> {train_file}")
    print(f"[*] Val set:   {len(val_set)} -> {val_file}")

    # 打包为 dataset.tar.gz
    archive_path = os.path.join(root, args.archive_name)
    print(f"[*] Archiving dataset package to: {archive_path}...")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(train_file, arcname="train.txt")
        tar.add(val_file, arcname="val.txt")
        tar.add(meta_path, arcname="metadata.csv")
        for fname, _, _ in valid_entries:
            wav_path = os.path.join(wavs_dir, fname)
            tar.add(wav_path, arcname=f"wavs/{fname}")

    sz_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"[+] Dataset package successfully generated: {archive_path} ({sz_mb:.2f} MB)")


if __name__ == "__main__":
    main()
