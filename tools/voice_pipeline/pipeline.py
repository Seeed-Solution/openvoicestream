#!/usr/bin/env python3
"""
tools/voice_pipeline/pipeline.py
通用化轻量声学模型音色微调调度引擎：
通过配置驱动（YAML/JSON）统一串联：
1. 教师模型音色设计与数据扩充 (Teacher TTS)
2. 自动化质检与数据打包 (Dataset Prep)
3. 训练后端调度与 ONNX 导出 (Training & Export)
"""

import os
import sys
import argparse
import subprocess
import json

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".json") or yaml is None:
            try:
                return json.load(f)
            except Exception:
                pass
        if yaml is not None:
            f.seek(0)
            return yaml.safe_load(f)
        raise RuntimeError("PyYAML not installed. Please install pyyaml or pass a .json config.")


def step_synthesize(cfg: dict):
    print("\n=== Step 1: Batch Synthesizing Dataset with Teacher TTS ===")
    teacher_cfg = cfg.get("teacher", {})
    corpus_cfg = cfg.get("corpus", {})
    proj_cfg = cfg.get("project", {})

    api_key_env = teacher_cfg.get("api_key_env", "Aliyun_General")
    api_key = (
        os.environ.get(api_key_env)
        or os.environ.get("Aliyun_General")
        or os.environ.get("Aliyun_apikey")
        or os.environ.get("DASHSCOPE_API_KEY")
    )
    if not api_key:
        print(f"Error: Missing environment variable {api_key_env} (or Aliyun_General/Aliyun_apikey)", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(proj_cfg.get("output_dir", "./output"), "wavs")
    texts_file = corpus_cfg.get("text_file", "./tools/voice_pipeline/corpus/texts_bilingual_balanced.txt")
    workers = corpus_cfg.get("workers", 4)
    provider = teacher_cfg.get("provider", "dashscope")
    model = teacher_cfg.get("model", "cosyvoice-v2" if provider == "dashscope" else "speech-01-turbo")
    voice = teacher_cfg.get("voice", "longxiaochun_v2" if provider == "dashscope" else "female-tianmei")
    base_url = teacher_cfg.get("base_url", "")

    cmd = [
        "python3", "tools/voice_pipeline/batch_synthesize.py",
        "--texts", texts_file,
        "--outdir", out_dir,
        "--provider", provider,
        "--voice", voice,
        "--model", model,
        "--workers", str(workers),
        "--api-key-env", api_key_env,
    ]
    if base_url:
        cmd.extend(["--base-url", base_url])

    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def step_prepare(cfg: dict):
    print("\n=== Step 2: Quality Gate & Preparing Dataset ===")
    proj_cfg = cfg.get("project", {})
    corpus_cfg = cfg.get("corpus", {})
    out_dir = proj_cfg.get("output_dir", "./output")
    split_ratio = corpus_cfg.get("split_ratio", [0.9, 0.1])[0]
    min_dur = corpus_cfg.get("min_duration_sec", 0.5)

    cmd = [
        "python3", "tools/voice_pipeline/prepare_dataset.py",
        "--data-dir", out_dir,
        "--split-ratio", str(split_ratio),
        "--min-duration", str(min_dur),
        "--archive-name", "dataset.tar.gz",
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def step_train(cfg: dict):
    print("\n=== Step 3: Executing Training & ONNX Export ===")
    train_cfg = cfg.get("training", {})
    proj_cfg = cfg.get("project", {})

    backend = train_cfg.get("backend", "colab")
    out_dir = proj_cfg.get("output_dir", "./output")
    dataset_tar = os.path.join(out_dir, "dataset.tar.gz")
    epochs = train_cfg.get("epochs", 20)

    if not os.path.exists(dataset_tar):
        print(f"Error: Dataset package {dataset_tar} not found. Run prepare first.", file=sys.stderr)
        sys.exit(1)

    if backend == "colab":
        session_name = f"matcha-{proj_cfg.get('voice_name', 'voice')}"
        gpu_type = train_cfg.get("gpu_type", "T4")

        cmd = [
            "./tools/voice_pipeline/remote_colab_train.sh",
            session_name, dataset_tar, out_dir, gpu_type, str(epochs)
        ]
        print(f"Executing Colab runner: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    elif backend == "local_gpu":
        cmd = [
            "python3", "tools/voice_pipeline/colab_runner/train_and_export.py",
            "--dataset", out_dir,
            "--output-dir", out_dir,
            "--epochs", str(epochs),
        ]
        print(f"Executing Local CUDA training: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        raise NotImplementedError(f"Unsupported training backend: {backend}")


def main():
    parser = argparse.ArgumentParser(description="Universal Matcha Voice Tuning Pipeline")
    parser.add_argument("action", choices=["synthesize", "prepare", "train", "all"], help="Pipeline action to run")
    parser.add_argument("--config", default="tools/voice_pipeline/config.template.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    voice_name = cfg.get("project", {}).get("voice_name")
    print(f"[*] Loaded configuration for voice: {voice_name}")

    if args.action == "synthesize":
        step_synthesize(cfg)
    elif args.action == "prepare":
        step_prepare(cfg)
    elif args.action == "train":
        step_train(cfg)
    elif args.action == "all":
        step_synthesize(cfg)
        step_prepare(cfg)
        step_train(cfg)
        final_onnx = os.path.join(cfg.get("project", {}).get("output_dir", "./output"), "model-steps-3.onnx")
        print(f"\n[+] Full pipeline completed successfully!")
        print(f"[+] Final ONNX model saved to: {final_onnx}")


if __name__ == "__main__":
    main()
