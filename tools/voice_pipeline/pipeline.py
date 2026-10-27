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
try:
    import yaml
except ImportError:
    yaml = None
import json

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
    print("=== Step 1: Batch Synthesizing Dataset with Teacher TTS ===")
    teacher_cfg = cfg.get("teacher", {})
    corpus_cfg = cfg.get("corpus", {})
    proj_cfg = cfg.get("project", {})

    api_key_env = teacher_cfg.get("api_key_env", "DASHSCOPE_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"Error: Missing environment variable {api_key_env}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(proj_cfg.get("output_dir", "./output"), "wavs")
    texts_file = corpus_cfg.get("text_file")
    workers = corpus_cfg.get("workers", 3)
    base_url = teacher_cfg.get("base_url")

    # 调用批量生成脚本
    cmd = [
        "python3", "tools/voice_pipeline/batch_synthesize.py",
        "--texts", texts_file,
        "--outdir", out_dir,
        "--workers", str(workers),
        "--base-url", base_url
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def step_prepare(cfg: dict):
    print("=== Step 2: Quality Gate & Preparing Dataset ===")
    proj_cfg = cfg.get("project", {})
    out_dir = proj_cfg.get("output_dir", "./output")
    cmd = ["python3", "tools/voice_pipeline/prepare_dataset.py"]
    subprocess.run(cmd, check=True)

def step_train(cfg: dict):
    print("=== Step 3: Executing Training & ONNX Export ===")
    train_cfg = cfg.get("training", {})
    backend = train_cfg.get("backend", "colab")
    proj_cfg = cfg.get("project", {})

    if backend == "colab":
        session_name = f"matcha-{proj_cfg.get('voice_name', 'voice')}"
        dataset_tar = os.path.join(proj_cfg.get("output_dir", "./output"), "zgl_dataset.tar.gz")
        out_dir = proj_cfg.get("output_dir", "./output")
        gpu_type = train_cfg.get("gpu_type", "T4")
        
        cmd = [
            "./tools/voice_pipeline/remote_colab_train.sh",
            session_name, dataset_tar, out_dir, gpu_type
        ]
        print(f"Executing Colab runner: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    elif backend == "local_gpu":
        cmd = ["python3", "tools/voice_pipeline/colab_runner/train_and_export.py"]
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
    print(f"[*] Loaded configuration for project: {cfg.get('project', {}).get('voice_name')}")

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
        print(f"\n[+] Full pipeline completed! Final model saved to: {cfg.get('project', {}).get('output_dir')}/model-steps-3.onnx")

if __name__ == "__main__":
    main()
