#!/usr/bin/env python3
"""
tools/voice_pipeline/batch_synthesize.py
生产级语料批量生成工具：
支持 DashScope (CosyVoice) 与 MiniMax (Speech-01-Turbo / T2A-v2) 教师大模型，
利用独立子进程隔离网络资源，彻底避免连接死锁，支持文本指纹断点续传与多进程并发。
"""

import os
import sys
import time
import argparse
import subprocess
import requests
import binascii
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_text_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:8]


def synthesize_worker_dashscope(
    text: str,
    outfile: str,
    voice_id: str,
    model_name: str,
    base_url: str,
    api_key_env: str,
) -> bool:
    """DashScope 独立子进程合成"""
    code = f'''
import os, sys, dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

api_key = os.environ.get("{api_key_env}") or os.environ.get("Aliyun_General") or os.environ.get("Aliyun_apikey") or os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    sys.exit(10)

dashscope.api_key = api_key
base_url = "{base_url}".strip()
if base_url:
    dashscope.base_http_api_url = base_url

synth = SpeechSynthesizer(
    model="{model_name}",
    voice="{voice_id}",
    format=AudioFormat.WAV_24000HZ_MONO_16BIT
)
try:
    data = synth.call("""{text}""")
    if data and len(data) > 1000:
        with open("{outfile}", "wb") as f:
            f.write(data)
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    sys.exit(2)
'''
    try:
        res = subprocess.run(
            ["uv", "run", "--with", "dashscope", "python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return res.returncode == 0
    except Exception:
        return False


def synthesize_worker_minimax(
    text: str,
    outfile: str,
    voice_id: str,
    model_name: str,
    api_key_env: str,
) -> bool:
    """MiniMax T2A-v2 语音合成接口（输出原生 24kHz WAV）"""
    api_key = os.environ.get(api_key_env) or os.environ.get("minimax_API")
    if not api_key:
        return False

    url = "https://api.minimax.chat/v1/t2a_v2"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name or "speech-01-turbo",
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 24000,
            "bitrate": 128000,
            "format": "wav",
            "channel": 1,
        },
    }
    for attempt in range(4):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                res_json = resp.json()
                code = res_json.get("base_resp", {}).get("status_code")
                if (
                    code == 0
                    and "data" in res_json
                    and "audio" in res_json["data"]
                ):
                    audio_bytes = binascii.unhexlify(res_json["data"]["audio"])
                    if len(audio_bytes) > 1000:
                        with open(outfile, "wb") as f:
                            f.write(audio_bytes)
                        return True
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return False


def main():
    parser = argparse.ArgumentParser(description="Batch Synthesize Training Corpus")
    parser.add_argument("--texts", required=True, help="Input text file path")
    parser.add_argument("--outdir", required=True, help="Output wav directory")
    parser.add_argument("--provider", default="dashscope", choices=["dashscope", "minimax"], help="TTS Provider")
    parser.add_argument("--voice", default="longxiaochun_v2", help="Voice ID / Name")
    parser.add_argument("--model", default="cosyvoice-v2", help="TTS Model name")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    parser.add_argument("--base-url", default="", help="Optional Maas endpoint URL")
    parser.add_argument("--api-key-env", default="Aliyun_General", help="Env var name for API key")
    args = parser.parse_args()

    if not os.path.exists(args.texts):
        print(f"Error: Text corpus file not found: {args.texts}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.texts, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"[*] Provider: {args.provider}, Lines: {len(lines)}, Workers: {args.workers}, Model: {args.model}, Voice: {args.voice}")

    # 读取已有 metadata.csv 做文本匹配校验
    parent_dir = os.path.dirname(os.path.abspath(args.outdir))
    meta_path = os.path.join(parent_dir, "metadata.csv")
    existing_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for l in f:
                parts = l.strip().split("|", 1)
                if len(parts) == 2:
                    existing_meta[parts[0]] = parts[1]

    tasks = []
    metadata = []
    for idx, text in enumerate(lines, 1):
        sample_name = f"sample_{idx:04d}.wav"
        sample_path = os.path.join(args.outdir, sample_name)
        # 严格检查：文件存在、体积正常、且原标注文本完全匹配
        if (
            os.path.exists(sample_path)
            and os.path.getsize(sample_path) > 1000
            and existing_meta.get(sample_name) == text
        ):
            metadata.append((sample_name, text))
        else:
            tasks.append((idx, text, sample_path, sample_name))

    print(f"[*] Existing valid completed: {len(metadata)}, To synthesize: {len(tasks)}")

    start_t = time.time()
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            if args.provider == "dashscope":
                future_to_item = {
                    executor.submit(
                        synthesize_worker_dashscope,
                        text,
                        sample_path,
                        args.voice,
                        args.model,
                        args.base_url,
                        args.api_key_env,
                    ): (idx, text, sample_path, sample_name)
                    for idx, text, sample_path, sample_name in tasks
                }
            else:
                future_to_item = {
                    executor.submit(
                        synthesize_worker_minimax,
                        text,
                        sample_path,
                        args.voice,
                        args.model or "speech-01-turbo",
                        args.api_key_env or "minimax_API",
                    ): (idx, text, sample_path, sample_name)
                    for idx, text, sample_path, sample_name in tasks
                }

            done_count = 0
            for future in as_completed(future_to_item):
                idx, text, sample_path, sample_name = future_to_item[future]
                ok = future.result()
                done_count += 1
                if ok:
                    print(f"[{done_count:03d}/{len(tasks):03d}] OK: {sample_name} -> {text[:22]}...")
                    metadata.append((sample_name, text))
                else:
                    print(f"[{done_count:03d}/{len(tasks):03d}] FAILED: {sample_name} -> {text[:22]}...")

    # 保存元数据 metadata.csv
    metadata.sort(key=lambda x: x[0])
    with open(meta_path, "w", encoding="utf-8") as f:
        for fname, text in metadata:
            f.write(f"{fname}|{text}\n")

    cost = time.time() - start_t
    print(f"[*] Synthesis complete in {cost:.1f}s. Valid samples: {len(metadata)}. Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
