#!/usr/bin/env python3
"""
tools/voice_pipeline/batch_synthesize.py
生产级语料批量生成工具：
利用独立子进程隔离 WebSocket/gRPC 资源，彻底避免连接死锁，支持断点续传与多进程并发。
"""

import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def synthesize_worker(text: str, outfile: str, voice_id: str, base_url: str) -> bool:
    code = f'''
import os, sys, dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

dashscope.api_key = os.environ.get("Aliyun_apikey")
dashscope.base_http_api_url = "{base_url}"

synth = SpeechSynthesizer(
    model="cosyvoice-v3.5-flash",
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
            timeout=25
        )
        return res.returncode == 0
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch Synthesize Training Corpus")
    parser.add_argument("--texts", default="voices/zgl_matcha_pipeline/texts_balanced.txt", help="Input text file")
    parser.add_argument("--outdir", default="voices/zgl_matcha_pipeline/wavs", help="Output wav directory")
    parser.add_argument("--voice-id", default="cosyvoice-v3.5-flash-vd-zgl01-f1a9852bfc69471783d64bf1d0e79192", help="Voice ID")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent workers")
    parser.add_argument("--base-url", default="https://llm-f26wpyjsm2nxfjf2.cn-beijing.maas.aliyuncs.com/api/v1", help="Maas endpoint")
    args = parser.parse_args()

    if not os.environ.get("Aliyun_apikey"):
        print("Error: Aliyun_apikey env var missing", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.texts, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"Total lines: {len(lines)}, Workers: {args.workers}, Voice: {args.voice_id[:25]}...")
    
    tasks = []
    metadata = []
    for idx, text in enumerate(lines, 1):
        wav_name = f"zgl_{idx:04d}.wav"
        wav_path = os.path.join(args.outdir, wav_name)
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
            metadata.append((wav_name, text))
        else:
            tasks.append((idx, text, wav_path, wav_name))

    print(f"Existing completed: {len(metadata)}, To synthesize: {len(tasks)}")
    
    start_t = time.time()
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_item = {
                executor.submit(synthesize_worker, text, wav_path, args.voice_id, args.base_url): (idx, text, wav_path, wav_name)
                for idx, text, wav_path, wav_name in tasks
            }
            done_count = 0
            for future in as_completed(future_to_item):
                idx, text, wav_path, wav_name = future_to_item[future]
                ok = future.result()
                done_count += 1
                if ok:
                    print(f"[{done_count}/{len(tasks)}] Generated {wav_name}: {text[:15]}...")
                    metadata.append((wav_name, text))
                else:
                    print(f"[{done_count}/{len(tasks)}] FAILED {wav_name}: {text[:15]}...")

    # 保存元数据
    metadata.sort(key=lambda x: x[0])
    meta_file = os.path.join(os.path.dirname(args.outdir), "metadata.csv")
    with open(meta_file, "w", encoding="utf-8") as f:
        for fname, text in metadata:
            f.write(f"{fname}|{text}\n")

    print(f"\nFinished! Total valid audio files: {len(metadata)}/{len(lines)} in {time.time() - start_t:.1f}s.")
    print(f"Metadata saved to: {meta_file}")

if __name__ == "__main__":
    main()
