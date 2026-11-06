#!/usr/bin/env python3
"""Assemble results/accuracy-unified-corpus.json from the per-device source
JSON files, using score_unified.py's scoring for every cell so every number
in the deliverable is computed the same way, once, from raw ref/hyp pairs.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run_score(json_path, concurrency, ids_file, mode, root_key=None):
    cmd = [sys.executable, str(HERE / "score_unified.py"),
           "--json", str(HERE / json_path), "--concurrency", str(concurrency),
           "--ids-file", str(HERE / ids_file), "--mode", mode]
    if root_key:
        cmd += ["--root-key", root_key]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


whisper_devices = [
    ("J3011", "orin-nano", "results/j3011-whisper-matched100-fixed.json", 1, None,
     "reComputer J3011 (Jetson Orin Nano 8GB Super), TensorRT bf16 encoder + CPU ONNX KV decoder, voxedge main@466f3e4, bench.py post-#95 (vad=none, accumulate-to-close)"),
    ("J4012", "orin-nx", "results/j4012-whisper-matched100-fixed.json", 1, None,
     "reComputer J4012 (Jetson Orin NX 16GB Super), TensorRT bf16 encoder + CPU ONNX KV decoder, voxedge main@466f3e4, bench.py post-#95 (vad=none, accumulate-to-close)"),
    ("RK3576", "rk3576", "results/rk3576-whisper-matched100-fixed.json", 1, None,
     "Seeed reComputer (RK3576), RKNN base10 encoder + CPU ONNX KV decoder, voxedge main@466f3e4, bench.py post-#95 (vad=none, accumulate-to-close)"),
    ("RK3588", "rk3588", "results/rk3588-whisper-matched100-fixed.json", 1, None,
     "Seeed reComputer (RK3588), RKNN base10 encoder + CPU ONNX KV decoder, voxedge main@466f3e4, bench.py post-#95 (vad=none, accumulate-to-close)"),
    ("R2000", "harvest-pi", "results/r2000-whisper-matched100-fixed.json", 1, None,
     "reComputer R2000 (Raspberry Pi 5 + Hailo-8), Hailo base encoder (5s window) + CPU ONNX KV decoder, voxedge main@466f3e4, bench.py post-#95 (vad=none, accumulate-to-close)"),
]

sensevoice_devices = [
    ("J3011", "orin-nano", "results/concurrency-orin-nano-ceiling.json", 8, "sensevoice_zh",
     "reComputer J3011 (Jetson Orin Nano 8GB Super), TensorRT SenseVoice, 100 of the 200-item corpus extracted by matching id+ref"),
    ("J4012", "orin-nx", "results/concurrency-orin-nx-ceiling.json", 8, "sensevoice_zh",
     "reComputer J4012 (Jetson Orin NX 16GB Super), TensorRT SenseVoice, 100 of the 200-item corpus extracted by matching id+ref"),
    ("RK3576", "rk3576", "results/concurrency-cat-remote-ceiling.json", 4, "sensevoice",
     "Seeed reComputer (RK3576), RKNN SenseVoice fp16-scaled encoder, native 100-item corpus"),
    ("RK3588", "rk3588", "results/concurrency-radxa-ceiling.json", 2, "sensevoice",
     "Seeed reComputer (RK3588), RKNN SenseVoice fp16-scaled encoder, native 100-item corpus"),
    ("R2000", "harvest-pi", "results/concurrency-harvest-pi-ceiling.json", 2, "sensevoice",
     "reComputer R2000 (Raspberry Pi 5), CPU ONNX SenseVoice, native 100-item corpus"),
]

report = {"whisper": {}, "sensevoice": {}}

for label, dev, jf, c, key, note in whisper_devices:
    r = run_score(jf, c, "whisper100_ids.txt", "wer", key)
    r["device"] = dev
    r["label"] = label
    r["note"] = note
    report["whisper"][label] = r

for label, dev, jf, c, key, note in sensevoice_devices:
    r = run_score(jf, c, "sensevoice100_ids.txt", "cer", key)
    r["device"] = dev
    r["label"] = label
    r["note"] = note
    report["sensevoice"][label] = r

out = HERE / "results" / "accuracy-unified-corpus.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(f"wrote {out}")
for section in ("whisper", "sensevoice"):
    print(f"== {section} ==")
    for label, r in report[section].items():
        print(f"  {label}: n={r['n']} agg={r['aggregate']*100:.2f}% mean={r['mean']*100:.2f}% p50={r['p50']*100:.2f}% (c={r['concurrency']})")
