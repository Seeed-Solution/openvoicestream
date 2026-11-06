#!/usr/bin/env python3
"""Transcribe WAVs with sherpa-onnx SenseVoice (RK3588). Resamples to 16k if needed."""
import os, sys, wave, numpy as np, sherpa_onnx

MODEL = os.environ.get(
    "SENSEVOICE_MODEL_DIR",
    "/opt/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
)

def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr

def resample(a, sr, target=16000):
    if sr == target:
        return a, sr
    idx = np.arange(0, len(a), sr / target)
    idx = idx[idx < len(a)].astype(np.int64)
    return a[idx], target

rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=f"{MODEL}/model.int8.onnx",
    tokens=f"{MODEL}/tokens.txt",
    use_itn=True,
    debug=False,
)

for path in sys.argv[1:]:
    a, sr = read_wav(path)
    a, sr = resample(a, sr)
    s = rec.create_stream()
    s.accept_waveform(sr, a)
    rec.decode_stream(s)
    print(f"{path}\t{s.result.text}")
