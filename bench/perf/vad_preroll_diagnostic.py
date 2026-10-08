"""Record frontend-VAD onset timing and production-style preroll intervals.

This is a diagnostic only: it feeds the real server VAD and never creates an
ASR backend or websocket.  Audio loading and 100 ms int16 quantisation match
``measure_v2v_unified.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Allow direct execution from the repository root or any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from server.core import vad as vad_mod


def load_audio(path: Path) -> tuple[np.ndarray, str, float]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)), audio,
        ).astype(np.float32)
    pcm_i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    return pcm_i16.astype(np.float32) / 32768.0, hashlib.sha256(
        pcm_i16.tobytes()
    ).hexdigest(), len(pcm_i16) / 16000.0


def diagnose(path: Path, *, chunk_ms: int, preroll_ms: int,
             silence_ms: int) -> dict:
    samples, feed_sha, duration_s = load_audio(path)
    chunk_samples = int(16000 * chunk_ms / 1000)
    preroll_cap = int(16000 * preroll_ms / 1000)
    detector = vad_mod.create_vad("silero", sample_rate=16000,
                                  silence_ms=silence_ms)
    ring: list[dict[str, int]] = []
    ring_samples = 0
    active = False
    rows: list[dict] = []
    onset: dict | None = None
    for start in range(0, len(samples), chunk_samples):
        end = min(start + chunk_samples, len(samples))
        chunk = samples[start:end]
        before = [dict(x) for x in ring]
        event = detector.process(chunk)
        speech_end_seen = event == vad_mod.VADSession.SPEECH_END
        row = {
            "chunk_index": len(rows),
            "start_sample": start,
            "end_sample": end,
            "start_s": start / 16000.0,
            "end_s": end / 16000.0,
            "event": event,
            "preroll_before": before,
        }
        if event == vad_mod.VADSession.SPEECH_START and onset is None:
            onset = {
                "chunk_index": len(rows),
                "trigger_interval": {"start_sample": start, "end_sample": end},
                "preroll_intervals": before,
                "preroll_samples": sum(x["end_sample"] - x["start_sample"] for x in before),
                "preroll_cap_samples": preroll_cap,
            }
            active = True
            ring = []
            ring_samples = 0
        # Match server/main.py: append only while no ASR turn is active, after
        # processing the current VAD chunk. The start event chunk is therefore
        # accepted as the trigger but is never inserted into this ring.
        if not active and not (event == vad_mod.VADSession.SPEECH_START):
            ring.append({"start_sample": start, "end_sample": end})
            ring_samples += end - start
            while ring_samples > preroll_cap and len(ring) > 1:
                old = ring.pop(0)
                ring_samples -= old["end_sample"] - old["start_sample"]
        row["preroll_after"] = [dict(x) for x in ring]
        row["preroll_after_samples"] = ring_samples
        rows.append(row)
        # Production keeps the current speech-end chunk in the active stream;
        # finalization/turn teardown happens later in the output task.
        if speech_end_seen:
            active = False

    return {
        "wav": str(path),
        "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "feed_sha256": feed_sha,
        "sample_rate": 16000,
        "chunk_ms": chunk_ms,
        "preroll_ms": preroll_ms,
        "preroll_cap_samples": preroll_cap,
        "vad_backend": "silero",
        "vad_silence_ms": silence_ms,
        "duration_s": duration_s,
        "onset": onset,
        "chunks": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--preroll-ms", type=int, default=300)
    ap.add_argument("--silence-ms", type=int, default=500)
    args = ap.parse_args()
    if args.chunk_ms <= 0 or args.preroll_ms < 0 or args.silence_ms < 0:
        ap.error("chunk-ms must be positive; silence/preroll must be non-negative")
    result = diagnose(args.wav, chunk_ms=args.chunk_ms,
                      preroll_ms=args.preroll_ms,
                      silence_ms=args.silence_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"out": str(args.out), "onset": result["onset"],
                      "feed_sha256": result["feed_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
