#!/usr/bin/env python3
"""Single-engine Qwen3 streaming window diagnostic.

This runner intentionally records prompt-reference text separately from ASR
output.  It is for an isolated RK3576 container and does not modify the
production backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from qwen3_asr_stream_eos_bench import build_engine


def load_audio(path: Path) -> tuple[np.ndarray, int, str]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        old = np.linspace(0, len(audio) - 1, len(audio))
        new = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(new, old, audio).astype(np.float32)
        sr = 16000
    # Match measure_v2v_unified.py and the production websocket path exactly:
    # float PCM is clipped, quantized to int16 with *32767, then reconstructed
    # by the receiver with /32768 before entering the ASR stream.
    pcm_i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    feed = pcm_i16.astype(np.float32) / 32768.0
    return feed, sr, hashlib.sha256(pcm_i16.tobytes()).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_case(engine, case: dict, group: dict, chunk_ms: int, realtime: bool) -> dict:
    from rkvoice_stream.backends.asr.qwen3.streaming import Qwen3TrueStreamingASRStream

    for key, value in group["env"].items():
        os.environ[key] = str(value)
    wav = Path(case["wav"])
    audio, sr, feed_sha256 = load_audio(wav)
    chunk_n = max(1, int(sr * chunk_ms / 1000))
    stream = Qwen3TrueStreamingASRStream(engine, language=case["language"])
    events: list[dict] = []

    original_find = stream._find_pause_cut
    original_run = stream._run_decoder
    original_decode = stream._decode_final
    original_commit = stream._commit_window_overflow

    def find_cut():
        cut = original_find()
        events.append({
            "event": "find_pause_cut",
            "frames": int(stream._total_encoder_frames),
            "cut": None if cut is None else int(cut),
            "cut_search_frames": int(stream._cut_search_frames),
        })
        return cut

    def run_decoder(full_embd, n_tokens, early_stop):
        result = original_run(full_embd, n_tokens, early_stop)
        events.append({
            "event": "decoder",
            "n_tokens": int(n_tokens),
            "early_stop": int(early_stop),
            "raw_text": str(result.get("text", "")),
            "aborted": bool(result.get("aborted", False)),
            "abort_reason": result.get("abort_reason", ""),
            "n_tokens_generated": result.get("n_tokens_generated"),
            "perf": result.get("perf") or {},
        })
        return result

    def decode_final(all_frames, stop_on_punctuation=True):
        before = len(events)
        text = original_decode(all_frames, stop_on_punctuation=stop_on_punctuation)
        events.append({
            "event": "decode_final",
            "frames": int(all_frames.shape[0]),
            "stop_on_punctuation": bool(stop_on_punctuation),
            "text": text,
            "decoder_events_since_previous": len(events) - before,
        })
        return text

    def commit_window():
        before_text = stream._window_committed_text
        before_frames = int(stream._total_encoder_frames)
        before_commits = int(stream._window_commits)
        ok = original_commit()
        events.append({
            "event": "window_commit",
            "ok": bool(ok),
            "frames_before": before_frames,
            "commits_before": before_commits,
            "commits_after": int(stream._window_commits),
            "text_before": before_text,
            "text_after": stream._window_committed_text,
            "pause_cuts": int(stream._pause_cuts),
            "seam_overlap": bool(stream._seam_overlap),
        })
        return ok

    stream._find_pause_cut = find_cut
    stream._run_decoder = run_decoder
    stream._decode_final = decode_final
    stream._commit_window_overflow = commit_window

    feed_start = time.perf_counter()
    for start in range(0, len(audio), chunk_n):
        t0 = time.perf_counter()
        stream.feed_audio(audio[start:start + chunk_n])
        if realtime:
            time.sleep(max(0.0, chunk_ms / 1000 - (time.perf_counter() - t0)))
    feed_wall_ms = (time.perf_counter() - feed_start) * 1000
    final_start = time.perf_counter()
    result = stream.finish()
    eos_to_final_ms = (time.perf_counter() - final_start) * 1000
    return {
        "case_id": case["id"],
        "language": case["language"],
        "wav": str(wav),
        "wav_sha256": sha256(wav),
        "feed_sha256": feed_sha256,
        "sample_rate": sr,
        "duration_s": len(audio) / sr,
        "reference_kind": "prompt-reference-not-human-ground-truth",
        "reference_text": case.get("reference_text", ""),
        "group": group,
        "chunk_ms": chunk_ms,
        "realtime": realtime,
        "feed_wall_ms": feed_wall_ms,
        "eos_to_final_ms": eos_to_final_ms,
        "final_text": result.get("text", ""),
        "stats": result.get("stats", {}),
        "stream_state": {
            "window_commits": int(stream._window_commits),
            "pause_cuts": int(stream._pause_cuts),
            "window_committed_text": stream._window_committed_text,
            "last_final_abort_reason": stream._last_final_abort_reason,
            "window_commit_error": stream._window_commit_error,
            "total_encoder_frames": int(stream._total_encoder_frames),
            "n_chunks": int(stream._n_chunks),
        },
        "effective_decoder": {
            key: getattr(engine.decoder, key, None)
            for key in (
                "_final_stop_on_punctuation", "_final_stop_min_chars",
                "_final_stop_min_chunks", "_max_context_len", "_max_new_tokens",
                "_embed_cache_reuse", "_async_mode",
            )
        },
        "events": events,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-wav")
    ap.add_argument("--mixed-wav")
    ap.add_argument("--long-reference", default="")
    ap.add_argument("--mixed-reference", default="")
    ap.add_argument("--model-dir", default=os.environ.get("ASR_MODEL_DIR", "/opt/asr/models"))
    ap.add_argument("--lib-path", default=os.environ.get("RKLLM_LIB_PATH", "/opt/asr/lib/librkllmrt.so"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--cases-json", help="JSON list of case objects; replaces the default two cases")
    ap.add_argument("--groups-json", help="JSON list of group objects; replaces the default groups")
    args = ap.parse_args()

    groups = [
        {"id": "roll20_baseline", "env": {
            "QWEN3_ASR_TRUE_ROLL_SEC": "20",
            "QWEN3_ASR_TRUE_ROLL_CUT_SEARCH_SEC": "2",
            "QWEN3_ASR_TRUE_ROLL_OVERLAP_SEC": "1",
        }},
        {"id": "roll5_cut2_overlap1", "env": {
            "QWEN3_ASR_TRUE_ROLL_SEC": "5",
            "QWEN3_ASR_TRUE_ROLL_CUT_SEARCH_SEC": "2",
            "QWEN3_ASR_TRUE_ROLL_OVERLAP_SEC": "1",
        }},
        {"id": "roll5_cut0_overlap1", "env": {
            "QWEN3_ASR_TRUE_ROLL_SEC": "5",
            "QWEN3_ASR_TRUE_ROLL_CUT_SEARCH_SEC": "0",
            "QWEN3_ASR_TRUE_ROLL_OVERLAP_SEC": "1",
        }},
        # Explicit diagnostic only: compare a larger no-pause acoustic overlap
        # without changing the candidate's default.
        {"id": "roll5_cut0_overlap2", "env": {
            "QWEN3_ASR_TRUE_ROLL_SEC": "5",
            "QWEN3_ASR_TRUE_ROLL_CUT_SEARCH_SEC": "0",
            "QWEN3_ASR_TRUE_ROLL_OVERLAP_SEC": "2",
        }},
    ]
    if args.groups_json:
        groups = json.loads(Path(args.groups_json).read_text(encoding="utf-8"))
    if args.cases_json:
        cases = json.loads(Path(args.cases_json).read_text(encoding="utf-8"))
    else:
        if not args.long_wav or not args.mixed_wav:
            ap.error("--long-wav and --mixed-wav are required unless --cases-json is used")
        cases = [
            {"id": "longtail", "wav": args.long_wav, "language": "English", "reference_text": args.long_reference},
            {"id": "mixed", "wav": args.mixed_wav, "language": "Chinese", "reference_text": args.mixed_reference},
        ]
    # Match the production backend's spelling for this setting.  The helper
    # historically used ASR_FINAL_STOP_ON_PUNCTUATION while qwen3_rk.py uses
    # ASR_FINAL_STOP_ON_PUNCT; default the direct diagnostic to the production
    # false value unless the caller explicitly supplies one.
    punct = os.environ.get("ASR_FINAL_STOP_ON_PUNCT", "0")
    os.environ["ASR_FINAL_STOP_ON_PUNCTUATION"] = punct

    # Build one decoder/engine and reuse it for all stream instances.
    ns = argparse.Namespace(
        model_dir=args.model_dir, platform=os.environ.get("ASR_PLATFORM", "rk3576"),
        lib_path=args.lib_path, decoder_quant=os.environ.get("ASR_DECODER_QUANT", "w8a8"),
        npu_core_mask=os.environ.get("ASR_NPU_CORE_MASK", "NPU_CORE_1"),
        enabled_cpus=int(os.environ.get("ASR_ENABLED_CPUS", "4")),
        max_context_len=int(os.environ.get("ASR_MAX_CONTEXT_LEN", "512")),
        max_new_tokens=int(os.environ.get("ASR_MAX_NEW_TOKENS", "64")),
    )
    engine = build_engine(ns)
    from rkvoice_stream.backends.asr.qwen3.streaming import Qwen3TrueStreamingASRStream
    source_path = Path(__import__(
        "inspect").getsourcefile(Qwen3TrueStreamingASRStream)
        or "rkvoice_stream/backends/asr/qwen3/streaming.py")
    source_sha256 = sha256(source_path)
    rows = []
    rows_path = Path(str(args.out) + ".jsonl")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.unlink(missing_ok=True)
    run_status = "passed"
    close_error = None
    try:
        for group in groups:
            for case in cases:
                try:
                    row = run_case(engine, case, group, args.chunk_ms, args.realtime)
                except Exception as exc:
                    run_status = "failed"
                    row = {
                        "case_id": case.get("id", ""), "language": case.get("language", ""),
                        "wav": case.get("wav", ""), "group": group, "status": "failed",
                        "error_type": type(exc).__name__, "error": str(exc),
                    }
                row.update({
                    "status": row.get("status", "passed"),
                    "argv": sys.argv,
                    "source_path": str(source_path),
                    "source_sha256": source_sha256,
                })
                rows.append(row)
                with rows_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
    finally:
        try:
            engine.close()
        except Exception as exc:
            run_status = "failed"
            close_error = {"type": type(exc).__name__, "error": str(exc)}
    payload = {
        "runner": "qwen3_asr_window_diagnostic",
        "status": run_status,
        "argv": sys.argv,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "groups": groups,
        "chunk_ms": args.chunk_ms,
        "realtime": args.realtime,
        "model_dir": args.model_dir,
        "rows": rows,
    }
    if close_error:
        payload["close_error"] = close_error
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out, "rows": len(rows), "status": run_status,
                      "model_dir": args.model_dir}, ensure_ascii=False))
    return 0 if run_status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
