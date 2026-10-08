#!/usr/bin/env python3
"""Bounded local Piper DP-bucket A/B measurement.

This runner measures one already assembled model directory at a time.  The
directory must contain the normal Piper bundle (config, frontend artifacts,
remainder and flow decoder); DP256's extracted/compact files are deliberately
not copied or assembled here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

EXPECTED_PIPER_SHA = "735c4d30b79bb0f5f53b40ab7068ae5678dae705d64a2417516a5dc709a9b082"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", action="append", required=True,
                   help="complete Piper language bundle; repeat for A/B bundles")
    p.add_argument("--label", action="append", default=[],
                   help="label matching each --model-dir (default: directory name)")
    p.add_argument("--cases", default="bench/perf/corpus/piper_dp_eval_prompts.json")
    p.add_argument("--case-count", type=int, default=8)
    p.add_argument("--case-ids", default=None,
                   help="comma-separated case ids; overrides --case-count")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--noise-scale", type=float, default=None)
    p.add_argument("--noise-w", type=float, default=None)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--frontend-mode", choices=("npu", "cpu"), default="npu",
                   help="frontend execution mode (default: npu)")
    p.add_argument("--save-all-wavs", action="store_true",
                   help="save sync and stream WAVs for every repetition")
    p.add_argument("--seed", type=int, default=None,
                   help="seed ONNX Runtime and NumPy before model sessions are built")
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_cases(path: Path, count: int) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("prompts", data) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("cases must contain a non-empty prompts list")
    return rows[:count]


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * fraction
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def resolve_actual_mode(model) -> str:
    """Report the production model route from its loaded flags."""
    if getattr(model, "_frontend_npu", False):
        return "frontend_npu"
    if getattr(model, "_hybrid", False):
        return "hybrid"
    return "legacy"


def bundle_files(root: Path, piper_source: Path, frontend_mode: str = "npu") -> dict[str, str]:
    if frontend_mode == "cpu":
        required = ["encoder.onnx", "model.onnx.json", "flow_decoder.rknn"]
    else:
        required = ["manifest.json", "model.onnx.json", "text_encoder.rknn",
                    "remainder.onnx", "flow_decoder.rknn"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete Piper frontend bundle {root}: missing {', '.join(missing)}"
        )
    out = {name: sha256(root / name) for name in required}
    out["piper_source.py"] = sha256(piper_source)
    return out


def run_bundle(root: Path, label: str, cases: list[dict], args: argparse.Namespace) -> dict:
    source_root_env = os.environ.get("PIPER_SOURCE_ROOT")
    source_root = Path(source_root_env) if source_root_env else None
    os.environ["PIPER_MODEL_DIR"] = str(root.parent)
    os.environ["PIPER_ENABLE_FRONTEND_NPU"] = "1" if args.frontend_mode == "npu" else "0"
    os.environ["PIPER_LANGUAGES"] = root.name
    if args.seed is not None:
        import onnxruntime as ort
        ort.set_seed(args.seed)
        np.random.seed(args.seed)
    if source_root is not None and source_root.is_dir():
        sys.path.insert(0, str(source_root))
    import rkvoice_stream.backends.tts.piper as piper_module
    piper_source = Path(piper_module.__file__).resolve()
    piper_sha = sha256(piper_source)
    if piper_sha != EXPECTED_PIPER_SHA:
        raise RuntimeError(f"unexpected piper runtime SHA {piper_sha}")
    files = bundle_files(root, piper_source, args.frontend_mode)
    # The production module reads these at import time.  Override only the
    # module globals for this isolated measurement so repeated --model-dir
    # bundles remain in one process without changing production code.
    piper_module.MODEL_DIR = str(root.parent)
    piper_module.PRELOAD_LANGS = [root.name]
    piper_module.DEFAULT_LANG = root.name
    PiperRKNNBackend = piper_module.PiperRKNNBackend

    backend = PiperRKNNBackend()
    preload_started = time.perf_counter()
    backend.preload()
    preload_ms = (time.perf_counter() - preload_started) * 1000
    try:
        if not backend.is_ready():
            raise RuntimeError(f"Piper bundle is not ready: {root}")
        if args.noise_w is not None:
            for model in backend._models.values():
                if hasattr(model, "noise_w"):
                    model.noise_w = args.noise_w
        model = backend._models[root.name]
        actual_mode = resolve_actual_mode(model)
        model_params = {"seq_len": int(model.seq_len), "noise_scale": float(model.noise_scale),
                        "noise_w": float(model.noise_w), "length_scale": float(model.length_scale),
                        "sample_rate": int(model.sample_rate), "actual_mode": actual_mode}
        rows = []
        out_dir = Path(args.output).with_suffix("") / label
        out_dir.mkdir(parents=True, exist_ok=True)
        partial_path = Path(args.output).with_suffix(Path(args.output).suffix + f".{label}.partial.json")
        for case in cases:
            try:
                text = str(case["text"])
                language = case.get("lang")
                kwargs = {}
                if args.noise_scale is not None:
                    kwargs["noise_scale"] = args.noise_scale
                if args.noise_w is not None:
                    model.noise_w = args.noise_w
                segment_source = "frontend_segments"
                if not hasattr(piper_module, "_frontend_segments"):
                    raise RuntimeError("production piper runtime lacks _frontend_segments")
                segments = list(piper_module._frontend_segments(text, model, {}))
                segment_tokens = []
                for segment in segments:
                    ph = piper_module.text_to_phonemes(segment, model.espeak_voice)
                    segment_tokens.append(len(piper_module.phonemes_to_ids(ph, model.phoneme_id_map)))
                rejoined = " ".join(segments)
                normalized = " ".join(text.split())
                compact = lambda value: "".join(value.split())
                if compact(rejoined) != compact(normalized):
                    raise RuntimeError(f"segment rejoin mismatch for {case.get('id')}")
                for _ in range(args.warmup):
                    backend.synthesize(text, speed=args.speed, language=language, **kwargs)
                sync_ms, stream_ms, sample_counts, sample_rates = [], [], [], []
                stream_sample_counts, stream_last_samples, stream_rates = [], [], []
                finite, token_counts, segment_counts, stream_ttfa_ms = True, [], [], []
                sync_wav = stream_wav = None
                rep_records = []
                for rep in range(args.reps):
                    effective_noise_scale = (args.noise_scale if args.noise_scale is not None
                                             else float(model.noise_scale))
                    effective_noise_w = float(args.noise_w if args.noise_w is not None
                                              else model.noise_w)
                    rep_params = {
                        "rep": rep, "noise_scale": effective_noise_scale,
                        "noise_w": effective_noise_w,
                        "length_scale": float(model.length_scale / max(args.speed, 0.1)),
                        "seq_len": int(model.seq_len), "actual_mode": actual_mode,
                        "segment_tokens": segment_tokens,
                    }
                    t0 = time.perf_counter()
                    wav, meta = backend.synthesize(text, speed=args.speed, language=language, **kwargs)
                    sync_ms.append((time.perf_counter() - t0) * 1000)
                    audio, rate = sf.read(__import__("io").BytesIO(wav), dtype="float32")
                    if int(rate) != 22050 or audio.shape[0] == 0:
                        raise RuntimeError(f"invalid sync audio rate/samples: {rate}/{audio.shape[0]}")
                    sample_counts.append(int(audio.shape[0])); sample_rates.append(int(rate))
                    if not bool(np.isfinite(audio).all()) or meta.get("error"):
                        raise RuntimeError(f"invalid sync output for case {case.get('id')}")
                    finite = finite and True
                    token_counts.append(int(meta.get("num_tokens", 0)))
                    if rep == 0: sync_wav = audio
                    t0 = time.perf_counter(); streamed = []; first_ms = None; stream_meta = []
                    for chunk, chunk_meta in backend.synthesize_stream(text, speed=args.speed, language=language, **kwargs):
                        if chunk_meta.get("error"):
                            raise RuntimeError(f"stream metadata error: {chunk_meta['error']}")
                        if first_ms is None: first_ms = (time.perf_counter() - t0) * 1000
                        streamed.append(np.asarray(chunk, dtype=np.float32)); stream_meta.append(chunk_meta)
                    stream_ms.append((time.perf_counter() - t0) * 1000)
                    if first_ms is None or not streamed: raise RuntimeError(f"empty Piper stream for case {case.get('id')}")
                    chunk_sizes = [int(x.shape[0]) for x in streamed]
                    if not all(bool(np.isfinite(x).all()) for x in streamed):
                        raise RuntimeError(f"non-finite stream output for case {case.get('id')}")
                    stream_ttfa_ms.append(first_ms); finite = finite and True
                    segment_counts.append(len(streamed))
                    stream_sample_counts.append(chunk_sizes)
                    stream_last_samples.append(chunk_sizes[-1])
                    stream_rates.append(sorted({int(m.get("sample_rate", 0)) for m in stream_meta}))
                    if rep == 0: stream_wav = np.concatenate(streamed)
                    rep_record = {
                        **rep_params,
                        "sync_ms": sync_ms[-1], "stream_ms": stream_ms[-1],
                        "stream_ttfa_ms": stream_ttfa_ms[-1],
                        "samples": sample_counts[-1],
                        "stream_samples": chunk_sizes,
                    }
                    if args.save_all_wavs:
                        rep_sync_path = out_dir / f"{case.get('id', 'case')}.rep{rep}.sync.wav"
                        rep_stream_path = out_dir / f"{case.get('id', 'case')}.rep{rep}.stream.wav"
                        sf.write(rep_sync_path, audio, 22050, subtype="PCM_16")
                        sf.write(rep_stream_path, np.concatenate(streamed), 22050, subtype="PCM_16")
                        rep_record["sync_wav"] = {"path": str(rep_sync_path), "sha256": sha256(rep_sync_path)}
                        rep_record["stream_wav"] = {"path": str(rep_stream_path), "sha256": sha256(rep_stream_path)}
                    rep_records.append(rep_record)
                if sync_wav is None or stream_wav is None: raise RuntimeError("missing first-rep audio")
                sync_path = out_dir / f"{case.get('id','case')}.sync.wav"
                stream_path = out_dir / f"{case.get('id','case')}.stream.wav"
                sf.write(sync_path, sync_wav, 22050, subtype="PCM_16"); sf.write(stream_path, stream_wav, 22050, subtype="PCM_16")
                row = {
                "id": case.get("id"), "lang": language, "text": text,
                "segment_source": segment_source, "segment_texts": segments,
                "segment_tokens": segment_tokens,
                "tokens": token_counts, "segments": segment_counts,
                "samples": sample_counts, "sample_rates": sample_rates, "finite": finite,
                "stream_samples": stream_sample_counts, "stream_last_samples": stream_last_samples,
                "stream_sample_rates": stream_rates,
                "sync_wav": {"path": str(sync_path), "sha256": sha256(sync_path)},
                "stream_wav": {"path": str(stream_path), "sha256": sha256(stream_path)},
                "actual_mode": actual_mode, "model_params": model_params,
                "rep_params": rep_records,
                "sync_ms": {"p50": statistics.median(sync_ms), "p90": percentile(sync_ms, .9)},
                "stream_ms": {"p50": statistics.median(stream_ms), "p90": percentile(stream_ms, .9)},
                "stream_ttfa_ms": {"p50": statistics.median(stream_ttfa_ms),
                                   "p90": percentile(stream_ttfa_ms, .9)},
                }
                if args.save_all_wavs:
                    row["sync_wavs"] = [rep["sync_wav"] for rep in rep_records]
                    row["stream_wavs"] = [rep["stream_wav"] for rep in rep_records]
            except Exception as exc:
                row = {"id": case.get("id"), "error": type(exc).__name__, "detail": str(exc)}
            rows.append(row)
            partial_path.write_text(json.dumps({"label": label, "model_dir": str(root),
                                                "files": files, "model_params": model_params,
                                                "rows": rows}, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        return {"label": label, "model_dir": str(root), "frontend_mode": args.frontend_mode,
                "files": files,
                "preload_ms": preload_ms, "warmup": args.warmup,
                "reps": args.reps, "noise_scale": args.noise_scale,
                "noise_w": args.noise_w, "seed": args.seed,
                "seed_note": ("onnxruntime and NumPy were seeded; CPU/NPU graphs are not\n"
                              "guaranteed to consume identical noise sequences"
                              if args.seed is not None else None),
                "save_all_wavs": args.save_all_wavs, "model_params": model_params, "rows": rows}
    finally:
        backend.cleanup()


def main() -> int:
    args = parse_args()
    if len(args.label) not in (0, len(args.model_dir)):
        raise SystemExit("--label must be supplied once per --model-dir")
    labels = args.label or [Path(x).name for x in args.model_dir]
    cases = load_cases(Path(args.cases), args.case_count)
    if args.case_ids:
        wanted = {x.strip() for x in args.case_ids.split(",") if x.strip()}
        cases = [x for x in cases if x.get("id") in wanted]
        if len(cases) != len(wanted):
            all_cases = load_cases(Path(args.cases), 10000)
            cases = [x for x in all_cases if x.get("id") in wanted]
        if len(cases) != len(wanted):
            missing = sorted(wanted - {x.get("id") for x in cases})
            raise SystemExit(f"unknown case ids: {', '.join(missing)}")
    if not cases:
        raise SystemExit("no evaluation cases selected")
    results = [run_bundle(Path(root).resolve(), label, cases, args)
               for root, label in zip(args.model_dir, labels)]
    out = {"cases_file": str(Path(args.cases).resolve()), "case_count": len(cases),
           "frontend_mode": args.frontend_mode, "seed": args.seed,
           "seed_note": ("onnxruntime and NumPy were seeded; CPU/NPU graphs are not\n"
                         "guaranteed to consume identical noise sequences"
                         if args.seed is not None else None),
           "results": results}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
    failed = [(x["label"], row.get("id"), row.get("error"))
              for x in results for row in x["rows"] if row.get("error")]
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "bundles": [x["label"] for x in results]}, ensure_ascii=False))
    if failed:
        print(json.dumps({"case_errors": failed}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
