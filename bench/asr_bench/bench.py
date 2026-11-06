#!/usr/bin/env python3
"""Generalized ASR benchmark for OpenVoiceStream's /asr/stream WebSocket.

Sends each audio segment as a "pseudo-streaming, non-streaming" pass: the
whole segment is fed in fixed-size binary chunks at 1.0x real-time pace (like
a live mic), then an empty binary frame signals end-of-segment, and the
client waits for the server's final. `?vad=none` is pinned so the client's
EOS frame is the only endpoint detector — see the "--server-vad" flag and
`server/main.py`'s "Streaming ASR endpointing" note for why running the
server VAD alongside a client that sends EOS loses text. This matches how
docs/perf-test-runbook.md and bench/perf/asr_stream_ws_bench.py already
exercise the service; this script generalizes that single-shot pattern to
N concurrent sessions and reports percentiles + RTF + CER/WER.

Protocol note (fact-checked 2026-09-09): this hits OpenVoiceStream's own
`/asr/stream` WebSocket (JSON keyed by `is_final`, no `type: connection/vad`
envelope). The `ws_api.md` doc referenced in the task describes a DIFFERENT
service (`sensecraft-asr-service`, Go, port 8080, `/ws`, `type:
connection|vad|final|error`) that sits in front of OpenVoiceStream as an app
compatibility shim in the retail_voice compose stack. There is no evidence
that shim exposes per-model timing/CER data any differently than the engine
underneath, and benchmarking model backends (SenseVoice vs Whisper) needs the
engine's own endpoint, not the shim. See docs/reports/retail-voice-asr-bench-matrix-2026-09-09.md
for the full trace (file:line citations for both services).

Usage:
    uv run bench.py --url ws://<rk3576-host>:8621 --model sensevoice --lang zh \
        --segments ../perf/corpus --concurrency 1
    uv run bench.py --url ws://<rk3576-host>:8621 --model sensevoice --lang zh \
        --segments ../perf/corpus --concurrency 1,2,4,8 --out results/rk3576-sensevoice-zh.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

try:
    import jiwer
except ImportError:  # pragma: no cover
    jiwer = None


# ---------------------------------------------------------------------------
# Corpus loading — reuses bench/perf/corpus/manifest.json's schema, and also
# accepts a flat "segments dir" with a manifest.json of the same shape (see
# bench/asr_bench/corpus/README.md for the AISHELL-1 / LibriSpeech subsets).
# ---------------------------------------------------------------------------

def load_items(segments_dir: Path, lang: str, category: str | None, limit: int | None) -> list[dict]:
    manifest_path = segments_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [f for f in manifest["files"] if f["lang"] == lang]
    if category:
        items = [f for f in items if f.get("category") == category]
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit(f"No items for lang={lang} category={category} in {manifest_path}")
    return items


#: Cap on the gap between two server messages once the first final has
#: arrived. Not applied before it, so a slow decode is not cut off.
_IDLE_GAP_S = 5.0


def load_pcm16(path: Path) -> bytes:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        x_old = np.linspace(0, len(audio) - 1, len(audio))
        x_new = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# Error rate
# ---------------------------------------------------------------------------

def error_rate(ref: str, hyp: str, lang: str) -> float:
    """CER for zh, WER for en, both after the same normalization on both sides.

    English references in LibriSpeech are uppercase and unpunctuated while ASR
    hypotheses carry ordinary casing and punctuation. Scoring them as-is
    charges every capital letter and every period as a recognition error
    (`HELLO WORLD` vs `Hello world.` scores WER 1.0), so case, punctuation and
    repeated whitespace are removed from reference and hypothesis alike before
    tokenization.
    """
    if jiwer is None:
        raise SystemExit(
            "jiwer is required for CER/WER scoring (declared in pyproject.toml). "
            "Install it — `uv sync` — instead of falling back to a different metric."
        )
    ref = ref.strip()
    hyp = hyp.strip()
    if not ref:
        return 0.0
    if lang == "zh":
        tr = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveWhiteSpace(replace_by_space=""),
            jiwer.ReduceToListOfListOfChars(),
        ])
        return jiwer.cer(ref, hyp, reference_transform=tr, hypothesis_transform=tr)
    tr = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    return jiwer.wer(ref, hyp, reference_transform=tr, hypothesis_transform=tr)


# ---------------------------------------------------------------------------
# One segment, one WS session
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    id: str
    lang: str
    duration_s: float
    feed_wall_ms: float
    eos_to_final_ms: float
    rtf: float
    text: str
    ref: str
    err: float
    ok: bool
    error: str | None = None
    # Number of `is_final` messages the server had already emitted before EOS.
    # Non-zero means decoding overlapped feeding, so `rtf` for that segment is
    # not a decode-only figure.
    pre_eos_finals: int = 0


async def run_segment(url: str, item: dict, segments_dir: Path, chunk_bytes: int, realtime: bool,
                      api_key: str = "", server_vad: bool = False) -> SegmentResult:
    duration_s = float(item.get("duration_s") or 0.0)
    ref = item.get("eval_transcript") or item["transcript"]
    lang = item["lang"]
    ws_url = f"{url.rstrip('/')}/asr/stream?language=auto&sample_rate=16000"
    # ENDPOINTING: pick exactly one detector. This client sends the EOS frame,
    # so it IS the detector; leaving the server VAD on as well is the
    # mutually-exclusive misconfiguration documented at server/main.py:5208
    # ("the detectors race ... text goes missing"). Under queueing the server's
    # mid-utterance final is delivered AFTER our EOS, and a client that stops
    # at the first is_final scores that fragment as the whole segment. Measured
    # on J4012 at c=24: 24 of 72 matched items came back as a strict prefix of
    # their c=1 text. ``--server-vad`` keeps the open-mic mode available; the
    # collection loop below then accumulates every final, which is what the
    # endpoint's contract requires of an open-mic client.
    if not server_vad:
        ws_url += "&vad=none"
    # OVS rejects the upgrade with 403 when OVS_API_KEYS is set and no key is
    # presented, which looks identical to a saturated backend in the results.
    if api_key:
        ws_url += f"&token={api_key}"

    try:
        # Decoding/resampling is blocking work; keep it off the event loop so a
        # concurrent sweep is not serialized by it, and keep it inside the
        # handler so one missing or corrupt WAV cannot abort the whole run.
        wav_path = segments_dir / item["filename"]
        pcm = await asyncio.to_thread(load_pcm16, wav_path)
        async with websockets.connect(ws_url, max_size=None, open_timeout=15) as ws:
            feed_start = time.perf_counter()
            bytes_per_ms = 16000 * 2 / 1000.0  # 16-bit mono @16kHz
            for start in range(0, len(pcm), chunk_bytes):
                chunk = pcm[start:start + chunk_bytes]
                t0 = time.perf_counter()
                await ws.send(chunk)
                if realtime:
                    chunk_ms = len(chunk) / bytes_per_ms
                    elapsed = (time.perf_counter() - t0) * 1000
                    await asyncio.sleep(max(0.0, (chunk_ms - elapsed) / 1000))
            feed_wall_ms = (time.perf_counter() - feed_start) * 1000

            # Drain whatever the server already queued while audio was being
            # fed. With ``vad=none`` nothing can arrive here; under
            # ``--server-vad`` these are real mid-utterance segments and the
            # count is reported so the caller knows decoding overlapped feeding.
            # This drain is NOT the place that guarantees completeness — a
            # server final delayed past EOS by queueing lands in the post-EOS
            # loop below, which is why that loop accumulates rather than
            # returning on the first one.
            pre_eos_finals: list[str] = []
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.01)
                except (asyncio.TimeoutError, TimeoutError):
                    break
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if msg.get("is_final") and msg.get("text"):
                    pre_eos_finals.append(msg["text"])

            eos_at = time.perf_counter()
            await ws.send(b"")  # empty frame = end-of-segment
            # Read until the server closes the stream, accumulating EVERY
            # final. The server closes right after the segment's EOS final, so
            # the close is the completion signal — not the absence of an
            # ``endpoint`` field, which a backend-driven midstream final
            # (server/main.py, the get_partial is_endpoint path) also lacks.
            # Returning on the first final silently dropped the rest of the
            # utterance whenever the server had a queued mid-utterance final
            # still to deliver.
            post_eos_finals: list[str] = []
            first_final_at = None
            stream_closed = False
            close_error = None
            server_error = None
            deadline = eos_at + 40
            # A gap cap, applied only once something has arrived. Before the
            # first final the whole deadline is available, so a slow decode is
            # not cut off at 5 s.
            idle_timeout = _IDLE_GAP_S
            while time.perf_counter() < deadline:
                remaining = max(0.1, deadline - time.perf_counter())
                budget = min(idle_timeout, remaining) if first_final_at else remaining
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=budget)
                except (asyncio.TimeoutError, TimeoutError):
                    break
                except websockets.ConnectionClosedOK:
                    stream_closed = True
                    break
                except websockets.ConnectionClosed as exc:
                    close_error = f"stream closed abnormally: {exc}"
                    break
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                # The server's terminal error frame carries is_final too
                # (server/main.py), and ``busy`` means the utterance was
                # dropped for backpressure. Either way this segment has no
                # transcript; recording it as a result would score an empty
                # or partial string as if the run had succeeded.
                kind = msg.get("type")
                if kind in ("error", "busy"):
                    server_error = f"server sent {kind}: {msg.get('reason') or msg.get('text') or ''}".strip()
                    continue
                if not msg.get("is_final"):
                    continue
                if first_final_at is None:
                    first_final_at = time.perf_counter()
                if msg.get("text"):
                    post_eos_finals.append(msg["text"])
            eos_to_final_ms = ((first_final_at or time.perf_counter()) - eos_at) * 1000
            joiner = "" if lang == "zh" else " "
            text = joiner.join([t for t in (*pre_eos_finals, *post_eos_finals) if t])
            # Complete means the server sent at least one final, sent no error
            # or busy frame, and closed the stream without an error code —
            # which it does only after the segment's own EOS final.
            complete = (first_final_at is not None and stream_closed
                        and server_error is None)
            err = error_rate(ref, text, lang)
            rtf = (eos_to_final_ms / 1000.0) / duration_s if duration_s > 0 else float("nan")
            return SegmentResult(
                id=item["id"], lang=lang, duration_s=duration_s,
                feed_wall_ms=feed_wall_ms, eos_to_final_ms=eos_to_final_ms,
                rtf=rtf, text=text, ref=ref, err=err, ok=complete,
                error=None if complete else (
                    server_error or close_error or (
                        "no final message before deadline" if first_final_at is None
                        else "server did not close the stream after EOS; the "
                             "transcript may be incomplete")),
                pre_eos_finals=len(pre_eos_finals),
            )
    except Exception as exc:  # noqa: BLE001 — record and continue, don't crash the run
        return SegmentResult(
            id=item["id"], lang=lang, duration_s=duration_s,
            feed_wall_ms=0.0, eos_to_final_ms=0.0, rtf=float("nan"),
            text="", ref=ref, err=1.0, ok=False, error=str(exc),
        )


# ---------------------------------------------------------------------------
# Concurrency runner: N workers each drain a shared queue of items once.
# ---------------------------------------------------------------------------

async def run_concurrency(url: str, items: list[dict], segments_dir: Path, concurrency: int,
                           chunk_bytes: int, realtime: bool, api_key: str = "",
                           server_vad: bool = False) -> dict:
    queue: asyncio.Queue = asyncio.Queue()
    for it in items:
        queue.put_nowait(it)
    results: list[SegmentResult] = []
    lock = asyncio.Lock()

    async def worker():
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            r = await run_segment(url, item, segments_dir, chunk_bytes, realtime, api_key, server_vad)
            async with lock:
                results.append(r)

    wall_start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall_s = time.perf_counter() - wall_start

    ok_results = [r for r in results if r.ok]
    err_count = len(results) - len(ok_results)
    lat = [r.eos_to_final_ms for r in ok_results]
    rtfs = [r.rtf for r in ok_results if r.rtf == r.rtf]  # filter NaN
    errs = [r.err for r in ok_results]

    def pct(values: list[float], p: float) -> float | None:
        if not values:
            return None
        s = sorted(values)
        k = (len(s) - 1) * p
        f, c = int(k), min(int(k) + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    total_audio_s = sum(r.duration_s for r in ok_results)
    return {
        "concurrency": concurrency,
        "segments": len(items),
        "ok": len(ok_results),
        "errors": err_count,
        "wall_s": wall_s,
        "throughput_segments_per_s": len(ok_results) / wall_s if wall_s > 0 else None,
        "throughput_audio_rtf_aggregate": total_audio_s / wall_s if wall_s > 0 else None,
        "final_latency_ms_p50": pct(lat, 0.50),
        "final_latency_ms_p95": pct(lat, 0.95),
        "final_latency_ms_mean": statistics.fmean(lat) if lat else None,
        "rtf_p50": pct(rtfs, 0.50),
        "rtf_p95": pct(rtfs, 0.95),
        "error_rate_mean": statistics.fmean(errs) if errs else None,
        "results": [r.__dict__ for r in results],
    }


def to_markdown(model: str, lang: str, url: str, runs: list[dict]) -> str:
    lines = [
        f"# ASR bench: {model} / {lang}",
        "",
        f"- Target: `{url}`",
        f"- Metric: final latency = audio-end -> `is_final` received (ms); RTF = final_latency / audio_duration",
        "",
        "| Concurrency | Segments | OK | Errors | p50 latency (ms) | p95 latency (ms) | RTF p50 | RTF p95 | Throughput (seg/s) | Mean error rate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        def f(v, nd=1):
            return "-" if v is None else f"{v:.{nd}f}"
        lines.append(
            f"| {r['concurrency']} | {r['segments']} | {r['ok']} | {r['errors']} | "
            f"{f(r['final_latency_ms_p50'])} | {f(r['final_latency_ms_p95'])} | "
            f"{f(r['rtf_p50'],3)} | {f(r['rtf_p95'],3)} | "
            f"{f(r['throughput_segments_per_s'],2)} | {f(r['error_rate_mean'],4)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="ws://host:port (no path) — /asr/stream is appended")
    p.add_argument("--api-key", default="", help="value for OVS_API_KEYS; without it a key-protected server rejects the upgrade with 403, which is indistinguishable from saturation in the results")
    p.add_argument("--model", required=True, choices=["sensevoice", "whisper"], help="label only, describes which OVS profile is running on --url")
    p.add_argument("--lang", required=True, choices=["zh", "en"])
    p.add_argument("--category", default=None, help="filter manifest 'category' (e.g. short/long); default: all")
    p.add_argument("--segments", required=True, help="dir containing manifest.json + wav files (see corpus/README.md)")
    p.add_argument("--concurrency", default="1", help="comma-separated list, e.g. 1,2,4,8")
    p.add_argument("--limit", type=int, default=None, help="cap segments per concurrency run")
    p.add_argument("--chunk-bytes", type=int, default=4096)
    p.add_argument("--no-realtime", action="store_true", help="send chunks as fast as possible instead of 1.0x real-time pace")
    p.add_argument("--server-vad", action="store_true", help="leave the server VAD on (open-mic mode) instead of pinning ?vad=none. The client still sends EOS, so the two detectors race — see server/main.py Streaming ASR endpointing. Only for measuring that mode deliberately.")
    p.add_argument("--out", default=None, help="write JSON results here; a sibling .md is also written")
    args = p.parse_args()

    segments_dir = Path(args.segments)
    items = load_items(segments_dir, args.lang, args.category, args.limit)
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    runs = []
    for c in levels:
        print(f"== concurrency={c} lang={args.lang} model={args.model} segments={len(items)} ==", flush=True)
        run = asyncio.run(run_concurrency(args.url, items, segments_dir, c, args.chunk_bytes,
                                          not args.no_realtime, args.api_key, args.server_vad))
        print(json.dumps({k: v for k, v in run.items() if k != "results"}, ensure_ascii=False, indent=2), flush=True)
        runs.append(run)

    payload = {
        "url": args.url,
        "model": args.model,
        "lang": args.lang,
        "category": args.category,
        "chunk_bytes": args.chunk_bytes,
        "realtime": not args.no_realtime,
        "runs": runs,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = out_path.with_suffix(".md")
        md_path.write_text(to_markdown(args.model, args.lang, args.url, runs), encoding="utf-8")
        print(f"wrote {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
