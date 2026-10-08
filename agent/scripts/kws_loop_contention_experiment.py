#!/usr/bin/env python3
"""Measure runtime-KWS detection loss as event-loop contention rises.

Runs the production classes — :class:`TappedAudioIO` for the mic fanout and
:class:`RuntimeKwsSource` for the spotter — against a fixed WAV, once per
event-loop load level, and reports how many mic chunks the KWS tap dropped
and how many wake words survived.

Why this shape:

* The mic feeder is a real thread that paces chunks in wall-clock time and
  hands them over with ``loop.call_soon_threadsafe``, exactly like the
  PortAudio input callback in :meth:`AudioIO.start_capture`. Feeding the
  queue from inside the loop would hide the effect under test, because a
  blocked loop would then also stop producing.
* ``RuntimeKwsSource.detect()`` is left exactly as production calls it
  (synchronously, on the loop thread). This script measures; it does not fix.
* The load generator burns CPU on the loop for ``duty`` of every slice, so
  ``--loads 0,20,50,80`` sweeps "loop is idle" to "loop is nearly saturated
  by other work" (ASR upload, TTS dispatch, WS traffic).

Typical run (inside the agent image, with this repo's ``agent/`` first on
PYTHONPATH so the tap drop counters are the instrumented ones)::

    python3 agent/scripts/kws_loop_contention_experiment.py \\
        --model-dir /opt/ovs/models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 \\
        --wav /data/human_zh.wav --phrase 你好小智 --expect 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

from ovs_agent.audio.tapped_audio_io import TappedAudioIO
from ovs_agent.kws import SherpaKwsBackend
from ovs_agent.wake_sources.runtime_kws import RuntimeKwsSource


class CountingKwsBackend(SherpaKwsBackend):
    """Production backend plus call/latency/raw-hit counters.

    Counted here rather than in the source so ``runtime_kws.py`` stays
    untouched: raw hits are what the spotter returned, while ``wakes`` below
    is what survived the source's cooldown.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.calls = 0
        self.raw_hits = 0
        self.detect_ms: list[float] = []

    def detect(self, stream, samples, sample_rate):
        started = time.perf_counter()
        keyword = super().detect(stream, samples, sample_rate)
        self.detect_ms.append((time.perf_counter() - started) * 1000.0)
        self.calls += 1
        if keyword:
            self.raw_hits += 1
        return keyword


class ProbeApp:
    """Minimal app surface RuntimeKwsSource needs: audio, config, wake()."""

    def __init__(self, audio, sample_rate: int) -> None:
        self.audio = audio
        self.config = SimpleNamespace(
            audio_input_sample_rate=sample_rate, wake_phrases=[]
        )
        self.wakes: list[float] = []

    async def wake(self, source: str = "") -> None:
        self.wakes.append(time.monotonic())


def read_pcm_chunks(wav_path: Path, chunk_ms: int) -> tuple[list[bytes], int, float]:
    with wave.open(str(wav_path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("experiment WAV must be mono 16-bit PCM")
        sample_rate = wav.getframerate()
        frames_per_chunk = int(sample_rate * chunk_ms / 1000)
        chunks: list[bytes] = []
        while data := wav.readframes(frames_per_chunk):
            chunks.append(data)
        duration_s = wav.getnframes() / sample_rate
    return chunks, sample_rate, duration_s


class MicFeeder(threading.Thread):
    """Wall-clock-paced stand-in for the PortAudio input callback thread."""

    def __init__(self, loop, audio, chunks, chunk_ms: int) -> None:
        super().__init__(name="mic-feeder", daemon=True)
        self._loop = loop
        self._audio = audio
        self._chunks = chunks
        self._period = chunk_ms / 1000.0
        self.done = threading.Event()
        self.late_ms: list[float] = []

    def run(self) -> None:
        start = time.perf_counter()
        for index, chunk in enumerate(self._chunks):
            deadline = start + index * self._period
            slack = deadline - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                self.late_ms.append(-slack * 1000.0)
            # Same handoff as AudioIO.start_capture's callback: the PCM is
            # produced off-loop and scheduled onto the loop thread.
            self._loop.call_soon_threadsafe(self._audio._safe_put, chunk)
        self.done.set()


async def _loop_lag_monitor(samples: list[float], interval_s: float = 0.02):
    """Record how far past its deadline each wakeup actually ran."""
    while True:
        started = time.perf_counter()
        await asyncio.sleep(interval_s)
        samples.append((time.perf_counter() - started - interval_s) * 1000.0)


async def _cpu_load(duty: float, slice_ms: float = 20.0):
    """Occupy the event loop for ``duty`` of every ``slice_ms`` window."""
    if duty <= 0:
        while True:
            await asyncio.sleep(3600)
    burn_s = slice_ms * duty / 1000.0
    idle_s = max(slice_ms * (1.0 - duty) / 1000.0, 0.0)
    while True:
        end = time.perf_counter() + burn_s
        acc = 0.0
        while time.perf_counter() < end:
            # Arbitrary arithmetic; the point is holding the loop thread.
            acc += 1.000000001
        await asyncio.sleep(idle_s)


async def _chatter(period_s: float = 0.005):
    """Constant await churn, standing in for WS/TTS callback traffic."""
    while True:
        await asyncio.sleep(period_s)


async def run_one(args, chunks, sample_rate: int, duty: float) -> dict:
    loop = asyncio.get_running_loop()
    model = args.model_dir
    audio = TappedAudioIO(input_sr=sample_rate, chunk_ms=args.chunk_ms)
    app = ProbeApp(audio, sample_rate)
    backend = CountingKwsBackend(
        {
            "tokens": str(model / "tokens.txt"),
            "encoder": str(model / args.encoder),
            "decoder": str(model / args.decoder),
            "joiner": str(model / args.joiner),
            "num_threads": args.num_threads,
            "keywords_score": args.score,
            "keywords_threshold": args.threshold,
            "num_trailing_blanks": 1,
        }
    )
    source = RuntimeKwsSource(
        app,
        phrases=[args.phrase],
        compiler_config={
            "tokens": str(model / "tokens.txt"),
            "lexicon": str(model / args.lexicon),
            "tokens_type": args.tokens_type,
        },
        backend=backend,
        cooldown_s=args.cooldown_s,
    )
    if not source.setup():
        raise RuntimeError("RuntimeKwsSource.setup() failed (model/phrase problem)")

    await source.start()
    # _run_once sleeps 0.5s before registering its tap; wait for the tap so
    # the feeder never runs against an unregistered fanout.
    deadline = time.monotonic() + 10.0
    while not audio.tap_stats() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    if not audio.tap_stats():
        raise RuntimeError("KWS tap never registered")

    lag_ms: list[float] = []
    background = [
        asyncio.create_task(_loop_lag_monitor(lag_ms), name="lag-monitor"),
        asyncio.create_task(_cpu_load(duty), name="sim-cpu-load"),
        asyncio.create_task(_chatter(), name="sim-ws-chatter"),
        asyncio.create_task(_chatter(), name="sim-tts-chatter"),
    ]

    silence = b"\x00" * len(chunks[0])
    tail = [silence] * int(args.tail_s * 1000 / args.chunk_ms)
    feeder = MicFeeder(loop, audio, chunks + tail, args.chunk_ms)
    wall_started = time.perf_counter()
    feeder.start()
    while not feeder.done.is_set():
        await asyncio.sleep(0.05)
    feed_end = time.monotonic()
    # Let the consumer finish whatever is still queued before reading counters.
    drain_deadline = time.monotonic() + args.drain_s
    while time.monotonic() < drain_deadline:
        if all(t["qsize"] == 0 for t in audio.tap_stats()):
            break
        await asyncio.sleep(0.05)
    wall_s = time.perf_counter() - wall_started

    stats = list(audio.tap_stats())
    for task in background:
        task.cancel()
    await asyncio.gather(*background, return_exceptions=True)
    await source.stop()

    tap = stats[0] if stats else {
        "name": "?", "offered": 0, "dropped": 0, "drop_ratio": 0.0, "qsize": 0
    }
    # Wakes that only landed after the audio had stopped are late, not
    # timely: on a live mic the user has already given up by then.
    late_wakes = [ts for ts in app.wakes if ts > feed_end]
    detect_ms = backend.detect_ms or [0.0]
    return {
        "load_pct": round(duty * 100),
        "tap": tap["name"],
        "chunks_offered": tap["offered"],
        "chunks_dropped": tap["dropped"],
        "drop_ratio_pct": round(tap["drop_ratio"] * 100, 2),
        "detect_calls": backend.calls,
        "raw_hits": backend.raw_hits,
        "wakes": len(app.wakes),
        "wakes_after_audio_end": len(late_wakes),
        "backlog_chunks_at_end": tap["qsize"],
        "expected": args.expect,
        "detect_ms_p50": round(statistics.median(detect_ms), 2),
        "detect_ms_p95": round(sorted(detect_ms)[int(len(detect_ms) * 0.95) - 1], 2),
        "detect_ms_max": round(max(detect_ms), 2),
        "loop_lag_ms_p50": round(statistics.median(lag_ms or [0.0]), 2),
        "loop_lag_ms_p95": round(sorted(lag_ms or [0.0])[int(len(lag_ms or [0.0]) * 0.95) - 1], 2),
        "loop_lag_ms_max": round(max(lag_ms or [0.0]), 2),
        "feeder_late_chunks": len(feeder.late_ms),
        "wall_s": round(wall_s, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--phrase", default="你好小智")
    parser.add_argument("--expect", type=int, default=8)
    parser.add_argument(
        "--loads",
        default="0,20,50,80",
        help="event-loop duty cycles to sweep, in percent",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--score", type=float, default=1.5)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    parser.add_argument("--tail-s", type=float, default=1.5)
    parser.add_argument("--drain-s", type=float, default=20.0)
    parser.add_argument("--tokens-type", default="phone+ppinyin")
    parser.add_argument("--lexicon", default="en.phone")
    parser.add_argument(
        "--encoder", default="encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
    )
    parser.add_argument("--decoder", default="decoder-epoch-13-avg-2-chunk-8-left-64.onnx")
    parser.add_argument(
        "--joiner", default="joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
    )
    args = parser.parse_args()

    chunks, sample_rate, duration_s = read_pcm_chunks(args.wav, args.chunk_ms)
    loads = [float(value) / 100.0 for value in args.loads.split(",") if value.strip()]
    print(
        f"# wav={args.wav} sr={sample_rate} dur={duration_s:.2f}s "
        f"chunks={len(chunks)} chunk_ms={args.chunk_ms} phrase={args.phrase!r} "
        f"expect={args.expect} threshold={args.threshold} score={args.score}",
        flush=True,
    )
    rows = []
    for duty in loads:
        for attempt in range(args.repeat):
            row = asyncio.run(run_one(args, chunks, sample_rate, duty))
            row["run"] = attempt
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    header = (
        f"{'load%':>6} {'run':>3} {'offered':>8} {'dropped':>8} {'drop%':>7} "
        f"{'wakes':>6} {'raw':>5} {'exp':>4} {'det_p95':>8} {'lag_p95':>8} {'lag_max':>8}"
    )
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        print(
            f"{row['load_pct']:>6} {row['run']:>3} {row['chunks_offered']:>8} "
            f"{row['chunks_dropped']:>8} {row['drop_ratio_pct']:>7} {row['wakes']:>6} "
            f"{row['raw_hits']:>5} {row['expected']:>4} {row['detect_ms_p95']:>8} "
            f"{row['loop_lag_ms_p95']:>8} {row['loop_lag_ms_max']:>8}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
