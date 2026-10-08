#!/usr/bin/env python3
"""Validate a fresh MOSS TensorRT engine set through the JSONL worker."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import select
import subprocess
import time
import wave
from array import array
from pathlib import Path


DEFAULT_TEXTS = (
    "今天天气很好，我们一起测试语音合成。",
    "语音合成的稳定性。",
    "说起咱北京的烤鸭啊，那可真是外焦里嫩。",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--codec-onnx-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--reserve", type=float, default=10.0)
    parser.add_argument("--max-slots", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-new-frames", type=int, default=None, help="Pass through to worker when supported")
    parser.add_argument("--worker-label", default="moss-v091", help="Evidence/request label")
    return parser.parse_args()


def process_identity(proc: subprocess.Popen[bytes]) -> dict:
    """Capture a positive identity that can be checked before TERM."""
    pid = proc.pid
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        exe_path = os.readlink(f"/proc/{pid}/exe")
        exe_sha = hashlib.sha256(Path(f"/proc/{pid}/exe").read_bytes()).hexdigest()
        argv = [part.decode(errors="replace") for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
        return {"pid": pid, "starttime_ticks": start_ticks, "exe": exe_path,
                "exe_sha256": exe_sha, "cmdline": argv,
                "valid": type(start_ticks) is int and start_ticks > 0 and bool(exe_path) and len(exe_sha) == 64}
    except Exception as exc:
        return {"pid": pid, "valid": False, "error": f"{type(exc).__name__}: {exc}"}


def same_identity(left: dict | None, right: dict | None) -> bool:
    keys = ("pid", "starttime_ticks", "exe", "exe_sha256", "cmdline")
    return bool(left and right and left.get("valid") is True and right.get("valid") is True
                and all(left.get(key) == right.get(key) for key in keys))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class EventReader:
    def __init__(self, proc: subprocess.Popen[bytes], raw_path: Path | None = None) -> None:
        if proc.stdout is None:
            raise RuntimeError("worker stdout is unavailable")
        self.proc = proc
        self.stdout = proc.stdout
        self.buffer = bytearray()
        self.raw_path = raw_path
        self.malformed_lines = 0
        self.eof = False

    def read(self, deadline: float) -> dict:
        while time.monotonic() < deadline:
            while b"\n" in self.buffer:
                line, _, remainder = self.buffer.partition(b"\n")
                self.buffer = bytearray(remainder)
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("JSON event must be an object")
                    return event
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # TensorRT/EdgeLLM diagnostics are not worker protocol.
                    self.malformed_lines += 1
                    continue
                except ValueError:
                    self.malformed_lines += 1
                    continue
            remaining = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select([self.stdout], [], [], remaining)
            if readable:
                chunk = os.read(self.stdout.fileno(), 65536)
                if chunk:
                    if self.raw_path is not None:
                        with self.raw_path.open("ab") as raw:
                            raw.write(chunk)
                    self.buffer.extend(chunk)
                    continue
                self.eof = True
                if self.proc.poll() is None:
                    raise RuntimeError("unexpected worker stdout EOF")
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker exited with status {self.proc.returncode}")
        raise TimeoutError("timed out waiting for MOSS worker event")

    def drain(self, deadline: float) -> None:
        """Drain post-done diagnostics so a noisy child cannot block on stdout."""
        while self.proc.poll() is None and time.monotonic() < deadline:
            readable, _, _ = select.select([self.stdout], [], [], min(0.1, max(0.0, deadline - time.monotonic())))
            if not readable:
                continue
            chunk = os.read(self.stdout.fileno(), 65536)
            if not chunk:
                self.eof = True
                return
            if self.raw_path is not None:
                with self.raw_path.open("ab") as raw:
                    raw.write(chunk)


def metrics(pcm: bytes, sample_rate: int, channels: int) -> dict:
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        raise RuntimeError("worker returned no PCM samples")
    scale = 32768.0
    rms = math.sqrt(sum((sample / scale) ** 2 for sample in samples) / len(samples))
    clipped = sum(abs(sample) >= 32767 for sample in samples) / len(samples)
    return {
        "bytes": len(pcm),
        "duration_s": len(samples) / channels / sample_rate,
        "rms": rms,
        "clipping_fraction": clipped,
    }


def write_wav(path: Path, pcm: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def main() -> int:
    args = parse_args()
    if (not math.isfinite(args.timeout) or args.timeout <= 0 or
            not math.isfinite(args.reserve) or args.reserve < 0 or args.reserve >= args.timeout or
            type(args.max_slots) is not int or args.max_slots <= 0 or
            (args.max_seq_len is not None and (type(args.max_seq_len) is not int or args.max_seq_len <= 0)) or
            (args.max_new_frames is not None and (type(args.max_new_frames) is not int or args.max_new_frames <= 0))):
        raise SystemExit("invalid finite timeout/reserve/worker limits")
    output_dir = Path(args.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise SystemExit(f"output directory exists or is a symlink: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    whole_deadline = started + args.timeout
    work_deadline = whole_deadline - args.reserve
    command = [
        args.worker,
        f"--engine-dir={args.engine_dir}",
        f"--tokenizer-model={args.engine_dir}/tokenizer.model",
        f"--codec-onnx-dir={args.codec_onnx_dir}",
        f"--max-slots={args.max_slots}",
    ]
    if args.max_seq_len is not None:
        command.append(f"--max-seq-len={args.max_seq_len}")
    stderr_path = output_dir / "worker.stderr.log"
    stdout_path = output_dir / "worker.stdout.raw"
    report = {"status": "FAIL", "command": command, "results": [], "worker_ready": None,
              "raw_stdout": str(stdout_path), "raw_stderr": str(stderr_path),
              "worker_label": args.worker_label, "whole_s": args.timeout,
              "reserve_s": args.reserve, "term_attempted": 0, "survivor": None,
              "started_mono": started}
    proc = None
    launch = None
    events = None
    error = None
    try:
        with stderr_path.open("x") as stderr:
            stdout_path.open("x").close()
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
            launch = process_identity(proc); report["launch_identity"] = launch
            if not launch.get("valid"):
                raise RuntimeError("invalid worker launch identity")
            events = EventReader(proc, stdout_path)
            ready = events.read(work_deadline)
            report["worker_ready"] = ready
            if ready.get("event") != "worker_ready" or type(ready.get("ok")) is not bool or not ready["ok"]:
                raise RuntimeError(f"unexpected worker readiness event: {ready}")
            sample_rate = ready.get("sample_rate"); channels = ready.get("channels")
            if type(sample_rate) is not int or sample_rate <= 0 or type(channels) is not int or channels <= 0:
                raise RuntimeError("worker_ready has invalid sample_rate/channels")
            results = []
            for index, text in enumerate(args.texts or DEFAULT_TEXTS, start=1):
                request_id = f"{args.worker_label}-{index}"
                request = {"id": request_id, "text": text, "stream": True, "stream_only": True,
                           "chunk_transport": "base64", "chunk_format": "pcm_s16le"}
                if args.max_new_frames is not None:
                    request["max_new_frames"] = args.max_new_frames
                if proc.stdin is None:
                    raise RuntimeError("worker stdin is unavailable")
                proc.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode()); proc.stdin.flush()
                pcm_parts: list[bytes] = []; done = None
                while done is None:
                    event = events.read(work_deadline)
                    if event.get("id") != request_id:
                        raise RuntimeError(f"event id mismatch: expected {request_id}, got {event.get('id')}")
                    if type(event.get("event")) is not str:
                        raise RuntimeError("event type is missing")
                    if type(event.get("ok")) is not bool:
                        raise RuntimeError("event ok must be boolean")
                    if not event["ok"]:
                        raise RuntimeError(f"MOSS request failed: {event}")
                    if event["event"] == "chunk":
                        if type(event.get("audio_b64")) is not str:
                            raise RuntimeError("chunk audio_b64 is missing")
                        try:
                            chunk = base64.b64decode(event["audio_b64"], validate=True)
                        except Exception as exc:
                            raise RuntimeError(f"invalid PCM base64: {exc}") from exc
                        if not chunk or len(chunk) % 2:
                            raise RuntimeError("chunk is not legal non-empty PCM16")
                        pcm_parts.append(chunk)
                    elif event["event"] == "done":
                        done = event
                    elif event["event"] != "ready":
                        raise RuntimeError(f"unexpected event type: {event['event']}")
                pcm = b"".join(pcm_parts)
                wav_path = output_dir / f"moss_v091_{index}.wav"
                write_wav(wav_path, pcm, sample_rate, channels)
                result = {"id": request_id, "text": text, "wav": str(wav_path), **metrics(pcm, sample_rate, channels),
                          "ttfa_ms": done.get("ttfa_ms"), "wall_ms": done.get("wall_ms"), "chunks": len(pcm_parts),
                          "finish_reason": done.get("finish_reason"), "eos_seen": done.get("eos_seen"),
                          "generated_frames": done.get("generated_frames"), "accepted_audio_frames": done.get("accepted_audio_frames"),
                          "sampled_frames": done.get("sampled_frames"), "sampler_stop": done.get("sampler_stop"),
                          "stop_signal_scope": done.get("stop_signal_scope"),
                          "effective_frame_cap": done.get("effective_frame_cap"),
                          "requested_max_new_frames": done.get("requested_max_new_frames"),
                          "prefill_seq_len": done.get("prefill_seq_len"), "decode_budget": done.get("decode_budget"),
                          "done_event": done}
                if result["rms"] <= 0.01 or result["clipping_fraction"] >= 0.01:
                    raise RuntimeError(f"invalid PCM quality: {result}")
                results.append(result)
            report["results"] = results
            report["status"] = "OBSERVED"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception as exc:
                error = error or f"stdin_close: {exc}"
            if proc.poll() is None:
                if events is not None:
                    try:
                        events.drain(work_deadline)
                    except Exception as exc:
                        error = error or f"stdout_drain: {exc}"
                try:
                    # Let a naturally completing worker use the work window. The
                    # reserved tail is the bounded cleanup window, so TERM is
                    # attempted at its start rather than after the whole budget.
                    proc.wait(timeout=max(0.0, work_deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    current = process_identity(proc)
                    if time.monotonic() < whole_deadline and same_identity(current, launch) and report["term_attempted"] == 0:
                        report["term_attempted"] = 1
                        try:
                            proc.terminate()
                            try:
                                proc.wait(timeout=max(0.0, whole_deadline - time.monotonic()))
                            except subprocess.TimeoutExpired:
                                report["survivor"] = process_identity(proc)
                        except ProcessLookupError:
                            pass
                    else:
                        report["survivor"] = current
            report["child_rc"] = proc.poll()
        if events is not None:
            report["malformed_lines"] = events.malformed_lines
            report["stdout_eof"] = events.eof
        report["error"] = error
        report["ended_mono"] = time.monotonic()
        report["status"] = "SURVIVOR_HANDOFF" if report.get("survivor") else ("FAIL" if error or report.get("child_rc") != 0 else report["status"])
        write_json(output_dir / "moss-v091-smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "OBSERVED" and report.get("child_rc") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
