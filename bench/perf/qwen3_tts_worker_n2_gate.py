#!/usr/bin/env python3
"""Direct Qwen3-TTS worker N=2 isolation, cancellation, and recovery gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import time
import traceback
import wave
from pathlib import Path
from typing import Any


ERROR_RE = re.compile(
    r"(CUDA(?: runtime)? (?:error|failure)|illegal memory access|"
    r"(?:TensorRT|\[TRT\]).*(?:\[E\]|error|fail)|segmentation fault|core dumped)",
    re.IGNORECASE,
)


def _write_wav(path: Path, pcm: bytes, sample_rate: int = 24000) -> dict[str, Any]:
    if not pcm or len(pcm) % 2:
        raise RuntimeError("B1 worker produced empty or unaligned PCM")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)
    with wave.open(str(path), "rb") as inp:
        if inp.getnchannels() != 1 or inp.getsampwidth() != 2 or inp.getframerate() != 24000:
            raise RuntimeError("B1 output WAV is not 24 kHz mono PCM16")
        frames = inp.getnframes()
    return {"path": str(path), "sha256": digest(path.read_bytes()), "size": path.stat().st_size,
            "sample_rate": 24000, "channels": 1, "sample_width": 2,
            "samples": frames, "duration_s": frames / 24000.0}


def _b1_request(args: argparse.Namespace) -> dict[str, Any]:
    req = {"id": "b1-0001", "text": args.text, "stream": True,
           "stream_only": True, "language": args.language,
           "chunk_transport": "base64", "chunk_format": "pcm_s16le"}
    if args.mode == "base":
        req["ref_audio"] = str(Path(args.ref_audio).resolve())
        req["ref_text"] = args.ref_text
    else:
        req["speaker"] = args.speaker
        if args.instruct:
            req["instruct"] = args.instruct
    return req


def _b1_provenance(args: argparse.Namespace) -> dict[str, str]:
    phase = {"base": "tts.base.b1", "customvoice": "tts.customvoice.b1"}[args.mode]
    return {"row_id": args.row_id, "phase": phase, "variant": args.mode, "closure_sha256": args.closure_sha256}


def run_b1(args: argparse.Namespace) -> int:
    """Run one v0.11 worker request and persist raw child/WAV evidence.

    This deliberately accepts the current ref_audio/ref_text and named speaker
    contract. It never passes max_slots or speaker_embedding_b64 to the child.
    """
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_stdout = output.with_suffix(output.suffix + ".stdout")
    raw_stderr = output.with_suffix(output.suffix + ".stderr")
    argv = json.loads(args.worker_argv) if args.worker_argv else [
        args.worker, "--cloneEncoderDir", args.clone_encoder_dir,
        "--checkpointDir", args.checkpoint_dir,
    ]
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
        raise RuntimeError("worker argv must be a nonempty string list")
    if any(x == "--max_slots" or x.startswith("--max_slots=") for x in argv):
        raise RuntimeError("B1 refuses unsupported --max_slots worker argv")
    env = os.environ.copy()
    if args.plugin_path:
        env["EDGELLM_PLUGIN_PATH"] = args.plugin_path
    stdout_stream = raw_stdout.open("wb")
    stderr_stream = raw_stderr.open("wb")
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=False, bufsize=0, env=env)
    assert proc.stdin and proc.stdout and proc.stderr
    events_q: queue.Queue[bytes | None] = queue.Queue()
    def capture_stdout() -> None:
        try:
            for raw in iter(proc.stdout.readline, b""):
                stdout_stream.write(raw); stdout_stream.flush(); events_q.put(raw)
        finally:
            events_q.put(None); stdout_stream.close()
    def capture_stderr() -> None:
        try:
            for raw in iter(proc.stderr.readline, b""):
                stderr_stream.write(raw); stderr_stream.flush()
        finally:
            stderr_stream.close()
    threading.Thread(target=capture_stdout, daemon=True).start()
    threading.Thread(target=capture_stderr, daemon=True).start()
    events: list[dict[str, Any]] = []
    pcm = bytearray(); first_chunk = None; terminal = None; ready = None
    expected_chunk_index = 0; final_chunk_seen = False
    request = _b1_request(args)
    send_at = None; deadline = time.monotonic() + args.timeout
    provenance = _b1_provenance(args)
    report = {"schema": "qwen3-tts-b1.v1", "status": "NOT_RUN", **provenance,
              "request": request, "events": [], "argv": argv,
              "child_pid": proc.pid, "termination_count": 0, "survivor": False,
              "raw_stdout": str(raw_stdout), "raw_stderr": str(raw_stderr)}
    try:
        # The worker must advertise readiness before the request clock starts.
        while ready is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise TimeoutError("B1 ready deadline exceeded")
            raw_bytes = events_q.get(timeout=remaining)
            if raw_bytes is None: raise RuntimeError("B1 worker closed before ready")
            raw = raw_bytes.decode("utf-8", errors="strict").rstrip("\r\n")
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed NDJSON: {exc}") from exc
            if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                raise RuntimeError("B1 event must be an object with event")
            events.append(event)
            if event["event"] == "ready":
                if ready is not None: raise RuntimeError("duplicate ready event")
                ready = event
        send_at = time.monotonic(); report["send_mono"] = send_at
        proc.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode()); proc.stdin.flush()
        deadline = send_at + args.timeout
        while terminal is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise TimeoutError("B1 terminal deadline exceeded")
            raw_bytes = events_q.get(timeout=remaining)
            if raw_bytes is None: raise RuntimeError("B1 worker closed without terminal event")
            raw = raw_bytes.decode("utf-8", errors="strict").rstrip("\r\n")
            try: event = json.loads(raw)
            except json.JSONDecodeError as exc: raise RuntimeError(f"malformed NDJSON: {exc}") from exc
            if not isinstance(event, dict) or not isinstance(event.get("event"), str): raise RuntimeError("B1 event object invalid")
            events.append(event); kind = event["event"]
            if kind == "ready": raise RuntimeError("duplicate/late ready event")
            if kind == "chunk":
                if event.get("id") != request["id"] or not isinstance(event.get("audio_b64"), str):
                    raise RuntimeError("B1 chunk id/audio_b64 invalid")
                try:
                    part = base64.b64decode(event["audio_b64"], validate=True)
                except Exception as exc:
                    raise RuntimeError("B1 chunk base64 invalid") from exc
                if event.get("sample_rate") != 24000 or type(event.get("samples")) is not int or event.get("samples") != len(part) // 2:
                    raise RuntimeError("B1 chunk sample metadata is invalid")
                if not part or len(part) % 2:
                    raise RuntimeError("B1 chunk PCM is empty or unaligned")
                if type(event.get("chunk_index")) is not int or event.get("chunk_index") != expected_chunk_index:
                    raise RuntimeError("B1 chunk_index is missing or not continuous")
                expected_chunk_index += 1
                if event.get("is_final") is not True and event.get("is_final") is not False:
                    raise RuntimeError("B1 chunk is_final is missing")
                if event.get("is_final") is True: final_chunk_seen = True
                if first_chunk is None:
                    first_chunk = time.monotonic()
                pcm.extend(part)
            elif kind in {"done", "error", "cancelled"}:
                if event.get("id") != request["id"]:
                    raise RuntimeError("B1 terminal id mismatch")
                terminal = event
        if terminal is None: raise RuntimeError("B1 worker closed without terminal event")
        try:
            late = events_q.get(timeout=0.05)
        except queue.Empty:
            late = None
        if late not in (None, b""):
            raise RuntimeError("B1 emitted data after terminal event")
        if terminal.get("event") != "done" or terminal.get("ok") is not True:
            raise RuntimeError(f"B1 terminal failure: {terminal}")
        if first_chunk is None or not pcm or not final_chunk_seen:
            raise RuntimeError("B1 requires ready, chunk, and done evidence")
        wav = _write_wav(output.with_suffix(".wav"), bytes(pcm))
        if type(terminal.get("samples")) is not int or terminal.get("samples") != wav["samples"]:
            raise RuntimeError("B1 terminal samples do not match reconstructed PCM")
        if terminal.get("sample_rate") != 24000 or terminal.get("audio_complete") is not True or terminal.get("last_chunk_was_final") is not True:
            raise RuntimeError("B1 terminal sample rate is not 24000")
        if type(terminal.get("chunk_count")) is not int or terminal.get("chunk_count") != expected_chunk_index or terminal.get("final_chunk_index") != expected_chunk_index - 1:
            raise RuntimeError("B1 terminal chunk finalization metadata is invalid")
        ended = time.monotonic()
        report = {"schema": "qwen3-tts-b1.v1", "status": "PASS", "functional_status": "PASS", "threshold_status": "OPEN", "qualification_status": "UNPROVEN", **provenance,
                  "request": request, "ready": ready,
                  "events": events, "terminal": terminal, "wav": wav,
                  "send_mono": send_at, "first_chunk_mono": first_chunk, "terminal_mono": ended,
                  "ttfa_s": first_chunk - send_at, "total_s": ended - send_at,
                  "rtf": (ended - send_at) / wav["duration_s"],
                  "child_pid": proc.pid, "child_rc": None, "termination_count": 0,
                  "survivor": False, "argv": argv, "raw_stdout": str(raw_stdout), "raw_stderr": str(raw_stderr)}
        proc.stdin.close(); proc.wait(timeout=max(1.0, min(10.0, args.timeout)))
        report["child_rc"] = proc.returncode
        report["status"] = "PASS" if proc.returncode == 0 else "UNPROVEN"
    except Exception as exc:
        reason = "B1 bounded read timeout" if isinstance(exc, queue.Empty) else str(exc)
        report.update({"reason": reason, "events": events, "first_chunk_mono": first_chunk,
                       "terminal_mono": time.monotonic(), "send_mono": send_at})
        if proc.poll() is None:
            proc.terminate(); report["termination_count"] = 1
            try: proc.wait(timeout=min(5.0, max(1.0, args.timeout)))
            except subprocess.TimeoutExpired: report["survivor"] = True
        report["child_rc"] = proc.returncode
    finally:
        if proc.stdin and not proc.stdin.closed:
            try: proc.stdin.close()
            except Exception: pass
        stdout_stream.flush() if not stdout_stream.closed else None
    report["raw_stdout_sha256"] = digest(raw_stdout.read_bytes())
    report["raw_stdout_size"] = raw_stdout.stat().st_size
    report["raw_stderr_sha256"] = digest(raw_stderr.read_bytes())
    report["raw_stderr_size"] = raw_stderr.stat().st_size
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return 0 if report.get("status") == "PASS" else 9


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("n2", "base", "customvoice"), default="n2")
    parser.add_argument("--worker-argv", help="JSON array for the exact v0.11 B1 worker argv")
    parser.add_argument("--clone-encoder-dir", default="")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--ref-audio", default="")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--text", default="B1 worker protocol check")
    parser.add_argument("--worker")
    parser.add_argument("--talker-dir")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--code2wav-dir")
    parser.add_argument("--cp-dir")
    parser.add_argument("--plugin-path", default="")
    speaker = parser.add_mutually_exclusive_group(required=False)
    speaker.add_argument("--speaker-embedding-b64-file")
    speaker.add_argument("--speaker")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", required=True)
    parser.add_argument("--row-id")
    parser.add_argument("--closure-sha256")
    args = parser.parse_args()

    if args.mode in {"base", "customvoice"}:
        if not isinstance(args.row_id, str) or args.row_id != {"base": "tts.base.b1", "customvoice": "tts.customvoice.b1"}[args.mode]:
            parser.error("B1 --row-id must match --mode")
        if not isinstance(args.closure_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", args.closure_sha256):
            parser.error("B1 --closure-sha256 must be a 64-hex closure digest")
        if not args.worker:
            parser.error("--worker is required for B1")
        if args.mode == "base" and (not args.ref_audio or not args.ref_text):
            parser.error("Base B1 requires --ref-audio and --ref-text")
        if args.mode == "customvoice" and not args.speaker:
            parser.error("CustomVoice B1 requires --speaker")
        if args.worker_argv:
            try:
                worker_argv = json.loads(args.worker_argv)
            except json.JSONDecodeError:
                parser.error("--worker-argv must be JSON")
            reserved = {"--row-id", "--closure-sha256"}
            if isinstance(worker_argv, list) and any(item in reserved or any(item.startswith(flag + "=") for flag in reserved) for item in worker_argv):
                parser.error("worker argv must not override reserved provenance flags")
        return run_b1(args)

    required_n2 = {
        "--talker-dir": args.talker_dir, "--tokenizer-dir": args.tokenizer_dir,
        "--code2wav-dir": args.code2wav_dir, "--cp-dir": args.cp_dir,
        "--plugin-path": args.plugin_path,
    }
    if not args.worker or not args.speaker and not args.speaker_embedding_b64_file:
        parser.error("N2 requires --worker and one speaker input")
    missing = [key for key, value in required_n2.items() if not value]
    if missing:
        parser.error("N2 missing " + ", ".join(missing))

    embedding = None
    if args.speaker_embedding_b64_file:
        embedding = Path(args.speaker_embedding_b64_file).read_text().strip()
        decoded = base64.b64decode(embedding, validate=True)
        if len(decoded) != 4096:
            raise RuntimeError(f"expected 4096 embedding bytes, got {len(decoded)}")

    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = args.plugin_path
    env["LD_PRELOAD"] = args.plugin_path + (
        f":{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else ""
    )
    env["EDGE_LLM_TTS_LAZY_CODE2WAV"] = "0"
    env["QWEN3_TTS_SEED"] = "42"
    cmd = [
        args.worker,
        "--talkerEngineDir",
        args.talker_dir,
        "--tokenizerDir",
        args.tokenizer_dir,
        "--code2wavEngineDir",
        args.code2wav_dir,
        "--codePredictorEngineDir",
        args.cp_dir,
        "--max_slots",
        "2",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdin and proc.stdout and proc.stderr
    condition = threading.Condition()
    states: dict[str, dict[str, Any]] = {}
    stderr_lines: list[str] = []
    reader_errors: list[str] = []
    ready: dict[str, Any] = {}

    def stdout_reader() -> None:
        try:
            for raw in proc.stdout:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                with condition:
                    if event.get("event") == "ready":
                        ready.update(event)
                    request_id = event.get("id") or event.get("request_id")
                    if request_id and request_id != "__worker__":
                        state = states.setdefault(
                            request_id, {"pcm": bytearray(), "chunks": 0}
                        )
                        state["last_event"] = event
                        if event.get("event") == "chunk":
                            state["chunks"] += 1
                            state["pcm"].extend(
                                base64.b64decode(event.get("audio_b64", ""))
                            )
                            state.setdefault("first_chunk_at", time.monotonic())
                        if event.get("event") in ("done", "error", "cancelled"):
                            state["terminal"] = event
                            state["done_at"] = time.monotonic()
                    condition.notify_all()
        except BaseException:
            with condition:
                reader_errors.append(traceback.format_exc())
                condition.notify_all()

    def stderr_reader() -> None:
        for raw in proc.stderr:
            stderr_lines.append(raw.rstrip())

    threading.Thread(target=stdout_reader, daemon=True).start()
    threading.Thread(target=stderr_reader, daemon=True).start()

    def wait_for(predicate, timeout: float, description: str) -> None:
        deadline = time.monotonic() + timeout
        with condition:
            while not predicate():
                if reader_errors:
                    raise RuntimeError(
                        f"worker stdout reader failed:\n{reader_errors[-1]}"
                    )
                returncode = proc.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"worker exited while waiting for {description}: "
                        f"returncode={returncode}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(description)
                condition.wait(min(remaining, 1.0))

    print("[phase] waiting for worker ready", flush=True)
    wait_for(lambda: bool(ready), 30, "worker ready timeout")
    if ready.get("max_slots") != 2:
        raise RuntimeError(f"worker did not create two slots: {ready}")
    print(f"[phase] worker ready: {ready}", flush=True)

    text_a = "这是并发验证请求甲，用于检查语音流隔离和取消恢复。"
    text_b = "这是并发验证请求乙，用于确认第二路语音能够独立完成。"

    def payload(request_id: str, text: str) -> dict[str, Any]:
        request = {
            "id": request_id,
            "text": text,
            "language": "chinese",
            "stream": True,
            "stream_only": True,
            "chunk_transport": "base64",
            "chunk_format": "pcm_s16le",
            "first_chunk_frames": 7,
            "chunk_frames": 10,
            "max_chunk_frames": 10,
            "max_audio_length": 50,
            "min_audio_length": 10,
            "talker_top_k": 1,
            "predictor_top_k": 1,
        }
        if embedding is not None:
            request["speaker_embedding_b64"] = embedding
        else:
            request["speaker"] = args.speaker
        return request

    def send(message: dict[str, Any]) -> float:
        sent_at = time.monotonic()
        proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        return sent_at

    def request(request_id: str, text: str) -> dict[str, Any]:
        states[request_id] = {"pcm": bytearray(), "chunks": 0}
        sent_at = send(payload(request_id, text))
        wait_for(
            lambda: "terminal" in states[request_id],
            args.timeout,
            f"request timeout: {request_id}",
        )
        state = states[request_id]
        terminal = state["terminal"]
        if not terminal.get("ok", False) or not state["pcm"]:
            raise RuntimeError(f"request failed: {request_id}: {terminal}")
        return {
            "id": request_id,
            "sent_at": sent_at,
            "done_at": state["done_at"],
            "ttfa_ms": (state["first_chunk_at"] - sent_at) * 1000,
            "total_ms": (state["done_at"] - sent_at) * 1000,
            "chunks": state["chunks"],
            "bytes": len(state["pcm"]),
            "sha256": digest(bytes(state["pcm"])),
        }

    print("[phase] baseline A", flush=True)
    baseline_a = request("baseline-a", text_a)
    print("[phase] baseline B", flush=True)
    baseline_b = request("baseline-b", text_b)
    if baseline_a["sha256"] == baseline_b["sha256"]:
        raise RuntimeError("distinct baseline prompts produced identical PCM")

    for request_id in ("concurrent-a", "concurrent-b"):
        states[request_id] = {"pcm": bytearray(), "chunks": 0}
    print("[phase] full N=2", flush=True)
    concurrent_sent = {
        "concurrent-a": send(payload("concurrent-a", text_a)),
        "concurrent-b": send(payload("concurrent-b", text_b)),
    }
    wait_for(
        lambda: all("terminal" in states[key] for key in ("concurrent-a", "concurrent-b")),
        args.timeout,
        "full N=2 timeout",
    )
    concurrent = {}
    for request_id, baseline in (
        ("concurrent-a", baseline_a),
        ("concurrent-b", baseline_b),
    ):
        state = states[request_id]
        if not state["terminal"].get("ok", False) or not state["pcm"]:
            raise RuntimeError(f"full N=2 failed: {request_id}: {state['terminal']}")
        concurrent[request_id] = {
            "ttfa_ms": (
                state["first_chunk_at"] - concurrent_sent[request_id]
            ) * 1000,
            "total_ms": (
                state["done_at"] - concurrent_sent[request_id]
            ) * 1000,
            "chunks": state["chunks"],
            "bytes": len(state["pcm"]),
            "sha256": digest(bytes(state["pcm"])),
            "matches_baseline": digest(bytes(state["pcm"])) == baseline["sha256"],
        }
    full_overlap = (
        states["concurrent-a"].get("first_chunk_at", float("inf"))
        < states["concurrent-b"]["done_at"]
        and states["concurrent-b"].get("first_chunk_at", float("inf"))
        < states["concurrent-a"]["done_at"]
    )
    if not full_overlap or not all(item["matches_baseline"] for item in concurrent.values()):
        raise RuntimeError(f"full N=2 isolation failed: {concurrent}")

    rounds = []
    for index in range(1, args.rounds + 1):
        cancel_id = f"round-{index:03d}-cancel-a"
        keep_id = f"round-{index:03d}-keep-b"
        recovery_id = f"round-{index:03d}-recovery-b"
        states[cancel_id] = {"pcm": bytearray(), "chunks": 0}
        states[keep_id] = {"pcm": bytearray(), "chunks": 0}
        cancel_sent_at = send(payload(cancel_id, text_a))
        keep_sent_at = send(payload(keep_id, text_b))
        wait_for(
            lambda: states[cancel_id]["chunks"] >= 1,
            args.timeout,
            f"cancel stream did not start: {cancel_id}",
        )
        cancel_at = send({"type": "cancel", "id": cancel_id})
        wait_for(
            lambda: "terminal" in states[cancel_id] and "terminal" in states[keep_id],
            args.timeout,
            f"cancel/keep timeout: round {index}",
        )
        keep = states[keep_id]
        recovery = request(recovery_id, text_b)
        keep_digest = digest(bytes(keep["pcm"]))
        ok = (
            bool(states[cancel_id]["pcm"])
            and keep["terminal"].get("ok", False)
            and bool(keep["pcm"])
            and keep_digest == baseline_b["sha256"]
            and recovery["sha256"] == baseline_b["sha256"]
            and cancel_at < keep["done_at"]
        )
        record = {
            "round": index,
            "ok": ok,
            "cancel_chunks": states[cancel_id]["chunks"],
            "cancel_terminal": states[cancel_id]["terminal"],
            "cancel_ttfa_ms": (
                states[cancel_id]["first_chunk_at"] - cancel_sent_at
            ) * 1000,
            "cancel_done_after_signal_ms": (
                states[cancel_id]["done_at"] - cancel_at
            ) * 1000,
            "keep_chunks": keep["chunks"],
            "keep_ttfa_ms": (
                keep["first_chunk_at"] - keep_sent_at
            ) * 1000,
            "keep_total_ms": (keep["done_at"] - keep_sent_at) * 1000,
            "keep_matches_baseline": keep_digest == baseline_b["sha256"],
            "recovery_ttfa_ms": recovery["ttfa_ms"],
            "recovery_total_ms": recovery["total_ms"],
            "recovery_matches_baseline": recovery["sha256"] == baseline_b["sha256"],
            "cancel_before_keep_done": cancel_at < keep["done_at"],
        }
        rounds.append(record)
        print(
            f"[{index:03d}/{args.rounds}] ok={ok} "
            f"cancel_chunks={record['cancel_chunks']} keep_chunks={record['keep_chunks']}",
            flush=True,
        )
        if not ok:
            raise RuntimeError(f"N=2 cancellation gate failed: {record}")

    proc.stdin.close()
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    error_hits = [line for line in stderr_lines if ERROR_RE.search(line)]
    report = {
        "ready": ready,
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "concurrent_sent_at": concurrent_sent,
        "concurrent": concurrent,
        "full_overlap": full_overlap,
        "rounds_requested": args.rounds,
        "rounds_passed": sum(bool(item["ok"]) for item in rounds),
        "rounds": rounds,
        "worker_returncode": proc.returncode,
        "stderr_error_hits": error_hits,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "ready", "concurrent", "full_overlap", "rounds_requested",
        "rounds_passed", "worker_returncode", "stderr_error_hits"
    )}, ensure_ascii=False, indent=2))
    return 0 if not error_hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
