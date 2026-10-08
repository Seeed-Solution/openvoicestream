#!/usr/bin/env python3
"""Bounded Kokoro HTTP qualification. No deployment or process management.

Run baseline and coexistence separately with identical corpora. This runner does
not launch the contender. Target RSS/disk must come from the companion sampler
in the target PID/mount namespace, NOT from the HTTP client's own process.
Protocol parsing derives from the archived http/stream/busy_acceptance probes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import http.client
import json
import math
from pathlib import Path
import socket
import statistics
import struct
import threading
import time
from urllib.parse import urlsplit

import numpy as np

PROTOCOLS = ("finite", "native", "openai-pcm", "openai-wav")


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(values, q))


def wav_info(raw):
    """Parse partial RIFF including OpenAI unknown-length (0xffffffff) data."""
    if len(raw) < 12:
        return None
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not RIFF WAVE")
    pos, valid_fmt = 12, False
    while pos + 8 <= len(raw):
        tag = raw[pos:pos + 4]
        size = struct.unpack_from("<I", raw, pos + 4)[0]
        start = pos + 8
        if tag == b"fmt " and len(raw) >= start + 16:
            if struct.unpack_from("<HHIIHH", raw, start) != (1, 1, 24000, 48000, 2, 16):
                raise ValueError("WAV must be 24kHz mono PCM16")
            valid_fmt = True
        if tag == b"data":
            if not valid_fmt:
                raise ValueError("WAV data before complete fmt")
            return start, min(size, max(0, len(raw) - start)), size
        pos = start + size + (size & 1)
    return None


def pcm_payload(raw, protocol, *, complete=True):
    if protocol == "native":
        if len(raw) < 4 or struct.unpack_from("<I", raw)[0] != 24000:
            raise ValueError("native sample-rate header must be 24000")
        pcm = raw[4:]
    elif protocol in ("finite", "openai-wav"):
        info = wav_info(raw)
        if info is None:
            raise ValueError("missing WAV data")
        offset, size, declared = info
        riff = struct.unpack_from("<I", raw, 4)[0]
        if complete:
            if protocol == "finite" and (declared == 0xffffffff or riff == 0xffffffff):
                raise ValueError("finite WAV must have finite lengths")
            if riff != 0xffffffff and riff + 8 != len(raw):
                raise ValueError("RIFF length mismatch")
            if declared != 0xffffffff and size != declared:
                raise ValueError("truncated WAV data")
        pcm = raw[offset:offset + size]
    else:
        pcm = raw
    if not pcm or len(pcm) % 2:
        raise ValueError("empty or odd PCM bytes")
    return pcm


def audio_stats(pcm):
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    normalized = samples / 32768
    rms = float(np.sqrt(np.mean(normalized ** 2)))
    clip = float(np.mean(np.abs(samples) >= 32767))
    silence = float(np.mean(np.abs(samples) <= 1))
    return {"pcm_sha256": hashlib.sha256(pcm).hexdigest(), "pcm_bytes": len(pcm),
            "sample_rate": 24000, "channels": 1, "sample_width": 2,
            "duration_s": len(samples) / 24000, "finite_samples": bool(np.isfinite(normalized).all()),
            "rms": rms, "peak": float(np.max(np.abs(normalized))),
            "clipping_fraction": clip, "silence_fraction": silence,
            "waveform_pass": rms >= 1e-5 and clip < .01 and silence < .99}


def check_http_completion(response, headers, received_bytes):
    """read1() returns EOF without raising for an incomplete Content-Length."""
    if getattr(response, "chunked", False):
        if getattr(response, "chunk_left", None) is not None:
            raise ValueError("incomplete HTTP chunked framing")
        return "chunked-complete"
    declared = headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError as exc:
            raise ValueError("invalid HTTP Content-Length") from exc
        remaining = getattr(response, "length", None)
        if length < 0 or received_bytes != length or remaining not in (None, 0):
            raise ValueError(f"incomplete HTTP Content-Length: declared={length}, received={received_bytes}, remaining={remaining}")
        return "content-length-complete"
    # EOF is the delimiter for a close-delimited HTTP body. It cannot prove
    # intended utterance length; only explicit framing can detect truncation.
    return "close-delimited-eof; intended utterance length unverified"


def request_body(case, protocol, model, capabilities):
    """Use only protocol fields and capability-declared optional controls."""
    tts = capabilities["tts"]
    language = case.get("language", case.get("route"))
    if not language:
        raise ValueError("case language/route required")
    if protocol.startswith("openai-"):
        fmt = protocol.split("-", 1)[1]
        if fmt not in tts.get("audio", {}).get("response_formats", []):
            raise ValueError(f"metadata does not declare {fmt}")
        body = {"model": model, "input": case["text"], "response_format": fmt}
        path = "/v1/audio/speech"
        # OpenAI has no language field. Keep intended corpus language in evidence,
        # but never claim the adapter selected that route or bundle voice.
    else:
        body = {"text": case["text"]}
        path = "/v1/tts" if protocol == "finite" else "/tts/stream"
        if protocol == "finite":
            body["model"] = model
        if tts.get("languages", {}).get("mode") == "multi_language":
            allowed = tts["languages"].get("values")
            if allowed is not None and language not in allowed:
                raise ValueError(f"metadata does not support language {language}")
            body["language"] = language
        elif language != tts.get("languages", {}).get("default"):
            raise ValueError(f"metadata does not declare language selection for {language}")
    if protocol != "finite" and not tts.get("streaming", {}).get("supported"):
        raise ValueError("metadata does not declare streaming")
    if "voice" in case:
        voices = tts.get("voices", {})
        allowed_voices = [v["id"] for v in voices.get("items", [])] + voices.get("aliases", [])
        if case["voice"] not in allowed_voices:
            raise ValueError("corpus voice is not advertised")
        body["voice"] = case["voice"]
    return path, body


class Client:
    def __init__(self, base_url, timeout=120, max_body_bytes=16 * 1024 * 1024):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("base-url must be an HTTP(S) URL without credentials/query/fragment")
        self.parsed, self.timeout, self.max_body_bytes = parsed, timeout, max_body_bytes
        self._active_lock = threading.Lock()
        self._active = set()

    def close_active(self):
        with self._active_lock:
            active = list(self._active)
        for abort in active:
            abort()

    def connection(self, timeout):
        kind = http.client.HTTPSConnection if self.parsed.scheme == "https" else http.client.HTTPConnection
        return kind(self.parsed.hostname, self.parsed.port, timeout=timeout)

    def request(self, path, body=None, *, protocol=None, deadline=None, partial=False, sent=None):
        start = time.monotonic()
        deadline = min(deadline or start + self.timeout, start + self.timeout)
        timeout = deadline - start
        row = {"http_status": 0, "first_pcm_ms": None, "status": "FAIL", "headers": {}}
        raw = bytearray()
        if timeout <= 0:
            return {**row, "error": "deadline expired", "wall_s": 0}, b""
        conn = self.connection(timeout)
        response = None
        transport = []
        def abort():
            # A deadline watchdog also bounds slow-drip HTTP headers/read1 calls.
            sock = transport[0] if transport else conn.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            conn.close()
        watchdog = threading.Timer(timeout, abort)
        watchdog.daemon = True
        with self._active_lock:
            self._active.add(abort)
        watchdog.start()
        try:
            conn.request("POST" if body is not None else "GET", self.parsed.path.rstrip("/") + path,
                         body=json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
                         headers={"Content-Type": "application/json", "Connection": "close"})
            if conn.sock is not None:
                transport.append(conn.sock)
            if sent is not None:
                sent.set()
            response = conn.getresponse()
            row.update(http_status=response.status, headers={k.lower(): v for k, v in response.getheaders()},
                       first_headers_ms=(time.monotonic() - start) * 1000)
            while time.monotonic() < deadline:
                chunk = response.read1(4096)
                if not chunk:
                    break
                raw.extend(chunk)
                if len(raw) > self.max_body_bytes:
                    raise ValueError("response exceeds bounded body limit")
                if response.status == 200 and protocol and row["first_pcm_ms"] is None:
                    if protocol in ("finite", "openai-wav"):
                        info = wav_info(raw)
                        has_pcm = info is not None and info[1] >= 2
                    else:
                        has_pcm = len(raw) >= (6 if protocol == "native" else 2)
                    if has_pcm:
                        row["first_pcm_ms"] = (time.monotonic() - start) * 1000
                        if partial:
                            break
            if time.monotonic() >= deadline:
                raise TimeoutError("absolute request deadline")
            if not partial:
                row["http_body_framing"] = check_http_completion(response, row["headers"], len(raw))
            else:
                row["http_body_framing"] = "deliberately-partial; completion not checked"
            if response.status != 200:
                row["error_body"] = bytes(raw[:4096]).decode(errors="replace")
            elif protocol:
                if protocol == "openai-pcm" and (row["headers"].get("content-type", "").split(";")[0], row["headers"].get("x-sample-rate"), row["headers"].get("x-audio-channels")) != ("audio/pcm", "24000", "1"):
                    raise ValueError("OpenAI PCM headers must declare 24kHz mono audio/pcm")
                pcm = pcm_payload(bytes(raw), protocol, complete=not partial)
                row.update(audio_stats(pcm))
                # A deliberately partial stream may consist only of leading
                # silence. Preserve its statistics but gate full utterances.
                row["status"] = "CANCELLED" if partial else ("PASS" if row["waveform_pass"] else "FAIL")
                if not partial and not row["waveform_pass"]:
                    row["error"] = "waveform integrity gate failed"
            else:
                row["status"] = "PASS"
        except Exception as exc:
            row.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        finally:
            abort()
            if response is not None:
                response.close()
            watchdog.cancel()
            with self._active_lock:
                self._active.discard(abort)
        row.update(wall_s=time.monotonic() - start, body_bytes=len(raw), body_sha256=hashlib.sha256(raw).hexdigest())
        if row.get("duration_s"):
            row["end_to_end_rtf"] = row["wall_s"] / row["duration_s"]
        return row, bytes(raw)

    def finite_disconnect(self, path, body, deadline):
        """Send a COMPLETE JSON request, then close TCP without reading response."""
        start = time.monotonic()
        conn = self.connection(min(self.timeout, max(.001, deadline - start)))
        try:
            conn.request("POST", self.parsed.path.rstrip("/") + path,
                         body=json.dumps(body, ensure_ascii=False).encode(),
                         headers={"Content-Type": "application/json", "Connection": "close"})
            # Give the server a bounded opportunity to begin processing. This is
            # not proof of native cancellation; server-side evidence is separate.
            time.sleep(min(.1, max(0, deadline - time.monotonic())))
            return {"status": "DISCONNECTED", "full_json_sent": True, "http_status": None,
                    "server_admission": "UNVERIFIED", "native_cancellation": "UNVERIFIED",
                    "wall_s": time.monotonic() - start, "disconnect_unix_s": time.time()}
        finally:
            if conn.sock:
                try:
                    conn.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            conn.close()


def aggregate(rows):
    successful = [r for r in rows if r.get("status") == "PASS" and r.get("kind") == "synthesis"]
    groups = defaultdict(list)
    intended_groups = defaultdict(list)
    for row in successful:
        groups[(row.get("verified_route", "unknown"), row["engine"], row["protocol"])].append(row)
        intended_groups[(row.get("intended_language", row["language"]), row["engine"], row["protocol"])].append(row)
    def summaries(source, label):
        return [{label: k[0], "engine": k[1], "protocol": k[2], "count": len(v),
                 "first_pcm_ms_p50": percentile([r["first_pcm_ms"] for r in v], 50),
                 "first_pcm_ms_p95": percentile([r["first_pcm_ms"] for r in v], 95),
                 "end_to_end_rtf_p50": percentile([r["end_to_end_rtf"] for r in v], 50),
                 "end_to_end_rtf_p95": percentile([r["end_to_end_rtf"] for r in v], 95)} for k, v in sorted(source.items())]
    audio = sum(r["duration_s"] for r in successful)
    return {"successes": len(successful), "weighting": "sum(request wall seconds) / sum(audio seconds); audio-duration weights; each corpus case x protocol visited round-robin",
            "weighted_end_to_end_rtf": sum(r["wall_s"] for r in successful) / audio if audio else None,
            "observed_case_protocol_counts": dict(Counter(r["case_id"] + "/" + r["protocol"] for r in successful)),
            "route_evidence_scope": "verified_route means an accepted explicit native language selector, not independent backend execution attestation; OpenAI auto route is unknown",
            "groups": summaries(groups, "verified_route"),
            "intended_language_groups": summaries(intended_groups, "intended_language")}


def route_coverage(rows, cases):
    successful = [r for r in rows if r.get("kind") == "synthesis" and r.get("status") == "PASS"]
    required = {(case.get("language", case.get("route")), protocol) for case in cases for protocol in PROTOCOLS}
    intended = {(r.get("intended_language", r.get("language")), r["protocol"]) for r in successful}
    verified = {(r["verified_route"], r["protocol"]) for r in successful if r.get("verified_route") not in (None, "unknown")}
    native_required = {(case["id"], case.get("language", case.get("route")), protocol)
                       for case in cases for protocol in ("finite", "native")}
    native_verified = {(r["case_id"], r.get("verified_route"), r["protocol"])
                       for r in successful if r["protocol"] in ("finite", "native")}
    return {"intended_language_protocol_coverage": required <= intended,
            "explicit_native_case_route_coverage": native_required <= native_verified,
            "missing_explicit_native_cases": [{"case_id": case, "route": route, "protocol": protocol}
                                              for case, route, protocol in sorted(native_required - native_verified)],
            "verified_route_protocol_coverage": required <= verified,
            "missing_verified_route_protocols": [{"route": route, "protocol": protocol} for route, protocol in sorted(required - verified)],
            "openai_route_control": [{"intended_route": route, "protocol": protocol,
                                      "control": "UNSUPPORTED: no advertised language selector",
                                      "directed_route_qualification": "UNVERIFIED"}
                                     for route, protocol in sorted(required) if protocol.startswith("openai")],
            "unverified_auto_route_successes": sum(r.get("verified_route") in (None, "unknown") for r in successful)}


def disk_gate(telemetry, platform):
    """No client-side growth may offset target disk consumption.

    This runner has no authenticated target/filesystem evidence attribution,
    so the conservative allowed target evidence growth is always zero.
    """
    start, end = telemetry.get("free_disk_start"), telemetry.get("free_disk_end")
    valid = telemetry.get("status") == "MEASURED" and isinstance(start, (int, float)) and isinstance(end, (int, float))
    passed = bool(valid and end >= start and (platform != "rk3576" or telemetry.get("free_disk_min", 0) >= 128 * 1024 ** 2))
    return {"pass": passed, "credited_target_evidence_bytes": 0,
            "allowance_basis": "none: target evidence growth is not associated with the sampled filesystem; client evidence is never credited"}


def telemetry_summary(path, start, end, warmup_s):
    if not path:
        return {"status": "UNVERIFIED", "reason": "target RSS/free disk telemetry not supplied"}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if start <= r["unix_s"] <= end]
    rows.sort(key=lambda r: r["unix_s"])
    valid = [r for r in rows if not r.get("error") and r.get("rss_bytes") is not None and r.get("free_disk_bytes") is not None]
    post = [r for r in valid if r["unix_s"] >= start + warmup_s]
    quarter = max(1, len(post) // 4)
    if len(post) < 4 or len(valid) != len(rows):
        return {"status": "UNVERIFIED", "reason": "insufficient/failed post-warmup target telemetry", "raw_series": rows}
    first = statistics.median(r["rss_bytes"] for r in post[:quarter])
    last = statistics.median(r["rss_bytes"] for r in post[-quarter:])
    # Target sampler v2 reports container_pid; retain pid compatibility for the
    # older standalone process sampler.
    identities = {(r.get("container_pid", r.get("pid")), r.get("process_start_ticks")) for r in valid}
    return {"status": "MEASURED", "raw_series": rows, "warmup_s": warmup_s,
            "process_identity_stable": len(identities) == 1 and None not in next(iter(identities)),
            "coverage_pass": valid[0]["unix_s"] <= start + 5 and valid[-1]["unix_s"] >= end - 5,
            "max_sample_gap_s": max(b["unix_s"] - a["unix_s"] for a, b in zip(valid, valid[1:])),
            "first_post_warmup_quarter_rss_median": first, "last_quarter_rss_median": last,
            "rss_plateau_pass": last <= first + 64 * 1024 * 1024,
            "free_disk_start": valid[0]["free_disk_bytes"], "free_disk_end": valid[-1]["free_disk_bytes"],
            "free_disk_min": min(r["free_disk_bytes"] for r in valid)}


def run(args):
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.cancel_cycles < 0 or args.min_success < 0:
        raise ValueError("positive finite seconds and nonnegative counts required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    corpus_raw = Path(args.corpus).read_bytes()
    cases = json.loads(corpus_raw)["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("nonempty corpus with unique IDs required")
    client = Client(args.base_url, args.request_timeout)
    started, start_unix = time.monotonic(), time.time()
    deadline = started + args.seconds
    workload_mode = getattr(args, "workload_mode", "measurement")
    report = {"schema": "kokoro.service-soak.v1", "status": "FAIL", "phase": args.phase,
              "platform": args.platform, "model": args.model, "base_url": args.base_url,
              "workload_mode": workload_mode,
              "requested_seconds": args.seconds, "started_unix_s": start_unix,
              "corpus_sha256": hashlib.sha256(corpus_raw).hexdigest(), "rows": [], "health": [], "failures": [],
              "scope": "HTTP waveform/recovery measurements; audio statistics are not intelligibility/MOS evidence",
              "native_cancellation": "UNVERIFIED without correlated backend evidence",
              "failure_injection": {"reboot": "NOT_TESTED", "power_loss": "NOT_TESTED", "native_hang": "NOT_INJECTED"}}
    stop = threading.Event()
    evidence_bytes = 0
    bodies = set()
    monitor = None
    worker = None
    save_lock = threading.Lock()
    def _save(row, raw):
        nonlocal evidence_bytes
        row["index"] = len(report["rows"])
        if raw and row.get("body_sha256") not in bodies:
            evidence_bytes += len(raw)
            if evidence_bytes > args.max_evidence_mib * 1024 ** 2:
                raise ValueError("bounded evidence budget exhausted")
            name = row["body_sha256"] + ".body"
            (out / name).write_bytes(raw)
            bodies.add(row["body_sha256"])
            row["body_file"] = name
        report["rows"].append(row)
        with (out / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        return row
    def save(row, raw=b""):
        with save_lock:
            return _save(row, raw)
    def synth(case, protocol, kind="synthesis", *, partial=False, request_deadline=None, sent=None):
        path, body = request_body(case, protocol, args.model, report["capabilities"])
        # No new workload request starts after phase deadline. An already-started
        # synthesis may drain for at most request_timeout, explicitly reported.
        row, raw = client.request(path, body, protocol=protocol, deadline=request_deadline or deadline + args.request_timeout, partial=partial, sent=sent)
        header_engine = row["headers"].get("x-kokoro-engine")
        row.update(kind=kind, case_id=case["id"], text=case["text"], language=case.get("language", case.get("route")),
                   intended_language=case.get("language", case.get("route")),
                   route=case.get("route"), protocol=protocol, request=body,
                   voice=body.get("voice", "server-default"),
                   requested_voice=body.get("voice"), verified_voice="unknown",
                   selected_route=body.get("language", "unknown-auto"),
                   verified_route=body["language"] if protocol in ("finite", "native") and row["status"] == "PASS" and "language" in body else "unknown",
                   engine=header_engine if protocol == "finite" and header_engine in ("npu", "cpu", "mixed") else "unknown")
        return save(row, raw)
    def health_monitor():
        while not stop.is_set() and time.monotonic() < deadline:
            row, _ = client.request("/health", deadline=min(deadline, time.monotonic() + 1))
            report["health"].append({"unix_s": time.time(), "http_status": row["http_status"], "latency_ms": row["wall_s"] * 1000, "error": row.get("error")})
            stop.wait(args.health_interval)
    def recovery(case):
        begin = time.monotonic()
        recovery_deadline = min(deadline, begin + 30)
        attempts = []
        while time.monotonic() < recovery_deadline:
            row = synth(case, "finite", "recovery", request_deadline=recovery_deadline)
            attempts.append(row["index"])
            if row["status"] == "PASS":
                return {"status": "PASS", "wall_s": time.monotonic() - begin, "attempt_rows": attempts}
            if row["http_status"] != 429:
                break
            # Each 429 remains an explicit recovery attempt, never a successful
            # synthesis. Only transient admission rejection may be polled.
            stop.wait(min(.25, max(0, recovery_deadline - time.monotonic())))
        raise RuntimeError(f"recovery failed within 30 seconds; attempt rows={attempts}")
    try:
        metadata, raw = client.request("/v1/capabilities", deadline=deadline)
        if metadata["status"] != "PASS":
            raise ValueError("capability discovery failed")
        report["capabilities"] = json.loads(raw)
        if report["capabilities"]["tts"].get("model_id") != args.model:
            raise ValueError("discovered model does not match --model")
        (out / "capabilities.json").write_bytes(raw)
        monitor = threading.Thread(target=health_monitor, daemon=True)
        monitor.start()
        # Validate all planned payloads before workload begins.
        for case in cases:
            for protocol in PROTOCOLS:
                request_body(case, protocol, args.model, report["capabilities"])
        long_case = max(cases, key=lambda c: len(c["text"]))
        short_case = min(cases, key=lambda c: len(c["text"]))
        reference = synth(long_case, "native", "cancellation-reference")
        if reference["status"] != "PASS":
            raise RuntimeError("cancellation reference synthesis failed")
        report["cancellations"] = []
        for cycle in range(args.cancel_cycles):
            if time.monotonic() >= deadline:
                raise TimeoutError("deadline before required cancellation cycles")
            health_before = len(report["health"])
            cycle_started = time.time()
            if cycle % 2 == 0:
                cancelled = synth(long_case, "native", "cancel-partial", partial=True)
                if cancelled["status"] != "CANCELLED" or not (0 < cancelled["pcm_bytes"] < reference["pcm_bytes"]):
                    raise RuntimeError("stream cancellation did not close after partial PCM")
                kind = "native-partial"
            else:
                path, body = request_body(long_case, "finite", args.model, report["capabilities"])
                cancelled = save({**client.finite_disconnect(path, body, deadline), "kind": "finite-disconnect",
                                  "protocol": "finite", "request": body, "case_id": long_case["id"]})
                kind = "finite-socket-disconnect"
            restored = recovery(short_case)
            report["cancellations"].append({"cycle": cycle, "kind": kind, "cancel_row": cancelled["index"],
                                            "started_unix_s": cycle_started, "finished_unix_s": time.time(),
                                            "health_row_range": [health_before, len(report["health"])], "recovery": restored})
        # Busy-rejection probe uses an actual concurrent finite request.
        sent = threading.Event()
        holder = []
        def hold():
            try:
                holder.append(synth(long_case, "finite", "busy-holder", sent=sent))
            except Exception as exc:
                holder.append({"status": "FAIL", "error": str(exc)})
        worker = threading.Thread(target=hold, daemon=True)
        worker.start()
        if not sent.wait(min(5, max(0, deadline - time.monotonic()))):
            raise RuntimeError("busy holder did not send request")
        stop.wait(min(.1, max(0, deadline - time.monotonic())))
        busy = synth(short_case, "finite", "busy-rejection")
        worker.join(max(0, min(args.request_timeout + 1, deadline - time.monotonic() + 1)))
        if worker.is_alive() or not holder or holder[0]["status"] != "PASS" or busy["http_status"] != 429:
            raise RuntimeError("busy overlap/reject-not-queue qualification failed")
        report["busy_recovery"] = recovery(short_case)
        invalid, raw = client.request("/v1/tts", {"model": args.model}, deadline=deadline)
        save({**invalid, "kind": "invalid-input"}, raw)
        if invalid["http_status"] not in (400, 422):
            raise RuntimeError("invalid input was not rejected")
        report["invalid_recovery"] = recovery(short_case)
        if workload_mode == "warmup":
            # A warmup is an explicit complete corpus pass plus all cancellation
            # cycles. It has its own report and is never mixed into measurement.
            for case in cases:
                for protocol in PROTOCOLS:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("warmup deadline before complete 45x4 corpus pass")
                    row = synth(case, protocol)
                    if row["status"] != "PASS":
                        raise RuntimeError(f"warmup synthesis failed at row {row['index']}; no automatic retry")
        else:
            index = 0
            while time.monotonic() < deadline:
                case = cases[(index // len(PROTOCOLS)) % len(cases)]
                protocol = PROTOCOLS[index % len(PROTOCOLS)]
                row = synth(case, protocol)
                index += 1
                if row["status"] != "PASS":
                    raise RuntimeError(f"synthesis failed at row {row['index']}; no automatic retry")
        report["status"] = "COMPLETE"
    except Exception as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
    finally:
        stop.set()
        if worker is not None and worker.is_alive():
            client.close_active()
            worker.join(2)
            if worker.is_alive():
                report["failures"].append("busy holder failed to drain after socket shutdown")
        if monitor:
            monitor.join(2)
            if monitor.is_alive():
                report["failures"].append("health monitor failed to drain")
        report["elapsed_s"] = time.monotonic() - started
        report["maximum_drain_grace_s"] = args.request_timeout
        report["finished_unix_s"] = time.time()
        report["aggregate"] = aggregate(report["rows"])
        report["http_status_counts"] = dict(Counter(str(r.get("http_status")) for r in report["rows"]))
        report["health_p95_ms"] = percentile([r["latency_ms"] for r in report["health"]], 95)
        report["evidence_body_bytes"] = evidence_bytes
        report["evidence_bytes_before_report"] = sum(path.stat().st_size for path in out.iterdir() if path.is_file())
        try:
            report["telemetry"] = telemetry_summary(args.telemetry_jsonl, start_unix, report["finished_unix_s"], args.warmup_seconds)
        except Exception as exc:
            report["telemetry"] = {"status": "UNVERIFIED", "error": str(exc)}
        telemetry = report["telemetry"]
        report["route_coverage"] = route_coverage(report["rows"], cases)
        report["disk_accounting"] = disk_gate(telemetry, args.platform)
        observed = {(r.get("case_id"), r.get("protocol")) for r in report["rows"] if r.get("kind") == "synthesis" and r["status"] == "PASS"}
        gates = {"completed": report["status"] == "COMPLETE" and not report["failures"],
                 "min_success": report["aggregate"]["successes"] >= args.min_success,
                 "all_protocols": set(r.get("protocol") for r in report["rows"] if r.get("kind") == "synthesis" and r["status"] == "PASS") == set(PROTOCOLS),
                 "corpus_protocol_coverage": all((case["id"], protocol) in observed for case in cases for protocol in PROTOCOLS),
                 "explicit_native_case_route_coverage": report["route_coverage"]["explicit_native_case_route_coverage"],
                 "cancel_cycles": len(report.get("cancellations", [])) >= args.cancel_cycles,
                 "health": bool(report["health"]) and all(r["http_status"] == 200 for r in report["health"]) and report["health_p95_ms"] < 500,
                 "target_telemetry": telemetry["status"] == "MEASURED" and telemetry.get("rss_plateau_pass", False) and telemetry.get("process_identity_stable", False) and telemetry.get("coverage_pass", False) and telemetry.get("max_sample_gap_s", float("inf")) <= 5,
                 "disk": report["disk_accounting"]["pass"]}
        if workload_mode == "warmup":
            # Target telemetry and disk accounting begin only after warmup.
            # The remaining gates prove the warmup itself was complete.
            gates.pop("target_telemetry")
            gates.pop("disk")
            report["warmup_contract"] = {
                "required_distinct_case_protocols": len(cases) * len(PROTOCOLS),
                "observed_distinct_case_protocols": len(observed),
                "required_cancel_cycles": args.cancel_cycles,
                "measurement_exclusion": "separate invocation completed before target sampler starts",
            }
        report["gates"] = gates
        report["service_measurement_gates"] = "PASS" if all(gates.values()) else "NOT_PASSED"
        # This report alone cannot certify native runtime logs/coexistence or
        # paired prechange performance. Do not label the whole gate PASS.
        report["qualification"] = "UNVERIFIED_EXTERNAL_GATES" if all(gates.values()) else "NOT_PASSED"
        report["performance_comparison"] = "UNVERIFIED: requires paired prechange approved-profile baseline; expanded corpus is baseline collection"
        report["coexistence_evidence"] = "UNVERIFIED: combine externally with authenticated contender lifetime and other-container load; phase label alone is not evidence"
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--platform", required=True, choices=("rk3576", "rk3588"))
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--cancel-cycles", type=int, default=20)
    parser.add_argument("--min-success", type=int, default=100)
    parser.add_argument("--phase", choices=("baseline", "coexistence"), default="baseline")
    parser.add_argument("--workload-mode", choices=("measurement", "warmup"), default="measurement")
    parser.add_argument("--telemetry-jsonl")
    parser.add_argument("--warmup-seconds", type=float, default=60)
    parser.add_argument("--health-interval", type=float, default=1)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--max-evidence-mib", type=float, default=64)
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in (args.health_interval, args.request_timeout, args.max_evidence_mib)) or args.warmup_seconds < 0:
        parser.error("interval/timeout/evidence budget must be positive and warmup nonnegative")
    report = run(args)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "health", "capabilities", "telemetry")}, allow_nan=False))
    return 0 if report["service_measurement_gates"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
