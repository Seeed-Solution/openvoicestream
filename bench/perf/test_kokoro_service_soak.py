"""Local synthetic transport tests, not speech quality or board qualification."""
import importlib.util
import http.client
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import struct
import threading
import time
from types import SimpleNamespace
import wave

import numpy as np
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


soak = load("kokoro_service_soak")
sampler = load("kokoro_proc_sampler")
PCM = (np.sin(np.arange(2400) / 20) * 10000).astype("<i2").tobytes()


def wav(pcm=PCM, unknown=False):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(pcm)
    raw = bytearray(buffer.getvalue())
    if unknown:
        struct.pack_into("<I", raw, 4, 0xffffffff)
        struct.pack_into("<I", raw, 40, 0xffffffff)
    return bytes(raw)


CAPS = {"tts": {"model_id": "kokoro", "languages": {"mode": "multi_language", "values": None},
                "audio": {"response_formats": ["wav", "pcm"]}, "streaming": {"supported": True},
                "voices": {"items": [{"id": 0}], "aliases": []}}}
CASE = {"id": "en-1", "route": "en-US", "text": "Hello world."}


@pytest.mark.parametrize("protocol,raw", [("finite", wav()), ("openai-wav", wav(unknown=True)), ("native", struct.pack("<I", 24000) + PCM), ("openai-pcm", PCM)])
def test_real_protocol_and_audio_statistics(protocol, raw):
    assert soak.pcm_payload(raw, protocol) == PCM
    assert soak.audio_stats(PCM)["waveform_pass"] is True


def test_wav_headers_are_not_audio_and_malformed_fails():
    assert soak.wav_info(wav()[:44])[1] == 0
    with pytest.raises(ValueError, match="finite lengths"):
        soak.pcm_payload(wav(unknown=True), "finite")
    with pytest.raises(ValueError, match="RIFF length"):
        soak.pcm_payload(wav()[:-2], "finite")
    with pytest.raises(ValueError, match="odd"):
        soak.pcm_payload(PCM[:-1], "openai-pcm")
    assert not soak.audio_stats(b"\0\0" * 100)["waveform_pass"]
    assert not soak.audio_stats(b"\xff\x7f" * 100)["waveform_pass"]


def test_metadata_driven_payload_and_openai_has_no_language():
    for protocol in soak.PROTOCOLS:
        _, body = soak.request_body({**CASE, "voice": 0, "speed": 2, "pitch": 1}, protocol, "kokoro", CAPS)
        assert "speed" not in body and "pitch" not in body
        assert ("language" in body) == (protocol in ("finite", "native"))
        assert body["voice"] == 0
    with pytest.raises(ValueError, match="voice"):
        soak.request_body({**CASE, "voice": "unadvertised"}, "finite", "kokoro", CAPS)


class FragmentedResponse:
    status = 200
    def __init__(self, fragments):
        self.fragments = iter(fragments)
    def getheaders(self):
        return []
    def read1(self, _):
        time.sleep(.005)
        return next(self.fragments, b"")
    def close(self):
        pass


@pytest.mark.parametrize("protocol,header", [("native", struct.pack("<I", 24000)), ("openai-wav", wav(unknown=True)[:44])])
def test_first_pcm_excludes_fragmented_headers(monkeypatch, protocol, header):
    response = FragmentedResponse([header[:1], header[1:], PCM[:2], PCM[2:]])
    class Connection:
        sock = None
        def request(self, *args, **kwargs): pass
        def getresponse(self): return response
        def close(self): pass
    client = soak.Client("http://localhost")
    monkeypatch.setattr(client, "connection", lambda _: Connection())
    row, raw = client.request("/test", {}, protocol=protocol, partial=True)
    assert row["status"] == "CANCELLED"
    assert row["pcm_bytes"] == 2
    assert row["first_pcm_ms"] >= 14
    assert len(raw) == len(header) + 2


@pytest.mark.parametrize("protocol,raw", [
    ("native", struct.pack("<I", 24000) + PCM),
    ("openai-pcm", PCM),
    ("openai-wav", wav(unknown=True)),
])
@pytest.mark.parametrize("framing", ["content-length", "chunked"])
def test_real_httpresponse_truncated_framing_is_not_audio_success(monkeypatch, protocol, raw, framing):
    # Use Python's real HTTPResponse.read1, which silently returns EOF with
    # a positive .length when the declared Content-Length was not received.
    headers = b"Content-Type: audio/pcm\r\nX-Sample-Rate: 24000\r\nX-Audio-Channels: 1\r\n"
    if framing == "content-length":
        headers += f"Content-Length: {len(raw) * 2}\r\n".encode()
        wire_body = raw
    else:
        headers += b"Transfer-Encoding: chunked\r\n"
        wire_body = f"{len(raw):x}\r\n".encode() + raw + b"\r\n"  # missing terminating zero chunk
    class Socket:
        def makefile(self, *args, **kwargs):
            return io.BytesIO(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n" + wire_body)
    response = http.client.HTTPResponse(Socket())
    response.begin()
    class Connection:
        sock = None
        def request(self, *args, **kwargs): pass
        def getresponse(self): return response
        def close(self): pass
    client = soak.Client("http://localhost")
    monkeypatch.setattr(client, "connection", lambda _: Connection())
    row, received = client.request("/test", {}, protocol=protocol)
    assert row["status"] == "FAIL"
    assert received == raw
    assert "Content-Length" in row["error"] if framing == "content-length" else "IncompleteRead" in row["error"]
    assert "duration_s" not in row  # Cannot enter aggregate as a valid utterance.


def test_partial_disconnect_does_not_require_full_http_content_length(monkeypatch):
    raw = struct.pack("<I", 24000) + PCM
    class Socket:
        def makefile(self, *args, **kwargs):
            return io.BytesIO(f"HTTP/1.1 200 OK\r\nContent-Length: {len(raw) * 2}\r\n\r\n".encode() + raw)
    response = http.client.HTTPResponse(Socket())
    response.begin()
    class Connection:
        sock = None
        def request(self, *args, **kwargs): pass
        def getresponse(self): return response
        def close(self): pass
    client = soak.Client("http://localhost")
    monkeypatch.setattr(client, "connection", lambda _: Connection())
    row, _ = client.request("/test", {}, protocol="native", partial=True)
    assert row["status"] == "CANCELLED"
    assert row["http_body_framing"].startswith("deliberately-partial")


def test_aggregate_reports_unknown_stream_engine_and_actual_weights():
    base = {"status": "PASS", "kind": "synthesis", "language": "en-US", "engine": "unknown", "protocol": "native", "case_id": "c", "first_pcm_ms": 20, "end_to_end_rtf": .5, "wall_s": 1, "duration_s": 2}
    result = soak.aggregate([base, {**base, "wall_s": 3, "duration_s": 3, "end_to_end_rtf": 1}, {**base, "status": "FAIL"}])
    assert result["successes"] == 2
    assert result["weighted_end_to_end_rtf"] == .8
    assert result["groups"][0]["engine"] == "unknown"


def test_openai_intended_languages_do_not_count_as_verified_routes():
    cases = [{"id": language, "route": language} for language in ("en-GB", "ja", "zh")]
    rows = []
    for case in cases:
        for protocol in soak.PROTOCOLS:
            explicit = protocol in ("finite", "native")
            rows.append({"status": "PASS", "kind": "synthesis", "case_id": case["id"], "language": case["route"],
                         "intended_language": case["route"], "verified_route": case["route"] if explicit else "unknown",
                         "selected_route": case["route"] if explicit else "unknown-auto", "engine": "unknown", "protocol": protocol,
                         "wall_s": 1, "duration_s": 2, "first_pcm_ms": 10, "end_to_end_rtf": .5})
    result = soak.aggregate(rows)
    openai = [group for group in result["groups"] if group["protocol"].startswith("openai")]
    assert len(openai) == 2 and all(group["verified_route"] == "unknown" and group["count"] == 3 for group in openai)
    assert {group["intended_language"] for group in result["intended_language_groups"]} == {"en-GB", "ja", "zh"}
    coverage = soak.route_coverage(rows, cases)
    assert coverage["intended_language_protocol_coverage"]
    assert coverage["explicit_native_case_route_coverage"]
    assert not coverage["verified_route_protocol_coverage"]
    assert len(coverage["missing_verified_route_protocols"]) == 6
    assert coverage["unverified_auto_route_successes"] == 6
    assert len(coverage["openai_route_control"]) == 6
    assert all(row["directed_route_qualification"] == "UNVERIFIED" for row in coverage["openai_route_control"])
    # A second case in the same language must have its own explicit-route
    # evidence; a language-level aggregate cannot satisfy case-level coverage.
    missing_case = cases + [{"id": "ja-extra", "route": "ja"}]
    assert not soak.route_coverage(rows, missing_case)["explicit_native_case_route_coverage"]


def test_client_evidence_cannot_offset_target_disk_loss():
    mib = 1024 ** 2
    telemetry = {"status": "MEASURED", "free_disk_start": 512 * mib,
                 "free_disk_end": 472 * mib, "free_disk_min": 472 * mib,
                 "client_evidence_bytes": 50 * mib, "evidence_body_bytes": 50 * mib}
    for platform in ("rk3576", "rk3588"):
        result = soak.disk_gate(telemetry, platform)
        assert not result["pass"]  # client +50 MiB cannot excuse target -40 MiB
        assert result["credited_target_evidence_bytes"] == 0
    assert soak.disk_gate({**telemetry, "free_disk_end": 512 * mib}, "rk3588")["pass"]


def test_target_telemetry_plateau_and_restart(tmp_path):
    rows = [{"unix_s": i, "pid": 99, "process_start_ticks": 20, "rss_bytes": 1000 + i, "free_disk_bytes": 200000000} for i in range(10)]
    path = tmp_path / "telemetry.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = soak.telemetry_summary(path, 0, 9, 1)
    assert result["rss_plateau_pass"] and result["process_identity_stable"]
    for row in rows:
        row["container_pid"] = row.pop("pid")
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert soak.telemetry_summary(path, 0, 9, 1)["process_identity_stable"]
    rows[-1]["process_start_ticks"] = 21
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert not soak.telemetry_summary(path, 0, 9, 1)["process_identity_stable"]
    assert soak.telemetry_summary(None, 0, 1, 0)["status"] == "UNVERIFIED"


def test_sampler_pid_is_read_only_and_stat_comm_can_contain_spaces(tmp_path):
    process = tmp_path / "101"
    process.mkdir()
    (process / "status").write_text("Name:\ttest\nVmRSS:\t42 kB\n")
    fields = ["S"] + ["0"] * 18 + ["12345"]
    (process / "stat").write_text("101 (name with ) parentheses) " + " ".join(fields))
    row = sampler.sample(101, tmp_path, proc_root=tmp_path)
    assert row["rss_bytes"] == 42 * 1024 and row["process_start_ticks"] == 12345


def test_real_tcp_complete_json_disconnect():
    # Real socket, not asyncio Task.cancel: receive declared bytes then EOF.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    received = []
    def server():
        conn, _ = listener.accept()
        conn.settimeout(2)
        with conn:
            raw = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                raw += chunk
            received.append(raw)
    worker = threading.Thread(target=server)
    worker.start()
    try:
        client = soak.Client(f"http://127.0.0.1:{listener.getsockname()[1]}")
        row = client.finite_disconnect("/v1/tts", {"model": "kokoro", "text": "你好。"}, time.monotonic() + 2)
        worker.join(2)
        assert not worker.is_alive() and row["full_json_sent"]
        assert row["server_admission"] == row["native_cancellation"] == "UNVERIFIED"
        head, body = received[0].split(b"\r\n\r\n", 1)
        length = int(next(line.split(b":", 1)[1] for line in head.split(b"\r\n") if line.startswith(b"Content-Length:")))
        assert len(body) == length and json.loads(body)["text"] == "你好。"
    finally:
        listener.close()


def test_absolute_deadline_bounds_slow_response_headers():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    stopped = threading.Event()
    def server():
        conn, _ = listener.accept()
        with conn:
            conn.recv(4096)
            try:
                for byte in b"HTTP/1.1 200 OK\r\n":
                    conn.send(bytes([byte]))
                    if stopped.wait(.05): break
            except OSError:
                pass
    worker = threading.Thread(target=server)
    worker.start()
    try:
        client = soak.Client(f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=.15)
        row, _ = client.request("/health")
        assert row["status"] == "FAIL" and row["wall_s"] < .6
    finally:
        stopped.set()
        worker.join(1)
        listener.close()


def test_runner_retains_failure_without_retry(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"cases": [CASE]}))
    calls = []
    def request(self, path, body=None, **kwargs):
        calls.append(path)
        if path == "/v1/capabilities":
            return {"status": "PASS"}, json.dumps(CAPS).encode()
        return {"status": "FAIL", "http_status": 503, "headers": {}, "wall_s": .01, "error": "injected"}, b""
    monkeypatch.setattr(soak.Client, "request", request)
    args = SimpleNamespace(out=str(tmp_path / "run"), corpus=str(corpus), seconds=.2, cancel_cycles=20, min_success=100,
                           base_url="http://localhost", request_timeout=1, phase="baseline", platform="rk3588", model="kokoro",
                           health_interval=.01, max_evidence_mib=1, telemetry_jsonl=None, warmup_seconds=0)
    result = soak.run(args)
    assert result["status"] == "FAIL" and result["aggregate"]["successes"] == 0
    assert calls.count("/tts/stream") == 1
    assert result["rows"][0]["http_status"] == 503
    assert (tmp_path / "run" / "report.json").is_file()


@pytest.mark.parametrize("workload_mode,target_telemetry", [
    ("measurement", False), ("measurement", True), ("warmup", False),
])
def test_bounded_full_mock_http_soak_all_protocols_cancel_busy_invalid(tmp_path, monkeypatch, workload_mode, target_telemetry):
    """Exercise actual TCP transport and recovery, never a real TTS backend."""
    admission = threading.Lock()
    payloads = []
    disconnects = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *_): pass
        def reply(self, code, raw, content_type="application/json", fragments=False):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            if content_type == "audio/pcm":
                self.send_header("X-Sample-Rate", "24000")
                self.send_header("X-Audio-Channels", "1")
            # Even a stale/pre-synthesis engine header must be ignored for streams.
            self.send_header("X-Kokoro-Engine", "npu")
            self.end_headers()
            try:
                if fragments:
                    for start in range(0, len(raw), 1024):
                        self.wfile.write(raw[start:start + 1024])
                        self.wfile.flush()
                        if start + 1024 < len(raw):
                            time.sleep(.006)
                else:
                    self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                disconnects.append(self.path)
        def do_GET(self):
            self.reply(200, json.dumps(CAPS if self.path == "/v1/capabilities" else {"status": "ok"}).encode())
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payloads.append((self.path, body))
            if "text" not in body and "input" not in body:
                self.reply(422, b'{"error":"missing text"}')
                return
            if not admission.acquire(blocking=False):
                self.reply(429, b'{"error":"busy"}')
                return
            try:
                if self.path == "/v1/tts":
                    time.sleep(.16)
                    self.reply(200, wav(), "audio/wav")
                elif self.path == "/tts/stream":
                    self.reply(200, struct.pack("<I", 24000) + PCM, "application/octet-stream", True)
                elif body["response_format"] == "pcm":
                    self.reply(200, PCM, "audio/pcm", True)
                else:
                    self.reply(200, wav(unknown=True), "audio/wav", True)
            finally:
                admission.release()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"cases": [CASE]}))
    args = SimpleNamespace(out=str(tmp_path / "run"), corpus=str(corpus), seconds=2, cancel_cycles=2, min_success=4,
                           base_url=f"http://127.0.0.1:{server.server_port}", request_timeout=1, phase="baseline", platform="rk3588", model="kokoro",
                           health_interval=.05, max_evidence_mib=1, telemetry_jsonl=None, warmup_seconds=0,
                           workload_mode=workload_mode)
    if target_telemetry:
        # Synthetic telemetry for hardgate composition only, not device evidence.
        monkeypatch.setattr(soak, "telemetry_summary", lambda *_: {
            "status": "MEASURED", "rss_plateau_pass": True, "process_identity_stable": True,
            "coverage_pass": True, "max_sample_gap_s": 1, "free_disk_start": 1024 ** 3,
            "free_disk_end": 1024 ** 3, "free_disk_min": 1024 ** 3})
    try:
        result = soak.run(args)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(1)
    assert result["status"] == "COMPLETE", result["failures"]
    assert result["gates"]["all_protocols"] and result["gates"]["min_success"]
    assert result["gates"]["cancel_cycles"]
    assert result["gates"]["corpus_protocol_coverage"]
    assert result["gates"]["explicit_native_case_route_coverage"]
    assert not result["route_coverage"]["verified_route_protocol_coverage"]
    expected_pass = target_telemetry or workload_mode == "warmup"
    assert result["service_measurement_gates"] == ("PASS" if expected_pass else "NOT_PASSED")
    assert result["qualification"] == ("UNVERIFIED_EXTERNAL_GATES" if expected_pass else "NOT_PASSED")
    if workload_mode == "warmup":
        assert "target_telemetry" not in result["gates"] and "disk" not in result["gates"]
        assert result["warmup_contract"]["observed_distinct_case_protocols"] == 4
    assert result["http_status_counts"]["429"]
    assert result["invalid_recovery"]["status"] == "PASS"
    assert {cycle["kind"] for cycle in result["cancellations"]} == {"native-partial", "finite-socket-disconnect"}
    assert all(r["engine"] == "unknown" for r in result["rows"] if r.get("protocol") in ("native", "openai-pcm", "openai-wav"))
    assert all("language" not in body for path, body in payloads if path == "/v1/audio/speech")
    assert all(r["verified_route"] == "unknown" and r["verified_voice"] == "unknown" for r in result["rows"] if r.get("protocol", "").startswith("openai"))
    assert result["elapsed_s"] < 3.2
