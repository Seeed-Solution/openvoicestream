from __future__ import annotations

import hashlib
import http.server
import socketserver
import sys
import threading
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.core import hf_artifacts


class _RedirectServer:
    def __init__(self, data: bytes, *, location_host: str | None = None, loop: bool = False):
        self.data = data
        self.location_host = location_host
        self.loop = loop
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            methods: list[tuple[str, str]] = []

            def _redirect(self):
                if owner.loop:
                    location = f"http://127.0.0.1:{server.server_address[1]}{self.path}"
                else:
                    host = owner.location_host or "127.0.0.1"
                    location = f"http://{host}:{server.server_address[1]}/payload"
                self.send_response(308)
                self.send_header("Location", location)
                self.end_headers()

            def do_GET(self):
                self.methods.append(("GET", self.path))
                if self.path == "/start" or owner.loop:
                    self._redirect()
                    return
                body = owner.data
                raw_range = self.headers.get("Range")
                if raw_range:
                    start, end = (int(v) for v in raw_range.removeprefix("bytes=").split("-"))
                    body = body[start : end + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(owner.data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_HEAD(self):
                self.methods.append(("HEAD", self.path))
                if self.path == "/start":
                    self._redirect()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(owner.data)))
                self.end_headers()

            def do_POST(self):
                self.methods.append(("POST", self.path))
                if self.path == "/start":
                    self._redirect()
                    return
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args):
                pass

        self.handler = Handler
        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        server = self.server
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def test_308_supports_get_and_head():
    with _RedirectServer(b"payload") as server:
        with hf_artifacts._open(f"{server.base_url}/start") as response:
            assert response.read() == b"payload"
        get_payload_count = server.handler.methods.count(("GET", "/payload"))
        with hf_artifacts._open(f"{server.base_url}/start", method="HEAD") as response:
            assert response.status == 200
        assert ("HEAD", "/payload") in server.handler.methods
        assert server.handler.methods.count(("GET", "/payload")) == get_payload_count


def test_regular_download_follows_308(tmp_path: Path, monkeypatch):
    data = b"regular artifact"
    with _RedirectServer(data) as server:
        monkeypatch.setattr(
            hf_artifacts,
            "file_url",
            lambda *_args, **_kwargs: f"{server.base_url}/start",
        )
        dest = tmp_path / "artifact.bin"
        assert hf_artifacts.download_file(
            "artifact.bin",
            dest,
            expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_size=len(data),
        ) == dest
        assert dest.read_bytes() == data


def test_308_preserves_post_method():
    with _RedirectServer(b"payload") as server:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            hf_artifacts._open(f"{server.base_url}/start", method="POST")
        assert exc_info.value.code == 308
        assert ("POST", "/payload") not in server.handler.methods


def test_308_is_checked_against_redirect_allowlist():
    with _RedirectServer(b"payload", location_host="localhost") as server:
        with pytest.raises(hf_artifacts.ArtifactError, match="not allowlisted"):
            hf_artifacts._open(
                f"{server.base_url}/start",
                allowed_redirect_hosts=frozenset({"127.0.0.1"}),
            )


def test_308_loop_has_a_hop_guard():
    with _RedirectServer(b"payload", loop=True) as server:
        with pytest.raises((urllib.error.HTTPError, hf_artifacts.ArtifactError)):
            hf_artifacts._open(f"{server.base_url}/start")


def test_resumable_download_follows_308_and_preserves_range(tmp_path: Path, monkeypatch):
    data = b"artifact" * (1 << 18)
    with _RedirectServer(data) as server:
        monkeypatch.setattr(
            hf_artifacts,
            "file_url",
            lambda *_args, **_kwargs: f"{server.base_url}/start",
        )
        dest = tmp_path / "artifact.bin"
        digest = hashlib.sha256(data).hexdigest()
        assert hf_artifacts.download_file(
            "artifact.bin",
            dest,
            expected_sha256=digest,
            expected_size=len(data),
            repo="org/repo",
            revision="main",
            resume=True,
            resume_key="308-resume-test",
        ) == dest
        assert dest.read_bytes() == data
        assert any(path == "/payload" for method, path in server.handler.methods if method == "GET")


def test_artifact_endpoint_precedence(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.example")
    monkeypatch.setenv("HF_ARTIFACT_ENDPOINT", "https://artifact.example/")
    assert hf_artifacts._endpoint() == "https://artifact.example"
    assert hf_artifacts._endpoint("https://explicit.example/") == "https://explicit.example"
