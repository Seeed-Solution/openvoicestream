"""Model-level v0.9.1 artifact source and downloader contracts."""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import sys
import tarfile
import threading
import time
import types
import re
from pathlib import Path

import pytest

from server.core import leaf_composition as lc
from server.core import qwen3_artifact_downloader as qad


ASR_REPO = "harvestsu/qwen3-asr-0.6b-jetson-artifacts"
ASR_REV = "9a82e1ae0fd8dce3ab090e66ae72e3b99ec9c9bf"


def _archive(files: dict[str, bytes]) -> bytes:
    """Build a tiny tar.gz payload with exact payload-relative paths."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tf:
        for rel, data in files.items():
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def _schema_v2(model_id: str, files: dict[str, bytes], archive_name: str = "payload.tar.gz") -> tuple[dict, bytes]:
    payload = _archive(files)
    manifest = {
        "schema_version": 2,
        "model_id": model_id,
        "files": {
            rel: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            for rel, data in files.items()
        },
        "payload": {
            "path": archive_name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        },
    }
    return manifest, payload


def _required_payload(required: list[str]) -> dict[str, bytes]:
    """Materialize file-or-directory requirements into a small payload."""
    result: dict[str, bytes] = {}
    for index, rel in enumerate(required):
        # Profile entries name directory roots for engine sets. A marker file
        # beneath the root exercises directory-presence verification while
        # extension-bearing entries remain regular files.
        if Path(rel).suffix:
            result[rel] = f"artifact-{index}".encode()
        else:
            result[f"{rel}/required.bin"] = f"artifact-{index}".encode()
    return result


def _install_mocks(monkeypatch, manifests: dict[tuple[str, str], tuple[dict, bytes]]):
    """Patch network calls and return fetch/download event lists."""
    fetches: list[tuple[str, str, str, str]] = []
    downloads: list[tuple[str, str, str]] = []

    def fake_fetch(model_id, *, repo=None, revision=None, manifest_path=None, endpoint=None, allowed_redirect_hosts=None):
        key = (str(repo), str(revision))
        fetches.append((str(model_id), key[0], key[1], str(manifest_path)))
        try:
            manifest, _ = manifests[key]
        except KeyError as exc:  # pragma: no cover - makes fixture failures clear
            raise AssertionError(f"unexpected manifest source {key}") from exc
        return json.loads(json.dumps(manifest))

    def fake_download(rel_path, dest, expected_sha256=None, expected_size=None, *, repo=None, revision=None, endpoint=None, allowed_redirect_hosts=None, **kwargs):
        key = (str(repo), str(revision))
        downloads.append((str(rel_path), key[0], key[1]))
        manifest, payload = manifests[key]
        assert rel_path == manifest["payload"]["path"]
        assert expected_sha256 == manifest["payload"]["sha256"]
        assert expected_size == manifest["payload"]["size"]
        Path(dest).write_bytes(payload)
        return Path(dest)

    from server.core import hf_artifacts

    monkeypatch.setattr(hf_artifacts, "fetch_manifest", fake_fetch)
    monkeypatch.setattr(hf_artifacts, "download_file", fake_download)
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")
    return fetches, downloads


def test_leaf_model_sources_keep_repositories_independent():
    registry = lc.load_registry()
    sources = lc.resolve_model_sources(
        [
            "asr.qwen3_asr_v091.orin-nx.n1",
            "tts.qwen3_tts_v091.orin-nx.n1",
        ],
        registry,
    )
    by_model = {source.model_id: source for source in sources}
    assert by_model["qwen3-asr"].repo == ASR_REPO
    assert by_model["qwen3-asr"].canonical_id == "qwen3-asr-0.6b"
    assert by_model["qwen3-tts-customvoice"].repo == (
        "harvestsu/qwen3-tts-0.6b-customvoice-jetson-artifacts"
    )
    assert by_model["qwen3-asr"].files != by_model["qwen3-tts-customvoice"].files


def test_schema_v2_payload_install_cache_hit_and_independent_repos(tmp_path, monkeypatch):
    requests = [
        {
            "model_id": "asr",
            "canonical_model_id": "asr-canonical",
            "repo": "org/asr-artifacts",
            "revision": "asr-rev",
            "required_files": ["engines/asr"],
        },
        {
            "model_id": "tts",
            "canonical_model_id": "tts-canonical",
            "repo": "org/tts-artifacts",
            "revision": "tts-rev",
            "required_files": ["engines/tts", "models/ref.bin"],
        },
    ]
    manifests = {}
    for request in requests:
        files = _required_payload(request["required_files"])
        manifests[(request["repo"], request["revision"])] = _schema_v2(
            request["canonical_model_id"], files
        )
    fetches, downloads = _install_mocks(monkeypatch, manifests)
    cache_root = tmp_path / "cache"
    requests = [
        {
            **request,
            "cache_root": str(cache_root),
            "root": str(tmp_path / request["canonical_model_id"]),
        }
        for request in requests
    ]

    assert qad.ensure_model_requests(requests)
    assert {event[1] for event in downloads} == {"org/asr-artifacts", "org/tts-artifacts"}
    assert (cache_root / "asr-canonical" / "manifest.json").is_file()
    assert (cache_root / "tts-canonical" / "manifest.json").is_file()
    first_fetches = len(fetches)
    first_downloads = len(downloads)

    # A valid manifest and all SHA-256 locks make the second invocation a
    # cache hit; no repository is touched again.
    assert qad.ensure_model_requests(requests)
    assert len(fetches) == first_fetches
    assert len(downloads) == first_downloads


def test_schema_v2_hash_drift_redownload_failure_leaves_cache_uninstalled(tmp_path, monkeypatch):
    required = ["engines/asr"]
    files = _required_payload(required)
    manifest, payload = _schema_v2("asr-canonical", files)
    key = ("org/asr-artifacts", "locked")
    manifests = {key: (manifest, payload)}
    _install_mocks(monkeypatch, manifests)
    cache_root = tmp_path / "cache"
    request = {
        "model_id": "asr",
        "canonical_model_id": "asr-canonical",
        "repo": key[0],
        "revision": key[1],
        "required_files": required,
        "cache_root": str(cache_root),
        "root": str(tmp_path / "runtime"),
    }
    qad.ensure_model_requests([request])
    installed = cache_root / "asr-canonical" / "engines/asr/required.bin"
    installed.write_bytes(b"tampered")

    from server.core import hf_artifacts

    def corrupt_download(rel_path, dest, **kwargs):
        Path(dest).write_bytes(b"not-the-payload")
        return Path(dest)

    monkeypatch.setattr(hf_artifacts, "download_file", corrupt_download)
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        qad.ensure_model_requests([request])
    # The old cache is retained until a complete replacement is verified.
    assert installed.read_bytes() == b"tampered"


def test_strict_revision_manifest_and_atomic_materialization(tmp_path, monkeypatch):
    revision = "0123456789abcdef0123456789abcdef01234567"
    required = ["engine.bin"]
    files = {"engine.bin": b"strict-engine"}
    manifest, payload = _schema_v2("strict-canonical", files)
    manifests = {("org/strict", revision): (manifest, payload)}
    _install_mocks(monkeypatch, manifests)
    cache_root = tmp_path / "cache"
    runtime = tmp_path / "runtime"
    request = {
        "model_id": "strict",
        "canonical_model_id": "strict-canonical",
        "repo": "org/strict",
        "revision": revision,
        "required_files": required,
        "cache_root": str(cache_root),
        "root": str(runtime),
        "strict": True,
    }
    assert qad.ensure_model_requests([request])
    repo_cache = "org-strict-" + hashlib.sha256(b"org/strict").hexdigest()
    active = cache_root / repo_cache / "strict-canonical" / revision / "engine.bin"
    assert active.read_bytes() == files["engine.bin"]
    assert runtime.is_symlink() and runtime.resolve() == active.parent.resolve()

    # A failed replacement leaves the previously materialized link and bytes.
    from server.core import hf_artifacts
    old_download = hf_artifacts.download_file

    def fail_download(*args, **kwargs):
        raise RuntimeError("fixture interrupted")

    monkeypatch.setattr(hf_artifacts, "download_file", fail_download)
    active.write_bytes(b"stale-active")
    active_before = active.read_bytes()
    with pytest.raises(RuntimeError, match="fixture interrupted"):
        qad.ensure_model_requests([request])
    assert active.read_bytes() == active_before
    assert runtime.is_symlink() and runtime.resolve() == active.parent.resolve()
    monkeypatch.setattr(hf_artifacts, "download_file", old_download)


@pytest.mark.parametrize("missing", [None, "_source", "model_id", "canonical_model_id", "repo", "revision"])
def test_strict_cache_requires_complete_source_identity_before_materialize(
    tmp_path, monkeypatch, missing
):
    model_id = "strict"
    canonical = "strict-canonical"
    repo = "org/strict"
    revision = "0123456789abcdef0123456789abcdef01234567"
    digest = hashlib.sha256(b"cached").hexdigest()
    cache = tmp_path / "cache" / qad._strict_cache_repo(repo) / canonical / revision
    cache.mkdir(parents=True)
    (cache / "engine.bin").write_bytes(b"cached")
    source = {
        "model_id": model_id,
        "canonical_model_id": canonical,
        "repo": repo,
        "revision": revision,
    }
    if missing is None:
        source_value = source
    elif missing == "_source":
        source_value = None
    else:
        source.pop(missing)
        source_value = source
    manifest = {
        "model_id": canonical,
        "_source": source_value,
        "files": {"engine.bin": {"sha256": digest, "size": 6}},
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")
    monkeypatch.setattr(
        qad,
        "_fetch_model_manifest",
        lambda *args, **kwargs: calls.append(args) or (_ for _ in ()).throw(
            RuntimeError("network must not repair invalid strict cache")
        ),
    )
    request = {
        "model_id": model_id,
        "canonical_model_id": canonical,
        "repo": repo,
        "revision": revision,
        "required_files": ["engine.bin"],
        "cache_root": str(tmp_path / "cache"),
        "root": str(tmp_path / "runtime"),
        "strict": True,
    }
    if missing is None:
        assert qad.ensure_model_requests([request]) is True
        assert calls == []
    else:
        with pytest.raises(RuntimeError, match="network must not repair"):
            qad.ensure_model_requests([request])
        assert calls


@pytest.mark.parametrize("field,value", [("model_id", "other"), ("canonical_model_id", "other-canonical"), ("repo", "org/other"), ("revision", "f" * 40)])
def test_strict_cache_rejects_source_identity_mismatch_before_materialize(
    tmp_path, monkeypatch, field, value
):
    model_id, canonical, repo = "strict", "strict-canonical", "org/strict"
    revision = "0123456789abcdef0123456789abcdef01234567"
    cache = tmp_path / "cache" / qad._strict_cache_repo(repo) / canonical / revision
    cache.mkdir(parents=True)
    (cache / "engine.bin").write_bytes(b"cached")
    source = {"model_id": model_id, "canonical_model_id": canonical, "repo": repo, "revision": revision}
    source[field] = value
    (cache / "manifest.json").write_text(json.dumps({
        "model_id": canonical,
        "_source": source,
        "files": {"engine.bin": {"sha256": hashlib.sha256(b"cached").hexdigest(), "size": 6}},
    }), encoding="utf-8")
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")
    monkeypatch.setattr(qad, "_fetch_model_manifest", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network must not repair mismatch")))
    with pytest.raises(RuntimeError, match="network must not repair"):
        qad.ensure_model_artifacts(model_id, repo, ["engine.bin"], revision=revision,
            canonical_model_id=canonical, cache_root=tmp_path / "cache", strict=True)


@pytest.mark.parametrize("revision", ["main", "", "abc", "g" * 41])
def test_strict_rejects_non_immutable_revision_before_fetch(revision, tmp_path, monkeypatch):
    called = False

    def fetch(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("strict validation must precede fetch")

    from server.core import hf_artifacts
    monkeypatch.setattr(hf_artifacts, "fetch_manifest", fetch)
    with pytest.raises(RuntimeError, match="immutable 40-hex"):
        qad.ensure_model_artifacts(
            "strict", "org/strict", ["engine.bin"], revision=revision,
            cache_root=tmp_path / "cache", strict=True,
        )
    assert not called


def test_strict_rejects_missing_or_bool_manifest_lock(tmp_path, monkeypatch):
    revision = "0123456789abcdef0123456789abcdef01234567"
    from server.core import hf_artifacts

    for metadata in ({"size": 4}, {"sha256": "0" * 64, "size": True}):
        monkeypatch.setattr(
            hf_artifacts, "fetch_manifest",
            lambda *args, metadata=metadata, **kwargs: {
                "model_id": "strict", "files": {"engine.bin": metadata}
            },
        )
        monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")
        with pytest.raises(RuntimeError, match="strict model manifest"):
            qad.ensure_model_artifacts(
                "strict", "org/strict", ["engine.bin"], revision=revision,
                cache_root=tmp_path / "cache", strict=True,
            )


@pytest.mark.parametrize("model_id,canonical,locks", [
    ("strict", "strict", {
        "engine.bin": {"sha256": "a" * 64, "size": 1},
        "models/strict/engine.bin": {"sha256": "a" * 64, "size": 1},
    }),
    ("flat", "flat", {
        "engine.bin": {"sha256": "a" * 64, "size": 1},
        "models/flat/engine.bin": {"sha256": "b" * 64, "size": 1},
    }),
])
def test_strict_rejects_normalized_manifest_alias_before_download(tmp_path, monkeypatch, model_id, canonical, locks):
    revision = "0123456789abcdef0123456789abcdef01234567"
    from server.core import hf_artifacts
    downloads = []
    monkeypatch.setattr(hf_artifacts, "fetch_manifest", lambda *args, **kwargs: {"model_id": canonical, "files": locks})
    monkeypatch.setattr(hf_artifacts, "download_file", lambda *args, **kwargs: downloads.append(args))
    active = tmp_path / "active"
    active.mkdir()
    (active / "engine.bin").write_bytes(b"old-active")
    runtime = tmp_path / "runtime"
    runtime.symlink_to(active, target_is_directory=True)
    with pytest.raises(RuntimeError, match="path collision after normalization"):
        qad.ensure_model_artifacts(
            model_id, f"org/{model_id}", ["engine.bin"], revision=revision,
            canonical_model_id=canonical, cache_root=tmp_path / "cache",
            root=runtime, strict=True,
        )
    assert downloads == []
    assert runtime.is_symlink() and runtime.resolve() == active.resolve()
    assert (active / "engine.bin").read_bytes() == b"old-active"


def test_strict_cache_separates_repo_and_rejects_raw_paths(tmp_path, monkeypatch):
    revision = "0123456789abcdef0123456789abcdef01234567"
    manifests = {}
    for repo, data in (("org/one", b"one"), ("org/two", b"two")):
        manifests[(repo, revision)] = _schema_v2("canonical", {"engine.bin": data})
    _install_mocks(monkeypatch, manifests)
    requests = [
        {"model_id": "m-" + repo.rsplit('/', 1)[-1], "canonical_model_id": "canonical", "repo": repo,
         "revision": revision, "required_files": ["engine.bin"], "strict": True,
         "cache_root": str(tmp_path / "cache"), "root": str(tmp_path / repo.replace('/', '-'))}
        for repo in ("org/one", "org/two")
    ]
    assert qad.ensure_model_requests(requests)
    repo_one = "org-one-" + hashlib.sha256(b"org/one").hexdigest()
    repo_two = "org-two-" + hashlib.sha256(b"org/two").hexdigest()
    assert (tmp_path / "cache" / repo_one / "canonical" / revision / "engine.bin").read_bytes() == b"one"
    assert (tmp_path / "cache" / repo_two / "canonical" / revision / "engine.bin").read_bytes() == b"two"

    from server.core import hf_artifacts
    monkeypatch.setattr(hf_artifacts, "fetch_manifest", lambda *args, **kwargs: {
        "model_id": "canonical", "files": {"../escape": {"sha256": "0" * 64, "size": 1}}
    })
    with pytest.raises(RuntimeError, match="unsafe"):
        qad.ensure_model_artifacts(
            "m", "org/path", ["engine.bin"], revision=revision,
            canonical_model_id="canonical", cache_root=tmp_path / "bad", strict=True,
        )


def test_strict_payload_path_rejects_raw_traversal_and_absolute(tmp_path):
    digest = "a" * 64
    for raw in ("../payload.tar.gz", "/payload.tar.gz", "dir/../payload.tar.gz"):
        with pytest.raises(RuntimeError, match="unsafe"):
            qad._archive_spec(
                {"payload": {"path": raw, "sha256": digest, "size": 1}},
                "model", "canonical", strict=True,
            )


@pytest.mark.parametrize("embedded", [
    {"model_id": "evil"},
    {"model_id": "canonical", "files": {"engine.bin": {"sha256": "b" * 64, "size": 99}}},
])
def test_strict_embedded_payload_manifest_conflict_preserves_active(tmp_path, monkeypatch, embedded):
    revision = "0123456789abcdef0123456789abcdef01234567"
    files = {"engine.bin": b"engine"}
    payload = _archive({"engine.bin": files["engine.bin"], "manifest.json": json.dumps(embedded).encode()})
    manifest = {
        "schema_version": 2, "model_id": "canonical",
        "files": {"engine.bin": {"sha256": hashlib.sha256(files["engine.bin"]).hexdigest(), "size": len(files["engine.bin"])}},
        "payload": {"path": "payload.tar.gz", "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)},
    }
    _install_mocks(monkeypatch, {("org/strict", revision): (manifest, payload)})
    cache = tmp_path / "cache"; runtime = tmp_path / "runtime"
    runtime_target = tmp_path / "old-active"; runtime_target.mkdir(); (runtime_target / "engine.bin").write_bytes(b"old")
    runtime.symlink_to(runtime_target, target_is_directory=True)
    request = {"model_id": "strict", "canonical_model_id": "canonical", "repo": "org/strict", "revision": revision,
               "required_files": ["engine.bin"], "cache_root": str(cache), "root": str(runtime), "strict": True}
    with pytest.raises(RuntimeError, match="embedded payload manifest"):
        qad.ensure_model_requests([request])
    assert runtime.is_symlink() and runtime.resolve() == runtime_target.resolve()
    assert (runtime_target / "engine.bin").read_bytes() == b"old"


def test_strict_endpoint_defaults_mirror_and_accepts_explicit(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert qad._strict_endpoint() == "https://hf-mirror.com"
    monkeypatch.setenv("HF_ENDPOINT", "")
    assert qad._strict_endpoint() == "https://hf-mirror.com"
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert qad._strict_endpoint() == "https://hf-mirror.com"
    monkeypatch.setenv("HF_ENDPOINT", "ftp://invalid")
    with pytest.raises(RuntimeError, match=r"http\(s\)"):
        qad._strict_endpoint()


def test_strict_redirect_policy_blocks_cross_host_and_allows_same_host():
    hits = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append((self.server.server_address, self.path))
            if self.path == "/redirect":
                self.send_response(302); self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/manifest"); self.end_headers(); return
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"model_id":"m","files":{}}')
        def log_message(self, *_): pass
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_port}/redirect"
        from server.core import hf_artifacts
        with pytest.raises(hf_artifacts.ArtifactError):
            hf_artifacts._open(url, timeout=2, allowed_redirect_hosts=frozenset({"localhost"}))
        assert len(hits) == 0
        with hf_artifacts._open(url, timeout=2, allowed_redirect_hosts=frozenset({"127.0.0.1"})) as response:
            assert response.read().startswith(b"{")
        assert len(hits) == 2
    finally:
        httpd.shutdown(); thread.join(timeout=2)


def test_strict_default_observed_mirror_redirect_hosts_are_exact(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert qad._strict_endpoint_hosts(qad._strict_endpoint()) == frozenset({
        "hf-mirror.com", "huggingface.co", "us.aws.cdn.hf.co",
        "cas-bridge.xethub.hf.co",
    })
    assert qad._strict_endpoint_hosts("https://custom.example") == frozenset({"custom.example"})


def test_strict_mirror_xet_bridge_redirect_is_exactly_allowlisted():
    from server.core import hf_artifacts

    hosts = qad._strict_endpoint_hosts("https://hf-mirror.com")
    handler = hf_artifacts._AllowedRedirect(hosts, "https")
    req = __import__("urllib.request", fromlist=["Request"]).Request(
        "https://hf-mirror.com/org/repo/resolve/rev/payload.tar"
    )
    redirected = handler.redirect_request(
        req, None, 302, "Found", {},
        "https://cas-bridge.xethub.hf.co/xet-bridge/payload.tar",
    )
    assert redirected.full_url.startswith("https://cas-bridge.xethub.hf.co/")

    for hostile in (
        "https://evil-cas-bridge.xethub.hf.co/payload.tar",
        "https://unknown.hf.co/payload.tar",
    ):
        with pytest.raises(hf_artifacts.ArtifactError, match="allowlisted"):
            handler.redirect_request(req, None, 302, "Found", {}, hostile)

    assert qad._strict_endpoint_hosts("https://custom.example") == frozenset({"custom.example"})


def test_https_redirect_cannot_downgrade_to_http():
    from server.core import hf_artifacts
    handler = hf_artifacts._AllowedRedirect(frozenset({"localhost"}), "https")
    req = __import__("urllib.request", fromlist=["Request"]).Request("https://localhost/start")
    with pytest.raises(hf_artifacts.ArtifactError, match="allowlisted"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://localhost/next")


def _resume_fixture(data: bytes, behavior):
    events = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            start, end = (int(match.group(1)), int(match.group(2))) if match else (0, len(data) - 1)
            events.append((start, end))
            status, body, content_range = behavior(len(events), start, end, data)
            self.send_response(status)
            if content_range:
                self.send_header("Content-Range", content_range)
            self.send_header("Content-Length", str(end - start + 1 if status == 206 else len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        def log_message(self, *_): pass
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    return httpd, thread, events


def test_strict_resume_interrupted_then_second_call_continues_without_prefix(tmp_path):
    from server.core import hf_artifacts
    data = bytes((i % 251 for i in range(5 * 1024 * 1024)))
    digest = hashlib.sha256(data).hexdigest()
    def behavior(index, start, end, payload):
        full = payload[start:end + 1]
        if index <= 3:
            return 206, full[:(1 << 20) + 123], f"bytes {start}-{end}/{len(payload)}"
        return 206, full, f"bytes {start}-{end}/{len(payload)}"
    httpd, thread, events = _resume_fixture(data, behavior)
    try:
        url = f"http://127.0.0.1:{httpd.server_port}"
        dest = tmp_path / "payload.tar.gz"
        kwargs = dict(expected_sha256=digest, expected_size=len(data), repo="org/model", revision="0" * 40,
                      endpoint=url, allowed_redirect_hosts=frozenset({"127.0.0.1"}), resume=True, resume_key="fixture")
        with pytest.raises(hf_artifacts.ArtifactError, match="short body"):
            hf_artifacts.download_file("payload.tar.gz", dest, **kwargs)
        assert not dest.exists()
        assert events[0][0] == 0 and events[-1][0] == 2 << 20
        assert hf_artifacts.download_file("payload.tar.gz", dest, **kwargs) == dest
        assert dest.read_bytes() == data
        assert events[-1][0] == 3 << 20
    finally:
        httpd.shutdown(); thread.join(timeout=2)


@pytest.mark.parametrize("socket_mode", ["missing", "detached"])
def test_strict_resume_body_without_live_socket_never_calls_blocking_read(
    tmp_path, monkeypatch, socket_mode
):
    from server.core import hf_artifacts

    body = b"body-without-live-socket"
    digest = hashlib.sha256(body).hexdigest()

    class Socket:
        def fileno(self):
            return -1

        def settimeout(self, _timeout):
            raise OSError("detached")

    class Response:
        status = 206
        headers = {
            "Content-Range": f"bytes 0-{len(body) - 1}/{len(body)}",
            "Content-Length": str(len(body)),
        }

        def __init__(self):
            self.read_calls = 0
            self.fp = None if socket_mode == "missing" else types.SimpleNamespace(
                raw=types.SimpleNamespace(_sock=Socket())
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return "http://127.0.0.1/payload"

        def read(self, *_args):
            self.read_calls += 1
            raise AssertionError("body read without a deadline-capable socket")

    response = Response()

    class Opener:
        def open(self, *_args, **_kwargs):
            return response

    monkeypatch.setattr(
        hf_artifacts.urllib.request, "build_opener", lambda *_args: Opener()
    )
    with pytest.raises(hf_artifacts.ArtifactError, match="socket timeout unavailable"):
        hf_artifacts.download_file(
            "payload.tar.gz",
            tmp_path / "payload.tar.gz",
            expected_sha256=digest,
            expected_size=len(body),
            repo="org/model",
            revision="0" * 40,
            endpoint="http://127.0.0.1",
            allowed_redirect_hosts=frozenset({"127.0.0.1"}),
            resume=True,
            resume_key=f"body-{socket_mode}",
            deadline=time.monotonic() + 1,
            max_retries=0,
        )
    assert response.read_calls == 0


def test_strict_resume_complete_range_with_closed_socket_does_not_read_again(
    tmp_path, monkeypatch
):
    from server.core import hf_artifacts

    body = b"complete-range"
    digest = hashlib.sha256(body).hexdigest()

    class Socket:
        def __init__(self):
            self.closed = False

        def fileno(self):
            return -1 if self.closed else 42

        def settimeout(self, _timeout):
            assert not self.closed

    class Response:
        status = 206
        headers = {
            "Content-Range": f"bytes 0-{len(body) - 1}/{len(body)}",
            "Content-Length": str(len(body)),
        }

        def __init__(self):
            self.sock = Socket()
            self.read_calls = 0
            self.fp = types.SimpleNamespace(raw=types.SimpleNamespace(_sock=self.sock))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return "http://127.0.0.1/payload"

        def read(self, *_args):
            self.read_calls += 1
            raise AssertionError("read() must not be used for deadline-bound ranges")

        def read1(self, *_args):
            self.read_calls += 1
            assert self.read_calls == 1
            self.sock.closed = True
            return body

    response = Response()

    class Opener:
        def open(self, *_args, **_kwargs):
            return response

    monkeypatch.setattr(
        hf_artifacts.urllib.request, "build_opener", lambda *_args: Opener()
    )
    dest = tmp_path / "payload.tar.gz"
    assert hf_artifacts.download_file(
        "payload.tar.gz",
        dest,
        expected_sha256=digest,
        expected_size=len(body),
        repo="org/model",
        revision="0" * 40,
        endpoint="http://127.0.0.1",
        allowed_redirect_hosts=frozenset({"127.0.0.1"}),
        resume=True,
        resume_key="closed-after-complete",
        deadline=time.monotonic() + 1,
        max_retries=0,
    ) == dest
    assert response.read_calls == 1
    assert dest.read_bytes() == body


@pytest.mark.parametrize("mode", ["chunked", "without-read1"])
def test_strict_resume_deadline_rejects_unbounded_response_before_body(
    tmp_path, monkeypatch, mode
):
    from server.core import hf_artifacts

    body = b"unbounded-response"
    digest = hashlib.sha256(body).hexdigest()

    class Socket:
        def fileno(self):
            return 42

        def settimeout(self, _timeout):
            pass

    class Response:
        status = 206
        headers = {
            "Content-Range": f"bytes 0-{len(body) - 1}/{len(body)}",
            "Content-Length": str(len(body)),
        }

        def __init__(self):
            self.read_calls = 0
            self.fp = types.SimpleNamespace(raw=types.SimpleNamespace(_sock=Socket()))
            if mode == "chunked":
                self.headers = {**self.headers, "Transfer-Encoding": "chunked"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return "http://127.0.0.1/payload"

        def read(self, *_args):
            self.read_calls += 1
            raise AssertionError("body read before bounded-response rejection")

        if mode == "chunked":
            def read1(self, *_args):
                self.read_calls += 1
                raise AssertionError("chunked body read before rejection")

    response = Response()

    class Opener:
        def open(self, *_args, **_kwargs):
            return response

    monkeypatch.setattr(
        hf_artifacts.urllib.request, "build_opener", lambda *_args: Opener()
    )
    expected = (
        "fixed-length deadline-bound response"
        if mode == "chunked"
        else "bounded body read unavailable"
    )
    with pytest.raises(hf_artifacts.ArtifactError, match=expected):
        hf_artifacts.download_file(
            "payload.tar.gz",
            tmp_path / "payload.tar.gz",
            expected_sha256=digest,
            expected_size=len(body),
            repo="org/model",
            revision="0" * 40,
            endpoint="http://127.0.0.1",
            allowed_redirect_hosts=frozenset({"127.0.0.1"}),
            resume=True,
            resume_key=f"unbounded-{mode}",
            deadline=time.monotonic() + 1,
            max_retries=0,
        )
    assert response.read_calls == 0


def test_strict_resume_real_range_slow_trickle_honors_total_deadline(tmp_path):
    from server.core import hf_artifacts

    body = b"0123456789abcdef"
    digest = hashlib.sha256(body).hexdigest()
    delay = 0.02

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.headers.get("Range") == f"bytes=0-{len(body) - 1}"
            self.send_response(206)
            self.send_header("Content-Range", f"bytes 0-{len(body) - 1}/{len(body)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for value in body:
                self.wfile.write(bytes((value,)))
                self.wfile.flush()
                time.sleep(delay)

        def log_message(self, *_args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        start = time.monotonic()
        with pytest.raises(hf_artifacts.ArtifactError, match="deadline exceeded"):
            hf_artifacts.download_file(
                "payload.tar.gz",
                tmp_path / "payload.tar.gz",
                expected_sha256=digest,
                expected_size=len(body),
                repo="org/model",
                revision="0" * 40,
                endpoint=f"http://127.0.0.1:{httpd.server_port}",
                allowed_redirect_hosts=frozenset({"127.0.0.1"}),
                resume=True,
                resume_key="slow-trickle",
                deadline=start + 0.08,
                max_retries=0,
            )
        elapsed = time.monotonic() - start
        assert elapsed < 0.20, elapsed
    finally:
        httpd.shutdown()
        thread.join(timeout=2)


@pytest.mark.parametrize("behavior,match", [
    (lambda _i, _s, _e, payload: (200, payload, None), "HTTP 206"),
    (lambda _i, s, e, payload: (206, payload[s:e + 1], f"bytes {s + 1}-{e + 1}/{len(payload)}"), "Content-Range"),
])
def test_strict_resume_rejects_ignored_range_or_wrong_content_range(tmp_path, behavior, match):
    from server.core import hf_artifacts
    data = b"resume-fixture" * 10000
    digest = hashlib.sha256(data).hexdigest()
    httpd, thread, events = _resume_fixture(data, behavior)
    try:
        dest = tmp_path / "payload.tar.gz"
        with pytest.raises(hf_artifacts.ArtifactError, match=match):
            hf_artifacts.download_file("payload.tar.gz", dest, expected_sha256=digest, expected_size=len(data),
                repo="org/model", revision="0" * 40, endpoint=f"http://127.0.0.1:{httpd.server_port}",
                allowed_redirect_hosts=frozenset({"127.0.0.1"}), resume=True, resume_key="fixture", max_retries=0)
        assert not dest.exists()
        assert events
    finally:
        httpd.shutdown(); thread.join(timeout=2)


def test_strict_resume_state_gap_and_symlink_are_rejected(tmp_path):
    from server.core import hf_artifacts
    dest = tmp_path / "payload.tar.gz"
    part, state, lock = hf_artifacts._resume_paths(dest, "fixture")
    part.write_bytes(b"partial")
    state.write_text(json.dumps({"identity": "fixture", "expected_size": 7, "expected_sha256": "0" * 64, "committed": 3}))
    with pytest.raises(hf_artifacts.ArtifactError, match="partial size mismatch"):
        hf_artifacts.download_file("payload.tar.gz", dest, expected_sha256="0" * 64, expected_size=7,
            repo="org/model", revision="0" * 40, endpoint="http://127.0.0.1:1",
            allowed_redirect_hosts=frozenset({"127.0.0.1"}), resume=True, resume_key="fixture")
    part.unlink()
    part.symlink_to(dest)
    with pytest.raises(hf_artifacts.ArtifactError, match="regular file"):
        hf_artifacts.download_file("payload.tar.gz", dest, expected_sha256="0" * 64, expected_size=7,
            repo="org/model", revision="0" * 40, endpoint="http://127.0.0.1:1",
            allowed_redirect_hosts=frozenset({"127.0.0.1"}), resume=True, resume_key="fixture")


def test_strict_resume_full_sha_failure_keeps_owned_partial(tmp_path):
    from server.core import hf_artifacts
    data = b"sha-failure" * 10000
    httpd, thread, events = _resume_fixture(data, lambda _i, s, e, payload: (206, payload[s:e + 1], f"bytes {s}-{e}/{len(payload)}"))
    try:
        dest = tmp_path / "payload.tar.gz"
        with pytest.raises(hf_artifacts.ArtifactError, match="SHA256/size"):
            hf_artifacts.download_file("payload.tar.gz", dest, expected_sha256="f" * 64, expected_size=len(data),
                repo="org/model", revision="0" * 40, endpoint=f"http://127.0.0.1:{httpd.server_port}",
                allowed_redirect_hosts=frozenset({"127.0.0.1"}), resume=True, resume_key="sha-failure")
        assert not dest.exists()
        assert any(p.name.endswith(".part") for p in tmp_path.iterdir())
    finally:
        httpd.shutdown(); thread.join(timeout=2)


def test_strict_large_huggingface_response_rejected_before_read(tmp_path, monkeypatch):
    from server.core import hf_artifacts
    class Response:
        def __init__(self): self.reads = 0
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def geturl(self): return "https://huggingface.co/org/repo/resolve/rev/payload.tar"
        def read(self, *_): self.reads += 1; raise AssertionError("body read before host rejection")
    response = Response()
    monkeypatch.setattr(hf_artifacts, "_open", lambda *args, **kwargs: response)
    with pytest.raises(hf_artifacts.ArtifactError, match="remained on huggingface.co"):
        hf_artifacts.download_file("payload.tar", tmp_path / "payload.tar", expected_sha256="a" * 64,
                                   expected_size=11 * 1024 * 1024, allowed_redirect_hosts=frozenset({"huggingface.co"}))
    assert response.reads == 0


def test_strict_resume_large_huggingface_response_rejected_before_read(tmp_path, monkeypatch):
    from server.core import hf_artifacts
    class Response:
        status = 206
        headers = {"Content-Range": "bytes 0-1048575/11534336", "Content-Length": str(1 << 20)}
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def geturl(self): return "https://huggingface.co/org/repo/resolve/rev/payload.tar"
        def read(self, *_): raise AssertionError("body read before host rejection")
    response = Response()
    class Opener:
        def open(self, *_args, **_kwargs): return response
    monkeypatch.setattr(hf_artifacts.urllib.request, "build_opener", lambda *_args: Opener())
    with pytest.raises(hf_artifacts.ArtifactError, match="remained on huggingface.co"):
        hf_artifacts.download_file("payload.tar", tmp_path / "payload.tar", expected_sha256="a" * 64,
            expected_size=11 * 1024 * 1024, repo="org/repo", revision="0" * 40,
            endpoint="https://hf-mirror.com", allowed_redirect_hosts=frozenset({"hf-mirror.com", "huggingface.co"}),
            resume=True, resume_key="large", deadline=time.monotonic() + 1)


def test_strict_noarchive_fetches_declared_files_and_wrong_hash_preserves_active(tmp_path, monkeypatch):
    revision = "0123456789abcdef0123456789abcdef01234567"
    content = b"strict-flat-engine"; digest = hashlib.sha256(content).hexdigest()
    manifest = {"model_id": "canonical", "files": {"engine.bin": {"sha256": digest, "size": len(content)}}}
    from server.core import hf_artifacts
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")
    monkeypatch.setattr(hf_artifacts, "fetch_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(hf_artifacts, "download_file", lambda rel, dest, **kwargs: (Path(dest).write_bytes(content), Path(dest))[1])
    cache = tmp_path / "cache"; runtime = tmp_path / "runtime"
    request = {"model_id":"flat", "canonical_model_id":"canonical", "repo":"org/flat", "revision":revision, "required_files":["engine.bin"], "cache_root":str(cache), "root":str(runtime), "strict":True}
    assert qad.ensure_model_requests([request])
    old_target = runtime.resolve(); assert (old_target / "engine.bin").read_bytes() == content
    bad = {"model_id":"canonical", "files":{"engine.bin":{"sha256":"a" * 64, "size":len(content)}}}
    monkeypatch.setattr(hf_artifacts, "fetch_manifest", lambda *args, **kwargs: bad)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        qad.ensure_model_requests([{**request, "revision": "fedcba9876543210fedcba9876543210fedcba98"}])
    assert runtime.resolve() == old_target


def test_strict_resume_recovers_data_state_gap_and_checks_history(tmp_path):
    from server.core import hf_artifacts
    data = bytes((i % 251 for i in range(2 * 1024 * 1024 + 17)))
    digest = hashlib.sha256(data).hexdigest()
    httpd, thread, events = _resume_fixture(data, lambda _i, s, e, payload: (206, payload[s:e + 1], f"bytes {s}-{e}/{len(payload)}"))
    try:
        dest = tmp_path / "gap.bin"
        identity = "gap"
        part, state, _ = hf_artifacts._resume_paths(dest, identity)
        first = data[:1 << 20]
        part.write_bytes(first + b"uncommitted-tail")
        state.write_text(json.dumps({
            "schema": "strict-resume-v2", "status": "active", "identity": identity,
            "expected_size": len(data), "expected_sha256": digest, "block_size": 1 << 20,
            "committed": 1 << 20, "blocks": [{"offset": 0, "size": 1 << 20, "sha256": hashlib.sha256(first).hexdigest()}],
        }))
        assert hf_artifacts.download_file("gap.bin", dest, expected_sha256=digest, expected_size=len(data),
            repo="org/model", revision="0" * 40, endpoint=f"http://127.0.0.1:{httpd.server_port}",
            allowed_redirect_hosts=frozenset({"127.0.0.1"}), resume=True, resume_key=identity) == dest
        assert dest.read_bytes() == data and events[0][0] == 1 << 20
    finally:
        httpd.shutdown(); thread.join(timeout=2)


def test_strict_resume_rejects_modified_committed_history(tmp_path):
    from server.core import hf_artifacts
    data = b"x" * (1 << 20)
    dest = tmp_path / "history.bin"; part, state, _ = hf_artifacts._resume_paths(dest, "history")
    part.write_bytes(b"y" * len(data))
    state.write_text(json.dumps({"schema": "strict-resume-v2", "status": "active", "identity": "history",
        "expected_size": len(data), "expected_sha256": hashlib.sha256(data).hexdigest(), "block_size": 1 << 20,
        "committed": len(data), "blocks": [{"offset": 0, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}))
    with pytest.raises(hf_artifacts.ArtifactError, match="block hash mismatch"):
        hf_artifacts.download_file("history.bin", dest, expected_sha256=hashlib.sha256(data).hexdigest(), expected_size=len(data),
            repo="org/model", revision="0" * 40, endpoint="http://127.0.0.1:1", allowed_redirect_hosts=frozenset({"127.0.0.1"}),
            resume=True, resume_key="history", deadline=time.monotonic() + 1)


def test_strict_resume_lock_deadline_is_bounded(tmp_path, monkeypatch):
    from server.core import hf_artifacts
    real_flock = hf_artifacts.fcntl.flock
    monkeypatch.setattr(hf_artifacts.fcntl, "flock", lambda _fd, _op: (_ for _ in ()).throw(BlockingIOError()))
    start = time.monotonic()
    with pytest.raises(hf_artifacts.ArtifactError, match="waiting for lock"):
        hf_artifacts.download_file("lock.bin", tmp_path / "lock.bin", expected_sha256="a" * 64, expected_size=1,
            repo="org/model", revision="0" * 40, endpoint="http://127.0.0.1:1", allowed_redirect_hosts=frozenset({"127.0.0.1"}),
            resume=True, resume_key="lock", deadline=time.monotonic() + 0.02)
    assert time.monotonic() - start < 0.5
    monkeypatch.setattr(hf_artifacts.fcntl, "flock", real_flock)


@pytest.mark.parametrize("swap", ["part", "state", "lock"])
def test_strict_resume_post_lock_path_swap_preserves_foreign_target(tmp_path, monkeypatch, swap):
    from server.core import hf_artifacts
    dest = tmp_path / "swap.bin"
    part, state, lock = hf_artifacts._resume_paths(dest, "swap")
    victim = tmp_path / f"victim-{swap}"
    victim.write_bytes(b"VICTIM")
    state.write_text(json.dumps({
        "schema": "strict-resume-v2", "status": "active", "identity": "swap",
        "expected_size": 1, "expected_sha256": "a" * 64, "block_size": 1 << 20,
        "committed": 0, "blocks": [],
    }))
    real_flock = hf_artifacts.fcntl.flock
    swapped = False
    def swap_after_lock(fd, op):
        nonlocal swapped
        real_flock(fd, op)
        if op & hf_artifacts.fcntl.LOCK_EX and not swapped:
            swapped = True
            target = {"part": part, "state": state, "lock": lock}[swap]
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(victim)
    monkeypatch.setattr(hf_artifacts.fcntl, "flock", swap_after_lock)
    with pytest.raises(hf_artifacts.ArtifactError):
        hf_artifacts.download_file(
            "swap.bin", dest, expected_sha256="a" * 64, expected_size=1,
            repo="org/model", revision="0" * 40, endpoint="http://127.0.0.1:1",
            allowed_redirect_hosts=frozenset({"127.0.0.1"}), resume=True,
            resume_key="swap", max_retries=0, deadline=time.monotonic() + 1,
        )
    assert victim.read_bytes() == b"VICTIM"


@pytest.mark.parametrize("foreign_kind", ["symlink", "file"])
def test_strict_state_foreign_unique_temp_is_never_overwritten(tmp_path, monkeypatch, foreign_kind):
    from server.core import hf_artifacts
    class UUID:
        hex = "fixed"
    monkeypatch.setattr(hf_artifacts.uuid, "uuid4", lambda: UUID())
    state = tmp_path / "state"
    foreign = tmp_path / ".state.fixed.new"
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"keep")
    if foreign_kind == "symlink":
        foreign.symlink_to(sentinel)
    else:
        foreign.write_bytes(b"foreign")
    with pytest.raises(FileExistsError):
        hf_artifacts._atomic_json(state, {"committed": 0})
    assert foreign.is_symlink() if foreign_kind == "symlink" else foreign.read_bytes() == b"foreign"
    assert sentinel.read_bytes() == b"keep"


def test_strict_noarchive_verifies_twelve_files_before_activation(tmp_path, monkeypatch):
    from server.core import hf_artifacts
    files = {f"engines/file-{i}.bin": f"payload-{i}".encode() for i in range(12)}
    manifest = {"model_id": "twelve", "files": {rel: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)} for rel, data in files.items()}}
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")
    monkeypatch.setattr(hf_artifacts, "fetch_manifest", lambda *args, **kwargs: manifest)
    def fetch(rel, dest, **kwargs):
        Path(dest).write_bytes(files[rel]); return Path(dest)
    monkeypatch.setattr(hf_artifacts, "download_file", fetch)
    runtime = tmp_path / "runtime"
    assert qad.ensure_model_artifacts("twelve", "org/twelve", files, revision="0" * 40,
        canonical_model_id="twelve", cache_root=tmp_path / "cache", root=runtime, strict=True)
    assert runtime.is_symlink() and all((runtime / rel).read_bytes() == data for rel, data in files.items())


def test_legacy_nested_manifest_falls_back_to_snapshot_download(tmp_path, monkeypatch):
    from server.core import hf_artifacts

    required = ["engines/legacy"]
    files = _required_payload(required)
    manifest, _ = _schema_v2("legacy", files)
    # Legacy repositories have no root manifest; the compatibility path asks
    # for models/<model_id>/manifest.json and then retries the root form.
    calls: list[str] = []

    def fake_fetch(model_id, *, repo=None, revision=None, manifest_path=None):
        calls.append(str(manifest_path or f"models/{model_id}/manifest.json"))
        if len(calls) == 1:
            raise hf_artifacts.ArtifactError("404")
        return {"model_id": "legacy", "files": manifest["files"]}

    def fake_snapshot_download(**kwargs):
        root = Path(kwargs["local_dir"])
        for rel, data in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return str(root)

    monkeypatch.setattr(hf_artifacts, "fetch_manifest", fake_fetch)
    # The runtime image installs huggingface_hub for the legacy snapshot path;
    # keep this unit test hermetic when only the stdlib HF resolver is present.
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")
    request = {
        "model_id": "legacy",
        "repo": "org/legacy-artifacts",
        "revision": "main",
        "required_files": required,
        "cache_root": str(tmp_path / "cache"),
        "root": str(tmp_path / "runtime"),
    }
    assert qad.ensure_model_requests([request])
    assert calls == ["manifest.json", "models/legacy/manifest.json"]
    assert (tmp_path / "cache" / "legacy" / "engines/legacy/required.bin").is_file()


@pytest.mark.parametrize(
    "profile_path",
    sorted(Path(__file__).resolve().parents[2].glob("configs/profiles/jetson-edgellm-v091-*.json")),
    ids=lambda path: Path(path).stem,
)
def test_every_formal_v091_profile_downloads_each_model_source_separately(
    profile_path: Path, tmp_path, monkeypatch
):
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    entries = profile.get("model_artifacts")
    assert isinstance(entries, list) and entries
    manifests: dict[tuple[str, str], tuple[dict, bytes]] = {}
    requests = []
    for index, entry in enumerate(entries):
        assert {
            "model_id",
            "repo",
            "revision",
            "canonical_model_id",
            "root",
            "required_files",
        } <= entry.keys()
        required = list(entry["required_files"])
        files = _required_payload(required)
        key = (entry["repo"], entry["revision"])
        manifests[key] = _schema_v2(entry["canonical_model_id"], files)
        requests.append(
            {
                **entry,
                "cache_root": str(tmp_path / "cache"),
                "root": str(tmp_path / "runtime" / entry["canonical_model_id"]),
            }
        )
    fetches, downloads = _install_mocks(monkeypatch, manifests)
    assert qad.ensure_model_requests(requests)
    repos = {entry["repo"] for entry in entries}
    assert {event[1] for event in downloads} == repos
    assert {event[1] for event in fetches} == repos
    for entry in entries:
        cache = tmp_path / "cache" / entry["canonical_model_id"]
        assert (cache / "manifest.json").is_file()
        # Every profile-declared required path is materialized under its own
        # canonical cache; no shared aggregate directory is used.
        for rel in entry["required_files"]:
            path = cache / rel
            assert path.exists(), (profile_path, rel)
