from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
import socketserver
import sys
import tarfile
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.core import qwen3_artifact_downloader as downloader


def _payload(path: Path, mode: str) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode=mode) as archive:
        data = b"plain archive payload\n"
        info = tarfile.TarInfo("config.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    path.write_bytes(payload.getvalue())
    return payload.getvalue()


@pytest.mark.parametrize(
    ("archive_name", "tar_mode"),
    [("payload.tar", "w:"), ("payload.tar.gz", "w:gz"), ("payload.tar.bz2", "w:bz2")],
)
def test_strict_download_extracts_declared_archive_format(
    monkeypatch, tmp_path: Path, archive_name: str, tar_mode: str
) -> None:
    payload = tmp_path / archive_name
    payload_bytes = _payload(payload, tar_mode)
    digest = hashlib.sha256(payload_bytes).hexdigest()
    file_digest = hashlib.sha256(b"plain archive payload\n").hexdigest()
    revision = "a" * 40
    manifest = {
        "model_id": "qwen3-asr-0.6b",
        "files": {"config.json": {"sha256": file_digest, "size": 22}},
        "payload": {
            "path": f"v011/orin-nx-sm87/asr-b2-20261005/{archive_name}",
            "sha256": digest,
            "size": len(payload_bytes),
        },
    }

    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(
        downloader,
        "_fetch_model_manifest",
        lambda *args, **kwargs: manifest,
    )

    from server.core import hf_artifacts

    def download_file(_remote: str, target: Path, **kwargs: object) -> None:
        target.write_bytes(payload_bytes)

    monkeypatch.setattr(hf_artifacts, "download_file", download_file)
    cache_root = tmp_path / "cache"
    runtime = tmp_path / "runtime"

    assert downloader.ensure_model_artifacts(
        "qwen3-asr-0.6b",
        "harvestsu/qwen3-asr-0.6b-jetson-artifacts",
        ("config.json",),
        revision=revision,
        cache_root=cache_root,
        root=runtime,
        strict=True,
    )
    assert (runtime / "config.json").read_bytes() == b"plain archive payload\n"
    assert json.loads((runtime / "manifest.json").read_text())["payload"]["path"].endswith(
        archive_name
    )


def test_strict_reuses_verified_legacy_gzip_name_without_http(
    monkeypatch, tmp_path: Path
) -> None:
    payload = tmp_path / "payload.tar"
    payload_bytes = _payload(payload, "w:")
    digest = hashlib.sha256(payload_bytes).hexdigest()
    file_digest = hashlib.sha256(b"plain archive payload\n").hexdigest()
    revision = "b" * 40
    repo = "harvestsu/qwen3-asr-0.6b-jetson-artifacts"
    model_id = "qwen3-asr-0.6b"
    manifest = {
        "model_id": model_id,
        "files": {"config.json": {"sha256": file_digest, "size": 22}},
        "payload": {"path": "payload.tar", "sha256": digest, "size": len(payload_bytes)},
    }
    cache_root = tmp_path / "cache"
    cache = cache_root / downloader._strict_cache_repo(repo) / model_id / revision
    cache.mkdir(parents=True)
    identity = "|".join((str(cache), repo, revision, "payload.tar", digest))
    resume_digest = hashlib.sha256(identity.encode()).hexdigest()
    legacy = cache.parent / f".qwen-archive-{resume_digest}.tar.gz"
    legacy.write_bytes(payload_bytes)

    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest)
    from server.core import hf_artifacts

    def no_http(*args, **kwargs):
        raise AssertionError("verified legacy archive should avoid HTTP")

    monkeypatch.setattr(hf_artifacts, "_open", no_http)
    runtime = tmp_path / "runtime"
    assert downloader.ensure_model_artifacts(
        model_id, repo, ("config.json",), revision=revision,
        cache_root=cache_root, root=runtime, strict=True,
    )
    target = cache.parent / f".qwen-archive-{resume_digest}.tar"
    assert target.is_file()
    assert target.stat().st_ino == legacy.stat().st_ino
    assert legacy.is_file()


def test_strict_rejects_wrong_legacy_hash_and_downloads_new_name(monkeypatch, tmp_path: Path) -> None:
    payload = tmp_path / "payload.tar"
    payload_bytes = _payload(payload, "w:")
    digest = hashlib.sha256(payload_bytes).hexdigest()
    file_digest = hashlib.sha256(b"plain archive payload\n").hexdigest()
    revision = "c" * 40
    repo = "harvestsu/qwen3-asr-0.6b-jetson-artifacts"
    model_id = "qwen3-asr-0.6b"
    manifest = {
        "model_id": model_id,
        "files": {"config.json": {"sha256": file_digest, "size": 22}},
        "payload": {"path": "payload.tar", "sha256": digest, "size": len(payload_bytes)},
    }
    cache_root = tmp_path / "cache"
    cache = cache_root / downloader._strict_cache_repo(repo) / model_id / revision
    cache.mkdir(parents=True)
    identity = "|".join((str(cache), repo, revision, "payload.tar", digest))
    resume_digest = hashlib.sha256(identity.encode()).hexdigest()
    legacy = cache.parent / f".qwen-archive-{resume_digest}.tar.gz"
    legacy.write_bytes(b"wrong")
    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest)
    from server.core import hf_artifacts
    monkeypatch.setattr(hf_artifacts, "download_file", lambda _remote, target, **kwargs: target.write_bytes(payload_bytes))
    runtime = tmp_path / "runtime"
    assert downloader.ensure_model_artifacts(
        model_id, repo, ("config.json",), revision=revision,
        cache_root=cache_root, root=runtime, strict=True,
    )
    target = cache.parent / f".qwen-archive-{resume_digest}.tar"
    assert target.is_file() and target.stat().st_ino != legacy.stat().st_ino
    assert legacy.read_bytes() == b"wrong"


def test_strict_archive_tmp_dir_cleans_only_verified_archive(monkeypatch, tmp_path: Path) -> None:
    payload = tmp_path / "payload.tar"
    payload_bytes = _payload(payload, "w:")
    digest = hashlib.sha256(payload_bytes).hexdigest()
    file_digest = hashlib.sha256(b"plain archive payload\n").hexdigest()
    revision = "d" * 40
    repo = "org/split"
    manifest = {
        "model_id": "split-model",
        "files": {"config.json": {"sha256": file_digest, "size": 22}},
        "payload": {"path": "payload.tar", "sha256": digest, "size": len(payload_bytes)},
    }
    tmp_archive_dir = tmp_path / "archive-tmp"
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(tmp_archive_dir))
    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest)
    from server.core import hf_artifacts

    seen: list[Path] = []

    def download_file(_remote: str, target: Path, **kwargs: object) -> None:
        seen.append(target)
        target.write_bytes(payload_bytes)

    monkeypatch.setattr(hf_artifacts, "download_file", download_file)
    assert downloader.ensure_model_artifacts(
        "split-model", repo, ("config.json",), revision=revision,
        cache_root=tmp_path / "cache", root=tmp_path / "runtime", strict=True,
    )
    assert seen and seen[0].parent == tmp_archive_dir
    assert seen[0].name.endswith(".tar") and not seen[0].exists()
    assert (tmp_path / "runtime" / "config.json").is_file()


def test_strict_archive_tmp_dir_retains_partial_after_failure(monkeypatch, tmp_path: Path) -> None:
    payload = tmp_path / "payload.tar"
    payload_bytes = _payload(payload, "w:")
    digest = hashlib.sha256(payload_bytes).hexdigest()
    file_digest = hashlib.sha256(b"plain archive payload\n").hexdigest()
    revision = "e" * 40
    repo = "org/split-failure"
    manifest = {
        "model_id": "split-failure",
        "files": {"config.json": {"sha256": file_digest, "size": 22}},
        "payload": {"path": "payload.tar", "sha256": digest, "size": len(payload_bytes)},
    }
    tmp_archive_dir = tmp_path / "archive-tmp"
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(tmp_archive_dir))
    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest)
    from server.core import hf_artifacts

    def failed_download(_remote: str, target: Path, **kwargs: object) -> None:
        target.write_bytes(payload_bytes[:11])
        raise TimeoutError("test timeout")

    monkeypatch.setattr(hf_artifacts, "download_file", failed_download)
    with pytest.raises(TimeoutError):
        downloader.ensure_model_artifacts(
            "split-failure", repo, ("config.json",), revision=revision,
            cache_root=tmp_path / "cache", root=tmp_path / "runtime", strict=True,
        )
    partials = list(tmp_archive_dir.iterdir())
    assert partials and partials[0].read_bytes() == payload_bytes[:11]


@pytest.mark.parametrize("value", ["relative/tmp", "bad-link"])
def test_strict_archive_tmp_dir_rejects_unsafe_path(monkeypatch, tmp_path: Path, value: str) -> None:
    if value == "bad-link":
        target = tmp_path / value
        target.symlink_to(tmp_path, target_is_directory=True)
        value = str(target)
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", value)
    with pytest.raises(RuntimeError, match="QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR|temporary path"):
        downloader._strict_archive_tmp_dir()


def test_strict_archive_tmp_dir_rejects_writable_parent_and_ancestor_symlink(
    monkeypatch, tmp_path: Path
) -> None:
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o777)
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(unsafe / "archive-tmp"))
    with pytest.raises(RuntimeError, match="writable|owned or sticky"):
        downloader._strict_archive_tmp_dir()
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(link / "archive-tmp"))
    with pytest.raises(RuntimeError, match="ancestor|directory"):
        downloader._strict_archive_tmp_dir()


def test_strict_legacy_archive_cross_filesystem_copy_preserves_source(tmp_path: Path, monkeypatch) -> None:
    legacy = tmp_path / "legacy.tar.gz"
    target = tmp_path / "target.tar"
    data = b"verified legacy archive"
    legacy.write_bytes(data)
    real_link = os.link
    calls = {"count": 0}

    def cross_fs_once(src, dst, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError(18, "cross-device link")
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(downloader.os, "link", cross_fs_once)
    assert downloader._adopt_legacy_archive(
        legacy, target, expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest()
    )
    assert target.read_bytes() == data and legacy.read_bytes() == data


def test_strict_resume_real_http_range_reuses_same_identity(tmp_path: Path, monkeypatch) -> None:
    data = b"a" * (2 * 1024 * 1024 + 123)
    ranges: list[tuple[int, int]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        calls = 0

        def do_GET(self):
            raw = self.headers.get("Range", "")
            start, end = (int(v) for v in raw.removeprefix("bytes=").split("-"))
            ranges.append((start, end))
            Handler.calls += 1
            body = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if Handler.calls == 1:
                self.wfile.write(body[: 1 << 20])
            else:
                self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    from server.core import hf_artifacts

    monkeypatch.setattr(
        hf_artifacts,
        "file_url",
        lambda *_args, **_kwargs: f"http://127.0.0.1:{server.server_address[1]}/payload.tar",
    )
    dest = tmp_path / "payload.tar"
    identity = "repo|revision|payload.tar|" + hashlib.sha256(data).hexdigest()
    try:
        assert hf_artifacts.download_file(
            "payload.tar", dest, expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_size=len(data), repo="org/model", revision="b" * 40,
            resume=True, resume_key=identity,
        ) == dest
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert ranges[0][0] == 0 and ranges[1][0] == 1 << 20
    assert dest.read_bytes() == data
    digest = hashlib.sha256(identity.encode()).hexdigest()
    assert not (tmp_path / f".{dest.name}.{digest}.part").exists()
    assert not (tmp_path / f".{dest.name}.{digest}.part.state").exists()
    assert (tmp_path / f".{dest.name}.{digest}.part.lock").exists()


@pytest.mark.parametrize("failure_point", ["_safe_extract", "_install", "_materialize"])
def test_strict_split_failure_retains_archive_and_unknown_sentinel(
    monkeypatch, tmp_path: Path, failure_point: str
) -> None:
    payload = tmp_path / "payload.tar"
    payload_bytes = _payload(payload, "w:")
    digest = hashlib.sha256(payload_bytes).hexdigest()
    file_digest = hashlib.sha256(b"plain archive payload\n").hexdigest()
    revision = "f" * 40
    manifest = {
        "model_id": "split-failure-stage",
        "files": {"config.json": {"sha256": file_digest, "size": 22}},
        "payload": {"path": "payload.tar", "sha256": digest, "size": len(payload_bytes)},
    }
    tmp_archive_dir = tmp_path / "archive-tmp"
    sentinel = tmp_archive_dir / "foreign.sentinel"
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(tmp_archive_dir))
    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest)
    from server.core import hf_artifacts
    monkeypatch.setattr(hf_artifacts, "download_file", lambda _remote, target, **kwargs: target.write_bytes(payload_bytes))
    if failure_point == "_safe_extract":
        monkeypatch.setattr(downloader, failure_point, lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("extract")))
    elif failure_point == "_install":
        monkeypatch.setattr(downloader, failure_point, lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("install")))
    else:
        monkeypatch.setattr(downloader, failure_point, lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("materialize")))
    tmp_archive_dir.mkdir()
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError):
        downloader.ensure_model_artifacts(
            "split-failure-stage", "org/failure-stage", ("config.json",), revision=revision,
            cache_root=tmp_path / "cache", root=tmp_path / "runtime", strict=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert any(item.name.startswith(".qwen-archive-") for item in tmp_archive_dir.iterdir())


def test_cleanup_claim_does_not_delete_replaced_original(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "archive-tmp" / ".qwen-archive-id.tar"
    archive.parent.mkdir()
    data = b"verified archive"
    archive.write_bytes(data)
    original = archive
    real_rename = os.rename

    def race(src, dst):
        real_rename(src, dst)
        if Path(src) == original:
            original.write_bytes(b"new same-directory sentinel")

    monkeypatch.setattr(downloader.os, "rename", race)
    downloader._cleanup_verified_archive(
        archive, archive.parent, expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest()
    )
    assert archive.read_bytes() == b"new same-directory sentinel"


def test_cleanup_permission_failure_restores_claimed_archive(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "archive-tmp" / ".qwen-archive-id.tar"
    archive.parent.mkdir()
    data = b"verified archive"
    archive.write_bytes(data)
    real_unlink = os.unlink

    def denied(path, *args, **kwargs):
        if str(path).endswith(".qwen-archive-id.tar") and ".qwen-archive-quarantine-" in str(path):
            raise PermissionError("test cleanup permission failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(downloader.os, "unlink", denied)
    with pytest.raises(PermissionError):
        downloader._cleanup_verified_archive(
            archive, archive.parent, expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest()
        )
    assert archive.read_bytes() == data


def test_cleanup_restore_race_preserves_new_original_and_claimed_archive(
    monkeypatch, tmp_path: Path
) -> None:
    archive = tmp_path / "archive-tmp" / ".qwen-archive-id.tar"
    archive.parent.mkdir()
    data = b"verified archive"
    archive.write_bytes(data)
    real_unlink = os.unlink
    real_link = os.link

    def denied(path, *args, **kwargs):
        if str(path).endswith(".qwen-archive-id.tar") and ".qwen-archive-quarantine-" in str(path):
            raise PermissionError("test cleanup permission failure")
        return real_unlink(path, *args, **kwargs)

    def race_restore(src, dst, *args, **kwargs):
        if Path(dst) == archive:
            archive.write_bytes(b"new original sentinel")
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(downloader.os, "unlink", denied)
    monkeypatch.setattr(downloader.os, "link", race_restore)
    with pytest.raises(PermissionError):
        downloader._cleanup_verified_archive(
            archive,
            archive.parent,
            expected_size=len(data),
            expected_sha256=hashlib.sha256(data).hexdigest(),
        )
    assert archive.read_bytes() == b"new original sentinel"
    quarantines = list(archive.parent.glob(".qwen-archive-quarantine-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / archive.name).read_bytes() == data


def test_cached_fastpath_retries_cleanup_without_network(monkeypatch, tmp_path: Path) -> None:
    payload = tmp_path / "payload.tar"
    payload_bytes = _payload(payload, "w:")
    digest = hashlib.sha256(payload_bytes).hexdigest()
    file_digest = hashlib.sha256(b"plain archive payload\n").hexdigest()
    revision = "1" * 40
    repo = "org/cached"
    manifest = {
        "model_id": "cached-model",
        "files": {"config.json": {"sha256": file_digest, "size": 22}},
        "payload": {"path": "payload.tar", "sha256": digest, "size": len(payload_bytes)},
    }
    tmp_archive_dir = tmp_path / "archive-tmp"
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(tmp_archive_dir))
    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest)
    from server.core import hf_artifacts
    monkeypatch.setattr(hf_artifacts, "download_file", lambda _remote, target, **kwargs: target.write_bytes(payload_bytes))
    cache_root = tmp_path / "cache"
    assert downloader.ensure_model_artifacts(
        "cached-model", repo, ("config.json",), revision=revision,
        cache_root=cache_root, root=tmp_path / "runtime", strict=True,
    )
    cache = cache_root / downloader._strict_cache_repo(repo) / "cached-model" / revision
    archive_spec = downloader._archive_spec(manifest, "cached-model", strict=True)
    archive_path = downloader._strict_archive_path(cache, repo, revision, archive_spec, tmp_archive_dir)
    archive_path.write_bytes(payload_bytes)
    monkeypatch.setattr(downloader, "_fetch_model_manifest", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network manifest fetch")))
    monkeypatch.setattr(hf_artifacts, "download_file", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network archive fetch")))
    assert downloader.ensure_model_artifacts(
        "cached-model", repo, ("config.json",), revision=revision,
        cache_root=cache_root, root=tmp_path / "runtime", strict=True,
    )
    assert not archive_path.exists()
