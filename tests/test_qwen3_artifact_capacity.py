from __future__ import annotations

import hashlib
import json
import os
import tarfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest

from server.core import qwen3_artifact_downloader as downloader


def test_capacity_groups_same_filesystem_and_reserves_once(monkeypatch, tmp_path: Path) -> None:
    paths = [tmp_path / "expanded", tmp_path / "archive"]
    for path in paths:
        path.mkdir()
    monkeypatch.setattr(downloader, "_filesystem_capacity", lambda path: (7, 150))

    downloader._strict_capacity_admit(
        [(paths[0], 60), (paths[1], 40)], reserve_bytes=50
    )


def test_capacity_groups_different_filesystems_separately(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(
        downloader,
        "_filesystem_capacity",
        lambda path: (1 if Path(path) == first else 2, 100),
    )

    downloader._strict_capacity_admit(
        [(first, 60), (second, 60)], reserve_bytes=40
    )


def test_capacity_uses_bavail_not_bfree(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        if Path(path) == target:
            return SimpleNamespace(st_dev=1)
        return real_stat(path, *args, **kwargs)

    with patch.object(downloader.os, "stat", fake_stat), patch.object(
        downloader.os,
        "statvfs",
        lambda path: SimpleNamespace(f_bavail=100, f_bfree=10_000, f_frsize=1),
    ):
        with pytest.raises(RuntimeError, match=r"required=101B available=100B"):
            downloader._strict_capacity_admit([(target, 101)], reserve_bytes=0)


def test_capacity_uses_lowest_repeated_probe_for_same_filesystem(
    monkeypatch, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    probes = iter(((1, 100), (1, 50)))
    monkeypatch.setattr(downloader, "_filesystem_capacity", lambda path: next(probes))

    with pytest.raises(RuntimeError, match=r"required=60B available=50B"):
        downloader._strict_capacity_admit(
            [(first, 20), (second, 40)], reserve_bytes=0
        )


def test_capacity_reserve_environment_requires_nonnegative_integer(monkeypatch) -> None:
    monkeypatch.delenv("QWEN3_ARTIFACT_CAPACITY_RESERVE_BYTES", raising=False)
    assert downloader._capacity_reserve_bytes() == 1 << 30
    for value in ("bad", "-1"):
        monkeypatch.setenv("QWEN3_ARTIFACT_CAPACITY_RESERVE_BYTES", value)
        with pytest.raises(RuntimeError, match="nonnegative integer"):
            downloader._capacity_reserve_bytes()


def _archive_manifest(
    payload: bytes, revision: str, archive_path: str = "v011/orin-nx-sm87/asr-b2-20261005/payload.tar"
) -> tuple[dict, str]:
    digest = hashlib.sha256(payload).hexdigest()
    file_bytes = b"tiny archive payload\n"
    file_digest = hashlib.sha256(file_bytes).hexdigest()
    manifest = {
        "model_id": "qwen3-asr-0.6b",
        "files": {"config.json": {"sha256": file_digest, "size": len(file_bytes)}},
        "payload": {
            "path": archive_path,
            "sha256": digest,
            "size": len(payload),
        },
        "_source": {
            "model_id": "qwen3-asr-0.6b",
            "repo": "org/asr",
            "revision": revision,
            "canonical_model_id": "qwen3-asr-0.6b",
        },
    }
    return manifest, file_digest


def _tiny_payload(*, gzip: bool = False) -> bytes:
    import io

    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz" if gzip else "w:") as archive:
        data = b"tiny archive payload\n"
        info = tarfile.TarInfo("config.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return out.getvalue()


def _legacy_archive_case(
    tmp_path: Path,
    revision: str,
    *,
    archive_path: str = "v011/orin-nx-sm87/asr-b2-20261005/payload.tar",
    payload: bytes | None = None,
):
    payload = _tiny_payload() if payload is None else payload
    manifest, _ = _archive_manifest(payload, revision, archive_path)
    repo = "org/asr"
    cache_root = tmp_path / "cache"
    cache = (
        cache_root
        / downloader._strict_cache_repo(repo)
        / "qwen3-asr-0.6b"
        / revision
    )
    cache.parent.mkdir(parents=True)
    archive = downloader._archive_spec(manifest, "qwen3-asr-0.6b", strict=True)
    assert archive is not None
    identity = "|".join(
        (str(cache), repo, revision, str(archive["path"]), str(archive["sha256"]).lower())
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()
    legacy = cache.parent / f".qwen-archive-{digest}.tar.gz"
    legacy.write_bytes(payload)
    return payload, manifest, repo, cache_root, cache, archive, legacy


def _install_archive_download_stubs(monkeypatch, manifest, calls):
    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(
        downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest
    )
    from server.core import hf_artifacts

    def download(_remote: str, target: Path, **kwargs: object) -> None:
        calls.append(target)
        if not target.exists():
            target.write_bytes(_tiny_payload())

    monkeypatch.setattr(hf_artifacts, "download_file", download)


def test_complete_legacy_archive_is_credited_only_on_same_filesystem(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "e" * 40
    payload, manifest, repo, cache_root, cache, archive, legacy = _legacy_archive_case(
        tmp_path, revision
    )
    archive_tmp_dir = tmp_path / "archive-tmp"
    archive_tmp_dir.mkdir(mode=0o700)
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(archive_tmp_dir))
    monkeypatch.setenv("QWEN3_ARTIFACT_CAPACITY_RESERVE_BYTES", "0")
    calls = []
    _install_archive_download_stubs(monkeypatch, manifest, calls)

    def capacity(path: Path):
        on_archive_fs = str(Path(path).resolve()).startswith(
            str(archive_tmp_dir.resolve()) + "/"
        ) or Path(path).resolve() == archive_tmp_dir.resolve()
        return (2, 0) if on_archive_fs else (1, 100_000)

    monkeypatch.setattr(downloader, "_filesystem_capacity", capacity)
    with pytest.raises(RuntimeError, match="required="):
        downloader.ensure_model_artifacts(
            "qwen3-asr-0.6b", repo, ("config.json",), revision=revision,
            cache_root=cache_root, strict=True,
        )
    target = downloader._strict_archive_path(cache, repo, revision, archive, archive_tmp_dir)
    assert not target.exists()
    assert legacy.read_bytes() == payload
    assert calls == []


def test_complete_legacy_archive_same_filesystem_avoids_duplicate_requirement(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "f" * 40
    payload, manifest, repo, cache_root, cache, archive, legacy = _legacy_archive_case(
        tmp_path, revision
    )
    archive_tmp_dir = tmp_path / "archive-tmp"
    archive_tmp_dir.mkdir(mode=0o700)
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(archive_tmp_dir))
    monkeypatch.setenv("QWEN3_ARTIFACT_CAPACITY_RESERVE_BYTES", "0")
    calls = []
    _install_archive_download_stubs(monkeypatch, manifest, calls)
    monkeypatch.setattr(downloader, "_filesystem_capacity", lambda path: (1, 21))

    assert downloader.ensure_model_artifacts(
        "qwen3-asr-0.6b", repo, ("config.json",), revision=revision,
        cache_root=cache_root, strict=True,
    )
    target = downloader._strict_archive_path(cache, repo, revision, archive, archive_tmp_dir)
    # The adopted archive is cleaned after extraction; the legacy resume copy
    # remains and proves that the same-filesystem credit path was used.
    assert not target.exists()
    assert legacy.read_bytes() == payload
    assert calls


def test_custom_tmp_gzip_legacy_archive_is_adopted_on_same_filesystem(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "c" * 40
    payload, manifest, repo, cache_root, cache, archive, legacy = _legacy_archive_case(
        tmp_path,
        revision,
        archive_path="v011/orin-nx-sm87/asr-b2-20261005/payload.tar.gz",
        payload=_tiny_payload(gzip=True),
    )
    archive_tmp_dir = tmp_path / "archive-tmp"
    archive_tmp_dir.mkdir(mode=0o700)
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(archive_tmp_dir))
    monkeypatch.setenv("QWEN3_ARTIFACT_CAPACITY_RESERVE_BYTES", "0")
    calls = []
    adopted_before_download = []
    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(
        downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest
    )
    from server.core import hf_artifacts

    def download(_remote: str, target: Path, **kwargs: object) -> None:
        calls.append(target)
        adopted_before_download.append(target.exists())

    monkeypatch.setattr(hf_artifacts, "download_file", download)
    monkeypatch.setattr(downloader, "_filesystem_capacity", lambda path: (1, 21))

    assert downloader.ensure_model_artifacts(
        "qwen3-asr-0.6b", repo, ("config.json",), revision=revision,
        cache_root=cache_root, strict=True,
    )
    assert adopted_before_download == [True]
    assert legacy.read_bytes() == payload


def test_custom_tmp_gzip_legacy_archive_cross_filesystem_needs_full_space(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "d" * 40
    payload, manifest, repo, cache_root, cache, archive, legacy = _legacy_archive_case(
        tmp_path,
        revision,
        archive_path="v011/orin-nx-sm87/asr-b2-20261005/payload.tar.gz",
        payload=_tiny_payload(gzip=True),
    )
    archive_tmp_dir = tmp_path / "archive-tmp"
    archive_tmp_dir.mkdir(mode=0o700)
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(archive_tmp_dir))
    monkeypatch.setenv("QWEN3_ARTIFACT_CAPACITY_RESERVE_BYTES", "0")
    calls = []
    _install_archive_download_stubs(monkeypatch, manifest, calls)

    def capacity(path: Path):
        on_archive_fs = Path(path).resolve() == archive_tmp_dir.resolve()
        return (2, 0) if on_archive_fs else (1, 100_000)

    monkeypatch.setattr(downloader, "_filesystem_capacity", capacity)
    with pytest.raises(RuntimeError, match="required="):
        downloader.ensure_model_artifacts(
            "qwen3-asr-0.6b", repo, ("config.json",), revision=revision,
            cache_root=cache_root, strict=True,
        )
    target = downloader._strict_archive_path(cache, repo, revision, archive, archive_tmp_dir)
    assert not target.exists()
    assert legacy.read_bytes() == payload
    assert calls == []


def test_invalid_target_does_not_credit_complete_legacy_archive(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "a" * 40
    payload, manifest, repo, cache_root, cache, archive, legacy = _legacy_archive_case(
        tmp_path, revision
    )
    archive_tmp_dir = tmp_path / "archive-tmp"
    archive_tmp_dir.mkdir(mode=0o700)
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(archive_tmp_dir))
    monkeypatch.setenv("QWEN3_ARTIFACT_CAPACITY_RESERVE_BYTES", "0")
    target = downloader._strict_archive_path(cache, repo, revision, archive, archive_tmp_dir)
    target.write_bytes(b"invalid target")
    calls = []
    _install_archive_download_stubs(monkeypatch, manifest, calls)
    monkeypatch.setattr(downloader, "_filesystem_capacity", lambda path: (1, 21))

    with pytest.raises(RuntimeError, match="required="):
        downloader.ensure_model_artifacts(
            "qwen3-asr-0.6b", repo, ("config.json",), revision=revision,
            cache_root=cache_root, strict=True,
        )
    assert target.read_bytes() == b"invalid target"
    assert legacy.read_bytes() == payload
    assert calls == []


def test_complete_declared_target_is_credited_without_archive_requirement(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "b" * 40
    payload, manifest, repo, cache_root, cache, archive, legacy = _legacy_archive_case(
        tmp_path, revision
    )
    legacy.unlink()
    archive_tmp_dir = tmp_path / "archive-tmp"
    archive_tmp_dir.mkdir(mode=0o700)
    monkeypatch.setenv("QWEN3_ARTIFACT_DOWNLOAD_TMP_DIR", str(archive_tmp_dir))
    monkeypatch.setenv("QWEN3_ARTIFACT_CAPACITY_RESERVE_BYTES", "0")
    target = downloader._strict_archive_path(cache, repo, revision, archive, archive_tmp_dir)
    target.write_bytes(payload)
    calls = []
    _install_archive_download_stubs(monkeypatch, manifest, calls)
    monkeypatch.setattr(downloader, "_filesystem_capacity", lambda path: (1, 21))

    assert downloader.ensure_model_artifacts(
        "qwen3-asr-0.6b", repo, ("config.json",), revision=revision,
        cache_root=cache_root, strict=True,
    )
    assert not target.exists()
    assert calls


def test_strict_capacity_rejects_before_download_or_staging(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _tiny_payload()
    revision = "a" * 40
    manifest, _ = _archive_manifest(payload, revision)
    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(
        downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest
    )
    calls = []
    from server.core import hf_artifacts

    monkeypatch.setattr(
        hf_artifacts,
        "download_file",
        lambda *args, **kwargs: calls.append(args),
    )
    monkeypatch.setattr(
        downloader, "_filesystem_capacity", lambda path: (1, 0)
    )

    cache_root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="strict artifact capacity insufficient"):
        downloader.ensure_model_artifacts(
            "qwen3-asr-0.6b",
            "org/asr",
            ("config.json",),
            revision=revision,
            cache_root=cache_root,
            strict=True,
        )
    assert not cache_root.exists()
    assert calls == []


def test_strict_tiny_archive_passes_capacity_and_extracts(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _tiny_payload()
    revision = "b" * 40
    monkeypatch.setenv("QWEN3_ARTIFACT_CAPACITY_RESERVE_BYTES", "0")
    manifest, _ = _archive_manifest(payload, revision)
    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(
        downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest
    )
    from server.core import hf_artifacts

    calls = []

    def download(_remote: str, target: Path, **kwargs: object) -> None:
        calls.append(target)
        target.write_bytes(payload)

    monkeypatch.setattr(hf_artifacts, "download_file", download)
    monkeypatch.setattr(
        downloader, "_filesystem_capacity", lambda path: (1, 20_000)
    )
    runtime = tmp_path / "runtime"
    assert downloader.ensure_model_artifacts(
        "qwen3-asr-0.6b",
        "org/asr",
        ("config.json",),
        revision=revision,
        cache_root=tmp_path / "cache",
        root=runtime,
        strict=True,
    )
    assert len(calls) == 1
    assert (runtime / "config.json").read_bytes() == b"tiny archive payload\n"


def test_strict_partial_archive_is_untouched_when_capacity_rejects(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _tiny_payload()
    revision = "c" * 40
    manifest, _ = _archive_manifest(payload, revision)
    repo = "org/asr"
    cache_root = tmp_path / "cache"
    cache = (
        cache_root
        / downloader._strict_cache_repo(repo)
        / "qwen3-asr-0.6b"
        / revision
    )
    cache.parent.mkdir(parents=True)
    cache.mkdir()
    active_marker = cache / "config.json"
    active_marker.write_bytes(b"active-cache")
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "qwen3-asr-0.6b",
                "files": {
                    "config.json": {
                        "sha256": hashlib.sha256(b"wrong").hexdigest(),
                        "size": 5,
                    }
                },
                "_source": {
                    "model_id": "qwen3-asr-0.6b",
                    "repo": repo,
                    "revision": revision,
                    "canonical_model_id": "qwen3-asr-0.6b",
                },
            }
        )
    )
    archive = downloader._archive_spec(manifest, "qwen3-asr-0.6b", strict=True)
    assert archive is not None
    partial = downloader._strict_archive_path(cache, repo, revision, archive, None)
    partial.write_bytes(payload[:7])
    before = partial.read_bytes()
    monkeypatch.setattr(downloader, "_is_enabled", lambda: True)
    monkeypatch.setattr(
        downloader, "_fetch_model_manifest", lambda *args, **kwargs: manifest
    )
    monkeypatch.setattr(
        downloader, "_filesystem_capacity", lambda path: (1, 0)
    )

    with pytest.raises(RuntimeError, match="strict artifact capacity insufficient"):
        downloader.ensure_model_artifacts(
            "qwen3-asr-0.6b",
            repo,
            ("config.json",),
            revision=revision,
            cache_root=cache_root,
            strict=True,
        )
    assert partial.read_bytes() == before
    assert active_marker.read_bytes() == b"active-cache"


def test_strict_cache_hit_skips_capacity_admission(monkeypatch, tmp_path: Path) -> None:
    revision = "d" * 40
    repo = "org/asr"
    cache = (
        tmp_path
        / downloader._strict_cache_repo(repo)
        / "qwen3-asr-0.6b"
        / revision
    )
    cache.mkdir(parents=True)
    data = b"cached\n"
    (cache / "config.json").write_bytes(data)
    manifest = {
        "model_id": "qwen3-asr-0.6b",
        "files": {
            "config.json": {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        },
        "_source": {
            "model_id": "qwen3-asr-0.6b",
            "repo": repo,
            "revision": revision,
            "canonical_model_id": "qwen3-asr-0.6b",
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        downloader,
        "_filesystem_capacity",
        lambda path: (_ for _ in ()).throw(AssertionError("cache hit admitted")),
    )

    assert downloader.ensure_model_artifacts(
        "qwen3-asr-0.6b",
        repo,
        ("config.json",),
        revision=revision,
        cache_root=tmp_path,
        strict=True,
    )
