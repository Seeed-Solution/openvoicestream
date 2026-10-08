from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_edgellm_v010_artifact.py"
# Some uv/pytest combinations choose ``tests/`` as the import root.  Keep
# this test hermetic when importing the downloader's schema helpers.
sys.path.insert(0, str(ROOT))
from server.core.qwen3_artifact_downloader import (
    _archive_spec,
    _manifest_files,
    _safe_extract,
)


def _run(payload: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(payload),
            str(output),
            "--artifact",
            "qwen3-tts-base-v010",
            "--model-id",
            "qwen3-tts-0.6b-base",
            "--repo",
            "harvestsu/qwen3-tts-0.6b-base-jetson-artifacts",
            "--source",
            "upstream=71dd1bae032e70771265917ec74d3ff4cad07a10",
            "--profile",
            "sm87-trt10.3-jp6.2-cuda12.6",
            "--provenance",
            '{"builder":"test","plugin":"v1"}',
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def _payload(root: Path) -> None:
    (root / "engines" / "nested").mkdir(parents=True)
    (root / "engines" / "talker.engine").write_bytes(b"talker\x00\x01")
    (root / "engines" / "nested" / "code_predictor.engine").write_bytes(
        b"predictor"
    )
    (root / "config.json").write_text('{"precision":"int4"}\n', encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_package_writes_schema_v2_manifest_and_verified_payload(tmp_path: Path) -> None:
    source = tmp_path / "payload"
    source.mkdir()
    _payload(source)
    output = tmp_path / "artifact"

    result = _run(source, output)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["artifact_kind"] == "qwen3-tts-base-v010"
    assert manifest["model_id"] == "qwen3-tts-0.6b-base"
    assert manifest["hf_repo_id"] == "harvestsu/qwen3-tts-0.6b-base-jetson-artifacts"
    assert manifest["source_revision"] == "upstream=71dd1bae032e70771265917ec74d3ff4cad07a10"
    assert manifest["engine_profile"] == "sm87-trt10.3-jp6.2-cuda12.6"
    assert manifest["provenance"] == {"builder": "test", "plugin": "v1"}
    assert not {"artifact", "repo", "source", "profile"} & manifest.keys()

    expected_files = {
        "config.json": source / "config.json",
        "engines/nested/code_predictor.engine": source / "engines/nested/code_predictor.engine",
        "engines/talker.engine": source / "engines/talker.engine",
    }
    assert list(manifest["files"]) == sorted(expected_files)
    for relative, path in expected_files.items():
        assert manifest["files"][relative] == {
            "sha256": _sha(path),
            "size": path.stat().st_size,
        }

    payload = output / "payload.tar"
    assert manifest["payload"] == {
        "path": "payload.tar",
        "sha256": _sha(payload),
        "size": payload.stat().st_size,
    }
    assert (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines() == [
        f"{_sha(output / 'manifest.json')}  manifest.json",
        f"{_sha(payload)}  payload.tar",
    ]
    checksum = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=output,
        capture_output=True,
        text=True,
    )
    assert checksum.returncode == 0, checksum.stderr

    # The generated shape is the same schema-v2 contract consumed by the
    # downloader: file locks are a mapping and payload is an archive lock.
    assert _manifest_files(manifest, manifest["model_id"]) == {
        relative: (metadata["sha256"], metadata["size"])
        for relative, metadata in manifest["files"].items()
    }
    assert _archive_spec(manifest, manifest["model_id"]) == manifest["payload"]

    with tarfile.open(payload, mode="r:") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == [
            "config.json",
            "engines",
            "engines/nested",
            "engines/nested/code_predictor.engine",
            "engines/talker.engine",
        ]
        for member in members:
            assert member.uid == 0
            assert member.gid == 0
            assert member.mtime == 0
            assert member.uname == ""
            assert member.gname == ""
            assert member.mode == (0o755 if member.isdir() else 0o644)
        assert archive.extractfile("config.json").read() == (source / "config.json").read_bytes()


def test_packaging_is_byte_deterministic_and_named_output_is_not_overwritten(
    tmp_path: Path,
) -> None:
    source = tmp_path / "payload"
    source.mkdir()
    _payload(source)
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert _run(source, first).returncode == 0
    assert _run(source, second).returncode == 0
    for name in ("payload.tar", "manifest.json", "SHA256SUMS"):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    sentinel = first / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    result = _run(source, first)
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_executable_mode_survives_cli_package_and_safe_extract(tmp_path: Path) -> None:
    source = tmp_path / "payload"
    source.mkdir()
    worker = source / "worker"
    worker.write_text("#!/bin/sh\nprintf worker-ok\n", encoding="utf-8")
    worker.chmod(0o4755 | stat.S_ISUID | stat.S_ISGID | stat.S_IWOTH)
    data = source / "data"
    data.write_bytes(b"data")
    data.chmod(0o6644 | stat.S_ISUID | stat.S_ISGID)
    output = tmp_path / "artifact"

    result = _run(source, output)

    assert result.returncode == 0, result.stderr
    with tarfile.open(output / "payload.tar", mode="r:") as archive:
        assert archive.getmember("worker").mode == 0o755
        assert archive.getmember("data").mode == 0o644
    extracted = tmp_path / "extracted"
    _safe_extract(output / "payload.tar", extracted)
    assert stat.S_IMODE(worker.stat().st_mode) & 0o111
    assert stat.S_IMODE((extracted / "worker").stat().st_mode) == 0o755
    assert stat.S_IMODE((extracted / "data").stat().st_mode) == 0o644
    assert subprocess.run([str(extracted / "worker")], capture_output=True, text=True).stdout == "worker-ok"


def test_packaging_rejects_executable_mode_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("packager", SCRIPT)
    assert spec and spec.loader
    packager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packager)
    source = tmp_path / "payload"
    source.mkdir()
    worker = source / "worker"
    worker.write_bytes(b"worker")
    worker.chmod(0o755)
    calls = 0
    original = packager._lstat

    def racing(path: Path):
        nonlocal calls
        result = original(path)
        if path == worker:
            calls += 1
            if calls == 2:
                worker.chmod(0o644)
        return result

    monkeypatch.setattr(packager, "_lstat", racing)
    with pytest.raises(packager.PackageError, match="changed while packaging"):
        packager.write_deterministic_payload_tar(
            source, [("worker", worker)], tmp_path / "payload.tar"
        )


@pytest.mark.parametrize("kind", ["empty", "symlink", "fifo", "directory_symlink"])
def test_packaging_rejects_unsafe_or_empty_payloads(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "payload"
    source.mkdir()
    if kind == "empty":
        pass
    elif kind == "symlink":
        (source / "link").symlink_to("missing.bin")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO is unavailable on this platform")
        os.mkfifo(source / "pipe")
    else:
        (source / "real").mkdir()
        (source / "link").symlink_to("real", target_is_directory=True)
    output = tmp_path / "artifact"

    result = _run(source, output)

    assert result.returncode != 0
    assert not output.exists()
    if kind == "empty":
        assert "empty" in result.stderr
    elif kind in {"symlink", "directory_symlink"}:
        assert "symbolic link" in result.stderr
    else:
        assert "non-regular" in result.stderr


def test_named_directory_options_and_provenance_file_are_supported(tmp_path: Path) -> None:
    source = tmp_path / "payload"
    source.mkdir()
    (source / "engine.plan").write_bytes(b"plan")
    output = tmp_path / "artifact"
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"source_sha":"abc","builder_sha":"def"}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--payload-dir",
            str(source),
            "--output-dir",
            str(output),
            "--artifact-id",
            "test-artifact",
            "--model",
            "test-model",
            "--repo",
            "org/test-model",
            "--source-revision",
            "source-sha",
            "--engine-profile",
            "test-profile",
            "--provenance-file",
            str(provenance),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "manifest.json").read_text())["provenance"] == {
        "source_sha": "abc",
        "builder_sha": "def",
    }


def test_profile_aware_runtime_contract_is_emitted(tmp_path: Path) -> None:
    source = tmp_path / "payload"
    source.mkdir()
    (source / "spec_base.engine").write_bytes(b"base")
    (source / "spec_draft.engine").write_bytes(b"draft")
    output = tmp_path / "artifact"

    result = _run(
        source,
        output,
        "--profile",
        "8k",
        "--engine-contract",
        '{"max_input_len":8192,"max_kv_cache_capacity":8192}',
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["engine_profile"] == "8k"
    assert manifest["engine_contract"] == {
        "max_input_len": 8192,
        "max_kv_cache_capacity": 8192,
    }


def test_repo_prefix_is_manifest_only_and_keeps_payload_members_flat(tmp_path: Path) -> None:
    source = tmp_path / "payload"
    source.mkdir()
    (source / "engine.plan").write_bytes(b"plan")
    output = tmp_path / "artifact"

    result = _run(source, output, "--repo-prefix", "v011/orin-nx-sm87/asr-b2-20261005")

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["payload"]["path"] == "v011/orin-nx-sm87/asr-b2-20261005/payload.tar"
    with tarfile.open(output / "payload.tar", mode="r:") as archive:
        assert [member.name for member in archive.getmembers()] == ["engine.plan"]


@pytest.mark.parametrize("prefix", ["/absolute", "../escape", "a/../b", "a//b", "a/./b", "a/", r"a\\..\\b"])
def test_repo_prefix_rejects_unsafe_raw_paths(tmp_path: Path, prefix: str) -> None:
    source = tmp_path / "payload"
    source.mkdir()
    (source / "engine.plan").write_bytes(b"plan")
    output = tmp_path / "artifact"

    result = _run(source, output, "--repo-prefix", prefix)

    assert result.returncode != 0
    assert "safe relative path" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("contract", ['"not-an-object"', "[]", "true"])
def test_engine_contract_must_be_json_object(
    tmp_path: Path, contract: str
) -> None:
    source = tmp_path / "payload"
    source.mkdir()
    (source / "engine.plan").write_bytes(b"plan")

    result = _run(
        source,
        tmp_path / "artifact",
        "--engine-contract",
        contract,
    )

    assert result.returncode != 0
    assert "engine-contract" in result.stderr


def test_non_regular_payload_root_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "not-a-directory"
    source.write_bytes(b"payload")
    result = _run(source, tmp_path / "artifact")
    assert result.returncode != 0
    assert "not a directory" in result.stderr


def test_existing_dangling_output_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "payload"
    source.mkdir()
    (source / "engine.plan").write_bytes(b"plan")
    output = tmp_path / "artifact"
    output.symlink_to("does-not-exist")

    result = _run(source, output)

    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert stat.S_ISLNK(output.lstat().st_mode)
