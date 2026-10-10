"""Profile-scoped download contract for external v0.9.1 engines."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import tarfile
from pathlib import Path

import pytest

from server.core import qwen3_artifact_downloader as qad


def _manifest(path: Path, *, digest: str) -> None:
    path.write_text(
        json.dumps(
            {
                "hf_repo_id": "example/engines",
                "revision": "locked",
                "artifact_sets": {
                    "orin-r6": {
                        "root": "/legacy/root",
                        "hf_prefix": "sets/orin-r6/v091",
                        "required_files": [
                            "engines/asr/llm.engine",
                            "engines/tts/llm.engine",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_download_uses_hf_prefix_profile_scope_and_stable_symlink(
    tmp_path, monkeypatch
):
    from server.core import qwen3_artifact_downloader as downloader
    import huggingface_hub

    artifact_root = tmp_path / "models" / "edgellm-v091"
    repo_root = tmp_path / "models"
    manifest = tmp_path / "manifest.json"
    engine = b"qualified-engine"
    digest = hashlib.sha256(engine).hexdigest()
    _manifest(manifest, digest=digest)

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        root = Path(kwargs["local_dir"]) / "sets/orin-r6/v091"
        engine_dir = root / "engines/asr"
        engine_dir.mkdir(parents=True)
        (engine_dir / "llm.engine").write_bytes(engine)
        (engine_dir / "config.json").write_text("{}", encoding="utf-8")
        config_digest = hashlib.sha256(b"{}").hexdigest()
        (root / "SHA256SUMS").write_text(
            f"{digest}  engines/asr/llm.engine\n"
            f"{config_digest}  engines/asr/config.json\n",
            encoding="utf-8",
        )
        return str(root)

    monkeypatch.setattr(downloader, "_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setenv("OVS_PROFILE", "jetson-edgellm-v091-matcha")
    monkeypatch.setenv("QWEN3_ARTIFACT_SET", "orin-r6")
    monkeypatch.setenv("QWEN3_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("QWEN3_REPO_CACHE_ROOT", str(repo_root))

    assert downloader.ensure_artifacts([str(artifact_root / "engines/asr/llm.engine")])
    assert artifact_root.is_symlink()
    assert (artifact_root / "engines/asr/config.json").is_file()
    assert calls[0]["allow_patterns"] == [
        "sets/orin-r6/v091/PROVENANCE.md",
        "sets/orin-r6/v091/SHA256SUMS",
        "sets/orin-r6/v091/engines/asr/**",
        "sets/orin-r6/v091/manifest.json",
    ]
    assert all("engines/tts" not in item for item in calls[0]["allow_patterns"])


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    from server.core import qwen3_artifact_downloader as downloader
    import huggingface_hub

    artifact_root = tmp_path / "models" / "edgellm-v091"
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, digest="0" * 64)

    def fake_snapshot_download(**kwargs):
        root = Path(kwargs["local_dir"]) / "sets/orin-r6/v091"
        target = root / "engines/asr/llm.engine"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"corrupt")
        (root / "SHA256SUMS").write_text(
            f"{'0' * 64}  engines/asr/llm.engine\n", encoding="utf-8"
        )

    monkeypatch.setattr(downloader, "_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setenv("OVS_PROFILE", "jetson-edgellm-v091-matcha")
    monkeypatch.setenv("QWEN3_ARTIFACT_SET", "orin-r6")
    monkeypatch.setenv("QWEN3_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("QWEN3_REPO_CACHE_ROOT", str(tmp_path / "models"))

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        downloader.ensure_artifacts(
            [str(artifact_root / "engines/asr/llm.engine")]
        )


def test_download_does_not_hide_nonempty_operator_artifact_root(
    tmp_path, monkeypatch
):
    from server.core import qwen3_artifact_downloader as downloader
    import huggingface_hub

    artifact_root = tmp_path / "models" / "edgellm-v091"
    artifact_root.mkdir(parents=True)
    (artifact_root / "operator-file").write_text("keep", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, digest="0" * 64)

    def fake_snapshot_download(**kwargs):
        root = Path(kwargs["local_dir"]) / "sets/orin-r6/v091"
        target = root / "engines/asr/llm.engine"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"engine")
        digest = hashlib.sha256(b"engine").hexdigest()
        (root / "SHA256SUMS").write_text(
            f"{digest}  engines/asr/llm.engine\n", encoding="utf-8"
        )

    monkeypatch.setattr(downloader, "_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setenv("OVS_PROFILE", "jetson-edgellm-v091-matcha")
    monkeypatch.setenv("QWEN3_ARTIFACT_SET", "orin-r6")
    monkeypatch.setenv("QWEN3_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("QWEN3_REPO_CACHE_ROOT", str(tmp_path / "models"))

    with pytest.raises(RuntimeError, match="exists and is not the HF set root"):
        downloader.ensure_artifacts(
            [str(artifact_root / "engines/asr/llm.engine")]
        )
    assert (artifact_root / "operator-file").read_text() == "keep"


def _mode_tar(tmp_path: Path, worker_mode: int = 0o755) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "payload.tar"
    with tarfile.open(archive, "w:") as tf:
        for name, data, mode in (
            ("worker/qwen3_tts_worker", b"worker", worker_mode),
            ("ordinary.txt", b"ordinary", 0o644),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            tf.addfile(info, io.BytesIO(data))
    return archive


def test_safe_extract_enforces_declared_native_and_ordinary_modes(tmp_path):
    archive = _mode_tar(tmp_path)
    dest = tmp_path / "dest"
    qad._safe_extract(
        archive,
        dest,
        expected_modes={"worker/qwen3_tts_worker": 0o755, "ordinary.txt": 0o644},
    )
    assert stat.S_IMODE((dest / "worker/qwen3_tts_worker").stat().st_mode) == 0o755
    assert stat.S_IMODE((dest / "ordinary.txt").stat().st_mode) == 0o644

    with pytest.raises(RuntimeError, match="payload mode mismatch"):
        qad._safe_extract(
            _mode_tar(tmp_path / "wrong"),
            tmp_path / "wrong-dest",
            expected_modes={"worker/qwen3_tts_worker": 0o644},
        )


def test_safe_extract_rejects_special_mode_bits(tmp_path):
    with pytest.raises(RuntimeError, match="unsafe model payload mode"):
        qad._safe_extract(_mode_tar(tmp_path / "special" , worker_mode=0o4755), tmp_path / "dest")


@pytest.mark.parametrize("value", [0o755, True, "755", "0777", "0750", None])
def test_manifest_mode_accepts_only_declared_string_modes(value):
    manifest = {"files": {"worker": {"sha256": "a" * 64, "size": 1, "mode": value}}}
    with pytest.raises(RuntimeError, match="manifest mode is invalid"):
        qad._manifest_modes(manifest, "model")


def test_old_manifest_without_mode_remains_compatible(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    worker = cache / "worker"
    worker.write_bytes(b"worker")
    worker.chmod(0o644)
    digest = hashlib.sha256(b"worker").hexdigest()
    manifest = {"model_id": "model", "files": {"worker": {"sha256": digest, "size": 6}}}
    qad._verify_model_files(cache, "model", ["worker"], manifest, strict=True)


def test_declared_modes_apply_only_to_new_staging_tree(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    worker = staging / "worker"
    worker.write_bytes(b"worker")
    worker.chmod(0o644)
    qad._apply_declared_modes(staging, {"worker": 0o755})
    assert stat.S_IMODE(worker.stat().st_mode) == 0o755


def test_declared_mode_mismatch_fails_before_runtime_activation(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    worker = cache / "worker"
    worker.write_bytes(b"worker")
    worker.chmod(0o644)
    digest = hashlib.sha256(b"worker").hexdigest()
    manifest = {"model_id": "model", "files": {"worker": {"sha256": digest, "size": 6, "mode": "0755"}}}
    active = tmp_path / "active-target"
    active.mkdir()
    runtime = tmp_path / "runtime"
    runtime.symlink_to(active, target_is_directory=True)
    with pytest.raises(RuntimeError, match="artifact mode mismatch"):
        qad._verify_model_files(cache, "model", ["worker"], manifest, strict=True)
    assert runtime.is_symlink() and runtime.resolve() == active.resolve()


def _strict_mode_download_fixture(monkeypatch, *, content: bytes = b"worker"):
    revision = "a" * 40
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "model_id": "tts-worker",
        "files": {
            "worker": {"sha256": digest, "size": len(content), "mode": "0755"},
        },
    }
    calls = {"manifest": 0, "download": 0}

    def fetch_manifest(*args, **kwargs):
        calls["manifest"] += 1
        return manifest

    def download_file(rel_path, dest, **kwargs):
        calls["download"] += 1
        assert rel_path == "worker"
        assert kwargs["expected_sha256"] == digest
        assert kwargs["expected_size"] == len(content)
        Path(dest).write_bytes(content)
        return Path(dest)

    from server.core import hf_artifacts

    monkeypatch.setattr(qad, "_fetch_model_manifest", fetch_manifest)
    monkeypatch.setattr(hf_artifacts, "download_file", download_file)
    monkeypatch.setenv("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1")
    return revision, manifest, calls


def test_strict_ensure_fresh_download_applies_declared_worker_mode(tmp_path, monkeypatch):
    revision, _manifest_data, calls = _strict_mode_download_fixture(monkeypatch)
    cache_root = tmp_path / "cache"
    runtime = tmp_path / "runtime"

    assert qad.ensure_model_artifacts(
        "tts-worker", "example/artifacts", ("worker",), revision=revision,
        root=runtime, cache_root=cache_root, strict=True,
    )

    # Resolve the repository directory through the same production helper so
    # this assertion remains tied to the cache contract.
    cache = cache_root / qad._strict_cache_repo("example/artifacts") / qad._strict_cache_model("tts-worker") / revision
    assert stat.S_IMODE((cache / "worker").stat().st_mode) == 0o755
    assert runtime.is_symlink() and runtime.resolve() == cache.resolve()
    assert calls == {"manifest": 1, "download": 1}


def test_strict_ensure_cache_hit_validates_mode_without_network(tmp_path, monkeypatch):
    revision, _manifest_data, calls = _strict_mode_download_fixture(monkeypatch)
    cache_root = tmp_path / "cache"
    runtime = tmp_path / "runtime"
    kwargs = dict(
        model_id="tts-worker", repo="example/artifacts", required_files=("worker",),
        revision=revision, root=runtime, cache_root=cache_root, strict=True,
    )
    assert qad.ensure_model_artifacts(**kwargs)
    calls_before = calls.copy()

    def fail_network(*args, **kwargs):
        raise AssertionError("cache hit must not access the network")

    monkeypatch.setattr(qad, "_fetch_model_manifest", fail_network)
    from server.core import hf_artifacts
    monkeypatch.setattr(hf_artifacts, "download_file", fail_network)
    assert qad.ensure_model_artifacts(**kwargs)
    assert calls == calls_before


def test_strict_ensure_wrong_mode_fetch_failure_keeps_active_symlink(tmp_path, monkeypatch):
    revision, _manifest_data, _calls = _strict_mode_download_fixture(monkeypatch)
    cache_root = tmp_path / "cache"
    runtime = tmp_path / "runtime"
    kwargs = dict(
        model_id="tts-worker", repo="example/artifacts", required_files=("worker",),
        revision=revision, root=runtime, cache_root=cache_root, strict=True,
    )
    assert qad.ensure_model_artifacts(**kwargs)
    active_before = runtime.resolve()
    worker = active_before / "worker"
    worker.chmod(0o644)

    def fail_network(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(qad, "_fetch_model_manifest", fail_network)
    with pytest.raises(RuntimeError, match="network unavailable"):
        qad.ensure_model_artifacts(**kwargs)
    assert runtime.is_symlink() and runtime.resolve() == active_before
    assert worker.read_bytes() == b"worker"
    assert stat.S_IMODE(worker.stat().st_mode) == 0o644
