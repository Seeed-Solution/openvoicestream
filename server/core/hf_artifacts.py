"""HuggingFace artifact downloader for OpenVoiceStream.

Downloads ONNX models and pre-built TRT engine bundles from HuggingFace,
with optional China mirror support via HF_ENDPOINT env. Designed to be
called from engine_resolver.

Layout convention on the artifact repo:
    <HF_REPO>/
        models/<model_id>/
            manifest.json              # files + SHA-256 + sizes
            <model-relative ONNX>      # raw / graph-surgery inputs for fallback rebuilds
            engines/<host_sig>.tar.gz  # pre-built engines for a specific host

Where host_sig is "sm<NN>-trt<X.Y>-jp<X.Y>-cuda<X.Y>" — see engine_resolver.

manifest.json schema (top level):
    {
      "model_id": "matcha-icefall-zh-en",
      "files": {
        "onnx/matcha_encoder_s64_trt.onnx": {"sha256": "...", "size": 12345},
        "model-steps-3.onnx": {"sha256": "...", "size": 12345},
        "engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz": {"sha256": "...", "size": 67890}
      }
    }
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Optional
import re
import time
import fcntl
import uuid
import stat

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_REPO = "harvestsu/seeed-local-voice-artifacts"
DEFAULT_REVISION = "main"

# hf-mirror.com rejects Python-urllib/x.y default User-Agent with 403.
# Use a hf_hub-style UA that mirrors what huggingface_hub sends.
_UA = "openvoicestream/1.0; hf_hub-emulating"


class _AllowedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: frozenset[str], initial_scheme: str):
        self.allowed_hosts = allowed_hosts
        self.initial_scheme = initial_scheme

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if (parsed.scheme not in {"http", "https"} or
                (self.initial_scheme == "https" and parsed.scheme != "https") or
                not parsed.hostname or parsed.hostname.casefold() not in self.allowed_hosts):
            raise ArtifactError(f"redirect host is not allowlisted: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url: str, timeout: float = 30.0, *, allowed_redirect_hosts: frozenset[str] | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    if allowed_redirect_hosts is None:
        return urllib.request.urlopen(req, timeout=timeout)
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.hostname.casefold() not in allowed_redirect_hosts:
        raise ArtifactError(f"endpoint host is not allowlisted: {url}")
    opener = urllib.request.build_opener(_AllowedRedirect(allowed_redirect_hosts, parsed.scheme))
    return opener.open(req, timeout=timeout)


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be fetched, verified, or extracted."""


def _endpoint(endpoint: Optional[str] = None) -> str:
    return str(endpoint or os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT)).rstrip("/")


def _repo() -> str:
    return os.environ.get("HF_ARTIFACT_REPO", DEFAULT_REPO).strip("/")


def _revision(revision: Optional[str] = None) -> str:
    return str(revision or os.environ.get("HF_ARTIFACT_REVISION") or DEFAULT_REVISION).strip("/")


def canonical_model_id(model_id: str) -> str:
    value = str(model_id or "").strip().replace("/", "--")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip(".-") or "model"


def model_cache_dir(cache_root: str | Path, model_id: str) -> Path:
    return Path(cache_root) / canonical_model_id(model_id)


def file_url(rel_path: str, *, repo: Optional[str] = None, revision: Optional[str] = None, endpoint: Optional[str] = None) -> str:
    """Build the HF resolve URL for a file inside the artifact repo."""
    return f"{_endpoint(endpoint)}/{str(repo or _repo()).strip('/')}/resolve/{_revision(revision)}/{rel_path.lstrip('/')}"


def _sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fetch_manifest(
    model_id: str, *, repo: Optional[str] = None, revision: Optional[str] = None,
    manifest_path: Optional[str] = None, endpoint: Optional[str] = None,
    allowed_redirect_hosts: frozenset[str] | None = None,
) -> dict:
    """Download and parse a model's manifest.json. Raises ArtifactError on failure."""
    rel = manifest_path or f"models/{model_id}/manifest.json"
    url = file_url(rel, repo=repo, revision=revision, endpoint=endpoint)
    try:
        with _open(url, timeout=30, allowed_redirect_hosts=allowed_redirect_hosts) as resp:
            data = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ArtifactError(f"manifest not found at {url}") from exc
        raise ArtifactError(f"HTTP {exc.code} fetching {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ArtifactError(f"network error fetching {url}: {exc}") from exc
    try:
        manifest = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid manifest JSON at {url}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactError(f"manifest JSON at {url} must be an object")
    manifest.setdefault("_source", {
        "model_id": str(model_id), "repo": str(repo or _repo()).strip("/"),
        "revision": _revision(revision), "path": rel,
    })
    return manifest


def download_file(
    rel_path: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    allowed_redirect_hosts: frozenset[str] | None = None,
    resume: bool = False,
    resume_key: Optional[str] = None,
    deadline: Optional[float] = None,
    max_retries: int = 2,
) -> Path:
    """Stream a file from HF into ``dest`` via a ``.tmp`` sibling then atomic rename.

    If ``expected_sha256`` is given, verifies after download and aborts on mismatch.
    Returns the final dest path.
    """
    if resume:
        return _download_file_resumable(
            rel_path, Path(dest), expected_sha256=expected_sha256,
            expected_size=expected_size, repo=repo, revision=revision,
            endpoint=endpoint, allowed_redirect_hosts=allowed_redirect_hosts,
            resume_key=resume_key, deadline=deadline, max_retries=max_retries,
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    url = file_url(rel_path, repo=repo, revision=revision, endpoint=endpoint)

    logger.info("downloading %s → %s", url, dest)
    try:
        with _open(url, timeout=60, allowed_redirect_hosts=allowed_redirect_hosts) as resp:
            final_host = urllib.parse.urlsplit(resp.geturl()).hostname
            if (allowed_redirect_hosts is not None and expected_size is not None and
                    expected_size > 10 * 1024 * 1024 and (final_host or "").casefold() == "huggingface.co"):
                tmp.unlink(missing_ok=True)
                raise ArtifactError("strict large artifact response remained on huggingface.co")
            with tmp.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        if exc.code == 404:
            raise ArtifactError(f"file not found: {url}") from exc
        raise ArtifactError(f"HTTP {exc.code} on {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise ArtifactError(f"network error on {url}: {exc}") from exc

    if expected_size is not None:
        got_size = tmp.stat().st_size
        if got_size != expected_size:
            tmp.unlink(missing_ok=True)
            raise ArtifactError(
                f"size mismatch for {rel_path}: expected {expected_size}, got {got_size}"
            )
    if expected_sha256:
        got = _sha256_file(tmp)
        if got != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ArtifactError(
                f"sha256 mismatch for {rel_path}: expected {expected_sha256}, got {got}"
            )
    os.replace(tmp, dest)
    return dest


def _resume_paths(dest: Path, identity: str) -> tuple[Path, Path, Path]:
    digest = hashlib.sha256(identity.encode()).hexdigest()
    part = dest.parent / f".{dest.name}.{digest}.part"
    state = part.with_suffix(part.suffix + ".state")
    lock = part.with_suffix(part.suffix + ".lock")
    return part, state, lock


def _atomic_json(path: Path, value: dict) -> None:
    # Use a unique, no-follow, exclusive temporary file.  A fixed ``.new``
    # name can be a stale/foreign symlink after a crash.
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.new")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(tmp), flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            fd = -1
            json.dump(value, out, sort_keys=True)
            out.flush(); os.fsync(out.fileno())
        os.replace(tmp, path)
    except Exception:
        if fd >= 0:
            os.close(fd)
        tmp.unlink(missing_ok=True)
        raise
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _same_path_fd(path: Path, fd: int, label: str) -> None:
    """Reject a path replacement or symlink after an owned fd was opened."""
    try:
        path_stat = os.lstat(path)
        fd_stat = os.fstat(fd)
    except OSError as exc:
        raise ArtifactError(f"strict resume {label} path changed") from exc
    if (not stat.S_ISREG(fd_stat.st_mode) or
            path_stat.st_dev != fd_stat.st_dev or path_stat.st_ino != fd_stat.st_ino):
        raise ArtifactError(f"strict resume {label} path changed")


def _open_pinned(path: Path, *, create: bool, label: str) -> int:
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(path), flags, 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ArtifactError(f"strict resume {label} is not a regular file")
        _same_path_fd(path, fd, label)
        return fd
    except FileExistsError:
        raise ArtifactError(f"strict resume {label} appeared during lock")
    except OSError as exc:
        raise ArtifactError(f"strict resume {label} is not a regular file") from exc


def _fd_sha256(fd: int, *, deadline: float | None = None) -> str:
    h = hashlib.sha256(); offset = 0
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise ArtifactError("strict resume deadline exceeded during hash")
        chunk = os.pread(fd, 1 << 20, offset)
        if not chunk: return h.hexdigest()
        h.update(chunk); offset += len(chunk)


def _fd_json(fd: int, label: str) -> dict:
    try:
        raw = os.pread(fd, 16 << 20, 0)
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"strict resume {label} is invalid") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"strict resume {label} is invalid")
    return value


def _download_file_resumable(
    rel_path: str, dest: Path, *, expected_sha256: Optional[str],
    expected_size: Optional[int], repo: Optional[str], revision: Optional[str],
    endpoint: Optional[str], allowed_redirect_hosts: frozenset[str] | None,
    resume_key: Optional[str], deadline: Optional[float], max_retries: int,
) -> Path:
    if expected_size is None or expected_sha256 is None:
        raise ArtifactError("strict resume requires expected size and SHA256")
    if expected_size <= 0 or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ArtifactError("strict resume lock is invalid")
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or (dest.exists() and not dest.is_file()):
        raise ArtifactError("strict resume destination is not a regular file")
    if dest.is_file():
        if dest.stat().st_size == expected_size and _sha256_file(dest) == expected_sha256.lower():
            return dest
        raise ArtifactError("strict resume destination identity/hash mismatch")
    identity = resume_key or "|".join((str(dest.parent.resolve()), str(repo or ""), str(revision or ""), rel_path, expected_sha256.lower()))
    part, state, lock = _resume_paths(dest, identity)
    for p in (part, state, lock):
        if p.is_symlink() or (p.exists() and not p.is_file()):
            raise ArtifactError("strict resume path is not a regular file")
    try:
        lockfd = _open_pinned(lock, create=not lock.exists(), label="lock")
    except ArtifactError:
        raise
    lockstream = os.fdopen(lockfd, "a+b", closefd=True)
    try:
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ArtifactError("strict resume deadline exceeded waiting for lock")
            else:
                remaining = None
            try:
                fcntl.flock(lockstream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                delay = min(0.05, remaining) if remaining is not None else 0.05
                if delay <= 0:
                    raise ArtifactError("strict resume deadline exceeded waiting for lock")
                time.sleep(delay)
        partfd = statefd = None
        try:
            _same_path_fd(lock, lockstream.fileno(), "lock")
            try:
                partfd = _open_pinned(part, create=False, label="partial")
            except ArtifactError:
                if part.exists(): raise
            try:
                statefd = _open_pinned(state, create=False, label="state")
            except ArtifactError:
                if state.exists(): raise
            block_size = 1 << 20
            if statefd is not None:
                metadata = _fd_json(statefd, "state")
                if metadata.get("identity") != identity or metadata.get("expected_size") != expected_size or metadata.get("expected_sha256") != expected_sha256.lower():
                    raise ArtifactError("strict resume identity mismatch")
                # Preserve the precise legacy corruption diagnostic while
                # rejecting the pre-v2 state format. Valid crash-gap
                # recovery is available only for v2 states with block
                # entries below.
                legacy_committed = metadata.get("committed")
                if (metadata.get("schema") != "strict-resume-v2" and
                        isinstance(legacy_committed, int) and not isinstance(legacy_committed, bool) and
                        partfd is not None and os.fstat(partfd).st_size != legacy_committed):
                    raise ArtifactError("strict resume partial size mismatch")
                if metadata.get("schema") != "strict-resume-v2" or metadata.get("status") != "active" or metadata.get("block_size") != block_size or not isinstance(metadata.get("blocks"), list):
                    raise ArtifactError("strict resume state schema/status is invalid")
                committed = metadata.get("committed")
                if isinstance(committed, bool) or not isinstance(committed, int):
                    raise ArtifactError("strict resume state committed is invalid")
                expected_offset = 0
                for entry in metadata["blocks"]:
                    if not isinstance(entry, dict) or set(entry) != {"offset", "size", "sha256"}:
                        raise ArtifactError("strict resume state block entry is invalid")
                    offset, size, digest = entry["offset"], entry["size"], entry["sha256"]
                    if (isinstance(offset, bool) or not isinstance(offset, int) or
                            isinstance(size, bool) or not isinstance(size, int) or size <= 0 or
                            offset != expected_offset or not re.fullmatch(r"[0-9a-f]{64}", str(digest))):
                        raise ArtifactError("strict resume state block entry is invalid")
                    expected_offset += size
                if committed != expected_offset:
                    raise ArtifactError("strict resume state committed offset is invalid")
            else:
                if partfd is not None:
                    raise ArtifactError("strict resume partial has no trusted state")
                partfd = _open_pinned(part, create=True, label="partial")
                committed = 0
                metadata = {"schema": "strict-resume-v2", "status": "active", "identity": identity, "expected_size": expected_size, "expected_sha256": expected_sha256.lower(), "block_size": block_size, "committed": 0, "blocks": []}
                _atomic_json(state, metadata)
                statefd = _open_pinned(state, create=False, label="state")
            if committed < 0 or committed > expected_size or partfd is None and committed:
                raise ArtifactError("strict resume state/partial mismatch")
            if partfd is not None:
                _same_path_fd(part, partfd, "partial")
                part_size = os.fstat(partfd).st_size
                if part_size < committed:
                    raise ArtifactError("strict resume partial size mismatch")
                if committed:
                    offset = 0
                    for entry in metadata["blocks"]:
                        chunk = os.pread(partfd, entry["size"], offset)
                        if len(chunk) != entry["size"] or hashlib.sha256(chunk).hexdigest() != entry["sha256"]:
                            raise ArtifactError("strict resume committed block hash mismatch")
                        offset += entry["size"]
                if part_size > committed:
                    # Data may have reached disk before the state replacement.
                    # It is owned by this identity; discard only the uncommitted
                    # tail after the trusted prefix has been checked.
                    _same_path_fd(part, partfd, "partial")
                    os.ftruncate(partfd, committed); os.fsync(partfd)
            url = file_url(rel_path, repo=repo, revision=revision, endpoint=endpoint)
            chunk_size = block_size; window = 8 << 20
            retries = 0
            while committed < expected_size:
                if deadline is not None and time.monotonic() >= deadline: raise ArtifactError("strict resume deadline exceeded")
                end = min(expected_size - 1, committed + window - 1)
                timeout = min(30.0, deadline - time.monotonic()) if deadline is not None else 30.0
                if timeout <= 0:
                    raise ArtifactError("strict resume deadline exceeded")
                req = urllib.request.Request(url, headers={"User-Agent": _UA, "Range": f"bytes={committed}-{end}"})
                try:
                    if allowed_redirect_hosts is None:
                        resp = urllib.request.urlopen(req, timeout=timeout)
                    else:
                        parsed = urllib.parse.urlsplit(url)
                        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.hostname.casefold() not in allowed_redirect_hosts:
                            raise ArtifactError("endpoint host is not allowlisted")
                        resp = urllib.request.build_opener(_AllowedRedirect(allowed_redirect_hosts, parsed.scheme)).open(req, timeout=timeout)
                    with resp:
                        final_host = urllib.parse.urlsplit(resp.geturl()).hostname
                        if (allowed_redirect_hosts is not None and expected_size > 10 * 1024 * 1024 and
                                (final_host or "").casefold() == "huggingface.co"):
                            raise ArtifactError("strict large artifact response remained on huggingface.co")
                        if resp.status != 206:
                            raise ArtifactError(f"strict resume requires HTTP 206, got {resp.status}")
                        cr = resp.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", cr)
                        length = int(resp.headers.get("Content-Length", "-1"))
                        if not match or int(match.group(1)) != committed or int(match.group(2)) != end or int(match.group(3)) != expected_size or length != end - committed + 1:
                            raise ArtifactError("strict resume Content-Range/Length mismatch")
                        if deadline is not None and resp.headers.get("Transfer-Encoding"):
                            raise ArtifactError("strict resume requires a fixed-length deadline-bound response")
                        sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
                        if sock is not None:
                            try:
                                if sock.fileno() < 0:
                                    sock = None
                            except OSError:
                                sock = None
                        pending = b""
                        os.lseek(partfd, 0, os.SEEK_END)
                        while True:
                            remaining_range = end - committed - len(pending) + 1
                            if remaining_range <= 0:
                                break
                            if deadline is not None and time.monotonic() >= deadline: raise ArtifactError("strict resume deadline exceeded")
                            if deadline is not None:
                                if sock is None:
                                    raise ArtifactError("strict resume socket timeout unavailable")
                                remaining_time = deadline - time.monotonic()
                                if remaining_time <= 0:
                                    raise ArtifactError("strict resume deadline exceeded")
                                try:
                                    sock.settimeout(min(30.0, remaining_time))
                                except OSError as exc:
                                    raise ArtifactError("strict resume socket timeout unavailable") from exc
                                read_body = getattr(resp, "read1", None)
                                if not callable(read_body):
                                    raise ArtifactError("strict resume bounded body read unavailable")
                            else:
                                read_body = resp.read
                            data = read_body(min(chunk_size, remaining_range))
                            if not data: break
                            pending += data
                            while len(pending) >= chunk_size and committed < end + 1:
                                block, pending = pending[:chunk_size], pending[chunk_size:]
                                _same_path_fd(part, partfd, "partial")
                                os.write(partfd, block); os.fsync(partfd); committed += len(block)
                                metadata["blocks"].append({"offset": committed - len(block), "size": len(block), "sha256": hashlib.sha256(block).hexdigest()})
                                metadata["committed"] = committed
                                _same_path_fd(state, statefd, "state")
                                _atomic_json(state, metadata)
                                os.close(statefd); statefd = _open_pinned(state, create=False, label="state")
                        if pending and committed + len(pending) == end + 1:
                            _same_path_fd(part, partfd, "partial")
                            os.write(partfd, pending); os.fsync(partfd); committed += len(pending)
                            metadata["blocks"].append({"offset": committed - len(pending), "size": len(pending), "sha256": hashlib.sha256(pending).hexdigest()})
                            metadata["committed"] = committed
                            _same_path_fd(state, statefd, "state")
                            _atomic_json(state, metadata)
                            os.close(statefd); statefd = _open_pinned(state, create=False, label="state")
                        if committed < end + 1: raise ArtifactError("strict resume short body")
                    retries = 0
                except (urllib.error.HTTPError, urllib.error.URLError, OSError, ArtifactError) as exc:
                    if partfd is not None:
                        _same_path_fd(part, partfd, "partial")
                        if os.fstat(partfd).st_size > committed:
                            os.ftruncate(partfd, committed); os.fsync(partfd)
                    retries += 1
                    if deadline is not None and time.monotonic() >= deadline:
                        raise ArtifactError("strict resume deadline exceeded") from exc
                    if retries > max_retries: raise
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ArtifactError("strict resume deadline exceeded during retry")
                        time.sleep(min(0.05 * (2 ** (retries - 1)), remaining))
            _same_path_fd(part, partfd, "partial")
            if os.fstat(partfd).st_size != expected_size or _fd_sha256(partfd, deadline=deadline) != expected_sha256.lower():
                raise ArtifactError("strict resume final SHA256/size mismatch")
            _same_path_fd(part, partfd, "partial")
            os.replace(part, dest); state.unlink(missing_ok=True)
            return dest
        finally:
            if partfd is not None: os.close(partfd)
            if statefd is not None: os.close(statefd)
            fcntl.flock(lockstream.fileno(), fcntl.LOCK_UN)
            lockstream.close()
    except Exception:
        lockstream.close()
        raise


def download_and_extract_tarball(
    rel_path: str,
    dest_dir: Path,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
) -> Path:
    """Download a .tar.gz from HF, verify SHA-256, extract into ``dest_dir``.

    Extraction is done into a temp directory first; on success the contents
    are moved atomically into ``dest_dir`` to avoid leaving a partial state.
    Returns the dest_dir path.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hf_extract_", dir=str(dest_dir.parent)) as tmpdir:
        tmpdir_path = Path(tmpdir)
        tarball = tmpdir_path / Path(rel_path).name
        download_file(
            rel_path,
            tarball,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            repo=repo,
            revision=revision,
        )

        extract_dir = tmpdir_path / "extracted"
        extract_dir.mkdir()
        with tarfile.open(tarball, "r:gz") as tf:
            # Reject absolute paths and ".." traversal.
            for member in tf.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ArtifactError(f"unsafe tar member: {member.name}")
            tf.extractall(extract_dir)

        # Move extracted contents into dest_dir, overwriting per-file.
        for item in extract_dir.iterdir():
            target = dest_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))

    return dest_dir
