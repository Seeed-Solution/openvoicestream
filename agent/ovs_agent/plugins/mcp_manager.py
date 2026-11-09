"""MCP server configuration management, secure persistence, sandboxed test execution, and restart watchdog.

Implements the spec defined in docs/specs/spec-dev-flow-mcp-and-frontend-v7.3.md:
- Platform validation (O_NOFOLLOW, O_DIRECTORY).
- Bijective error mapping (SAFE_CLIENT_MESSAGES, HTTP_STATUS_MAP).
- Unified exceptions (DashboardApiError, SecurityError, ConfigPersistenceError).
- Desensitized boundary decorator (@mcp_api_boundary).
- Safe atomic file I/O primitives.
- Persistent recovery marker and cleanup primitives.
- Stdio anti-TOCTOU execution using /proc/self/fd/<fd>.
- SSE anti-SSRF client with pinned DNS.
- Two-phase atomic commit with dual-hash domain validation.
- Algebraic restart watchdog with generation tracking.
"""
from __future__ import annotations

import asyncio
import copy
import errno
import functools
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import aiohttp
from aiohttp import web
import yaml

logger = logging.getLogger(__name__)

# ── 1. Platform Capability & Environment Verification ──────────────────────

def validate_platform_capabilities() -> None:
    """Validate mandatory POSIX file flags to guard against silent degradation."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("Linux/POSIX environment required: os.O_NOFOLLOW and os.O_DIRECTORY are mandatory")

validate_platform_capabilities()

MAX_CONFIG_SIZE = 2 * 1024 * 1024  # 2MB hard limit

# ── 2. Error Mappings & Bijective Consistency ─────────────────────────────

SAFE_CLIENT_MESSAGES: dict[str, str] = {
    "UNAUTHORIZED": "Authentication required or invalid admin token",
    "FORBIDDEN": "Request forbidden or CSRF validation failed",
    "FILE_TOO_LARGE": "Configuration size exceeds 2MB limit",
    "CONFIG_PARSE_ERROR": "Configuration syntax is invalid",
    "SECURITY_VIOLATION": "Security validation failed",
    "INVALID_PARAM": "Invalid request parameter",
    "ETAG_MISMATCH": "Configuration was modified concurrently, please refresh",
    "CONCURRENT_MODIFICATION": "Operation conflict, please retry",
    "TOO_MANY_REQUESTS": "Too many concurrent requests, please retry later",
    "SERVICE_UNAVAILABLE": "Service is shutting down or unavailable",
    "CONNECTION_FAILED": "Failed to connect to MCP server or discovery failed",
    "TEST_TIMEOUT": "MCP server test connection timed out",
    "DISK_FULL": "Host storage space is exhausted",
    "TMP_CREATE_FAILED": "Failed to create temporary configuration file",
    "TMP_CLEANUP_FAILED": "Failed to clean up temporary configuration files",
    "FD_CLOSE_FAILED": "Failed to close file descriptor safely",
    "FCHMOD_FAILED": "Failed to set secure file permissions",
    "FSYNC_FAILED": "Failed to synchronize configuration file to storage",
    "DIR_FSYNC_FAILED": "Failed to synchronize directory metadata to storage",
    "WRITE_FAILED": "Failed to write configuration data",
    "SHORT_WRITE": "Incomplete configuration write encountered",
    "REPLACE_FAILED": "Failed to replace configuration file atomically",
    "READ_FAILED": "Failed to read configuration file",
    "CONFIG_RECOVERY_REQUIRED": "Configuration state requires manual recovery; locked in read-only mode",
    "INTERNAL_ERROR": "Internal server error occurred",
    "MISSING_ARGUMENT": "参数缺失，stdio 模式需要提供非空 args 列表",
    "INVALID_MODULE_NAME": "Python 模块名格式不正确，必须为合法的模块标识符",
    "INVALID_SCRIPT_PATH": "Python 脚本路径不合法，必须以 .py 结尾且不能以选项参数开头",
    "SCRIPT_NOT_FOUND": "指定的 Python 脚本文件不存在，请检查路径是否正确",
    "SCRIPT_NOT_READABLE": "Python 脚本文件无读取权限，请检查文件权限",
    "ILLEGAL_ARGUMENT": "包含不支持的参数格式，严禁使用 -c 参数",
    "FORBIDDEN_ENV": "禁止设置动态链接器相关环境变量",
}

HTTP_STATUS_MAP: dict[str, int] = {
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "FILE_TOO_LARGE": 400,
    "CONFIG_PARSE_ERROR": 400,
    "SECURITY_VIOLATION": 400,
    "INVALID_PARAM": 400,
    "ETAG_MISMATCH": 409,
    "CONCURRENT_MODIFICATION": 409,
    "TOO_MANY_REQUESTS": 429,
    "SERVICE_UNAVAILABLE": 503,
    "CONNECTION_FAILED": 502,
    "TEST_TIMEOUT": 504,
    "DISK_FULL": 500,
    "TMP_CREATE_FAILED": 500,
    "TMP_CLEANUP_FAILED": 500,
    "FD_CLOSE_FAILED": 500,
    "FCHMOD_FAILED": 500,
    "FSYNC_FAILED": 500,
    "DIR_FSYNC_FAILED": 500,
    "WRITE_FAILED": 500,
    "SHORT_WRITE": 500,
    "REPLACE_FAILED": 500,
    "READ_FAILED": 500,
    "CONFIG_RECOVERY_REQUIRED": 500,
    "INTERNAL_ERROR": 500,
    "MISSING_ARGUMENT": 400,
    "INVALID_MODULE_NAME": 400,
    "INVALID_SCRIPT_PATH": 400,
    "SCRIPT_NOT_FOUND": 400,
    "SCRIPT_NOT_READABLE": 400,
    "ILLEGAL_ARGUMENT": 400,
    "FORBIDDEN_ENV": 400,
}

VALID_HTTP_STATUSES: set[int] = {400, 401, 403, 409, 429, 500, 502, 503, 504}


def validate_error_maps_consistency() -> None:
    """Module-level bijective assertion on error codes and HTTP statuses."""
    assert isinstance(VALID_HTTP_STATUSES, set) and len(VALID_HTTP_STATUSES) > 0, (
        "VALID_HTTP_STATUSES must be non-empty set"
    )
    assert all(type(s) is int for s in VALID_HTTP_STATUSES), "All status codes must be int"
    if not SAFE_CLIENT_MESSAGES or not HTTP_STATUS_MAP:
        raise RuntimeError("Error mappings must not be empty")
    if set(SAFE_CLIENT_MESSAGES.keys()) != set(HTTP_STATUS_MAP.keys()):
        raise RuntimeError(
            f"Mapping mismatch! in_msg_only={set(SAFE_CLIENT_MESSAGES) - set(HTTP_STATUS_MAP)}, "
            f"in_status_only={set(HTTP_STATUS_MAP) - set(SAFE_CLIENT_MESSAGES)}"
        )
    for code, status in HTTP_STATUS_MAP.items():
        if type(status) is not int or status not in VALID_HTTP_STATUSES:
            raise RuntimeError(f"Invalid integer status code {status} configured for {code}")


validate_error_maps_consistency()

# ── 3. Unified Exceptions ──────────────────────────────────────────────────

class DashboardApiError(Exception):
    def __init__(self, code: str, internal_message: str, errno: int | None = None):
        if code not in HTTP_STATUS_MAP:
            raise RuntimeError(f"FATAL: Unregistered error code passed to DashboardApiError: {code}")
        super().__init__(internal_message)
        self.code = code
        self.internal_message = internal_message
        self.errno = errno

    @property
    def status(self) -> int:
        return HTTP_STATUS_MAP[self.code]

    def to_response(self, request_id: str) -> web.Response:
        client_msg = SAFE_CLIENT_MESSAGES[self.code]
        return web.json_response(
            {
                "ok": False,
                "code": self.code,
                "error": client_msg,
                "request_id": request_id,
            },
            status=self.status,
        )


class SecurityError(DashboardApiError):
    def __init__(self, internal_message: str, code: str = "SECURITY_VIOLATION"):
        super().__init__(code=code, internal_message=internal_message)


class ConfigPersistenceError(DashboardApiError):
    def __init__(self, code: str, internal_message: str, errno: int | None = None):
        super().__init__(code=code, internal_message=internal_message, errno=errno)


def mcp_api_boundary(handler):
    """Global boundary decorator: intercepts all errors, masks client messages, attaches request_id."""
    @functools.wraps(handler)
    async def wrapper(*args, **kwargs) -> web.Response:
        request_id = secrets.token_hex(8)
        try:
            return await handler(*args, **kwargs)
        except DashboardApiError as e:
            logger.warning(
                "MCP API error [%s] %s: %s (errno=%s)",
                request_id,
                e.code,
                e.internal_message,
                e.errno,
            )
            return e.to_response(request_id)
        except json.JSONDecodeError as e:
            logger.warning("JSON syntax error [%s]: %s", request_id, e)
            err = DashboardApiError("CONFIG_PARSE_ERROR", f"JSONDecodeError: {e}")
            return err.to_response(request_id)
        except Exception as e:
            logger.exception("Unhandled MCP API exception [%s]: %s", request_id, e)
            err = DashboardApiError("INTERNAL_ERROR", str(e))
            return err.to_response(request_id)

    return wrapper

# ── 4. Low-Level Safe Atomic File Primitives ───────────────────────────────

def _safe_close(fd: int, raise_on_error: bool = False) -> None:
    """Safely close file descriptor; EBADF is ignored, other errors can optionally raise FD_CLOSE_FAILED."""
    try:
        os.close(fd)
    except OSError as e:
        if e.errno != errno.EBADF:
            logger.warning("Error closing fd %d: %s", fd, e)
            if raise_on_error:
                raise ConfigPersistenceError("FD_CLOSE_FAILED", f"Close fd {fd} failed: {e}", e.errno) from e


def _safe_open_readonly(path: str) -> int:
    """Open regular file read-only with O_NOFOLLOW and O_CLOEXEC."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("os.O_NOFOLLOW is mandatory")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(path, flags)
    except OSError as e:
        raise ConfigPersistenceError("READ_FAILED", f"Open {path} failed: {e}", e.errno) from e


def _safe_fchmod(fd: int, mode: int) -> None:
    """Set exact file permission mode on an open descriptor."""
    try:
        os.fchmod(fd, mode)
    except OSError as e:
        raise ConfigPersistenceError("FCHMOD_FAILED", f"fchmod failed on fd {fd}: {e}", e.errno) from e


def _safe_fsync_dir(dirpath: str) -> None:
    """Fsync directory metadata to physical media."""
    if not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("os.O_DIRECTORY is mandatory")
    dfd = -1
    try:
        dfd = os.open(dirpath, os.O_DIRECTORY)
        os.fsync(dfd)
    except OSError as e:
        raise ConfigPersistenceError("DIR_FSYNC_FAILED", f"Fsync dir {dirpath} failed: {e}", e.errno) from e
    finally:
        if dfd >= 0:
            _safe_close(dfd)


def _open_atomic_tmp(path: str) -> int:
    """Open new atomic temp file with 0o600, O_EXCL, O_NOFOLLOW, and O_CLOEXEC."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("os.O_NOFOLLOW is mandatory for atomic file persistence")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = -1
    try:
        fd = os.open(path, flags, 0o600)
        _safe_fchmod(fd, 0o600)
        st = os.fstat(fd)
        if stat.S_IMODE(st.st_mode) != 0o600:
            raise ConfigPersistenceError("FCHMOD_FAILED", f"Exact 0600 mode verification failed: {oct(st.st_mode)}")
        return fd
    except Exception as e:
        if fd >= 0:
            _safe_close(fd)
        try:
            os.unlink(path)
        except OSError as unlink_err:
            if unlink_err.errno != errno.ENOENT:
                logger.critical("FATAL: Failed to unlink tmp file %s: %s", path, unlink_err)
                raise ConfigPersistenceError(
                    "TMP_CLEANUP_FAILED",
                    f"Failed to clean up temporary file: {unlink_err}",
                    unlink_err.errno,
                ) from unlink_err
        if isinstance(e, DashboardApiError):
            raise
        errno_val = getattr(e, "errno", None)
        raise ConfigPersistenceError("TMP_CREATE_FAILED", f"Create tmp file failed: {e}", errno_val) from e


def _safe_fsync_file(fd: int) -> None:
    """Fsync open file descriptor to disk."""
    try:
        os.fsync(fd)
    except OSError as e:
        code = "DISK_FULL" if e.errno == errno.ENOSPC else "FSYNC_FAILED"
        raise ConfigPersistenceError(code, f"Fsync file failed: {e}", e.errno) from e


def _safe_replace(src: str, dst: str) -> None:
    """Atomically replace src -> dst, falling back to direct copy on bind mounts."""
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno == errno.EBUSY:
            logger.warning(
                "Target %s is a single-file bind mount point; replacing via direct copy (atomic rename degraded)",
                dst,
            )
            # Docker single-file bind mount point: cannot rename over mount point.
            # Fall back to copying content directly into the existing file inode.
            # Pre-read original content so we can restore dst if copy or fsync fails.
            original_dst_bytes: bytes | None = None
            try:
                flags_ro = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
                ro_fd = os.open(dst, flags_ro)
                try:
                    original_dst_bytes = _read_all_bounded(ro_fd, MAX_CONFIG_SIZE)
                finally:
                    os.close(ro_fd)
            except Exception:
                pass

            try:
                flags = os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
                dst_fd = os.open(dst, flags)
                try:
                    with open(src, "rb") as s_fp, open(dst_fd, "wb", closefd=False) as d_fp:
                        shutil.copyfileobj(s_fp, d_fp)
                        d_fp.flush()
                        os.fsync(dst_fd)
                finally:
                    os.close(dst_fd)
                try:
                    shutil.copymode(src, dst)
                except OSError:
                    pass
                try:
                    os.unlink(src)
                except OSError:
                    pass
                return
            except OSError as write_err:
                # If copy/fsync failed after truncation, restore the original content
                if original_dst_bytes is not None:
                    try:
                        restore_flags = os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
                        r_fd = os.open(dst, restore_flags)
                        try:
                            _write_all(r_fd, original_dst_bytes)
                            _safe_fsync_file(r_fd)
                        finally:
                            os.close(r_fd)
                        logger.info("Successfully restored original content of %s after failed bind-mount copy", dst)
                    except Exception as rest_err:
                        logger.critical("FATAL: Failed to restore %s after failed bind-mount copy: %s", dst, rest_err)
                raise ConfigPersistenceError(
                    "REPLACE_FAILED",
                    f"Replace (bind-mount copy) {src} -> {dst} failed: {write_err}",
                    write_err.errno,
                ) from write_err
        raise ConfigPersistenceError("REPLACE_FAILED", f"Replace {src} -> {dst} failed: {e}", e.errno) from e


def _write_all(fd: int, data: bytes) -> None:
    """Write all bytes to fd, retrying on InterruptedError and catching ENOSPC."""
    pos = 0
    while pos < len(data):
        try:
            n = os.write(fd, data[pos:])
        except InterruptedError:
            continue
        except OSError as e:
            code = "DISK_FULL" if e.errno == errno.ENOSPC else "WRITE_FAILED"
            raise ConfigPersistenceError(code, f"Write failed: {e}", errno=e.errno) from e
        if n <= 0:
            raise ConfigPersistenceError("SHORT_WRITE", "Short write encountered")
        pos += n


def _read_all_bounded(fd: int, max_bytes: int = MAX_CONFIG_SIZE) -> bytes:
    """Read all bytes up to max_bytes, raising FILE_TOO_LARGE if exceeded."""
    buf = bytearray()
    while True:
        try:
            chunk = os.read(fd, 65536)
        except InterruptedError:
            continue
        except OSError as e:
            raise ConfigPersistenceError("READ_FAILED", f"Read failed: {e}", errno=e.errno) from e
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise ConfigPersistenceError("FILE_TOO_LARGE", f"File content exceeds {max_bytes} bytes limit")
    return bytes(buf)

# ── 5. Recovery Marker & Startup Cleanup Primitives ────────────────────────

def fatal_config_recovery_exit(reason: str, exit_code: int = 78) -> None:
    """Flush logs, stream buffers, and exit with code 78 to block supervisor loop."""
    logger.critical("FATAL CONFIG FAILURE: %s. Process exiting with code %d", reason, exit_code)
    try:
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    sys.exit(exit_code)


MARKER_REGEX = re.compile(
    r"^CONFIG_RECOVERY_REQUIRED:v1:reason=([a-zA-Z0-9_\-\.]{1,64}):timestamp=([0-9]+\.[0-9]+)\n$"
)
MARKER_TMP_REGEX = re.compile(r"^\.config_recovery_required\.marker_tmp\.[0-9a-f]{32}$")


def write_persistent_recovery_marker(config_dir: str, reason: str) -> None:
    """Atomically write .config_recovery_required; wrapped in outer try/except, exiting with 78 on failure."""
    marker_path = os.path.join(config_dir, ".config_recovery_required")
    marker_tmp = f"{marker_path}.marker_tmp.{uuid.uuid4().hex}"
    try:
        if not re.match(r"^[a-zA-Z0-9_\-\.]{1,64}$", reason):
            raise ValueError(f"Invalid reason format: {reason}")
        fd = -1
        try:
            fd = _open_atomic_tmp(marker_tmp)
            payload = f"CONFIG_RECOVERY_REQUIRED:v1:reason={reason}:timestamp={time.time():.6f}\n".encode("utf-8")
            _write_all(fd, payload)
            _safe_fsync_file(fd)
        finally:
            if fd >= 0:
                _safe_close(fd, raise_on_error=True)
        _safe_replace(marker_tmp, marker_path)
        _safe_fsync_dir(config_dir)
    except SystemExit:
        raise
    except Exception as e:
        try:
            os.unlink(marker_tmp)
        except OSError as unl_err:
            if unl_err.errno != errno.ENOENT:
                logger.critical("FATAL: Failed to unlink marker tmp file %s: %s", marker_tmp, unl_err)
        fatal_config_recovery_exit(f"Failed to commit persistent recovery marker: {e}")


def check_persistent_recovery_marker(config_dir: str) -> bool:
    """Scan and verify .config_recovery_required or orphaned marker_tmp files."""
    if not os.path.exists(config_dir):
        return False
    marker_path = os.path.join(config_dir, ".config_recovery_required")

    # 1. Scan for orphaned marker_tmp files
    try:
        with os.scandir(config_dir) as entries:
            for entry in entries:
                if MARKER_TMP_REGEX.match(entry.name):
                    entry_type = (
                        "symlink"
                        if entry.is_symlink()
                        else "dir"
                        if entry.is_dir()
                        else "file"
                        if entry.is_file()
                        else "other"
                    )
                    logger.critical(
                        "FATAL: Found orphaned recovery marker tmp %s (type=%s); persistent lock enforced",
                        entry.path,
                        entry_type,
                    )
                    return True
    except OSError as e:
        raise ConfigPersistenceError(
            "CONFIG_RECOVERY_REQUIRED", f"Failed to scan config dir for marker tmp: {e}", e.errno
        ) from e

    # 2. Check formal marker file
    try:
        fd = _safe_open_readonly(marker_path)
    except ConfigPersistenceError as e:
        if e.errno == errno.ENOENT:
            return False
        raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", f"Corrupted recovery marker: {e}", e.errno) from e

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Recovery marker is not a regular file")
        if st.st_uid not in {os.getuid(), 0}:
            raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Untrusted recovery marker owner")
        if stat.S_IMODE(st.st_mode) != 0o600:
            raise ConfigPersistenceError(
                "CONFIG_RECOVERY_REQUIRED", f"Recovery marker mode must be exactly 0600, got {oct(st.st_mode)}"
            )
        if st.st_size > 4096:
            raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Recovery marker size exceeds 4KB limit")
        raw_bytes = _read_all_bounded(fd, 4096)
        try:
            data = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as dec_err:
            raise ConfigPersistenceError(
                "CONFIG_RECOVERY_REQUIRED", f"Recovery marker decode failed: {dec_err}"
            ) from dec_err

        m = MARKER_REGEX.match(data)
        if not m:
            raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Recovery marker semantic structure corrupted")
        ts = float(m.group(2))
        if ts <= 0 or not math.isfinite(ts):
            raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Recovery marker timestamp invalid")
        return True
    finally:
        _safe_close(fd)


ORPHANED_TMP_REGEX = re.compile(r"^.*\.(backup\.|restore\.)?tmp\.[0-9a-f]{32}$")


def cleanup_orphaned_tmp_files(config_dir: str) -> None:
    """Startup scanning and cleanup of orphaned configuration tmp files."""
    if not os.path.exists(config_dir):
        return
    if check_persistent_recovery_marker(config_dir):
        raise ConfigPersistenceError(
            "CONFIG_RECOVERY_REQUIRED", "Persistent recovery marker detected; configuration locked"
        )

    try:
        with os.scandir(config_dir) as entries:
            for entry in entries:
                if ORPHANED_TMP_REGEX.match(entry.name) and not MARKER_TMP_REGEX.match(entry.name):
                    if entry.is_symlink():
                        write_persistent_recovery_marker(config_dir, "symlink_in_tmp")
                        raise ConfigPersistenceError("TMP_CLEANUP_FAILED", f"Symlink detected in tmp file: {entry.path}")
                    if not entry.is_file():
                        continue
                    try:
                        os.unlink(entry.path)
                        logger.info("Cleaned orphaned tmp file: %s", entry.path)
                    except OSError as e:
                        if e.errno != errno.ENOENT:
                            logger.critical("FATAL: Failed to clean orphaned tmp file %s: %s", entry.path, e)
                            write_persistent_recovery_marker(config_dir, "unlink_tmp_failed")
                            raise ConfigPersistenceError(
                                "TMP_CLEANUP_FAILED", f"Startup tmp cleanup failed: {e}", e.errno
                            ) from e
    except OSError as scan_err:
        logger.critical("FATAL: Failed to scan config directory for tmp files: %s", scan_err)
        write_persistent_recovery_marker(config_dir, "scan_dir_failed")
        raise ConfigPersistenceError(
            "TMP_CLEANUP_FAILED", f"Scan config dir failed: {scan_err}", scan_err.errno
        ) from scan_err

# ── 6. stdio Execution & Anti-TOCTOU ────────────────────────────────────────

ALLOWED_MCP_MODULES: set[str] = {"ovs_agent.mcp.servers", "ha_mcp", "device_mcp"}
SHELL_METACHARS: set[str] = set(";&|`$><\n\r")
MODULE_NAME_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")


def redact_sensitive_text(text: str) -> str:
    """Redact sensitive patterns (tokens, passwords, secrets, api keys) from text."""
    if not text:
        return ""
    # 0. PEM private keys
    text = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
        "******",
        text,
    )
    # 1. Double-quoted values with escaped quotes (supporting := or space separator)
    text = re.sub(
        r'(?i)(["\']?(?:api[_-]?key|token|secret|password|passwd|auth)[_-]?(?:key|token|id)?["\']?(?:\s*[:=]\s*|\s+))"([^"\\]*(?:\\.[^"\\]*)*)"',
        r'\1"******"',
        text,
    )
    # 2. Single-quoted values with escaped quotes (supporting := or space separator)
    text = re.sub(
        r"(?i)([\"']?(?:api[_-]?key|token|secret|password|passwd|auth)[_-]?(?:key|token|id)?[\"']?(?:\s*[:=]\s*|\s+))'([^'\\]*(?:\\.[^'\\]*)*)'",
        r"\1'******'",
        text,
    )
    # 3. Unquoted key: value / key=value / key <space> value
    # Supports indented continuation lines even across blank lines
    text = re.sub(
        r'(?i)(["\']?(?:api[_-]?key|token|secret|password|passwd|auth)[_-]?(?:key|token|id)?["\']?(?:\s*[:=]\s*|\s+))([^\s,;}{"\'\r\n][^\r\n]*(?:\r?\n(?:[ \t]*\r?\n)*[ \t]+[^\r\n]+)*)',
        r'\1"******"',
        text,
    )
    # 4. Bearer tokens with base64 / urlsafe chars (+, /, =)
    text = re.sub(r'(?i)(bearer\s+)[a-zA-Z0-9_\-\.+/=]{8,}', r'\1******', text)
    # 5. sk- tokens
    text = re.sub(r'(?i)(sk-[a-zA-Z0-9_\-]{8,})', r'sk-******', text)
    # 6. ghp_ tokens
    text = re.sub(r'\b(ghp_[a-zA-Z0-9]{30,}|github_pat_[a-zA-Z0-9_]{30,})\b', r'******', text)
    return text


FORBIDDEN_ENV_PREFIXES: tuple[str, ...] = ("PYTHON", "LD_", "DYLD_", "PERL5")
FORBIDDEN_ENV_EXACT: frozenset[str] = frozenset({
    "PATH",
    "VIRTUAL_ENV",
    "BROWSER",
    "NODE_OPTIONS",
    "PERL5",
    "IFS",
    "ENV",
    "BASH_ENV",
})
ENV_KEY_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")


def validate_mcp_env(env: Any) -> None:
    """Validate stdio MCP environment variables.

    Enforces:
    1. Key must match ^[A-Za-z_][A-Za-z0-9_]*$.
    2. Prohibits environment variables that can hijack Python interpreter or dynamic linkers:
       - Prefix match (case-insensitive): PYTHON*, LD_*, DYLD_*, PERL5*
       - Exact match (case-insensitive): PATH, VIRTUAL_ENV, BROWSER, NODE_OPTIONS, PERL5, IFS, ENV, BASH_ENV
       Raises SecurityError if any violating key is encountered.
    3. env values must be strings; raises DashboardApiError("INVALID_PARAM", ...) otherwise.
    """
    if env is None:
        return
    if not isinstance(env, dict):
        raise DashboardApiError("INVALID_PARAM", "env must be a dictionary")

    for k, v in env.items():
        if not isinstance(k, str) or not ENV_KEY_REGEX.match(k):
            raise SecurityError(f"Forbidden environment variable key: {k}")

        upper_k = k.upper()
        if upper_k == "PYTHONUNBUFFERED":
            continue
        if (
            any(upper_k.startswith(p) for p in FORBIDDEN_ENV_PREFIXES)
            or upper_k in FORBIDDEN_ENV_EXACT
        ):
            raise SecurityError(f"Forbidden environment variable key: {k}", code="FORBIDDEN_ENV")

        if not isinstance(v, str):
            raise DashboardApiError(
                "INVALID_PARAM",
                f"env value for key '{k}' must be a string, got {type(v).__name__}",
            )


def normalize_stdio_mcp_config(
    command: str | None,
    args: Sequence[str],
    env: dict[str, str] | None = None,
    config_dir: str | None = None,
) -> tuple[str, list[str], dict[str, str], str]:
    """Validate and normalize stdio config into (command, args, env, cwd)."""
    # 1. 基本结构检查：args 必须为非空字符串列表
    if not isinstance(args, (list, tuple)) or not args:
        raise DashboardApiError("MISSING_ARGUMENT", "参数缺失，stdio 模式需要提供非空 args 列表")
    for arg in args:
        if not isinstance(arg, str):
            raise DashboardApiError("MISSING_ARGUMENT", "参数缺失，stdio 模式需要提供非空 args 列表")

    # 2. 禁止参数与特殊字符：
    for arg in args:
        if any(c in SHELL_METACHARS for c in arg):
            raise DashboardApiError("SECURITY_VIOLATION", f"stdio args contain shell metacharacter: {arg}")
    for arg in args:
        if arg == "-c" or arg.startswith("-c"):
            raise DashboardApiError("ILLEGAL_ARGUMENT", "包含不支持的参数格式，严禁使用 -c 参数")

    # 3. 模式匹配与区分：
    base_dir = config_dir if config_dir else os.getcwd()
    if args[0] == "-m":
        if len(args) < 2:
            raise DashboardApiError("MISSING_ARGUMENT", "参数缺失，stdio 模式需要提供非空 args 列表")
        if not MODULE_NAME_REGEX.match(args[1]):
            raise DashboardApiError("INVALID_MODULE_NAME", "Python 模块名格式不正确，必须为合法的模块标识符")
        norm_args = list(args)
        cwd = base_dir
    else:
        if args[0].startswith("-"):
            raise DashboardApiError("INVALID_SCRIPT_PATH", "Python 脚本路径不合法，必须以 .py 结尾且不能以选项参数开头")
        if not re.search(r"\.py$", args[0], re.I):
            raise DashboardApiError("INVALID_SCRIPT_PATH", "Python 脚本路径不合法，必须以 .py 结尾且不能以选项参数开头")
        abs_path = os.path.abspath(args[0]) if os.path.isabs(args[0]) else os.path.abspath(os.path.join(base_dir, args[0]))
        if not os.path.isfile(abs_path):
            raise DashboardApiError("SCRIPT_NOT_FOUND", "指定的 Python 脚本文件不存在，请检查路径是否正确")
        if not os.access(abs_path, os.R_OK):
            raise DashboardApiError("SCRIPT_NOT_READABLE", "Python 脚本文件无读取权限，请检查文件权限")
        resolved_script = os.path.realpath(abs_path)
        norm_args = [resolved_script, *args[1:]]
        cwd = os.path.dirname(resolved_script)

    # 4. Command 解释器规范化（保留 virtualenv 解释器路径）
    if command and str(command).strip() and str(command).strip() != sys.executable:
        target_cmd = str(command).strip()
        exe_fd, venv_bin, _ = inspect_and_open_verified_exe(target_cmd)
        _safe_close(exe_fd)
        if not os.path.isabs(target_cmd):
            resolved = shutil.which(target_cmd)
            norm_command = resolved if resolved else target_cmd
        else:
            norm_command = target_cmd
        cmd_dir = os.path.dirname(os.path.abspath(norm_command))
        venv_root = os.path.dirname(cmd_dir)
        if os.path.isfile(os.path.join(venv_root, "pyvenv.cfg")):
            venv_bin = cmd_dir
    else:
        norm_command = sys.executable
        venv_bin = os.path.dirname(sys.executable)

    # 5. 环境变量校验与构建
    norm_env = build_mcp_env(venv_bin, env)

    return norm_command, norm_args, norm_env, cwd


def validate_stdio_args(
    args: Sequence[str],
    project_root: str | None = None,
    env: dict[str, Any] | None = None,
    config_dir: str | None = None,
) -> None:
    """Validate stdio args leveraging normalize_stdio_mcp_config."""
    normalize_stdio_mcp_config(
        command=None,
        args=args,
        env=env,
        config_dir=config_dir or project_root,
    )


def inspect_and_open_verified_exe(exe_path: str | None = None) -> tuple[int, str, list[tuple[str, int, int]]]:
    """Inspect sys.executable or whitelisted Python interpreter, verify parent directory ownership/permissions, and return open fd."""
    if not exe_path or not str(exe_path).strip():
        target_exe = sys.executable
    else:
        target_exe = str(exe_path).strip()
        if not os.path.isabs(target_exe):
            resolved = shutil.which(target_exe)
            if not resolved:
                raise SecurityError(f"Command executable not found: {target_exe}")
            target_exe = resolved

    real_exe = os.path.realpath(target_exe)
    real_sys_exe = os.path.realpath(sys.executable)
    real_prefix = os.path.realpath(sys.prefix)
    real_base = os.path.realpath(getattr(sys, "base_prefix", sys.prefix))
    try:
        in_prefix = os.path.commonpath([real_exe, real_prefix]) == real_prefix
        in_base = os.path.commonpath([real_exe, real_base]) == real_base
        if not (in_prefix or in_base or real_exe == real_sys_exe):
            raise SecurityError(f"Command executable {real_exe} is not an authorized Python interpreter in sys.prefix or sys.base_prefix")
    except ValueError as e:
        raise SecurityError(f"commonpath validation failed for executable {real_exe}: {e}") from e

    basename = os.path.basename(real_exe)
    if not (basename.startswith("python") or real_exe == real_sys_exe):
        raise SecurityError(f"Command executable {basename} is not an allowed Python interpreter")

    # Open with O_NOFOLLOW without O_CLOEXEC so child process can inherit it
    try:
        exe_fd = os.open(real_exe, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as e:
        raise SecurityError(f"Failed to open real_exe securely: {e}") from e

    try:
        st = os.fstat(exe_fd)
        if not stat.S_ISREG(st.st_mode):
            raise SecurityError("Executable is not a regular file")
        if st.st_uid not in {os.getuid(), 0}:
            raise SecurityError(f"Untrusted owner {st.st_uid} on executable")
        if (st.st_mode & 0o022) != 0:
            raise SecurityError("Executable is group/world writable")

        expected_uids = {os.getuid(), 0}
        chain = [(real_exe, st.st_dev, st.st_ino)]
        curr = os.path.dirname(real_exe)
        while True:
            try:
                pst = os.stat(curr)
            except OSError as e:
                raise SecurityError(f"stat failed on parent {curr}: {e}") from e
            if pst.st_uid not in expected_uids:
                raise SecurityError(f"Untrusted owner on parent {curr}")
            if (pst.st_mode & 0o022) != 0:
                raise SecurityError(f"Parent directory {curr} is group/world writable")
            chain.append((curr, pst.st_dev, pst.st_ino))
            parent = os.path.dirname(curr)
            if parent == curr:
                break
            curr = parent

        venv_bin = os.path.dirname(real_exe)
        if not os.path.isdir(venv_bin):
            venv_bin = os.path.join(real_prefix, "bin")
        return exe_fd, venv_bin, chain
    except Exception:
        _safe_close(exe_fd)
        raise


def build_mcp_env(venv_bin: str, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Construct a clean execution environment for stdio subprocess."""
    if extra_env is not None:
        validate_mcp_env(extra_env)
    env = os.environ.copy()
    for k in list(env.keys()):
        upper_k = k.upper()
        if upper_k.startswith("LD_") or upper_k.startswith("DYLD_"):
            env.pop(k, None)
    current_path = env.get("PATH", "")
    if venv_bin not in current_path.split(os.pathsep):
        env["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"
    env["VIRTUAL_ENV"] = os.path.dirname(venv_bin)
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        for k, v in extra_env.items():
            if isinstance(k, str) and isinstance(v, str):
                env[k] = v
    return env


async def launch_stdio_mcp_subprocess(
    args: Sequence[str],
    env: dict[str, str] | None = None,
    command: str | None = None,
    project_root: str | None = None,
    config_dir: str | None = None,
) -> asyncio.subprocess.Process:
    """Launch stdio MCP subprocess."""
    norm_cmd, norm_args, norm_env, cwd = normalize_stdio_mcp_config(
        command=command,
        args=args,
        env=env,
        config_dir=config_dir or project_root,
    )
    exe_fd = -1
    try:
        exe_fd, _, _ = inspect_and_open_verified_exe(norm_cmd)
        proc = await asyncio.create_subprocess_exec(
            norm_cmd,
            *norm_args,
            env=norm_env,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        return proc
    finally:
        if exe_fd >= 0:
            _safe_close(exe_fd)

# ── 7. SSE Client & Anti-SSRF Protection ───────────────────────────────────

PROHIBITED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("ff00::/8"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::/128"),
]

PROHIBITED_METADATA_HOSTS = {
    "metadata.google.internal",
    "169.254.169.254",
    "instance-data",
}


async def validate_and_pin_sse_url(url: str) -> tuple[str, str, int]:
    """Validate URL scheme and resolve DNS asynchronously, rejecting private/loopback/metadata destinations."""
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise DashboardApiError("INVALID_PARAM", "URL scheme must be http or https")
    if not parsed.hostname or parsed.hostname.lower() in PROHIBITED_METADATA_HOSTS:
        raise DashboardApiError("INVALID_PARAM", "Target host is prohibited")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    loop = asyncio.get_running_loop()
    try:
        addrinfo = await loop.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise DashboardApiError("CONNECTION_FAILED", f"DNS resolution failed: {e}") from e

    if not addrinfo:
        raise DashboardApiError("CONNECTION_FAILED", f"No address records found for {parsed.hostname}")

    for item in addrinfo:
        ip_str = item[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as e:
            raise DashboardApiError("INVALID_PARAM", f"Invalid IP format: {ip_str}") from e

        if getattr(ip, "ipv4_mapped", None) is not None:
            ip = ip.ipv4_mapped

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise DashboardApiError(
                "INVALID_PARAM",
                f"Target host resolves to prohibited private or loopback address: {ip_str}",
            )

        for net in PROHIBITED_NETWORKS:
            if ip in net:
                raise DashboardApiError(
                    "INVALID_PARAM",
                    f"Target host resolves to prohibited private or loopback address: {ip_str}",
                )

    pinned_ip = addrinfo[0][4][0]
    return pinned_ip, parsed.hostname, port


class PinnedResolver(aiohttp.resolver.AbstractResolver):
    """Custom aiohttp DNS resolver pinning destination hostname directly to pinned_ip."""

    def __init__(self, pinned_ip: str, port: int):
        self._pinned_ip = pinned_ip
        self._port = port

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list[dict[str, Any]]:
        return [
            {
                "hostname": host,
                "host": self._pinned_ip,
                "port": port or self._port,
                "family": socket.AF_INET if ":" not in self._pinned_ip else socket.AF_INET6,
                "proto": socket.IPPROTO_TCP,
                "flags": socket.AI_NUMERICHOST,
            }
        ]

    async def close(self) -> None:
        pass

# ── 8. Secret Masking & Secret Retention ───────────────────────────────────

SECRET_KEY_KEYWORDS = ("KEY", "TOKEN", "SECRET", "AUTH", "PASS", "PWD", "CREDENTIAL")


def is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(kw in upper for kw in SECRET_KEY_KEYWORDS)


def _mask_sensitive_dict(data: Any) -> Any:
    """Recursively mask dictionary keys matching sensitive keywords across arbitrary dicts and lists."""
    if isinstance(data, dict):
        return {
            k: ("******" if (isinstance(k, str) and is_secret_key(k)) else _mask_sensitive_dict(v))
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [_mask_sensitive_dict(item) for item in data]
    return data


def sanitize_mcp_url(url: str) -> str:
    """Mask credentials in URL userinfo and query parameters."""
    if not isinstance(url, str) or not url:
        return url
    try:
        parsed = urlsplit(url)
    except Exception:
        return url

    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, host_port = netloc.rsplit("@", 1)
        if ":" in userinfo:
            user, _ = userinfo.split(":", 1)
            userinfo = f"{user}:******"
        else:
            userinfo = "******"
        netloc = f"{userinfo}@{host_port}"

    new_query = parsed.query
    if parsed.query:
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        masked_pairs = []
        for k, v in pairs:
            if is_secret_key(k) and v:
                masked_pairs.append((k, "******"))
            else:
                masked_pairs.append((k, v))
        new_query = urlencode(masked_pairs, safe="*")

    return urlunsplit((parsed.scheme, netloc, parsed.path, new_query, parsed.fragment))


def _expand_for_validation(val: Any) -> Any:
    """Expand ${VAR} / ${VAR:-default} for validation, replacing unset vars with safe placeholder."""
    from ovs_agent.config import _ENV_RE
    if isinstance(val, str):
        def repl(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            env_val = os.environ.get(var)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            return f"__ENV_{var}__"
        return _ENV_RE.sub(repl, val)
    if isinstance(val, dict):
        return {k: _expand_for_validation(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_expand_for_validation(v) for v in val]
    return val


def _expand_url_for_validation(url: str) -> str:
    """Expand ${VAR} / ${VAR:-default} for SSE URL validation, using safe fallback if unset."""
    from ovs_agent.config import _ENV_RE
    if not isinstance(url, str):
        return url

    def repl(m: re.Match[str]) -> str:
        var, default = m.group(1), m.group(2)
        env_val = os.environ.get(var)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        return "127.0.0.1:8000"

    expanded = _ENV_RE.sub(repl, url)
    if not (expanded.startswith("http://") or expanded.startswith("https://")):
        expanded = f"http://{expanded}"
    return expanded


def sanitize_error_text(err: str, server_cfg: dict[str, Any] | None = None) -> str:
    """Redact sensitive URLs, credentials, and token strings from error text."""
    if not isinstance(err, str) or not err:
        return err

    # 1. Mask known secrets from server_cfg if provided
    if server_cfg and isinstance(server_cfg, dict):
        env = server_cfg.get("env")
        if isinstance(env, dict):
            for k, v in env.items():
                if isinstance(v, str) and len(v) > 3 and v != "******" and is_secret_key(k):
                    err = err.replace(v, "******")
        headers = server_cfg.get("headers")
        if isinstance(headers, dict):
            for k, v in headers.items():
                if isinstance(v, str) and len(v) > 3 and v != "******":
                    err = err.replace(v, "******")

    # 2. Redact URLs in error text
    def url_repl(m: re.Match[str]) -> str:
        raw_url = m.group(0)
        return sanitize_mcp_url(raw_url)

    err = re.sub(r"https?://[^\s\"'<>]+", url_repl, err)

    # 3. Redact common header/query credential patterns
    err = re.sub(r"(bearer\s+)[A-Za-z0-9_\-\.]{8,}", r"\1******", err, flags=re.IGNORECASE)
    err = re.sub(r"((?:api[_-]?key|token|secret|password|passwd|auth)=)[^&\s\"'<>]+", r"\1******", err, flags=re.IGNORECASE)
    return err


def sanitize_server_config(server: dict[str, Any]) -> dict[str, Any]:
    """Sanitize server configuration for client-facing GET responses."""
    if not isinstance(server, dict):
        return {"name": str(server)}
    s = copy.deepcopy(server)
    if "url" in s and isinstance(s["url"], str):
        s["url"] = sanitize_mcp_url(s["url"])
    if "env" in s and isinstance(s["env"], dict):
        s["env"] = _mask_sensitive_dict(s["env"])
    if "headers" in s and isinstance(s["headers"], dict):
        s["has_headers"] = bool(s["headers"])
        s["configured_header_keys"] = list(s["headers"].keys())
        del s["headers"]
    for k, v in list(s.items()):
        if k not in ("url", "env", "has_headers", "configured_header_keys"):
            s[k] = _mask_sensitive_dict(v)
    return s


def _normalize_cmd_str(cmd: str | None) -> str | None:
    if not cmd:
        return None
    try:
        resolved = shutil.which(cmd)
        if resolved:
            return os.path.normpath(resolved)
        return os.path.normpath(cmd)
    except Exception:
        return cmd


def _normalize_args_for_comparison(args: Sequence[str] | None, config_dir: str | None = None) -> list[str] | None:
    if args is None:
        return None
    if not isinstance(args, (list, tuple)) or not args:
        return []
    res = list(args)
    if res and res[0] != "-m" and isinstance(res[0], str) and re.search(r"\.py$", res[0], re.I):
        base_dir = config_dir if config_dir else os.getcwd()
        abs_path = os.path.abspath(res[0]) if os.path.isabs(res[0]) else os.path.abspath(os.path.join(base_dir, res[0]))
        try:
            res[0] = os.path.realpath(abs_path)
        except Exception:
            res[0] = abs_path
    return res


def is_mcp_target_matched(
    new_cfg: dict[str, Any],
    existing_cfg: dict[str, Any] | None,
    config_dir: str | None = None,
) -> bool:
    """Check if new_cfg targets the same destination as existing_cfg."""
    if not existing_cfg or not isinstance(existing_cfg, dict):
        return False
    existing_transport = existing_cfg.get("transport", "stdio").strip().lower()
    req_transport = new_cfg.get("transport", existing_transport).strip().lower()
    if req_transport != existing_transport:
        return False
    if req_transport == "sse":
        req_url = new_cfg.get("url")
        existing_url = existing_cfg.get("url")
        if req_url is not None and existing_url is not None:
            if req_url != existing_url and req_url != sanitize_mcp_url(existing_url):
                return False
        return True
    if req_transport == "stdio":
        req_cmd = new_cfg.get("command")
        existing_cmd = existing_cfg.get("command")
        req_args = new_cfg.get("args")
        existing_args = existing_cfg.get("args")

        norm_req_cmd = _normalize_cmd_str(req_cmd)
        norm_exist_cmd = _normalize_cmd_str(existing_cmd)

        if norm_req_cmd is not None and norm_exist_cmd is not None and norm_req_cmd != norm_exist_cmd:
            return False

        norm_req_args = _normalize_args_for_comparison(req_args, config_dir=config_dir)
        norm_exist_args = _normalize_args_for_comparison(existing_args, config_dir=config_dir)

        if norm_req_args is not None and norm_exist_args is not None and norm_req_args != norm_exist_args:
            return False
        return True
    return False


def _check_no_masked_credentials(server_cfg: dict[str, Any]) -> None:
    """Raise DashboardApiError if any masked credential is present."""
    url = server_cfg.get("url")
    if isinstance(url, str) and "******" in url:
        raise DashboardApiError(
            "INVALID_PARAM",
            "Cannot retain masked credential '******' in URL when server target changes; please provide new URL with credentials",
        )
    env = server_cfg.get("env")
    if isinstance(env, dict):
        for k, v in env.items():
            if v == "******":
                raise DashboardApiError(
                    "INVALID_PARAM",
                    f"Cannot retain masked credential '******' for env '{k}' when server target changes; please provide new secret",
                )
    hdrs = server_cfg.get("headers")
    if isinstance(hdrs, dict):
        for k, v in hdrs.items():
            if v == "******":
                raise DashboardApiError(
                    "INVALID_PARAM",
                    f"Cannot retain masked credential '******' for header '{k}' when server target changes; please provide new secret",
                )


def merge_server_secrets(
    new_server: dict[str, Any],
    existing_server: dict[str, Any] | None,
    config_dir: str | None = None,
) -> dict[str, Any]:
    """Perform atomic secret retention when updating server configuration."""
    if not isinstance(new_server, dict):
        raise DashboardApiError("INVALID_PARAM", "new_server must be a dictionary")
    validate_mcp_env(new_server.get("env"))
    s = copy.deepcopy(new_server)
    if not existing_server or not is_mcp_target_matched(s, existing_server, config_dir=config_dir):
        _check_no_masked_credentials(s)
        return s

    # URL secret preservation
    new_url = s.get("url")
    if isinstance(new_url, str) and "******" in new_url and existing_server:
        exist_url = existing_server.get("url")
        if isinstance(exist_url, str):
            if new_url == sanitize_mcp_url(exist_url):
                s["url"] = exist_url
            else:
                raise DashboardApiError(
                    "INVALID_PARAM",
                    "URL contains masked credentials that do not match existing server URL",
                )

    # Merge env
    if "env" in s:
        new_env = s["env"]
        if not isinstance(new_env, dict):
            raise DashboardApiError("INVALID_PARAM", "Invalid secret value type: env must be dict")
        merged_env = dict(existing_server.get("env") or {})
        for k, v in new_env.items():
            if not isinstance(v, str):
                raise DashboardApiError(
                    "INVALID_PARAM",
                    f"Invalid secret value type for env key {k}: expected str, got {type(v).__name__}",
                )
            if v == "******":
                pass
            elif v == "":
                merged_env.pop(k, None)
            else:
                merged_env[k] = v
        validate_mcp_env(merged_env)
        s["env"] = merged_env
    elif "env" in existing_server:
        s["env"] = copy.deepcopy(existing_server["env"])
        validate_mcp_env(s["env"])

    # Merge headers
    if "headers" in s:
        new_hdrs = s["headers"]
        if not isinstance(new_hdrs, dict):
            raise DashboardApiError("INVALID_PARAM", "Invalid secret value type: headers must be dict")
        merged_hdrs = dict(existing_server.get("headers") or {})
        for k, v in new_hdrs.items():
            if not isinstance(v, str):
                raise DashboardApiError("INVALID_PARAM", f"Invalid secret value type for header {k}")
            if v == "******":
                pass
            elif v == "":
                merged_hdrs.pop(k, None)
            else:
                merged_hdrs[k] = v
        s["headers"] = merged_hdrs
    elif "headers" in existing_server:
        s["headers"] = copy.deepcopy(existing_server["headers"])

    return s

# ── 9. Sandboxed Test Execution ────────────────────────────────────────────

_PENDING_CLEANUP_TASKS: set[asyncio.Task] = set()
_PENDING_REAP_TASKS: set[asyncio.Task] = set()


async def _supervise_pending_reap(proc: asyncio.subprocess.Process, pid: int) -> None:
    """Supervise reap of process that did not exit within initial 1.0s wait."""
    try:
        await asyncio.wait_for(proc.wait(), timeout=10.0)
        logger.info("Background reap confirmed for MCP process %d", pid)
    except asyncio.TimeoutError:
        logger.error("MCP process %d failed to reap after background timeout 10.0s", pid)
    except Exception as e:
        logger.error("Error in background reap for MCP process %d: %s", pid, e)


async def _cleanup_process(
    proc: asyncio.subprocess.Process,
    pgid: int,
    drain_task: asyncio.Task | None,
) -> str:
    # 返回值: REAP_CONFIRMED | REAP_TIMEOUT_PENDING | REAP_FAILED
    # 1. 尝试优雅终止整个进程组 (SIGTERM)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    # 2. 组级存活判定：基于 os.killpg(pgid, 0) 等待最多 1.5s 优雅期
    deadline = asyncio.get_running_loop().time() + 1.5
    while asyncio.get_running_loop().time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            pass
        await asyncio.sleep(0.05)

    # 3. 升级强制击杀整个进程组 (SIGKILL)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    # 4. 回收父进程并显式标记状态，超时则转交后台监督
    reap_status = "REAP_CONFIRMED"
    try:
        await asyncio.wait_for(proc.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        reap_status = "REAP_TIMEOUT_PENDING"
        logger.warning(
            "MCP parent process %d failed to reap within 1.0s after SIGKILL; launching background supervisor",
            proc.pid,
        )
        reap_task = asyncio.create_task(_supervise_pending_reap(proc, proc.pid))
        _PENDING_REAP_TASKS.add(reap_task)

        def _on_reap_done(t: asyncio.Task) -> None:
            try:
                if not t.cancelled():
                    _ = t.exception()
            except Exception:
                pass
            finally:
                _PENDING_REAP_TASKS.discard(t)

        reap_task.add_done_callback(_on_reap_done)
    except Exception as e:
        reap_status = "REAP_FAILED"
        logger.warning("Error waiting for MCP process %d to exit: %s", proc.pid, e)

    # 5. 清理 stderr 排水任务与管道
    if drain_task:
        if not drain_task.done():
            drain_task.cancel()
            try:
                await asyncio.wait_for(drain_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        try:
            if drain_task.done() and not drain_task.cancelled():
                _ = drain_task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    return reap_status


async def test_stdio_mcp_server(
    args: Sequence[str],
    env: dict[str, str] | None = None,
    timeout: float = 10.0,
    command: str | None = None,
    project_root: str | None = None,
    config_dir: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Execute stdio discovery in an isolated process group with clean killpg teardown."""
    proc = None
    pgid = None
    drain_task = None
    stderr_tail = bytearray()
    drain_error: Exception | None = None
    first_cancel: asyncio.CancelledError | None = None

    async def _drain_stderr(stream: asyncio.StreamReader) -> None:
        nonlocal stderr_tail, drain_error
        try:
            while True:
                chunk = await stream.read(1024)
                if not chunk:
                    break
                stderr_tail.extend(chunk)
                if len(stderr_tail) > 4096:
                    del stderr_tail[:-4096]
        except asyncio.CancelledError:
            raise
        except Exception as e:
            drain_error = e
            logger.warning("MCP stderr drain encountered error: %s", e)
            raise

    try:
        proc = await launch_stdio_mcp_subprocess(
            args=args,
            env=env,
            command=command,
            config_dir=config_dir or project_root,
        )
        pgid = proc.pid
        if proc.stderr is not None:
            drain_task = asyncio.create_task(_drain_stderr(proc.stderr))

        async def _query_tools() -> list[dict[str, Any]]:
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ovs-mcp-test", "version": "1.0"},
                },
            }
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write((json.dumps(init_req) + "\n").encode("utf-8"))
            await proc.stdin.drain()

            async def _read_jsonrpc_response(expected_id: int, phase: str) -> dict[str, Any]:
                line_count = 0
                while line_count < 100:
                    line_count += 1
                    line_bytes = await proc.stdout.readline()
                    if not line_bytes:
                        if drain_error is not None:
                            raise DashboardApiError("CONNECTION_FAILED", f"MCP stderr drain error: {drain_error}")
                        raw_err = stderr_tail.decode("utf-8", errors="replace").strip()
                        if raw_err:
                            redacted = redact_sensitive_text(raw_err)
                            raise DashboardApiError("CONNECTION_FAILED", f"Empty response from MCP server; stderr: {redacted}")
                        raise DashboardApiError("CONNECTION_FAILED", f"Empty response from MCP server during {phase}")
                    if drain_error is not None:
                        raise DashboardApiError("CONNECTION_FAILED", f"MCP stderr drain error: {drain_error}")
                    line_str = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    try:
                        msg = json.loads(line_str)
                    except Exception as e:
                        raise DashboardApiError("CONNECTION_FAILED", f"Invalid JSON during {phase}: {e}") from e
                    if isinstance(msg, dict) and msg.get("id") == expected_id:
                        return msg
                raise DashboardApiError(
                    "CONNECTION_FAILED",
                    f"MCP server did not return response for request id {expected_id} during {phase}",
                )

            init_resp = await _read_jsonrpc_response(1, "initialize")
            if "error" in init_resp:
                raise DashboardApiError("CONNECTION_FAILED", f"MCP initialize error: {init_resp['error']}")

            notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            proc.stdin.write((json.dumps(notif) + "\n").encode("utf-8"))
            await proc.stdin.drain()

            list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            proc.stdin.write((json.dumps(list_req) + "\n").encode("utf-8"))
            await proc.stdin.drain()

            list_resp = await _read_jsonrpc_response(2, "tools/list")
            if "error" in list_resp:
                raise DashboardApiError("CONNECTION_FAILED", f"MCP tools/list error: {list_resp['error']}")

            raw_tools = list_resp.get("result", {}).get("tools", [])
            tools = []
            for t in raw_tools:
                tools.append(
                    {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                    }
                )
            return tools

        query_task = asyncio.create_task(_query_tools())
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        try:
            wait_tasks = [query_task]
            if drain_task and not drain_task.done():
                wait_tasks.append(drain_task)
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                if not query_task.done():
                    query_task.cancel()
                raise DashboardApiError("TEST_TIMEOUT", "MCP server test connection timed out")

            if drain_task and drain_task in done:
                drain_exc = drain_task.exception() if not drain_task.cancelled() else None
                if drain_exc or drain_error:
                    if not query_task.done():
                        query_task.cancel()
                    err = drain_exc or drain_error
                    raise DashboardApiError("CONNECTION_FAILED", f"MCP stderr drain failed: {err}")

            if query_task not in done:
                remaining = timeout - (loop.time() - start_time)
                if remaining <= 0:
                    if not query_task.done():
                        query_task.cancel()
                    raise DashboardApiError("TEST_TIMEOUT", "MCP server test connection timed out")
                done2, _ = await asyncio.wait([query_task], timeout=remaining)
                if not done2:
                    if not query_task.done():
                        query_task.cancel()
                    raise DashboardApiError("TEST_TIMEOUT", "MCP server test connection timed out")

            if query_task.cancelled():
                raise asyncio.CancelledError()
            q_exc = query_task.exception()
            if q_exc:
                if isinstance(q_exc, DashboardApiError):
                    raise q_exc
                raise DashboardApiError("CONNECTION_FAILED", f"MCP test connection failed: {q_exc}") from q_exc
            return query_task.result()
        except asyncio.CancelledError as ce:
            if not query_task.done():
                query_task.cancel()
            if first_cancel is None:
                first_cancel = ce
            raise
    except asyncio.CancelledError as ce:
        if first_cancel is None:
            first_cancel = ce
        raise
    finally:
        if proc is not None and pgid is not None:
            def _on_background_cleanup_done(t: asyncio.Task) -> None:
                """Explicitly observe result/exception to prevent unretrieved task exception warnings."""
                try:
                    if not t.cancelled():
                        exc = t.exception()
                        if exc:
                            logger.error("Background MCP cleanup task failed: %s", exc)
                        else:
                            res = t.result()
                            logger.info("Background MCP cleanup completed with status: %s", res)
                except Exception as e:
                    logger.error("Error inspecting background cleanup task completion: %s", e)
                finally:
                    _PENDING_CLEANUP_TASKS.discard(t)

            cleanup_task = asyncio.create_task(_cleanup_process(proc, pgid, drain_task))
            loop = asyncio.get_running_loop()
            cleanup_deadline = loop.time() + 3.5
            cleanup_exc: Exception | None = None
            cleanup_status: str | None = None

            while not cleanup_task.done():
                remaining = cleanup_deadline - loop.time()
                if remaining <= 0:
                    logger.error(
                        "MCP cleanup task exceeded budget 3.5s; status=CLEANUP_PENDING_TIMEOUT; transferring to background supervisor"
                    )
                    _PENDING_CLEANUP_TASKS.add(cleanup_task)
                    cleanup_task.add_done_callback(_on_background_cleanup_done)
                    break
                try:
                    cleanup_status = await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=remaining)
                except asyncio.CancelledError as ce:
                    if first_cancel is None:
                        first_cancel = ce  # 锁定首个原始取消异常
                except asyncio.TimeoutError:
                    logger.error(
                        "MCP cleanup task timed out against budget; status=CLEANUP_PENDING_TIMEOUT; transferring to background supervisor"
                    )
                    _PENDING_CLEANUP_TASKS.add(cleanup_task)
                    cleanup_task.add_done_callback(_on_background_cleanup_done)
                    break
                except Exception as exc:
                    cleanup_exc = exc
                    logger.exception("MCP cleanup task raised exception: %s", exc)
                    break

            # 若前台正常等待完成，必须显式消费其异常或返回值
            if cleanup_task.done() and not cleanup_task.cancelled():
                if cleanup_exc is None:
                    cleanup_exc = cleanup_task.exception()
                if cleanup_exc is None and cleanup_status is None:
                    cleanup_status = cleanup_task.result()

            if first_cancel is not None:
                if cleanup_exc is not None:
                    first_cancel.__context__ = cleanup_exc
                raise first_cancel
            elif cleanup_exc is not None:
                raise cleanup_exc


async def test_sse_mcp_server(
    url: str, headers: dict[str, str] | None = None, timeout: float = 10.0
) -> list[dict[str, Any]]:
    """Test SSE MCP server with DNS pinning, no proxy, redirect blocking, and protocol handshake."""
    pinned_ip, _, port = await validate_and_pin_sse_url(url)
    resolver = PinnedResolver(pinned_ip, port)
    connector = aiohttp.TCPConnector(resolver=resolver)
    try:
        req_headers = dict(headers or {})
        req_headers.setdefault("Accept", "text/event-stream, application/json")
        async with aiohttp.ClientSession(connector=connector, trust_env=False) as session:
            try:
                async with session.get(
                    url,
                    headers=req_headers,
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status in (301, 302, 303, 307, 308):
                        raise DashboardApiError(
                            "CONNECTION_FAILED", f"Redirect not allowed on SSE endpoint (status {resp.status})"
                        )
                    if resp.status >= 400:
                        raise DashboardApiError("CONNECTION_FAILED", f"SSE endpoint returned HTTP {resp.status}")

                    content_type = resp.headers.get("Content-Type", "")
                    if "text/event-stream" not in content_type:
                        raise DashboardApiError(
                            "CONNECTION_FAILED",
                            f"Invalid content-type for MCP SSE server: expected 'text/event-stream', got '{content_type}'",
                        )

                    # Handshake: Read lines until finding endpoint event
                    endpoint_path = None
                    current_event = None
                    line_count = 0
                    while line_count < 100:
                        line_count += 1
                        raw_line = await resp.content.readline()
                        if not raw_line:
                            break
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if line.startswith("event:"):
                            current_event = line.split(":", 1)[1].strip()
                        elif line.startswith("data:") and current_event == "endpoint":
                            endpoint_path = line.split(":", 1)[1].strip()
                            break
                        elif not line:
                            current_event = None

                    if not endpoint_path:
                        raise DashboardApiError(
                            "CONNECTION_FAILED",
                            "SSE connection established but server did not emit required MCP 'endpoint' event",
                        )

                    # Validate post target and enforce same-origin to prevent credential exfiltration
                    post_url = urljoin(url, endpoint_path)
                    orig_parts = urlsplit(url)
                    post_parts = urlsplit(post_url)
                    orig_port = orig_parts.port or (443 if orig_parts.scheme == "https" else 80)
                    post_port = post_parts.port or (443 if post_parts.scheme == "https" else 80)
                    if (
                        orig_parts.scheme != post_parts.scheme
                        or (orig_parts.hostname or "").lower() != (post_parts.hostname or "").lower()
                        or orig_port != post_port
                    ):
                        raise DashboardApiError(
                            "CONNECTION_FAILED",
                            f"SSE message endpoint '{post_url}' is cross-origin; cross-origin targets are prohibited",
                        )
                    await validate_and_pin_sse_url(post_url)

                    async def _read_sse_rpc_response(expected_id: int) -> dict[str, Any]:
                        line_count = 0
                        data_lines: list[str] = []
                        while line_count < 200:
                            line_count += 1
                            raw_line = await resp.content.readline()
                            if not raw_line:
                                break
                            line = raw_line.decode("utf-8", errors="replace").strip()
                            if line.startswith("data:"):
                                data_lines.append(line.split(":", 1)[1].strip())
                                combined = "\n".join(data_lines)
                                try:
                                    msg = json.loads(combined)
                                    if isinstance(msg, dict) and msg.get("id") == expected_id:
                                        return msg
                                except Exception:
                                    pass
                            elif not line and data_lines:
                                try:
                                    msg = json.loads("\n".join(data_lines))
                                    if isinstance(msg, dict) and msg.get("id") == expected_id:
                                        return msg
                                except Exception:
                                    pass
                                data_lines = []
                        raise DashboardApiError(
                            "CONNECTION_FAILED",
                            f"MCP SSE server did not return response for request id {expected_id}",
                        )

                    # Initialize POST
                    init_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "openvoicestream", "version": "1.0"},
                        },
                    }
                    post_headers = dict(headers or {})
                    post_headers["Content-Type"] = "application/json"
                    async with session.post(
                        post_url,
                        json=init_payload,
                        headers=post_headers,
                        allow_redirects=False,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as p_resp:
                        if p_resp.status in (301, 302, 303, 307, 308):
                            raise DashboardApiError(
                                "CONNECTION_FAILED",
                                f"Redirect not allowed on MCP message endpoint (status {p_resp.status})",
                            )
                        if p_resp.status >= 400:
                            raise DashboardApiError(
                                "CONNECTION_FAILED", f"MCP initialize POST failed with HTTP {p_resp.status}"
                            )

                    # Await initialize response on SSE stream
                    init_resp = await _read_sse_rpc_response(1)
                    if "error" in init_resp:
                        raise DashboardApiError(
                            "CONNECTION_FAILED", f"MCP initialize rejected: {init_resp['error']}"
                        )

                    # Send notifications/initialized
                    notify_payload = {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                        "params": {},
                    }
                    async with session.post(
                        post_url,
                        json=notify_payload,
                        headers=post_headers,
                        allow_redirects=False,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as p_resp:
                        if p_resp.status in (301, 302, 303, 307, 308):
                            raise DashboardApiError(
                                "CONNECTION_FAILED",
                                f"Redirect not allowed on MCP notifications endpoint (status {p_resp.status})",
                            )
                        if p_resp.status >= 400:
                            raise DashboardApiError(
                                "CONNECTION_FAILED",
                                f"MCP notifications/initialized POST failed with HTTP {p_resp.status}",
                            )

                    # Request tools/list
                    tools_payload = {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/list",
                        "params": {},
                    }
                    async with session.post(
                        post_url,
                        json=tools_payload,
                        headers=post_headers,
                        allow_redirects=False,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as p_resp:
                        if p_resp.status in (301, 302, 303, 307, 308):
                            raise DashboardApiError(
                                "CONNECTION_FAILED",
                                f"Redirect not allowed on MCP tools/list endpoint (status {p_resp.status})",
                            )
                        if p_resp.status >= 400:
                            raise DashboardApiError(
                                "CONNECTION_FAILED", f"MCP tools/list POST failed with HTTP {p_resp.status}"
                            )

                    # Read tools response from SSE stream
                    tools_resp = await _read_sse_rpc_response(2)
                    if "error" in tools_resp:
                        raise DashboardApiError(
                            "CONNECTION_FAILED", f"MCP tools/list rejected: {tools_resp['error']}"
                        )
                    raw_tools = tools_resp.get("result", {}).get("tools", [])
                    tools = [
                        {"name": t.get("name", ""), "description": t.get("description", "")}
                        for t in raw_tools
                        if isinstance(t, dict)
                    ]
                    return tools
            except asyncio.TimeoutError:
                raise DashboardApiError("TEST_TIMEOUT", "MCP server test connection timed out")
            except DashboardApiError:
                raise
            except Exception as e:
                raise DashboardApiError("CONNECTION_FAILED", f"Failed to connect to SSE MCP server: {e}") from e
    finally:
        await connector.close()

# ── 10. McpServerManager: Dual-Log Atomic Persistence ──────────────────────

class McpServerManager:
    """Manages active vs persisted MCP servers, two-phase atomic persistence, and request authorization."""

    def __init__(
        self,
        config_path: str | Path,
        active_servers: list[dict[str, Any]] | None = None,
        admin_token: str | None = None,
    ):
        self.config_path = str(Path(config_path).resolve())
        self.config_dir = os.path.dirname(self.config_path)
        self._admin_token = admin_token or secrets.token_hex(32)
        if isinstance(active_servers, dict):
            self._active_mcp_servers = [{"name": k, **v} if isinstance(v, dict) else {"name": k} for k, v in active_servers.items()]
        elif isinstance(active_servers, list):
            self._active_mcp_servers = copy.deepcopy(active_servers)
        else:
            self._active_mcp_servers = []
        self._mcp_config_lock = asyncio.Lock()
        self._test_semaphore = asyncio.Semaphore(2)
        self._config_recovery_required = False

        # Load initial persisted state from disk
        self._persisted_mcp_servers: list[dict[str, Any]] = []
        self._persisted_file_sha256 = hashlib.sha256(b"").hexdigest()
        self._persisted_config_etag = self.calculate_etag([])

        if os.path.exists(self.config_path):
            try:
                fd = _safe_open_readonly(self.config_path)
                try:
                    raw_bytes = _read_all_bounded(fd, MAX_CONFIG_SIZE)
                    self._persisted_mcp_servers = self.parse_config(raw_bytes)
                    self._persisted_file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
                    self._persisted_config_etag = self.calculate_etag(self._persisted_mcp_servers)
                finally:
                    _safe_close(fd)
            except Exception as e:
                logger.warning("Could not read initial config %s: %s", self.config_path, e)
                self._persisted_mcp_servers = copy.deepcopy(self._active_mcp_servers)
                self._persisted_config_etag = self.calculate_etag(self._persisted_mcp_servers)
        else:
            self._persisted_mcp_servers = copy.deepcopy(self._active_mcp_servers)
            self._persisted_config_etag = self.calculate_etag(self._persisted_mcp_servers)

    @property
    def admin_token(self) -> str:
        return self._admin_token

    @property
    def active_servers(self) -> list[dict[str, Any]]:
        return self._active_mcp_servers

    @property
    def persisted_servers(self) -> list[dict[str, Any]]:
        return self._persisted_mcp_servers

    @property
    def persisted_config_etag(self) -> str:
        return self._persisted_config_etag

    @property
    def persisted_file_sha256(self) -> str:
        return self._persisted_file_sha256

    @property
    def config_recovery_required(self) -> bool:
        return self._config_recovery_required

    @property
    def pending_restart(self) -> bool:
        from ovs_agent.config import _expand_env
        expanded_persisted = [_expand_env(copy.deepcopy(s)) for s in self._persisted_mcp_servers]
        return self._active_mcp_servers != expanded_persisted

    @property
    def active_servers_sanitized(self) -> list[dict[str, Any]]:
        return [sanitize_server_config(s) for s in self._active_mcp_servers]

    @property
    def persisted_servers_sanitized(self) -> list[dict[str, Any]]:
        return [sanitize_server_config(s) for s in self._persisted_mcp_servers]

    @staticmethod
    def calculate_etag(servers: list[dict[str, Any]]) -> str:
        """Calculate canonical semantic ETag (RFC 7232)."""
        canonical_json = json.dumps(servers, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @staticmethod
    def calculate_file_sha256(raw_bytes: bytes) -> str:
        """Calculate binary file hash."""
        return hashlib.sha256(raw_bytes).hexdigest()

    @staticmethod
    def parse_config(raw_bytes: bytes) -> list[dict[str, Any]]:
        """Parse configuration bytes from YAML or JSON format."""
        if not raw_bytes.strip():
            return []
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ConfigPersistenceError("CONFIG_PARSE_ERROR", f"Config file is not valid UTF-8: {e}") from e
        try:
            doc = yaml.safe_load(text)
        except Exception as e:
            raise ConfigPersistenceError("CONFIG_PARSE_ERROR", f"YAML/JSON parse failed: {e}") from e
        if doc is None:
            return []
        if isinstance(doc, list):
            return doc
        if isinstance(doc, dict):
            mcp_val = doc.get("mcp_servers")
            if isinstance(mcp_val, dict):
                return [{"name": k, **v} if isinstance(v, dict) else {"name": k} for k, v in mcp_val.items()]
            elif isinstance(mcp_val, list):
                return list(mcp_val)
            return []
        raise ConfigPersistenceError("CONFIG_PARSE_ERROR", f"Unexpected config root type: {type(doc).__name__}")

    def serialize_config(self, servers: list[dict[str, Any]], prior_raw_bytes: bytes) -> bytes:
        """Serialize configuration preserving other sections of the YAML file."""
        doc = {}
        if prior_raw_bytes.strip():
            try:
                loaded = yaml.safe_load(prior_raw_bytes.decode("utf-8"))
                if isinstance(loaded, dict):
                    doc = loaded
            except Exception:
                doc = {}
        doc["mcp_servers"] = servers
        out = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
        return out.encode("utf-8")

    def startup_check(self) -> None:
        """Run startup check and clean orphaned tmp files; locks system if marker exists."""
        try:
            cleanup_orphaned_tmp_files(self.config_dir)
        except ConfigPersistenceError as e:
            if e.code == "CONFIG_RECOVERY_REQUIRED":
                self._config_recovery_required = True
            raise

    def verify_admin_token(self, request: web.Request) -> None:
        """Verify X-Admin-Token header using constant-time comparison."""
        token = request.headers.get("X-Admin-Token", "")
        if not token or not secrets.compare_digest(token, self._admin_token):
            raise DashboardApiError("UNAUTHORIZED", "Missing or invalid admin token")

    def verify_csrf(self, request: web.Request) -> None:
        """Verify Origin and Referer headers against Host."""
        req_host = request.headers.get("Host", "")
        if request.method in ("POST", "DELETE", "PATCH", "PUT"):
            origin = request.headers.get("Origin")
            referer = request.headers.get("Referer")
            if not origin and not referer:
                raise DashboardApiError("FORBIDDEN", "CSRF check failed: missing Origin and Referer")
            if origin:
                origin_host = urlsplit(origin).netloc
                if origin_host != req_host:
                    raise DashboardApiError(
                        "FORBIDDEN", f"CSRF check failed: Origin host {origin_host} != Host {req_host}"
                    )
            if referer:
                referer_host = urlsplit(referer).netloc
                if referer_host != req_host:
                    raise DashboardApiError(
                        "FORBIDDEN", f"CSRF check failed: Referer host {referer_host} != Host {req_host}"
                    )
        else:  # GET / HEAD
            origin = request.headers.get("Origin")
            referer = request.headers.get("Referer")
            if origin:
                origin_host = urlsplit(origin).netloc
                if origin_host != req_host:
                    raise DashboardApiError(
                        "FORBIDDEN", f"CSRF check failed on GET: Origin host {origin_host} != Host {req_host}"
                    )
            if referer:
                referer_host = urlsplit(referer).netloc
                if referer_host != req_host:
                    raise DashboardApiError(
                        "FORBIDDEN", f"CSRF check failed on GET: Referer host {referer_host} != Host {req_host}"
                    )

    def verify_if_match(self, if_match_or_request: str | web.Request) -> None:
        """Verify If-Match ETag for optimistic concurrency control."""
        if hasattr(if_match_or_request, "headers"):
            raw = if_match_or_request.headers.get("If-Match", "")
        else:
            raw = str(if_match_or_request or "")
        clean = raw.strip('"')
        if not clean or clean != self._persisted_config_etag:
            raise DashboardApiError("ETAG_MISMATCH", "Configuration was modified concurrently, please refresh")

    async def commit_servers(self, new_servers: list[dict[str, Any]]) -> None:
        """Two-phase atomic commit with dual-hash verification and rollback."""
        async with self._mcp_config_lock:
            await self._commit_servers_locked(new_servers)

    async def _commit_servers_locked(self, new_servers: list[dict[str, Any]]) -> None:
        """Two-phase atomic commit with dual-hash verification and rollback (caller must hold _mcp_config_lock)."""
        if self._config_recovery_required or check_persistent_recovery_marker(self.config_dir):
            self._config_recovery_required = True
            raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "System locked into read-only")

        target_path = self.config_path
        dirpath = self.config_dir

        # Pre-flight size checks
        prior_raw_bytes = b""
        if os.path.exists(target_path):
            fd = -1
            try:
                fd = _safe_open_readonly(target_path)
                prior_raw_bytes = _read_all_bounded(fd, MAX_CONFIG_SIZE)
            finally:
                if fd >= 0:
                    _safe_close(fd)

        if len(prior_raw_bytes) > MAX_CONFIG_SIZE:
            raise ConfigPersistenceError("FILE_TOO_LARGE", "Existing file too large")

        # Check if on-disk MCP config was modified externally since last read
        on_disk_mcp_servers = self.parse_config(prior_raw_bytes)
        on_disk_etag = self.calculate_etag(on_disk_mcp_servers)
        if on_disk_etag != self._persisted_config_etag:
            # Refresh internal cached state to reflect external changes and reject stale write
            self._persisted_mcp_servers = on_disk_mcp_servers
            self._persisted_file_sha256 = hashlib.sha256(prior_raw_bytes).hexdigest()
            self._persisted_config_etag = on_disk_etag
            raise DashboardApiError(
                "ETAG_MISMATCH",
                "Configuration file was modified externally on disk, please refresh",
            )

        new_raw_bytes = self.serialize_config(new_servers, prior_raw_bytes)
        if len(new_raw_bytes) > MAX_CONFIG_SIZE:
            raise ConfigPersistenceError("FILE_TOO_LARGE", "New config payload too large")

        expected_prior_file_sha256 = hashlib.sha256(prior_raw_bytes).hexdigest()
        expected_prior_config_etag = self.calculate_etag(self.parse_config(prior_raw_bytes))

        # Phase 1: Atomic Backup Commit
        backup_tmp = f"{target_path}.backup.tmp.{uuid.uuid4().hex}"
        backup_path = f"{target_path}.backup"
        try:
            backup_tmp_fd = -1
            try:
                backup_tmp_fd = _open_atomic_tmp(backup_tmp)
                _write_all(backup_tmp_fd, prior_raw_bytes)
                _safe_fsync_file(backup_tmp_fd)
            finally:
                if backup_tmp_fd >= 0:
                    _safe_close(backup_tmp_fd, raise_on_error=True)

            _safe_replace(backup_tmp, backup_path)
            _safe_fsync_dir(dirpath)
        except Exception as exc:
            try:
                os.unlink(backup_tmp)
            except OSError:
                pass

            # Phase 1 failed before mutating target_path.
            # If target_path is untouched, do NOT enter recovery mode or lock configuration.
            target_intact = False
            if expected_prior_file_sha256 is not None and os.path.exists(target_path):
                t_fd = -1
                try:
                    t_fd = _safe_open_readonly(target_path)
                    cur_bytes = _read_all_bounded(t_fd, MAX_CONFIG_SIZE)
                    if hashlib.sha256(cur_bytes).hexdigest() == expected_prior_file_sha256:
                        target_intact = True
                except Exception:
                    pass
                finally:
                    if t_fd >= 0:
                        _safe_close(t_fd)
            elif expected_prior_file_sha256 is None and not os.path.exists(target_path):
                target_intact = True

            if target_intact:
                logger.error("Phase 1 backup creation failed: %s; target configuration is untouched", exc)
                raise

            # Only if target_path was damaged or untrusted do we check backup and enter recovery
            backup_valid = False
            if os.path.exists(backup_path):
                bfd = -1
                try:
                    bfd = _safe_open_readonly(backup_path)
                    b_bytes = _read_all_bounded(bfd, MAX_CONFIG_SIZE)
                    if hashlib.sha256(b_bytes).hexdigest() == expected_prior_file_sha256:
                        backup_valid = True
                except Exception:
                    backup_valid = False
                finally:
                    if bfd >= 0:
                        _safe_close(bfd)
            if not backup_valid:
                self._config_recovery_required = True
                try:
                    write_persistent_recovery_marker(dirpath, "phase1_backup_corrupted")
                except SystemExit:
                    raise
                except Exception as marker_err:
                    logger.critical("FATAL: Failed to write recovery marker: %s", marker_err)
                logger.critical("FATAL: Phase 1 backup corrupted, locked into read-only")
                raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Phase 1 backup corrupted")
            raise

        # Phase 2: Atomic Target Commit
        tmp_path = f"{target_path}.tmp.{uuid.uuid4().hex}"
        try:
            tmp_fd = -1
            try:
                tmp_fd = _open_atomic_tmp(tmp_path)
                _write_all(tmp_fd, new_raw_bytes)
                _safe_fsync_file(tmp_fd)
            finally:
                if tmp_fd >= 0:
                    _safe_close(tmp_fd, raise_on_error=True)

            _safe_replace(tmp_path, target_path)
            _safe_fsync_dir(dirpath)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            # Rollback / restore from backup
            try:
                restore_tmp = f"{target_path}.restore.tmp.{uuid.uuid4().hex}"
                dst_fd = -1
                try:
                    dst_fd = _open_atomic_tmp(restore_tmp)
                    src_fd = -1
                    try:
                        src_fd = _safe_open_readonly(backup_path)
                        st = os.fstat(src_fd)
                        if not stat.S_ISREG(st.st_mode):
                            raise ConfigPersistenceError("READ_FAILED", "Backup file is not a regular file")
                        if st.st_uid not in {os.getuid(), 0}:
                            raise SecurityError("Backup file owner untrusted")
                        if stat.S_IMODE(st.st_mode) != 0o600:
                            raise SecurityError(f"Backup file mode must be exactly 0600, got {oct(st.st_mode)}")
                        backup_bytes = _read_all_bounded(src_fd, MAX_CONFIG_SIZE)
                    finally:
                        if src_fd >= 0:
                            _safe_close(src_fd)

                    if hashlib.sha256(backup_bytes).hexdigest() != expected_prior_file_sha256:
                        raise ConfigPersistenceError("READ_FAILED", "Backup file binary hash corrupted")
                    _write_all(dst_fd, backup_bytes)
                    _safe_fsync_file(dst_fd)
                finally:
                    if dst_fd >= 0:
                        _safe_close(dst_fd, raise_on_error=True)

                _safe_replace(restore_tmp, target_path)
                _safe_fsync_dir(dirpath)

                restored_data = self.parse_config(backup_bytes)
                restored_etag = self.calculate_etag(restored_data)
                if restored_etag != expected_prior_config_etag:
                    raise ConfigPersistenceError("READ_FAILED", "Restored semantic config ETag mismatch")

                self._persisted_mcp_servers = restored_data
                self._persisted_config_etag = restored_etag
                self._persisted_file_sha256 = expected_prior_file_sha256
            except Exception as recovery_err:
                self._config_recovery_required = True
                write_persistent_recovery_marker(dirpath, "phase2_recovery_failed")
                logger.critical("FATAL: Config recovery failed, locked into read-only: %s", recovery_err)
                raise
            raise

        # Successful Phase 2: sync memory state
        self._persisted_mcp_servers = copy.deepcopy(new_servers)
        self._persisted_file_sha256 = hashlib.sha256(new_raw_bytes).hexdigest()
        self._persisted_config_etag = self.calculate_etag(new_servers)

    async def test_server(
        self, server_cfg: dict[str, Any], project_root: str | None = None
    ) -> dict[str, Any]:
        """Run connectivity and tool discovery test with semaphore concurrency control."""
        if not isinstance(server_cfg, dict):
            raise DashboardApiError("INVALID_PARAM", "server_cfg must be a dictionary")
        validate_mcp_env(server_cfg.get("env"))

        # Validate and bound timeout_s before semaphore acquisition
        raw_timeout = server_cfg.get("timeout_s", 10.0)
        try:
            timeout = float(raw_timeout)
        except (ValueError, TypeError):
            raise DashboardApiError("INVALID_PARAM", "timeout_s must be a valid number")
        if not math.isfinite(timeout) or timeout < 0.1 or timeout > 60.0:
            raise DashboardApiError("INVALID_PARAM", "timeout_s must be a finite number between 0.1 and 60.0 seconds")

        if self._test_semaphore._value <= 0:
            raise DashboardApiError("TOO_MANY_REQUESTS", "Concurrent MCP test limit reached")
        await self._test_semaphore.acquire()
        try:
            validate_mcp_env(server_cfg.get("env"))
            name = server_cfg.get("name")
            effective_cfg = copy.deepcopy(server_cfg)
            if name and isinstance(name, str):
                existing = next((s for s in self._persisted_mcp_servers if s.get("name") == name), None)
                if not existing:
                    existing = next((s for s in self._active_mcp_servers if s.get("name") == name), None)
                if existing:
                    target_matched = is_mcp_target_matched(server_cfg, existing, config_dir=self.config_dir)
                    if target_matched:
                        effective_cfg = merge_server_secrets(effective_cfg, existing, config_dir=self.config_dir)
                        if not effective_cfg.get("command") and existing.get("command"):
                            effective_cfg["command"] = existing["command"]
                        if not effective_cfg.get("transport") and existing.get("transport"):
                            effective_cfg["transport"] = existing["transport"]
                        if not effective_cfg.get("args") and existing.get("args"):
                            effective_cfg["args"] = copy.deepcopy(existing["args"])
                        if not effective_cfg.get("url") and existing.get("url"):
                            effective_cfg["url"] = existing["url"]
                    else:
                        logger.warning(
                            "MCP test target for '%s' was modified or does not match saved target; "
                            "persisted credentials will NOT be restored.",
                            name,
                        )

            # Resolve environment variable substitutions (${VAR} / ${VAR:-default})
            # for connection testing while leaving persisted configuration in raw template form
            from ovs_agent.config import _expand_env
            effective_cfg = _expand_env(effective_cfg)

            validate_mcp_env(effective_cfg.get("env"))
            transport = effective_cfg.get("transport", "stdio").strip().lower()
            timeout = float(effective_cfg.get("timeout_s", 10.0))
            if transport == "stdio":
                args = effective_cfg.get("args")
                if not args or not isinstance(args, (list, tuple)):
                    raise DashboardApiError("INVALID_PARAM", "stdio transport requires args")
                command = effective_cfg.get("command")
                test_kwargs: dict[str, Any] = {}
                if command and command != sys.executable:
                    if not isinstance(command, str) or not command.strip():
                        raise DashboardApiError("INVALID_PARAM", "command must be a non-empty string if provided")
                    try:
                        fd, _, _ = inspect_and_open_verified_exe(command)
                        _safe_close(fd)
                    except DashboardApiError:
                        raise
                    except Exception as e:
                        raise SecurityError(f"Command verification failed for {command}: {e}")
                    test_kwargs["command"] = command
                test_kwargs["project_root"] = project_root or self.config_dir

                env = effective_cfg.get("env")
                validate_mcp_env(env)
                validate_stdio_args(args, project_root=test_kwargs["project_root"], env=env)
                tools = await test_stdio_mcp_server(
                    args=args, env=env, timeout=timeout, **test_kwargs
                )
            elif transport == "sse":
                url = effective_cfg.get("url")
                if not url:
                    raise DashboardApiError("INVALID_PARAM", "sse transport requires url")
                if "******" in url and existing and existing.get("url"):
                    if url == sanitize_mcp_url(existing["url"]):
                        url = existing["url"]
                        effective_cfg["url"] = url
                    else:
                        raise DashboardApiError("INVALID_PARAM", "URL contains masked credentials that do not match existing server URL")
                headers = effective_cfg.get("headers")
                tools = await test_sse_mcp_server(url=url, headers=headers, timeout=timeout)
            else:
                raise DashboardApiError("INVALID_PARAM", f"Unsupported transport: {transport}")

            return {
                "ok": True,
                "tools_count": len(tools),
                "tools": tools,
            }
        finally:
            self._test_semaphore.release()

    async def add_or_update_server(
        self, server_data: dict[str, Any], if_match: str | None = None
    ) -> list[dict[str, Any]]:
        """Add new server or update existing server with secret retention."""
        if not isinstance(server_data, dict):
            raise DashboardApiError("INVALID_PARAM", "server_data must be a dictionary")

        name = server_data.get("name")
        if not isinstance(name, str) or not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", name):
            raise DashboardApiError("INVALID_PARAM", "Server name must be 1-64 alphanumeric characters, hyphens or underscores")

        transport = server_data.get("transport", "stdio").strip().lower()
        if transport not in ("stdio", "sse"):
            raise DashboardApiError("INVALID_PARAM", f"Unsupported transport: {transport}")

        validated_server = copy.deepcopy(server_data)
        validated_server["transport"] = transport

        if transport == "stdio":
            command = validated_server.get("command")
            args = validated_server.get("args")
            env = validated_server.get("env")
            expanded_cmd = _expand_for_validation(command)
            expanded_args = _expand_for_validation(args)
            expanded_env = _expand_for_validation(env)
            norm_cmd, norm_args, _, _ = normalize_stdio_mcp_config(
                command=expanded_cmd,
                args=expanded_args,
                env=expanded_env,
                config_dir=self.config_dir,
            )
            if command and "${" in command:
                validated_server["command"] = command
            else:
                validated_server["command"] = norm_cmd

            args_to_store = list(args) if isinstance(args, (list, tuple)) else []
            if args_to_store and norm_args and "${" not in str(args_to_store[0]):
                args_to_store[0] = norm_args[0]
            validated_server["args"] = args_to_store
        elif transport == "sse":
            validate_mcp_env(validated_server.get("env"))
            url = validated_server.get("url")
            if not url or not isinstance(url, str):
                raise DashboardApiError("INVALID_PARAM", "sse transport requires url string")
            existing = next((s for s in self._persisted_mcp_servers if s.get("name") == name), None)
            if existing and "******" in url and existing.get("url"):
                if url == sanitize_mcp_url(existing["url"]):
                    url = existing["url"]
                    validated_server["url"] = url
                else:
                    raise DashboardApiError("INVALID_PARAM", "URL contains masked credentials that do not match existing server URL")
            url_to_validate = _expand_url_for_validation(url)
            await validate_and_pin_sse_url(url_to_validate)

        async with self._mcp_config_lock:
            if if_match is not None:
                self.verify_if_match(if_match)

            new_servers = copy.deepcopy(self._persisted_mcp_servers)
            idx = next((i for i, s in enumerate(new_servers) if s.get("name") == name), None)
            if idx is not None:
                existing = new_servers[idx]
                merged = merge_server_secrets(validated_server, existing, config_dir=self.config_dir)
                new_servers[idx] = merged
            else:
                merged = merge_server_secrets(validated_server, None, config_dir=self.config_dir)
                new_servers.append(merged)

            validate_mcp_env(merged.get("env"))
            await self._commit_servers_locked(new_servers)
            return self._persisted_mcp_servers

    async def patch_server(
        self, name: str, patch_data: dict[str, Any], if_match: str | None = None
    ) -> list[dict[str, Any]]:
        """Patch an existing MCP server configuration by name with secret retention."""
        if not isinstance(patch_data, dict):
            raise DashboardApiError("INVALID_PARAM", "patch_data must be a dictionary")

        if not isinstance(name, str) or not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", name):
            raise DashboardApiError("INVALID_PARAM", "Invalid server name")

        async with self._mcp_config_lock:
            if if_match is not None:
                self.verify_if_match(if_match)

            new_servers = copy.deepcopy(self._persisted_mcp_servers)
            idx = next((i for i, s in enumerate(new_servers) if s.get("name") == name), None)
            if idx is None:
                raise DashboardApiError("INVALID_PARAM", f"Server {name} not found")

            existing = new_servers[idx]
            target_name = patch_data.get("name", name)
            if not isinstance(target_name, str) or not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", target_name):
                raise DashboardApiError("INVALID_PARAM", "Server name must be 1-64 alphanumeric characters, hyphens or underscores")

            if target_name != name and any(s.get("name") == target_name for s in new_servers):
                raise DashboardApiError("INVALID_PARAM", f"Server {target_name} already exists")

            transport = patch_data.get("transport", existing.get("transport", "stdio")).strip().lower()
            if transport not in ("stdio", "sse"):
                raise DashboardApiError("INVALID_PARAM", f"Unsupported transport: {transport}")

            merged_data = copy.deepcopy(existing)
            merged_data.update(patch_data)
            merged_data["name"] = target_name
            merged_data["transport"] = transport

            if transport == "stdio":
                command = merged_data.get("command")
                args = merged_data.get("args")
                env = merged_data.get("env")
                expanded_cmd = _expand_for_validation(command)
                expanded_args = _expand_for_validation(args)
                expanded_env = _expand_for_validation(env)
                norm_cmd, norm_args, _, _ = normalize_stdio_mcp_config(
                    command=expanded_cmd,
                    args=expanded_args,
                    env=expanded_env,
                    config_dir=self.config_dir,
                )
                if command and "${" in command:
                    merged_data["command"] = command
                else:
                    merged_data["command"] = norm_cmd

                args_to_store = list(args) if isinstance(args, (list, tuple)) else []
                if args_to_store and norm_args and "${" not in str(args_to_store[0]):
                    args_to_store[0] = norm_args[0]
                merged_data["args"] = args_to_store
            elif transport == "sse":
                validate_mcp_env(merged_data.get("env"))
                url = merged_data.get("url")
                if not url or not isinstance(url, str):
                    raise DashboardApiError("INVALID_PARAM", "sse transport requires url string")
                if "******" in url and existing and existing.get("url"):
                    if url == sanitize_mcp_url(existing["url"]):
                        url = existing["url"]
                        merged_data["url"] = url
                    else:
                        raise DashboardApiError("INVALID_PARAM", "URL contains masked credentials that do not match existing server URL")
                url_to_validate = _expand_url_for_validation(url)
                await validate_and_pin_sse_url(url_to_validate)

            if not is_mcp_target_matched(merged_data, existing, config_dir=self.config_dir):
                logger.warning(
                    "Target changed for server '%s'; dropping existing credentials", target_name
                )
                if "headers" not in patch_data:
                    merged_data.pop("headers", None)
                if "env" not in patch_data:
                    merged_data.pop("env", None)

            final_server = merge_server_secrets(merged_data, existing, config_dir=self.config_dir)
            validate_mcp_env(final_server.get("env"))
            new_servers[idx] = final_server

            await self._commit_servers_locked(new_servers)
            return self._persisted_mcp_servers

    async def delete_server(
        self, name: str, if_match: str | None = None
    ) -> list[dict[str, Any]]:
        """Delete named server from persisted configuration."""
        async with self._mcp_config_lock:
            if if_match is not None:
                self.verify_if_match(if_match)

            new_servers = [s for s in self._persisted_mcp_servers if s.get("name") != name]
            if len(new_servers) == len(self._persisted_mcp_servers):
                raise DashboardApiError("INVALID_PARAM", f"Server {name} not found")

            await self._commit_servers_locked(new_servers)
            return self._persisted_mcp_servers

# ── 11. Algebraic Restart Watchdog ─────────────────────────────────────────

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "idle": {"restarting"},
    "restarting": {"restart_failed", "idle"},
    "restart_failed": {"restarting"},
}


class RestartWatchdog:
    """State machine and algebraic watchdog governing graceful agent process restarts."""

    def __init__(self, exit_fn: Any = None):
        self._exit_fn = exit_fn
        self._restart_lock = asyncio.Lock()
        self._restart_generation = 0
        self._is_restarting = "idle"
        self._shutting_down = False
        self._tracked_tasks: set[asyncio.Task] = set()

    @property
    def is_restarting(self) -> str:
        return self._is_restarting

    @property
    def generation(self) -> int:
        return self._restart_generation

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    async def transition_state(self, gen: int, new_state: str, reason: str = "") -> bool:
        """Algebraic state transition with generation guard."""
        async with self._restart_lock:
            if self._restart_generation != gen:
                logger.warning(
                    "Ignoring stale state transition to %s (gen %d != %d, reason=%s)",
                    new_state,
                    gen,
                    self._restart_generation,
                    reason,
                )
                return False
            allowed = ALLOWED_TRANSITIONS.get(self._is_restarting, set())
            if new_state not in allowed:
                logger.error(
                    "Illegal restart state transition: %s -> %s (gen %d, reason=%s)",
                    self._is_restarting,
                    new_state,
                    gen,
                    reason,
                )
                return False
            logger.info(
                "Restart state transition: %s -> %s (gen %d, reason=%s)",
                self._is_restarting,
                new_state,
                gen,
                reason,
            )
            self._is_restarting = new_state
            return True

    async def trigger_restart(self, exit_fn: Any = None) -> None:
        """Schedule delayed graceful termination with watchdog checking process exit."""
        actual_exit_fn = exit_fn or self._exit_fn
        async with self._restart_lock:
            if self._shutting_down:
                raise DashboardApiError("SERVICE_UNAVAILABLE", "Agent is shutting down")
            if self._is_restarting not in ("idle", "restart_failed"):
                raise DashboardApiError("CONCURRENT_MODIFICATION", "Agent is already restarting")
            self._restart_generation += 1
            current_gen = self._restart_generation
            self._is_restarting = "restarting"

        async def _do_restart(gen: int):
            try:
                await asyncio.sleep(0.5)
                logger.warning("Agent restart requested; executing exit function...")
                if callable(actual_exit_fn):
                    try:
                        actual_exit_fn(0)
                    except TypeError:
                        actual_exit_fn()
                else:
                    try:
                        os.kill(os.getpid(), signal.SIGTERM)
                    except Exception:
                        os._exit(0)

                # Check process alive
                await asyncio.sleep(2.0)
                try:
                    os.kill(os.getpid(), 0)
                    is_alive = True
                except OSError:
                    is_alive = False
                if is_alive:
                    await self.transition_state(gen, "restart_failed", reason="process_alive_timeout")
                    logger.error("Agent process still alive after restart signal")
            except Exception as e:
                logger.error("restart execution failed: %s", e)
                await self.transition_state(gen, "restart_failed", reason=f"exception:{e}")

        coro = _do_restart(current_gen)
        try:
            shutdown_task = asyncio.create_task(coro, name="delayed-restart")
            self._tracked_tasks.add(shutdown_task)
            shutdown_task.add_done_callback(
                lambda t: self._tracked_tasks.discard(t) if hasattr(self, "_tracked_tasks") else None
            )

            def _on_shutdown_done(t: asyncio.Task):
                if getattr(self, "_shutting_down", False):
                    return
                if t.cancelled() or t.exception():
                    fail_task = asyncio.create_task(
                        self.transition_state(current_gen, "restart_failed", reason="task_done_error")
                    )
                    self._tracked_tasks.add(fail_task)
                    fail_task.add_done_callback(
                        lambda ft: self._tracked_tasks.discard(ft) if hasattr(self, "_tracked_tasks") else None
                    )

            shutdown_task.add_done_callback(_on_shutdown_done)
        except Exception as e:
            try:
                coro.close()
            except Exception:
                pass
            await self.transition_state(current_gen, "idle", reason=f"create_task_failed:{e}")
            raise DashboardApiError("INTERNAL_ERROR", f"Failed to schedule restart: {e}") from e

    async def shutdown(self) -> None:
        """Gracefully cancel and drain all tracked tasks during application shutdown."""
        async with self._restart_lock:
            self._shutting_down = True

        while self._tracked_tasks:
            current_batch = list(self._tracked_tasks)
            for task in current_batch:
                if not task.done():
                    task.cancel()
            if current_batch:
                await asyncio.gather(*current_batch, return_exceptions=True)
            for task in current_batch:
                self._tracked_tasks.discard(task)
