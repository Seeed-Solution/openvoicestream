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
from urllib.parse import urlsplit

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
    """Atomically replace src -> dst."""
    try:
        os.replace(src, dst)
    except OSError as e:
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
        if (
            any(upper_k.startswith(p) for p in FORBIDDEN_ENV_PREFIXES)
            or upper_k in FORBIDDEN_ENV_EXACT
        ):
            raise SecurityError(f"Forbidden environment variable key: {k}")

        if not isinstance(v, str):
            raise DashboardApiError(
                "INVALID_PARAM",
                f"env value for key '{k}' must be a string, got {type(v).__name__}",
            )


def validate_stdio_args(
    args: Sequence[str],
    project_root: str | None = None,
    env: dict[str, Any] | None = None,
) -> None:
    """Validate stdio args: disallow -c, shell metacharacters, enforcing Mode A or Mode B."""
    if env is not None:
        validate_mcp_env(env)
    if not isinstance(args, (list, tuple)) or not args:
        raise DashboardApiError("INVALID_PARAM", "stdio args must be a non-empty list of strings")
    for arg in args:
        if not isinstance(arg, str):
            raise DashboardApiError("INVALID_PARAM", "stdio args must contain only strings")
        if any(c in SHELL_METACHARS for c in arg):
            raise SecurityError(f"stdio args contain shell metacharacter: {arg}")
        if arg.startswith("-c"):
            raise SecurityError(f"stdio args contain prohibited -c flag or prefix: {arg}")

    # Mode A: Python standard module
    if args[0] == "-m":
        if len(args) < 2 or args[1] not in ALLOWED_MCP_MODULES:
            raise SecurityError(
                f"Module {args[1] if len(args) > 1 else ''} not in allowed MCP modules"
            )
        return

    # Mode B: Script file strictly within agent/mcp_servers/
    if args[0].startswith("-"):
        raise SecurityError(f"Prohibited flag or option in script mode: {args[0]}")
    if not args[0].endswith(".py"):
        raise SecurityError(f"stdio script must end with .py: {args[0]}")

    if project_root is not None:
        p = os.path.realpath(project_root)
        if os.path.basename(p) == "agent":
            allowed_dir = os.path.join(p, "mcp_servers")
        else:
            allowed_dir = os.path.join(p, "agent", "mcp_servers")
    else:
        env_root = os.environ.get("OVS_MCP_PROJECT_ROOT")
        if env_root:
            p = os.path.realpath(env_root)
            if os.path.basename(p) == "agent":
                allowed_dir = os.path.join(p, "mcp_servers")
            else:
                allowed_dir = os.path.join(p, "agent", "mcp_servers")
        else:
            agent_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
            allowed_dir = os.path.join(agent_dir, "mcp_servers")
    allowed_dir = os.path.realpath(allowed_dir)

    if os.path.isabs(args[0]):
        script_path = os.path.realpath(args[0])
    else:
        candidate = os.path.realpath(args[0])
        if os.path.isfile(candidate):
            script_path = candidate
        elif project_root and os.path.isfile(os.path.join(project_root, args[0])):
            script_path = os.path.realpath(os.path.join(project_root, args[0]))
        else:
            script_path = candidate

    if not os.path.isfile(script_path):
        raise SecurityError(f"Script file does not exist or is not a regular file: {script_path}")

    try:
        rel = os.path.relpath(script_path, allowed_dir)
        if rel.startswith("..") or rel == ".":
            raise SecurityError(
                f"Script path {script_path} is outside allowed directory {allowed_dir}"
            )
    except ValueError:
        raise SecurityError(
            f"Script path {script_path} is outside allowed directory {allowed_dir}"
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
) -> asyncio.subprocess.Process:
    """Launch stdio MCP subprocess via /proc/self/fd/<fd> preventing TOCTOU attacks."""
    validate_stdio_args(args, project_root=project_root, env=env)
    if env is not None:
        validate_mcp_env(env)
    exe_fd = -1
    try:
        exe_fd, venv_bin, _ = inspect_and_open_verified_exe(command)
        proc_exe = f"/proc/self/fd/{exe_fd}"
        mcp_env = build_mcp_env(venv_bin, env)
        proc = await asyncio.create_subprocess_exec(
            proc_exe,
            *args,
            env=mcp_env,
            pass_fds=(exe_fd,),
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


def sanitize_server_config(server: dict[str, Any]) -> dict[str, Any]:
    """Sanitize server configuration for client-facing GET responses."""
    if not isinstance(server, dict):
        return {"name": str(server)}
    s = copy.deepcopy(server)
    if "env" in s and isinstance(s["env"], dict):
        s["env"] = _mask_sensitive_dict(s["env"])
    if "headers" in s and isinstance(s["headers"], dict):
        s["has_headers"] = bool(s["headers"])
        s["configured_header_keys"] = list(s["headers"].keys())
        del s["headers"]
    for k, v in list(s.items()):
        if k not in ("env", "has_headers", "configured_header_keys"):
            s[k] = _mask_sensitive_dict(v)
    return s


def merge_server_secrets(
    new_server: dict[str, Any], existing_server: dict[str, Any] | None
) -> dict[str, Any]:
    """Perform atomic secret retention when updating server configuration."""
    if not isinstance(new_server, dict):
        raise DashboardApiError("INVALID_PARAM", "new_server must be a dictionary")
    validate_mcp_env(new_server.get("env"))
    s = copy.deepcopy(new_server)
    if not existing_server:
        return s

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

async def test_stdio_mcp_server(
    args: Sequence[str],
    env: dict[str, str] | None = None,
    timeout: float = 10.0,
    command: str | None = None,
    project_root: str | None = None,
) -> list[dict[str, Any]]:
    """Execute stdio discovery in an isolated process group with clean killpg teardown."""
    if env is not None:
        validate_mcp_env(env)
    proc = None
    pgid = None
    try:
        proc = await launch_stdio_mcp_subprocess(args, env, command=command, project_root=project_root)
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = None

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

            init_line = await proc.stdout.readline()
            if not init_line:
                raise DashboardApiError("CONNECTION_FAILED", "Empty response from MCP server during initialize")
            try:
                init_resp = json.loads(init_line.decode("utf-8"))
            except Exception as e:
                raise DashboardApiError("CONNECTION_FAILED", f"Invalid JSON during initialize: {e}") from e
            if "error" in init_resp:
                raise DashboardApiError("CONNECTION_FAILED", f"MCP initialize error: {init_resp['error']}")

            notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            proc.stdin.write((json.dumps(notif) + "\n").encode("utf-8"))
            await proc.stdin.drain()

            list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            proc.stdin.write((json.dumps(list_req) + "\n").encode("utf-8"))
            await proc.stdin.drain()

            list_line = await proc.stdout.readline()
            if not list_line:
                raise DashboardApiError("CONNECTION_FAILED", "Empty response from MCP server during tools/list")
            try:
                list_resp = json.loads(list_line.decode("utf-8"))
            except Exception as e:
                raise DashboardApiError("CONNECTION_FAILED", f"Invalid JSON during tools/list: {e}") from e
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

        try:
            tools = await asyncio.wait_for(_query_tools(), timeout=timeout)
            return tools
        except asyncio.TimeoutError:
            raise DashboardApiError("TEST_TIMEOUT", "MCP server test connection timed out")
        except DashboardApiError:
            raise
        except Exception as e:
            raise DashboardApiError("CONNECTION_FAILED", f"MCP test connection failed: {e}") from e
    finally:
        if proc is not None and pgid is not None:
            is_alive = (proc.returncode if hasattr(proc, "returncode") else proc.poll()) is None
            if is_alive:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                    await asyncio.wait_for(proc.wait(), timeout=0.5)
                except (ProcessLookupError, PermissionError, asyncio.TimeoutError):
                    pass
            is_alive = (proc.returncode if hasattr(proc, "returncode") else proc.poll()) is None
            if is_alive:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    await asyncio.wait_for(proc.wait(), timeout=0.5)
                except (ProcessLookupError, PermissionError, asyncio.TimeoutError):
                    pass


async def test_sse_mcp_server(
    url: str, headers: dict[str, str] | None = None, timeout: float = 10.0
) -> list[dict[str, Any]]:
    """Test SSE MCP server with DNS pinning, no proxy, and redirect blocking."""
    pinned_ip, _, port = await validate_and_pin_sse_url(url)
    resolver = PinnedResolver(pinned_ip, port)
    connector = aiohttp.TCPConnector(resolver=resolver)
    try:
        async with aiohttp.ClientSession(connector=connector, trust_env=False) as session:
            try:
                async with session.get(
                    url,
                    headers=headers,
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
                    if "application/json" in content_type:
                        data = await resp.json()
                        tools = data.get("tools", [])
                        return [
                            {"name": t.get("name", ""), "description": t.get("description", "")} for t in tools
                        ]
                    return []
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
        return self._active_mcp_servers != self._persisted_mcp_servers

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
        except Exception:
            try:
                os.unlink(backup_tmp)
            except OSError:
                pass

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
                    existing_transport = existing.get("transport", "stdio").strip().lower()
                    req_transport = server_cfg.get("transport", existing_transport).strip().lower()

                    target_matched = False
                    if req_transport == existing_transport:
                        if req_transport == "sse":
                            req_url = server_cfg.get("url")
                            existing_url = existing.get("url")
                            if req_url is None or req_url == existing_url:
                                target_matched = True
                        elif req_transport == "stdio":
                            req_cmd = server_cfg.get("command")
                            existing_cmd = existing.get("command")
                            req_args = server_cfg.get("args")
                            existing_args = existing.get("args")

                            cmd_matched = (req_cmd is None or req_cmd == existing_cmd)
                            args_matched = (req_args is None or req_args == existing_args)
                            if cmd_matched and args_matched:
                                target_matched = True

                    if target_matched:
                        effective_cfg = merge_server_secrets(effective_cfg, existing)
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
                if project_root:
                    test_kwargs["project_root"] = project_root

                env = effective_cfg.get("env")
                validate_mcp_env(env)
                validate_stdio_args(args, project_root=project_root, env=env)
                tools = await test_stdio_mcp_server(
                    args=args, env=env, timeout=timeout, **test_kwargs
                )
            elif transport == "sse":
                url = effective_cfg.get("url")
                if not url:
                    raise DashboardApiError("INVALID_PARAM", "sse transport requires url")
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
        validate_mcp_env(server_data.get("env"))

        name = server_data.get("name")
        if not isinstance(name, str) or not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", name):
            raise DashboardApiError("INVALID_PARAM", "Server name must be 1-64 alphanumeric characters, hyphens or underscores")

        transport = server_data.get("transport", "stdio").strip().lower()
        if transport not in ("stdio", "sse"):
            raise DashboardApiError("INVALID_PARAM", f"Unsupported transport: {transport}")

        validated_server = copy.deepcopy(server_data)
        validated_server["transport"] = transport
        validate_mcp_env(validated_server.get("env"))

        if transport == "stdio":
            command = validated_server.get("command")
            if not command:
                validated_server["command"] = sys.executable
            else:
                if not isinstance(command, str) or not command.strip():
                    raise DashboardApiError("INVALID_PARAM", "command must be a non-empty string if provided")
                try:
                    fd, _, _ = inspect_and_open_verified_exe(command)
                    _safe_close(fd)
                except DashboardApiError:
                    raise
                except Exception as e:
                    raise SecurityError(f"Command verification failed for {command}: {e}")

            args = validated_server.get("args")
            if not args or not isinstance(args, (list, tuple)):
                raise DashboardApiError("INVALID_PARAM", "stdio transport requires args list")
            validate_stdio_args(args, env=validated_server.get("env"))
        elif transport == "sse":
            url = validated_server.get("url")
            if not url or not isinstance(url, str):
                raise DashboardApiError("INVALID_PARAM", "sse transport requires url string")
            await validate_and_pin_sse_url(url)

        async with self._mcp_config_lock:
            if if_match is not None:
                self.verify_if_match(if_match)

            new_servers = copy.deepcopy(self._persisted_mcp_servers)
            idx = next((i for i, s in enumerate(new_servers) if s.get("name") == name), None)
            if idx is not None:
                existing = new_servers[idx]
                merged = merge_server_secrets(validated_server, existing)
                new_servers[idx] = merged
            else:
                merged = merge_server_secrets(validated_server, None)
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
        validate_mcp_env(patch_data.get("env"))

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
            validate_mcp_env(merged_data.get("env"))

            if transport == "stdio":
                command = merged_data.get("command")
                if not command:
                    merged_data["command"] = sys.executable
                else:
                    if not isinstance(command, str) or not command.strip():
                        raise DashboardApiError("INVALID_PARAM", "command must be a non-empty string if provided")
                    try:
                        fd, _, _ = inspect_and_open_verified_exe(command)
                        _safe_close(fd)
                    except DashboardApiError:
                        raise
                    except Exception as e:
                        raise SecurityError(f"Command verification failed for {command}: {e}")

                args = merged_data.get("args")
                if not args or not isinstance(args, (list, tuple)):
                    raise DashboardApiError("INVALID_PARAM", "stdio transport requires args list")
                validate_stdio_args(args, env=merged_data.get("env"))
            elif transport == "sse":
                url = merged_data.get("url")
                if not url or not isinstance(url, str):
                    raise DashboardApiError("INVALID_PARAM", "sse transport requires url string")
                await validate_and_pin_sse_url(url)

            final_server = merge_server_secrets(merged_data, existing)
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
