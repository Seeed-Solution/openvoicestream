"""Comprehensive test suite for MCP server configuration management, persistence, sandboxing, and watchdog.

Strictly covers scenarios T1 through T10 as specified in docs/specs/spec-dev-flow-mcp-and-frontend-v7.3.md:
- T1: Session Token & CSRF Validation (401/403 and request_id assertions)
- T2: Stdio Whitelist, /proc/self/fd Anti-TOCTOU, Parent Chain Validation, and FD Lifecycle
- T3: SSE Anti-SSRF, DNS Pinning, and Proxy / Redirect Blocking
- T4: Secret Masking and Credential Retention (Secret Merge)
- T5: Strong ETag / If-Match Optimistic Locking (409 Conflict)
- T6: Sandboxed Process Group Lifecycle and Clean Teardown
- T7: Concurrent Test Limit (asyncio.Semaphore(2) -> 429)
- T8: Dual-Hash Domains, 2MB Bounds, Phase 1/2 Atomic Rollback, and State Sync
- T9: Runtime State Separation and Algebraic Restart Watchdog
- T10: Platform POSIX Capabilities, Startup Markers, and Bijective Error Map Consistency
"""
from __future__ import annotations

import asyncio
import copy
import errno
import hashlib
import json
import os
import signal
import socket
import stat
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from aiohttp import web
import pytest
import yaml

from ovs_agent.config import Config
from ovs_agent.event_bus import EventBus
from ovs_agent.app_mode import AppMode, ModeManager
from ovs_agent.plugins.debug_dashboard import DebugDashboardPlugin
from ovs_agent.plugins.mcp_manager import (
    ALLOWED_MCP_MODULES,
    ALLOWED_TRANSITIONS,
    HTTP_STATUS_MAP,
    MAX_CONFIG_SIZE,
    SAFE_CLIENT_MESSAGES,
    VALID_HTTP_STATUSES,
    ConfigPersistenceError,
    DashboardApiError,
    McpServerManager,
    PinnedResolver,
    RestartWatchdog,
    SecurityError,
    _cleanup_process,
    _mask_sensitive_dict,
    _open_atomic_tmp,
    _read_all_bounded,
    _safe_close,
    _safe_fsync_dir,
    _safe_fsync_file,
    _safe_open_readonly,
    _safe_replace,
    _supervise_pending_reap,
    _write_all,
    _PENDING_CLEANUP_TASKS,
    _PENDING_REAP_TASKS,
    check_persistent_recovery_marker,
    cleanup_orphaned_tmp_files,
    fatal_config_recovery_exit,
    inspect_and_open_verified_exe,
    launch_stdio_mcp_subprocess,
    mcp_api_boundary,
    merge_server_secrets,
    normalize_stdio_mcp_config,
    redact_sensitive_text,
    sanitize_server_config,
    test_sse_mcp_server as run_sse_test,
    test_stdio_mcp_server as run_stdio_test,
    validate_and_pin_sse_url,
    validate_error_maps_consistency,
    validate_mcp_env,
    validate_platform_capabilities,
    validate_stdio_args,
    write_persistent_recovery_marker,
)


class _DummyChat(AppMode):
    name = "chat"
    display_name = "对话"
    icon = "💬"
    description = "test chat"

    async def on_user_utterance(self, ctx, text):
        return None


def _mk_test_app(port: int, config: Config):
    app = MagicMock()
    app.config = config
    app.events = EventBus()
    app.audio = SimpleNamespace(_in_queue=None, stop_playback=AsyncMock())
    app.slv = SimpleNamespace(
        _ws=None, reconnect=AsyncMock(), abort=AsyncMock(), send_text=AsyncMock()
    )
    app.session = SimpleNamespace(history=[])
    app.broadcast = AsyncMock()
    mgr = ModeManager(lambda: None)
    chat = _DummyChat()
    mgr.register(chat)
    mgr._current = chat
    app.modes = mgr
    app.plugins = []
    return app


@pytest.fixture
async def dashboard_env(unused_tcp_port, tmp_path):
    """Spins up a test DebugDashboardPlugin with an active McpServerManager."""
    config_file = tmp_path / "agent.yaml"
    initial_servers = [
        {
            "name": "srv1",
            "transport": "stdio",
            "args": ["-m", "ovs_agent.mcp.servers"],
            "env": {"API_KEY": "secret_abc", "NORMAL_VAR": "val1"},
        }
    ]
    initial_doc = {"mcp_servers": copy.deepcopy(initial_servers)}
    config_file.write_text(yaml.safe_dump(initial_doc), encoding="utf-8")

    cfg = Config(
        metadata={"dashboard_port": unused_tcp_port},
        mcp_servers=copy.deepcopy(initial_servers),
    )
    cfg._source_path = config_file
    app = _mk_test_app(unused_tcp_port, cfg)
    exit_calls = []
    app._restart_exit_fn = lambda code=0: exit_calls.append(code)

    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()

    base_url = f"http://127.0.0.1:{unused_tcp_port}"
    token = plugin._mcp_admin_token

    yield {
        "base_url": base_url,
        "token": token,
        "config_file": config_file,
        "tmp_path": tmp_path,
        "plugin": plugin,
        "app": app,
        "exit_calls": exit_calls,
        "host": f"127.0.0.1:{unused_tcp_port}",
    }

    await plugin.stop()


# ── T1: Session Token & CSRF Validation ────────────────────────────────────

@pytest.mark.asyncio
async def test_t1_session_token_and_csrf(dashboard_env):
    """T1: Missing/invalid token returns 401; forged Origin/Referer returns 403; all carry request_id."""
    base = dashboard_env["base_url"]
    token = dashboard_env["token"]
    host = dashboard_env["host"]

    async with aiohttp.ClientSession() as s:
        # 1. GET without X-Admin-Token -> 401
        resp = await s.get(f"{base}/api/mcp/servers")
        assert resp.status == 401
        data = await resp.json()
        assert data["ok"] is False
        assert data["code"] == "UNAUTHORIZED"
        assert "request_id" in data
        assert data["error"] == SAFE_CLIENT_MESSAGES["UNAUTHORIZED"]

        # 2. GET with wrong X-Admin-Token -> 401
        resp = await s.get(f"{base}/api/mcp/servers", headers={"X-Admin-Token": "bad_token"})
        assert resp.status == 401
        data = await resp.json()
        assert data["ok"] is False
        assert data["code"] == "UNAUTHORIZED"
        assert "request_id" in data

        # 3. GET with valid token but forged Referer -> 403
        resp = await s.get(
            f"{base}/api/mcp/servers",
            headers={"X-Admin-Token": token, "Referer": "http://evil.com/dashboard"},
        )
        assert resp.status == 403
        data = await resp.json()
        assert data["ok"] is False
        assert data["code"] == "FORBIDDEN"
        assert "request_id" in data

        # 4. GET with valid token and matching Referer -> 200
        resp = await s.get(
            f"{base}/api/mcp/servers",
            headers={"X-Admin-Token": token, "Referer": f"http://{host}/"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["ok"] is True

        # 5. GET with valid token and no Referer/Origin (direct API) -> 200
        resp = await s.get(f"{base}/api/mcp/servers", headers={"X-Admin-Token": token})
        assert resp.status == 200
        data = await resp.json()
        assert data["ok"] is True
        etag = data["etag"]

        # 6. POST without X-Admin-Token -> 401
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={"Origin": f"http://{host}", "If-Match": etag},
            json={"server": {"name": "test_post", "transport": "stdio", "args": ["-m", "ha_mcp"]}},
        )
        assert resp.status == 401
        data = await resp.json()
        assert data["ok"] is False
        assert data["code"] == "UNAUTHORIZED"
        assert "request_id" in data

        # 7. POST with valid token but forged Origin -> 403
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={"X-Admin-Token": token, "Origin": "http://attacker.com", "If-Match": etag},
            json={"server": {"name": "test_post", "transport": "stdio", "args": ["-m", "ha_mcp"]}},
        )
        assert resp.status == 403
        data = await resp.json()
        assert data["ok"] is False
        assert data["code"] == "FORBIDDEN"
        assert "request_id" in data

        # 8. POST with valid token but forged Referer -> 403
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={
                "X-Admin-Token": token,
                "Referer": "http://evil.com/page",
                "If-Match": etag,
            },
            json={"server": {"name": "test_post", "transport": "stdio", "args": ["-m", "ha_mcp"]}},
        )
        assert resp.status == 403
        data = await resp.json()
        assert data["code"] == "FORBIDDEN"


# ── T2: Stdio Whitelist, /proc/self/fd Anti-TOCTOU, Parent Chain ────────────

@pytest.mark.asyncio
async def test_t2_stdio_whitelist_and_proc_fd_exec(dashboard_env, tmp_path):
    """T2: Non-whitelisted args, -c, and shell metachars rejected (400); /proc/self/fd verified."""
    # 1. Parameter whitelist verification
    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["-c", "import os"])
    assert exc_info.value.code == "ILLEGAL_ARGUMENT"

    # S1: -cCODE joined form rejected
    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["-cimport os"])
    assert exc_info.value.code == "ILLEGAL_ARGUMENT"

    # Shell metacharacter takes precedence over -c prefix
    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["-cimport os; print(1)"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["-m", "ha_mcp", "-cbad"])
    assert exc_info.value.code == "ILLEGAL_ARGUMENT"

    # S1: -mMODULE joined form rejected
    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["-mha_mcp"])
    assert exc_info.value.code == "INVALID_SCRIPT_PATH"

    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["-m", "ha_mcp; rm -rf /"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["-m", "123invalid_module"])
    assert exc_info.value.code == "INVALID_MODULE_NAME"

    # S1: Non-existent script rejected
    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["/tmp/malicious_non_existent.py"])
    assert exc_info.value.code == "SCRIPT_NOT_FOUND"

    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["agent/mcp_servers/does_not_exist.py"])
    assert exc_info.value.code == "SCRIPT_NOT_FOUND"

    # S1: Non-.py script rejected
    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["agent/mcp_servers/test.sh"])
    assert exc_info.value.code == "INVALID_SCRIPT_PATH"

    # S1: Valid Mode B script passes
    mock_mcp_dir = tmp_path / "agent" / "mcp_servers"
    mock_mcp_dir.mkdir(parents=True, exist_ok=True)
    valid_script = mock_mcp_dir / "valid_server.py"
    valid_script.write_text("# valid script\n", encoding="utf-8")
    validate_stdio_args([str(valid_script)], project_root=str(tmp_path))

    # Valid Mode A args pass
    validate_stdio_args(["-m", "ovs_agent.mcp.servers"])
    validate_stdio_args(["-m", "ha_mcp"])
    validate_stdio_args(["-m", "custom_thirdparty_module"])

    # 2. inspect_and_open_verified_exe success
    exe_fd, venv_bin, chain = inspect_and_open_verified_exe()
    try:
        assert exe_fd >= 0
        assert os.path.isdir(venv_bin)
        assert len(chain) > 0
        st = os.fstat(exe_fd)
        assert stat.S_ISREG(st.st_mode)
    finally:
        _safe_close(exe_fd)

    # 3. Parent directory permission tampering raises SecurityError
    with patch("os.stat") as mock_stat:
        fake_st = os.stat(sys.executable)
        # World writable mode on parent
        tampered_st = os.stat_result((
            fake_st.st_mode | 0o002,
            fake_st.st_ino,
            fake_st.st_dev,
            fake_st.st_nlink,
            os.getuid(),
            fake_st.st_gid,
            fake_st.st_size,
            fake_st.st_atime,
            fake_st.st_mtime,
            fake_st.st_ctime,
        ))
        mock_stat.return_value = tampered_st
        with pytest.raises(SecurityError) as sec_exc:
            inspect_and_open_verified_exe()
        assert sec_exc.value.code == "SECURITY_VIOLATION"
        assert sec_exc.value.status == 400

    # 4. FD lifecycle: exe_fd unconditionally closed in finally
    opened_fds = []
    real_open = os.open

    def tracking_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened_fds.append(fd)
        return fd

    with patch("os.open", side_effect=tracking_open):
        proc = await launch_stdio_mcp_subprocess(["-m", "ovs_agent.mcp.servers"])
        assert len(opened_fds) > 0
        target_fd = opened_fds[-1]
        # In parent process, target_fd must already be closed by finally block
        with pytest.raises(OSError) as close_err:
            os.fstat(target_fd)
        assert close_err.value.errno == errno.EBADF

        # Terminate child
        if (proc.returncode if hasattr(proc, "returncode") else proc.poll()) is None:
            proc.terminate()
            await proc.wait()

    # 5. Stdio command validation: only trusted Python interpreter allowed
    with pytest.raises(SecurityError):
        inspect_and_open_verified_exe("/bin/bash")

    with pytest.raises(SecurityError):
        inspect_and_open_verified_exe("curl")

    # Current Python interpreter passes
    fd_ok, _, _ = inspect_and_open_verified_exe(sys.executable)
    _safe_close(fd_ok)

    # Empty command defaults to sys.executable and passes
    fd_ok2, _, _ = inspect_and_open_verified_exe("")
    _safe_close(fd_ok2)

    # Testing server with malicious command is blocked
    mgr = dashboard_env["plugin"]._mcp_manager
    with pytest.raises(SecurityError):
        await mgr.test_server({"transport": "stdio", "command": "/bin/bash", "args": ["-m", "ha_mcp"]})

    # Adding server with malicious command is blocked
    with pytest.raises(SecurityError):
        await mgr.add_or_update_server(
            {"name": "malicious", "transport": "stdio", "command": "/bin/sh", "args": ["-m", "ha_mcp"]}
        )


# ── T3: SSE Anti-SSRF, DNS Pinning, and Proxy / Redirect Blocking ──────────

@pytest.mark.asyncio
async def test_t3_sse_ssrf_and_proxy_blocking():
    """T3: Resolving to private/loopback/metadata blocked (400); DNS pinned; redirects refused."""
    # 1. Scheme checks
    with pytest.raises(DashboardApiError) as exc:
        await validate_and_pin_sse_url("ftp://example.com/sse")
    assert exc.value.code == "INVALID_PARAM"

    # 2. Private IP destinations
    private_urls = [
        "http://127.0.0.1:8080/sse",
        "http://10.0.0.1/sse",
        "http://172.16.0.1/sse",
        "http://192.168.1.1/sse",
        "http://169.254.169.254/sse",
        "http://metadata.google.internal/sse",
        "http://[::1]/sse",
        "http://[::ffff:127.0.0.1]/sse",
        "http://[::ffff:192.168.1.1]/sse",
    ]
    for u in private_urls:
        with pytest.raises(DashboardApiError) as exc:
            await validate_and_pin_sse_url(u)
        assert exc.value.code == "INVALID_PARAM"
        assert exc.value.status == 400

    # 3. PinnedResolver binds the first resolved IP
    with patch("socket.getaddrinfo") as mock_gai:
        mock_gai.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.35", 80)),
        ]
        pinned_ip, host, port = await validate_and_pin_sse_url("http://example.com/sse")
        assert pinned_ip == "93.184.216.34"
        assert host == "example.com"
        assert port == 80

        resolver = PinnedResolver(pinned_ip, port)
        resolved = await resolver.resolve("example.com")
        assert resolved[0]["host"] == "93.184.216.34"

    # 4. Redirect blocking
    app = web.Application()
    async def redir_handler(request):
        raise web.HTTPFound("/somewhere_else")
    app.router.add_get("/sse", redir_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assigned_port = site._server.sockets[0].getsockname()[1]

    try:
        # Bypass validate_and_pin_sse_url loopback check just to test redirect blocking
        with patch(
            "ovs_agent.plugins.mcp_manager.validate_and_pin_sse_url",
            return_value=("127.0.0.1", "127.0.0.1", assigned_port),
        ):
            with pytest.raises(DashboardApiError) as exc:
                await run_sse_test(f"http://127.0.0.1:{assigned_port}/sse")
            assert exc.value.code == "CONNECTION_FAILED"
            assert exc.value.status == 502
    finally:
        await runner.cleanup()


# ── T4: Secret Masking and Secret Retention (Secret Merge) ─────────────────

@pytest.mark.asyncio
async def test_t4_secret_masking_and_retention(dashboard_env):
    """T4: Secret keys desensitized to ****** on GET; retained on update; deleted on empty string."""
    base = dashboard_env["base_url"]
    token = dashboard_env["token"]
    host = dashboard_env["host"]

    async with aiohttp.ClientSession() as s:
        # 1. GET verifies API_KEY is masked to ******
        resp = await s.get(f"{base}/api/mcp/servers", headers={"X-Admin-Token": token})
        assert resp.status == 200
        data = await resp.json()
        servers = data["servers"]
        srv1 = next(srv for srv in servers if srv["name"] == "srv1")
        assert srv1["env"]["API_KEY"] == "******"
        assert srv1["env"]["NORMAL_VAR"] == "val1"
        etag = data["etag"]

        # 2. POST update sending ****** retains original secret_abc
        update_payload = {
            "server": {
                "name": "srv1",
                "transport": "stdio",
                "args": ["-m", "ovs_agent.mcp.servers"],
                "env": {"API_KEY": "******", "NORMAL_VAR": "val_updated"},
            }
        }
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": etag},
            json=update_payload,
        )
        assert resp.status == 200
        data = await resp.json()
        etag2 = data["etag"]

        # Verify underlying persisted state on disk holds secret_abc
        manager = dashboard_env["plugin"]._mcp_manager
        persisted_srv1 = next(srv for srv in manager.persisted_servers if srv["name"] == "srv1")
        assert persisted_srv1["env"]["API_KEY"] == "secret_abc"
        assert persisted_srv1["env"]["NORMAL_VAR"] == "val_updated"

        # 3. POST update sending empty string deletes API_KEY
        del_payload = {
            "server": {
                "name": "srv1",
                "transport": "stdio",
                "args": ["-m", "ovs_agent.mcp.servers"],
                "env": {"API_KEY": "", "NORMAL_VAR": "val_updated"},
            }
        }
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": etag2},
            json=del_payload,
        )
        assert resp.status == 200
        persisted_srv1 = next(srv for srv in manager.persisted_servers if srv["name"] == "srv1")
        assert "API_KEY" not in persisted_srv1["env"]

        # 4. Invalid secret value type raises 400
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={
                "X-Admin-Token": token,
                "Origin": f"http://{host}",
                "If-Match": manager.persisted_config_etag,
            },
            json={
                "server": {
                    "name": "srv1",
                    "transport": "stdio",
                    "args": ["-m", "ovs_agent.mcp.servers"],
                    "env": {"API_KEY": 12345},
                }
            },
        )
        assert resp.status == 400
        data = await resp.json()
        assert data["code"] == "INVALID_PARAM"


@pytest.mark.asyncio
async def test_t4_patch_and_desensitized_test_connection(dashboard_env):
    """T4 Extension: PATCH /api/mcp/servers/{name} with secret retention, and test connection with desensitized creds."""
    base = dashboard_env["base_url"]
    token = dashboard_env["token"]
    host = dashboard_env["host"]
    manager = dashboard_env["plugin"]._mcp_manager

    async with aiohttp.ClientSession() as s:
        # 1. GET current etag
        resp = await s.get(f"{base}/api/mcp/servers", headers={"X-Admin-Token": token})
        assert resp.status == 200
        data = await resp.json()
        etag = data["etag"]

        # 2. PATCH without X-Admin-Token -> 401
        resp = await s.patch(
            f"{base}/api/mcp/servers/srv1",
            headers={"Origin": f"http://{host}", "If-Match": etag},
            json={"server": {"timeout_s": 15.0}},
        )
        assert resp.status == 401

        # 3. PATCH with forged Origin -> 403
        resp = await s.patch(
            f"{base}/api/mcp/servers/srv1",
            headers={"X-Admin-Token": token, "Origin": "http://evil.com", "If-Match": etag},
            json={"server": {"timeout_s": 15.0}},
        )
        assert resp.status == 403

        # 4. PATCH nonexistent server -> 400
        resp = await s.patch(
            f"{base}/api/mcp/servers/nonexistent",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": etag},
            json={"server": {"timeout_s": 15.0}},
        )
        assert resp.status == 400
        data = await resp.json()
        assert data["code"] == "INVALID_PARAM"

        # 5. Valid PATCH: update timeout_s and add NEW_KEY while sending API_KEY as ******
        resp = await s.patch(
            f"{base}/api/mcp/servers/srv1",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": etag},
            json={
                "server": {
                    "timeout_s": 15.0,
                    "env": {"API_KEY": "******", "NEW_KEY": "new_val"},
                }
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["ok"] is True
        assert data["pending_restart"] is True
        new_etag = data["etag"]
        assert new_etag != etag

        # Check persisted server retained API_KEY == "secret_abc"
        persisted = next(srv for srv in manager.persisted_servers if srv["name"] == "srv1")
        assert persisted["env"]["API_KEY"] == "secret_abc"
        assert persisted["env"]["NEW_KEY"] == "new_val"
        assert persisted["timeout_s"] == 15.0

        # 6. Test connection with desensitized credentials:
        mock_called_env = {}

        async def mock_stdio_test(args, env=None, timeout=10.0, command=None, project_root=None):
            nonlocal mock_called_env
            mock_called_env = dict(env or {})
            return [{"name": "tool_ok", "description": ""}]

        with patch("ovs_agent.plugins.mcp_manager.test_stdio_mcp_server", side_effect=mock_stdio_test):
            resp = await s.post(
                f"{base}/api/mcp/test",
                headers={"X-Admin-Token": token, "Origin": f"http://{host}"},
                json={"name": "srv1", "transport": "stdio", "env": {"API_KEY": "******"}},
            )
            assert resp.status == 200
            res = await resp.json()
            assert res["ok"] is True
            assert res["tools_count"] == 1
            assert mock_called_env.get("API_KEY") == "secret_abc"


# ── T5: Strong ETag / If-Match Optimistic Locking ───────────────────────────

@pytest.mark.asyncio
async def test_t5_etag_optimistic_locking(dashboard_env):
    """T5: Missing or mismatched If-Match on POST/PATCH/DELETE raises 409 Conflict."""
    base = dashboard_env["base_url"]
    token = dashboard_env["token"]
    host = dashboard_env["host"]

    async with aiohttp.ClientSession() as s:
        # 1. Missing If-Match on POST -> 409
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}"},
            json={"server": {"name": "srv_new", "transport": "stdio", "args": ["-m", "ha_mcp"]}},
        )
        assert resp.status == 409
        data = await resp.json()
        assert data["code"] == "ETAG_MISMATCH"
        assert data["error"] == SAFE_CLIENT_MESSAGES["ETAG_MISMATCH"]
        assert "request_id" in data

        # 2. Stale If-Match on POST -> 409
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": "stale_etag_123"},
            json={"server": {"name": "srv_new", "transport": "stdio", "args": ["-m", "ha_mcp"]}},
        )
        assert resp.status == 409
        data = await resp.json()
        assert data["code"] == "ETAG_MISMATCH"

        # 3. Missing If-Match on PATCH -> 409
        resp = await s.patch(
            f"{base}/api/mcp/servers/srv1",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}"},
            json={"server": {"timeout_s": 20.0}},
        )
        assert resp.status == 409
        data = await resp.json()
        assert data["code"] == "ETAG_MISMATCH"

        # 4. Stale If-Match on PATCH -> 409
        resp = await s.patch(
            f"{base}/api/mcp/servers/srv1",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": "stale_etag_123"},
            json={"server": {"timeout_s": 20.0}},
        )
        assert resp.status == 409
        data = await resp.json()
        assert data["code"] == "ETAG_MISMATCH"

        # 5. Stale If-Match on DELETE -> 409
        resp = await s.delete(
            f"{base}/api/mcp/servers/srv1",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": "stale_etag_123"},
        )
        assert resp.status == 409
        data = await resp.json()
        assert data["code"] == "ETAG_MISMATCH"


# ── T6: Sandboxed Process Group Lifecycle and Clean Teardown ───────────────

@pytest.mark.asyncio
async def test_t6_sandboxed_test_and_process_cleanup(tmp_path):
    """T6: Normal exit handles ProcessLookupError idempotently; timeout kills group; failure -> 502."""
    # Write a mock test MCP server script in isolated tmp_path
    script_dir = tmp_path / "agent" / "mcp_servers"
    script_dir.mkdir(parents=True, exist_ok=True)
    server_script = script_dir / "test_mock_server.py"
    server_script.write_text(
        """import sys, json
# Handshake loop
while True:
    line = sys.stdin.readline()
    if not line: break
    msg = json.loads(line)
    if msg.get("method") == "initialize":
        resp = {"jsonrpc": "2.0", "id": msg["id"], "result": {"capabilities": {}}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif msg.get("method") == "notifications/initialized":
        pass
    elif msg.get("method") == "tools/list":
        resp = {"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [{"name": "mock_tool", "description": "A mock tool"}]}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
        break
""",
        encoding="utf-8",
    )

    # 1. Normal execution: discovery succeeds and process group is cleaned
    tools = await run_stdio_test(
        args=[str(server_script)],
        timeout=5.0,
        project_root=str(tmp_path),
    )
    assert len(tools) == 1
    assert tools[0]["name"] == "mock_tool"

    # 2. Timeout case: server sleeps without responding
    sleep_script = script_dir / "test_sleep_server.py"
    sleep_script.write_text("import time; time.sleep(30)", encoding="utf-8")
    with pytest.raises(DashboardApiError) as exc:
        await run_stdio_test(args=[str(sleep_script)], timeout=0.2, project_root=str(tmp_path))
    assert exc.value.code == "TEST_TIMEOUT"
    assert exc.value.status == 504

    # 3. Failure case: server exits immediately with error
    fail_script = script_dir / "test_fail_server.py"
    fail_script.write_text("import sys; sys.exit(1)", encoding="utf-8")
    with pytest.raises(DashboardApiError) as exc:
        await run_stdio_test(args=[str(fail_script)], timeout=2.0, project_root=str(tmp_path))
    assert exc.value.code == "CONNECTION_FAILED"
    assert exc.value.status == 502


# ── T7: Concurrent Test Limit (Semaphore) ──────────────────────────────────

@pytest.mark.asyncio
async def test_t7_concurrent_test_semaphore(tmp_path):
    """T7: Concurrent limit of 2; the 3rd simultaneous request raises TOO_MANY_REQUESTS (429)."""
    cfg_file = tmp_path / "agent.yaml"
    cfg_file.write_text("mcp_servers: []\n", encoding="utf-8")
    mgr = McpServerManager(cfg_file)

    # Mock a long running test connection
    async def mock_long_test(args, env=None, timeout=10.0, **kwargs):
        await asyncio.sleep(0.3)
        return [{"name": "slow_tool", "description": ""}]

    with patch("ovs_agent.plugins.mcp_manager.test_stdio_mcp_server", side_effect=mock_long_test):
        srv_req = {"transport": "stdio", "args": ["-m", "ha_mcp"]}

        async def run_req():
            return await mgr.test_server(srv_req)

        # Launch 3 requests concurrently
        t1 = asyncio.create_task(run_req())
        t2 = asyncio.create_task(run_req())
        # Let t1 and t2 acquire semaphore slots
        await asyncio.sleep(0.01)
        # 3rd request must be rejected immediately
        with pytest.raises(DashboardApiError) as exc:
            await run_req()
        assert exc.value.code == "TOO_MANY_REQUESTS"
        assert exc.value.status == 429

        res1, res2 = await asyncio.gather(t1, t2)
        assert res1["ok"] is True
        assert res2["ok"] is True


# ── T8: Dual-Hash Domains, 2MB Bounds, Phase 1/2 Rollback, State Sync ─────

@pytest.mark.asyncio
async def test_t8_dual_hash_and_atomic_rollback(tmp_path):
    """T8: Formatted YAML vs Canonical JSON dual hashes; 2MB limit; Phase 1/2 rollback & state sync."""
    cfg_file = tmp_path / "agent.yaml"
    # Write YAML with extra comments and indentation
    yaml_content = """# Comment line 1
mcp_servers:
  - name: s1
    transport: stdio
    args:
      - '-m'
      - ha_mcp
"""
    cfg_file.write_text(yaml_content, encoding="utf-8")

    mgr = McpServerManager(cfg_file)
    raw_bytes = cfg_file.read_bytes()
    expected_file_sha = hashlib.sha256(raw_bytes).hexdigest()
    expected_etag = mgr.calculate_etag(mgr.parse_config(raw_bytes))

    # Dual hash domains are completely independent
    assert mgr.persisted_file_sha256 == expected_file_sha
    assert mgr.persisted_config_etag == expected_etag
    assert mgr.persisted_file_sha256 != mgr.persisted_config_etag

    # 1. 2MB size limit on new payload
    huge_servers = [{"name": f"s_{i}", "transport": "stdio", "args": ["a" * 1000]} for i in range(2500)]
    with pytest.raises(ConfigPersistenceError) as exc:
        await mgr.commit_servers(huge_servers)
    assert exc.value.code == "FILE_TOO_LARGE"
    assert exc.value.status == 400

    # 2. Phase 1 failure when target is untouched does NOT trigger CONFIG_RECOVERY_REQUIRED
    real_replace = _safe_replace

    def replace_fail_on_backup(src, dst):
        if dst.endswith(".backup"):
            raise OSError(errno.EIO, "I/O error during backup replace")
        return real_replace(src, dst)

    with patch("ovs_agent.plugins.mcp_manager._safe_replace", side_effect=replace_fail_on_backup):
        with pytest.raises(OSError):
            await mgr.commit_servers([{"name": "new_s", "transport": "stdio", "args": ["-m", "ha_mcp"]}])
        # Target was untouched: must not enter recovery mode or lock configuration
        assert not mgr.config_recovery_required
        marker_file = tmp_path / ".config_recovery_required"
        assert not marker_file.exists()
        assert mgr.persisted_servers[0]["name"] == "s1"

    # 3. Phase 2 failure: rollback restores state and syncs memory
    failed_once = False
    real_replace = _safe_replace

    def replace_fail_on_phase2(src, dst):
        nonlocal failed_once
        if not failed_once and "tmp" in src and dst == str(cfg_file):
            failed_once = True
            raise OSError(errno.EIO, "Simulated Phase 2 disk failure")
        return real_replace(src, dst)

    with patch("ovs_agent.plugins.mcp_manager._safe_replace", side_effect=replace_fail_on_phase2):
        with pytest.raises(OSError):
            await mgr.commit_servers([{"name": "new_s2", "transport": "stdio", "args": ["-m", "ha_mcp"]}])

        # State should be restored to prior state
        assert mgr.persisted_file_sha256 == expected_file_sha
        assert mgr.persisted_config_etag == expected_etag
        assert mgr.persisted_servers[0]["name"] == "s1"
        assert not mgr.config_recovery_required


# ── T9: Runtime State Separation and Algebraic Restart Watchdog ────────────

@pytest.mark.asyncio
async def test_t9_runtime_separation_and_restart_watchdog():
    """T9: pending_restart flag; state machine white-list; create_task rollback; shutdown 503."""
    # 1. Algebraic State Machine Transitions
    watchdog = RestartWatchdog()
    gen = watchdog.generation

    # idle -> restarting allowed
    assert await watchdog.transition_state(gen, "restarting", reason="start") is True
    assert watchdog.is_restarting == "restarting"

    # restarting -> restart_failed allowed
    assert await watchdog.transition_state(gen, "restart_failed", reason="timeout") is True
    assert watchdog.is_restarting == "restart_failed"

    # restart_failed -> idle is ILLEGAL (only restarting allowed)
    assert await watchdog.transition_state(gen, "idle", reason="illegal") is False
    assert watchdog.is_restarting == "restart_failed"

    # Stale generation rejected
    assert await watchdog.transition_state(gen - 1, "restarting", reason="stale") is False

    # restart_failed -> restarting allowed
    assert await watchdog.transition_state(gen, "restarting", reason="retry") is True

    # restarting -> idle allowed (on schedule error)
    assert await watchdog.transition_state(gen, "idle", reason="rollback") is True

    # 2. create_task failure safely rolls back to idle
    with patch("asyncio.create_task", side_effect=RuntimeError("Loop error")):
        with pytest.raises(DashboardApiError) as exc:
            await watchdog.trigger_restart()
        assert exc.value.code == "INTERNAL_ERROR"
        assert watchdog.is_restarting == "idle"

    # 3. Shutdown prevents new restart requests (returns 503)
    await watchdog.shutdown()
    assert watchdog.shutting_down is True
    with pytest.raises(DashboardApiError) as exc:
        await watchdog.trigger_restart()
    assert exc.value.code == "SERVICE_UNAVAILABLE"
    assert exc.value.status == 503


@pytest.mark.asyncio
async def test_t9_restart_agent_endpoint_auth_and_shutdown(dashboard_env):
    """T9 API: restart_agent missing token -> 401, forged CSRF -> 403, shutting_down -> 503."""
    base = dashboard_env["base_url"]
    token = dashboard_env["token"]
    host = dashboard_env["host"]

    async with aiohttp.ClientSession() as s:
        # 1. Missing X-Admin-Token -> 401
        r = await s.post(f"{base}/api/control/restart_agent")
        assert r.status == 401
        data = await r.json()
        assert data["code"] == "UNAUTHORIZED"

        # 2. Invalid X-Admin-Token -> 401
        r = await s.post(f"{base}/api/control/restart_agent", headers={"X-Admin-Token": "bad_token"})
        assert r.status == 401
        data = await r.json()
        assert data["code"] == "UNAUTHORIZED"

        # 3. Valid token, forged Origin -> 403
        r = await s.post(
            f"{base}/api/control/restart_agent",
            headers={"X-Admin-Token": token, "Origin": "http://evil.com"},
        )
        assert r.status == 403
        data = await r.json()
        assert data["code"] == "FORBIDDEN"

        # 4. Valid token, forged Referer -> 403
        r = await s.post(
            f"{base}/api/control/restart_agent",
            headers={"X-Admin-Token": token, "Referer": "http://evil.com/page"},
        )
        assert r.status == 403
        data = await r.json()
        assert data["code"] == "FORBIDDEN"

        # 5. Shutting down state -> 503
        plugin = dashboard_env["plugin"]
        plugin._restart_watchdog._shutting_down = True
        try:
            r = await s.post(
                f"{base}/api/control/restart_agent",
                headers={"X-Admin-Token": token, "Origin": f"http://{host}"},
            )
            assert r.status == 503
            data = await r.json()
            assert data["code"] == "SERVICE_UNAVAILABLE"
        finally:
            plugin._restart_watchdog._shutting_down = False


# ── T10: Platform POSIX Capabilities, Startup Markers, Error Map Checks ────

@pytest.mark.asyncio
async def test_t10_platform_capabilities_and_markers(tmp_path, monkeypatch):
    """T10: O_NOFOLLOW/O_DIRECTORY mandatory; marker scans lock system; Exit 78 on marker write failure."""
    # 1. Platform capabilities
    validate_platform_capabilities()
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    with pytest.raises(RuntimeError, match="os.O_NOFOLLOW and os.O_DIRECTORY are mandatory"):
        validate_platform_capabilities()
    monkeypatch.undo()

    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
    with pytest.raises(RuntimeError, match="os.O_NOFOLLOW and os.O_DIRECTORY are mandatory"):
        validate_platform_capabilities()
    monkeypatch.undo()

    # 2. Error map bijective consistency
    validate_error_maps_consistency()

    # 3. Persistent marker scanning and validation
    cfg_dir = str(tmp_path)
    assert check_persistent_recovery_marker(cfg_dir) is False

    # Write valid marker
    write_persistent_recovery_marker(cfg_dir, "test_reason")
    assert check_persistent_recovery_marker(cfg_dir) is True

    # Cleanup raises CONFIG_RECOVERY_REQUIRED
    with pytest.raises(ConfigPersistenceError) as exc:
        cleanup_orphaned_tmp_files(cfg_dir)
    assert exc.value.code == "CONFIG_RECOVERY_REQUIRED"

    # 4. Orphaned marker_tmp detection
    marker_file = tmp_path / ".config_recovery_required"
    marker_file.unlink()
    assert check_persistent_recovery_marker(cfg_dir) is False

    orphan_tmp = tmp_path / ".config_recovery_required.marker_tmp.0123456789abcdef0123456789abcdef"
    orphan_tmp.write_text("dummy", encoding="utf-8")
    assert check_persistent_recovery_marker(cfg_dir) is True

    orphan_tmp.unlink()

    # 5. Marker write failure triggers Exit 78
    with patch("ovs_agent.plugins.mcp_manager._safe_replace", side_effect=OSError("Disk failed")):
        with pytest.raises(SystemExit) as exit_exc:
            write_persistent_recovery_marker(cfg_dir, "fail_test")
        assert exit_exc.value.code == 78

    # 6. Startup check failure in DebugDashboardPlugin triggers Exit 78 hard block
    fake_app = MagicMock()
    fake_app.config = Config(metadata={"dashboard_port": 18001})
    plugin = DebugDashboardPlugin(fake_app)
    plugin.setup()
    with patch.object(
        plugin._mcp_manager,
        "startup_check",
        side_effect=ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Marker exists"),
    ):
        with pytest.raises(SystemExit) as exit_exc:
            await plugin.start()
        assert exit_exc.value.code == 78


# ── M-b, Desensitization, Boundary & Cleanup Regex Reinforcement Tests ─────

@pytest.mark.asyncio
async def test_mb_test_target_tampering_refuses_credentials(dashboard_env):
    """M-b: Restoring credentials in test_server is refused if the test target is tampered with."""
    base = dashboard_env["base_url"]
    token = dashboard_env["token"]
    host = dashboard_env["host"]
    manager = dashboard_env["plugin"]._mcp_manager

    etag = manager.persisted_config_etag
    async with aiohttp.ClientSession() as s:
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": etag},
            json={
                "server": {
                    "name": "target_secure_srv",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "ha_mcp"],
                    "env": {"API_KEY": "super_secret_token_123"},
                }
            },
        )
        assert resp.status == 200

        mock_called_env = {}

        async def mock_stdio_test(args, env=None, timeout=10.0, command=None, project_root=None):
            nonlocal mock_called_env
            mock_called_env = dict(env or {})
            return [{"name": "tool_ok", "description": ""}]

        with patch("ovs_agent.plugins.mcp_manager.test_stdio_mcp_server", side_effect=mock_stdio_test):
            # 1. Unmodified target (same command & args) -> credentials restored
            mock_called_env.clear()
            resp = await s.post(
                f"{base}/api/mcp/test",
                headers={"X-Admin-Token": token, "Origin": f"http://{host}"},
                json={
                    "name": "target_secure_srv",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "ha_mcp"],
                    "env": {"API_KEY": "******"},
                },
            )
            assert resp.status == 200
            assert mock_called_env.get("API_KEY") == "super_secret_token_123"

            # 2. Tampered args -> credentials MUST NOT be injected
            mock_called_env.clear()
            resp = await s.post(
                f"{base}/api/mcp/test",
                headers={"X-Admin-Token": token, "Origin": f"http://{host}"},
                json={
                    "name": "target_secure_srv",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "device_mcp"],
                    "env": {"API_KEY": "******"},
                },
            )
            assert resp.status == 200
            assert mock_called_env.get("API_KEY") != "super_secret_token_123"
            assert mock_called_env.get("API_KEY") == "******"

            # 3. Tampered command -> credentials MUST NOT be injected
            mock_called_env.clear()
            resp = await s.post(
                f"{base}/api/mcp/test",
                headers={"X-Admin-Token": token, "Origin": f"http://{host}"},
                json={
                    "name": "target_secure_srv",
                    "transport": "stdio",
                    "command": "/usr/bin/python3",
                    "args": ["-m", "ha_mcp"],
                    "env": {"API_KEY": "******"},
                },
            )
            assert mock_called_env.get("API_KEY") != "super_secret_token_123"


def test_mask_sensitive_dict_deep_nesting():
    """Verify recursive masking on deeply nested dicts, lists, and lists of dicts."""
    raw = {
        "plain_field": "visible",
        "api_key": "top_secret",
        "nested_dict": {
            "password": "pwd",
            "normal_sub": 123,
            "inner": {
                "token": "inner_token",
            },
        },
        "nested_list": [
            {"secret": "item_secret", "name": "item1"},
            [{"auth_credential": "deep_in_list", "val": 42}],
            "plain_string_item",
            100,
        ],
    }
    masked = _mask_sensitive_dict(raw)
    assert masked["plain_field"] == "visible"
    assert masked["api_key"] == "******"
    assert masked["nested_dict"]["password"] == "******"
    assert masked["nested_dict"]["normal_sub"] == 123
    assert masked["nested_dict"]["inner"]["token"] == "******"
    assert masked["nested_list"][0]["secret"] == "******"
    assert masked["nested_list"][0]["name"] == "item1"
    assert masked["nested_list"][1][0]["auth_credential"] == "******"
    assert masked["nested_list"][1][0]["val"] == 42
    assert masked["nested_list"][2] == "plain_string_item"
    assert masked["nested_list"][3] == 100


@pytest.mark.asyncio
async def test_api_reconnect_boundary_success_and_error(dashboard_env):
    """Verify @mcp_api_boundary on _api_reconnect handles both success and exception uniformly."""
    base = dashboard_env["base_url"]
    token = dashboard_env["token"]
    host = dashboard_env["host"]
    plugin = dashboard_env["plugin"]

    async with aiohttp.ClientSession() as s:
        # 1. Success case
        resp = await s.post(
            f"{base}/api/control/reconnect",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data.get("ok") is True

        # 2. Error case intercepted by boundary
        with patch.object(plugin.app.slv, "reconnect", side_effect=RuntimeError("SLV bus broken")):
            resp = await s.post(
                f"{base}/api/control/reconnect",
                headers={"X-Admin-Token": token, "Origin": f"http://{host}"},
            )
            assert resp.status == 500
            err_data = await resp.json()
            assert err_data["ok"] is False
            assert err_data["code"] == "INTERNAL_ERROR"
            assert err_data["error"] == SAFE_CLIENT_MESSAGES["INTERNAL_ERROR"]
            assert "request_id" in err_data


def test_cleanup_orphaned_tmp_files_exact_regex(tmp_path):
    """M-c: cleanup_orphaned_tmp_files only cleans tmp files matching exact regex and preserves normal files/dirs."""
    # 1. Valid temp file matching regex
    tmp1 = tmp_path / "agent.yaml.tmp.0123456789abcdef0123456789abcdef"
    tmp1.write_text("temp", encoding="utf-8")

    tmp2 = tmp_path / "agent.yaml.backup.tmp.fedcba9876543210fedcba9876543210"
    tmp2.write_text("backup", encoding="utf-8")

    tmp3 = tmp_path / "agent.yaml.restore.tmp.abcdef0123456789abcdef0123456789"
    tmp3.write_text("restore", encoding="utf-8")

    # 2. Files that MUST NOT be deleted
    normal_file = tmp_path / "notes.tmp"
    normal_file.write_text("important user note", encoding="utf-8")

    normal_file2 = tmp_path / "custom.tmp.txt"
    normal_file2.write_text("custom", encoding="utf-8")

    normal_dir = tmp_path / "dir.tmp"
    normal_dir.mkdir()

    cleanup_orphaned_tmp_files(str(tmp_path))

    # Matched files deleted
    assert not tmp1.exists()
    assert not tmp2.exists()
    assert not tmp3.exists()

    # Normal files and directories preserved
    assert normal_file.exists()
    assert normal_file2.exists()
    assert normal_dir.exists()


# ── Tests for Defect 1: stdio env injection protection ─────────────────────

def test_validate_mcp_env_unit():
    """Unit tests for validate_mcp_env:
    - None and valid dict succeed.
    - Non-dict env or non-string values raise DashboardApiError(INVALID_PARAM).
    - Keys violating ^[A-Za-z_][A-Za-z0-9_]*$ raise SecurityError.
    - Dangerous keys (PYTHON*, LD_*, DYLD_*, PATH, VIRTUAL_ENV, BROWSER, NODE_OPTIONS, PERL5, IFS, ENV, BASH_ENV)
      case-insensitively raise SecurityError.
    """
    # 1. Valid cases
    validate_mcp_env(None)
    validate_mcp_env({})
    validate_mcp_env({"FOO": "bar", "_BAR": "baz", "TEST_VAR_123": "val"})

    # 2. Non-dict env raises INVALID_PARAM
    for bad_env in ["not_a_dict", 123, True, ["a", "b"]]:
        with pytest.raises(DashboardApiError) as exc_info:
            validate_mcp_env(bad_env)
        assert exc_info.value.code == "INVALID_PARAM"

    # 3. Non-string values raise INVALID_PARAM
    for bad_val in [123, True, None, ["a"], {"k": "v"}]:
        with pytest.raises(DashboardApiError) as exc_info:
            validate_mcp_env({"VALID_KEY": bad_val})
        assert exc_info.value.code == "INVALID_PARAM"

    # 4. Invalid key identifiers raise SecurityError
    for invalid_key in ["123_STARTS_WITH_DIGIT", "HAS-HYPHEN", "HAS SPACE", "HAS=EQUALS", "", "HAS;SEMI", "PATH\n", "FOO\nBAR", "VAR\n"]:
        with pytest.raises(SecurityError) as exc_info:
            validate_mcp_env({invalid_key: "value"})
        assert exc_info.value.code == "SECURITY_VIOLATION"
        assert "Forbidden environment variable key" in exc_info.value.internal_message

    # Explicit check for "PATH\n" trailing newline bypass prevention (\Z)
    with pytest.raises(SecurityError) as exc_info:
        validate_mcp_env({"PATH\n": "/usr/bin"})
    assert exc_info.value.code == "SECURITY_VIOLATION"
    assert "Forbidden environment variable key" in exc_info.value.internal_message

    # 5. Prohibited dangerous keys blocked case-insensitively
    prohibited_keys = [
        "PYTHONWARNINGS",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONOPTIMIZE",
        "pythonpath",
        "PythonWarnings",
        "python_custom",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "ld_preload",
        "Ld_Library_Path",
        "DYLD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "dyld_insert_libraries",
        "PATH",
        "path",
        "Path",
        "VIRTUAL_ENV",
        "virtual_env",
        "Virtual_Env",
        "BROWSER",
        "browser",
        "Browser",
        "NODE_OPTIONS",
        "node_options",
        "Node_Options",
        "PERL5",
        "perl5",
        "PERL5LIB",
        "perl5lib",
        "PERL5OPT",
        "IFS",
        "ifs",
        "Ifs",
        "ENV",
        "env",
        "Env",
        "BASH_ENV",
        "bash_env",
        "Bash_Env",
    ]
    for key in prohibited_keys:
        with pytest.raises(SecurityError) as exc_info:
            validate_mcp_env({key: "exploit_payload"})
        assert exc_info.value.code == "FORBIDDEN_ENV"
        assert f"Forbidden environment variable key: {key}" in exc_info.value.internal_message


@pytest.mark.asyncio
async def test_mcp_env_injection_blocked_in_server_management_and_api(dashboard_env):
    """End-to-end and API verification that prohibited env keys are blocked in add_or_update, patch, test_server."""
    base = dashboard_env["base_url"]
    token = dashboard_env["token"]
    host = dashboard_env["host"]
    manager = dashboard_env["plugin"]._mcp_manager

    # 1. Direct call to add_or_update_server with dangerous env
    with pytest.raises(SecurityError):
        await manager.add_or_update_server({
            "name": "bad_srv_1",
            "transport": "stdio",
            "args": ["-m", "ha_mcp"],
            "env": {"LD_PRELOAD": "/evil/lib.so"},
        })

    # Direct call with non-string value
    with pytest.raises(DashboardApiError) as exc_info:
        await manager.add_or_update_server({
            "name": "bad_srv_val",
            "transport": "stdio",
            "args": ["-m", "ha_mcp"],
            "env": {"CUSTOM_KEY": 12345},
        })
    assert exc_info.value.code == "INVALID_PARAM"

    # 2. HTTP POST /api/mcp/servers with dangerous env -> 400 SECURITY_VIOLATION
    etag = manager.persisted_config_etag
    async with aiohttp.ClientSession() as s:
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": etag},
            json={
                "server": {
                    "name": "bad_http_srv",
                    "transport": "stdio",
                    "args": ["-m", "ha_mcp"],
                    "env": {"PYTHONPATH": "/malicious/path"},
                }
            },
        )
        assert resp.status == 400
        data = await resp.json()
        assert data["ok"] is False
        assert data["code"] == "FORBIDDEN_ENV"

        # 3. HTTP PATCH /api/mcp/servers/srv1 with dangerous env -> 400 FORBIDDEN_ENV
        resp = await s.patch(
            f"{base}/api/mcp/servers/srv1",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": etag},
            json={
                "server": {
                    "env": {"PATH": "/evil/bin"},
                }
            },
        )
        assert resp.status == 400
        data = await resp.json()
        assert data["ok"] is False
        assert data["code"] == "FORBIDDEN_ENV"

        # 4. HTTP POST /api/mcp/test with dangerous env -> 400 FORBIDDEN_ENV
        resp = await s.post(
            f"{base}/api/mcp/test",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}"},
            json={
                "name": "srv1",
                "transport": "stdio",
                "args": ["-m", "ovs_agent.mcp.servers"],
                "env": {"BROWSER": "/evil/browser"},
            },
        )
        assert resp.status == 400
        data = await resp.json()
        assert data["ok"] is False
        assert data["code"] == "FORBIDDEN_ENV"


# ── Tests for Defect 2: _persist_agent_settings_to_yaml 0600 mode and cleanup ──

def test_persist_agent_settings_to_yaml_mode_0600_and_cleanup(tmp_path):
    """Verify _persist_agent_settings_to_yaml:
    - Atomically creates and enforces exact 0600 file mode.
    - Correctly serializes updated settings into YAML.
    - Detects failure to enforce 0600 mode and raises SecurityError.
    - Cleans up temporary uuid-named file on any error.
    """
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "pipeline_mode: always_on\nsleep_timeout_s: 30.0\nsystem_prompt: TEST\n",
        encoding="utf-8",
    )
    # Give it permissive permissions initially
    os.chmod(cfg_file, 0o644)

    # 1. Normal persist succeeds and enforces exact 0o600
    DebugDashboardPlugin._persist_agent_settings_to_yaml(
        cfg_file, {"sleep_timeout_s": 50.0, "wake_reply_text": "hello"}
    )
    st = cfg_file.stat()
    assert stat.S_IMODE(st.st_mode) == 0o600

    import yaml
    with cfg_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["sleep_timeout_s"] == 50.0
    assert data["wake_reply_text"] == "hello"
    assert data["pipeline_mode"] == "always_on"
    assert data["system_prompt"] == "TEST"

    # Verify no stray tmp files left
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0

    # 2. Verify SecurityError if 0600 mode check fails on target file, and verify rollback to prior content
    orig_fstat = os.fstat
    fstat_calls = [0]

    def mock_fstat(fd):
        fstat_calls[0] += 1
        st_res = orig_fstat(fd)
        # Mock mode only on verification step (second call) to simulate mode tampering on target
        if fstat_calls[0] == 2:
            return os.stat_result((
                0o100644,  # st_mode: regular file with 0644 instead of 0600
                st_res.st_ino,
                st_res.st_dev,
                st_res.st_nlink,
                st_res.st_uid,
                st_res.st_gid,
                st_res.st_size,
                st_res.st_atime,
                st_res.st_mtime,
                st_res.st_ctime,
            ))
        return st_res

    with patch("ovs_agent.plugins.debug_dashboard.os.fstat", side_effect=mock_fstat):
        with pytest.raises(SecurityError) as exc_info:
            DebugDashboardPlugin._persist_agent_settings_to_yaml(
                cfg_file, {"sleep_timeout_s": 99.0}
            )
        assert fstat_calls[0] >= 2
        assert "Failed to enforce 0600 mode on config file" in str(exc_info.value)

    # Assert destination file content was rolled back to original content before modification
    with cfg_file.open("r", encoding="utf-8") as f:
        rolled_back_data = yaml.safe_load(f)
    assert rolled_back_data["sleep_timeout_s"] == 50.0
    assert rolled_back_data["wake_reply_text"] == "hello"

    # 3. Exception cleanup on replace failure
    with patch("ovs_agent.plugins.mcp_manager.os.replace", side_effect=OSError(errno.EIO, "I/O error")):
        with pytest.raises(ConfigPersistenceError) as exc_info:
            DebugDashboardPlugin._persist_agent_settings_to_yaml(
                cfg_file, {"sleep_timeout_s": 77.0}
            )
        assert exc_info.value.code == "REPLACE_FAILED"

    # Verify all temporary files are cleaned up after exception
    leftover_tmps = list(tmp_path.glob("*.tmp*"))
    assert len(leftover_tmps) == 0

    # 4. Exception raised when temporary file cleanup fails
    orig_unlink = os.unlink

    def mock_unlink_fail(p):
        if ".tmp." in str(p):
            raise OSError(errno.EACCES, "Permission denied on unlink")
        return orig_unlink(p)

    with patch("ovs_agent.plugins.mcp_manager.os.replace", side_effect=OSError(errno.EIO, "I/O error")), \
         patch("ovs_agent.plugins.debug_dashboard.os.unlink", side_effect=mock_unlink_fail):
        with pytest.raises(ConfigPersistenceError) as exc_info:
            DebugDashboardPlugin._persist_agent_settings_to_yaml(
                cfg_file, {"sleep_timeout_s": 88.0}
            )
        assert exc_info.value.code == "TMP_CLEANUP_FAILED"
        assert exc_info.value.errno == errno.EACCES

    # Clean up any leftover test files
    for f in tmp_path.glob("*.tmp*"):
        try:
            orig_unlink(f)
        except OSError:
            pass


def test_persist_agent_settings_to_yaml_rollback_failure(tmp_path):
    """Verify that when config replacement succeeds but verification fails (bad mode)
    and rollback fails during _safe_replace:
    - write_persistent_recovery_marker is called and writes .config_recovery_required
    - ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED") is raised with cause.
    """
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "pipeline_mode: always_on\nsleep_timeout_s: 30.0\nsystem_prompt: TEST\n",
        encoding="utf-8",
    )
    os.chmod(cfg_file, 0o600)

    orig_fstat = os.fstat
    fstat_calls = [0]

    def mock_fstat(fd):
        fstat_calls[0] += 1
        st_res = orig_fstat(fd)
        # Mock mode on verification step (second call) to simulate 0644 mode tampering on target
        if fstat_calls[0] == 2:
            return os.stat_result((
                0o100644,
                st_res.st_ino,
                st_res.st_dev,
                st_res.st_nlink,
                st_res.st_uid,
                st_res.st_gid,
                st_res.st_size,
                st_res.st_atime,
                st_res.st_mtime,
                st_res.st_ctime,
            ))
        return st_res

    import ovs_agent.plugins.debug_dashboard as dbg_mod
    orig_safe_replace = dbg_mod._safe_replace
    target_replace_calls = [0]

    def mock_safe_replace(src, dst):
        if dst == str(cfg_file):
            target_replace_calls[0] += 1
            if target_replace_calls[0] == 2:
                # Fail specifically on the rollback replace step
                raise OSError(errno.EIO, "Simulated rollback disk write failure")
        return orig_safe_replace(src, dst)

    with patch("ovs_agent.plugins.debug_dashboard.os.fstat", side_effect=mock_fstat), \
         patch("ovs_agent.plugins.debug_dashboard._safe_replace", side_effect=mock_safe_replace):
        with pytest.raises(ConfigPersistenceError) as exc_info:
            DebugDashboardPlugin._persist_agent_settings_to_yaml(
                cfg_file, {"sleep_timeout_s": 99.0}
            )

    assert exc_info.value.code == "CONFIG_RECOVERY_REQUIRED"
    assert "Target configuration replaced but rollback failed" in str(exc_info.value)
    assert target_replace_calls[0] == 2
    assert fstat_calls[0] >= 2

    # Assert that persistent recovery marker was committed
    assert check_persistent_recovery_marker(str(tmp_path)) is True
    marker_file = tmp_path / ".config_recovery_required"
    assert marker_file.exists()
    content = marker_file.read_text(encoding="utf-8")
    assert "reason=rollback_failed" in content


def test_persist_agent_settings_to_yaml_rollback_and_marker_both_fail(tmp_path):
    """Verify that when config replacement succeeds but verification fails, rollback fails,
    and write_persistent_recovery_marker raises SystemExit(78) via fatal_config_recovery_exit:
    - The SystemExit is caught internally and does not leak out of the function.
    - ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED") is reliably raised with rollback failure cause.
    """
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "pipeline_mode: always_on\nsleep_timeout_s: 30.0\nsystem_prompt: TEST\n",
        encoding="utf-8",
    )
    os.chmod(cfg_file, 0o600)

    orig_fstat = os.fstat
    fstat_calls = [0]

    def mock_fstat(fd):
        fstat_calls[0] += 1
        st_res = orig_fstat(fd)
        # Mock mode on verification step (second call) to simulate 0644 mode tampering on target
        if fstat_calls[0] == 2:
            return os.stat_result((
                0o100644,
                st_res.st_ino,
                st_res.st_dev,
                st_res.st_nlink,
                st_res.st_uid,
                st_res.st_gid,
                st_res.st_size,
                st_res.st_atime,
                st_res.st_mtime,
                st_res.st_ctime,
            ))
        return st_res

    import ovs_agent.plugins.debug_dashboard as dbg_mod
    orig_safe_replace = dbg_mod._safe_replace
    target_replace_calls = [0]

    def mock_safe_replace(src, dst):
        if dst == str(cfg_file):
            target_replace_calls[0] += 1
            if target_replace_calls[0] == 2:
                # Fail specifically on the rollback replace step
                raise OSError(errno.EIO, "Simulated rollback disk write failure")
        return orig_safe_replace(src, dst)

    def mock_write_marker_fatal(config_dir, reason):
        # Simulate fatal_config_recovery_exit raising SystemExit(78)
        fatal_config_recovery_exit(f"Simulated write_persistent_recovery_marker failure for {reason}")

    with patch("ovs_agent.plugins.debug_dashboard.os.fstat", side_effect=mock_fstat), \
         patch("ovs_agent.plugins.debug_dashboard._safe_replace", side_effect=mock_safe_replace), \
         patch("ovs_agent.plugins.debug_dashboard.write_persistent_recovery_marker", side_effect=mock_write_marker_fatal):
        with pytest.raises(ConfigPersistenceError) as exc_info:
            DebugDashboardPlugin._persist_agent_settings_to_yaml(
                cfg_file, {"sleep_timeout_s": 99.0}
            )

    assert exc_info.value.code == "CONFIG_RECOVERY_REQUIRED"
    assert "Target configuration replaced but rollback failed" in str(exc_info.value)
    assert target_replace_calls[0] == 2
    assert fstat_calls[0] >= 2


# ── Spec v2.5 Verification Tests ──────────────────────────────────────────

def test_normalize_stdio_mode_a(tmp_path):
    """Verify Mode A parses valid module identifier and sets cwd to config_dir."""
    cfg_dir = str(tmp_path)
    cmd, args, env, cwd = normalize_stdio_mcp_config(
        command=None,
        args=["-m", "ovs_agent.mcp.servers", "--extra"],
        env={"CUSTOM_VAR": "val1"},
        config_dir=cfg_dir,
    )
    assert cmd == sys.executable
    assert args == ["-m", "ovs_agent.mcp.servers", "--extra"]
    assert cwd == cfg_dir
    assert env["CUSTOM_VAR"] == "val1"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert "VIRTUAL_ENV" in env


def test_normalize_stdio_mode_b_relative(tmp_path):
    """Verify Mode B resolves relative script against config_dir and sets cwd to script directory."""
    sub_dir = tmp_path / "scripts"
    sub_dir.mkdir()
    script_file = sub_dir / "my_server.py"
    script_file.write_text("#!/usr/bin/env python\nprint('hello')\n", encoding="utf-8")

    cmd, args, env, cwd = normalize_stdio_mcp_config(
        command=None,
        args=["scripts/my_server.py", "--flag"],
        config_dir=str(tmp_path),
    )
    assert cmd == sys.executable
    assert args == [str(script_file.resolve()), "--flag"]
    assert cwd == str(sub_dir.resolve())


def test_validation_precedence(tmp_path):
    """Verify precedence: MISSING_ARGUMENT -> SECURITY_VIOLATION -> ILLEGAL_ARGUMENT -> INVALID_MODULE_NAME -> SCRIPT_NOT_FOUND."""
    # 1. Missing argument: empty list
    with pytest.raises(DashboardApiError) as exc:
        normalize_stdio_mcp_config(command=None, args=[])
    assert exc.value.code == "MISSING_ARGUMENT"

    # 2. Metacharacter vs -c: metacharacter takes precedence
    with pytest.raises(DashboardApiError) as exc:
        normalize_stdio_mcp_config(command=None, args=["-c; rm -rf /"])
    assert exc.value.code == "SECURITY_VIOLATION"

    # 3. -c vs invalid module name: -c takes precedence
    with pytest.raises(DashboardApiError) as exc:
        normalize_stdio_mcp_config(command=None, args=["-c", "import sys"])
    assert exc.value.code == "ILLEGAL_ARGUMENT"

    # 4. Mode A: invalid module name takes precedence before checking existence
    with pytest.raises(DashboardApiError) as exc:
        normalize_stdio_mcp_config(command=None, args=["-m", "123_invalid_module!"])
    assert exc.value.code == "INVALID_MODULE_NAME"

    # 5. Mode B: non-existent file
    with pytest.raises(DashboardApiError) as exc:
        normalize_stdio_mcp_config(command=None, args=["does_not_exist.py"], config_dir=str(tmp_path))
    assert exc.value.code == "SCRIPT_NOT_FOUND"


def test_unreadable_file_rejected(tmp_path):
    """Verify unreadable script raises SCRIPT_NOT_READABLE."""
    script_file = tmp_path / "unreadable.py"
    script_file.write_text("print('test')\n", encoding="utf-8")
    with patch("os.access", return_value=False):
        with pytest.raises(DashboardApiError) as exc:
            normalize_stdio_mcp_config(command=None, args=[str(script_file)])
        assert exc.value.code == "SCRIPT_NOT_READABLE"


@pytest.mark.asyncio
async def test_stderr_drain_under_heavy_output(tmp_path):
    """Verify child writing 128KB unbuffered stderr output does not deadlock and succeeds."""
    server_py = tmp_path / "heavy_stderr_server.py"
    server_py.write_text(
        """import sys, json
sys.stderr.write("X" * (128 * 1024))
sys.stderr.flush()

for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": req["id"],
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "test-heavy", "version": "1.0"},
            }
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        resp = {
            "jsonrpc": "2.0",
            "id": req["id"],
            "result": {"tools": [{"name": "heavy_tool", "description": "heavy test"}]}
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
""",
        encoding="utf-8",
    )
    tools = await run_stdio_test(
        args=[str(server_py)],
        timeout=10.0,
        config_dir=str(tmp_path),
    )
    assert len(tools) == 1
    assert tools[0]["name"] == "heavy_tool"


@pytest.mark.asyncio
async def test_stderr_drain_error_handling():
    """Verify stderr read exception converts to CONNECTION_FAILED without hanging."""
    mock_proc = MagicMock()
    mock_proc.pid = 99994
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = MagicMock()
    async def mock_hang_stdout():
        await asyncio.sleep(100)
        return b""
    mock_proc.stdout.readline = AsyncMock(side_effect=mock_hang_stdout)
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = AsyncMock(side_effect=OSError("Simulated broken pipe"))

    with patch("ovs_agent.plugins.mcp_manager.launch_stdio_mcp_subprocess", AsyncMock(return_value=mock_proc)) as mock_launch, \
         patch("ovs_agent.plugins.mcp_manager._cleanup_process", AsyncMock(return_value="REAP_CONFIRMED")):
        with pytest.raises(DashboardApiError) as exc:
            await run_stdio_test(args=["-m", "ha_mcp"], timeout=5.0)
        assert exc.value.code == "CONNECTION_FAILED"
        assert "Simulated broken pipe" in exc.value.internal_message
        assert mock_launch.call_count == 1


def test_stderr_tail_redaction():
    """Verify sensitive tokens and secrets are redacted from diagnostic text."""
    raw_text = (
        "Error in server: Authorization: Bearer secret_token_value_12345678\n"
        "Config token=my_secret_token_12345\n"
        "OpenAI API key sk-ant-api03-abcdef1234567890abcdef\n"
        "GitHub token ghp_123456789012345678901234567890123456\n"
        'JSON data: {"token": "sensitive-json-token", "password": "two word secret", "api_key": "sec-12345"}'
    )
    redacted = redact_sensitive_text(raw_text)
    assert "secret_token_value_12345678" not in redacted
    assert "my_secret_token_12345" not in redacted
    assert "sk-ant-api03-abcdef1234567890abcdef" not in redacted
    assert "ghp_123456789012345678901234567890123456" not in redacted
    assert "sensitive-json-token" not in redacted
    assert "two word secret" not in redacted
    assert "sec-12345" not in redacted
    assert "******" in redacted


@pytest.mark.asyncio
async def test_process_group_cleanup_parent_exited_surviving_children(tmp_path):
    """Verify process group cleanup successfully kills stubborn grandchildren via SIGKILL even after parent exited."""
    child_py = tmp_path / "stubborn_tree.py"
    child_py.write_text(
        f"""import sys, time, subprocess, signal
signal.signal(signal.SIGTERM, signal.SIG_IGN)
cmd = [sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"]
subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True)
time.sleep(0.1)
sys.exit(0)
""",
        encoding="utf-8",
    )
    proc = await launch_stdio_mcp_subprocess(args=[str(child_py)], config_dir=str(tmp_path))
    pgid = proc.pid
    await proc.wait()
    assert proc.returncode is not None

    try:
        os.killpg(pgid, 0)
        group_alive = True
    except ProcessLookupError:
        group_alive = False
    assert group_alive, "Child process should still be alive in process group"

    status = await _cleanup_process(proc, pgid, drain_task=None)
    assert status == "REAP_CONFIRMED"

    deadline = asyncio.get_running_loop().time() + 1.0
    group_gone = False
    while asyncio.get_running_loop().time() < deadline:
        try:
            os.killpg(pgid, 0)
            await asyncio.sleep(0.05)
        except ProcessLookupError:
            group_gone = True
            break
    assert group_gone, "Process group should be completely reaped after cleanup"


@pytest.mark.asyncio
async def test_process_group_cleanup_under_repeated_cancellation():
    """Verify repeated cancellation preserves first cancel and completes cleanup."""
    mock_proc = MagicMock()
    mock_proc.pid = 99999
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = MagicMock()

    read_event = asyncio.Event()
    async def mock_readline():
        await read_event.wait()
        return b""
    mock_proc.stdout.readline = AsyncMock(side_effect=mock_readline)
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = AsyncMock(return_value=b"")

    cleanup_completed = False
    async def mock_cleanup(p, pg, dt):
        nonlocal cleanup_completed
        await asyncio.sleep(0.2)
        cleanup_completed = True
        return "REAP_CONFIRMED"

    with patch("ovs_agent.plugins.mcp_manager.launch_stdio_mcp_subprocess", AsyncMock(return_value=mock_proc)) as mock_launch, \
         patch("os.getpgid", return_value=99999), \
         patch("ovs_agent.plugins.mcp_manager._cleanup_process", side_effect=mock_cleanup):

        task = asyncio.create_task(run_stdio_test(["-m", "ovs_agent.mcp.servers"]))
        await asyncio.sleep(0.05)
        task.cancel("cancel-1")
        await asyncio.sleep(0.05)
        task.cancel("cancel-2")

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert mock_launch.call_count == 1
        assert cleanup_completed, "Cleanup task should have executed completely"
        assert exc_info.value.args == ("cancel-1",), "First cancellation must be preserved"


@pytest.mark.asyncio
async def test_process_group_cleanup_failure_with_cancellation_precedence():
    """Verify cleanup exception is attached to context while CancelledError retains precedence."""
    mock_proc = MagicMock()
    mock_proc.pid = 99998
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = MagicMock()
    read_event = asyncio.Event()
    async def mock_hang_read():
        await read_event.wait()
        return b""
    mock_proc.stdout.readline = AsyncMock(side_effect=mock_hang_read)
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = AsyncMock(return_value=b"")

    cleanup_error = RuntimeError("Internal cleanup error")
    async def mock_cleanup_fail(p, pg, dt):
        await asyncio.sleep(0.1)
        raise cleanup_error

    with patch("ovs_agent.plugins.mcp_manager.launch_stdio_mcp_subprocess", AsyncMock(return_value=mock_proc)) as mock_launch, \
         patch("os.getpgid", return_value=99998), \
         patch("ovs_agent.plugins.mcp_manager._cleanup_process", side_effect=mock_cleanup_fail):

        task = asyncio.create_task(run_stdio_test(["-m", "ovs_agent.mcp.servers"]))
        await asyncio.sleep(0.05)
        task.cancel("outer-cancel")

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert mock_launch.call_count == 1
        assert exc_info.value.args == ("outer-cancel",)
        assert exc_info.value.__context__ is cleanup_error, "Cleanup error must be attached to __context__"


@pytest.mark.asyncio
async def test_process_group_cleanup_deadline_timeout_transfer_and_observation():
    """Verify cleanup task exceeding 3.5s budget is transferred to background tasks and properly observed."""
    mock_proc = MagicMock()
    mock_proc.pid = 99997
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = MagicMock()
    mock_proc.stdout.readline = AsyncMock(return_value=b"")
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = AsyncMock(return_value=b"")

    cleanup_done_event = asyncio.Event()
    async def mock_slow_cleanup(p, pg, dt):
        await cleanup_done_event.wait()
        return "REAP_CONFIRMED"

    real_time = asyncio.get_running_loop().time
    time_offset = 0.0
    def fake_time():
        return real_time() + time_offset

    with patch("ovs_agent.plugins.mcp_manager.launch_stdio_mcp_subprocess", AsyncMock(return_value=mock_proc)) as mock_launch, \
         patch("os.getpgid", return_value=99997), \
         patch("ovs_agent.plugins.mcp_manager._cleanup_process", side_effect=mock_slow_cleanup):

        time_offset = 10.0  # Immediately exceed 3.5s budget
        with patch.object(asyncio.get_running_loop(), "time", side_effect=fake_time):
            with pytest.raises(DashboardApiError) as exc:
                await run_stdio_test(["-m", "ovs_agent.mcp.servers"])
            assert exc.value.code == "CONNECTION_FAILED"

        assert mock_launch.call_count == 1
        assert len(_PENDING_CLEANUP_TASKS) == 1
        pending_task = next(iter(_PENDING_CLEANUP_TASKS))
        assert not pending_task.done()

        cleanup_done_event.set()
        await pending_task
        await asyncio.sleep(0.01)
        assert len(_PENDING_CLEANUP_TASKS) == 0


@pytest.mark.asyncio
async def test_reap_failure_status_tracking_and_supervision():
    """Verify process failing to reap within 1.0s marks REAP_TIMEOUT_PENDING and launches supervised task."""
    mock_proc = MagicMock()
    mock_proc.pid = 99996
    reap_event = asyncio.Event()
    async def mock_wait():
        await reap_event.wait()
        return 0
    mock_proc.wait = AsyncMock(side_effect=mock_wait)

    with patch("os.killpg"):
        status = await _cleanup_process(mock_proc, 99996, drain_task=None)
        assert status == "REAP_TIMEOUT_PENDING"
        assert len(_PENDING_REAP_TASKS) == 1

        reap_task = next(iter(_PENDING_REAP_TASKS))
        assert not reap_task.done()

        reap_event.set()
        await reap_task
        await asyncio.sleep(0.01)
        assert len(_PENDING_REAP_TASKS) == 0


def test_symlink_virtualenv_preservation(tmp_path):
    """Verify virtualenv path and pyvenv.cfg are preserved via symlinked Python."""
    venv_dir = tmp_path / "mock_venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    symlink_python = bin_dir / "python3"
    symlink_python.symlink_to(sys.executable)
    (venv_dir / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")

    cmd, args, env, cwd = normalize_stdio_mcp_config(
        command=str(symlink_python),
        args=["-m", "ha_mcp"],
    )
    assert cmd == str(symlink_python)
    assert env["VIRTUAL_ENV"] == str(venv_dir)


def test_case_insensitive_script_extension(tmp_path):
    """Verify .py and .PY extensions are both accepted."""
    script = tmp_path / "UPPERCASE_SERVER.PY"
    script.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    cmd, args, env, cwd = normalize_stdio_mcp_config(
        command=None,
        args=[str(script)],
        config_dir=str(tmp_path),
    )
    assert args[0] == str(script.resolve())


def test_mcp_bridge_plugin_parity(tmp_path):
    """Verify McpBridgePlugin leverages normalize_stdio_mcp_config and initializes cwd appropriately."""
    from ovs_agent.mcp.plugin import McpBridgePlugin

    sub_dir = tmp_path / "custom_mcp"
    sub_dir.mkdir()
    script = sub_dir / "test_script.py"
    script.write_text("print(1)\n", encoding="utf-8")

    config_yaml = tmp_path / "agent.yaml"
    config_yaml.write_text("dummy: true\n", encoding="utf-8")

    cfg = Config()
    cfg._source_path = config_yaml
    cfg.mcp_servers = [
        {
            "name": "relative_server",
            "transport": "stdio",
            "args": ["custom_mcp/test_script.py"],
        }
    ]
    mock_app = MagicMock()
    mock_app.config = cfg

    plugin = McpBridgePlugin(mock_app)
    ok = plugin.setup()
    assert ok is True
    assert len(plugin._connections) == 1
    conn = plugin._connections[0]
    assert conn.cwd == str(sub_dir.resolve())
    assert conn.args[0] == str(script.resolve())


@pytest.mark.asyncio
async def test_stderr_drain_failure_races_and_causes_immediate_connection_failed():
    """Verify drain task exception immediately causes CONNECTION_FAILED without waiting for stdout timeout."""
    mock_proc = MagicMock()
    mock_proc.pid = 99995
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = MagicMock()
    # Stdout readline hangs forever
    async def mock_hang_stdout():
        await asyncio.sleep(100)
        return b""
    mock_proc.stdout.readline = AsyncMock(side_effect=mock_hang_stdout)
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = AsyncMock(side_effect=OSError("Immediate stderr failure"))

    with patch("ovs_agent.plugins.mcp_manager.launch_stdio_mcp_subprocess", AsyncMock(return_value=mock_proc)) as mock_launch, \
         patch("os.getpgid", return_value=99995), \
         patch("ovs_agent.plugins.mcp_manager._cleanup_process", AsyncMock(return_value="REAP_CONFIRMED")):
        start = asyncio.get_running_loop().time()
        with pytest.raises(DashboardApiError) as exc:
            await run_stdio_test(["-m", "ovs_agent.mcp.servers"], timeout=5.0)
        elapsed = asyncio.get_running_loop().time() - start
        assert elapsed < 1.0, f"Expected immediate failure, took {elapsed}s"
        assert exc.value.code == "CONNECTION_FAILED"
        assert "MCP stderr drain failed" in exc.value.internal_message
        assert mock_launch.call_count == 1


def test_stderr_redaction_quoted_json():
    """Verify quoted JSON keys and values are completely redacted."""
    json_text = '{"token": "secret_abc_123", "password": "super secret password", "apiKey": "xyz_987"}'
    redacted = redact_sensitive_text(json_text)
    assert "secret_abc_123" not in redacted
    assert "super secret password" not in redacted
    assert "xyz_987" not in redacted
    assert '"******"' in redacted

    case1 = '{"password": "alpha\'BRAVO_PRIVATE"}'
    assert redact_sensitive_text(case1) == '{"password": "******"}'

    case2 = r'{"token": "foo\"bar"}'
    assert redact_sensitive_text(case2) == '{"token": "******"}'

    assert redact_sensitive_text("password=alpha'BRAVO_PRIVATE") == 'password="******"'
    assert redact_sensitive_text("token BRAVO_PRIVATE") == 'token "******"'
    assert redact_sensitive_text('token "BRAVO_PRIVATE"') == 'token "******"'
    assert redact_sensitive_text("token 'BRAVO_PRIVATE'") == "token '******'"
    assert redact_sensitive_text("Bearer abcdefgh/BRAVO_PRIVATE+tail==") == "Bearer ******"

    multiline = "api_key: secret_line1\n    continuation_line2\n    continuation_line3"
    assert redact_sensitive_text(multiline) == 'api_key: "******"'

    multiline_blank = "api_key: first_line\n    second_line\n\n    BRAVO_PRIVATE"
    assert redact_sensitive_text(multiline_blank) == 'api_key: "******"'

    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
    assert redact_sensitive_text(pem) == "******"


@pytest.mark.asyncio
async def test_add_server_empty_args_yields_missing_argument(dashboard_env):
    """Verify add_or_update_server yields MISSING_ARGUMENT for empty args list even when forbidden env is present."""
    base = dashboard_env["base_url"]
    token = dashboard_env["token"]
    host = dashboard_env["host"]
    manager = dashboard_env["plugin"]._mcp_manager
    etag = manager.persisted_config_etag

    async with aiohttp.ClientSession() as s:
        resp = await s.post(
            f"{base}/api/mcp/servers",
            headers={"X-Admin-Token": token, "Origin": f"http://{host}", "If-Match": etag},
            json={
                "server": {
                    "name": "empty_args_srv",
                    "transport": "stdio",
                    "args": [],
                    "env": {"LD_PRELOAD": "/evil/preload.so"},
                }
            },
        )
        assert resp.status == 400
        data = await resp.json()
        assert data["code"] == "MISSING_ARGUMENT"


def test_safe_replace_ebusy_bind_mount_fallback(tmp_path):
    """Verify _safe_replace falls back to copying content directly into target
    when os.replace fails with [Errno 16] EBUSY (simulating Docker bind-mounted files)."""
    from ovs_agent.plugins.mcp_manager import ConfigPersistenceError, _safe_replace

    src_file = tmp_path / "config.yaml.tmp.12345"
    dst_file = tmp_path / "config.yaml"

    dst_file.write_text("initial: original_content\n", encoding="utf-8")
    src_file.write_text("updated: bind_mount_content\n", encoding="utf-8")
    os.chmod(src_file, 0o640)

    orig_replace = os.replace

    def mock_replace(src, dst):
        if dst == str(dst_file):
            raise OSError(errno.EBUSY, "Device or resource busy (simulated bind mount)")
        return orig_replace(src, dst)

    with patch("ovs_agent.plugins.mcp_manager.os.replace", side_effect=mock_replace):
        _safe_replace(str(src_file), str(dst_file))

    # Verify dst_file got the new content
    assert dst_file.read_text(encoding="utf-8") == "updated: bind_mount_content\n"
    # Verify src_file was cleaned up
    assert not src_file.exists()
    # Verify copymode preserved src mode (0640)
    st = dst_file.stat()
    assert stat.S_IMODE(st.st_mode) == 0o640

    # Non-EBUSY error should raise directly
    src_file.write_text("retry\n", encoding="utf-8")

    def mock_replace_eio(src, dst):
        raise OSError(errno.EIO, "Disk I/O error")

    with patch("ovs_agent.plugins.mcp_manager.os.replace", side_effect=mock_replace_eio):
        with pytest.raises(ConfigPersistenceError) as exc_info:
            _safe_replace(str(src_file), str(dst_file))
        assert exc_info.value.errno == errno.EIO


@pytest.mark.asyncio
async def test_target_changed_clears_old_credentials(dashboard_env):
    """Codex Review Finding P1: Changing MCP server URL or command must not leak old credentials."""
    manager = dashboard_env["plugin"]._mcp_manager

    # 1. Add initial SSE server with Authorization header
    with patch("ovs_agent.plugins.mcp_manager.validate_and_pin_sse_url", return_value=("1.2.3.3", "api.alpha.com", 443)):
        await manager.add_or_update_server({
            "name": "sse_auth_test",
            "transport": "sse",
            "url": "https://api.alpha.com/sse",
            "headers": {"Authorization": "Bearer alpha_secret_token"},
        })

    # Verify secret is saved
    servers = manager.persisted_servers
    srv = next(s for s in servers if s["name"] == "sse_auth_test")
    assert srv["headers"]["Authorization"] == "Bearer alpha_secret_token"

    # 2. Patch URL to a new target WITHOUT providing new headers -> old headers must be dropped
    with patch("ovs_agent.plugins.mcp_manager.validate_and_pin_sse_url", return_value=("1.2.3.4", "api.beta.com", 443)):
        await manager.patch_server("sse_auth_test", {
            "url": "https://api.beta.com/sse",
        })

    servers = manager.persisted_servers
    srv = next(s for s in servers if s["name"] == "sse_auth_test")
    assert srv["url"] == "https://api.beta.com/sse"
    assert "headers" not in srv or not srv.get("headers")

    # 3. Patch URL to another target while attempting to send masked '******' -> must raise INVALID_PARAM
    with patch("ovs_agent.plugins.mcp_manager.validate_and_pin_sse_url", return_value=("1.2.3.5", "api.gamma.com", 443)):
        with pytest.raises(DashboardApiError) as exc_info:
            await manager.patch_server("sse_auth_test", {
                "url": "https://api.gamma.com/sse",
                "headers": {"Authorization": "******"},
            })
        assert exc_info.value.code == "INVALID_PARAM"


@pytest.mark.asyncio
async def test_async_timeout_python_310_compatibility():
    """Codex Review Finding P2: Verify timeout works even when asyncio.timeout is unavailable (Python 3.10)."""
    from ovs_agent.plugins.debug_dashboard import _async_timeout

    # Test under simulated Python 3.10 where hasattr(asyncio, 'timeout') is bypassed
    with patch("asyncio.timeout", None):
        t0 = time.monotonic()
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            async with _async_timeout(0.05):
                await asyncio.sleep(0.3)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.25


@pytest.mark.asyncio
async def test_sse_test_rejects_html_and_verifies_handshake():
    """Codex Review Finding P2: SSE testing must reject plain HTML pages."""
    app = web.Application()

    async def html_handler(request):
        return web.Response(text="<html><body>Welcome</body></html>", content_type="text/html")

    app.router.add_get("/not_mcp", html_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        with patch("ovs_agent.plugins.mcp_manager.validate_and_pin_sse_url", return_value=("127.0.0.1", "127.0.0.1", port)):
            with pytest.raises(DashboardApiError) as exc_info:
                await run_sse_test(f"http://127.0.0.1:{port}/not_mcp")
            assert exc_info.value.code == "CONNECTION_FAILED"
            assert "expected 'text/event-stream'" in str(exc_info.value)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_sse_rejects_cross_origin_endpoint():
    """Codex Review Finding P1: Reject cross-origin SSE message endpoints."""
    app = web.Application()
    stop_event = asyncio.Event()

    async def sse_handler(request):
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b"event: endpoint\ndata: https://evil.attacker.com/messages\n\n")
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
        return resp

    app.router.add_get("/sse", sse_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        with patch("ovs_agent.plugins.mcp_manager.validate_and_pin_sse_url", return_value=("127.0.0.1", "127.0.0.1", port)):
            with pytest.raises(DashboardApiError) as exc_info:
                await run_sse_test(f"http://127.0.0.1:{port}/sse")
            assert exc_info.value.code == "CONNECTION_FAILED"
            assert "cross-origin targets are prohibited" in str(exc_info.value)
    finally:
        stop_event.set()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_sse_full_handshake_lifecycle_and_tool_discovery():
    """Codex Review Finding P2: Verify full MCP initialization lifecycle handshake over SSE."""
    app = web.Application()
    app["initialized_received"] = False
    stop_event = asyncio.Event()

    async def sse_handler(request):
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b"event: endpoint\ndata: /messages\n\n")
        request.app["sse_resp"] = resp
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
        return resp

    async def msg_handler(request):
        body = await request.json()
        sse = request.app.get("sse_resp")
        method = body.get("method")
        msg_id = body.get("id")

        if method == "initialize":
            res = json.dumps({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "test-mock", "version": "1.0"},
                }
            })
            await sse.write(f"data: {res}\n\n".encode("utf-8"))
            return web.Response(status=200, text="ok")
        elif method == "notifications/initialized":
            request.app["initialized_received"] = True
            return web.Response(status=200, text="ok")
        elif method == "tools/list":
            assert request.app["initialized_received"] is True
            res = json.dumps({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {"name": "fetch_weather", "description": "Fetches current weather"},
                        {"name": "calculate", "description": "Calculates math expression"}
                    ]
                }
            })
            await sse.write(f"data: {res}\n\n".encode("utf-8"))
            return web.Response(status=200, text="ok")
        return web.Response(status=400, text="unknown method")

    app.router.add_get("/sse", sse_handler)
    app.router.add_post("/messages", msg_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        with patch("ovs_agent.plugins.mcp_manager.validate_and_pin_sse_url", return_value=("127.0.0.1", "127.0.0.1", port)):
            tools = await run_sse_test(f"http://127.0.0.1:{port}/sse")
            assert len(tools) == 2
            assert tools[0]["name"] == "fetch_weather"
            assert tools[1]["name"] == "calculate"
    finally:
        stop_event.set()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_sse_handshake_initialize_error_rejected():
    """Verify that initialize rejection in SSE response raises CONNECTION_FAILED."""
    app = web.Application()
    stop_event = asyncio.Event()

    async def sse_handler(request):
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b"event: endpoint\ndata: /messages\n\n")
        request.app["sse_resp"] = resp
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
        return resp

    async def msg_handler(request):
        body = await request.json()
        sse = request.app.get("sse_resp")
        msg_id = body.get("id")
        res = json.dumps({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32600, "message": "Unsupported protocol version"}
        })
        await sse.write(f"data: {res}\n\n".encode("utf-8"))
        return web.Response(status=200, text="ok")

    app.router.add_get("/sse", sse_handler)
    app.router.add_post("/messages", msg_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        with patch("ovs_agent.plugins.mcp_manager.validate_and_pin_sse_url", return_value=("127.0.0.1", "127.0.0.1", port)):
            with pytest.raises(DashboardApiError) as exc_info:
                await run_sse_test(f"http://127.0.0.1:{port}/sse")
            assert exc_info.value.code == "CONNECTION_FAILED"
            assert "MCP initialize rejected" in str(exc_info.value)
    finally:
        stop_event.set()
        await runner.cleanup()


def test_safe_replace_bind_mount_copy_failure_restores_original(tmp_path):
    """Codex Review Finding P1: If bind-mount fallback copy fails, dst must be restored to original content."""
    from ovs_agent.plugins.mcp_manager import ConfigPersistenceError, _safe_replace

    src_file = tmp_path / "config.yaml.tmp.copy_fail"
    dst_file = tmp_path / "config.yaml"

    original_text = "original_key: persistent_safe_value\n"
    dst_file.write_text(original_text, encoding="utf-8")
    src_file.write_text("corrupted_key: bad_write\n", encoding="utf-8")

    orig_replace = os.replace

    def mock_replace(src, dst):
        if dst == str(dst_file):
            raise OSError(errno.EBUSY, "Simulated bind mount")
        return orig_replace(src, dst)

    def mock_copyfileobj(fsrc, fdst, length=None):
        # Truncate occurs on open, now simulate disk full during write
        raise OSError(errno.ENOSPC, "No space left on device")

    with patch("ovs_agent.plugins.mcp_manager.os.replace", side_effect=mock_replace), \
         patch("ovs_agent.plugins.mcp_manager.shutil.copyfileobj", side_effect=mock_copyfileobj):
        with pytest.raises(ConfigPersistenceError) as exc_info:
            _safe_replace(str(src_file), str(dst_file))
        assert exc_info.value.code == "REPLACE_FAILED"

    # Destination file must have been restored to its original content!
    assert dst_file.read_text(encoding="utf-8") == original_text


@pytest.mark.asyncio
async def test_stdio_mcp_test_handles_interleaved_notifications(tmp_path):
    """Codex Review Finding P2: Stdio MCP test must match request ID and skip interleaved notifications."""
    from ovs_agent.plugins.mcp_manager import test_stdio_mcp_server

    script = tmp_path / "mock_mcp_server.py"
    script.write_text(
        """import sys, json

while True:
    line = sys.stdin.readline()
    if not line:
        break
    msg = json.loads(line)
    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        # Send initialize response
        resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "mock", "version": "1.0"}}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        # Send an interleaved notification BEFORE the tools/list response!
        notif = {"jsonrpc": "2.0", "method": "notifications/tools/list_changed", "params": {}}
        sys.stdout.write(json.dumps(notif) + "\\n")
        sys.stdout.flush()

        # Then send actual response for id
        resp = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [{"name": "discovered_tool", "description": "Successfully parsed tool"}]
            }
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
""",
        encoding="utf-8",
    )

    tools = await test_stdio_mcp_server(
        command=sys.executable,
        args=[str(script)],
        timeout=3.0,
    )
    assert len(tools) == 1
    assert tools[0]["name"] == "discovered_tool"
    assert tools[0]["description"] == "Successfully parsed tool"


@pytest.mark.asyncio
async def test_env_var_substitution_in_pending_restart_and_test(tmp_path, monkeypatch):
    """Codex Review Finding P2: Persisted ${VAR} settings must be expanded for active comparison and testing."""
    monkeypatch.setenv("TEST_MCP_SECRET", "super_secret_123")
    monkeypatch.setenv("TEST_MCP_HOST", "127.0.0.1")

    cfg_file = tmp_path / "agent-config.yaml"
    cfg_file.write_text(
        """
mcp_servers:
  - name: test_sub
    transport: stdio
    command: python3
    args:
      - /opt/voice/test.py
    env:
      TOKEN: ${TEST_MCP_SECRET}
      HOST: ${TEST_MCP_HOST:-localhost}
""",
        encoding="utf-8",
    )

    # Active servers as loaded by load_config() (already expanded)
    active_servers = [
        {
            "name": "test_sub",
            "transport": "stdio",
            "command": "python3",
            "args": ["/opt/voice/test.py"],
            "env": {
                "TOKEN": "super_secret_123",
                "HOST": "127.0.0.1",
            },
        }
    ]

    manager = McpServerManager(config_path=str(cfg_file), active_servers=active_servers)
    # 1. pending_restart must be False because expanded values match active_servers
    assert manager.pending_restart is False

    # 2. Persisted servers retain raw template syntax
    assert manager.persisted_servers[0]["env"]["TOKEN"] == "${TEST_MCP_SECRET}"

    # 3. In test_server, effective_cfg expands ${VAR}
    mock_script = tmp_path / "mock_echo_env.py"
    mock_script.write_text(
        """import os, sys, json
msg = json.loads(sys.stdin.readline())
token = os.environ.get("TOKEN", "")
sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "mock", "version": "1.0"}}}) + "\\n")
sys.stdout.flush()
sys.stdin.readline() # initialized
msg2 = json.loads(sys.stdin.readline())
sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg2["id"], "result": {"tools": [{"name": f"token_{token}", "description": "tool"}]}}) + "\\n")
sys.stdout.flush()
""",
        encoding="utf-8",
    )

    test_payload = {
        "name": "test_sub",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(mock_script)],
        "env": {"TOKEN": "${TEST_MCP_SECRET}"},
    }
    res = await manager.test_server(test_payload)
    assert len(res["tools"]) == 1
    assert res["tools"][0]["name"] == "token_super_secret_123"


@pytest.mark.asyncio
async def test_disabling_spoken_replies_clears_metadata_fallbacks(dashboard_env):
    """Codex Review Finding P2: Saving empty reply text must clear metadata fallback fields."""
    plugin = dashboard_env["plugin"]
    cfg = plugin.app.config
    # Seed metadata fallbacks
    cfg.metadata = {
        "wake_reply_text": "在呢",
        "wakeword": {"reply_text": "在呢"},
        "exit_reply_text": "再见",
    }
    cfg.wake_reply_text = "在呢"
    cfg.exit_reply_text = "再见"

    base_url = dashboard_env["base_url"]

    # Clear both replies via settings API
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/agent/settings",
            json={"wake_reply_text": "", "exit_reply_text": ""},
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True
            assert data["wake_reply_text"] == ""
            assert data["exit_reply_text"] == ""

    # Verify cfg and metadata fallbacks were cleared
    assert cfg.wake_reply_text == ""
    assert cfg.exit_reply_text == ""
    assert "wake_reply_text" not in cfg.metadata
    assert "reply_text" not in cfg.metadata.get("wakeword", {})
    assert "exit_reply_text" not in cfg.metadata

    # Verify app_base resolver returns empty string (disabled)
    from ovs_agent.app_base import BaseApp
    assert BaseApp._get_wake_reply_text(plugin.app) == ""
    assert BaseApp._get_exit_reply_text(plugin.app) == ""


@pytest.mark.asyncio
async def test_active_servers_populates_runtime_status_from_bridge(dashboard_env):
    """Codex Review Finding P2: active_servers must reflect actual bridge connection and tool status."""
    plugin = dashboard_env["plugin"]
    base_url = dashboard_env["base_url"]
    token = dashboard_env["token"]

    # Inject mock bridge plugin into app.plugins
    class MockConnection:
        def __init__(self, name, is_conn, tools, err=None):
            self.name = name
            self.is_connected = is_conn
            self._tools = tools
            self._error = err

    class MockBridgePlugin:
        name = "mcp_bridge"
        def __init__(self):
            self._connections = [
                MockConnection("live_server", True, [{"name": "weather", "description": "Weather tool"}]),
                MockConnection("dead_server", False, [], err="Process exited with code 1"),
            ]
        def get_server_runtime_status(self, name):
            conn = next((c for c in self._connections if c.name == name), None)
            if conn and conn.is_connected:
                return {"status": "connected", "connected": True, "tools": conn._tools, "error": None}
            return {"status": "failed", "connected": False, "tools": [], "error": getattr(conn, "_error", "Stopped")}

    plugin.app.plugins.append(MockBridgePlugin())
    plugin._mcp_manager._active_mcp_servers = [
        {"name": "live_server", "transport": "stdio", "command": "python3"},
        {"name": "dead_server", "transport": "stdio", "command": "python3"},
    ]

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/api/mcp/servers", headers={"X-Admin-Token": token}) as resp:
            assert resp.status == 200
            data = await resp.json()
            active = {s["name"]: s for s in data["active_servers"]}

            assert active["live_server"]["status"] == "connected"
            assert active["live_server"]["connected"] is True
            assert len(active["live_server"]["tools"]) == 1
            assert active["live_server"]["tools"][0]["name"] == "weather"

            assert active["dead_server"]["status"] == "failed"
            assert active["dead_server"]["connected"] is False
            assert active["dead_server"]["tools"] == []
            assert "Process exited" in active["dead_server"]["error"]


@pytest.mark.asyncio
async def test_external_disk_modification_triggers_etag_mismatch_and_refreshes_state(tmp_path):
    """Codex Review Finding P2: Reject stale writes when config was modified externally on disk."""
    cfg_file = tmp_path / "agent-config.yaml"
    cfg_file.write_text(
        """
mcp_servers:
  - name: s1
    transport: stdio
    command: python3
""",
        encoding="utf-8",
    )

    manager = McpServerManager(config_path=str(cfg_file), active_servers=[])
    old_etag = manager.persisted_config_etag

    # External modification on disk
    cfg_file.write_text(
        """
mcp_servers:
  - name: s1
    transport: stdio
    command: python3
  - name: s_external
    transport: stdio
    command: python3
""",
        encoding="utf-8",
    )

    # Calling commit_servers must detect the on-disk ETag change
    with pytest.raises(DashboardApiError) as exc_info:
        await manager.commit_servers([{"name": "s1_edited", "transport": "stdio", "command": "python3"}])

    assert exc_info.value.code == "ETAG_MISMATCH"
    assert "modified externally" in str(exc_info.value)
    # Manager's cached state must have refreshed to the external disk content
    assert manager.persisted_config_etag != old_etag
    assert len(manager.persisted_servers) == 2
    assert manager.persisted_servers[1]["name"] == "s_external"


@pytest.mark.asyncio
async def test_sse_test_rejects_stream_without_endpoint_event():
    """Codex Review Finding P2: SSE test must reject streams that do not emit endpoint event."""
    app = web.Application()

    async def sse_no_endpoint_handler(request):
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        # Emits a comment/ping instead of endpoint event, then closes stream
        await resp.write(b": ping\n\n")
        return resp

    app.router.add_get("/sse", sse_no_endpoint_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        with patch("ovs_agent.plugins.mcp_manager.validate_and_pin_sse_url", return_value=("127.0.0.1", "127.0.0.1", port)):
            with pytest.raises(DashboardApiError) as exc_info:
                await run_sse_test(f"http://127.0.0.1:{port}/sse", timeout=2.0)
            assert exc_info.value.code == "CONNECTION_FAILED"
            assert "did not emit required MCP 'endpoint' event" in str(exc_info.value)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_patch_relative_script_retains_credentials(tmp_path):
    """Codex Review Finding P2: Target matching must normalize relative script paths before comparing."""
    cfg_file = tmp_path / "agent-config.yaml"
    script = tmp_path / "tool.py"
    script.write_text("print('hello')\n", encoding="utf-8")

    cfg_file.write_text(
        """
mcp_servers:
  - name: rel_server
    transport: stdio
    command: python3
    args:
      - tool.py
    env:
      API_KEY: my_precious_secret
""",
        encoding="utf-8",
    )

    manager = McpServerManager(config_path=str(cfg_file), active_servers=[])
    # Patch with masked secret and relative script path
    patch_payload = {
        "args": ["tool.py"],
        "env": {"API_KEY": "******", "NEW_VAR": "new_val"},
    }
    updated = await manager.patch_server("rel_server", patch_payload)
    srv = next(s for s in updated if s["name"] == "rel_server")
    # Secret must be retained!
    assert srv["env"]["API_KEY"] == "my_precious_secret"
    assert srv["env"]["NEW_VAR"] == "new_val"


def test_bridge_status_surfaces_tool_discovery_failure():
    """Codex Review Finding P2: Surface discovery failures when list_tools fails after connect."""
    from ovs_agent.mcp.plugin import McpBridgePlugin

    class MockApp:
        plugins = []
        config = None

    class MockConn:
        def __init__(self):
            self.name = "failing_server"
            self.is_connected = True
            self._tools = []
            self._error = "list_tools timeout or protocol error"

    bridge = McpBridgePlugin(MockApp(), connections=[MockConn()])
    status = bridge.get_server_runtime_status("failing_server")
    assert status["status"] == "failed"
    assert status["connected"] is False
    assert status["tools"] == []
    assert "protocol error" in status["error"]


@pytest.mark.asyncio
async def test_stdio_env_var_template_args_preserved_without_metachar_rejection(tmp_path):
    """Codex Review Finding P2: Validate expanded MCP settings while preserving stored templates."""
    cfg_file = tmp_path / "agent-config.yaml"
    script = tmp_path / "server.py"
    script.write_text("print('mock server')\n", encoding="utf-8")

    cfg_file.write_text(
        """
mcp_servers: []
""",
        encoding="utf-8",
    )

    manager = McpServerManager(config_path=str(cfg_file), active_servers=[])
    server_data = {
        "name": "templated_server",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(script), "--service-url", "${SERVICE_URL:-http://127.0.0.1:8080}"],
    }

    # Should succeed without SECURITY_VIOLATION on '$'
    updated = await manager.add_or_update_server(server_data)
    srv = next(s for s in updated if s["name"] == "templated_server")
    # Must preserve the exact template string in persisted configuration
    assert srv["args"][1] == "--service-url"
    assert srv["args"][2] == "${SERVICE_URL:-http://127.0.0.1:8080}"


def test_sanitize_mcp_url_and_secret_retention(tmp_path):
    """Codex Review Finding P2: Redact credentials embedded in MCP URLs and retain on match."""
    from ovs_agent.plugins.mcp_manager import (
        sanitize_mcp_url,
        sanitize_server_config,
        merge_server_secrets,
        is_mcp_target_matched,
        DashboardApiError,
    )

    raw_url = "https://admin:my_secret_pass@api.example.com/sse?api_key=token-xyz-123&format=json"
    masked = sanitize_mcp_url(raw_url)
    assert "my_secret_pass" not in masked
    assert "token-xyz-123" not in masked
    assert masked == "https://admin:******@api.example.com/sse?api_key=******&format=json"

    # Sanitize config dictionary
    server = {"name": "sse_srv", "transport": "sse", "url": raw_url}
    sanitized = sanitize_server_config(server)
    assert sanitized["url"] == masked

    # Target matching: masked URL matching sanitized existing URL should match
    assert is_mcp_target_matched({"transport": "sse", "url": masked}, server) is True
    # Tampered URL should not match
    assert is_mcp_target_matched({"transport": "sse", "url": "https://other.com/sse"}, server) is False

    # Secret retention on matching target
    merged = merge_server_secrets({"name": "sse_srv", "transport": "sse", "url": masked}, server)
    assert merged["url"] == raw_url

    # Masked URL on changed target must be rejected
    with pytest.raises(DashboardApiError) as exc_info:
        merge_server_secrets({"name": "sse_srv", "transport": "sse", "url": masked}, None)
    assert exc_info.value.code == "INVALID_PARAM"


@pytest.mark.asyncio
async def test_agent_settings_includes_instance_id_and_restart_state(dashboard_env):
    """Codex Review Finding P2: Verify process replacement before reporting restart success."""
    base_url = dashboard_env["base_url"]
    plugin = dashboard_env["plugin"]

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/api/agent/settings") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert "instance_id" in data
            assert isinstance(data["instance_id"], str)
            assert len(data["instance_id"]) > 0
            assert data["restart_failed"] is False


@pytest.mark.asyncio
async def test_sse_env_var_template_url_preserved_in_add_and_patch(tmp_path):
    """Codex Review Finding P2: Expand SSE URL templates before validating saved configurations."""
    cfg_file = tmp_path / "agent-config.yaml"
    cfg_file.write_text("mcp_servers: []\n", encoding="utf-8")

    manager = McpServerManager(config_path=str(cfg_file), active_servers=[])
    server_data = {
        "name": "templated_sse",
        "transport": "sse",
        "url": "${SSE_ENDPOINT:-http://127.0.0.1:8000/sse}",
    }

    with patch("ovs_agent.plugins.mcp_manager.validate_and_pin_sse_url", return_value=("127.0.0.1", "127.0.0.1", 8000)):
        updated = await manager.add_or_update_server(server_data)
        srv = next(s for s in updated if s["name"] == "templated_sse")
        assert srv["url"] == "${SSE_ENDPOINT:-http://127.0.0.1:8000/sse}"

        # Test patching with unset template variable
        patched = await manager.patch_server(
            "templated_sse",
            {"url": "${UNSET_SSE_URL}"},
        )
        srv2 = next(s for s in patched if s["name"] == "templated_sse")
        assert srv2["url"] == "${UNSET_SSE_URL}"


@pytest.mark.asyncio
async def test_active_servers_sanitizes_runtime_error_credentials(dashboard_env):
    """Codex Review Finding P2: Sanitize runtime connection errors before returning them."""
    plugin = dashboard_env["plugin"]
    base_url = dashboard_env["base_url"]
    token = dashboard_env["token"]

    class MockConn:
        def __init__(self, name, err):
            self.name = name
            self.is_connected = False
            self._tools = []
            self._error = err

    class MockBridge:
        name = "mcp_bridge"
        def __init__(self, conn):
            self._conn = conn
        def get_server_runtime_status(self, name):
            if name == self._conn.name:
                return {"status": "failed", "connected": False, "tools": [], "error": self._conn._error}
            return {"status": "stopped", "connected": False, "tools": [], "error": None}

    raw_err = (
        "Failed to connect to https://admin:super_secret_pwd@127.0.0.1:8000/sse?api_key=token-secret-999: "
        "Bearer sensitive_token_abc12345 unauthorized"
    )
    plugin.app.plugins.append(MockBridge(MockConn("sse_error_server", raw_err)))
    plugin._mcp_manager._active_mcp_servers = [
        {
            "name": "sse_error_server",
            "transport": "sse",
            "url": "https://admin:super_secret_pwd@127.0.0.1:8000/sse?api_key=token-secret-999",
            "env": {"API_KEY": "token-secret-999"},
        }
    ]

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/api/mcp/servers", headers={"X-Admin-Token": token}) as resp:
            assert resp.status == 200
            data = await resp.json()
            active = next(s for s in data["active_servers"] if s["name"] == "sse_error_server")
            err_msg = active["error"]
            assert "super_secret_pwd" not in err_msg
            assert "token-secret-999" not in err_msg
            assert "sensitive_token_abc12345" not in err_msg
            assert "******" in err_msg


@pytest.mark.asyncio
async def test_mcp_test_timeout_validation_and_bounds(tmp_path):
    """Codex Review Finding P2: Validate and bound MCP test timeouts before slot acquisition."""
    cfg_file = tmp_path / "agent-config.yaml"
    cfg_file.write_text("mcp_servers: []\n", encoding="utf-8")
    manager = McpServerManager(config_path=str(cfg_file), active_servers=[])

    invalid_timeouts = [0, -5, "Infinity", "-Infinity", "nan", 100, "abc"]
    for t in invalid_timeouts:
        with pytest.raises(DashboardApiError) as exc_info:
            await manager.test_server({"name": "t", "transport": "stdio", "timeout_s": t, "args": ["tool.py"]})
        assert exc_info.value.code == "INVALID_PARAM"
        assert "timeout_s" in str(exc_info.value)
        # Verify semaphore was not leaked or decremented
        assert manager._test_semaphore._value == 2


def test_bridge_and_test_runners_preserve_tool_schemas():
    """Codex Review Finding P2: Preserve tool schemas in dashboard API responses."""
    from ovs_agent.mcp.plugin import McpBridgePlugin

    class MockApp:
        plugins = []
        config = None

    class MockConn:
        def __init__(self):
            self.name = "schema_server"
            self.is_connected = True
            self._tools = [
                {
                    "name": "lookup_user",
                    "description": "Fetch user by ID",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"user_id": {"type": "string"}},
                        "required": ["user_id"],
                    },
                }
            ]
            self._error = None

    bridge = McpBridgePlugin(MockApp(), connections=[MockConn()])
    status = bridge.get_server_runtime_status("schema_server")
    assert status["status"] == "connected"
    assert len(status["tools"]) == 1
    tool = status["tools"][0]
    assert tool["name"] == "lookup_user"
    assert tool["description"] == "Fetch user by ID"
    assert "parameters" in tool
    assert tool["parameters"]["properties"]["user_id"]["type"] == "string"
    assert "inputSchema" in tool
    assert tool["inputSchema"]["required"] == ["user_id"]


@pytest.mark.asyncio
async def test_agent_settings_retries_persistence_after_earlier_failure(dashboard_env, tmp_path):
    """Codex Review Finding P2: Retry persistence for settings left unsaved by an earlier failure."""
    plugin = dashboard_env["plugin"]
    base_url = dashboard_env["base_url"]

    # Setup source YAML config file
    cfg_file = tmp_path / "agent-config.yaml"
    cfg_file.write_text("wake_reply_text: 'old_wake'\n", encoding="utf-8")
    plugin.app.config._source_path = str(cfg_file)
    plugin.app.config.wake_reply_text = "old_wake"

    async with aiohttp.ClientSession() as session:
        # First attempt: persistence raises an OSError (e.g. disk error)
        with patch.object(
            plugin,
            "_persist_agent_settings_to_yaml",
            side_effect=OSError("Disk full or I/O error"),
        ):
            async with session.post(
                f"{base_url}/api/agent/settings",
                json={"wake_reply_text": "retried_wake"},
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["persisted"] is False
                assert "Disk full" in data.get("persist_error", "")
                # Runtime updated
                assert plugin.app.config.wake_reply_text == "retried_wake"
                assert "wake_reply_text" in plugin._unsaved_agent_settings_fields

        # Second attempt: sending the same payload after resolving the disk issue
        async with session.post(
            f"{base_url}/api/agent/settings",
            json={"wake_reply_text": "retried_wake"},
        ) as resp2:
            assert resp2.status == 200
            data2 = await resp2.json()
            # Persistence must be retried and succeed!
            assert data2["persisted"] is True
            assert "wake_reply_text" in data2["changed"]
            assert "wake_reply_text" not in plugin._unsaved_agent_settings_fields
            # Verify on-disk file was actually updated
            assert "retried_wake" in cfg_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_reply_alternatives_list_preserved_when_saving_unrelated_settings(dashboard_env, tmp_path):
    """Codex Review Finding P2: Preserve reply alternatives when saving unrelated settings."""
    import yaml
    plugin = dashboard_env["plugin"]
    base_url = dashboard_env["base_url"]

    # Source YAML has wake_reply_text as a list of alternatives
    cfg_file = tmp_path / "agent-config.yaml"
    initial_yaml = {
        "wake_reply_text": ["你好", "我在"],
        "sleep_timeout_s": 30.0,
    }
    cfg_file.write_text(yaml.safe_dump(initial_yaml, allow_unicode=True), encoding="utf-8")
    plugin.app.config._source_path = str(cfg_file)
    plugin.app.config.wake_reply_text = ["你好", "我在"]
    plugin.app.config.sleep_timeout_s = 30.0

    async with aiohttp.ClientSession() as session:
        # GET returns comma-joined string for the input box
        async with session.get(f"{base_url}/api/agent/settings") as get_resp:
            assert get_resp.status == 200
            data = await get_resp.json()
            assert data["wake_reply_text"] == "你好, 我在"

        # POST submits the comma-joined string alongside an unrelated edit (sleep_timeout_s)
        payload = {
            "wake_reply_text": "你好, 我在",
            "sleep_timeout_s": 45.0,
        }
        async with session.post(f"{base_url}/api/agent/settings", json=payload) as post_resp:
            assert post_resp.status == 200
            res = await post_resp.json()
            assert res["ok"] is True
            assert res["persisted"] is True
            # wake_reply_text was not actually edited; must NOT be in changed list
            assert "wake_reply_text" not in res["changed"]
            assert "sleep_timeout_s" in res["changed"]

            # Runtime config must preserve original list
            assert plugin.app.config.wake_reply_text == ["你好", "我在"]
            assert plugin.app.config.sleep_timeout_s == 45.0

            # YAML file must preserve original list
            disk_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            assert disk_data["wake_reply_text"] == ["你好", "我在"]
            assert disk_data["sleep_timeout_s"] == 45.0





