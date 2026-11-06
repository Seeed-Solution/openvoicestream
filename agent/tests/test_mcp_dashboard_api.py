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
    _mask_sensitive_dict,
    _open_atomic_tmp,
    _read_all_bounded,
    _safe_close,
    _safe_fsync_dir,
    _safe_fsync_file,
    _safe_open_readonly,
    _safe_replace,
    _write_all,
    check_persistent_recovery_marker,
    cleanup_orphaned_tmp_files,
    fatal_config_recovery_exit,
    inspect_and_open_verified_exe,
    launch_stdio_mcp_subprocess,
    mcp_api_boundary,
    merge_server_secrets,
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
    assert exc_info.value.code == "SECURITY_VIOLATION"

    # S1: -cCODE joined form rejected
    with pytest.raises(SecurityError) as exc_info:
        validate_stdio_args(["-cimport os; print(1)"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    with pytest.raises(SecurityError) as exc_info:
        validate_stdio_args(["-m", "ha_mcp", "-cbad"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    # S1: -mMODULE joined form rejected
    with pytest.raises(SecurityError) as exc_info:
        validate_stdio_args(["-mha_mcp"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["-m", "ha_mcp; rm -rf /"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["-m", "unauthorized_module"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    # S1: External script / arbitrary script in project_root rejected
    with pytest.raises(DashboardApiError) as exc_info:
        validate_stdio_args(["/tmp/malicious.py"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    with pytest.raises(SecurityError) as exc_info:
        validate_stdio_args(["agent/ovs_agent/plugins/mcp_manager.py"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    # S1: Non-existent script or non-.py script in agent/mcp_servers/ rejected
    with pytest.raises(SecurityError) as exc_info:
        validate_stdio_args(["agent/mcp_servers/does_not_exist.py"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    with pytest.raises(SecurityError) as exc_info:
        validate_stdio_args(["agent/mcp_servers/test.sh"])
    assert exc_info.value.code == "SECURITY_VIOLATION"

    # S1: Valid Mode B script strictly within agent/mcp_servers/ passes
    mock_mcp_dir = tmp_path / "agent" / "mcp_servers"
    mock_mcp_dir.mkdir(parents=True, exist_ok=True)
    valid_script = mock_mcp_dir / "valid_server.py"
    valid_script.write_text("# valid script\n", encoding="utf-8")
    validate_stdio_args([str(valid_script)], project_root=str(tmp_path))

    # Valid Mode A args pass
    validate_stdio_args(["-m", "ovs_agent.mcp.servers"])
    validate_stdio_args(["-m", "ha_mcp"])

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

    # 2. Phase 1 failure with corrupted backup triggers CONFIG_RECOVERY_REQUIRED
    real_replace = _safe_replace

    def replace_fail_on_backup(src, dst):
        if dst.endswith(".backup"):
            raise OSError(errno.EIO, "I/O error during backup replace")
        return real_replace(src, dst)

    with patch("ovs_agent.plugins.mcp_manager._safe_replace", side_effect=replace_fail_on_backup):
        with pytest.raises(ConfigPersistenceError) as exc:
            await mgr.commit_servers([{"name": "new_s", "transport": "stdio", "args": ["-m", "ha_mcp"]}])
        assert exc.value.code == "CONFIG_RECOVERY_REQUIRED"
        assert exc.value.status == 500
        assert mgr.config_recovery_required is True

    # 3. Phase 2 failure: rollback restores state and syncs memory
    mgr._config_recovery_required = False
    marker_file = tmp_path / ".config_recovery_required"
    marker_file.unlink(missing_ok=True)

    # Let Phase 1 succeed, but fail during Phase 2 _safe_replace into target_path
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
        assert exc_info.value.code == "SECURITY_VIOLATION"
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
        assert data["code"] == "SECURITY_VIOLATION"

        # 3. HTTP PATCH /api/mcp/servers/srv1 with dangerous env -> 400 SECURITY_VIOLATION
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
        assert data["code"] == "SECURITY_VIOLATION"

        # 4. HTTP POST /api/mcp/test with dangerous env -> 400 SECURITY_VIOLATION
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
        assert data["code"] == "SECURITY_VIOLATION"


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


