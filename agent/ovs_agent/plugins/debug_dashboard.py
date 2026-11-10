"""Debug dashboard plugin v2: live agent observability + manual controls.

Browse to http://localhost:<dashboard_port>/ during local dev to see:
  - ConvState pill, latency cards, mic RMS chart, TTS indicator, errors
  - Filterable event log
  - Chat / Events / History tabs
  - Manual control buttons (reconnect / abort / send text)

Strictly a debugging aid. No auth, no TLS, no CORS handling.
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import logging
import os
import secrets
import stat
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..config import DEFAULT_EXIT_WORDS
from ..plugin import Plugin
from ..state import ConvState
from .mcp_manager import (
    ConfigPersistenceError,
    DashboardApiError,
    McpServerManager,
    RestartWatchdog,
    SecurityError,
    _open_atomic_tmp,
    _read_all_bounded,
    _safe_close,
    _safe_fchmod,
    _safe_fsync_dir,
    _safe_fsync_file,
    _safe_open_readonly,
    _safe_replace,
    _write_all,
    check_persistent_recovery_marker,
    fatal_config_recovery_exit,
    mcp_api_boundary,
    sanitize_error_text,
    write_persistent_recovery_marker,
)

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def _async_timeout(delay: float):
    """Asyncio timeout compatible with Python >= 3.10."""
    if getattr(asyncio, "timeout", None) is not None:
        async with asyncio.timeout(delay):
            yield
    else:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        handle = loop.call_later(delay, task.cancel)
        try:
            yield
        except asyncio.CancelledError:
            raise asyncio.TimeoutError() from None
        finally:
            handle.cancel()

_STATIC_DIR = Path(__file__).with_name("static")
_INDEX_HTML = _STATIC_DIR / "dashboard.html"

_AGENT_SETTINGS_KEYS = (
    "pipeline_mode",
    "sleep_timeout_s",
    "stop_words",
    "wake_reply_text",
    "exit_words",
    "exit_reply_text",
)


class DebugDashboardPlugin(Plugin):
    name = "debug_dashboard"

    def __init__(self, app) -> None:  # noqa: ANN001
        super().__init__(app)
        self._port: int = 18000
        self._runner = None  # aiohttp AppRunner
        self._site = None
        self._browser_clients: set = set()
        self._stats_task: asyncio.Task | None = None

        # ── per-turn timing scratch ──
        self._turn_id: int = 0
        self._last_speech_end_ts_ms: int | None = None
        self._last_utterance_ts_ms: int | None = None
        self._first_token_ts_ms: int | None = None
        self._first_tts_audio_ts_ms: int | None = None
        # ── rolling latency history (most-recent 3) ──
        self._latency_history: dict[str, deque] = {
            "asr": deque(maxlen=3),
            "ttft": deque(maxlen=3),
            "ttfa": deque(maxlen=3),
            "rtt": deque(maxlen=3),
        }
        # ── persistent error log (browser-side filters can hide them) ──
        self._errors: list[dict[str, Any]] = []
        # ── session start (used by browser via stats; here for completeness) ──
        self._session_started_ms: int = int(time.time() * 1000)
        # ── rolling cache-hit percentages (most-recent 20) ──
        self._cache_pct_history: deque = deque(maxlen=20)
        # ── per-turn TTS metrics ──
        self._tts_sentence_count: int = 0
        self._tts_bytes_current: int = 0
        self._tts_bytes_last: int = 0
        self._tts_last_sample_rate: int = 24000
        self._tts_last_duration_s: float = 0.0
        # ── idempotent start guard (avoids double-subscribe on EventBus
        #    when the plugin is started twice without an intervening stop) ──
        self._started: bool = False
        # ── lazy HTTP client to SLV (used by /api/tts/* proxy routes) ──
        self._slv_http: Any = None  # aiohttp.ClientSession
        # ── MCP server management and algebraic restart watchdog ──
        self._mcp_manager: Any = None
        self._mcp_admin_token: str | None = None
        self._restart_watchdog: Any = None
        # ── LLM endpoint configuration lock and restart status ──
        self._llm_config_lock: asyncio.Lock = asyncio.Lock()
        self._llm_config_pending_restart: bool = False
        self._instance_id: str = uuid.uuid4().hex
        self._unsaved_agent_settings_fields: set[str] = set()
        # ── control API auth keys ──
        self._control_api_keys: list[str] = []
        # ── startup pipeline mode (retained until process restart) ──
        self._startup_pipeline_mode: str | None = None
        cfg = getattr(app, "config", None)
        if cfg is not None:
            raw_mode = getattr(cfg, "pipeline_mode", None)
            if isinstance(raw_mode, str) and raw_mode:
                self._startup_pipeline_mode = raw_mode

    # ── lifecycle ─────────────────────────────────────────────────

    def setup(self) -> bool:
        try:
            meta = getattr(self.app.config, "metadata", {}) or {}
            self._port = int(meta.get("dashboard_port", 18000))
        except Exception:
            self._port = 18000

        try:
            config_path = getattr(getattr(self.app, "config", None), "_source_path", None)
            if not config_path:
                config_path = "agent.yaml"
            active_servers = getattr(getattr(self.app, "config", None), "mcp_servers", None) or []
            self._mcp_manager = McpServerManager(
                config_path=config_path,
                active_servers=active_servers,
            )
            self._mcp_admin_token = self._mcp_manager.admin_token
            exit_fn = getattr(self.app, "_restart_exit_fn", None)
            self._restart_watchdog = RestartWatchdog(exit_fn=exit_fn)
        except Exception:
            logger.exception("Failed to initialize McpServerManager in setup")
            self._mcp_manager = None
            self._mcp_admin_token = None
            self._restart_watchdog = None

        try:
            self._control_api_keys = self._load_control_api_keys()
        except Exception:
            self._control_api_keys = []

        return True

    async def start(self) -> None:
        if self._started:
            logger.warning(
                "DebugDashboardPlugin.start() called twice without stop(); "
                "ignoring duplicate start to avoid double EventBus subscribe"
            )
            return
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp not installed -- debug_dashboard disabled")
            return
        self._started = True

        if self._mcp_manager is not None:
            try:
                self._mcp_manager.startup_check()
            except ConfigPersistenceError as e:
                logger.critical("Startup MCP check encountered recovery marker: %s", e)
                fatal_config_recovery_exit(f"Startup check failed: {e}", exit_code=78)

        if not self._control_api_keys:
            self._control_api_keys = self._load_control_api_keys()

        @web.middleware
        async def control_auth_middleware(request, handler):
            if request.path.startswith("/api/control/"):
                keys = self._get_control_api_keys()
                if keys:
                    if not self._check_control_auth(request, keys):
                        return web.json_response(
                            {"ok": False, "error": "unauthorized"},
                            status=401,
                        )
            return await handler(request)

        web_app = web.Application(middlewares=[control_auth_middleware])
        web_app.router.add_get("/", self._handle_index)
        web_app.router.add_get("/ws", self._handle_ws)

        # MCP configuration and sandbox endpoints
        web_app.router.add_get("/api/mcp/servers", self._api_mcp_servers_get)
        web_app.router.add_post("/api/mcp/servers", self._api_mcp_servers_post)
        web_app.router.add_patch("/api/mcp/servers/{name}", self._api_mcp_servers_patch)
        web_app.router.add_delete("/api/mcp/servers/{name}", self._api_mcp_servers_delete)
        web_app.router.add_post("/api/mcp/test", self._api_mcp_test)

        # Control endpoints (POST).
        web_app.router.add_post("/api/control/reconnect", self._api_reconnect)
        web_app.router.add_post("/api/control/restart_mic", self._api_restart_mic)
        web_app.router.add_post("/api/control/restart_agent", self._api_restart_agent)
        web_app.router.add_post("/api/control/abort", self._api_abort)
        web_app.router.add_post("/api/control/send_text", self._api_send_text)
        web_app.router.add_post(
            "/api/control/inject_user_text", self._api_inject_user_text
        )
        # Mic-less audio inject. Always routed, but the handler refuses unless
        # OVS_AGENT_DEBUG_INJECT=1, so production is inert until enabled (+
        # restart). Unlike inject_user_text this is the only endpoint that
        # exercises a SERVER-LOOP deployment faithfully — see debug_inject.
        web_app.router.add_post("/api/control/inject_wav", self._api_inject_wav)
        web_app.router.add_get("/api/session/history", self._api_session_history)
        web_app.router.add_post("/api/session/clear", self._api_session_clear)
        # AppMode framework endpoints.
        web_app.router.add_get("/api/modes", self._api_modes_list)
        web_app.router.add_post("/api/control/mode", self._api_mode_switch)
        # Per-mode override editor (system_prompt etc.).
        web_app.router.add_get(
            "/api/modes/{name}/overrides", self._api_mode_overrides_get
        )
        web_app.router.add_post(
            "/api/modes/{name}/overrides", self._api_mode_overrides_post
        )
        # pipeline_mode endpoints (wake_word / push_to_talk).
        web_app.router.add_post("/api/control/wake", self._api_wake)
        web_app.router.add_post("/api/control/sleep", self._api_sleep)
        web_app.router.add_post("/api/control/ptt/start", self._api_ptt_start)
        web_app.router.add_post("/api/control/ptt/end", self._api_ptt_end)
        # TTS speaker / voice-clone proxy routes (forward to SLV).
        web_app.router.add_get("/api/tts/speakers", self._api_tts_speakers_list)
        web_app.router.add_get("/api/tts/runtime", self._api_tts_runtime_get)
        web_app.router.add_patch("/api/tts/runtime", self._api_tts_runtime_patch)
        web_app.router.add_post(
            "/api/tts/clone/embedding", self._api_tts_clone_embedding
        )
        web_app.router.add_post(
            "/api/tts/speakers/register", self._api_tts_speakers_register
        )
        web_app.router.add_delete(
            "/api/tts/speakers/{speaker_id}", self._api_tts_speakers_delete
        )
        # Errors management + agent settings.
        web_app.router.add_post("/api/errors/clear", self._api_errors_clear)
        web_app.router.add_get("/api/agent/settings", self._api_agent_settings_get)
        web_app.router.add_post("/api/agent/settings", self._api_agent_settings_post)
        web_app.router.add_get("/api/wakeword/runtime", self._api_wakeword_runtime_get)
        web_app.router.add_post("/api/wakeword/validate", self._api_wakeword_validate)
        web_app.router.add_patch("/api/wakeword/runtime", self._api_wakeword_runtime_patch)
        web_app.router.add_post("/api/wakeword/test-tone", self._api_wakeword_test_tone)
        web_app.router.add_post("/api/llm/probe", self._api_llm_probe)
        web_app.router.add_get("/api/llm/config", self._api_llm_config_get)
        web_app.router.add_post("/api/llm/config", self._api_llm_config_post)
        web_app.router.add_post("/api/llm/test", self._api_llm_test_post)
        web_app.router.add_get(
            "/api/translator/runtime", self._api_translator_runtime_get
        )
        web_app.router.add_patch(
            "/api/translator/runtime", self._api_translator_runtime_patch
        )

        # Static assets (css/js).
        if _STATIC_DIR.exists():
            web_app.router.add_static("/static/", str(_STATIC_DIR), show_index=False)

        self._runner = web.AppRunner(web_app)
        await self._runner.setup()
        # Default-bind to loopback. The dashboard exposes unauthenticated
        # control endpoints (e.g. /api/control/inject_user_text can trigger
        # arbitrary tool/arm actions); binding to 0.0.0.0 by default would
        # let any host on the LAN drive the robot. Operators who need
        # remote access can opt-in via OVS_DEBUG_DASHBOARD_BIND=0.0.0.0
        # (and should put a firewall / SSH tunnel in front).
        bind_host = os.environ.get("OVS_DEBUG_DASHBOARD_BIND", "127.0.0.1").strip() or "127.0.0.1"
        self._site = web.TCPSite(self._runner, bind_host, self._port)
        await self._site.start()
        logger.info("debug_dashboard listening on http://%s:%d", bind_host, self._port)
        if bind_host != "127.0.0.1":
            logger.warning(
                "debug_dashboard bound to %s — /api/control/inject_user_text "
                "and other control endpoints are UNAUTHENTICATED. Restrict "
                "access via firewall / SSH tunnel.",
                bind_host,
            )

        self._stats_task = asyncio.create_task(self._stats_loop(), name="dashboard-stats")

        # Subscribe to EventBus signals that don't have a plugin-hook bridge
        # (session.trim_history / edge_llm prefix_cache disable). Relay them
        # to browser clients so the LLM-health card can reflect them live.
        bus = getattr(self.app, "events", None)
        if bus is not None:
            try:
                bus.subscribe("on_session_trimmed", self._on_bus_session_trimmed)
                bus.subscribe(
                    "on_prefix_cache_disabled", self._on_bus_prefix_cache_disabled
                )
                bus.subscribe("on_echo_recovery", self._on_bus_echo_recovery)
            except Exception:  # pragma: no cover - defensive
                logger.debug("event bus subscribe failed", exc_info=True)

        await super().start()

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        bus = getattr(self.app, "events", None)
        if bus is not None:
            try:
                bus.unsubscribe("on_session_trimmed", self._on_bus_session_trimmed)
                bus.unsubscribe(
                    "on_prefix_cache_disabled", self._on_bus_prefix_cache_disabled
                )
                bus.unsubscribe("on_echo_recovery", self._on_bus_echo_recovery)
            except Exception:  # pragma: no cover - defensive
                pass

        if self._stats_task is not None:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stats_task = None

        for ws in list(self._browser_clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._browser_clients.clear()

        if self._slv_http is not None:
            try:
                await self._slv_http.close()
            except Exception:
                pass
            self._slv_http = None

        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:
                pass
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None

        if self._restart_watchdog is not None:
            try:
                await self._restart_watchdog.shutdown()
            except Exception:
                pass

        await super().stop()

    # ── HTTP / WS handlers ───────────────────────────────────────

    async def _handle_index(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            text = _INDEX_HTML.read_text(encoding="utf-8")
        except FileNotFoundError:
            return web.Response(status=500, text="dashboard.html not found")
        token = self._mcp_admin_token or ""
        meta_tag = f'<meta name="ovs-admin-token" content="{token}">'
        if "<head>" in text:
            text = text.replace("<head>", f"<head>\n    {meta_tag}", 1)
        elif "</head>" in text:
            text = text.replace("</head>", f"    {meta_tag}\n</head>", 1)
        return web.Response(
            text=text,
            content_type="text/html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, private",
            },
        )

    async def _handle_ws(self, request):  # noqa: ANN001
        from aiohttp import web
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._browser_clients.add(ws)
        # Send initial snapshot so late-connecting browsers don't see
        # stale "–" placeholders until the next event happens to fire.
        try:
            state_val = "idle"
            try:
                s = getattr(self.app, "_state", None)
                if s is not None:
                    state_val = getattr(s, "value", str(s))
            except Exception:
                state_val = "idle"
            mode_name = None
            mode_list: list[dict] = []
            try:
                mgr = getattr(self.app, "modes", None)
                if mgr is not None:
                    mode_name = mgr.current_name
                    mode_list = mgr.list_all()
            except Exception:
                mode_name = None
            pipeline_mode = "always_on"
            sleep_timeout_s = 30.0
            try:
                cfg = getattr(self.app, "config", None)
                if cfg is not None:
                    pipeline_mode = getattr(cfg, "pipeline_mode", "always_on")
                    sleep_timeout_s = float(getattr(cfg, "sleep_timeout_s", 30.0))
            except Exception:
                pass
            snapshot = {
                "event": "snapshot",
                "ts": int(time.time() * 1000),
                "data": {
                    "state": state_val,
                    "reconnect_count": getattr(self.app, "_slv_reconnect_count", 0),
                    "session_started_ms": self._session_started_ms,
                    "errors": list(self._errors),
                    "mode": mode_name,
                    "modes": mode_list,
                    "pipeline_mode": pipeline_mode,
                    "sleep_timeout_s": sleep_timeout_s,
                    "llm_availability": self._build_llm_availability_payload(),
                    "prefix_cache_disabled": self._read_prefix_cache_disabled(),
                },
            }
            await ws.send_str(json.dumps(snapshot, ensure_ascii=False))
        except Exception:
            logger.debug("snapshot send failed", exc_info=True)
        try:
            async for _msg in ws:
                pass  # ignore inbound
        finally:
            self._browser_clients.discard(ws)
        return ws

    # ── control auth ──────────────────────────────────────────────

    @staticmethod
    def _parse_keys(raw: Any) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, (list, tuple)):
            return [str(k).strip() for k in raw if str(k).strip()]
        return [part.strip() for part in str(raw).split(",") if part.strip()]

    def _load_control_api_keys(self) -> list[str]:
        raw_env = os.environ.get("OVS_CONTROL_API_KEY")
        if raw_env is not None and raw_env.strip():
            keys = self._parse_keys(raw_env)
            if keys:
                return keys
        meta = getattr(getattr(self.app, "config", None), "metadata", {}) or {}
        raw_meta = meta.get("control_api_key")
        if raw_meta:
            keys = self._parse_keys(raw_meta)
            if keys:
                return keys
        return []

    def _get_control_api_keys(self) -> list[str]:
        raw_env = os.environ.get("OVS_CONTROL_API_KEY")
        if raw_env is not None and raw_env.strip():
            keys = self._parse_keys(raw_env)
            if keys:
                return keys
        if self._control_api_keys:
            return self._control_api_keys
        meta = getattr(getattr(self.app, "config", None), "metadata", {}) or {}
        raw_meta = meta.get("control_api_key")
        if raw_meta:
            keys = self._parse_keys(raw_meta)
            if keys:
                return keys
        return []

    @staticmethod
    def _extract_bearer_token(header: str | None) -> str | None:
        if not header:
            return None
        parts = header.split(None, 1)
        if len(parts) != 2:
            return None
        if parts[0].lower() != "bearer":
            return None
        return parts[1].strip()

    def _check_control_auth(self, request: Any, keys: list[str]) -> bool:
        candidates: list[str] = []
        auth_header = request.headers.get("Authorization")
        if auth_header:
            token = self._extract_bearer_token(auth_header)
            if token:
                candidates.append(token)
        alt_header = request.headers.get("X-OVS-Control-Token")
        if alt_header and alt_header.strip():
            candidates.append(alt_header.strip())
        if not candidates:
            return False

        matched = False
        for candidate in candidates:
            cand_b = candidate.encode("utf-8", errors="replace")
            for key in keys:
                key_b = key.encode("utf-8", errors="replace")
                if secrets.compare_digest(cand_b, key_b):
                    matched = True
        return matched

    # ── /api/control/* ──────────────────────────────────────────

    @mcp_api_boundary
    async def _api_reconnect(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            try:
                await self.app.audio.stop_playback()
            except Exception:
                pass
            try:
                self.app._first_tts_seen = False
            except Exception:
                pass
            try:
                self.app._set_state(ConvState.IDLE)
            except Exception:
                pass
            await self.app.slv.reconnect()
            # Fresh session: clear the discard latch stop_playback above
            # armed, so the next turn's TTS isn't silently dropped.
            try:
                arm = getattr(self.app.audio, "arm_for_next_turn", None)
                if callable(arm):
                    arm()
            except Exception:  # pragma: no cover - defensive
                pass
            return web.json_response({"ok": True})
        except DashboardApiError:
            raise
        except Exception as e:
            raise DashboardApiError("INTERNAL_ERROR", f"Reconnect failed: {e}")

    @mcp_api_boundary
    async def _api_abort(self, request):  # noqa: ANN001
        from aiohttp import web
        from .mcp_manager import DashboardApiError
        errs: list[str] = []
        try:
            await self.app.slv.abort()
        except Exception as e:
            errs.append(f"abort: {e}")
        try:
            await self.app.audio.stop_playback()
        except Exception as e:
            errs.append(f"stop_playback: {e}")
        if errs:
            raise DashboardApiError("INTERNAL_ERROR", f"Abort failed: {'; '.join(errs)}")
        return web.json_response({"ok": True})

    @mcp_api_boundary
    async def _api_restart_mic(self, request):  # noqa: ANN001
        from aiohttp import web
        from .mcp_manager import DashboardApiError
        restart = getattr(self.app, "restart_mic_capture", None)
        if not callable(restart):
            raise DashboardApiError(
                "SERVICE_UNAVAILABLE", "restart_mic_capture unavailable"
            )
        try:
            await restart("dashboard")
            return web.json_response({"ok": True})
        except Exception as e:
            raise DashboardApiError("INTERNAL_ERROR", f"restart_mic failed: {e}")

    @mcp_api_boundary
    async def _api_restart_agent(self, request):  # noqa: ANN001
        import asyncio
        import os
        import signal
        from aiohttp import web
        from .mcp_manager import DashboardApiError

        if self._mcp_manager is None:
            raise DashboardApiError("SERVICE_UNAVAILABLE", "MCP manager not initialized")
        self._mcp_manager.verify_admin_token(request)
        self._mcp_manager.verify_csrf(request)

        exit_fn = getattr(self.app, "_restart_exit_fn", None)
        if self._restart_watchdog is not None:
            await self._restart_watchdog.trigger_restart(exit_fn=exit_fn)
        else:
            async def _delayed_exit() -> None:
                await asyncio.sleep(0.5)
                logger.warning("Agent restart requested via dashboard API; terminating process...")
                if callable(exit_fn):
                    exit_fn()
                    return
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                except Exception:
                    os._exit(0)

            asyncio.create_task(_delayed_exit(), name="agent-restart-delayed-exit")

        return web.json_response(
            {"ok": True, "message": "Agent restarting in 500ms, poll /api/agent/settings for recovery"}
        )

    @mcp_api_boundary
    async def _api_mcp_servers_get(self, request):  # noqa: ANN001
        from aiohttp import web
        from .mcp_manager import DashboardApiError
        if self._mcp_manager is None:
            raise DashboardApiError("SERVICE_UNAVAILABLE", "MCP manager not initialized")
        self._mcp_manager.verify_admin_token(request)
        self._mcp_manager.verify_csrf(request)

        bridge = next(
            (p for p in getattr(getattr(self, "app", None), "plugins", []) if getattr(p, "name", None) == "mcp_bridge"),
            None,
        )
        active_list = []
        for s in self._mcp_manager.active_servers_sanitized:
            entry = dict(s)
            if bridge is not None and hasattr(bridge, "get_server_runtime_status"):
                st = bridge.get_server_runtime_status(entry.get("name", ""))
                entry["status"] = st.get("status", "unknown")
                entry["connected"] = st.get("connected", False)
                entry["tools"] = st.get("tools", [])
                if st.get("error"):
                    raw_cfg = next((raw for raw in self._mcp_manager.active_servers if raw.get("name") == entry.get("name")), None)
                    entry["error"] = sanitize_error_text(str(st["error"]), server_cfg=raw_cfg)
            else:
                entry["status"] = "failed"
                entry["connected"] = False
                entry["tools"] = []
                entry["error"] = "McpBridgePlugin not active (missing 'mcp' dependency or disabled)"
            active_list.append(entry)

        return web.json_response({
            "ok": True,
            "pending_restart": self._mcp_manager.pending_restart,
            "active_servers": active_list,
            "servers": self._mcp_manager.persisted_servers_sanitized,
            "etag": self._mcp_manager.persisted_config_etag,
        })

    @mcp_api_boundary
    async def _api_mcp_servers_post(self, request):  # noqa: ANN001
        from aiohttp import web
        from .mcp_manager import DashboardApiError
        if self._mcp_manager is None:
            raise DashboardApiError("SERVICE_UNAVAILABLE", "MCP manager not initialized")
        self._mcp_manager.verify_admin_token(request)
        self._mcp_manager.verify_csrf(request)
        body = await request.json()
        server_data = body.get("server") if isinstance(body, dict) and "server" in body else body
        if not isinstance(server_data, dict):
            raise DashboardApiError("INVALID_PARAM", "Request body must contain server configuration object")
        if_match = request.headers.get("If-Match", "")
        await self._mcp_manager.add_or_update_server(server_data, if_match=if_match)
        return web.json_response({
            "ok": True,
            "pending_restart": self._mcp_manager.pending_restart,
            "etag": self._mcp_manager.persisted_config_etag,
            "servers": self._mcp_manager.persisted_servers_sanitized,
        })

    @mcp_api_boundary
    async def _api_mcp_servers_patch(self, request):  # noqa: ANN001
        from aiohttp import web
        from .mcp_manager import DashboardApiError
        if self._mcp_manager is None:
            raise DashboardApiError("SERVICE_UNAVAILABLE", "MCP manager not initialized")
        self._mcp_manager.verify_admin_token(request)
        self._mcp_manager.verify_csrf(request)
        name = request.match_info.get("name", "")
        if not name:
            raise DashboardApiError("INVALID_PARAM", "Missing server name")
        body = await request.json()
        patch_data = body.get("server") if isinstance(body, dict) and "server" in body else body
        if not isinstance(patch_data, dict):
            raise DashboardApiError("INVALID_PARAM", "Request body must contain server configuration object")
        if_match = request.headers.get("If-Match", "")
        await self._mcp_manager.patch_server(name, patch_data, if_match=if_match)
        return web.json_response({
            "ok": True,
            "pending_restart": self._mcp_manager.pending_restart,
            "etag": self._mcp_manager.persisted_config_etag,
            "servers": self._mcp_manager.persisted_servers_sanitized,
        })

    @mcp_api_boundary
    async def _api_mcp_servers_delete(self, request):  # noqa: ANN001
        from aiohttp import web
        from .mcp_manager import DashboardApiError
        if self._mcp_manager is None:
            raise DashboardApiError("SERVICE_UNAVAILABLE", "MCP manager not initialized")
        self._mcp_manager.verify_admin_token(request)
        self._mcp_manager.verify_csrf(request)
        name = request.match_info.get("name", "")
        if not name:
            raise DashboardApiError("INVALID_PARAM", "Missing server name")
        if_match = request.headers.get("If-Match", "")
        await self._mcp_manager.delete_server(name, if_match=if_match)
        return web.json_response({
            "ok": True,
            "pending_restart": self._mcp_manager.pending_restart,
            "etag": self._mcp_manager.persisted_config_etag,
            "servers": self._mcp_manager.persisted_servers_sanitized,
        })

    @mcp_api_boundary
    async def _api_mcp_test(self, request):  # noqa: ANN001
        from aiohttp import web
        from .mcp_manager import DashboardApiError
        if self._mcp_manager is None:
            raise DashboardApiError("SERVICE_UNAVAILABLE", "MCP manager not initialized")
        self._mcp_manager.verify_admin_token(request)
        self._mcp_manager.verify_csrf(request)
        body = await request.json()
        if not isinstance(body, dict):
            raise DashboardApiError("INVALID_PARAM", "Request body must be a JSON object")
        res = await self._mcp_manager.test_server(body)
        return web.json_response(res)

    @mcp_api_boundary
    async def _api_send_text(self, request):  # noqa: ANN001
        from aiohttp import web
        from .mcp_manager import DashboardApiError
        try:
            body = await request.json()
        except Exception as e:
            raise DashboardApiError("CONFIG_PARSE_ERROR", f"Bad JSON: {e}")
        text = (body or {}).get("text")
        if not isinstance(text, str) or not text.strip():
            raise DashboardApiError("INVALID_PARAM", "missing text")
        # Typed-text bypasses ASR, so no ASRFinal will clear the playback
        # discard latch a prior barge-in / abort / sleep may have armed.
        # Clear it explicitly so this turn's TTS is actually audible.
        try:
            arm = getattr(self.app.audio, "arm_for_next_turn", None)
            if callable(arm):
                arm()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            await self.app.slv.send_text(text)
            await self.app.slv.flush_tts()
            return web.json_response({"ok": True})
        except Exception as e:
            raise DashboardApiError("INTERNAL_ERROR", f"send_text failed: {e}")

    async def _api_inject_wav(self, request):  # noqa: ANN001
        """DEBUG-ONLY: POST raw WAV bytes → fed to the SLV as a spoken utterance.

        Gated behind ``OVS_AGENT_DEBUG_INJECT=1`` (default OFF) because in a
        server-loop deployment this reaches the LLM and therefore whatever the
        app's tools can actuate — lights, locks, an arm.
        """
        from aiohttp import web

        from ..debug_inject import inject_wav

        if os.environ.get("OVS_AGENT_DEBUG_INJECT") != "1":
            return web.json_response(
                {"ok": False, "error": "inject disabled; set "
                 "OVS_AGENT_DEBUG_INJECT=1 and restart the agent"},
                status=403,
            )
        try:
            result = await inject_wav(self.app, await request.read())
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except RuntimeError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=409)
        return web.json_response(result)

    async def _api_inject_user_text(self, request):  # noqa: ANN001
        # DEBUG-ONLY: bypasses ASR/wake-word for end-to-end testing without a mic.
        # Production deployments should gate this behind an auth check or env flag.
        """Debug-only: fake an ASRFinal so the dialogue runner (and any
        tool-call / preamble path) executes end-to-end without a real mic.

        Unlike /api/control/send_text (which only forwards the string to
        SLV's TTS-ish ``send_text``), this synthesises a real
        ``ASRFinal`` event and feeds it through ``_dispatch_one`` — i.e.
        exactly the entrypoint a server-VAD ASR final would hit. The
        result: full LLM turn, tool_call dispatch, preamble emission,
        arm action — same code paths a spoken utterance walks.
        """
        from aiohttp import web

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"}, status=400)
        text = (body or {}).get("text")
        if not isinstance(text, str) or not text.strip():
            return web.json_response(
                {"ok": False, "error": "missing text"}, status=422
            )
        language = (body or {}).get("language")
        if language is not None and not isinstance(language, str):
            return web.json_response(
                {"ok": False, "error": "language must be string"}, status=422
            )

        app = self.app
        cfg = getattr(app, "config", None)
        try:
            server_loop = bool(cfg.server_loop_enabled()) if cfg is not None else False
        except AttributeError:
            server_loop = bool(getattr(cfg, "server_loop", False))
        if server_loop:
            return web.json_response(
                {
                    "ok": False,
                    "status": "unsupported",
                    "error": (
                        "inject_user_text cannot exercise server-loop; use real "
                        "mic/ASR or a server-side injection path"
                    ),
                },
                status=409,
            )
        # Reject if a prior LLM turn is still in flight — injecting a
        # second utterance mid-turn would race the FSM and confuse the
        # preamble / tool-dispatch sequencing this endpoint exists to
        # validate.
        turn_task = getattr(app, "_llm_turn_task", None)
        if turn_task is not None and not turn_task.done():
            return web.json_response(
                {"ok": False, "status": "busy",
                 "error": "previous llm_turn still running"},
                status=409,
            )

        # Clear discard latch (typed-text bypass of ASR doesn't run the
        # normal arm_for_next_turn path in _dispatch_one's ASRFinal
        # branch — _dispatch_one does it itself, but harmless to do here
        # too in case state is mid-SPEAKING). Also wake if pipeline_mode
        # gated us into SLEEPING — _dispatch_one would otherwise drop the
        # event with no LLM turn.
        try:
            arm = getattr(app.audio, "arm_for_next_turn", None)
            if callable(arm):
                arm()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            if getattr(app, "_state", ConvState.IDLE) == ConvState.SLEEPING:
                wake = getattr(app, "wake", None)
                if callable(wake):
                    await wake(source="inject_user_text")
        except Exception:
            logger.debug("pre-inject wake failed", exc_info=True)

        try:
            from ..slv_client import ASRFinal
        except Exception as e:  # pragma: no cover - defensive
            return web.json_response(
                {"ok": False, "error": f"ASRFinal import failed: {e}"},
                status=500,
            )

        evt = ASRFinal(text=text, session_complete=False, language=language)
        logger.info(
            "inject_user_text: faking ASRFinal text=%r language=%r",
            text, language,
        )
        try:
            # Dispatch in a fire-and-forget task: _dispatch_one spawns
            # the LLM turn via asyncio.create_task and returns, so doing
            # this inline is fine. We still create_task so the HTTP
            # response isn't blocked by dispatcher work (broadcast etc).
            asyncio.create_task(
                app._dispatch_one(evt), name="inject-user-text"
            )
        except Exception as e:
            return web.json_response(
                {"ok": False, "error": str(e)}, status=500
            )
        return web.json_response(
            {"ok": True, "status": "dispatched", "text": text}
        )

    async def _api_modes_list(self, request):  # noqa: ANN001
        from aiohttp import web
        mgr = getattr(self.app, "modes", None)
        if mgr is None:
            return web.json_response([])
        try:
            return web.json_response(mgr.list_all())
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_mode_switch(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"}, status=400)
        name = (body or {}).get("name")
        if not isinstance(name, str) or not name.strip():
            return web.json_response({"ok": False, "error": "missing name"}, status=400)
        mgr = getattr(self.app, "modes", None)
        if mgr is None:
            return web.json_response(
                {"ok": False, "error": "app has no ModeManager"}, status=400
            )
        try:
            await mgr.switch(name)
            return web.json_response({"ok": True, "current": mgr.current_name})
        except KeyError:
            return web.json_response(
                {"ok": False, "error": f"unknown mode: {name}"}, status=404
            )
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    # ── per-mode override editor ────────────────────────────────
    _OVERRIDE_KEYS = ("system_prompt", "temperature", "max_history", "barge_in_enabled")
    _AGENT_SETTINGS_KEYS = ("pipeline_mode", "sleep_timeout_s", "stop_words")
    _PIPELINE_MODES = ("always_on", "wake_word", "push_to_talk")

    def _build_override_payload(self, mode) -> dict[str, Any]:
        class_default = {k: getattr(mode, k, None) for k in self._OVERRIDE_KEYS}
        overrides = getattr(self.app.config, "mode_overrides", None) or {}
        current = {}
        if isinstance(overrides, dict):
            mo = overrides.get(mode.name) or {}
            if isinstance(mo, dict):
                current = {k: mo[k] for k in mo if k in self._OVERRIDE_KEYS}
        effective = dict(class_default)
        for k, v in current.items():
            effective[k] = v
        return {
            "name": mode.name,
            "display_name": mode.display_name,
            "icon": mode.icon,
            "effective": effective,
            "class_default": class_default,
            "current_override": current,
        }

    async def _api_mode_overrides_get(self, request):  # noqa: ANN001
        from aiohttp import web
        name = request.match_info.get("name", "")
        mgr = getattr(self.app, "modes", None)
        if mgr is None:
            return web.json_response(
                {"ok": False, "error": "app has no ModeManager"}, status=400
            )
        mode = mgr.get(name)
        if mode is None:
            return web.json_response(
                {"ok": False, "error": f"unknown mode: {name}"}, status=404
            )
        return web.json_response(self._build_override_payload(mode))

    async def _api_mode_overrides_post(self, request):  # noqa: ANN001
        from aiohttp import web
        name = request.match_info.get("name", "")
        mgr = getattr(self.app, "modes", None)
        if mgr is None:
            return web.json_response(
                {"ok": False, "error": "app has no ModeManager"}, status=400
            )
        mode = mgr.get(name)
        if mode is None:
            return web.json_response(
                {"ok": False, "error": f"unknown mode: {name}"}, status=404
            )
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "body must be object"}, status=400)
        # Ensure mode_overrides dict exists and is mutable.
        cfg = self.app.config
        overrides = getattr(cfg, "mode_overrides", None)
        if not isinstance(overrides, dict):
            overrides = {}
            try:
                cfg.mode_overrides = overrides
            except Exception:
                pass
        mo = overrides.get(name)
        if not isinstance(mo, dict):
            mo = {}
        for k in self._OVERRIDE_KEYS:
            if k not in body:
                continue
            v = body[k]
            if v is None:
                mo.pop(k, None)
            else:
                mo[k] = v
        if mo:
            overrides[name] = mo
        else:
            overrides.pop(name, None)
        # Try to persist to YAML.
        persisted = False
        persist_error: str | None = None
        src = getattr(cfg, "_source_path", None)
        if src is not None:
            try:
                self._persist_overrides_to_yaml(Path(src), overrides)
                persisted = True
            except Exception as e:  # pragma: no cover - defensive
                persist_error = str(e)
                logger.exception("persist mode_overrides failed")
        # Broadcast plugin hook.
        try:
            await self.app.broadcast(
                "on_mode_override_change",
                {"name": name, "override": mo, "persisted": persisted},
            )
        except Exception:
            logger.exception("on_mode_override_change broadcast failed")
        payload = self._build_override_payload(mode)
        payload["persisted"] = persisted
        if persist_error is not None:
            payload["persist_error"] = persist_error
        return web.json_response(payload)

    @staticmethod
    def _persist_overrides_to_yaml(path: Path, overrides: dict[str, Any]) -> None:
        """Re-write the YAML config file, replacing only mode_overrides.

        Loads with safe_load, mutates, dumps with default_flow_style=False.
        Comments in the file will be lost (pyyaml limitation) — this is
        acceptable for a developer-tool endpoint.
        """
        import yaml

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config root not a mapping: {type(raw).__name__}")
        if overrides:
            raw["mode_overrides"] = overrides
        else:
            raw.pop("mode_overrides", None)
        # Write atomically: tmp + rename.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                raw, f,
                default_flow_style=False, allow_unicode=True, sort_keys=False,
            )
        _safe_replace(str(tmp), str(path))

    async def _api_wake(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            await self.app.wake(source="dashboard")
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_sleep(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            await self.app.sleep()
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_ptt_start(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            # Wake first (no-op if already awake), then jump straight to
            # LISTENING — bypassing the client-VAD speech_min wait so the
            # very first PCM chunk after the button-press is streamed.
            await self.app.wake(source="ptt")
            try:
                self.app._ptt_explicit_eos_pending = True
            except Exception:
                pass
            try:
                self.app._set_state(ConvState.LISTENING)
            except Exception:
                pass
            # Force VAD into "speech" mode so mic_pump starts streaming
            # immediately — without this, _update_vad would still wait
            # for client_vad_speech_min_ms of energy before transmitting.
            try:
                self.app._vad_state = "speech"
                self.app._vad_speech_ms = 0
                self.app._vad_silence_ms = 0
                self.app._vad_eos_sent = False
                # Allow asr_eos for the new turn (dedupe is per-turn).
                self.app._eos_sent_this_turn = False
            except Exception:
                pass
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_ptt_end(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            # Explicit end: send asr_eos to SLV (Paraformer will finalise
            # the buffered audio) and let the normal dispatch path take
            # care of THINKING → SPEAKING → IDLE → SLEEPING via the
            # auto-sleep timer.
            # Use the dedupe helper so a client VAD silence trigger +
            # this explicit end can't both fire asr_eos and race the
            # SLV state machine.
            try:
                helper = getattr(self.app, "send_asr_eos_once", None)
                if callable(helper):
                    await helper()
                else:
                    await self.app.slv.asr_eos()
            except Exception:
                logger.debug("ptt/end asr_eos send failed", exc_info=True)
            try:
                self.app._ptt_explicit_eos_pending = False
            except Exception:
                pass
            try:
                self.app._set_state(ConvState.THINKING)
                # Symmetric with the server-VAD / client-VAD THINKING
                # entries: arm the thinking-watchdog so a wedged SLV
                # TTS can't strand this PTT turn forever.
                arm = getattr(self.app, "_arm_thinking_watchdog", None)
                if callable(arm):
                    arm()
            except Exception:
                pass
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    # ── SLV TTS proxy (speakers + voice clone) ──────────────────

    def _slv_http_base(self) -> str:
        cfg = getattr(self.app, "config", None)
        base = getattr(cfg, "slv_http_base", None) if cfg is not None else None
        return base or "http://localhost:8621"

    def _admin_headers(self) -> dict[str, str]:
        """Inject X-Admin-Key from OVS_ADMIN_KEY env when set.

        Loopback SLV deployments don't need the key (admin_auth bypasses
        loopback unconditionally); remote deployments do. We always pass
        the header when the env var is set — harmless on loopback.
        """
        key = os.environ.get("OVS_ADMIN_KEY", "").strip()
        return {"X-Admin-Key": key} if key else {}

    async def _get_slv_http(self):
        """Lazily create aiohttp.ClientSession on first use."""
        if self._slv_http is None:
            import aiohttp
            self._slv_http = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30.0),
            )
        return self._slv_http

    async def _proxy_json(self, method: str, path: str, *,
                          json_body: Any = None,
                          admin: bool = False):
        """Forward a JSON request to SLV; return (status, payload).

        Payload is the parsed JSON, or {"error": <text>} if SLV's body
        isn't JSON.
        """
        from aiohttp import web  # noqa: F401 — needed by callers in same file
        sess = await self._get_slv_http()
        url = self._slv_http_base().rstrip("/") + path
        headers = self._admin_headers() if admin else {}
        try:
            async with sess.request(
                method, url, json=json_body, headers=headers,
            ) as resp:
                ct = resp.headers.get("Content-Type", "")
                if "application/json" in ct:
                    return resp.status, await resp.json()
                txt = await resp.text()
                return resp.status, {"error": txt or f"HTTP {resp.status}"}
        except Exception as e:
            logger.warning("SLV proxy %s %s failed: %s", method, path, e)
            return 502, {"error": f"SLV unreachable: {e}"}

    async def _api_tts_speakers_list(self, request):  # noqa: ANN001
        from aiohttp import web
        status, payload = await self._proxy_json("GET", "/tts/speakers")
        return web.json_response(payload, status=status)

    async def _api_tts_runtime_get(self, request):  # noqa: ANN001
        from aiohttp import web
        status, payload = await self._proxy_json(
            "GET", "/admin/tts/runtime", admin=True,
        )
        return web.json_response(payload, status=status)

    async def _api_tts_runtime_patch(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be object"}, status=400)
        # Whitelist fields so we don't smuggle arbitrary payloads upstream.
        allowed = {"speaker_id", "speed", "pitch_shift"}
        upstream = {k: body[k] for k in body if k in allowed}
        status, payload = await self._proxy_json(
            "PATCH", "/admin/tts/runtime", json_body=upstream, admin=True,
        )
        # SLV's v2v WS path resolves tts_speaker_id once from CLIENT_CONFIG
        # (main.py: "avoids mid-session changes affecting later sentences"),
        # so /admin/tts/runtime alone does NOT change the voice of the
        # ongoing WS conversation. We additionally mutate our own slv_config
        # and reconnect so the new speaker_id rides on the next CLIENT_CONFIG.
        # Also warm the worker for this speaker — qwen3-tts is deterministic
        # post-warm but the first synth per speaker differs ("cold worker"
        # state). Warming here trades ~600ms switch latency for stable voice
        # on the user's next utterance.
        if status < 400 and "speaker_id" in upstream:
            sid = upstream["speaker_id"]
            try:
                slv = getattr(self.app, "slv", None)
                if slv is not None and isinstance(slv.config, dict):
                    slv.config["tts_speaker_id"] = (
                        None if sid is None else int(sid)
                    )
                    # Warm + reconnect in parallel — reconnect doesn't touch
                    # the TTS worker, so they're independent.
                    await asyncio.gather(
                        self._warm_tts_speaker(sid),
                        slv.reconnect(),
                        return_exceptions=True,
                    )
            except Exception as e:
                logger.warning("post-patch slv reconnect failed: %s", e)
        return web.json_response(payload, status=status)

    async def _warm_tts_speaker(self, speaker_id) -> None:
        """One dummy /tts/stream call so qwen3-tts worker is warm for this
        speaker. After this, subsequent synths are byte-identical (verified
        via md5). Non-fatal — switch still proceeds if warmup fails."""
        if speaker_id is None:
            return
        try:
            import aiohttp
            sess = await self._get_slv_http()
            url = self._slv_http_base().rstrip("/") + "/tts/stream"
            body = {
                "text": "你好",
                "speaker_id": int(speaker_id),
                "language": "chinese",
            }
            async with sess.post(
                url, json=body, timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                # Drain so the connection returns to the pool.
                await resp.read()
                logger.info(
                    "tts warmup for speaker %s: HTTP %s", speaker_id, resp.status,
                )
        except Exception as e:
            logger.info("tts warmup for speaker %s skipped: %s", speaker_id, e)

    async def _api_tts_clone_embedding(self, request):  # noqa: ANN001
        """Forward a multipart WAV upload to SLV's /tts/clone/embedding."""
        from aiohttp import web, FormData
        # Re-package the incoming multipart as a fresh FormData so aiohttp
        # owns the boundary + Content-Length.
        try:
            reader = await request.multipart()
        except Exception as e:
            return web.json_response({"error": f"bad multipart: {e}"}, status=400)
        field = None
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "file":
                field = part
                break
        if field is None:
            return web.json_response({"error": "missing 'file' field"}, status=400)
        data = await field.read(decode=False)
        if not data:
            return web.json_response({"error": "empty file"}, status=400)

        sess = await self._get_slv_http()
        url = self._slv_http_base().rstrip("/") + "/tts/clone/embedding"
        form = FormData()
        form.add_field(
            "file", data,
            filename=field.filename or "reference.wav",
            content_type=field.headers.get("Content-Type", "audio/wav"),
        )
        try:
            async with sess.post(url, data=form) as resp:
                ct = resp.headers.get("Content-Type", "")
                if "application/json" in ct:
                    return web.json_response(await resp.json(), status=resp.status)
                txt = await resp.text()
                return web.json_response(
                    {"error": txt or f"HTTP {resp.status}"}, status=resp.status,
                )
        except Exception as e:
            logger.warning("clone/embedding proxy failed: %s", e)
            return web.json_response(
                {"error": f"SLV unreachable: {e}"}, status=502,
            )

    async def _api_tts_speakers_register(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be object"}, status=400)
        status, payload = await self._proxy_json(
            "POST", "/tts/speakers/register", json_body=body,
        )
        return web.json_response(payload, status=status)

    async def _api_tts_speakers_delete(self, request):  # noqa: ANN001
        from aiohttp import web
        sid = request.match_info.get("speaker_id", "")
        if not sid:
            return web.json_response({"error": "missing speaker_id"}, status=400)
        status, payload = await self._proxy_json(
            "DELETE", f"/tts/speakers/{sid}",
        )
        return web.json_response(payload, status=status)

    # ── errors / agent-settings endpoints ───────────────────────

    async def _api_errors_clear(self, request):  # noqa: ANN001
        from aiohttp import web
        self._errors.clear()
        try:
            await self._broadcast("errors_cleared", None)
        except Exception:
            pass
        return web.json_response({"ok": True})

    _AGENT_SETTINGS_KEYS = _AGENT_SETTINGS_KEYS

    def _build_agent_settings_payload(self) -> dict[str, Any]:
        cfg = getattr(self.app, "config", None)
        pipeline_mode = "always_on"
        active = getattr(self.app, "_active_pipeline_mode", None)
        if not isinstance(active, str):
            active = getattr(self.app, "pipeline_mode", None)
        if not isinstance(active, str):
            active = self._startup_pipeline_mode
        pipeline_mode_active = active if isinstance(active, str) else None
        sleep_timeout_s = 30.0
        stop_words: list[str] = []
        wake_reply_text = ""
        exit_words: list[str] = list(DEFAULT_EXIT_WORDS)
        exit_reply_text = ""
        if cfg is not None:
            pipeline_mode = getattr(cfg, "pipeline_mode", "always_on")
            if not self._startup_pipeline_mode and isinstance(pipeline_mode, str):
                self._startup_pipeline_mode = pipeline_mode
            if not pipeline_mode_active:
                pipeline_mode_active = self._startup_pipeline_mode or pipeline_mode
            try:
                sleep_timeout_s = float(getattr(cfg, "sleep_timeout_s", 30.0))
            except Exception:
                sleep_timeout_s = 30.0
            raw_sw = getattr(cfg, "stop_words", None)
            if isinstance(raw_sw, (list, tuple)):
                stop_words = [str(x) for x in raw_sw]
            elif raw_sw is not None:
                stop_words = []
            raw_ew = getattr(cfg, "exit_words", None)
            if isinstance(raw_ew, (list, tuple)):
                exit_words = [str(x) for x in raw_ew]
            elif raw_ew is None:
                exit_words = list(DEFAULT_EXIT_WORDS)
            
            raw_wrt = getattr(cfg, "wake_reply_text", "")
            if not raw_wrt:
                meta = getattr(cfg, "metadata", {}) or {}
                raw_wrt = meta.get("wake_reply_text") or meta.get("wakeword", {}).get("reply_text", "")
            wake_reply_text = self._reply_text_str(raw_wrt)

            raw_ert = getattr(cfg, "exit_reply_text", "")
            if not raw_ert:
                meta = getattr(cfg, "metadata", {}) or {}
                raw_ert = meta.get("exit_reply_text", "")
            exit_reply_text = self._reply_text_str(raw_ert)
        else:
            if not pipeline_mode_active:
                pipeline_mode_active = pipeline_mode

        restart_failed = False
        restart_error = ""
        if self._restart_watchdog is not None:
            if getattr(self._restart_watchdog, "is_restarting", "") == "restart_failed":
                restart_failed = True
                restart_error = "Watchdog detected restart termination failure"

        return {
            "instance_id": self._instance_id,
            "restart_failed": restart_failed,
            "restart_error": restart_error,
            "pipeline_mode": pipeline_mode,
            "pipeline_mode_active": pipeline_mode_active,
            "sleep_timeout_s": sleep_timeout_s,
            "stop_words": stop_words,
            "wake_reply_text": wake_reply_text,
            "exit_words": exit_words,
            "exit_reply_text": exit_reply_text,
        }

    @staticmethod
    def _reply_text_str(raw: Any) -> str:
        # Config also accepts a list (random choice per wake); the dashboard
        # edits a single line, so show the alternatives comma-joined.
        if isinstance(raw, (list, tuple)):
            return ", ".join(str(x).strip() for x in raw if str(x).strip())
        return str(raw or "")

    def _runtime_kws(self):  # noqa: ANN001
        for plugin in getattr(self.app, "plugins", []) or []:
            if getattr(plugin, "name", "") == "runtime_kws":
                return plugin
        return None

    async def _api_wakeword_runtime_get(self, request):  # noqa: ANN001
        from aiohttp import web
        source = self._runtime_kws()
        if source is None:
            return web.json_response({
                "available": False,
                "enabled": False,
                "running": False,
                "phrases": [],
            })
        payload = dict(source.status())
        payload["enabled"] = True
        return web.json_response(payload)

    async def _api_wakeword_validate(self, request):  # noqa: ANN001
        from aiohttp import web
        source = self._runtime_kws()
        if source is None:
            return web.json_response(
                {"ok": False, "error": "runtime wake-word backend is not enabled"},
                status=409,
            )
        try:
            body = await request.json()
            phrases = body.get("phrases") if isinstance(body, dict) else None
            if not isinstance(phrases, list):
                raise ValueError("phrases must be a list")
            normalised = await source.validate_phrases(phrases)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "phrases": list(normalised)})

    async def _api_wakeword_runtime_patch(self, request):  # noqa: ANN001
        from aiohttp import web
        source = self._runtime_kws()
        if source is None:
            return web.json_response(
                {"ok": False, "error": "runtime wake-word backend is not enabled"},
                status=409,
            )
        try:
            body = await request.json()
            phrases = body.get("phrases") if isinstance(body, dict) else None
            if not isinstance(phrases, list):
                raise ValueError("phrases must be a list")
            normalised = await source.update_phrases(phrases)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        persisted = False
        persist_error = None
        try:
            self._persist_runtime_wakeword(list(normalised))
            persisted = True
        except Exception as exc:
            # Hot update has already succeeded. A read-only library deployment
            # should keep working now and clearly report that restart will
            # restore the configured phrase.
            persist_error = str(exc)
            logger.warning("runtime wake-word state was not persisted: %s", exc)
        payload = dict(source.status())
        payload.update({"ok": True, "enabled": True, "persisted": persisted})
        if persist_error:
            payload["persist_error"] = persist_error
        return web.json_response(payload)

    def _persist_runtime_wakeword(self, phrases: list[str]) -> None:
        meta = getattr(getattr(self.app, "config", None), "metadata", {}) or {}
        wake = meta.get("wakeword", {}) or {}
        path = Path(str(wake.get("state_path") or "/var/lib/ovs-agent/wakeword.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(tmp, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, "phrases": phrases}, stream, ensure_ascii=False)
                stream.write("\n")
            os.chmod(tmp, 0o600)
            _safe_replace(str(tmp), str(path))
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    async def _api_wakeword_test_tone(self, request):  # noqa: ANN001
        from aiohttp import web
        play = getattr(self.app, "_play_wake_tone", None)
        if not callable(play):
            return web.json_response({"ok": False, "error": "wake tone unavailable"}, status=409)
        play()
        return web.json_response({"ok": True})

    async def _api_agent_settings_get(self, request):  # noqa: ANN001
        from aiohttp import web
        return web.json_response(self._build_agent_settings_payload())

    async def _api_agent_settings_post(self, request):  # noqa: ANN001
        import math
        from aiohttp import web

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "body must be object"}, status=400)
        cfg = getattr(self.app, "config", None)
        if cfg is None:
            return web.json_response({"ok": False, "error": "no config"}, status=500)

        # Global pre-validation: validate all fields first.
        # If any field is invalid, immediately return HTTP 400 without updating cfg.
        for k in body:
            if k not in _AGENT_SETTINGS_KEYS:
                return web.json_response(
                    {"ok": False, "error": f"unrecognized field: {k}"}, status=400
                )

        normalized: dict[str, Any] = {}

        if "pipeline_mode" in body:
            pm = body["pipeline_mode"]
            if not isinstance(pm, str) or pm not in self._PIPELINE_MODES:
                return web.json_response(
                    {"ok": False, "error": f"pipeline_mode must be one of {list(self._PIPELINE_MODES)}"},
                    status=400,
                )
            normalized["pipeline_mode"] = pm

        if "sleep_timeout_s" in body:
            st = body["sleep_timeout_s"]
            if isinstance(st, bool) or not isinstance(st, (int, float)):
                return web.json_response(
                    {"ok": False, "error": "sleep_timeout_s must be a number (not bool)"},
                    status=400,
                )
            if not math.isfinite(st):
                return web.json_response(
                    {"ok": False, "error": "sleep_timeout_s must be a finite number"},
                    status=400,
                )
            if not (0.0 <= st <= 86400.0):
                return web.json_response(
                    {"ok": False, "error": "sleep_timeout_s must be between 0.0 and 86400.0"},
                    status=400,
                )
            normalized["sleep_timeout_s"] = 0.0 if st == 0.0 else float(st)

        for text_field in ("wake_reply_text", "exit_reply_text"):
            if text_field in body:
                tf = body[text_field]
                if not isinstance(tf, str):
                    return web.json_response(
                        {"ok": False, "error": f"{text_field} must be a string"},
                        status=400,
                    )
                tf_clean = tf.strip()
                if len(tf_clean) > 200 or len(tf) > 200:
                    return web.json_response(
                        {"ok": False, "error": f"{text_field} length must be <= 200"},
                        status=400,
                    )
                normalized[text_field] = tf_clean

        for list_field in ("stop_words", "exit_words"):
            if list_field in body:
                lw = body[list_field]
                if not isinstance(lw, list):
                    return web.json_response(
                        {"ok": False, "error": f"{list_field} must be a list"},
                        status=400,
                    )
                cleaned: list[str] = []
                seen: set[str] = set()
                for item in lw:
                    if not isinstance(item, str):
                        return web.json_response(
                            {"ok": False, "error": f"{list_field} items must be strings"},
                            status=400,
                        )
                    item_s = item.strip()
                    if not item_s:
                        continue
                    if len(item_s) > 50 or len(item) > 50:
                        return web.json_response(
                            {"ok": False, "error": f"{list_field} item length must be <= 50"},
                            status=400,
                        )
                    if item_s not in seen:
                        seen.add(item_s)
                        cleaned.append(item_s)
                if len(cleaned) > 100:
                    return web.json_response(
                        {"ok": False, "error": f"{list_field} cannot exceed 100 items"},
                        status=400,
                    )
                normalized[list_field] = cleaned

        # Compute changed: normalized values that differ from current cfg,
        # were left unsaved by an earlier failure, or differ from on-disk configuration.
        src = getattr(cfg, "_source_path", None)
        current = self._build_agent_settings_payload()
        changed: list[str] = []
        for k in _AGENT_SETTINGS_KEYS:
            if k in normalized:
                differs_runtime = (normalized[k] != current.get(k))
                is_unsaved = (k in self._unsaved_agent_settings_fields)
                if differs_runtime or is_unsaved:
                    changed.append(k)
                elif src is not None and os.path.exists(src):
                    try:
                        import yaml
                        with open(src, "r", encoding="utf-8") as f:
                            disk_raw = yaml.safe_load(f) or {}
                        if isinstance(disk_raw, dict) and k in disk_raw:
                            disk_val = disk_raw.get(k)
                            if k in ("wake_reply_text", "exit_reply_text"):
                                disk_str = self._reply_text_str(disk_val)
                                if disk_str != normalized[k]:
                                    changed.append(k)
                                    self._unsaved_agent_settings_fields.add(k)
                            elif disk_val != normalized[k]:
                                changed.append(k)
                                self._unsaved_agent_settings_fields.add(k)
                    except Exception:
                        pass

        if not changed:
            payload = self._build_agent_settings_payload()
            payload["ok"] = True
            payload["persisted"] = False
            payload["changed"] = []
            return web.json_response(payload)

        # Apply updates to cfg
        for k in changed:
            val = normalized[k]
            if k in ("wake_reply_text", "exit_reply_text") and isinstance(val, str) and "," in val:
                items = [x.strip() for x in val.split(",") if x.strip()]
                if len(items) > 1:
                    val = items
            setattr(cfg, k, val)

        # If spoken reply texts are explicitly disabled (cleared to empty),
        # clear any fallback in metadata so it cannot resurrect the reply.
        meta = getattr(cfg, "metadata", None)
        if isinstance(meta, dict):
            if "wake_reply_text" in changed and not normalized.get("wake_reply_text"):
                meta.pop("wake_reply_text", None)
                if isinstance(meta.get("wakeword"), dict):
                    meta["wakeword"].pop("reply_text", None)
            if "exit_reply_text" in changed and not normalized.get("exit_reply_text"):
                meta.pop("exit_reply_text", None)

        # Invalidate stop words matcher cache only if word lists changed
        if "stop_words" in changed or "exit_words" in changed:
            try:
                self.app._stop_words_cache = None
            except Exception:
                pass

        persisted = False
        persist_error: str | None = None
        src = getattr(cfg, "_source_path", None)
        if src is not None and changed:
            try:
                self._persist_agent_settings_to_yaml(
                    Path(src), {k: normalized[k] for k in changed}
                )
                persisted = True
                self._unsaved_agent_settings_fields.difference_update(changed)
            except Exception as e:
                persist_error = str(e)
                persisted = False
                self._unsaved_agent_settings_fields.update(changed)
                logger.warning("persist agent_settings failed: %s", e)

        payload = self._build_agent_settings_payload()
        payload["ok"] = True
        payload["persisted"] = persisted
        payload["changed"] = list(changed)
        if persist_error is not None:
            payload["persist_error"] = persist_error

        try:
            await self.app.broadcast(
                "on_agent_settings_change",
                dict(payload),
            )
        except Exception:
            logger.exception("on_agent_settings_change broadcast failed")

        return web.json_response(payload)

    @staticmethod
    def _persist_agent_settings_to_yaml(path: Path, updated: dict[str, Any]) -> None:
        """Re-write the YAML config replacing only the changed agent-setting keys."""
        import yaml

        path = Path(path)
        dirpath = str(path.parent)
        if check_persistent_recovery_marker(dirpath):
            raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "System locked into read-only due to recovery marker")

        read_fd = _safe_open_readonly(str(path))
        try:
            prior_bytes = _read_all_bounded(read_fd)
        finally:
            _safe_close(read_fd, raise_on_error=True)

        raw = yaml.safe_load(prior_bytes) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config root not a mapping: {type(raw).__name__}")
        for k, v in updated.items():
            if k in ("wake_reply_text", "exit_reply_text") and isinstance(v, str) and "," in v:
                items = [x.strip() for x in v.split(",") if x.strip()]
                if len(items) > 1:
                    v = items
            raw[k] = v
            # If reply text is cleared, also remove fallback keys from metadata in YAML
            if k == "wake_reply_text" and not v and isinstance(raw.get("metadata"), dict):
                raw["metadata"].pop("wake_reply_text", None)
                if isinstance(raw["metadata"].get("wakeword"), dict):
                    raw["metadata"]["wakeword"].pop("reply_text", None)
            elif k == "exit_reply_text" and not v and isinstance(raw.get("metadata"), dict):
                raw["metadata"].pop("exit_reply_text", None)

        serialized = yaml.safe_dump(
            raw,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ).encode("utf-8")

        tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"
        fd = -1
        replaced = False
        try:
            fd = _open_atomic_tmp(tmp_path)
            _write_all(fd, serialized)
            _safe_fsync_file(fd)
            _safe_close(fd, raise_on_error=True)
            fd = -1

            _safe_replace(tmp_path, str(path))
            replaced = True

            verify_fd = _safe_open_readonly(str(path))
            try:
                st = os.fstat(verify_fd)
                if stat.S_IMODE(st.st_mode) != 0o600:
                    raise SecurityError("Failed to enforce 0600 mode on config file")
            finally:
                _safe_close(verify_fd, raise_on_error=True)

            _safe_fsync_dir(str(path.parent))
        except Exception:
            if fd >= 0:
                _safe_close(fd)
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError as unl_err:
                if unl_err.errno != errno.ENOENT:
                    logger.critical("FATAL: Failed to unlink tmp file %s: %s", tmp_path, unl_err)
                    raise ConfigPersistenceError(
                        "TMP_CLEANUP_FAILED",
                        f"Failed to clean up temporary file {tmp_path}: {unl_err}",
                        unl_err.errno,
                    ) from unl_err

            if not replaced:
                # If replacement failed (e.g. during bind-mount copy), check if target was partially written
                try:
                    if os.path.exists(path):
                        v_fd = _safe_open_readonly(str(path))
                        try:
                            cur_bytes = _read_all_bounded(v_fd)
                        finally:
                            _safe_close(v_fd)
                        if cur_bytes != prior_bytes:
                            # Target content diverged; force rollback to restore prior_bytes
                            replaced = True
                except Exception:
                    replaced = True

            if replaced:
                # Replacement succeeded or target was partially written but verification/copy failed.
                # Must perform safe rollback to restore prior_bytes.
                restore_tmp = f"{path}.tmp.{uuid.uuid4().hex}"
                r_fd = -1
                try:
                    r_fd = _open_atomic_tmp(restore_tmp)
                    _write_all(r_fd, prior_bytes)
                    _safe_fsync_file(r_fd)
                    _safe_close(r_fd, raise_on_error=True)
                    r_fd = -1
                    _safe_replace(restore_tmp, str(path))
                    _safe_fsync_dir(str(path.parent))
                except Exception as rb_err:
                    logger.critical("FATAL: Rollback failed for %s: %s", path, rb_err)
                    try:
                        write_persistent_recovery_marker(str(path.parent), "rollback_failed")
                    except (Exception, SystemExit) as marker_err:
                        logger.critical("FATAL: Failed to write recovery marker during rollback failure: %s", marker_err)
                    raise ConfigPersistenceError(
                        "CONFIG_RECOVERY_REQUIRED",
                        f"Target configuration replaced but rollback failed: {rb_err}",
                    ) from rb_err
                finally:
                    if r_fd >= 0:
                        _safe_close(r_fd)
                    try:
                        if os.path.exists(restore_tmp):
                            os.unlink(restore_tmp)
                    except OSError as rb_unl_err:
                        if rb_unl_err.errno != errno.ENOENT:
                            logger.critical("FATAL: Failed to unlink restore tmp file %s: %s", restore_tmp, rb_unl_err)
                            raise ConfigPersistenceError(
                                "TMP_CLEANUP_FAILED",
                                f"Failed to clean up temporary file {restore_tmp}: {rb_unl_err}",
                                rb_unl_err.errno,
                            ) from rb_unl_err
            raise

    # ── translator runtime endpoints (Phase 2a) ──────────────────
    #
    # GET  /api/translator/runtime   → current backend / src / tgt + the
    #                                  full list of supported FLORES
    #                                  targets (from translator/lang_map).
    # PATCH /api/translator/runtime  → mutate target language at runtime.
    #
    # Only ``tgt_lang`` is mutable today. ``src_lang`` is decided
    # per-utterance from the ASR-detected language (with the config
    # default as fallback). ``backend`` / ``url`` / ``timeout_s`` are
    # set at process start and not hot-swappable here.

    def _build_translator_runtime_payload(self) -> dict[str, Any]:
        from ..translator.lang_map import supported_target_languages
        cfg = getattr(self.app, "config", None)
        backend = "noop"
        src_lang = "zho_Hans"
        tgt_lang = "eng_Latn"
        url = "http://localhost:9001"
        timeout_s = 5.0
        if cfg is not None:
            backend = str(getattr(cfg, "translator_backend", backend))
            src_lang = str(getattr(cfg, "translator_src_lang", src_lang))
            tgt_lang = str(getattr(cfg, "translator_tgt_lang", tgt_lang))
            url = str(getattr(cfg, "translator_url", url))
            try:
                timeout_s = float(getattr(cfg, "translator_timeout_s", timeout_s))
            except Exception:
                timeout_s = 5.0
        return {
            "backend": backend,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "url": url,
            "timeout_s": timeout_s,
            "supported_targets": supported_target_languages(),
        }

    async def _api_translator_runtime_get(self, request):  # noqa: ANN001
        from aiohttp import web
        return web.json_response(self._build_translator_runtime_payload())

    async def _api_translator_runtime_patch(self, request):  # noqa: ANN001
        import re

        from aiohttp import web

        from ..translator.lang_map import FLORES_DISPLAY_NAMES

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "bad json"}, status=400
            )
        if not isinstance(body, dict):
            return web.json_response(
                {"ok": False, "error": "body must be object"}, status=400
            )
        if "tgt_lang" not in body:
            return web.json_response(
                {"ok": False, "error": "tgt_lang is required"}, status=400
            )
        tgt = body["tgt_lang"]
        if not isinstance(tgt, str) or not re.match(
            r"^[a-z]{3}_[A-Z][a-z]{3}$", tgt
        ):
            return web.json_response(
                {"ok": False,
                 "error": "tgt_lang must match NLLB format (e.g. 'eng_Latn')"},
                status=400,
            )
        if tgt not in FLORES_DISPLAY_NAMES:
            return web.json_response(
                {"ok": False,
                 "error": f"tgt_lang {tgt!r} not in supported targets"},
                status=400,
            )
        cfg = getattr(self.app, "config", None)
        if cfg is None:
            return web.json_response(
                {"ok": False, "error": "no config"}, status=500
            )
        old = getattr(cfg, "translator_tgt_lang", None)
        cfg.translator_tgt_lang = tgt
        # Persist to YAML if we have a source path (same pattern as
        # _api_agent_settings_post). Restart-survives the runtime swap.
        persisted = False
        persist_error: str | None = None
        src = getattr(cfg, "_source_path", None)
        if src is not None and old != tgt:
            try:
                self._persist_agent_settings_to_yaml(
                    Path(src), {"translator_tgt_lang": tgt}
                )
                persisted = True
            except Exception as e:  # pragma: no cover - defensive
                persist_error = str(e)
                logger.exception("persist translator_tgt_lang failed")
        logger.info(
            "translator_tgt_lang switched: %r → %r (persisted=%s)",
            old, tgt, persisted,
        )
        # Broadcast so any subscribed dashboard reflects the change live.
        bcast_payload: dict[str, Any] = {
            **self._build_translator_runtime_payload(),
            "persisted": persisted,
        }
        if persist_error is not None:
            bcast_payload["persist_error"] = persist_error
        try:
            await self.app.broadcast(
                "on_translator_runtime_change", bcast_payload
            )
        except Exception:
            logger.exception("on_translator_runtime_change broadcast failed")
        resp = {"ok": True, "tgt_lang": tgt, "persisted": persisted}
        if persist_error is not None:
            resp["persist_error"] = persist_error
        return web.json_response(resp)

    # ── LLM endpoint configuration and connectivity probe (v2.2) ──

    _LLM_ALLOWED_BACKENDS = ("edge_llm", "openai", "openai_compat", "noop")
    _LLM_ALLOWED_KEYS = {"llm_backend", "llm_base_url", "llm_model", "llm_api_key"}

    def _verify_csrf(self, request: web.Request) -> str | None:
        """Verify Origin and Referer headers against Host."""
        req_host = request.headers.get("Host", "").strip()
        if not req_host:
            return "CSRF check failed: missing Host header"

        origin = request.headers.get("Origin")
        referer = request.headers.get("Referer")

        if not origin and not referer:
            return "CSRF check failed: missing Origin and Referer headers"

        if origin:
            try:
                origin_host = urlsplit(origin).netloc
            except Exception:
                return "CSRF check failed: malformed Origin header"
            if not origin_host or origin_host != req_host:
                return f"CSRF check failed: Origin host {origin_host!r} != Host {req_host!r}"

        if referer:
            try:
                referer_host = urlsplit(referer).netloc
            except Exception:
                return "CSRF check failed: malformed Referer header"
            if not referer_host or referer_host != req_host:
                return f"CSRF check failed: Referer host {referer_host!r} != Host {req_host!r}"

        return None

    @staticmethod
    def _origin_tuple(url: str | None) -> tuple[str, str, int] | None:
        if not url or not isinstance(url, str):
            return None
        try:
            split = urlsplit(url)
            scheme = split.scheme.lower()
            hostname = (split.hostname or "").lower()
            if not scheme or not hostname:
                return None
            port = split.port or (443 if scheme == "https" else 80)
            return (scheme, hostname, port)
        except Exception:
            return None

    @classmethod
    def _validate_llm_payload(cls, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise ValueError("Payload must be a JSON object")

        unknown = set(body.keys()) - cls._LLM_ALLOWED_KEYS
        if unknown:
            raise ValueError(f"Unknown fields in payload: {sorted(unknown)}")

        for req in ("llm_backend", "llm_base_url", "llm_model"):
            if req not in body:
                raise ValueError(f"Missing required field: {req}")

        backend = body["llm_backend"]
        if not isinstance(backend, str) or backend not in cls._LLM_ALLOWED_BACKENDS:
            raise ValueError(
                f"Invalid llm_backend: {backend!r}. Must be one of {cls._LLM_ALLOWED_BACKENDS}"
            )

        base_url = body["llm_base_url"]
        if not isinstance(base_url, str):
            raise ValueError("llm_base_url must be a string")
        base_url = base_url.strip()
        if not base_url:
            raise ValueError("llm_base_url cannot be empty")

        try:
            split = urlsplit(base_url)
        except Exception as e:
            raise ValueError(f"Invalid llm_base_url: {e}")

        if split.scheme.lower() not in ("http", "https"):
            raise ValueError(
                f"Invalid llm_base_url scheme: {split.scheme!r}. Must be http or https"
            )

        if not split.netloc:
            raise ValueError("llm_base_url netloc must not be empty")

        if split.username is not None or split.password is not None or "@" in split.netloc:
            raise ValueError("llm_base_url must not contain userinfo")

        try:
            port = split.port
        except ValueError as e:
            raise ValueError(f"Invalid port in llm_base_url: {e}")

        if port is not None and not (1 <= port <= 65535):
            raise ValueError(f"llm_base_url port out of range (1..65535): {port}")

        if not split.hostname:
            raise ValueError("llm_base_url must contain a valid hostname")

        normalized_url = base_url.rstrip("/")

        model = body["llm_model"]
        if not isinstance(model, str):
            raise ValueError("llm_model must be a string")
        model = model.strip()
        if not (1 <= len(model) <= 128):
            raise ValueError("llm_model length must be between 1 and 128 characters")

        api_key = body.get("llm_api_key")
        if api_key is not None:
            if not isinstance(api_key, str):
                raise ValueError("llm_api_key must be a string or null")
            if len(api_key) > 512:
                raise ValueError("llm_api_key length exceeds 512 characters")

        return {
            "llm_backend": backend,
            "llm_base_url": normalized_url,
            "llm_model": model,
            "llm_api_key": api_key,
        }

    async def _api_llm_config_get(self, request: web.Request) -> web.Response:
        from aiohttp import web

        cfg = getattr(self.app, "config", None)
        fn = getattr(cfg, "server_loop_enabled", None)
        server_loop = bool(fn()) if callable(fn) else bool(getattr(cfg, "server_loop", False))
        backend = str(getattr(cfg, "llm_backend", "edge_llm") or "edge_llm")
        base_url = str(getattr(cfg, "llm_base_url", "http://localhost:8000/v1") or "http://localhost:8000/v1")
        model = str(getattr(cfg, "llm_model", "qwen2.5-3b-instruct") or "qwen2.5-3b-instruct")
        saved_key = getattr(cfg, "llm_api_key", None)
        has_api_key = bool(saved_key and str(saved_key).strip() and str(saved_key).strip() != "EMPTY")
        api_key_masked = "●●●●●●" if has_api_key else ""
        pending_restart = getattr(self, "_llm_config_pending_restart", False)

        return web.json_response({
            "ok": True,
            "llm_backend": backend,
            "llm_base_url": base_url,
            "llm_model": model,
            "has_api_key": has_api_key,
            "api_key_masked": api_key_masked,
            "pending_restart": pending_restart,
            "server_loop": server_loop,
            "editable": not server_loop,
        })

    async def _api_llm_config_post(self, request: web.Request) -> web.Response:
        from aiohttp import web

        csrf_err = self._verify_csrf(request)
        if csrf_err:
            return web.json_response({"ok": False, "error": csrf_err}, status=403)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body"}, status=400)

        try:
            validated = self._validate_llm_payload(body)
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)

        if self._llm_config_lock is None:
            self._llm_config_lock = asyncio.Lock()

        async with self._llm_config_lock:
            cfg = getattr(self.app, "config", None)
            if cfg is None:
                return web.json_response({"ok": False, "error": "配置未加载"}, status=500)

            fn = getattr(cfg, "server_loop_enabled", None)
            server_loop = bool(fn()) if callable(fn) else bool(getattr(cfg, "server_loop", False))
            if server_loop:
                return web.json_response(
                    {
                        "ok": False,
                        "error": "当前运行在服务端回环模式（server_loop: true），LLM 端点由语音服务端统一管理，禁止在客户端修改",
                    },
                    status=403,
                )

            src = getattr(cfg, "_source_path", None)
            if not src or not Path(src).is_file():
                return web.json_response(
                    {"ok": False, "error": "配置文件路径不可用，无法将端点改动持久化至磁盘以备重启"},
                    status=400,
                )

            current_backend = getattr(cfg, "llm_backend", "edge_llm")
            current_url = getattr(cfg, "llm_base_url", "http://localhost:8000/v1")
            current_model = getattr(cfg, "llm_model", "qwen2.5-3b-instruct")
            current_key = getattr(cfg, "llm_api_key", "EMPTY")
            if current_key is None:
                current_key = "EMPTY"

            new_backend = validated["llm_backend"]
            new_url = validated["llm_base_url"]
            new_model = validated["llm_model"]
            raw_key = validated["llm_api_key"]

            saved_origin = self._origin_tuple(current_url)
            new_origin = self._origin_tuple(new_url)
            dest_matched = (
                saved_origin is not None
                and new_origin is not None
                and saved_origin == new_origin
            )

            key_cleared = False
            if raw_key is None or raw_key == "●●●●●●":
                if dest_matched:
                    final_key = current_key
                else:
                    final_key = "EMPTY"
                    if current_key and current_key != "EMPTY":
                        key_cleared = True
            elif raw_key == "" or raw_key.strip() == "EMPTY":
                final_key = "EMPTY"
                if current_key and current_key != "EMPTY":
                    key_cleared = True
            else:
                final_key = raw_key.strip()

            changed: list[str] = []
            if new_backend != current_backend:
                changed.append("llm_backend")
            if new_url != current_url:
                changed.append("llm_base_url")
            if new_model != current_model:
                changed.append("llm_model")
            if final_key != current_key:
                changed.append("llm_api_key")

            if not changed:
                return web.json_response({
                    "ok": True,
                    "changed": [],
                    "pending_restart": self._llm_config_pending_restart,
                    "key_cleared": False,
                })

            field_map = {
                "llm_backend": new_backend,
                "llm_base_url": new_url,
                "llm_model": new_model,
                "llm_api_key": final_key,
            }
            updated_fields = {k: field_map[k] for k in changed}

            try:
                self._persist_agent_settings_to_yaml(Path(src), updated_fields)
            except Exception as e:
                logger.error("Failed to persist LLM config to %s: %s", src, type(e).__name__)
                return web.json_response({"ok": False, "error": "持久化失败: 无法写入配置文件，请检查磁盘与权限"}, status=500)

            for k in changed:
                setattr(cfg, k, field_map[k])
            self._llm_config_pending_restart = True

            bcast_payload: dict[str, Any] = {
                "ok": True,
                "changed": changed,
                "llm_backend": cfg.llm_backend,
                "llm_base_url": cfg.llm_base_url,
                "llm_model": cfg.llm_model,
                "has_api_key": bool(cfg.llm_api_key and cfg.llm_api_key != "EMPTY"),
                "pending_restart": self._llm_config_pending_restart,
                "key_cleared": key_cleared,
            }
            try:
                if hasattr(self.app, "broadcast"):
                    res = self.app.broadcast("on_llm_config_change", bcast_payload)
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                logger.exception("on_llm_config_change broadcast failed")

            return web.json_response({
                "ok": True,
                "changed": changed,
                "pending_restart": self._llm_config_pending_restart,
                "key_cleared": key_cleared,
            })

    async def _api_llm_test_post(self, request: web.Request) -> web.Response:
        from aiohttp import web
        import aiohttp

        csrf_err = self._verify_csrf(request)
        if csrf_err:
            return web.json_response({"ok": False, "error": csrf_err}, status=403)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body"}, status=400)

        try:
            validated = self._validate_llm_payload(body)
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)

        cfg = getattr(self.app, "config", None)
        saved_url = getattr(cfg, "llm_base_url", "") if cfg else ""
        saved_key = getattr(cfg, "llm_api_key", "EMPTY") if cfg else "EMPTY"
        if saved_key is None:
            saved_key = "EMPTY"

        target_backend = validated["llm_backend"]
        target_url = validated["llm_base_url"]
        target_model = validated["llm_model"]
        raw_key = validated["llm_api_key"]

        saved_origin = self._origin_tuple(saved_url)
        target_origin = self._origin_tuple(target_url)
        dest_matched = (
            saved_origin is not None
            and target_origin is not None
            and saved_origin == target_origin
        )

        if raw_key is None or raw_key == "●●●●●●":
            if dest_matched:
                test_key = saved_key
            else:
                test_key = "EMPTY"
        elif raw_key == "" or raw_key.strip() == "EMPTY":
            test_key = "EMPTY"
        else:
            test_key = raw_key.strip()

        if target_backend == "noop":
            return web.json_response({
                "ok": True,
                "status": "noop",
                "latency_ms": 0,
                "message": "noop 后端：跳过连通性探测",
            })

        async def _read_bounded(stream, limit: int = 65536) -> bytes:
            buf = bytearray()
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > limit:
                    raise ValueError("响应体超过 64KB 上限")
            return bytes(buf)

        t0 = time.monotonic()
        try:
            async with _async_timeout(5.0):
                headers = {"Accept": "application/json"}
                if test_key and test_key != "EMPTY":
                    headers["Authorization"] = f"Bearer {test_key}"

                async with aiohttp.ClientSession() as session:
                    models_url = f"{target_url}/models"
                    need_fallback = False
                    try:
                        async with session.get(models_url, headers=headers, allow_redirects=False) as resp:
                            latency_ms = max(1, int((time.monotonic() - t0) * 1000))
                            if resp.status in (301, 302, 303, 307, 308):
                                return web.json_response({
                                    "ok": False,
                                    "status": "redirect_blocked",
                                    "latency_ms": latency_ms,
                                    "error": f"端点返回重定向 (HTTP {resp.status})，出于凭据安全已禁止跟随重定向",
                                })
                            elif resp.status == 200:
                                try:
                                    body_bytes = await _read_bounded(resp.content, 64 * 1024)
                                    data = json.loads(body_bytes.decode("utf-8", errors="replace"))
                                except Exception:
                                    return web.json_response({
                                        "ok": False,
                                        "status": "invalid_response",
                                        "latency_ms": latency_ms,
                                        "error": "端点响应无法解析为 JSON 或超出大小上限",
                                    })

                                model_ids: set[str] = set()
                                if isinstance(data, dict):
                                    items = data.get("data")
                                    if isinstance(items, list):
                                        for item in items:
                                            if isinstance(item, dict) and "id" in item:
                                                model_ids.add(str(item["id"]))
                                            elif isinstance(item, str):
                                                model_ids.add(item)
                                    elif "models" in data and isinstance(data["models"], list):
                                        for item in data["models"]:
                                            if isinstance(item, dict) and "id" in item:
                                                model_ids.add(str(item["id"]))
                                            elif isinstance(item, str):
                                                model_ids.add(item)
                                elif isinstance(data, list):
                                    for item in data:
                                        if isinstance(item, dict) and "id" in item:
                                            model_ids.add(str(item["id"]))
                                        elif isinstance(item, str):
                                            model_ids.add(item)

                                if target_model in model_ids:
                                    return web.json_response({
                                        "ok": True,
                                        "status": "available",
                                        "latency_ms": latency_ms,
                                        "message": f"连接成功，模型 {target_model} 可用",
                                    })
                                else:
                                    return web.json_response({
                                        "ok": False,
                                        "status": "model_not_found",
                                        "latency_ms": latency_ms,
                                        "error": f"端点连接成功，但模型列表中未找到 {target_model}",
                                    })
                            elif resp.status == 401:
                                return web.json_response({
                                    "ok": False,
                                    "status": "unauthorized",
                                    "latency_ms": latency_ms,
                                    "error": "认证失败 (HTTP 401)：API 密钥无效或未授权",
                                })
                            elif resp.status == 404:
                                need_fallback = True
                            else:
                                return web.json_response({
                                    "ok": False,
                                    "status": f"http_{resp.status}",
                                    "latency_ms": latency_ms,
                                    "error": f"端点返回错误状态码 HTTP {resp.status}",
                                })
                    except aiohttp.ClientConnectorError as e:
                        latency_ms = max(1, int((time.monotonic() - t0) * 1000))
                        err_msg = str(e)
                        if test_key and test_key != "EMPTY":
                            err_msg = err_msg.replace(test_key, "●●●●●●")
                        return web.json_response({
                            "ok": False,
                            "status": "connection_refused",
                            "latency_ms": latency_ms,
                            "error": f"无法连接到端点: {err_msg}",
                        })
                    except (asyncio.TimeoutError, TimeoutError):
                        raise
                    except Exception as e:
                        latency_ms = max(1, int((time.monotonic() - t0) * 1000))
                        err_msg = str(e)
                        if test_key and test_key != "EMPTY":
                            err_msg = err_msg.replace(test_key, "●●●●●●")
                        return web.json_response({
                            "ok": False,
                            "status": "connection_error",
                            "latency_ms": latency_ms,
                            "error": f"端点连接发生异常: {err_msg}",
                        })

                    if need_fallback:
                        chat_url = f"{target_url}/chat/completions"
                        chat_payload = {
                            "model": target_model,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                        }
                        chat_headers = dict(headers)
                        chat_headers["Content-Type"] = "application/json"
                        try:
                            async with session.post(
                                chat_url,
                                headers=chat_headers,
                                json=chat_payload,
                                allow_redirects=False,
                            ) as chat_resp:
                                latency_ms = max(1, int((time.monotonic() - t0) * 1000))
                                if chat_resp.status in (301, 302, 303, 307, 308):
                                    return web.json_response({
                                        "ok": False,
                                        "status": "redirect_blocked",
                                        "latency_ms": latency_ms,
                                        "error": f"端点返回重定向 (HTTP {chat_resp.status})，出于凭据安全已禁止跟随重定向",
                                    })
                                elif chat_resp.status == 200:
                                    try:
                                        chat_bytes = await _read_bounded(chat_resp.content, 64 * 1024)
                                        chat_data = json.loads(chat_bytes.decode("utf-8", errors="replace"))
                                    except Exception:
                                        return web.json_response({
                                            "ok": False,
                                            "status": "invalid_response",
                                            "latency_ms": latency_ms,
                                            "error": "端点响应无法解析为 JSON 或超出大小上限",
                                        })
                                    choices = chat_data.get("choices") if isinstance(chat_data, dict) else None
                                    if isinstance(choices, list) and len(choices) > 0:
                                        return web.json_response({
                                            "ok": True,
                                            "status": "available",
                                            "latency_ms": latency_ms,
                                            "message": f"连接成功，模型 {target_model} 响应正常",
                                        })
                                    else:
                                        return web.json_response({
                                            "ok": False,
                                            "status": "unexpected_response",
                                            "latency_ms": latency_ms,
                                            "error": "端点响应异常: /chat/completions 未返回有效的 choices 列表",
                                        })
                                elif chat_resp.status == 401:
                                    return web.json_response({
                                        "ok": False,
                                        "status": "unauthorized",
                                        "latency_ms": latency_ms,
                                        "error": "认证失败 (HTTP 401)：API 密钥无效或未授权",
                                    })
                                elif chat_resp.status == 404:
                                    return web.json_response({
                                        "ok": False,
                                        "status": "not_found",
                                        "latency_ms": latency_ms,
                                        "error": "端点连接成功，但 /models 与 /chat/completions 均返回 404",
                                    })
                                else:
                                    return web.json_response({
                                        "ok": False,
                                        "status": f"http_{chat_resp.status}",
                                        "latency_ms": latency_ms,
                                        "error": f"端点返回错误状态码 HTTP {chat_resp.status}",
                                    })
                        except aiohttp.ClientConnectorError as e:
                            latency_ms = max(1, int((time.monotonic() - t0) * 1000))
                            err_msg = str(e)
                            if test_key and test_key != "EMPTY":
                                err_msg = err_msg.replace(test_key, "●●●●●●")
                            return web.json_response({
                                "ok": False,
                                "status": "connection_refused",
                                "latency_ms": latency_ms,
                                "error": f"无法连接到端点: {err_msg}",
                            })
                        except (asyncio.TimeoutError, TimeoutError):
                            raise
                        except Exception as e:
                            latency_ms = max(1, int((time.monotonic() - t0) * 1000))
                            err_msg = str(e)
                            if test_key and test_key != "EMPTY":
                                err_msg = err_msg.replace(test_key, "●●●●●●")
                            return web.json_response({
                                "ok": False,
                                "status": "connection_error",
                                "latency_ms": latency_ms,
                                "error": f"端点连接发生异常: {err_msg}",
                            })
        except (asyncio.TimeoutError, TimeoutError):
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return web.json_response({
                "ok": False,
                "status": "timeout",
                "latency_ms": max(elapsed_ms, 5000),
                "error": "连接超时（超过 5.0 秒单一截止时间），请检查目标服务地址是否可达",
            })
        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            err_msg = str(e)
            if test_key and test_key != "EMPTY":
                err_msg = err_msg.replace(test_key, "●●●●●●")
            return web.json_response({
                "ok": False,
                "status": "error",
                "latency_ms": elapsed_ms,
                "error": f"测试端点失败: {err_msg}",
            })

    async def _api_session_history(self, request):  # noqa: ANN001
        from aiohttp import web
        system_prompt = getattr(self.app.config, "system_prompt", "")
        history = list(getattr(self.app.session, "history", []) or [])
        items: list[dict[str, str]] = []
        if system_prompt:
            items.append({"role": "system", "content": system_prompt})
        items.extend(history)
        return web.json_response(items)

    async def _api_session_clear(self, request):  # noqa: ANN001
        """Drop accumulated conversation history.

        Useful when in-context learning has latched onto a degenerate
        echo pattern (e.g. several identical assistant turns make the
        LLM keep repeating that same reply). Leaves the system prompt
        and active mode untouched."""
        from aiohttp import web
        session = getattr(self.app, "session", None)
        if session is None:
            return web.json_response(
                {"ok": False, "error": "no session"}, status=500
            )
        cleared = 0
        try:
            history = getattr(session, "history", None)
            if isinstance(history, list):
                cleared = len(history)
                history.clear()
            # Invalidate any cached prefix so the next turn doesn't
            # try to resume from a (now-stale) KV warm-up.
            for attr in ("cache_warmed", "_cache_warmed", "prefix_cache_disabled"):
                if hasattr(session, attr):
                    try:
                        setattr(session, attr, False if "warmed" in attr else False)
                    except Exception:  # pragma: no cover - defensive
                        pass
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, "cleared": cleared})

    # ── broadcast helpers ────────────────────────────────────────

    def _schedule_broadcast(self, event: str, data: Any) -> None:
        try:
            asyncio.get_running_loop().create_task(self._broadcast(event, data))
        except RuntimeError:
            pass

    async def _broadcast(self, event: str, data: Any = None) -> None:
        if not self._browser_clients:
            return
        try:
            payload = json.dumps(
                {"event": event, "data": _safe(data), "ts": int(time.time() * 1000)},
                ensure_ascii=False,
            )
        except Exception:
            return
        for ws in list(self._browser_clients):
            try:
                if ws.closed:
                    self._browser_clients.discard(ws)
                    continue
                await ws.send_str(payload)
            except Exception:
                self._browser_clients.discard(ws)

    async def _emit_latency(self, kind: str, ms: float) -> None:
        if ms < 0:
            return
        self._latency_history[kind].append(float(ms))
        await self._broadcast("latency", {"kind": kind, "ms": int(ms)})

    def _tts_metrics_payload(self) -> dict[str, Any]:
        return {
            "sentence_count": self._tts_sentence_count,
            "bytes_current": self._tts_bytes_current,
            "bytes_last": self._tts_bytes_last,
            "last_duration_s": round(self._tts_last_duration_s, 2),
            "sample_rate": self._tts_last_sample_rate,
        }

    async def _stats_loop(self) -> None:
        try:
            while True:
                mic_qsize = None
                try:
                    q = getattr(self.app.audio, "_in_queue", None)
                    if q is not None:
                        mic_qsize = q.qsize()
                except Exception:
                    mic_qsize = None

                ws_state = "closed"
                try:
                    slv = getattr(self.app, "slv", None)
                    raw_ws = getattr(slv, "_ws", None) if slv is not None else None
                    if raw_ws is None:
                        ws_state = "closed"
                    elif getattr(raw_ws, "closed", False):
                        ws_state = "closed"
                    else:
                        ws_state = "open"
                except Exception:
                    ws_state = "closed"

                await self._broadcast(
                    "stats",
                    {"mic_queue_depth": mic_qsize, "slv_ws_state": ws_state},
                )
                await self._broadcast("tts_metrics", self._tts_metrics_payload())
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise

    # ── plugin observer hooks (forward to browser + compute latency) ─

    async def on_user_speech_start(self) -> None:
        await self._broadcast("on_user_speech_start", None)

    async def on_user_partial(self, text: str) -> None:
        await self._broadcast("on_user_partial", text)

    async def on_user_speech_end_client(self, data: dict) -> None:
        try:
            self._last_speech_end_ts_ms = int((data or {}).get("ts") or time.time() * 1000)
        except Exception:
            self._last_speech_end_ts_ms = int(time.time() * 1000)
        await self._broadcast("on_user_speech_end_client", data)

    async def on_user_utterance(self, text: str) -> None:
        now_ms = int(time.time() * 1000)
        self._turn_id += 1
        self._last_utterance_ts_ms = now_ms
        self._first_token_ts_ms = None
        self._first_tts_audio_ts_ms = None
        # Reset per-turn TTS counters at the start of a new user turn.
        self._tts_sentence_count = 0
        self._tts_bytes_current = 0
        # ASR latency: speech end → final utterance.
        if self._last_speech_end_ts_ms is not None:
            await self._emit_latency("asr", now_ms - self._last_speech_end_ts_ms)
        await self._broadcast("on_user_utterance", text)

    async def on_user_stop_intent(self, data: str) -> None:
        await self._broadcast("on_user_stop_intent", data)

    async def on_assistant_token(self, token: str) -> None:
        if self._first_token_ts_ms is None and self._last_utterance_ts_ms is not None:
            self._first_token_ts_ms = int(time.time() * 1000)
            await self._emit_latency(
                "ttft", self._first_token_ts_ms - self._last_utterance_ts_ms
            )
        await self._broadcast("on_assistant_token", token)

    async def on_assistant_sentence(self, sentence: str) -> None:
        self._tts_sentence_count += 1
        await self._broadcast(
            "on_assistant_sentence",
            sentence,
        )
        await self._broadcast(
            "tts_metrics",
            self._tts_metrics_payload(),
        )

    async def on_assistant_sentence_start(self, sentence: str) -> None:
        await self._broadcast("on_assistant_sentence_start", sentence)

    async def on_assistant_done(self) -> None:
        # Snapshot per-turn TTS metrics into "last". Keep the current-turn
        # total visible while idle; reset it only when the next user turn
        # starts in on_user_utterance().
        self._tts_bytes_last = self._tts_bytes_current
        duration_s = self._tts_last_duration_s
        if self._tts_last_sample_rate > 0:
            # 16-bit PCM mono → 2 bytes per sample.
            duration_s = self._tts_bytes_last / float(self._tts_last_sample_rate) / 2.0
            self._tts_last_duration_s = duration_s
        await self._broadcast("tts_metrics", self._tts_metrics_payload())
        await self._broadcast("on_assistant_done", None)

    async def on_tts_audio_frame(self, data: dict) -> None:
        if self._first_tts_audio_ts_ms is None:
            self._first_tts_audio_ts_ms = int(time.time() * 1000)
            if self._first_token_ts_ms is not None:
                await self._emit_latency(
                    "ttfa", self._first_tts_audio_ts_ms - self._first_token_ts_ms
                )
            if self._last_speech_end_ts_ms is not None:
                await self._emit_latency(
                    "rtt", self._first_tts_audio_ts_ms - self._last_speech_end_ts_ms
                )
        # Track bytes for the per-turn TTS metrics card.
        try:
            d = data or {}
            frame_len = int(d.get("frame_len") or 0)
            if frame_len > 0:
                self._tts_bytes_current += frame_len
            sr = int(d.get("sample_rate") or 0)
            if sr > 0:
                self._tts_last_sample_rate = sr
        except Exception:
            pass
        await self._broadcast("on_tts_audio_frame", data)

    async def on_error(self, exc: BaseException) -> None:
        # Prefer the exception's str() representation when it carries a
        # human-readable message (e.g. our LLMTimeoutError-derived
        # RuntimeError); fall back to repr() so empty/bare exceptions
        # still produce something useful.
        try:
            s = str(exc)
        except Exception:
            s = ""
        msg = s if s else repr(exc)
        # Detect TypedLLMError-style payloads (or any exception that
        # exposes a dict-like ``.payload``). Browser clients prefer the
        # richer dict so they can colour-code by ``type``; the persistent
        # error log keeps both for snapshot-replay to late clients.
        payload_attr = getattr(exc, "payload", None)
        if isinstance(payload_attr, dict):
            # Build a defensive copy so plugin modifications can't leak
            # back into the original exception.
            data: dict = {
                "type": str(payload_attr.get("type") or "unknown"),
                "message": str(payload_attr.get("message") or msg),
                "exc_class": str(
                    payload_attr.get("exc_class") or type(exc).__name__
                ),
                "timestamp": payload_attr.get("timestamp"),
            }
            for k, v in payload_attr.items():
                data.setdefault(k, v)
        else:
            data = {
                "type": "unknown",
                "message": msg,
                "exc_class": type(exc).__name__,
                "timestamp": time.time(),
            }
        entry = {"ts": int(time.time() * 1000), "msg": data["message"], **data}
        self._errors.append(entry)
        if len(self._errors) > 50:
            self._errors.pop(0)
        await self._broadcast("on_error", data)

    async def on_state_change(self, data: dict) -> None:
        await self._broadcast("on_state_change", data)

    async def on_slv_reconnect(self, data: dict) -> None:
        await self._broadcast("on_slv_reconnect", data)

    async def on_mic_rms(self, data: dict) -> None:
        await self._broadcast("on_mic_rms", data)

    async def on_mode_change(self, data: dict) -> None:
        await self._broadcast("on_mode_change", data)

    async def on_mode_override_change(self, data: dict) -> None:
        await self._broadcast("on_mode_override_change", data)

    async def on_mode_registered(self, data: dict) -> None:
        await self._broadcast("mode_registered", data)

    async def on_agent_settings_change(self, data: dict) -> None:
        await self._broadcast("on_agent_settings_change", data)

    async def on_transcribed(self, data: dict) -> None:
        await self._broadcast("on_transcribed", data)

    async def on_translation(self, data: dict) -> None:
        # live_caption / simul_interpret + InterpreterMode bilingual output.
        await self._broadcast("on_translation", data)

    async def on_wake(self, data: dict) -> None:
        await self._broadcast("on_wake", data)

    async def on_sleep(self, data) -> None:
        await self._broadcast("on_sleep", data)

    # ── LLM availability / session-trim / prefix-cache relays ──────

    def _build_llm_availability_payload(self) -> dict[str, Any]:
        """Snapshot of current LLM availability for late-connecting clients."""
        payload: dict[str, Any] = {
            "state": "unknown",
            "last_ok_ts": None,
            "consecutive_failures": 0,
            "probe_interval_s": None,
        }
        avail = getattr(self.app, "llm_availability", None) if self.app else None
        if avail is not None:
            try:
                state = getattr(avail, "state", None)
                payload["state"] = (
                    state.value if hasattr(state, "value") else str(state or "unknown")
                )
                payload["last_ok_ts"] = getattr(avail, "last_ok_ts", None)
                payload["consecutive_failures"] = int(
                    getattr(avail, "consecutive_failures", 0) or 0
                )
                interval = getattr(avail, "interval_s", None)
                payload["probe_interval_s"] = (
                    float(interval) if interval is not None else None
                )
            except Exception:  # pragma: no cover - defensive
                logger.debug("availability snapshot failed", exc_info=True)
        return payload

    def _read_prefix_cache_disabled(self) -> bool:
        try:
            sess = getattr(self.app, "session", None) if self.app else None
            if sess is None:
                return False
            return bool(getattr(sess, "prefix_cache_disabled", False))
        except Exception:  # pragma: no cover - defensive
            return False

    async def on_llm_availability_change(self, data: dict) -> None:
        await self._broadcast("on_llm_availability_change", data)

    def _on_bus_session_trimmed(self, data: Any) -> None:
        self._schedule_broadcast("on_session_trimmed", data)

    def _on_bus_prefix_cache_disabled(self, data: Any) -> None:
        self._schedule_broadcast("on_prefix_cache_disabled", data)

    def _on_bus_echo_recovery(self, data: Any) -> None:
        self._schedule_broadcast("on_echo_recovery", data)

    async def _api_llm_probe(self, request):  # noqa: ANN001
        from aiohttp import web

        avail = getattr(self.app, "llm_availability", None) if self.app else None
        if avail is None:
            return web.json_response(
                {"error": "llm_availability plugin not active"}, status=503
            )
        try:
            await avail.force_probe()
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response(self._build_llm_availability_payload())

    async def on_llm_cache_metrics(self, data: dict) -> None:
        # edge-llm-chat-service payload shape:
        #   {"prefill": {"reused_tokens": N, "computed_tokens": M}, ...}
        # Older / flat shapes:
        #   {"hit_tokens": N, "total_tokens": T} or {"cache_hit_tokens": N}
        if not isinstance(data, dict):
            await self._broadcast("on_llm_cache_metrics", data)
            return
        prefill = data.get("prefill") or {}
        if not isinstance(prefill, dict):
            prefill = {}
        reused = (
            prefill.get("reused_tokens")
            or data.get("hit_tokens")
            or data.get("cache_hit_tokens")
            or 0
        )
        computed = prefill.get("computed_tokens") or 0
        if reused or computed:
            total = int(reused) + int(computed)
        else:
            total = int(data.get("total_tokens") or data.get("input_tokens") or 0)
        reused_i = int(reused)
        pct: float | None = (
            round(100.0 * reused_i / total, 1) if total else None
        )
        if pct is not None:
            self._cache_pct_history.append(pct)
        avg_pct: float | None = (
            round(sum(self._cache_pct_history) / len(self._cache_pct_history), 1)
            if self._cache_pct_history
            else None
        )
        await self._broadcast(
            "on_llm_cache_metrics",
            {
                "hit_pct": pct,
                "avg_pct": avg_pct,
                "reused": reused_i,
                "total": total,
                "raw": data,
            },
        )


def _safe(value: Any) -> Any:
    """Coerce to a JSON-serialisable form (best effort)."""
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


__all__ = ["DebugDashboardPlugin"]
