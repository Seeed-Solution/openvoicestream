"""Dashboard /api/errors/clear and /api/agent/settings endpoints + ModeManager
late-registration broadcast."""
from __future__ import annotations

import asyncio
import os
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from aiohttp import web
import pytest
import yaml

from ovs_agent.app_mode import AppMode, ModeManager
from ovs_agent.config import Config, load_config
from ovs_agent.event_bus import EventBus
from ovs_agent.plugins.debug_dashboard import DebugDashboardPlugin


class _Chat(AppMode):
    name = "chat"
    display_name = "对话"
    icon = "💬"
    description = "test chat"

    async def on_user_utterance(self, ctx, text):
        return None


class _Other(AppMode):
    name = "other"
    display_name = "其他"
    icon = "🔧"
    description = "another mode"

    async def on_user_utterance(self, ctx, text):
        return None


def _mk_app(port: int, config: Config):
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
    chat = _Chat()
    mgr.register(chat)
    mgr._current = chat
    app.modes = mgr
    app.plugins = []
    return app


class _RuntimeKws:
    name = "runtime_kws"

    def __init__(self):
        self.phrases = ["你好小智"]

    def status(self):
        return {"available": True, "running": True, "phrases": self.phrases}

    async def validate_phrases(self, phrases):
        clean = [str(value).strip() for value in phrases if str(value).strip()]
        if not clean:
            raise ValueError("at least one phrase is required")
        return tuple(clean)

    async def update_phrases(self, phrases):
        self.phrases = list(await self.validate_phrases(phrases))
        return tuple(self.phrases)


@pytest.mark.asyncio
async def test_runtime_wakeword_update_persists_and_test_tone_plays(
    unused_tcp_port, tmp_path
):
    state_path = tmp_path / "wakeword.json"
    cfg = Config(
        metadata={
            "dashboard_port": unused_tcp_port,
            "wakeword": {"state_path": str(state_path)},
        },
        pipeline_mode="wake_word",
    )
    app = _mk_app(unused_tcp_port, cfg)
    source = _RuntimeKws()
    app.plugins = [source]
    app._play_wake_tone = MagicMock()
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                base + "/api/wakeword/validate", json={"phrases": [" Hey Seeed "]}
            )
            assert response.status == 200
            assert (await response.json())["phrases"] == ["Hey Seeed"]

            response = await session.patch(
                base + "/api/wakeword/runtime",
                json={"phrases": ["你好小智", "Hey Seeed"]},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload["persisted"] is True

            response = await session.post(base + "/api/wakeword/test-tone")
            assert response.status == 200
        app._play_wake_tone.assert_called_once_with()
        saved = state_path.read_text(encoding="utf-8")
        assert "你好小智" in saved and "Hey Seeed" in saved
        assert state_path.stat().st_mode & 0o777 == 0o600
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_errors_clear_endpoint(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    # Seed some errors directly.
    plugin._errors.extend([{"ts": 1, "msg": "boom"}, {"ts": 2, "msg": "kaboom"}])
    assert len(plugin._errors) == 2
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(base + "/api/errors/clear")
            assert r.status == 200
            data = await r.json()
            assert data == {"ok": True}
        assert plugin._errors == []
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_restart_agent_endpoint(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    exit_called = []
    app._restart_exit_fn = lambda: exit_called.append(True)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        headers = {
            "X-Admin-Token": plugin._mcp_admin_token,
            "Origin": f"http://127.0.0.1:{unused_tcp_port}",
        }
        async with aiohttp.ClientSession() as s:
            r = await s.post(base + "/api/control/restart_agent", headers=headers)
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is True
            assert "restarting in 500ms" in data["message"]
        await asyncio.sleep(0.6)
        assert exit_called == [True]
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_agent_settings_get_returns_six_fields(unused_tcp_port):
    cfg = Config(
        metadata={"dashboard_port": unused_tcp_port},
        pipeline_mode="wake_word",
        sleep_timeout_s=42.5,
        stop_words=["停", "stop"],
        exit_words=["退下", "bye"],
        wake_reply_text="在呢",
        exit_reply_text="那我先退下了",
    )
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.get(base + "/api/agent/settings")
            assert r.status == 200
            data = await r.json()
        assert data["pipeline_mode"] == "wake_word"
        assert data["sleep_timeout_s"] == 42.5
        assert data["stop_words"] == ["停", "stop"]
        assert data["exit_words"] == ["退下", "bye"]
        assert data["wake_reply_text"] == "在呢"
        assert data["exit_reply_text"] == "那我先退下了"
    finally:
        await plugin.stop()


test_agent_settings_get_returns_three_fields = test_agent_settings_get_returns_six_fields


@pytest.mark.asyncio
async def test_agent_settings_post_updates_config(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/agent/settings",
                json={
                    "pipeline_mode": "wake_word",
                    "sleep_timeout_s": 12.0,
                    "stop_words": ["a", "b ", "  "],
                    "exit_words": [" 退下", ""],
                    "wake_reply_text": " 在呢 ",
                    "exit_reply_text": "",
                },
            )
            assert r.status == 200
            data = await r.json()
        assert cfg.pipeline_mode == "wake_word"
        assert cfg.sleep_timeout_s == 12.0
        assert cfg.stop_words == ["a", "b"]  # blank dropped, trimmed
        assert cfg.exit_words == ["退下"]
        assert cfg.wake_reply_text == "在呢"
        assert cfg.exit_reply_text == ""
        assert data["exit_words"] == ["退下"]
        # Persistence skipped because no _source_path.
        assert data["persisted"] is False
        # Broadcast fired.
        called = [c for c in app.broadcast.call_args_list
                  if c.args[0] == "on_agent_settings_change"]
        assert called, "broadcast should have fired on_agent_settings_change"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_agent_settings_post_persists_to_yaml(tmp_path, unused_tcp_port):
    src = tmp_path / "config.yaml"
    src.write_text(
        "slv_url: ws://x/y\n"
        "system_prompt: GLOBAL\n"
        "pipeline_mode: always_on\n"
        "sleep_timeout_s: 30.0\n",
        encoding="utf-8",
    )
    cfg = load_config(src)
    cfg.metadata = {"dashboard_port": unused_tcp_port}
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/agent/settings",
                json={"sleep_timeout_s": 99.0, "stop_words": ["x", "y"]},
            )
            assert r.status == 200
            data = await r.json()
        assert data["persisted"] is True

        import yaml
        with src.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert raw["sleep_timeout_s"] == 99.0
        assert raw["stop_words"] == ["x", "y"]
        assert raw["system_prompt"] == "GLOBAL"
        assert raw["pipeline_mode"] == "always_on"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_agent_settings_post_validates_pipeline_mode(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/agent/settings",
                json={"pipeline_mode": "garbage"},
            )
            assert r.status == 400
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_inject_user_text_rejects_server_loop(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port}, server_loop=True)
    app = _mk_app(unused_tcp_port, cfg)
    app._dispatch_one = AsyncMock()
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/control/inject_user_text",
                json={"text": "请点头。", "language": "zh"},
            )
            data = await r.json()
        assert r.status == 409
        assert data["status"] == "unsupported"
        assert "server-loop" in data["error"]
        app._dispatch_one.assert_not_awaited()
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_mode_manager_register_silent_before_start():
    """Registering modes before start() must NOT broadcast on_mode_registered."""
    broadcasts: list = []

    async def _bcast(name, payload):
        broadcasts.append((name, payload))

    def _factory():
        return SimpleNamespace(broadcast=_bcast, mode_manager=None)

    mgr = ModeManager(_factory)
    mgr.register(_Chat())
    mgr.register(_Other())
    assert broadcasts == []
    assert mgr._started is False


@pytest.mark.asyncio
async def test_mode_manager_register_broadcasts_after_start():
    """Registering AFTER start() must broadcast on_mode_registered."""
    broadcasts: list = []

    async def _bcast(name, payload):
        broadcasts.append((name, payload))

    def _factory():
        return SimpleNamespace(broadcast=_bcast, mode_manager=None)

    mgr = ModeManager(_factory)
    mgr.register(_Chat())
    await mgr.start("chat")
    assert mgr._started is True
    # on_mode_change fired from switch().
    assert any(b[0] == "on_mode_change" for b in broadcasts)

    n_before = len(broadcasts)
    mgr.register(_Other())

    # Late-registration broadcast is fire-and-forget; yield to let it run.
    import asyncio
    await asyncio.sleep(0.01)
    new_b = broadcasts[n_before:]
    reg_events = [b for b in new_b if b[0] == "on_mode_registered"]
    assert len(reg_events) == 1
    assert reg_events[0][1]["name"] == "other"
    assert reg_events[0][1]["display_name"] == "其他"


@pytest.mark.asyncio
async def test_tts_metrics_emitted_on_audio_frame(unused_tcp_port):
    """on_tts_audio_frame should accumulate bytes; on_assistant_done snapshots."""
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    # Don't start HTTP server — just drive hooks directly.
    await plugin.on_user_utterance("hi")
    await plugin.on_assistant_sentence("hello there.")
    await plugin.on_tts_audio_frame({"sample_rate": 24000, "frame_len": 48000})
    await plugin.on_tts_audio_frame({"sample_rate": 24000, "frame_len": 24000})
    assert plugin._tts_sentence_count == 1
    assert plugin._tts_bytes_current == 72000
    await plugin.on_assistant_done()
    assert plugin._tts_bytes_last == 72000
    assert plugin._tts_bytes_current == 72000
    assert plugin._tts_last_duration_s == 1.5
    await plugin.on_user_utterance("next")
    assert plugin._tts_bytes_current == 0


@pytest.mark.asyncio
async def test_agent_settings_post_rejects_bad_exit_fields(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(base + "/api/agent/settings", json={"exit_words": "bye"})
            assert r.status == 400
            r = await s.post(base + "/api/agent/settings", json={"wake_reply_text": 1})
            assert r.status == 400
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_wakeword_runtime_get_disabled_returns_four_fields(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    app.plugins = []  # No runtime_kws plugin
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.get(base + "/api/wakeword/runtime")
            assert r.status == 200
            data = await r.json()
            assert data == {
                "available": False,
                "enabled": False,
                "running": False,
                "phrases": [],
            }
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_agent_settings_post_strict_validation_and_atomicity(unused_tcp_port):
    cfg = Config(
        metadata={"dashboard_port": unused_tcp_port},
        pipeline_mode="always_on",
        sleep_timeout_s=30.0,
        stop_words=["停"],
        exit_words=["退下"],
        wake_reply_text="在呢",
        exit_reply_text="我走了",
    )
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            # 1. sleep_timeout_s validation: bool is forbidden (True / False)
            for bad_val in (True, False, "30", -0.1, 86400.1, float("inf"), float("nan"), None):
                r = await s.post(base + "/api/agent/settings", json={"sleep_timeout_s": bad_val})
                assert r.status == 400, f"Expected 400 for sleep_timeout_s={bad_val!r}"

            # 2. pipeline_mode validation
            for bad_val in ("unknown_mode", 123, True, None, []):
                r = await s.post(base + "/api/agent/settings", json={"pipeline_mode": bad_val})
                assert r.status == 400, f"Expected 400 for pipeline_mode={bad_val!r}"

            # 3. wake_reply_text / exit_reply_text validation: non-str and > 200 chars
            for field_name in ("wake_reply_text", "exit_reply_text"):
                for bad_val in (123, True, None, ["reply"], "a" * 201):
                    r = await s.post(base + "/api/agent/settings", json={field_name: bad_val})
                    assert r.status == 400, f"Expected 400 for {field_name}={bad_val!r}"

            # 4. stop_words / exit_words validation: non-list, item non-str, item > 50 chars, count > 100
            for field_name in ("stop_words", "exit_words"):
                for bad_val in (
                    "not_a_list",
                    123,
                    True,
                    None,
                    {"words": []},
                    [123],
                    ["valid", True],
                    ["valid", None],
                    ["valid", 3.14],
                    ["x" * 51],
                    [f"word_{i}" for i in range(101)],
                ):
                    r = await s.post(base + "/api/agent/settings", json={field_name: bad_val})
                    assert r.status == 400, f"Expected 400 for {field_name}={bad_val!r}"

            # 5. Unrecognized field
            r = await s.post(base + "/api/agent/settings", json={"random_invalid_key": 42})
            assert r.status == 400

            # 6. ATOMICITY: valid field mixed with invalid field must NOT modify cfg
            r = await s.post(
                base + "/api/agent/settings",
                json={"pipeline_mode": "wake_word", "sleep_timeout_s": True},
            )
            assert r.status == 400
            assert cfg.pipeline_mode == "always_on"
            assert cfg.sleep_timeout_s == 30.0

            r = await s.post(
                base + "/api/agent/settings",
                json={"wake_reply_text": "新问候", "exit_words": "not_a_list"},
            )
            assert r.status == 400
            assert cfg.wake_reply_text == "在呢"
            assert cfg.exit_words == ["退下"]

            # Broadcast was not triggered by failed requests
            bcast_calls = [
                c for c in app.broadcast.call_args_list
                if c.args[0] == "on_agent_settings_change"
            ]
            assert len(bcast_calls) == 0
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_agent_settings_post_changed_and_noop(unused_tcp_port):
    cfg = Config(
        metadata={"dashboard_port": unused_tcp_port},
        pipeline_mode="wake_word",
        sleep_timeout_s=30.0,
        stop_words=["停", "stop"],
        exit_words=["退下"],
        wake_reply_text="在呢",
        exit_reply_text="再见",
    )
    app = _mk_app(unused_tcp_port, cfg)
    app._stop_words_cache = ("dummy", "cache")
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            # 1. No changes (empty dict or identical values) -> not changed -> 200, no broadcast
            r = await s.post(base + "/api/agent/settings", json={})
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is True
            assert data["persisted"] is False
            assert data["changed"] == []
            assert app.broadcast.call_count == 0

            # Identical values, whitespace-stripped to same value
            r = await s.post(
                base + "/api/agent/settings",
                json={
                    "pipeline_mode": "wake_word",
                    "sleep_timeout_s": 30.0,
                    "wake_reply_text": " 在呢 ",
                    "stop_words": ["停", "stop"],
                },
            )
            assert r.status == 200
            data = await r.json()
            assert data["changed"] == []
            assert app.broadcast.call_count == 0

            # -0.0 normalizes to 0.0: when sleep_timeout_s is 0.0, sending -0.0 is a noop
            cfg.sleep_timeout_s = 0.0
            r = await s.post(base + "/api/agent/settings", json={"sleep_timeout_s": -0.0})
            assert r.status == 200
            data = await r.json()
            assert data["changed"] == []
            assert app.broadcast.call_count == 0

            # 2. Changed values -> calculates exact diff, sets _stop_words_cache to None, broadcasts
            r = await s.post(
                base + "/api/agent/settings",
                json={
                    "sleep_timeout_s": 15.0,
                    "stop_words": [" a ", "b", "a", ""],  # strips, filters empty, deduplicates
                    "exit_words": [],  # empty list allowed
                },
            )
            assert r.status == 200
            data = await r.json()
            assert set(data["changed"]) == {"sleep_timeout_s", "stop_words", "exit_words"}
            assert data["stop_words"] == ["a", "b"]
            assert data["exit_words"] == []
            assert cfg.stop_words == ["a", "b"]
            assert cfg.exit_words == []
            assert cfg.sleep_timeout_s == 15.0
            assert app._stop_words_cache is None
            assert app.broadcast.call_count == 1
            assert app.broadcast.call_args[0][0] == "on_agent_settings_change"
            assert app.broadcast.call_args[0][1]["changed"] == data["changed"]
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_agent_settings_post_yaml_persistence_and_degradation(tmp_path, unused_tcp_port):
    src = tmp_path / "config.yaml"
    src.write_text(
        "slv_url: ws://x/y\n"
        "pipeline_mode: always_on\n"
        "sleep_timeout_s: 30.0\n"
        "wake_reply_text: ''\n"
        "exit_reply_text: ''\n",
        encoding="utf-8",
    )
    cfg = load_config(src)
    cfg.metadata = {"dashboard_port": unused_tcp_port}
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            # 1. Successful persistence: atomic tmp file replacement
            r = await s.post(
                base + "/api/agent/settings",
                json={"wake_reply_text": "您好", "exit_reply_text": "再会"},
            )
            assert r.status == 200
            data = await r.json()
            assert data["persisted"] is True
            assert "persist_error" not in data

            import yaml
            with src.open("r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)
            assert saved["wake_reply_text"] == "您好"
            assert saved["exit_reply_text"] == "再会"
            assert saved["slv_url"] == "ws://x/y"

            # 2. Persistence degradation on write error:
            # Point _source_path to a non-existent directory where creation fails
            cfg._source_path = tmp_path / "missing_dir" / "does_not_exist.yaml"
            r = await s.post(
                base + "/api/agent/settings",
                json={"sleep_timeout_s": 55.0},
            )
            assert r.status == 200
            data = await r.json()
            assert data["persisted"] is False
            assert "persist_error" in data
            assert len(data["persist_error"]) > 0
            assert cfg.sleep_timeout_s == 55.0  # In-memory update still took effect!
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_llm_config_get_and_post_persistence(tmp_path, unused_tcp_port):
    """Test GET and POST /api/llm/config endpoints:
    - GET returns masked key if key present, empty if key unset or EMPTY.
    - POST enforces CSRF validation against Host.
    - POST payload schema & validation (bad backend, bad URL, userinfo, port out of range, bad model, unknown keys -> 400).
    - POST missing cfg._source_path rejected with 400.
    - POST destination change clears key to EMPTY if omitted/masked.
    - POST atomic YAML persistence (0600 mode), pending_restart setting, and on_llm_config_change broadcast.
    - POST no-op save retains pending_restart without writing to disk.
    - POST disk write failure rolls back and does not mutate in-memory cfg.
    """
    src = tmp_path / "config.yaml"
    src.write_text(
        "slv_url: ws://x/y\n"
        "llm_backend: edge_llm\n"
        "llm_base_url: http://localhost:8000/v1\n"
        "llm_model: qwen2.5-3b-instruct\n"
        "llm_api_key: secret_initial_token\n",
        encoding="utf-8",
    )
    cfg = load_config(src)
    cfg.metadata = {"dashboard_port": unused_tcp_port}
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()

    base = f"http://127.0.0.1:{unused_tcp_port}"
    host = f"127.0.0.1:{unused_tcp_port}"
    headers = {"Host": host, "Origin": f"http://{host}"}

    try:
        async with aiohttp.ClientSession() as s:
            # 1. GET /api/llm/config with active secret key -> masked
            r = await s.get(f"{base}/api/llm/config")
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is True
            assert data["llm_backend"] == "edge_llm"
            assert data["llm_base_url"] == "http://localhost:8000/v1"
            assert data["llm_model"] == "qwen2.5-3b-instruct"
            assert data["has_api_key"] is True
            assert data["api_key_masked"] == "●●●●●●"
            assert "secret_initial_token" not in str(data)
            assert data["pending_restart"] is False

            # GET with EMPTY key -> unmasked empty
            cfg.llm_api_key = "EMPTY"
            r = await s.get(f"{base}/api/llm/config")
            assert r.status == 200
            data = await r.json()
            assert data["has_api_key"] is False
            assert data["api_key_masked"] == ""
            cfg.llm_api_key = "secret_initial_token"

            # 2. CSRF rejection on POST /api/llm/config
            # Missing Origin and Referer -> 403
            r = await s.post(
                f"{base}/api/llm/config",
                json={
                    "llm_backend": "edge_llm",
                    "llm_base_url": "http://localhost:8000/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                },
            )
            assert r.status == 403
            data = await r.json()
            assert data["ok"] is False
            assert "CSRF" in data["error"]

            # Mismatched Origin -> 403
            r = await s.post(
                f"{base}/api/llm/config",
                headers={"Origin": "http://evil.com"},
                json={
                    "llm_backend": "edge_llm",
                    "llm_base_url": "http://localhost:8000/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                },
            )
            assert r.status == 403

            # Mismatched Referer -> 403
            r = await s.post(
                f"{base}/api/llm/config",
                headers={"Referer": "http://evil.com/settings"},
                json={
                    "llm_backend": "edge_llm",
                    "llm_base_url": "http://localhost:8000/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                },
            )
            assert r.status == 403

            # 3. Payload validation failures -> 400
            # Invalid JSON
            r = await s.post(
                f"{base}/api/llm/config",
                headers=headers,
                data="not json",
            )
            assert r.status == 400

            # Bad backend
            r = await s.post(
                f"{base}/api/llm/config",
                headers=headers,
                json={
                    "llm_backend": "invalid_engine",
                    "llm_base_url": "http://localhost:8000/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                },
            )
            assert r.status == 400
            data = await r.json()
            assert "Invalid llm_backend" in data["error"]

            # Bad URL scheme (ftp)
            r = await s.post(
                f"{base}/api/llm/config",
                headers=headers,
                json={
                    "llm_backend": "edge_llm",
                    "llm_base_url": "ftp://localhost:8000/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                },
            )
            assert r.status == 400
            data = await r.json()
            assert "scheme" in data["error"]

            # URL containing userinfo (http://user:pass@host)
            r = await s.post(
                f"{base}/api/llm/config",
                headers=headers,
                json={
                    "llm_backend": "edge_llm",
                    "llm_base_url": "http://user:pass@localhost:8000/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                },
            )
            assert r.status == 400
            data = await r.json()
            assert "userinfo" in data["error"]

            # Port out of range
            r = await s.post(
                f"{base}/api/llm/config",
                headers=headers,
                json={
                    "llm_backend": "edge_llm",
                    "llm_base_url": "http://localhost:99999/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                },
            )
            assert r.status == 400
            data = await r.json()
            assert "port" in data["error"].lower()

            # Empty model
            r = await s.post(
                f"{base}/api/llm/config",
                headers=headers,
                json={
                    "llm_backend": "edge_llm",
                    "llm_base_url": "http://localhost:8000/v1",
                    "llm_model": "   ",
                },
            )
            assert r.status == 400
            data = await r.json()
            assert "llm_model" in data["error"]

            # Unknown keys
            r = await s.post(
                f"{base}/api/llm/config",
                headers=headers,
                json={
                    "llm_backend": "edge_llm",
                    "llm_base_url": "http://localhost:8000/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                    "extra_key": "injected",
                },
            )
            assert r.status == 400
            data = await r.json()
            assert "Unknown fields" in data["error"]

            # 4. Missing cfg._source_path rejection -> 400
            cfg._source_path = None
            r = await s.post(
                f"{base}/api/llm/config",
                headers=headers,
                json={
                    "llm_backend": "edge_llm",
                    "llm_base_url": "http://localhost:8000/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                },
            )
            assert r.status == 400
            data = await r.json()
            assert data["ok"] is False
            assert "配置文件路径不可用" in data["error"]
            cfg._source_path = src

            # 5. Destination change clears key to EMPTY if key omitted/masked
            r = await s.post(
                f"{base}/api/llm/config",
                headers=headers,
                json={
                    "llm_backend": "openai_compat",
                    "llm_base_url": "http://remote-server:9000/v1/",
                    "llm_model": "qwen2.5-7b-instruct",
                    "llm_api_key": "●●●●●●",
                },
            )
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is True
            assert data["key_cleared"] is True
            assert set(data["changed"]) == {"llm_backend", "llm_base_url", "llm_model", "llm_api_key"}
            assert data["pending_restart"] is True
            assert plugin._llm_config_pending_restart is True
            assert cfg.llm_backend == "openai_compat"
            assert cfg.llm_base_url == "http://remote-server:9000/v1"  # Trailing slash stripped
            assert cfg.llm_model == "qwen2.5-7b-instruct"
            assert cfg.llm_api_key == "EMPTY"

            # Check YAML persistence & mode 0600
            with src.open("r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)
            assert saved["llm_backend"] == "openai_compat"
            assert saved["llm_base_url"] == "http://remote-server:9000/v1"
            assert saved["llm_model"] == "qwen2.5-7b-instruct"
            assert saved["llm_api_key"] == "EMPTY"
            st = os.stat(src)
            assert stat.S_IMODE(st.st_mode) == 0o600

            # Check broadcast
            assert app.broadcast.call_count >= 1
            assert app.broadcast.call_args[0][0] == "on_llm_config_change"
            bcast_payload = app.broadcast.call_args[0][1]
            assert bcast_payload["key_cleared"] is True
            assert bcast_payload["llm_backend"] == "openai_compat"

            # 6. No-op save retains pending_restart without writing to disk
            app.broadcast.reset_mock()
            r = await s.post(
                f"{base}/api/llm/config",
                headers=headers,
                json={
                    "llm_backend": "openai_compat",
                    "llm_base_url": "http://remote-server:9000/v1",
                    "llm_model": "qwen2.5-7b-instruct",
                    "llm_api_key": "EMPTY",
                },
            )
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is True
            assert data["changed"] == []
            assert data["pending_restart"] is True
            assert app.broadcast.call_count == 0

            # 7. Disk write failure rolls back and does not mutate in-memory cfg -> 500
            with patch.object(
                plugin,
                "_persist_agent_settings_to_yaml",
                side_effect=OSError("Disk write error"),
            ):
                r = await s.post(
                    f"{base}/api/llm/config",
                    headers=headers,
                    json={
                        "llm_backend": "noop",
                        "llm_base_url": "http://remote-server:9000/v1",
                        "llm_model": "qwen2.5-7b-instruct",
                        "llm_api_key": "EMPTY",
                    },
                )
                assert r.status == 500
                data = await r.json()
                assert data["ok"] is False
                assert "持久化失败" in data["error"]
                assert cfg.llm_backend == "openai_compat"  # Not mutated!
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_llm_test_probe_and_security_boundary(tmp_path, unused_tcp_port):
    """Test POST /api/llm/test endpoint:
    - CSRF rejection (missing Origin/Referer -> 403).
    - Backend 'noop' immediate ok without probe.
    - Credential origin isolation (dest changed + masked key -> strictly EMPTY, no Authorization header; dest matched + masked key -> reuses saved key).
    - Redirect blocking (allow_redirects=False -> status: redirect_blocked).
    - Models listing check (status: available when target model exists, status: model_not_found when absent).
    - Fallback to /chat/completions when /models returns 404 (status: available when choices non-empty).
    - 5.0s deadline timeout handling (status: timeout).
    - Probing does not tamper with in-memory cfg or YAML file.
    """
    src = tmp_path / "config.yaml"
    src.write_text(
        "slv_url: ws://x/y\n"
        "llm_backend: edge_llm\n"
        "llm_base_url: http://localhost:8000/v1\n"
        "llm_model: qwen2.5-3b-instruct\n"
        "llm_api_key: saved_probe_secret\n",
        encoding="utf-8",
    )
    cfg = load_config(src)
    cfg.metadata = {"dashboard_port": unused_tcp_port}
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()

    base = f"http://127.0.0.1:{unused_tcp_port}"
    host = f"127.0.0.1:{unused_tcp_port}"
    headers = {"Host": host, "Origin": f"http://{host}"}

    # Setup mock remote LLM server
    recorded_requests: list[dict[str, Any]] = []
    models_status = 200
    models_data: dict[str, Any] = {"data": [{"id": "target-model-1"}]}
    chat_status = 200
    chat_data: dict[str, Any] = {"choices": [{"message": {"content": "pong"}}]}
    redirect_target: str | None = None
    server_sleep_s: float = 0.0

    mock_app = web.Application()

    async def handle_models(request: web.Request) -> web.Response:
        if server_sleep_s > 0:
            await asyncio.sleep(server_sleep_s)
        recorded_requests.append({
            "path": request.path,
            "headers": dict(request.headers),
            "method": request.method,
        })
        if redirect_target:
            raise web.HTTPFound(redirect_target)
        if models_status == 404:
            return web.json_response({"error": "not found"}, status=404)
        if models_status == 401:
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response(models_data, status=models_status)

    async def handle_chat(request: web.Request) -> web.Response:
        if server_sleep_s > 0:
            await asyncio.sleep(server_sleep_s)
        body = await request.json()
        recorded_requests.append({
            "path": request.path,
            "headers": dict(request.headers),
            "body": body,
            "method": request.method,
        })
        if redirect_target:
            raise web.HTTPFound(redirect_target)
        if chat_status == 404:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(chat_data, status=chat_status)

    mock_app.router.add_get("/v1/models", handle_models)
    mock_app.router.add_post("/v1/chat/completions", handle_chat)

    mock_runner = web.AppRunner(mock_app)
    await mock_runner.setup()
    mock_site = web.TCPSite(mock_runner, "127.0.0.1", 0)
    await mock_site.start()
    mock_port = mock_site._server.sockets[0].getsockname()[1]
    mock_url = f"http://127.0.0.1:{mock_port}/v1"

    try:
        async with aiohttp.ClientSession() as s:
            # 1. CSRF rejection on POST /api/llm/test
            r = await s.post(
                f"{base}/api/llm/test",
                json={
                    "llm_backend": "noop",
                    "llm_base_url": "http://localhost:8000/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                },
            )
            assert r.status == 403
            data = await r.json()
            assert data["ok"] is False
            assert "CSRF" in data["error"]

            # 2. Backend 'noop' immediate ok
            r = await s.post(
                f"{base}/api/llm/test",
                headers=headers,
                json={
                    "llm_backend": "noop",
                    "llm_base_url": "http://localhost:8000/v1",
                    "llm_model": "qwen2.5-3b-instruct",
                },
            )
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is True
            assert data["status"] == "noop"
            assert data["latency_ms"] == 0

            # 3. /models query success: target model present
            recorded_requests.clear()
            models_data = {"data": [{"id": "target-model-1"}, {"id": "other-model"}]}
            r = await s.post(
                f"{base}/api/llm/test",
                headers=headers,
                json={
                    "llm_backend": "openai_compat",
                    "llm_base_url": mock_url,
                    "llm_model": "target-model-1",
                    "llm_api_key": "custom_probe_key",
                },
            )
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is True
            assert data["status"] == "available"
            assert "可用" in data["message"]
            assert recorded_requests[-1]["path"] == "/v1/models"
            assert recorded_requests[-1]["headers"].get("Authorization") == "Bearer custom_probe_key"

            # 4. /models query target model absent -> model_not_found
            recorded_requests.clear()
            models_data = {"data": [{"id": "different-model"}]}
            r = await s.post(
                f"{base}/api/llm/test",
                headers=headers,
                json={
                    "llm_backend": "openai_compat",
                    "llm_base_url": mock_url,
                    "llm_model": "target-model-1",
                    "llm_api_key": "custom_probe_key",
                },
            )
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is False
            assert data["status"] == "model_not_found"
            assert "未找到" in data["error"]

            # 5. /models returns 404, fallback to /chat/completions
            recorded_requests.clear()
            models_status = 404
            chat_status = 200
            chat_data = {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}
            r = await s.post(
                f"{base}/api/llm/test",
                headers=headers,
                json={
                    "llm_backend": "openai_compat",
                    "llm_base_url": mock_url,
                    "llm_model": "target-model-1",
                    "llm_api_key": "custom_probe_key",
                },
            )
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is True
            assert data["status"] == "available"
            assert "响应正常" in data["message"]
            assert len(recorded_requests) == 2
            assert recorded_requests[0]["path"] == "/v1/models"
            assert recorded_requests[1]["path"] == "/v1/chat/completions"
            assert recorded_requests[1]["body"]["model"] == "target-model-1"
            assert recorded_requests[1]["body"]["messages"] == [{"role": "user", "content": "ping"}]
            models_status = 200

            # 6. Redirect blocking (allow_redirects=False -> status: redirect_blocked)
            redirect_target = "http://malicious-server.com/intercept"
            r = await s.post(
                f"{base}/api/llm/test",
                headers=headers,
                json={
                    "llm_backend": "openai_compat",
                    "llm_base_url": mock_url,
                    "llm_model": "target-model-1",
                },
            )
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is False
            assert data["status"] == "redirect_blocked"
            assert "重定向" in data["error"]
            redirect_target = None

            # 7. Credential origin isolation
            # A. Target destination differs from saved origin -> strictly EMPTY, no Authorization header
            cfg.llm_base_url = "http://127.0.0.1:59999/v1"  # Different port!
            cfg.llm_api_key = "saved_probe_secret"
            recorded_requests.clear()
            models_data = {"data": [{"id": "target-model-1"}]}
            r = await s.post(
                f"{base}/api/llm/test",
                headers=headers,
                json={
                    "llm_backend": "openai_compat",
                    "llm_base_url": mock_url,
                    "llm_model": "target-model-1",
                    "llm_api_key": "●●●●●●",  # Masked key
                },
            )
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is True
            assert recorded_requests[-1]["headers"].get("Authorization") is None

            # B. Target destination matches saved origin -> reuses saved key
            cfg.llm_base_url = mock_url  # Matching origin!
            recorded_requests.clear()
            r = await s.post(
                f"{base}/api/llm/test",
                headers=headers,
                json={
                    "llm_backend": "openai_compat",
                    "llm_base_url": mock_url,
                    "llm_model": "target-model-1",
                    "llm_api_key": "●●●●●●",  # Masked key
                },
            )
            assert r.status == 200
            data = await r.json()
            assert data["ok"] is True
            assert recorded_requests[-1]["headers"].get("Authorization") == "Bearer saved_probe_secret"

            # 8. 5.0s deadline timeout handling
            server_sleep_s = 0.5
            real_timeout = getattr(asyncio, "timeout", None)
            if real_timeout is not None:
                patch_ctx = patch("asyncio.timeout", side_effect=lambda s: real_timeout(0.1))
            else:
                from ovs_agent.plugins.debug_dashboard import _async_timeout
                patch_ctx = patch("ovs_agent.plugins.debug_dashboard._async_timeout", side_effect=lambda s: _async_timeout(0.1))
            with patch_ctx:
                r = await s.post(
                    f"{base}/api/llm/test",
                    headers=headers,
                    json={
                        "llm_backend": "openai_compat",
                        "llm_base_url": mock_url,
                        "llm_model": "target-model-1",
                    },
                )
                assert r.status == 200
                data = await r.json()
                assert data["ok"] is False
                assert data["status"] == "timeout"
                assert "连接超时" in data["error"]
            server_sleep_s = 0.0

            # 9. Ensure probe did not tamper with in-memory cfg or YAML file
            assert cfg.llm_backend == "edge_llm"
            assert cfg.llm_model == "qwen2.5-3b-instruct"
            with src.open("r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)
            assert saved["llm_backend"] == "edge_llm"
    finally:
        await mock_runner.cleanup()
        await plugin.stop()
