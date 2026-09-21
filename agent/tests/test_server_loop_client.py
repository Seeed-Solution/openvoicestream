"""Server-loop client mode (#37 Phase 2-product, behind OVS_AGENT_SERVER_LOOP).

Covers the three contracts of the flag-gated server-loop mode:

  1. Advertise: on session open the agent uploads its tool schemas +
     system prompt + llm params via CLIENT_TOOL_ADVERTISE.
  2. Remote dispatch: a SERVER_TOOL_CALL frame runs the local handler and
     a CLIENT_TOOL_RESULT (correct call_id + result) is returned.
  3. Off (default): the flag resolves False, advertise is a no-op, and the
     local LLM/tool loop still runs (zero behaviour change).
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ovs_agent import Config, Session
from ovs_agent.app_base import BaseApp
from ovs_agent.app_mode import ModeContext, ModeManager
from ovs_agent.modes import ChatMode
from ovs_agent.llm.base import LLMBackend
from ovs_agent.slv_client import ServerToolCall
from ovs_agent.tools import ToolRegistry


async def _drain_tool_tasks(app) -> None:
    """#F4: SERVER_TOOL_CALL now dispatches in a background task, so
    ``_dispatch_one`` returns before the CLIENT_TOOL_RESULT is sent. Await the
    tracked tool tasks so assertions observe the completed result."""
    while app._pending_tool_tasks:
        await asyncio.gather(*list(app._pending_tool_tasks), return_exceptions=True)


# ── fakes ────────────────────────────────────────────────────────────


class FakeSLV:
    """Captures every frame the agent sends in server-loop mode."""

    def __init__(self) -> None:
        self.advertised: list[dict] = []
        self.tool_results: list[dict] = []
        self.text_frames: list[str] = []
        self.flushed = 0
        self.aborted = 0

    async def advertise_tools(self, tools, *, system_prompt=None, llm_params=None):
        self.advertised.append(
            {"tools": tools, "system_prompt": system_prompt, "llm_params": llm_params}
        )

    async def send_tool_result(self, call_id, name, *, ok, result=None, error=None):
        self.tool_results.append(
            {"id": call_id, "name": name, "ok": ok, "result": result, "error": error}
        )

    async def send_text(self, text: str) -> None:
        self.text_frames.append(text)

    async def flush_tts(self) -> None:
        self.flushed += 1

    async def abort(self, *, keep_asr: bool = False) -> None:
        self.aborted += 1


class FakeLLM(LLMBackend):
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.stream_called = False

    async def stream(self, messages, **kw):  # type: ignore[override]
        self.stream_called = True
        for t in self.tokens:
            yield t


async def _noop_broadcast(*args, **kwargs):
    return None


def _make_app(server_loop: bool, registry: ToolRegistry) -> BaseApp:
    """Build a BaseApp without running __init__ (avoids audio/WS setup)."""
    cfg = Config(
        system_prompt="SYS",
        server_loop=server_loop,
        tools_max_iterations=4,
        llm_model="qwen-test",
    )
    app = BaseApp.__new__(BaseApp)
    app.config = cfg
    app.tool_registry = registry
    app.session = Session()
    app.events = type("E", (), {"emit": lambda *a, **k: None})()
    app.modes = None
    app.slv = FakeSLV()
    return app


def _ten_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for i in range(10):
        # Distinct closure per tool name.
        def _make(name):
            @reg.tool(name=name, description=f"tool {name}")
            def _h(value: str = "x") -> dict:
                return {"ran": name, "value": value}

        _make(f"tool_{i}")
    return reg


# ── 1. advertise ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_server_loop_advertises_tools_with_system_prompt():
    reg = _ten_tool_registry()
    app = _make_app(server_loop=True, registry=reg)

    await app._advertise_tools_if_server_loop()

    assert len(app.slv.advertised) == 1
    payload = app.slv.advertised[0]
    # All 10 tool schemas present, OpenAI shape.
    assert len(payload["tools"]) == 10
    names = {t["function"]["name"] for t in payload["tools"]}
    assert names == {f"tool_{i}" for i in range(10)}
    assert all(t["type"] == "function" for t in payload["tools"])
    # System prompt carried.
    assert payload["system_prompt"] == "SYS"
    # LLM params bundle carries the model + max iterations.
    assert payload["llm_params"]["model"] == "qwen-test"
    assert payload["llm_params"]["max_tool_iterations"] == 4


@pytest.mark.asyncio
async def test_off_mode_does_not_advertise():
    reg = _ten_tool_registry()
    app = _make_app(server_loop=False, registry=reg)

    await app._advertise_tools_if_server_loop()

    # Flag off → nothing sent over the WS.
    assert app.slv.advertised == []


@pytest.mark.asyncio
async def test_env_override_enables_advertise(monkeypatch):
    reg = _ten_tool_registry()
    app = _make_app(server_loop=False, registry=reg)  # config says off
    monkeypatch.setenv("OVS_AGENT_SERVER_LOOP", "1")  # env forces on

    await app._advertise_tools_if_server_loop()

    assert len(app.slv.advertised) == 1


# ── 2. remote SERVER_TOOL_CALL → CLIENT_TOOL_RESULT ──────────────────


@pytest.mark.asyncio
async def test_server_tool_call_dispatches_local_handler_and_returns_result():
    calls: list[dict] = []
    reg = ToolRegistry()

    @reg.tool(name="wave", description="wave the arm")
    def wave(side: str = "left") -> dict:
        calls.append({"side": side})
        return {"started": True, "action": "wave", "side": side}

    app = _make_app(server_loop=True, registry=reg)

    evt = ServerToolCall(id="call_x", name="wave", arguments={"side": "right"})
    # Route through the real dispatch entry point.
    await app._dispatch_one(evt)
    await _drain_tool_tasks(app)  # #F4: tool now dispatched in a background task

    # Local handler actually ran with the server-supplied args.
    assert calls == [{"side": "right"}]
    # Exactly one CLIENT_TOOL_RESULT, correct call_id + result.
    assert len(app.slv.tool_results) == 1
    res = app.slv.tool_results[0]
    assert res["id"] == "call_x"
    assert res["name"] == "wave"
    assert res["ok"] is True
    assert res["result"] == {"started": True, "action": "wave", "side": "right"}


@pytest.mark.asyncio
async def test_server_tool_call_unknown_tool_returns_error_result():
    reg = ToolRegistry()
    app = _make_app(server_loop=True, registry=reg)

    evt = ServerToolCall(id="c2", name="does_not_exist", arguments={})
    await app._dispatch_one(evt)
    await _drain_tool_tasks(app)  # #F4

    assert len(app.slv.tool_results) == 1
    res = app.slv.tool_results[0]
    assert res["id"] == "c2"
    assert res["ok"] is False
    assert "unknown tool" in (res["error"] or "")


@pytest.mark.asyncio
async def test_server_tool_call_handler_failure_returns_error_result():
    reg = ToolRegistry()

    @reg.tool(name="grip")
    def grip() -> dict:
        return {"success": False, "error": "serial bus unavailable"}

    app = _make_app(server_loop=True, registry=reg)

    evt = ServerToolCall(id="c3", name="grip", arguments={})
    await app._dispatch_one(evt)
    await _drain_tool_tasks(app)  # #F4

    res = app.slv.tool_results[0]
    assert res["ok"] is False
    assert res["error"] == "serial bus unavailable"


@pytest.mark.asyncio
async def test_server_tool_call_trigger_guard_blocks_wrong_motion_mapping():
    calls: list[dict] = []
    reg = ToolRegistry()

    @reg.tool(
        name="point_at",
        description=(
            'Point forward. Triggers: "指向", "指一下", "point at". '
            'Do not use for head nodding: "点头" is unsupported.'
        ),
    )
    def point_at() -> dict:
        calls.append({"ran": True})
        return {"started": True, "action": "point_at"}

    app = _make_app(server_loop=True, registry=reg)
    app.config.tool_trigger_guard = True
    app._last_user_utterance_text = "请点头。"

    evt = ServerToolCall(id="c_guard", name="point_at", arguments={})
    await app._dispatch_one(evt)
    await _drain_tool_tasks(app)

    assert calls == []
    res = app.slv.tool_results[0]
    assert res["id"] == "c_guard"
    assert res["ok"] is False
    assert "no trigger phrase" in (res["error"] or "")


@pytest.mark.asyncio
async def test_server_tool_call_is_dispatched_in_background_non_blocking():
    """#F4: _dispatch_one must NOT block on the tool's execution — it schedules
    a tracked background task and returns immediately, so the dispatch loop keeps
    draining events (TTS audio / partials / control frames) while the tool runs."""
    reg = ToolRegistry()
    started = asyncio.Event()
    release = asyncio.Event()

    @reg.tool(name="slow")
    async def slow() -> dict:
        started.set()
        await release.wait()
        return {"started": True}

    app = _make_app(server_loop=True, registry=reg)
    evt = ServerToolCall(id="cb", name="slow", arguments={})

    await app._dispatch_one(evt)  # must return before the tool finishes
    await asyncio.wait_for(started.wait(), timeout=1.0)
    # Handler is running but the dispatch entry already returned: the result is
    # NOT sent yet, and the task is tracked (strong ref → no GC).
    assert app.slv.tool_results == [], "dispatch blocked on the tool (not backgrounded)"
    assert len(app._pending_tool_tasks) == 1

    release.set()
    await _drain_tool_tasks(app)
    assert len(app.slv.tool_results) == 1
    assert app.slv.tool_results[0]["id"] == "cb"
    assert app.slv.tool_results[0]["ok"] is True
    assert not app._pending_tool_tasks  # done-callback pruned the set


# ── 3. off mode: local LLM loop still runs (zero behaviour change) ────


@pytest.mark.asyncio
async def test_off_mode_runs_local_llm_loop():
    cfg = Config(system_prompt="SYS", server_loop=False)
    slv = FakeSLV()
    llm = FakeLLM(["你", "好"])
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()

    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")

    await mgr.current.on_user_utterance(ctx, "hi")

    # Local LLM was invoked, tokens streamed, history grew — legacy path.
    assert llm.stream_called is True
    assert slv.text_frames == ["你", "好"]
    assert slv.flushed == 1
    assert session.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "你好"},
    ]


@pytest.mark.asyncio
async def test_server_loop_skips_local_llm_loop():
    cfg = Config(system_prompt="SYS", server_loop=True)
    slv = FakeSLV()
    llm = FakeLLM(["你", "好"])
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()

    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")

    await mgr.current.on_user_utterance(ctx, "hi")

    # Server owns the loop now: agent did NOT call its LLM, did NOT push
    # text to TTS, and did NOT touch local history.
    assert llm.stream_called is False
    assert slv.text_frames == []
    assert slv.flushed == 0
    assert session.history == []


# ── config flag resolution ───────────────────────────────────────────


def test_config_server_loop_default_off():
    assert Config().server_loop is False
    assert Config().server_loop_enabled() is False


def test_config_env_overrides(monkeypatch):
    cfg = Config(server_loop=False)
    monkeypatch.setenv("OVS_AGENT_SERVER_LOOP", "true")
    assert cfg.server_loop_enabled() is True
    monkeypatch.setenv("OVS_AGENT_SERVER_LOOP", "off")
    assert cfg.server_loop_enabled() is False
    # Field still wins when env unset.
    monkeypatch.delenv("OVS_AGENT_SERVER_LOOP", raising=False)
    assert Config(server_loop=True).server_loop_enabled() is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        # ── truthy ────────────────────────────────────────────────
        ("1", True),
        ('"1"', True),       # ROOT CAUSE: --env-file delivers literal quotes
        ("'1'", True),       # single-quote variant
        (" 1 ", True),       # surrounding whitespace
        ('" 1 "', True),     # quotes + inner whitespace
        ("true", True),
        ('"true"', True),
        ("'true'", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        # ── falsy ─────────────────────────────────────────────────
        ("0", False),
        ('"0"', False),
        ("false", False),
        ('"false"', False),
        ("no", False),
        ("off", False),
        ("", False),
        ('""', False),       # empty quoted string normalizes to empty
    ],
)
def test_server_loop_env_quote_stripping(monkeypatch, raw, expected):
    """OVS_AGENT_SERVER_LOOP must parse literal-quoted env-file values.

    Regression guard for the 2026-05-31 production activation bug: values
    injected via ``--env-file`` arrive in os.environ carrying literal quote
    characters (e.g. the 3-char string '"1"'), and the old ``.strip().lower()``
    parser left the quotes in place → never matched "1" → server-loop silently
    disabled in prod while quote-free ``-e FLAG=1`` isolated runs passed.
    """
    monkeypatch.setenv("OVS_AGENT_SERVER_LOOP", raw)
    # config field is False, so any True result comes purely from the env.
    assert Config(server_loop=False).server_loop_enabled() is expected


def test_server_loop_env_unset_defers_to_field(monkeypatch):
    """Unset env defers to the config field for both truth values."""
    monkeypatch.delenv("OVS_AGENT_SERVER_LOOP", raising=False)
    assert Config(server_loop=False).server_loop_enabled() is False
    assert Config(server_loop=True).server_loop_enabled() is True


# ── 4. LLM warmup skip in server-loop mode (2026-05-31 fix) ──────────


class _WarmupRecordingLLM(LLMBackend):
    """LLM backend that records whether warmup() was invoked."""

    def __init__(self) -> None:
        self.warmup_called = False

    async def stream(self, messages, **kw):  # type: ignore[override]
        if False:  # pragma: no cover - never streamed in these tests
            yield ""

    async def warmup(self, *, system_prompt="", tools=None, enable_thinking=False):  # type: ignore[override]
        self.warmup_called = True
        return {"cache_warmed": True, "graph_warmed": True}


def _make_warmup_app(server_loop: bool) -> BaseApp:
    app = _make_app(server_loop=server_loop, registry=ToolRegistry())
    app.llm = _WarmupRecordingLLM()
    # _make_app already wires config/session; plugins not needed for warmup.
    return app


@pytest.mark.asyncio
async def test_warmup_skipped_in_server_loop_mode(caplog):
    """server-loop on → local LLM warmup must NOT run (avoids the misleading
    'Session budget OK' / edge-llm traffic that derailed the 2026-05-31 triage)."""
    import logging

    app = _make_warmup_app(server_loop=True)
    with caplog.at_level(logging.INFO):
        await app._maybe_run_llm_warmup()

    assert app.llm.warmup_called is False
    assert any(
        "skipping local LLM warmup" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_warmup_runs_in_client_loop_mode(monkeypatch):
    """client-loop (default) → local LLM warmup runs exactly as before."""
    monkeypatch.delenv("OVS_AGENT_SERVER_LOOP", raising=False)
    app = _make_warmup_app(server_loop=False)

    await app._maybe_run_llm_warmup()

    assert app.llm.warmup_called is True
    # Side effects of a successful warmup still applied (behaviour unchanged).
    assert app.session.prefix_cache_warmed is True
    assert app.session.graph_warmed is True


@pytest.mark.asyncio
async def test_warmup_env_quoted_flag_skips_warmup(monkeypatch):
    """The production failure shape: --env-file delivers '"1"' (with quotes).
    With the quote-stripping fix this resolves True → warmup is skipped."""
    monkeypatch.setenv("OVS_AGENT_SERVER_LOOP", '"1"')
    app = _make_warmup_app(server_loop=False)  # config off; env forces on

    await app._maybe_run_llm_warmup()

    assert app.llm.warmup_called is False


# ── 5. advertise skip log + boot diagnostic ──────────────────────────


@pytest.mark.asyncio
async def test_off_mode_advertise_logs_skip_reason(caplog, monkeypatch):
    """Flag off → advertise no-ops AND logs a diagnosable skip reason."""
    import logging

    monkeypatch.delenv("OVS_AGENT_SERVER_LOOP", raising=False)
    app = _make_app(server_loop=False, registry=_ten_tool_registry())
    with caplog.at_level(logging.INFO):
        await app._advertise_tools_if_server_loop()

    assert app.slv.advertised == []
    assert any(
        "server-loop mode OFF" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_boot_diagnostic_logs_pid_raw_flag_and_resolved(caplog, monkeypatch):
    """Boot diagnostic must expose pid + repr(raw flag) + resolved decision so
    the next prod incident is one grep away. repr() makes quotes visible."""
    import logging
    import os

    monkeypatch.setenv("OVS_AGENT_SERVER_LOOP", '"1"')
    app = _make_app(server_loop=False, registry=ToolRegistry())
    with caplog.at_level(logging.INFO):
        app._log_boot_diagnostic()

    msgs = [r.getMessage() for r in caplog.records]
    boot = next((m for m in msgs if m.startswith("boot:")), None)
    assert boot is not None
    assert f"pid={os.getpid()}" in boot
    # repr() of the raw value preserves the literal quotes from --env-file.
    assert "OVS_AGENT_SERVER_LOOP(raw)='\"1\"'" in boot
    # Quote-stripping fix → resolves enabled despite literal quotes.
    assert "resolved server_loop_enabled=True" in boot
