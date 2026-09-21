"""The LLM warmup must send the tool set the turns will send.

rk3588 + RK1828 devkit (2026-09-21): the conversation app sends no tools on its
turns (tools_enabled=False), but the warmup sent every registered tool. The
RK1828 worker keeps registered tools across requests, so every later tool-less
turn was rendered with the tool preamble (prefill 220 vs 28 tokens for the same
request) and the model "called" set_mode as plain text that reached TTS.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ovs_agent.app_base import BaseApp


class _Registry:
    def __init__(self) -> None:
        self.allows: list = []

    def list_openai_tools(self, allow=None):
        self.allows.append(allow)
        names = ["set_mode", "get_time"] if allow is None else sorted(allow)
        return [{"type": "function", "function": {"name": n}} for n in names]


class _LLM:
    def __init__(self) -> None:
        self.tools_seen: list = []

    async def warmup(self, *, system_prompt, tools, enable_thinking):
        self.tools_seen.append(tools)
        return None


def _app(**cfg) -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.config = SimpleNamespace(
        default_mode="chat", mode_overrides={}, system_prompt="sp",
        tools_enabled=False, tools_default_allowlist=[], server_loop=False, **cfg,
    )
    app.llm = _LLM()
    app.tool_registry = _Registry()
    app.session = None
    app._validate_session_budget = lambda sp, tools: None  # type: ignore[assignment]
    return app


@pytest.mark.asyncio
async def test_no_tools_on_turns_means_no_tools_in_warmup():
    app = _app()
    await app._maybe_run_llm_warmup()
    assert app.llm.tools_seen == [None]
    assert app.tool_registry.allows == [], "registry must not even be asked"


@pytest.mark.asyncio
async def test_tools_enabled_without_allowlist_warms_every_tool():
    app = _app()
    app.config.tools_enabled = True
    await app._maybe_run_llm_warmup()
    assert app.tool_registry.allows == [None]
    assert [t["function"]["name"] for t in app.llm.tools_seen[0]] == ["set_mode", "get_time"]


@pytest.mark.asyncio
async def test_mode_override_wins_over_global_default():
    app = _app()
    app.config.mode_overrides = {"chat": {"tools_enabled": True, "tools_allowlist": ["get_time"]}}
    await app._maybe_run_llm_warmup()
    assert app.tool_registry.allows == [{"get_time"}]

    off = _app(); off.config.tools_enabled = True
    off.config.mode_overrides = {"chat": {"tools_enabled": False}}
    await off._maybe_run_llm_warmup()
    assert off.llm.tools_seen == [None]



@pytest.mark.asyncio
async def test_null_override_inherits_the_global_setting():
    """A turn treats a present-but-None override as "not set" (Codex review)."""
    app = _app()
    app.config.tools_enabled = True
    app.config.mode_overrides = {"chat": {"tools_enabled": None}}
    await app._maybe_run_llm_warmup()
    assert app.tool_registry.allows == [None]


class _Mode:
    def __init__(self, name, **attrs):
        self.name = name
        for k, v in attrs.items():
            setattr(self, k, v)


@pytest.mark.asyncio
async def test_mode_object_defaults_and_actual_startup_mode_are_used():
    """Turns fall back to attributes on the active mode, and the ModeManager
    starts the first registered mode when default_mode is missing."""
    app = _app(); app.config.default_mode = "missing"
    app.modes = SimpleNamespace(
        _current=None,
        _modes={"robot": _Mode("robot", tools_enabled=True, tools_allowlist=["get_time"])},
    )
    await app._maybe_run_llm_warmup()
    assert app.tool_registry.allows == [{"get_time"}]

    started = _app()
    started.modes = SimpleNamespace(_current=_Mode("chat", tools_enabled=False), _modes={})
    started.config.tools_enabled = True
    await started._maybe_run_llm_warmup()
    assert started.llm.tools_seen == [None]
