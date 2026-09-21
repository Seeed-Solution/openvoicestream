"""The availability probe must not evict a local runtime's prompt cache.

RK1828 (2026-09-21): the runtime keeps one prompt prefix in its KV cache.
The 30 s chat probe ("." / max_tokens=1) replaced it, so the next real turn
re-prefilled the whole system prompt + history: 168 ms → 1367 ms TTFT after a
single probe, 2.0-2.6 s per turn in conversation. The probe now prefers a
``/health`` that reports worker status, and skips a cycle after a real success.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from ovs_agent.plugins import llm_availability as mod
from ovs_agent.plugins.llm_availability import AvailabilityState, LLMAvailabilityPlugin


def _app(mode: str | None = None):
    cfg = SimpleNamespace(
        llm_base_url="http://127.0.0.1:1828/v1",
        llm_api_key="",
        llm_model="Qwen3-4B",
        llm_availability_probe_interval_s=30.0,
        llm_availability_probe_timeout_s=0.5,
        llm_availability_failures_to_down=3,
    )
    if mode is not None:
        cfg.llm_availability_probe_mode = mode
    return SimpleNamespace(config=cfg, events=None, llm_availability=None)


def _plugin(monkeypatch, handler, mode=None) -> tuple[LLMAvailabilityPlugin, list[str]]:
    seen: list[str] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        seen.append(f"{req.method} {req.url.path}")
        return handler(req)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        mod.httpx, "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(_handler), **kw),
    )
    p = LLMAvailabilityPlugin(_app(mode))
    p._wake_evt = asyncio.Event()
    p._probe_lock = asyncio.Lock()
    return p, seen


def _chat_ok() -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})


@pytest.mark.asyncio
async def test_auto_uses_health_and_never_touches_chat(monkeypatch):
    def h(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "model": "Qwen3-4B"})
        return _chat_ok()

    p, seen = _plugin(monkeypatch, h)
    assert await p._probe() is True
    assert await p._probe() is True
    assert seen == ["GET /health", "GET /health"], "chat probe would evict the prompt cache"


@pytest.mark.asyncio
async def test_health_status_unavailable_is_a_failure(monkeypatch):
    p, _ = _plugin(monkeypatch, lambda req: httpx.Response(
        200, json={"status": "unavailable"}))
    assert await p._probe() is False


@pytest.mark.asyncio
async def test_auto_falls_back_to_chat_once_when_health_missing(monkeypatch):
    def h(req):
        if req.url.path == "/health":
            return httpx.Response(404, text="not found")
        return _chat_ok()

    p, seen = _plugin(monkeypatch, h)
    assert await p._probe() is True
    assert await p._probe() is True
    assert seen == ["GET /health", "POST /v1/chat/completions",
                    "POST /v1/chat/completions"], "detect once, then stay on chat"


@pytest.mark.asyncio
async def test_health_200_without_status_is_not_trusted(monkeypatch):
    """A generic 200 page is not a worker report — fall back to chat."""
    def h(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"hello": "world"})
        return _chat_ok()

    p, seen = _plugin(monkeypatch, h)
    assert await p._probe() is True
    assert seen == ["GET /health", "POST /v1/chat/completions"]


@pytest.mark.asyncio
async def test_chat_mode_never_calls_health(monkeypatch):
    p, seen = _plugin(monkeypatch, lambda req: _chat_ok(), mode="chat")
    assert await p._probe() is True
    assert seen == ["POST /v1/chat/completions"]


@pytest.mark.asyncio
async def test_recent_real_success_skips_probe_without_double_counting(monkeypatch):
    p, seen = _plugin(monkeypatch, lambda req: _chat_ok(), mode="chat")
    p.state = AvailabilityState.DOWN
    p.report_request_success()                     # DOWN → RECOVERING
    assert p.state == AvailabilityState.RECOVERING

    p._stopped = False
    task = asyncio.create_task(p.run())
    await asyncio.sleep(0.05)                      # one loop iteration
    p._stopped = True
    p._wake_evt.set()
    await asyncio.wait_for(task, 1.0)

    assert seen == [], "a real success within the interval must skip the probe"
    assert p.state == AvailabilityState.RECOVERING, (
        "a skipped cycle must not count as a second confirmation"
    )


@pytest.mark.asyncio
async def test_health_503_while_detecting_does_not_latch_chat(monkeypatch):
    """A server that is still booting answers /health 503 once; detection must
    retry /health later instead of using the chat probe for the whole session."""
    calls = {"health": 0}

    def h(req):
        if req.url.path == "/health":
            calls["health"] += 1
            if calls["health"] == 1:
                return httpx.Response(503, json={"detail": "starting"})
            return httpx.Response(200, json={"status": "ok"})
        return _chat_ok()

    p, seen = _plugin(monkeypatch, h)
    assert await p._probe() is True          # 503 → chat this cycle only
    assert await p._probe() is True          # /health retried, now confirmed
    assert await p._probe() is True
    assert seen == ["GET /health", "POST /v1/chat/completions",
                    "GET /health", "GET /health"]
