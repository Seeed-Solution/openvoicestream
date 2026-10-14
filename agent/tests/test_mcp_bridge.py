"""Tests for MCP (Model Context Protocol) integration in ovs_agent."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mcp")

from ovs_agent.config import Config
from ovs_agent.mcp.client import McpConnection
from ovs_agent.mcp.plugin import McpBridgePlugin
from ovs_agent.tools.registry import ToolRegistry


class FakeTextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeCallToolResult:
    def __init__(self, text: str, is_error: bool = False) -> None:
        self.content = [FakeTextContent(text)]
        self.isError = is_error


class FakeToolMeta:
    def __init__(self, name: str, description: str, schema: dict) -> None:
        self.name = name
        self.description = description
        self.inputSchema = schema


class FakeListToolsResult:
    def __init__(self, tools: list[FakeToolMeta]) -> None:
        self.tools = tools


@pytest.fixture
def clean_registry():
    return ToolRegistry()


@pytest.fixture
def fake_app():
    app = MagicMock()
    app.config = Config()
    return app


def test_mcp_plugin_setup_disabled_when_empty(fake_app, clean_registry):
    fake_app.config.mcp_servers = []
    plugin = McpBridgePlugin(fake_app, registry=clean_registry)
    assert plugin.setup() is False
    assert len(clean_registry.list_names()) == 0


def test_mcp_plugin_setup_handles_invalid_config(fake_app, clean_registry):
    fake_app.config.mcp_servers = "not-a-list"
    plugin = McpBridgePlugin(fake_app, registry=clean_registry)
    assert plugin.setup() is False


@pytest.mark.asyncio
async def test_mcp_connection_lifecycle_and_dispatch(clean_registry):
    # Setup mock session
    mock_session = AsyncMock()
    mock_session.list_tools.return_value = FakeListToolsResult([
        FakeToolMeta(
            name="get_weather",
            description="Get current weather for a location",
            schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        )
    ])
    mock_session.call_tool.return_value = FakeCallToolResult("Sunny 24C in Shenzhen")

    class FakeAsyncContext:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    conn = McpConnection(
        name="weather_service",
        transport="stdio",
        command="fake-cmd",
        session_factory=lambda: FakeAsyncContext(),
    )
    conn._preamble_text = "Checking weather..."
    conn._response_mode = "await"

    app = MagicMock()
    plugin = McpBridgePlugin(app, registry=clean_registry, connections=[conn])
    assert plugin.setup() is True

    # Start plugin - connects and registers tool
    await plugin.start()

    assert clean_registry.has("get_weather")
    registered = clean_registry._tools["get_weather"]
    assert registered.description == "Get current weather for a location"
    assert registered.preamble_text == "Checking weather..."
    assert registered.response_mode == "await"

    # Verify OpenAI tool schema export
    openai_tools = clean_registry.list_openai_tools()
    assert len(openai_tools) == 1
    assert openai_tools[0]["function"]["name"] == "get_weather"
    assert "city" in openai_tools[0]["function"]["parameters"]["properties"]

    # Verify dispatch
    res = await clean_registry.dispatch("get_weather", {"city": "Shenzhen"}, ctx=None)
    assert res == {"success": True, "result": "Sunny 24C in Shenzhen"}
    mock_session.call_tool.assert_awaited_once_with("get_weather", arguments={"city": "Shenzhen"})

    # Stop plugin - unregisters tool
    await plugin.stop()
    assert not clean_registry.has("get_weather")
    assert len(clean_registry.list_names()) == 0


@pytest.mark.asyncio
async def test_mcp_tool_error_handling(clean_registry):
    mock_session = AsyncMock()
    mock_session.list_tools.return_value = FakeListToolsResult([
        FakeToolMeta(
            name="failing_tool",
            description="A tool that returns isError=True",
            schema={"type": "object", "properties": {}},
        )
    ])
    mock_session.call_tool.return_value = FakeCallToolResult("device offline", is_error=True)

    class FakeAsyncContext:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    conn = McpConnection(
        name="failing_service",
        transport="sse",
        url="http://fake:8000/sse",
        session_factory=lambda: FakeAsyncContext(),
    )
    plugin = McpBridgePlugin(MagicMock(), registry=clean_registry, connections=[conn])
    await plugin.start()

    res = await clean_registry.dispatch("failing_tool", {}, ctx=None)
    assert res["success"] is False
    assert "device offline" in res["error"]

    await plugin.stop()


@pytest.mark.asyncio
async def test_mcp_tool_timeout_handling(clean_registry):
    mock_session = AsyncMock()
    mock_session.list_tools.return_value = FakeListToolsResult([
        FakeToolMeta(
            name="slow_tool",
            description="A slow tool that times out",
            schema={"type": "object", "properties": {}},
        )
    ])

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10.0)

    mock_session.call_tool.side_effect = _hang

    class FakeAsyncContext:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    conn = McpConnection(
        name="slow_service",
        transport="stdio",
        command="slow-cmd",
        timeout_s=0.1,  # Short timeout for test
        session_factory=lambda: FakeAsyncContext(),
    )
    plugin = McpBridgePlugin(MagicMock(), registry=clean_registry, connections=[conn])
    await plugin.start()

    res = await clean_registry.dispatch("slow_tool", {}, ctx=None)
    assert res["success"] is False
    assert "timed out" in res["error"]

    await plugin.stop()


def test_mcp_config_parsing(clean_registry):
    app = MagicMock()
    app.config = Config(
        mcp_servers=[
            {
                "name": "home_assistant",
                "transport": "sse",
                "url": "http://127.0.0.1:8123/sse",
                "headers": {"Authorization": "Bearer test-token"},
                "timeout_s": 5.0,
                "preamble_text": "正在执行智能家居指令。",
            },
            {
                "name": "local_arm",
                "transport": "stdio",
                "command": "python3",
                "args": ["-m", "arm_mcp"],
                "timeout_s": 15.0,
            },
        ]
    )

    plugin = McpBridgePlugin(app, registry=clean_registry)
    assert plugin.setup() is True
    assert len(plugin._connections) == 2

    conn1 = plugin._connections[0]
    assert conn1.name == "home_assistant"
    assert conn1.transport == "sse"
    assert conn1.url == "http://127.0.0.1:8123/sse"
    assert conn1.timeout_s == 5.0
    assert getattr(conn1, "_preamble_text", "") == "正在执行智能家居指令。"

    conn2 = plugin._connections[1]
    assert conn2.name == "local_arm"
    assert conn2.transport == "stdio"
    assert conn2.command == "python3"
    assert conn2.args == ["-m", "arm_mcp"]
    assert conn2.timeout_s == 15.0
