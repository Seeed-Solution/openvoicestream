"""Real end-to-end integration test with a live subprocess MCP server (stdio transport)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from ovs_agent.mcp.client import McpConnection
from ovs_agent.mcp.plugin import McpBridgePlugin
from ovs_agent.tools.registry import ToolRegistry

SERVER_SCRIPT = """
from mcp.server.mcpserver import MCPServer

server = MCPServer("test_device_service")

@server.tool()
def compute_sum(x: int, y: int) -> int:
    '''Compute sum of two numbers.'''
    return x + y

@server.tool()
def get_hardware_status() -> str:
    '''Return current edge hardware status.'''
    return "RK3576 NPU ONLINE"

if __name__ == "__main__":
    server.run(transport="stdio")
"""


@pytest.mark.asyncio
async def test_real_stdio_mcp_server_e2e():
    """Verify real stdio child process spawning, JSON-RPC handshake, and dispatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        srv_file = Path(tmpdir) / "test_mcp_server.py"
        srv_file.write_text(SERVER_SCRIPT, encoding="utf-8")

        registry = ToolRegistry()
        conn = McpConnection(
            name="live_server",
            transport="stdio",
            command=sys.executable,
            args=[str(srv_file)],
            timeout_s=5.0,
        )
        conn._preamble_text = "好的，正在计算。"
        conn._response_mode = "await"

        class DummyApp:
            def __init__(self):
                self.config = None

        plugin = McpBridgePlugin(DummyApp(), registry=registry, connections=[conn])
        assert plugin.setup() is True

        # 1. Start: Spawns child process, handshakes MCP, registers tools
        await plugin.start()

        assert registry.has("compute_sum")
        assert registry.has("get_hardware_status")

        # 2. Check OpenAI-compatible Schema
        tools_schema = registry.list_openai_tools()
        names = [t["function"]["name"] for t in tools_schema]
        assert "compute_sum" in names
        assert "get_hardware_status" in names

        # 3. Call tool via registry dispatch (simulating what LLM does)
        result = await registry.dispatch("compute_sum", {"x": 17, "y": 25}, ctx=None)
        assert result["success"] is True
        assert "42" in result["result"]

        # Call second tool
        hw_result = await registry.dispatch("get_hardware_status", {}, ctx=None)
        assert hw_result["success"] is True
        assert "RK3576 NPU ONLINE" in hw_result["result"]

        # 4. Stop: Cleanly terminates child process and unregisters tools
        await plugin.stop()
        assert not registry.has("compute_sum")
        assert not registry.has("get_hardware_status")


@pytest.mark.asyncio
async def test_real_sse_mcp_server_e2e():
    """Verify live HTTP/SSE server connection, tool discovery, and dispatch."""
    import asyncio
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("live_sse_device")

    @server.tool()
    def toggle_relay(switch_id: int, state: bool) -> dict:
        """Control a hardware relay switch."""
        return {"switch": switch_id, "state": state, "status": "ok"}

    server_task = asyncio.create_task(
        server.run_sse_async(host="127.0.0.1", port=19876, sse_path="/sse")
    )
    await asyncio.sleep(0.5)

    try:
        registry = ToolRegistry()
        conn = McpConnection(
            name="sse_device",
            transport="sse",
            url="http://127.0.0.1:19876/sse",
            timeout_s=5.0,
        )
        conn._preamble_text = "好的，正在切换继电器。"

        class DummyApp:
            def __init__(self):
                self.config = None

        plugin = McpBridgePlugin(DummyApp(), registry=registry, connections=[conn])
        assert plugin.setup() is True

        await plugin.start()
        assert registry.has("toggle_relay")

        tools_schema = registry.list_openai_tools()
        assert any(t["function"]["name"] == "toggle_relay" for t in tools_schema)

        res = await registry.dispatch("toggle_relay", {"switch_id": 2, "state": True}, ctx=None)
        assert res["success"] is True
        assert "ok" in res["result"]

        await plugin.stop()
        assert not registry.has("toggle_relay")
    finally:
        server_task.cancel()
        try:
            await server_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_mcp_full_dialogue_runner_flow():
    """Verify stream_with_tools executes real MCP tool and completes 2-round voice dialogue."""
    from ovs_agent.llm import LLMEvent
    from ovs_agent.session import Session
    from ovs_agent.tools import ToolCallCtx, stream_with_tools

    class _FakeLLM:
        def __init__(self, script):
            self._script = list(script)
            self._call_idx = 0

        async def stream_events(self, messages, **kwargs):
            events = self._script[self._call_idx]
            self._call_idx += 1
            for ev in events:
                yield ev

    with tempfile.TemporaryDirectory() as tmpdir:
        srv_file = Path(tmpdir) / "test_mcp_server.py"
        srv_file.write_text(SERVER_SCRIPT, encoding="utf-8")

        registry = ToolRegistry()
        conn = McpConnection(
            name="live_server",
            transport="stdio",
            command=sys.executable,
            args=[str(srv_file)],
            timeout_s=5.0,
        )
        conn._preamble_text = "好的，正在计算。"

        class DummyApp:
            def __init__(self):
                self.config = None

        plugin = McpBridgePlugin(DummyApp(), registry=registry, connections=[conn])
        await plugin.start()

        # Script: Round 1 emits tool call -> Round 2 receives tool output and emits reply text
        script = [
            [
                LLMEvent(
                    kind="tool_call_delta",
                    tool_call_index=0,
                    tool_call_id="call_sum_123",
                    name="compute_sum",
                    arguments='{"x": 10, "y": 20}',
                ),
                LLMEvent(kind="finish", finish_reason="tool_calls"),
            ],
            [
                LLMEvent(kind="text", text="计算结果是30。"),
                LLMEvent(kind="finish", finish_reason="stop"),
            ],
        ]

        session = Session()
        llm = _FakeLLM(script)
        preambles_heard = []
        tokens_heard = []

        async def on_tok(tok):
            tokens_heard.append(tok)

        async def on_pre(text):
            preambles_heard.append(text)

        final_text = await stream_with_tools(
            llm,
            [{"role": "user", "content": "10加20等于多少"}],
            session=session,
            registry=registry,
            allowed_tools=None,
            ctx=ToolCallCtx(session=session),
            on_assistant_token=on_tok,
            on_tool_preamble=on_pre,
        )

        # Verify immediate voice preamble was triggered
        assert preambles_heard == ["好的，正在计算。"]
        # Verify final synthesized text
        assert final_text == "计算结果是30。"
        assert "".join(tokens_heard) == "计算结果是30。"

        # Verify session history captured tool round-trip cleanly
        assert len(session.history) == 3
        assert session.history[0]["role"] == "assistant"
        assert session.history[0]["tool_calls"][0]["function"]["name"] == "compute_sum"
        assert session.history[1]["role"] == "tool"
        assert "30" in session.history[1]["content"]
        assert session.history[2] == {"role": "assistant", "content": "计算结果是30。"}

        await plugin.stop()


