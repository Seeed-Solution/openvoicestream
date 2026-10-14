"""MCP Client connection wrapper supporting stdio and SSE transports."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def _extract_content(result: Any) -> str:
    """Extract plain text from CallToolResult content."""
    content_list = getattr(result, "content", [])
    if not isinstance(content_list, (list, tuple)):
        return str(result)
    texts: list[str] = []
    for item in content_list:
        text = getattr(item, "text", None)
        if text is not None:
            texts.append(str(text))
        elif hasattr(item, "data"):
            texts.append(f"[{getattr(item, 'type', 'binary')} data]")
        else:
            texts.append(str(item))
    return "\n".join(texts) if texts else ""


class McpConnection:
    """Manages an active MCP ClientSession over stdio or SSE."""

    def __init__(
        self,
        name: str,
        *,
        transport: str = "stdio",
        command: str | None = None,
        args: Sequence[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 10.0,
        session_factory: Any | None = None,
    ) -> None:
        self.name = name
        self.transport = transport.lower()
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {}) if env else None
        self.url = url
        self.headers = dict(headers or {}) if headers else None
        self.timeout_s = float(timeout_s)
        self._session_factory = session_factory
        self._stack: contextlib.AsyncExitStack | None = None
        self._session: Any = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def connect(self) -> None:
        """Establish connection to the MCP server and initialize the session."""
        if self._connected:
            return

        stack = contextlib.AsyncExitStack()
        try:
            if self._session_factory is not None:
                # Testing hook: caller supplied a custom session factory
                session = await stack.enter_async_context(self._session_factory())
            elif self.transport == "stdio":
                if not self.command:
                    raise ValueError(f"MCP server '{self.name}' stdio transport requires 'command'")
                try:
                    from mcp import ClientSession, StdioServerParameters
                    from mcp.client.stdio import stdio_client
                except ImportError as exc:
                    raise RuntimeError(
                        "mcp package is required for MCP integration. Install with 'uv pip install mcp'."
                    ) from exc

                server_params = StdioServerParameters(
                    command=self.command,
                    args=self.args,
                    env=self.env,
                )
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(server_params)
                )
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
            elif self.transport == "sse":
                if not self.url:
                    raise ValueError(f"MCP server '{self.name}' sse transport requires 'url'")
                try:
                    from mcp import ClientSession
                    from mcp.client.sse import sse_client
                except ImportError as exc:
                    raise RuntimeError(
                        "mcp package is required for MCP integration. Install with 'uv pip install mcp'."
                    ) from exc

                read_stream, write_stream = await stack.enter_async_context(
                    sse_client(self.url, headers=self.headers)
                )
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
            else:
                raise ValueError(
                    f"Unsupported MCP transport '{self.transport}' for server '{self.name}'. "
                    "Supported transports: 'stdio', 'sse'."
                )

            self._stack = stack
            self._session = session
            self._connected = True
            logger.info("Connected to MCP server '%s' via %s", self.name, self.transport)
        except Exception:
            await stack.aclose()
            self._connected = False
            self._session = None
            self._stack = None
            raise

    async def list_tools(self) -> list[dict[str, Any]]:
        """Query available tools from the server."""
        if not self._connected or self._session is None:
            raise RuntimeError(f"MCP server '{self.name}' is not connected")

        resp = await asyncio.wait_for(self._session.list_tools(), timeout=self.timeout_s)
        raw_tools = getattr(resp, "tools", []) or []
        out: list[dict[str, Any]] = []
        for t in raw_tools:
            name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "")
            description = getattr(t, "description", None) or (
                t.get("description") if isinstance(t, dict) else ""
            ) or ""
            schema = (
                getattr(t, "input_schema", None)
                or getattr(t, "inputSchema", None)
                or (t.get("input_schema") if isinstance(t, dict) else None)
                or (t.get("inputSchema") if isinstance(t, dict) else None)
                or {}
            )
            out.append({
                "name": str(name),
                "description": str(description),
                "parameters": dict(schema) if isinstance(schema, dict) else {},
            })
        return out

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool and return a JSON-friendly dict."""
        if not self._connected or self._session is None:
            return {"success": False, "error": f"MCP server '{self.name}' is not connected"}

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments=arguments),
                timeout=self.timeout_s,
            )
            is_error = getattr(result, "isError", False)
            text_out = _extract_content(result)
            if is_error:
                return {"success": False, "error": text_out or "tool error"}
            return {"success": True, "result": text_out}
        except asyncio.TimeoutError:
            logger.warning("MCP tool %s.%s timed out after %.1fs", self.name, name, self.timeout_s)
            return {
                "success": False,
                "error": f"MCP tool '{name}' timed out after {self.timeout_s:.1f}s",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP tool %s.%s raised %r", self.name, name, exc)
            return {"success": False, "error": str(exc)}

    async def close(self) -> None:
        """Close connection and release resources."""
        self._connected = False
        self._session = None
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                logger.debug("Error closing MCP connection '%s'", self.name, exc_info=True)
            finally:
                self._stack = None
        logger.info("Closed MCP connection '%s'", self.name)
