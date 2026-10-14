"""McpBridgePlugin — registers tools from configured MCP servers into ToolRegistry."""
from __future__ import annotations

import logging
from typing import Any

from ovs_agent.plugin import Plugin
from ovs_agent.tools import Tool, default_registry

from .client import McpConnection

logger = logging.getLogger(__name__)


class McpBridgePlugin(Plugin):
    """Bridge one or more Model Context Protocol (MCP) servers into the agent ToolRegistry."""

    name = "mcp_bridge"

    def __init__(self, app, *, registry=None, connections: list[McpConnection] | None = None) -> None:
        super().__init__(app)
        self.registry = registry or default_registry
        self._connections: list[McpConnection] = connections or []
        self._registered_tools: set[str] = set()

    def setup(self) -> bool:
        """Check configuration and prerequisites."""
        if self._connections:
            # Pre-injected connections (e.g. tests)
            return True

        servers_cfg = getattr(getattr(self.app, "config", None), "mcp_servers", None)
        if not servers_cfg:
            logger.debug("no mcp_servers configured; McpBridgePlugin disabled")
            return False

        if not isinstance(servers_cfg, (list, tuple)):
            logger.warning("mcp_servers configuration must be a list; got %s", type(servers_cfg).__name__)
            return False

        # Verify mcp package availability early
        try:
            import mcp  # noqa: F401
        except ImportError:
            logger.warning(
                "mcp_servers are configured but the 'mcp' package is not installed. "
                "Install with 'uv pip install mcp' or 'uv run --extra mcp ...'. "
                "MCP tools will not be registered."
            )
            return False

        self._connections.clear()
        for idx, item in enumerate(servers_cfg):
            if not isinstance(item, dict):
                logger.warning("ignoring invalid mcp_server entry at index %d: %r", idx, item)
                continue
            name = str(item.get("name") or f"mcp_{idx}")
            transport = str(item.get("transport") or "stdio").strip().lower()
            try:
                conn = McpConnection(
                    name=name,
                    transport=transport,
                    command=item.get("command"),
                    args=item.get("args"),
                    env=item.get("env"),
                    url=item.get("url"),
                    headers=item.get("headers"),
                    timeout_s=float(item.get("timeout_s", 10.0)),
                )
                # Stash extra tool attributes from server config
                conn._preamble_text = str(item.get("preamble_text") or "")
                conn._completion_text = str(item.get("completion_text") or "")
                conn._response_mode = str(item.get("response_mode") or "await")
                self._connections.append(conn)
            except Exception:
                logger.exception("failed to initialize MCP connection config for %r", name)

        if not self._connections:
            logger.info("no valid MCP servers found; skipping McpBridgePlugin")
            return False

        logger.info("McpBridgePlugin setup complete with %d configured server(s)", len(self._connections))
        return True

    async def start(self) -> None:
        """Connect to MCP servers and register discovered tools into the registry."""
        await super().start()
        for conn in self._connections:
            try:
                await conn.connect()
                tools = await conn.list_tools()
                for tool_meta in tools:
                    tname = tool_meta["name"]
                    if not tname:
                        continue
                    if self.registry.has(tname):
                        logger.warning(
                            "MCP tool '%s' from server '%s' overrides existing registered tool",
                            tname,
                            conn.name,
                        )

                    # Create caller closure capturing conn and tool name
                    async def _make_caller(c=conn, target_name=tname):
                        async def _caller(**kwargs):
                            return await c.call_tool(target_name, kwargs)
                        return _caller

                    caller_fn = await _make_caller(conn, tname)
                    # Use provided parameters JSON Schema; ensure minimum dict
                    parameters = tool_meta.get("parameters") or {"type": "object", "properties": {}}

                    tool_obj = Tool(
                        name=tname,
                        description=tool_meta.get("description") or f"Tool from MCP server {conn.name}",
                        parameters=parameters,
                        fn=caller_fn,
                        timeout_s=conn.timeout_s,
                        preamble_text=getattr(conn, "_preamble_text", ""),
                        completion_text=getattr(conn, "_completion_text", ""),
                        response_mode=getattr(conn, "_response_mode", "await"),
                    )

                    self.registry._tools[tname] = tool_obj
                    self._registered_tools.add(tname)
                    logger.info("Registered MCP tool: %s (from %s)", tname, conn.name)
            except Exception:
                logger.exception("Failed to connect or discover tools from MCP server '%s'", conn.name)

        logger.info("McpBridgePlugin started; registered %d tool(s)", len(self._registered_tools))

    async def stop(self) -> None:
        """Unregister MCP tools and cleanly shut down all connections."""
        await super().stop()
        # Unregister tools added by this plugin
        for tname in list(self._registered_tools):
            self.registry.unregister(tname)
        self._registered_tools.clear()

        # Close all active connections
        for conn in self._connections:
            try:
                await conn.close()
            except Exception:
                logger.debug("Error while closing connection %s", conn.name, exc_info=True)

        logger.info("McpBridgePlugin stopped and cleaned up all MCP tools")
