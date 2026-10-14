# Tool Calling — User Guide

`ovs_agent` supports local, in-process **tool calling**: the
LLM can decide to invoke a Python function (a "tool") instead of (or in
addition to) producing text. The result is fed back into the same
conversation and the LLM either answers or chains another tool.

This is the OpenAI-compatible `tool_calls` flow, but every tool runs in
the agent process — no MCP server, no subprocess.

## When to enable it

Tools are **off by default**. Enable them in your YAML config when you
want voice users to be able to trigger actions ("what time is it",
"switch to translation mode", …). For pure-chat agents leave it off:
the framework bypasses the tool runner entirely and the LLM stream
goes straight to TTS, byte-equivalent to the pre-tool implementation.

```yaml
tools_enabled: true
tools_default_allowlist:
  - time_now
  - set_mode
tools_max_iterations: 5     # safety cap per user turn
```

The allowlist controls which registered tools the LLM is allowed to call:

| `tools_enabled` | `tools_default_allowlist` | Behavior |
|---|---|---|
| `false` | _anything_ | No tools sent to LLM |
| `true`  | `[a, b]` | Only `a` and `b` (must be registered) |
| `true`  | `[]` or unset | **All registered tools** |

The "all" case is the right default for single-mode solutions whose
plugins register exactly the tools they want exposed — no need to repeat
every action name in YAML.

Per-mode override (only the named mode sees these tools):

```yaml
mode_overrides:
  chat:
    tools_enabled: true
    tools_allowlist: [time_now]
  interpreter:
    tools_enabled: false    # translation mode shouldn't call tools
```

Resolution order per turn: `mode_overrides[<mode>]` → global config.

## Built-in tools

| Name        | Signature                | What it does |
|-------------|--------------------------|--------------|
| `time_now`  | `() -> {"now": iso8601}` | Returns the agent's local time. |
| `set_mode`  | `(mode_name: str)`       | Switches the active `AppMode`. |

They live in `agent/ovs_agent/tools/builtin.py` and are
registered on `default_registry` at import time.

## Writing a custom tool

```python
# my_plugin.py
from ovs_agent.tools import default_registry

@default_registry.tool(description="Turn the kitchen light on or off.")
def kitchen_light(on: bool) -> dict:
    """Returns the new state of the light."""
    _hardware_set("kitchen", on)
    return {"success": True, "on": on}
```

That's it. The decorator inspects type hints to build the JSON Schema
that goes to the LLM. The function's docstring (or the explicit
`description=`) becomes the tool description.

### Type → schema mapping

| Python                | JSON Schema       |
|-----------------------|-------------------|
| `str`                 | `"string"`        |
| `int`                 | `"integer"`       |
| `float`               | `"number"`        |
| `bool`                | `"boolean"`       |
| `list[T]` / `list`    | `"array"`         |
| `dict[…]` / `dict`    | `"object"`        |
| `Literal["a", "b"]`   | `"string"` + enum |
| `Optional[T]`         | same as `T`       |

Parameters with no default become `"required"`.

### Async tools

```python
@default_registry.tool(timeout_s=2.5)
async def weather(city: str) -> dict:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"https://api.example.com/weather?q={city}")
    return r.json()
```

`timeout_s` caps the call — exceeding it yields
`{"success": False, "error": "tool weather timed out after 2.5s"}`,
fed back to the LLM so it can self-recover (try a different city,
apologise to the user, etc.).

### The `ctx` parameter

Declare `ctx` (or `ctx: ToolCallCtx`) in your signature and the
registry will inject it without exposing it to the LLM:

```python
from ovs_agent.tools import ToolCallCtx

@default_registry.tool(description="Switch agent mode.")
def set_mode(mode_name: str, ctx: ToolCallCtx) -> dict:
    if mode_name not in ctx.mode_manager.available_modes():
        return {"success": False, "error": f"unknown mode: {mode_name}"}
    ctx.mode_manager.request_switch(mode_name)
    return {"success": True, "mode": mode_name}
```

`ToolCallCtx` carries:

- `session` — the current `Session` (read-only inspect of history).
- `mode_manager` — switch / inspect available `AppMode`s.
- `event_bus` — emit custom events back to dashboard plugins.
- `config` — the live `Config`.

## Per-session stability for prefix_cache

The edge-llm server-side prefix_cache keys on the full
`(system_prompt, tools, …history)` shape. If you mutate the allowlist
*during a session* the next request misses cache (safe — no crash,
just slower). The runner additionally forces `prefix_cache=False` on
all loop iterations after the first within a single user turn, because
the shape changes once `assistant(tool_calls)` and `tool` messages
appear.

Best practice: declare the allowlist once per mode in YAML and leave
it there.

## Dashboard events

When tools are enabled the agent emits these events on the EventBus
**and** broadcasts them to plugins:

| Event                       | Payload |
|-----------------------------|---------|
| `tool_call_started`         | `{id, name, arguments_json}` |
| `tool_call_completed`       | `{id, name, ok, duration_ms}` |
| `tool_call_error`           | `{id, name, error}` (only when `ok=False`) |
| `on_tool_iteration_limit`   | `{iterations, dropped, sid}` |

The dashboard plugin renders the most recent 5 calls in a "Tools"
panel.

## TTS behaviour with preamble text

If the LLM emits text before a tool call ("好的，我查一下…"), the
preamble streams to TTS immediately. The user hears it, then there's a
silence window for tool execution, then the post-tool answer continues
on the same TTS stream. We accept this trade — holding the preamble
would add latency to every text-only turn.

For long-latency tools, return a `display` field your code path can
speak as a filler later.

## Limitations

- **Sequential execution.** Tool A finishes before tool B starts.
  Voice latency budget doesn't tolerate parallel coordination
  complexity. If the LLM emits two `tool_calls` in one assistant
  message they run one after another.
- **In-process only.** No MCP server, no subprocess. Every tool is
  trusted code in the agent's process. MCP is a future adapter.
- **Iteration cap.** `tools_max_iterations` (default 5) per turn —
  exceeding it rolls the partial round back and returns empty text
  (the user sees no reply). Tune up if you build a tool-heavy
  workflow.
- **No streaming results.** Tool results are short JSON; they're not
  re-streamed to TTS. Only the LLM's *answer* about the tool result
  reaches the speaker.
- **Cancellation poisoning.** Barge-in during a tool call rolls back
  the in-flight `assistant(tool_calls)` + `tool` messages from
  `Session.history` so the next turn sees a clean shape.

  **Tool authors MUST NOT swallow `asyncio.CancelledError`.** If a tool
  catches it (e.g. via `except BaseException` or a bare `except:`), the
  barge-in latency extends to the tool's full runtime — the runner can
  only finish rolling back once the tool actually returns. Use
  `try/finally` for cleanup; let `CancelledError` propagate.

## Complete example

`config.yaml`:

```yaml
slv_url: ws://localhost:8621/v2v/stream
llm_backend: edge_llm
llm_base_url: http://localhost:8000/v1
llm_model: qwen2.5-3b-instruct

tools_enabled: true
tools_default_allowlist: [time_now, kitchen_light]
tools_max_iterations: 5

mode_overrides:
  chat:
    system_prompt: |
      You are a helpful voice assistant. You can call tools to query
      time or control the kitchen light. Reply concisely.
```

`my_plugin.py` (loaded via the plugin discovery mechanism):

```python
from ovs_agent.tools import default_registry

@default_registry.tool(description="Turn the kitchen light on or off.")
def kitchen_light(on: bool) -> dict:
    _hardware_set("kitchen", on)
    return {"success": True, "on": on}
```

Voice: "关一下厨房的灯" → LLM emits `tool_calls=[kitchen_light(on=false)]`
→ runner dispatches → `{"success": true, "on": false}` → LLM emits
"好的，厨房灯已关。" → TTS speaks the answer.

## Model Context Protocol (MCP) integration

`ovs_agent` can bridge external MCP servers as tools via `McpBridgePlugin`. Tools exposed by configured MCP servers are dynamically registered into `default_registry` and surfaced to the LLM during dialogue turns.

When no `mcp_servers` are configured, the MCP subsystem remains completely inactive with **zero runtime overhead**.

### Transports

1. **`sse` (Recommended for remote / decoupled services):**
   Communicates over HTTP + Server-Sent Events. Best for connecting to tools hosted on a central server, Home Assistant, or independent microservices without polluting the edge device's Docker image.
2. **`stdio` (For local scripts / binaries):**
   Launches the tool as a subprocess and speaks JSON-RPC over `stdin` and `stdout`.

### Configuration in YAML

```yaml
tools_enabled: true
tools_default_allowlist: []  # expose all registered tools including MCP tools

mcp_servers:
  # Remote Home Assistant or custom API via SSE
  - name: home_assistant
    transport: sse
    url: "http://192.168.1.100:8000/sse"
    timeout_s: 5.0
    preamble_text: "好的，正在处理。"  # Spoken acknowledgement emitted via TTS immediately

  # Local Python script via stdio
  - name: local_device
    transport: stdio
    command: python3
    args: ["/opt/tools/device_mcp.py"]
    timeout_s: 10.0
```

### Installation

MCP support is included via the optional extra:

```bash
uv run --extra mcp ovs-agent run conversation
```

