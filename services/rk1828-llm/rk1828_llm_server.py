"""OpenAI-compatible streaming HTTP shim for Qwen3-4B on the RK1828 PCIe EP.

Owns one persistent ``rknn_qwen3_demo`` server-mode subprocess (init once) and
exposes ``POST /v1/chat/completions`` (SSE streaming + non-streaming) and
``GET /v1/models``.

Worker IPC protocol (see examples/Qwen3/cpp/main.cc):
  stderr : "READY 1" handshake once model init completes; all diagnostics.
  stdin  : one request line per turn: ``<max_new_tokens>\\t<escaped prompt>``
  stdout : per token ``[uint32 LE len][utf8 bytes]``; ``0xFFFFFFFE`` = EOS.

The EP is a single-context device: every request is serialised on a lock.
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import math
import os
import queue
import re
import struct
import subprocess
import threading
import time
import uuid
from typing import Any, Iterator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

LOG = logging.getLogger("rk1828-llm")

END_OF_STREAM = 0xFFFFFFFE
_LEN = struct.Struct("<I")
MAX_FRAME_BYTES = 8 * 1024 * 1024  # a bigger length prefix means stdout desync
MODEL_ID = "Qwen3-4B"

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# A spoken reply fits in ~96 tokens; a JSON tool call does not. Requests that
# carry tools are floored at this so a caller tuned for speech does not
# truncate every call it asks for.
TOOL_MIN_MAX_TOKENS = 320

# Model init pushes ~3.2 GB over PCIe to the EP, so the READY handshake is slow
# by nature and the ceiling has to be a deployment knob: a slower disk or a
# bigger model needs more, and a host/client runtime skew needs LESS (it never
# becomes ready at all, and every attempt degrades the EP from 8 cores to 4 —
# see BUILD.md 'Single-EP exclusivity').
READY_TIMEOUT_S = 180.0
START_ATTEMPTS = 3
# After the last attempt, wait before exiting. The container's restart policy
# would otherwise re-enter this loop immediately and keep hammering a card that
# cannot load, forever, at three failed loads per cycle.
FAIL_COOLDOWN_S = 300.0


def _env_float(name: str, default: float) -> float:
    """Read a positive, finite float from the environment.

    ``float()`` happily returns nan/inf, and both compare False against ``<= 0``,
    so an explicit isfinite check is what keeps a typo from disabling the
    timeout entirely.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        LOG.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if not math.isfinite(val) or val <= 0:
        LOG.warning("%s=%r is not a positive finite number; using %s", name, raw, default)
        return default
    return val


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        LOG.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default
    if val < 1:
        LOG.warning("%s=%r must be >= 1; using %s", name, raw, default)
        return default
    return val


class WorkerError(RuntimeError):
    pass


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


class Qwen3Worker:
    def __init__(
        self,
        binary: str,
        model_dir: str,
        core_mask: str = "ff",
        max_context: int = 2048,
        start_attempts: int = START_ATTEMPTS,
        ready_timeout: float = READY_TIMEOUT_S,
    ) -> None:
        self.binary = binary
        self.model_dir = model_dir
        self.core_mask = core_mask
        self.max_context = max_context
        self.start_attempts = start_attempts
        self.ready_timeout = ready_timeout
        self.proc: Optional[subprocess.Popen] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self.last_stderr: List[str] = []

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        last: Optional[BaseException] = None
        for attempt in range(1, self.start_attempts + 1):
            try:
                self._spawn()
                LOG.info("worker ready (attempt %d)", attempt)
                return
            except BaseException as exc:  # noqa: BLE001
                last = exc
                LOG.error("worker start attempt %d failed: %s", attempt, exc)
                self.stop()
                if attempt < self.start_attempts:
                    time.sleep(2.0 * attempt)
        raise WorkerError(
            f"RK1828 worker failed to start after {self.start_attempts} attempts "
            f"(ready_timeout={self.ready_timeout:g}s each): {last}\n"
            "  If every attempt timed out after the SAME duration and the worker's "
            "last line was 'init qwen3 llm model', the model never loaded at all: "
            "the usual cause is a host/client RKNN3 runtime skew (the image's "
            "librknn3_api*.so vs the host's rknn3 package). The worker then blocks "
            "in read() on @transfer_proxy3 with no error. Compare the versions the "
            "entrypoint logs at startup, and mount the host's lib dir read-only.\n"
            "  A genuinely slow load (attempt 2 faster than attempt 1) instead needs "
            "a higher RK1828_READY_TIMEOUT.\n"
            "  Otherwise the EP may be degraded; a clean host reboot is likely required."
        )

    def _spawn(self) -> None:
        self._ready.clear()
        env = dict(os.environ)
        libdir = os.path.join(os.path.dirname(self.binary), "lib")
        env["LD_LIBRARY_PATH"] = f"{libdir}:/lib:" + env.get("LD_LIBRARY_PATH", "")
        args = [
            self.binary,
            self.model_dir,
            "--core-mask",
            self.core_mask,
            "--max-context",
            str(self.max_context),
            "-",
        ]
        LOG.info("spawning worker: %s", " ".join(args))
        self.proc = subprocess.Popen(
            args,
            cwd=os.path.dirname(self.binary),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="worker-stderr", daemon=True
        )
        self._stderr_thread.start()

        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if self._ready.wait(0.5):
                return
            if self.proc.poll() is not None:
                raise WorkerError(
                    f"worker exited during init rc={self.proc.returncode} "
                    f"tail={self.last_stderr[-6:]}"
                )
        raise WorkerError(f"worker READY handshake timed out tail={self.last_stderr[-6:]}")

    def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            self.last_stderr.append(line)
            if len(self.last_stderr) > 200:
                del self.last_stderr[:100]
            # Case-INsensitive: the TTS binary emits "ready", this one "READY 1".
            if "ready" in line.lower():
                self._ready.set()
            LOG.info("[worker] %s", line)

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        self._ready.clear()
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()

    def is_ready(self) -> bool:
        return (
            self.proc is not None
            and self.proc.poll() is None
            and self._ready.is_set()
        )

    # ── framed IO ────────────────────────────────────────────────────────
    def _read_exact(self, n: int) -> bytes:
        assert self.proc and self.proc.stdout
        buf = b""
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise WorkerError(
                    f"worker stdout EOF (rc={self.proc.returncode}) "
                    f"tail={self.last_stderr[-6:]}"
                )
            buf += chunk
        return buf

    def _run_request(self, prompt: str, max_new_tokens: int, q: "queue.Queue",
                     tools: Optional[List[dict]] = None) -> None:
        """Drive one request to its EOS frame, pushing pieces into ``q``.

        Runs on its own thread and owns the worker lock for the whole request.
        Decoupling it from the HTTP response iterator is load-bearing: if the
        client disconnects mid-stream (``curl | head``), this thread still reads
        every remaining frame through the EOS sentinel before releasing the lock.
        Abandoning a half-read request would leave the next one reading the
        previous request's tokens (desync) or block forever on the lock.
        """
        try:
            with self._lock:
                if not self.is_ready():
                    raise WorkerError("worker is not ready")
                assert self.proc and self.proc.stdin
                if tools:
                    # V2: hand the tool schema to the runtime, which renders it
                    # through the model's own Jinja chat template so the
                    # canonical Qwen3 preamble lands in a real system block.
                    # keep=0 and a single turn: the runtime accepts only one
                    # input per run, and keeping history would make this shared
                    # single-EP session stateful.
                    tools_json = json.dumps(tools, ensure_ascii=False)
                    line = (
                        "V2\t"
                        f"{max_new_tokens}\t0\t{_escape(tools_json)}\t1\t\t"
                        f"{_escape(prompt)}\n"
                    ).encode("utf-8")
                else:
                    line = f"{max_new_tokens}\t{_escape(prompt)}\n".encode("utf-8")
                self.proc.stdin.write(line)
                self.proc.stdin.flush()

                dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    (length,) = _LEN.unpack(self._read_exact(4))
                    if length == END_OF_STREAM:
                        tail = dec.decode(b"", final=True)
                        if tail:
                            q.put(("text", tail))
                        break
                    if length > MAX_FRAME_BYTES:
                        raise WorkerError(
                            f"frame length {length} > MAX_FRAME_BYTES={MAX_FRAME_BYTES}: "
                            "stdout desync (stray text on the frame channel)"
                        )
                    piece = dec.decode(self._read_exact(length))
                    if piece:
                        q.put(("text", piece))
        except BaseException as exc:  # noqa: BLE001
            LOG.exception("request failed")
            q.put(("error", str(exc)))
        finally:
            q.put(("done", None))

    def generate(
        self, prompt: str, max_new_tokens: int, timeout: float = 600.0,
        tools: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """Serialised streaming generation. Yields decoded text pieces."""
        q: "queue.Queue" = queue.Queue()
        threading.Thread(
            target=self._run_request,
            args=(prompt, max_new_tokens, q, tools),
            daemon=True,
        ).start()
        deadline = time.time() + timeout
        while True:
            try:
                kind, payload = q.get(timeout=max(1.0, deadline - time.time()))
            except queue.Empty:
                raise WorkerError(f"request timed out after {timeout}s")
            if kind == "text":
                yield payload
            elif kind == "error":
                raise WorkerError(payload)
            else:
                return


# ── tool calling ─────────────────────────────────────────────────────────
# The tool schema is handed to the runtime via rknn3_session_set_function_tools
# (worker V2 line), which renders it through the model's OWN Jinja chat template
# from the GGUF -- the canonical Qwen3 `{%- if tools %}` branch -- so the
# preamble lands in a real system block.
#
# This replaced a hand-written copy of that preamble injected into the user
# turn. Both work (the hand-written one measured 12/12 at temperature 0, because
# it reproduced the template's own wording), but the template is the model's own
# and does not have to be kept in sync by hand.
#
# Measured on device 2026-07-31: with tools registered, prefill for "打开客厅灯"
# goes 15 -> 239 tokens and the model emits a well-formed <tool_call>; without,
# it answers in prose and calls nothing.
#
# The model still emits the call as TEXT in the output stream, so the splitter
# below is required either way.
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


def _partial_tail(text: str, sentinel: str) -> int:
    """Length of the longest proper prefix of `sentinel` that ends `text`."""
    for k in range(min(len(sentinel) - 1, len(text)), 0, -1):
        if text.endswith(sentinel[:k]):
            return k
    return 0


class ToolCallSplitter:
    """Incrementally split a token stream into content text and tool calls.

    Used only when the request carried tools, so a plain chat request streams
    exactly as before — the measured first-token latency depends on not
    buffering that path.

    Holds back a partial `<tool_call>` prefix so a sentinel straddling two token
    pieces is never leaked to the client as content.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_call = False
        self.bodies: List[str] = []
        self.truncated = False

    def feed(self, piece: str) -> str:
        self._buf += piece
        out: List[str] = []
        while True:
            if self._in_call:
                end = self._buf.find(TOOL_CLOSE)
                if end < 0:
                    return "".join(out)
                self.bodies.append(self._buf[:end])
                self._buf = self._buf[end + len(TOOL_CLOSE) :]
                self._in_call = False
                continue
            start = self._buf.find(TOOL_OPEN)
            if start >= 0:
                out.append(self._buf[:start])
                self._buf = self._buf[start + len(TOOL_OPEN) :]
                self._in_call = True
                continue
            hold = _partial_tail(self._buf, TOOL_OPEN)
            if hold:
                out.append(self._buf[:-hold])
                self._buf = self._buf[-hold:]
            else:
                out.append(self._buf)
                self._buf = ""
            return "".join(out)

    def flush(self) -> str:
        if self._in_call:
            # Unterminated call: the generation hit max_tokens mid-JSON. Drop it
            # rather than forwarding half a call — a malformed `arguments` would
            # be dispatched by the caller as if it were real. The warning is the
            # signal that max_tokens is too low for this tool set.
            LOG.warning(
                "dropping truncated tool call (%d bytes); raise max_tokens",
                len(self._buf),
            )
            self._buf = ""
            self.truncated = True
            return ""
        tail, self._buf = self._buf, ""
        return tail


def to_openai_tool_calls(bodies: List[str]) -> List[dict]:
    calls: List[dict] = []
    for body in bodies:
        try:
            obj = json.loads(body.strip())
        except ValueError:
            LOG.warning("unparseable tool call body: %r", body[:200])
            continue
        name = obj.get("name")
        if not name:
            LOG.warning("tool call without a name: %r", body[:200])
            continue
        args = obj.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args if args is not None else {}, ensure_ascii=False)
        calls.append(
            {
                "index": len(calls),
                "id": "call_" + uuid.uuid4().hex[:20],
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
    return calls


# ── prompt assembly ──────────────────────────────────────────────────────
# The RKNN3 runtime applies the Qwen3 ChatML template itself (verified: an
# 11-token user string prefills as 24 tokens) and `enable_thinking=false` is set
# on the C++ side, so no <|im_start|> tags are injected here.  Multi-turn
# history / system prompts are flattened into the single prompt string.
def _render_turn(m: dict) -> str:
    role = m.get("role")
    content = str(m.get("content") or "").strip()
    if role == "tool":
        # Qwen3 carries tool results in <tool_response> tags. Without this the
        # result never reaches the model and the second round of a tool turn
        # answers from nothing.
        return f"Tool: <tool_response>\n{content}\n</tool_response>"
    if role == "assistant":
        blocks = [
            f"{TOOL_OPEN}\n"
            + json.dumps(
                {
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments"),
                },
                ensure_ascii=False,
            )
            + f"\n{TOOL_CLOSE}"
            for tc in (m.get("tool_calls") or [])
        ]
        return "Assistant: " + "\n".join([content, *blocks]).strip()
    return f"User: {content}"


def build_prompt(messages: List[dict]) -> str:
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    system = [m for m in messages if m.get("role") == "system"]
    turns = [m for m in messages if m.get("role") in ("user", "assistant", "tool")]
    if not turns:
        raise HTTPException(status_code=400, detail="no user/assistant messages")

    parts: List[str] = []
    for m in system:
        parts.append(str(m.get("content") or "").strip())
    # No tool preamble here: the runtime renders it from the model's own chat
    # template when the schema is registered (see the V2 line in _run_request).
    # Single plain user turn keeps the exact prompt shape every latency figure
    # was measured on; anything richer gets role tags.
    if len(turns) == 1 and turns[0].get("role") == "user":
        parts.append(str(turns[0].get("content") or ""))
    else:
        for m in turns[:-1]:
            parts.append(_render_turn(m))
        last = turns[-1]
        if last.get("role") == "user":
            parts.append(str(last.get("content") or ""))
        else:
            parts.append(_render_turn(last))
    return "\n\n".join(p for p in parts if p)


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[dict]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # Without these two declared, Pydantic drops them silently: the model then
    # never sees the tools and answers "done!" without ever emitting a call.
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None


app = FastAPI(title="RK1828 Qwen3-4B OpenAI shim")
WORKER: Optional[Qwen3Worker] = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if (WORKER and WORKER.is_ready()) else "unavailable",
        "model": MODEL_ID,
        "device": "rk1828",
    }


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "rk1828",
            }
        ],
    }


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if WORKER is None or not WORKER.is_ready():
        raise HTTPException(status_code=503, detail="RK1828 worker not ready")

    prompt = build_prompt(req.messages)
    max_new = int(req.max_tokens or 512)
    if req.tools and max_new < TOOL_MIN_MAX_TOKENS:
        # A JSON tool call does not fit in the ~96 tokens that suffice for a
        # spoken reply, and a truncated call is discarded outright, so the turn
        # would silently do nothing.
        LOG.warning(
            "max_tokens=%d is too low for tool calls; raising to %d",
            max_new,
            TOOL_MIN_MAX_TOKENS,
        )
        max_new = TOOL_MIN_MAX_TOKENS
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if not req.stream:
        text = "".join(WORKER.generate(prompt, max_new, tools=req.tools))
        text = _THINK_RE.sub("", text)
        tool_calls: List[dict] = []
        if req.tools:
            splitter = ToolCallSplitter()
            text = splitter.feed(text) + splitter.flush()
            tool_calls = to_openai_tool_calls(splitter.bodies)
        message: dict = {"role": "assistant", "content": text.strip() or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    def event_stream() -> Iterator[str]:
        yield _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )
        n = 0
        splitter = ToolCallSplitter() if req.tools else None

        def _content_chunk(text: str) -> str:
            return _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }
            )

        try:
            for piece in WORKER.generate(prompt, max_new, tools=req.tools):
                # Defensive: the runtime has enable_thinking=false, but drop any
                # literal think tags rather than surfacing them to the client.
                if piece in ("<think>", "</think>"):
                    continue
                n += 1
                if splitter is not None:
                    piece = splitter.feed(piece)
                    if not piece:
                        continue
                yield _content_chunk(piece)
            if splitter is not None:
                tail = splitter.flush()
                if tail:
                    yield _content_chunk(tail)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("generation failed")
            yield _sse({"error": {"message": str(exc), "type": "worker_error"}})
        tool_calls = to_openai_tool_calls(splitter.bodies) if splitter else []
        for tc in tool_calls:
            # Emitted whole rather than as name/argument fragments: the call is
            # only recognisable once </tool_call> has arrived, so there is
            # nothing to gain from splitting it, and the consumer accumulates
            # per-index either way.
            yield _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [tc]},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        yield _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": n,
                    "total_tokens": n,
                },
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--binary",
        default="/home/radxa/rk1828/rknn3-model-zoo/install/rk3588_linux_aarch64/"
        "rknn_Qwen3_demo/rknn_qwen3_demo",
    )
    ap.add_argument(
        "--model-dir",
        default="/home/radxa/rk1828/rknn3-model-zoo/install/rk3588_linux_aarch64/"
        "rknn_Qwen3_demo/model",
    )
    ap.add_argument("--core-mask", default="ff")
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1828)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    global WORKER
    ready_timeout = _env_float("RK1828_READY_TIMEOUT", READY_TIMEOUT_S)
    start_attempts = _env_int("RK1828_START_ATTEMPTS", START_ATTEMPTS)
    WORKER = Qwen3Worker(
        binary=args.binary,
        model_dir=args.model_dir,
        core_mask=args.core_mask,
        max_context=args.max_context,
        start_attempts=start_attempts,
        ready_timeout=ready_timeout,
    )
    LOG.info(
        "worker start: attempts=%d ready_timeout=%gs max_context=%d core_mask=%s",
        start_attempts, ready_timeout, args.max_context, args.core_mask,
    )
    try:
        WORKER.start()
    except WorkerError as exc:
        LOG.error("%s", exc)
        cooldown = _env_float("RK1828_FAIL_COOLDOWN_S", FAIL_COOLDOWN_S)
        LOG.error(
            "sleeping %gs before exit so the restart policy does not immediately "
            "load the model again (set RK1828_FAIL_COOLDOWN_S=1 to opt out)",
            cooldown,
        )
        time.sleep(cooldown)
        raise SystemExit(1)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
