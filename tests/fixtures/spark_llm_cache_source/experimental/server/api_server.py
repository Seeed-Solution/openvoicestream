# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
OpenAI-compatible HTTP server for TensorRT Edge-LLM.

Endpoints:
    GET  /health                  - Health check
    GET  /v1/models               - List available models
    POST /v1/chat/completions     - Chat completion (OpenAI-compatible)

Usage (standalone)::

    python -m experimental.server \\
        --model Qwen/Qwen3-1.7B --port 8000

Usage (from LLM object)::

    from experimental.server import LLM
    llm = LLM(model="Qwen/Qwen3-1.7B")
    llm.serve(port=8000)
"""

import argparse
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .engine import SamplingParams, finish_reason_name, _GEN_LOCK
from .tool_calling import (ToolConfig, parse_assistant_output,
                           validate_tool_request)

logger = logging.getLogger("edgellm.api_server")

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
IM_END_TOKEN = "<|im_end|>"


def _split_reasoning_and_content(text: str):
    """Split model output into (reasoning_content, content) around <think> tags."""
    think_open = text.find(THINK_OPEN_TAG)
    think_close = text.find(THINK_CLOSE_TAG)
    if think_open != -1 and think_close != -1 and think_close > think_open:
        reasoning = text[think_open + len(THINK_OPEN_TAG):think_close].strip()
        content = text[think_close + len(THINK_CLOSE_TAG):].strip()
        return reasoning, content or None
    return None, text.strip() if text.strip() else None


def _create_app(llm_instance):
    """Create a FastAPI app backed by the given LLM instance."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI is required for the server. "
                           "Install: pip install fastapi uvicorn") from exc

    app = FastAPI(
        title="TensorRT Edge-LLM Server",
        version="0.1.0",
        description=
        "OpenAI-compatible inference server powered by TensorRT Edge-LLM",
    )

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model": llm_instance.model_dir,
            "speculative_decoding": llm_instance.has_draft_model,
        }

    @app.get("/v1/models")
    def list_models():
        return {
            "object":
            "list",
            "data": [{
                "id": llm_instance._model_id,
                "object": "model",
                "owned_by": "tensorrt-edgellm",
            }],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: Dict[str, Any]):
        messages = body.get("messages", [])
        if not messages:
            return JSONResponse(status_code=400,
                                content={"error": "messages required"})

        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        top_k = body.get("top_k", 50)
        max_tokens = body.get("max_tokens", 2048)
        stream = body.get("stream", False)
        enable_thinking = body.get("enable_thinking", False)
        disable_spec_decode = body.get("disable_spec_decode", False)
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")

        # OpenAI-compatible "stop": null | str | list[str]. Reject other types with 400.
        stop_raw = body.get("stop")
        stop: List[str] = []
        if stop_raw is None:
            pass
        elif isinstance(stop_raw, str):
            stop = [stop_raw]
        elif isinstance(stop_raw, list) and all(
                isinstance(s, str) for s in stop_raw):
            stop = stop_raw
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "'stop' must be a string or array of strings"
                })

        try:
            tool_config = validate_tool_request(messages, tools, tool_choice)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
            )

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            disable_spec_decode=disable_spec_decode,
            stop=stop,
        )

        if stream:
            return StreamingResponse(
                _generate_stream_sse(
                    llm_instance,
                    messages,
                    params,
                    response_id,
                    enable_thinking,
                    tool_config=tool_config,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        try:
            request = llm_instance._make_generation_request(
                messages,
                params,
                tools=tool_config.tools,
                tool_choice=tool_config.tool_choice,
                tool_config=tool_config,
            )
            with _GEN_LOCK:
                response = llm_instance._runtime.handle_request(request)
        except (ValueError, KeyError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid messages: {exc}"},
            )
        except Exception as exc:
            logger.exception("Inference failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})

        raw_text = response.output_texts[0] if response.output_texts else ""
        output_text = raw_text.replace(IM_END_TOKEN, "")
        output_ids = response.output_ids[0] if response.output_ids else []
        completion_tokens = len(output_ids)

        message_body, has_tool_calls = _build_message_body(
            output_text, tool_config, llm_instance.model_dir)

        finish_reason = (finish_reason_name(llm_instance._rt,
                                            response.finish_reasons[0])
                         if response.finish_reasons else "stop")
        if has_tool_calls:
            finish_reason = "tool_calls"

        # ``prompt_tokens`` is reported as 0 because the runtime response does
        # not expose tokenised prompt ids; ``total_tokens`` is then equal to
        # ``completion_tokens``. SDKs that validate the schema (existence of
        # the three fields) succeed; consumers that compute cost from
        # ``prompt_tokens`` will see 0 until the runtime is extended.
        return {
            "id":
            response_id,
            "object":
            "chat.completion",
            "created":
            int(time.time()),
            "model":
            os.path.basename(llm_instance.model_dir) or llm_instance.model_dir,
            "choices": [{
                "index": 0,
                "message": message_body,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            },
        }

    return app


class _ThinkingStateMachine:
    """Tracks <think>...</think> boundaries across streaming deltas."""

    def __init__(self, thinking_enabled: bool):
        self._enabled = thinking_enabled
        self._in_think = False
        self._think_opened = False
        self._buf = ""

    def feed(self, text: str):
        """Yield (field, text) pairs: field is 'reasoning' or 'content'."""
        if not self._enabled:
            yield "content", text
            return

        self._buf += text
        while self._buf:
            if not self._in_think:
                idx = self._buf.find(THINK_OPEN_TAG)
                if idx == -1:
                    if len(self._buf) > len(THINK_OPEN_TAG):
                        safe = self._buf[:-len(THINK_OPEN_TAG)]
                        self._buf = self._buf[len(safe):]
                        if safe and self._think_opened:
                            yield "content", safe
                        elif safe:
                            yield "content", safe
                    break
                if idx > 0 and self._think_opened:
                    yield "content", self._buf[:idx]
                elif idx > 0:
                    yield "content", self._buf[:idx]
                self._buf = self._buf[idx + len(THINK_OPEN_TAG):]
                self._in_think = True
                self._think_opened = True
            else:
                idx = self._buf.find(THINK_CLOSE_TAG)
                if idx == -1:
                    if len(self._buf) > len(THINK_CLOSE_TAG):
                        safe = self._buf[:-len(THINK_CLOSE_TAG)]
                        self._buf = self._buf[len(safe):]
                        if safe:
                            yield "reasoning", safe
                    break
                if idx > 0:
                    yield "reasoning", self._buf[:idx]
                self._buf = self._buf[idx + len(THINK_CLOSE_TAG):]
                self._in_think = False

    def flush(self):
        """Flush remaining buffer at end of stream."""
        if self._buf:
            field = "reasoning" if self._in_think else "content"
            yield field, self._buf
            self._buf = ""


def _generate_stream_sse(llm_instance,
                         messages,
                         params,
                         response_id,
                         enable_thinking,
                         tool_config: Optional[ToolConfig] = None):
    """Yield real SSE chunks via StreamChannel streaming."""
    yield _sse_chunk(response_id, {"role": "assistant"})

    if tool_config is not None and tool_config.parse_output:
        yield from _generate_tool_stream_sse(llm_instance, messages, params,
                                             response_id, tool_config)
        return

    sm = _ThinkingStateMachine(enable_thinking)
    finish_reason: Optional[str] = None
    stream_tools = tool_config.tools if tool_config else None
    stream_tool_choice = tool_config.tool_choice if tool_config else None

    try:
        for delta in llm_instance.generate_stream(
                messages,
                params,
                tools=stream_tools,
                tool_choice=stream_tool_choice):
            if delta.text:
                for field, text in sm.feed(delta.text):
                    yield _sse_chunk(response_id, {field: text})
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"
    except Exception:
        logger.exception("Streaming inference failed")
        finish_reason = "error"

    for field, text in sm.flush():
        yield _sse_chunk(response_id, {field: text})

    yield _sse_chunk(response_id, {}, finish_reason=finish_reason or "stop")
    yield "data: [DONE]\n\n"


def _generate_tool_stream_sse(llm_instance, messages, params, response_id,
                              tool_config: ToolConfig):
    text_parts: List[str] = []
    finish_reason: Optional[str] = None
    try:
        for delta in llm_instance.generate_stream(
                messages,
                params,
                tools=tool_config.tools,
                tool_choice=tool_config.tool_choice):
            if delta.text:
                text_parts.append(delta.text)
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"
    except Exception:
        logger.exception("Streaming inference failed")
        finish_reason = "error"

    output_text = "".join(text_parts).replace(IM_END_TOKEN, "")
    parsed = parse_assistant_output(output_text, tool_config,
                                    llm_instance.model_dir)
    tool_index = 0
    for event in parsed.events:
        if event["type"] == "reasoning" and event["text"]:
            yield _sse_chunk(response_id, {"reasoning": event["text"]})
        elif event["type"] == "content" and event["text"]:
            yield _sse_chunk(response_id, {"content": event["text"]})
        elif event["type"] == "tool_call":
            call = event["tool_call"]
            yield _sse_chunk(
                response_id, {
                    "tool_calls": [{
                        "index": tool_index,
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": "",
                        },
                    }]
                })
            if call.arguments:
                yield _sse_chunk(
                    response_id, {
                        "tool_calls": [{
                            "index": tool_index,
                            "function": {
                                "arguments": call.arguments,
                            },
                        }]
                    })
            tool_index += 1

    finish = "tool_calls" if tool_index else finish_reason or "stop"
    yield _sse_chunk(response_id, {}, finish_reason=finish)
    yield "data: [DONE]\n\n"


def _build_message_body(output_text: str, tool_config: ToolConfig,
                        model_dir: str) -> Tuple[Dict[str, Any], bool]:
    if not tool_config.parse_output:
        reasoning, answer = _split_reasoning_and_content(output_text)
        message_body: Dict[str, Any] = {"role": "assistant"}
        if reasoning is not None:
            message_body["reasoning"] = reasoning
        message_body["content"] = (
            (answer if answer is not None else reasoning) or "")
        return message_body, False

    parsed = parse_assistant_output(output_text, tool_config, model_dir)
    tool_calls = [call.to_openai() for call in parsed.tool_calls]
    message_body: Dict[str, Any] = {"role": "assistant"}
    if parsed.reasoning:
        message_body["reasoning"] = parsed.reasoning
    content = parsed.content.strip()
    message_body["content"] = content if content or not tool_calls else None
    if tool_calls:
        message_body["tool_calls"] = tool_calls
    return message_body, bool(tool_calls)


def _sse_chunk(response_id: str,
               delta: dict,
               finish_reason: Optional[str] = None):
    choice: Dict[str, Any] = {"delta": delta, "index": 0}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    payload = {"id": response_id, "choices": [choice]}
    return f"data: {json.dumps(payload)}\n\n"


def run_server(llm_instance, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the OpenAI-compatible server."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required. Install: pip install uvicorn") from exc

    app = _create_app(llm_instance)
    logger.info("Starting server on %s:%d ...", host, port)
    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="TensorRT Edge-LLM OpenAI-compatible server")
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local checkpoint path",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=4096,
        help="Max input sequence length",
    )
    parser.add_argument("--max-batch-size",
                        type=int,
                        default=1,
                        help="Max batch size")
    parser.add_argument(
        "--max-kv-cache-capacity",
        type=int,
        default=8192,
        help="Max KV cache capacity",
    )
    parser.add_argument(
        "--spec-decode-engine-dir",
        dest="spec_decode_engine_dir",
        default="",
        help=
        "Pre-built speculative decoding engine dir (EAGLE, MTP, or DFlash)",
    )
    parser.add_argument("--draft-top-k",
                        type=int,
                        default=10,
                        help="Speculative decoding: tokens per predecessor")
    parser.add_argument("--draft-step",
                        type=int,
                        default=6,
                        help="Speculative decoding: number of draft steps")
    parser.add_argument("--verify-tree-size",
                        type=int,
                        default=60,
                        help="Speculative decoding: verification tree size")
    args = parser.parse_args()

    from .engine import LLM

    llm = LLM(
        model=args.model,
        max_input_len=args.max_input_len,
        max_batch_size=args.max_batch_size,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        eagle_engine_dir=args.spec_decode_engine_dir,
        draft_top_k=args.draft_top_k,
        draft_step=args.draft_step,
        verify_tree_size=args.verify_tree_size,
    )
    llm.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
