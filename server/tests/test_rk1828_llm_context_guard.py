"""RK1828 LLM service: the KV cache the runtime allocated vs the configured context.

rk3588 devkit, 2026-09-21: the worker ran with --max-context 8192, the export
only had a 2048 kvcache group, and the runtime picked 2048 with a warning. As
the conversation grew past ~2048 tokens prefix reuse was lost (TTFT 181 ms ->
2.4 s), then replies came back empty with HTTP 200.
"""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import struct
import sys

import pytest
from fastapi.testclient import TestClient

SVC = pathlib.Path(__file__).resolve().parents[2] / "services" / "rk1828-llm"

RUNTIME_WARNING = ("W RKNNAPI(24): Using closest attention kvcache group_id = 0 for requested "
                   "max_context_len=8192 (chosen kvcache_buffer_lens: 2048)")


@pytest.fixture
def m(monkeypatch):
    monkeypatch.delenv("RK1828_TOOL_CALL_FORMAT", raising=False)
    spec = importlib.util.spec_from_file_location("rk1828_llm_server_ctx", SVC / "rk1828_llm_server.py")
    mod = importlib.util.module_from_spec(spec)
    # Registered so pydantic can resolve the request model's postponed annotations.
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


class _FakeProc:
    def __init__(self, frames: bytes):
        self.stdout = io.BytesIO(frames)
        self.stdin = io.BytesIO()

    def poll(self):
        return None


def _frames(*parts) -> bytes:
    out = b""
    for p in parts:
        if isinstance(p, int):
            out += struct.pack("<I", p)
        elif isinstance(p, tuple):
            out += struct.pack("<III", *p)
        else:
            b = p.encode()
            out += struct.pack("<I", len(b)) + b
    return out


def _worker(m, frames: bytes = b"", max_context=8192, **kw):
    w = m.Qwen3Worker(binary="/x", model_dir="/m", max_context=max_context, **kw)
    w.proc = _FakeProc(frames)
    w._ready.set()
    return w


# ── guard 1: know the real context ─────────────────────────────────────────

def test_context_comes_from_the_worker_query_line(m):
    w = _worker(m)
    w._note_context_line("CONTEXT effective=2048 requested=8192 source=query")
    assert (w.effective_context, w.context_source) == (2048, "runtime_query")


def test_runtime_warning_is_the_fallback_for_older_workers(m):
    w = _worker(m)
    w._note_context_line(RUNTIME_WARNING)
    assert (w.effective_context, w.context_source) == (2048, "runtime_log")


def test_unqueried_context_is_not_trusted_and_larger_groups_do_not_raise_it(m):
    w = _worker(m)
    w._note_context_line("CONTEXT effective=8192 requested=8192 source=none")
    assert (w.effective_context, w.context_source) == (8192, "unverified")
    w._note_context_line("CONTEXT effective=16384 requested=8192 source=query")
    assert w.effective_context == 8192


def test_smaller_kvcache_refuses_to_start_by_default(m, caplog):
    w = _worker(m)
    w._note_context_line(RUNTIME_WARNING)
    with caplog.at_level("ERROR"), pytest.raises(m.ContextTooSmall):
        w.check_context()
    assert "2048" in caplog.text and "8192" in caplog.text
    assert "RK1828_MAX_CONTEXT=2048" in caplog.text


def test_opt_out_runs_with_the_real_length(m, caplog):
    w = _worker(m, allow_smaller_kvcache=True)
    w._note_context_line(RUNTIME_WARNING)
    with caplog.at_level("ERROR"):
        w.check_context()
    assert "Running with 2048" in caplog.text
    assert w.effective_context == 2048


def test_start_does_not_retry_a_too_small_kvcache(m, monkeypatch):
    """Reloading gives the same KV cache; retrying only costs three 3 GB loads."""
    w = m.Qwen3Worker(binary="/x", model_dir="/m", max_context=8192, start_attempts=3)
    calls = []

    def spawn():
        calls.append(1)
        w.ctx_queried = 2048

    monkeypatch.setattr(w, "_spawn", spawn)
    with pytest.raises(m.ContextTooSmall):
        w.start()
    assert len(calls) == 1


def test_health_reports_both_numbers(m):
    w = _worker(m, allow_smaller_kvcache=True)
    w._note_context_line("CONTEXT effective=2048 requested=8192 source=query")
    m.WORKER = w
    h = TestClient(m.app).get("/health").json()
    assert h["status"] == "ok"
    assert (h["max_context"], h["requested_max_context"], h["context_source"]) == (
        2048, 8192, "runtime_query")


# ── guard 2: no silent empties ─────────────────────────────────────────────

def _chat(m, frames: bytes, **body):
    m.WORKER = _worker(m, frames, max_context=2048)
    body = {"messages": [{"role": "user", "content": "hi"}], **body}
    return TestClient(m.app).post("/v1/chat/completions", json=body)


def test_overflow_is_http_400_context_length_exceeded(m):
    r = _chat(m, _frames(m.CONTEXT_EXCEEDED, (1900, 320, 2048), m.END_OF_STREAM))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "context_length_exceeded"
    assert err["type"] == "invalid_request_error"
    assert "2048" in err["message"] and "1900" in err["message"] and "320" in err["message"]


def test_overflow_in_a_stream_is_an_error_event(m):
    r = _chat(m, _frames(m.CONTEXT_EXCEEDED, (1900, 96, 2048), m.END_OF_STREAM), stream=True)
    assert r.status_code == 200
    events = [json.loads(line[6:]) for line in r.text.splitlines()
              if line.startswith("data: {")]
    errors = [e["error"] for e in events if "error" in e]
    assert len(errors) == 1 and errors[0]["code"] == "context_length_exceeded"
    assert not any(c.get("delta", {}).get("content") for e in events for c in e.get("choices", []))


def test_refusal_leaves_the_stream_in_sync(m):
    w = _worker(m, _frames(m.CONTEXT_EXCEEDED, (1900, 96, 2048), m.END_OF_STREAM,
                           "ok", m.REQUEST_STATS, (30, 1, 2048), m.END_OF_STREAM))
    with pytest.raises(m.ContextLengthExceeded) as ei:
        list(w.generate("long", 96))
    assert (ei.value.prompt_tokens, ei.value.max_new_tokens, ei.value.context) == (1900, 96, 2048)
    stats: dict = {}
    assert "".join(w.generate("short", 96, stats=stats)) == "ok"
    assert stats == {"prompt_tokens": 30, "completion_tokens": 1, "context": 2048}


def test_usage_reports_worker_token_counts(m):
    r = _chat(m, _frames("Hi", "!", m.REQUEST_STATS, (42, 2, 2048), m.END_OF_STREAM))
    assert r.status_code == 200
    assert r.json()["usage"] == {"prompt_tokens": 42, "completion_tokens": 2, "total_tokens": 44}


def test_empty_reply_near_the_limit_is_logged(m, caplog):
    with caplog.at_level("WARNING"):
        r = _chat(m, _frames(m.REQUEST_STATS, (2010, 0, 2048), m.END_OF_STREAM))
    assert r.status_code == 200
    assert "KV cache is probably full" in caplog.text
    assert m.warn_if_empty_near_limit({"prompt_tokens": 100, "context": 2048}, "", []) is False
    assert m.warn_if_empty_near_limit({"prompt_tokens": 2010, "context": 2048}, "ok", []) is False
