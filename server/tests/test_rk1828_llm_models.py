"""RK1828 LLM service: model types (Qwen3-4B, Qwen3.5-4B) and tool-call formats.

Qwen3 writes a tool call as JSON between <tool_call> tags; Qwen3.5 writes
<function=...><parameter=...>. On the rk3588 devkit (2026-09-21) the shim only
parsed JSON, so all three Qwen3.5 tool calls in the benchmark were dropped.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

SVC = pathlib.Path(__file__).resolve().parents[2] / "services" / "rk1828-llm"


def _shim(monkeypatch, fmt: str | None = None):
    if fmt is None:
        monkeypatch.delenv("RK1828_TOOL_CALL_FORMAT", raising=False)
    else:
        monkeypatch.setenv("RK1828_TOOL_CALL_FORMAT", fmt)
    spec = importlib.util.spec_from_file_location("rk1828_llm_server_t", SVC / "rk1828_llm_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


QWEN35_BODY = """
<function=set_mode>
<parameter=mode_name>
transcribe
</parameter>
</function>
"""


def test_parses_qwen3_json_and_qwen35_xml(monkeypatch):
    m = _shim(monkeypatch)
    assert m.parse_tool_call_body('{"name": "set_mode", "arguments": {"mode_name": "chat"}}') == (
        "set_mode", '{"mode_name": "chat"}')
    assert m.parse_tool_call_body(QWEN35_BODY) == ("set_mode", '{"mode_name": "transcribe"}')
    assert m.parse_tool_call_body("<function=get_time>\n</function>") == ("get_time", "{}")


def test_xml_values_keep_their_types(monkeypatch):
    m = _shim(monkeypatch)
    body = ("<function=move>\n<parameter=x>\n0.25\n</parameter>\n<parameter=fast>\ntrue\n</parameter>\n"
            "<parameter=label>\nred box\n</parameter>\n<parameter=pose>\n{\"yaw\": 90}\n</parameter>\n</function>")
    name, args = m.parse_tool_call_body(body)
    assert name == "move"
    assert json.loads(args) == {"x": 0.25, "fast": True, "label": "red box", "pose": {"yaw": 90}}


def test_garbage_bodies_are_rejected(monkeypatch):
    m = _shim(monkeypatch)
    for body in ["", "not json", '{"arguments": {}}', "<function=>\n</function>", "<function=x"]:
        assert m.parse_tool_call_body(body) is None, body


def test_splitter_extracts_a_qwen35_call_split_across_pieces(monkeypatch):
    m = _shim(monkeypatch)
    sp = m.ToolCallSplitter()
    text = "".join(sp.feed(p) for p in ["Sure.<tool", "_call>\n<function=set_", "mode>\n<parameter=mode_name>\nchat",
                                        "\n</parameter>\n</function>\n</tool_", "call>"]) + sp.flush()
    assert text == "Sure."
    calls = m.to_openai_tool_calls(sp.bodies)
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == [
        ("set_mode", {"mode_name": "chat"})]


@pytest.mark.parametrize("fmt", ["json", "xml"])
def test_history_rendering_round_trips(monkeypatch, fmt):
    m = _shim(monkeypatch, fmt)
    block = m.render_tool_call("set_mode", '{"mode_name": "interpreter", "n": 2}')
    assert block.startswith(m.TOOL_OPEN) and block.endswith(m.TOOL_CLOSE)
    if fmt == "xml":
        assert "<function=set_mode>" in block
    body = block[len(m.TOOL_OPEN):-len(m.TOOL_CLOSE)]
    name, args = m.parse_tool_call_body(body)
    assert name == "set_mode" and json.loads(args) == {"mode_name": "interpreter", "n": 2}


def test_model_id_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("RK1828_MODEL_ID", "Qwen3.5-4B")
    assert _shim(monkeypatch).MODEL_ID == "Qwen3.5-4B"
    monkeypatch.delenv("RK1828_MODEL_ID")
    assert _shim(monkeypatch).MODEL_ID == "Qwen3-4B"


MANIFESTS = sorted((SVC / "artifacts").glob("*.json"))


def test_there_is_a_manifest_per_model():
    assert {p.stem for p in MANIFESTS} >= {"Qwen3-4B", "Qwen3.5-4B"}


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.stem)
def test_manifest_describes_one_four_file_export(path):
    d = json.loads(path.read_text())
    mid = d["model_id"]
    assert mid == path.stem, "entrypoint looks the manifest up by RK1828_MODEL_ID"
    names = {f["name"] for f in d["files"]}
    assert names == {f"{mid}.rknn", f"{mid}.weight", f"{mid}.tokenizer.gguf", f"{mid}.embed.bin"}
    for f in d["files"]:
        assert f["size_bytes"] > 0 and len(f["sha256"]) == 64
    rt = d["runtime"]
    assert rt["tool_call_format"] in ("json", "xml")
    assert rt["max_context"] > 0
    assert (rt["kv_checkpoint_interval"] == 0) == (rt["kv_checkpoint_count"] == 0)
    if rt["kv_checkpoint_interval"]:
        assert rt["kv_checkpoint_interval"] % 128 == 0
        assert rt["kv_checkpoint_count"] <= rt["max_context"] // rt["kv_checkpoint_interval"]
    pub = d.get("published_at")
    assert pub is None or pub["prefix"].startswith("rk1828/")



class _FakeProc:
    """A worker whose stdout is a fixed byte string of frames."""

    def __init__(self, frames: bytes):
        import io
        self.stdout = io.BytesIO(frames)
        self.stdin = io.BytesIO()

    def poll(self):
        return None


def _frames(*parts) -> bytes:
    import struct
    out = b""
    for p in parts:
        if isinstance(p, int):
            out += struct.pack("<I", p)
        else:
            b = p.encode()
            out += struct.pack("<I", len(b)) + b
    return out


def _worker(m, frames: bytes):
    w = m.Qwen3Worker(binary="/x", model_dir="/m", max_context=4096)
    w.proc = _FakeProc(frames)
    w._ready.set()
    return w


def test_failed_run_is_an_error_not_an_empty_answer(monkeypatch):
    """Regression (rk3588, 2026-09-21): a prompt longer than the KV cache failed
    in the worker and the client got HTTP 200 with empty content."""
    m = _shim(monkeypatch)
    w = _worker(m, _frames(m.REQUEST_FAILED, m.END_OF_STREAM, "next", m.END_OF_STREAM))
    with pytest.raises(m.WorkerError, match="context"):
        list(w.generate("long prompt", 16))
    # The EOS after the failure was consumed: the next request is not cut off.
    assert "".join(w.generate("short prompt", 16)) == "next"


def test_normal_run_still_streams(monkeypatch):
    m = _shim(monkeypatch)
    w = _worker(m, _frames("Hel", "lo", m.END_OF_STREAM))
    assert "".join(w.generate("hi", 16)) == "Hello"


def test_tool_less_request_never_returns_or_speaks_a_tool_call(monkeypatch, caplog):
    """The runtime keeps tools registered for the session, so a tool-less request
    can still produce a <tool_call>. It must not reach the client as a call or
    as text (it was spoken on the devkit, 2026-09-21)."""
    m = _shim(monkeypatch)
    sp = m.ToolCallSplitter()
    text = "".join(sp.feed(p) for p in ["<tool_call>\n<function=set_mode>\n<parameter=mode_name>\n",
                                        "transcribe\n</parameter>\n</function>\n</tool_call>", "OK."]) + sp.flush()
    assert text == "OK."
    with caplog.at_level("WARNING"):
        assert m.tool_calls_for(None, sp) == []
    assert "carried no tools" in caplog.text
    # With tools the same body is a real call.
    assert m.tool_calls_for([{"type": "function"}], sp)[0]["function"]["name"] == "set_mode"


def test_plain_text_is_not_held_back(monkeypatch):
    m = _shim(monkeypatch)
    sp = m.ToolCallSplitter()
    assert sp.feed("Hello") == "Hello"
    assert sp.feed(" <b>") == " <b>"
    assert sp.feed(" a <to") == " a "      # could be the start of <tool_call>
    assert sp.feed("ast") == "<toast"
