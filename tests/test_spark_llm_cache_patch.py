"""Behavioral checks for the Spark system-prompt KV-cache overlay."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

ROOT = Path(__file__).parents[1]
PATCH = ROOT / "third_party/jetson-voice-engine/engine-overlay-v010/patches/spark-llm-system-cache-control.patch"
SNAPSHOT = ROOT / "tests/fixtures/spark_llm_cache_source/experimental/server"
OLD_PATCH = ROOT / "third_party/jetson-voice-engine/engine-overlay-v010/patches/spark-llm-serialize-syscache-pybind.patch"


def _candidate_tree() -> Path:
    temp = Path(tempfile.mkdtemp(prefix="spark-llm-candidate-"))
    server = temp / "experimental/server"
    server.mkdir(parents=True)
    shutil.copytree(SNAPSHOT, server, dirs_exist_ok=True)
    (temp / "experimental/__init__.py").write_text("")
    (server / "__init__.py").write_text("")
    (server / "tool_calling.py").write_text(
        "class ToolConfig:\n"
        "    def __init__(self):\n"
        "        self.tools = None\n"
        "        self.tool_choice = None\n"
        "        self.parse_output = False\n"
        "def parse_assistant_output(*args, **kwargs): return None\n"
        "def validate_tool_request(*args, **kwargs): return ToolConfig()\n"
    )
    (server / "tool_chat_template.py").write_text(
        "class ToolChatTemplateFormatter: pass\n"
        "def needs_tool_chat_template(*args, **kwargs): return False\n"
    )
    (server / "audio_preprocess.py").write_text(
        "def load_audio_buffers(runtime, messages): return []\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=temp, check=True)
    subprocess.run(["git", "add", "."], cwd=temp, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.invalid", "-c", "user.name=test", "commit", "-qm", "snapshot"],
        cwd=temp,
        check=True,
    )
    subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=temp, check=True)
    subprocess.run(["git", "apply", str(PATCH)], cwd=temp, check=True)
    return temp


@contextmanager
def _candidate_modules():
    temp = _candidate_tree()
    sys.path.insert(0, str(temp))
    try:
        yield importlib.import_module("experimental.server.engine"), importlib.import_module("experimental.server.api_server")
    finally:
        sys.path.remove(str(temp))
        for name in list(sys.modules):
            if name == "experimental" or name.startswith("experimental."):
                sys.modules.pop(name, None)
        shutil.rmtree(temp, ignore_errors=True)


def test_historical_serialize_then_cache_control_sequence_applies() -> None:
    temp = Path(tempfile.mkdtemp(prefix="spark-llm-sequence-"))
    try:
        server = temp / "experimental/server"
        server.mkdir(parents=True)
        shutil.copytree(SNAPSHOT, server, dirs_exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=temp, check=True)
        subprocess.run(["git", "add", "."], cwd=temp, check=True)
        subprocess.run(["git", "-c", "user.email=test@example.invalid", "-c", "user.name=test", "commit", "-qm", "post-old"], cwd=temp, check=True)
        subprocess.run(["git", "apply", "-R", str(OLD_PATCH)], cwd=temp, check=True)
        subprocess.run(["git", "apply", "--check", str(OLD_PATCH)], cwd=temp, check=True)
        subprocess.run(["git", "apply", str(OLD_PATCH)], cwd=temp, check=True)
        subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=temp, check=True)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def test_patch_applies_to_persisted_spark_snapshot_without_recount() -> None:
    temp = _candidate_tree()
    try:
        engine = (temp / "experimental/server/engine.py").read_text()
        api = (temp / "experimental/server/api_server.py").read_text()
        assert "save_system_prompt_kv_cache: bool = False" in engine
        assert "request.save_system_prompt_kv_cache = params.save_system_prompt_kv_cache" in engine
        assert "cache_prompt must be a boolean" in api
        assert "system_prompt_cache_unsupported" in api
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def test_real_request_builder_preserves_messages_and_flag() -> None:
    with _candidate_modules() as (engine, _api):
        class Request:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class GenerationRequest:
            pass

        Runtime = type("Runtime", (), {"Request": Request, "LLMGenerationRequest": GenerationRequest})

        llm = object.__new__(engine.LLM)
        llm._rt = Runtime()
        llm._prepare_messages_for_runtime = lambda *args, **kwargs: (["system", "user"], [], True, True)
        messages = [{"role": "system", "content": "keep me"}, {"role": "user", "content": "hello"}]
        observed = []
        for enabled in (False, True):
            request = llm._make_generation_request(
                messages, engine.SamplingParams(save_system_prompt_kv_cache=enabled)
            )
            assert request.save_system_prompt_kv_cache is enabled
            assert request.requests[0].messages == ["system", "user"]
            observed.append(request.save_system_prompt_kv_cache)
        print("builder_flags", observed, "messages", ["system", "user"])


def _fake_llm(runtime_error=None):
    class Runtime:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def handle_request(self, request):
            if runtime_error is not None:
                raise runtime_error
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.03)
                return SimpleNamespace(output_texts=["ok"], output_ids=[[1]], finish_reasons=[])
            finally:
                with self.lock:
                    self.active -= 1

    runtime = Runtime()
    calls = []

    class LLM:
        model_dir = "stub-model"
        _model_id = "stub"
        has_draft_model = False

        def __init__(self):
            self._runtime = runtime

        def _make_generation_request(self, messages, params, **kwargs):
            calls.append((messages, params))
            return SimpleNamespace(messages=messages, params=params)

    return LLM(), runtime, calls


def test_real_api_alias_validation_error_mapping_and_system_content() -> None:
    with _candidate_modules() as (_engine, api):
        llm, _runtime, calls = _fake_llm()
        client = TestClient(api._create_app(llm))
        base = {"messages": [{"role": "system", "content": "system text"}, {"role": "user", "content": "hi"}]}
        for body, expected in ((base, False), ({**base, "cache_prompt": True}, True), ({**base, "save_system_prompt_kv_cache": True}, True), ({**base, "save_system_prompt_kv_cache": False}, False), ({**base, "save_system_prompt_kv_cache": False, "cache_prompt": True}, False)):
            response = client.post("/v1/chat/completions", json=body)
            assert response.status_code == 200, response.text
            assert calls[-1][0][0]["content"] == "system text"
            assert calls[-1][1].save_system_prompt_kv_cache is expected

        invalid = client.post("/v1/chat/completions", json={**base, "cache_prompt": "yes"})
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "invalid_cache_prompt"

        unsupported_llm, _, _ = _fake_llm(RuntimeError("HybridCacheManager::captureKVCache currently only supports kHALF KV cache"))
        unsupported_client = TestClient(api._create_app(unsupported_llm))
        assert unsupported_client.post("/v1/chat/completions", json={**base, "cache_prompt": True}).status_code == 422
        failed_llm, _, _ = _fake_llm(RuntimeError("different native failure"))
        failed_client = TestClient(api._create_app(failed_llm))
        unexpected_status = failed_client.post("/v1/chat/completions", json=base).status_code
        print("api_cases", {"absent": 200, "false": 200, "alias_true": 200, "canonical_true": 200, "invalid": invalid.status_code, "known_unsupported": 422, "unexpected": unexpected_status})
        assert unexpected_status == 500


def test_real_api_generation_lock_serializes_concurrent_requests() -> None:
    with _candidate_modules() as (_engine, api):
        llm, runtime, _calls = _fake_llm()
        client = TestClient(api._create_app(llm))
        body = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: client.post("/v1/chat/completions", json=body), range(2)))
        assert [response.status_code for response in responses] == [200, 200]
        print("concurrency", {"statuses": [response.status_code for response in responses], "max_active": runtime.max_active})
        assert runtime.max_active == 1
