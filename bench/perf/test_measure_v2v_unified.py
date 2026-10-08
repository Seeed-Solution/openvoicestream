import json
import asyncio
import threading
from collections import deque
import sys
from pathlib import Path

import pytest
import websocket

sys.path.insert(0, str(Path(__file__).parent))
from measure_v2v_unified import run_once


class FakeWS:
    def __init__(self, *, include_tts_done=True, completion_probe=None, audio_on_text=False):
        self.sent = []
        self.events = deque()
        self.include_tts_done = include_tts_done
        self.completion_probe = completion_probe
        self.audio_on_text = audio_on_text
        self._early_audio_sent = False
        self.audio_before_llm_done = False
        self.closed = False
        self.multi_utterance = False
        self.asr_finals = []
        self.binary_bytes = 0

    def settimeout(self, _timeout):
        pass

    def send(self, raw):
        data = json.loads(raw)
        self.sent.append(data)
        if data["type"] == "config":
            self.multi_utterance = bool(data.get("multi_utterance"))
        if data["type"] == "text" and self.audio_on_text and not self._early_audio_sent:
            self.events.append((16000).to_bytes(4, "little") + b"early-pcm")
            self._early_audio_sent = True
        if data["type"] == "asr_eos":
            complete = not self.multi_utterance
            self.asr_finals.append(complete)
            self.events.append(json.dumps({
                "type": "asr_final",
                "text": "what time is it",
                "session_complete": complete,
            }))
        elif data["type"] == "tts_flush":
            self.events.append((16000).to_bytes(4, "little") + b"pcm-a")
            self.events.append(b"pcm-b")
            if self.include_tts_done:
                self.events.append(json.dumps({
                    "type": "tts_done",
                    "session_complete": self.multi_utterance,
                }))

    def send_binary(self, _data):
        self.binary_bytes += len(_data)
        # Model the VAD endpoint caused by the benchmark's silence tail.
        if self.multi_utterance and _data and not any(_data) and not self.asr_finals:
            self.asr_finals.append(False)
            self.events.append(json.dumps({
                "type": "asr_final",
                "text": "what time is it",
                "session_complete": False,
            }))

    def recv(self):
        if self.events:
            event = self.events.popleft()
            if isinstance(event, bytes) and self.completion_probe is not None:
                self.audio_before_llm_done = self.audio_before_llm_done or not self.completion_probe()
                if hasattr(SlowFakeLLM, "release"):
                    SlowFakeLLM.release.set()
            return event
        raise websocket.WebSocketTimeoutException()

    def close(self):
        self.closed = True


class FakeLLM:
    async def stream(self, messages):
        assert messages[-1] == {"role": "user", "content": "what time is it"}
        yield "It is "
        yield "noon."


class SlowFakeLLM:
    done = False
    release = threading.Event()

    async def stream(self, _messages):
        type(self).done = False
        type(self).release.clear()
        yield "It is "
        await asyncio.to_thread(type(self).release.wait)
        yield "noon."
        type(self).done = True


class FailingLLM:
    async def stream(self, _messages):
        raise RuntimeError("provider URL and key must not escape")
        yield  # pragma: no cover


class NeverEndingLLM:
    cancelled = False
    closed = False
    started = threading.Event()

    async def stream(self, _messages):
        type(self).started.set()
        try:
            await asyncio.sleep(3600)
            yield "never"
        finally:
            type(self).cancelled = True

    async def aclose(self):
        type(self).closed = True


class SlowTokenLLM:
    cancelled = False
    closed = False
    started = threading.Event()

    async def stream(self, _messages):
        type(self).started.set()
        try:
            yield "first"
            await asyncio.sleep(3600)
        finally:
            type(self).cancelled = True

    async def aclose(self):
        type(self).closed = True


class DisconnectWS(FakeWS):
    def send_binary(self, data):
        super().send_binary(data)
        if self.multi_utterance and data and not any(data):
            # A server configured in single mode would close after this VAD
            # final; use this fixture to prove the client reports that loss.
            self.events.clear()
            self.events.append(json.dumps({
                "type": "asr_final",
                "text": "what time is it",
                "session_complete": True,
            }))
            self.disconnect_after_final = True

    def send(self, raw):
        super().send(raw)
        if json.loads(raw)["type"] == "asr_eos":
            # Model the real single-utterance server: its final has no
            # persistent-session continuation and the websocket closes before
            # client-owned text/tts_flush can be delivered.
            self.events.clear()
            self.events.append(json.dumps({
                "type": "asr_final",
                "text": "what time is it",
                "session_complete": True,
            }))
            self.disconnect_after_final = True

    def recv(self):
        if self.events:
            return self.events.popleft()
        if getattr(self, "disconnect_after_final", False):
            return ""
        raise websocket.WebSocketTimeoutException()


class SendFailWS(FakeWS):
    def send(self, raw):
        if json.loads(raw).get("type") == "text":
            raise RuntimeError("send closed")
        return super().send(raw)


def test_client_llm_calls_backend_once_and_collects_complete_pcm():
    wav = b"\x01\x00" * 160
    ws = FakeWS()
    result = run_once(
        "fake:8621",
        wav,
        0.01,
        tts_enabled=False,
        realtime=False,
        timeout=0.1,
        client_llm=True,
        ws_factory=lambda *_args, **_kwargs: ws,
        llm_backend_factory=FakeLLM,
    )

    assert result.error is None
    assert result.assistant_text == "It is noon."
    assert result.tts_complete is True
    assert result.tts_sample_rate == 16000
    assert result.tts_pcm_bytes == len(b"pcm-a") + len(b"pcm-b")
    assert result.tts_last_pcm_bytes == len(b"pcm-b")
    assert [x for x in ws.sent if x["type"] == "text"] == [
        {"type": "text", "text": "It is "},
        {"type": "text", "text": "noon."},
    ]
    assert sum(x["type"] == "tts_flush" for x in ws.sent) == 1
    assert not any(x.get("text") == "what time is it" for x in ws.sent)
    assert ws.multi_utterance is True
    assert ws.asr_finals == [False]
    assert ws.binary_bytes == len(wav)
    assert sum(x["type"] == "asr_eos" for x in ws.sent) == 1


def test_client_llm_missing_tts_done_fails():
    ws = FakeWS(include_tts_done=False)
    result = run_once(
        "fake:8621",
        b"\x01\x00" * 160,
        0.01,
        realtime=False,
        timeout=0.1,
        client_llm=True,
        ws_factory=lambda *_args, **_kwargs: ws,
        llm_backend_factory=FakeLLM,
    )
    assert result.error == "timeout"
    assert result.tts_pcm_bytes > 0
    assert result.tts_complete is False


def test_pcm_is_consumed_before_llm_finishes():
    ws = FakeWS(completion_probe=lambda: SlowFakeLLM.done, audio_on_text=True)
    result = run_once(
        "fake:8621", b"\x01\x00" * 160, 0.01, realtime=False, timeout=1.0,
        client_llm=True, ws_factory=lambda *_args, **_kwargs: ws,
        llm_backend_factory=SlowFakeLLM,
    )
    assert result.error is None
    assert result.tts_pcm_bytes > 0
    assert SlowFakeLLM.done is True
    assert ws.audio_before_llm_done is True
    assert result.tts_last_pcm_ms is not None
    assert result.client_llm_ttft_ms is not None
    assert result.client_llm_ttft_ms < 100


def test_client_llm_exception_is_sanitized():
    ws = FakeWS()
    result = run_once(
        "fake:8621", b"\x01\x00" * 160, 0.01, realtime=False, timeout=0.1,
        client_llm=True, ws_factory=lambda *_args, **_kwargs: ws,
        llm_backend_factory=FailingLLM,
    )
    assert result.error == "client LLM failed: RuntimeError"
    assert "provider URL" not in result.error


def test_vad_none_is_forwarded_in_v2v_config():
    ws = FakeWS()
    result = run_once(
        "fake:8621", b"\x01\x00" * 160, 0.01,
        vad="none", realtime=False, timeout=0.1,
        ws_factory=lambda *_args, **_kwargs: ws,
    )

    assert result.error is None
    assert ws.sent[0] == {
        "type": "config",
        "asr_language": "Chinese",
        "sample_rate": 16000,
        "vad": "none",
        "vad_silence_ms": 500,
        "multi_utterance": False,
    }


def test_client_llm_early_disconnect_is_failure():
    result = run_once(
        "fake:8621", b"\x01\x00" * 160, 0.01, realtime=False, timeout=0.1,
        client_llm=True, ws_factory=lambda *_args, **_kwargs: DisconnectWS(),
        llm_backend_factory=FakeLLM,
    )
    assert result.error == "server closed"


def test_client_llm_timeout_cancels_backend():
    NeverEndingLLM.cancelled = False
    NeverEndingLLM.closed = False
    NeverEndingLLM.started.clear()
    ws = FakeWS()
    result = run_once(
        "fake:8621", b"\x01\x00" * 160, 0.01, realtime=False, timeout=0.03,
        client_llm=True, ws_factory=lambda *_args, **_kwargs: ws,
        llm_backend_factory=NeverEndingLLM,
    )
    assert result.error == "timeout"
    assert NeverEndingLLM.cancelled is True
    assert NeverEndingLLM.closed is True


def test_client_llm_disconnect_cancels_never_ending_backend():
    NeverEndingLLM.cancelled = False
    NeverEndingLLM.closed = False
    NeverEndingLLM.started.clear()

    class DisconnectAfterLLMStart(DisconnectWS):
        def recv(self):
            if self.events:
                return self.events.popleft()
            NeverEndingLLM.started.wait(0.2)
            return ""

    result = run_once(
        "fake:8621", b"\x01\x00" * 160, 0.01, realtime=False, timeout=1.0,
        client_llm=True,
        ws_factory=lambda *_args, **_kwargs: DisconnectAfterLLMStart(),
        llm_backend_factory=NeverEndingLLM,
    )
    assert result.error == "server closed"
    assert NeverEndingLLM.cancelled is True
    assert NeverEndingLLM.closed is True


def test_client_llm_send_failure_cancels_running_backend():
    SlowTokenLLM.cancelled = False
    SlowTokenLLM.closed = False
    SlowTokenLLM.started.clear()
    result = run_once(
        "fake:8621", b"\x01\x00" * 160, 0.01, realtime=False, timeout=1.0,
        client_llm=True, ws_factory=lambda *_args, **_kwargs: SendFailWS(),
        llm_backend_factory=SlowTokenLLM,
    )
    assert result.error == "transport failed: RuntimeError"
    assert SlowTokenLLM.started.is_set()
    assert SlowTokenLLM.cancelled is True
    assert SlowTokenLLM.closed is True


def test_client_llm_rejects_multi_and_server_loop():
    with pytest.raises(ValueError, match="single-turn"):
        run_once("fake:8621", b"", 0, client_llm=True, multi_count=2)
    with pytest.raises(ValueError, match="single-turn"):
        run_once("fake:8621", b"", 0, client_llm=True, server_loop=True)
