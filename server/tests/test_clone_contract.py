"""Contract tests for the modality-specific native v1 clone endpoints."""

from __future__ import annotations

import base64
import asyncio
import io
import struct
import threading
import time
from contextlib import asynccontextmanager

import pytest


def _wav(payload: bytes = b"\x00\x00\x01\x00", *, sample_rate: int = 48_000, channels: int = 1) -> bytes:
    block_align = channels * 2
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        16,
    )
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


class _Backend:
    name = "fake-qwen"
    model_id = "qwen3-tts-0.6b-base"
    sample_rate = 24_000
    channels = 1
    supports_voice_cloning = True
    capabilities = {"voice_clone", "streaming"}

    def __init__(self):
        self.clone_calls = []
        self.stream_calls = []

    def is_ready(self):
        return True

    def has_capability(self, cap):
        return getattr(cap, "value", cap) in self.capabilities

    def rate_pitch_caps(self):
        return True, True

    def clone_voice(self, text, language=None, **kwargs):
        self.clone_calls.append((text, language, kwargs))
        return b"RIFFfake", {"duration": 1}

    def generate_streaming(self, text, language=None, cancel_event=None, **kwargs):
        self.stream_calls.append((text, language, cancel_event, kwargs))
        yield b"\x01\x00\x02\x00"
        while cancel_event is not None and cancel_event.is_set():
            return
        yield b"\x03\x00\x04\x00"


class _Moss(_Backend):
    name = "fake-moss"
    model_id = "moss-tts-nano-v1"
    sample_rate = 48_000
    channels = 1
    # Reference encoder contract is intentionally distinct from synthesized
    # output shape; production reads the same values from codec metadata.
    reference_sample_rate = 48_000
    reference_channels = 1

    def clone_voice(self, text, language=None, **kwargs):
        self.clone_calls.append((text, language, kwargs))
        assert kwargs["reference_audio"] == b"\x00\x00\x01\x00"
        assert kwargs["reference_sample_rate"] == 48_000
        return b"RIFFmoss", {"duration": 1}


class _BaseReference(_Backend):
    def supports_reference_audio_cloning(self):
        return True

    def clone_voice(self, text, language=None, **kwargs):
        self.clone_calls.append((text, language, kwargs))
        assert kwargs["reference_text"] == "reference transcript"
        assert kwargs["reference_audio_wav"].startswith(b"RIFF")
        return b"RIFFbase", {"duration": 1}


@pytest.fixture()
def clone_client(monkeypatch):
    pytest.importorskip("prometheus_client")
    from fastapi.testclient import TestClient
    from server import main
    from server.core import coordinator, session_limiter, tts_runtime

    backend = _Backend()

    class Manager:
        state = "ready"

        def get_backend_unsafe(self):
            return backend

        def is_ready(self):
            return True

        @asynccontextmanager
        async def acquire(self):
            yield backend

    manager = Manager()
    async def ensure():
        return manager

    monkeypatch.setattr(main, "_ensure_tts_manager_started", ensure)
    coordinator._coordinator = None
    coordinator.init_coordinator({"mode": "concurrent"})
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"max_concurrent_sessions": 4})
    tts_runtime.reset_overrides()
    client = TestClient(main.app)
    try:
        yield client, backend, manager
    finally:
        tts_runtime.reset_overrides()
        session_limiter._reset_for_tests()


def test_embedding_clone_is_strict_and_model_scoped(clone_client):
    client, backend, _manager = clone_client
    embedding = struct.pack("<ff", 0.25, -0.5)
    response = client.post(
        "/v1/tts/clone/embedding",
        json={
            "model": backend.model_id,
            "text": "hello",
            "embedding_b64": base64.b64encode(embedding).decode(),
            "dim": 2,
        },
    )
    assert response.status_code == 200, response.text
    assert backend.clone_calls[-1][2]["speaker_embedding"] == embedding

    mismatch = client.post(
        "/v1/tts/clone/embedding",
        json={
            "model": backend.model_id,
            "text": "hello",
            "embedding_b64": base64.b64encode(embedding).decode(),
            "dim": 3,
        },
    )
    assert mismatch.status_code == 400
    assert mismatch.json()["error"]["code"] == "invalid_profile"

    malformed = client.post(
        "/v1/tts/clone/embedding",
        json={
            "model": backend.model_id,
            "text": "hello",
            "speaker_embedding_b64": "not base64!",
        },
    )
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "invalid_profile"

    unknown = client.post(
        "/v1/tts/clone/embedding",
        json={
            "model": "qwen3-tts-customvoice",
            "text": "hello",
            "embedding_b64": base64.b64encode(embedding).decode(),
        },
    )
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "unknown_model"


def test_embedding_limit_rejects_before_base64_decode(monkeypatch):
    from server import main

    monkeypatch.setenv("OVS_API_MAX_PROFILE_BYTES", "3")
    called = False

    def forbidden_decode(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("oversized embedding must not be decoded")

    monkeypatch.setattr(main.base64, "b64decode", forbidden_decode)
    req = main.V1CloneEmbeddingRequest(
        model="qwen3-tts-0.6b-base",
        text="x",
        embedding_b64="A" * 8,
    )
    with pytest.raises(Exception) as caught:
        main._v1_decode_embedding(req)
    assert getattr(caught.value, "status_code", None) == 413
    assert called is False


def test_moss_reference_contract_uses_codec_metadata_not_output_shape(tmp_path, monkeypatch):
    from server import main

    meta = tmp_path / "codec_browser_onnx_meta.json"
    meta.write_text(
        '{"codec_config":{"sample_rate":48000,"channels":2}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("MOSS_CODEC_META_PATH", str(meta))

    class Backend:
        # Deliberately different synthesized output/downmix contract.
        sample_rate = 24_000
        channels = 1

    assert main._v1_moss_codec_contract(Backend()) == (48_000, 2)


def test_moss_reference_strips_wav_and_rejects_codec_mismatch(clone_client):
    client, _backend, manager = clone_client
    moss = _Moss()
    manager.get_backend_unsafe = lambda: moss
    manager.acquire = lambda: _manager_cm(moss)
    reference = _wav()
    response = client.post(
        "/v1/tts/clone/reference",
        files={"file": ("ref.wav", reference, "audio/wav")},
        data={"model": moss.model_id, "text": "hello"},
    )
    assert response.status_code == 200, response.text
    assert moss.clone_calls[-1][2]["reference_audio"] == b"\x00\x00\x01\x00"

    wrong_rate = client.post(
        "/v1/tts/clone/reference",
        files={"file": ("ref.wav", _wav(sample_rate=16_000), "audio/wav")},
        data={"model": moss.model_id, "text": "hello"},
    )
    assert wrong_rate.status_code == 400
    assert wrong_rate.json()["error"]["code"] == "audio_contract_mismatch"


def test_moss_reference_rejects_ambiguous_duplicate_fmt(monkeypatch):
    from server import main

    moss = _Moss()
    wav = _wav()
    fmt_chunk = wav[12:36]
    forged_body = fmt_chunk + wav[12:]
    forged = b"RIFF" + struct.pack("<I", 4 + len(forged_body)) + b"WAVE" + forged_body
    with pytest.raises(Exception) as caught:
        main._v1_parse_moss_reference_wav(forged, moss)
    assert getattr(caught.value, "code", None) == "invalid_audio"


def test_base_reference_preserves_wav_and_requires_transcript(clone_client):
    client, _backend, manager = clone_client
    base = _BaseReference()
    manager.get_backend_unsafe = lambda: base
    manager.acquire = lambda: _manager_cm(base)
    reference = _wav(b"\x00\x00" * 1280, sample_rate=16_000)
    response = client.post(
        "/v1/tts/clone/reference",
        files={"file": ("ref.wav", reference, "audio/wav")},
        data={"model": base.model_id, "text": "hello", "ref_text": "reference transcript"},
    )
    assert response.status_code == 200, response.text
    assert base.clone_calls[-1][2]["reference_audio_wav"] == reference

    missing = client.post(
        "/v1/tts/clone/reference",
        files={"file": ("ref.wav", reference, "audio/wav")},
        data={"model": base.model_id, "text": "hello"},
    )
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == "missing_required_parameter"


def test_base_reference_wav_rejects_trailing_and_truncated_bytes_but_accepts_metadata():
    from server import main

    payload = b"\x00\x00" * 2400  # 100 ms at 24 kHz, mono PCM16
    valid = _wav(payload, sample_rate=24_000)
    junk = valid + b"outside-riff"
    with pytest.raises(Exception) as trailing:
        main._v1_parse_base_reference_wav(junk)
    assert getattr(trailing.value, "code", None) == "invalid_audio"

    data_pos = valid.index(b"data")
    truncated = valid[: data_pos + 8] + payload[:-2]
    with pytest.raises(Exception) as short:
        main._v1_parse_base_reference_wav(truncated)
    assert getattr(short.value, "code", None) == "invalid_audio"

    fmt_end = valid.index(b"data")
    metadata = b"JUNK" + struct.pack("<I", 3) + b"abc" + b"\x00"
    with_metadata = valid[:fmt_end] + metadata + valid[fmt_end:]
    with_metadata = with_metadata[:4] + struct.pack("<I", len(with_metadata) - 8) + with_metadata[8:]
    assert main._v1_parse_base_reference_wav(with_metadata) == with_metadata


def test_base_reference_text_uses_ref_text_validation_contract(monkeypatch):
    from server import main

    monkeypatch.setenv("OVS_API_MAX_TEXT_BYTES", "3")
    with pytest.raises(Exception) as too_long:
        main._v1_validate_text("你好", param="ref_text")
    assert getattr(too_long.value, "param", None) == "ref_text"
    with pytest.raises(Exception) as surrogate:
        main._v1_validate_text("\ud800", param="ref_text")
    assert getattr(surrogate.value, "param", None) == "ref_text"


def _manager_cm(backend):
    @asynccontextmanager
    async def manager_cm():
        yield backend

    return manager_cm()


def test_spark_raw_embedding_fails_before_backend(monkeypatch, clone_client):
    client, backend, manager = clone_client
    backend.model_id = "sparktts-0p5b"
    backend.clone_calls.clear()
    response = client.post(
        "/v1/tts/clone/embedding",
        json={
            "model": backend.model_id,
            "text": "hello",
            "embedding_b64": base64.b64encode(b"\x00" * 8).decode(),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_clone_mode"
    assert not backend.clone_calls


def test_clone_stream_passes_cancel_event_and_releases_session(clone_client):
    client, backend, _manager = clone_client
    embedding = base64.b64encode(struct.pack("<f", 0.5)).decode()
    with client.stream(
        "POST",
        "/v1/tts/clone/embedding/stream",
        json={"model": backend.model_id, "text": "hello", "embedding_b64": embedding},
    ) as response:
        assert response.status_code == 200
        payload = b"".join(response.iter_bytes())
        assert payload.startswith(struct.pack("<I", backend.sample_rate))
        assert payload[4:]
    # Starlette closes the response generator; the backend receives the
    # shared cancellation event even when the client stops after first PCM.
    assert backend.stream_calls
    event = backend.stream_calls[-1][2]
    assert event is not None
    for _ in range(20):
        if event.is_set():
            break
        time.sleep(0.01)
    assert event.is_set()


def test_clone_stream_backend_error_is_json_before_audio_header(clone_client):
    client, backend, _manager = clone_client

    def fail_before_pcm(text, language=None, cancel_event=None, **kwargs):
        raise RuntimeError("private backend failure")
        yield b""  # pragma: no cover - keep this a generator function

    backend.generate_streaming = fail_before_pcm
    embedding = base64.b64encode(struct.pack("<f", 0.5)).decode()
    response = client.post(
        "/v1/tts/clone/embedding/stream",
        json={"model": backend.model_id, "text": "hello", "embedding_b64": embedding},
    )
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "backend_error"
    assert "private backend failure" not in response.text


@pytest.mark.parametrize(
    ("first_chunk", "code"),
    [("not-bytes", "invalid_backend_audio"), (b"\x00", "invalid_backend_audio")],
)
def test_clone_stream_rejects_invalid_first_pcm_before_header(clone_client, first_chunk, code):
    client, backend, _manager = clone_client

    def invalid_stream(text, language=None, cancel_event=None, **kwargs):
        yield first_chunk

    backend.generate_streaming = invalid_stream
    embedding = base64.b64encode(struct.pack("<f", 0.5)).decode()
    response = client.post(
        "/v1/tts/clone/embedding/stream",
        json={"model": backend.model_id, "text": "hello", "embedding_b64": embedding},
    )
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == code


def test_clone_stream_skips_empty_chunk_then_surfaces_backend_error_before_header(clone_client):
    client, backend, _manager = clone_client

    def empty_then_fail(text, language=None, cancel_event=None, **kwargs):
        yield b""
        raise RuntimeError("failure after keepalive")

    backend.generate_streaming = empty_then_fail
    embedding = base64.b64encode(struct.pack("<f", 0.5)).decode()
    response = client.post(
        "/v1/tts/clone/embedding/stream",
        json={"model": backend.model_id, "text": "hello", "embedding_b64": embedding},
    )
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "backend_error"


def test_clone_stream_saturation_is_json_before_audio_header(clone_client):
    from server.core import session_limiter

    client, backend, _manager = clone_client
    limiter = session_limiter.get_limiter()
    token = limiter.try_acquire()
    assert token is not None
    try:
        embedding = base64.b64encode(struct.pack("<f", 0.5)).decode()
        response = client.post(
            "/v1/tts/clone/embedding/stream",
            json={"model": backend.model_id, "text": "hello", "embedding_b64": embedding},
        )
    finally:
        token.release()
    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["retry-after"] == "5"
    assert response.json()["error"]["code"] == "too_many_sessions"


class _StreamRequest:
    def __init__(self):
        self.messages = asyncio.Queue()

    async def receive(self):
        return await self.messages.get()

    async def disconnect(self):
        await self.messages.put({"type": "http.disconnect"})


class _LeaseManager:
    def __init__(self, backend):
        self.backend = backend
        self.entered = 0
        self.exited = 0

    @asynccontextmanager
    async def acquire(self):
        self.entered += 1
        try:
            yield self.backend
        finally:
            self.exited += 1


class _LeaseCoordinator:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    @asynccontextmanager
    async def acquire(self, _slot):
        self.entered += 1
        try:
            yield
        finally:
            self.exited += 1


class _CancellableBackend(_Backend):
    def __init__(self, *, wait_before_first=False):
        super().__init__()
        self.wait_before_first = wait_before_first

    def generate_streaming(self, text, language=None, cancel_event=None, **kwargs):
        self.stream_calls.append((text, language, cancel_event, kwargs))
        if self.wait_before_first:
            while cancel_event is not None and not cancel_event.is_set():
                time.sleep(0.005)
            return
        yield b"\x01\x00\x02\x00"
        while cancel_event is not None and not cancel_event.is_set():
            time.sleep(0.005)


@pytest.mark.parametrize("manager_path", [True, False])
def test_clone_stream_post_first_cancel_releases_manager_or_legacy(monkeypatch, manager_path):
    """Closing after the first PCM sets cancel and drains before lease release."""
    pytest.importorskip("prometheus_client")
    from server import main
    from server.core import coordinator as coordinator_module, session_limiter
    from server.core import tts_service

    backend = _CancellableBackend()
    manager = _LeaseManager(backend)
    coord = _LeaseCoordinator()
    monkeypatch.setattr(coordinator_module, "get_coordinator", lambda: coord)
    monkeypatch.setattr(main, "_ensure_tts_manager_started", lambda: _async_result(manager if manager_path else None))
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: backend)
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"max_concurrent_sessions": 1})
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(main, "_get_tts_stream_executor", lambda: executor)

    async def run():
        request = _StreamRequest()
        response = await main._v1_clone_stream_impl(
            request,
            text="hello",
            language=None,
            prepare=lambda _backend: {},
            endpoint="/v1/test/clone",
        )
        iterator = response.body_iterator
        assert await iterator.__anext__() == struct.pack("<I", backend.sample_rate)
        assert await iterator.__anext__() == b"\x01\x00\x02\x00"
        await request.disconnect()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(iterator.__anext__(), timeout=1)
        for _ in range(100):
            if session_limiter.get_limiter().active == 0 and coord.exited:
                break
            await asyncio.sleep(0.005)

    try:
        asyncio.run(run())
    finally:
        executor.shutdown(wait=True)
        session_limiter._reset_for_tests()
    assert backend.stream_calls[-1][2].is_set()
    assert coord.entered == coord.exited == 1
    assert session_limiter.get_limiter() is None or session_limiter.get_limiter().active == 0
    if manager_path:
        assert manager.entered == manager.exited == 1


@pytest.mark.parametrize("manager_path", [True, False])
def test_clone_stream_pre_first_cancel_releases_every_lease(monkeypatch, manager_path):
    pytest.importorskip("prometheus_client")
    from server import main
    from server.core import coordinator as coordinator_module, session_limiter
    from server.core import tts_service

    backend = _CancellableBackend(wait_before_first=True)
    manager = _LeaseManager(backend)
    coord = _LeaseCoordinator()
    monkeypatch.setattr(coordinator_module, "get_coordinator", lambda: coord)
    monkeypatch.setattr(main, "_ensure_tts_manager_started", lambda: _async_result(manager if manager_path else None))
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: backend)
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"max_concurrent_sessions": 1})
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(main, "_get_tts_stream_executor", lambda: executor)

    async def run():
        request = _StreamRequest()
        pending = asyncio.create_task(
            main._v1_clone_stream_impl(
                request,
                text="hello",
                language=None,
                prepare=lambda _backend: {},
                endpoint="/v1/test/clone",
            )
        )
        await asyncio.sleep(0.05)
        await request.disconnect()
        with pytest.raises(Exception) as caught:
            await asyncio.wait_for(pending, timeout=1)
        assert getattr(caught.value, "status_code", None) == 503
        for _ in range(100):
            if session_limiter.get_limiter().active == 0 and coord.exited:
                break
            await asyncio.sleep(0.005)

    try:
        asyncio.run(run())
    finally:
        executor.shutdown(wait=True)
        session_limiter._reset_for_tests()
    assert backend.stream_calls[-1][2].is_set()
    assert coord.entered == coord.exited == 1
    if manager_path:
        assert manager.entered == manager.exited == 1


@pytest.mark.parametrize("manager_path", [True, False])
def test_clone_stream_queued_not_started_is_cancelled_without_lease_leak(monkeypatch, manager_path):
    pytest.importorskip("prometheus_client")
    from server import main
    from server.core import coordinator as coordinator_module, session_limiter
    from server.core import tts_service
    from concurrent.futures import ThreadPoolExecutor

    backend = _CancellableBackend()
    manager = _LeaseManager(backend)
    coord = _LeaseCoordinator()
    monkeypatch.setattr(coordinator_module, "get_coordinator", lambda: coord)
    monkeypatch.setattr(main, "_ensure_tts_manager_started", lambda: _async_result(manager if manager_path else None))
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: backend)
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"max_concurrent_sessions": 1})
    executor = ThreadPoolExecutor(max_workers=1)
    blocker_started = threading.Event()
    unblock = threading.Event()

    def blocker():
        blocker_started.set()
        unblock.wait(2)

    executor.submit(blocker)
    blocker_started.wait(1)
    monkeypatch.setattr(main, "_get_tts_stream_executor", lambda: executor)

    async def run():
        request = _StreamRequest()
        pending = asyncio.create_task(
            main._v1_clone_stream_impl(
                request,
                text="hello",
                language=None,
                prepare=lambda _backend: {},
                endpoint="/v1/test/clone",
            )
        )
        await asyncio.sleep(0.05)
        await request.disconnect()
        with pytest.raises(Exception) as caught:
            await asyncio.wait_for(pending, timeout=1)
        assert getattr(caught.value, "status_code", None) == 503
        # The queued backend job was never started; cancellation can release
        # admission immediately without waiting for the unrelated blocker.
        assert session_limiter.get_limiter().active == 0
        assert coord.entered == coord.exited == 1

    try:
        asyncio.run(run())
    finally:
        unblock.set()
        executor.shutdown(wait=True)
        session_limiter._reset_for_tests()
    assert not backend.stream_calls
    if manager_path:
        assert manager.entered == manager.exited == 1


async def _async_result(value):
    return value
