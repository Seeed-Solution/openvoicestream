"""Integration tests for the PR4 main.py wiring.

These tests bypass the heavy ``@app.on_event("startup")`` (which would try to
download models, load real ASR/TTS, etc.) by manually installing fake
BackendManager singletons before the TestClient is constructed. The TestClient
intentionally does NOT use ``with`` so the startup event never fires.

Coverage:
* /tts goes through ``tts_manager().acquire()`` (the fake records calls)
* tts_runtime overrides take effect on /tts when payload omits speaker_id
* explicit request speaker_id beats the runtime override
* /admin/backend/status returns both kinds
* /admin/backend/reload validates ``kind``
* admin auth: TestClient default host=testclient → 403 when key unset
* admin auth: loopback host bypass works
* /admin/backend/reload swaps to a fresh backend instance
* successful reload swaps the backend instance
* /admin/tts/speakers/reload calls into the speakers module
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from types import MethodType
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------

class _FakeTTSBackend:
    name = "fake-tts"
    model_id = "qwen3-tts"  # match speaker table used in tts_speakers.json
    sample_rate = 16000
    # PR5: opt in so existing reload tests still pass.
    supports_hot_reload = True

    def __init__(self) -> None:
        # Advertise everything by default; individual tests can override.
        from server.core.tts_backend import TTSCapability
        self.capabilities = {TTSCapability.STREAMING, TTSCapability.VOICE_CLONE}
        self._ready = False
        self.synthesize_calls: list[dict] = []
        self.streaming_calls: list[dict] = []
        self.clone_calls: list[dict] = []
        self.unloaded = False
        # Hook the test can set to observe inflight_http during a call.
        self.inflight_observer = None

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        self._ready = True

    def unload(self) -> None:
        self.unloaded = True
        self._ready = False

    def has_capability(self, cap) -> bool:
        return cap in self.capabilities

    def synthesize(self, text, **kwargs):
        self.synthesize_calls.append({"text": text, **kwargs})
        if self.inflight_observer is not None:
            self.inflight_observer()
        return b"\x00\x00" * 16, {"duration": 0.001, "inference_time": 0.001, "rtf": 1.0}

    def generate_streaming(self, text, **kwargs):
        self.streaming_calls.append({"text": text, **kwargs})
        if self.inflight_observer is not None:
            self.inflight_observer()
        yield b"\x00\x00" * 8

    def clone_voice(self, text, speaker_embedding, language=None, **kwargs):
        self.clone_calls.append(
            {"text": text, "speaker_embedding": speaker_embedding, "language": language, **kwargs}
        )
        if self.inflight_observer is not None:
            self.inflight_observer()
        return b"\x00\x00" * 16, {"duration": 0.001, "inference_time": 0.001, "rtf": 1.0}


class _FakeASRBackend:
    name = "fake-asr"
    sample_rate = 16000
    # PR5: opt in so existing reload tests still pass.
    supports_hot_reload = True

    def __init__(self) -> None:
        self.capabilities = set()
        self._ready = False
        self.unloaded = False

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        self._ready = True

    def unload(self) -> None:
        self.unloaded = True
        self._ready = False

    def has_capability(self, cap) -> bool:
        return cap in self.capabilities


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _install_managers(asr=None, tts=None):
    """Reset module-level managers and install fakes (started)."""
    from server.core import backend_manager as bm
    from server.core import coordinator as coord_mod
    from server.core import session_limiter as sl_mod
    bm._reset_for_tests()
    # Ensure the coordinator singleton exists (default concurrent policy);
    # endpoint code calls get_coordinator() unconditionally.
    coord_mod._coordinator = None  # type: ignore[attr-defined]
    coord_mod.init_coordinator({"mode": "concurrent"})
    # Stale-test fix: the session limiter was introduced after this test was
    # written. The HTTP admission path (acquire_http) and WS path now require a
    # global limiter that is normally initialized in @app.on_event("startup")
    # (server/main.py:671), which these tests intentionally bypass. Without it
    # every /tts(/clone) call short-circuits to 503 session_limiter_unavailable.
    # Initialize it here (parallel to the coordinator init above) with a high
    # ceiling so admission never throttles the wiring assertions under test.
    sl_mod._reset_for_tests()
    sl_mod.init_limiter({"max_concurrent_sessions": 64})

    asr_be = asr or _FakeASRBackend()
    tts_be = tts or _FakeTTSBackend()

    bm.init_backend_managers(
        tts_factory=lambda: tts_be,
        tts_preloader=lambda b: b.preload(),
        tts_unloader=lambda b: b.unload(),
        asr_factory=lambda: asr_be,
        asr_preloader=lambda b: b.preload(),
        asr_unloader=lambda b: b.unload(),
    )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(bm.tts_manager().start())
        loop.run_until_complete(bm.asr_manager().start())
    finally:
        loop.close()
    return asr_be, tts_be


def _tts_stream_registries():
    from server import main as appmod

    return {
        "_tts_stream_cleanup_tasks": appmod._tts_stream_cleanup_tasks,
        "_tts_stream_watcher_tasks": appmod._tts_stream_watcher_tasks,
        "_tts_framed_stream_owners": appmod._tts_framed_stream_owners,
        "_tts_stream_orphaned_cleanups": appmod._tts_stream_orphaned_cleanups,
    }


def _describe_stream_registries() -> str:
    return "; ".join(
        f"{name}={sorted(map(repr, registry))}"
        for name, registry in _tts_stream_registries().items()
        if registry
    )


async def _drain_tts_stream_registries(timeout: float = 2.0) -> None:
    """Await this loop's background stream cleanup while the loop still runs.

    Callers first unblock every fake backend, so a healthy cleanup finishes
    promptly.  Anything still registered after ``timeout`` is a real leak and
    is reported by the ``_isolate_tts_stream_registries`` teardown.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        registries = _tts_stream_registries()
        pending = {
            task
            for name in ("_tts_stream_cleanup_tasks", "_tts_stream_watcher_tasks")
            for task in registries[name]
            if not task.done() and task.get_loop() is loop
        }
        orphaned = [
            record
            for record in list(registries["_tts_stream_orphaned_cleanups"])
            if record.loop is loop
        ]
        remaining = deadline - loop.time()
        if (not pending and not orphaned) or remaining <= 0:
            break
        if pending:
            await asyncio.wait(pending, timeout=min(remaining, 0.05))
        else:
            # Orphaned cleanups release from a watcher thread via
            # call_soon_threadsafe; give the loop a turn to run it.
            await asyncio.sleep(0.01)
    # Let done-callbacks deregister finished tasks before the loop closes.
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def _run_stream_exercise(exercise, *unblock: threading.Event) -> None:
    """``asyncio.run(exercise())`` that settles stream cleanup in-loop.

    Whether ``exercise`` passes or fails, unblock the fake backends and await
    the background cleanup tasks before ``asyncio.run`` closes the loop, so
    no task bound to this loop outlives the test.
    """

    async def _main():
        try:
            await exercise()
        finally:
            for event in unblock:
                event.set()
            await _drain_tts_stream_registries()

    asyncio.run(_main())


@pytest.fixture(autouse=True)
def _isolate_tts_stream_registries():
    """Fail a test that leaves background stream state behind.

    ``server.main`` holds strong references to background stream cleanup
    tasks, disconnect watchers and framed-stream owners in module-level sets.
    Each test runs its own event loop, so an entry that survives the test is
    bound to a closed loop; a later ``gather(*_tts_stream_cleanup_tasks)``
    would raise ``The future belongs to a different loop``.  Tests that drive
    ``tts_stream`` directly settle cleanup inside their loop via
    ``_run_stream_exercise``; anything left here is a leak.  The leaking test
    fails with the residual entries, and the registries are cleared so the
    leak does not cascade into unrelated tests.
    """
    residual = _describe_stream_registries()
    assert not residual, f"TTS stream registries not empty before test: {residual}"
    yield
    residual = _describe_stream_registries()
    for registry in _tts_stream_registries().values():
        registry.clear()
    if residual:
        pytest.fail(
            "test leaked background TTS stream state past its event loop: "
            + residual,
            pytrace=False,
        )


@pytest.fixture
def client(monkeypatch):
    from server.core import tts_runtime, tts_service
    tts_runtime.reset_overrides()

    asr_be, tts_be = _install_managers()

    # Some endpoints still inspect tts_service.is_ready() / get_backend()
    # in the partial-wiring fallback path. Wire it to the fake.
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: tts_be)
    monkeypatch.setattr(tts_service, "is_configured", lambda: True)
    monkeypatch.setattr(tts_service, "_backend", tts_be, raising=False)

    from server.main import app
    from server.core.admin_auth import require_admin

    async def _allow():
        return None

    app.dependency_overrides[require_admin] = _allow

    c = TestClient(app)
    c.tts_be = tts_be   # type: ignore[attr-defined]
    c.asr_be = asr_be   # type: ignore[attr-defined]
    try:
        yield c
    finally:
        app.dependency_overrides.pop(require_admin, None)
        tts_runtime.reset_overrides()
        from server.core import backend_manager as bm
        bm._reset_for_tests()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tts_endpoint_uses_manager_acquire(client):
    """A POST to /tts should call backend.synthesize via the manager path."""
    r = client.post("/tts", json={"text": "hello"})
    assert r.status_code == 200, r.text
    assert len(client.tts_be.synthesize_calls) == 1
    call = client.tts_be.synthesize_calls[0]
    assert call["text"] == "hello"


def test_tts_runtime_override_applied(client):
    """PATCH /admin/tts/runtime sets default_speaker_id → /tts picks it up."""
    # 2301 is a known preset id for qwen3-tts in the bundled speakers table.
    r = client.patch("/admin/tts/runtime", json={"speaker_id": 2301})
    assert r.status_code == 200, r.text

    client.tts_be.synthesize_calls.clear()
    r = client.post("/tts", json={"text": "hi"})
    assert r.status_code == 200, r.text
    call = client.tts_be.synthesize_calls[-1]
    # speaker_kwargs_for_id translates 2301 into a backend-specific kwarg
    # (either ``speaker_id`` or an embedding). The key thing is it's NOT
    # the default speaker id (which would be 0 for qwen3-tts).
    if "speaker_id" in call:
        assert call["speaker_id"] == 2301


def test_tts_request_param_overrides_runtime(client):
    """An explicit speaker_id in the request wins over the runtime override."""
    client.patch("/admin/tts/runtime", json={"speaker_id": 2301})
    client.tts_be.synthesize_calls.clear()

    r = client.post("/tts", json={"text": "hi", "speaker_id": 0})
    assert r.status_code == 200, r.text
    call = client.tts_be.synthesize_calls[-1]
    if "speaker_id" in call:
        assert call["speaker_id"] == 0


def test_admin_backend_status(client):
    r = client.get("/admin/backend/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "tts" in body and "asr" in body
    assert body["tts"]["state"] == "ready"
    assert body["asr"]["state"] == "ready"
    assert body["tts"]["backend_name"] == "fake-tts"
    assert body["asr"]["backend_name"] == "fake-asr"


def test_admin_backend_reload_unknown_kind(client):
    r = client.post("/admin/backend/reload", json={"kind": "xxx", "profile": "p"})
    # Pydantic literal rejection → 422.
    assert r.status_code in (400, 422), r.text


def test_admin_backend_reload_missing_auth_non_loopback(monkeypatch):
    """TestClient default host=testclient + no OVS_ADMIN_KEY → 403."""
    monkeypatch.delenv("OVS_ADMIN_KEY", raising=False)
    from server.core import tts_runtime, tts_service
    tts_runtime.reset_overrides()
    _install_managers()
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)

    from server.main import app
    # No dependency_overrides this time → real require_admin runs.
    c = TestClient(app)
    r = c.post("/admin/backend/reload", json={"kind": "tts", "profile": "fake"})
    assert r.status_code == 403, r.text
    from server.core import backend_manager as bm
    bm._reset_for_tests()


def test_admin_backend_reload_loopback_allowed(monkeypatch, tmp_path):
    """Loopback client.host bypasses the OVS_ADMIN_KEY check."""
    monkeypatch.delenv("OVS_ADMIN_KEY", raising=False)
    from server.core import tts_runtime, tts_service, profile_loader
    tts_runtime.reset_overrides()
    asr_be, tts_be = _install_managers()
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: tts_be)

    # Stub apply_profile / current_profile / path resolver so reload doesn't
    # touch the real configs/profiles tree.
    import tempfile, json as _json
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    tmp.write(_json.dumps({"name": "any", "tts_backend": "kokoro"}))
    tmp.flush()
    from pathlib import Path as _Path
    from server.core import backend_manager as bm
    monkeypatch.setattr(bm, "_resolve_profile_path", lambda ref: _Path(tmp.name))
    monkeypatch.setattr(
        profile_loader, "current_profile",
        lambda: {"name": "live", "tts_backend": "kokoro"},
    )
    monkeypatch.setattr(
        profile_loader, "apply_profile",
        lambda ref, *, overrides=None, resolve_engines=False, kind=None: None,
    )

    from server.main import app
    c = TestClient(app)

    from server.core import admin_auth
    monkeypatch.setattr(admin_auth, "_is_loopback", lambda host: True)

    r = c.post("/admin/backend/reload", json={"kind": "tts", "profile": "any"})
    # Reload should succeed (fakes preload trivially) and return status dict.
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in ("reloaded", "rolled_back")
    from server.core import backend_manager as bm
    bm._reset_for_tests()


def test_admin_backend_reload_success_swaps_backend(client, monkeypatch):
    """Successful reload returns ``reloaded`` and bumps backend instance."""
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader, "current_profile",
        lambda: {"name": "p1", "tts_backend": "kokoro"},
    )
    monkeypatch.setattr(
        profile_loader, "apply_profile",
        lambda ref, *, overrides=None, resolve_engines=False, kind=None: None,
    )

    # Pre-set a synthetic profile path that parses to the same backend kind.
    import tempfile, json as _json
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    tmp.write(_json.dumps({"name": "p2", "tts_backend": "kokoro"}))
    tmp.flush()
    from pathlib import Path as _Path
    from server.core import backend_manager as bm
    monkeypatch.setattr(bm, "_resolve_profile_path", lambda ref: _Path(tmp.name))

    r = client.post("/admin/backend/reload", json={"kind": "tts", "profile": "p2"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "reloaded"
    assert body["kind"] == "tts"


def test_admin_tts_speakers_reload(client):
    """The speakers reload route still works under the new wiring."""
    r = client.post("/admin/tts/speakers/reload")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reloaded"] is True


# ---------------------------------------------------------------------------
# FIX_1 / FIX_2 / FIX_3 regression tests (PR4b)
# ---------------------------------------------------------------------------

def _observe_inflight(client):
    """Return a hook the fake backend can call mid-request to record inflight_http."""
    from server.core.backend_manager import tts_manager
    observed: list[int] = []

    def hook():
        observed.append(tts_manager().status()["inflight_http"])

    client.tts_be.inflight_observer = hook
    return observed


def test_tts_stream_uses_manager_acquire(client):
    """FIX_1: /tts/stream must bump tts_manager().inflight_http during the call."""
    observed = _observe_inflight(client)
    r = client.post("/tts/stream", json={"text": "hello"})
    assert r.status_code == 200, r.text
    # Consume the body so the StreamingResponse generator runs to completion
    # (TestClient eagerly buffers, so by the time we read .content the
    # generator has finished — but observed[] is populated as a side effect).
    _ = r.content
    assert observed, "backend.generate_streaming was never called"
    assert all(n >= 1 for n in observed), f"expected inflight>=1 during call, got {observed}"
    # And streaming kwargs were forwarded (no stray speed/pitch since override unset)
    assert client.tts_be.streaming_calls, "generate_streaming not invoked"


def test_tts_stream_disconnect_waits_for_backend_slot_before_recovery(
    monkeypatch,
):
    """A disconnected stream must finish backend cleanup before admission
    tokens are returned.

    This models the production WorkerIO race without a GPU: request 1 emits
    one chunk, then remains blocked inside ``next(gen)``.  Cross-thread
    ``gen.close()`` cannot interrupt an already executing generator, so the
    endpoint must pass the shared cancel event into the backend.  The backend
    can then cancel its worker while blocked and cleanup can retain all leases
    until the real worker slot is free.
    """
    from server.core import tts_service
    from server.core.worker_io import PoolSaturatedError
    from server.main import TTSRequest, tts_stream
    from starlette.requests import Request

    class _SingleSlotBackend(_FakeTTSBackend):
        def __init__(self) -> None:
            super().__init__()
            self._slot = threading.Lock()
            self.cleaned = threading.Event()

        def generate_streaming(self, text, **kwargs):
            if not self._slot.acquire(blocking=False):
                raise PoolSaturatedError(1)
            self.streaming_calls.append({"text": text, **kwargs})
            try:
                yield b"\x01\x00" * 8
                if text == "first":
                    cancel_event = kwargs.get("cancel_event")
                    assert cancel_event is not None
                    # Mirror a worker thread blocked in its next IPC read.
                    # The out-of-band event must release it;
                    # generator.close cannot.
                    assert cancel_event.wait(timeout=1.0)
                yield b"\x02\x00" * 8
            finally:
                self._slot.release()
                self.cleaned.set()

    backend = _SingleSlotBackend()
    _install_managers(tts=backend)
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: backend)
    monkeypatch.setattr(tts_service, "is_configured", lambda: True)
    monkeypatch.setattr(tts_service, "_backend", backend, raising=False)

    async def _exercise():
        receive_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def _receive():
            return await receive_queue.get()

        request = Request(
            {"type": "http", "method": "POST", "path": "/tts/stream",
             "headers": []},
            _receive,
        )
        response = await tts_stream(
            TTSRequest(text="first", language="en"), request, None
        )
        body = response.body_iterator
        rate_header = await body.__anext__()
        assert len(rate_header) == 4
        pcm = await body.__anext__()
        assert pcm

        await receive_queue.put({"type": "http.disconnect"})
        await body.aclose()
        assert backend.cleaned.is_set(), (
            "HTTP cleanup returned before the backend slot was released"
        )

        recovery_receive_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def _never_disconnect():
            return await recovery_receive_queue.get()

        recovery_request = Request(
            {"type": "http", "method": "POST", "path": "/tts/stream",
             "headers": []},
            _never_disconnect,
        )
        recovery = await tts_stream(
            TTSRequest(text="recovery", language="en"),
            recovery_request,
            None,
        )
        chunks = [chunk async for chunk in recovery.body_iterator]
        assert len(chunks) >= 2
        assert b"".join(chunks[1:]), "recovery stream returned empty PCM"

    try:
        _run_stream_exercise(_exercise)
    finally:
        from server.core import backend_manager as bm
        bm._reset_for_tests()


def test_v2v_deadline_propagates_cancel_and_releases_backend_slot(client, monkeypatch):
    """The real v2v handler must cancel a blocked stream before the next turn.

    The fake backend yields one chunk, then waits for ``cancel_event`` while
    holding a single slot.  The first turn therefore reaches the v2v sentence
    deadline.  The second turn proves that the handler waited for backend
    cleanup rather than releasing the manager lease while native work was
    still active.
    """
    backend = client.tts_be
    slot = threading.Lock()
    calls = []
    cancel_events = []
    cleaned = threading.Event()
    manager_inflight_at_cleanup = []

    def _stream(self, text, **kwargs):
        cancel_event = kwargs.get("cancel_event")
        assert cancel_event is not None
        calls.append(text)
        cancel_events.append(cancel_event)
        if text == "second":
            assert cleaned.wait(3.0), "second sentence raced the first native future"
        assert slot.acquire(blocking=False), "previous v2v synth still owns slot"
        try:
            yield b"\x01\x00" * 8
            if len(calls) == 1:
                assert cancel_event.wait(1.0), "v2v deadline did not reach backend"
                # The client deadline must return while the native future is
                # still alive; its manager and coordinator leases stay held
                # until this slow cancellation path actually exits.
                time.sleep(2.2)
                raise RuntimeError("late backend failure after cancellation")
            yield b"\x02\x00" * 8
        finally:
            from server.core.backend_manager import tts_manager
            manager_inflight_at_cleanup.append(
                tts_manager().status()["inflight_http"]
            )
            slot.release()
            cleaned.set()

    backend.generate_streaming = MethodType(_stream, backend)
    from server.core import coordinator as coord_mod
    coordinator = coord_mod.init_coordinator({"mode": "serialized"})
    monkeypatch.setenv("OVS_TTS_SENTENCE_TIMEOUT_S", "0.1")
    with client.websocket_connect("/v2v/stream") as ws:
        ws.send_json({
            "type": "config", "asr_language": None, "tts_language": "en",
            "vad": "none", "multi_utterance": True,
        })
        ws.send_json({"type": "text", "text": "first"})
        ws.send_json({"type": "tts_flush"})
        first_events = []
        while True:
            frame = ws.receive()
            if frame.get("bytes") is not None:
                continue
            event = json.loads(frame["text"])
            first_events.append(event)
            if event.get("type") == "tts_done":
                break
        assert any(e.get("type") == "error" for e in first_events), first_events
        from server.core.backend_manager import tts_manager
        assert tts_manager().status()["inflight_http"] == 1
        assert not cleaned.is_set(), "deadline waited for the slow native future"
        assert coordinator._lock is not None and coordinator._lock.locked()

        # Enqueue the next sentence while the detached owner is still
        # sleeping. It must wait for the old lease, then use fresh closures.
        monkeypatch.setenv("OVS_TTS_SENTENCE_TIMEOUT_S", "3.0")
        ws.send_json({"type": "text", "text": "second"})
        ws.send_json({"type": "tts_flush"})
        second_events = []
        while True:
            frame = ws.receive()
            if frame.get("bytes") is not None:
                continue
            event = json.loads(frame["text"])
            second_events.append(event)
            if event.get("type") == "tts_done":
                break
        assert calls == ["first", "second"]
        assert cancel_events[0] is not cancel_events[1]
        assert cancel_events[0].is_set()
        assert not cancel_events[1].is_set()
        assert not any(e.get("type") == "error" for e in second_events), second_events
        assert cleaned.wait(1.0), "manager returned before backend slot cleanup"
        assert manager_inflight_at_cleanup == [2, 1], manager_inflight_at_cleanup
        for _ in range(20):
            if tts_manager().status()["inflight_http"] == 0:
                break
            time.sleep(0.05)
        assert tts_manager().status()["inflight_http"] == 0
        assert coordinator._lock is not None and not coordinator._lock.locked()
        ws.send_json({"type": "asr_eos"})


def _speakable_takes_language() -> bool:
    import inspect

    from voxedge.engine.tts_sequencer import _to_speakable

    return "language" in inspect.signature(_to_speakable).parameters


@pytest.mark.skipif(
    not _speakable_takes_language(),
    reason="zh URL spelling needs voxedge>=0.0.15a2 (_to_speakable(language=))",
)
def test_v2v_tts_speakable_normalizes_markdown_before_backend(client):
    """V2V sends speakable text to TTS while protocol events keep the source."""
    backend = client.tts_be
    with client.websocket_connect("/v2v/stream") as ws:
        ws.send_json({
            "type": "config", "asr_language": None, "tts_language": "zh",
            "vad": "none", "multi_utterance": True,
        })
        source = "3. **访问权威气象网站**，如中国气象网 (www.nmc.cn)。"
        ws.send_json({"type": "text", "text": source})
        ws.send_json({"type": "tts_flush"})
        events = []
        while True:
            frame = ws.receive()
            if frame.get("bytes") is not None:
                continue
            event = json.loads(frame["text"])
            events.append(event)
            if event.get("type") == "tts_done":
                break

    assert [call["text"] for call in backend.streaming_calls] == [
        "访问权威气象网站，如中国气象网 (w w w 点 n m c 点 c n)。"
    ]
    started = [e for e in events if e.get("type") == "tts_started"]
    done = [e for e in events if e.get("type") == "tts_sentence_done"]
    assert started[0]["sentence"] == source
    assert done[0]["sentence"] == source


def test_v2v_tts_drops_markup_only_and_keeps_plain_text(client):
    """Markup-only input never starts TTS; ordinary text remains unchanged."""
    backend = client.tts_be
    with client.websocket_connect("/v2v/stream") as ws:
        ws.send_json({
            "type": "config", "asr_language": None, "tts_language": "en",
            "vad": "none", "multi_utterance": True,
        })
        ws.send_json({"type": "text", "text": "***"})
        ws.send_json({"type": "tts_flush"})
        events = []
        while True:
            frame = ws.receive()
            if frame.get("bytes") is not None:
                continue
            event = json.loads(frame["text"])
            events.append(event)
            if event.get("type") == "tts_done":
                break
        assert not any(e.get("type") == "tts_started" for e in events)
        assert not backend.streaming_calls

        ws.send_json({"type": "text", "text": "plain text"})
        ws.send_json({"type": "tts_flush"})
        while True:
            frame = ws.receive()
            if frame.get("bytes") is None and json.loads(frame["text"]).get("type") == "tts_done":
                break

    assert [call["text"] for call in backend.streaming_calls] == ["plain text"]


def test_v2v_deadline_while_acquiring_does_not_start_cancelled_sentence(client, monkeypatch):
    """A sentence canceled while waiting for a lease never starts native TTS."""
    import server.main as main_mod

    backend = client.tts_be
    calls = []

    def _stream(self, text, **kwargs):
        calls.append(text)
        yield b"\x01\x00" * 8

    backend.generate_streaming = MethodType(_stream, backend)

    class _DelayedAcquire:
        async def __aenter__(self):
            await asyncio.sleep(0.3)
            return backend

        async def __aexit__(self, *exc):
            return False

    class _DelayedManager:
        def acquire(self):
            return _DelayedAcquire()

        def register_ws(self, _handle):
            pass

        def unregister_ws(self, _handle):
            pass

    monkeypatch.setattr(main_mod, "_try_tts_manager", lambda: _DelayedManager())
    monkeypatch.setenv("OVS_TTS_SENTENCE_TIMEOUT_S", "0.05")
    with client.websocket_connect("/v2v/stream") as ws:
        ws.send_json({
            "type": "config", "asr_language": None, "tts_language": "en",
            "vad": "none", "multi_utterance": True,
        })
        ws.send_json({"type": "text", "text": "acquire-wait"})
        ws.send_json({"type": "tts_flush"})
        events = []
        while True:
            frame = ws.receive()
            if frame.get("bytes") is not None:
                continue
            event = json.loads(frame["text"])
            events.append(event)
            if event.get("type") == "tts_done":
                break
        assert any(event.get("type") == "error" for event in events)
        assert not any(event.get("type") == "tts_started" for event in events)
        time.sleep(0.4)
    assert calls == []


# Upper bounds for cross-thread handshakes in the V2V lifecycle test.  They
# return as soon as the event fires; the bound only limits a hung run.
_V2V_WAIT_S = 5.0


def _ws_receive(ws, timeout: float = _V2V_WAIT_S):
    """``ws.receive()`` with a deadline instead of blocking forever."""
    import anyio

    async def _receive():
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()

    return ws.portal.call(_receive)


@pytest.mark.parametrize("trigger", ["barge_in", "websocket_disconnect"])
def test_v2v_barge_in_and_disconnect_retain_slow_cleanup_leases(client, trigger):
    """V2V abort/disconnect retain leases until a canceled backend unwinds."""
    backend = client.tts_be
    cancel_seen = threading.Event()
    awaiting_cancel = threading.Event()
    cleaned = threading.Event()
    release_cleanup = threading.Event()
    inflight_at_cleanup = []

    def _stream(self, text, **kwargs):
        cancel_event = kwargs.get("cancel_event")
        assert cancel_event is not None
        try:
            yield b"\x01\x00" * 8
            # The executor resumed the generator past its first chunk.  The
            # V2V loop stops pulling chunks once its stop event is set, so a
            # trigger sent before this point would close the generator at the
            # yield instead of reaching the backend's cooperative cancel.
            awaiting_cancel.set()
            assert cancel_event.wait(_V2V_WAIT_S), "V2V trigger did not reach backend"
            cancel_seen.set()
            # Slow native unwind: held until the test has observed that the
            # V2V session did not wait for it.
            release_cleanup.wait(2 * _V2V_WAIT_S)
            raise RuntimeError("late failure after V2V cancellation")
        finally:
            from server.core.backend_manager import tts_manager
            inflight_at_cleanup.append(tts_manager().status()["inflight_http"])
            cleaned.set()

    backend.generate_streaming = MethodType(_stream, backend)
    try:
        with client.websocket_connect("/v2v/stream") as ws:
            ws.send_json({
                "type": "config", "asr_language": None, "tts_language": "en",
                "vad": "none", "multi_utterance": True,
            })
            ws.send_json({"type": "text", "text": "trigger"})
            ws.send_json({"type": "tts_flush"})
            events = []
            while True:
                frame = _ws_receive(ws)
                if frame.get("bytes") is not None:
                    continue
                event = json.loads(frame["text"])
                events.append(event)
                if event.get("type") == "tts_started":
                    break
            # ``tts_started`` is sent before the synth job is submitted.  Trigger
            # only once the backend is inside its slow native call.
            assert awaiting_cancel.wait(_V2V_WAIT_S)
            if trigger == "barge_in":
                ws.send_json({"type": "abort"})
                assert cancel_seen.wait(_V2V_WAIT_S)
                assert not cleaned.is_set(), "barge-in waited for slow native cleanup"
                release_cleanup.set()
                while True:
                    frame = _ws_receive(ws)
                    if frame.get("bytes") is not None:
                        continue
                    event = json.loads(frame["text"])
                    events.append(event)
                    if event.get("type") == "tts_done":
                        break
                assert not any(event.get("type") == "tts_sentence_done" for event in events)
            else:
                ws.close()

            if trigger == "websocket_disconnect":
                assert cancel_seen.wait(_V2V_WAIT_S)
                assert not cleaned.is_set(), "disconnect waited for slow native cleanup"
                release_cleanup.set()
            assert any(event.get("type") == "tts_started" for event in events)
            assert not any(event.get("error", "").startswith("tts: late failure") for event in events)
    finally:
        release_cleanup.set()

    assert cleaned.wait(_V2V_WAIT_S)
    assert inflight_at_cleanup == [1]


def test_tts_cleanup_does_not_wait_for_unstarted_prefetch():
    """A queued job that never touched the backend cannot retain its leases."""
    from server.main import _finish_tts_stream_cleanup

    async def _exercise():
        loop = asyncio.get_running_loop()
        queued = loop.create_future()
        started = threading.Event()
        released = asyncio.Event()

        async def _release():
            released.set()

        completed = await _finish_tts_stream_cleanup(
            [(queued, started)],
            _release,
            wait_s=0.05,
        )
        assert completed is True
        assert released.is_set()
        assert not queued.done()
        queued.cancel()

    asyncio.run(_exercise())


async def _wait_for_task_named(qualname_suffix: str, timeout: float = 5.0):
    """Handshake: return once a task running ``*qualname_suffix`` exists."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        for task in asyncio.all_tasks():
            if task.get_coro().__qualname__.endswith(qualname_suffix):
                return task
        assert loop.time() < deadline, f"no task {qualname_suffix!r} started"
        await asyncio.sleep(0)


def _wire_direct_tts_test_backend(monkeypatch, backend, *, session_limit=2):
    """Install a backend for direct async ``tts_stream`` calls."""
    from server.core import session_limiter, tts_service

    _install_managers(tts=backend)
    session_limiter._limiter = session_limiter.SessionLimiter(session_limit)
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: backend)
    monkeypatch.setattr(tts_service, "is_configured", lambda: True)
    monkeypatch.setattr(tts_service, "_backend", backend, raising=False)


def test_tts_stream_serial_multisentence_submits_next_and_reaches_eof(monkeypatch):
    """A one-slot pipeline must submit sentence N+1 after N completes."""
    from server import main as appmod
    from starlette.requests import Request

    backend = _FakeTTSBackend()
    _wire_direct_tts_test_backend(monkeypatch, backend)
    monkeypatch.setenv("OVS_TTS_STREAM_PREFETCH", "1")
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(appmod, "_get_tts_stream_executor", lambda: executor)

    async def _exercise():
        receive_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def _receive():
            return await receive_queue.get()

        response = await appmod.tts_stream(
            appmod.TTSRequest(
                text="First sentence. Second sentence.", language="en"
            ),
            Request(
                {"type": "http", "method": "POST", "path": "/tts/stream",
                 "headers": []},
                _receive,
            ),
            None,
        )

        async def _drain():
            return [chunk async for chunk in response.body_iterator]

        chunks = await asyncio.wait_for(_drain(), timeout=0.5)
        assert len(chunks) == 3  # sample-rate header + one chunk per sentence
        assert [call["text"] for call in backend.streaming_calls] == [
            "First sentence.",
            "Second sentence.",
        ]

    try:
        _run_stream_exercise(_exercise)
    finally:
        executor.shutdown(wait=True)
        from server.core import backend_manager as bm
        bm._reset_for_tests()


def test_tts_stream_saturation_is_429_before_rate_header(monkeypatch):
    """Pool saturation must be an HTTP status, never 200 + rate-only body."""
    from server.core.worker_io import PoolSaturatedError
    from server.main import TTSRequest, tts_stream
    from starlette.requests import Request

    class _SaturatedBackend(_FakeTTSBackend):
        def generate_streaming(self, text, **kwargs):
            raise PoolSaturatedError(2)
            yield  # pragma: no cover - keep this a generator function

    backend = _SaturatedBackend()
    _wire_direct_tts_test_backend(monkeypatch, backend)

    async def _exercise():
        receive_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def _receive():
            return await receive_queue.get()

        response = await tts_stream(
            TTSRequest(text="busy", language="en"),
            Request(
                {"type": "http", "method": "POST", "path": "/tts/stream",
                 "headers": []},
                _receive,
            ),
            None,
        )
        assert response.status_code == 429
        assert b"tts_backend_busy" in response.body

    try:
        _run_stream_exercise(_exercise)
    finally:
        from server.core import backend_manager as bm
        bm._reset_for_tests()


def test_tts_stream_legacy_saturation_is_429(monkeypatch):
    """The manager-less fallback must not swallow backend saturation."""
    from server import main as appmod
    from server.core import coordinator as coord_mod, session_limiter, tts_service
    from server.core.worker_io import PoolSaturatedError
    from starlette.requests import Request

    class _SaturatedLegacyBackend(_FakeTTSBackend):
        def generate_streaming(self, text, **kwargs):
            raise PoolSaturatedError(1)
            yield  # pragma: no cover

    backend = _SaturatedLegacyBackend()
    session_limiter._limiter = session_limiter.SessionLimiter(2)
    coord_mod.init_coordinator({"mode": "concurrent"})
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: backend)
    monkeypatch.setattr(tts_service, "get_sample_rate", lambda: backend.sample_rate)
    monkeypatch.setattr(tts_service, "has_capability", lambda _cap: True)

    async def _no_manager():
        return None

    monkeypatch.setattr(appmod, "_ensure_tts_manager_started", _no_manager)

    async def _exercise():
        receive_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def _receive():
            return await receive_queue.get()

        response = await appmod.tts_stream(
            appmod.TTSRequest(text="busy legacy", language="en"),
            Request(
                {"type": "http", "method": "POST", "path": "/tts/stream",
                 "headers": []},
                _receive,
            ),
            None,
        )
        assert response.status_code == 429
        assert b"tts_backend_busy" in response.body
        assert session_limiter.get_limiter().active == 0

    _run_stream_exercise(_exercise)


def test_tts_stream_cleanup_timeout_quarantines_exclusive_backend(
    monkeypatch,
):
    """A stuck ``next(gen)`` must not busy-loop or release exclusive leases.

    Foreground cleanup times out quickly, while the background cleanup retains
    the TTS session, BackendManager inflight count, and coordinator lock.  Once
    the backend unwinds, all three are released and ASR can acquire the
    exclusive coordinator.
    """
    from server import main as appmod
    from server.core import backend_manager as bm, coordinator as coord_mod
    from server.core import session_limiter
    from starlette.requests import Request

    release_backend = threading.Event()

    class _StuckBackend(_FakeTTSBackend):
        def generate_streaming(self, text, **kwargs):
            try:
                yield b"\x01\x00" * 8
                # Signal once the executor has resumed past the first chunk.  The
                # disconnect watcher closes a generator still suspended at its
                # yield, which legitimately unwinds the backend at once.
                self.stuck.set()
                release_backend.wait(timeout=5)
                yield b"\x02\x00" * 8
            finally:
                self.cleaned.set()

        def __init__(self):
            super().__init__()
            self.cleaned = threading.Event()
            self.stuck = threading.Event()

    backend = _StuckBackend()
    _wire_direct_tts_test_backend(monkeypatch, backend, session_limit=2)
    coordinator = coord_mod.init_coordinator({"mode": "serialized"})
    monkeypatch.setenv("OVS_TTS_STREAM_CLEANUP_WAIT_S", "0.02")

    async def _exercise():
        receive_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def _receive():
            return await receive_queue.get()

        response = await appmod.tts_stream(
            appmod.TTSRequest(text="stuck", language="en"),
            Request(
                {"type": "http", "method": "POST", "path": "/tts/stream",
                 "headers": []},
                _receive,
            ),
            None,
        )
        body = response.body_iterator
        assert await body.__anext__()  # sample rate
        assert await body.__anext__()  # first PCM
        assert await asyncio.to_thread(backend.stuck.wait, 1)
        await receive_queue.put({"type": "http.disconnect"})

        started = time.perf_counter()
        await body.aclose()
        assert time.perf_counter() - started < 0.25
        assert appmod._tts_stream_cleanup_tasks
        assert session_limiter.get_limiter().active == 1
        assert bm.tts_manager().status()["inflight_http"] == 1

        # A quarantined stream must also block hot reload.  Reload previously
        # hard-proceeded after its drain timeout and unloaded the backend while
        # this executor thread still used it.
        manager = bm.tts_manager()
        manager._drain_timeout_s = 0.02
        with pytest.raises(HTTPException) as reload_error:
            await manager.reload(None)
        assert getattr(reload_error.value, "status_code", None) == 409
        assert reload_error.value.detail["error"] == "backend_drain_timeout"
        assert manager.get_backend_unsafe() is backend
        assert backend.unloaded is False

        entered_asr = asyncio.Event()

        async def _acquire_asr():
            async with coordinator.acquire("asr"):
                entered_asr.set()

        asr_task = asyncio.create_task(_acquire_asr())
        await asyncio.sleep(0.03)
        assert not entered_asr.is_set(), (
            "serialized coordinator released before TTS executor cleanup"
        )

        release_backend.set()
        await asyncio.wait_for(
            asyncio.gather(*list(appmod._tts_stream_cleanup_tasks)),
            timeout=1,
        )
        await asyncio.wait_for(asr_task, timeout=1)
        assert entered_asr.is_set()
        assert session_limiter.get_limiter().active == 0
        assert bm.tts_manager().status()["inflight_http"] == 0

    try:
        _run_stream_exercise(_exercise, release_backend)
    finally:
        release_backend.set()
        bm._reset_for_tests()


def test_tts_stream_n2_reconnect_during_cleanup_is_429_then_recovers(
    monkeypatch,
):
    """With two HTTP admissions, a quarantined single backend slot is honest.

    The reconnect may consume session #2, but pre-header priming must translate
    backend saturation to 429.  Once background cleanup returns the backend
    slot, the next request streams valid PCM.
    """
    from server import main as appmod
    from server.core import backend_manager as bm, coordinator as coord_mod
    from server.core.worker_io import PoolSaturatedError
    from starlette.requests import Request

    backend_slot = threading.Lock()
    release_first = threading.Event()
    first_stuck = threading.Event()
    first_call = True

    class _OneSlotBackend(_FakeTTSBackend):
        def generate_streaming(self, text, **kwargs):
            nonlocal first_call
            if not backend_slot.acquire(blocking=False):
                raise PoolSaturatedError(1)
            try:
                yield b"\x01\x00" * 8
                if first_call:
                    first_call = False
                    # Signal once the executor has resumed past the first chunk.  The
                    # disconnect watcher closes a generator still suspended at its
                    # yield, which legitimately unwinds the backend at once.
                    first_stuck.set()
                    release_first.wait(timeout=5)
                    yield b"\x02\x00" * 8
            finally:
                backend_slot.release()

    backend = _OneSlotBackend()
    _wire_direct_tts_test_backend(monkeypatch, backend, session_limit=2)
    coord_mod.init_coordinator({"mode": "concurrent"})
    monkeypatch.setenv("OVS_TTS_STREAM_CLEANUP_WAIT_S", "0.02")

    def _request(receive):
        return Request(
            {"type": "http", "method": "POST", "path": "/tts/stream",
             "headers": []},
            receive,
        )

    async def _exercise():
        first_rx: asyncio.Queue[dict] = asyncio.Queue()

        async def _first_receive():
            return await first_rx.get()

        first_response = await appmod.tts_stream(
            appmod.TTSRequest(text="first", language="en"),
            _request(_first_receive),
            None,
        )
        first_body = first_response.body_iterator
        await first_body.__anext__()
        await first_body.__anext__()
        assert await asyncio.to_thread(first_stuck.wait, 1)
        await first_rx.put({"type": "http.disconnect"})
        await first_body.aclose()
        assert appmod._tts_stream_cleanup_tasks

        second_rx: asyncio.Queue[dict] = asyncio.Queue()

        async def _second_receive():
            return await second_rx.get()

        busy = await appmod.tts_stream(
            appmod.TTSRequest(text="reconnect-too-soon", language="en"),
            _request(_second_receive),
            None,
        )
        assert busy.status_code == 429
        assert b"tts_backend_busy" in busy.body

        release_first.set()
        await asyncio.wait_for(
            asyncio.gather(*list(appmod._tts_stream_cleanup_tasks)),
            timeout=1,
        )

        recovery_rx: asyncio.Queue[dict] = asyncio.Queue()

        async def _recovery_receive():
            return await recovery_rx.get()

        recovery = await appmod.tts_stream(
            appmod.TTSRequest(text="recovered", language="en"),
            _request(_recovery_receive),
            None,
        )
        chunks = [chunk async for chunk in recovery.body_iterator]
        assert recovery.status_code == 200
        assert len(chunks) >= 2
        assert b"".join(chunks[1:])

    try:
        _run_stream_exercise(_exercise, release_first)
    finally:
        release_first.set()
        bm._reset_for_tests()


def test_tts_stream_legacy_disconnect_quarantines_until_backend_unwinds(
    monkeypatch,
):
    """The no-manager fallback must retain session/coordinator leases too."""
    from server import main as appmod
    from server.core import backend_manager as bm, coordinator as coord_mod
    from server.core import session_limiter
    from starlette.requests import Request

    release_backend = threading.Event()

    class _StuckLegacyBackend(_FakeTTSBackend):
        def __init__(self):
            super().__init__()
            self.cleaned = threading.Event()
            self.stuck = threading.Event()

        def generate_streaming(self, text, **kwargs):
            try:
                yield b"\x01\x00" * 8
                # Signal once the executor has resumed past the first chunk.  The
                # disconnect watcher closes a generator still suspended at its
                # yield, which legitimately unwinds the backend at once.
                self.stuck.set()
                release_backend.wait(timeout=5)
                yield b"\x02\x00" * 8
            finally:
                self.cleaned.set()

    backend = _StuckLegacyBackend()
    _wire_direct_tts_test_backend(monkeypatch, backend, session_limit=2)
    bm._tts_manager = None
    coordinator = coord_mod.init_coordinator({"mode": "serialized"})
    monkeypatch.setenv("OVS_TTS_STREAM_CLEANUP_WAIT_S", "0.02")

    async def _exercise():
        receive_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def _receive():
            return await receive_queue.get()

        response = await appmod.tts_stream(
            appmod.TTSRequest(text="legacy stuck", language="en"),
            Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/tts/stream",
                    "headers": [],
                },
                _receive,
            ),
            None,
        )
        body = response.body_iterator
        assert await body.__anext__()
        assert await body.__anext__()
        assert await asyncio.to_thread(backend.stuck.wait, 1)
        await receive_queue.put({"type": "http.disconnect"})
        await body.aclose()

        assert appmod._tts_stream_cleanup_tasks
        assert session_limiter.get_limiter().active == 1
        entered_asr = asyncio.Event()

        async def _acquire_asr():
            async with coordinator.acquire("asr"):
                entered_asr.set()

        asr_task = asyncio.create_task(_acquire_asr())
        await asyncio.sleep(0.03)
        assert not entered_asr.is_set()

        release_backend.set()
        await asyncio.wait_for(
            asyncio.gather(*list(appmod._tts_stream_cleanup_tasks)),
            timeout=1,
        )
        await asyncio.wait_for(asr_task, timeout=1)
        assert backend.cleaned.is_set()
        assert session_limiter.get_limiter().active == 0

    try:
        _run_stream_exercise(_exercise, release_backend)
    finally:
        release_backend.set()
        bm._reset_for_tests()


def test_tts_stream_disconnect_drains_both_prefetched_generators(monkeypatch):
    """Disconnect cleanup must retain leases until every prefetched job exits."""
    from server import main as appmod
    from server.core import backend_manager as bm, coordinator as coord_mod
    from starlette.requests import Request

    release_backend = threading.Event()
    both_started = threading.Event()
    both_stuck = threading.Event()
    state_lock = threading.Lock()
    started = 0
    stuck = 0
    cleaned = 0

    class _TwoSentenceBackend(_FakeTTSBackend):
        def generate_streaming(self, text, **kwargs):
            nonlocal started, stuck, cleaned
            with state_lock:
                started += 1
                if started == 2:
                    both_started.set()
            try:
                yield b"\x01\x00" * 8
                # Signal once the executor has resumed past the first chunk.  The
                # disconnect watcher closes a generator still suspended at its
                # yield, which legitimately unwinds the backend at once.
                with state_lock:
                    stuck += 1
                    if stuck == 2:
                        both_stuck.set()
                release_backend.wait(timeout=5)
                yield b"\x02\x00" * 8
            finally:
                with state_lock:
                    cleaned += 1

    backend = _TwoSentenceBackend()
    _wire_direct_tts_test_backend(monkeypatch, backend, session_limit=2)
    coord_mod.init_coordinator({"mode": "concurrent"})
    monkeypatch.setenv("OVS_TTS_STREAM_CLEANUP_WAIT_S", "0.02")
    executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(appmod, "_get_tts_stream_executor", lambda: executor)

    async def _exercise():
        receive_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def _receive():
            return await receive_queue.get()

        response = await appmod.tts_stream(
            appmod.TTSRequest(
                text="First sentence. Second sentence.",
                language="en",
            ),
            Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/tts/stream",
                    "headers": [],
                },
                _receive,
            ),
            None,
        )
        body = response.body_iterator
        assert await body.__anext__()
        assert await body.__anext__()
        assert await asyncio.to_thread(both_started.wait, 1)
        assert await asyncio.to_thread(both_stuck.wait, 1)

        await receive_queue.put({"type": "http.disconnect"})
        await body.aclose()
        assert appmod._tts_stream_cleanup_tasks
        assert bm.tts_manager().status()["inflight_http"] == 1

        release_backend.set()
        await asyncio.wait_for(
            asyncio.gather(*list(appmod._tts_stream_cleanup_tasks)),
            timeout=1,
        )
        assert started == 2
        assert cleaned == 2
        assert bm.tts_manager().status()["inflight_http"] == 0

    try:
        _run_stream_exercise(_exercise, release_backend)
    finally:
        release_backend.set()
        executor.shutdown(wait=True)
        bm._reset_for_tests()


def test_tts_stream_loop_shutdown_keeps_quarantine_until_stuck_workers_exit(
    monkeypatch,
):
    """Loop shutdown with two stuck workers must neither stall nor unquarantine.

    ``asyncio.run`` shutdown cancels every task alive at that moment,
    including the drain child of the first job.  Cleanup must stop draining
    (a drain task for the second job would be outside the cancellation
    snapshot) but must not release leases while the executor workers still
    run: the unfinished jobs are handed to a tracked orphaned-cleanup record
    that keeps the session, BackendManager inflight count and coordinator
    held until both workers exit and close their generators.
    """
    from server import main as appmod
    from server.core import backend_manager as bm, coordinator as coord_mod
    from server.core import session_limiter
    from starlette.requests import Request

    release_backend = threading.Event()
    both_stuck = threading.Event()
    state_lock = threading.Lock()
    stuck = 0
    closed = 0

    class _TwoStuckWorkersBackend(_FakeTTSBackend):
        def generate_streaming(self, text, **kwargs):
            nonlocal stuck, closed
            try:
                yield b"\x01\x00" * 8
                with state_lock:
                    stuck += 1
                    if stuck == 2:
                        both_stuck.set()
                release_backend.wait(timeout=10)
                yield b"\x02\x00" * 8
            finally:
                with state_lock:
                    closed += 1

    backend = _TwoStuckWorkersBackend()
    _wire_direct_tts_test_backend(monkeypatch, backend, session_limit=2)
    coordinator = coord_mod.init_coordinator({"mode": "serialized"})
    monkeypatch.setenv("OVS_TTS_STREAM_CLEANUP_WAIT_S", "0.02")
    executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(appmod, "_get_tts_stream_executor", lambda: executor)

    async def _exercise():
        receive_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def _receive():
            return await receive_queue.get()

        response = await appmod.tts_stream(
            appmod.TTSRequest(
                text="First sentence. Second sentence.", language="en"
            ),
            Request(
                {"type": "http", "method": "POST", "path": "/tts/stream",
                 "headers": []},
                _receive,
            ),
            None,
        )
        body = response.body_iterator
        assert await body.__anext__()
        assert await body.__anext__()
        assert await asyncio.to_thread(both_stuck.wait, 5)
        await receive_queue.put({"type": "http.disconnect"})
        await body.aclose()
        assert appmod._tts_stream_cleanup_tasks
        await _wait_for_task_named("_consume_job")

        # What asyncio.run() shutdown does: cancel every task alive now.
        cleanup_tasks = set(appmod._tts_stream_cleanup_tasks)
        for task in asyncio.all_tasks() - {asyncio.current_task()}:
            task.cancel()
        done, _ = await asyncio.wait(cleanup_tasks, timeout=1)
        assert done == cleanup_tasks, "cleanup stalled after loop shutdown"
        await asyncio.sleep(0)

        # Workers still running: leases retained, nothing untracked.
        assert len(appmod._tts_stream_orphaned_cleanups) == 1
        (record,) = appmod._tts_stream_orphaned_cleanups
        assert len(record.jobs) == 2
        assert not record.workers_finished.is_set()
        assert closed == 0
        assert session_limiter.get_limiter().active == 1
        assert bm.tts_manager().status()["inflight_http"] == 1
        entered_asr = asyncio.Event()

        async def _acquire_asr():
            async with coordinator.acquire("asr"):
                entered_asr.set()

        asr_task = asyncio.create_task(_acquire_asr())
        await asyncio.sleep(0.05)
        assert not entered_asr.is_set(), "coordinator released under live workers"

        # Workers exit: generators close, then the record releases the leases.
        release_backend.set()
        assert await asyncio.to_thread(record.released.wait, 5)
        await asyncio.wait_for(asr_task, timeout=1)
        assert closed == 2
        assert not appmod._tts_stream_orphaned_cleanups
        assert session_limiter.get_limiter().active == 0
        assert bm.tts_manager().status()["inflight_http"] == 0

    try:
        _run_stream_exercise(_exercise, release_backend)
    finally:
        release_backend.set()
        executor.shutdown(wait=True)
        bm._reset_for_tests()


def test_tts_stream_disconnect_before_executor_start_skips_backend(
    monkeypatch,
):
    """Disconnect while queued must not enter the backend generator."""
    from server import main as appmod
    from starlette.requests import Request

    calls = 0

    class _CountingBackend(_FakeTTSBackend):
        def generate_streaming(self, text, **kwargs):
            nonlocal calls
            calls += 1
            yield b"\x01\x00" * 8

    backend = _CountingBackend()
    _wire_direct_tts_test_backend(monkeypatch, backend)

    executor = ThreadPoolExecutor(max_workers=1)
    unblock_executor = threading.Event()
    blocker = executor.submit(unblock_executor.wait)
    monkeypatch.setattr(appmod, "_get_tts_stream_executor", lambda: executor)

    async def _exercise():
        receive_queue: asyncio.Queue[dict] = asyncio.Queue()
        await receive_queue.put({"type": "http.disconnect"})

        async def _receive():
            return await receive_queue.get()

        asyncio.get_running_loop().call_later(0.05, unblock_executor.set)
        response = await appmod.tts_stream(
            appmod.TTSRequest(text="cancel-before-run", language="en"),
            Request(
                {"type": "http", "method": "POST", "path": "/tts/stream",
                 "headers": []},
                _receive,
            ),
            None,
        )
        # The peer is already gone; importantly, we do not fabricate a
        # successful rate-only response for a stream that emitted no PCM.
        assert response.status_code == 503
        assert calls == 0

    try:
        _run_stream_exercise(_exercise, unblock_executor)
    finally:
        unblock_executor.set()
        blocker.result(timeout=1)
        executor.shutdown(wait=True)
        from server.core import backend_manager as bm
        bm._reset_for_tests()


def test_health_reads_cancel_counter_from_active_backend_worker_io(client):
    """The voxedge backend owns a different WorkerIO class than server.core."""

    class _ActiveWorkerIO:
        _cancel_count = 7
        _cancel_count_lock = threading.Lock()

    client.tts_be._worker_io = _ActiveWorkerIO()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["tts_worker_cancel_count"] == 7


def test_tts_clone_uses_manager_acquire(client):
    """FIX_1: /tts/clone must bump inflight_http via mgr.acquire()."""
    import base64
    observed = _observe_inflight(client)
    payload = {
        "text": "hi",
        "speaker_embedding_b64": base64.b64encode(b"\x00" * 16).decode(),
    }
    r = client.post("/tts/clone", json=payload)
    assert r.status_code == 200, r.text
    assert observed, "clone_voice was never called"
    assert all(n >= 1 for n in observed), observed
    assert client.tts_be.clone_calls


def test_tts_clone_stream_uses_manager_acquire(client):
    """FIX_1: /tts/clone/stream must bump inflight_http via mgr.acquire()."""
    import base64
    observed = _observe_inflight(client)
    payload = {
        "text": "hi",
        "speaker_embedding_b64": base64.b64encode(b"\x00" * 16).decode(),
    }
    r = client.post("/tts/clone/stream", json=payload)
    assert r.status_code == 200, r.text
    _ = r.content
    assert observed
    assert all(n >= 1 for n in observed), observed
    assert client.tts_be.streaming_calls


def test_runtime_speed_override_applied_to_tts(client):
    """FIX_2: PATCH /admin/tts/runtime speed=1.5 → backend sees speed=1.5."""
    r = client.patch("/admin/tts/runtime", json={"speed": 1.5})
    assert r.status_code == 200, r.text
    client.tts_be.synthesize_calls.clear()
    r = client.post("/tts", json={"text": "hi"})
    assert r.status_code == 200, r.text
    call = client.tts_be.synthesize_calls[-1]
    assert call.get("speed") == 1.5, f"expected speed=1.5 in {call}"


def test_runtime_pitch_override_applied_to_tts(client):
    """FIX_2: PATCH /admin/tts/runtime pitch_shift=3 → backend sees pitch_shift=3."""
    r = client.patch("/admin/tts/runtime", json={"pitch_shift": 3.0})
    assert r.status_code == 200, r.text
    client.tts_be.synthesize_calls.clear()
    r = client.post("/tts", json={"text": "hi"})
    assert r.status_code == 200, r.text
    call = client.tts_be.synthesize_calls[-1]
    assert call.get("pitch_shift") == 3.0, f"expected pitch_shift=3.0 in {call}"


def test_request_speed_overrides_runtime(client):
    """FIX_2: request payload beats runtime override (speed=2.0 > runtime 1.5)."""
    r = client.patch("/admin/tts/runtime", json={"speed": 1.5})
    assert r.status_code == 200, r.text
    client.tts_be.synthesize_calls.clear()
    r = client.post("/tts", json={"text": "hi", "speed": 2.0})
    assert r.status_code == 200, r.text
    call = client.tts_be.synthesize_calls[-1]
    assert call.get("speed") == 2.0, f"expected request speed=2.0 to win, got {call}"


def test_lazy_tts_first_request_starts_manager(monkeypatch):
    """FIX_3: LAZY-style startup leaves manager in INIT → first /tts drives it READY."""
    from server.core import backend_manager as bm, coordinator as coord_mod, tts_runtime, tts_service

    tts_runtime.reset_overrides()
    bm._reset_for_tests()
    coord_mod._coordinator = None  # type: ignore[attr-defined]
    coord_mod.init_coordinator({"mode": "concurrent"})
    # Stale-test fix: /tts admission now requires a session limiter (see
    # _install_managers comment). Init it so the request reaches the lazy-start
    # path instead of short-circuiting to 503 session_limiter_unavailable.
    from server.core import session_limiter as sl_mod
    sl_mod._reset_for_tests()
    sl_mod.init_limiter({"max_concurrent_sessions": 64})

    tts_be = _FakeTTSBackend()
    asr_be = _FakeASRBackend()
    bm.init_backend_managers(
        tts_factory=lambda: tts_be,
        tts_preloader=lambda b: b.preload(),
        tts_unloader=lambda b: b.unload(),
        asr_factory=lambda: asr_be,
        asr_preloader=lambda b: b.preload(),
        asr_unloader=lambda b: b.unload(),
    )

    # IMPORTANT: do NOT call mgr.start() — simulate LAZY_TTS skipping startup preload.
    assert bm.tts_manager().state.value == "init"
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: tts_be)

    # Reset lazy-start lock between tests in the same process.
    import server.main as _main_mod
    _main_mod._tts_lazy_start_lock = None

    from server.main import app
    from server.core.admin_auth import require_admin
    app.dependency_overrides[require_admin] = lambda: None

    try:
        c = TestClient(app)
        r = c.post("/tts", json={"text": "hi"})
        assert r.status_code == 200, r.text
        assert bm.tts_manager().state.value == "ready"
        assert tts_be._ready is True
        assert tts_be.synthesize_calls
    finally:
        app.dependency_overrides.pop(require_admin, None)
        bm._reset_for_tests()
        tts_runtime.reset_overrides()


def test_lazy_tts_exclusive_evicts_asr_before_preload():
    """Exclusive lazy TTS must free ASR residency before loading TTS."""
    from server.core import backend_manager as bm, coordinator as coord_mod
    import server.main as main_mod

    events: list[str] = []
    bm._reset_for_tests()
    coord_mod._coordinator = None  # type: ignore[attr-defined]

    asr_be = _FakeASRBackend()
    asr_be.preload()
    original_asr_unload = asr_be.unload

    def asr_unload():
        events.append("unload-asr")
        original_asr_unload()

    asr_be.unload = asr_unload  # type: ignore[method-assign]
    tts_be = _FakeTTSBackend()

    def tts_factory():
        events.append("factory-tts")
        return tts_be

    def tts_preload(backend):
        events.append("preload-tts")
        backend.preload()

    bm.init_backend_managers(
        tts_factory=tts_factory,
        tts_preloader=tts_preload,
        tts_unloader=lambda b: b.unload(),
        asr_factory=lambda: asr_be,
        asr_preloader=lambda b: b.preload(),
        asr_unloader=lambda b: b.unload(),
    )
    coordinator = coord_mod.init_coordinator({"mode": "exclusive"})
    coordinator.register_backend("asr", lambda: asr_be)
    # This mirrors LAZY_TTS startup: no TTS singleton exists yet.
    coordinator.register_backend("tts", lambda: None)
    main_mod._tts_lazy_start_lock = None

    try:
        manager = asyncio.run(main_mod._ensure_tts_manager_started())
        assert manager is bm.tts_manager()
        assert events == ["unload-asr", "factory-tts", "preload-tts"]
        assert not asr_be.is_ready()
        assert tts_be.is_ready()
    finally:
        bm._reset_for_tests()


# ---------------------------------------------------------------------------
# FIX_3_completion: FAILED / start-fail manager must NOT silently fall back
# to legacy tts_service.synthesize. Operators need a real 503.
# ---------------------------------------------------------------------------

def test_failed_manager_returns_503_not_fallback(monkeypatch):
    """Manager in FAILED state → /tts gets 503, legacy tts_service is NOT used."""
    from server.core import backend_manager as bm, coordinator as coord_mod, tts_runtime, tts_service

    tts_runtime.reset_overrides()
    bm._reset_for_tests()
    coord_mod._coordinator = None  # type: ignore[attr-defined]
    coord_mod.init_coordinator({"mode": "concurrent"})
    # Stale-test fix: init session limiter so /tts reaches the manager-failed
    # branch instead of short-circuiting to 503 session_limiter_unavailable.
    from server.core import session_limiter as sl_mod
    sl_mod._reset_for_tests()
    sl_mod.init_limiter({"max_concurrent_sessions": 64})

    # TTS factory that always fails → start() flips state to FAILED.
    def _bad_factory():
        raise RuntimeError("boom-tts")

    asr_be = _FakeASRBackend()
    bm.init_backend_managers(
        tts_factory=_bad_factory,
        tts_preloader=lambda b: None,
        tts_unloader=lambda b: None,
        asr_factory=lambda: asr_be,
        asr_preloader=lambda b: b.preload(),
        asr_unloader=lambda b: b.unload(),
    )

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError):
            loop.run_until_complete(bm.tts_manager().start())
    finally:
        loop.close()
    assert bm.tts_manager().state.value == "failed"

    # Sentinel: if endpoint silently fell back, it would call tts_service.synthesize.
    legacy_called = {"n": 0}

    def _legacy_synth(*args, **kwargs):
        legacy_called["n"] += 1
        return b"\x00\x00" * 8, {"duration": 0.001, "inference_time": 0.001, "rtf": 1.0}

    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "is_configured", lambda: True)
    monkeypatch.setattr(tts_service, "synthesize", _legacy_synth)
    monkeypatch.setattr(tts_service, "_backend", _FakeTTSBackend(), raising=False)

    import server.main as _main_mod
    _main_mod._tts_lazy_start_lock = None

    from server.main import app
    from server.core.admin_auth import require_admin
    app.dependency_overrides[require_admin] = lambda: None
    try:
        c = TestClient(app)
        r = c.post("/tts", json={"text": "hi"})
        assert r.status_code == 503, r.text
        body = r.json()
        # FastAPI wraps HTTPException.detail under "detail".
        detail = body.get("detail", body)
        assert isinstance(detail, dict)
        assert detail.get("error") == "tts_manager_failed", detail
        assert detail.get("state") == "failed", detail
        # The legacy path must NOT have been reached.
        assert legacy_called["n"] == 0, "FAILED manager must not silently fall back to tts_service"
    finally:
        app.dependency_overrides.pop(require_admin, None)
        bm._reset_for_tests()
        tts_runtime.reset_overrides()


def test_manager_start_failure_first_request_raises_503(monkeypatch):
    """Manager INIT + factory fails on lazy start → /tts gets 503 (not 200 via legacy)."""
    from server.core import backend_manager as bm, coordinator as coord_mod, tts_runtime, tts_service

    tts_runtime.reset_overrides()
    bm._reset_for_tests()
    coord_mod._coordinator = None  # type: ignore[attr-defined]
    coord_mod.init_coordinator({"mode": "concurrent"})
    # Stale-test fix: init session limiter so /tts reaches the lazy-start
    # failure branch instead of short-circuiting to session_limiter_unavailable.
    from server.core import session_limiter as sl_mod
    sl_mod._reset_for_tests()
    sl_mod.init_limiter({"max_concurrent_sessions": 64})

    def _bad_factory():
        raise RuntimeError("preload-blew-up")

    asr_be = _FakeASRBackend()
    bm.init_backend_managers(
        tts_factory=_bad_factory,
        tts_preloader=lambda b: None,
        tts_unloader=lambda b: None,
        asr_factory=lambda: asr_be,
        asr_preloader=lambda b: b.preload(),
        asr_unloader=lambda b: b.unload(),
    )

    # IMPORTANT: do NOT call start() — keep manager in INIT so the endpoint
    # triggers lazy start, which will fail.
    assert bm.tts_manager().state.value == "init"

    legacy_called = {"n": 0}

    def _legacy_synth(*args, **kwargs):
        legacy_called["n"] += 1
        return b"\x00\x00" * 8, {"duration": 0.001, "inference_time": 0.001, "rtf": 1.0}

    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "is_configured", lambda: True)
    monkeypatch.setattr(tts_service, "synthesize", _legacy_synth)
    monkeypatch.setattr(tts_service, "_backend", _FakeTTSBackend(), raising=False)

    import server.main as _main_mod
    _main_mod._tts_lazy_start_lock = None

    from server.main import app
    from server.core.admin_auth import require_admin
    app.dependency_overrides[require_admin] = lambda: None
    try:
        c = TestClient(app)
        r = c.post("/tts", json={"text": "hi"})
        assert r.status_code == 503, r.text
        body = r.json()
        detail = body.get("detail", body)
        assert isinstance(detail, dict)
        assert detail.get("error") == "tts_manager_start_failed", detail
        # After start() failed, manager should be FAILED.
        assert bm.tts_manager().state.value == "failed"
        assert legacy_called["n"] == 0, "start() failure must not silently fall back to tts_service"
    finally:
        app.dependency_overrides.pop(require_admin, None)
        bm._reset_for_tests()
        tts_runtime.reset_overrides()
