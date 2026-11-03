"""Transport-neutral execution primitives for HTTP speech APIs.

The service historically performed backend selection, coordinator admission and
the actual decode directly in each FastAPI handler.  Versioned APIs must not
duplicate those ownership rules, otherwise a legacy and a v1 request can race
through different manager paths.  This module intentionally knows nothing
about FastAPI request/response objects: callers supply the already-resolved
request values and serialize the typed result themselves.

The manager arguments are deliberately duck-typed.  In production they are
``BackendManager`` instances; tests can inject tiny fakes without importing the
heavy backend implementations.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSExecutionResult:
    """Audio bytes and backend metadata returned by one synthesis call."""

    audio: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)
    backend: str | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class ASRExecutionResult:
    """Transcription text and metadata returned by one decode call."""

    text: str
    language: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    backend: str | None = None


class APIExecutionError(RuntimeError):
    """A transport-neutral domain error for a speech execution operation."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "execution_error",
        param: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.param = param
        self.headers = dict(headers or {})


class BackendNotReadyError(APIExecutionError):
    def __init__(self, kind: str, detail: str | None = None) -> None:
        message = f"{kind.upper()} backend not available"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(
            message,
            status_code=503,
            code="backend_not_ready",
        )


class BackendBusyError(APIExecutionError):
    def __init__(self, max_slots: int | None = None) -> None:
        message = "backend is busy"
        if max_slots is not None:
            message = f"{message}; maximum concurrent slots: {max_slots}"
        super().__init__(
            message,
            status_code=429,
            code="backend_busy",
            headers={"Retry-After": "1"},
        )
        self.max_slots = max_slots


class _TransportDisconnected(BaseException):
    """Internal control flow for a client that closed a finite HTTP request."""


KOKORO_BACKEND_NAMES: frozenset[str] = frozenset({
    "rk:kokoro_convonly",
    "kokoro_convonly",
    "rk:kokoro_dualbucket",
    "kokoro_dualbucket",
})


def _is_kokoro_convonly_cancelled(exc: BaseException) -> bool:
    """Match the backend cooperative-cancel sentinel by exact class identity."""
    module_name = "rkvoice_stream.backends.tts.kokoro_convonly"
    cls = type(exc)
    if cls.__module__ != module_name or cls.__name__ != "ConvOnlyCancelled":
        return False
    module = sys.modules.get(module_name)
    return module is not None and getattr(module, "ConvOnlyCancelled", None) is cls


def _pool_saturated(exc: BaseException) -> bool:
    return bool(
        getattr(exc, "status", None) == 4429
        or type(exc).__name__ == "PoolSaturatedError"
    )


def _backend_name(backend: object) -> str | None:
    value = getattr(backend, "name", None)
    return str(value) if value is not None else None


async def _execute_kokoro_call(call, cancel_event, disconnect_awaitable):
    """Own native work and receive children until both have actually stopped."""
    import anyio

    async def own_work():
        future = asyncio.get_running_loop().run_in_executor(
            None, contextvars.copy_context().run, call,
        )
        watcher_scope = anyio.CancelScope()
        watcher = None

        async def watch_disconnect():
            # Level cancellation reaches receive()'s nested AnyIO task groups;
            # a bare Task.cancel() only delivers one edge to their parent.
            with watcher_scope:
                value = disconnect_awaitable() if callable(disconnect_awaitable) else disconnect_awaitable
                value = await value
                if isinstance(value, Mapping):
                    return value.get("type") == "http.disconnect"
                return bool(value)
            return False

        try:
            if disconnect_awaitable is not None:
                watcher = asyncio.create_task(watch_disconnect(), name="finite-tts-disconnect")
                done, _ = await asyncio.wait((future, watcher), return_when=asyncio.FIRST_COMPLETED)
                # Native completion wins a same-turn race.  Its real result or
                # exception must not be replaced by a concurrently observed
                # transport disconnect.
                if future in done:
                    return await future
                if watcher in done and watcher.result():
                    cancel_event.set()
                    try:
                        await future
                    except BaseException as exc:
                        if not (
                            cancel_event.is_set()
                            and _is_kokoro_convonly_cancelled(exc)
                        ):
                            logger.error(
                                "finite TTS backend failed after client disconnect",
                                exc_info=(type(exc), exc, exc.__traceback__),
                            )
                    raise _TransportDisconnected()
            return await future
        except BaseException:
            cancel_event.set()
            raise
        finally:
            # This owner is independent from request cancellation. Always join
            # both resources, including watcher errors and native failures.
            if watcher is not None:
                watcher_scope.cancel()
                try:
                    await watcher
                except BaseException:
                    pass
            try:
                await future
            except BaseException:
                pass

    owner = asyncio.create_task(own_work(), name="finite-tts-native-owner")
    try:
        # Polling retains ownership without propagating cancellation into the
        # child.  The timer also guarantees an event-loop wake after executor
        # completion on platforms whose quiet-loop self-pipe wake can lag.
        while not owner.done():
            await asyncio.sleep(0.01)
        return owner.result()
    except asyncio.CancelledError:
        cancel_event.set()
        # AnyIO middleware cancels repeatedly at checkpoints.  Shield that
        # scope, and tolerate repeated explicit asyncio Task.cancel() calls,
        # without ever detaching the owner or freeing its backend leases.
        with anyio.CancelScope(shield=True):
            while not owner.done():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    continue
            try:
                owner.result()
            except BaseException:
                pass
        raise


async def execute_tts(
    *,
    text: str,
    language: str | None,
    voice_kwargs: Mapping[str, Any],
    manager: Any | None,
    legacy_service: Any | None,
    coordinator: Any,
    prepare: Callable[[Any], Mapping[str, Any]] | None = None,
    cancel_event: Any | None = None,
    disconnect_awaitable: Any | None = None,
) -> TTSExecutionResult:
    """Execute one non-streaming TTS request under shared ownership.

    ``manager`` is the ready ``BackendManager`` returned by the caller's
    startup routine.  If no manager exists (the ASR-only/legacy deployment),
    ``legacy_service`` supplies the already-ready singleton backend.  Both
    paths acquire the same coordinator lease; manager-backed requests also
    hold ``manager.acquire()`` for the whole synchronous synthesis operation.
    """

    async def _call(backend: Any, synthesize) -> TTSExecutionResult:
        kwargs = dict(prepare(backend) if prepare is not None else voice_kwargs)
        is_kokoro = _backend_name(backend) in KOKORO_BACKEND_NAMES
        active_cancel_event = cancel_event
        if is_kokoro and active_cancel_event is None:
            active_cancel_event = threading.Event()
        if is_kokoro and active_cancel_event is not None:
            # The adapter accepts this transport-neutral thread Event and
            # forwards it to rkvoice-stream.  Other backends never see it.
            kwargs.setdefault("cancel_event", active_cancel_event)
        async with coordinator.acquire("tts"):
            try:
                call = functools.partial(synthesize, text=text, language=language, **kwargs)
                if is_kokoro:
                    audio, metadata = await _execute_kokoro_call(
                        call, active_cancel_event, disconnect_awaitable,
                    )
                else:
                    audio, metadata = call()
            except BaseException:
                # Keep the original backend exception for legacy callers.  A
                # versioned serializer can map the duck-typed saturation
                # exception to ``BackendBusyError`` without changing the
                # historical /tts traceback/status surface.
                raise
        return TTSExecutionResult(
            audio=audio,
            metadata=metadata or {},
            backend=_backend_name(backend),
            sample_rate=getattr(backend, "sample_rate", None),
        )

    if manager is not None:
        async with manager.acquire() as backend:
            return await _call(backend, backend.synthesize)

    backend = None
    if legacy_service is not None:
        try:
            if legacy_service.is_ready():
                backend = legacy_service.get_backend()
        except Exception:
            backend = None
    if backend is None:
        raise BackendNotReadyError("tts")
    return await _call(backend, legacy_service.synthesize)


async def execute_tts_clone(
    *,
    text: str,
    language: str | None,
    clone_kwargs: Mapping[str, Any],
    manager: Any | None,
    legacy_service: Any | None,
    coordinator: Any,
    prepare: Callable[[Any], Mapping[str, Any]] | None = None,
) -> TTSExecutionResult:
    """Execute one voice-clone synthesis under the normal ownership rules.

    Clone backends intentionally expose a separate ``clone_voice`` method
    from ordinary ``synthesize``.  Keeping this tiny sibling of
    :func:`execute_tts` lets the native clone routes share the manager,
    coordinator and cancellation-safe admission paths without making the
    legacy serializer parse a FastAPI response.
    """

    async def _call(backend: Any) -> TTSExecutionResult:
        kwargs = dict(prepare(backend) if prepare is not None else clone_kwargs)
        async with coordinator.acquire("tts"):
            audio, metadata = backend.clone_voice(
                text=text,
                language=language,
                **kwargs,
            )
        return TTSExecutionResult(
            audio=audio,
            metadata=metadata or {},
            backend=_backend_name(backend),
            sample_rate=getattr(backend, "sample_rate", None),
        )

    if manager is not None:
        async with manager.acquire() as backend:
            return await _call(backend)

    if legacy_service is not None:
        try:
            if legacy_service.is_ready():
                backend = legacy_service.get_backend()
            else:
                backend = None
        except Exception:
            backend = None
        if backend is not None:
            return await _call(backend)
    raise BackendNotReadyError("tts")


async def _decode(backend: Any, audio: bytes, language: str) -> Any:
    """Decode one clip off the event loop, segmenting it when it is too long.

    Two things happen here rather than in each transport, because this is the
    one place every non-streaming path (native /asr, /v1/asr, the OpenAI
    adapter) funnels through:

    * ``backend.transcribe()`` is blocking. Called inline it freezes the event
      loop for the whole decode, which for an offline clip is seconds — long
      enough to starve /readyz and /metrics and have the container healthcheck
      (interval 30s / timeout 5s / retries 3) flag it mid-request.
    * Fixed-shape engines silently drop audio past their input limit.
      SenseVoice TRT/RKNN takes 344 LFR frames, about 20.4 s, and simply
      truncates the rest with no error and no log line: measured on device, a
      26.57 s clip and a 21.21 s clip returned byte-identical text. The
      segmenter splits over-long clips at VAD silence, decodes each piece and
      rejoins; ``OVS_ASR_AUTO_SEGMENT=0`` restores the single-pass behaviour.

    Backends that already chunk internally are left alone by the segmenter --
    paraformer carries CIF state across its own chunks, so an outer split would
    reset exactly the continuity it maintains.
    """
    loop = asyncio.get_running_loop()

    def _run() -> Any:
        from server.core import asr_segmenter as _seg
        try:
            segmented = _seg.maybe_transcribe_segmented(
                backend, audio, language=language
            )
        except Exception:
            logger.exception(
                "asr auto-segmentation failed; falling back to single-pass decode"
            )
            segmented = None
        if segmented is not None:
            return _SegmentedResult(segmented)
        return backend.transcribe(audio, language=language)

    return await loop.run_in_executor(None, _run)


class _SegmentedResult:
    """Adapts a SegmentedTranscription to the transcribe() result shape."""

    __slots__ = ("text", "language", "meta")

    def __init__(self, segmented: Any) -> None:
        self.text = segmented.text
        self.language = segmented.language
        self.meta = segmented.as_meta()


async def execute_asr(
    *,
    audio: bytes,
    language: str,
    manager: Any | None,
    legacy_backend: Any | None,
    coordinator: Any,
    metrics_module: Any | None = None,
    prepare: Callable[[Any], None] | None = None,
) -> ASRExecutionResult:
    """Execute one non-streaming ASR request under shared ownership."""

    async def _call(backend: Any) -> ASRExecutionResult:
        if prepare is not None:
            prepare(backend)
        async with coordinator.acquire("asr"):
            # Preserve the legacy metric: decode time excludes coordinator
            # queueing and begins only once the backend may execute.
            started = time.perf_counter()
            try:
                result = await _decode(backend, audio, language)
            except BaseException:
                raise
        if metrics_module is not None:
            try:
                metrics_module.record_asr_decode_duration(
                    _backend_name(backend) or "asr",
                    time.perf_counter() - started,
                )
            except Exception:
                pass
        return ASRExecutionResult(
            text=result.text,
            language=result.language,
            metadata=result.meta or {},
            backend=_backend_name(backend),
        )

    if manager is not None:
        async with manager.acquire() as backend:
            return await _call(backend)

    if legacy_backend is None:
        raise BackendNotReadyError("asr")
    try:
        if not legacy_backend.is_ready():
            raise BackendNotReadyError("asr")
    except APIExecutionError:
        raise
    except Exception:
        raise BackendNotReadyError("asr")
    return await _call(legacy_backend)


async def read_bounded_upload(upload: Any, *, max_bytes: int, chunk_size: int = 1024 * 1024) -> bytes:
    """Read an UploadFile-like object while enforcing decoded-byte bounds.

    The helper deliberately ignores ``Content-Length`` and counts bytes
    returned by the decoded upload stream.  It works with Starlette's async
    ``UploadFile`` and simple test doubles implementing ``read(size)``.
    """

    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise APIExecutionError(
                "uploaded body is not binary",
                status_code=400,
                code="invalid_audio",
                param="file",
            )
        data = bytes(chunk)
        total += len(data)
        if total > max_bytes:
            raise APIExecutionError(
                f"uploaded audio exceeds the {max_bytes} byte limit",
                status_code=413,
                code="payload_too_large",
                param="file",
            )
        chunks.append(data)
    return b"".join(chunks)
