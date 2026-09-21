"""FastAPI speech service: ASR + TTS with pluggable backends."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse, StreamingResponse
from server.core.api_execution import (
    _TransportDisconnected,
    _is_kokoro_convonly_cancelled,
)
from pydantic import BaseModel
try:  # pydantic v2; the fallback keeps source tools on v1 importable
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment,misc]
class _WSHandle:
    """Lightweight WS-session handle for BackendManager.register_ws().

    Replaces ``types.SimpleNamespace`` here because Python 3.10's
    SimpleNamespace lacks ``__weakref__`` (added in 3.11), and
    BackendManager._ws_handles is a WeakSet. The Jetson image still
    ships Python 3.10.12, so any handle stored in a WeakSet must be a
    plain class.
    """
    __slots__ = ("websocket", "task", "__weakref__")

    def __init__(self, websocket, task):
        self.websocket = websocket
        self.task = task


# /v2v admission-time eviction (limit=1 single-client deployments, e.g.
# voice-arm): track live /v2v WS holders so a fresh connection can reclaim a
# slot leaked by a zombie peer — app-layer reader dead but the websockets
# protocol layer still auto-ponging, so uvicorn ping/pong never times it out
# and the held slot never releases. See _v2v_evict_and_reacquire. Plain set
# (not WeakSet): every holder is removed on _v2v_release_early's
# finally-guaranteed path.
_V2V_HOLDERS: "set" = set()
_v2v_evict_lock = None  # lazily created asyncio.Lock (no top-level asyncio import)


def _get_v2v_evict_lock():
    # Lazy init is atomic across coroutines: no await between check and set.
    global _v2v_evict_lock
    if _v2v_evict_lock is None:
        import asyncio as _asyncio
        _v2v_evict_lock = _asyncio.Lock()
    return _v2v_evict_lock


def _v2v_evict_enabled() -> bool:
    """Admission-time eviction is opt-in and only safe for limit==1.

    Gated on ``OVS_V2V_EVICT_ON_FULL``. Only enabled when the resolved
    session limit is 1 (single conversant): then a newcomer arriving against
    a full slot can only be that same client reconnecting after abandoning a
    dead session, so evicting the holder is always correct. For limit>=2
    (multi-tenant) a newcomer cannot prove the holder is stale → stays off.
    """
    from server.core.session_limiter import get_limiter
    if not _env_truthy(os.environ.get("OVS_V2V_EVICT_ON_FULL")):
        return False
    lim = get_limiter()
    return lim is not None and lim.limit == 1


async def _v2v_evict_and_reacquire(endpoint: str, *, timeout_s: float = 2.0):
    """Evict stale /v2v holder(s) and retry admission once → ``(token, info)``.

    Serialised by a module lock so two simultaneous newcomers don't both
    evict. Closes each holder's WS (1012) and cancels its handler task; the
    holder's own outer finally (``_v2v_release_early``) releases the slot,
    which we then reclaim via a bounded poll. On timeout returns
    ``(None, info)`` and the caller renders a normal 4429.
    """
    import asyncio as _asyncio
    from server.core.session_limiter import try_acquire_ws_token

    async with _get_v2v_evict_lock():
        # Re-check under the lock: a prior waiter may have already freed the
        # slot, in which case we just take it without evicting anyone.
        token, info = try_acquire_ws_token(endpoint)
        if token is not None:
            return token, info
        holders = list(_V2V_HOLDERS)
        if not holders:
            # Slot held by something we don't own (e.g. an in-flight /tts
            # HTTP request sharing the limiter) — not ours to evict.
            return None, info
        logger.warning(
            "v2v admission full; evicting %d stale holder(s) (limit=1, endpoint=%s)",
            len(holders), endpoint,
        )
        for h in holders:
            _ws = getattr(h, "websocket", None)
            if _ws is not None:
                try:
                    res = _ws.close(code=1012, reason="evicted: superseded by new session")
                    if _asyncio.iscoroutine(res):
                        await res
                except Exception:
                    logger.debug("v2v evict: ws.close raised", exc_info=True)
            _task = getattr(h, "task", None)
            if _task is not None and not _task.done():
                _task.cancel()
        # Bounded wait for the evicted holder's finally to release the slot.
        steps = max(1, int(timeout_s / 0.05))
        for _ in range(steps):
            await _asyncio.sleep(0.05)
            token, info = try_acquire_ws_token(endpoint)
            if token is not None:
                return token, info
        logger.warning(
            "v2v evict: slot not released within %.1fs; rejecting newcomer", timeout_s,
        )
        return None, info


from typing import Literal, Optional

# Week 2: configure logging (JSON or text) from OVS_LOG_FORMAT before
# any other module emits a startup log. Falls back gracefully to the
# legacy text format if the env var is unset/invalid.
from server.core.logging_config import (  # noqa: E402  (must precede app creation)
    setup_logging,
    set_request_context,
    reset_request_context,
    request_id_from_headers,
    generate_request_id,
    mask_url_query,
)

setup_logging()
logger = logging.getLogger(__name__)


def _keep_asr_max_s() -> float:
    """Longest in-flight ASR utterance an ``abort(keep_asr)`` may keep.

    ``OVS_V2V_ABORT_KEEP_ASR_MAX_S`` (default 5.0 s). Non-finite or negative
    values fall back to the default: ``float("nan") <= 0`` is False, so a
    plain comparison would let ``nan`` through and silently disable the bound.
    """
    import math

    raw = os.getenv("OVS_V2V_ABORT_KEEP_ASR_MAX_S", "")
    try:
        value = float(raw) if raw.strip() else 5.0
    except ValueError:
        return 5.0
    if not math.isfinite(value) or value < 0:
        return 5.0
    return value


app = FastAPI(title="Jetson Speech Service", version="2.0.0")


# Week 2: HTTP middleware injects/propagates X-Request-ID and stores it
# in the request_id contextvar so every log line from the handler can
# include it. Never reads request body. Probes (/livez /readyz /health
# /metrics) are NOT skipped because we still want the response header.
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    inbound = request_id_from_headers(request.headers)
    request_id = inbound or generate_request_id()
    tokens = set_request_context(request_id=request_id)
    try:
        try:
            response = await call_next(request)
        except Exception:
            # Make sure the request_id is visible in the exception log
            # before we propagate so operators can correlate.
            logger.exception(
                "unhandled exception in request: %s",
                mask_url_query(str(request.url)),
            )
            raise
        # Add the response header. Streaming responses are passed through
        # unchanged; the generator captures its own context.
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        reset_request_context(tokens)


# Week 1 production hardening: optional API-key auth for public voice
# endpoints. Disabled when OVS_API_KEYS is unset/empty. See
# docs/archive/specs/prod-hardening-week1.md Deliverable 1.
def _require_api_key(request: Request) -> None:
    from server.core.api_auth import check_http
    check_http(request)


class TTSRequest(BaseModel):
    text: str
    sid: int | None = None
    speaker_id: int | None = None
    speaker_embedding_b64: str | None = None
    speed: float | None = None
    pitch: float | None = None
    language: str | None = None
    # Named voice selector (string). For SparkTTS this routes to a registered clone
    # VoiceProfile (voice_id, e.g. "clone:alice") when it hits the backend's voice
    # registry; otherwise the backend may interpret it as a controllable style spec.
    voice: str | None = None


class CloneRequest(BaseModel):
    text: str
    speaker_embedding_b64: str  # base64-encoded speaker embedding
    language: str | None = None


class CloneStreamRequest(BaseModel):
    text: str
    speaker_embedding_b64: str  # base64-encoded speaker embedding
    language: str | None = None
    streaming_profile: str | None = None
    first_chunk_frames: int | None = None
    chunk_frames: int | None = None


_asr_backend = None
_rk_profile_status: dict | None = None


def _current_rk_profile_status() -> dict:
    """Return the bounded RK profile contract status for observability."""
    try:
        from server.core.profile_loader import current_profile
        from server.core.rk_profile_contract import runtime_status

        return runtime_status(current_profile() or {}, os.environ)
    except Exception as exc:  # pragma: no cover - diagnostics must not crash probes
        logger.warning("RK profile contract status unavailable: %s", exc)
        return {
            "required": False,
            "profile": None,
            "device": None,
            "contract": "unavailable",
            "verified": False,
            "settings": {},
            "missing_profile": [],
            "missing_runtime": [],
            "mismatches": {},
        }

# Dedicated single-thread executor for streaming TTS (T3 fix).
# Default asyncio executor spawns multiple worker threads; each new thread
# observes a cold CUDA per-thread context for the C++ TRT engine, which
# inflates streaming prefill from ~16ms (warm) to 33-122ms (cold) under
# any concurrency. Pinning streaming TTS to a single worker keeps the
# CUDA context warm across all requests.
_tts_stream_executor: ThreadPoolExecutor | None = None

# Dedicated single-thread executor for streaming ASR.  Without this,
# concurrent WS connections dispatch ASR work to separate IO threads,
# each racing on _ASR_CUDA_STREAM (process-global singleton) leading to
# CUDA Graph capture failures.  One worker serialises all ASR ops on a
# consistent thread with a warm CUDA context.
_asr_executor: ThreadPoolExecutor | None = None


class _VoiceCloneUnsupportedError(Exception):
    """Raised by ``_request_voice_kwargs`` when the request carries a
    ``speaker_embedding_b64`` but the active backend explicitly disables
    voice cloning. Callers translate this into a 400 JSON response with
    the unified capability payload (Bug 3 fix — /tts and /tts/stream).
    """

    def __init__(self, backend) -> None:  # type: ignore[no-untyped-def]
        self.backend = backend
        backend_name = getattr(backend, "name", "tts")
        super().__init__(
            f"Current TTS backend ({backend_name}) does not support voice cloning"
        )


def _voice_clone_unsupported_payload(backend) -> dict:  # type: ignore[no-untyped-def]
    backend_name = getattr(backend, "name", "tts")
    msg = (
        f"Current TTS backend ({backend_name}) does not support voice "
        "cloning. Use a built-in speaker_id via /tts, or switch to a "
        "clone-capable backend."
    )
    return {
        "error": msg,
        "required_capability": "voice_clone",
        "backend": backend_name,
        "supports_voice_cloning": False,
    }


def _request_voice_kwargs(req: TTSRequest, *, backend=None) -> dict:
    """Resolve TTS kwargs for one synth call.

    Mixes (in priority order):
      request payload > runtime overrides > speaker-table default

    Returns a dict combining the backend-specific speaker kwargs (from
    :func:`speaker_kwargs_for_id`) plus ``speed`` / ``pitch_shift`` when the
    merge (request payload + runtime overrides) yields a value. Callers
    should ``**``-spread the result into ``synthesize`` / ``generate_streaming``
    and must NOT additionally pass ``speed`` / ``pitch_shift`` from the raw
    request, or runtime overrides will be silently discarded (FIX_2).

    ``backend`` is the live TTS backend (from BackendManager.acquire()); when
    omitted we fall back to ``tts_service`` / env so the helper still works
    if called outside an acquire() scope.
    """
    from server.core.tts_speakers import speaker_kwargs_for_id
    from server.core.tts_runtime import merge_tts_request_kwargs

    speaker_id = req.speaker_id if req.speaker_id is not None else req.sid
    if speaker_id is not None and req.speaker_embedding_b64:
        raise ValueError("speaker_id and speaker_embedding_b64 cannot be used together")
    if req.speaker_embedding_b64:
        # Bug 3 fix: pre-response capability gate for /tts and /tts/stream.
        # Without this, /tts on a non-clone backend (CustomVoice) raises
        # NotImplementedError → FastAPI 500; /tts/stream is worse because
        # response headers are already on the wire when the worker thread
        # raises, leaving the client with a half-written stream.
        if backend is not None and getattr(backend, "supports_voice_cloning", True) is False:
            from server.core.tts_backend import TTSCapability
            if not backend.has_capability(TTSCapability.VOICE_CLONE):
                raise _VoiceCloneUnsupportedError(backend)
        try:
            return {"speaker_embedding": base64.b64decode(req.speaker_embedding_b64)}
        except Exception as exc:
            raise ValueError("Invalid base64 speaker_embedding_b64") from exc

    if backend is not None:
        model_id = backend.model_id
    else:
        from server.core import tts_service
        if tts_service.is_ready():
            model_id = tts_service.get_backend().model_id
        else:
            model_id = os.environ.get("OVS_TTS_MODEL_ID") or "qwen3-tts"

    merged = merge_tts_request_kwargs(
        request_speaker_id=speaker_id,
        request_speed=req.speed,
        request_pitch_shift=getattr(req, "pitch", None),
        model_id=model_id,
    )
    # Translate merged speaker_id into backend-specific kwargs (speaker_id
    # for preset, speaker_embedding for an embedding-typed entry, etc.).
    out: dict = speaker_kwargs_for_id(merged["speaker_id"], model_id)
    # FIX_2: thread merged speed / pitch_shift through so PATCH /admin/tts/runtime
    # actually takes effect. Only include keys that resolved to a non-None value
    # so backends keep using their intrinsic defaults when nothing was set.
    if merged.get("speed") is not None:
        out["speed"] = merged["speed"]
    if merged.get("pitch_shift") is not None:
        out["pitch_shift"] = merged["pitch_shift"]
    # Named voice selector (e.g. SparkTTS clone "voice_id"). Forwarded as a string so
    # a clone-capable backend can route it through its voice registry. Backends that
    # don't recognise it ignore the extra kwarg.
    voice = getattr(req, "voice", None)
    if voice:
        # Server-side embedding-profile resolution: if `voice` names an
        # embedding-profile enrolled via /tts/voices/enroll (CPU-ONNX path),
        # load its raw float32 speaker vector and forward it as
        # `speaker_embedding` — the Qwen3 BASE backend has no voice registry but
        # already consumes raw embeddings. SparkTTS `global_ids` clones and any
        # other opaque selector fall through as a plain `voice` passthrough.
        from server.core import sparktts_voices
        # Scope the lookup to the active canonical model.  Without this, the
        # loader's Base default could route a registered Qwen embedding into
        # Spark/MOSS after a hot switch.
        emb = sparktts_voices.load_embedding_voice(voice, model_id=model_id)
        if emb is not None:
            if backend is not None and getattr(backend, "supports_voice_cloning", True) is False:
                from server.core.tts_backend import TTSCapability
                if not backend.has_capability(TTSCapability.VOICE_CLONE):
                    raise _VoiceCloneUnsupportedError(backend)
            out["speaker_embedding"] = emb
            out.pop("voice", None)
        else:
            out["voice"] = voice
    return out


def _peek_tts_backend():
    """Return the live TTS backend for metadata/CPU-only ops (no slot acquire).

    Prefers the BackendManager's current backend (``get_backend_unsafe`` — safe
    for readiness/metadata queries per its contract), falling back to the legacy
    ``tts_service`` backend. Used by /tts/voices/enroll to reach the CPU-ONNX
    ``extract_speaker_embedding`` without holding a synthesis slot. Returns
    ``None`` when no backend is ready.
    """
    # Distinguish an absent manager (ASR-only/legacy startup, where the
    # singleton has never been installed) from an installed manager that is
    # currently INIT/FAILED/DRAINING.  In the latter case a stale
    # ``tts_service._backend`` must never leak into discovery metadata.
    mgr = _get_tts_manager()
    if mgr is not None:
        if not mgr.is_ready():
            return None
        try:
            return mgr.get_backend_unsafe()
        except Exception:
            return None
    from server.core import tts_service
    if tts_service.is_ready():
        return tts_service.get_backend()
    return None


def _get_asr_backend():
    return _asr_backend


_tts_lazy_start_lock = None  # asyncio.Lock; created on first use


def _try_tts_manager():
    """Return the TTS BackendManager if it is initialised+ready, else None.

    Kept for ASR-only profiles where TTS isn't wired at all. For LAZY_TTS the
    ``_ensure_tts_manager_started`` coroutine should be awaited first so the
    manager is in READY state before this is consulted.
    """
    mgr = _get_tts_manager()
    return mgr if mgr is not None and mgr.is_ready() else None


def _get_tts_manager():
    """Return the installed TTS manager, including non-ready states."""
    try:
        from server.core.backend_manager import tts_manager  # local import; PR3 module
        return tts_manager()
    except RuntimeError:
        return None


def _manager_state_value(manager) -> str | None:
    """Return a safe, non-secret manager state label for capability output."""
    raw = getattr(manager, "state", None)
    value = getattr(raw, "value", raw)
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


async def _ensure_tts_manager_started():
    """FIX_3 / FIX_3_completion: drive the TTS BackendManager to READY.

    Return values:
      * ``mgr`` — manager exists and is READY (caller must use ``acquire``).
      * ``None`` — manager was never installed (ASR-only profile, or
        ``init_backend_managers`` wasn't called). Caller may fall back to the
        legacy ``tts_service.synthesize`` path.

    Raises ``HTTPException(503)`` when the manager exists but is *not*
    serviceable — FAILED, DRAINING, RELOADING, or when ``start()`` fails to
    bring an INIT-state manager to READY. This is intentional: a FAILED
    manager indicates a configuration / resource problem and silently
    falling back to legacy ``tts_service`` would bypass the drain contract
    and mask the failure from operators.
    """
    import asyncio as _asyncio
    global _tts_lazy_start_lock
    try:
        from server.core.backend_manager import tts_manager, BackendState
        mgr = tts_manager()
    except RuntimeError:
        # Manager singleton never installed → legacy fallback is OK.
        return None

    if mgr.is_ready():
        return mgr

    # FAILED is non-recoverable here — surface as 503, never fall through to
    # legacy tts_service (which would skip drain / hide the failure).
    if mgr.state == BackendState.FAILED:
        raise HTTPException(
            status_code=503,
            detail={"error": "tts_manager_failed", "state": "failed"},
        )

    # DRAINING / RELOADING are transient — surface 503 so the client retries.
    if mgr.state != BackendState.INIT:
        raise HTTPException(
            status_code=503,
            detail={"error": "tts_manager_unavailable", "state": mgr.state.value},
        )

    if _tts_lazy_start_lock is None:
        _tts_lazy_start_lock = _asyncio.Lock()
    async with _tts_lazy_start_lock:
        if mgr.is_ready():
            return mgr
        if mgr.state == BackendState.FAILED:
            raise HTTPException(
                status_code=503,
                detail={"error": "tts_manager_failed", "state": "failed"},
            )
        if mgr.state != BackendState.INIT:
            raise HTTPException(
                status_code=503,
                detail={"error": "tts_manager_unavailable", "state": mgr.state.value},
            )
        try:
            # Exclusive profiles keep ASR resident at startup and lazy-load
            # TTS. Evict ASR before manager.start() preloads TTS; waiting for
            # the endpoint's later coordinator section would briefly
            # co-reside both models and can OOM a 16GB Orin shared with GDN.
            from server.core.coordinator import get_coordinator
            async with get_coordinator().acquire("tts"):
                pass
            await mgr.start()
        except Exception as exc:
            logger.exception("lazy TTS manager.start() failed")
            # start() failure flips state to FAILED. Surface 503 instead of
            # silently falling back to legacy tts_service.
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "tts_manager_start_failed",
                    "state": mgr.state.value,
                },
            ) from exc
    if mgr.is_ready():
        return mgr
    # Defensive: start() returned without exception but state isn't READY.
    raise HTTPException(
        status_code=503,
        detail={"error": "tts_manager_unavailable", "state": mgr.state.value},
    )


def _try_asr_manager():
    """Return the ASR BackendManager if it is initialised+ready, else None."""
    mgr = _get_asr_manager()
    return mgr if mgr is not None and mgr.is_ready() else None


def _get_asr_manager():
    """Return the installed ASR manager, including non-ready states."""
    try:
        from server.core.backend_manager import asr_manager
        return asr_manager()
    except RuntimeError:
        return None


def _resolve_tts_stream_max_workers() -> tuple[int, str | None, str]:
    """Resolve the TTS stream executor `max_workers` from env + the
    currently-loaded backend's concurrency_capability. Returns
    `(workers, backend_name_or_None, source_label)`.

    Spec docs/specs/concurrency-capability-framework.md §5: executor cap
    and the WorkerIO semaphore must derive from the same capability
    source. Resolution precedence:

      1. backend-specific env (OVS_TTS_STREAM_MAX_WORKERS_{KOKORO,MATCHA,
         QWEN3,MOSS}) if matching, then global OVS_TTS_STREAM_MAX_WORKERS
         — env values are CLAMPED to the backend ceiling.
      2. backend.concurrency_capability(profile).max_concurrent (None ->
         legacy global default of 2 since the executor needs a finite N).
      3. legacy default ``2``.

    Codex Week 3 BLOCKER 4 lazy-startup workflow is preserved: the
    one-shot post-ready refresh still applies because we re-read both
    backend name and capability each call until the flag flips.
    """
    try:
        from server.core import tts_service as _tts_svc
        backend_name = (
            (_tts_svc.backend_name() or "").lower()
            if _tts_svc.is_ready()
            else ""
        )
    except Exception:
        backend_name = ""

    # Delegate to the shared resolver (spec §5). Profile lookup +
    # capability fallback + env clamp now live in one place — see
    # ``server.core.capability_resolver``.
    try:
        from server.core.profile_loader import current_profile
        from server.core.capability_resolver import resolve_executor_for_tts
        prof = current_profile()
    except Exception:
        prof = {}
        from server.core.capability_resolver import resolve_executor_for_tts

    workers, name, src = resolve_executor_for_tts(
        profile=prof, tts_backend_name=backend_name or None,
    )
    # Surface the legacy WARNING for env > ceiling clamps. The resolver
    # already produced the warning text; we reproduce it here at WARNING
    # level to preserve the prior log surface for ops dashboards.
    try:
        from server.core.capability_resolver import resolve as _resolve_cap
        snapshot = _resolve_cap(
            profile=prof, tts_backend_name=backend_name or None,
        )
        for w in snapshot.clamp_warnings:
            if w.startswith("TTS executor:"):
                logger.warning(w)
    except Exception:
        pass
    return workers, name, src


# Tracks whether the cached executor was created BEFORE the TTS backend
# name could be resolved. If True, the first /tts/stream call that lands
# with a ready backend will refresh the executor so backend-specific
# OVS_TTS_STREAM_MAX_WORKERS_* envs actually take effect.
_tts_stream_executor_resolved_backend: bool = False


def _prefetch_window_allows(
    next_to_submit: int, current_idx: int, prefetch_max: int
) -> bool:
    """May sentence `next_to_submit` be handed to the executor yet?

    Sentences 0..next_to_submit-1 have been submitted and `current_idx` is the
    one being drained, so the number already in flight *ahead* of the current
    one is `next_to_submit - 1 - current_idx`.  Allow another while that stays
    under the window.

    The comparison used to be written against `next_to_submit - current_idx`
    with a strict `<`, which is off by one: with prefetch_max=1 -- the value on
    every RK device, where the TTS executor has a single worker -- it evaluated
    1 < 1 and never submitted anything after sentence 0.  The drain loop then
    awaited a queue nothing would ever fill, so /tts/stream deadlocked on *any*
    multi-sentence input and held the single session slot until the client gave
    up.  The old comment claimed a window of 1 degraded to serial synthesis; it
    degraded to synthesizing the first sentence and hanging.
    """
    return (next_to_submit - 1 - current_idx) < prefetch_max


def _kokoro_hybrid_pipeline_enabled(sentence_count: int) -> bool:
    """Return whether one HTTP request may use Kokoro's cross-sentence pipeline.

    ``rk.tts`` is the product-layer backend name even when the wrapped
    rkvoice-stream engine is Kokoro, so backend identity is not a reliable
    selector here. Keep this opt-in tied to the two profile environment values
    that identify the wrapped backend and explicitly enable its pipeline.
    Single-sentence requests retain the existing streaming path because there
    is no cross-sentence work to overlap.
    """
    return (
        sentence_count > 1
        and os.environ.get("TTS_BACKEND") == "kokoro_rknn"
        and os.environ.get("KOKORO_HYBRID_PIPELINE") == "1"
    )


def _get_tts_stream_executor() -> ThreadPoolExecutor:
    global _tts_stream_executor, _tts_stream_executor_resolved_backend
    # Codex Week 3 BLOCKER 4: if the cached executor was built before the
    # TTS backend was identifiable, try once more now that backend_name()
    # may resolve. This lets backend-specific env overrides apply even
    # when the executor was lazily touched during early startup.
    if _tts_stream_executor is not None and not _tts_stream_executor_resolved_backend:
        try:
            from server.core import tts_service as _tts_svc
            backend_ready_now = _tts_svc.is_ready() and bool(_tts_svc.backend_name())
        except Exception:
            backend_ready_now = False
        if backend_ready_now:
            new_workers, backend_name, env_used = _resolve_tts_stream_max_workers()
            if new_workers != _tts_stream_executor._max_workers:
                logger.info(
                    "TTS executor: refreshing max_workers %d → %d "
                    "(backend=%s, env=%s) after TTS service became ready",
                    _tts_stream_executor._max_workers,
                    new_workers, backend_name, env_used,
                )
                old = _tts_stream_executor
                _tts_stream_executor = ThreadPoolExecutor(
                    max_workers=new_workers,
                    thread_name_prefix="tts-stream",
                )
                # Best-effort shutdown of the old executor without
                # blocking; in-flight tasks finish naturally.
                try:
                    old.shutdown(wait=False, cancel_futures=False)
                except Exception:
                    pass
            else:
                logger.info(
                    "TTS executor: backend=%s resolved post-init "
                    "(env=%s, max_workers=%d, no change needed)",
                    backend_name, env_used, new_workers,
                )
            _tts_stream_executor_resolved_backend = True
    if _tts_stream_executor is None:
        # Phase 3b-B-4 part-4 INVESTIGATION RESULT: lifting max_workers above
        # 1 exposes a deeper bug in the C++ stateful Code2WavRunner reset
        # path. Two concurrent /tts/stream requests cause:
        #
        #   CUDA runtime error in cudaMemsetAsync(state.read.rawPointer(), ...)
        #   an illegal memory access was encountered
        #
        # The C++ engine slot pools (Phase 3b-B-1) + worker thread-dispatch
        # (Phase 3b-B-2) + per-slot Code2Wav (Phase 3b-B-4 part-2 commit
        # `5e1323f`) all carry the right per-slot data, but per-slot
        # StatefulCode2WavRunner state buffer initialization isn't actually
        # multi-slot safe yet — that's the next bottleneck to fix. Until
        # that's resolved, keep this serializing at the HTTP layer so the
        # worker never sees two in-flight requests simultaneously. The
        # `OVS_TTS_WORKER_CONCURRENCY` env is wired all the way down (engine
        # pools sized to it, worker dispatcher uses it, _WorkerIO semaphore
        # picks it up) but its only practical effect today is making the
        # cold-start eager-init less of a spike when the cap eventually
        # rises.
        # Phase B C1+C2+C3+C5 landed (fork commits e1abd90, fff8a38,
        # 99cf14a) — per-request locals for the talker + CP scratch
        # tensors plus C5 Code2Wav worker mutex. Real N=2 throughput
        # IS achievable on Orin NX: empirically 1.3-1.5× single-client
        # TTFA on the first N=2 request-pair after restart (within the
        # ≤ 1.5× spec gate). Audio MD5 byte-identical baseline at N=1.
        # Caveat: sustained N=2 (3+ consecutive bursts) still shows
        # cumulative state corruption from residual shared state
        # (mSamplingWorkspace and/or TRT context sharing inside the
        # CodePredictor engine slot pool, not yet traced). Default
        # max_workers=2 lets the optimization apply; if you observe
        # CUDA errors in production, set OVS_TTS_STREAM_MAX_WORKERS=1
        # to fall back to the C5b runtime-mutex stability gate.
        # Week 3 spec §D1: backend-specific override env > global > default.
        # Lets ops force one backend single-slot without muting the others.
        max_workers, backend_name, env_used = _resolve_tts_stream_max_workers()
        if backend_name:
            logger.info(
                "TTS executor: backend=%s using %s=%d",
                backend_name, env_used, max_workers,
            )
            _tts_stream_executor_resolved_backend = True
        else:
            logger.info(
                "TTS executor: backend not yet resolved at init; using %s=%d "
                "(will refresh once TTS service is ready)",
                env_used, max_workers,
            )
        _tts_stream_executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tts-stream",
        )
    return _tts_stream_executor


def _resolve_asr_max_workers() -> tuple[int, str]:
    """Resolve the ASR executor ``max_workers`` from the active ASR backend's
    ``concurrency_capability`` (slot-pool ceiling N). Returns
    ``(workers, source_label)``.

    Same capability source as the C++ worker's ``--max_slots N`` + the
    backend's ``WorkerIO`` Semaphore(N) (see
    ``trt_edge_llm_asr.concurrency_capability``): precedence is env
    ``EDGE_LLM_ASR_MAX_CONCURRENT`` → profile ``asr_max_slots`` → default 1.
    Reading it through the unified ``capability_resolver`` keeps the executor
    cap, the WorkerIO semaphore, and the session-limiter ceiling derived from
    one place (spec docs/specs/concurrency-capability-framework.md §5).

    Default 1 → byte-equivalent to the legacy single-thread executor, so an
    unset ``asr_max_slots`` is fully backward compatible (no behavior change).
    """
    try:
        from server.core.profile_loader import current_profile
        from server.core.capability_resolver import resolve as _resolve_cap
        prof = current_profile()
    except Exception:
        return 1, "default(no-profile)"
    try:
        snapshot = _resolve_cap(profile=prof)
        n = snapshot.asr_cap.max_concurrent
    except Exception:
        return 1, "default(resolve-failed)"
    if not isinstance(n, int) or n < 1:
        # max_concurrent=None (capability undeclared) → legacy single slot.
        return 1, "default(asr_cap=None)"
    return n, "asr_cap.max_concurrent"


# Tracks whether the cached ASR executor was sized BEFORE the ASR backend /
# profile could be resolved. If True, the first /asr/stream (or /v2v) call
# that lands with a resolvable capability re-sizes the executor once so
# ``asr_max_slots`` / ``EDGE_LLM_ASR_MAX_CONCURRENT`` actually take effect even
# when the executor was lazily touched during early startup (mirrors the TTS
# executor's BLOCKER-4 post-ready refresh).
_asr_executor_resolved_capability: bool = False


def _get_asr_executor() -> ThreadPoolExecutor:
    global _asr_executor, _asr_executor_resolved_capability
    # One-shot post-ready refresh: if the cached executor was built before the
    # ASR capability was resolvable, re-size it now (the profile / backend may
    # have become available since). Idempotent — flips the flag on first
    # successful resolution.
    if _asr_executor is not None and not _asr_executor_resolved_capability:
        new_workers, src = _resolve_asr_max_workers()
        if not src.startswith("default("):
            _asr_executor_resolved_capability = True
            if new_workers != _asr_executor._max_workers:
                logger.info(
                    "ASR executor: refreshing max_workers %d → %d (source=%s) "
                    "after capability became resolvable",
                    _asr_executor._max_workers, new_workers, src,
                )
                old = _asr_executor
                _asr_executor = ThreadPoolExecutor(
                    max_workers=new_workers, thread_name_prefix="asr-stream"
                )
                # In-flight ASR ops finish on the old executor naturally.
                try:
                    old.shutdown(wait=False, cancel_futures=False)
                except Exception:
                    pass
    if _asr_executor is None:
        # max_workers=N from the ASR slot-pool ceiling. Lifting this above 1
        # is SAFE under the slot-pool worker: each decoder slot's CUDA graph
        # is captured once at ``initSlotPool`` time (per-slot, NOT per-request),
        # so concurrent run_in_executor dispatch never races on graph capture.
        # The single-thread serialization that this executor historically
        # enforced (process-global _ASR_CUDA_STREAM capture race) is obsolete
        # for slot-pool backends — keeping max_workers=1 here would otherwise
        # serialize accept_waveform/finalize and silently defeat the
        # worker-side slot-pool + WorkerIO Semaphore(N) concurrency.
        # Default N=1 (asr_max_slots unset) → byte-equivalent to the legacy
        # single-thread executor, so this is backward compatible.
        max_workers, src = _resolve_asr_max_workers()
        if not src.startswith("default("):
            _asr_executor_resolved_capability = True
            logger.info(
                "ASR executor: max_workers=%d (source=%s)", max_workers, src,
            )
        else:
            logger.info(
                "ASR executor: max_workers=%d (source=%s; will refresh once "
                "ASR capability is resolvable)", max_workers, src,
            )
        _asr_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="asr-stream"
        )
    return _asr_executor


def _is_pool_saturated(exc: BaseException) -> tuple[bool, int | None]:
    """Duck-type a backend ``PoolSaturatedError`` (ASR or TTS — two distinct
    classes in different modules) without importing either.

    Both backends define ``PoolSaturatedError(RuntimeError)`` with a class
    attribute ``status = 4429`` and an instance attribute ``max_slots``. They
    are deliberately NOT ``WorkerProtocolError`` subclasses, so the session
    manager does not treat them as worker faults and does NOT trigger a
    destructive kill+respawn — a saturation is "backend busy", not a protocol
    error. We mirror that here: recognize by ``status == 4429`` (+ class name
    as a belt-and-braces check) and surface a clean 4429 reject, never a
    worker restart.

    Returns ``(is_saturated, max_slots_or_None)``.
    """
    if getattr(exc, "status", None) == 4429 or type(exc).__name__ == "PoolSaturatedError":
        ms = getattr(exc, "max_slots", None)
        return True, ms if isinstance(ms, int) else None
    return False, None


def _unpack_finalize_result(raw):
    """Normalise ``ASRStream.finalize()`` return to ``(text, language)``.

    Per the ASRStream ABC contract, backends MUST return ``(text, language)``.
    Some migrated or third-party backends historically returned a
    StreamSession-style dict; accept that shape here to avoid silently
    unpacking dict keys as transcript/language.
    """
    if isinstance(raw, dict):
        return raw.get("text") or "", raw.get("language")
    text, lang = raw
    return text or "", lang


def _default_vad_backend() -> str:
    return (
        os.environ.get("OVS_VAD_BACKEND")
        or "silero"
    ).strip() or "silero"


def _vad_preroll_ms() -> int:
    """Pre-speech audio (ms) replayed into the ASR stream on a frontend-VAD
    speech-start, so silero's onset-detection latency does not clip the first
    word. silero only fires SPEECH_START after the word onset has crossed its
    threshold; the leading frames were consumed by ``vad.process()`` while the
    ASR turn was not yet open and never reached the decoder. 0 disables.
    (Real-machine 2026-06-15: systematic first-word drop on the reBot demo.)"""
    raw = os.environ.get("OVS_VAD_PREROLL_MS") or "300"
    try:
        return max(0, int(raw))
    except ValueError:
        return 300


def _default_vad_silence_ms() -> int:
    raw = os.environ.get("OVS_VAD_SILENCE_MS") or "400"
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid OVS_VAD_SILENCE_MS=%r; using 400", raw)
        return 400
    return max(0, value)


def _flag_or(value, env_default: bool) -> bool:
    """Resolve an optional per-connection flag against an env default.

    Mirrors the ?vad= convention: a value (when present) overrides the env
    default; absent (None) → use the env default. Accepts a ``?flag=`` query
    string (/asr/stream) or a JSON bool/value from the v2v ``config`` frame.
    """
    if value is None:
        return env_default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


async def _augment_final_payload(
    payload, raw_text, seg, punct_on, spk_on, sample_rate,
    *, diarizer=None, seg_start=None, seg_end=None,
):
    """Apply optional punctuation + speaker embedding to a *final* payload.

    Mutates ``payload`` in place: rewrites ``payload['text']`` with restored
    punctuation, and adds {speaker_embedding, embedding_model, dim, normalized}
    from the utterance audio in ``seg`` (a list of float32 numpy chunks). A
    no-op (zero cost, behavior identical to before) when both flags are off.

    P0b/P1 additions (purely additive — existing fields untouched, off-path is
    byte-identical): when speaker embedding is on, the segment's session-relative
    ``start``/``end`` seconds are added. When ``diarizer`` is provided (diarize
    on), the embedding is fed to the online diarizer and ``speaker`` /
    ``speaker_conf`` are added too.

    Runs the CPU models in the default executor so the event loop and the ASR
    decode executor are not blocked. Never raises to the caller.
    """
    if not punct_on and not spk_on:
        return
    loop = asyncio.get_event_loop()
    if punct_on and raw_text:
        try:
            from server.core import punctuation as _punct
            payload["text"] = await loop.run_in_executor(
                None, _punct.add_punctuation, raw_text
            )
        except Exception:
            logger.exception("punctuation on final failed; keeping raw text")
    if spk_on and seg:
        try:
            import numpy as _np
            from server.core import speaker_embedding as _spk
            samples = _np.concatenate(seg) if len(seg) > 1 else seg[0]
            emb = await loop.run_in_executor(
                None, _spk.compute_embedding, samples, sample_rate
            )
            # P0b: tag the segment with its session-relative time window so a
            # consumer (or the diarizer below) can order the transcript.
            if seg_start is not None and seg_end is not None:
                payload["start"] = round(float(seg_start), 3)
                payload["end"] = round(float(seg_end), 3)
            if emb is not None:
                payload.update(_spk.embedding_payload(emb))
                # P1: online blind diarization — assign a speaker label.
                if diarizer is not None:
                    try:
                        _ds = diarizer.assign(
                            emb, float(seg_start or 0.0), float(seg_end or 0.0)
                        )
                        payload["speaker"] = _ds.speaker
                        payload["speaker_conf"] = round(float(_ds.confidence), 3)
                    except Exception:
                        logger.exception("diarize assign on final failed; skipping label")
        except Exception:
            logger.exception("speaker embedding on final failed; skipping")

@app.on_event("shutdown")
async def shutdown_watchdog():
    """Cancel the GPU watchdog background task on app shutdown.

    Best-effort: any errors here are swallowed because the process is
    going away regardless.
    """
    try:
        from server.core import gpu_watchdog as _gw
        await _gw.stop()
    except Exception:
        logger.debug("gpu_watchdog stop raised during shutdown", exc_info=True)


@app.on_event("startup")
async def startup():
    global _asr_backend, _rk_profile_status

    try:
        from server.core.profile_loader import apply_profile_from_env, current_profile
        apply_profile_from_env()
    except Exception as exc:
        logger.error("Failed to apply OpenVoiceStream profile: %s", exc)
        raise

    # RK release profiles own the true-streaming/NPU/Matcha contract.  Keep a
    # bounded status snapshot for /health and /v1/capabilities and make any
    # mismatch visible at boot instead of allowing a batch-mode downgrade to
    # look like a healthy profile selection.
    _rk_profile_status = _current_rk_profile_status()
    if _rk_profile_status.get("required"):
        from server.core.rk_profile_contract import format_failure
        if _rk_profile_status.get("verified"):
            logger.info(
                "RK profile contract verified: profile=%s device=%s contract=%s settings=%s",
                _rk_profile_status.get("profile"),
                _rk_profile_status.get("device"),
                _rk_profile_status.get("contract"),
                _rk_profile_status.get("settings"),
            )
        else:
            logger.error("RK profile contract INVALID: %s", format_failure(_rk_profile_status))

    # TRACK 1 SLICE 2 (gated): if the active profile carries a `composition`
    # block, validate it (fail fast, before any download/limiter), apply the
    # leaf-derived env with env-wins precedence, and capture the union-pull
    # file list for the downloader below. A profile WITHOUT `composition` is a
    # strict no-op here — the flat path is byte-for-byte unchanged.
    _composition_pull_files: list[str] = []
    try:
        from server.core.composition_boot import apply_composition
        _composition_pull_files = apply_composition(current_profile())
    except Exception as exc:
        logger.error("Composition validation/apply failed: %s", exc)
        raise

    # Week 1 production hardening: initialise the global session limiter
    # immediately after profile application, BEFORE model downloads and
    # backend preload. A bad limit value (zero/negative/non-int env) MUST
    # fail startup early. See docs/archive/specs/prod-hardening-week1.md
    # Deliverable 2.
    try:
        from server.core.session_limiter import init_limiter
        init_limiter(current_profile())
    except Exception as exc:
        logger.error("SessionLimiter init failed: %s", exc)
        raise

    # Initialise the execution coordinator from the loaded profile's
    # execution_policy block. Default to concurrent (no lock) when the
    # profile does not declare one — matches the previous behaviour.
    from server.core.coordinator import init_coordinator, get_coordinator
    _prof = current_profile()
    init_coordinator(
        _prof.get("execution_policy", {"mode": "concurrent"}),
        profile=_prof,
    )

    # ASR inference gate: decouples "how many sessions are connected" from
    # "how many ASR inferences run at once". Sized from the ASR backend's
    # max_concurrent (in-flight inference ceiling), NOT from the session
    # ceiling. On a single shared RKNN context this stays 1 while the session
    # limiter admits N; per-utterance work then queues here in FIFO order
    # instead of being rejected at connect time.
    try:
        from server.core.asr_infer_gate import init_asr_inference_gate
        from server.core.capability_resolver import resolve as _resolve_cap
        _rc = _resolve_cap(
            profile=_prof,
            policy=_prof.get("execution_policy"),
        )
        init_asr_inference_gate(
            concurrency=_rc.asr_infer_concurrency,
            max_waiting=_rc.asr_queue_depth,
        )
        _set_asr_sentence_level_locking(_rc)
    except Exception:
        logger.exception(
            "ASR inference gate init failed; falling back to serial default"
        )

    # Week 2: launch the GPU/NPU watchdog background task. Failures here
    # never block startup — the task is purely diagnostic.
    try:
        from server.core import gpu_watchdog as _gw
        await _gw.start()
    except Exception:
        logger.exception("gpu_watchdog: start() failed; continuing without watchdog")

    # Rockchip userspace runtime is vendored in the RK image. Validate it
    # before importing rkvoice-stream backends so version/hash mismatches fail
    # with a clear operator action instead of opaque native runtime errors.
    if (current_profile().get("env") or {}).get("LANGUAGE_MODE") == "rk" or os.environ.get("LANGUAGE_MODE") == "rk":
        from server.core.rk_runtime import check_rk_runtime
        check_rk_runtime(current_profile())

    # Log language mode configuration
    language_mode = os.environ.get("LANGUAGE_MODE", "zh_en")
    logger.info("=" * 60)
    logger.info("LANGUAGE_MODE: %s", language_mode)
    logger.info(
        "VAD default: backend=%s silence_ms=%d",
        _default_vad_backend(),
        _default_vad_silence_ms(),
    )
    if language_mode == "multilanguage":
        logger.info("  → Using Qwen3 TTS + ASR (52 languages, voice cloning)")
    else:
        logger.info("  → Using Sherpa TTS + ASR (zh/en mode)")
    logger.info("=" * 60)

    from server.core import model_downloader
    model_dir = os.environ.get("MODEL_DIR", "/opt/models")
    # TRACK 1 SLICE 2 (gated): when composition mode is active, the leaf
    # union-pull list is folded in additively. The leaf env (EDGE_LLM_*) was
    # already emitted to os.environ above, so the existing artifact-provisioning
    # path picks the right files up unchanged; here we only surface the
    # declared union for the operator. Empty (and silent) on the flat path.
    if _composition_pull_files:
        logger.info(
            "composition: %d union-pull file(s) required: %s",
            len(_composition_pull_files), _composition_pull_files,
        )
    if _composition_pull_files:
        model_downloader.ensure_models(
            language_mode, model_dir, qwen3_required_files=_composition_pull_files
        )
    else:
        model_downloader.ensure_models(language_mode, model_dir)

    # Resolve any TRT engines declared by the active profile. Must run
    # AFTER model_downloader (ONNX inputs may be needed for fallback
    # compile) and BEFORE any backend module is imported by the factories
    # (backends read engine paths from env vars at module import time).
    try:
        from server.core.engine_resolver import resolve_all
        resolved = resolve_all(current_profile())
        if resolved:
            logger.info("engine_resolver: resolved %d engine(s)", len(resolved))
            for env_var, path in resolved.items():
                logger.info("  %s → %s", env_var, path)
    except Exception as exc:
        logger.error("engine_resolver failed: %s", exc)
        raise

    # ASR backend (load before TTS to avoid ORT session conflicts)
    # Note: create_asr_backend() will auto-select based on LANGUAGE_MODE
    try:
        from server.core.asr_backend import create_asr_backend
        _asr_backend = create_asr_backend()  # Let it auto-detect from LANGUAGE_MODE
        logger.info("Pre-loading ASR (%s)...", _asr_backend.name)
        _asr_backend.preload()
        logger.info("ASR backend: %s (capabilities: %s)",
                     _asr_backend.name, [c.value for c in _asr_backend.capabilities])

        # Warm up ASR executor thread so its CUDA per-thread context is
        # initialised before the first streaming request.  Without this the
        # very first accept_waveform pays a cold-context tax on encoder.
        # SKIP_ASR_WARMUP=1 skips this on memory-constrained devices (Nano 8GB):
        # saves ~300-400 MB at startup, costs ~100ms one-time cold-context tax
        # on the very first ASR request.
        if os.environ.get("SKIP_ASR_WARMUP", "").lower() in ("1", "true", "yes"):
            logger.info("ASR streaming warmup skipped (SKIP_ASR_WARMUP set).")
        else:
            _asyncio = __import__("asyncio")
            _executor = _get_asr_executor()

            def _warm_asr():
                warm_target = _asr_backend
                backend_warmup = getattr(warm_target, "warmup", None)
                if not callable(backend_warmup):
                    # Voxedge RK adapters wrap the concrete rkvoice-stream
                    # backend.  After preload(), the inner backend is live and
                    # may expose a real RKNN/RKLLM inference warmup hook.
                    inner = getattr(_asr_backend, "_inner", None)
                    inner_warmup = getattr(inner, "warmup", None)
                    if callable(inner_warmup):
                        warm_target = inner
                        backend_warmup = inner_warmup

                if callable(backend_warmup):
                    try:
                        backend_warmup()
                        logger.info(
                            "ASR backend warmup completed (%s via %s).",
                            type(warm_target).__name__,
                            type(_asr_backend).__name__,
                        )
                    except Exception as exc:
                        logger.warning("ASR backend warm-up failed: %s", exc)
                    return

                # Some backends (e.g. SherpaASRBackend) don't expose a
                # transcribe_audio convenience method; their warmup is
                # implicit in preload(). Skip silently to avoid log noise.
                if not hasattr(_asr_backend, "transcribe_audio"):
                    logger.info(
                        "ASR warmup skipped: %s has no transcribe_audio (preload already warmed).",
                        type(_asr_backend).__name__,
                    )
                    return
                try:
                    import numpy as _np
                    silence = _np.zeros(16000, dtype=_np.float32)
                    _asr_backend.transcribe_audio(silence)
                    # Note: warmup primes ONE executor thread's CUDA context.
                    # With a slot-pool backend (max_workers=N) per-slot CUDA
                    # graphs are captured at initSlotPool, not per-thread, so a
                    # single warm thread is sufficient.
                    logger.info("ASR streaming executor warmed up (CUDA primed).")
                except Exception as exc:
                    logger.warning("ASR warm-up failed: %s", exc)

            await _asyncio.get_event_loop().run_in_executor(_executor, _warm_asr)
    except Exception as e:
        logger.warning("ASR backend failed: %s", e)

    from server.core import tts_service
    if not tts_service.is_configured():
        logger.info("ASR-only mode: profile declares no tts_backend; TTS endpoints will return 503.")
    elif os.environ.get("LAZY_TTS", "").lower() in ("1", "true", "yes"):
        logger.info("TTS preload skipped (LAZY_TTS set); will load on first request.")
    else:
        logger.info("Pre-loading TTS model...")
        tts_service.preload()

    # Warm up the dedicated streaming-TTS executor thread so its CUDA
    # per-thread context is initialized before the first /tts/stream
    # request lands. Without this, the very first streaming request
    # pays a ~30ms cold-context tax on prefill.
    # Skip when LAZY_TTS or ASR-only — TTS not loaded yet, can't warm what isn't there.
    if not tts_service.is_configured():
        pass  # ASR-only mode, no TTS warmup
    elif os.environ.get("LAZY_TTS", "").lower() in ("1", "true", "yes"):
        logger.info("TTS streaming warmup skipped (LAZY_TTS).")
    else:
      try:
        from server.core.tts_backend import TTSCapability
        if tts_service.has_capability(TTSCapability.STREAMING):
            backend = tts_service.get_backend()
            executor = _get_tts_stream_executor()

            def _warm_stream():
                try:
                    # Run one tiny streaming synthesis on the executor
                    # thread to materialize CUDA context state.
                    stream_kwargs = {}
                    profile = os.environ.get("EDGE_LLM_TTS_WARMUP_STREAMING_PROFILE")
                    if profile:
                        stream_kwargs["streaming_profile"] = profile
                    warmup_text = os.environ.get("EDGE_LLM_TTS_WARMUP_TEXT", "你好")
                    for _ in backend.generate_streaming(warmup_text, **stream_kwargs):
                        pass
                except Exception as exc:  # pragma: no cover
                    logger.warning("TTS streaming warm-up failed: %s", exc)

            import asyncio as _asyncio
            await _asyncio.get_event_loop().run_in_executor(executor, _warm_stream)
            logger.info("TTS streaming executor warmed up (1 thread, CUDA primed).")
      except Exception as exc:  # pragma: no cover
        logger.warning("TTS streaming executor warm-up skipped: %s", exc)

    # ── Optional capabilities (punctuation / speaker embedding) ─────────
    # Opt-in, default-OFF. Eager-load ONLY when enabled so the first request
    # doesn't pay download + init latency; when disabled these are never
    # imported beyond the cheap env check (zero memory / behavior change).
    # Fire-and-forget (NOT awaited): a slow/large model download (~294MB) must
    # not block startup completion / /readyz. The model becomes ready in the
    # background; a request arriving first falls back to the same lazy load.
    try:
        from server.core import punctuation as _punct
        if _punct.punctuation_enabled():
            logger.info("Punctuation enabled (OVS_PUNCT); pre-loading model in background...")
            asyncio.get_event_loop().run_in_executor(None, _punct.preload)
    except Exception as exc:  # pragma: no cover
        logger.warning("Punctuation preload skipped: %s", exc)
    try:
        from server.core import speaker_embedding as _spk
        if _spk.speaker_embedding_enabled():
            logger.info("Speaker embedding enabled (OVS_SPEAKER_EMB); pre-loading model in background...")
            asyncio.get_event_loop().run_in_executor(None, _spk.preload)
    except Exception as exc:  # pragma: no cover
        logger.warning("Speaker embedding preload skipped: %s", exc)

    # Register backend getters with the coordinator so 'exclusive' policy can
    # call unload() on the dormant slot. Lambdas resolve lazily so they cope
    # with backends loaded after this point (LAZY_TTS).
    try:
        from server.core import tts_service as _tts_service_mod
        coord = get_coordinator()
        coord.register_backend("asr", lambda: _asr_backend)
        coord.register_backend("tts", lambda: _tts_service_mod._backend)
    except Exception as exc:  # pragma: no cover
        logger.warning("Coordinator backend registration skipped: %s", exc)

    # ── BackendManager wiring (PR4) ─────────────────────────────────────
    # Wrap the already-preloaded ASR/TTS instances in lifecycle managers so
    # /admin/backend/reload + acquire()-based request gating can drain
    # inflight work and hot-swap backends. The factories below return the
    # *current* singleton; on a reload the manager will call them again
    # after unloading the previous one, by which point tts_service /
    # _asr_backend have been re-bound to fresh instances. preloader is a
    # no-op on initial start (already loaded above); on reload the factory
    # invokes the real backend factory which performs its own preload.
    try:
        from server.core import backend_manager as _bm
        from server.core import tts_service as _tts_service_mod
        from server.core.asr_backend import create_asr_backend as _create_asr
        from server.core.tts_backend import create_tts_backend as _create_tts

        # On reload, build a fresh instance and rebind the legacy module
        # globals so downstream code (which still reads tts_service /
        # _asr_backend directly) sees the new backend.
        def _asr_factory():
            global _asr_backend
            if _asr_backend is None:
                _asr_backend = _create_asr()
            return _asr_backend

        def _asr_preloader(b):
            # Initial start: already preloaded above. On reload, the
            # factory returns a freshly constructed (un-preloaded)
            # instance, so we call preload() here.
            if not b.is_ready():
                b.preload()

        def _asr_unloader(b):
            global _asr_backend
            try:
                b.unload()
            finally:
                _asr_backend = None

        def _tts_factory():
            if _tts_service_mod._backend is None:
                _tts_service_mod._backend = _create_tts()
            return _tts_service_mod._backend

        def _tts_preloader(b):
            if not b.is_ready():
                b.preload()

        def _tts_unloader(b):
            try:
                b.unload()
            finally:
                _tts_service_mod._backend = None

        # FIX_4_completion: seed both managers with the profile ref used at
        # startup (OVS_PROFILE_JSON / OVS_PROFILE / OVS_PROFILE_DEFAULT). Same
        # precedence as profile_loader.apply_profile_from_env so rollback
        # re-applies via the identical source.
        _initial_profile_ref = (
            os.environ.get("OVS_PROFILE_JSON")
            or os.environ.get("OVS_PROFILE")
            or os.environ.get("OVS_PROFILE_DEFAULT")
        )
        _bm.init_backend_managers(
            tts_factory=_tts_factory,
            tts_preloader=_tts_preloader,
            tts_unloader=_tts_unloader,
            asr_factory=_asr_factory,
            asr_preloader=_asr_preloader,
            asr_unloader=_asr_unloader,
            initial_profile_ref=_initial_profile_ref,
        )

        # Bring up managers. ASR is always started if a backend exists;
        # TTS respects ASR-only profiles and LAZY_TTS env (matches the
        # legacy preload skip above).
        if _asr_backend is not None:
            await _bm.asr_manager().start()
        if tts_service.is_configured() and tts_service.is_ready():
            await _bm.tts_manager().start()
    except Exception as exc:  # pragma: no cover
        logger.warning("BackendManager wiring skipped: %s", exc)

    logger.info("Speech service ready.")


# ── Health & Capabilities ────────────────────────────────────────

# RFC 8594 deprecation hint pointing /health users at /readyz.
_HEALTH_DEPRECATION_HEADERS = {
    "Deprecation": "true",
    "Link": '</readyz>; rel="successor-version"',
}


def _env_truthy(value: "str | None") -> bool:
    """Truthy check for an env-flag value, tolerant of ``--env-file`` quoting.

    Production injects flags via ``--env-file`` whose values can carry literal
    quotes, so ``FLAG="1"`` arrives in ``os.environ`` as the 3-char string
    ``"1"``. A plain ``.strip().lower()`` leaves the quotes in place, so a
    quoted truthy value silently reads as False — exactly the 2026-05-31
    server-loop activation bug (agent-side fix in
    ``ovs_agent/config.py``). Strip a single matched outer quote
    pair (after whitespace) before comparing; the ``1``/``true`` semantics are
    unchanged for unquoted values.
    """
    if value is None:
        return False
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("\"", "'"):
        v = v[1:-1].strip()
    return v.lower() in ("1", "true", "yes", "on")


def _metrics_requires_key() -> bool:
    """Return True when ``OVS_METRICS_REQUIRE_KEY`` opts into API-key
    protection for ``/metrics``. Default-off so standard Prometheus
    scrapes work without auth."""
    return _env_truthy(os.environ.get("OVS_METRICS_REQUIRE_KEY"))


# Operator-level default for the backend-owned endpoint VAD threshold, in ms.
# Distinct from the profile's ``VAD_ENDPOINT_SILENCE_MS`` on purpose: the
# profile loader re-stamps its own keys at startup, so a compose env entry for
# that key never reaches the backend (measured 2026-09-13). This one is read
# per session and injected through the ASR stream options, so a deployment (or
# one client) can tune it without editing the image profile.
_VAD_ENDPOINT_OPERATOR_ENV = "OVS_V2V_VAD_ENDPOINT_SILENCE_MS"


def _coerce_vad_endpoint_ms(raw: "object", *, source: str) -> "int | None":
    """Parse a backend endpoint threshold in ms; None when unset/invalid/0.

    Invalid or out-of-range values are ignored with a warning rather than
    failing the session — a bad tuning hint must never break a conversation.
    """
    if raw is None:
        return None
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning(
            "v2v: ignoring invalid vad_endpoint_silence_ms=%r from %s (want int ms)",
            raw,
            source,
        )
        return None
    if value < 0 or value > 60000:
        logger.warning(
            "v2v: ignoring out-of-range vad_endpoint_silence_ms=%d from %s (0-60000 ms)",
            value,
            source,
        )
        return None
    if value == 0:
        return None
    return value


def _v2v_asr_stream_options(cfg: "dict[str, object]") -> dict:
    """Session-scoped ASR stream options from the parsed v2v config.

    Today this is just ``vad_endpoint_silence_ms`` — the backend endpoint
    threshold for ASR backends that own their own endpoint VAD (RK Qwen3).
    Without a per-session channel the only lever was the profile, which then
    stamped the env value back at startup, so a client could not tune it at
    all (see server/core/rk_profile_contract.py for the measured cost: 400 ms
    cut a 1.5 s mid-sentence pause into two turns).

    Precedence: the client's session value wins; otherwise the operator default
    ``OVS_V2V_VAD_ENDPOINT_SILENCE_MS`` applies. The operator variable is
    deliberately NOT named ``VAD_ENDPOINT_SILENCE_MS``: that key is part of the
    baked profile and the profile loader re-stamps it at startup, so a compose
    ``environment:`` entry for it is silently ignored (measured 2026-09-13).
    The operator default must stay above the client VAD's silence window (the
    shipped agent uses 600 ms) or the backend wins the endpoint race and cuts
    utterances in half — see docs/CONFIGURATION.md "Streaming ASR
    endpointing".
    """
    explicit = _coerce_vad_endpoint_ms(
        cfg.get("vad_endpoint_silence_ms"), source="session config"
    )
    if explicit is not None:
        return {"vad_endpoint_silence_ms": explicit}
    operator = _coerce_vad_endpoint_ms(
        os.environ.get(_VAD_ENDPOINT_OPERATOR_ENV), source=_VAD_ENDPOINT_OPERATOR_ENV
    )
    if operator is not None:
        return {"vad_endpoint_silence_ms": operator}
    return {}


def _endpoint_detector_conflict(vad_backend: "str | None", asr_be) -> "str | None":
    """Single-endpoint-detector guard for the voxedge engine path.

    Returns the operator-facing warning text when the client asked for a
    server-side VAD while the ASR backend already owns endpointing, else
    ``None``. Two detectors racing is the most expensive misconfiguration of
    this API: whichever fires first wins, the loser's segment is silently
    truncated, and neither side logs an error
    (docs/CONFIGURATION.md "Streaming ASR endpointing: pick exactly one
    detector").

    The legacy (non-engine) path already honours
    ``prefer_backend_endpoint_vad`` at runtime and keeps its behaviour; the
    engine path has no such reconciliation, so the caller gates on this.
    Pure function: no I/O, no env reads — the strict-mode reject lives at the
    call site.
    """
    if asr_be is None:
        return None
    if not bool(getattr(asr_be, "prefer_backend_endpoint_vad", False)):
        return None
    vad = str(vad_backend or "").strip().lower()
    if vad in ("", "none", "off", "disabled"):
        return None
    return (
        f"v2v(engine): client requested server VAD '{vad_backend}' "
        "but the ASR backend owns endpointing "
        "(prefer_backend_endpoint_vad=True) — two endpoint "
        "detectors would race and silently truncate turns. Send "
        "vad=none (Realtime V2: turn_detection.type='none') and "
        "let the backend endpoint, or use a client-side VAD. "
        "See docs/CONFIGURATION.md 'Streaming ASR endpointing'."
    )


@app.get("/metrics")
async def metrics_endpoint(request: Request) -> Response:
    """Prometheus text exposition.

    Default unprotected (standard Prometheus scrape pattern). Set
    ``OVS_METRICS_REQUIRE_KEY=true`` to require the same API key used
    by public voice endpoints — ``Authorization: Bearer <key>``.

    Read-only: never blocks on backend locks, never acquires a session
    slot, never runs GPU probes. Returns 200 even while ``/readyz`` is
    503 so operators can scrape during incidents.
    """
    if _metrics_requires_key():
        # Reuse the existing HTTP auth path; it raises 401 (with a
        # ``ovs_auth_rejected_total{endpoint="/metrics"}`` bump) when
        # the token is missing or invalid.
        from server.core.api_auth import check_http
        check_http(request)

    from server.core import metrics as _metrics_mod
    body = _metrics_mod.render_prometheus()
    return Response(content=body, media_type=_metrics_mod.prometheus_content_type())


@app.get("/livez")
async def livez():
    """Process-liveness probe (always 200 while the route is reachable).

    No backend / GPU / model / profile dependency. Use this for
    orchestrator liveness restart policy; ``/readyz`` controls traffic
    admission instead.
    """
    return JSONResponse({"status": "ok"})


@app.get("/readyz")
async def readyz():
    """Readiness probe: 200 only when the service should receive traffic.

    Ready iff:
      * Required BackendManager(s) report READY (ASR always; TTS unless
        the profile is ASR-only or ``LAZY_TTS=1``).
      * The global session limiter has free capacity.
      * ``gpu_watchdog.is_ok()`` returns True.

    Read-only: never acquires a session slot, never mutates limiter
    state. Returns 503 with stable ``reasons[]`` otherwise (see spec
    Deliverable 3).
    """
    from server.core import backend_manager as _bm_mod
    from server.core import session_limiter as _sl_mod
    from server.core import gpu_watchdog as _gw_mod
    from server.core import tts_service

    reasons: list[str] = []

    # BackendManager readiness — only managers that are *required* for
    # the active profile.
    try:
        asr_mgr = _bm_mod.asr_manager()
    except Exception:
        asr_mgr = None
    try:
        tts_mgr = _bm_mod.tts_manager()
    except Exception:
        tts_mgr = None

    asr_required = _get_asr_backend() is not None
    lazy_tts = os.environ.get("LAZY_TTS", "").lower() in ("1", "true", "yes")
    tts_required = tts_service.is_configured() and not lazy_tts

    if asr_required:
        if asr_mgr is None:
            reasons.append("backend_manager_unavailable")
        elif not asr_mgr.is_ready():
            reasons.append("backend_not_ready")
    if tts_required:
        if tts_mgr is None:
            if "backend_manager_unavailable" not in reasons:
                reasons.append("backend_manager_unavailable")
        elif not tts_mgr.is_ready():
            if "backend_not_ready" not in reasons:
                reasons.append("backend_not_ready")

    # Session capacity.
    limiter = _sl_mod.get_limiter()
    if limiter is None:
        reasons.append("session_limiter_unavailable")
    elif limiter.available <= 0:
        reasons.append("sessions_full")

    # GPU/NPU watchdog (Week 2: real background-checked status).
    wd_detail = None
    try:
        if not _gw_mod.is_ok():
            reasons.append("gpu_watchdog_failed")
        try:
            wd_detail = _gw_mod.status()
        except Exception:
            wd_detail = None
    except Exception:
        reasons.append("gpu_watchdog_failed")

    if reasons:
        body = {"status": "not_ready", "reasons": reasons}
        if wd_detail is not None:
            body["details"] = {"gpu_watchdog": wd_detail}
        return JSONResponse(body, status_code=503)
    return JSONResponse({"status": "ready"})


@app.get("/health")
async def health():
    from server.core import tts_service

    rk_profile_status = _rk_profile_status or _current_rk_profile_status()
    result = {
        "tts": tts_service.is_ready(),
        "tts_backend": tts_service.backend_name() if tts_service.is_ready() else None,
        "tts_capabilities": [c.value for c in tts_service.capabilities()] if tts_service.is_ready() else [],
    }
    if rk_profile_status.get("required"):
        result["runtime_profile"] = rk_profile_status

    # Part D disconnect-watcher instrumentation: expose the counter from the
    # WorkerIO class actually used by the active backend. The TRT-Edge-LLM
    # backend now lives in voxedge and therefore no longer shares the class
    # variable in server.core.worker_io.
    try:
        backend = tts_service.get_backend() if tts_service.is_ready() else None
        worker_io = getattr(backend, "_worker_io", None)
        worker_io_cls = type(worker_io) if worker_io is not None else None
        if (
            worker_io_cls is not None
            and hasattr(worker_io_cls, "_cancel_count")
            and hasattr(worker_io_cls, "_cancel_count_lock")
        ):
            with worker_io_cls._cancel_count_lock:
                result["tts_worker_cancel_count"] = worker_io_cls._cancel_count
        else:
            from server.core.worker_io import WorkerIO
            with WorkerIO._cancel_count_lock:
                result["tts_worker_cancel_count"] = WorkerIO._cancel_count
    except Exception:
        pass

    # ASR
    try:
        from server.core.asr_backend import create_asr_backend
        asr_be = _get_asr_backend()
        result["asr"] = asr_be.is_ready() if asr_be else False
        result["asr_backend"] = asr_be.name if asr_be and asr_be.is_ready() else None
        result["asr_capabilities"] = [c.value for c in asr_be.capabilities] if asr_be and asr_be.is_ready() else []
        # Capability signal for clients that must pick exactly ONE endpoint
        # detector (docs/CONFIGURATION.md "Streaming ASR endpointing"): when
        # this is true, a client asking for its own server VAD would race the
        # backend's, so it should send vad:"none" (and run a client-side VAD if
        # it needs one). Offline/accumulate backends report false and need the
        # server VAD to ever finish a turn.
        result["asr_owns_endpointing"] = bool(
            getattr(asr_be, "prefer_backend_endpoint_vad", False)
        ) if asr_be and asr_be.is_ready() else None
        if asr_be and asr_be.is_ready() and hasattr(asr_be, "providers"):
            result["asr_providers"] = asr_be.providers
    except Exception:
        result["asr"] = False
        result["asr_backend"] = None
        result["asr_capabilities"] = []

    # /health is preserved for backward-compat but deprecated; orchestrators
    # should migrate to /readyz (RFC 8594 Deprecation hint).
    return JSONResponse(result, headers=_HEALTH_DEPRECATION_HEADERS)


@app.get("/asr/capabilities")
async def asr_capabilities(_: None = Depends(_require_api_key)):
    """Return ASR backend info and supported capabilities."""
    asr_mgr = _get_asr_manager()
    if asr_mgr is None:
        asr_be = _get_asr_backend()
    elif asr_mgr.is_ready():
        try:
            asr_be = asr_mgr.get_backend_unsafe()
        except Exception:
            asr_be = None
    else:
        asr_be = None
    if not asr_be or not asr_be.is_ready():
        return JSONResponse({"error": "ASR not ready"}, status_code=503)
    caps = {
        "backend": asr_be.name,
        "capabilities": [c.value for c in asr_be.capabilities],
        "sample_rate": asr_be.sample_rate,
    }
    if hasattr(asr_be, "providers"):
        caps["providers"] = asr_be.providers
    return caps


def _collect_route_paths(routes, _depth: int = 0) -> set[str]:
    """Every path reachable from ``routes``, descending into included routers.

    FastAPI >= ~0.13x no longer copies a sub-router's routes into
    ``app.routes``: ``include_router()`` leaves a single lazy holder whose own
    ``path`` is empty and whose children live one level down. Reading
    ``app.routes`` flat therefore misses everything mounted that way -- here
    the whole OpenAI-compatible surface, so /v1/capabilities stopped
    advertising "openai-audio" and a client negotiating on it would conclude
    the server has no OpenAI API. Descending keeps this working on both the
    old and the new behaviour; requirements.txt pins only a lower bound
    (fastapi>=0.115.0), so both are live.
    """
    found: set[str] = set()
    if _depth > 4:  # defensive: routers do not nest this deep
        return found
    for route in routes or ():
        path = str(getattr(route, "path", "") or "")
        if path:
            found.add(path)
        child = None
        for holder in (
            route,
            getattr(route, "original_router", None),  # FastAPI's _IncludedRouter
            getattr(route, "router", None),
            getattr(route, "app", None),
        ):
            child = getattr(holder, "routes", None) if holder is not None else None
            if child:
                break
        if child:
            found |= _collect_route_paths(child, _depth + 1)
    return found


def _registered_capability_api_versions() -> list[str]:
    """Derive advertised capability API versions from registered routes.

    This keeps the Phase A document honest if a deployment omits a router or
    when later phases add the OpenAI-compatible audio surface.  The helper is
    intentionally based on ``app.routes`` rather than a hard-coded phase
    switch, and is evaluated only after the application has been assembled.
    """
    paths = _collect_route_paths(app.routes)
    versions: list[str] = []
    if "/tts/capabilities" in paths or "/asr/capabilities" in paths:
        versions.append("legacy")
    if "/v1/capabilities" in paths:
        versions.append("v1")
    openai_audio_routes = {
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/models",
    }
    if openai_audio_routes.issubset(paths):
        versions.append("openai-audio")
    return versions


@app.get("/tts/capabilities")
async def tts_capabilities(_: None = Depends(_require_api_key)):
    """Return TTS backend info and supported capabilities."""
    from server.core.tts_speakers import available_speakers
    from server.core.api_capabilities import build_capabilities
    from server.core import tts_service

    tts_mgr = _get_tts_manager()
    if tts_mgr is None:
        if not tts_service.is_ready():
            return JSONResponse({"error": "TTS not ready"}, status_code=503)
        backend = tts_service.get_backend()
        backend_name = tts_service.backend_name()
        caps = [c.value for c in tts_service.capabilities()]
        sample_rate = tts_service.get_sample_rate()
    elif tts_mgr.is_ready():
        try:
            backend = tts_mgr.get_backend_unsafe()
        except Exception:
            return JSONResponse({"error": "TTS not ready"}, status_code=503)
        if not backend or not backend.is_ready():
            return JSONResponse({"error": "TTS not ready"}, status_code=503)
        backend_name = getattr(backend, "name", "tts")
        caps = [getattr(c, "value", str(c)) for c in getattr(backend, "capabilities", ())]
        sample_rate = getattr(backend, "sample_rate", None)
    else:
        # Preserve the legacy response shape/status while avoiding a stale
        # tts_service singleton during INIT/FAILED/DRAINING manager states.
        return JSONResponse({"error": "TTS not ready"}, status_code=503)

    structured = build_capabilities(
        tts_backend=backend,
        tts_ready=True,
        tts_configured=True,
    )["tts"]
    cloning = structured["cloning"]
    return {
        "backend": backend_name,
        "model_id": backend.model_id,
        "capabilities": caps,
        # Keep the legacy flat keys, but derive them from the structured
        # contract so CustomVoice/MOSS/Spark cannot drift from discovery.
        "supports_voice_cloning": bool(cloning["supported"]),
        "supports_voice_enrollment": bool(cloning["enrollment"]["supported"]),
        "sample_rate": sample_rate,
        "speakers": available_speakers(backend.model_id),
    }


@app.get("/v1/capabilities")
async def v1_capabilities(_: None = Depends(_require_api_key)):
    """Return structured ASR/TTS capabilities during any startup state.

    Unlike the legacy component routes this endpoint is intentionally 200 for
    lazy, ASR-only, TTS-only and failed backend states.  The builder reads
    current state without triggering a lazy preload and keeps the legacy flat
    routes untouched.
    """
    from server.core import session_limiter, tts_runtime
    from server.core.api_capabilities import build_capabilities
    from server.core.profile_loader import current_profile

    try:
        profile = current_profile() or {}
    except Exception:
        profile = {}

    # Discovery must use the manager-owned backend when a manager exists.  A
    # non-ready manager is represented as unavailable, even if the legacy
    # tts_service/asr globals still point at a stale pre-reload instance.
    tts_mgr = _get_tts_manager()
    if tts_mgr is None:
        tts_backend = _peek_tts_backend()  # legacy fallback only when absent
        tts_manager_state = None
    elif tts_mgr.is_ready():
        try:
            tts_backend = tts_mgr.get_backend_unsafe()
        except Exception:
            tts_backend = None
        tts_manager_state = None if tts_backend is not None else _manager_state_value(tts_mgr)
    else:
        tts_backend = None
        tts_manager_state = _manager_state_value(tts_mgr)

    asr_mgr = _get_asr_manager()
    if asr_mgr is None:
        asr_backend = _get_asr_backend()  # legacy fallback only when absent
        asr_manager_state = None
    elif asr_mgr.is_ready():
        try:
            asr_backend = asr_mgr.get_backend_unsafe()
        except Exception:
            asr_backend = None
        asr_manager_state = None if asr_backend is not None else _manager_state_value(asr_mgr)
    else:
        asr_backend = None
        asr_manager_state = _manager_state_value(asr_mgr)
    try:
        limiter = session_limiter.get_limiter()
    except Exception:
        limiter = None
    try:
        runtime_speaker_id = tts_runtime.get_overrides().default_speaker_id
    except Exception:
        runtime_speaker_id = None
    return build_capabilities(
        tts_backend=tts_backend,
        asr_backend=asr_backend,
        tts_ready=bool(tts_backend and tts_backend.is_ready()),
        asr_ready=bool(asr_backend and asr_backend.is_ready()),
        tts_configured=bool(profile.get("tts_backend")) or tts_mgr is not None,
        asr_configured=bool(profile.get("asr_backend")) or asr_mgr is not None,
        limiter=limiter,
        profile=profile,
        runtime_speaker_id=runtime_speaker_id,
        tts_manager_state=tts_manager_state,
        asr_manager_state=asr_manager_state,
        api_versions=_registered_capability_api_versions(),
    )


# ── Speaker Management ─────────────────────────────────────────────


class RegisterSpeakerRequest(BaseModel):
    speaker_embedding_b64: str
    label: str | None = None
    speaker_id: int | None = None


@app.get("/tts/speakers")
async def tts_speakers_list(_: None = Depends(_require_api_key)):
    """List all speakers registered for the active TTS model."""
    from server.core import tts_service
    from server.core.tts_speakers import available_speakers, default_speaker_id
    tts_mgr = _get_tts_manager()
    if tts_mgr is None:
        if not tts_service.is_ready():
            return JSONResponse({"error": "TTS not ready"}, status_code=503)
        backend = tts_service.get_backend()
    elif tts_mgr.is_ready():
        try:
            backend = tts_mgr.get_backend_unsafe()
        except Exception:
            return JSONResponse({"error": "TTS not ready"}, status_code=503)
        if not backend or not backend.is_ready():
            return JSONResponse({"error": "TTS not ready"}, status_code=503)
    else:
        return JSONResponse({"error": "TTS not ready"}, status_code=503)
    from server.core.api_capabilities import build_capabilities
    structured = build_capabilities(
        tts_backend=backend,
        tts_ready=True,
        tts_configured=True,
    )["tts"]
    return {
        "model_id": backend.model_id,
        "default_speaker_id": default_speaker_id(backend.model_id),
        "speakers": available_speakers(backend.model_id),
        "supports_voice_cloning": bool(structured["cloning"]["supported"]),
    }


@app.post("/tts/speakers/register")
async def tts_speakers_register(
    req: RegisterSpeakerRequest,
    _: None = Depends(_require_api_key),
):
    """Register a voice-clone embedding as a persistent speaker.

    Accepts a base64-encoded speaker embedding (from /tts/clone/embedding)
    and assigns it a permanent speaker_id for subsequent /tts calls.
    """
    import base64
    from server.core import tts_service
    from server.core.tts_backend import TTSCapability
    from server.core.tts_speakers import register_speaker

    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)
    if not tts_service.has_capability(TTSCapability.VOICE_CLONE):
        # MEDIUM: use the unified capability-aware response (400 when the
        # backend *explicitly* disables cloning, 501 when it merely lacks
        # the capability) so /tts/speakers/register matches /tts/clone*.
        unsupported, _ok = _voice_clone_unsupported_response()
        if unsupported is not None:
            return unsupported

    try:
        emb = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    backend = tts_service.get_backend()
    try:
        spec = register_speaker(
            model_id=backend.model_id,
            payload=req.speaker_embedding_b64,
            label=req.label or "",
            meta={"dim": len(emb) // 4, "dtype": "float32"},
            speaker_id=req.speaker_id,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {
        "speaker_id": spec.id,
        "type": spec.type,
        "label": spec.label,
        "model_id": backend.model_id,
    }


@app.delete("/tts/speakers/{speaker_id}")
async def tts_speakers_delete(
    speaker_id: int,
    _: None = Depends(_require_api_key),
):
    """Delete a registered embedding speaker. Preset speakers cannot be deleted."""
    from server.core import tts_service
    from server.core.tts_speakers import unregister_speaker
    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)

    backend = tts_service.get_backend()
    try:
        ok = unregister_speaker(backend.model_id, speaker_id)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    if not ok:
        return JSONResponse({"error": f"Speaker {speaker_id} not found"}, status_code=404)
    return {"deleted": True, "speaker_id": speaker_id}


# ── SparkTTS clone voices (reference-token VoiceProfile registry, spec §4.4) ──
# Distinct from /tts/speakers (preset/embedding) and /tts/clone (embedding-based,
# CustomVoice). These manage host-enrolled VoiceProfiles selected at synth time via
# the `voice` field (e.g. {"text": "...", "voice": "clone:alice"}).

@app.get("/tts/voices")
async def tts_voices_list(_: None = Depends(_require_api_key)):
    """List registered SparkTTS clone voices (VoiceProfiles)."""
    from server.core import sparktts_voices
    return {"voices": sparktts_voices.list_voices(),
            "voices_dir": sparktts_voices.voices_dir()}


@app.post("/tts/voices/profile")
async def tts_voices_register_profile(
    profile_json: UploadFile = File(..., description="VoiceProfile .json"),
    profile_npz: UploadFile = File(..., description="VoiceProfile .npz"),
    voice_id: str | None = Form(None),
    _: None = Depends(_require_api_key),
):
    """Register a host-enrolled VoiceProfile (json + npz). Always available — no torch.

    The analysis chain runs on a GPU host (enroll_voice.py); this endpoint persists the
    resulting pair into the shared voices dir and reloads the live backend registry.
    """
    from server.core import sparktts_voices
    jb = await profile_json.read()
    nb = await profile_npz.read()
    try:
        res = sparktts_voices.register_from_profile_files(jb, nb, voice_id=voice_id)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return res


@app.post("/tts/voices/enroll")
async def tts_voices_enroll(
    file: UploadFile = File(..., description="reference wav (3-15s, single speaker)"),
    voice_id: str = Form(...),
    ref_text: str | None = Form(None),
    _: None = Depends(_require_api_key),
):
    """Enroll a clone voice from reference audio.

    Two paths, tried in order:

      1. **CPU-ONNX (torch-less, preferred on Jetson TRT)** — when the active TTS
         backend exposes a usable ``extract_speaker_embedding`` (its
         ``supports_voice_enrollment`` is true because a speaker-encoder ONNX is
         present). Produces a float32[1024] embedding on ONNX Runtime and persists
         it as an *embedding-profile* the server resolves at synth time.
      2. **PyTorch SparkTTS analysis chain** — falls back to the in-process
         wav2vec2 + BiCodec enroller (host deployments with the torch stack).

    Returns 501 only when neither path is available, with guidance to POST a
    host-generated ``.json + .npz`` to /tts/voices/profile.
    """
    from server.core import sparktts_voices
    audio = await file.read()

    # Path 1: CPU-ONNX extractor on the active backend (no torch required).
    backend = _peek_tts_backend()
    if backend is not None and getattr(backend, "supports_voice_enrollment", False):
        try:
            embedding = backend.extract_speaker_embedding(audio)
        except NotImplementedError:
            embedding = None
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # extractor blew up — surface, don't silently torch-fall
            logger.warning("ONNX speaker embedding extraction failed", exc_info=True)
            return JSONResponse(
                {"error": f"speaker embedding extraction failed: {exc}"},
                status_code=500,
            )
        if embedding:
            try:
                embedding_kwargs = {
                    "sample_rate": getattr(backend, "sample_rate", 24000),
                    "ref_text": ref_text,
                    "source_meta": {
                        "method": "onnx_speaker_encoder",
                        "backend": getattr(backend, "name", None),
                    },
                }
                # New Spark workers persist the active canonical model on the
                # profile so capability discovery can reject cross-model
                # embeddings.  Retry the legacy wheel signature only when
                # running an older worker that lacks this keyword.
                try:
                    res = sparktts_voices.register_embedding_voice(
                        voice_id,
                        embedding,
                        model_id=getattr(backend, "model_id", None),
                        **embedding_kwargs,
                    )
                except TypeError as exc:
                    if "model_id" not in str(exc):
                        raise
                    res = sparktts_voices.register_embedding_voice(
                        voice_id,
                        embedding,
                        **embedding_kwargs,
                    )
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            res["method"] = "onnx_speaker_encoder"
            return res

    # Path 2: in-process PyTorch SparkTTS enrollment (host deployments).
    try:
        res = sparktts_voices.enroll_from_audio(audio, voice_id, ref_text=ref_text)
    except sparktts_voices.EnrollmentUnavailable as exc:
        return JSONResponse(
            {"error": str(exc), "hint": "POST /tts/voices/profile with a host-enrolled .json+.npz"},
            status_code=501,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    res["method"] = "sparktts_pytorch"
    return res


@app.delete("/tts/voices/{voice_id:path}")
async def tts_voices_delete(voice_id: str, _: None = Depends(_require_api_key)):
    """Delete a clone voice's VoiceProfile (json + npz) and reload the registry."""
    from server.core import sparktts_voices
    if not sparktts_voices.delete_voice(voice_id):
        return JSONResponse({"error": f"clone voice {voice_id!r} not found"}, status_code=404)
    return {"deleted": True, "voice_id": voice_id}


# ── TTS ──────────────────────────────────────────────────────────

@app.post("/tts")
async def tts(req: TTSRequest, _: None = Depends(_require_api_key), request: Request = None):
    from server.core import tts_service
    from server.core.coordinator import get_coordinator
    from server.core.session_limiter import acquire_http

    try:
        async with acquire_http("/tts"):
            return await _tts_synthesize(req, request=request)
    except _TransportDisconnected:
        logger.info(
            "route=/tts client disconnected; finite cancellation drained"
        )
        return Response(status_code=204)


async def _execute_tts_core(
    req: TTSRequest,
    *,
    manager=None,
    voice_kwargs: dict | None = None,
    prepare=None,
    request: Request | None = None,
):
    """Run the shared transport-neutral non-streaming TTS execution core."""
    from server.core import tts_service
    from server.core.api_execution import execute_tts
    from server.core.coordinator import get_coordinator
    import threading

    async def _disconnect():
        while True:
            message = await request.receive()
            if message.get("type") == "http.disconnect":
                return True

    return await execute_tts(
        text=req.text,
        language=req.language,
        voice_kwargs=voice_kwargs or {},
        manager=manager,
        legacy_service=tts_service,
        coordinator=get_coordinator(),
        prepare=prepare,
        cancel_event=threading.Event(),
        disconnect_awaitable=_disconnect if request is not None else None,
    )


def _tts_response_headers(metadata: dict | None, *, backend: str | None = None, mode: str | None = None) -> dict[str, str]:
    """Build common TTS headers, including long32 routing diagnostics."""
    meta = metadata or {}
    convonly = backend in {"rk.tts", "rk:kokoro_convonly", "kokoro_convonly"} and meta.get("backend") == "kokoro_convonly"
    if convonly:
        import math
        def finite(value):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            try:
                return math.isfinite(value)
            except (OverflowError, TypeError):
                return False
        audio = meta.get("audio_s") if finite(meta.get("audio_s")) else 0
        infer = meta.get("total_ms") / 1000 if finite(meta.get("total_ms")) else 0
        full = meta.get("full_rtf") if finite(meta.get("full_rtf")) else 0
    headers = {
        "X-Audio-Duration": str(audio if convonly else meta.get("duration", meta.get("duration_s", 0))),
        "X-Inference-Time": str(infer if convonly else meta.get("inference_time", meta.get("inference_time_s", meta.get("wall_ms", 0) / 1000 if meta.get("wall_ms") is not None else 0))),
        "X-RTF": str(full if convonly else (meta.get("rtf", meta.get("total_rtf", meta.get("generator_rtf", 0))) or 0)),
    }
    if convonly:
        valid_number = finite
        sha = meta.get("manifest_sha256")
        if isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha): headers["X-Kokoro-Manifest-SHA256"] = sha
        values = (("platform", "X-Kokoro-Platform", lambda v: isinstance(v, str) and v in {"rk3576", "rk3588"}), ("selected_T", "X-Kokoro-Selected-T", valid_number), ("frontend_ms", "X-Kokoro-Frontend-Ms", valid_number), ("prefix_ms", "X-Kokoro-Prefix-Ms", valid_number), ("tail_ms", "X-Kokoro-Tail-Ms", valid_number), ("istft_ms", "X-Kokoro-Istft-Ms", valid_number), ("wav_ms", "X-Kokoro-Wav-Ms", valid_number), ("preload_ms", "X-Kokoro-Preload-Ms", valid_number))
        for key, header, check in values:
            value = meta.get(key, meta.get("T")) if key == "selected_T" else meta.get(key)
            if check(value): headers[header] = str(value)
        engine = meta.get("engine")
        if isinstance(engine, str) and engine in {"npu", "cpu", "mixed"}:
            headers["X-Kokoro-Engine"] = engine
        fallback = meta.get("fallback")
        if isinstance(fallback, bool):
            headers["X-Kokoro-Fallback"] = "1" if fallback else "0"
        segments = meta.get("segments")
        segment_count = meta.get("segment_count")
        if segment_count is None and isinstance(segments, list):
            segment_count = len(segments)
        if (isinstance(segment_count, int) and not isinstance(segment_count, bool)
                and 1 <= segment_count <= 256
                and (segments is None or isinstance(segments, list) and len(segments) == segment_count)):
            headers["X-Kokoro-Segments"] = str(segment_count)
        cpu_ms = meta.get("cpu_generator_ms")
        if valid_number(cpu_ms) and 0 <= cpu_ms <= 86_400_000:
            headers["X-Kokoro-CPU-Generator-Ms"] = str(cpu_ms)
        return headers
    # The product-layer result backend is ``rk.tts`` in production (the
    # wrapped implementation's name is ``kokoro_rknn``). Require both the
    # known RK product/backend identity and the long32 metadata marker so an
    # unrelated backend cannot opt in merely by returning a T field.
    if backend not in {"rk.tts", "kokoro_rknn"} or mode != "long32" or meta.get("backend") != "kokoro_long32":
        return headers

    def _bounded(value, *, item_limit: int = 64) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            values = [str(item) for item in value]
            if len(values) > item_limit:
                values = values[:item_limit] + [f"+{len(values) - item_limit} more"]
            value = ",".join(values)
        rendered = str(value)
        return rendered if len(rendered) <= 512 else rendered[:500] + "…"

    selected = meta.get("selected_T", meta.get("T"))
    headers["X-Kokoro-Selected-T"] = _bounded("" if selected is None else selected)
    headers["X-Kokoro-Route"] = _bounded(meta.get("route_ts", ()))
    headers["X-Kokoro-Snap-Ratio"] = _bounded(meta.get("snap_ratio", ""))
    headers["X-Kokoro-Fallback"] = "1" if bool(meta.get("fallback")) else "0"
    return headers


async def _tts_synthesize(req: TTSRequest, *, request: Request | None = None):
    """Legacy serializer around the shared transport-neutral TTS core."""
    from server.core import tts_service
    from server.core.coordinator import get_coordinator

    mgr = await _ensure_tts_manager_started()
    if mgr is not None:
        try:
            result = await _execute_tts_core(
                req,
                manager=mgr,
                prepare=lambda backend: _request_voice_kwargs(req, backend=backend),
                request=request,
            )
        except _VoiceCloneUnsupportedError as exc:
            return JSONResponse(
                _voice_clone_unsupported_payload(exc.backend),
                status_code=400,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        # Week 2: record server-side TTS RTF for /metrics.
        try:
            from server.core import metrics as _m
            _m.record_tts_rtf(result.backend or "tts", float(result.metadata.get("rtf", 0) or 0))
        except Exception:
            pass
    else:
        # Manager not initialised (ASR-only or wiring failed at startup) —
        # legacy tts_service path. Kept for ASR-only profiles where the
        # TTS manager is intentionally never started; LAZY_TTS is now handled
        # by _ensure_tts_manager_started above.
        try:
            result = await _execute_tts_core(
                req,
                manager=None,
                prepare=lambda backend: _request_voice_kwargs(req, backend=backend),
                request=request,
            )
        except _VoiceCloneUnsupportedError as exc:
            return JSONResponse(
                _voice_clone_unsupported_payload(exc.backend),
                status_code=400,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            from server.core import metrics as _m
            _m.record_tts_rtf(result.backend or tts_service.backend_name() or "tts", float(result.metadata.get("rtf", 0) or 0))
        except Exception:
            pass
    return Response(
        content=result.audio,
        media_type="audio/wav",
        headers=_tts_response_headers(result.metadata, backend=result.backend, mode=result.metadata.get("mode")),
    )


class NativeTTSRequest(BaseModel):
    """Strict body for the versioned native synthesis endpoint."""

    model: str
    text: str
    voice: int | str | None = None
    speaker_id: int | None = None
    sid: int | None = None
    speed: float | None = None
    pitch: float | None = None
    language: str | None = None

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            extra = "forbid"


def _v1_limit(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using %d", env_name, raw, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r; using %d", env_name, raw, default)
        return default
    return value


def _v1_validate_text(text: str) -> None:
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        from server.core.api_execution import APIExecutionError
        raise APIExecutionError(
            "text must be valid UTF-8",
            status_code=400,
            code="invalid_text",
            param="text",
        ) from exc
    max_bytes = _v1_limit("OVS_API_MAX_TEXT_BYTES", 64 * 1024)
    if size > max_bytes:
        from server.core.api_execution import APIExecutionError
        raise APIExecutionError(
            f"text exceeds the {max_bytes} byte limit",
            status_code=413,
            code="payload_too_large",
            param="text",
        )


def _v1_backend_model(backend: object) -> str:
    from server.core.tts_speakers import canonical_model_id
    value = getattr(backend, "model_id", None)
    if not value:
        raise ValueError("active TTS backend does not expose model_id")
    return canonical_model_id(str(value))


def _v1_resolve_tts_backend(manager):
    from server.core import tts_service
    from server.core.api_execution import APIExecutionError

    if manager is not None:
        try:
            return manager.get_backend_unsafe()
        except Exception as exc:
            raise APIExecutionError(
                "TTS backend is not ready",
                status_code=503,
                code="backend_not_ready",
            ) from exc
    if not tts_service.is_ready():
        raise APIExecutionError(
            "TTS backend is not ready",
            status_code=503,
            code="backend_not_ready",
        )
    return tts_service.get_backend()


def _v1_check_model(requested: str, active_model: str) -> None:
    from server.core.api_execution import APIExecutionError
    from server.core.tts_speakers import canonical_model_id

    if canonical_model_id(requested) != canonical_model_id(active_model):
        raise APIExecutionError(
            f"model {requested!r} is not the active TTS model",
            status_code=404,
            code="unknown_model",
            param="model",
        )


def _v1_resolve_voice_kwargs(req: NativeTTSRequest, backend: object) -> dict:
    """Strictly resolve a native v1 voice and translate it to backend kwargs."""
    from server.core.api_execution import APIExecutionError
    from server.core.tts_speakers import resolve_speaker_selector

    active_model = _v1_backend_model(backend)
    selector_count = sum(
        value is not None for value in (req.voice, req.speaker_id, req.sid)
    )
    if selector_count > 1:
        raise APIExecutionError(
            "voice, speaker_id and sid are mutually exclusive",
            status_code=400,
            code="duplicate_voice_selector",
            param="voice",
        )
    selector = req.voice
    if selector is None:
        selector = req.speaker_id if req.speaker_id is not None else req.sid

    if selector is None or (isinstance(selector, str) and not selector.strip()):
        # Preserve the established precedence: request > runtime override >
        # model default > backend intrinsic.  Resolving the model default here
        # would turn it into an explicit request and mask the runtime override.
        return _request_voice_kwargs(
            TTSRequest(
                text=req.text,
                speed=req.speed,
                pitch=req.pitch,
                language=req.language,
            ),
            backend=backend,
        )

    if isinstance(selector, str) and "spark" in active_model:
        genders = ("female", "male")
        levels = ("very_low", "low", "moderate", "high", "very_high")
        styles = {f"{gender}_{pitch}_{speed}" for gender in genders for pitch in levels for speed in levels}
        style = selector.strip().lower()
        if style in styles:
            # Spark consumes controllable styles through voice/speaker.  Remove
            # the default preset so the backend does not choose it before the
            # explicit style; continuous speed/pitch remain DSP controls.
            native_req = TTSRequest(
                text=req.text,
                speed=req.speed,
                pitch=req.pitch,
                language=req.language,
                voice=style,
            )
            kwargs = _request_voice_kwargs(native_req, backend=backend)
            kwargs.pop("speaker_id", None)
            kwargs.pop("speaker", None)
            kwargs["voice"] = style
            return kwargs

    profile = None
    if isinstance(selector, str) and selector.strip():
        try:
            from server.core import sparktts_voices
            profile = next(
                (
                    item
                    for item in sparktts_voices.list_voices(
                        model_id=active_model,
                        compatible_model=active_model,
                    )
                    if isinstance(item, dict) and item.get("voice_id") == selector
                ),
                None,
            )
        except Exception:
            profile = None

    if profile is not None:
        from server.core.api_execution import APIExecutionError
        from server.core import sparktts_voices

        expected_type = None
        if active_model == sparktts_voices.QWEN_BASE_MODEL_ID:
            expected_type = sparktts_voices.EMBEDDING_PROFILE_TYPE
        elif "spark" in active_model:
            expected_type = sparktts_voices.SPARK_PROFILE_TYPE
        if expected_type is None or profile.get("profile_type") != expected_type:
            raise APIExecutionError(
                f"voice {selector!r} is not a compatible profile for model {active_model!r}",
                status_code=404,
                code="unsupported_voice",
                param="voice",
            )
        native_req = TTSRequest(
            text=req.text,
            speed=req.speed,
            pitch=req.pitch,
            language=req.language,
            voice=selector,
        )
        kwargs = _request_voice_kwargs(native_req, backend=backend)
        # A registered profile is the complete voice selector.  Do not mix it
        # with the model's default preset/intrinsic selector.
        kwargs.pop("speaker_id", None)
        kwargs.pop("speaker", None)
        return kwargs

    try:
        spec = resolve_speaker_selector(selector, active_model)
    except (TypeError, ValueError) as exc:
        raise APIExecutionError(
            str(exc),
            status_code=404,
            code="unsupported_voice",
            param="voice",
        ) from exc
    native_req = TTSRequest(
        text=req.text,
        speed=req.speed,
        pitch=req.pitch,
        language=req.language,
        speaker_id=spec.id if spec is not None else None,
    )
    try:
        return _request_voice_kwargs(native_req, backend=backend)
    except _VoiceCloneUnsupportedError as exc:
        raise APIExecutionError(
            str(exc),
            status_code=400,
            code="unsupported_voice",
            param="voice",
        ) from exc
    except ValueError as exc:
        raise APIExecutionError(
            str(exc),
            status_code=400,
            code="unsupported_voice",
            param="voice",
        ) from exc


def _v1_validate_controls(req: NativeTTSRequest, backend: object) -> None:
    from server.core.api_execution import APIExecutionError

    if req.speed is not None:
        try:
            speed = float(req.speed)
        except (TypeError, ValueError):
            raise APIExecutionError(
                "speed must be numeric",
                status_code=400,
                code="unsupported_control",
                param="speed",
            )
        if speed != speed or speed in (float("inf"), float("-inf")) or not 0.25 <= speed <= 4.0:
            raise APIExecutionError(
                "speed must be in [0.25, 4.0]",
                status_code=400,
                code="unsupported_control",
                param="speed",
            )
    if req.pitch is not None:
        try:
            pitch = float(req.pitch)
        except (TypeError, ValueError):
            raise APIExecutionError(
                "pitch must be numeric",
                status_code=400,
                code="unsupported_control",
                param="pitch",
            )
        if pitch != pitch or pitch in (float("inf"), float("-inf")) or not -24.0 <= pitch <= 24.0:
            raise APIExecutionError(
                "pitch must be in [-24, 24] semitones",
                status_code=400,
                code="unsupported_control",
                param="pitch",
            )
    if req.speed is not None or req.pitch is not None:
        if not callable(getattr(backend, "rate_pitch_caps", None)):
            raise APIExecutionError(
                "the active backend does not support speed/pitch controls",
                status_code=400,
                code="unsupported_control",
                param="speed" if req.speed is not None else "pitch",
            )


def _native_error_response(exc: BaseException) -> JSONResponse:
    """Serialize a native v1 domain/admission failure without OpenAI coupling."""
    from fastapi import HTTPException
    from server.core.api_execution import APIExecutionError

    if isinstance(exc, APIExecutionError):
        status = exc.status_code
        code = exc.code
        message = exc.message
        param = exc.param
        headers = exc.headers
    elif _is_pool_saturated(exc)[0]:
        status = 429
        code = "backend_busy"
        message = "backend is busy"
        param = None
        headers = {"Retry-After": "1"}
    elif isinstance(exc, HTTPException):
        status = int(exc.status_code)
        detail = exc.detail
        if isinstance(detail, dict):
            code = str(detail.get("error") or "http_error")
            if status >= 500:
                public_messages = {
                    "tts_manager_start_failed": "TTS backend failed to start",
                    "tts_manager_failed": "TTS backend is not ready",
                    "tts_manager_unavailable": "TTS backend is temporarily unavailable",
                }
                message = public_messages.get(code, "service unavailable")
            else:
                message = str(detail.get("message") or detail.get("detail") or code)
        else:
            code = "http_error"
            # HTTPException.detail may be a scalar supplied by a third-party
            # startup/backend path.  Never expose that value for a 5xx native
            # response; manager/backend exceptions can contain local paths or
            # engine diagnostics.  Keep the stable public class used by the
            # structured-detail branch above.
            message = "service unavailable" if status >= 500 else str(detail)
        param = None
        headers = dict(exc.headers or {})
    else:
        logger.exception("native v1 execution failed", exc_info=exc)
        status = 503
        code = "backend_error"
        message = "backend execution failed"
        param = None
        headers = {"Retry-After": "1"}
    body = {"error": {"code": code, "message": message}}
    if param:
        body["error"]["param"] = param
    return JSONResponse(body, status_code=status, headers=headers)


class V1CloneEmbeddingRequest(BaseModel):
    """Strict Qwen3-TTS Base embedding-clone request.

    ``embedding_b64`` is the versioned spelling.  The legacy
    ``speaker_embedding_b64`` spelling remains accepted as an explicit
    compatibility alias, but both fields may not be sent together.
    """

    model: str | None = None
    text: str | None = None
    embedding_b64: str | None = None
    speaker_embedding_b64: str | None = None
    dim: int | None = None
    language: str | None = None
    speed: float | None = None
    pitch: float | None = None

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            extra = "forbid"


def _v1_clone_error(
    message: str,
    *,
    status_code: int = 400,
    code: str = "clone_error",
    param: str | None = None,
    headers: dict[str, str] | None = None,
):
    from server.core.api_execution import APIExecutionError

    return APIExecutionError(
        message,
        status_code=status_code,
        code=code,
        param=param,
        headers=headers,
    )


def _v1_decode_embedding(req: V1CloneEmbeddingRequest) -> bytes:
    """Decode and validate one strict little-endian float32 embedding."""
    import math
    import struct

    if req.embedding_b64 is not None and req.speaker_embedding_b64 is not None:
        raise _v1_clone_error(
            "embedding_b64 and speaker_embedding_b64 are mutually exclusive",
            code="multiple_profiles",
            param="embedding_b64",
        )
    value = req.embedding_b64
    field = "embedding_b64"
    if value is None:
        value = req.speaker_embedding_b64
        field = "speaker_embedding_b64"
    if not isinstance(value, str) or not value:
        raise _v1_clone_error(
            "a base64 float32 embedding is required",
            code="missing_required_parameter",
            param="embedding_b64",
        )
    max_bytes = _v1_limit("OVS_API_MAX_PROFILE_BYTES", 16 * 1024 * 1024)
    # Strict base64 has a deterministic expansion ratio.  Reject an encoded
    # body that cannot possibly decode within the configured limit before
    # allocating a second, decoded copy of an attacker-controlled JSON value.
    max_encoded_chars = 4 * ((max_bytes + 2) // 3)
    if len(value) > max_encoded_chars:
        raise _v1_clone_error(
            f"embedding exceeds the {max_bytes} byte limit",
            status_code=413,
            code="payload_too_large",
            param=field,
        )
    try:
        # validate=True rejects non-alphabet characters and malformed padding;
        # unlike the legacy endpoint this never silently discards separators.
        decoded = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise _v1_clone_error(
            "embedding must be valid base64",
            code="invalid_profile",
            param=field,
        ) from exc
    if len(decoded) > max_bytes:
        raise _v1_clone_error(
            f"embedding exceeds the {max_bytes} byte limit",
            status_code=413,
            code="payload_too_large",
            param=field,
        )
    if not decoded or len(decoded) % 4:
        raise _v1_clone_error(
            "embedding must contain little-endian float32 values",
            code="invalid_profile",
            param=field,
        )
    actual_dim = len(decoded) // 4
    if req.dim is not None and (req.dim <= 0 or req.dim != actual_dim):
        raise _v1_clone_error(
            f"embedding dim {req.dim!r} does not match decoded float32 count {actual_dim}",
            code="invalid_profile",
            param="dim",
        )
    # Check finiteness without allocating a tuple for a 16 MiB profile.  The
    # service contract consumes float32 little-endian values, not arbitrary
    # opaque bytes; NaN/Inf vectors are rejected before they reach TRT.
    for offset in range(0, len(decoded), 4 * 1024):
        chunk = decoded[offset : offset + 4 * 1024]
        values = struct.iter_unpack("<f", chunk)
        if any(not math.isfinite(item[0]) for item in values):
            raise _v1_clone_error(
                "embedding values must be finite float32 numbers",
                code="invalid_profile",
                param=field,
            )
    return decoded


def _v1_backend_has_capability(backend: object, capability: object) -> bool:
    try:
        has = getattr(backend, "has_capability", None)
        if callable(has):
            return bool(has(capability))
    except Exception:
        return False
    values = getattr(backend, "capabilities", ()) or ()
    wanted = getattr(capability, "value", capability)
    return any(getattr(item, "value", item) == wanted for item in values)


def _v1_require_clone_backend(backend: object, mode: str) -> str:
    """Fail closed before clone reaches an optional/unimplemented method."""
    from server.core.api_execution import APIExecutionError
    from server.core.tts_backend import TTSCapability, TTSBackend
    from server.core.tts_speakers import canonical_model_id

    active_model = _v1_backend_model(backend)
    if mode == "embedding":
        expected = "qwen3-tts-0.6b-base"
        valid_model = canonical_model_id(active_model) == expected
    elif mode == "reference_audio":
        valid_model = "moss" in canonical_model_id(active_model).lower()
    else:
        valid_model = False
    if not valid_model:
        raise APIExecutionError(
            f"clone mode {mode!r} is not supported by active model {active_model!r}",
            status_code=400,
            code="unsupported_clone_mode",
            param="model",
        )
    if getattr(backend, "supports_voice_cloning", True) is False:
        raise APIExecutionError(
            "the active backend does not support voice cloning",
            status_code=400,
            code="unsupported_clone_mode",
            param="model",
        )
    if not _v1_backend_has_capability(backend, TTSCapability.VOICE_CLONE):
        raise APIExecutionError(
            "the active backend does not advertise voice cloning",
            status_code=400,
            code="unsupported_clone_mode",
            param="model",
        )
    if not callable(getattr(backend, "clone_voice", None)):
        raise APIExecutionError(
            "the active backend has no usable clone implementation",
            status_code=400,
            code="unsupported_clone_mode",
            param="model",
        )
    implementation = getattr(type(backend), "clone_voice", None)
    if implementation is TTSBackend.clone_voice:
        raise APIExecutionError(
            "the active backend has no usable clone implementation",
            status_code=400,
            code="unsupported_clone_mode",
            param="model",
        )
    return active_model


def _v1_clone_control_kwargs(req: V1CloneEmbeddingRequest | NativeTTSRequest, backend: object) -> dict:
    """Validate shared controls and return backend clone kwargs."""
    native = NativeTTSRequest(
        model=_v1_backend_model(backend),
        text=req.text or "",
        speed=req.speed,
        pitch=req.pitch,
        language=req.language,
    )
    _v1_validate_controls(native, backend)
    result: dict = {}
    if req.speed is not None:
        result["speed"] = req.speed
    if req.pitch is not None:
        result["pitch_shift"] = req.pitch
    return result


def _v1_moss_codec_contract(backend: object) -> tuple[int, int]:
    """Return the active MOSS reference PCM contract (sample rate, channels)."""
    import json
    from pathlib import Path

    # Output audio and reference-codec audio are separate contracts: MOSS can
    # downmix worker output while its reference encoder still requires the
    # codec's native channel count.  Prefer explicit backend properties when
    # a newer voxedge wheel exposes them, then read the same codec metadata
    # file consumed by the C++ worker.  Never fall back to backend.sample_rate
    # or backend.channels, which describe synthesized output.
    try:
        sample_rate = int(getattr(backend, "reference_sample_rate"))
    except (TypeError, ValueError, AttributeError):
        sample_rate = 0
    try:
        channels = int(getattr(backend, "reference_channels"))
    except (TypeError, ValueError, AttributeError):
        channels = 0

    candidates: list[Path] = []
    explicit_meta = os.environ.get("MOSS_CODEC_META_PATH")
    if explicit_meta:
        candidates.append(Path(explicit_meta))
    codec_dir = getattr(backend, "_codec_onnx_dir", None) or os.environ.get("MOSS_CODEC_ONNX_DIR")
    if codec_dir:
        candidates.append(Path(str(codec_dir)) / "codec_browser_onnx_meta.json")
    if sample_rate <= 0 or channels <= 0:
        for path in candidates:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                codec = document.get("codec_config", document)
                meta_rate = int(codec.get("sample_rate", codec.get("sampleRate")))
                meta_channels = int(codec.get("channels"))
            except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
                continue
            if sample_rate <= 0:
                sample_rate = meta_rate
            if channels <= 0:
                channels = meta_channels
            break
    if sample_rate <= 0 or channels <= 0:
        raise _v1_clone_error(
            "active MOSS codec contract is unavailable",
            status_code=503,
            code="codec_contract_unavailable",
        )
    return sample_rate, channels


def _v1_parse_moss_reference_wav(raw: bytes, backend: object) -> tuple[bytes, int]:
    """Strictly parse a PCM16 RIFF/WAVE and strip its container header."""
    import struct

    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise _v1_clone_error("reference file must be binary", code="invalid_audio", param="file")
    data = bytes(raw)
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise _v1_clone_error("reference file must be a RIFF/WAVE stream", code="invalid_audio", param="file")
    riff_size = struct.unpack_from("<I", data, 4)[0]
    riff_end = riff_size + 8
    if riff_size < 4 or riff_end != len(data):
        raise _v1_clone_error("reference WAV has an invalid RIFF length", code="invalid_audio", param="file")
    fmt = None
    pcm = None
    offset = 12
    while offset < riff_end:
        if offset + 8 > riff_end:
            raise _v1_clone_error("reference WAV chunk header is truncated", code="invalid_audio", param="file")
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        start = offset + 8
        end = start + size
        padded_end = end + (size & 1)
        if end > riff_end or padded_end > riff_end:
            raise _v1_clone_error("reference WAV chunk is truncated", code="invalid_audio", param="file")
        if chunk_id == b"fmt ":
            if fmt is not None:
                raise _v1_clone_error("reference WAV contains multiple fmt chunks", code="invalid_audio", param="file")
            if size < 16:
                raise _v1_clone_error("reference WAV fmt chunk is too short", code="invalid_audio", param="file")
            fmt = struct.unpack_from("<HHIIHH", data, start)
        elif chunk_id == b"data":
            if pcm is not None:
                raise _v1_clone_error("reference WAV contains multiple data chunks", code="invalid_audio", param="file")
            pcm = data[start:end]
        offset = padded_end
    if fmt is None or pcm is None:
        raise _v1_clone_error("reference WAV must contain fmt and data chunks", code="invalid_audio", param="file")
    audio_format, channels, sample_rate, byte_rate, block_align, bits = fmt
    expected_rate, expected_channels = _v1_moss_codec_contract(backend)
    if audio_format != 1 or bits != 16:
        raise _v1_clone_error("reference WAV must be PCM16", code="invalid_audio", param="file")
    if channels != expected_channels or sample_rate != expected_rate:
        raise _v1_clone_error(
            f"reference WAV must be {expected_rate} Hz/{expected_channels} channel(s)",
            code="audio_contract_mismatch",
            param="file",
        )
    if block_align != channels * 2 or byte_rate != sample_rate * block_align:
        raise _v1_clone_error("reference WAV PCM metadata is invalid", code="invalid_audio", param="file")
    if not pcm or len(pcm) % block_align:
        raise _v1_clone_error("reference WAV data is not aligned to PCM frames", code="invalid_audio", param="file")
    return pcm, sample_rate


_V1_MANAGER_UNSET = object()


async def _v1_clone_stream_impl(
    request: Request,
    *,
    text: str,
    language: str | None,
    prepare,
    endpoint: str,
    preload=None,
    manager_override=_V1_MANAGER_UNSET,
):
    """Single-job clone stream with the native stream ownership contract.

    This intentionally shares the established disconnect watcher, executor
    drain and lease-release helpers with ``/tts/stream``.  A queued executor
    job is marked before backend access and is allowed to observe the shared
    cancellation event without retaining a manager/coordinator lease.
    """
    import asyncio as _asyncio
    import struct as _struct
    import threading as _threading
    from server.core import tts_service
    from server.core import metrics as _metrics
    from server.core.coordinator import get_coordinator
    from server.core.session_limiter import get_limiter
    from server.core.tts_backend import TTSCapability

    limiter = get_limiter()
    session_token = limiter.try_acquire() if limiter is not None else None
    if limiter is not None and session_token is None:
        snap = limiter.snapshot()
        try:
            _metrics.inc_sessions_rejected("http")
        except Exception:
            pass
        return _native_error_response(
            _v1_clone_error(
                "too many concurrent sessions",
                status_code=429,
                code="too_many_sessions",
                headers={"Retry-After": "5"},
            )
        )

    def release_session():
        if session_token is not None:
            session_token.release()

    manager_cm = None
    coordinator_cm = None
    resources_released = False
    cleanup_started = False
    try:
        # Multipart reference uploads are consumed only after the session
        # token is owned.  This keeps large decoded bodies within the same
        # admission budget as synthesis rather than bypassing it.
        if preload is not None:
            await preload()
        # OpenAI's speech adapter reuses this transport-neutral stream helper
        # after it has already resolved the active manager (to decide whether
        # the backend can stream).  Accepting an explicit override keeps that
        # path single-owner and avoids a second lazy-start/acquire decision;
        # clone routes retain the historical sentinel/default behavior.
        manager = (
            await _ensure_tts_manager_started()
            if manager_override is _V1_MANAGER_UNSET
            else manager_override
        )
        if manager is not None:
            manager_cm = manager.acquire()
            backend = await manager_cm.__aenter__()
        else:
            if not tts_service.is_ready():
                raise _v1_clone_error("TTS backend is not ready", status_code=503, code="backend_not_ready")
            backend = tts_service.get_backend()
        if not _v1_backend_has_capability(backend, TTSCapability.STREAMING):
            raise _v1_clone_error(
                "the active backend does not support streaming",
                status_code=501,
                code="streaming_unavailable",
            )
        stream_kwargs = dict(prepare(backend))
        try:
            stream_sample_rate = int(getattr(backend, "sample_rate"))
        except (TypeError, ValueError, AttributeError):
            stream_sample_rate = 0
        if stream_sample_rate <= 0:
            raise _v1_clone_error(
                "active backend has no valid sample-rate contract",
                status_code=503,
                code="audio_contract_unavailable",
            )
        coordinator_cm = get_coordinator().acquire("tts")
        await coordinator_cm.__aenter__()

        async def release_resources():
            nonlocal resources_released
            if resources_released:
                return
            resources_released = True
            if coordinator_cm is not None:
                try:
                    await coordinator_cm.__aexit__(None, None, None)
                except BaseException:
                    pass
            if manager_cm is not None:
                await _safe_cleanup_acquire_and_session(manager_cm, release_session)
            else:
                release_session()

        async def stream():
            nonlocal cleanup_started
            cancel_flag = _threading.Event()
            active_gens: list = []
            gen_lock = _threading.Lock()
            watcher_task = None
            executor_jobs: list = []
            loop = _asyncio.get_event_loop()
            queue: _asyncio.Queue = _asyncio.Queue()

            async def disconnect_watcher():
                try:
                    while not cancel_flag.is_set():
                        message = await request.receive()
                        if message.get("type") == "http.disconnect":
                            cancel_flag.set()
                            # A queued executor job has not touched the
                            # backend yet.  Wake the prefetching coroutine so
                            # it can enter its finally block immediately;
                            # _finish_tts_stream_cleanup intentionally skips
                            # waiting on this not-started future.
                            if not started.is_set():
                                loop.call_soon_threadsafe(queue.put_nowait, None)
                            with gen_lock:
                                generators = list(active_gens)
                            for generator in generators:
                                try:
                                    generator.close()
                                except Exception:
                                    logger.debug("clone stream generator close failed", exc_info=True)
                            return
                except _asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug("clone stream disconnect watcher failed", exc_info=True)

            started = _threading.Event()

            def run_backend():
                started.set()
                generator = None
                try:
                    if cancel_flag.is_set():
                        return
                    generator = backend.generate_streaming(
                        text,
                        language=language,
                        cancel_event=cancel_flag,
                        **stream_kwargs,
                    )
                    with gen_lock:
                        cancelled_before_register = cancel_flag.is_set()
                        if not cancelled_before_register:
                            active_gens.append(generator)
                    if cancelled_before_register:
                        try:
                            generator.close()
                        finally:
                            generator = None
                        return
                    for chunk in generator:
                        if cancel_flag.is_set():
                            # Keep draining until the backend observes its
                            # cancel event and reaches its terminal state.
                            continue
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except BaseException as exc:
                    if not isinstance(exc, (_asyncio.CancelledError, GeneratorExit)):
                        loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    if generator is not None:
                        try:
                            generator.close()
                        except Exception:
                            logger.debug("clone stream generator close failed", exc_info=True)
                        with gen_lock:
                            try:
                                active_gens.remove(generator)
                            except ValueError:
                                pass
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            try:
                watcher_task = _asyncio.create_task(disconnect_watcher())
                future = loop.run_in_executor(_get_tts_stream_executor(), run_backend)
                executor_jobs.append((future, started))
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    if isinstance(chunk, BaseException):
                        raise chunk
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise _v1_clone_error(
                            "TTS backend returned a non-binary PCM chunk",
                            status_code=503,
                            code="invalid_backend_audio",
                        )
                    pcm = bytes(chunk)
                    # Empty keepalive chunks are not audio and must not satisfy
                    # the pre-header prime.  Continue until real PCM, a
                    # terminal backend error, or end-of-stream is observed.
                    if not pcm:
                        continue
                    if len(pcm) % 2:
                        raise _v1_clone_error(
                            "TTS backend returned an unaligned PCM16 chunk",
                            status_code=503,
                            code="invalid_backend_audio",
                        )
                    yield pcm
            finally:
                cancel_flag.set()
                with gen_lock:
                    generators = list(active_gens)
                for generator in generators:
                    try:
                        generator.close()
                    except Exception:
                        logger.debug("clone stream generator close failed", exc_info=True)
                await _stop_tts_disconnect_watcher(watcher_task)
                cleanup_started = True
                await _finish_tts_stream_cleanup(executor_jobs, release_resources)

        # Prime one real PCM chunk before returning the response.  This keeps
        # backend startup/saturation failures on the JSON error path instead
        # of returning a successful four-byte header with no audio, and it
        # guarantees the manager/coordinator/session leases are already owned
        # by a live stream when the response is handed to Starlette.
        pcm_stream = stream()
        try:
            first_pcm = await pcm_stream.__anext__()
        except StopAsyncIteration:
            try:
                await pcm_stream.aclose()
            except BaseException:
                pass
            raise _v1_clone_error(
                "TTS backend returned no PCM chunks",
                status_code=503,
                code="clone_stream_start_failed",
            )
        except BaseException:
            try:
                await pcm_stream.aclose()
            except BaseException:
                pass
            raise

        async def framed_stream():
            try:
                yield _struct.pack("<I", stream_sample_rate)
                yield first_pcm
                async for chunk in pcm_stream:
                    yield chunk
            finally:
                try:
                    await pcm_stream.aclose()
                except BaseException:
                    pass

        return StreamingResponse(framed_stream(), media_type="application/octet-stream")
    except BaseException:
        if not resources_released and not cleanup_started:
            if coordinator_cm is not None:
                try:
                    await coordinator_cm.__aexit__(None, None, None)
                except BaseException:
                    pass
            if manager_cm is not None:
                try:
                    await manager_cm.__aexit__(None, None, None)
                except BaseException:
                    pass
            release_session()
        raise


async def _v1_clone_embedding_impl(req: V1CloneEmbeddingRequest, *, stream: bool, request: Request | None = None):
    from server.core import tts_service
    from server.core.api_execution import execute_tts_clone
    from server.core.coordinator import get_coordinator
    from server.core.session_limiter import acquire_http

    if not req.model:
        raise _v1_clone_error("model is required", code="missing_required_parameter", param="model")
    if not req.text:
        raise _v1_clone_error("text is required", code="missing_required_parameter", param="text")
    _v1_validate_text(req.text)
    embedding_holder: list[bytes | None] = [None]

    async def preload():
        # Decode and scan the embedding only after HTTP admission.  The
        # encoded-length preflight inside _v1_decode_embedding prevents a
        # second oversized allocation; the admission lease also bounds the
        # finite-value CPU scan under concurrent load.
        embedding_holder[0] = _v1_decode_embedding(req)

    def prepare(backend):
        active_model = _v1_backend_model(backend)
        _v1_check_model(req.model or "", active_model)
        _v1_require_clone_backend(backend, "embedding")
        embedding = embedding_holder[0]
        if embedding is None:
            raise _v1_clone_error("embedding was not decoded", code="invalid_profile", param="embedding_b64")
        kwargs = _v1_clone_control_kwargs(req, backend)
        kwargs["speaker_embedding"] = embedding
        return kwargs

    if stream:
        if request is None:
            raise RuntimeError("clone stream requires request")
        return await _v1_clone_stream_impl(
            request,
            text=req.text,
            language=req.language,
            prepare=prepare,
            endpoint="/v1/tts/clone/embedding/stream",
            preload=preload,
        )
    async with acquire_http("/v1/tts/clone/embedding"):
        await preload()
        manager = await _ensure_tts_manager_started()
        result = await execute_tts_clone(
            text=req.text,
            language=req.language,
            clone_kwargs={},
            manager=manager,
            legacy_service=tts_service,
            coordinator=get_coordinator(),
            prepare=prepare,
        )
    return Response(
        content=result.audio,
        media_type="audio/wav",
        headers=_tts_response_headers(result.metadata, backend=result.backend, mode=result.metadata.get("mode")),
    )


@app.post("/v1/tts")
async def v1_tts(req: NativeTTSRequest, _: None = Depends(_require_api_key), request: Request = None):
    """Strict native v1 TTS alias sharing legacy execution ownership."""
    from server.core.session_limiter import acquire_http

    try:
        _v1_validate_text(req.text)
        native_req = TTSRequest(
            text=req.text,
            speed=req.speed,
            pitch=req.pitch,
            language=req.language,
        )
        async with acquire_http("/v1/tts"):
            mgr = await _ensure_tts_manager_started()

            def _prepare(backend):
                active_model = _v1_backend_model(backend)
                _v1_check_model(req.model, active_model)
                _v1_validate_controls(req, backend)
                return _v1_resolve_voice_kwargs(req, backend)

            result = await _execute_tts_core(
                native_req,
                manager=mgr,
                prepare=_prepare,
                request=request,
            )
        try:
            from server.core import metrics as _m
            _m.record_tts_rtf(
                result.backend or "tts",
                float(result.metadata.get("rtf", 0) or 0),
            )
        except Exception:
            pass
        return Response(
            content=result.audio,
            media_type="audio/wav",
            headers=_tts_response_headers(result.metadata, backend=result.backend, mode=result.metadata.get("mode")),
        )
    except _TransportDisconnected:
        logger.info(
            "route=/v1/tts client disconnected; finite cancellation drained"
        )
        return Response(status_code=204)
    except Exception as exc:
        return _native_error_response(exc)


@app.get("/v1/tts/capabilities")
async def v1_tts_capabilities(_: None = Depends(_require_api_key)):
    """Versioned alias for the additive TTS capabilities response."""
    return await tts_capabilities(None)


@app.get("/v1/tts/speakers")
async def v1_tts_speakers(_: None = Depends(_require_api_key)):
    """Versioned alias for the active model's speaker catalog."""
    return await tts_speakers_list(None)


@app.post("/v1/tts/clone/embedding")
async def v1_tts_clone_embedding(
    req: V1CloneEmbeddingRequest,
    _: None = Depends(_require_api_key),
):
    """Qwen3-TTS Base clone using a strict float32 embedding payload."""
    try:
        return await _v1_clone_embedding_impl(req, stream=False)
    except Exception as exc:
        return _native_error_response(exc)


@app.post("/v1/tts/clone/embedding/stream")
async def v1_tts_clone_embedding_stream(
    req: V1CloneEmbeddingRequest,
    request: Request,
    _: None = Depends(_require_api_key),
):
    """Native framed PCM stream for Qwen3-TTS Base embedding clone."""
    try:
        return await _v1_clone_embedding_impl(req, stream=True, request=request)
    except Exception as exc:
        return _native_error_response(exc)


async def _v1_clone_reference_impl(
    file: UploadFile,
    *,
    model: str | None,
    text: str | None,
    language: str | None,
    speed: float | None,
    pitch: float | None,
    stream: bool,
    request: Request | None = None,
):
    from server.core import tts_service
    from server.core.api_execution import execute_tts_clone, read_bounded_upload
    from server.core.coordinator import get_coordinator
    from server.core.session_limiter import acquire_http

    if not model:
        raise _v1_clone_error("model is required", code="missing_required_parameter", param="model")
    if not text:
        raise _v1_clone_error("text is required", code="missing_required_parameter", param="text")
    _v1_validate_text(text)
    # The active MOSS codec contract is checked later inside the manager
    # lease.  Streaming routes fill this holder from ``preload`` after their
    # session token is acquired; non-streaming routes fill it inside their
    # acquire_http context below.
    raw_holder: list[bytes | None] = [None]

    async def preload():
        raw_holder[0] = await read_bounded_upload(
            file,
            max_bytes=_v1_limit("OVS_API_MAX_PROFILE_BYTES", 16 * 1024 * 1024),
        )

    control_req = V1CloneEmbeddingRequest(
        model=model,
        text=text,
        language=language,
        speed=speed,
        pitch=pitch,
    )

    def prepare(backend):
        active_model = _v1_backend_model(backend)
        _v1_check_model(model, active_model)
        _v1_require_clone_backend(backend, "reference_audio")
        if raw_holder[0] is None:
            raise _v1_clone_error("reference file was not read", code="invalid_audio", param="file")
        pcm, sample_rate = _v1_parse_moss_reference_wav(raw_holder[0], backend)
        kwargs = _v1_clone_control_kwargs(control_req, backend)
        if stream:
            kwargs.update(
                {
                    "ref_audio_b64": base64.b64encode(pcm).decode("ascii"),
                    "ref_audio_sample_rate": sample_rate,
                }
            )
        else:
            kwargs.update(
                {
                    "reference_audio": pcm,
                    "reference_sample_rate": sample_rate,
                }
            )
        return kwargs

    if stream:
        if request is None:
            raise RuntimeError("clone stream requires request")
        return await _v1_clone_stream_impl(
            request,
            text=text,
            language=language,
            prepare=prepare,
            endpoint="/v1/tts/clone/reference/stream",
            preload=preload,
        )
    async with acquire_http("/v1/tts/clone/reference"):
        await preload()
        manager = await _ensure_tts_manager_started()
        result = await execute_tts_clone(
            text=text,
            language=language,
            clone_kwargs={},
            manager=manager,
            legacy_service=tts_service,
            coordinator=get_coordinator(),
            prepare=prepare,
        )
    return Response(
        content=result.audio,
        media_type="audio/wav",
        headers=_tts_response_headers(result.metadata, backend=result.backend, mode=result.metadata.get("mode")),
    )


@app.post("/v1/tts/clone/reference")
async def v1_tts_clone_reference(
    file: UploadFile | None = File(None),
    model: str | None = Form(None),
    text: str | None = Form(None),
    language: str | None = Form(None),
    speed: float | None = Form(None),
    pitch: float | None = Form(None),
    _: None = Depends(_require_api_key),
):
    """MOSS clone from an exact PCM16 WAV codec contract."""
    try:
        if file is None:
            raise _v1_clone_error(
                "file is required",
                code="missing_required_parameter",
                param="file",
            )
        return await _v1_clone_reference_impl(
            file,
            model=model,
            text=text,
            language=language,
            speed=speed,
            pitch=pitch,
            stream=False,
        )
    except Exception as exc:
        return _native_error_response(exc)


@app.post("/v1/tts/clone/reference/stream")
async def v1_tts_clone_reference_stream(
    request: Request,
    file: UploadFile | None = File(None),
    model: str | None = Form(None),
    text: str | None = Form(None),
    language: str | None = Form(None),
    speed: float | None = Form(None),
    pitch: float | None = Form(None),
    _: None = Depends(_require_api_key),
):
    """Native framed PCM stream for MOSS reference-audio clone."""
    try:
        if file is None:
            raise _v1_clone_error(
                "file is required",
                code="missing_required_parameter",
                param="file",
            )
        return await _v1_clone_reference_impl(
            file,
            model=model,
            text=text,
            language=language,
            speed=speed,
            pitch=pitch,
            stream=True,
            request=request,
        )
    except Exception as exc:
        return _native_error_response(exc)


async def _safe_cleanup_acquire_and_session(acquire_cm, release_session_fn):
    """Codex round-4 GAP B: best-effort serial cleanup helper for TTS
    streaming endpoints.

    Both ``acquire_cm.__aexit__()`` and ``release_session_fn()`` must run
    even if one of them raises. The previous pattern wrote them on
    consecutive lines without protection — if ``__aexit__`` raised
    (BackendManager bug, GeneratorExit re-raised, etc.) the slot release
    was silently skipped. Using two independent try/except blocks
    guarantees both run.
    """
    try:
        await acquire_cm.__aexit__(None, None, None)
    except BaseException:
        pass
    try:
        release_session_fn()
    except BaseException:
        pass


_tts_stream_cleanup_tasks: set = set()
_tts_stream_watcher_tasks: set = set()
_tts_framed_stream_owners: set = set()


async def _tts_stream_queue_get(queue):
    """Read a worker bridge queue with a bounded wake-up heartbeat."""
    import asyncio

    while True:
        try:
            return queue.get_nowait()
        except asyncio.QueueEmpty:
            try:
                return await asyncio.wait_for(queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue


def _track_tts_stream_cleanup(task) -> None:
    """Keep a timed-out/cancelled stream cleanup alive until it releases its
    backend leases.

    A detached asyncio task is otherwise only weakly referenced by the event
    loop.  The strong-reference set also gives tests/health diagnostics a
    deterministic view of quarantined stream cleanups.
    """
    _tts_stream_cleanup_tasks.add(task)

    def _finish_cleanup_tracking(done_task):
        _tts_stream_cleanup_tasks.discard(done_task)
        if done_task.cancelled():
            return
        try:
            error = done_task.exception()
        except BaseException:
            return
        if error is not None:
            logger.error(
                "background TTS stream cleanup failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    task.add_done_callback(_finish_cleanup_tracking)


def _track_tts_stream_watcher(task) -> None:
    """Keep a late disconnect watcher alive without mixing it with cleanup.

    ``_tts_stream_cleanup_tasks`` is also used as a synchronization point by
    lifecycle tests and diagnostics.  A canceled ``receive()`` watcher is not
    backend cleanup and must not make that synchronization point contain a
    canceled future.
    """
    _tts_stream_watcher_tasks.add(task)
    task.add_done_callback(_tts_stream_watcher_tasks.discard)


async def _stop_tts_disconnect_watcher(watcher_task) -> None:
    """Stop a raw-ASGI disconnect watcher without delaying HTTP EOF.

    Uvicorn's ``request.receive()`` may be inside a cancellation-shielded wait.
    An unbounded ``await watcher_task`` then deadlocks normal stream completion:
    the client waits for the terminating HTTP chunk while the server waits for
    the client's eventual ``http.disconnect``.  Give cooperative cancellation
    a short window, then retain the task in the existing strong-reference set;
    it will observe the disconnect after EOF and remove itself.
    """
    import asyncio

    if watcher_task is None:
        return
    watcher_task.cancel()
    try:
        raw_wait_s = os.environ.get("OVS_TTS_DISCONNECT_WATCHER_WAIT_S", "0.1")
        wait_s = max(0.01, float(raw_wait_s))
    except ValueError:
        wait_s = 0.1
    try:
        await asyncio.wait_for(asyncio.shield(watcher_task), timeout=wait_s)
    except asyncio.TimeoutError:
        _track_tts_stream_watcher(watcher_task)
    except asyncio.CancelledError:
        # Either the watcher honored cancellation, or this stream's own ASGI
        # task is being cancelled.  In both cases do not delay teardown.
        if not watcher_task.done():
            _track_tts_stream_watcher(watcher_task)


async def _run_tts_stream_cleanup(executor_jobs, release_resources) -> None:
    """Drain started executor jobs before releasing all stream leases."""
    import asyncio

    async def _await_uncancellable(awaitable):
        """Finish cleanup work even if the owner task is cancelled again."""
        task = asyncio.ensure_future(awaitable)
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Cancellation belongs to the disconnect/ASGI owner.  The
                # cleanup task must retain leases until the backend future and
                # release callback have actually completed.
                continue
        try:
            return task.result()
        except asyncio.CancelledError:
            # Cancellation is expected only for an executor future or child
            # cleanup task explicitly canceled during stream teardown.
            return None

    async def _consume_job(future, finished=None):
        """Consume one possibly-canceled future without cancellation fanout."""
        try:
            # Cancellation of asyncio's run_in_executor wrapper does not
            # prove that its executor thread stopped.  For started jobs, wait
            # for the completion event set by _run's own finally block before
            # treating a cancelled/done wrapper as drained.
            while finished is not None and not finished.is_set():
                await asyncio.sleep(0.05)
            # Poll with a bounded timer instead of relying solely on the
            # executor thread's self-pipe wakeup. On quiet event loops the
            # thread can already be idle while its asyncio wrapper remains
            # pending until the next timer or I/O wake.
            while not future.done():
                await asyncio.wait(
                    {future}, timeout=0.05, return_when=asyncio.ALL_COMPLETED
                )
            future.result()
        except asyncio.CancelledError:
            # A canceled executor future is already drained.  Do not let that
            # expected cancellation fan out to the other cleanup jobs.
            try:
                future.exception()
            except asyncio.CancelledError:
                pass

    async def _drain_and_release():
        try:
            if executor_jobs:
                # A manager-branch stream can prefetch its next sentence into
                # the shared executor.  After a disconnect, a job that has
                # not started yet is harmless: its first action is to observe
                # the shared cancel flag and return without touching the
                # backend.  Waiting for such a queued job can nevertheless
                # retain the HTTP session lease behind unrelated long-running
                # streams for several seconds.
                #
                # Callers may therefore pass ``(future, started_event)`` or
                # ``(future, started_event, finished_event)`` records. Drain
                # jobs that have actually started, but do not hold leases for
                # untouched queue entries. There is no unsafe race here: a job
                # that starts after this snapshot sees the already-raised
                # cancel flag before it accesses the backend.
                jobs_to_drain = []
                for job in executor_jobs:
                    if isinstance(job, tuple):
                        if len(job) == 3:
                            future, started, finished = job
                            # A job that never started cannot touch the
                            # backend, including a queued wrapper cancelled
                            # before the executor accepts it.
                            if not started.is_set():
                                continue
                            jobs_to_drain.append((future, finished))
                            continue
                        future, started = job
                        if not started.is_set() and not future.done():
                            continue
                        jobs_to_drain.append((future, None))
                    else:
                        jobs_to_drain.append((job, None))
                if jobs_to_drain:
                    for future, finished in jobs_to_drain:
                        await _await_uncancellable(
                            _consume_job(future, finished)
                        )
        finally:
            await _await_uncancellable(release_resources())

    await _drain_and_release()


class _TTSStreamCleanupOwner:
    """Start one strongly-referenced cleanup independently of the ASGI task."""

    def __init__(self, executor_jobs, release_resources):
        self._executor_jobs = executor_jobs
        self._release_resources = release_resources
        self._task = None

    @property
    def started(self) -> bool:
        return self._task is not None

    def start(self):
        import asyncio

        if self._task is None:
            self._task = asyncio.create_task(
                _run_tts_stream_cleanup(
                    self._executor_jobs, self._release_resources
                )
            )
            _track_tts_stream_cleanup(self._task)
        return self._task

    async def wait(self, *, wait_s: float | None = None) -> bool:
        import asyncio

        if wait_s is None:
            try:
                wait_s = float(
                    os.environ.get("OVS_TTS_STREAM_CLEANUP_WAIT_S", "1.0")
                )
            except ValueError:
                wait_s = 1.0
        wait_s = max(0.01, wait_s)
        cleanup_task = self.start()
        try:
            await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=wait_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False
        cleanup_task.result()
        return True


class _TTSFramedStreamOwner:
    """Keep one inner stream reachable and drain it through one serial owner."""

    def __init__(self, inner_stream, request_cancel=None):
        self._inner_stream = inner_stream
        self._request_cancel = request_cancel
        self._cancel_requested = False
        self._current_task = None
        self._close_task = None
        # Register before the outer framed generator can become unreachable.
        # This prevents CPython from scheduling an independent finalizer for
        # the inner async generator while the outer finalizer starts cleanup.
        _tts_framed_stream_owners.add(self)

    async def next(self):
        import asyncio

        task = asyncio.create_task(self._inner_stream.__anext__())
        self._current_task = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and self._current_task is task:
                self._current_task = None

    def start_close(self):
        import asyncio

        # Tell the producer to unwind before serially draining it.  Cancelling
        # an in-flight __anext__ injects CancelledError/athrow into the async
        # generator.  Explicitly calling aclose() after that can also race the
        # async-generator finalizer when an abandoned response is collected.
        # Both TTS branches provide their thread-safe cancel Event.set callback
        # here, so the producer can reach its own terminal state and run its
        # finally blocks without either forced-close path.
        if not self._cancel_requested:
            self._cancel_requested = True
            if self._request_cancel is not None:
                self._request_cancel()

        if self._close_task is None:
            async def _drain_to_terminal():
                try:
                    task = self._current_task
                    while True:
                        if task is None:
                            try:
                                await self._inner_stream.__anext__()
                            except (StopAsyncIteration, asyncio.CancelledError):
                                return
                            except BaseException as exc:
                                if (
                                    self._cancel_requested
                                    and _is_kokoro_convonly_cancelled(exc)
                                ):
                                    return
                                raise
                            # The producer returned one last chunk after the
                            # cancel request; discard it and continue directly.
                            continue
                        while not task.done():
                            try:
                                await asyncio.shield(task)
                            except (StopAsyncIteration, asyncio.CancelledError):
                                continue
                            except BaseException:
                                # Retrieve and classify it after task.done().
                                continue
                        try:
                            task.result()
                        except (StopAsyncIteration, asyncio.CancelledError):
                            return
                        except BaseException as exc:
                            if (
                                self._cancel_requested
                                and _is_kokoro_convonly_cancelled(exc)
                            ):
                                return
                            raise
                        finally:
                            if self._current_task is task:
                                self._current_task = None
                        # A cooperative producer may have completed one final
                        # chunk while observing cancellation.  Discard it and
                        # keep sole ownership of subsequent __anext__ calls
                        # until the generator reaches its terminal state.
                        task = None
                finally:
                    _tts_framed_stream_owners.discard(self)

            self._close_task = asyncio.create_task(_drain_to_terminal())
            _track_tts_stream_cleanup(self._close_task)
        return self._close_task

    async def close(self) -> None:
        import asyncio

        had_inflight = self._current_task is not None
        task = self.start_close()
        if not had_inflight:
            # A disconnect can arrive while the outer generator is suspended
            # at a yielded PCM chunk.  Starting the next inner call is required
            # to let it observe cooperative cancellation, but a backend that
            # is already inside synchronous work must not hold ASGI teardown.
            # The tracked task and owner registry retain the entire cleanup;
            # its completion callback reports any non-cooperative error.
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                return
            if not task.done():
                try:
                    wait_s = max(
                        0.01,
                        float(
                            os.environ.get(
                                "OVS_TTS_STREAM_CLEANUP_WAIT_S", "1.0"
                            )
                        ),
                    )
                except ValueError:
                    wait_s = 1.0
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task), timeout=wait_s
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    return
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Repeated ASGI cancellation belongs to the framed response;
                # the strongly-referenced close owner must keep draining.
                continue
        task.result()


async def _framed_tts_stream(
    inner_stream,
    sample_rate: int,
    first_pcm: bytes,
    request_cancel=None,
):
    """Frame PCM while explicitly owning every in-flight inner next task."""
    owner = _TTSFramedStreamOwner(inner_stream, request_cancel)
    try:
        yield sample_rate.to_bytes(4, "little")
        yield first_pcm
        while True:
            try:
                chunk = await owner.next()
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        await owner.close()


async def _finish_tts_stream_cleanup(
    executor_jobs,
    release_resources,
    *,
    wait_s: float | None = None,
) -> bool:
    """Drain sync streaming jobs, then release coordinator/backend/session.

    The foreground wait is bounded.  If the ASGI task is cancelled again or a
    backend generator is stuck, the cleanup continues in a strongly-referenced
    background task which *retains* all admission/coordinator/backend leases.
    This quarantines the backend safely: reload/exclusive switching cannot
    unload it while the executor still uses it, and reconnects are rejected by
    normal admission or pre-header saturation rather than receiving empty PCM.

    Returns True when cleanup completed in the foreground, False when handed
    off to the background.
    """
    import asyncio

    if wait_s is None:
        try:
            wait_s = float(os.environ.get("OVS_TTS_STREAM_CLEANUP_WAIT_S", "1.0"))
        except ValueError:
            wait_s = 1.0
    wait_s = max(0.01, wait_s)

    cleanup_task = asyncio.create_task(
        _run_tts_stream_cleanup(executor_jobs, release_resources)
    )
    try:
        done, _pending = await asyncio.wait(
            {cleanup_task}, timeout=wait_s, return_when=asyncio.ALL_COMPLETED
        )
        if cleanup_task in done:
            cleanup_task.result()
            return True
    except asyncio.CancelledError:
        pass
    if not cleanup_task.done():
        _track_tts_stream_cleanup(cleanup_task)
        return False
    cleanup_task.result()
    return True


def _tts_stream_error_response(exc: BaseException):
    """Map a pre-header TTS streaming failure to an honest HTTP response."""
    saturated, max_slots = _is_pool_saturated(exc)
    if saturated:
        return JSONResponse(
            {
                "error": "tts_backend_busy",
                "detail": str(exc),
                "max_slots": max_slots,
            },
            status_code=429,
            headers={"Retry-After": "1"},
        )
    logger.exception("tts/stream failed before first PCM chunk", exc_info=exc)
    return JSONResponse(
        {"error": "tts_stream_start_failed", "detail": str(exc)},
        status_code=503,
        headers={"Retry-After": "1"},
    )


@app.options("/tts/stream")
async def tts_stream_options():
    return Response(status_code=200)


@app.post("/tts/stream")
async def tts_stream(
    req: TTSRequest,
    request: Request,
    _: None = Depends(_require_api_key),
):
    """Stream TTS as raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks."""
    import asyncio
    import struct
    import time
    from server.core import tts_service
    from server.core.tts_backend import TTSCapability
    from server.core.coordinator import get_coordinator
    from server.core.session_limiter import get_limiter
    from server.core import metrics as _metrics

    # Reject-not-queue: acquire session slot BEFORE setup work. Slot
    # ownership is handed to the StreamingResponse generator's finally
    # block so it releases on disconnect / exception / normal end.
    _sl = get_limiter()
    _session_token = None
    if _sl is not None:
        _session_token = _sl.try_acquire()
        if _session_token is None:
            snap = _sl.snapshot()
            _metrics.inc_sessions_rejected("http")
            return JSONResponse(
                {"error": "too_many_sessions",
                 "current": snap["active"], "limit": snap["limit"]},
                status_code=429,
                headers={"Retry-After": "5"},
            )

    def _release_session():
        if _session_token is not None:
            _session_token.release()

    # FIX_1+FIX_3: prefer the BackendManager path so /admin/backend/reload's
    # drain logic sees streaming requests in flight. Fall back to the legacy
    # tts_service path only when the manager isn't initialised (ASR-only).
    #
    # Codex MUST-FIX 1 (Week 4 round 2): catch BaseException so CancelledError
    # also releases the slot. Python 3.8+ CancelledError is a BaseException
    # subclass, not Exception, so `except Exception` would silently leak the
    # slot on client cancel mid-setup.
    try:
        mgr = await _ensure_tts_manager_started()

        # Capability gate uses tts_service so the response shape stays
        # consistent — both paths read the same underlying backend.
        if not tts_service.has_capability(TTSCapability.STREAMING):
            _release_session()
            return JSONResponse(
                {"error": "Streaming not supported by current backend",
                 "required_capability": "streaming"},
                status_code=501,
            )
    except BaseException:
        _release_session()
        raise

    # Sentence-level streaming: split the request text into sentences (via
    # pysbd when the language is supported, regex fallback otherwise) and
    # call the TTS backend per sentence.
    from server.core.v2v import SentenceBuffer
    sbuf = SentenceBuffer(language=req.language)
    sentences = list(sbuf.add(req.text or "")) + list(sbuf.flush())
    if not sentences:
        _release_session()

        async def _empty_text_header():
            yield struct.pack("<I", tts_service.get_sample_rate())

        return StreamingResponse(
            _empty_text_header(), media_type="application/octet-stream"
        )

    # Kokoro's high-performance mode overlaps the CPU tail of sentence N with
    # the NPU preparation of sentence N+1 inside one backend generator. Pass
    # the SentenceBuffer result as an explicit contract rather than asking the
    # backend to split raw text again; the original request text is preserved
    # for metadata and language-specific normalization. Every other path keeps
    # the existing one-job-per-sentence behavior.
    kokoro_hybrid_pipeline = _kokoro_hybrid_pipeline_enabled(len(sentences))
    backend_jobs = [req.text] if kokoro_hybrid_pipeline else sentences

    if mgr is not None:
        # Acquire OUTSIDE the generator so inflight_http is bumped synchronously
        # at endpoint entry — otherwise it would only increment when the client
        # starts iterating the StreamingResponse, and reload drain could miss it.
        # Codex MUST-FIX 1 (Week 4 round 2): wrap acquire_cm.__aenter__() so
        # if it raises (FAILED/DRAINING manager) the session slot is released.
        # Previously this await sat outside the try block.
        acquire_cm = mgr.acquire()
        try:
            backend = await acquire_cm.__aenter__()
        except BaseException:
            _release_session()
            raise
        _release_stream_resources = None
        cleanup_started = False
        cleanup_owner = None
        framed_cancel = None
        try:
            try:
                voice_kwargs = _request_voice_kwargs(req, backend=backend)
            except _VoiceCloneUnsupportedError as exc:
                # Bug 3 fix: pre-response 400 — better than 500 mid-stream
                # after StreamingResponse already wrote the sample-rate
                # header. Capability gate fires before any bytes go out.
                await _safe_cleanup_acquire_and_session(acquire_cm, _release_session)
                return JSONResponse(
                    _voice_clone_unsupported_payload(exc.backend),
                    status_code=400,
                )
            except ValueError as exc:
                # Codex round-4 GAP B: best-effort serial cleanup so
                # __aexit__ raising cannot skip _release_session().
                await _safe_cleanup_acquire_and_session(acquire_cm, _release_session)
                return JSONResponse({"error": str(exc)}, status_code=400)
            sr = backend.sample_rate
            coordinator_cm = get_coordinator().acquire("tts")
            try:
                await coordinator_cm.__aenter__()
            except BaseException:
                await _safe_cleanup_acquire_and_session(
                    acquire_cm, _release_session
                )
                raise
            resources_released = False

            async def _release_stream_resources():
                nonlocal resources_released
                if resources_released:
                    return
                resources_released = True
                errors = []
                try:
                    await coordinator_cm.__aexit__(None, None, None)
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    errors.append(exc)
                try:
                    await acquire_cm.__aexit__(None, None, None)
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    errors.append(exc)
                try:
                    _release_session()
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    errors.append(exc)
                if errors:
                    for extra in errors[1:]:
                        logger.error("additional TTS stream release failure", exc_info=extra)
                    raise errors[0]

            async def stream():
                nonlocal cleanup_started, cleanup_owner, framed_cancel
                # Part D disconnect watcher (spec §3): Starlette cancellation
                # does not reliably close the inner sync generator running in
                # _tts_stream_executor — so poll request.is_disconnected()
                # every 100 ms and explicitly close the generator on
                # disconnect. The for-loop break path in _run calls .close()
                # on the wrapped generator, which raises GeneratorExit into
                # _generate_streaming_single() and triggers
                # _WorkerIO.cancel(req_id) (trt_edge_llm_tts.py:1255-1269).
                #
                # [Sentence pipeline parallelism] Single-user multi-sentence
                # streaming used to be strictly serial: sentence N drains
                # before sentence N+1 is even submitted. Slot 1 of the worker
                # pool (sized to OVS_TTS_WORKER_CONCURRENCY=2) sat idle in
                # the typical single-client case. Now: submit a sliding
                # window of `prefetch` sentences and drain their chunk queues
                # in order. Chunk order on the wire is unchanged (sentence
                # N's chunks are yielded before sentence N+1's), so audio
                # MD5 is byte-identical to the serial baseline. The win is
                # wall-clock: while sentence N's audio is being yielded to
                # the client, sentence N+1's prefill + early decode is
                # already running on the second slot.
                import threading as _threading
                cancel_flag = _threading.Event()
                framed_cancel = cancel_flag.set
                # Active sync generators (one per in-flight sentence). The
                # disconnect watcher must close ALL of them so each
                # underlying _generate_streaming_single() receives
                # GeneratorExit and emits worker_io.cancel(req_id).
                active_gens: list = []
                gen_lock = _threading.Lock()
                watcher_task: asyncio.Task | None = None
                # Keep strong references to every executor job submitted for
                # this HTTP response.  Releasing the BackendManager/session
                # admission tokens before these jobs finish creates a false
                # "idle" window: a reconnect is admitted while the cancelled
                # worker request still owns its WorkerIO slot, and the new
                # stream is then reduced to HTTP 200 + empty PCM by the
                # saturation handling in _run().
                executor_jobs: list[
                    tuple[
                        asyncio.Future,
                        "_threading.Event",
                        "_threading.Event",
                    ]
                ] = []
                cleanup_owner = _TTSStreamCleanupOwner(
                    executor_jobs, _release_stream_resources
                )

                async def _disconnect_watcher():
                    nonlocal cleanup_started
                    # Directly drain the ASGI receive channel; Starlette's
                    # is_disconnected() uses a tight cancel-scope that often
                    # misses uvicorn's http.disconnect events under
                    # StreamingResponse on Python 3.10. Blocking on raw
                    # request.receive() is reliable: uvicorn pushes
                    # http.disconnect there as soon as the socket closes.
                    logger.info("tts/stream: disconnect watcher started")
                    try:
                        while not cancel_flag.is_set():
                            try:
                                message = await request.receive()
                            except Exception:
                                logger.debug(
                                    "disconnect watcher receive() failed",
                                    exc_info=True,
                                )
                                return
                            if message.get("type") == "http.disconnect":
                                cancel_flag.set()
                                # The raw receive watcher may consume the only
                                # disconnect before StreamingResponse cancels
                                # (or resumes) its body iterator.  Establish
                                # an independent cleanup owner before touching
                                # generators so leases cannot be orphaned.
                                cleanup_started = True
                                cleanup_owner.start()
                                # Snapshot the set under lock then close
                                # outside to avoid holding it during slow
                                # close() calls.
                                with gen_lock:
                                    gens = list(active_gens)
                                for g in gens:
                                    try:
                                        g.close()
                                    except Exception:
                                        logger.debug(
                                            "disconnect watcher gen.close() raised",
                                            exc_info=True,
                                        )
                                logger.info(
                                    "tts/stream: client disconnected — cancel flag raised (%d gens closed)",
                                    len(gens),
                                )
                                return
                    except asyncio.CancelledError:
                        pass

                try:
                    # The coordinator lease was entered before priming this
                    # generator and is released only after executor cleanup.
                    # Keep this block solely to preserve the pipeline's
                    # existing indentation.
                    if True:
                        if not sentences:
                            return
                        loop = asyncio.get_event_loop()
                        watcher_task = asyncio.create_task(_disconnect_watcher())
                        # Week 2: TTFA timer starts after admission (post sr
                        # header), observed once when the first real PCM
                        # chunk passes the boundary.
                        _ttfa_t0 = time.perf_counter()
                        _ttfa_recorded = False

                        # Pipeline window: max sentences in flight at once.
                        # Capped by the TTS stream executor size so we never
                        # block waiting for an executor slot.
                        # OVS_TTS_STREAM_PREFETCH overrides; default mirrors
                        # max_workers (2).
                        #
                        # CRITICAL: we do NOT pre-submit sentence 1 alongside
                        # sentence 0. If both prefills run simultaneously the
                        # GPU contention also hits sentence 0, so the TTFA
                        # of the very first chunk regresses (~520ms → ~920ms
                        # in early tests). Instead, sentence i+1 is submitted
                        # the moment sentence i emits its FIRST chunk — i.e.
                        # sentence i has cleared prefill and is in decode/
                        # Code2Wav, so its first audio is already on the way.
                        # This keeps sentence 0's TTFA at the single-sentence
                        # baseline while still overlapping sentence i+1's
                        # prefill with sentence i's decode.
                        executor = _get_tts_stream_executor()
                        # max(1, ...): a window of 0 would submit nothing
                        # after sentence 0 and deadlock the drain loop below.
                        prefetch_max = max(1, min(
                            int(os.environ.get(
                                "OVS_TTS_STREAM_PREFETCH",
                                str(executor._max_workers),
                            )),
                            len(backend_jobs),
                        ))

                        def _submit(idx: int, q: "asyncio.Queue[bytes | None]"):
                            text = backend_jobs[idx]
                            started = _threading.Event()
                            finished = _threading.Event()

                            def _run():
                                # Set this before inspecting cancel_flag so
                                # cleanup can distinguish an active backend
                                # user from a prefetch job still queued in the
                                # shared executor.
                                started.set()
                                gen = None
                                try:
                                    if cancel_flag.is_set():
                                        return
                                    stream_kwargs = {
                                        "language": req.language,
                                        "cancel_event": cancel_flag,
                                        **voice_kwargs,
                                    }
                                    if kokoro_hybrid_pipeline:
                                        stream_kwargs["segments"] = sentences
                                    gen = backend.generate_streaming(
                                        text, **stream_kwargs
                                    )
                                    with gen_lock:
                                        cancelled_before_register = (
                                            cancel_flag.is_set()
                                        )
                                        if not cancelled_before_register:
                                            active_gens.append(gen)
                                    if cancelled_before_register:
                                        gen.close()
                                        gen = None
                                        return
                                    for chunk in gen:
                                        if cancel_flag.is_set():
                                            # The backend has the same event
                                            # and sends an out-of-band worker
                                            # cancel while blocked in IPC.
                                            # Discard any final audio chunk,
                                            # but keep iterating until it
                                            # consumes the worker's terminal
                                            # cancelled/done event. Releasing
                                            # leases earlier creates a false
                                            # idle window while the C++ slot
                                            # is still occupied.
                                            continue
                                        loop.call_soon_threadsafe(q.put_nowait, chunk)
                                except Exception as _e:
                                    if (
                                        cancel_flag.is_set()
                                        and _is_kokoro_convonly_cancelled(_e)
                                    ):
                                        return
                                    # Slot-pool saturation is "backend busy",
                                    # not a synth fault: log at warning (no
                                    # stacktrace) and do NOT trigger a worker
                                    # restart. The HTTP response headers are
                                    # already in flight here, so the stream
                                    # ends gracefully (empty/short audio) rather
                                    # than a 429 — the WS paths surface 4429.
                                    _sat, _ms = _is_pool_saturated(_e)
                                    if _sat:
                                        logger.warning(
                                            "tts/stream slot-pool saturated for "
                                            "sentence=%r (max_slots=%s)",
                                            text[:80], _ms,
                                        )
                                    else:
                                        logger.exception(
                                            "tts/stream synthesis failed for sentence=%r",
                                            text,
                                        )
                                    loop.call_soon_threadsafe(q.put_nowait, _e)
                                finally:
                                    try:
                                        if gen is not None:
                                            try:
                                                gen.close()
                                            except Exception:
                                                logger.debug(
                                                    "gen.close() in _run raised",
                                                    exc_info=True,
                                                )
                                            with gen_lock:
                                                try:
                                                    active_gens.remove(gen)
                                                except ValueError:
                                                    pass
                                        loop.call_soon_threadsafe(
                                            q.put_nowait, None
                                        )
                                    finally:
                                        finished.set()

                            executor_jobs.append((
                                loop.run_in_executor(executor, _run),
                                started,
                                finished,
                            ))

                        # Allocate queues. Submit ONLY sentence 0 to start —
                        # sentence 1+ will be submitted as sentence i emits
                        # its first chunk (see comment above for rationale).
                        queues: list[
                            asyncio.Queue[bytes | BaseException | None]
                        ] = [
                            asyncio.Queue() for _ in range(len(backend_jobs))
                        ]
                        next_to_submit = 1
                        _submit(0, queues[0])

                        def _maybe_prefetch(*, after_completion: bool = False):
                            nonlocal next_to_submit
                            if (
                                next_to_submit < len(backend_jobs)
                                and not cancel_flag.is_set()
                                and _prefetch_window_allows(
                                    next_to_submit, current_idx, prefetch_max
                                )
                            ):
                                _submit(next_to_submit, queues[next_to_submit])
                                next_to_submit += 1

                        # Drain in order. Submit sentence i+1 as soon as
                        # sentence i emits its first audio chunk.
                        for current_idx in range(len(backend_jobs)):
                            if cancel_flag.is_set():
                                break
                            q = queues[current_idx]
                            first_chunk_seen = False
                            while True:
                                chunk = await _tts_stream_queue_get(q)
                                if chunk is None:
                                    break
                                if isinstance(chunk, BaseException):
                                    raise chunk
                                if not first_chunk_seen:
                                    _maybe_prefetch()
                                    first_chunk_seen = True
                                    if not _ttfa_recorded:
                                        try:
                                            from server.core import metrics as _m2
                                            _m2.record_tts_ttfa(
                                                getattr(backend, "name", "tts"),
                                                time.perf_counter() - _ttfa_t0,
                                            )
                                        except Exception:
                                            pass
                                        _ttfa_recorded = True
                                yield chunk
                            # Also try after sentence completes, in case it
                            # produced zero chunks (degenerate path).
                            _maybe_prefetch(after_completion=True)
                finally:
                    # StreamingResponse cancellation does not guarantee that
                    # the raw ASGI disconnect watcher wins the race. Always
                    # raise the shared flag here as well. Executor workers
                    # observe it after their next worker chunk, break the
                    # sync-generator loop and close the backend generator,
                    # which sends the cooperative WorkerIO cancel request.
                    # Without this, a TTFA-only/disconnected HTTP client can
                    # leave full synthesis running invisibly and occupy a
                    # worker slot, making the next N=2 pair look serialized.
                    cancel_flag.set()
                    cleanup_started = True
                    cleanup_owner.start()
                    await _stop_tts_disconnect_watcher(watcher_task)
                    await cleanup_owner.wait()

            pcm_stream = stream()
            try:
                first_pcm = await pcm_stream.__anext__()
            except StopAsyncIteration:
                return _tts_stream_error_response(
                    RuntimeError("TTS backend returned no PCM chunks")
                )
            except BaseException as exc:
                try:
                    await pcm_stream.aclose()
                except BaseException:
                    pass
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return _tts_stream_error_response(exc)

            return StreamingResponse(
                _framed_tts_stream(
                    pcm_stream, sr, first_pcm, framed_cancel
                ),
                media_type="application/octet-stream",
            )
        except BaseException:
            # MUST-FIX 1 round 2: cover CancelledError (BaseException) too.
            # MUST-FIX 1 round 3: each cleanup must be best-effort so a
            # failing __aexit__ / release cannot mask the original
            # exception or short-circuit subsequent cleanups.
            if _release_stream_resources is not None and not cleanup_started:
                await _release_stream_resources()
            elif _release_stream_resources is None:
                try:
                    await acquire_cm.__aexit__(None, None, None)
                except BaseException:
                    pass
                try:
                    _release_session()
                except BaseException:
                    pass
            raise

    # Manager not initialised — legacy direct-backend path.
    backend = tts_service.get_backend()
    sr = tts_service.get_sample_rate()
    try:
        voice_kwargs = _request_voice_kwargs(req, backend=backend)
    except _VoiceCloneUnsupportedError as exc:
        # Bug 3 fix: pre-response capability gate on the legacy path too.
        _release_session()
        return JSONResponse(
            _voice_clone_unsupported_payload(exc.backend),
            status_code=400,
        )
    except ValueError as exc:
        _release_session()
        return JSONResponse({"error": str(exc)}, status_code=400)

    legacy_coordinator_cm = get_coordinator().acquire("tts")
    try:
        await legacy_coordinator_cm.__aenter__()
    except BaseException:
        _release_session()
        raise
    legacy_resources_released = False

    async def _release_legacy_stream_resources():
        nonlocal legacy_resources_released
        if legacy_resources_released:
            return
        legacy_resources_released = True
        errors = []
        try:
            await legacy_coordinator_cm.__aexit__(None, None, None)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            errors.append(exc)
        try:
            _release_session()
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            errors.append(exc)
        if errors:
            for extra in errors[1:]:
                logger.error("additional legacy TTS stream release failure", exc_info=extra)
            raise errors[0]

    framed_cancel = None

    async def stream_legacy():
        nonlocal framed_cancel
        # Part D disconnect watcher — mirrors the manager-branch logic above.
        import threading as _threading
        cancel_flag = _threading.Event()
        framed_cancel = cancel_flag.set
        gen_holder: list = [None]
        gen_lock = _threading.Lock()
        watcher_task: asyncio.Task | None = None
        executor_jobs: list[
            tuple[asyncio.Future, "_threading.Event", "_threading.Event"]
        ] = []
        cleanup_owner = _TTSStreamCleanupOwner(
            executor_jobs, _release_legacy_stream_resources
        )

        async def _disconnect_watcher():
            logger.info("tts/stream (legacy): disconnect watcher started")
            try:
                while not cancel_flag.is_set():
                    try:
                        message = await request.receive()
                    except Exception:
                        logger.debug(
                            "legacy disconnect watcher receive() failed",
                            exc_info=True,
                        )
                        return
                    if message.get("type") == "http.disconnect":
                        cancel_flag.set()
                        cleanup_owner.start()
                        with gen_lock:
                            g = gen_holder[0]
                        if g is not None:
                            try:
                                g.close()
                            except Exception:
                                logger.debug(
                                    "legacy disconnect watcher gen.close() raised",
                                    exc_info=True,
                                )
                        logger.info(
                            "tts/stream (legacy): client disconnected — cancel flag raised"
                        )
                        return
            except asyncio.CancelledError:
                pass

        try:
            if True:
                loop = asyncio.get_event_loop()
                watcher_task = asyncio.create_task(_disconnect_watcher())
                for sentence in backend_jobs:
                    if cancel_flag.is_set():
                        break
                    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
                    started = _threading.Event()
                    finished = _threading.Event()

                    def _run(text=sentence):
                        started.set()
                        gen = None
                        try:
                            if cancel_flag.is_set():
                                return
                            stream_kwargs = {
                                "language": req.language,
                                "cancel_event": cancel_flag,
                                **voice_kwargs,
                            }
                            if kokoro_hybrid_pipeline:
                                stream_kwargs["segments"] = sentences
                            gen = backend.generate_streaming(
                                text, **stream_kwargs
                            )
                            with gen_lock:
                                cancelled_before_register = cancel_flag.is_set()
                                if not cancelled_before_register:
                                    gen_holder[0] = gen
                            if cancelled_before_register:
                                gen.close()
                                gen = None
                                return
                            for chunk in gen:
                                if cancel_flag.is_set():
                                    # Drain to the real worker terminal while
                                    # dropping post-disconnect PCM; see the
                                    # manager branch above.
                                    continue
                                loop.call_soon_threadsafe(queue.put_nowait, chunk)
                        except Exception as exc:
                            if (
                                cancel_flag.is_set()
                                and _is_kokoro_convonly_cancelled(exc)
                            ):
                                return
                            saturated, max_slots = _is_pool_saturated(exc)
                            if saturated:
                                logger.warning(
                                    "tts/stream legacy slot-pool saturated "
                                    "for sentence=%r (max_slots=%s)",
                                    text[:80], max_slots,
                                )
                            else:
                                logger.exception(
                                    "tts/stream legacy synthesis failed "
                                    "for sentence=%r",
                                    text,
                                )
                            loop.call_soon_threadsafe(
                                queue.put_nowait, exc
                            )
                        finally:
                            try:
                                if gen is not None:
                                    try:
                                        gen.close()
                                    except Exception:
                                        logger.debug(
                                            "legacy gen.close() in _run raised",
                                            exc_info=True,
                                        )
                                with gen_lock:
                                    gen_holder[0] = None
                                loop.call_soon_threadsafe(
                                    queue.put_nowait, None
                                )
                            finally:
                                finished.set()

                    executor_jobs.append(
                        (
                            loop.run_in_executor(
                                _get_tts_stream_executor(), _run
                            ),
                            started,
                            finished,
                        )
                    )

                    while True:
                        chunk = await _tts_stream_queue_get(queue)
                        if chunk is None:
                            break
                        if isinstance(chunk, BaseException):
                            raise chunk
                        yield chunk
        finally:
            # Mirror the manager path: endpoint cancellation must propagate
            # to the sync generator even when request.receive() misses the
            # http.disconnect event.
            cancel_flag.set()
            cleanup_owner.start()
            await _stop_tts_disconnect_watcher(watcher_task)
            await cleanup_owner.wait()

    pcm_stream = stream_legacy()
    try:
        first_pcm = await pcm_stream.__anext__()
    except StopAsyncIteration:
        return _tts_stream_error_response(
            RuntimeError("TTS backend returned no PCM chunks")
        )
    except BaseException as exc:
        try:
            await pcm_stream.aclose()
        except BaseException:
            pass
        if isinstance(exc, asyncio.CancelledError):
            raise
        return _tts_stream_error_response(exc)

    return StreamingResponse(
        _framed_tts_stream(
            pcm_stream, sr, first_pcm, framed_cancel
        ),
        media_type="application/octet-stream",
    )


# ── Voice Clone ───��──────────────────────────────────────────────


def _voice_clone_unsupported_response():
    """Build the unified 400/501 JSON response for backends without voice clone.

    Returns a tuple ``(response, supports_clone)`` where ``supports_clone`` is
    True iff the active backend advertises VOICE_CLONE capability. Callers
    early-return the response when supports_clone is False.
    """
    from server.core import tts_service
    from server.core.tts_backend import TTSCapability

    if tts_service.has_capability(TTSCapability.VOICE_CLONE):
        return None, True

    backend = tts_service.get_backend() if tts_service.is_ready() else None
    supports_clone = getattr(backend, "supports_voice_cloning", None)
    if supports_clone is False:
        msg = (
            f"Current TTS backend ({tts_service.backend_name()}) does not support voice "
            "cloning. Switch to MOSS or another clone-capable backend, or use a built-in "
            "speaker_id via /tts."
        )
        return JSONResponse(
            {"error": msg,
             "required_capability": "voice_clone",
             "backend": tts_service.backend_name(),
             "supports_voice_cloning": False},
            status_code=400,
        ), False
    return JSONResponse(
        {"error": "Voice cloning not supported by current backend",
         "required_capability": "voice_clone",
         "backend": tts_service.backend_name()},
        status_code=501,
    ), False


@app.post("/tts/clone")
async def tts_clone(req: CloneRequest, _: None = Depends(_require_api_key)):
    """Synthesize with voice cloning. Requires voice_clone capability."""
    import base64
    from server.core import tts_service
    from server.core.tts_backend import TTSCapability
    from server.core.coordinator import get_coordinator
    from server.core.session_limiter import acquire_http

    async with acquire_http("/tts/clone"):
        return await _tts_clone_impl(req)


async def _tts_clone_impl(req: CloneRequest):
    import base64
    from server.core import tts_service
    from server.core.tts_backend import TTSCapability
    from server.core.coordinator import get_coordinator

    unsupported, _ok = _voice_clone_unsupported_response()
    if unsupported is not None:
        return unsupported

    try:
        speaker_embedding = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    # FIX_1: route through manager.acquire() so reload drain sees this request.
    mgr = await _ensure_tts_manager_started()
    if mgr is not None:
        async with mgr.acquire() as backend:
            async with get_coordinator().acquire("tts"):
                wav_bytes, meta = backend.clone_voice(
                    text=req.text,
                    speaker_embedding=speaker_embedding,
                    language=req.language,
                )
    else:
        async with get_coordinator().acquire("tts"):
            wav_bytes, meta = tts_service.clone_voice(
                text=req.text,
                speaker_embedding=speaker_embedding,
                language=req.language,
            )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers=_tts_response_headers(meta, backend="kokoro_rknn" if meta.get("mode") == "long32" else None, mode=meta.get("mode")),
    )


@app.post("/tts/clone/embedding")
async def tts_extract_embedding(
    file: UploadFile = File(...),
    _: None = Depends(_require_api_key),
):
    """Extract speaker embedding from reference audio WAV.

    Returns base64-encoded speaker embedding that can be reused
    across multiple /tts/clone calls.
    """
    import base64
    from server.core import tts_service
    from server.core.tts_backend import TTSCapability
    from server.core.session_limiter import acquire_http

    unsupported, _ok = _voice_clone_unsupported_response()
    if unsupported is not None:
        return unsupported

    async with acquire_http("/tts/clone/embedding"):
        audio_bytes = await file.read()
        from server.core.coordinator import get_coordinator
        async with get_coordinator().acquire("tts"):
            embedding = tts_service.extract_speaker_embedding(audio_bytes)
        return {
            "speaker_embedding_b64": base64.b64encode(embedding).decode(),
            "embedding_size": len(embedding),
        }


@app.post("/tts/clone/stream")
async def tts_clone_stream(
    req: CloneStreamRequest,
    _: None = Depends(_require_api_key),
):
    """Stream TTS with voice cloning.

    Returns raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks.
    Requires voice_clone capability.
    """
    import asyncio
    import struct
    import base64
    from server.core import tts_service
    from server.core.tts_backend import TTSCapability
    from server.core.session_limiter import get_limiter
    from server.core import metrics as _metrics

    unsupported, _ok = _voice_clone_unsupported_response()
    if unsupported is not None:
        return unsupported

    if not tts_service.has_capability(TTSCapability.STREAMING):
        return JSONResponse(
            {"error": "Streaming not supported by current backend",
             "required_capability": "streaming"},
            status_code=501,
        )

    try:
        speaker_embedding = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    # Reject-not-queue admission gate. Slot lifetime spans the entire
    # streaming response — release happens in the generator finally.
    _sl = get_limiter()
    _session_token = None
    if _sl is not None:
        _session_token = _sl.try_acquire()
        if _session_token is None:
            snap = _sl.snapshot()
            _metrics.inc_sessions_rejected("http")
            return JSONResponse(
                {"error": "too_many_sessions",
                 "current": snap["active"], "limit": snap["limit"]},
                status_code=429,
                headers={"Retry-After": "5"},
            )

    def _release_session():
        if _session_token is not None:
            _session_token.release()

    from server.core.coordinator import get_coordinator

    stream_kwargs: dict = {
        "speaker_embedding": speaker_embedding,
        "language": req.language,
    }
    if req.first_chunk_frames is not None:
        stream_kwargs["first_chunk_frames"] = req.first_chunk_frames
    if req.chunk_frames is not None:
        stream_kwargs["chunk_frames"] = req.chunk_frames
    if req.streaming_profile is not None:
        stream_kwargs["streaming_profile"] = req.streaming_profile

    # FIX_1: enter manager.acquire() at endpoint scope so reload drain
    # observes the inflight streaming request immediately.
    #
    # Codex MUST-FIX 1: lazy-start can raise — release the just-acquired
    # session slot rather than leaking it on FAILED/DRAINING manager.
    try:
        mgr = await _ensure_tts_manager_started()
    except BaseException:
        # MUST-FIX 1 round 2: cover CancelledError (BaseException) too.
        _release_session()
        raise
    if mgr is not None:
        acquire_cm = mgr.acquire()
        try:
            backend = await acquire_cm.__aenter__()
        except BaseException:
            _release_session()
            raise
        try:
            sr = backend.sample_rate

            async def stream():
                try:
                    async with get_coordinator().acquire("tts"):
                        yield struct.pack("<I", sr)
                        loop = asyncio.get_event_loop()
                        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

                        def _run():
                            try:
                                for chunk in backend.generate_streaming(req.text, **stream_kwargs):
                                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                            except Exception:
                                logger.exception("tts/clone/stream synthesis failed")
                            finally:
                                loop.call_soon_threadsafe(queue.put_nowait, None)

                        loop.run_in_executor(_get_tts_stream_executor(), _run)
                        while True:
                            chunk = await queue.get()
                            if chunk is None:
                                break
                            yield chunk
                finally:
                    # Codex round-4 GAP B: best-effort serial cleanup so
                    # __aexit__ raising cannot skip _release_session().
                    await _safe_cleanup_acquire_and_session(acquire_cm, _release_session)

            return StreamingResponse(stream(), media_type="application/octet-stream")
        except BaseException:
            # MUST-FIX 1 round 2: cover CancelledError (BaseException) too.
            # MUST-FIX 1 round 3: best-effort cleanups so neither
            # __aexit__ nor _release_session can mask the original
            # exception or skip the other release path.
            try:
                await acquire_cm.__aexit__(None, None, None)
            except BaseException:
                pass
            try:
                _release_session()
            except BaseException:
                pass
            raise

    # Legacy fallback (manager not initialised).
    sr = tts_service.get_sample_rate()
    backend = tts_service.get_backend()

    async def stream_legacy():
        try:
            async with get_coordinator().acquire("tts"):
                yield struct.pack("<I", sr)
                loop = asyncio.get_event_loop()
                queue: asyncio.Queue[bytes | None] = asyncio.Queue()

                def _run():
                    try:
                        for chunk in backend.generate_streaming(req.text, **stream_kwargs):
                            loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    except Exception:
                        logger.exception("tts/clone/stream synthesis failed")
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                loop.run_in_executor(_get_tts_stream_executor(), _run)

                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
        finally:
            _release_session()

    return StreamingResponse(stream_legacy(), media_type="application/octet-stream")


# ── ASR ──────────────────────────────────────────────────────────

@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Query("auto"),
    _: None = Depends(_require_api_key),
):
    from server.core.session_limiter import acquire_http
    async with acquire_http("/asr"):
        try:
            return await _asr_impl(file, language)
        except Exception as exc:
            saturated, max_slots = _is_pool_saturated(exc)
            if not saturated:
                raise
            try:
                from server.core import metrics as _metrics
                _metrics.inc_sessions_rejected("http")
            except Exception:
                pass
            return JSONResponse(
                status_code=429,
                content={
                    "error": "pool_saturated",
                    "status": 4429,
                    "max_slots": max_slots,
                },
            )


async def _asr_impl(file: UploadFile, language: str):
    audio_bytes = await file.read()
    try:
        result = await _execute_asr_core(audio_bytes, language)
    except Exception as exc:
        from server.core.api_execution import BackendNotReadyError
        if isinstance(exc, BackendNotReadyError):
            return JSONResponse(
                status_code=503,
                content={"error": "ASR backend not available"},
            )
        raise
    return {
        "text": result.text,
        "language": result.language,
        "backend": result.backend,
        **dict(result.metadata),
    }


async def _execute_asr_core(audio_bytes: bytes, language: str, *, prepare=None, manager_override=None):
    """Run the shared transport-neutral non-streaming ASR execution core."""
    from server.core.api_execution import execute_asr
    from server.core.coordinator import get_coordinator
    from server.core import metrics as _metrics

    installed_mgr = _get_asr_manager() if manager_override is None else manager_override
    if installed_mgr is not None and not installed_mgr.is_ready():
        from server.core.api_execution import BackendNotReadyError
        raise BackendNotReadyError("asr")
    mgr = installed_mgr if installed_mgr is not None else None
    asr_be = _get_asr_backend() if mgr is None else None
    return await execute_asr(
        audio=audio_bytes,
        language=language,
        manager=mgr,
        legacy_backend=asr_be,
        coordinator=get_coordinator(),
        metrics_module=_metrics,
        prepare=prepare,
    )


def _v1_active_asr_model(backend: object) -> str:
    from server.core.api_capabilities import canonical_asr_model_id
    value = getattr(backend, "model_id", None)
    if not value:
        try:
            from server.core.profile_loader import current_profile
            value = (current_profile() or {}).get("asr_model_id")
        except Exception:
            value = None
    value = value or os.environ.get("OVS_ASR_MODEL_ID") or getattr(backend, "name", None)
    return canonical_asr_model_id(str(value or "asr"))


@app.post("/v1/asr")
async def v1_asr(
    file: UploadFile = File(...),
    model: str = Query(...),
    language: str = Query("auto"),
    _: None = Depends(_require_api_key),
):
    """Strict native v1 ASR alias with a decoded-byte bounded upload reader."""
    from server.core.api_execution import APIExecutionError, read_bounded_upload
    from server.core.session_limiter import acquire_http

    try:
        async with acquire_http("/v1/asr"):
            manager = _get_asr_manager()
            audio = await read_bounded_upload(
                file,
                max_bytes=_v1_limit("OVS_API_MAX_AUDIO_BYTES", 32 * 1024 * 1024),
            )

            def _prepare(backend):
                active_model = _v1_active_asr_model(backend)
                from server.core.api_capabilities import canonical_asr_model_id
                if canonical_asr_model_id(model) != active_model:
                    raise APIExecutionError(
                        f"model {model!r} is not the active ASR model",
                        status_code=404,
                        code="unknown_model",
                        param="model",
                    )

            result = await _execute_asr_core(
                audio,
                language,
                prepare=_prepare,
                manager_override=manager,
            )
        return {
            "text": result.text,
            "language": result.language,
            "backend": result.backend,
            **dict(result.metadata),
        }
    except Exception as exc:
        return _native_error_response(exc)


# ── Punctuation (optional, opt-in, stateless) ───────────────────────

class PunctuateRequest(BaseModel):
    text: str


@app.post("/punctuate")
async def punctuate(req: PunctuateRequest, _: None = Depends(_require_api_key)):
    """Restore punctuation on a text string (CT-Transformer).

    Offline counterpart to the ``?punctuate=true`` streaming flag. Stateless,
    pure text-in / text-out. Returns the original text unchanged if the model
    is unavailable.
    """
    from server.core import punctuation as _punct
    text = await asyncio.get_running_loop().run_in_executor(
        None, _punct.add_punctuation, req.text
    )
    return {"text": text, "model": _punct.PUNCT_MODEL_NAME}


# ── Speaker embedding (optional, opt-in, stateless) ─────────────────

@app.post("/speaker/embedding")
async def speaker_embedding(
    file: UploadFile = File(...),
    sample_rate: int = Query(16000),
    _: None = Depends(_require_api_key),
):
    """Extract a speaker-embedding vector (CAM++) from an audio blob.

    Accepts a PCM16 WAV or raw int16 PCM (``?sample_rate=`` for the latter).
    Returns ``{embedding_b64, embedding_model, dim, normalized}``. OVS only
    emits the vector — matching / identity is the consumer's job. Returns 503
    if the model is unavailable.
    """
    from server.core import speaker_embedding as _spk
    from server.core.session_limiter import acquire_http

    audio_bytes = await file.read()
    async with acquire_http("/speaker/embedding"):
        loop = asyncio.get_running_loop()

        def _run():
            samples = _spk.decode_audio_to_16k_mono(audio_bytes, fallback_sr=sample_rate)
            return _spk.compute_embedding(samples, 16000)

        emb = await loop.run_in_executor(None, _run)
    if emb is None:
        return JSONResponse(
            status_code=503,
            content={"error": "speaker embedding unavailable (model not loaded or audio too short)"},
        )
    payload = _spk.embedding_payload(emb)
    # Endpoint contract uses embedding_b64; the streaming final uses
    # speaker_embedding for the same value (see stream handlers).
    payload["embedding_b64"] = payload.pop("speaker_embedding")
    return payload


# ── Diarization (optional, opt-in, blind clustering) ────────────────

@app.post("/diarize")
async def diarize(
    file: UploadFile = File(...),
    sample_rate: int = Query(16000),
    num_speakers: Optional[int] = Query(None),
    return_embeddings: bool = Query(False),
    _: None = Depends(_require_api_key),
):
    """Offline blind speaker diarization of a (possibly multi-speaker) clip.

    Accepts a PCM16 WAV or raw int16 PCM (``?sample_rate=`` for the latter).
    Internally: VAD/energy-segment → CAM++ embedding per segment → numpy
    agglomerative clustering. ``?num_speakers=`` pins the cluster count when
    known; ``?return_embeddings=true`` attaches per-segment vectors. Returns
    spec §4.2: ``{num_speakers, segments:[{start,end,speaker,confidence}],
    embedding_model, dim}``. Blind clustering only — identification (mapping
    ``spk_N`` → a name) is the consumer's responsibility (default off).
    """
    from server.core import diarization as _diar
    from server.core import speaker_embedding as _spk
    from server.core.session_limiter import acquire_http

    audio_bytes = await file.read()
    async with acquire_http("/diarize"):
        loop = asyncio.get_running_loop()

        def _run():
            samples = _spk.decode_audio_to_16k_mono(audio_bytes, fallback_sr=sample_rate)
            return _diar.diarize_audio(samples, 16000, num_speakers=num_speakers)

        segments = await loop.run_in_executor(None, _run)
    return _diar.diarize_response(segments, return_embeddings=return_embeddings)


@app.websocket("/asr/stream")
async def asr_stream(
    ws: WebSocket,
    language: str = "auto",
    sample_rate: int = 16000,
    vad: Optional[str] = None,           # default from OVS_VAD_BACKEND
    vad_silence_ms: Optional[int] = None,
    punctuate: Optional[str] = None,          # default from OVS_PUNCT
    speaker_embedding: Optional[str] = None,  # default from OVS_SPEAKER_EMB
    diarize: Optional[str] = None,            # default from OVS_DIARIZE
):
    """Streaming ASR via WebSocket.

    Client sends: raw int16 PCM bytes
    Client sends: empty bytes b"" to signal end
    Server sends: JSON {"text": "...", "is_final": bool, "is_stable": bool}

    ENDPOINTING: PICK EXACTLY ONE DETECTOR. The two modes below are mutually
    exclusive; running both is the most expensive misconfiguration in this API
    because it truncates transcripts without logging anything on either side.
    See docs/CONFIGURATION.md "Streaming ASR endpointing".

      * Open-mic (server decides). ``?vad_silence_ms=900`` — server VAD
        auto-finalizes on silence, emits a per-segment final, then swaps in a
        fresh ASR stream and keeps the socket open. The client never sends the
        EOS frame. Default silence threshold 400 ms. For clients that are a
        dumb audio pipe (browser demo, live caption).

      * Client-driven (client decides). ``?vad=none`` — no server VAD is
        created; the server finalizes ONLY on the empty b"" frame. For clients
        that already run their own VAD, which usually know things the server
        cannot (wake word just fired, device is playing TTS, session state).

    Running both: the detectors race. When the server wins it starts a NEW
    segment on its side while the client still believes one utterance is in
    progress; from then on the two disagree about which segment is current and
    text goes missing — the head, the tail, or nearly everything, depending on
    whether the client honours, overwrites, or accumulates the server's
    mid-utterance final. Raising ``vad_silence_ms`` does not fix this, it only
    makes the server lose the race more often.

    A VAD-capable client that must stay in open-mic mode has to ACCUMULATE every
    final the server sends — each is a complete segment and the server has
    already moved on. ``{"type":"vad_endpoint"}`` is emitted before the finalize
    compute, so it is also the right signal for flipping client UI state.

    Optional, default-off enrichments applied to the *final* payload only:
    ``?punctuate=true`` inlines punctuation into final.text; ``?speaker_embedding=true``
    adds {speaker_embedding, embedding_model, dim, normalized}. Both default to
    their OVS_PUNCT / OVS_SPEAKER_EMB env values; the query overrides per
    connection (same convention as ?vad=).

    Requires an ASR backend with STREAMING capability.
    """
    import asyncio
    import numpy as np
    from server.core.asr_backend import ASRCapability
    from server.core.api_auth import check_ws
    from server.core.session_limiter import try_acquire_ws

    # Auth runs BEFORE accept (when possible). check_ws() accepts+closes
    # 4401 on failure so the WS hand-off is deterministic.
    if not await check_ws(ws):
        return

    # Week 2: capture/generate request id before accept so the very
    # first WS log line carries the same correlator as later logs.
    _ws_request_id = request_id_from_headers(ws.headers) or generate_request_id()
    _ws_ctx_tokens = set_request_context(request_id=_ws_request_id)

    await ws.accept()

    # Reject-not-queue admission gate.
    _session_token = await try_acquire_ws(ws, "/asr/stream")
    if _session_token is None:
        reset_request_context(_ws_ctx_tokens)
        return

    # Week 2: track active streaming WS for /metrics. Paired decrement
    # lives in the finally block at the bottom of this handler.
    try:
        from server.core import metrics as _m_ws
        _m_ws.inc_active_ws_sessions()
        _ws_metric_taken = True
    except Exception:
        _ws_metric_taken = False

    # Register this WS session with the BackendManager (if available) so a
    # subsequent /admin/backend/reload can force-close it (code 1012) and
    # cancel the handler task instead of waiting forever for drain.
    _asr_mgr = _try_asr_manager()
    _ws_handle = _WSHandle(websocket=ws, task=asyncio.current_task())
    if _asr_mgr is not None:
        _asr_mgr.register_ws(_ws_handle)
    vad_backend = vad if vad is not None else _default_vad_backend()
    vad_silence = _default_vad_silence_ms() if vad_silence_ms is None else max(0, int(vad_silence_ms))

    # Optional final-payload enrichments: query overrides env default (off).
    from server.core import punctuation as _punct_mod
    from server.core import speaker_embedding as _spk_mod
    from server.core import diarization as _diar_mod
    punct_on = _flag_or(punctuate, _punct_mod.punctuation_enabled())
    spk_on = _flag_or(speaker_embedding, _spk_mod.speaker_embedding_enabled())
    diarize_on = _flag_or(diarize, _diar_mod.diarize_enabled())
    # Diarization clusters over per-segment embeddings, so it implies embedding.
    if diarize_on:
        spk_on = True

    # Choose backend: prefer ASR backend with STREAMING, fall back to sherpa
    asr_be = _get_asr_backend()
    use_backend_stream = (
        asr_be is not None
        and asr_be.is_ready()
        and asr_be.has_capability(ASRCapability.STREAMING)
    )

    # Lazy-init server-side VAD only if requested. Reuses the shared
    # singleton model from server.core.vad — no extra model load per
    # connection.
    vad_session = None
    if vad_backend and vad_backend not in ("none", "off", "disabled"):
        try:
            from server.core import vad as vad_mod
            vad_session = vad_mod.create_vad(
                vad_backend, sample_rate=sample_rate, silence_ms=vad_silence
            )
        except Exception as e:
            logger.warning("VAD '%s' init failed (%s); falling back to forced-EOS", vad_backend, e)
            vad_session = None

    # #41 P3 (DEFERRED): on an ABRUPT TCP drop (no close frame), the slot
    # release below is gated on ws.receive() returning, which can stall
    # ~60s until the OS TCP stack times out. There is no clean Starlette-
    # level active-disconnect event that fires earlier — ws.client_state
    # only flips after receive() processes a websocket.disconnect message,
    # and uvicorn cannot synthesize that message before the transport dies.
    # The only earlier signal would be a fixed receive() timeout, which is
    # explicitly rejected: a slow-but-alive client (long inter-frame
    # silence mid-utterance) would be falsely killed. The /v2v/stream P1
    # fix above removes the back-to-back accumulation that actually drove
    # the 4429 storm; the residual /asr/stream lag affects only one stuck
    # slot per abrupt-dropped connection and self-heals on TCP timeout.
    # TODO(#41): if a kernel-level liveness signal becomes available
    # (TCP_USER_TIMEOUT probe / ASGI extension), gate release on it.
    try:
        if use_backend_stream:
            from server.core.coordinator import get_coordinator
            if _asr_sentence_level_locking:
                # Slot is taken per utterance inside _asr_stream_backend.
                await _asr_stream_backend(
                    ws, asr_be, language, sample_rate, vad_session,
                    punct_on=punct_on, spk_on=spk_on, diarize_on=diarize_on,
                    per_utterance_slot=True,
                )
            else:
                async with get_coordinator().acquire("asr"):
                    await _asr_stream_backend(
                        ws, asr_be, language, sample_rate, vad_session,
                        punct_on=punct_on, spk_on=spk_on, diarize_on=diarize_on,
                    )
        else:
            await ws.send_json({"error": "no streaming ASR available"})
            await ws.close()
    finally:
        # best-effort cleanup: if unregister_ws raises, _session_token.release()
        # must still run (slot leak guard symmetric to /tts/stream pattern).
        if _asr_mgr is not None:
            try:
                _asr_mgr.unregister_ws(_ws_handle)
            except BaseException:
                pass
        if _session_token is not None:
            try:
                _session_token.release()
            except BaseException:
                pass
        if _ws_metric_taken:
            try:
                from server.core import metrics as _m_ws
                _m_ws.dec_active_ws_sessions()
            except Exception:
                pass
        reset_request_context(_ws_ctx_tokens)


# ---------------------------------------------------------------------------
# Sentence-level ASR locking
#
# Before this, ``/asr/stream`` held the coordinator "asr" slot for the entire
# WebSocket, so one connected client meant no other client could connect even
# while the NPU sat idle between utterances. Sentence-level locking holds the
# slot only around each utterance's finalize (the only place a shared-runtime
# offline backend actually touches the NPU), and admits N sessions that queue
# their per-utterance work in ``asr_infer_gate``.
#
# Backend-agnostic and opt-in: it engages whenever the resolver says more
# sessions may be admitted than may be inside the ASR runtime at once, which a
# backend expresses as ``supports_parallel=False`` with ``max_concurrent=N``.
# A backend that declares nothing keeps 1:1 and the old connection-level lock,
# so no existing deployment changes behaviour.
#
# What the slot covers is the finalize path only — the one place an
# accumulate-then-transcribe backend touches its runtime. A backend that also
# infers inside ``get_partial()`` (called on the event-loop thread, outside
# the slot) must NOT declare max_concurrent > 1 with supports_parallel=False.
# ---------------------------------------------------------------------------

_asr_sentence_level_locking: bool = False


def _set_asr_sentence_level_locking(resolved) -> None:
    """Decide connection-level vs sentence-level locking once, at startup."""
    global _asr_sentence_level_locking

    cap = resolved.asr_cap
    if resolved.coordinator_mode == "exclusive":
        # ``exclusive`` is a residency contract: entering the slot unloads the
        # opposite modality. Re-acquiring per utterance would thrash
        # unload/preload on every sentence, so keep the connection-level hold.
        _asr_sentence_level_locking = False
    elif cap.max_concurrent is None:
        # No fixed session cap. Only meaningful to gate when inference is
        # serial; a parallel, uncapped backend (which is also what "no ASR
        # backend declared" resolves to) has nothing to queue behind.
        _asr_sentence_level_locking = not cap.supports_parallel
    else:
        # Gate whenever we admit more sessions than may be inside the runtime.
        # Keyed on the RESOLVED in-flight limit, not on supports_parallel, so
        # an operator lowering OVS_ASR_INFER_CONCURRENCY below the session
        # count actually gets the queue rather than being silently ignored.
        _asr_sentence_level_locking = (
            cap.max_concurrent > resolved.asr_infer_concurrency
        )
    logger.info(
        "ASR locking granularity: %s (asr sessions=%s, in-flight=%s, "
        "queue depth=%s, mode=%s)",
        "sentence" if _asr_sentence_level_locking else "connection",
        cap.max_concurrent,
        resolved.asr_infer_concurrency,
        resolved.asr_queue_depth,
        resolved.coordinator_mode,
    )


class _AsrSlotJobs:
    """Executor futures started inside one utterance slot.

    ``run_in_executor`` cannot cancel a thread that has already started. If the
    handler task is cancelled (server shutdown, backend drain) the ``await``
    raises at once while the worker is still inside the backend. Releasing the
    inference slot there would hand the shared runtime to the next session on
    top of a live inference. Recording the futures here lets the slot wait for
    the worker before it releases.
    """

    __slots__ = ("futures",)

    def __init__(self) -> None:
        # concurrent.futures.Future, deliberately NOT the asyncio wrapper.
        # Cancelling the awaiting coroutine marks the asyncio future CANCELLED
        # even when the thread could not be stopped, so the wrapper reports
        # done() while the worker is still running — it cannot tell us when the
        # backend is free. The thread-pool future can.
        self.futures: list = []

    async def run(self, fn, *args):
        cf = _get_asr_executor().submit(fn, *args)
        self.futures.append(cf)
        return await asyncio.wrap_future(cf)

    async def drain(self) -> None:
        pending = [f for f in self.futures if not f.done()]
        self.futures.clear()
        if not pending:
            return
        pending = [asyncio.wrap_future(cf) for cf in pending]
        # ``shield`` keeps the wait itself from being cancelled again. Best
        # effort: if the caller is cancelled a second time we re-raise rather
        # than hang, which is the shutdown path where a second inference no
        # longer matters.
        try:
            await asyncio.shield(
                asyncio.gather(*pending, return_exceptions=True)
            )
        except asyncio.CancelledError:
            logger.warning(
                "ASR slot: cancelled while waiting for an in-flight inference "
                "to finish; releasing the slot with a worker still running"
            )


async def _send_asr_busy(ws, reason: str) -> None:
    """Tell the client this utterance was dropped for backpressure.

    Distinct from an error: the session stays open and the next utterance is
    processed normally. Clients that ignore the frame simply see a missing
    final, which is the same outcome as before but without the server growing
    an unbounded inference backlog.
    """
    try:
        await ws.send_json({
            "type": "busy",
            "reason": "asr_queue_full",
            "endpoint": reason,
        })
    except Exception:
        logger.debug("ASR busy notice send failed (client gone)", exc_info=True)


@asynccontextmanager
async def _asr_no_slot(jobs: "_AsrSlotJobs"):
    """No-op slot: the caller already holds the connection-level lock.

    Still drains ``jobs``, so a cancelled handler does not return to the
    caller — which then releases the connection-level lock — while a worker
    thread is still inside the backend.
    """
    try:
        yield 0.0
    finally:
        await jobs.drain()


@asynccontextmanager
async def _asr_utterance_slot(jobs: "_AsrSlotJobs"):
    """Hold the ASR execution slot for exactly one utterance's inference.

    Gate first (bounded, rejects with ``InferenceQueueFull`` when the backlog
    is full), coordinator lock second (unbounded ``asyncio.Lock``). That order
    matters: it is what keeps the backlog bounded. With the gate at
    concurrency=1 at most one ASR task can ever be parked on the coordinator
    lock, so the unbounded wait is never reachable from this path.
    """
    from server.core.asr_infer_gate import get_asr_inference_gate
    from server.core.coordinator import get_coordinator

    async with get_asr_inference_gate().acquire() as waited:
        async with get_coordinator().acquire("asr"):
            try:
                yield waited
            finally:
                # Release only once the executor worker has actually left the
                # backend — see _AsrSlotJobs.
                await jobs.drain()


async def _asr_stream_backend(
    ws: WebSocket,
    asr_be,
    language: str,
    sample_rate: int,
    vad_session=None,
    punct_on: bool = False,
    spk_on: bool = False,
    diarize_on: bool = False,
    per_utterance_slot: bool = False,
):
    """Streaming ASR using ASR backend (accumulate-then-transcribe).

    Supports a ``reset`` control command: the client may send a JSON text
    message ``{"command": "reset"}`` at any time.  This discards the
    current stream and creates a fresh one without closing the WebSocket.

    When ``spk_on`` is set, raw utterance audio is buffered ("one utterance,
    cleared on each finalize") so a speaker embedding can be computed per final.
    The buffer is NOT touched when ``spk_on`` is False (zero overhead).
    """
    import asyncio
    import json as _json
    import numpy as np

    from server.core import diarization as _diar_mod_be
    from server.core.asr_infer_gate import InferenceQueueFull

    # ``per_utterance_slot``: the caller did NOT take the coordinator slot for
    # the connection, so each finalize below must take it (plus the bounded
    # inference gate) for the duration of that one utterance's inference. When
    # False the caller holds the slot for the whole connection and the slot
    # context manager here is a no-op — identical to the pre-change code path.
    _slot_cm = _asr_utterance_slot if per_utterance_slot else _asr_no_slot

    def _slot():
        """One utterance's slot, with a fresh executor-job tracker."""
        jobs = _AsrSlotJobs()
        return _slot_cm(jobs), jobs

    stream = asr_be.create_stream(language=language)
    logger.info("ASR stream opened (backend=%s)", asr_be.name)

    # Per-utterance audio buffer for speaker embedding. Only populated when
    # spk_on; cleared after every finalize / reset (see _augment_final_payload).
    _seg: list = []

    # P0b/P1: session-relative timeline. ``_t_samples`` counts every sample fed
    # to the buffer; a segment spans [_t_samples - len(_seg), _t_samples] / sr.
    # Only advanced when spk_on (zero overhead otherwise). One OnlineDiarizer
    # per connection holds the running speaker centroids when diarize_on.
    _t_samples: int = 0
    _diarizer = _diar_mod_be.make_session_diarizer() if diarize_on else None

    def _seg_window():
        """Session-relative (start, end) seconds for the current buffer."""
        if not sample_rate:
            return 0.0, 0.0
        _len = sum(int(len(s)) for s in _seg)
        end = _t_samples / float(sample_rate)
        start = (_t_samples - _len) / float(sample_rate)
        return start, end

    # Close-frame override: set to 4429 on slot-pool saturation so the finally
    # block closes with a reject-not-queue code instead of the default 1000.
    _asr_close_code: int | None = None
    _asr_close_reason: str | None = None

    try:
        while True:
            msg = await ws.receive()

            # ── Text message: control command ──
            if "text" in msg and msg["text"]:
                try:
                    cmd = _json.loads(msg["text"])
                except (ValueError, TypeError):
                    continue
                if cmd.get("command") == "reset":
                    # Release per-stream GPU resources from the old stream
                    # before swapping (no-op for backends without close()).
                    try:
                        _old_close = getattr(stream, "close", None)
                        if _old_close is not None:
                            _old_close()
                    except Exception:
                        logger.exception("ASR reset: old stream close raised")
                    stream = asr_be.create_stream(language=language)
                    _seg.clear()
                    await ws.send_json({
                        "type": "reset",
                        "text": "",
                        "is_final": True,
                        "is_stable": True,
                        "reset": True,
                    })
                    logger.debug("ASR stream reset by client command (backend=%s)", asr_be.name)
                elif cmd.get("command") == "end_utterance" or (cmd.get("type") or "").lower() == "eou":
                    _loop = asyncio.get_event_loop()
                    force_endpoint = getattr(stream, "force_endpoint", None)
                    _cm, _jobs = _slot()
                    try:
                        async with _cm:
                            if force_endpoint is not None:
                                final_text = await _jobs.run(force_endpoint)
                                detected_language = None
                            else:
                                await _jobs.run(stream.prepare_finalize)
                                raw_final = await _jobs.run(stream.finalize)
                                final_text, detected_language = _unpack_finalize_result(raw_final)
                    except InferenceQueueFull:
                        # Same cleanup as the VAD rejection path: without it the
                        # rejected audio stays in the stream and in _seg, and
                        # the next successful final would splice two utterances
                        # together (and the buffer would keep growing under
                        # sustained overload).
                        await _send_asr_busy(ws, "end_utterance")
                        try:
                            _old_close = getattr(stream, "close", None)
                            if _old_close is not None:
                                _old_close()
                        except Exception:
                            logger.exception("ASR busy re-arm: stream close raised")
                        stream = asr_be.create_stream(language=language)
                        _seg.clear()
                        if vad_session is not None:
                            try:
                                vad_session.reset()
                            except Exception:
                                logger.debug("VAD reset after busy raised", exc_info=True)
                        continue
                    payload = {
                        "type": "final",
                        "text": final_text,
                        "is_final": True,
                        "is_stable": True,
                    }
                    if detected_language:
                        payload["language"] = detected_language
                    _st, _en = _seg_window()
                    await _augment_final_payload(
                        payload, final_text, _seg, punct_on, spk_on, sample_rate,
                        diarizer=_diarizer, seg_start=_st, seg_end=_en,
                    )
                    _seg.clear()
                    await ws.send_json(payload)
                    logger.debug("ASR utterance endpoint forced (backend=%s)", asr_be.name)
                continue

            # ── Binary message: audio data ──
            # Check the ASGI type, not the payload. A disconnect message has no
            # "bytes" key at all, so `.get("bytes", b"")` returned b"" and fell
            # through to the empty-payload branch below — which means "end of
            # audio" and runs a full finalize. On an offline backend that is a
            # whole transcription, held against a single ASR slot, for a client
            # that has already gone.
            if msg.get("type") == "websocket.disconnect":
                logger.debug("ASR stream: client disconnected; abandoning utterance")
                break
            data = msg.get("bytes")
            if data is None:
                # Neither audio nor a recognised control frame.
                continue

            if len(data) == 0:
                # End of audio — pre-encode tail, then decode
                _loop = asyncio.get_event_loop()
                _cm, _jobs = _slot()
                try:
                    async with _cm:
                        await _jobs.run(stream.prepare_finalize)
                        raw_final = await _jobs.run(stream.finalize)
                except InferenceQueueFull:
                    await _send_asr_busy(ws, "eos")
                    break
                final_text, detected_language = _unpack_finalize_result(raw_final)
                payload = {
                    "type": "final",
                    "text": final_text,
                    "is_final": True,
                    "is_stable": True,
                }
                if detected_language:
                    payload["language"] = detected_language
                _st, _en = _seg_window()
                await _augment_final_payload(
                    payload, final_text, _seg, punct_on, spk_on, sample_rate,
                    diarizer=_diarizer, seg_start=_st, seg_end=_en,
                )
                _seg.clear()
                await ws.send_json(payload)
                # P1: optional session-end blind-diarization summary (relabel()).
                if _diarizer is not None:
                    _summary = _diar_mod_be.summary_payload(_diarizer)
                    if _summary is not None:
                        try:
                            await ws.send_json(_summary)
                        except Exception:
                            pass
                break

            # Buffer audio (run in thread to avoid blocking event loop)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if spk_on:
                _seg.append(samples)
                _t_samples += int(len(samples))
            _loop = asyncio.get_event_loop()
            await _loop.run_in_executor(_get_asr_executor(), stream.accept_waveform, sample_rate, samples)

            # Server-side VAD endpoint detection (opt-in via ?vad=)
            if vad_session is not None:
                from server.core.vad import VADSession
                event = vad_session.process(samples)
                if event == VADSession.SPEECH_END:
                    # Emit vad_endpoint BEFORE finalize so the client can split
                    # VAD silence-wait from ASR compute time.
                    await ws.send_json({"type": "vad_endpoint"})
                    _cm, _jobs = _slot()
                    try:
                        async with _cm:
                            await _jobs.run(stream.prepare_finalize)
                            raw_final = await _jobs.run(stream.finalize)
                    except InferenceQueueFull:
                        # Backlog full: drop this utterance rather than grow an
                        # unbounded queue. Re-arm the stream/VAD exactly as the
                        # success path does so the session stays usable.
                        await _send_asr_busy(ws, "vad")
                        try:
                            _old_close = getattr(stream, "close", None)
                            if _old_close is not None:
                                _old_close()
                        except Exception:
                            logger.exception("ASR busy re-arm: stream close raised")
                        stream = asr_be.create_stream(language=language)
                        _seg.clear()
                        try:
                            vad_session.reset()
                        except Exception:
                            logger.debug("VAD reset after busy raised", exc_info=True)
                        continue
                    final_text, detected_language = _unpack_finalize_result(raw_final)
                    try:
                        payload = {
                            "type": "final",
                            "text": final_text,
                            "is_final": True,
                            "is_stable": True,
                            "endpoint": "vad",
                        }
                        if detected_language:
                            payload["language"] = detected_language
                        _st, _en = _seg_window()
                        await _augment_final_payload(
                            payload, final_text, _seg, punct_on, spk_on, sample_rate,
                            diarizer=_diarizer, seg_start=_st, seg_end=_en,
                        )
                        await ws.send_json(payload)
                    except Exception:
                        # Client gone during a slow finalize (e.g. TRT-EdgeLLM
                        # on Jetson) — nothing to send to, close out.
                        break
                    # Multi-utterance: reset the ASR stream + VAD and KEEP the
                    # socket open for the next utterance. Previously this path
                    # `break`'d — closing after every server-VAD endpoint — which
                    # forced clients (e.g. the live-caption page) to reconnect per
                    # sentence. The finalize above uses prepare_finalize+finalize
                    # (complete text), unlike the end_utterance force_endpoint path.
                    try:
                        _old_close = getattr(stream, "close", None)
                        if _old_close is not None:
                            _old_close()
                    except Exception:
                        logger.exception("ASR VAD endpoint: stream close raised")
                    stream = asr_be.create_stream(language=language)
                    _seg.clear()
                    try:
                        vad_session.reset()
                    except Exception:
                        logger.debug("VAD reset after endpoint raised", exc_info=True)
                    continue

            # Check for partial results
            partial_text, is_endpoint = stream.get_partial()
            if partial_text:
                if is_endpoint:
                    payload = {
                        "type": "final",
                        "text": partial_text,
                        "is_final": True,
                        "is_stable": True,
                    }
                    _st, _en = _seg_window()
                    await _augment_final_payload(
                        payload, partial_text, _seg, punct_on, spk_on, sample_rate,
                        diarizer=_diarizer, seg_start=_st, seg_end=_en,
                    )
                    _seg.clear()
                    await ws.send_json(payload)
                    # Re-arm exactly like the frontend-VAD endpoint path above.
                    # Backends with sticky endpoint results (RK chunk-confirm's
                    # get_partial() keeps returning is_final=True with the
                    # composed text after _finalize_utterance) otherwise re-emit
                    # the same final every poll — a "final storm" that feeds one
                    # stale embedding per ~400ms into the online diarizer.
                    try:
                        _old_close = getattr(stream, "close", None)
                        if _old_close is not None:
                            _old_close()
                    except Exception:
                        logger.exception("ASR backend endpoint: stream close raised")
                    stream = asr_be.create_stream(language=language)
                    try:
                        vad_session.reset()
                    except Exception:
                        logger.debug(
                            "VAD reset after backend endpoint raised", exc_info=True
                        )
                else:
                    await ws.send_json({
                        "type": "partial",
                        "text": partial_text,
                        "is_final": False,
                        "is_stable": False,
                    })

    except WebSocketDisconnect:
        logger.debug("ASR stream client disconnected (backend=%s)", asr_be.name)
    except Exception as e:
        # Slot-pool saturation: the backend rejected a begin because every
        # decoder slot is busy (PoolSaturatedError, status 4429). This is a
        # "backend busy" condition, NOT a worker fault — surface it as a clean
        # 4429 close (matching SessionLimiter's reject-not-queue semantics) and
        # do NOT trigger a destructive worker restart (the PoolSaturatedError
        # class is intentionally off the WorkerProtocolError lineage so the
        # session manager already avoids the rebuild path).
        _saturated, _max_slots = _is_pool_saturated(e)
        if _saturated:
            logger.warning(
                "ASR stream slot-pool saturated (backend=%s, max_slots=%s); "
                "rejecting with 4429",
                asr_be.name, _max_slots,
            )
            try:
                from server.core import metrics as _m_sat
                _m_sat.inc_sessions_rejected("ws")
            except Exception:
                pass
            _reason = _json.dumps({"error": "pool_saturated", "max_slots": _max_slots})
            _asr_close_code = 4429
            _asr_close_reason = _reason
        else:
            logger.error("ASR stream error (backend=%s): %s", asr_be.name, e, exc_info=True)
            # Surface backend failures to clients as structured terminal frames.
            # Otherwise benchmark clients only observe a socket close and lose the
            # actual failure reason.
            try:
                await ws.send_json({
                    "type": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "backend": asr_be.name,
                    "is_final": True,
                    "is_stable": True,
                })
            except Exception:
                pass
    finally:
        # Release per-stream TRT contexts and device buffers (no-op for
        # backends without per-stream GPU resources).
        try:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        except Exception:
            logger.exception("ASR stream close raised")
        try:
            if _asr_close_code is not None:
                await ws.close(code=_asr_close_code, reason=_asr_close_reason)
            else:
                await ws.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────
# Unified V2V WebSocket: ASR + TTS + VAD + barge-in
# ──────────────────────────────────────────────────────────────────────


class _RealtimeV2WebSocketProxy:
    """Translate voxedge's current V1 transport frames to Realtime V2.

    The engine remains transport/protocol agnostic while its internal event
    names are migrated. This proxy is deliberately thin and uses the same
    ``RealtimeV2EventAdapter`` as the legacy orchestration path.
    """

    def __init__(self, ws, adapter):
        self._ws = ws
        self._adapter = adapter
        self._binary_seen = False
        self._pending_messages = []
        self._tool_registry = None
        self._advertised_tool_names = set()

    def bind_tool_registry(self, registry) -> None:
        self._tool_registry = registry

    def __getattr__(self, name):
        return getattr(self._ws, name)

    async def receive(self):
        import json as _json
        from server.core import v2v as v2v_proto

        while True:
            if self._pending_messages:
                return self._pending_messages.pop(0)
            msg = await self._ws.receive()
            text = msg.get("text")
            if not text:
                return msg
            try:
                payload = _json.loads(text)
            except (ValueError, TypeError):
                return msg
            typ = payload.get("type")
            if typ == v2v_proto.CLIENT_INPUT_AUDIO_BUFFER_COMMIT:
                payload = {"type": v2v_proto.CLIENT_ASR_EOS}
            elif typ == v2v_proto.CLIENT_RESPONSE_CANCEL:
                self._adapter.mark_cancelled("client_cancelled")
                payload = {"type": v2v_proto.CLIENT_ABORT}
            elif typ == v2v_proto.CLIENT_INPUT_AUDIO_BUFFER_CLEAR:
                payload = {"type": v2v_proto.CLIENT_ABORT}
            elif typ == v2v_proto.CLIENT_SESSION_UPDATE:
                session = payload.get("session")
                session = session if isinstance(session, dict) else {}
                extension = session.get("x_v2v")
                extension = extension if isinstance(extension, dict) else {}
                tools_supplied = "tools" in session
                tools = list(session.get("tools") or [])
                new_names = {
                    str(
                        (entry.get("function") or entry).get("name") or ""
                    )
                    for entry in tools
                    if isinstance(entry, dict)
                    and isinstance(entry.get("function") or entry, dict)
                }
                new_names.discard("")
                if tools_supplied and self._tool_registry is not None:
                    for removed in self._advertised_tool_names - new_names:
                        self._tool_registry.unregister(removed)
                if tools_supplied:
                    self._advertised_tool_names = new_names
                payload = {
                    "type": v2v_proto.CLIENT_TOOL_ADVERTISE,
                    "tools": [
                        {
                            **entry,
                            **(
                                entry.get("x_v2v")
                                if isinstance(entry.get("x_v2v"), dict)
                                else {}
                            ),
                        }
                        for entry in tools
                        if isinstance(entry, dict)
                    ],
                    "system_prompt": session.get("instructions"),
                    "llm_params": dict(extension.get("llm_params") or {}),
                    # Initial/tool-schema updates warm the stable prefix.
                    # Reachy's per-turn instructions-only update must not race
                    # response.create with a second GPU warm-up.
                    "warm_prefix": tools_supplied,
                }
                await self._ws.send_json(self._adapter.session_updated(session))
            elif typ == v2v_proto.CLIENT_CONVERSATION_ITEM_CREATE:
                item = payload.get("item")
                item = item if isinstance(item, dict) else {}
                if item.get("type") != "function_call_output":
                    await self._ws.send_json(self._adapter.translate({
                        "type": v2v_proto.SERVER_ERROR,
                        "code": "unsupported_conversation_item",
                        "error": "only function_call_output items are accepted",
                        "param": "item.type",
                    })[0])
                    continue
                output = item.get("output", "{}")
                try:
                    result = _json.loads(output) if isinstance(output, str) else output
                except (ValueError, TypeError):
                    result = {"ok": True, "result": {"output": str(output)}}
                result = result if isinstance(result, dict) else {"result": result}
                ok = bool(result.get("ok", True))
                payload = {
                    "type": v2v_proto.CLIENT_TOOL_RESULT,
                    "call_id": item.get("call_id"),
                    "id": item.get("call_id"),
                    "name": result.get("name", ""),
                    "ok": ok,
                }
                if ok:
                    value = result.get("result", result)
                    payload["result"] = value if isinstance(value, dict) else {"value": value}
                else:
                    payload["error"] = str(result.get("error") or "tool execution failed")
            elif typ == v2v_proto.CLIENT_DIRECT_SPEAK:
                speech = payload.get("speech")
                speech = speech if isinstance(speech, dict) else {}
                text_value = str(speech.get("text") or "")
                self._adapter.mark_direct_speak()
                payload = {"type": v2v_proto.CLIENT_TEXT, "text": text_value}
                self._pending_messages.append({
                    "type": "websocket.receive",
                    "text": _json.dumps({"type": v2v_proto.CLIENT_TTS_FLUSH}),
                })
            elif typ == v2v_proto.CLIENT_CONVERSATION_ITEM_TRUNCATE:
                try:
                    audio_end_ms = max(0, int(payload.get("audio_end_ms", 0)))
                except (TypeError, ValueError):
                    await self._ws.send_json(self._adapter.translate({
                        "type": v2v_proto.SERVER_ERROR,
                        "code": "invalid_audio_end_ms",
                        "error": "audio_end_ms must be a non-negative integer",
                        "param": "audio_end_ms",
                    })[0])
                    continue
                await self._ws.send_json({
                    "type": v2v_proto.SERVER_CONVERSATION_ITEM_TRUNCATED,
                    "event_id": self._adapter._event_id(),
                    "item_id": payload.get("item_id"),
                    "content_index": payload.get("content_index", 0),
                    "audio_end_ms": audio_end_ms,
                })
                continue
            elif typ == v2v_proto.CLIENT_CONVERSATION_RESET:
                self._adapter.mark_cancelled("conversation_reset")
                await self._ws.send_json({
                    "type": v2v_proto.SERVER_CONVERSATION_RESET_DONE,
                    "event_id": self._adapter._event_id(),
                })
                payload = {"type": v2v_proto.CLIENT_ABORT}
            msg = dict(msg)
            msg["text"] = _json.dumps(payload)
            return msg

    async def send_json(self, payload):
        for event in self._adapter.translate(payload):
            await self._ws.send_json(event)

    async def send_bytes(self, data):
        # voxedge currently sends the negotiated sample rate as a standalone
        # uint32 first frame. Realtime V2 declares it in the session instead.
        if not self._binary_seen and len(data) == 4:
            self._binary_seen = True
            return
        self._binary_seen = True
        await self._ws.send_bytes(data)

    async def close(self, code=None, reason=None):
        kwargs = {}
        if code is not None:
            kwargs["code"] = code
        if reason is not None:
            kwargs["reason"] = reason
        await self._ws.close(**kwargs)


async def _v2v_stream_via_engine(
    ws,
    cfg: dict,
    *,
    asr_language,
    tts_language,
    tts_language_norm,
    sample_rate: int,
    vad_backend,
    vad_silence_ms: int,
    multi_utterance: bool,
    coord,
    asr_stream_options: "dict | None" = None,
):
    """Phase 1b feature-flag path: drive /v2v/stream via voxedge's
    importable :class:`ConversationEngine` instead of the in-handler
    dispatcher/asr_out_task/tts_out_task.

    Called from ``v2v_stream`` ONLY when ``OVS_V2V_ENGINE == "voxedge"``,
    AFTER auth + admission (``try_acquire_ws``) + config parse have already
    succeeded. Admission/4429/auth/close-code bookkeeping stays in the caller
    (transport layer); this function only orchestrates the conversation.

    Backends are taken from the SAME already-resolved process singletons the
    legacy path uses — ASR via ``_get_asr_backend()``, TTS via
    ``tts_service.get_backend()``, VAD via ``vad_mod.create_vad(...)`` — so a
    hardware A/B between the two paths is same-backend, same-config.

    Two LLM modes, selected by the ``OVS_V2V_SERVER_LOOP`` env flag:

    * **OFF (default)** — no LLM is wired (``tool_registry=None``, no ``llm``
      backend): production /v2v runs the LLM client-side and re-feeds tokens
      via CLIENT_TEXT, so the engine stays a "client text → TTS" pass-through.
      This is a hard zero-behavior-change contract for the existing path.
    * **ON** (``OVS_V2V_SERVER_LOOP=1``, #37 Phase 2-product) — the server owns
      ASR→LLM(+tools)→TTS: an edge-llm adapter (``server.core.edge_llm_backend``)
      drives the LLM hop and a ``ToolRegistry`` runs the server-side multi-turn
      tool pump (spec §2/§4). Tool schema starts empty (client-advertised tools
      are the next step); the pump path is live so registered tools flow
      without re-wiring.
    """
    import asyncio

    from server.core import tts_service
    from server.core.asr_backend import ASRCapability
    from server.core.tts_backend import TTSCapability
    from server.core import vad as vad_mod

    from voxedge.engine.conversation import ConversationEngine
    from voxedge.engine.coordinator import BackendCoordinator
    from voxedge.transport.base import WebSocketTransport

    # ── Resolve backends from existing singletons (mirror legacy Stage 2) ──
    asr_be = None
    if asr_language:
        asr_be = _get_asr_backend()
        if (
            asr_be is None
            or not asr_be.is_ready()
            or not asr_be.has_capability(ASRCapability.STREAMING)
        ):
            await ws.send_json({
                "type": "error",
                "error": "asr_language requested but no streaming ASR backend ready",
            })
            try:
                await ws.close(code=1011)
            except BaseException:
                pass
            return

    tts_be = None
    if tts_language:
        if not tts_service.is_ready() or not tts_service.has_capability(
            TTSCapability.STREAMING
        ):
            await ws.send_json({
                "type": "error",
                "error": "tts_language requested but no streaming TTS backend ready",
            })
            try:
                await ws.close(code=1011)
            except BaseException:
                pass
            return
        tts_be = tts_service.get_backend()

    # VAD: build the per-connection session via the SAME factory the legacy
    # path uses (executor hop — silero ONNX first-load is ~500ms and would
    # otherwise stall the loop). Then wrap it in a one-method backend shim so
    # the engine's ``vad_be.create_session(silence_ms=...)`` contract is met
    # without re-implementing VAD. ValueError = hard config error → reject;
    # any other init failure falls back to no-VAD (legacy behavior).
    vad_session = None
    if asr_language:
        try:
            _loop_init = asyncio.get_event_loop()
            vad_session = await _loop_init.run_in_executor(
                None,
                lambda: vad_mod.create_vad(
                    vad_backend, sample_rate=sample_rate, silence_ms=vad_silence_ms
                ),
            )
        except ValueError as e:
            await ws.send_json({"type": "error", "error": f"VAD config: {e}"})
            try:
                await ws.close(code=1003)
            except BaseException:
                pass
            return
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "v2v(engine) VAD init (%s) failed: %s — running without VAD",
                vad_backend, e,
            )
            vad_session = None

    class _PrebuiltVADBackend:
        """Adapt an already-created app VADSession to voxedge's VADBackend.

        The legacy handler creates the per-connection ``VADSession`` eagerly
        (so the executor hop above absorbs silero's first-load cost). voxedge's
        engine instead expects a *backend* whose ``create_session`` mints the
        session. This shim returns the pre-built session — the app VADSession
        already duck-types voxedge's VADSession (process/SPEECH_START/
        SPEECH_END); the engine only calls ``.process`` on it.
        """

        def __init__(self, session):
            self._session = session

        @property
        def name(self) -> str:
            return getattr(self._session, "backend", "vad")

        def create_session(self, sample_rate: int = 16000, silence_ms: int = 400, **kwargs):
            return self._session

    # ── Coordinator: resolve mode from the live backends (spec §3.1). Use
    # the same requested mode the process-wide coordinator resolved to, so the
    # engine path honors the active profile's execution_policy. ──
    requested_mode = getattr(coord, "mode", "concurrent")
    engine_coord = BackendCoordinator.from_backends(
        asr=asr_be, tts=tts_be, requested_mode=requested_mode
    )

    # ── Build backends dict (any subset; no LLM — client-driven text path) ──
    backends: dict = {}
    if asr_be is not None:
        backends["asr"] = asr_be
    if tts_be is not None:
        backends["tts"] = tts_be
    if vad_session is not None:
        backends["vad"] = _PrebuiltVADBackend(vad_session)

    # ── Server-side LLM+tool loop (spec §2/§4, #37 Phase 2-product step 1).
    # Gated on OVS_V2V_SERVER_LOOP (default OFF). When OFF, NOTHING below runs:
    # backends has no "llm", tool_registry stays None, and the engine is the
    # exact client-text→TTS pass-through it was before this feature — a hard
    # zero-behavior-change contract for the existing /v2v path. When ON, the
    # server owns ASR→LLM(+tools)→TTS: an edge-llm adapter drives the LLM hop
    # and a ToolRegistry runs the multi-turn tool pump server-side. ──
    server_loop = _env_truthy(os.environ.get("OVS_V2V_SERVER_LOOP"))
    tool_registry = None
    server_system_prompt = None
    server_llm_params: dict = {}
    if server_loop:
        from server.core.edge_llm_backend import EdgeLLMBackend
        from voxedge.engine.tool_registry import ToolRegistry

        llm_be = EdgeLLMBackend(
            base_url=os.environ.get("EDGE_LLM_BASE_URL") or None,
            model=os.environ.get("EDGE_LLM_MODEL", "qwen3"),
            enable_thinking=_env_truthy(
                os.environ.get("OVS_V2V_LLM_ENABLE_THINKING")
            ),
        )
        backends["llm"] = llm_be
        # Tool registry: empty by default (client-advertised tools are the
        # NEXT step). A non-None registry is what switches the engine to the
        # server-side tool pump; with zero tools it sends an empty tools list
        # (== no tools) so the LLM just answers — but the pump path is live so
        # later-registered tools (or client-advertised) flow without re-wiring.
        tool_registry = ToolRegistry()
        bind_registry = getattr(ws, "bind_tool_registry", None)
        if callable(bind_registry):
            bind_registry(tool_registry)
        # System prompt + LLM params for the server loop. Sourced from env/
        # config (interface for the future VoiceArm prompt port, spec §5); the
        # engine itself never reads env (spec §2 — params are injected here).
        server_system_prompt = os.environ.get("OVS_V2V_SYSTEM_PROMPT") or None
        _temp = os.environ.get("OVS_V2V_LLM_TEMPERATURE")
        if _temp:
            try:
                server_llm_params["temperature"] = float(_temp)
            except ValueError:
                pass
        _max_tok = os.environ.get("OVS_V2V_LLM_MAX_TOKENS")
        if _max_tok:
            try:
                server_llm_params["max_tokens"] = int(_max_tok)
            except ValueError:
                pass

    # ── Timeouts: mirror the legacy env defaults so the watchdogs behave
    # identically across the A/B. Engine reads NO env itself (spec §2). ──
    def _env_float(key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, default))
        except (TypeError, ValueError):
            return default

    timeouts = {
        "asr_turn": _env_float("OVS_ASR_TURN_TIMEOUT_S", 45.0),
        "tts_chunk": _env_float("OVS_TTS_CHUNK_TIMEOUT_S", 10.0),
        "tts_sentence": _env_float("OVS_TTS_SENTENCE_TIMEOUT_S", 15.0),
    }

    # ── #15 parity: resolve the TTS speaker/voice/speed + buffer selection the
    # SAME way the legacy handler does (server/main.py config-parse block +
    # tts_buffer selection), then forward to the engine. The engine applies
    # them in _SentenceWorker exactly like legacy stream_kwargs
    # (tts_speaker_kwargs wins; tts_voice deprecated fallback; tts_speed). ──
    tts_voice = cfg.get("tts_voice")
    tts_speaker_id = cfg.get("tts_speaker_id")
    tts_speed = cfg.get("tts_speed")
    tts_speaker_kwargs: dict = {}
    if tts_speaker_id is not None and tts_language and tts_be is not None:
        from server.core.tts_speakers import speaker_kwargs_for_id
        if tts_service.is_ready():
            tts_speaker_kwargs = speaker_kwargs_for_id(
                int(tts_speaker_id), tts_service.get_backend().model_id
            )
    low_latency_tts = os.environ.get(
        "OVS_TTS_LOW_LATENCY_CHUNKING", "1"
    ).lower() not in ("0", "false", "no", "off")
    # V1 has always auto-started on ASR final. Realtime V2 can negotiate
    # create_response=false so applications such as Reachy can update dynamic
    # visual instructions before explicitly sending response.create.
    auto_create_response = not (
        isinstance(ws, _RealtimeV2WebSocketProxy)
        and not bool(cfg.get("_create_response", False))
    )

    engine = ConversationEngine(
        backends=backends,
        tool_registry=tool_registry,
        system_prompt=server_system_prompt,
        llm_params=server_llm_params,
        multi_utterance=multi_utterance,
        timeouts=timeouts,
        silence_ms=vad_silence_ms,
        vad_preroll_ms=_vad_preroll_ms(),
        asr_language=asr_language or "auto",
        asr_stream_options=asr_stream_options,
        tts_language=tts_language_norm,
        coordinator=engine_coord,
        tts_speaker_kwargs=tts_speaker_kwargs,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
        low_latency_tts=low_latency_tts,
        auto_create_response=auto_create_response,
    )

    logger.info(
        "v2v stream opened (engine path: asr=%s tts=%s vad=%s mode=%s multi=%s)",
        asr_language or "off",
        tts_language or "off",
        vad_backend if (asr_language and vad_session is not None) else "off",
        engine_coord.mode,
        multi_utterance,
    )

    # Resolve the half-open idle watchdog from product env and inject it — the
    # voxedge transport no longer reads env itself. None/invalid → its 90s
    # default (same value/parse as before, behaviour unchanged).
    _idle_raw = os.environ.get("OVS_V2V_IDLE_TIMEOUT_S")
    try:
        _idle_timeout_s = float(_idle_raw) if _idle_raw not in (None, "") else 90.0
    except (TypeError, ValueError):
        _idle_timeout_s = 90.0

    # The engine drives the conversation and closes the transport (which closes
    # the WS) at the end. Admission release stays in the caller's finally.
    await engine.run(WebSocketTransport(ws, idle_timeout_s=_idle_timeout_s))


@app.websocket("/v2v/stream")
async def v2v_stream(ws: WebSocket):
    """Unified bi-directional WebSocket: speech in, partials + audio out.

    Realtime V2 clients request the ``seeed.realtime.v2`` WebSocket
    subprotocol and use the session/response lifecycle documented in
    ``docs/api/realtime-v2.md``. Connections without a subprotocol remain on
    the migration-only V1 dialect in ``docs/api/v2v-stream.md``.

    Legacy minimum viable patterns:

      TTS-only (LLM token stream → audio):
        send {"type":"config", "tts_language":"zh"}
        send {"type":"text", "text":"..."} repeatedly
        send {"type":"tts_flush"}; await binary chunks + tts_done

      ASR-only (mic → text, with auto VAD endpoint):
        send {"type":"config", "asr_language":"zh"}
        send PCM binary chunks
        await asr_partial / asr_endpoint / asr_final

      V2V (full duplex):
        config with both asr_language + tts_language
        interleave binary (mic) with text (LLM tokens)
        receive partials, endpoints, audio
        send {"type":"abort"} to barge-in
    """
    import asyncio
    import json as _json
    import struct
    import numpy as np

    from server.core import tts_service, v2v as v2v_proto
    from server.core.asr_backend import ASRCapability
    from server.core.tts_backend import TTSCapability
    from server.core import vad as vad_mod
    from server.core.coordinator import get_coordinator

    coord = get_coordinator()

    from server.core.api_auth import check_ws
    from server.core.session_limiter import try_acquire_ws_token, close_ws_rejected

    # Auth before accept; deterministic 4401 close on failure.
    if not await check_ws(ws):
        return

    # Week 2: request id context for V2V WS.
    _v2v_request_id = request_id_from_headers(ws.headers) or generate_request_id()
    _v2v_ctx_tokens = set_request_context(request_id=_v2v_request_id)

    offered_subprotocols = {
        part.strip()
        for part in ws.headers.get("sec-websocket-protocol", "").split(",")
        if part.strip()
    }
    realtime_v2 = v2v_proto.REALTIME_V2_SUBPROTOCOL in offered_subprotocols
    await ws.accept(
        subprotocol=v2v_proto.REALTIME_V2_SUBPROTOCOL if realtime_v2 else None
    )

    # Idle / half-open watchdog for the two un-timed ws.receive() sites below
    # (config-phase + steady-state dispatcher). A dead or half-open client that
    # never sends another frame would otherwise wedge ws.receive() forever,
    # which holds the SessionLimiter admission slot open permanently (the slot
    # release lives in the orchestration finally, only reachable after the
    # receive loop exits) → back-to-back 4429 rejections until a manual restart.
    #
    # Default 90s is deliberately LONGER than the agent thinking-watchdog (20s)
    # and the LLM stream-idle timeout (30s) so a slow-but-alive turn is never
    # killed; this only fires on a truly silent socket. A timeout funnels into
    # the SAME teardown a normal client CLOSE / WebSocketDisconnect takes.
    def _v2v_env_float(_k: str, _d: float) -> float:
        try:
            return float(os.environ.get(_k, _d))
        except (TypeError, ValueError):
            return _d
    _v2v_idle_timeout_s = _v2v_env_float("OVS_V2V_IDLE_TIMEOUT_S", 90.0)

    # Reject-not-queue admission gate. When the slot is full AND admission-time
    # eviction is enabled (OVS_V2V_EVICT_ON_FULL + limit==1 single-client, e.g.
    # voice-arm), reclaim a slot leaked by a zombie holder before giving up:
    # the newcomer can only be that same client reconnecting after abandoning a
    # dead session. Otherwise behaves exactly like the legacy single-shot gate.
    _v2v_session_token, _admit_info = try_acquire_ws_token("/v2v/stream")
    if (
        _v2v_session_token is None
        and _admit_info.get("reason") == "too_many"
        and _v2v_evict_enabled()
    ):
        _v2v_session_token, _admit_info = await _v2v_evict_and_reacquire("/v2v/stream")
    if _v2v_session_token is None:
        await close_ws_rejected(ws, "/v2v/stream", _admit_info)
        try:
            reset_request_context(_v2v_ctx_tokens)
        except BaseException:
            pass
        return

    # Week 2: active WS gauge increment. Paired decrement is in both the
    # early-exit helper (_v2v_release_early) and the final cleanup block.
    try:
        from server.core import metrics as _m_v2v
        _m_v2v.inc_active_ws_sessions()
        _v2v_ws_metric_taken = True
    except Exception:
        _v2v_ws_metric_taken = False

    # Register this v2v session with whichever BackendManager(s) are
    # available so /admin/backend/reload of either kind can hard-close
    # the WS (code 1012) instead of letting the connection linger.
    _v2v_asr_mgr = _try_asr_manager()
    _v2v_tts_mgr = _try_tts_manager()
    _v2v_handle = _WSHandle(websocket=ws, task=asyncio.current_task())
    if _v2v_asr_mgr is not None:
        _v2v_asr_mgr.register_ws(_v2v_handle)
    if _v2v_tts_mgr is not None:
        _v2v_tts_mgr.register_ws(_v2v_handle)
    # Track as a live /v2v holder so a future newcomer can evict this session
    # if it leaks its slot (see _v2v_evict_and_reacquire). Removed in
    # _v2v_release_early (finally-guaranteed).
    _V2V_HOLDERS.add(_v2v_handle)

    realtime_adapter = None
    realtime_provider_name = os.environ.get("OVS_REALTIME_PROVIDER", "local").lower()
    if realtime_v2:
        try:
            _output_sr = (
                int(tts_service.get_sample_rate())
                if tts_service.is_ready() and hasattr(tts_service, "get_sample_rate")
                else 16000
            )
        except (TypeError, ValueError):
            _output_sr = 16000
        provider_label = (
            realtime_provider_name
            if realtime_provider_name in {"openai", "qwen"}
            else "local-cascade"
        )
        provider_capabilities = None
        if realtime_provider_name in {"openai", "qwen"}:
            from server.core.realtime_provider import create_provider_adapter
            provider_capabilities = create_provider_adapter(
                realtime_provider_name
            ).capabilities()
        provider_model_defaults = {
            "openai": "gpt-realtime-2.1",
            "qwen": "qwen-audio-3.0-realtime-flash",
        }
        realtime_adapter = v2v_proto.RealtimeV2EventAdapter(
            provider=provider_label,
            model=(
                os.environ.get(f"OVS_REALTIME_{realtime_provider_name.upper()}_MODEL", "")
                or provider_model_defaults.get(realtime_provider_name, provider_label)
            ),
            input_sample_rate=16000,
            output_sample_rate=_output_sr,
            capabilities_override=provider_capabilities,
        )
        await ws.send_json(realtime_adapter.session_created())

    # MUST-FIX 1 round 2: make release idempotent + a nonlocal flag so the
    # outer setup try/except BaseException can safely cover CancelledError
    # without double-releasing on the normal path.
    _v2v_released = {"done": False}

    def _v2v_release_early():
        # Release admission resources for early-exit paths that bypass
        # the main try/finally below. Idempotent: safe to call multiple
        # times (e.g. once from an inner branch, once from outer cancel
        # guard).
        if _v2v_released["done"]:
            return
        _v2v_released["done"] = True
        try:
            _V2V_HOLDERS.discard(_v2v_handle)
        except Exception:
            pass
        if _v2v_asr_mgr is not None:
            try:
                _v2v_asr_mgr.unregister_ws(_v2v_handle)
            except Exception:
                pass
        if _v2v_tts_mgr is not None:
            try:
                _v2v_tts_mgr.unregister_ws(_v2v_handle)
            except Exception:
                pass
        if _v2v_session_token is not None:
            try:
                _v2v_session_token.release()
            except Exception:
                pass
        if _v2v_ws_metric_taken:
            try:
                from server.core import metrics as _m_v2v
                _m_v2v.dec_active_ws_sessions()
            except Exception:
                pass
        # Note: do not reset_request_context here because the early-exit
        # helper may be called inside try/finally that itself resets;
        # caller is responsible for context cleanup.

    # MUST-FIX 1 round 2: wrap setup + main loop so any CancelledError
    # mid-setup (BaseException) still triggers admission cleanup via the
    # idempotent _v2v_release_early() helper in the outer finally.
    try:
        # ── Stage 1: receive initial config ─────────────────────────────
        try:
            first_msg = await asyncio.wait_for(
                ws.receive(), timeout=_v2v_idle_timeout_s
            )
        except asyncio.TimeoutError:
            # Half-open / silent client during the config handshake: treat
            # exactly like a client disconnect so the admission slot is
            # released instead of wedging this receive forever.
            logger.warning(
                "v2v: idle timeout (%.0fs) awaiting config frame — "
                "releasing slot as half-open client",
                _v2v_idle_timeout_s,
            )
            _v2v_release_early(); return
        except WebSocketDisconnect:
            _v2v_release_early(); return
        cfg_text = first_msg.get("text", "")
        if not cfg_text:
            await ws.close(code=1003); _v2v_release_early(); return
        try:
            cfg = _json.loads(cfg_text)
        except (ValueError, TypeError):
            await ws.close(code=1003); _v2v_release_early(); return
        if realtime_v2:
            if cfg.get("type") != v2v_proto.CLIENT_SESSION_UPDATE:
                await ws.send_json(realtime_adapter.translate({
                    "type": v2v_proto.SERVER_ERROR,
                    "error": "first client message must be session.update",
                    "code": "invalid_session_handshake",
                })[0])
                await ws.close(code=1003); _v2v_release_early(); return
            try:
                cfg = v2v_proto.session_update_to_legacy_config(cfg)
            except (ValueError, TypeError) as exc:
                await ws.send_json(realtime_adapter.translate({
                    "type": v2v_proto.SERVER_ERROR,
                    "error": str(exc),
                    "code": "invalid_session_update",
                })[0])
                await ws.close(code=1003); _v2v_release_early(); return
        elif cfg.get("type") != v2v_proto.CLIENT_CONFIG:
            await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                "error": "first message must be a config frame"})
            await ws.close(code=1003); _v2v_release_early(); return
    
        # Codex MUST-FIX 1: config parsing below can raise ValueError/TypeError
        # on bad client input (e.g. non-int sample_rate). Without this guard,
        # the exception escapes the slot-acquired region without releasing the
        # session token / decrementing the active-WS gauge / unregistering from
        # BackendManagers.
        punct_on = False
        spk_on = False
        diarize_on = False
        try:
            asr_language    = cfg.get("asr_language")  # e.g. "zh" / "Chinese" / "en" / "auto" / None
            tts_language    = cfg.get("tts_language")  # truthy = enable TTS; "auto" = let backend detect
            # "auto" enables TTS but tells downstream (sentence buffer + backend) to
            # not assume a language — backends with auto-detect (e.g. qwen3) will pick
            # one from the text content; SentenceBuffer falls back to a regex splitter.
            tts_language_norm = None if tts_language == "auto" else tts_language
            # Normalize common client-supplied aliases to the lowercase full names
            # qwen3 TTS expects ("chinese"/"english"/...). Sherpa TTS ignores the
            # value entirely, so this is a no-op there.
            if tts_language_norm:
                _TTS_LANG_ALIAS = {
                    "zh": "chinese", "zh-cn": "chinese", "zh-hans": "chinese",
                    "en": "english", "en-us": "english", "en-gb": "english",
                    "ja": "japanese", "jp": "japanese",
                    "ko": "korean", "kr": "korean",
                }
                key = tts_language_norm.strip().lower()
                tts_language_norm = _TTS_LANG_ALIAS.get(key, key)
            tts_voice       = cfg.get("tts_voice")
            tts_speaker_id = cfg.get("tts_speaker_id")
            tts_speed       = cfg.get("tts_speed")
            # Resolve speaker once at config time — avoids mid-session changes
            # (e.g. unregister) affecting later sentences in the same session.
            tts_speaker_kwargs: dict = {}
            if tts_speaker_id is not None and tts_language:
                from server.core.tts_speakers import speaker_kwargs_for_id
                if tts_service.is_ready():
                    tts_speaker_kwargs = speaker_kwargs_for_id(
                        int(tts_speaker_id), tts_service.get_backend().model_id
                    )
            sample_rate     = int(cfg.get("sample_rate", 16000))
            preroll_cap     = int(sample_rate * _vad_preroll_ms() / 1000)
            vad_backend     = cfg.get("vad", _default_vad_backend() if asr_language else "none")
            vad_silence_ms  = int(cfg.get("vad_silence_ms", _default_vad_silence_ms()))
            multi_utterance = bool(cfg.get("multi_utterance", False))
            # Optional, default-off final-payload enrichments (config overrides env).
            from server.core import punctuation as _punct_mod
            from server.core import speaker_embedding as _spk_mod
            from server.core import diarization as _diar_mod
            punct_on = _flag_or(cfg.get("punctuate"), _punct_mod.punctuation_enabled())
            spk_on   = _flag_or(cfg.get("speaker_embedding"), _spk_mod.speaker_embedding_enabled())
            diarize_on = _flag_or(cfg.get("diarize"), _diar_mod.diarize_enabled())
            # Diarization clusters over embeddings, so it implies embedding.
            if diarize_on:
                spk_on = True
        except (ValueError, TypeError) as _cfg_exc:
            try:
                await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                    "error": f"invalid config field: {_cfg_exc}"})
            except Exception:
                pass
            try:
                await ws.close(code=1003)
            except Exception:
                pass
            _v2v_release_early()
            try:
                reset_request_context(_v2v_ctx_tokens)
            except BaseException:
                pass
            return
    
        if not asr_language and not tts_language:
            await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                "error": "config must enable asr_language and/or tts_language"})
            await ws.close(code=1003); _v2v_release_early(); return

        if realtime_v2:
            canonical_session = cfg.get("_canonical_session")
            canonical_session = (
                canonical_session if isinstance(canonical_session, dict) else {}
            )
            if realtime_provider_name in {"openai", "qwen"}:
                from server.core.realtime_relay import relay_cloud_realtime
                try:
                    await relay_cloud_realtime(
                        ws,
                        provider_name=realtime_provider_name,
                        canonical_session=canonical_session,
                        downstream_adapter=realtime_adapter,
                        input_rate=sample_rate,
                        create_response=bool(cfg.get("_create_response", False)),
                        interrupt_response=bool(
                            cfg.get("_interrupt_response", True)
                        ),
                    )
                finally:
                    _v2v_release_early()
                    try:
                        reset_request_context(_v2v_ctx_tokens)
                    except BaseException:
                        pass
                return
            requested_create_response = bool(cfg.get("_create_response", False))
            server_loop_available = (
                os.environ.get("OVS_V2V_ENGINE") == "voxedge"
                and _env_truthy(os.environ.get("OVS_V2V_SERVER_LOOP"))
            )
            if requested_create_response and not server_loop_available:
                await ws.send_json(realtime_adapter.translate({
                    "type": v2v_proto.SERVER_ERROR,
                    "code": "unsupported_create_response",
                    "error": (
                        "create_response=true requires OVS_V2V_ENGINE=voxedge "
                        "and OVS_V2V_SERVER_LOOP=1"
                    ),
                    "param": "session.audio.input.turn_detection.create_response",
                })[0])
                await ws.close(code=1003); _v2v_release_early(); return
            realtime_adapter.input_sample_rate = sample_rate
            await ws.send_json(realtime_adapter.session_updated(
                canonical_session,
                create_response=(
                    requested_create_response and server_loop_available
                ),
                interrupt_response=bool(cfg.get("_interrupt_response", True)),
            ))

        # ── Phase 1b: optional voxedge ConversationEngine path ──────────
        # Feature-flag a parallel implementation that delegates the V2V
        # orchestration to voxedge's importable ConversationEngine instead of
        # the in-handler dispatcher/asr_out_task/tts_out_task below. Gated on
        # OVS_V2V_ENGINE == "voxedge"; ANY other value (incl. unset) takes the
        # untouched legacy path so existing behavior is bit-for-bit unchanged.
        #
        # Admission/auth/4429/close-code stay in THIS handler (transport
        # layer); the engine never touches them. We branch only AFTER config
        # parse + try_acquire_ws so the engine path inherits a held slot, and
        # we release that slot (+ unregister managers, dec metrics, reset ctx)
        # in the engine path's own finally — mirroring the legacy finally.
        if os.environ.get("OVS_V2V_ENGINE") == "voxedge":
            # Single-endpoint-detector guard (docs/CONFIGURATION.md
            # "Streaming ASR endpointing: pick exactly one detector").
            # Default: loud warning (back-compat). With
            # OVS_V2V_SINGLE_ENDPOINT_STRICT=1: reject the session.
            _ep_msg = _endpoint_detector_conflict(
                vad_backend, _get_asr_backend() if asr_language else None
            )
            if _ep_msg is not None:
                if _env_truthy(os.environ.get("OVS_V2V_SINGLE_ENDPOINT_STRICT")):
                    logger.error(_ep_msg + " (strict mode: rejecting session)")
                    try:
                        await ws.send_json({
                            "type": v2v_proto.SERVER_ERROR,
                            "code": "endpoint_detector_conflict",
                            "error": _ep_msg,
                        })
                        await ws.close(code=1003)
                    except BaseException:
                        pass
                    _v2v_release_early()
                    return
                logger.warning(_ep_msg)
            try:
                await _v2v_stream_via_engine(
                    _RealtimeV2WebSocketProxy(ws, realtime_adapter)
                    if realtime_v2
                    else ws,
                    cfg,
                    asr_language=asr_language,
                    tts_language=tts_language,
                    tts_language_norm=tts_language_norm,
                    sample_rate=sample_rate,
                    vad_backend=vad_backend,
                    vad_silence_ms=vad_silence_ms,
                    multi_utterance=multi_utterance,
                    coord=coord,
                    asr_stream_options=_v2v_asr_stream_options(cfg),
                )
            finally:
                # Same admission/teardown bookkeeping the legacy finally does:
                # release the SessionLimiter slot, unregister from the
                # BackendManager(s), decrement the active-WS gauge. Idempotent.
                try:
                    _v2v_release_early()
                except BaseException:
                    pass
                try:
                    reset_request_context(_v2v_ctx_tokens)
                except BaseException:
                    pass
                logger.info("v2v stream closed (engine path)")
            return

        # ── Stage 2: bring up the backends ──────────────────────────────
        asr_be = None
        asr_manager = None  # ASRSessionManager — owns per-utterance lifecycle.
        asr_enabled = False
        vad = None
        if asr_language:
            asr_be = _get_asr_backend()
            if asr_be is None or not asr_be.is_ready() or not asr_be.has_capability(ASRCapability.STREAMING):
                await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                    "error": "asr_language requested but no streaming ASR backend ready"})
                # Codex round-4 GAP A: ws.close() itself can raise (e.g. socket
                # already torn down) — must not skip _v2v_release_early() or
                # the session slot leaks. Wrap close, then release unconditionally.
                try:
                    await ws.close(code=1011)
                except BaseException:
                    pass
                _v2v_release_early()
                return
            # Defer stream creation until first speech-start (or first audio
            # without VAD) — the manager creates a fresh stream per utterance.
            from server.core.asr_session_manager import (
                ASRSessionManager,
                ASRSessionUnavailable,
            )
            asr_manager = ASRSessionManager(
                backend=asr_be,
                language=asr_language,
                coord=coord,
                executor=_get_asr_executor(),
                # Same session-scoped options the engine path forwards: the
                # legacy path is what the shipped RK/Jetson composes actually
                # run (no OVS_V2V_ENGINE=voxedge), so ignoring the documented
                # override here would make it silently dead.
                stream_options=_v2v_asr_stream_options(cfg),
            )
            asr_enabled = True
            # VAD init runs in executor: silero ONNX first-load takes ~500ms and
            # would otherwise stall the event loop. ValueError (e.g. unsupported
            # sample rate) is a hard config error → reject and close. Other init
            # failures fall back to no-VAD with a warning.
            try:
                _loop_init = asyncio.get_event_loop()
                vad = await _loop_init.run_in_executor(
                    None,
                    lambda: vad_mod.create_vad(vad_backend, sample_rate=sample_rate, silence_ms=vad_silence_ms),
                )
            except ValueError as e:
                await ws.send_json({"type": v2v_proto.SERVER_ERROR, "error": f"VAD config: {e}"})
                await ws.close(code=1003); _v2v_release_early(); return
            except Exception as e:
                logger.warning("v2v VAD init (%s) failed: %s — running without VAD", vad_backend, e)
                vad = None
    
        tts_be = None
        tts_buffer = None
        if tts_language:
            if not tts_service.is_ready() or not tts_service.has_capability(TTSCapability.STREAMING):
                await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                    "error": "tts_language requested but no streaming TTS backend ready"})
                # Codex round-4 GAP A: same guard as ASR backend-not-ready
                # above — ws.close() raising must not skip slot release.
                try:
                    await ws.close(code=1011)
                except BaseException:
                    pass
                _v2v_release_early()
                return
            tts_be = tts_service.get_backend()
            low_latency_tts = os.environ.get("OVS_TTS_LOW_LATENCY_CHUNKING", "1").lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
            if low_latency_tts:
                tts_buffer = v2v_proto.LowLatencyTTSBuffer(language=tts_language_norm)
            else:
                tts_buffer = v2v_proto.SentenceBuffer(language=tts_language_norm)
    
        logger.info("v2v stream opened (asr=%s tts=%s vad=%s spk_id=%s spk_kwargs=%s)",
                    asr_language or "off", tts_language or "off", vad_backend if asr_language else "off",
                    tts_speaker_id, list(tts_speaker_kwargs.keys()) if tts_speaker_kwargs else None)
    
        # ── Stage 3: per-connection state + write serialization ─────────
        send_lock = asyncio.Lock()
        state = {
            # Per-utterance ASR endpoint signalling. Replaces the old
            # asr_eos / vad_endpoint / vad_endpoint_pending flags now that
            # ASRSessionManager owns the stream lifecycle.
            "asr_session_closed": False,   # client explicitly ended ASR (asr_eos or ws close)
            "endpoint_pending":   None,    # ("vad" | "client_eos"), set by dispatcher,
                                           # consumed by asr_out_task
            "endpoint_pending_gen": None,  # generation tag for endpoint_pending;
                                           # if it no longer matches asr_active_gen
                                           # by the time asr_out_task observes it,
                                           # the endpoint belongs to a preempted
                                           # utterance and must NOT fire finalize
                                           # against the new one (gen-race fix,
                                           # codex root-cause 2026-05-19).
            "asr_prepare_task": None,      # optional same-generation
                                           # prepare_finalize task for low
                                           # dialogue EOU latency.
            "asr_prepare_gen": None,
            "tts_flush":        False,   # set when client tts_flush
            "current_tts_task": None,    # running TTS synth task (cancellable)
            "current_tts_stop": None,    # threading.Event to signal synth thread
                                         # to stop on barge-in (avoids orphan
                                         # synth blocking the TTS executor)
            "tts_started":      False,   # tts_started frame sent for current sentence
            "client_closed":    False,
            "asr_active":       False,   # tracks whether manager.on_speech_start
                                         # has been called for the current utterance
            "asr_active_gen":   0,       # generation tagged onto asr_active so a
                                         # stale finalize doesn't clear a fresh
                                         # utterance's asr_active flag (BUG 2)
            "asr_started_once": False,  # single-turn backend endpoint streams
                                         # must not reopen on trailing silence
            "asr_audio_samples_accepted": 0,  # accepted audio duration for
                                         # per-stream frontend EOU safety gate
            "preroll":          [],      # ring of recent pre-speech sample
                                         # frames; replayed on frontend-VAD
                                         # speech-start to back-fill the word
                                         # onset silero clips (first-word drop)
            "preroll_samples":  0,       # total samples held in "preroll"
            "endpoint_finalize_pending": False,  # set when dispatcher already
                                         # accepted the speech-end chunk and the
                                         # asr_out_task should finalize next tick
                                         # (BUG 3: avoids the flag-set/audio-accept
                                         # race that lost the tail of utterances)
            # SLV #2 (2026-05-27): per-ASR-turn wall-clock deadline. Set when a
            # turn starts (on_speech_start), cleared when it cleanly finalizes
            # / cancels. asr_out_task polls this every tick; on expiry it forces
            # a cancel + restart_worker so the WorkerIO semaphore slot is freed
            # and the v2v handler can unwind (releases the SessionLimiter slot,
            # preventing 4429 mute from a stuck Qwen3 inference).
            "asr_turn_started_at": None,
            # Optional speaker-embedding: per-utterance audio buffer (only
            # populated when spk_on; cleared after each finalize). sample_rate
            # captured here so the finalize closure needn't reach for it.
            "spk_seg": [],
            "sample_rate": sample_rate,
            # P0b/P1: session-cumulative speech-sample counter for diarization
            # timestamps. NOT reset per turn (unlike asr_audio_samples_accepted)
            # so start/end are relative to session start. TODO(P5): inter-turn
            # silence is not counted here, so the timeline is speech-compacted —
            # fine for ordering blind clusters, not wall-clock accurate.
            "diar_samples": 0,
        }
        # One OnlineDiarizer per v2v session when diarize_on (holds running
        # centroids); None otherwise (zero overhead).
        _v2v_diarizer = _diar_mod.make_session_diarizer() if diarize_on else None
        tts_q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
    
        async def send_json(payload):
            async with send_lock:
                try:
                    if realtime_v2:
                        for event in realtime_adapter.translate(payload):
                            await ws.send_json(event)
                    else:
                        await ws.send_json(payload)
                except Exception:
                    state["client_closed"] = True
    
        async def send_bytes(data):
            async with send_lock:
                try:
                    # V1 sends a standalone uint32 sample-rate header once.
                    # V2 negotiated the format in session.updated, so suppress
                    # that header and keep every binary frame pure PCM.
                    if realtime_v2 and not state.get("v2_binary_seen") and len(data) == 4:
                        state["v2_binary_seen"] = True
                        return
                    state["v2_binary_seen"] = True
                    await ws.send_bytes(data)
                except Exception:
                    state["client_closed"] = True
    
        async def send_error(msg):
            await send_json({"type": v2v_proto.SERVER_ERROR, "error": msg})

        def _clear_asr_prepare_state() -> None:
            state["asr_prepare_task"] = None
            state["asr_prepare_gen"] = None

        def _schedule_asr_prepare(reason: str) -> None:
            """Run stream.prepare_finalize ahead of EOS when supported.

            The normal generation-gated finalize path remains authoritative.
            This helper only hides the expensive tail encode/decoder prep under
            client or frontend-VAD EOU lead time.
            """
            if asr_manager is None or not state.get("asr_active"):
                return
            gen = int(state.get("asr_active_gen") or 0)
            existing = state.get("asr_prepare_task")
            if (
                existing is not None
                and not existing.done()
                and state.get("asr_prepare_gen") == gen
            ):
                return

            async def _run_prepare() -> None:
                try:
                    async with coord.acquire("asr"):
                        fn = getattr(asr_manager, "prepare_finalize_for_generation", None)
                        if fn is not None:
                            ran_gen, prepared = await fn(gen)
                        else:
                            if getattr(asr_manager, "current_generation", None) != gen:
                                return
                            stream = getattr(asr_manager, "stream", None)
                            prepare = getattr(stream, "prepare_finalize", None)
                            if prepare is None:
                                return
                            loop2 = asyncio.get_event_loop()
                            await loop2.run_in_executor(_get_asr_executor(), prepare)
                            ran_gen, prepared = gen, True
                    logger.debug(
                        "v2v ASR prepare_finalize finished reason=%s gen=%s prepared=%s",
                        reason,
                        ran_gen,
                        prepared,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug(
                        "v2v ASR prepare_finalize failed reason=%s gen=%s",
                        reason,
                        gen,
                        exc_info=True,
                    )

            state["asr_prepare_gen"] = gen
            state["asr_prepare_task"] = asyncio.create_task(_run_prepare())

        def _asr_stream_prefers_backend_endpoint_vad() -> bool:
            """Return True only for streams that explicitly own endpoint VAD.

            Some ASR streams, notably RK Qwen3 true-streaming, run their own
            endpoint detector over the exact encoder buffer they need for final
            decode. For those streams an outer VAD SPEECH_START during an
            active turn is still a client/TTS barge-in signal, but it must not
            preempt and recreate the ASR stream. Legacy backends do not expose
            this opt-in flag and keep the old preempt behavior.
            """
            if asr_manager is None:
                return False
            stream = getattr(asr_manager, "stream", None)
            return bool(getattr(stream, "prefer_backend_endpoint_vad", False))

        def _asr_manager_idle() -> bool:
            """True only when the manager is terminally IDLE (no live turn).

            The desync we heal lands SPECIFICALLY in IDLE: a cancel-timeout
            worker restart or an exhausted rebuild ladder both end in IDLE while
            our per-connection ``asr_active`` / ``asr_active_gen`` still point at
            the now-dead generation — so a fresh speech-start gets swallowed as
            barge-in-only, the generation never advances, and every finalize is
            rejected until a full container restart.

            We key off IDLE and NOT merely "not ACTIVE": FINALIZING /
            CANCELLING / ERROR_REBUILD are transient states a HEALTHY turn
            passes through, and treating those as a desync would misfire the
            re-open / reconcile on a live utterance (backend-agnostic safety).
            """
            st = getattr(asr_manager, "state", None)
            return str(getattr(st, "value", st)) == "idle"

        def _asr_backend_prefers_backend_endpoint_vad() -> bool:
            return bool(getattr(asr_be, "prefer_backend_endpoint_vad", False))

        def _asr_stream_allows_frontend_eou_finalize() -> bool:
            if asr_manager is None:
                return False
            stream = getattr(asr_manager, "stream", None)
            return bool(getattr(stream, "allow_frontend_eou_finalize", False))

        def _asr_stream_frontend_eou_min_audio_s() -> float:
            if asr_manager is None:
                return 0.0
            stream = getattr(asr_manager, "stream", None)
            try:
                return max(
                    0.0,
                    float(getattr(stream, "frontend_eou_min_audio_s", 0.0) or 0.0),
                )
            except (TypeError, ValueError):
                return 0.0
    
        # ── Stage 4: tasks ──────────────────────────────────────────────
    
        async def dispatcher():
            """Receive incoming binary (audio) + text (control) frames."""
            try:
                while not state["client_closed"]:
                    try:
                        msg = await asyncio.wait_for(
                            ws.receive(), timeout=_v2v_idle_timeout_s
                        )
                    except asyncio.TimeoutError:
                        # Half-open / silent client mid-session: no frame for
                        # the idle window. Treat identically to a client
                        # disconnect (see the websocket.disconnect branch
                        # below) so the work tasks are cancelled and the
                        # SessionLimiter slot releases fast instead of wedging
                        # this receive forever.
                        logger.warning(
                            "v2v: idle timeout (%.0fs) awaiting client frame — "
                            "releasing slot as half-open client",
                            _v2v_idle_timeout_s,
                        )
                        state["client_closed"] = True
                        for _wt in work_tasks:
                            if not _wt.done():
                                _wt.cancel()
                        break
                    if state["client_closed"]:
                        break
                    if msg.get("type") == "websocket.disconnect":
                        state["client_closed"] = True
                        # Fast slot release: cancel the (possibly worker-blocked)
                        # work tasks so the SessionLimiter slot frees within ~1s
                        # instead of waiting up to OVS_ASR_TURN_TIMEOUT_S (45s)
                        # for the per-turn deadline. asr_out_task's blocking
                        # awaits (finalize / get_partial via run_in_executor) are
                        # not shielded, so cancellation propagates immediately.
                        for _wt in work_tasks:
                            if not _wt.done():
                                _wt.cancel()
                        break
                    # binary → ASR input
                    data = msg.get("bytes")
                    if data:
                        if not asr_enabled:
                            continue  # ignored in TTS-only mode
                        # After session close, drop further audio. Spec: client
                        # must open a new WebSocket to start another session.
                        if state["asr_session_closed"]:
                            continue
                        if (
                            not multi_utterance
                            and state["asr_started_once"]
                            and not state["asr_active"]
                        ):
                            continue
                        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        speech_started_now = False
                        speech_ended_now = False
                        if vad is not None:
                            event = vad.process(samples)
                            if event == vad_mod.VADSession.SPEECH_START:
                                # Notify client FIRST so it can stop buffering /
                                # playing TTS audio, then perform the server-side
                                # barge-in (cancel in-flight TTS, open fresh ASR).
                                await send_json({
                                    "type": v2v_proto.SERVER_VAD_EVENT,
                                    "event": v2v_proto.VAD_EVENT_SPEECH_START,
                                })
                                if realtime_v2 and realtime_adapter is not None:
                                    realtime_adapter.mark_cancelled("turn_detected")
                                if not multi_utterance and state["asr_started_once"]:
                                    if (
                                        state["asr_active"]
                                        and _asr_stream_prefers_backend_endpoint_vad()
                                    ):
                                        speech_started_now = True
                                    continue
                                # Auto barge-in: cancel any in-flight TTS, then
                                # open a fresh ASR utterance (pre-empts any
                                # still-active session per spec).
                                t = state["current_tts_task"]
                                if t is not None and not t.done():
                                    t.cancel()
                                stop = state["current_tts_stop"]
                                if stop is not None:
                                    stop.set()
                                if (
                                    state["asr_active"]
                                    and _asr_stream_prefers_backend_endpoint_vad()
                                    # Only keep accumulating if the manager
                                    # hasn't terminally dropped to IDLE. If it
                                    # self-dropped (worker restart / rebuild
                                    # exhausted) while asr_active is still True,
                                    # treating this as barge-in-only would never
                                    # re-open the stream → generation frozen, ASR
                                    # wedged. Fall through to on_speech_start.
                                    and not _asr_manager_idle()
                                ):
                                    # Backend-owned endpoint streams keep
                                    # accumulating audio across outer VAD
                                    # speech blips. Treat this event as
                                    # barge-in for TTS/client only.
                                    speech_started_now = True
                                else:
                                    try:
                                        async with coord.acquire("asr"):
                                            new_gen = await asr_manager.on_speech_start()
                                    except ASRSessionUnavailable as e:
                                        # Race #1: ASR worker rebuild ladder
                                        # exhausted — surface to client + skip
                                        # this turn rather than silently
                                        # accepting audio with no transcript.
                                        logger.warning(
                                            "v2v: on_speech_start failed (VAD): %s", e
                                        )
                                        await send_json({
                                            "type": v2v_proto.SERVER_ERROR,
                                            "error": "asr_unavailable",
                                        })
                                        state["asr_active"] = False
                                        state["endpoint_pending"] = None
                                        state["endpoint_pending_gen"] = None
                                        continue
                                    # Clear any stale endpoint from the previous
                                    # utterance — a VAD speech-end that was pending
                                    # finalize while this new speech-start preempted
                                    # it must NOT cause asr_out_task to call
                                    # finalize() against the fresh generation
                                    # (codex root-cause 2026-05-19: stale endpoint
                                    # firing on the wrong generation).
                                    state["endpoint_pending"] = None
                                    state["endpoint_pending_gen"] = None
                                    _clear_asr_prepare_state()
                                    state["asr_active"] = True
                                    state["asr_active_gen"] = new_gen
                                    state["asr_audio_samples_accepted"] = 0
                                    state["asr_turn_started_at"] = loop.time()
                                    state["asr_started_once"] = True
                                    speech_started_now = True
                                    # Back-fill the speech onset: replay the
                                    # pre-speech preroll ring into the fresh
                                    # stream BEFORE this chunk so the decoder
                                    # sees the full word silero clipped while
                                    # latching SPEECH_START (first-word drop,
                                    # real-machine 2026-06-15). Chronological
                                    # order is preserved: preroll frames first,
                                    # then the trigger chunk fed at the normal
                                    # accept_audio() below.
                                    if state["preroll"]:
                                        pre = np.concatenate(state["preroll"])
                                        async with coord.acquire("asr"):
                                            await asr_manager.accept_audio(pre)
                                        state["asr_audio_samples_accepted"] += int(len(pre))
                                        if spk_on:
                                            state["spk_seg"].append(pre)
                                            state["diar_samples"] += int(len(pre))
                                    state["preroll"] = []
                                    state["preroll_samples"] = 0
                            elif event == vad_mod.VADSession.SPEECH_END:
                                # Defer setting endpoint_pending until AFTER we
                                # accept this final chunk below — otherwise the
                                # asr_out_task observes the flag and calls
                                # finalize() while the tail audio is still
                                # in-flight, silently dropping it (BUG 3).
                                speech_ended_now = True
                        # No-VAD mode opens lazily on first audio. Backends
                        # that own endpoint VAD need the same first-frame
                        # behavior even when frontend VAD is enabled; otherwise
                        # the frontend VAD speech_start gate drops leading
                        # context and shifts the final encoder buffer.
                        if (
                            (vad is None or _asr_backend_prefers_backend_endpoint_vad())
                            and not state["asr_active"]
                            and state["endpoint_pending"] is None
                            # `asr_started_once` is a one-shot latch that is never
                            # reset, so in no-VAD mode it would open the ASR stream
                            # only for the FIRST utterance — the 2nd+ utterance on a
                            # persistent multi_utterance session would find
                            # asr_active=False and never re-open, so accept_audio()
                            # below is skipped and the audio is silently dropped
                            # (0 partial/final). In multi_utterance the session is
                            # explicitly kept alive for more turns, so re-open every
                            # utterance. (`not asr_active` above prevents double-open.)
                            and (multi_utterance or not state["asr_started_once"])
                        ):
                            try:
                                async with coord.acquire("asr"):
                                    new_gen = await asr_manager.on_speech_start()
                            except ASRSessionUnavailable as e:
                                # Race #1: same as VAD path above.
                                logger.warning(
                                    "v2v: on_speech_start failed (no-VAD): %s", e
                                )
                                await send_json({
                                    "type": v2v_proto.SERVER_ERROR,
                                    "error": "asr_unavailable",
                                })
                                state["asr_active"] = False
                                continue
                            state["endpoint_pending"] = None
                            state["endpoint_pending_gen"] = None
                            _clear_asr_prepare_state()
                            state["asr_active"] = True
                            state["asr_active_gen"] = new_gen
                            state["asr_audio_samples_accepted"] = 0
                            state["asr_turn_started_at"] = loop.time()
                            state["asr_started_once"] = True
                        if state["asr_active"]:
                            async with coord.acquire("asr"):
                                await asr_manager.accept_audio(samples)
                            state["asr_audio_samples_accepted"] += int(len(samples))
                            if spk_on:
                                state["spk_seg"].append(samples)
                                state["diar_samples"] += int(len(samples))
                        # Now safe to flag the endpoint — audio chunk that
                        # carried the speech-end has been delivered to the
                        # stream. asr_out_task will pick this up on the next
                        # poll and call finalize().
                        if speech_ended_now:
                            backend_owns_endpoint = _asr_stream_prefers_backend_endpoint_vad()
                            accepted_audio_s = (
                                state.get("asr_audio_samples_accepted", 0)
                                / max(float(sample_rate), 1.0)
                            )
                            frontend_eou_may_finalize = (
                                not backend_owns_endpoint
                                or (
                                    _asr_stream_allows_frontend_eou_finalize()
                                    and accepted_audio_s >= _asr_stream_frontend_eou_min_audio_s()
                                )
                            )
                            if frontend_eou_may_finalize:
                                _schedule_asr_prepare("vad_speech_end")
                                state["endpoint_pending"] = "vad"
                                state["endpoint_pending_gen"] = state["asr_active_gen"]
                                if not multi_utterance:
                                    state["asr_session_closed"] = True
                            elif not multi_utterance:
                                # Keep accepting trailing silence into the
                                # active backend-owned stream so its endpoint
                                # detector can fire. Reopen is already blocked
                                # by asr_started_once once the stream becomes
                                # inactive.
                                pass
                            # Notify client of VAD speech_end so it can update
                            # its state machine (e.g. show "thinking" indicator,
                            # await asr_final). Sent AFTER endpoint_pending is
                            # latched to keep ordering deterministic w.r.t. the
                            # asr_final that follows from asr_out_task.
                            await send_json({
                                "type": v2v_proto.SERVER_VAD_EVENT,
                                "event": v2v_proto.VAD_EVENT_SPEECH_END,
                            })
                        # While no ASR turn is open, keep a short rolling ring of
                        # the most recent frames so the next frontend-VAD
                        # speech-start can replay the onset (see open branch). We
                        # only buffer pre-speech audio: once asr_active, frames go
                        # straight to accept_audio(), so the trigger chunk is fed
                        # exactly once and never double-counted.
                        if preroll_cap > 0 and not state["asr_active"]:
                            state["preroll"].append(samples)
                            state["preroll_samples"] += int(len(samples))
                            while (
                                state["preroll_samples"] > preroll_cap
                                and len(state["preroll"]) > 1
                            ):
                                state["preroll_samples"] -= int(
                                    len(state["preroll"].pop(0))
                                )
                        continue
                    # text → JSON control
                    text = msg.get("text", "")
                    if not text:
                        continue
                    try:
                        payload = _json.loads(text)
                    except (ValueError, TypeError):
                        continue
                    typ = payload.get("type")
                    if realtime_v2:
                        if typ == v2v_proto.CLIENT_INPUT_AUDIO_BUFFER_COMMIT:
                            typ = v2v_proto.CLIENT_ASR_EOS
                        elif typ == v2v_proto.CLIENT_RESPONSE_CANCEL:
                            realtime_adapter.mark_cancelled("client_cancelled")
                            typ = v2v_proto.CLIENT_ABORT
                        elif typ == v2v_proto.CLIENT_INPUT_AUDIO_BUFFER_CLEAR:
                            typ = v2v_proto.CLIENT_ABORT
                        elif typ == v2v_proto.CLIENT_SESSION_UPDATE:
                            session = payload.get("session")
                            session = session if isinstance(session, dict) else {}
                            await send_json(realtime_adapter.session_updated(session))
                            continue
                        elif typ == v2v_proto.CLIENT_DIRECT_SPEAK:
                            speech = payload.get("speech")
                            speech = speech if isinstance(speech, dict) else {}
                            text_value = str(speech.get("text") or "")
                            realtime_adapter.mark_direct_speak()
                            if tts_buffer is not None and text_value:
                                for sentence in tts_buffer.add(text_value):
                                    await tts_q.put(sentence)
                                for sentence in tts_buffer.flush():
                                    await tts_q.put(sentence)
                            state["tts_flush"] = True
                            continue
                        elif typ == v2v_proto.CLIENT_CONVERSATION_ITEM_TRUNCATE:
                            try:
                                audio_end_ms = max(
                                    0, int(payload.get("audio_end_ms", 0))
                                )
                            except (TypeError, ValueError):
                                await send_json({
                                    "type": v2v_proto.SERVER_ERROR,
                                    "code": "invalid_audio_end_ms",
                                    "error": (
                                        "audio_end_ms must be a non-negative integer"
                                    ),
                                    "param": "audio_end_ms",
                                })
                                continue
                            await send_json({
                                "type": v2v_proto.SERVER_CONVERSATION_ITEM_TRUNCATED,
                                "item_id": payload.get("item_id"),
                                "content_index": payload.get("content_index", 0),
                                "audio_end_ms": audio_end_ms,
                            })
                            continue
                        elif typ == v2v_proto.CLIENT_CONVERSATION_RESET:
                            realtime_adapter.mark_cancelled("conversation_reset")
                            await send_json({
                                "type": v2v_proto.SERVER_CONVERSATION_RESET_DONE,
                            })
                            typ = v2v_proto.CLIENT_ABORT
                    if typ == v2v_proto.CLIENT_PING:
                        # Idle keepalive: intentionally does nothing. Arriving
                        # here already reset the idle watchdog (the ws.receive()
                        # above returned), which is the whole point — a live but
                        # silent client (wake-word mode, quiet room) must not be
                        # reaped as half-open. Handled explicitly rather than
                        # relying on the unknown-type fall-through so a future
                        # refactor that rejects unknown types can't silently
                        # start killing keepalives.
                        continue
                    if typ == v2v_proto.CLIENT_TEXT and tts_buffer is not None:
                        for sentence in tts_buffer.add(payload.get("text", "")):
                            await tts_q.put(sentence)
                    elif typ == v2v_proto.CLIENT_TTS_FLUSH:
                        if tts_buffer is not None:
                            for sentence in tts_buffer.flush():
                                await tts_q.put(sentence)
                        state["tts_flush"] = True
                    elif typ == v2v_proto.CLIENT_ASR_EOS:
                        state["endpoint_pending"] = "client_eos"
                        state["endpoint_pending_gen"] = state["asr_active_gen"]
                        if not multi_utterance:
                            state["asr_session_closed"] = True
                    elif typ in {
                        getattr(v2v_proto, "CLIENT_ASR_PREPARE", "asr_prepare"),
                        "prepare",
                        "pre_eou",
                        "prepare_finalize",
                    }:
                        _schedule_asr_prepare("client_prepare")
                    elif typ == v2v_proto.CLIENT_ABORT:
                        t = state["current_tts_task"]
                        if t is not None and not t.done():
                            t.cancel()
                        stop = state["current_tts_stop"]
                        if stop is not None:
                            stop.set()
                        # Drain queue so flush doesn't replay queued sentences
                        while not tts_q.empty():
                            try: tts_q.get_nowait()
                            except asyncio.QueueEmpty: break
                        # Cancel the in-flight ASR utterance too — spec: barge-in
                        # discards pending finals and resets to IDLE.
                        #
                        # ...unless the client asked us to keep it. In a
                        # speech-driven barge-in the in-flight utterance is the
                        # one the user is interrupting WITH, and cancelling it
                        # discards every word spoken before the client's
                        # barge-in threshold fired: measured on rk3588
                        # (2026-09-21), "Stop, please answer in one sentence."
                        # reached the LLM as "One sentence." because this cancel
                        # restarted the ASR stream 0.3 s after the barge-in. The
                        # client says which kind of abort this is; see
                        # ``keep_asr`` in server/core/v2v.py.
                        #
                        # Bounded, because keeping the stream also keeps whatever
                        # it accumulated while TTS was playing. A speech-driven
                        # barge-in fires 0.5-2.5 s after the user starts talking
                        # (echo guard + partial latency; 2.4 s measured on RK),
                        # so a longer utterance is mostly pre-barge-in audio:
                        # speaker echo on boards without AEC, or noise. Past the
                        # bound we cancel as before. This is engine-independent:
                        # keeping the stream is the same path as a user who kept
                        # talking, which every ASR engine already handles
                        # (Jetson TRT rotates segments at segment_cap_sec=5.5 s,
                        # RK rolls/commits its window, whisper/sherpa segment).
                        keep_asr = bool(payload.get("keep_asr"))
                        if keep_asr:
                            kept_s = int(
                                state.get("asr_audio_samples_accepted") or 0
                            ) / float(getattr(asr_be, "sample_rate", 16000) or 16000)
                            max_keep_s = _keep_asr_max_s()
                            if kept_s > max_keep_s:
                                logger.info(
                                    "v2v abort(keep_asr) ignored: in-flight "
                                    "utterance is %.1fs > %.1fs, most of it "
                                    "predates the barge-in; cancelling ASR",
                                    kept_s, max_keep_s,
                                )
                                keep_asr = False
                            else:
                                logger.info(
                                    "v2v abort(keep_asr): TTS cancelled, ASR "
                                    "utterance kept (%.1fs so far)", kept_s,
                                )
                        if (
                            not keep_asr
                            and asr_manager is not None
                            and state["asr_active"]
                        ):
                            async with coord.acquire("asr"):
                                await asr_manager.cancel("bargein")
                            state["asr_active"] = False
                            state["asr_audio_samples_accepted"] = 0
                            state["asr_turn_started_at"] = None
                            _clear_asr_prepare_state()
            except WebSocketDisconnect:
                state["client_closed"] = True
                # See websocket.disconnect branch above: cancel work tasks so
                # the SessionLimiter slot releases fast on client disconnect.
                for _wt in work_tasks:
                    if not _wt.done():
                        _wt.cancel()

        async def asr_out_task():
            """Drive partial polling + per-utterance finalize via the manager.
    
            Each utterance is its own ``ASRSessionManager`` stream. We poll
            the *active* stream (manager.stream) for partials, then on an
            endpoint trigger (VAD speech-end, client asr_eos, or backend
            is_endpoint) we call ``manager.finalize()`` which destroys the
            stream and returns the final text.
            """
            last_streamed_final = None
            # SLV #2 (2026-05-27): wall-clock per-turn deadline. Env-driven so
            # we can dial up on slow Jetsons. Covers the gap where the ASR
            # backend gets wedged inside WorkerIO (qwen3_asr_worker stuck on
            # an inference, GPU OOM deadlock, or stdout reader hung) — none of
            # the existing per-stream timeouts (WorkerIO q.get 60s) fire when
            # the worker is *producing* events that never lead to a final.
            asr_turn_timeout_s = float(
                os.getenv("OVS_ASR_TURN_TIMEOUT_S", "45.0")
            )
            while not state["client_closed"]:
                # ── Wall-clock turn deadline ────────────────────────────
                # Active turn started but hasn't produced a final within
                # the deadline → force cancel + worker restart so the
                # WorkerIO semaphore slot is freed and this handler can
                # unwind cleanly (releases the SessionLimiter slot).
                turn_started = state.get("asr_turn_started_at")
                if (
                    state.get("asr_active")
                    and turn_started is not None
                    and (loop.time() - turn_started) > asr_turn_timeout_s
                ):
                    elapsed = loop.time() - turn_started
                    logger.warning(
                        "v2v ASR turn exceeded %.1fs wall-clock (elapsed=%.1fs); "
                        "aborting turn + force-cancel ASR session",
                        asr_turn_timeout_s, elapsed,
                    )
                    # Step 1: try cooperative cancel with a tight budget.
                    if asr_manager is not None:
                        try:
                            await asyncio.wait_for(
                                asr_manager.cancel("turn_timeout"),
                                timeout=2.0,
                            )
                        except (asyncio.TimeoutError, Exception) as _exc:
                            # cancel itself jammed → escalate to worker
                            # restart directly. restart_worker uses the
                            # default executor (not the wedged ASR slot),
                            # so it cannot deadlock here.
                            logger.error(
                                "v2v ASR cancel timed out / failed (%s); "
                                "force-restarting worker",
                                _exc,
                            )
                            try:
                                fn = getattr(asr_be, "restart_worker", None)
                                if fn is not None:
                                    loop2 = asyncio.get_event_loop()
                                    await loop2.run_in_executor(None, fn)
                            except Exception:
                                logger.exception(
                                    "v2v ASR restart_worker after turn timeout failed"
                                )
                    # Step 2: clear state + emit a final so the client side
                    # cancels its turn promptly (instead of waiting for
                    # its own thinking watchdog). Treat as empty final.
                    state["asr_active"] = False
                    state["asr_audio_samples_accepted"] = 0
                    state["asr_turn_started_at"] = None
                    state["endpoint_pending"] = None
                    state["endpoint_pending_gen"] = None
                    _clear_asr_prepare_state()
                    try:
                        await send_error(
                            f"asr: per-turn deadline {asr_turn_timeout_s:.0f}s "
                            f"exceeded"
                        )
                    except Exception:
                        logger.exception("send_error after asr turn timeout failed")
                    if multi_utterance and not state["asr_session_closed"]:
                        # Multi-utterance: keep running; the next speech_start
                        # will issue a new generation.
                        await asyncio.sleep(0.05)
                        continue
                    else:
                        # Single-utterance: terminate the task. The outer
                        # try/finally will release the slot.
                        return

                # Pull a stream snapshot under the manager's lock so we can
                # tag any partial with the generation it came from. If the
                # generation has advanced by emit-time, drop the partial —
                # it belongs to an utterance that's already been replaced
                # (BUG 4: stale-stream partial leak).
                partial, is_endpoint, partial_gen = "", False, 0
                if state["asr_active"]:
                    try:
                        async with coord.acquire("asr"):
                            partial_gen, partial, is_endpoint = (
                                await asr_manager.get_partial_for_generation()
                            )
                    except Exception:
                        partial, is_endpoint, partial_gen = "", False, 0
                    if partial and partial_gen == asr_manager.current_generation \
                            and partial_gen == state["asr_active_gen"]:
                        await send_json({"type": v2v_proto.SERVER_ASR_PARTIAL,
                                         "text": partial, "is_stable": bool(is_endpoint)})
    
                endpoint_reason = state["endpoint_pending"]
                # Gen-race gate: if endpoint_pending was stamped against a
                # generation that has since been preempted (VAD speech-start
                # of a new utterance, or post-worker-restart on_speech_start),
                # drop it on the floor instead of firing finalize against the
                # *new* active utterance. Without this gate the new utterance
                # gets finalized too early and the manager rejects the result
                # with "finalize result discarded (state=ACTIVE)"
                # (codex root-cause 2026-05-19).
                if (
                    endpoint_reason
                    and state.get("endpoint_pending_gen") is not None
                    and state.get("endpoint_pending_gen") != state["asr_active_gen"]
                ):
                    state["endpoint_pending"] = None
                    state["endpoint_pending_gen"] = None
                    endpoint_reason = None
    
                endpoint_fired = (
                    bool(endpoint_reason)
                    or (is_endpoint and state["asr_active"])
                )
    
                if endpoint_fired:
                    if not multi_utterance:
                        # Single-turn sessions must close the input side as
                        # soon as any endpoint wins. Otherwise queued trailing
                        # silence can arrive after asr_active is cleared and
                        # lazily open a second backend stream.
                        state["asr_session_closed"] = True
                    if (
                        is_endpoint
                        and not endpoint_reason
                        and not multi_utterance
                    ):
                        # Backend-owned endpointing has already observed
                        # enough tail inside the stream. Close the input side
                        # before finalize so queued trailing silence cannot
                        # lazily open a second ASR stream in this single-turn
                        # session.
                        state["asr_session_closed"] = True
                    # Drain pending flag now to avoid double-firing.
                    state["endpoint_pending"] = None
                    state["endpoint_pending_gen"] = None
                    # Emit asr_endpoint only for VAD / backend endpoints,
                    # not client-driven eos.
                    if endpoint_reason != "client_eos":
                        await send_json({"type": v2v_proto.SERVER_ASR_ENDPOINT})
    
                    if state["asr_active"]:
                        finalize_gen = state["asr_active_gen"]
                        prep_task = state.get("asr_prepare_task")
                        if (
                            prep_task is not None
                            and state.get("asr_prepare_gen") == finalize_gen
                            and not prep_task.done()
                        ):
                            try:
                                await prep_task
                            except Exception:
                                pass
                        async with coord.acquire("asr"):
                            ran_gen, final_text, finalize_accepted, detected_language = (
                                await asr_manager.finalize_with_status(
                                    endpoint_reason or "backend_endpoint"
                                )
                            )
                        # Only clear asr_active if the generation we finalized
                        # is still the active one. If a new speech_start
                        # bumped the generation while finalize was in flight,
                        # leaving asr_active=True is correct — audio for the
                        # new utterance must continue to flow (BUG 2).
                        if finalize_accepted and state["asr_active_gen"] == finalize_gen:
                            state["asr_active"] = False
                            state["asr_audio_samples_accepted"] = 0
                            state["asr_turn_started_at"] = None
                            _clear_asr_prepare_state()
                    else:
                        final_text = ""
                        ran_gen = state["asr_active_gen"]
                        finalize_accepted = True
                        detected_language = None

                    if not finalize_accepted:
                        logger.info(
                            "suppressing discarded asr_final from gen=%s current_gen=%s reason=%s",
                            ran_gen,
                            state["asr_active_gen"],
                            endpoint_reason or "backend_endpoint",
                        )
                        # Self-heal desync: the manager can fall to IDLE on its
                        # own (cancel-timeout worker restart / rebuild ladder
                        # exhausted) while we still hold asr_active at the dead
                        # generation. Left alone, every finalize is rejected and
                        # — for backends that treat a fresh speech-start as
                        # barge-in-only while asr_active=True — the generation
                        # never advances, wedging ASR until a full container
                        # restart. Reconcile so the next speech-start opens a
                        # fresh turn (self-heal without a restart). IDLE-only so
                        # a preempt that left the manager ACTIVE on a NEW gen
                        # (legit in-flight turn) is never torn down here.
                        if _asr_manager_idle():
                            state["asr_active"] = False
                            state["asr_audio_samples_accepted"] = 0
                            state["asr_turn_started_at"] = None
                            state["endpoint_pending"] = None
                            state["endpoint_pending_gen"] = None
                            _clear_asr_prepare_state()
                        continue

                    # Optional, default-off final enrichments. No-op (and no
                    # buffered audio) unless a flag is on, so the default v2v
                    # path is byte-identical. Punctuation rewrites final_text
                    # (also improves downstream LLM input); speaker embedding
                    # is computed once from the per-utterance audio buffer and
                    # merged into whichever final_payload is sent below.
                    _spk_fields: dict = {}
                    if punct_on and final_text:
                        try:
                            from server.core import punctuation as _punct
                            final_text = await loop.run_in_executor(
                                None, _punct.add_punctuation, final_text
                            )
                        except Exception:
                            logger.exception("v2v punctuation failed; keeping raw text")
                    if spk_on and state.get("spk_seg"):
                        try:
                            from server.core import speaker_embedding as _spk
                            _seg = state["spk_seg"]
                            _seg_all = np.concatenate(_seg) if len(_seg) > 1 else _seg[0]
                            _sr = int(state.get("sample_rate", 16000))
                            _emb = await loop.run_in_executor(
                                None, _spk.compute_embedding, _seg_all, _sr,
                            )
                            if _emb is not None:
                                _spk_fields = _spk.embedding_payload(_emb)
                                # P0b: session-relative segment time window.
                                _seg_len = int(len(_seg_all))
                                _d_end = state["diar_samples"] / float(_sr) if _sr else 0.0
                                _d_start = (state["diar_samples"] - _seg_len) / float(_sr) if _sr else 0.0
                                _spk_fields["start"] = round(_d_start, 3)
                                _spk_fields["end"] = round(_d_end, 3)
                                # P1: online blind diarization speaker label.
                                if diarize_on and _v2v_diarizer is not None:
                                    try:
                                        _ds = _v2v_diarizer.assign(_emb, _d_start, _d_end)
                                        _spk_fields["speaker"] = _ds.speaker
                                        _spk_fields["speaker_conf"] = round(float(_ds.confidence), 3)
                                    except Exception:
                                        logger.exception("v2v diarize assign failed; skipping label")
                        except Exception:
                            logger.exception("v2v speaker embedding failed; skipping")
                    state["spk_seg"] = []
                    # TODO(P2/v2v): emit a diarization_summary (relabel()) on
                    # session close. Deferred — the v2v close path is complex
                    # (multi-utterance / barge-in); /asr/stream already does it.

                    # Multi-utterance: mid-session finals carry
                    # session_complete=False; close-out final on
                    # asr_session_closed carries True.
                    if multi_utterance:
                        is_closing = state["asr_session_closed"]
                        if is_closing:
                            duplicate = (final_text or "") == (last_streamed_final or "")
                            final_payload = {
                                "type": v2v_proto.SERVER_ASR_FINAL,
                                "text": final_text or "",
                                "session_complete": True,
                                "duplicate_of_streamed": duplicate,
                            }
                            if detected_language:
                                final_payload["language"] = detected_language
                            final_payload.update(_spk_fields)
                            await send_json(final_payload)
                            return
                        else:
                            final_payload = {
                                "type": v2v_proto.SERVER_ASR_FINAL,
                                "text": final_text or "",
                                "session_complete": False,
                            }
                            if detected_language:
                                final_payload["language"] = detected_language
                            final_payload.update(_spk_fields)
                            await send_json(final_payload)
                            last_streamed_final = final_text or ""
                            # keep the loop running for the next utterance
                    else:
                        final_payload = {
                            "type": v2v_proto.SERVER_ASR_FINAL,
                            "text": final_text or "",
                        }
                        if detected_language:
                            final_payload["language"] = detected_language
                        final_payload.update(_spk_fields)
                        await send_json(final_payload)
                        state["client_closed"] = True
                        return
    
                # Exit only when the session is closed and there's nothing
                # left to finalize — single-utterance terminates above on
                # the endpoint; multi-utterance terminates on close-out.
                if state["asr_session_closed"] and not state["asr_active"]:
                    return
    
                await asyncio.sleep(0.05)
    
        async def tts_out_task():
            """Drain sentence queue → synthesize → emit audio.
    
            Sends the 4-byte sample-rate header on first successful synth
            (NOT first attempted synth — so a cancelled-mid-flight first
            sentence doesn't leave the client with a header but no audio).
            """
            sr_header_sent = False
            while not state["client_closed"]:
                # Exit when client said flush and the queue is drained.
                if state["tts_flush"] and tts_q.empty():
                    # Multi-utterance: per-turn flush ends one turn but not
                    # the SESSION. Reset the sticky flag, emit a per-turn
                    # tts_done (session_complete=False mirroring ASR's
                    # mid-session final at :1763-1779), then loop back to
                    # wait for the next turn. Without this the task returns
                    # after round 1 → asyncio.gather() unblocks → WS closes,
                    # which is the "TTS stuck after round 1" bug.
                    if multi_utterance and not state.get("asr_session_closed", False):
                        state["tts_flush"] = False
                        if not state["client_closed"]:
                            await send_json({
                                "type": v2v_proto.SERVER_TTS_DONE,
                                "session_complete": False,
                            })
                        continue
                    break
                try:
                    sentence = await asyncio.wait_for(tts_q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                audio_queue: asyncio.Queue = asyncio.Queue()
                # Likely #1 fix: signal the synth thread to stop mid-iteration
                # on barge-in (single-thread TTS executor would otherwise be
                # blocked by the orphaned generator until it completes).
                import threading as _threading
                stop_event = _threading.Event()
                state["current_tts_stop"] = stop_event
    
                def _run_synth(s, synth_be):
                    try:
                        stream_kwargs = {"language": tts_language_norm}
                        if tts_speaker_kwargs:
                            stream_kwargs.update(tts_speaker_kwargs)
                        elif tts_voice is not None:
                            stream_kwargs["voice"] = tts_voice  # deprecated
                        if tts_speed is not None:    stream_kwargs["speed"] = tts_speed
                        # FIX_C: use the manager-acquired backend so a reload
                        # waiting for drain sees this request as inflight.
                        for chunk in synth_be.generate_streaming(s, **stream_kwargs):
                            if stop_event.is_set():
                                break
                            loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)
                    except Exception as e:
                        # TTS slot-pool saturation is "backend busy", NOT a
                        # synth failure — log at warning (no stacktrace) and tag
                        # the queue item so the drain loop emits a clean 4429
                        # signal rather than a destructive error. The
                        # PoolSaturatedError class is off the WorkerProtocolError
                        # lineage, so no worker restart is triggered here.
                        _sat, _ms = _is_pool_saturated(e)
                        if _sat:
                            logger.warning(
                                "v2v tts slot-pool saturated for sentence=%r "
                                "(max_slots=%s)", s[:80], _ms,
                            )
                            loop.call_soon_threadsafe(
                                audio_queue.put_nowait,
                                ("__saturated__", _ms),
                            )
                        else:
                            logger.exception("v2v tts synthesis failed for sentence=%r", s)
                            loop.call_soon_threadsafe(audio_queue.put_nowait, ("__error__", str(e)))
                    finally:
                        loop.call_soon_threadsafe(audio_queue.put_nowait, None)
    
                async def drain():
                    nonlocal sr_header_sent
                    # PR5 / FIX_C: take BackendManager.acquire() *per utterance*
                    # so admin reload's drain logic sees this synth as inflight.
                    # Per-utterance (vs per-session) is intentional: v2v sessions
                    # can run for minutes; holding acquire across the whole
                    # session would block every reload until the user hangs up.
                    # _v2v_tts_mgr is captured from the enclosing scope (set
                    # earlier from _try_tts_manager()); fall back to the
                    # already-bound tts_be when manager wiring is absent (partial
                    # config / tests).
                    tts_mgr_local = _v2v_tts_mgr
                    if tts_mgr_local is not None:
                        acquire_cm = tts_mgr_local.acquire()
                        synth_backend = await acquire_cm.__aenter__()
                    else:
                        acquire_cm = None
                        synth_backend = tts_be
                    try:
                        # Coord lock per-sentence: cheap on concurrent profiles;
                        # serializes sentences against ASR on serialized profiles.
                        async with coord.acquire("tts"):
                            if not sr_header_sent:
                                sr = tts_service.get_sample_rate() if hasattr(tts_service, "get_sample_rate") else 16000
                                await send_bytes(struct.pack("<I", sr))
                                sr_header_sent = True
                            await send_json({"type": v2v_proto.SERVER_TTS_STARTED, "sentence": sentence})
                            loop.run_in_executor(_get_tts_stream_executor(), _run_synth, sentence, synth_backend)
                            state["tts_started"] = True
                            # Watchdog: if the synth thread doesn't produce
                            # a chunk within this many seconds, treat the
                            # sentence as failed and continue. Without this
                            # a wedged TTS backend (model load issue, GPU
                            # OOM, etc.) leaves the client (and any
                            # downstream agent) waiting forever on a
                            # promise the server can never fulfil.
                            tts_chunk_timeout_s = float(
                                os.getenv("OVS_TTS_CHUNK_TIMEOUT_S", "10.0")
                            )
                            while True:
                                try:
                                    item = await asyncio.wait_for(
                                        audio_queue.get(),
                                        timeout=tts_chunk_timeout_s,
                                    )
                                except asyncio.TimeoutError:
                                    logger.warning(
                                        "v2v tts watchdog: no chunk within %.1fs for "
                                        "sentence=%r — aborting synth and emitting error",
                                        tts_chunk_timeout_s, sentence[:80],
                                    )
                                    stop_event.set()
                                    await send_error(
                                        f"tts: synth produced no chunks within "
                                        f"{tts_chunk_timeout_s:.0f}s"
                                    )
                                    break
                                if item is None:
                                    break
                                if isinstance(item, tuple) and item[0] == "__saturated__":
                                    # Backend busy (slot-pool saturated). Surface
                                    # a typed reject-not-queue signal; do NOT
                                    # tear down the worker. The session stays
                                    # alive so the client can retry.
                                    try:
                                        await send_json({
                                            "type": v2v_proto.SERVER_ERROR,
                                            "error": "pool_saturated",
                                            "status": 4429,
                                            "max_slots": item[1],
                                        })
                                    except Exception:
                                        pass
                                    break
                                if isinstance(item, tuple) and item[0] == "__error__":
                                    await send_error(f"tts: {item[1]}")
                                    break
                                await send_bytes(item)
                            await send_json({"type": v2v_proto.SERVER_TTS_SENTENCE_DONE, "sentence": sentence})
                    finally:
                        if acquire_cm is not None:
                            try:
                                await acquire_cm.__aexit__(None, None, None)
                            except Exception:
                                logger.exception("v2v tts acquire exit failed")
    
                task = asyncio.create_task(drain())
                state["current_tts_task"] = task
                # Outer per-sentence deadline. Codex review 2026-05-26
                # caught the gap: the inner ``audio_queue.get()`` watchdog
                # only fires AFTER ``tts_started`` is emitted (drain() has
                # already passed backend acquire + coord acquire +
                # send_json). If the wedge is in any of those earlier
                # steps — backend-manager acquire blocked by a stuck
                # reload, ``coord.acquire`` deadlock, or Matcha's
                # pre-yield ORT/TRT setup — the client never sees
                # ``tts_started`` and waits forever. Wrap the whole drain
                # in a deadline that covers ALL of the above; tuning via
                # env so we can dial up on slow Jetsons.
                tts_sentence_timeout_s = float(
                    os.getenv("OVS_TTS_SENTENCE_TIMEOUT_S", "15.0")
                )
                try:
                    await asyncio.wait_for(task, timeout=tts_sentence_timeout_s)
                except asyncio.TimeoutError:
                    logger.warning(
                        "v2v tts: per-sentence deadline %.1fs exceeded for "
                        "sentence=%r — cancelling drain and continuing",
                        tts_sentence_timeout_s, sentence[:80],
                    )
                    stop_event.set()
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    # Try to send an error event so the client side
                    # cancels its turn promptly (instead of waiting for
                    # its own thinking watchdog at 20s).
                    if not state["client_closed"]:
                        try:
                            await send_error(
                                f"tts: per-sentence deadline {tts_sentence_timeout_s:.0f}s "
                                f"exceeded"
                            )
                        except Exception:
                            logger.exception("send_error after tts deadline failed")
                except asyncio.CancelledError:
                    # Barge-in: tell the synth thread to break out of the
                    # generator loop, then drain any chunks it produced
                    # before noticing the flag.
                    stop_event.set()
                    try:
                        while True:
                            item = audio_queue.get_nowait()
                            if item is None: break
                    except asyncio.QueueEmpty:
                        pass
                finally:
                    state["current_tts_task"] = None
                    state["current_tts_stop"] = None
            if not state["client_closed"]:
                # Session-final tts_done. In multi-utterance mode tag it as
                # session_complete=True so the client can distinguish it from
                # the per-turn dones emitted above. Single-utterance mode
                # omits the field for backward compatibility.
                payload = {"type": v2v_proto.SERVER_TTS_DONE}
                if multi_utterance:
                    payload["session_complete"] = True
                await send_json(payload)
    
        # ── Stage 5: orchestrate ────────────────────────────────────────
        # Bug #3 fix: dispatcher loops on ws.receive() forever (only exits
        # on disconnect). If we asyncio.gather all three, the server hangs
        # after asr_final / tts_done. Spawn work tasks separately, wait for
        # them, then cancel the dispatcher.
        dispatcher_task = asyncio.create_task(dispatcher())
        work_tasks = []
        if asr_enabled:
            work_tasks.append(asyncio.create_task(asr_out_task()))
        if tts_be is not None:
            work_tasks.append(asyncio.create_task(tts_out_task()))
    
        # NIT 3 round 3: track whether the V2V loop exited via a server
        # error so the WebSocket close frame carries the standard 1011
        # "internal error" code rather than the default 1005/1000.
        _v2v_server_error = False
        # Override the close code: a slot-pool saturation that bubbles up
        # through the ASR/TTS work tasks should close 4429 (reject-not-queue),
        # not 1011 (server error). None = use the default close logic below.
        _v2v_close_code: int | None = None
        try:
            if work_tasks:
                try:
                    await asyncio.gather(*work_tasks, return_exceptions=False)
                except asyncio.CancelledError:
                    # Work tasks were cancelled by the dispatcher on client
                    # disconnect (fast slot release) — not a server error and
                    # not a cancellation of this handler. The finally below
                    # still releases the SessionLimiter slot. Re-raise only if
                    # THIS handler was genuinely cancelled (no client close).
                    if not state["client_closed"]:
                        raise
            else:
                # No work tasks (shouldn't happen — config rejected earlier),
                # just keep the dispatcher running until the client closes.
                await dispatcher_task
        except Exception as e:
            # Slot-pool saturation bubbling up from ASR finalize / TTS synth:
            # treat as "backend busy", NOT a server error. Emit a typed 4429
            # reject and close 4429 (reject-not-queue) — do NOT trigger the
            # 1011 path or a destructive worker restart.
            _sat, _ms = _is_pool_saturated(e)
            if _sat:
                _v2v_close_code = 4429
                logger.warning(
                    "v2v stream slot-pool saturated (max_slots=%s); rejecting "
                    "with 4429", _ms,
                )
                try:
                    from server.core import metrics as _m_v2v_sat
                    _m_v2v_sat.inc_sessions_rejected("ws")
                except Exception:
                    pass
                try:
                    await send_json({
                        "type": v2v_proto.SERVER_ERROR,
                        "error": "pool_saturated",
                        "status": 4429,
                        "max_slots": _ms,
                    })
                except Exception:
                    pass
            else:
                _v2v_server_error = True
                logger.error("v2v stream error: %s", e, exc_info=True)
                try:
                    await send_error(f"{type(e).__name__}: {e}")
                except Exception:
                    pass
        finally:
            if not dispatcher_task.done():
                dispatcher_task.cancel()
                try:
                    await dispatcher_task
                except (asyncio.CancelledError, Exception):
                    pass
            for t in work_tasks:
                if not t.done():
                    t.cancel()
            # Tell the synth thread to bail (if running) so the TTS executor
            # frees up for the next connection.
            stop = state["current_tts_stop"]
            if stop is not None:
                stop.set()
            # Race #6: await the cancelled work tasks before releasing the
            # ASR/TTS slot. Otherwise the next WS connection on this
            # SessionLimiter slot can grab the limited ASR executor while
            # the previous worker thread is still running, producing
            # spurious "ASR busy" or worker-protocol errors on the very
            # first turn of the new connection.
            if work_tasks:
                try:
                    await asyncio.gather(*work_tasks, return_exceptions=True)
                except Exception:
                    pass
            # #41 P1: release the SessionLimiter admission slot (and paired
            # WS gauge / manager registration) BEFORE the blocking cleanup
            # below (asr_manager.cancel + ws.close). Those steps can take a
            # while on abrupt disconnects; holding the admission slot across
            # them is what caused back-to-back runs to pile up and hit 4429.
            # All releases here are idempotent (SessionToken._released token,
            # _v2v_ws_metric_taken flag, manager unregister is best-effort),
            # so the final cleanup guard below remains safe to run again.
            if _v2v_asr_mgr is not None:
                try:
                    _v2v_asr_mgr.unregister_ws(_v2v_handle)
                except BaseException:
                    pass
            if _v2v_tts_mgr is not None:
                try:
                    _v2v_tts_mgr.unregister_ws(_v2v_handle)
                except BaseException:
                    pass
            if _v2v_session_token is not None:
                try:
                    _v2v_session_token.release()
                except BaseException:
                    pass
            if _v2v_ws_metric_taken:
                try:
                    from server.core import metrics as _m_v2v
                    _m_v2v.dec_active_ws_sessions()
                    _v2v_ws_metric_taken = False
                except Exception:
                    pass
            # Cancel any in-flight ASR utterance before closing the socket
            # so the worker doesn't leak the session.
            # #41 P2: bound the cancel with a 2s timeout (aligns with the
            # turn-timeout path above) so a wedged worker cannot stall this
            # teardown. Admission is already released above, so a slow cancel
            # here no longer blocks the next connection's slot.
            if asr_manager is not None and state.get("asr_active"):
                try:
                    await asyncio.wait_for(
                        asr_manager.cancel("ws_close"),
                        timeout=2.0,
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
            try:
                if _v2v_close_code is not None:
                    # Slot-pool saturation → reject-not-queue close.
                    await ws.close(
                        code=_v2v_close_code,
                        reason='{"error":"pool_saturated"}',
                    )
                elif _v2v_server_error:
                    await ws.close(code=1011)
                else:
                    await ws.close()
            except Exception:
                pass
            if _v2v_asr_mgr is not None:
                try:
                    _v2v_asr_mgr.unregister_ws(_v2v_handle)
                except BaseException:
                    pass
            if _v2v_tts_mgr is not None:
                try:
                    _v2v_tts_mgr.unregister_ws(_v2v_handle)
                except BaseException:
                    pass
            if _v2v_session_token is not None:
                try:
                    _v2v_session_token.release()
                except BaseException:
                    pass
            if _v2v_ws_metric_taken:
                try:
                    from server.core import metrics as _m_v2v
                    _m_v2v.dec_active_ws_sessions()
                    _v2v_ws_metric_taken = False
                except Exception:
                    pass
            try:
                reset_request_context(_v2v_ctx_tokens)
            except BaseException:
                pass
            logger.info("v2v stream closed")
    except BaseException:
        # MUST-FIX 1 round 2: covers CancelledError (BaseException) raised
        # mid-setup before the inner main try/finally is established. The
        # release helper is idempotent so this is safe even on the normal
        # path where the inner finally already released.
        # MUST-FIX 1 round 3: wrap each cleanup in best-effort try/except
        # so a failing helper cannot mask the original exception or
        # short-circuit subsequent cleanups.
        try:
            _v2v_release_early()
        except BaseException:
            pass
        try:
            reset_request_context(_v2v_ctx_tokens)
        except BaseException:
            pass
        raise


# ── Admin: TTS runtime overrides ────────────────────────────────────────────

class TTSRuntimePatch(BaseModel):
    speaker_id: Optional[int] = None
    speed: Optional[float] = None
    pitch_shift: Optional[float] = None


def _current_tts_model_id() -> Optional[str]:
    from server.core import tts_service
    if not tts_service.is_ready():
        return None
    try:
        return tts_service.get_backend().model_id
    except Exception:
        return None


def _effective_tts_values(model_id: Optional[str]) -> dict:
    from server.core import tts_runtime
    from server.core.tts_speakers import default_speaker_id
    snap = tts_runtime.get_overrides()
    if snap.default_speaker_id is not None:
        eff_speaker = snap.default_speaker_id
    elif model_id is not None:
        try:
            eff_speaker = default_speaker_id(model_id)
        except Exception:
            eff_speaker = None
    else:
        eff_speaker = None
    return {
        "speaker_id": eff_speaker,
        "speed": snap.default_speed,
        "pitch_shift": snap.default_pitch_shift,
    }


def _admin_dep():
    from server.core.admin_auth import require_admin
    return require_admin


@app.get("/admin/tts/runtime")
async def admin_tts_runtime_get(_: None = Depends(_admin_dep())):
    from server.core import tts_runtime
    snap = tts_runtime.get_overrides()
    model_id = _current_tts_model_id()
    return {
        "model_id": model_id,
        "overrides": {
            "speaker_id": snap.default_speaker_id,
            "speed": snap.default_speed,
            "pitch_shift": snap.default_pitch_shift,
            "updated_at": snap.updated_at,
        },
        "effective": _effective_tts_values(model_id),
    }


@app.patch("/admin/tts/runtime")
async def admin_tts_runtime_patch(
    req: TTSRuntimePatch,
    _: None = Depends(_admin_dep()),
):
    from server.core import tts_runtime
    fields = req.model_fields_set
    kwargs: dict = {}
    if "speaker_id" in fields:
        kwargs["speaker_id"] = req.speaker_id
    if "speed" in fields:
        kwargs["speed"] = req.speed
    if "pitch_shift" in fields:
        kwargs["pitch_shift"] = req.pitch_shift
    model_id = _current_tts_model_id()
    if model_id is not None:
        kwargs["model_id"] = model_id
    try:
        snap = tts_runtime.update_overrides(**kwargs)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    return {
        "model_id": model_id,
        "overrides": {
            "speaker_id": snap.default_speaker_id,
            "speed": snap.default_speed,
            "pitch_shift": snap.default_pitch_shift,
            "updated_at": snap.updated_at,
        },
        "effective": _effective_tts_values(model_id),
    }


@app.post("/admin/tts/speakers/reload")
async def admin_tts_speakers_reload(
    _: None = Depends(_admin_dep()),
):
    from server.core.tts_speakers import reload_speakers, available_speakers
    reload_speakers()
    model_id = _current_tts_model_id()
    count = 0
    if model_id is not None:
        try:
            count = len(available_speakers(model_id))
        except Exception:
            count = 0
    return {"reloaded": True, "model_id": model_id, "count": count}


# ── Admin: Backend hot-reload ───────────────────────────────────────────────

class BackendReloadRequest(BaseModel):
    kind: Literal["tts", "asr"]
    profile: str
    drain_timeout_s: Optional[float] = None


@app.post("/admin/backend/reload")
async def admin_backend_reload(
    payload: BackendReloadRequest,
    _: None = Depends(_admin_dep()),
):
    from server.core.backend_manager import tts_manager, asr_manager

    if payload.kind == "tts":
        mgr = tts_manager()
    else:  # "asr"  (Literal already constrains the values)
        mgr = asr_manager()
    # drain_timeout_s override is plumbed into the request schema for
    # forward compatibility; the manager does not yet expose a setter,
    # so we ignore it for now (TODO: surface a per-call drain timeout
    # on BackendManager.reload).
    return await mgr.reload(payload.profile, reason="admin")


@app.get("/admin/backend/status")
async def admin_backend_status(_: None = Depends(_admin_dep())):
    from server.core.backend_manager import tts_manager, asr_manager
    return {"tts": tts_manager().status(), "asr": asr_manager().status()}


@app.get("/admin/backend/loadable")
async def admin_backend_loadable(_: None = Depends(_admin_dep())):
    """Classify every server-side profile by whether *this* SLV can actually
    load it, split per kind (tts / asr).

    For each profile JSON under ``configs/profiles/`` we ask each manager to
    preview its own half (``_load_profile_kind``) and run the same artifact
    pre-flight the hot-reload path uses (``find_missing_artifacts``). A profile
    is *loadable* for a kind when no expected artifact path is missing.

    Because tts_manager loads the TTS side and asr_manager loads the ASR side,
    the same profile can be loadable for one kind and not the other (e.g. the
    ASR engine is absent) — which is exactly what the demo portal needs to
    only offer switch targets that will succeed.

    A single broken profile never fails the whole endpoint: load/parse errors
    are captured per-profile under ``invalid``.
    """
    from server.core.backend_manager import tts_manager, asr_manager
    from server.core import backend_manager as _bm_mod
    from server.core import profile_loader as _pl
    from pathlib import Path as _Path

    repo_root = _Path(_bm_mod.__file__).resolve().parents[2]
    profiles_dir = repo_root / "configs" / "profiles"
    names: list[str] = []
    if profiles_dir.is_dir():
        names = sorted(p.stem for p in profiles_dir.glob("*.json"))

    def _classify(mgr) -> dict:
        loadable: list[str] = []
        unloadable: list[dict] = []
        invalid: list[dict] = []
        for name in names:
            try:
                preview = mgr._load_profile_kind(name)
                missing = _pl.find_missing_artifacts(preview, kind=mgr.name)
            except Exception as exc:  # noqa: BLE001 — one bad profile ≠ dead endpoint
                invalid.append({"name": name, "error": str(exc)})
                continue
            if missing:
                unloadable.append({"name": name, "missing": missing})
            else:
                loadable.append(name)
        return {"loadable": loadable, "unloadable": unloadable, "invalid": invalid}

    return {"tts": _classify(tts_manager()), "asr": _classify(asr_manager())}


# Phase B: install the deliberately small OpenAI-compatible audio adapter
# after all native resolver/execution helpers have been defined.  The adapter
# imports ``server.main`` lazily from request handlers, so this registration
# does not create a module cycle and keeps the legacy routes untouched.
from server.api.openai_compat import register as _register_openai_compat  # noqa: E402

_register_openai_compat(app)
