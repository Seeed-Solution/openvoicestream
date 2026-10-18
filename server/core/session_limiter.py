"""Process-wide concurrent voice-session limiter.

Reject-not-queue: when the configured limit is reached, additional voice
sessions are rejected immediately (HTTP 429 / WS 4429). This protects
GPU/NPU runtime stability, executor pressure, and latency on edge
devices.

See ``docs/archive/specs/prod-hardening-week1.md`` Deliverable 2.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from server.core import metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Limit derivation
# ---------------------------------------------------------------------------

# Target → default limit (spec §Deliverable 2 Defaults).
_TARGET_DEFAULTS: dict[str, int] = {
    "orin-nx": 2,
    "orin-nano": 1,
    "rk": 1,
    "desktop": 4,
}
_UNKNOWN_DEFAULT = 1


def _infer_target(profile: dict | None) -> str:
    """Best-effort target classification from profile + env.

    Returns one of the keys in ``_TARGET_DEFAULTS`` or ``"unknown"``.
    """
    profile = profile or {}
    name = str(profile.get("name") or "").lower()
    env_block = profile.get("env") or {}

    if "orin-nx" in name or "orin_nx" in name:
        return "orin-nx"
    if "orin-nano" in name or "orin_nano" in name:
        return "orin-nano"
    if "rk" in name or "rockchip" in name or "radxa" in name:
        return "rk"
    if "desktop" in name or "ci" in name:
        return "desktop"

    # Env hints
    rk_platform = (env_block.get("RK_PLATFORM") or os.environ.get("RK_PLATFORM") or "").lower()
    if rk_platform:
        return "rk"
    lang_mode = (env_block.get("LANGUAGE_MODE") or os.environ.get("LANGUAGE_MODE") or "").lower()
    if lang_mode == "rk":
        return "rk"

    return "unknown"


def _parse_int(value: str | int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s == "":
        return None
    try:
        return int(s)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc


def resolve_limit(profile: dict | None = None) -> int:
    """Resolve the effective session limit.

    Spec §3 + §7: ceiling comes from
    ``min(asr.max_concurrent, tts.max_concurrent)`` (``None`` treated as
    ``+inf``). Profile and env overrides MAY ONLY DOWNGRADE; any attempt
    to exceed the ceiling is warn-logged and silently clamped. If the
    backend ceiling cannot be determined (no profile, import failure),
    fall back to the legacy ``_TARGET_DEFAULTS`` table.

    Precedence (after clamping): env override > profile field > ceiling.
    Raises ``ValueError`` for non-int, zero, or negative values.

    Implementation: delegates to ``capability_resolver.resolve`` so all
    three downstream callers (limiter, coordinator, executor) share one
    capability snapshot. Pre-validate env/profile values before calling
    the resolver so the historical ``ValueError`` messages remain
    surfaced even when the resolver isn't asked about that field.
    """
    from server.core.capability_resolver import resolve as _resolve_cap

    profile = profile or {}

    # Preserve the original ``ValueError`` surface for sanity checks even
    # though the resolver re-validates internally. Tests rely on these
    # being raised by ``resolve_limit`` directly.
    _parse_int(os.environ.get("OVS_MAX_CONCURRENT_SESSIONS"),
               label="OVS_MAX_CONCURRENT_SESSIONS")
    _parse_int(profile.get("max_concurrent_sessions"),
               label="profile.max_concurrent_sessions")

    resolved = _resolve_cap(profile=profile)
    for w in resolved.clamp_warnings:
        if "OVS_MAX_CONCURRENT_SESSIONS" in w or "max_concurrent_sessions" in w:
            logger.warning("session_limiter: %s", w)
    ceiling = resolved.session_ceiling
    if ceiling is None:
        # Resolver returns None only when both backends declare max=None
        # AND no profile/env override applies. Fall back to legacy target
        # table (behaviour preserved for mac dev-shell etc.).
        target = _infer_target(profile)
        return _TARGET_DEFAULTS.get(target, _UNKNOWN_DEFAULT)
    return ceiling


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------

class SessionLimiter:
    """Non-blocking voice-session admission gate.

    Internally tracks an integer count under a lock; ``try_acquire`` is
    immediate (never blocks). Slot release is idempotent (guarded by a
    per-token flag) so generator ``finally`` blocks plus exception paths
    cannot drive ``active`` negative.
    """

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError(f"SessionLimiter limit must be > 0, got {limit}")
        self._limit = limit
        self._active = 0
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def available(self) -> int:
        with self._lock:
            return max(0, self._limit - self._active)

    def try_acquire(self) -> "SessionToken | None":
        """Return a token on success, ``None`` when full.

        Never blocks. Never raises.
        """
        with self._lock:
            if self._active >= self._limit:
                return None
            self._active += 1
            current = self._active
        # Metrics are best-effort: a metric failure must NOT leak a slot
        # nor desync limiter state. See codex MUST-FIX 2.
        try:
            metrics.inc_sessions_active()
        except Exception:
            logger.warning("session_limiter: inc_sessions_active() raised", exc_info=True)
        return SessionToken(self, current)

    def _release(self, token: "SessionToken") -> None:
        # Double-release guard: check + set under the lock to prevent a
        # concurrent double release racing past the early-return.
        with self._lock:
            if token._released:
                return
            token._released = True
            if self._active > 0:
                self._active -= 1
        try:
            metrics.dec_sessions_active()
        except Exception:
            logger.warning("session_limiter: dec_sessions_active() raised", exc_info=True)

    def snapshot(self) -> dict:
        with self._lock:
            return {"limit": self._limit, "active": self._active}


class SessionToken:
    """Opaque release handle. Use ``release()`` or ``async with``."""

    __slots__ = ("_limiter", "_acquired_at_count", "_released")

    def __init__(self, limiter: SessionLimiter, acquired_at_count: int) -> None:
        self._limiter = limiter
        self._acquired_at_count = acquired_at_count
        self._released = False

    def release(self) -> None:
        self._limiter._release(self)

    async def __aenter__(self) -> "SessionToken":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_limiter: SessionLimiter | None = None
_init_lock = threading.Lock()


def init_limiter(profile: dict | None = None) -> SessionLimiter:
    """Initialize the global limiter from profile/env. Fails fast on bad config."""
    global _limiter
    limit = resolve_limit(profile)
    with _init_lock:
        _limiter = SessionLimiter(limit)
    # #41 P4: surface the resolved env override + effective ceiling so an
    # env/cap mismatch (e.g. OVS_MAX_CONCURRENT_SESSIONS=2 against a cap of 1)
    # is diagnosable from a single startup line. This is observability only —
    # the clamp logic in resolve_limit() is unchanged. A 4429 storm under
    # such a mismatch is backend-pool saturation, not a stuck slot.
    _env_override = os.environ.get("OVS_MAX_CONCURRENT_SESSIONS")
    _profile_override = (profile or {}).get("max_concurrent_sessions")
    logger.info(
        "SessionLimiter initialized: effective_limit=%d "
        "(env OVS_MAX_CONCURRENT_SESSIONS=%r, profile.max_concurrent_sessions=%r)",
        limit, _env_override, _profile_override,
    )
    return _limiter


def get_limiter() -> SessionLimiter | None:
    """Return the global limiter, or ``None`` if not yet initialized.

    Read-only callers (e.g. ``/readyz``) should tolerate ``None`` during
    startup; admission paths should treat missing limiter as a startup
    bug.
    """
    return _limiter


def _reset_for_tests() -> None:
    global _limiter
    _limiter = None


# ---------------------------------------------------------------------------
# Admission helpers used by server.main
# ---------------------------------------------------------------------------

@asynccontextmanager
async def acquire_http(endpoint: str) -> AsyncIterator[SessionToken]:
    """Context manager wrapping ``try_acquire`` for HTTP handlers.

    On rejection raises ``HTTPException(429)``. The slot is released on
    every exit path (success, exception, generator close).
    """
    from fastapi import HTTPException
    limiter = _limiter
    if limiter is None:
        # Defensive: missing limiter after startup is a configuration bug.
        raise HTTPException(
            status_code=503,
            detail={"error": "session_limiter_unavailable"},
        )
    token = limiter.try_acquire()
    if token is None:
        snap = limiter.snapshot()
        try:
            metrics.inc_sessions_rejected("http")
        except Exception:
            logger.warning("session_limiter: inc_sessions_rejected(http) raised", exc_info=True)
        logger.warning(
            "session_limiter: HTTP 429 endpoint=%s active=%d limit=%d",
            endpoint, snap["active"], snap["limit"],
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "too_many_sessions",
                "current": snap["active"],
                "limit": snap["limit"],
            },
            headers={"Retry-After": "5"},
        )
    try:
        yield token
    finally:
        token.release()


def try_acquire_ws_token(endpoint: str) -> tuple["SessionToken | None", dict]:
    """Acquire a WS slot WITHOUT touching the socket.

    Returns ``(token, info)``. On success ``token`` is a SessionToken and
    ``info`` is ``{"reason": "ok"}``. On rejection ``token`` is ``None`` and
    ``info`` carries ``{"reason": "no_limiter"}`` or
    ``{"reason": "too_many", "snapshot": {...}}`` so the caller can decide
    whether to evict-and-retry before rendering the rejection close frame
    (see ``close_ws_rejected``). Never blocks, never closes the socket.

    This split exists so the /v2v admission path can attempt admission-time
    eviction of a stale holder: the legacy ``try_acquire_ws`` closed the
    *incoming* socket the instant the slot was full, leaving no room to
    reclaim the leaked slot first.
    """
    limiter = _limiter
    if limiter is None:
        return None, {"reason": "no_limiter"}
    token = limiter.try_acquire()
    if token is None:
        return None, {"reason": "too_many", "snapshot": limiter.snapshot()}
    return token, {"reason": "ok"}


async def close_ws_rejected(ws, endpoint: str, info: dict) -> None:
    """Close a WS that failed admission, mirroring ``try_acquire_ws`` codes.

    ``info`` is the dict returned by ``try_acquire_ws_token``. Emits the
    rejection metric + warning log for the ``too_many`` case so behaviour is
    identical to the pre-split single-shot path.
    """
    import json as _json
    from server.core import metrics as _metrics
    reason = info.get("reason")
    if reason == "no_limiter":
        try:
            await ws.close(code=1011, reason='{"error":"session_limiter_unavailable"}')
        except Exception:
            pass
        return
    snap = info.get("snapshot") or {"active": 0, "limit": 0}
    # V10 lifecycle instrumentation (observation only): correlate the WS
    # rejection with the admitted-connection stream via the safe ephemeral
    # id(ws). Wire JSON / schema / rejection code / limits / counts below
    # are UNCHANGED.
    logger.info(
        "session_limiter: WS 4429 reject conn=%s monotonic=%.6f endpoint=%s "
        "current=%d limit=%d",
        id(ws), time.monotonic(), endpoint, snap["active"], snap["limit"],
    )
    try:
        _metrics.inc_sessions_rejected("ws")
    except Exception:
        logger.warning("session_limiter: inc_sessions_rejected(ws) raised", exc_info=True)
    logger.warning(
        "session_limiter: WS 4429 endpoint=%s active=%d limit=%d",
        endpoint, snap["active"], snap["limit"],
    )
    payload = _json.dumps({
        "error": "too_many_sessions",
        "current": snap["active"],
        "limit": snap["limit"],
    })
    try:
        await ws.close(code=4429, reason=payload)
    except Exception:
        pass


async def try_acquire_ws(ws, endpoint: str) -> SessionToken | None:
    """Acquire a slot for a WS session. Caller must have already accepted.

    On rejection: closes the WS with 4429 (reason JSON) and returns
    ``None``. Caller MUST return without further work. Thin wrapper over
    ``try_acquire_ws_token`` + ``close_ws_rejected`` (unchanged behaviour for
    callers that don't want admission-time eviction, e.g. /asr/stream).
    """
    token, info = try_acquire_ws_token(endpoint)
    if token is None:
        await close_ws_rejected(ws, endpoint, info)
        return None
    return token
