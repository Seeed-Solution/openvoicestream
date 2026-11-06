"""Unified concurrency-capability resolver.

MIGRATION TWIN — this is the ACTIVE, profile/env-aware resolver used by the HTTP
server. Its env-free counterpart is ``voxedge.engine.capability_resolver`` (the
convergence target). Both follow docs/specs/concurrency-capability-framework.md.
Prefer extending the voxedge copy; see ARCHITECTURE.md → "Known structural debt".


Three sites previously re-implemented backend-capability resolution:

- ``server.core.session_limiter`` — ``min(asr.max_concurrent, tts.max_concurrent)``
  plus profile/env clamp + warn (spec §3 + §7).
- ``server.core.coordinator`` — downgrade ``concurrent`` → ``serialized`` when
  either backend declares ``supports_parallel=False`` or ``max_concurrent<=1``
  (spec §4). ``exclusive`` is always honored as-is.
- ``server.main._resolve_tts_stream_max_workers`` — executor max_workers aligned
  with the TTS backend capability so the executor and the WorkerIO semaphore
  share the same ceiling source (spec §5 end).

This module centralises the lookup so future field additions (e.g. VRAM
budget — spec §1 ``vram_mb_per_slot``) or profile-key renames only touch
one place. Behaviour is byte-equivalent to the previous three call sites;
each caller is migrated in a follow-up commit.

The resolver is intentionally side-effect-free: it returns warnings as a
list of strings, and lets each caller decide how/where to log them. That
lets ``coordinator`` keep its INFO-level "downgrading…" message and
``session_limiter`` keep its WARNING-level clamp message without forcing a
single log style on all callers.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

from server.core.concurrency_capability import ConcurrencyCapability


logger = logging.getLogger(__name__)


CoordinatorMode = Literal["concurrent", "serialized", "exclusive"]


# ---------------------------------------------------------------------------
# Legacy target → default ceiling table.
#
# Mirrors ``session_limiter._TARGET_DEFAULTS``. Kept in sync here so callers
# that hit the "no backend declared" fallback get identical numbers.
# ---------------------------------------------------------------------------
_TARGET_DEFAULTS: dict[str, int] = {
    "orin-nx": 2,
    "orin-nano": 1,
    "rk": 1,
    "desktop": 4,
}
_UNKNOWN_DEFAULT = 1


def _infer_target(profile: Mapping[str, object] | None) -> str:
    """Best-effort target classification — copied verbatim from limiter."""
    profile = profile or {}
    name = str(profile.get("name") or "").lower()
    env_block = profile.get("env") or {}
    if not isinstance(env_block, Mapping):
        env_block = {}

    if "orin-nx" in name or "orin_nx" in name:
        return "orin-nx"
    if "orin-nano" in name or "orin_nano" in name:
        return "orin-nano"
    if "rk" in name or "rockchip" in name or "recomputer" in name:
        return "rk"
    if "desktop" in name or "ci" in name:
        return "desktop"

    rk_platform = (
        str(env_block.get("RK_PLATFORM") or "")
        or os.environ.get("RK_PLATFORM", "")
    ).lower()
    if rk_platform:
        return "rk"
    lang_mode = (
        str(env_block.get("LANGUAGE_MODE") or "")
        or os.environ.get("LANGUAGE_MODE", "")
    ).lower()
    if lang_mode == "rk":
        return "rk"

    return "unknown"


# ---------------------------------------------------------------------------
# Backend-class resolution (lazy import; never instantiates)
# ---------------------------------------------------------------------------


def _resolve_backend_class(
    profile: Mapping[str, object] | None,
    key: str,
    registry: Mapping[str, tuple[str, str]],
):
    """Return the backend class for ``profile[key]`` via lazy import.

    Returns ``None`` when the profile is missing the key, the registry has
    no matching entry, or the lazy import fails.
    """
    if not isinstance(profile, Mapping):
        return None
    spec = profile.get(key)
    if not spec or spec not in registry:
        return None
    module_path, cls_name = registry[spec]
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)
    except Exception as exc:
        logger.debug(
            "capability_resolver: %s import failed for %r: %s", key, spec, exc
        )
        return None


def _capability_for(
    cls,
    profile: Mapping[str, object] | None,
    spec: Optional[str] = None,
    kind: Optional[str] = None,
) -> ConcurrencyCapability:
    """Resolve a backend's ConcurrencyCapability with graceful fallback.

    voxedge backends declare ``concurrency_capability`` as an instance method
    (env-free, reads injected config), so calling it as a classmethod raises.
    When ``spec``/``kind`` are given we build the config from profile and read
    the capability off a (cheap) instance; otherwise we fall back to the legacy
    classmethod call.
    """
    if cls is None:
        return ConcurrencyCapability.default()
    if spec and kind:
        try:
            from server.core.voxedge_backend_config import concurrency_capability_for_spec
            cap = concurrency_capability_for_spec(spec, cls, kind, profile)
            if cap is not None:
                return cap
        except Exception as exc:
            logger.debug(
                "capability_resolver: voxedge capability resolve failed for %s: %s",
                spec, exc,
            )
    try:
        return cls.concurrency_capability(profile)
    except Exception as exc:
        logger.debug(
            "capability_resolver: concurrency_capability() raised for %s: %s",
            getattr(cls, "__name__", cls), exc,
        )
        return ConcurrencyCapability.default()


def _aggregate_ceiling(
    asr_cap: ConcurrencyCapability,
    tts_cap: ConcurrencyCapability,
    extra: Optional[list[tuple[str, ConcurrencyCapability]]] = None,
) -> tuple[Optional[int], str]:
    """Compute ``min(max_concurrent)`` across asr, tts and any ``extra`` caps.

    ``None`` means "no fixed cap" (treated as +inf per spec §1) and is dropped
    from the ``min()``. Returns ``(ceiling, label)`` where ``label`` is the
    human-readable diagnostic used in logs (e.g. ``asr=4,tts=2,diar=3``).

    Back-compat: with ``extra`` omitted (or ``None``) the result — both the
    ceiling and the label string — is byte-identical to the original
    two-backend implementation.
    """
    items: list[tuple[str, Optional[int]]] = [
        ("asr", asr_cap.max_concurrent),
        ("tts", tts_cap.max_concurrent),
    ]
    if extra:
        items.extend((label, cap.max_concurrent) for label, cap in extra)

    label = ",".join(
        f"{name}={'inf' if n is None else n}" for name, n in items
    )
    finite = [n for _, n in items if n is not None]
    ceiling = min(finite) if finite else None
    return ceiling, label


def _parallel_ok(cap: ConcurrencyCapability) -> bool:
    """Mirror of coordinator._parallel_ok — spec §4 wording."""
    if not cap.supports_parallel:
        return False
    if cap.max_concurrent is None:
        return True
    return cap.max_concurrent > 1


# ---------------------------------------------------------------------------
# Resolved snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedCapability:
    """Snapshot of aggregated backend capability + caller-specific decisions.

    Fields:

    - ``session_ceiling``: post-clamp value for the session admission gate
      (spec §3). May be ``None`` if neither backend declares a cap and no
      explicit override or target-default applies — callers that need a
      finite N (limiter, executor) should fall back to ``UNKNOWN_DEFAULT``.
    - ``executor_max_workers``: finite N >=1 for ThreadPoolExecutor sizing
      (spec §5 alignment with WorkerIO semaphore).
    - ``coordinator_mode``: resolved execution policy (spec §4) honoring
      profile ``exclusive`` and downgrading ``concurrent`` when either
      backend cannot run in parallel.
    - ``ceiling_source``: short diagnostic string for log lines.
    - ``clamp_warnings``: list of formatted warning messages produced by
      the resolution. Callers log these at their preferred level.
    - ``asr_cap`` / ``tts_cap``: the raw capability descriptors (handy for
      diagnostics and for the coordinator INFO log).
    """

    session_ceiling: Optional[int]
    executor_max_workers: int
    coordinator_mode: CoordinatorMode
    ceiling_source: str
    asr_cap: ConcurrencyCapability
    tts_cap: ConcurrencyCapability
    clamp_warnings: list[str] = field(default_factory=list)
    diar_cap: Optional[ConcurrencyCapability] = None
    # -- ASR inference gate (session admission vs in-flight inference) ------
    # ``session_ceiling`` above is how many connections may be OPEN.
    # ``asr_infer_concurrency`` is how many of them may be INSIDE the ASR
    # runtime at the same time. They were the same number until the gate
    # existed; a backend declaring ``supports_parallel=False`` with
    # ``max_concurrent=N`` now means "admit N, run one at a time".
    # ``asr_queue_depth`` bounds the FIFO backlog; ``None`` = unbounded.
    asr_infer_concurrency: int = 1
    asr_queue_depth: Optional[int] = None


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def _parse_positive_int(value, *, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        n = value
    else:
        s = str(value).strip()
        if s == "":
            return None
        try:
            n = int(s)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    if n <= 0:
        raise ValueError(f"{label} must be > 0, got {n}")
    return n


def _parse_non_negative_int(value, *, label: str) -> Optional[int]:
    """Like ``_parse_positive_int`` but admits 0."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        n = value
    else:
        s = str(value).strip()
        if s == "":
            return None
        try:
            n = int(s)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    if n < 0:
        raise ValueError(f"{label} must be >= 0, got {n}")
    return n


def resolve(
    *,
    profile: Mapping[str, object] | None,
    policy: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
    tts_backend_name: str | None = None,
) -> ResolvedCapability:
    """Resolve aggregate capability + per-caller-derived values.

    Args:
        profile: profile dict (must contain ``asr_backend`` / ``tts_backend``
            keys for capability lookup; otherwise the legacy ``_TARGET_DEFAULTS``
            table is used).
        policy: optional ``execution_policy`` dict from the profile. When
            absent or missing ``mode``, the coordinator mode defaults to
            ``concurrent`` and downgrades per spec §4.
        env: optional mapping to override ``os.environ`` (for tests). Reads
            ``OVS_MAX_CONCURRENT_SESSIONS`` for the session ceiling and
            ``OVS_TTS_STREAM_MAX_WORKERS[_<BACKEND>]`` for the executor cap.
        tts_backend_name: optional currently-loaded TTS backend name
            (lowercase). Used to pick the backend-specific
            ``OVS_TTS_STREAM_MAX_WORKERS_<KOKORO|MATCHA|QWEN3|MOSS>`` env.

    Returns:
        ``ResolvedCapability`` — see field docs.
    """
    env_map: Mapping[str, str] = env if env is not None else os.environ
    profile = profile or {}
    if policy is None and isinstance(profile.get("execution_policy"), Mapping):
        policy = profile["execution_policy"]
    cross_modal_overlap = bool(
        isinstance(policy, Mapping)
        and policy.get("mode") == "concurrent"
        and policy.get("cross_modal_overlap") is True
    )

    # ---- Capability lookup ------------------------------------------------
    from server.core.asr_backend import _ASR_REGISTRY
    from server.core.tts_backend import _TTS_REGISTRY

    asr_cls = _resolve_backend_class(profile, "asr_backend", _ASR_REGISTRY)
    tts_cls = _resolve_backend_class(profile, "tts_backend", _TTS_REGISTRY)
    asr_cap = _capability_for(asr_cls, profile, spec=profile.get("asr_backend"), kind="asr")
    tts_cap = _capability_for(tts_cls, profile, spec=profile.get("tts_backend"), kind="tts")

    has_asr_backend = bool(profile.get("asr_backend"))
    has_tts_backend = bool(profile.get("tts_backend"))
    has_declared_backends = has_asr_backend or has_tts_backend

    # A profile may intentionally expose only one modality. An absent backend
    # consumes no session capacity and must therefore be the neutral +inf
    # element for both the aggregate ceiling and coordinator-mode check. Keep
    # the conservative N=1 fallback only for a backend that was declared but
    # failed capability resolution.
    inactive_cap = ConcurrencyCapability(
        supports_parallel=True,
        max_concurrent=None,
        is_stateful=False,
        requires_exclusive_device=False,
        scaling_mode="per_call_isolated",
    )
    if not has_asr_backend:
        asr_cap = inactive_cap
    if not has_tts_backend:
        tts_cap = inactive_cap

    # Diarization (opt-in, default-off) optionally tightens the ceiling. When
    # disabled — the common path — ``diar_cap`` is None and is NOT passed to
    # ``_aggregate_ceiling``, so the aggregate is byte-identical to a deploy
    # that never declared diarization. Never-raise: a failure → None → no-op.
    diar_cap: Optional[ConcurrencyCapability] = None
    try:
        from server.core.diarization import diarization_concurrency_capability
        diar_cap = diarization_concurrency_capability(profile, env_map)
    except Exception:
        diar_cap = None

    if cross_modal_overlap and has_asr_backend and has_tts_backend:
        # Cross-modal capacity is additive: one ASR-only session and one
        # TTS-only session consume different backend slots. The legacy min()
        # aggregate describes same-backend fan-out and incorrectly clamps a
        # validated ASR(N=1)+TTS(N=1) pair to one process-wide session.
        finite_modal = [
            cap.max_concurrent
            for cap in (asr_cap, tts_cap)
            if cap.max_concurrent is not None
        ]
        ceiling = (
            sum(finite_modal)
            if len(finite_modal) == 2
            else None
        )
        ceiling_source = (
            "cross_modal:"
            f"asr={'inf' if asr_cap.max_concurrent is None else asr_cap.max_concurrent}"
            "+"
            f"tts={'inf' if tts_cap.max_concurrent is None else tts_cap.max_concurrent}"
        )
        if diar_cap is not None and diar_cap.max_concurrent is not None:
            ceiling = (
                diar_cap.max_concurrent
                if ceiling is None
                else min(ceiling, diar_cap.max_concurrent)
            )
            ceiling_source += f",diar={diar_cap.max_concurrent}"
    elif has_declared_backends:
        extra = [("diar", diar_cap)] if diar_cap is not None else None
        ceiling, ceiling_source = _aggregate_ceiling(asr_cap, tts_cap, extra)
    else:
        target = _infer_target(profile)
        ceiling = _TARGET_DEFAULTS.get(target, _UNKNOWN_DEFAULT)
        ceiling_source = f"target_default={target}"

    # ---- Session ceiling (limiter §3 + §7) --------------------------------
    warnings: list[str] = []
    env_session = _parse_positive_int(
        env_map.get("OVS_MAX_CONCURRENT_SESSIONS"),
        label="OVS_MAX_CONCURRENT_SESSIONS",
    )
    profile_session = _parse_positive_int(
        profile.get("max_concurrent_sessions"),
        label="profile.max_concurrent_sessions",
    )

    if env_session is not None:
        if ceiling is not None and env_session > ceiling:
            warnings.append(
                f"OVS_MAX_CONCURRENT_SESSIONS={env_session} exceeds backend"
                f" ceiling ({ceiling_source}) → clamping to {ceiling}"
            )
            session_ceiling: Optional[int] = ceiling
        else:
            session_ceiling = env_session
    elif profile_session is not None:
        if ceiling is not None and profile_session > ceiling:
            warnings.append(
                f"profile.max_concurrent_sessions={profile_session} exceeds"
                f" backend ceiling ({ceiling_source}) → clamping to {ceiling}"
            )
            session_ceiling = ceiling
        else:
            session_ceiling = profile_session
    else:
        if ceiling is None:
            target = _infer_target(profile)
            session_ceiling = _TARGET_DEFAULTS.get(target, _UNKNOWN_DEFAULT)
        else:
            session_ceiling = ceiling

    # ---- Coordinator mode (spec §4) ---------------------------------------
    requested = "concurrent"
    if isinstance(policy, Mapping):
        requested = str(policy.get("mode", "concurrent"))

    if requested == "exclusive":
        coordinator_mode: CoordinatorMode = "exclusive"
    elif requested == "concurrent" and has_declared_backends and not (
        (_parallel_ok(asr_cap) and _parallel_ok(tts_cap))
        # An explicit, device-qualified cross-modal profile overrides the
        # conservative per-backend same-modality parallel flags. It does not
        # increase either backend's own slot count.
        or (
            cross_modal_overlap
            and has_asr_backend
            and has_tts_backend
        )
    ):
        coordinator_mode = "serialized"
    else:
        # Either the requested mode is already non-concurrent (serialized)
        # or both backends are parallel-capable. ``serialized`` and other
        # explicit values pass through unchanged.
        if requested == "concurrent":
            coordinator_mode = "concurrent"
        elif requested == "serialized":
            coordinator_mode = "serialized"
        else:
            # Unknown modes pass through as-is (typed as concurrent for
            # mypy — but the BackendCoordinator inspects the raw string).
            coordinator_mode = requested  # type: ignore[assignment]

    # ---- Executor max_workers (spec §5) -----------------------------------
    # When no TTS backend is declared in profile, treat the cap as unknown
    # (None) instead of the conservative default (max=1). Legacy
    # ``_resolve_tts_stream_max_workers`` only consulted capability when
    # profile.tts_backend resolved; otherwise it fell back to the legacy
    # default of 2 / env value un-clamped. Preserve that surface.
    tts_declared = isinstance(profile, Mapping) and profile.get("tts_backend") in (
        _TTS_REGISTRY if isinstance(profile, Mapping) else {}
    )
    exec_cap = tts_cap if tts_declared else ConcurrencyCapability(
        supports_parallel=False, max_concurrent=None,
    )
    executor_max_workers, exec_warning = _resolve_executor_max_workers(
        tts_cap=exec_cap,
        tts_backend_name=tts_backend_name or "",
        env_map=env_map,
    )
    if exec_warning:
        warnings.append(exec_warning)

    # ---- ASR inference gate sizing ---------------------------------------
    #
    # Backend-agnostic on purpose: nothing here names a backend, a platform or
    # an env beyond the two operator overrides. A backend opts in purely by
    # what it declares.
    #
    # ``max_concurrent`` keeps its existing meaning — the session admission
    # ceiling, which is what the limiter has always read it as.
    # ``supports_parallel`` answers the separate question of whether those
    # admitted sessions may be inside the runtime simultaneously. So:
    #
    #   supports_parallel=False, max_concurrent=N  → admit N, infer 1 at a
    #       time, queue the rest FIFO. (Shared single-runtime backends: one
    #       RKNN context, one sherpa recognizer, one TRT engine.)
    #   supports_parallel=True,  max_concurrent=N  → admit N, infer N.
    #
    # The gate never widens concurrency past what the backend declared, so a
    # backend that says nothing keeps 1:1 and never reaches the queue at all.
    if asr_cap.supports_parallel:
        # ``None`` here means "no fixed cap": inference is as wide as the
        # sessions we admit, so the gate has nothing to narrow.
        asr_infer_concurrency = (
            asr_cap.max_concurrent
            or session_ceiling
            or _UNKNOWN_DEFAULT
        )
    else:
        asr_infer_concurrency = 1

    env_infer = _parse_positive_int(
        env_map.get("OVS_ASR_INFER_CONCURRENCY"),
        label="OVS_ASR_INFER_CONCURRENCY",
    )
    if env_infer is not None:
        if env_infer > asr_infer_concurrency:
            warnings.append(
                f"OVS_ASR_INFER_CONCURRENCY={env_infer} exceeds what the ASR"
                f" backend declares (supports_parallel="
                f"{asr_cap.supports_parallel}, max_concurrent="
                f"{asr_cap.max_concurrent}) → clamping to"
                f" {asr_infer_concurrency}"
            )
        else:
            asr_infer_concurrency = env_infer

    # Default backlog: one queued utterance per admitted session beyond the
    # running ones. That is the deepest queue reachable when every client hits
    # an endpoint at the same instant, so it never rejects a legitimate burst,
    # and it is still bounded — unlike an unbounded await on a stalled
    # backend.
    if session_ceiling is None:
        asr_queue_depth: Optional[int] = None
    else:
        asr_queue_depth = max(0, session_ceiling - asr_infer_concurrency)
    # Non-negative, not positive: 0 is a meaningful setting here — "never
    # queue, reject the moment the runtime is busy".
    env_depth = _parse_non_negative_int(
        env_map.get("OVS_ASR_INFER_QUEUE_DEPTH"),
        label="OVS_ASR_INFER_QUEUE_DEPTH",
    )
    if env_depth is not None:
        asr_queue_depth = env_depth

    return ResolvedCapability(
        session_ceiling=session_ceiling,
        executor_max_workers=executor_max_workers,
        coordinator_mode=coordinator_mode,
        ceiling_source=ceiling_source,
        asr_cap=asr_cap,
        tts_cap=tts_cap,
        clamp_warnings=warnings,
        diar_cap=diar_cap,
        asr_infer_concurrency=asr_infer_concurrency,
        asr_queue_depth=asr_queue_depth,
    )


def _resolve_executor_max_workers(
    *,
    tts_cap: ConcurrencyCapability,
    tts_backend_name: str,
    env_map: Mapping[str, str],
) -> tuple[int, Optional[str]]:
    """Mirror of ``server.main._resolve_tts_stream_max_workers`` (spec §5).

    Precedence:
      1. backend-specific env var (KOKORO/MATCHA/QWEN3/MOSS) if matched,
         else global ``OVS_TTS_STREAM_MAX_WORKERS``.
      2. ``tts_cap.max_concurrent`` (None → fall through).
      3. legacy default ``2``.

    Env values exceeding the backend ceiling are clamped + warned.
    """
    name = (tts_backend_name or "").lower()
    env_used: Optional[str] = None
    env_str: Optional[str] = None
    for suffix, env_name in (
        ("kokoro", "OVS_TTS_STREAM_MAX_WORKERS_KOKORO"),
        ("matcha", "OVS_TTS_STREAM_MAX_WORKERS_MATCHA"),
        ("qwen3", "OVS_TTS_STREAM_MAX_WORKERS_QWEN3"),
        ("moss", "OVS_TTS_STREAM_MAX_WORKERS_MOSS"),
    ):
        if suffix in name and env_map.get(env_name):
            env_str = env_map[env_name]
            env_used = env_name
            break
    if env_str is None and env_map.get("OVS_TTS_STREAM_MAX_WORKERS"):
        env_str = env_map["OVS_TTS_STREAM_MAX_WORKERS"]
        env_used = "OVS_TTS_STREAM_MAX_WORKERS"

    cap_max = tts_cap.max_concurrent  # may be None

    if env_str is not None:
        try:
            n = int(env_str)
        except (TypeError, ValueError):
            n = 2
        warning: Optional[str] = None
        if cap_max is not None and n > cap_max:
            warning = (
                f"TTS executor: {env_used}={n} exceeds backend ceiling"
                f" {cap_max}; clamping"
            )
            n = cap_max
        return max(1, n), warning

    if cap_max is not None:
        return max(1, cap_max), None

    return 2, None


# ---------------------------------------------------------------------------
# Caller-specific projections (thin wrappers — keep migration diffs small)
# ---------------------------------------------------------------------------


def resolve_executor_for_tts(
    *,
    profile: Mapping[str, object] | None,
    tts_backend_name: str | None,
    env: Mapping[str, str] | None = None,
) -> tuple[int, Optional[str], str]:
    """Return ``(max_workers, backend_name_or_None, source_label)``.

    Matches the legacy return shape of
    ``server.main._resolve_tts_stream_max_workers`` so the call-site change
    is a one-liner.
    """
    env_map: Mapping[str, str] = env if env is not None else os.environ
    name = (tts_backend_name or "").lower()
    prof_map = profile or {}
    registry = _lazy_tts_registry()
    tts_declared = (
        isinstance(prof_map, Mapping) and prof_map.get("tts_backend") in registry
    )
    if tts_declared:
        tts_cls = _resolve_backend_class(prof_map, "tts_backend", registry)
        cap_for_exec = _capability_for(tts_cls, prof_map, spec=prof_map.get("tts_backend"), kind="tts")
    else:
        cap_for_exec = ConcurrencyCapability(
            supports_parallel=False, max_concurrent=None,
        )
    workers, _warn = _resolve_executor_max_workers(
        tts_cap=cap_for_exec,
        tts_backend_name=name,
        env_map=env_map,
    )

    # Determine the source label the legacy function emitted.
    env_used: Optional[str] = None
    for suffix, env_name in (
        ("kokoro", "OVS_TTS_STREAM_MAX_WORKERS_KOKORO"),
        ("matcha", "OVS_TTS_STREAM_MAX_WORKERS_MATCHA"),
        ("qwen3", "OVS_TTS_STREAM_MAX_WORKERS_QWEN3"),
        ("moss", "OVS_TTS_STREAM_MAX_WORKERS_MOSS"),
    ):
        if suffix in name and env_map.get(env_name):
            env_used = env_name
            break
    if env_used is None and env_map.get("OVS_TTS_STREAM_MAX_WORKERS"):
        env_used = "OVS_TTS_STREAM_MAX_WORKERS"

    if env_used is not None:
        source = env_used
    else:
        # Match legacy: "concurrency_capability" when cap was the source,
        # else "default". Only consult capability when profile declares it.
        if tts_declared and cap_for_exec.max_concurrent is not None:
            source = "concurrency_capability"
        else:
            source = "default"
    return workers, (name or None), source


def _lazy_tts_registry() -> Mapping[str, tuple[str, str]]:
    """Defer import so this module can be imported during early startup."""
    from server.core.tts_backend import _TTS_REGISTRY
    return _TTS_REGISTRY
