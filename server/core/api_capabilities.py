"""Versioned, transport-neutral ASR/TTS capability descriptions.

The legacy capability endpoints intentionally keep their historical 503 and
flat response shapes.  This module is used by ``GET /v1/capabilities`` to
produce an additive schema that is safe to call during lazy startup, backend
failure, or ASR-only/TTS-only operation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from server.core.tts_speakers import (
    available_speakers,
    canonical_model_id,
    default_speaker_id,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
API_VERSIONS = ["legacy", "v1"]
ApiVersion = Literal["legacy", "v1", "openai-audio"]
VoiceKind = Literal["preset", "embedding", "intrinsic", "voice_profile"]
VoiceSource = Literal["model", "live", "preset", "registered_embedding", "backend_intrinsic", "none"]
ControlImplementation = Literal["backend", "dsp"]
CloneMode = Literal["embedding", "reference_audio", "voice_profile"]
EnrollmentMethod = Literal["reference_audio", "voice_profile"]

_ASR_MODEL_ALIASES = {
    "qwen3-asr-0.6b": "qwen3-asr",
    "qwen/qwen3-asr-0.6b": "qwen3-asr",
}


def canonical_asr_model_id(model_id: str) -> str:
    value = str(model_id).strip()
    return _ASR_MODEL_ALIASES.get(value.casefold(), value)


class _StrictModel(BaseModel):
    """Nested schema models reject accidental undocumented fields."""

    model_config = ConfigDict(extra="forbid")


class AudioCapabilities(_StrictModel):
    sample_rate: int | None
    channels: int | None
    sample_format: Literal["pcm_s16le"]
    response_formats: list[Literal["wav", "pcm"]]


class LanguageCapabilities(_StrictModel):
    mode: Literal["single_language", "multi_language", "unknown"]
    values: list[str] | None
    default: str | None


class VoiceItem(_StrictModel):
    id: int | str
    type: VoiceKind
    label: str
    payload: str | None = None
    source: VoiceSource | None = None
    model_id: str | None = None
    compatible_models: list[str] | None = None
    profile_type: str | None = None
    sample_rate: int | None = None
    source_meta: dict[str, Any] | None = None


class VoiceDefault(_StrictModel):
    speaker_id: int | str | None
    source: VoiceSource


class VoiceCapabilities(_StrictModel):
    default: VoiceDefault
    items: list[VoiceItem]
    aliases: list[str]


class ControlCapability(_StrictModel):
    supported: bool
    min: float | int | None = None
    max: float | int | None = None
    unit: str | None = None
    implementation: ControlImplementation | None


StyleDimension = Literal["gender", "pitch", "speed"]
StyleGender = Literal["male", "female"]
StyleLevel = Literal["very_low", "low", "moderate", "high", "very_high"]


class StyleValues(_StrictModel):
    gender: list[StyleGender]
    pitch: list[StyleLevel]
    speed: list[StyleLevel]


class StyleCapabilities(_StrictModel):
    supported: bool
    dimensions: list[StyleDimension]
    values: StyleValues


class ControlsCapabilities(_StrictModel):
    speed: ControlCapability
    pitch: ControlCapability
    style: StyleCapabilities | None = None


class CloneEnrollment(_StrictModel):
    supported: bool
    methods: list[EnrollmentMethod]


class CloningCapabilities(_StrictModel):
    supported: bool
    modes: list[CloneMode]
    enrollment: CloneEnrollment


class StreamingCapabilities(_StrictModel):
    supported: bool
    native_wire_format: str | None = None


class ConcurrencyCapabilities(_StrictModel):
    backend_max_concurrent: int | None
    admission_limit: int | None
    active: int | None
    available: int | None


class TTSCapabilities(_StrictModel):
    model_id: str | None
    ready: bool
    backend: str | None
    capabilities: list[str]
    audio: AudioCapabilities
    languages: LanguageCapabilities
    voices: VoiceCapabilities
    cloning: CloningCapabilities
    controls: ControlsCapabilities
    streaming: StreamingCapabilities
    concurrency: ConcurrencyCapabilities
    failure_class: str | None


class IFBHealthSnapshot(_StrictModel):
    """Source-backed ASR IFB health counters.

    ``None`` means unknown/unproven: a counter that is absent, non-integer, or
    negative is never fabricated as ``0``.  Field names and integer types are
    taken from the canonical actual snapshot
    (``phases.E_lifecycle.health_final`` in the v011 b2 results).
    """

    admitted_mid_flight: int | None = None
    resident: int | None = None
    queued: int | None = None
    failed: int | None = None
    cancelled: int | None = None
    completed: int | None = None
    stalls_founder_only: int | None = None
    stalls_guided: int | None = None
    stalls_incompatible: int | None = None
    stalls_no_capacity: int | None = None


class ASRRuntimeIFB(_StrictModel):
    """Actual native-worker IFB ready provenance for a configured ASR.

    Additive, ASR-only, and strictly typed.  Emitted only when the backend
    exposes a usable ``runtime_diagnostics`` snapshot; absent (not null) for
    every other backend.  Deliberately excludes worker PID, raw traces, and
    external session/work identifiers.
    """

    source: Literal["native_worker_ready"]
    validation_required: bool
    reported_ifb: bool | None = None
    reported_max_slots: int | None = None
    reported_gpu_slots: int | None = None
    reported_engine_max_batch_size: int | None = None
    contract_verified: bool
    latest: IFBHealthSnapshot | None = None
    observed_health: bool


class ASRCapabilities(_StrictModel):
    model_id: str | None
    ready: bool
    backend: str | None
    capabilities: list[str]
    audio: AudioCapabilities
    languages: LanguageCapabilities
    streaming: StreamingCapabilities
    concurrency: ConcurrencyCapabilities
    failure_class: str | None
    # Additive optional provenance: absent (not null) unless the backend
    # supplies runtime IFB diagnostics.
    runtime_ifb: ASRRuntimeIFB | None = None


class RuntimeProfileCapabilities(_StrictModel):
    """Non-secret startup contract status for hardware-specific profiles."""

    required: bool
    profile: str | None
    device: str | None
    contract: str
    verified: bool
    settings: dict[str, str]
    missing_profile: list[str]
    missing_runtime: list[str]
    mismatches: dict[str, dict[str, str]]


class EmptyComponent(_StrictModel):
    """The only valid shape for an unconfigured component: an empty object."""

    pass


class CapabilitiesResponse(_StrictModel):
    """Pydantic contract for the versioned capabilities document."""

    object: Literal["capabilities"]
    schema_version: Literal["1.0"]
    api_versions: list[ApiVersion]
    # An unconfigured component is represented by {}, while a configured
    # component is always validated against its complete nested contract.
    # Do not use a broad dict branch here: it would let malformed configured
    # components silently bypass all nested validation.
    tts: TTSCapabilities | EmptyComponent
    asr: ASRCapabilities | EmptyComponent
    runtime_profile: RuntimeProfileCapabilities | None = None


# Public aliases used by schema fixtures/integrators.
TTSComponent = TTSCapabilities
ASRComponent = ASRCapabilities
IFBRuntimeComponent = ASRRuntimeIFB


def _safe_ready(backend: object | None, explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    if backend is None:
        return False
    try:
        return bool(backend.is_ready())  # type: ignore[attr-defined]
    except Exception:
        return False


def _safe_caps(backend: object | None) -> set[str]:
    if backend is None:
        return set()
    try:
        raw = getattr(backend, "capabilities", ())
        return {str(getattr(cap, "value", cap)) for cap in raw}
    except Exception:
        return set()


def _safe_name(backend: object | None) -> str | None:
    if backend is None:
        return None
    try:
        value = getattr(backend, "name", None)
        return str(value) if value is not None else None
    except Exception:
        return None


def _safe_sample_rate(backend: object | None) -> int | None:
    if backend is None:
        return None
    try:
        value = getattr(backend, "sample_rate", None)
        return int(value) if value is not None else None
    except Exception:
        return None


def _safe_channels(backend: object | None) -> int:
    if backend is not None:
        for attr in ("channels", "_channels"):
            try:
                value = getattr(backend, attr)
                return int(value)
            except (AttributeError, TypeError, ValueError):
                pass
    return 1


def _safe_model_id(backend: object | None, profile: Mapping[str, Any] | None, kind: str) -> str | None:
    if backend is not None:
        try:
            value = getattr(backend, "model_id", None)
            if value:
                return (
                    canonical_model_id(str(value))
                    if kind == "tts"
                    else canonical_asr_model_id(str(value))
                )
        except Exception:
            pass
    if profile:
        key = "tts_model_id" if kind == "tts" else "asr_model_id"
        value = profile.get(key)
        if value:
            return (
                canonical_model_id(str(value))
                if kind == "tts"
                else canonical_asr_model_id(str(value))
            )
    env_key = "OVS_TTS_MODEL_ID" if kind == "tts" else "OVS_ASR_MODEL_ID"
    value = os.environ.get(env_key)
    if value:
        return canonical_model_id(value) if kind == "tts" else canonical_asr_model_id(value)
    return None


def _concurrency_capability(backend: object | None, profile: Mapping[str, Any] | None = None) -> object | None:
    if backend is None:
        return None
    method = getattr(backend, "concurrency_capability", None)
    if not callable(method):
        return None
    try:
        return method()
    except TypeError:
        try:
            return method(profile)
        except Exception:
            return None
    except Exception:
        return None


def _backend_ceiling(backend: object | None, profile: Mapping[str, Any] | None = None) -> int | None:
    cap = _concurrency_capability(backend, profile)
    if cap is None:
        # A fake/minimal backend with no declaration is conservatively one
        # slot, matching ConcurrencyCapability.default().
        return 1 if backend is not None else None
    try:
        value = getattr(cap, "max_concurrent", None)
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return 1


def _limiter_snapshot(limiter: object | None) -> dict[str, int | None]:
    if limiter is None:
        return {"admission_limit": None, "active": None, "available": None}
    try:
        limit = int(getattr(limiter, "limit"))
    except (AttributeError, TypeError, ValueError):
        limit = None
    try:
        active = int(getattr(limiter, "active"))
    except (AttributeError, TypeError, ValueError):
        active = None
    try:
        available = int(getattr(limiter, "available"))
    except (AttributeError, TypeError, ValueError):
        available = None if limit is None or active is None else max(0, limit - active)
    return {"admission_limit": limit, "active": active, "available": available}


def _audio_contract(backend: object | None, *, sample_rate: int | None = None) -> dict[str, Any]:
    return {
        "sample_rate": _safe_sample_rate(backend) if sample_rate is None else sample_rate,
        "channels": _safe_channels(backend),
        "sample_format": "pcm_s16le",
        "response_formats": ["wav", "pcm"],
    }


def _languages_contract(model_id: str | None, caps: set[str]) -> dict[str, Any]:
    if "multi_language" in caps or (model_id or "").startswith(("qwen3-", "moss-", "spark")):
        mode = "multi_language"
    else:
        mode = "single_language"
    # Do not invent a language list: backends do not expose a stable one yet.
    return {"mode": mode, "values": None, "default": "auto"}


def _manager_failure_class(state: str | None) -> str | None:
    """Map a BackendManager state to a stable, non-secret failure class."""
    if state is None:
        return None
    normalized = str(state).strip().lower()
    if normalized in {"init", "failed", "draining", "reloading"}:
        return f"backend_manager_{normalized}"
    return "backend_manager_unavailable"


def _voice_contract(
    model_id: str | None,
    backend: object | None,
    caps: set[str],
    runtime_speaker_id: int | None = None,
) -> dict[str, Any]:
    mid = canonical_model_id(model_id or "")
    try:
        items = available_speakers(mid) if mid else []
    except Exception:
        items = []
    default_id = runtime_speaker_id if runtime_speaker_id is not None else (default_speaker_id(mid) if mid else None)
    if default_id is not None and mid:
        # Runtime overrides are stored as canonical ids when an active model
        # is known, but normalize legacy aliases defensively for direct callers.
        try:
            from server.core.tts_speakers import speaker_spec_for_id
            resolved = speaker_spec_for_id(int(default_id), mid)
            if resolved is not None:
                default_id = resolved.id
        except (TypeError, ValueError):
            pass
    normalized_items: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    for item in items:
        normalized = {
            "id": item.get("id"),
            "type": item.get("type"),
            "label": item.get("label", ""),
            "source": "model",
            "model_id": mid,
            "compatible_models": [mid] if mid else [],
        }
        # Keep legacy/backend payloads discoverable without making them
        # required by the structured schema.
        if "payload" in item:
            normalized["payload"] = item["payload"]
        normalized_items.append(normalized)
        try:
            by_id[int(item["id"])] = normalized
        except (KeyError, TypeError, ValueError):
            pass
    # Spark's profile registry is a live data source in addition to the model
    # preset table.  Merge it into the same voice item contract and mark the
    # active model compatibility explicitly; this keeps stale/different-model
    # profiles from being mistaken for built-ins.
    try:
        from server.core import sparktts_voices
        try:
            # New workers filter by the active canonical model and preserve
            # profile metadata.  Keep a compatibility fallback for older
            # wheels whose list_voices() has no keyword parameters.
            live_items = sparktts_voices.list_voices(
                model_id=mid,
                compatible_model=mid,
            )
        except TypeError:
            live_items = sparktts_voices.list_voices()
    except Exception:
        live_items = []

    for profile in live_items or []:
        if not isinstance(profile, Mapping):
            continue
        voice_id = profile.get("voice_id")
        if not isinstance(voice_id, str) or not voice_id:
            continue
        profile_model_raw = profile.get("model_id")
        profile_model = (
            canonical_model_id(str(profile_model_raw))
            if profile_model_raw
            else None
        )
        compatible_raw = profile.get("compatible_models")
        compatible = (
            [canonical_model_id(str(v)) for v in compatible_raw]
            if isinstance(compatible_raw, (list, tuple, set))
            else []
        )
        profile_type = str(
            profile.get("profile_type") or profile.get("type") or ""
        ).strip()
        # Registered Base embeddings are model-scoped.  A profile without an
        # explicit canonical model is not advertised for Base, because it
        # could otherwise route a vector into the wrong backend.  Spark
        # profiles may come from legacy registries that omit model metadata;
        # the keyword-filtering worker call above is authoritative for those.
        if "qwen3-tts-0.6b-base" == mid:
            if profile_type != "speaker_embedding" or profile_model != mid:
                continue
            if compatible and mid not in compatible:
                continue
            live_item: dict[str, Any] = {
                "id": voice_id,
                "type": "embedding",
                "label": str(profile.get("label") or voice_id),
                "source": "registered_embedding",
                "model_id": mid,
                "compatible_models": [mid],
                "profile_type": "speaker_embedding",
                "sample_rate": profile.get("sample_rate"),
                "source_meta": profile.get("source_meta"),
            }
            normalized_items.append(live_item)
            continue

        if "spark" not in mid:
            continue
        if profile_type != "voice_profile":
            continue
        if profile_model and profile_model != mid:
            continue
        if compatible and mid not in compatible:
            continue
        live_item = {
            "id": voice_id,
            "type": "voice_profile",
            "label": str(profile.get("label") or voice_id),
            "source": "live",
            "model_id": profile_model or mid,
            "compatible_models": compatible or [mid],
            "profile_type": profile_type or None,
            "sample_rate": profile.get("sample_rate"),
            "source_meta": profile.get("source_meta"),
        }
        normalized_items.append(live_item)
    selected = by_id.get(int(default_id)) if default_id is not None else None
    if default_id is None:
        source = "none"
    elif selected and selected.get("type") == "intrinsic":
        source = "backend_intrinsic"
    elif selected and selected.get("type") == "embedding":
        source = "registered_embedding"
    else:
        source = "preset"

    if "spark" in mid:
        modes = ["voice_profile"] if "voice_clone" in caps else []
    elif "moss" in mid:
        # MOSS's clone contract is reference-audio prompt conditioning, never
        # a reusable embedding selector.
        modes = ["reference_audio"] if "voice_clone" in caps else []
    elif "qwen3-tts-0.6b-base" == mid:
        modes = ["embedding"] if "voice_clone" in caps else []
        if (
            "voice_clone" in caps
            and backend is not None
            and bool(getattr(backend, "supports_reference_audio_cloning", False))
        ):
            modes.append("reference_audio")
    else:
        modes = ["embedding"] if "voice_clone" in caps else []
    enrollment = bool(getattr(backend, "supports_voice_enrollment", False)) if backend else False
    return {
        "voices": {
            "default": {"speaker_id": default_id, "source": source},
            "items": normalized_items,
            "aliases": (["0"] if mid == "qwen3-tts-customvoice" else []),
        },
        "cloning": {
            "supported": bool(modes),
            "modes": modes,
            "enrollment": {"supported": enrollment, "methods": ["reference_audio"] if enrollment else []},
        },
    }


def _controls_contract(backend: object | None, model_id: str | None) -> dict[str, Any]:
    # ``rate_pitch_caps`` reports whether the backend has native controls; a
    # false value does not mean the public control is unavailable.  The
    # service applies its DSP fallback in that case (notably Base/MOSS/Spark),
    # so both controls remain advertised as supported with an explicit
    # implementation label.
    speed_native = pitch_native = False
    method = getattr(backend, "rate_pitch_caps", None) if backend else None
    has_native_contract = callable(method)
    if has_native_contract:
        try:
            speed_native, pitch_native = (bool(v) for v in method())
        except Exception:
            pass
    # A backend that does not expose the rate/pitch capability hook has no
    # public implementation contract yet.  Do not advertise controls for it.
    supported = has_native_contract
    controls: dict[str, Any] = {
        "speed": {
            "supported": supported,
            "min": 0.25,
            "max": 4.0,
            "unit": None,
            "implementation": ("backend" if speed_native else "dsp") if supported else None,
        },
        "pitch": {
            "supported": supported,
            "min": -24,
            "max": 24,
            "unit": "semitone",
            "implementation": ("backend" if pitch_native else "dsp") if supported else None,
        },
    }
    if model_id and "spark" in model_id:
        controls["style"] = {
            "supported": True,
            "dimensions": ["gender", "pitch", "speed"],
            "values": {"gender": ["male", "female"], "pitch": ["very_low", "low", "moderate", "high", "very_high"], "speed": ["very_low", "low", "moderate", "high", "very_high"]},
        }
    return controls


def _tts_component(
    backend: object | None,
    *,
    ready: bool,
    limiter: object | None,
    profile: Mapping[str, Any] | None,
    runtime_speaker_id: int | None = None,
    manager_state: str | None = None,
) -> dict[str, Any]:
    model_id = _safe_model_id(backend, profile, "tts")
    caps = _safe_caps(backend)
    component: dict[str, Any] = {
        "model_id": model_id,
        "ready": ready,
        "backend": _safe_name(backend),
        "capabilities": sorted(caps),
        "audio": _audio_contract(backend),
        "languages": _languages_contract(model_id, caps),
        **_voice_contract(model_id, backend, caps, runtime_speaker_id),
        "controls": _controls_contract(backend, model_id),
        "streaming": {
            "supported": "streaming" in caps,
            "native_wire_format": "u32le_sample_rate+pcm_s16le",
        },
        "concurrency": {
            "backend_max_concurrent": _backend_ceiling(backend, profile),
            **_limiter_snapshot(limiter),
        },
        # Explicit nullable field keeps the response shape stable for both
        # ready and failed configured backends.
        "failure_class": None,
    }
    if not ready:
        component["failure_class"] = (
            _manager_failure_class(manager_state)
            or ("backend_not_ready" if backend is not None else "backend_unavailable")
        )
    return component


_IFB_HEALTH_FIELDS = (
    "admitted_mid_flight",
    "resident",
    "queued",
    "failed",
    "cancelled",
    "completed",
    "stalls_founder_only",
    "stalls_guided",
    "stalls_incompatible",
    "stalls_no_capacity",
)


def _strict_int(value: Any) -> int | None:
    """Return ``value`` only when it is an int (bool rejected) >= 0.

    Anything else (bool, float, string, negative, None) is unproven and maps
    to ``None`` — never fabricated as zero.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _ifb_health_snapshot(health: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _strict_int(health.get(name)) for name in _IFB_HEALTH_FIELDS}


def _runtime_ifb_contract(
    backend: object | None,
    *,
    configured_slots: int | None,
) -> dict[str, Any] | None:
    """Build the additive ``asr.runtime_ifb`` provenance, or ``None``.

    Returns ``None`` unless the backend exposes a callable
    ``runtime_diagnostics`` that yields a mapping with IFB opt-in context
    (``require_ifb``).  Wrong types or exceptions are swallowed into a safe
    concise warning; the rest of the capability document is preserved and no
    exception text or backend internals (PID, traces, ids) ever escape.
    """
    method = getattr(backend, "runtime_diagnostics", None)
    if not callable(method):
        return None
    try:
        diagnostics = method()
    except Exception:
        logger.warning("ASR runtime_diagnostics() raised; IFB provenance omitted")
        return None
    if not isinstance(diagnostics, Mapping) or "require_ifb" not in diagnostics:
        return None

    validation_required = diagnostics.get("require_ifb")
    if not isinstance(validation_required, bool):
        logger.warning("ASR runtime_diagnostics require_ifb malformed; IFB provenance unproven")
        validation_required = False

    validated = diagnostics.get("validated_ready")
    if not isinstance(validated, Mapping):
        validated = {}

    reported_ifb = validated.get("ifb")
    reported_ifb = reported_ifb if isinstance(reported_ifb, bool) else None
    reported_max_slots = _strict_int(validated.get("max_slots"))
    reported_gpu_slots = _strict_int(validated.get("gpu_slots"))
    reported_batch = _strict_int(validated.get("engine_max_batch_size"))

    # Liveness must be explicitly PROVEN by the backend.  ``worker_alive`` is
    # not yet returned by the live backend (root must add proven Popen.poll()
    # metadata); until then missing/anything-but-True leaves verification
    # false and readiness UNPROVEN.  Cached ``ready`` alone is never enough.
    worker_alive = diagnostics.get("worker_alive")

    contract_verified = bool(
        validation_required is True
        and diagnostics.get("ready") is True
        and worker_alive is True
        and reported_ifb is True
        and reported_max_slots is not None
        and reported_gpu_slots is not None
        and reported_max_slots == reported_gpu_slots
        and configured_slots is not None
        and reported_max_slots == configured_slots
        and reported_batch is not None
        and reported_batch >= reported_max_slots
    )

    raw_health = diagnostics.get("latest_asr_ifb_health")
    if isinstance(raw_health, Mapping):
        latest = _ifb_health_snapshot(raw_health)
        observed_health = True
    else:
        latest = None
        observed_health = False

    return {
        "source": "native_worker_ready",
        "validation_required": validation_required,
        "reported_ifb": reported_ifb,
        "reported_max_slots": reported_max_slots,
        "reported_gpu_slots": reported_gpu_slots,
        "reported_engine_max_batch_size": reported_batch,
        "contract_verified": contract_verified,
        "latest": latest,
        "observed_health": observed_health,
    }


def _asr_component(
    backend: object | None,
    *,
    ready: bool,
    limiter: object | None,
    profile: Mapping[str, Any] | None,
    manager_state: str | None = None,
) -> dict[str, Any]:
    model_id = _safe_model_id(backend, profile, "asr")
    caps = _safe_caps(backend)
    configured_slots = _backend_ceiling(backend, profile)
    component: dict[str, Any] = {
        "model_id": model_id,
        "ready": ready,
        "backend": _safe_name(backend),
        "capabilities": sorted(caps),
        "audio": _audio_contract(backend),
        "languages": _languages_contract(model_id, caps),
        "streaming": {"supported": "streaming" in caps},
        "concurrency": {
            "backend_max_concurrent": configured_slots,
            **_limiter_snapshot(limiter),
        },
        "failure_class": None,
    }
    runtime_ifb = _runtime_ifb_contract(backend, configured_slots=configured_slots)
    if runtime_ifb is not None:
        component["runtime_ifb"] = runtime_ifb
    if not ready:
        component["failure_class"] = (
            _manager_failure_class(manager_state)
            or ("backend_not_ready" if backend is not None else "backend_unavailable")
        )
    return component


def build_capabilities(
    *,
    tts_backend: object | None = None,
    asr_backend: object | None = None,
    tts_ready: bool | None = None,
    asr_ready: bool | None = None,
    tts_configured: bool | None = None,
    asr_configured: bool | None = None,
    limiter: object | None = None,
    profile: Mapping[str, Any] | None = None,
    runtime_speaker_id: int | None = None,
    tts_manager_state: str | None = None,
    asr_manager_state: str | None = None,
    api_versions: list[str] | tuple[str, ...] | None = None,
    runtime_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured capability document without mutating service state."""
    if profile is None:
        try:
            from server.core.profile_loader import current_profile
            profile = current_profile() or {}
        except Exception:
            profile = {}

    # Callers must provide live backend objects explicitly.  This keeps the
    # schema builder pure and prevents stale module singletons or lazy service
    # access from changing a capability response.
    if tts_ready is None:
        tts_ready = _safe_ready(tts_backend, None)
    if asr_ready is None:
        asr_ready = _safe_ready(asr_backend, None)
    if tts_configured is None:
        tts_configured = bool(profile.get("tts_backend"))
    if asr_configured is None:
        asr_configured = bool(profile.get("asr_backend"))

    if limiter is None:
        try:
            from server.core.session_limiter import get_limiter
            limiter = get_limiter()
        except Exception:
            limiter = None

    tts_configured = bool(tts_configured)
    asr_configured = bool(asr_configured)
    tts_ready = bool(tts_ready)
    asr_ready = bool(asr_ready)
    # An explicitly supplied backend is itself a configured component even if
    # a test/fake service does not expose is_configured().
    if tts_backend is not None:
        tts_configured = True
    if asr_backend is not None:
        asr_configured = True

    document: dict[str, Any] = {
        "object": "capabilities",
        "schema_version": SCHEMA_VERSION,
        "api_versions": list(API_VERSIONS),
        "tts": _tts_component(
            tts_backend,
            ready=tts_ready,
            limiter=limiter,
            profile=profile,
            runtime_speaker_id=runtime_speaker_id,
            manager_state=tts_manager_state,
        ) if tts_configured else {},
        "asr": _asr_component(
            asr_backend,
            ready=asr_ready,
            limiter=limiter,
            profile=profile,
            manager_state=asr_manager_state,
        ) if asr_configured else {},
    }
    if runtime_profile is None:
        try:
            from server.core.rk_profile_contract import runtime_status
            _runtime_profile = runtime_status(profile, os.environ)
        except Exception:
            _runtime_profile = None
    else:
        _runtime_profile = runtime_profile
    # Keep the additive field absent for non-RK profiles, preserving the
    # compact legacy capability document while exposing the full contract for
    # the profiles where env shadowing can change streaming semantics.
    if _runtime_profile and _runtime_profile.get("required"):
        document["runtime_profile"] = dict(_runtime_profile)
    # Validate shape now, but return a plain JSON-compatible dict so callers
    # can preserve FastAPI's existing response serialization behaviour.
    document["api_versions"] = list(API_VERSIONS if api_versions is None else api_versions)
    result = CapabilitiesResponse.model_validate(document).model_dump(mode="json")
    # The additive ASR provenance field must be ABSENT (not null) whenever the
    # backend does not supply runtime IFB diagnostics.  Stripping the default
    # null here keeps every other existing optional field's serialization
    # behaviour (e.g. ``runtime_profile``) untouched.
    asr = result.get("asr")
    if isinstance(asr, dict) and asr.get("runtime_ifb") is None:
        asr.pop("runtime_ifb", None)
    return result


# A short descriptive alias for tests and integrations that call the builder
# by its schema-oriented name.
build_capability_document = build_capabilities
