"""ASR backend abstraction with capability discovery.

Mirrors the TTS backend pattern (tts_backend.py).
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from server.core.concurrency_capability import ConcurrencyCapability

logger = logging.getLogger(__name__)


class ASRCapability(str, Enum):
    OFFLINE = "offline"
    STREAMING = "streaming"
    TIMESTAMPS = "timestamps"
    MULTI_LANGUAGE = "multi_language"
    LANGUAGE_ID = "language_id"


class TranscriptionResult:
    def __init__(self, text: str, language: Optional[str] = None, **meta):
        self.text = text
        self.language = language
        self.meta = meta


class ASRStream(ABC):
    """A streaming ASR session that accumulates audio and produces text."""

    prefer_backend_endpoint_vad: bool = False
    """Whether this stream owns endpoint VAD and should not be finalized by
    frontend VAD speech_end events."""

    allow_frontend_eou_finalize: bool = False
    """Whether frontend VAD speech_end may finalize this stream even when the
    backend owns endpoint VAD. Default False preserves backend-owned endpoint
    semantics for opt-in streams until a backend explicitly accepts hybrid
    frontend/backend endpointing."""

    frontend_eou_min_audio_s: float = 0.0
    """Minimum accepted audio duration before frontend VAD speech_end may
    finalize a backend-owned endpoint stream."""

    immediate_client_eos_cancel_safe: bool = False
    """Whether partial abort can run outside normal ASR serialization."""

    @abstractmethod
    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        """Feed audio samples (float32, [-1,1]) into the stream."""
        ...

    @abstractmethod
    def finalize(self) -> tuple[str, Optional[str]]:
        """Signal end-of-audio.

        Returns ``(final_text, detected_language)``. ``detected_language`` is
        the human-readable language name (e.g. ``"Chinese"``, ``"English"``)
        if the backend supports language ID and detected one, otherwise
        ``None``. Backends without language detection return ``(text, None)``.
        """
        ...

    def get_partial(self) -> tuple[str, bool]:
        """Return (partial_text, is_endpoint). Default: no partial results."""
        return "", False

    def prepare_finalize(self) -> None:
        """Pre-encode remaining audio buffer so finalize() only runs decoder.

        Optional optimization — finalize() works without calling this first.
        """
        pass

    def cancel_and_finalize(self) -> None:
        """Hard-cancel any in-flight partial decode and skip residual tail encode.

        Used by barge-in / client-initiated stop paths where waiting for the
        pending decode wastes hundreds of ms. Default: no-op (subclasses that
        run async final decodes — e.g. RK true-streaming — override).
        """
        pass

    def cancel(self) -> None:
        """Symmetric alias for cancel_and_finalize().

        Lets callers (e.g. ASRSessionManager) treat cancel as a first-class
        operation without forcing every backend to implement both methods.
        Default: delegate to cancel_and_finalize().
        """
        self.cancel_and_finalize()

    def close(self) -> None:
        """Release per-stream resources (TRT exec contexts, device buffers).

        Default: no-op. Backends whose stream owns per-instance GPU resources
        (e.g. paraformer_trt's _ParaformerCtxBundle) override this to drop
        them deterministically. Safe to call multiple times.
        """
        pass


class ASRBackend(ABC):

    # PR5 / FIX_A: see TTSBackend.supports_hot_reload. Default False; backends
    # whose unload() actually releases GPU/NPU resources should set True.
    supports_hot_reload: bool = False
    prefer_backend_endpoint_vad: bool = False
    """Whether streams from this backend should receive audio from the first
    frame even when frontend VAD is enabled, then let backend endpointing own
    finalization. Default False preserves legacy frontend-VAD semantics."""

    allow_frontend_eou_finalize: bool = False
    """Whether frontend VAD speech_end may finalize streams from this backend
    even when ``prefer_backend_endpoint_vad`` is true."""

    frontend_eou_min_audio_s: float = 0.0
    """Minimum accepted audio duration before frontend EOU can finalize."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def capabilities(self) -> set[ASRCapability]: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def preload(self) -> None: ...

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult: ...

    def create_stream(
        self, language: str = "auto", stream_options: Optional[dict] = None
    ) -> ASRStream:
        """Create a streaming ASR session. Requires STREAMING capability.

        ``stream_options`` is session-scoped and optional; backends that own
        their own endpoint VAD may consume ``vad_endpoint_silence_ms`` from it
        (see ``docs/CONFIGURATION.md`` "Streaming ASR endpointing"). Legacy
        backends ignore the argument.
        """
        raise NotImplementedError(f"{self.name} does not support streaming")

    def has_capability(self, cap: ASRCapability) -> bool:
        return cap in self.capabilities

    def unload(self) -> None:
        """Release GPU/NPU resources. Override in backends that hold shared
        hardware so the BackendCoordinator's 'exclusive' mode can hand the
        device to another backend. Default is a no-op — backends without
        an unload() stay resident, which is fine for 'concurrent' and
        'serialized' modes."""
        pass

    @classmethod
    def concurrency_capability(
        cls, profile: Optional[dict] = None
    ) -> ConcurrencyCapability:
        """Describe runtime concurrency properties.

        Classmethod (not instance property) so the scheduler can read the
        ceiling before ``preload()``. Default is conservative (N=1,
        serialized) — backends opt in by overriding. See
        ``docs/specs/concurrency-capability-framework.md`` Section 2.
        """
        return ConcurrencyCapability.default()


_ASR_REGISTRY: Dict[str, Tuple[str, str]] = {
    "jetson.trt_edge_llm":   ("voxedge.backends.jetson.trt_edge_llm_asr", "TRTEdgeLLMASRBackend"),
    "jetson.paraformer_trt": ("voxedge.backends.jetson.paraformer_trt", "ParaformerTRTBackend"),
    "jetson.sensevoice_trt": ("voxedge.backends.jetson.sensevoice_trt", "SenseVoiceTRTBackend"),
    "cpu.sherpa_asr":        ("voxedge.backends.sherpa.asr",          "SherpaASRBackend"),
    "rk.asr":                ("voxedge.backends.rk.asr",              "RKASRBackend"),
    # One class, three encoder execution paths. The spec picks the path; the
    # config builder supplies the window and the boundary guard that path's
    # compiled graph expects.
    "hailo.whisper":         ("voxedge.backends.whisper",             "WhisperASR"),
    "rk.whisper":            ("voxedge.backends.whisper",             "WhisperASR"),
    "jetson.whisper_trt":    ("voxedge.backends.whisper",             "WhisperASR"),
}


def _lazy_import(module_path: str, attr: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def create_asr_backend() -> ASRBackend:
    """Factory: instantiate the ASR backend declared by the loaded profile.

    Reads ``asr_backend`` from server.core.profile_loader.current_profile().
    The value must be a registry key (e.g. ``jetson.trt_edge_llm``). Raises
    ValueError if no profile is loaded, the key is missing, or unknown.
    """
    from server.core.profile_loader import current_profile
    spec = current_profile().get("asr_backend")
    if not spec:
        raise ValueError("Profile must declare 'asr_backend'")
    if spec not in _ASR_REGISTRY:
        raise ValueError(f"Unknown asr_backend: {spec!r}")
    module_path, cls_name = _ASR_REGISTRY[spec]
    logger.info("Creating ASR backend %s (%s.%s)", spec, module_path, cls_name)
    cls = _lazy_import(module_path, cls_name)
    # voxedge backends are env-free: build their config from env/profile in the
    # product layer (preserves voxedge's zero-env property). Other specs keep
    # their legacy os.environ-reading __init__.
    if spec == "jetson.trt_edge_llm":
        from server.core.voxedge_backend_config import build_trt_edge_llm_asr_config
        config = build_trt_edge_llm_asr_config(profile=current_profile())
        return cls(config=config)
    if spec == "jetson.paraformer_trt":
        from server.core.voxedge_backend_config import build_paraformer_trt_config
        config = build_paraformer_trt_config(profile=current_profile())
        return cls(config=config)
    if spec == "jetson.sensevoice_trt":
        from server.core.voxedge_backend_config import build_sensevoice_trt_config
        config = build_sensevoice_trt_config(profile=current_profile())
        return cls(config=config)
    if spec == "cpu.sherpa_asr":
        from server.core.voxedge_backend_config import build_sherpa_asr_config
        config = build_sherpa_asr_config(profile=current_profile())
        return cls(config=config)
    if spec == "rk.asr":
        from server.core.voxedge_backend_config import build_rk_asr_config
        config = build_rk_asr_config(profile=current_profile())
        return cls(config=config)
    if spec in ("hailo.whisper", "rk.whisper", "jetson.whisper_trt"):
        from server.core.voxedge_backend_config import build_config_for_spec
        return cls(config=build_config_for_spec(spec, "asr", current_profile()))
    return cls()
