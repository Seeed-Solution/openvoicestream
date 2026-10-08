"""Speaker-embedding extraction — product-layer shim over voxedge.

The inference engine (sherpa-onnx CAM++ wrapper) + stateless helpers live in
``voxedge.capabilities.speaker_embedding`` and are env-free. This product-layer
module keeps the deployment concerns: the ``OVS_SPEAKER_EMB`` feature flag
(default off), model-path resolution, and lazy on-demand download (honoring
HF_ENDPOINT mirrors). Opt-in, default-OFF, lazy-loaded.

OVS is stateless — it emits the raw embedding + metadata only; matching/identity
lives on the consumer side. Public API is unchanged so callers need no edits;
the stateless helpers are re-exported from voxedge.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from server.core.env_helpers import truthy

# Stateless helpers + model id come straight from voxedge (single source).
try:
    from voxedge.capabilities.speaker_embedding import (  # noqa: F401
        SPEAKER_MODEL_NAME,
        decode_audio_to_16k_mono,
        embedding_payload,
        encode_embedding,
        pcm16_to_float32,
        resample_linear,
    )
except Exception:  # voxedge optional at import time
    SPEAKER_MODEL_NAME = "campplus_sv_zh_en_3dspeaker"

logger = logging.getLogger(__name__)

_HF_URL_DEFAULT = (
    "{endpoint}/csukuangfj/speaker-embedding-models/resolve/main/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)

_embedder = None        # cached voxedge SpeakerEmbedder
_lock = threading.Lock()
_load_failed = False
_embedder_backend = None
_embedder_fallback = False


class SpeakerBackendError(RuntimeError):
    """A requested speaker-embedding backend is unavailable or failed."""


class SpeakerEmbeddingInputError(ValueError):
    """Audio cannot be represented by the selected embedding profile."""


def _trt_engine_file() -> str:
    """Path to a prebuilt CAM++ TRT engine (a *file*, not a dir).

    When set and the file exists, the Jetson TRT backend is preferred over the
    sherpa CPU path. Unset/empty (the default, and every non-Jetson image) keeps
    the existing sherpa behavior byte-for-byte.
    """
    return os.environ.get("DIAR_CAMPPLUS_ENGINE_FILE", "").strip()


def speaker_embedding_backend() -> str:
    value = os.environ.get("OVS_SPEAKER_EMB_BACKEND", "auto").strip().lower()
    if value in {"auto", "jetson_trt", "cpu_sherpa"}:
        return value
    raise SpeakerBackendError(f"unsupported speaker embedding backend: {value}")


def speaker_embedding_strict() -> bool:
    return speaker_embedding_backend() == "jetson_trt"


def embedding_metadata() -> dict:
    if _embedder_backend == "jetson_trt":
        return {"embedding_backend": "jetson_trt", "embedding_device": "cuda",
                "embedding_fallback": False}
    if _embedder_backend == "cpu_sherpa":
        return {"embedding_backend": "cpu_sherpa", "embedding_device": "cpu",
                "embedding_fallback": bool(_embedder_fallback)}
    return {}


class _TRTEmbedderAdapter:
    """Expose ``JetsonCampplusTRT`` under the ``SpeakerEmbedder.compute()`` API
    so ``compute_embedding`` stays backend-agnostic (no caller edits)."""

    def __init__(self, ext):
        self._ext = ext

    def ready(self) -> bool:
        return self._ext.ready()

    @property
    def dim(self) -> int:
        return self._ext.dim

    @property
    def frame_bounds(self):
        bounds = getattr(self._ext, "frame_bounds", None)
        if bounds is None:
            raise SpeakerBackendError("CAM++ TRT profile bounds are unavailable")
        return tuple(int(v) for v in bounds)

    def compute(self, samples, sample_rate):
        # JetsonCampplusTRT.extract: mono float32 [-1,1] -> 192-d L2-norm | None.
        return self._ext.extract(samples, sample_rate)


def speaker_embedding_enabled() -> bool:
    """Global default, from ``OVS_SPEAKER_EMB`` (default off). Overridable per
    connection via ``?speaker_embedding=`` / v2v config field.
    """
    return truthy(os.environ.get("OVS_SPEAKER_EMB", ""))


def _model_path() -> str:
    explicit = os.environ.get("OVS_SPEAKER_EMB_MODEL")
    if explicit:
        return explicit
    base = os.environ.get("MODEL_DIR", "/opt/models")
    return os.path.join(base, "speaker", "campplus.onnx")


def _hf_url() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return _HF_URL_DEFAULT.format(endpoint=endpoint)


def _ensure_model(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    if os.environ.get("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1").strip().lower() in {
        "0", "false", "no", "off"
    }:
        raise SpeakerBackendError(
            f"speaker CPU artifact is missing and OVS_AUTO_DOWNLOAD_ARTIFACTS is disabled: {path}"
        )
    import shutil
    import subprocess

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = _hf_url()
    logger.info("Speaker model missing; downloading %s -> %s", url, path)
    tmp = path + ".part"
    if shutil.which("curl"):
        # -L follows the HF-mirror 302 → LFS/CDN store; timeouts so a stuck or
        # unreachable mirror fails fast (feature degrades to off) instead of
        # hanging forever and wedging startup readiness.
        subprocess.run(
            ["curl", "-fSL", "--connect-timeout", "20", "--max-time", "1800",
             "--retry", "3", "-o", tmp, url],
            check=True, timeout=1900,
        )
    else:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    os.replace(tmp, path)
    logger.info("Speaker model ready (%d bytes).", os.path.getsize(path))


def _get_embedder():
    global _embedder, _load_failed, _embedder_backend, _embedder_fallback
    backend = speaker_embedding_backend()
    strict = backend == "jetson_trt"
    cache_ok = _embedder_backend == backend or backend == "auto"
    if _embedder is not None and cache_ok:
        return _embedder
    if _load_failed and not strict:
        return None
    with _lock:
        cache_ok = _embedder_backend == backend or backend == "auto"
        if _embedder is not None and cache_ok:
            return _embedder
        if _load_failed and not strict:
            return None
        engine_file = _trt_engine_file()
        if strict:
            path = Path(engine_file)
            if not engine_file or not path.is_file() or path.stat().st_size <= 0:
                raise SpeakerBackendError("strict CAM++ TRT backend requires a non-empty local engine file")
            try:
                from voxedge.capabilities.embedding_extractor import JetsonCampplusTRT
                ext = JetsonCampplusTRT(engine_file, strict=True)
                if not ext.ready():
                    raise SpeakerBackendError("CAM++ TRT engine is not ready")
                _embedder = _TRTEmbedderAdapter(ext)
                _embedder_backend = "jetson_trt"
                _load_failed = False
                return _embedder
            except (SpeakerBackendError, SpeakerEmbeddingInputError):
                raise
            except Exception as exc:
                raise SpeakerBackendError("CAM++ TRT backend initialization failed") from exc
        # Legacy auto mode retains the existing TRT-preferred/CPU-fallback path.
        if (speaker_embedding_backend() != "cpu_sherpa" and engine_file
                and os.path.isfile(engine_file) and os.path.getsize(engine_file) > 0):
            try:
                from voxedge.capabilities.embedding_extractor import JetsonCampplusTRT
                ext = JetsonCampplusTRT(engine_file)
                if ext.ready():
                    _embedder = _TRTEmbedderAdapter(ext)
                    _embedder_backend = "jetson_trt"
                    logger.info("Speaker embedding via Jetson TRT engine (%s).", engine_file)
                    return _embedder
                _embedder_fallback = True
            except Exception:
                _embedder_fallback = True
                logger.exception("Jetson TRT speaker backend init failed; falling back to sherpa CPU.")
        try:
            from voxedge.capabilities.speaker_embedding import SpeakerEmbedder
            path = _model_path()
            _ensure_model(path)
            num_threads = int(os.environ.get("OVS_SPEAKER_THREADS", "2"))
            emb = SpeakerEmbedder(path, num_threads=num_threads)
            if not emb.ready():
                _load_failed = True
                return None
            _embedder = emb
            _embedder_backend = "cpu_sherpa"
            if backend == "cpu_sherpa":
                _embedder_fallback = False
        except SpeakerBackendError:
            if os.environ.get("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1").strip().lower() in {
                "0", "false", "no", "off"
            }:
                raise
            _load_failed = True
            logger.exception("Failed to init speaker embedding; feature disabled.")
            return None
        except Exception:
            _load_failed = True
            logger.exception("Failed to init speaker embedding; feature disabled.")
            return None
    return _embedder


def preload() -> bool:
    """Eagerly load (call at startup only when enabled). Returns readiness."""
    return _get_embedder() is not None


def embedding_dim() -> int:
    emb = _get_embedder()
    return emb.dim if emb is not None else 0


def compute_embedding(samples, sample_rate: int):
    """L2-normalized float32 vector for one utterance, or None. Never raises."""
    emb = _get_embedder()
    if emb is None:
        return None
    try:
        result = emb.compute(samples, sample_rate)
    except (SpeakerBackendError, SpeakerEmbeddingInputError):
        raise
    except Exception as exc:
        # Bridge the extractor's typed errors into the product API without
        # matching exception text. The optional import keeps CPU-only images
        # importable.
        try:
            from voxedge.capabilities import embedding_extractor as _extractor
            input_error = getattr(_extractor, "EmbeddingInputError", ())
            backend_error = getattr(_extractor, "EmbeddingBackendError", ())
            if input_error and isinstance(exc, input_error):
                raise SpeakerEmbeddingInputError(str(exc)) from exc
            if backend_error and isinstance(exc, backend_error):
                raise SpeakerBackendError(str(exc)) from exc
        except (SpeakerEmbeddingInputError, SpeakerBackendError):
            raise
        except Exception:
            pass
        if speaker_embedding_strict():
            raise SpeakerBackendError("CAM++ TRT inference failed") from exc
        raise
    if result is None and speaker_embedding_strict():
        raise SpeakerEmbeddingInputError("audio is outside the CAM++ TRT profile")
    return result
