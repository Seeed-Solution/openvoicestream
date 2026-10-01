"""Server-side VAD with a shared singleton model.

Used by:
- WS /v2v/stream — auto-detect end-of-speech to finalize ASR
- WS /asr/stream?vad=silero — opt-in VAD for the ASR-only endpoint

Design: we load silero-vad's ONNX model DIRECTLY via onnxruntime —
intentionally avoiding the `silero_vad` pip package because it
pulls PyTorch + CUDA wheels (200+ MB on aarch64). Our containers
already ship onnxruntime, so the marginal cost is the 2.3 MB model
file bundled at `app/core/assets/silero_vad.onnx`.

The model loads once per process (~30 ms) and the InferenceSession
is shared across every WS connection. Per-connection state (LSTM
hidden state h+c, silence timer) is held in a Session object.

webrtcvad fallback is stateless C extension (`pip install
webrtcvad-wheels`), tiny but less accurate on noisy / non-English.

Backends:
- 'silero' — multilingual, accurate, 2.3 MB ONNX
- 'webrtcvad' — pure C, ~50 KB, low-overhead fallback
- 'none' — disable VAD (caller must finalize ASR explicitly)
"""
from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────
# Bundled silero-vad ONNX model location
# ───────────────────────────────────────────────────────────────
_SILERO_ONNX_PATH = Path(__file__).parent / "assets" / "silero_vad.onnx"


# ───────────────────────────────────────────────────────────────
# Singleton: ONNX InferenceSession loaded once per process
# ───────────────────────────────────────────────────────────────
_silero_session = None
_silero_load_lock = threading.Lock()


def _get_silero_session():
    """Lazy-load the silero VAD ONNX session; subsequent callers reuse it."""
    global _silero_session
    if _silero_session is not None:
        return _silero_session
    with _silero_load_lock:
        if _silero_session is not None:
            return _silero_session
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "silero VAD requires onnxruntime. Add `pip install "
                "onnxruntime` to your image, or use vad=webrtcvad / "
                "vad=none."
            ) from e
        model_path = os.environ.get("SILERO_VAD_ONNX_PATH", str(_SILERO_ONNX_PATH))
        if not Path(model_path).exists():
            raise RuntimeError(
                f"silero VAD ONNX model not found at {model_path}. "
                "Set SILERO_VAD_ONNX_PATH or bundle the model at "
                "app/core/assets/silero_vad.onnx."
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _silero_session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        logger.info(
            "silero-vad ONNX session loaded (shared, %s)", model_path
        )
        return _silero_session


# ───────────────────────────────────────────────────────────────
# Per-connection VAD sessions
# ───────────────────────────────────────────────────────────────

class VADSession:
    """Abstract base. Feed PCM, get speech / endpoint events."""

    SPEECH_START   = "speech_start"
    SPEECH_END     = "speech_end"      # silence threshold crossed
    NONE           = None              # no transition this chunk

    def process(self, samples: np.ndarray) -> Optional[str]:
        """Feed one chunk of int16 or float32 PCM (16 kHz mono assumed).

        Returns one of:
          SPEECH_START  — first speech frame after silence (onset)
          SPEECH_END    — silence sustained past threshold (endpoint)
          None          — no transition this chunk
        """
        raise NotImplementedError

    def process_events(self, samples: np.ndarray) -> list[tuple[str, int]]:
        """Feed PCM and return ordered ``(event, sample_offset)`` pairs.

        This compatibility default is for older third-party/fake sessions that
        implement only ``process``.  Concrete streaming VADs override it to
        preserve every transition within a PCM block.  The offset fallback is
        the end of the supplied block because an older implementation does not
        expose the transition position.
        """
        event = self.process(samples)
        return [(event, int(len(samples)))] if event is not None else []

    def reset(self) -> None:
        """Reset state (e.g., after a forced finalize)."""
        raise NotImplementedError


class SileroVADSession(VADSession):
    """silero-vad streaming wrapper.

    Runs the silero ONNX model directly via onnxruntime — no PyTorch
    dependency. Per-connection state = LSTM hidden state (h, c) + a
    silence-duration counter; the InferenceSession itself is shared
    with every other session in this process (singleton above).
    """

    # The bundled Silero ONNX graph follows the official 16 kHz wrapper:
    # 512 samples per step with 64 samples of preceding context.  The
    # 256-sample step is the 8 kHz protocol, even though this graph accepts
    # the shorter dynamic input shape without raising an error.
    WINDOW_16K = 512
    CONTEXT_16K = 64

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        silence_ms: int = 400,
        speech_pad_ms: int = 100,   # accepted for API parity; not used here
    ):
        if sample_rate != 16000:
            raise ValueError(f"silero VAD expects 16 kHz, got {sample_rate}")
        self._session = _get_silero_session()
        self._sr = np.array(sample_rate, dtype=np.int64)
        # LSTM state: [2 (=h+c), batch=1, hidden=128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._threshold = float(threshold)
        # Silence counter in steps (512 samples = 32 ms at 16 kHz).  Ceil so
        # an endpoint never fires before the configured silence duration.
        self._silence_step_threshold = max(1, math.ceil(silence_ms / 32))
        self._silence_steps = 0
        self._in_speech = False
        self._leftover: np.ndarray = np.empty(0, dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_16K), dtype=np.float32)

    def process_events(self, samples: np.ndarray) -> list[tuple[str, int]]:
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        previous_leftover = int(self._leftover.size)
        buf = np.concatenate([self._leftover, samples])
        events: list[tuple[str, int]] = []
        i = 0
        while i + self.WINDOW_16K <= len(buf):
            window = buf[i : i + self.WINDOW_16K].reshape(1, -1)
            model_input = np.concatenate([self._context, window], axis=1)
            out, new_state = self._session.run(
                None,
                {"input": model_input, "state": self._state, "sr": self._sr},
            )
            self._state = new_state
            self._context = model_input[:, -self.CONTEXT_16K:]
            prob = float(out[0, 0])
            speech = prob >= self._threshold
            if speech:
                if not self._in_speech:
                    self._in_speech = True
                    offset = min(max(i + self.WINDOW_16K - previous_leftover, 0), len(samples))
                    events.append((self.SPEECH_START, int(offset)))
                self._silence_steps = 0
            else:
                if self._in_speech:
                    self._silence_steps += 1
                    if self._silence_steps >= self._silence_step_threshold:
                        self._in_speech = False
                        self._silence_steps = 0
                        offset = min(max(i + self.WINDOW_16K - previous_leftover, 0), len(samples))
                        events.append((self.SPEECH_END, int(offset)))
            i += self.WINDOW_16K
        self._leftover = buf[i:]
        return events

    def process(self, samples: np.ndarray) -> Optional[str]:
        events = self.process_events(samples)
        return events[-1][0] if events else None

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._silence_steps = 0
        self._in_speech = False
        self._leftover = np.empty(0, dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_16K), dtype=np.float32)


class WebRTCVADSession(VADSession):
    """webrtcvad streaming wrapper. Per-frame is_speech check + a
    silence-frame counter."""

    FRAME_MS = 30   # webrtcvad supports 10/20/30 ms frames
    FRAME_BYTES_16K = 16000 * (FRAME_MS / 1000) * 2  # int16 samples * 2 bytes

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,    # 0..3, higher = more aggressive
        silence_ms: int = 400,
    ):
        import webrtcvad
        self._vad = webrtcvad.Vad(int(aggressiveness))
        self._sr = sample_rate
        self._frame_bytes = int(sample_rate * (self.FRAME_MS / 1000) * 2)
        self._silence_frames_threshold = max(1, silence_ms // self.FRAME_MS)
        self._silence_count = 0
        self._leftover_bytes = b""
        self._in_speech = False

    def process_events(self, samples: np.ndarray) -> list[tuple[str, int]]:
        if samples.dtype == np.float32:
            samples = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        elif samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        previous_leftover_samples = len(self._leftover_bytes) // 2
        buf = self._leftover_bytes + samples.tobytes()
        events: list[tuple[str, int]] = []
        i = 0
        while i + self._frame_bytes <= len(buf):
            frame = buf[i : i + self._frame_bytes]
            speech = self._vad.is_speech(frame, self._sr)
            if speech:
                if not self._in_speech:
                    self._in_speech = True
                    offset = min(max((i + self._frame_bytes) // 2 - previous_leftover_samples, 0), len(samples))
                    events.append((self.SPEECH_START, int(offset)))
                self._silence_count = 0
            else:
                if self._in_speech:
                    self._silence_count += 1
                    if self._silence_count >= self._silence_frames_threshold:
                        self._in_speech = False
                        offset = min(max((i + self._frame_bytes) // 2 - previous_leftover_samples, 0), len(samples))
                        events.append((self.SPEECH_END, int(offset)))
                        self._silence_count = 0
            i += self._frame_bytes
        self._leftover_bytes = buf[i:]
        return events

    def process(self, samples: np.ndarray) -> Optional[str]:
        events = self.process_events(samples)
        return events[-1][0] if events else None

    def reset(self) -> None:
        self._silence_count = 0
        self._leftover_bytes = b""
        self._in_speech = False


def create_vad(
    backend: str = "silero",
    sample_rate: int = 16000,
    silence_ms: int = 400,
    **kwargs,
) -> Optional[VADSession]:
    """Factory. Returns None for backend='none'."""
    if backend in (None, "", "none", "off", "disabled"):
        return None
    if backend == "silero":
        return SileroVADSession(sample_rate=sample_rate, silence_ms=silence_ms, **kwargs)
    if backend in ("webrtc", "webrtcvad"):
        return WebRTCVADSession(sample_rate=sample_rate, silence_ms=silence_ms, **kwargs)
    raise ValueError(f"unknown VAD backend: {backend!r}; use 'silero' | 'webrtcvad' | 'none'")
