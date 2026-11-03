"""Audio feature extraction and vocoder reconstruction module.

Universal audio processing and Mel feature extraction module adhering to the
Matcha-TTS and Vocos acoustic contracts:
1. Physical Mel: 80-band filterbank, fmin=0.0, fmax=8000.0, power=1.0 (magnitude)
2. Unnormalized Log-Mel: m = ln(max(S_mel, eps)), eps=1e-10 (Vocos input domain & ONNX output domain)
3. Normalized Flow-Matching domain: z = (m - MEL_MEAN) / MEL_SIGMA (CFM training & MAS alignment domain)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import kaldi_native_fbank as knf
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

# Acoustic Feature Hyperparameters (Acoustic Contract)
SAMPLE_RATE: int = 16000
N_FFT: int = 1024
HOP_LENGTH: int = 256
WIN_LENGTH: int = 1024
WINDOW: str = "hann"
N_MELS: int = 80
FMIN: float = 0.0
FMAX: float = 8000.0
POWER: float = 1.0
CENTER: bool = True
PAD_MODE: str = "reflect"

# Flow-Matching Normalization Constants
MEL_MEAN: float = -2.9521978
MEL_SIGMA: float = 5.446792

# Frozen Spectral Floor Constant (Validated in Step 0 Baseline)
# Prevents low-level truncation cliffs while retaining clean silence (< -60 dBFS)
DEFAULT_EPS: float = 1e-10

# Default Vocos ONNX checkpoint
DEFAULT_VOCOS_PATH: str = "models/matcha-icefall-zh-en/vocos-16khz-univ.onnx"

# Module-level session cache
_CACHED_VOCOS_SESS: Optional[ort.InferenceSession] = None
_CACHED_VOCOS_PATH: Optional[str] = None
_CACHED_ISTFT: Optional[knf.IStft] = None


def get_istft_instance() -> knf.IStft:
    """Get or create cached kaldi_native_fbank IStft instance."""
    global _CACHED_ISTFT
    if _CACHED_ISTFT is None:
        stft_cfg = knf.StftConfig(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window_type=WINDOW,
            center=CENTER,
            pad_mode=PAD_MODE,
            normalized=False,
        )
        _CACHED_ISTFT = knf.IStft(stft_cfg)
    return _CACHED_ISTFT


def get_vocos_session(vocos_path: Union[str, os.PathLike, ort.InferenceSession] = DEFAULT_VOCOS_PATH) -> ort.InferenceSession:
    """Get or create cached ONNX InferenceSession for Vocos vocoder."""
    global _CACHED_VOCOS_SESS, _CACHED_VOCOS_PATH
    if isinstance(vocos_path, ort.InferenceSession):
        return vocos_path

    path_str = str(vocos_path)
    if _CACHED_VOCOS_SESS is None or _CACHED_VOCOS_PATH != path_str:
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Vocos ONNX model not found: {path_str}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        _CACHED_VOCOS_SESS = ort.InferenceSession(path_str, sess_options=opts, providers=["CPUExecutionProvider"])
        _CACHED_VOCOS_PATH = path_str
    return _CACHED_VOCOS_SESS


def load_audio(
    wav_input: Union[str, os.PathLike, np.ndarray],
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Load audio from file path or array, convert to mono, and resample with soxr_vhq.

    Args:
        wav_input: Path to audio file or existing 1D/2D numpy array (assumed 16kHz if array).
        target_sr: Target sample rate (default: 16000).

    Returns:
        1D float32 numpy array normalized in [-1.0, 1.0].
    """
    if isinstance(wav_input, (str, os.PathLike)):
        path = str(wav_input)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        # soundfile is fast and handles headers cleanly
        y, orig_sr = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if orig_sr != target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_vhq")
    elif isinstance(wav_input, np.ndarray):
        y = wav_input.astype(np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    else:
        raise TypeError(f"Unsupported wav_input type: {type(wav_input)}")

    return np.ascontiguousarray(y, dtype=np.float32)


def extract_log_mel(
    wav_input: Union[str, os.PathLike, np.ndarray],
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Extract unnormalized log-mel spectrogram (Vocos input domain and ONNX output domain).

    Mathematical formulation:
        S_mel = MelFilterBank * |STFT(s)|
        m = ln(max(S_mel, eps))

    Args:
        wav_input: Path to audio file or 1D audio waveform.
        eps: Floor clipping constant to prevent log(0). Default: 1e-10.

    Returns:
        np.ndarray [80, T]: Unnormalized log-mel spectrogram in float32.
    """
    y = load_audio(wav_input, target_sr=SAMPLE_RATE)
    # librosa.feature.melspectrogram with Slaney mel filterbank and Hann window
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=POWER,
        center=CENTER,
        pad_mode=PAD_MODE,
        norm="slaney",
        htk=False,
    )
    log_mel = np.log(np.maximum(mel, eps)).astype(np.float32)
    return log_mel


def extract_norm_mel(
    wav_input: Union[str, os.PathLike, np.ndarray],
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Extract normalized log-mel spectrogram for CFM training and MAS alignment domain.

    Mathematical formulation:
        z = (m - MEL_MEAN) / MEL_SIGMA

    Args:
        wav_input: Path to audio file or 1D audio waveform.
        eps: Floor clipping constant. Default: 1e-10.

    Returns:
        np.ndarray [80, T]: Standardized log-mel spectrogram (mean ~ 0, std ~ 1).
    """
    log_mel = extract_log_mel(wav_input, eps=eps)
    norm_mel = ((log_mel - MEL_MEAN) / MEL_SIGMA).astype(np.float32)
    return norm_mel


def denorm_mel(norm_mel: np.ndarray) -> np.ndarray:
    """Convert normalized log-mel z back to unnormalized log-mel m for Vocos input.

    Mathematical formulation:
        m = z * MEL_SIGMA + MEL_MEAN
    """
    return (norm_mel * MEL_SIGMA + MEL_MEAN).astype(np.float32)


def extract_mel(
    wav_input: Union[str, os.PathLike, np.ndarray],
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Convenience alias for extract_norm_mel (standard training/MAS input)."""
    return extract_norm_mel(wav_input, eps=eps)


def vocos_reconstruct(
    log_mel: np.ndarray,
    vocos_path: Union[str, os.PathLike, ort.InferenceSession] = DEFAULT_VOCOS_PATH,
) -> np.ndarray:
    """Reconstruct waveform from unnormalized log-mel spectrogram using Vocos ONNX + kaldi_native_fbank IStft.

    Args:
        log_mel: Unnormalized log-mel spectrogram of shape [80, T] or [1, 80, T].
        vocos_path: Path to vocos-16khz-univ.onnx or pre-instantiated InferenceSession.

    Returns:
        np.ndarray: Reconstructed 1D audio waveform in float32 at 16000 Hz.
    """
    sess = get_vocos_session(vocos_path)
    istft = get_istft_instance()

    mel_arr = np.asarray(log_mel, dtype=np.float32)
    if mel_arr.ndim == 2:
        if mel_arr.shape[0] != N_MELS:
            raise ValueError(f"Expected mel channel dimension {N_MELS}, got {mel_arr.shape[0]}")
        mels_in = mel_arr[np.newaxis, :, :]  # [1, 80, T]
    elif mel_arr.ndim == 3:
        if mel_arr.shape[1] != N_MELS:
            raise ValueError(f"Expected mel channel dimension {N_MELS}, got {mel_arr.shape[1]}")
        mels_in = mel_arr
    else:
        raise ValueError(f"Invalid log_mel shape: {mel_arr.shape}, expected [80, T] or [1, 80, T]")

    mag, x, y = sess.run(["mag", "x", "y"], {"mels": mels_in})
    num_frames = mag.shape[2]

    # Batch size 1 reconstruction
    real = (mag * x)[0].transpose().reshape(-1).tolist()
    imag = (mag * y)[0].transpose().reshape(-1).tolist()

    res = knf.StftResult(real=real, imag=imag, num_frames=num_frames)
    audio = np.array(istft(res), dtype=np.float32)
    return audio
