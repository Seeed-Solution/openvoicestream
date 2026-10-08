"""Unit tests for the diarization product shim (server/core/diarization.py).

CPU-only, no real CAM++ model: ``compute_embedding`` is monkeypatched to return
synthetic prototype vectors so we exercise the session orchestration, offline
segmentation + clustering, and the feature-flag default-off behaviour without a
profile or model download. The clustering kernel itself lives in voxedge and is
tested there; here we verify the product layer wires it correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from server.core import diarization as diar
from server.core import speaker_embedding as spk

DIM = 192


def _proto(idx: int) -> np.ndarray:
    """A unit basis vector — distinct prototypes are orthogonal (cosine 0)."""
    v = np.zeros(DIM, dtype=np.float32)
    v[idx] = 1.0
    return v


PROTO_A = _proto(0)
PROTO_B = _proto(1)


# ── feature flag (default OFF) ───────────────────────────────────────────────

def test_diarize_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OVS_DIARIZE", raising=False)
    assert diar.diarize_enabled() is False


def test_diarize_enabled_via_env(monkeypatch):
    monkeypatch.setenv("OVS_DIARIZE", "true")
    assert diar.diarize_enabled() is True
    monkeypatch.setenv("OVS_DIARIZE", "0")
    assert diar.diarize_enabled() is False


# ── online session orchestration ─────────────────────────────────────────────

def test_session_diarizer_assigns_multiple_speakers(monkeypatch):
    monkeypatch.delenv("OVS_DIARIZE", raising=False)  # params from defaults
    d = diar.make_session_diarizer()
    assert d is not None

    # A, B, A → two distinct speakers, A re-identified on its second turn.
    s0 = d.assign(PROTO_A, 0.0, 1.0)
    s1 = d.assign(PROTO_B, 1.5, 2.5)
    s2 = d.assign(PROTO_A, 3.0, 4.0)

    assert s0.speaker == "spk_0"
    assert s1.speaker == "spk_1"
    assert s2.speaker == "spk_0"            # same speaker as s0
    assert d.num_speakers == 2
    # Confidence is the cosine to the cluster centroid (orthogonal protos → ~1).
    assert s2.confidence >= 0.9


def test_summary_payload_relabels(monkeypatch):
    d = diar.make_session_diarizer()
    d.assign(PROTO_A, 0.0, 1.0)
    d.assign(PROTO_B, 1.5, 2.5)
    d.assign(PROTO_A, 3.0, 4.0)
    summary = diar.summary_payload(d)
    assert summary is not None
    assert summary["type"] == "diarization_summary"
    assert summary["num_speakers"] == 2
    assert len(summary["segments"]) == 3
    # Segments are time-ordered with the contract fields.
    for seg in summary["segments"]:
        assert set(seg) >= {"start", "end", "speaker", "confidence"}


def test_summary_payload_none_for_empty():
    d = diar.make_session_diarizer()
    assert diar.summary_payload(d) is None
    assert diar.summary_payload(None) is None


# ── offline diarize_audio (segmentation + clustering) ────────────────────────

def _make_two_speaker_audio(sr: int = 16000):
    """Three 0.6 s speech spans (A, B, A) separated by 0.5 s of silence.

    Speaker identity is encoded in amplitude so the monkeypatched embedder can
    return the right prototype from the audio slice alone.
    """
    def tone(amp: float, dur: float):
        t = np.arange(int(sr * dur)) / sr
        return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    a1 = tone(0.5, 0.6)   # speaker A (loud)
    b = tone(0.2, 0.6)    # speaker B (quiet)
    a2 = tone(0.5, 0.6)   # speaker A again
    return np.concatenate([a1, silence, b, silence, a2]), sr


def _fake_embed(samples, sample_rate):
    rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float32) ** 2)))
    return PROTO_A if rms > 0.25 else PROTO_B


def test_diarize_audio_segments_and_clusters(monkeypatch):
    monkeypatch.setattr(spk, "compute_embedding", _fake_embed)
    audio, sr = _make_two_speaker_audio()

    segs = diar.diarize_audio(audio, sr)
    # Three speech spans detected, two distinct speakers.
    assert len(segs) == 3
    speakers = [s.speaker for s in segs]
    assert len(set(speakers)) == 2
    # First and last span are the same (loud) speaker; middle is the other.
    assert speakers[0] == speakers[2]
    assert speakers[1] != speakers[0]
    # Time-ordered, monotonic, within clip bounds.
    assert segs[0].start < segs[1].start < segs[2].start
    assert segs[-1].end <= len(audio) / sr + 1e-3


def test_diarize_audio_respects_num_speakers(monkeypatch):
    monkeypatch.setattr(spk, "compute_embedding", _fake_embed)
    audio, sr = _make_two_speaker_audio()
    segs = diar.diarize_audio(audio, sr, num_speakers=1)
    assert len({s.speaker for s in segs}) == 1   # forced into one cluster


def test_diarize_response_envelope(monkeypatch):
    monkeypatch.setattr(spk, "compute_embedding", _fake_embed)
    audio, sr = _make_two_speaker_audio()
    segs = diar.diarize_audio(audio, sr)

    resp = diar.diarize_response(segs, return_embeddings=False)
    assert resp["num_speakers"] == 2
    assert resp["embedding_model"]
    assert all("embedding_b64" not in s for s in resp["segments"])

    resp_emb = diar.diarize_response(segs, return_embeddings=True)
    assert resp_emb["dim"] == DIM
    assert all("embedding_b64" in s for s in resp_emb["segments"])


def test_diarize_audio_empty_when_no_embeddings(monkeypatch):
    # Model unavailable → compute_embedding returns None → empty result.
    monkeypatch.setattr(spk, "compute_embedding", lambda s, sr: None)
    audio, sr = _make_two_speaker_audio()
    assert diar.diarize_audio(audio, sr) == []


def test_diarize_audio_silent_input(monkeypatch):
    monkeypatch.setattr(spk, "compute_embedding", _fake_embed)
    silent = np.zeros(16000, dtype=np.float32)
    assert diar.diarize_audio(silent, 16000) == []


def test_strict_segmenter_failure_is_typed(monkeypatch):
    monkeypatch.setenv("OVS_SPEAKER_EMB_BACKEND", "jetson_trt")
    monkeypatch.setattr(diar, "_KERNEL_OK", True)
    monkeypatch.setattr(spk, "_get_embedder", lambda: type("E", (), {"frame_bounds": (40, 4000)})())
    monkeypatch.setattr(diar, "_segment_audio", lambda *args: (_ for _ in ()).throw(RuntimeError("segmenter")))
    with pytest.raises(spk.SpeakerBackendError):
        diar.diarize_audio(np.ones(16000, dtype=np.float32), 16000)


def test_strict_kernel_missing_is_typed(monkeypatch):
    monkeypatch.setenv("OVS_SPEAKER_EMB_BACKEND", "jetson_trt")
    monkeypatch.setattr(diar, "_KERNEL_OK", False)
    with pytest.raises(spk.SpeakerBackendError):
        diar.diarize_audio(np.ones(16000, dtype=np.float32), 16000)


def test_strict_frame_count_matches_nx_snip_false_probe():
    # Values are from the installed kaldi_native_fbank 1.22.3 NX probe,
    # root-nx-installed-fbank-frame-probe-r2.raw.
    cases = {
        640000: 4000, 640080: 4001, 640240: 4002, 640400: 4003,
        479840: 2999, 479999: 3000, 480000: 3000,
    }
    assert {n: diar._trt_frame_count(n, 16000) for n in cases} == cases


def test_strict_chunks_use_bounds_and_cover_without_gaps():
    samples = np.zeros(640240, dtype=np.float32)
    chunks = diar._strict_span_chunks(samples, 0.0, len(samples) / 16000.0,
                                       16000, (40, 4000))
    assert len(chunks) == 2
    assert chunks[0][1] == 0.0
    assert chunks[-1][2] == len(samples) / 16000.0
    for i, (piece, start, end) in enumerate(chunks):
        assert diar._trt_frame_count(len(piece), 16000) <= 4000
        if i:
            assert start == chunks[i - 1][2]


def test_strict_chunks_do_not_add_frames_across_snip_false_boundaries():
    # NX fbank observation: two independent 640240-sample inputs each have
    # 4002 frames. A whole-span frame sum would incorrectly call this fixed
    # (4002,4002) profile impossible.
    samples = np.zeros(1280480, dtype=np.float32)
    chunks = diar._strict_span_chunks(samples, 0.0, len(samples) / 16000.0,
                                       16000, (4002, 4002))
    assert [len(piece) for piece, _, _ in chunks] == [640240, 640240]
    assert all(diar._trt_frame_count(len(piece), 16000) == 4002
               for piece, _, _ in chunks)
    assert chunks[0][2] == chunks[1][1]
    assert chunks[-1][2] == len(samples) / 16000.0


@pytest.mark.parametrize("samples, bounds, accepted", [
    (479840, (3000, 3000), False),  # NX observation is 2999 frames.
    (479999, (3000, 3000), True),
    (480000, (3000, 3000), True),
    (6400, (40, 1500), True),       # exact min-frame short span.
])
def test_strict_sample_bounds_short_tail_and_unpartitionable(samples, bounds, accepted):
    audio = np.zeros(samples, dtype=np.float32)
    if accepted:
        chunks = diar._strict_span_chunks(audio, 0.0, samples / 16000.0,
                                           16000, bounds)
        assert sum(len(piece) for piece, _, _ in chunks) == samples
        assert all(bounds[0] <= diar._trt_frame_count(len(piece), 16000) <= bounds[1]
                   for piece, _, _ in chunks)
    else:
        with pytest.raises(spk.SpeakerEmbeddingInputError):
            diar._strict_span_chunks(audio, 0.0, samples / 16000.0,
                                     16000, bounds)
