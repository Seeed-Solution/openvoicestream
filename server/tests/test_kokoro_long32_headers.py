from server.main import _tts_response_headers


def _meta(**extra):
    value = {"backend": "kokoro_long32", "T": [400], "route_ts": [400, 640], "snap_ratio": [0.0], "fallback": False}
    value.update(extra)
    return value


def test_rk_product_long32_headers_include_route_snap_and_fallback_state():
    headers = _tts_response_headers(_meta(), backend="rk.tts", mode="long32")
    assert headers["X-Kokoro-Selected-T"] == "400"
    assert headers["X-Kokoro-Route"] == "400,640"
    assert headers["X-Kokoro-Snap-Ratio"] == "0.0"
    assert headers["X-Kokoro-Fallback"] == "0"
    assert "X-RTF" in headers


def test_long32_fallback_is_explicit_and_multisegment_headers_are_bounded():
    headers = _tts_response_headers(
        _meta(T=list(range(400, 500)), selected_T=list(range(400, 500)), route_ts=list(range(400, 500)), snap_ratio=[0.1] * 100, fallback=True),
        backend="rk.tts", mode="long32",
    )
    assert headers["X-Kokoro-Fallback"] == "1"
    assert len(headers["X-Kokoro-Selected-T"]) <= 512
    assert len(headers["X-Kokoro-Route"]) <= 512
    assert len(headers["X-Kokoro-Snap-Ratio"]) <= 512


def test_unrelated_backend_with_long32_shaped_metadata_gets_no_kokoro_headers():
    headers = _tts_response_headers(_meta(), backend="other.tts", mode="long32")
    assert not any(key.startswith("X-Kokoro-") for key in headers)

def test_convonly_headers_use_audio_and_total_units_not_tail_rtf():
    headers = _tts_response_headers({"backend":"kokoro_convonly", "audio_s":2.5, "total_ms":1250, "full_rtf":0.5, "tail_rtf":99, "manifest_sha256":"a"*64, "platform":"rk3576", "selected_T":416, "frontend_ms":1.0, "prefix_ms":2.0, "tail_ms":3.0, "istft_ms":4.0, "wav_ms":5.0, "preload_ms":6.0}, backend="rk:kokoro_convonly")
    assert headers["X-Audio-Duration"] == "2.5"
    assert headers["X-Inference-Time"] == "1.25"
    assert headers["X-RTF"] == "0.5"
    assert headers["X-Kokoro-Manifest-SHA256"] == "a"*64
    assert headers["X-Kokoro-Platform"] == "rk3576"
    assert headers["X-Kokoro-Preload-Ms"] == "6.0"

def test_convonly_invalid_diagnostics_are_skipped_and_old_long32_unchanged():
    headers = _tts_response_headers({"backend":"kokoro_convonly", "audio_s":float("nan"), "total_ms":float("inf"), "full_rtf":float("nan"), "manifest_sha256":"bad", "platform":"bad", "frontend_ms":float("nan")}, backend="kokoro_convonly")
    assert headers["X-Audio-Duration"] == "0" and headers["X-Inference-Time"] == "0" and headers["X-RTF"] == "0"
    assert "X-Kokoro-Manifest-SHA256" not in headers and "X-Kokoro-Platform" not in headers
    old = _tts_response_headers(_meta(), backend="rk.tts", mode="long32")
    assert old["X-Kokoro-Route"] == "400,640" and old["X-Kokoro-Fallback"] == "0"


def test_convonly_actual_t_field_and_overflowing_numbers():
    headers = _tts_response_headers(
        {"backend": "kokoro_convonly", "T": 368, "audio_s": 10**400,
         "total_ms": 10**400, "full_rtf": 10**400, "tail_ms": 10**400},
        backend="rk:kokoro_convonly",
    )
    assert headers["X-Kokoro-Selected-T"] == "368"
    assert headers["X-Audio-Duration"] == "0"
    assert headers["X-Inference-Time"] == "0"
    assert headers["X-RTF"] == "0"
    assert "X-Kokoro-Tail-Ms" not in headers


def test_convonly_marker_cannot_override_unrelated_backend():
    headers = _tts_response_headers(
        {"backend": "kokoro_convonly", "T": 368, "duration": 2,
         "inference_time": 1, "rtf": 0.5, "full_rtf": 99},
        backend="other.tts",
    )
    assert headers == {"X-Audio-Duration": "2", "X-Inference-Time": "1", "X-RTF": "0.5"}
