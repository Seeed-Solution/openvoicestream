import json
import math
import struct
import wave
from pathlib import Path

import piper_dp_asr_loopback as runner


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\0\0" * 160)


def test_iter_inputs_expands_stream_wavs_and_preserves_missing_row(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _wav(a); _wav(b)
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"results": [{"label": "dp256", "rows": [
        {"id": "case-a", "stream_wavs": [{"path": str(a)}, {"path": str(b)}]},
        {"id": "missing"},
    ]}]}))
    rows = list(runner.iter_inputs([bench], sync=False))
    assert [(row.get("case"), row.get("rep")) for row in rows[:2]] == [("case-a", 0), ("case-a", 1)]
    assert rows[0]["label"] == "dp256"
    assert rows[2]["case"] == "missing"
    assert "no selected WAV" in rows[2]["_error"]


def test_post_asr_uses_file_multipart_and_language_query(tmp_path):
    wav = tmp_path / "sample.wav"
    _wav(wav)
    seen = {}

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def read(self): return b'{"text":"ok","backend":"mock-asr"}'

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = request.data
        seen["timeout"] = timeout
        return Response()

    original = runner.urllib.request.urlopen
    runner.urllib.request.urlopen = fake_urlopen
    try:
        result = runner.post_asr("http://mock", wav, "English", 5)
    finally:
        runner.urllib.request.urlopen = original
    assert result["http_status"] == 200
    assert result["text"] == "ok"
    assert result["backend"] == "mock-asr"
    assert seen["url"] == "http://mock/asr?language=English"
    assert b'name="file"' in seen["body"]
    assert b"RIFF" in seen["body"]


def test_missing_wav_is_recorded_as_failure(tmp_path):
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"rows": [{"id": "gone", "stream_wav": {"path": str(tmp_path / "gone.wav")}}]}))
    rows = list(runner.iter_inputs([bench], sync=False))
    assert len(rows) == 1
    assert rows[0]["wav"].name == "gone.wav"
    try:
        runner.wav_info(rows[0]["wav"])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing WAV must fail when consumed")


def test_http_200_without_text_is_failed_and_keeps_body(tmp_path):
    wav = tmp_path / "sample.wav"
    _wav(wav)

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def read(self): return b'{"backend":"mock-asr"}'

    original = runner.urllib.request.urlopen
    runner.urllib.request.urlopen = lambda _request, timeout: Response()
    try:
        result = runner.post_asr("http://mock", wav, "English", 5)
    finally:
        runner.urllib.request.urlopen = original
    assert result["http_status"] == 200
    assert result["error"] == "response JSON has no non-empty text"
    assert result["response_body"] == '{"backend":"mock-asr"}'


def test_missing_benchmark_is_a_failure_item(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    rows = list(runner.iter_inputs([missing], sync=False))
    assert len(rows) == 1
    assert rows[0]["_error"].startswith("benchmark parse failed:")


def test_wav_info_pcm16_peak_rms_and_saturation(tmp_path):
    wav = tmp_path / "levels.wav"
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(struct.pack("<hhhh", 32767, -32768, 0, 0))
    info = runner.wav_info(wav)
    assert info["channels"] == 1
    assert info["sample_width"] == 2
    assert info["peak_abs"] == 32768
    assert info["saturated_samples"] == 2
    assert math.isclose(info["rms"], ((32767**2 + 32768**2) / 4) ** 0.5)
