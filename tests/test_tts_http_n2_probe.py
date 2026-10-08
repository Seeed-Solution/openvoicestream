import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kokoro_stream_probe", ROOT / "bench/perf/tts_http_n2_probe.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_cases_accepts_top_level_list(tmp_path):
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps([{"id": "ja-1", "route": "ja", "lang": "ja", "voice": "jf_alpha", "text": "こんにちは"}]))
    cases = MODULE.load_cases(manifest)
    assert cases[0]["route"] == "ja"
    assert cases[0]["voice"] == "jf_alpha"


@pytest.mark.parametrize("payload,rate", [(b"\x00", 24000), (b"\x00\x00", 0)])
def test_pcm_stats_rejects_invalid_framing(payload, rate):
    with pytest.raises(ValueError):
        MODULE.pcm_stats(payload, rate)


def test_pcm_stats_reports_real_pcm_quality():
    stats = MODULE.pcm_stats(b"\x00\x00\xff\x7f", 24000)
    assert stats["finite"] is True
    assert stats["duration_s"] == pytest.approx(2 / 24000)
    assert stats["peak"] == pytest.approx(32767 / 32768)
    assert stats["silence_fraction"] == pytest.approx(0.5)


def test_run_one_maps_route_to_language_without_sending_route(monkeypatch):
    class Raw:
        done = False
        def read(self, size):
            if size == 4: return (24000).to_bytes(4, "little")
            if size == 1: return b"\x00"
            if not self.done:
                self.done = True; return b"\x00"
            return b""

    class Response:
        status_code = 200
        raw = Raw()
        def raise_for_status(self): pass
        def close(self): pass

    seen = {}
    def post(url, **kwargs):
        seen.update(kwargs); return Response()
    monkeypatch.setattr(MODULE.requests, "post", post)
    result = MODULE.run_one("http://mock", {"id": "gb", "route": "b", "text": "hello"}, 1)
    assert seen["json"] == {"text": "hello", "language": "en-GB"}
    assert "route" not in seen["json"]
    assert result["complete"] is True and result["audio_stats"]["duration_s"] == pytest.approx(1 / 24000)
