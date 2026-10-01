"""Regression tests for the explicit Silero VAD dependency contract."""

import sys

import pytest


def test_silero_reports_missing_onnxruntime(monkeypatch):
    from server.core import vad

    monkeypatch.setattr(vad, "_silero_session", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(RuntimeError, match="silero VAD requires onnxruntime"):
        vad.SileroVADSession(sample_rate=16000)


def test_none_backend_does_not_import_onnxruntime():
    from server.core import vad

    assert vad.create_vad("none", sample_rate=16000) is None
