"""Single-endpoint-detector guard for the voxedge engine path.

Two endpoint detectors on one /v2v/stream session race, and whichever fires
first silently truncates the other one's segment — no error on either side
(docs/CONFIGURATION.md "Streaming ASR endpointing: pick exactly one detector";
measured truncation on RK3588 in docs, and in
bench/parity/v2v_wav_inject.py runs on 2026-09-13).

`_endpoint_detector_conflict` is the pure decision the engine-path handler
calls before dispatching to `_v2v_stream_via_engine`: warning text when the
client asked for a server VAD while the ASR backend already owns endpointing,
else None.
"""
from __future__ import annotations

from server.main import _endpoint_detector_conflict


class _Backend:
    """Minimal stand-in for an ASR backend adapter."""

    def __init__(self, owns_endpoint: bool) -> None:
        self.prefer_backend_endpoint_vad = owns_endpoint


def test_no_conflict_when_backend_does_not_own_endpointing():
    for vad in ("none", "silero", "off", None, ""):
        assert _endpoint_detector_conflict(vad, _Backend(False)) is None


def test_no_conflict_when_client_did_not_ask_for_server_vad():
    # The single-detector arrangement: client owns it (vad:none), so a
    # backend-owned endpoint is exactly what the docs prescribe.
    for vad in ("none", "off", "disabled", None, "", "  NONE  "):
        assert _endpoint_detector_conflict(vad, _Backend(True)) is None


def test_conflict_when_both_sides_would_endpoint():
    msg = _endpoint_detector_conflict("silero", _Backend(True))
    assert msg is not None
    assert "silero" in msg
    assert "prefer_backend_endpoint_vad" in msg
    assert "docs/CONFIGURATION.md" in msg


def test_no_conflict_without_a_backend():
    # ASR disabled / not resolved yet: nothing to race with.
    assert _endpoint_detector_conflict("silero", None) is None


def test_backend_without_the_flag_attribute_is_treated_as_legacy():
    class _Legacy:
        pass

    assert _endpoint_detector_conflict("silero", _Legacy()) is None
