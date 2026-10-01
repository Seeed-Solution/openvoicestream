"""Focused contract tests for the bundled Silero streaming wrapper."""

import numpy as np
import sys
import types


class _FakeSession:
    def __init__(self):
        self.inputs = []

    def run(self, _outputs, inputs):
        self.inputs.append(inputs["input"].copy())
        # Keep the fake in speech for the shape/context assertions.
        return [np.array([[0.9]], dtype=np.float32), inputs["state"]]


class _SequenceSession(_FakeSession):
    def __init__(self, probabilities):
        super().__init__()
        self.probabilities = iter(probabilities)

    def run(self, _outputs, inputs):
        self.inputs.append(inputs["input"].copy())
        probability = next(self.probabilities)
        return [np.array([[probability]], dtype=np.float32), inputs["state"]]


def test_silero_16k_uses_official_frame_and_context(monkeypatch):
    from server.core import vad

    fake = _FakeSession()
    monkeypatch.setattr(vad, "_silero_session", fake)
    session = vad.SileroVADSession(sample_rate=16000, silence_ms=400)

    session.process(np.ones(1000, dtype=np.float32))
    assert session.WINDOW_16K == 512
    assert session.CONTEXT_16K == 64
    assert session._silence_step_threshold == 13
    assert [x.shape for x in fake.inputs] == [(1, 576)]
    np.testing.assert_array_equal(fake.inputs[0][0, :64], np.zeros(64))

    session.process(np.ones(536, dtype=np.float32))
    assert [x.shape for x in fake.inputs] == [(1, 576), (1, 576), (1, 576)]
    np.testing.assert_array_equal(fake.inputs[1][0, :64], np.ones(64))
    np.testing.assert_array_equal(fake.inputs[2][0, :64], np.ones(64))


def test_silero_silence_duration_uses_ceil_and_reset_clears_context(monkeypatch):
    from server.core import vad

    fake = _FakeSession()
    monkeypatch.setattr(vad, "_silero_session", fake)
    session = vad.SileroVADSession(sample_rate=16000, silence_ms=401)
    assert session._silence_step_threshold == 13
    session.process(np.ones(512, dtype=np.float32))
    assert np.any(session._context)

    session.reset()
    assert session._silence_steps == 0
    assert session._in_speech is False
    assert session._leftover.size == 0
    np.testing.assert_array_equal(session._context, np.zeros((1, 64)))


def test_silero_process_events_preserves_end_then_start_in_one_chunk(monkeypatch):
    from server.core import vad

    # Start a turn, then cross the two-frame (64 ms) silence threshold and
    # start another turn before the same input block is returned.
    fake = _SequenceSession([0.9, 0.0, 0.0, 0.9])
    monkeypatch.setattr(vad, "_silero_session", fake)
    session = vad.SileroVADSession(sample_rate=16000, silence_ms=64)

    events = session.process_events(np.ones(4 * 512, dtype=np.float32))

    assert events == [
        (vad.VADSession.SPEECH_START, 512),
        (vad.VADSession.SPEECH_END, 1536),
        (vad.VADSession.SPEECH_START, 2048),
    ]
    # Legacy callers retain the old last-event behavior.
    fake = _SequenceSession([0.9, 0.0, 0.0, 0.9])
    monkeypatch.setattr(vad, "_silero_session", fake)
    session = vad.SileroVADSession(sample_rate=16000, silence_ms=64)
    assert session.process(np.ones(4 * 512, dtype=np.float32)) == vad.VADSession.SPEECH_START


def test_silero_event_offset_accounts_for_previous_leftover(monkeypatch):
    from server.core import vad

    fake = _SequenceSession([0.9])
    monkeypatch.setattr(vad, "_silero_session", fake)
    session = vad.SileroVADSession(sample_rate=16000)

    assert session.process_events(np.ones(100, dtype=np.float32)) == []
    # The frame ends at concatenated offset 512, but 100 samples belonged to
    # the preceding call, so its offset in this call is 412.
    assert session.process_events(np.ones(412, dtype=np.float32)) == [
        (vad.VADSession.SPEECH_START, 412)
    ]


def test_webrtc_process_events_preserves_order_and_legacy_result(monkeypatch):
    from server.core import vad

    class _FakeWebRTCVad:
        values = iter([True, False, False, True])

        def __init__(self, _aggressiveness):
            pass

        def is_speech(self, _frame, _sample_rate):
            return next(self.values)

    monkeypatch.setitem(sys.modules, "webrtcvad", types.SimpleNamespace(Vad=_FakeWebRTCVad))
    session = vad.WebRTCVADSession(sample_rate=16000, silence_ms=60)
    samples = np.zeros(4 * 480, dtype=np.int16)

    assert session.process_events(samples) == [
        (vad.VADSession.SPEECH_START, 480),
        (vad.VADSession.SPEECH_END, 1440),
        (vad.VADSession.SPEECH_START, 1920),
    ]

    _FakeWebRTCVad.values = iter([True, False, False, True])
    session = vad.WebRTCVADSession(sample_rate=16000, silence_ms=60)
    assert session.process(samples) == vad.VADSession.SPEECH_START
