"""Focused tests for REAL native stream cancellation on the canonical
``_TRTEdgeLLMStreamingASRStream`` (U2 attempt2).

Status: AUTHORED for attempt2; run ONCE by the authoring worker. A fresh
independent reviewer must re-run on a frozen tree.

Attempt2 corrections pinned here (on top of the attempt1 coverage):

  * BEGIN-WRITTEN and CANCEL-REQUESTED are SEPARATE per-SID Events. The begin
    ``on_written`` callback leaves the cancel-requested marker UNSET.
  * ``arm_cancel`` sets ONLY cancel-requested (never the begin barrier).
  * An UNARMED ``cancelled`` terminal is rejected (``WorkerProtocolError``) and
    does NOT fabricate an intent/confirmation.
  * Rotation commit/reset/SID-switch is linearized under the state lock; a
    cancel armed during a blocked rotation preserves committed text and does
    NOT launch a new SID.
  * Ordinary state mutation shares the same lock as the armed check; text
    postprocess happens OUTSIDE the lock. A cancel armed while postprocess is
    blocked leaves normal text unchanged.
  * ``close`` after a CONFIRMED receipt sends ZERO ``end``; an ``end`` failure
    does NOT mark the stream closed; an armed-but-unconfirmed cancel sends no
    ``end`` and stays open.
  * A duplicate concurrent ``request_cancel`` issues only ONE writer.
  * Per-SID ordinary in-flight counts are exact and return to zero after a real
    overlap.
  * A confirmed receipt is idempotent: a second ``request_cancel`` returns the
    retained actual receipt without a duplicate native cancel.

The canonical class is imported from ``voxedge.backends.jetson.trt_edge_llm_asr``
and runs against a controlled ``FakeBackend`` (no GPU, no subprocess). Blocking
uses real ``threading.Event`` handshakes; every thread is joined with a finite
budget. Fake IO handlers assert the stream's state lock is NOT held during IO.
"""

from __future__ import annotations

import importlib
import threading
import time

import numpy as np
import pytest

_asr_mod = importlib.import_module("voxedge.backends.jetson.trt_edge_llm_asr")
TRTEdgeLLMASRConfig = _asr_mod.TRTEdgeLLMASRConfig
WorkerExitError = _asr_mod.WorkerExitError
WorkerProtocolError = _asr_mod.WorkerProtocolError
_Stream = _asr_mod._TRTEdgeLLMStreamingASRStream

CANON_TIMEOUT = 5.0


def _assert_lock_free(stream) -> None:
    """Fail if the stream's state lock is currently held by the caller thread."""
    acquired = stream._state_lock.acquire(timeout=0.0)
    if acquired:
        stream._state_lock.release()
    else:
        raise AssertionError("state lock held during worker IO")


class FakeBackend:
    """Controlled backend for the real stream class.

    Handlers run WITHOUT the stream's state lock (the tests assert this via
    ``_assert_lock_free``). ``cancel_barrier`` blocks the actual cancel helper
    so tests can control timeout/overlap deterministically.
    """

    def __init__(self, *, segment_cap_sec: float = 0.0):
        cfg = TRTEdgeLLMASRConfig()
        cfg.stream_chunk_sec = 0.01
        cfg.segment_cap_sec = segment_cap_sec
        self._config = cfg
        self.begin_calls: list[dict] = []
        self.chunk_calls: list[dict] = []
        self.end_calls: list[dict] = []
        self.begin_handler = None       # callable(ev, on_written) -> dict | None
        self.chunk_handler = None       # callable(ev) -> dict
        self.end_handler = None         # callable(ev) -> dict
        self.cancel_calls: list[str] = []
        self.cancel_handler = None      # callable(sid, timeout) -> dict
        self.cancel_raises: BaseException | None = None
        self.cancel_barrier = threading.Event()
        self.restart_count = 0
        self.stream_ref = None          # set by tests when lock assertions wanted

    def _assert(self):
        if self.stream_ref is not None:
            _assert_lock_free(self.stream_ref)

    def _worker_request(self, input_data, *, expected_cancel=None, on_written=None):
        event = input_data.get("event")
        self._assert()
        if event == "begin":
            self.begin_calls.append(input_data)
            if self.begin_handler is not None:
                custom = self.begin_handler(input_data, on_written)
                if custom is not None:
                    return custom
            if on_written is not None:
                on_written()
            return {"event": "begin_ack", "id": input_data.get("id")}
        if event == "chunk":
            self.chunk_calls.append(input_data)
            if self.chunk_handler is not None:
                return self.chunk_handler(input_data)
            return {"event": "partial", "id": input_data.get("id"), "text": "x"}
        if event == "end":
            self.end_calls.append(input_data)
            if self.end_handler is not None:
                return self.end_handler(input_data)
            return {"event": "done", "id": input_data.get("id")}
        raise AssertionError(f"unexpected event {event!r}")

    def _worker_cancel_and_wait(self, sid: str, timeout_s: float) -> dict:
        self.cancel_calls.append(sid)
        if not self.cancel_barrier.wait(timeout_s):
            raise TimeoutError(f"fake cancel barrier timed out for {sid}")
        if self.cancel_raises is not None:
            raise self.cancel_raises
        if self.cancel_handler is not None:
            return self.cancel_handler(sid, timeout_s)
        return {"event": "cancelled", "id": sid, "ok": False, "epoch": 3}

    def _strip_language_prefix(self, text: str):
        return text, None

    def _postprocess_text(self, text: str):
        return text, None

    def transcribe(self, wav_bytes, language="auto"):
        raise AssertionError("offline rescue must not run for an armed cancel")


def _new_stream(backend: FakeBackend, language: str = "auto"):
    stream = _Stream(backend, language=language)
    backend.stream_ref = stream
    return stream


# ---------------------------------------------------------------------------
# 1. Genuine overlap: blocked ordinary consumer + control cancel
# ---------------------------------------------------------------------------


def test_blocked_chunk_and_control_cancel_returns_exact_receipt():
    backend = FakeBackend()
    backend.cancel_barrier.set()
    backend.cancel_handler = (
        lambda sid, t: {"event": "cancelled", "id": sid, "ok": False, "epoch": 11}
    )

    chunk_started = threading.Event()
    release_chunk = threading.Event()
    sid_seen: list[str] = []

    def chunk_handler(ev):
        sid_seen.append(ev["id"])
        chunk_started.set()
        assert release_chunk.wait(CANON_TIMEOUT)
        return {"event": "cancelled", "id": ev["id"], "ok": False, "epoch": 11}

    backend.chunk_handler = chunk_handler
    stream = _new_stream(backend)
    assert len(backend.begin_calls) == 1

    results: dict = {}

    def run_chunk():
        try:
            results["chunk"] = stream._send_chunk(last=False)
        except BaseException as exc:  # noqa: BLE001
            results["chunk_exc"] = exc

    def run_cancel():
        try:
            results["receipt"] = stream.request_cancel(CANON_TIMEOUT)
        except BaseException as exc:  # noqa: BLE001
            results["cancel_exc"] = exc

    t_chunk = threading.Thread(target=run_chunk, name="u2-chunk", daemon=True)
    t_cancel = threading.Thread(target=run_cancel, name="u2-cancel", daemon=True)
    t_chunk.start()
    assert chunk_started.wait(CANON_TIMEOUT)
    t_cancel.start()
    deadline = time.monotonic() + CANON_TIMEOUT
    while not backend.cancel_calls and time.monotonic() < deadline:
        time.sleep(0.005)
    assert backend.cancel_calls, "control cancel never reached the backend"
    release_chunk.set()

    t_chunk.join(timeout=CANON_TIMEOUT)
    t_cancel.join(timeout=CANON_TIMEOUT)
    assert not t_chunk.is_alive(), "ordinary chunk thread did not join"
    assert not t_cancel.is_alive(), "control cancel thread did not join"
    assert "chunk_exc" not in results, results
    assert "cancel_exc" not in results, results

    assert results["receipt"] == {
        "event": "cancelled",
        "id": sid_seen[0],
        "ok": False,
        "epoch": 11,
    }
    assert results["chunk"]["event"] == "cancelled"
    assert results["chunk"]["id"] == sid_seen[0]
    assert stream._final_text == ""
    assert stream._partial_text == ""
    assert stream._closed is False
    assert stream._cancel_confirm is not None
    assert stream._cancel_confirm["sid"] == sid_seen[0]
    # Attempt2: in-flight accounting returns to zero after the real overlap.
    assert stream._inflight_for(sid_seen[0]) == 0


# ---------------------------------------------------------------------------
# 2. Timeout: intent armed, no confirm/closed/restart
# ---------------------------------------------------------------------------


def test_timeout_leaves_intent_armed_no_confirm_no_close_no_restart():
    backend = FakeBackend()
    stream = _new_stream(backend)

    with pytest.raises(TimeoutError):
        stream.request_cancel(0.1)

    assert stream._cancel_confirm is None
    assert stream._cancel_exit is False
    assert stream._closed is False
    assert stream._cancel_intent is not None
    assert stream._control_inflight is False
    assert backend.restart_count == 0
    assert stream.finalize() == ("", None)
    stream.close()
    assert stream._closed is False


# ---------------------------------------------------------------------------
# 3. Exit is distinct from timeout
# ---------------------------------------------------------------------------


def test_worker_exit_is_distinct_from_timeout():
    backend = FakeBackend()
    backend.cancel_barrier.set()
    backend.cancel_raises = WorkerExitError("worker gone")
    stream = _new_stream(backend)

    with pytest.raises(WorkerExitError):
        stream.request_cancel(CANON_TIMEOUT)

    assert stream._cancel_exit is True
    assert stream._cancel_confirm is None
    assert stream._closed is False
    assert stream._control_inflight is False


# ---------------------------------------------------------------------------
# 4. Invalid / mismatch fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "1", None])
def test_invalid_timeout_rejects_before_state_change(bad):
    backend = FakeBackend()
    stream = _new_stream(backend)
    assert stream._cancel_intent is None
    with pytest.raises(ValueError):
        stream.request_cancel(bad)
    assert stream._cancel_intent is None
    assert stream._cancel_armed_sids == set()
    assert backend.cancel_calls == []


def test_mismatched_or_malformed_receipt_rejected():
    backend = FakeBackend()
    backend.cancel_barrier.set()
    backend.cancel_handler = lambda sid, t: {"event": "cancelled", "id": "OTHER", "ok": False}
    stream = _new_stream(backend)
    with pytest.raises(WorkerProtocolError):
        stream.request_cancel(CANON_TIMEOUT)
    assert stream._cancel_confirm is None
    assert stream._cancel_intent is not None
    assert stream._control_inflight is False


def test_ok_true_receipt_is_not_a_cancel_receipt():
    backend = FakeBackend()
    backend.cancel_barrier.set()
    backend.cancel_handler = lambda sid, t: {"event": "cancelled", "id": sid, "ok": True}
    stream = _new_stream(backend)
    with pytest.raises(WorkerProtocolError):
        stream.request_cancel(CANON_TIMEOUT)
    assert stream._cancel_confirm is None


def test_non_dict_receipt_rejected():
    backend = FakeBackend()
    backend.cancel_barrier.set()
    backend.cancel_handler = lambda sid, t: "not-a-dict"
    stream = _new_stream(backend)
    with pytest.raises(WorkerProtocolError):
        stream.request_cancel(CANON_TIMEOUT)
    assert stream._cancel_confirm is None


# ---------------------------------------------------------------------------
# 5. Begin-write barrier: timeout then later success
# ---------------------------------------------------------------------------


def test_begin_write_barrier_timeout_then_success():
    backend = FakeBackend()
    captured_written: list = []

    def begin_handler(ev, on_written):
        captured_written.append(on_written)
        return {"event": "begin_ack", "id": ev["id"]}

    backend.begin_handler = begin_handler
    stream = _new_stream(backend)
    captured = stream._session_id

    with pytest.raises(TimeoutError):
        stream.request_cancel(0.1)
    assert not stream._begin_written_event(captured).is_set()
    assert stream._cancel_confirm is None
    assert backend.cancel_calls == []
    assert stream._cancel_intent["sid"] == captured
    assert stream._control_inflight is False

    backend.cancel_barrier.set()
    assert captured_written and captured_written[0] is not None
    captured_written[0]()
    assert stream._begin_written_event(captured).is_set()
    receipt = stream.request_cancel(CANON_TIMEOUT)
    assert receipt["id"] == captured
    assert backend.cancel_calls == [captured]
    assert stream._cancel_confirm["sid"] == captured


# ---------------------------------------------------------------------------
# 6. Attempt2: marker separation
# ---------------------------------------------------------------------------


def test_begin_written_callback_leaves_cancel_requested_unset():
    backend = FakeBackend()
    stream = _new_stream(backend)
    sid = stream._session_id
    assert stream._begin_written_event(sid).is_set()
    # The ordinary begin must NOT have armed cancellation.
    assert not stream._cancel_requested_event(sid).is_set()
    assert not stream._cancel_armed_for(sid)
    assert stream._cancel_intent is None
    # An ordinary partial/final still mutates normally (no false suppression).
    backend.chunk_handler = lambda ev: {"event": "partial", "id": ev["id"], "text": "ok"}
    stream._send_chunk(last=False)
    assert stream._partial_text == "ok"


def test_arm_cancel_does_not_fake_begin_written():
    backend = FakeBackend()

    def begin_handler(ev, on_written):
        return {"event": "begin_ack", "id": ev["id"]}  # never fires barrier

    backend.begin_handler = begin_handler
    stream = _new_stream(backend)
    sid = stream.arm_cancel()
    assert stream._cancel_requested_event(sid).is_set()
    assert not stream._begin_written_event(sid).is_set()
    # cancel cannot be written until the begin barrier is genuinely satisfied.
    with pytest.raises(TimeoutError):
        stream.request_cancel(0.1)
    assert backend.cancel_calls == []


def test_unarmed_cancelled_receipt_is_rejected():
    backend = FakeBackend()
    backend.chunk_handler = lambda ev: {
        "event": "cancelled",
        "id": ev["id"],
        "ok": False,
        "epoch": 1,
    }
    stream = _new_stream(backend)
    with pytest.raises(WorkerProtocolError):
        stream._send_chunk(last=False)
    assert stream._cancel_intent is None
    assert stream._cancel_confirm is None


# ---------------------------------------------------------------------------
# 7. Cancellation during rotation preserves committed text / no new SID
# ---------------------------------------------------------------------------


def test_cancellation_during_rotation_does_not_begin_new_sid():
    backend = FakeBackend(segment_cap_sec=0.001)
    rotation_chunk_seen = threading.Event()
    release_rotation = threading.Event()

    def chunk_handler(ev):
        if ev.get("last"):
            rotation_chunk_seen.set()
            assert release_rotation.wait(CANON_TIMEOUT)
            return {"event": "final", "id": ev["id"], "text": "hello"}
        return {"event": "partial", "id": ev["id"], "text": "hel"}

    backend.chunk_handler = chunk_handler
    stream = _new_stream(backend)
    old_sid = stream._session_id

    errors: list = []

    def run_accept():
        try:
            stream.accept_waveform(16000, np.zeros(32, dtype=np.float32))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t_accept = threading.Thread(target=run_accept, name="u2-rotate-accept", daemon=True)
    t_accept.start()
    assert rotation_chunk_seen.wait(CANON_TIMEOUT)
    stream.arm_cancel()
    release_rotation.set()
    t_accept.join(timeout=CANON_TIMEOUT)
    assert not t_accept.is_alive()
    assert errors == []
    assert len(backend.begin_calls) == 1
    assert stream._session_id == old_sid
    # Attempt2: blocked rotation must leave committed text UNCHANGED.
    assert stream._committed_text == ""


def test_blocked_rotation_with_prior_commit_preserves_committed_text():
    backend = FakeBackend(segment_cap_sec=0.001)
    stream = _new_stream(backend)
    stream._committed_text = "PREVIOUS"  # simulate one committed segment
    rotation_seen = threading.Event()
    release = threading.Event()

    def chunk_handler(ev):
        rotation_seen.set()
        assert release.wait(CANON_TIMEOUT)
        return {"event": "final", "id": ev["id"], "text": "NEWTEXT"}

    backend.chunk_handler = chunk_handler

    errors: list = []

    def run_accept():
        try:
            stream.accept_waveform(16000, np.zeros(32, dtype=np.float32))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=run_accept, name="u2-rotate2", daemon=True)
    t.start()
    assert rotation_seen.wait(CANON_TIMEOUT)
    stream.arm_cancel()
    release.set()
    t.join(timeout=CANON_TIMEOUT)
    assert errors == []
    assert stream._committed_text == "PREVIOUS"
    assert len(backend.begin_calls) == 1


# ---------------------------------------------------------------------------
# 8. Postprocess blocked then arm leaves normal text unchanged
# ---------------------------------------------------------------------------


def test_postprocess_blocked_then_arm_leaves_normal_text_unchanged():
    backend = FakeBackend()
    post_seen = threading.Event()
    release_post = threading.Event()

    def slow_post(text):
        post_seen.set()
        assert release_post.wait(CANON_TIMEOUT)
        return text, None

    backend._postprocess_text = slow_post
    backend.chunk_handler = lambda ev: {"event": "final", "id": ev["id"], "text": "LATE"}

    stream = _new_stream(backend)
    errors: list = []

    def run_chunk():
        try:
            stream._send_chunk(last=True)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=run_chunk, name="u2-post", daemon=True)
    t.start()
    assert post_seen.wait(CANON_TIMEOUT)
    stream.arm_cancel()
    release_post.set()
    t.join(timeout=CANON_TIMEOUT)
    assert errors == []
    assert stream._final_text == ""
    assert stream._closed is False


# ---------------------------------------------------------------------------
# 9. Late final / armed accept
# ---------------------------------------------------------------------------


def test_late_final_after_armed_cancel_suppresses_text_and_rescue():
    backend = FakeBackend()
    backend.chunk_handler = lambda ev: {
        "event": "final",
        "id": ev["id"],
        "text": "SHOULD-NOT-APPEAR",
    }
    stream = _new_stream(backend)
    stream.arm_cancel()
    stream.accept_waveform(16000, np.zeros(16000, dtype=np.float32))
    assert backend.chunk_calls == []
    resp = stream._send_chunk(last=True)
    assert resp["event"] == "final"
    assert stream._final_text == ""
    assert stream._closed is False
    assert stream.finalize() == ("", None)


def test_armed_rotation_terminal_does_not_issue_new_begin():
    backend = FakeBackend()
    stream = _new_stream(backend)
    stream.arm_cancel()
    backend.chunk_handler = lambda ev: {"event": "segment_rotation", "id": ev["id"]}
    sid_before = stream._session_id
    stream._send_chunk(last=True)
    assert stream._session_id == sid_before
    assert len(backend.begin_calls) == 1


# ---------------------------------------------------------------------------
# 10. Close discipline
# ---------------------------------------------------------------------------


def test_close_marks_closed_normally_when_uncancelled():
    backend = FakeBackend()
    stream = _new_stream(backend)
    stream.close()
    assert stream._closed is True
    assert any(e.get("event") == "end" for e in backend.end_calls)


def test_confirmed_close_sends_zero_end():
    backend = FakeBackend()
    backend.cancel_barrier.set()
    stream = _new_stream(backend)
    sid = stream._session_id
    stream.request_cancel(CANON_TIMEOUT)
    assert stream._cancel_confirm["sid"] == sid
    end_before = len(backend.end_calls)
    stream.close()
    assert stream._closed is True
    assert len(backend.end_calls) == end_before  # ZERO redundant end


def test_close_does_not_falsely_mark_closed_after_unconfirmed_cancel():
    backend = FakeBackend()
    stream = _new_stream(backend)
    with pytest.raises(TimeoutError):
        stream.request_cancel(0.1)
    assert stream._cancel_confirm is None
    stream.close()
    assert stream._closed is False
    assert all(e.get("event") != "end" for e in backend.end_calls)


def test_end_failure_does_not_mark_closed():
    backend = FakeBackend()

    def end_handler(ev):
        raise WorkerExitError("end failed")

    backend.end_handler = end_handler
    stream = _new_stream(backend)
    stream.close()
    assert stream._closed is False  # not falsely marked closed


# ---------------------------------------------------------------------------
# 11. Duplicate control / idempotence
# ---------------------------------------------------------------------------


def test_duplicate_control_issues_one_writer():
    backend = FakeBackend()
    entered = threading.Event()
    release = threading.Event()

    def cancel_handler(sid, t):
        entered.set()
        assert release.wait(CANON_TIMEOUT)
        return {"event": "cancelled", "id": sid, "ok": False, "epoch": 1}

    backend.cancel_handler = cancel_handler
    backend.cancel_barrier.set()
    stream = _new_stream(backend)

    results: dict = {}

    def first():
        try:
            results["first"] = stream.request_cancel(CANON_TIMEOUT)
        except BaseException as exc:  # noqa: BLE001
            results["first_exc"] = exc

    t1 = threading.Thread(target=first, name="u2-c1", daemon=True)
    t1.start()
    assert entered.wait(CANON_TIMEOUT)
    # Second concurrent control must be rejected without a second writer.
    with pytest.raises(RuntimeError):
        stream.request_cancel(CANON_TIMEOUT)
    assert len(backend.cancel_calls) == 1
    release.set()
    t1.join(timeout=CANON_TIMEOUT)
    assert not t1.is_alive()
    assert "first_exc" not in results, results
    assert results["first"]["event"] == "cancelled"


def test_confirmed_receipt_is_idempotent_no_duplicate_cancel():
    backend = FakeBackend()
    backend.cancel_barrier.set()
    stream = _new_stream(backend)
    first = stream.request_cancel(CANON_TIMEOUT)
    assert len(backend.cancel_calls) == 1
    second = stream.request_cancel(CANON_TIMEOUT)
    assert second == first
    assert len(backend.cancel_calls) == 1  # NO duplicate native cancel


# ---------------------------------------------------------------------------
# 12. Ordinary regression / isolation / legacy delegate
# ---------------------------------------------------------------------------


def test_ordinary_uncancelled_path_regression():
    backend = FakeBackend()
    responses = [
        {"event": "partial", "text": "he"},
        {"event": "final", "text": "hello"},
    ]

    def chunk_handler(ev):
        return dict(responses.pop(0), id=ev["id"])

    backend.chunk_handler = chunk_handler
    # Offline rescue is a legitimate UNCANCELLED path (a 1-word final with enough
    # buffered audio): return empty so the streaming text is retained.
    backend.transcribe = lambda wav_bytes, language="auto": type(
        "R", (), {"text": "", "language": None}
    )()
    stream = _new_stream(backend)
    stream._audio_accum = np.zeros(16000, dtype=np.float32)  # buffered audio
    stream._send_chunk(last=False)  # ordinary partial
    assert stream._partial_text == "he"
    text, _lang = stream.finalize()
    assert text == "hello"
    # A successful streaming 'final' sets _closed; finalize()'s close() is then a
    # no-op (this is the pre-existing normal path — no end is sent after final).
    assert stream._closed is True
    assert stream._inflight_for(stream._session_id) == 0


def test_stream_isolation():
    backend = FakeBackend()
    s1 = _new_stream(backend)
    s2 = _new_stream(backend)
    assert s1._session_id != s2._session_id

    s1.arm_cancel()
    assert s1._cancel_armed_for(s1._session_id)
    assert not s2._cancel_armed_for(s2._session_id)
    assert s2._cancel_intent is None
    s2.accept_waveform(16000, np.zeros(320, dtype=np.float32))
    assert s2._cancel_intent is None
    assert len(backend.chunk_calls) >= 1


def test_cancel_and_finalize_delegates_to_native_cancel():
    backend = FakeBackend()
    backend.cancel_barrier.set()
    stream = _new_stream(backend)
    captured = stream._session_id
    stream.cancel_and_finalize()
    assert backend.cancel_calls == [captured]
    assert stream._cancel_confirm["sid"] == captured
    assert stream._final_text == ""
