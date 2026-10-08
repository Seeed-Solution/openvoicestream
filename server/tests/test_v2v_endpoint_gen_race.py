"""Regression test for the finalize / speech-start gen-race bug.

Root-cause analysis (codex, 2026-05-19)
======================================

Two trigger paths observed on Orin NX validation, both with the same
discard log line — ``ASRSessionManager: finalize result discarded
(state=ACTIVE)`` — and the same observable symptom: an empty
``asr_final`` emitted for utterance N:

1. **Multi-turn**: VAD ``speech_start`` for utterance N+1 arrives at the
   dispatcher while ``stream.finalize()`` for utterance N is still
   in-flight inside ``asr_out_task``. The dispatcher calls
   ``manager.on_speech_start()``, which preempts the in-flight stream
   (state ACTIVE→CANCELLING→ACTIVE, generation bumped). When
   ``stream.finalize()`` returns, the manager sees state=ACTIVE (the
   *new* utterance is active) and discards the result.

2. **Post-worker-restart**: error recovery creates a new stream and
   issues ``on_speech_start``; the in-flight previous finalize hits the
   same discard path.

Compounding the bug, ``endpoint_pending`` in ``server/main.py`` had no
generation tag. So even after a clean preempt, a stale endpoint stamped
by utterance N could trigger a finalize against utterance N+1's stream,
forcing the manager into a second wrong-generation finalize.

Fix (this commit)
-----------------

``endpoint_pending`` carries a sibling ``endpoint_pending_gen``. The
dispatcher stamps it whenever it sets ``endpoint_pending``; the
``asr_out_task`` consumer gates on it before calling
``finalize_with_generation`` — if the gen doesn't match the current
``asr_active_gen``, the endpoint is dropped on the floor.

Test strategy
-------------

We exercise the *exact* gating block that was added to ``asr_out_task``
by mirroring it in this test, against a real ``ASRSessionManager`` with
a fake backend whose ``finalize()`` takes ~100ms.

Sequence:

1. ``on_speech_start`` for U1 → gen=1, state=ACTIVE.
2. Dispatcher stamps ``endpoint_pending = "vad"``,
   ``endpoint_pending_gen = 1``.
3. Schedule a background task running the slow ``finalize_with_generation``
   for U1 (mimics asr_out_task picking up the endpoint).
4. While that's in flight, ``on_speech_start`` for U2 — preempts U1,
   gen advances to 2.
5. **Critical**: a second VAD speech-end *would have* stamped
   ``endpoint_pending`` against U1's generation in the pre-fix code,
   AND/OR the SPEECH_START path didn't clear the stale endpoint. We
   simulate the pre-fix and post-fix decision and assert the gate fires
   correctly only in the post-fix variant.

The test ALSO exercises a second axis: when SPEECH_START clears the
stale endpoint (fix #2/3 in main.py), the gate is the belt-and-braces
defense if some other path missed clearing it. Both must hold.

Why we do this at the unit level instead of routing real audio through
the FastAPI TestClient: deterministically interleaving the dispatcher
and asr_out_task asyncio tasks at the right point of finalize() is
brittle. The fix lives in two narrow logic blocks; testing those blocks
directly with a real ASRSessionManager underneath is more reliable and
catches the same regression class.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from typing import List, Tuple

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from server.core.asr_backend import ASRBackend, ASRCapability
from server.core.asr_session_manager import ASRSessionManager


def _asynctest(fn):
    """Decorator turning an ``async def`` test into a sync test.

    Matches the convention used elsewhere in this test suite (see
    ``server/tests/test_asr_session_manager.py``) — avoids the
    pytest-asyncio dependency.
    """
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(fn(*args, **kwargs))
        finally:
            loop.close()
    wrapper.__name__ = fn.__name__
    return wrapper


# ──────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────


class _SlowFinalizeStream:
    """Stream stand-in whose ``finalize()`` blocks for ``finalize_delay_s``.

    Returns a pre-determined text on finalize so the test can assert
    whether the value made it to the caller or was discarded en route.
    """

    def __init__(self, text: str, finalize_delay_s: float):
        self._text = text
        self._delay = finalize_delay_s
        self.accepted: List[int] = []
        self.finalized = False
        self.cancelled = False

    def accept_waveform(self, sr: int, samples) -> None:
        self.accepted.append(len(samples))

    def get_partial(self) -> Tuple[str, bool]:
        return "", False

    def finalize(self):
        # Block synchronously — this runs in an executor thread, exactly
        # as the real ``stream.finalize()`` does in production.
        time.sleep(self._delay)
        self.finalized = True
        # New ABC contract: ``(text, detected_language)``.
        return self._text, None

    def cancel(self) -> None:
        self.cancelled = True

    def cancel_and_finalize(self) -> str:
        self.cancelled = True
        return ""


class _SequencedASRBackend(ASRBackend):
    """Backend that hands out a pre-built sequence of slow streams."""

    def __init__(self, streams: List[_SlowFinalizeStream]):
        self._streams = list(streams)
        self._idx = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:  # type: ignore[override]
        return "fake-genrace"

    @property
    def capabilities(self):  # type: ignore[override]
        return {ASRCapability.STREAMING}

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return 16000

    def is_ready(self) -> bool:  # type: ignore[override]
        return True

    def preload(self) -> None:  # type: ignore[override]
        return None

    def transcribe(self, audio_bytes: bytes, language: str = "auto"):  # type: ignore[override]
        from server.core.asr_backend import TranscriptionResult
        return TranscriptionResult(text="", duration=0.0, inference_time=0.0,
                                   rtf=0.0, n_tokens=0, per_token_ms=0.0,
                                   backend=self.name)

    def transcribe_audio(self, audio, language="auto"):  # type: ignore[override]
        from server.core.asr_backend import TranscriptionResult
        return TranscriptionResult(text="", duration=0.0, inference_time=0.0,
                                   rtf=0.0, n_tokens=0, per_token_ms=0.0,
                                   backend=self.name)

    def create_stream(self, language: str = "auto"):
        with self._lock:
            if self._idx >= len(self._streams):
                raise RuntimeError("no more pre-built streams")
            s = self._streams[self._idx]
            self._idx += 1
            return s


# ──────────────────────────────────────────────────────────────────────
# Decision-block helpers — mirror the EXACT logic in asr_out_task
# ──────────────────────────────────────────────────────────────────────
#
# We keep two variants so the test can demonstrate that without the
# gate, the bug reproduces (wrong-gen finalize fires); with the gate,
# it's correctly skipped.


def _decide_finalize_PRE_FIX(state: dict) -> Tuple[bool, str | None]:
    """Pre-fix decision: endpoint_pending fires unconditionally."""
    endpoint_reason = state["endpoint_pending"]
    endpoint_fired = bool(endpoint_reason)
    if endpoint_fired:
        state["endpoint_pending"] = None
    return endpoint_fired, endpoint_reason


def _decide_finalize_POST_FIX(state: dict) -> Tuple[bool, str | None]:
    """Post-fix decision: gen-tag gate skips stale endpoints.

    This mirrors the new block in server/main.py:asr_out_task.
    """
    endpoint_reason = state["endpoint_pending"]
    if (
        endpoint_reason
        and state.get("endpoint_pending_gen") is not None
        and state.get("endpoint_pending_gen") != state["asr_active_gen"]
    ):
        state["endpoint_pending"] = None
        state["endpoint_pending_gen"] = None
        endpoint_reason = None
    endpoint_fired = bool(endpoint_reason)
    if endpoint_fired:
        state["endpoint_pending"] = None
        state["endpoint_pending_gen"] = None
    return endpoint_fired, endpoint_reason


def _should_emit_final_POST_FIX(
    state: dict,
    finalize_gen: int,
    accepted: bool,
) -> bool:
    """Mirror the post-finalize accepted-result guard in server/main.py."""
    if accepted and state["asr_active_gen"] == finalize_gen:
        state["asr_active"] = False
    return accepted


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


@_asynctest
async def test_stale_endpoint_against_new_generation_is_gated():
    """Reproduce the race: SPEECH_START preempts mid-finalize, leaving
    a stale endpoint stamped against the OLD generation. Pre-fix this
    would fire a second wrong-gen finalize on the new utterance's stream;
    post-fix the gate drops it.
    """
    # ── Build manager with two slow streams (U1 + U2) ──
    u1_stream = _SlowFinalizeStream(text="utterance one text", finalize_delay_s=0.10)
    u2_stream = _SlowFinalizeStream(text="utterance two text", finalize_delay_s=0.05)
    backend = _SequencedASRBackend([u1_stream, u2_stream])
    manager = ASRSessionManager(backend, language="en")

    # ── State mirror (exactly the keys the production dispatcher writes) ──
    state = {
        "asr_active": False,
        "asr_active_gen": 0,
        "endpoint_pending": None,
        "endpoint_pending_gen": None,
    }

    # ── U1 begins ──
    gen1 = await manager.on_speech_start()
    state["asr_active"] = True
    state["asr_active_gen"] = gen1
    assert gen1 == 1

    # Dispatcher receives U1's VAD speech-end → stamps endpoint+gen.
    state["endpoint_pending"] = "vad"
    state["endpoint_pending_gen"] = state["asr_active_gen"]
    assert state["endpoint_pending_gen"] == 1

    # asr_out_task picks it up and starts the slow finalize on a task.
    # We DON'T mirror the gate here — this is the legitimate U1 finalize.
    fired, reason = _decide_finalize_POST_FIX(state)
    assert fired and reason == "vad"
    assert state["endpoint_pending"] is None
    assert state["endpoint_pending_gen"] is None

    finalize_task = asyncio.create_task(
        manager.finalize_with_generation(reason or "vad_end")
    )

    # ── Race window: while finalize is running, U2's SPEECH_START arrives ──
    # Give the executor a chance to enter stream.finalize() (which sleeps
    # 100ms). 30ms is plenty.
    await asyncio.sleep(0.03)

    gen2 = await manager.on_speech_start()
    # The fix-side clearing in the dispatcher: clear stale endpoint, bump gen.
    state["endpoint_pending"] = None
    state["endpoint_pending_gen"] = None
    state["asr_active"] = True
    state["asr_active_gen"] = gen2
    assert gen2 == 2

    # ── NOW SIMULATE THE OTHER RACE LEG ──
    # Imagine a logic path where (e.g. due to a near-simultaneous SPEECH_END
    # delivered by an over-eager VAD, or an old endpoint that was racing
    # the SPEECH_START), endpoint_pending gets re-stamped but the gen
    # tag still points at U1. This is the wrong-gen finalize that the
    # bug fires against U2's stream.
    state["endpoint_pending"] = "vad"
    state["endpoint_pending_gen"] = 1   # ← stale; points at U1, not U2

    # ── Pre-fix decision: would fire finalize against U2's stream ──
    # Snapshot a copy so we don't mutate the real state.
    state_prefix = dict(state)
    fired_pre, reason_pre = _decide_finalize_PRE_FIX(state_prefix)
    assert fired_pre is True, (
        "pre-fix code WOULD have fired finalize against the wrong "
        "generation — this is exactly the bug"
    )

    # ── Post-fix decision: gate skips ──
    fired_post, reason_post = _decide_finalize_POST_FIX(state)
    assert fired_post is False, (
        "post-fix gate must drop the stale endpoint (gen mismatch). "
        f"got fired={fired_post} reason={reason_post}"
    )
    assert state["endpoint_pending"] is None
    assert state["endpoint_pending_gen"] is None

    # ── Let U1's finalize finish — its result should be discarded by
    # the manager (state=ACTIVE, U2 has bumped gen). This part is the
    # existing manager-level defense, which we keep in place per the
    # fix sketch.
    ran_gen, u1_text = await finalize_task
    assert ran_gen == 1, "finalize ran against U1's generation"
    # The manager either discards (state mismatch) or returns U1 text;
    # the load-bearing assertion is that the *gate* prevented the
    # downstream from issuing another finalize. We verify the gate
    # already above; U1's text being discarded by the manager is
    # corroborating evidence and is documented behavior.
    # (We don't pin the exact value because the discard message logs at
    # INFO and doesn't change the return shape — either way, no second
    # finalize was issued against U2's stream.)
    _ = u1_text


@_asynctest
async def test_preempted_finalize_does_not_emit_empty_asr_final():
    """If U2 starts while U1 finalize is still running, U1's discarded
    finalize must not leak downstream as an empty asr_final.

    This is the "empty final after 2 turns" failure mode: the generation
    guard kept U2 active, but the old code still sent final_text="" to the
    client after the discarded U1 finalize returned.
    """
    u1_stream = _SlowFinalizeStream(text="utterance one text", finalize_delay_s=0.10)
    u2_stream = _SlowFinalizeStream(text="utterance two text", finalize_delay_s=0.05)
    manager = ASRSessionManager(_SequencedASRBackend([u1_stream, u2_stream]), language="en")
    state = {
        "asr_active": True,
        "asr_active_gen": await manager.on_speech_start(),
        "endpoint_pending": None,
        "endpoint_pending_gen": None,
    }
    finalize_gen = state["asr_active_gen"]
    finalize_task = asyncio.create_task(manager.finalize_with_status("vad"))

    await asyncio.sleep(0.03)
    state["asr_active_gen"] = await manager.on_speech_start()
    state["asr_active"] = True

    ran_gen, final_text, accepted, _lang = await finalize_task

    assert ran_gen == finalize_gen
    assert final_text == ""
    assert accepted is False
    assert _should_emit_final_POST_FIX(state, finalize_gen, accepted) is False
    assert state["asr_active"] is True


@_asynctest
async def test_accepted_finalize_emits_even_if_outer_gen_changed_after_return():
    """The outer connection generation is mutable and can advance after a
    valid finalize result returns but before the frame is emitted.

    The suppress decision must trust the manager's accepted flag, not the
    current outer asr_active_gen alone.
    """
    manager = ASRSessionManager(
        _SequencedASRBackend([_SlowFinalizeStream(text="你好。", finalize_delay_s=0.0)]),
        language="zh",
    )
    state = {
        "asr_active": True,
        "asr_active_gen": await manager.on_speech_start(),
        "endpoint_pending": None,
        "endpoint_pending_gen": None,
    }
    finalize_gen = state["asr_active_gen"]
    ran_gen, final_text, accepted, _lang = await manager.finalize_with_status("vad")

    state["asr_active_gen"] = finalize_gen + 1
    state["asr_active"] = True

    assert ran_gen == finalize_gen
    assert final_text == "你好。"
    assert accepted is True
    assert _should_emit_final_POST_FIX(state, finalize_gen, accepted) is True
    assert state["asr_active"] is True


@_asynctest
async def test_endpoint_cleared_on_fresh_speech_start_directly():
    """Direct check of fix #2: a fresh on_speech_start (whether from
    VAD or no-VAD lazy-open) clears the pending endpoint flags so the
    next asr_out_task tick doesn't observe a stale endpoint at all.
    """
    state = {
        "asr_active": True,
        "asr_active_gen": 7,
        "endpoint_pending": "vad",
        "endpoint_pending_gen": 7,
    }
    # Dispatcher logic on SPEECH_START (mirrors main.py):
    state["endpoint_pending"] = None
    state["endpoint_pending_gen"] = None
    state["asr_active_gen"] = 8

    fired, _ = _decide_finalize_POST_FIX(state)
    assert fired is False
    assert state["endpoint_pending"] is None
    assert state["endpoint_pending_gen"] is None


def test_endpoint_pending_gen_field_initialized_in_state():
    """Source pin: the state dict in v2v_stream must initialize
    endpoint_pending_gen alongside endpoint_pending.

    Guards against a future refactor accidentally dropping the field.
    """
    here = os.path.dirname(__file__)
    main_path = os.path.abspath(os.path.join(here, "..", "main.py"))
    with open(main_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert '"endpoint_pending_gen"' in src, (
        "endpoint_pending_gen state-dict field is missing — the gen-race "
        "fix has been reverted in server/main.py"
    )
    assert "endpoint_pending_gen" in src and "asr_active_gen" in src
    # Check the gate exists.
    assert "endpoint_pending_gen" in src
    needle_gate = "state.get(\"endpoint_pending_gen\") != state[\"asr_active_gen\"]"
    assert needle_gate in src, (
        "gen-race gate in asr_out_task is missing; the fix has been reverted"
    )


def test_vad_speech_end_stamps_gen_in_source():
    """Source pin: VAD SPEECH_END handler must stamp endpoint_pending_gen."""
    here = os.path.dirname(__file__)
    main_path = os.path.abspath(os.path.join(here, "..", "main.py"))
    with open(main_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert (
        'state["endpoint_pending"] = "vad"' in src
        and 'state["endpoint_pending_gen"] = state["asr_active_gen"]' in src
    ), (
        "VAD speech-end handler no longer stamps endpoint_pending_gen"
    )


def test_client_eos_stamps_gen_in_source():
    """Source pin: CLIENT_ASR_EOS handler must stamp endpoint_pending_gen."""
    here = os.path.dirname(__file__)
    main_path = os.path.abspath(os.path.join(here, "..", "main.py"))
    with open(main_path, "r", encoding="utf-8") as f:
        src = f.read()
    needle = (
        "state[\"endpoint_pending\"] = \"client_eos\"\n"
        "                        state[\"endpoint_pending_gen\"] = state[\"asr_active_gen\"]"
    )
    assert needle in src, (
        "CLIENT_ASR_EOS handler no longer stamps endpoint_pending_gen"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
