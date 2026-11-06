"""Regression tests for the /tts/stream sentence prefetch window.

The window decides when sentence i+1 may be handed to the TTS executor while
sentence i is still draining. It was written as

    (next_to_submit - current_idx) < prefetch_max

which is off by one: sentences 0..next_to_submit-1 are already submitted, so
the count in flight *ahead* of the current one is next_to_submit-1-current_idx.
With prefetch_max=1 -- the value on every RK device, where the TTS executor has
a single worker -- the old form evaluated 1 < 1 and never submitted anything
after sentence 0. The drain loop then awaited a queue that nothing would ever
fill and /tts/stream deadlocked on *any* multi-sentence input, holding the
single session slot until the client timed out.

Measured on RK3588 before the fix: "Hello there. How are you?" (two short
sentences, 1.4 s of audio) hung after emitting only sentence 0.
"""

from __future__ import annotations

import pytest

from server.main import _prefetch_window_allows


def _simulate(n_sentences: int, prefetch_max: int) -> list[int]:
    """Replay the handler's submit schedule; return the submitted indices.

    Mirrors the real loop: submit 0 up front, then call the window check once
    when the current sentence emits its first chunk and once when it finishes.
    """
    submitted = [0]
    next_to_submit = 1

    def maybe(current_idx: int) -> None:
        nonlocal next_to_submit
        if next_to_submit < n_sentences and _prefetch_window_allows(
            next_to_submit, current_idx, prefetch_max
        ):
            submitted.append(next_to_submit)
            next_to_submit += 1

    for current_idx in range(n_sentences):
        if current_idx not in submitted:
            # The drain loop would block here forever.
            raise AssertionError(
                f"sentence {current_idx} awaited but never submitted "
                f"(prefetch_max={prefetch_max}, submitted={submitted})"
            )
        maybe(current_idx)  # first chunk of the current sentence
        maybe(current_idx)  # sentence finished
    return submitted


@pytest.mark.parametrize("n_sentences", [2, 3, 5, 12])
def test_every_sentence_is_submitted_with_a_single_worker(n_sentences):
    """prefetch_max=1 is the RK case that deadlocked."""
    assert _simulate(n_sentences, prefetch_max=1) == list(range(n_sentences))


@pytest.mark.parametrize("prefetch_max", [1, 2, 3, 8])
@pytest.mark.parametrize("n_sentences", [1, 2, 4, 9])
def test_every_sentence_is_submitted_for_any_window(n_sentences, prefetch_max):
    assert _simulate(n_sentences, prefetch_max) == list(range(n_sentences))


def test_window_stays_bounded():
    """The point of the window is to cap work in flight, not to remove it.

    With prefetch_max=1 exactly one sentence may run ahead of the one being
    drained -- that is what keeps sentence 0's TTFA at the single-sentence
    baseline while overlapping sentence 1's prefill.
    """
    assert _prefetch_window_allows(1, 0, 1) is True     # submit sentence 1
    assert _prefetch_window_allows(2, 0, 1) is False    # not sentence 2 as well
    assert _prefetch_window_allows(2, 1, 1) is True     # once 1 is current

    assert _prefetch_window_allows(2, 0, 2) is True     # window of 2 allows both
    assert _prefetch_window_allows(3, 0, 2) is False


def test_old_condition_would_have_deadlocked():
    """Witness for the regression, so this cannot quietly come back."""
    next_to_submit, current_idx, prefetch_max = 1, 0, 1
    old = (next_to_submit - current_idx) < prefetch_max
    new = _prefetch_window_allows(next_to_submit, current_idx, prefetch_max)
    assert old is False and new is True
