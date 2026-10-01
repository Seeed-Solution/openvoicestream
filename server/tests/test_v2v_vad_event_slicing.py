"""Focused tests for V2V ordered VAD block slicing and pending safeguards."""

import numpy as np


def test_v2v_split_vad_block_is_ordered_and_non_overlapping():
    from server.main import _split_vad_block

    samples = np.arange(10, dtype=np.float32)
    parts = _split_vad_block(
        samples,
        [("speech_end", 4), ("speech_start", 7)],
    )

    assert [(len(part), event) for part, event in parts] == [
        (4, "speech_end"),
        (3, "speech_start"),
        (3, None),
    ]
    np.testing.assert_array_equal(
        np.concatenate([part for part, _event in parts]), samples
    )


def test_v2v_split_vad_block_does_not_add_trailing_empty_segment():
    from server.main import _split_vad_block

    parts = _split_vad_block(
        np.arange(4, dtype=np.float32),
        [("speech_end", 4)],
    )

    assert [(len(part), event) for part, event in parts] == [(4, "speech_end")]


def test_v2v_source_has_bounded_pending_and_bounded_finalize():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()

    assert '"pending_turns": []' in src
    assert '"pending_limit_samples"' in src
    assert "_reject_pending_audio" in src
    assert "await asyncio.wait_for(" in src
    assert "asr_manager.finalize_with_status(" in src
