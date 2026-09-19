import os
import sys

# Make project root (parent of app/) importable so `from server.core.* import`
# resolves the same way as in production (uvicorn server.main:app).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import copy

import pytest


@pytest.fixture(autouse=True)
def _restore_profile_loader_globals():
    """Put ``profile_loader``'s module state back after every test.

    ``apply_profile`` writes three module-level globals, and a test that loads a
    profile leaves them set for everything that runs afterwards — including
    tests in the sibling ``tests/`` directory, which pytest collects after this
    one. Measured 2026-09-18: ``test_profile_plugin_ownership.py`` left
    ``_CURRENT_PROFILE = jetson-edgellm-v091-moss`` behind, whose executor
    ceiling of 2 then clamped ``tests/test_tts_stream_executor_resolve.py``'s
    expected 3 and failed six tests in two files — but only in a combined run,
    which is why running either directory alone looked clean.

    Snapshot-and-restore rather than clear-to-empty: a module- or session-scoped
    fixture that applies a profile once for a whole file is captured in the
    snapshot taken before each of that file's tests, so it survives.
    """
    from server.core import profile_loader as _pl

    saved = (
        copy.deepcopy(_pl._CURRENT_PROFILE),
        set(_pl._APPLIED_KEYS),
        set(_pl._OWNED_OVERRIDES),
    )
    try:
        yield
    finally:
        _pl._CURRENT_PROFILE = saved[0]
        _pl._APPLIED_KEYS.clear()
        _pl._APPLIED_KEYS.update(saved[1])
        _pl._OWNED_OVERRIDES.clear()
        _pl._OWNED_OVERRIDES.update(saved[2])
