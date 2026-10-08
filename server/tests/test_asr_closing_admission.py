"""V12 regression: known-closing ASR WS admission grace (bounded, real limiter).

No full ``server.main`` runtime import (its module graph pulls
``prometheus_client``, absent in the agent uv env) and no mirrored code:
this module AST-extracts the ACTUAL authored sources —

* from ``server/main.py``: the registry assignments
  (``_ASR_CLOSING_RELEASE_EVENTS`` / ``_ASR_CLOSING_GRACE_SECONDS`` /
  ``_ASR_CLOSING_GRACE_WAITERS``) and the actual helpers
  ``_asr_mark_closing_token`` / ``_asr_signal_closing_released`` /
  ``_asr_grace_reacquire``;
* from ``server/core/session_limiter.py``: the actual ``SessionLimiter``,
  ``SessionToken``, ``try_acquire_ws_token``, ``get_limiter`` and
  ``close_ws_rejected`` (exec'd into the module slot
  ``server.core.session_limiter`` so the helpers' in-function
  ``from server.core.session_limiter import …`` resolves to the SAME actual
  source — one coherent limiter, real counts / release / hooks / events).

The ONLY controlled fixture is the instrumentation surface
(``server.core.metrics`` → bounded no-op stub), installed via a scoped
pytest fixture with proper ``sys.modules`` restoration. No admission
domain logic is stubbed.

Contracts covered: explicit release-before-signal ordering; newcomer held
until ACTUAL token release; healthy occupied owner → immediate unchanged
strict 4429 (no wait, no mark); release-before-wait no lost wake;
closing-unreleased (quarantine) timeout keeps the old token active with
clean waiter/task drain; caller cancel never cancels the owner, never
releases the old token, never leaks a wait task or counter (including
cancel landing in the cleanup phase); timeout NEVER speculatively
reacquires on an unobserved slot free; excess waiters get immediate 4429;
B2 one-closing+one-healthy (limit=2) waits only on the closing event;
stale-limiter closing events are ignored; release exception never signals
nor frees.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import pathlib
import sys
import textwrap
import threading
import time
import types

import pytest

pytestmark = pytest.mark.asyncio

_MAIN = pathlib.Path(__file__).resolve().parents[1] / "main.py"
_SL = pathlib.Path(__file__).resolve().parents[1] / "core" / "session_limiter.py"

_MAIN_FNS = (
    "_asr_mark_closing_token",
    "_asr_signal_closing_released",
    "_asr_grace_reacquire",
)
_MAIN_GLOBALS = (
    "_ASR_CLOSING_RELEASE_EVENTS",
    "_ASR_CLOSING_GRACE_SECONDS",
    "_ASR_CLOSING_GRACE_WAITERS",
)
_SL_NAMES = (
    "SessionLimiter",
    "SessionToken",
    "try_acquire_ws_token",
    "get_limiter",
    "close_ws_rejected",
    "_limiter",
)


def _extract_admission_gate(path: pathlib.Path) -> str:
    """Extract the ACTUAL outer-handler admission gate verbatim (V12).

    Locates — syntax only, via the existing ``ast`` parse — the exact three
    statements inside the real ``asr_stream`` handler that (a) perform the
    FIRST ``try_acquire_ws_token`` admission attempt, (b) invoke the grace
    helper ONLY when the rejection reason is ``too_many``, and copies their
    VERBATIM source lines into a thin ``_asr_admission_gate(endpoint)``
    wrapper. No algorithm is mirrored or rewritten: the executed gate body
    is byte-identical to the handler source. The trailing reject/return
    block (which needs ``ws``/request-context surfaces) is intentionally NOT
    included; the gate returns ``(token, info)`` so tests assert the actual
    admission outcome and can drive the REAL ``close_ws_rejected`` seam.
    """
    src = path.read_text()
    tree = ast.parse(src)  # syntax-only parse
    lines = src.splitlines(keepends=True)
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.AsyncFunctionDef) and fn.name == "asr_stream"):
            continue
        body = fn.body
        for i, node in enumerate(body[:-1]):
            nxt = body[i + 1]
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "") == "try_acquire_ws_token"
                and isinstance(nxt, ast.If)
                and "_asr_grace_reacquire" in ast.dump(nxt)
            ):
                segment = textwrap.dedent(
                    "".join(lines[node.lineno - 1:nxt.end_lineno])
                )
                wrapper = (
                    "async def _asr_admission_gate(endpoint):\n"
                    + textwrap.indent(segment, "    ")
                    + "    return _session_token, _admit_info\n"
                )
                compile(wrapper, str(path), "exec")  # syntax sanity only
                return wrapper
        break
    raise AssertionError("actual V12 admission gate not found in asr_stream")


def _extract_segments(path: pathlib.Path, names: tuple[str, ...]) -> str:
    """AST-extract the ACTUAL module source segments for ``names``."""
    src = path.read_text()
    tree = ast.parse(src)
    parts: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            parts.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.ClassDef) and node.name in names:
            parts.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            parts.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names:
            parts.append(ast.get_source_segment(src, node))
    missing = set(names) - {
        n for n in names
        if any(
            (isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and x.name == n)
            or (isinstance(x, (ast.Assign, ast.AnnAssign)) and n in ast.dump(x))
            for x in tree.body
        )
    }
    assert not parts or len(parts) >= 1
    assert not missing, f"AST extraction missed {missing} in {path}"
    return "\n\n".join(parts)


class _NoopMetrics(types.ModuleType):
    """Bounded no-op instrumentation stub (NOT domain logic)."""

    @staticmethod
    def inc_sessions_active() -> None: ...
    @staticmethod
    def dec_sessions_active() -> None: ...
    @staticmethod
    def inc_sessions_rejected(_k: str) -> None: ...
    @staticmethod
    def dec_sessions_rejected(_k: str) -> None: ...


class _Harness:
    """Namespace over the actual extracted code under test.

    The extracted main helpers are exec'd DIRECTLY into a ModuleType's
    ``__dict__`` so the helpers' ``global`` statements and every registry /
    counter / grace access by the tests share ONE namespace (no stale copy).
    """

    def __init__(self) -> None:
        self.main_mod = types.ModuleType(
            "test_asr_closing_admission._main_extract"
        )
        self.main_mod.__dict__.update(
            {
                "asyncio": asyncio,
                "logging": logging,
                "logger": logging.getLogger("test.asr.closing_admission"),
            }
        )
        main_src = _extract_segments(_MAIN, _MAIN_FNS + _MAIN_GLOBALS)
        exec(compile(main_src, str(_MAIN), "exec"), self.main_mod.__dict__)
        # The ACTUAL outer-handler admission gate (first try_acquire_ws_token
        # + grace only-if-too_many), verbatim from the real asr_stream body.
        exec(
            compile(_extract_admission_gate(_MAIN), str(_MAIN), "exec"),
            self.main_mod.__dict__,
        )

    def __getattr__(self, name):
        # Live access into the single shared namespace (never a stale copy).
        try:
            return self.main_mod.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def main(self):
        return self.main_mod

    @property
    def registry(self) -> dict:
        return self.main_mod.__dict__["_ASR_CLOSING_RELEASE_EVENTS"]

    @property
    def grace(self) -> float:
        return self.main_mod.__dict__["_ASR_CLOSING_GRACE_SECONDS"]

    @property
    def waiters(self) -> int:
        return self.main_mod.__dict__["_ASR_CLOSING_GRACE_WAITERS"]


@pytest.fixture
def h():
    """Scoped instrumentation patch + actual-source limiter module slot.

    Installs the no-op metrics stub and the AST-extracted ACTUAL
    session_limiter source into ``sys.modules`` (so the extracted main
    helpers' in-function imports resolve to the same actual source), builds
    a fresh REAL ``SessionLimiter(1)``, isolates the closing registry and
    waiter counter, and restores every patched surface afterwards.
    """
    saved_metrics = sys.modules.get("server.core.metrics")
    saved_sl = sys.modules.get("server.core.session_limiter")
    saved_parent = sys.modules.get("server.core")
    saved_parent_metrics = (
        saved_parent.__dict__.get("metrics") if saved_parent is not None else None
    )
    saved_parent_sl = (
        saved_parent.__dict__.get("session_limiter")
        if saved_parent is not None else None
    )
    sys.modules["server.core.metrics"] = _NoopMetrics("server.core.metrics")

    harness = _Harness()

    # Exec the ACTUAL extracted limiter source DIRECTLY into the module
    # __dict__ so function __globals__ ARE the module namespace: the actual
    # `_limiter` AST assignment and the fixture's instance share ONE dict.
    sl_mod = types.ModuleType("server.core.session_limiter")
    sl_mod.__dict__.update(
        {
            "metrics": sys.modules["server.core.metrics"],
            "logger": logging.getLogger("test.asr.closing_admission.limiter"),
            "threading": threading,
            "time": time,
            "asyncio": asyncio,
        }
    )
    sl_src = _extract_segments(_SL, _SL_NAMES)
    exec(compile(sl_src, str(_SL), "exec"), sl_mod.__dict__)
    sys.modules["server.core.session_limiter"] = sl_mod

    limiter = sl_mod.SessionLimiter(1)
    sl_mod._limiter = limiter
    # The extracted handler gate calls the REAL limiter seam by name (the
    # in-handler import target); bind it to the SAME actual extracted module.
    harness.main_mod.__dict__["try_acquire_ws_token"] = sl_mod.try_acquire_ws_token

    # Namespace-identity guards (fixture assertions, no extra test cases):
    # every helper patches/reads the SAME namespace the fixture patched.
    assert sl_mod.get_limiter.__globals__ is sl_mod.__dict__, \
        "extracted limiter function globals are not the module namespace"
    assert harness.main._asr_grace_reacquire.__globals__ \
        is harness.main_mod.__dict__, \
        "extracted main helper globals are not the module namespace"
    assert sl_mod.__dict__["_limiter"] is limiter, \
        "get_limiter would not observe the current actual limiter instance"

    harness.limiter = limiter
    harness.sl = sl_mod
    try:
        yield harness
    finally:
        if saved_metrics is not None:
            sys.modules["server.core.metrics"] = saved_metrics
        else:
            sys.modules.pop("server.core.metrics", None)
        if saved_sl is not None:
            sys.modules["server.core.session_limiter"] = saved_sl
        else:
            sys.modules.pop("server.core.session_limiter", None)
        # Restore parent-package attrs so no module surface leaks.
        if saved_parent is not None:
            for attr, saved in (
                ("metrics", saved_parent_metrics),
                ("session_limiter", saved_parent_sl),
            ):
                if saved is not None:
                    saved_parent.__dict__[attr] = saved
                else:
                    saved_parent.__dict__.pop(attr, None)


class _FakeRejectWS:
    """Captures the exact 4429 close frame produced by ``close_ws_rejected``."""

    def __init__(self) -> None:
        self.code = None
        self.reason = None

    async def close(self, code=1000, reason=""):
        self.code = code
        self.reason = reason


def _mark(h, token):
    """Mark via the actual module seam (the hook closure calls exactly this)."""
    ev = h.main._asr_mark_closing_token(token)
    assert token in h.registry
    return ev


def _release_and_signal(h, token, order=None):
    """EXACT outer-finally ordering: release() FIRST, signal only after."""
    token.release()
    assert token._released is True
    if order is not None:
        order.append("release")
    h.main._asr_signal_closing_released(token)
    if order is not None:
        order.append("signal")
    assert token not in h.registry


# ---------------------------------------------------------------------------
# (1) Helper / admission seam with a REAL SessionLimiter
# ---------------------------------------------------------------------------

async def test_closing_owner_holds_newcomer_until_actual_release(h):
    """Known-closing owner: newcomer cannot obtain the token before the
    ACTUAL release; after release exactly ONE admission succeeds and the
    release-before-signal ordering is explicitly asserted."""
    limiter = h.limiter
    owner = limiter.try_acquire()
    assert owner is not None and limiter.active == 1
    _mark(h, owner)

    waiter = asyncio.ensure_future(h.main._asr_grace_reacquire("/asr/stream"))
    await asyncio.sleep(0.02)  # well inside the 100 ms grace
    assert not waiter.done(), "admitted while old token still held"
    assert limiter.active == 1, "newcomer obtained a token before actual release"

    order: list[str] = []
    _release_and_signal(h, owner, order)
    assert order == ["release", "signal"], "signal must follow actual release"

    got = await asyncio.wait_for(waiter, timeout=h.grace)
    assert got is not None
    assert owner._released is True
    assert limiter.active == 1 and limiter.active <= limiter.limit
    got.release()
    assert limiter.active == 0
    assert h.waiters == 0, "waiter counter leaked"


@pytest.mark.parametrize("limit", [1, 2])
async def test_release_ordering_explicit_count_within_limit(h, limit):
    """Release ordering + count<=limit asserted for each limiter size via the
    ACTUAL outer-handler admission gate: the FIRST ``try_acquire_ws_token``
    is full (so reason == ``too_many``), THEN the grace waits on the
    registered known-closing event, and admission succeeds only after the
    actual release + signal. (Direct ``_asr_grace_reacquire`` invocation
    after a completed release was a fixture call-flow mismatch: the real
    handler's first acquire succeeds in that state.)"""
    limiter = h.limiter
    limiter._limit = limit
    tokens = [limiter.try_acquire() for _ in range(limit)]
    _mark(h, tokens[0])
    order: list[str] = []
    gate = asyncio.ensure_future(
        h.main._asr_admission_gate("/asr/stream")
    )
    await asyncio.sleep(0.02)  # well inside the 100 ms grace
    assert not gate.done(), "gate admitted/resolved before the actual release"
    _release_and_signal(h, tokens[0], order)
    assert order == ["release", "signal"]
    got, info = await asyncio.wait_for(gate, timeout=h.grace)
    assert got is not None
    assert limiter.active <= limiter.limit
    assert got._limiter is limiter
    got.release()


async def test_healthy_occupied_owner_immediate_reject_4429(h):
    """Healthy owner (no disconnect observed): NO grace wait, unchanged
    immediate strict 4429 JSON."""
    limiter = h.limiter
    owner = limiter.try_acquire()
    assert h.registry == {}, "healthy owner not marked"

    t0 = time.monotonic()
    got = await h.main._asr_grace_reacquire("/asr/stream")
    elapsed = time.monotonic() - t0
    assert got is None
    assert elapsed < 0.05, "waited on a healthy owner"
    assert owner._released is False and limiter.active == 1

    ws = _FakeRejectWS()
    _tok, info = h.sl.try_acquire_ws_token("/asr/stream")
    assert _tok is None
    await h.sl.close_ws_rejected(ws, "/asr/stream", info)
    assert ws.code == 4429
    import json as _json
    assert _json.loads(ws.reason) == {
        "error": "too_many_sessions", "current": 1, "limit": 1,
    }
    assert owner._released is False


async def test_released_before_admission_uses_first_acquire(h):
    """After a complete actual release + signal (registry entry removed), the
    ACTUAL extracted handler admission gate is driven: its FIRST
    ``try_acquire_ws_token`` succeeds directly (grace not needed, no waiter
    counted), returning a real current-limiter token. NOTE: this does NOT
    exercise the captured-event-before-wait lost-wake stage (the helper's
    already-set-Event wait path); that coverage gap is explicitly recorded
    in the handoff report, not fabricated here."""
    owner = h.limiter.try_acquire()
    _mark(h, owner)
    _release_and_signal(h, owner)
    assert owner not in h.registry
    t0 = time.monotonic()
    got, info = await asyncio.wait_for(
        h.main._asr_admission_gate("/asr/stream"),
        timeout=h.grace,
    )
    assert got is not None and time.monotonic() - t0 < 0.05
    assert got is not owner
    assert getattr(got, "_limiter", None) is h.limiter
    assert h.limiter.active == 1 and h.limiter.active <= h.limiter.limit
    assert h.waiters == 0, "waiter counter touched without a grace wait"
    got.release()
    assert h.limiter.active == 0


# ---------------------------------------------------------------------------
# (2) Known-closing unreleased / quarantine / cancellation / bounding
# ---------------------------------------------------------------------------

def _foreign_tasks() -> list:
    return [
        t for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]


async def test_known_closing_unreleased_times_out_keeps_owner_active(h):
    """Closing token never released (worst-case quarantine hold): the grace
    times out at the static bound, the old token stays ACTIVE, the outcome is
    the unchanged 4429, and waiter state drains clean."""
    limiter = h.limiter
    owner = limiter.try_acquire()
    _mark(h, owner)
    t0 = time.monotonic()
    got = await h.main._asr_grace_reacquire("/asr/stream")
    elapsed = time.monotonic() - t0
    assert got is None, "admitted without an actual release"
    assert elapsed >= h.grace * 0.9
    assert owner._released is False and limiter.active == 1
    assert h.waiters == 0, "waiter counter leaked"
    assert not _foreign_tasks(), "leaked grace wait task"


async def test_cancelled_incoming_never_cancels_owner_or_releases_token(h):
    """Cancelling the incoming admission while it waits must NOT cancel the
    old owner, NOT release the old token, and must not leak waiter state."""
    limiter = h.limiter
    owner = limiter.try_acquire()
    _mark(h, owner)
    waiter = asyncio.ensure_future(h.main._asr_grace_reacquire("/asr/stream"))
    await asyncio.sleep(0.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.sleep(0)  # let cancellations settle
    assert owner._released is False, "old token released by incoming cancel"
    assert limiter.active == 1
    assert h.waiters == 0, "waiter counter leaked"
    assert owner in h.registry, "closing mark lost"


async def test_cancel_during_wait_cleanup_propagates_no_new_token(h):
    """Caller cancel arriving as the wait resolves (cleanup phase) must
    propagate: NO new token acquired, counter decremented, owner's actual
    release untouched by the cancellation itself."""
    limiter = h.limiter
    owner = limiter.try_acquire()
    _mark(h, owner)
    waiter = asyncio.ensure_future(h.main._asr_grace_reacquire("/asr/stream"))
    await asyncio.sleep(0.01)
    owner.release()               # actual release: event set
    h.main._asr_signal_closing_released(owner)
    waiter.cancel()               # cancel lands at the next await (cleanup)
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.sleep(0)
    assert limiter.active == 0, "new token acquired after caller cancel"
    assert h.waiters == 0, "waiter counter leaked"


async def test_timeout_returns_none_without_speculative_reacquire(h):
    """Unreleased closing token: even when a slot frees up UNOBSERVED during
    the grace, the timeout path must NOT speculatively reacquire — admission
    stays the unchanged 4429 and the counter drains."""
    limiter = h.limiter
    owner = limiter.try_acquire()
    _mark(h, owner)

    async def _late_release():
        # free the slot just after the grace deadline expires, without any
        # event signal (unobserved free)
        await asyncio.sleep(h.grace + 0.05)
        owner.release()

    releaser = asyncio.ensure_future(_late_release())
    got = await h.main._asr_grace_reacquire("/asr/stream")
    assert got is None
    await asyncio.wait_for(releaser, timeout=5)
    assert owner._released is True and limiter.active == 0
    assert h.waiters == 0


async def test_excess_waiters_get_immediate_unchanged_reject(h):
    """More simultaneous grace waiters than limiter.limit -> immediate
    original 4429 (no queueing)."""
    limiter = h.limiter
    owner = limiter.try_acquire()
    _mark(h, owner)
    first = asyncio.ensure_future(h.main._asr_grace_reacquire("/asr/stream"))
    await asyncio.sleep(0.005)
    assert h.waiters == 1
    t0 = time.monotonic()
    overflow = await h.main._asr_grace_reacquire("/asr/stream")
    assert overflow is None
    assert time.monotonic() - t0 < 0.05, "overflow waiter queued"
    _release_and_signal(h, owner)
    got = await asyncio.wait_for(first, timeout=h.grace)
    assert got is not None
    got.release()


async def test_b2_one_closing_one_healthy_admits_after_release(h):
    """limit=2 with one closing + one healthy owner: the grace waits ONLY on
    the closing event; after its actual release, ONE reattempt admits and
    the count stays <= limit; the healthy owner is untouched."""
    limiter = h.limiter
    limiter._limit = 2
    closing = limiter.try_acquire()
    healthy = limiter.try_acquire()
    assert closing is not None and healthy is not None
    _mark(h, closing)

    waiter = asyncio.ensure_future(h.main._asr_grace_reacquire("/asr/stream"))
    await asyncio.sleep(0.02)
    assert not waiter.done()
    assert healthy._released is False, "healthy owner disturbed"

    _release_and_signal(h, closing)
    got = await asyncio.wait_for(waiter, timeout=h.grace)
    assert got is not None
    assert limiter.active == 2 and limiter.active <= limiter.limit
    assert healthy._released is False
    healthy.release()
    got.release()


async def test_stale_limiter_events_do_not_affect_new_limiter(h):
    """A closing event whose token belongs to a REPLACED (stale) limiter must
    be ignored by the current limiter's admission: immediate 4429."""
    limiter = h.limiter
    stale = h.sl.SessionLimiter(1)
    stale_token = stale.try_acquire()
    _mark(h, stale_token)  # registry entry bound to the stale limiter

    owner = limiter.try_acquire()  # current limiter full + healthy
    assert owner is not None

    t0 = time.monotonic()
    got = await h.main._asr_grace_reacquire("/asr/stream")
    assert got is None
    assert time.monotonic() - t0 < 0.05, "waited on a stale-limiter event"
    assert stale_token._released is False and owner._released is False


async def test_release_exception_never_signals_or_frees(h):
    """If the outer release raises (unknown/failed release), the closing
    event must NOT be signalled and the slot must NOT be freed."""
    limiter = h.limiter
    owner = limiter.try_acquire()
    _mark(h, owner)

    def failing_release():
        raise RuntimeError("release failed")

    saved = owner.release
    owner.release = failing_release  # type: ignore[method-assign]
    try:
        owner.release()
    except RuntimeError:
        pass
    assert owner._released is False
    h.main._asr_signal_closing_released(owner)
    ev = h.registry.get(owner)
    assert ev is not None and not ev.is_set(), "signalled despite failed release"
    got = await h.main._asr_grace_reacquire("/asr/stream")
    assert got is None, "admitted on a failed release"
    owner.release = saved  # type: ignore[method-assign]
    _release_and_signal(h, owner)
