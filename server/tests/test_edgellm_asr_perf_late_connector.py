"""AUTHORED-UNRUN late-connector ownership tests for
bench/perf/edgellm_asr_ws_perf_gate.py ``_connect_owned``.

Scope (root takeback task): three SOURCE-EVIDENCED defects in the owned
connector, never addressed by the CLI/schema repairs:

  A. The external-cancel path used to re-raise BEFORE the late-ownership
     done-callback was attached: a cancellation-suppressing connector that
     late-completed with a live resource after the caller left leaked it
     (never tracked, never closed, never ledgered).
  B. The late close task was scheduled with NO remaining-cleanup-deadline
     guard, potentially creating a new async task after the caller's whole
     deadline.
  C. A synchronous ``late_close`` returning None (the HTTP writer without a
     supported ``wait_closed``) was ledgered closed=True — invented cleanup
     proof; the unknown-observability case must stay UNPROVEN.

Plus the cross-cutting contracts: the ownership callback is registered
immediately after connector-task creation BEFORE any wait; cancellation is
requested EXACTLY ONCE across timeout/external-cancel; a normal (claimed)
connector is never early-closed or double-owned.

These tests drive the ACTUAL driver functions (``_connect_owned``,
``_late_http_close``) and the ACTUAL ``AsyncLifetimeRegistry`` on ONE
owned event loop per test (``new_event_loop`` + ``run_until_complete``,
no ``asyncio.run`` Runner cancel-gather). Only in-process fake connectors /
connections are used; there are no source-string assertions. Suppressing
fakes are released ONLY AFTER the suppressed-state assertions and BEFORE
loop teardown, so teardown is always finite.

Two FINAL-BOUNDARY regressions (root takeback contracts A and B):

  A. Callback-before-cancel race: the ownership done-callback already ran
     (tracked the late resource) while the caller was NOT yet abandoned;
     when the caller is then externally cancelled BEFORE its normal claim,
     abandonment must STILL reach the SAME idempotent close decision —
     the resource must be closed EXACTLY ONCE inside the positive cleanup
     budget (never re-tracked, never double-attempted, never left open).
  B. A synchronous close initializer may consume the remaining cleanup
     budget and return an UNSTARTED coroutine. Immediately BEFORE
     ensure_future the caller's OWN absolute cleanup deadline is checked
     AGAIN: the unstarted coroutine is disposed, ZERO new tasks are
     created, and the ACTUAL resource stays retained with a truthful
     UNPROVEN close (never close_requested, never default-closed).

Root-review hardening of boundary A (second pass):

  C. The ownership callback's resource track is IDEMPOTENT (id-presence
     check): a scheduled callback running AFTER the external-cancel branch
     already made the decision can never re-track and RESET the close
     ledger (close_requested/closed/close_failed back to open).
  D. A RAISING synchronous close initializer inside the external-cancel
     manual decision never masks the original CancelledError: the failure
     is ledgered against the ACTUAL resource and the cancellation still
     propagates exactly once.

AUTHORED UNRUN: an independent review and one isolated focused execution
are required before any claim.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import pytest

_DRIVER_PATH = (
    Path(__file__).resolve().parents[2]
    / "bench" / "perf" / "edgellm_asr_ws_perf_gate.py"
)


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "edgellm_asr_ws_perf_gate_late_connector", _DRIVER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def run_owned(coro):
    """One owned loop per test: no asyncio.run Runner cancel-gather, no
    default-executor join; the loop is closed only after the scenario has
    released every suppressor (finite teardown by construction)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        # Safety net only: scenarios must leave nothing pending. Cancel any
        # straggler once and give it one bounded loop slice before close.
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.sleep(0.05))
        loop.close()


class FakeWSConn:
    """Fake WS connection whose supported close is an async coroutine."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        return self._close_coro()

    async def _close_coro(self) -> None:
        await asyncio.sleep(0)


class FakeWSConnCloseFails(FakeWSConn):
    async def _close_coro(self) -> None:
        raise RuntimeError("close boom")


class FakeWriterNoWaitClosed:
    """Fake HTTP StreamWriter WITHOUT a wait_closed observation contract."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class SuppressingConnector:
    """Fake connector that suppresses the first cancellation and later —
    only when the TEST releases it — completes with a live resource."""

    def __init__(self, resource: Any, release: asyncio.Event) -> None:
        self.resource = resource
        self.release = release
        self.cancel_count = 0

    async def __call__(self):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancel_count += 1
        # Suppressed: stay alive until the test releases, then hand back
        # the live resource AFTER the caller has already left.
        await self.release.wait()
        return self.resource


async def _settle(predicate, attempts: int = 200) -> bool:
    """Bounded loop settle: poll a ledger predicate, never unbounded."""
    for _ in range(attempts):
        await asyncio.sleep(0.005)
        if predicate():
            return True
    return predicate()


# ──────────────────────────────────────────────────────────────────────
# Defect A: external cancel -> late connector -> resource retained + closed
# ──────────────────────────────────────────────────────────────────────


def test_external_cancel_late_connector_retained_and_closed_in_cleanup_budget():
    async def scenario() -> None:
        registry = gate.AsyncLifetimeRegistry("ext-cancel")
        resource = FakeWSConn()
        release = asyncio.Event()
        connector = SuppressingConnector(resource, release)
        now = time.monotonic()
        request_deadline = now + 5.0
        cleanup_deadline = now + 6.0  # positive EXISTING cleanup budget
        connect_task = asyncio.ensure_future(
            gate._connect_owned(
                connector, request_deadline, "connect", registry,
                kind="ws-connection",
                late_close=lambda c: c.close(),
                cleanup_deadline_mono=cleanup_deadline,
            )
        )
        await asyncio.sleep(0.05)  # connector task actually started
        connect_task.cancel()  # external caller cancellation
        with pytest.raises(asyncio.CancelledError):
            await connect_task
        # Cancellation DELIVERY to the suppressing connector is async.
        assert await _settle(lambda: connector.cancel_count == 1)
        # Suppressed-state assertions BEFORE release: cancellation requested
        # EXACTLY ONCE, the late resource not yet acquired, nothing closed.
        assert connector.cancel_count == 1
        assert id(resource) not in registry.resources
        assert resource.close_calls == 0
        # Release AFTER asserting pending ownership, BEFORE teardown.
        release.set()
        settled = await _settle(
            lambda: registry.resources.get(id(resource), {}).get("close_state")
            == "closed"
        )
        assert settled, registry.snapshot()
        entry = registry.resources[id(resource)]
        assert entry["kind"] == "ws-connection-late"
        assert entry["close_state"] == "closed"
        assert entry["close_detail"] == "late close observed complete"
        assert resource.close_calls == 1  # exactly ONE close attempt
        assert not registry.pending_entries()

    run_owned(scenario())


# ──────────────────────────────────────────────────────────────────────
# Defect B: late completion AFTER the cleanup deadline -> zero new async
# tasks, resource retained UNPROVEN (and cancellation still exactly once)
# ──────────────────────────────────────────────────────────────────────


def test_late_completion_after_cleanup_deadline_creates_no_task_unproven():
    async def scenario() -> None:
        registry = gate.AsyncLifetimeRegistry("expired-cleanup")
        resource = FakeWSConn()
        release = asyncio.Event()
        connector = SuppressingConnector(resource, release)
        now = time.monotonic()
        request_deadline = now + 0.1
        cleanup_deadline = now + 0.1  # expires before the late completion
        with pytest.raises(TimeoutError):
            await gate._connect_owned(
                connector, request_deadline, "connect", registry,
                kind="ws-connection",
                late_close=lambda c: c.close(),
                cleanup_deadline_mono=cleanup_deadline,
            )
        # Cancellation requested exactly once on the timeout path (delivery
        # to the suppressing connector is async); still pending.
        assert await _settle(lambda: connector.cancel_count == 1)
        # A later registry drain must NOT cancel a second time.
        await registry.drain(0.05)
        assert connector.cancel_count == 1
        # Wait until the cleanup deadline has genuinely passed.
        while time.monotonic() < cleanup_deadline + 0.05:
            await asyncio.sleep(0.01)
        task_count_before = len(registry.tasks)
        # Release AFTER asserting pending ownership, BEFORE teardown.
        release.set()
        settled = await _settle(lambda: id(resource) in registry.resources)
        assert settled, registry.snapshot()
        await asyncio.sleep(0.02)  # let any (forbidden) close task appear
        entry = registry.resources[id(resource)]
        assert entry["kind"] == "ws-connection-late"
        # UNPROVEN: not closed, no close attempt made after expiry.
        assert entry["close_state"] != "closed"
        assert "cleanup deadline expired" in (entry["close_detail"] or "")
        assert resource.close_calls == 0
        # Zero NEW async tasks: only the callback ran, no close task.
        assert len(registry.tasks) == task_count_before

    run_owned(scenario())


# ──────────────────────────────────────────────────────────────────────
# Defect C: HTTP late close without wait_closed -> sync close ONCE but the
# resource is NOT recorded closed (UNPROVEN stays unproven)
# ──────────────────────────────────────────────────────────────────────


def test_late_http_close_without_wait_closed_sync_close_once_not_closed():
    async def scenario() -> None:
        registry = gate.AsyncLifetimeRegistry("http-no-wait-closed")
        writer = FakeWriterNoWaitClosed()
        resource = (None, writer)
        release = asyncio.Event()
        connector = SuppressingConnector(resource, release)
        now = time.monotonic()
        with pytest.raises(TimeoutError):
            await gate._connect_owned(
                connector, now + 0.1, "HTTP connect", registry,
                kind="http-reader-writer",
                late_close=gate._late_http_close,  # ACTUAL driver helper
                cleanup_deadline_mono=now + 5.0,  # positive cleanup budget
            )
        assert await _settle(lambda: connector.cancel_count == 1)
        assert id(resource) not in registry.resources
        release.set()  # release AFTER pending assertions, BEFORE teardown
        settled = await _settle(lambda: id(resource) in registry.resources)
        assert settled, registry.snapshot()
        await asyncio.sleep(0.02)
        entry = registry.resources[id(resource)]
        # The supported sync close WAS initiated exactly once ...
        assert writer.close_calls == 1
        # ... but with no observation contract the close stays UNPROVEN:
        # close_requested, NEVER the old invented closed=True.
        assert entry["close_state"] == "close_requested"
        assert "UNPROVEN" in (entry["close_detail"] or "")
        assert entry["close_state"] != "closed"

    run_owned(scenario())


# ──────────────────────────────────────────────────────────────────────
# Callback close failure -> error retained in the ledger, not swallowed
# ──────────────────────────────────────────────────────────────────────


def test_late_close_failure_is_retained_as_ledger_error():
    async def scenario() -> None:
        registry = gate.AsyncLifetimeRegistry("close-fails")
        resource = FakeWSConnCloseFails()
        release = asyncio.Event()
        connector = SuppressingConnector(resource, release)
        now = time.monotonic()
        with pytest.raises(TimeoutError):
            await gate._connect_owned(
                connector, now + 0.1, "connect", registry,
                kind="ws-connection",
                late_close=lambda c: c.close(),
                cleanup_deadline_mono=now + 5.0,
            )
        release.set()  # release AFTER the abandonment, BEFORE teardown
        settled = await _settle(
            lambda: registry.resources.get(id(resource), {}).get("close_state")
            == "close_failed"
        )
        assert settled, registry.snapshot()
        entry = registry.resources[id(resource)]
        assert entry["close_state"] == "close_failed"
        assert "late close failed: RuntimeError" in (entry["close_detail"] or "")
        assert resource.close_calls == 1  # exactly ONE close attempt

    run_owned(scenario())


# ──────────────────────────────────────────────────────────────────────
# Normal connector: caller claims the resource -> no early close, no
# double ownership (exactly one ledger entry, normal kind, still open)
# ──────────────────────────────────────────────────────────────────────


def test_normal_connector_not_early_closed_not_double_owned():
    async def scenario() -> None:
        registry = gate.AsyncLifetimeRegistry("normal")
        resource = FakeWSConn()

        async def quick_connect():
            await asyncio.sleep(0)
            return resource

        now = time.monotonic()
        got = await gate._connect_owned(
            quick_connect, now + 5.0, "connect", registry,
            kind="ws-connection",
            late_close=lambda c: c.close(),
            cleanup_deadline_mono=now + 6.0,
        )
        assert got is resource
        await asyncio.sleep(0.05)  # ownership callback already ran by now
        assert resource.close_calls == 0  # never early-closed
        entries = [e for e in registry.resources.values()
                   if e["obj"] is resource]
        assert len(entries) == 1  # single ownership record (id-keyed)
        assert entries[0]["kind"] == "ws-connection"  # normal, not -late
        assert entries[0]["close_state"] == "open"
        assert not registry.pending_entries()

    run_owned(scenario())


# ──────────────────────────────────────────────────────────────────────
# Final boundary A: callback-before-cancel race. The ownership callback
# runs while the caller is NOT yet abandoned (tracks, no decision); the
# caller is then externally cancelled before its normal claim — the SAME
# idempotent decision must still close the resource EXACTLY ONCE inside
# the positive cleanup budget.
# ──────────────────────────────────────────────────────────────────────


def test_callback_already_run_then_external_cancel_still_closes_once_in_budget():
    async def scenario() -> None:
        registry = gate.AsyncLifetimeRegistry("callback-before-cancel")
        resource = FakeWSConn()
        holder: dict[str, Any] = {}

        async def completing_connector():
            await asyncio.sleep(0)  # let the caller enter its wait
            loop = asyncio.get_running_loop()
            # Queue the caller cancellation BEFORE this task's completion
            # callbacks run: FIFO loop order guarantees the ownership
            # done-callback executes FIRST (abandoned still False — it
            # tracks and returns), and only THEN is CancelledError
            # delivered to the waiting caller.
            loop.call_soon(holder["caller"].cancel)
            return resource

        async def caller():
            now = time.monotonic()
            return await gate._connect_owned(
                completing_connector, now + 5.0, "connect", registry,
                kind="ws-connection",
                late_close=lambda c: c.close(),
                cleanup_deadline_mono=now + 5.0,  # positive cleanup budget
            )

        outer = asyncio.ensure_future(caller())
        holder["caller"] = outer
        with pytest.raises(asyncio.CancelledError):
            await outer
        settled = await _settle(
            lambda: registry.resources.get(id(resource), {}).get("close_state")
            == "closed"
        )
        assert settled, registry.snapshot()
        entry = registry.resources[id(resource)]
        # Abandonment DID reenter the close decision after the callback had
        # already run: closed exactly once inside the positive budget.
        assert entry["kind"] == "ws-connection-late"
        assert entry["close_state"] == "closed"
        assert entry["close_detail"] == "late close observed complete"
        assert resource.close_calls == 1  # no double attempt / double cancel
        entries = [e for e in registry.resources.values()
                   if e["obj"] is resource]
        assert len(entries) == 1  # idempotent: never re-tracked
        assert not registry.pending_entries()

    run_owned(scenario())


# ──────────────────────────────────────────────────────────────────────
# Manual close on external cancel: a RAISING synchronous close initializer
# must never mask the original CancelledError — the failure is retained
# against the ACTUAL resource and the cancellation still propagates.
# ──────────────────────────────────────────────────────────────────────


def test_manual_close_on_external_cancel_raises_sync_init_propagates_cancel():
    async def scenario() -> None:
        registry = gate.AsyncLifetimeRegistry("manual-close-raise")
        resource = FakeWSConn()
        holder: dict[str, Any] = {}

        async def completing_connector():
            await asyncio.sleep(0)  # let the caller enter its wait
            loop = asyncio.get_running_loop()
            # Same deterministic FIFO ordering as boundary A: the ownership
            # done-callback runs FIRST (tracks, no decision), then the
            # external cancellation reaches the waiting caller, whose
            # except branch runs the manual shared decision.
            loop.call_soon(holder["caller"].cancel)
            return resource

        def raising_sync_close(res: Any) -> None:
            res.close_calls += 1
            raise RuntimeError("sync close boom")

        async def caller():
            now = time.monotonic()
            return await gate._connect_owned(
                completing_connector, now + 5.0, "connect", registry,
                kind="ws-connection",
                late_close=raising_sync_close,
                cleanup_deadline_mono=now + 5.0,  # positive cleanup budget
            )

        outer = asyncio.ensure_future(caller())
        holder["caller"] = outer
        # The ORIGINAL CancelledError propagates, not the close failure.
        with pytest.raises(asyncio.CancelledError):
            await outer
        settled = await _settle(
            lambda: registry.resources.get(id(resource), {}).get("close_state")
            == "close_failed"
        )
        assert settled, registry.snapshot()
        entry = registry.resources[id(resource)]
        assert entry["kind"] == "ws-connection-late"  # ACTUAL resource retained
        assert entry["close_state"] == "close_failed"
        assert "late close unschedulable/failed: RuntimeError" in (
            entry["close_detail"] or ""
        )
        assert resource.close_calls == 1  # exactly ONE close attempt
        entries = [e for e in registry.resources.values()
                   if e["obj"] is resource]
        assert len(entries) == 1  # idempotent: never re-tracked/reset
        assert not registry.pending_entries()

    run_owned(scenario())


# ──────────────────────────────────────────────────────────────────────
# Final boundary B: a synchronous close initializer consumes the remaining
# cleanup budget and returns an UNSTARTED coroutine. Immediately before
# ensure_future the absolute deadline is checked AGAIN: the coroutine is
# disposed, ZERO new tasks are created, the ACTUAL resource stays retained
# with a truthful UNPROVEN close.
# ──────────────────────────────────────────────────────────────────────


def test_sync_close_initializer_consuming_budget_disposes_unstarted_coroutine(
    monkeypatch,
):
    async def scenario() -> None:
        registry = gate.AsyncLifetimeRegistry("sync-init-expiry")
        resource = FakeWSConn()
        release = asyncio.Event()
        connector = SuppressingConnector(resource, release)
        # Fake monotonic clock: ONLY time.monotonic is patched (restored by
        # the monkeypatch fixture). gate.time is the SHARED stdlib time
        # module, and the event loop's internal clock (BaseEventLoop.time)
        # reads time.monotonic from it too — so the fake must stay anchored
        # to the real monotonic clock to keep loop timers progressing. It
        # returns real monotonic + a deterministic offset that jumps +10s
        # once the sync close initializer consumes the budget.
        real_monotonic = time.monotonic
        clock = {"t": 0.0}

        def fake_monotonic() -> float:
            return real_monotonic() + clock["t"]

        monkeypatch.setattr(gate.time, "monotonic", fake_monotonic)
        calls = {"init": 0, "body": 0}

        def late_close(res: Any):
            calls["init"] += 1
            clock["t"] += 10.0  # offset jump: budget consumed...

            async def _close() -> None:
                calls["body"] += 1
                res.close_calls += 1

            return _close()  # ...then returns an UNSTARTED coroutine

        now = fake_monotonic()
        with pytest.raises(TimeoutError):
            await gate._connect_owned(
                connector, now + 0.1, "connect", registry,
                kind="ws-connection",
                late_close=late_close,
                cleanup_deadline_mono=now + 1.0,  # still positive at call
            )
        assert await _settle(lambda: connector.cancel_count == 1)
        task_count_before = len(registry.tasks)
        release.set()  # release AFTER pending assertions, BEFORE teardown
        settled = await _settle(lambda: id(resource) in registry.resources)
        assert settled, registry.snapshot()
        await asyncio.sleep(0.02)  # let any (forbidden) close task appear
        entry = registry.resources[id(resource)]
        assert calls["init"] == 1  # the sync initializer ran exactly once
        assert calls["body"] == 0  # unstarted coroutine NEVER executed
        assert resource.close_calls == 0
        assert entry["kind"] == "ws-connection-late"  # retained, not closed
        # Truthful UNPROVEN: disposed, not close_requested, never closed.
        assert entry["close_state"] == "close_failed"
        assert "UNPROVEN" in (entry["close_detail"] or "")
        assert "synchronous close initializer" in (entry["close_detail"] or "")
        # ZERO new async tasks created after the expiry.
        assert len(registry.tasks) == task_count_before
        assert not registry.pending_entries()

    run_owned(scenario())


# ──────────────────────────────────────────────────────────────────────
# Cancellation is requested EXACTLY ONCE across the whole connector
# lifetime, including an explicit late drain.
# ──────────────────────────────────────────────────────────────────────


def test_connector_cancellation_requested_exactly_once():
    async def scenario() -> None:
        registry = gate.AsyncLifetimeRegistry("cancel-once")
        resource = FakeWSConn()
        release = asyncio.Event()
        connector = SuppressingConnector(resource, release)
        now = time.monotonic()
        with pytest.raises(TimeoutError):
            await gate._connect_owned(
                connector, now + 0.1, "connect", registry,
                kind="ws-connection",
                late_close=lambda c: c.close(),
                cleanup_deadline_mono=now + 5.0,
            )
        assert await _settle(lambda: connector.cancel_count == 1)
        # Explicit drains (the per-utterance cleanup path) must observe the
        # recorded cancel request and never issue a second one.
        await registry.drain(0.05)
        await registry.drain(0.05)
        assert connector.cancel_count == 1
        # Release AFTER asserting the single-cancel ownership state and let
        # the late close complete so teardown is finite.
        release.set()
        settled = await _settle(
            lambda: registry.resources.get(id(resource), {}).get("close_state")
            == "closed"
        )
        assert settled, registry.snapshot()
        assert connector.cancel_count == 1  # still exactly once at the end

    run_owned(scenario())
