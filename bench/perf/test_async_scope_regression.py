#!/usr/bin/env python3
"""Regression tests: driver child-scope drain + HTTP tuple close accounting.

Executes the ACTUAL driver class/functions from
``bench/perf/edgellm_asr_ws_perf_gate.py`` (importlib-loaded, no mirror/AST
clone). Only the nested connection opener is monkeypatched with fakes; all
registry, drain, connect-ownership, cleanup and probe logic is the real
driver code. Loopback-free (no sockets at all), no device/network.

Covers:
  1. a normal probe helper drain cannot cancel the CLI top task/supervisor
  2. a completing B2-style helper preserves a sibling blocked in header read
  3. child-local snapshot excludes the parent; root snapshot includes ALL
     actual refs forwarded by both helpers
  4. a helper-returned cancellation-suppressing connector stays retained
     pending in the ROOT ledger (forces async-lifetime NOTQUALIFIED basis)
  5. observed writer wait_closed marks the ORIGINAL (reader, writer) tuple
     closed; timeout preserves open (never faked by close() alone)
  6. run_gate's child scope drain never cancels the CLI current task or the
     supervisor; the current task is excluded even if (accidentally)
     registered in the drained scope

Fixture honesty contract (v011 correction): every fake serves a COMPLETE,
valid HTTP response (accurate Content-Length, valid JSON body with text);
a blocked sibling blocks genuinely INSIDE the first header read until its
release event; every test's ``finally`` releases all fixture events and
bounded-joins all retained tasks so an assertion failure can never hang a
sibling, connector or writer cleanup.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Optional

_DRIVER_PATH = (
    Path(__file__).resolve().parents[2] / "bench" / "perf" / "edgellm_asr_ws_perf_gate.py"
)


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "edgellm_asr_ws_perf_gate_scope_regression", _DRIVER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


# ──────────────────────────────────────────────────────────────────────
# Fakes (connection transport ONLY; all driver logic is real)
# ──────────────────────────────────────────────────────────────────────

_BODY = b'{"text": "ok", "status": "ready"}'


class FakeReader:
    """Reader serving ONE complete valid HTTP 200 response (headers AND
    full JSON body, accurate Content-Length), then EOF. With
    ``blocked_until`` the FIRST read (the header read) blocks until the
    event is released — the sibling is genuinely stuck in the header
    phase, never after an empty header."""

    def __init__(self, blocked_until: Optional[asyncio.Event] = None) -> None:
        self._resp = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(_BODY)}\r\n\r\n"
        ).encode() + _BODY
        self._served = False
        self._blocked_until = blocked_until
        self.block_entered = asyncio.Event()
        self.header_served = asyncio.Event()

    async def read(self, n: int) -> bytes:
        if self._blocked_until is not None and not self._served:
            self.block_entered.set()
            await self._blocked_until.wait()
        if not self._served:
            self._served = True
            self.header_served.set()
            return self._resp
        return b""


class FakeWriter:
    def __init__(self, wait_closed_s: float = 0.0,
                 block_until: Optional[asyncio.Event] = None) -> None:
        self._delay_s = wait_closed_s
        self._block_until = block_until
        self.close_called = False
        self.wait_closed_called = False

    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def close(self) -> None:
        self.close_called = True

    async def wait_closed(self) -> None:
        self.wait_closed_called = True
        if self._block_until is not None:
            await self._block_until.wait()
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)


def _entry_by_phase(snap: dict[str, Any], phase: str) -> Optional[dict[str, Any]]:
    for t in snap["tasks"]:
        if t["phase"] == phase:
            return t
    return None


def _pair_entry(snap: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The ACTUAL shared (reader, writer) tuple entry. Depending on a
    benign completion/callback ordering inside _connect_owned, the SAME
    shared entry object is first registered either as 'http-reader-writer'
    (claim branch first) or 'http-reader-writer-late' (ownership callback
    first); the claim branch then reuses the shared entry either way."""
    for r in snap["resources"]:
        if r["kind"] in ("http-reader-writer", "http-reader-writer-late"):
            return r
    return None


def _entry_by_kind(snap: dict[str, Any], kind: str) -> dict[str, Any]:
    for r in snap["resources"]:
        if r["kind"] == kind:
            return r
    raise AssertionError(f"resource kind {kind!r} missing: {snap['resources']}")


async def _join_all(root: gate.AsyncLifetimeRegistry, timeout_s: float = 2.0) -> None:
    """Fixture cleanup: request cancellation ONCE per still-pending task,
    then bounded-join ALL retained actual refs. Never unbounded."""
    pending = [t for t in list(root.tasks) if not t.done()]
    for t in pending:
        t.cancel()
    if pending:  # empty-set guard: asyncio.wait raises on an empty set
        await asyncio.wait(pending, timeout=timeout_s)


# ──────────────────────────────────────────────────────────────────────
# 1. Normal probe does not cancel top/supervisor
# ──────────────────────────────────────────────────────────────────────

def test_normal_probe_does_not_cancel_top_or_supervisor():
    async def scenario():
        root = gate.AsyncLifetimeRegistry("cli:probe-scope")
        writer = FakeWriter()

        async def fake_open_conn(parsed, deadline_mono, registry=None):
            return FakeReader(), writer

        real_open = gate._default_open_conn
        gate._default_open_conn = fake_open_conn
        try:
            async def top() -> None:
                # run_gate/probe wiring: the shared CLI registry is passed
                # down; _http_get_json scopes itself as a CHILD internally.
                out = await gate._http_get_json(
                    "http://127.0.0.1:1/v1/capabilities",
                    time.monotonic() + 10.0, root,
                )
                assert out.get("request_error") is None, out
                assert out.get("status") == 200, out

            top_task = asyncio.create_task(top(), name="run_gate")
            root.track_task(top_task, "run_gate top-level")

            async def supervise() -> None:
                await asyncio.wait({top_task})

            sup = asyncio.create_task(supervise(), name="cli_supervise")
            root.track_task(sup, "cli supervise")
            try:
                await asyncio.wait({sup}, timeout=15.0)
            finally:
                gate._default_open_conn = real_open
                await _join_all(root)
        finally:
            gate._default_open_conn = real_open

        snap = root.snapshot()
        top_e = _entry_by_phase(snap, "run_gate top-level")
        sup_e = _entry_by_phase(snap, "cli supervise")
        assert top_e["state"] == "done" and not top_e["cancel_requested"], top_e
        assert sup_e["state"] == "done" and not sup_e["cancel_requested"], sup_e
        # Observed close closed BOTH the writer and the original tuple.
        assert _entry_by_kind(snap, "stdlib-stream-writer")["close_state"] == "closed"
        # STRICT normal-success label: a successfully claimed owned pair is
        # exactly 'http-reader-writer'; the provisional '-late' label must
        # have been promoted by the claim. Accepting either label here would
        # mask the labeling regression.
        pair_e = _entry_by_kind(snap, "http-reader-writer")
        assert pair_e["close_state"] == "closed", snap
        assert not any(
            r["kind"] == "http-reader-writer-late" for r in snap["resources"]
        ), snap
        assert snap["open_resource_count"] == 0, snap
        assert snap["pending_count"] == 0, snap

    asyncio.run(asyncio.wait_for(scenario(), 20.0))


# ──────────────────────────────────────────────────────────────────────
# 1b. Focused late->normal promotion: exact normal label, shared dict/ref
#     identity, preserved close ledger, abandoned stays late
# ──────────────────────────────────────────────────────────────────────

def test_track_resource_promotes_late_only_on_normal_claim():
    """Focused registry semantics for the evidence-labeling fix.

    Covers exactly what the independent review flagged: the ownership
    callback-first provisional ``kind + '-late'`` entry must be promoted to
    the normal kind on a successful claim, WITHOUT replacing the shared
    entry object, releasing the actual ``obj`` reference, or resetting the
    close ledger; normal must never be downgraded; an abandoned
    (unclaimed) resource must remain ``-late``.
    """
    async def scenario():
        root = gate.AsyncLifetimeRegistry("cli:label-identity")
        child = root.child("run:label-identity")

        # Callback-before-claim ordering: provisional late entry lives in the
        # child scope and is shared (same dict) into the root.
        resource = object()
        child.track_resource(resource, "http-reader-writer-late", "owned pair")
        entry = child.resources[id(resource)]
        # ACTUAL shared dict/ref identity: child and ancestor forward the
        # very same mutable entry, holding the actual live resource.
        assert root.resources[id(resource)] is entry
        assert entry["obj"] is resource
        assert entry["kind"] == "http-reader-writer-late"
        # A close ledger observed before the claim must survive promotion.
        child.note_resource_closed(resource, True, "observed wait_closed")
        assert entry["close_state"] == "closed"

        # Corrupt the ledger state/detail to prove promotion does not touch
        # them; then promote via the normal successful claim.
        entry["close_detail"] = "observed wait_closed sentinel"
        child.track_resource(resource, "http-reader-writer", "owned pair")
        assert entry["kind"] == "http-reader-writer", entry
        assert child.resources[id(resource)] is entry, "shared dict replaced"
        assert root.resources[id(resource)] is entry, "shared dict replaced"
        assert entry["obj"] is resource, "actual ref released/replaced"
        assert entry["close_state"] == "closed", "close_state reset on promote"
        assert entry["close_detail"] == "observed wait_closed sentinel"

        # Re-tracking the normal kind again is a no-op (no downgrade).
        child.track_resource(resource, "http-reader-writer", "owned pair")
        assert entry["kind"] == "http-reader-writer"
        # A normal claim must never be downgraded to the late variant.
        child.track_resource(resource, "http-reader-writer-late", "owned pair")
        assert entry["kind"] == "http-reader-writer", entry

        # Abandoned/unclaimed resource: provisionally late, never claimed by
        # a normal track_resource -> stays late (no accidental promotion).
        abandoned = object()
        root.track_resource(abandoned, "http-reader-writer-late", "abandoned pair")
        a_entry = root.resources[id(abandoned)]
        assert a_entry["kind"] == "http-reader-writer-late", a_entry
        assert a_entry["close_state"] == "open"

        # A DIFFERENT kind's '-late' entry is never rewritten by an
        # unrelated normal kind.
        other = object()
        root.track_resource(other, "ws-reader-writer-late", "ws pair")
        o_entry = root.resources[id(other)]
        root.track_resource(other, "http-reader-writer", "http pair")
        assert o_entry["kind"] == "ws-reader-writer-late", o_entry

        snap = root.snapshot()
        kinds = [r["kind"] for r in snap["resources"]]
        assert kinds.count("http-reader-writer") == 1, kinds
        assert "http-reader-writer-late" in kinds, kinds
        assert "ws-reader-writer-late" in kinds, kinds

    asyncio.run(asyncio.wait_for(scenario(), 15.0))


# ──────────────────────────────────────────────────────────────────────
# 2. Helper A completion preserves sibling B blocked in header read
# ──────────────────────────────────────────────────────────────────────

def test_helper_completion_preserves_blocked_sibling():
    async def scenario():
        root = gate.AsyncLifetimeRegistry("cli:sibling")
        release_b = asyncio.Event()

        async def fast_conn(parsed, deadline_mono, registry=None):
            return FakeReader(), FakeWriter()

        blocked_reader = FakeReader(blocked_until=release_b)

        async def blocked_conn(parsed, deadline_mono, registry=None):
            return blocked_reader, FakeWriter()

        real_open = gate._default_open_conn
        pcm = b"\x00\x01" * 100
        task_a = task_b = None
        try:
            async def member(name, opener):
                return await gate.run_http_utterance(
                    "http://127.0.0.1:1/asr", pcm, wav_name="t.wav",
                    order=0, file="t.wav", item_id=name, warm=True,
                    pair_index=0, audio_s=0.01, transcript=None,
                    deadline_mono=time.monotonic() + 10.0,
                    open_conn=opener, registry=root,
                )

            # The B2 wiring: both member tasks tracked in the SHARED root
            # registry (pair phases), each helper scopes itself as a child.
            task_a = root.track_task(
                asyncio.create_task(member("a", fast_conn), name="b2 a"),
                "b2 pair 0 member a",
            )
            task_b = root.track_task(
                asyncio.create_task(member("b", blocked_conn), name="b2 b"),
                "b2 pair 0 member b",
            )
            # B is genuinely parked INSIDE its first (header) read.
            await asyncio.wait_for(blocked_reader.block_entered.wait(), 5.0)

            done, _ = await asyncio.wait({task_a}, timeout=15.0)
            assert task_a in done and task_a.exception() is None
            res_a = task_a.result()
            assert res_a.ok, res_a.error

            # A's helper drain must NOT have cancelled sibling B or the
            # ledger ancestor entries; B is STILL in the header phase.
            snap_mid = root.snapshot()
            b_e = _entry_by_phase(snap_mid, "b2 pair 0 member b")
            assert b_e is not None and not b_e["done"], b_e
            assert not b_e["cancel_requested"], b_e
            assert not blocked_reader.header_served.is_set(), (
                "sibling B must be blocked BEFORE its header was served"
            )
            top_like = _entry_by_phase(snap_mid, "cli top")  # none registered
            assert top_like is None

            release_b.set()
            done_b, _ = await asyncio.wait({task_b}, timeout=15.0)
            assert task_b in done_b and task_b.exception() is None
            res_b = task_b.result()
            assert res_b.ok, res_b.error
            assert not task_b.cancelled()
        finally:
            gate._default_open_conn = real_open
            release_b.set()  # an assertion failure must never hang B
            if task_a is not None and task_b is not None:
                await _join_all(root)

    asyncio.run(asyncio.wait_for(scenario(), 30.0))


# ──────────────────────────────────────────────────────────────────────
# 3. Child snapshot local; root snapshot complete with actual refs
# ──────────────────────────────────────────────────────────────────────

def test_child_snapshot_local_root_snapshot_complete():
    async def scenario():
        root = gate.AsyncLifetimeRegistry("cli:snap")
        top = asyncio.create_task(asyncio.sleep(3600), name="cli top")
        root.track_task(top, "run_gate top-level")
        ev_a, ev_b = asyncio.Event(), asyncio.Event()
        returned: list[asyncio.Task] = []
        try:
            async def helper(label, ev: asyncio.Event):
                child = root.child(f"helper:{label}")
                t = child.track_task(
                    asyncio.create_task(ev.wait()), f"{label} work"
                )
                child.track_resource(object(), "test-resource", label)
                local = child.snapshot()
                assert _entry_by_phase(local, "run_gate top-level") is None, local
                assert local["task_count"] == 1, local
                return t

            t_a, t_b = await helper("a", ev_a), await helper("b", ev_b)
            returned.extend([t_a, t_b])
            # Root holds the ACTUAL refs after the helpers returned.
            assert t_a in root.tasks and t_b in root.tasks
            snap = root.snapshot()
            assert _entry_by_phase(snap, "run_gate top-level") is not None
            assert _entry_by_phase(snap, "a work") is not None
            assert _entry_by_phase(snap, "b work") is not None
            kinds = [r["kind"] for r in snap["resources"]]
            assert "test-resource" in kinds
            assert snap["task_count"] == 3 and snap["pending_count"] == 3
        finally:
            top.cancel()
            ev_a.set()
            ev_b.set()
            if returned:
                await asyncio.gather(top, *returned, return_exceptions=True)
            await _join_all(root)

    asyncio.run(asyncio.wait_for(scenario(), 15.0))


# ──────────────────────────────────────────────────────────────────────
# 4. Suppression connector stays retained pending in the ROOT ledger
# ──────────────────────────────────────────────────────────────────────

def test_suppressing_connector_retained_pending_in_root():
    async def scenario():
        root = gate.AsyncLifetimeRegistry("cli:suppress")
        release = asyncio.Event()

        async def suppressing_conn(parsed, deadline_mono, registry=None):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Suppress EXACTLY ONE cancellation by parking on a
                # RELEASEABLE event (never an anonymous forever-event):
                # after release the connector completes with an actual
                # transport so the driver's late-close path stays honest.
                await release.wait()
                return FakeReader(), FakeWriter()

        real_open = gate._default_open_conn
        gate._default_open_conn = suppressing_conn
        try:
            try:
                await gate._http_get_json(
                    "http://127.0.0.1:1/v1/readyz",
                    time.monotonic() + 0.15, root,
                )
            except TimeoutError:
                pass  # expected: the suppressing connector exceeds the budget
        finally:
            gate._default_open_conn = real_open

        snap = root.snapshot()
        assert snap["pending_count"] >= 1, snap
        conn_e = _entry_by_phase(snap, "probe HTTP connect")
        assert conn_e is not None, snap
        assert conn_e["cancel_requested"], conn_e
        # Same basis outcome_to_dict uses: pending forces NOTQUALIFIED.
        async_lifetime_pending = bool(
            snap["pending_count"] or snap["open_resource_count"]
        )
        assert async_lifetime_pending
        # Cleanup for the fixture itself: capture the pending ACTUAL refs
        # BEFORE releasing the suppressed connector, release, then bounded
        # join everything (empty-set guarded).
        pending = [t for t in list(root.tasks) if not t.done()]
        release.set()
        if pending:
            await asyncio.wait(pending, timeout=2.0)
        await _join_all(root)

    asyncio.run(asyncio.wait_for(scenario(), 20.0))


# ──────────────────────────────────────────────────────────────────────
# 5. Tuple close accounting: observed vs timeout
# ──────────────────────────────────────────────────────────────────────

def test_tuple_closed_on_observed_wait_closed_only():
    async def observed():
        root = gate.AsyncLifetimeRegistry("cli:tuple-ok")
        writer = FakeWriter(wait_closed_s=0.0)

        async def fake_open(parsed, dl, registry=None):
            return FakeReader(), writer

        real_open = gate._default_open_conn
        gate._default_open_conn = fake_open
        try:
            await gate._http_get_json(
                "http://127.0.0.1:1/x", time.monotonic() + 5.0, root
            )
        finally:
            gate._default_open_conn = real_open
            await _join_all(root)
        assert writer.close_called and writer.wait_closed_called
        snap_ok = root.snapshot()
        assert _entry_by_kind(snap_ok, "stdlib-stream-writer")["close_state"] == "closed"
        pair_e = _pair_entry(snap_ok)
        assert pair_e is not None and pair_e["close_state"] == "closed", snap_ok
        assert snap_ok["open_resource_count"] == 0

    async def timed_out():
        root = gate.AsyncLifetimeRegistry("cli:tuple-stuck")
        release_wc = asyncio.Event()
        writer = FakeWriter(block_until=release_wc)  # held until release

        async def fake_open(parsed, dl, registry=None):
            return FakeReader(), writer

        real_open = gate._default_open_conn
        gate._default_open_conn = fake_open
        try:
            await gate._http_get_json(
                "http://127.0.0.1:1/x", time.monotonic() + 0.2, root
            )
        finally:
            gate._default_open_conn = real_open
            # The close-wait task was cancel-requested by the driver but
            # FakeWriter does not suppress; release anyway so no fixture
            # event can ever dangle past the test, then bounded-join.
            release_wc.set()
            await _join_all(root)
        snap_stuck = root.snapshot()
        assert writer.close_called  # close WAS called …
        # … but the close was never OBSERVED: no false closed anywhere.
        assert _entry_by_kind(snap_stuck, "stdlib-stream-writer")["close_state"] != "closed"
        pair_e = _pair_entry(snap_stuck)
        assert pair_e is not None and pair_e["close_state"] != "closed", snap_stuck

    asyncio.run(asyncio.wait_for(observed(), 15.0))
    asyncio.run(asyncio.wait_for(timed_out(), 15.0))


# ──────────────────────────────────────────────────────────────────────
# 6. run_gate child-scope drain never cancels CLI self/supervisor
# ──────────────────────────────────────────────────────────────────────

def test_run_gate_child_scope_does_not_cancel_cli_self_or_supervisor():
    async def scenario():
        root = gate.AsyncLifetimeRegistry("cli:childscope")
        blocker_started = asyncio.Event()
        scope_refs: dict[str, Any] = {}

        async def top() -> None:
            # EXACT run_gate wiring: the run scope is a CHILD of the CLI
            # root; the run-level drain acts on the child only.
            run_scope = root.child("run:childscope")
            scope_refs["run_scope"] = run_scope

            async def blocker() -> None:
                blocker_started.set()
                await asyncio.Event().wait()

            bt = run_scope.track_task(
                asyncio.create_task(blocker(), name="owned helper"),
                "owned helper",
            )
            scope_refs["bt"] = bt
            await blocker_started.wait()
            # The current task (top) is ALSO registered in the drained
            # scope (accidental-registration guard): drain must neither
            # cancel it nor self-wait on it.
            run_scope.track_task(asyncio.current_task(), "run_gate top-level")
            await run_scope.drain(0.5)  # run-level drain, bounded
            assert bt.done() or bt.cancelled()

        top_task = asyncio.create_task(top(), name="run_gate")
        root.track_task(top_task, "run_gate top-level")

        async def supervise() -> None:
            await asyncio.wait({top_task})

        sup = asyncio.create_task(supervise(), name="cli_supervise")
        root.track_task(sup, "cli supervise")
        try:
            await asyncio.wait({sup}, timeout=15.0)
        finally:
            # Release the (already cancelled) blocker deterministically.
            await _join_all(root)

        snap = root.snapshot()
        top_e = _entry_by_phase(snap, "run_gate top-level")
        sup_e = _entry_by_phase(snap, "cli supervise")
        assert top_e["state"] == "done" and not top_e["cancel_requested"], top_e
        assert sup_e["state"] == "done" and not sup_e["cancel_requested"], sup_e
        owned_e = _entry_by_phase(snap, "owned helper")
        assert owned_e is not None, snap
        assert owned_e["state"] == "cancelled", owned_e
        # Root aggregate retains the ACTUAL refs of every task created in
        # this scenario (no trivial self-membership tautology).
        # Actual-ref membership against the KNOWN child tasks: every task
        # created in this scenario must be retained by the root aggregate,
        # and the child scope must retain its OWN registrations locally.
        run_scope = scope_refs["run_scope"]
        bt = scope_refs["bt"]
        assert top_task in root.tasks, "CLI top actual ref missing in root"
        assert sup in root.tasks, "supervisor actual ref missing in root"
        assert bt in root.tasks, "owned helper actual ref missing in root"
        assert bt in run_scope.tasks, "owned helper ref missing in child scope"
        assert top_task in run_scope.tasks, (
            "accidentally-registered current task ref missing in child scope"
        )
        assert len(root.tasks) >= 3, sorted(
            e["phase"] for e in snap["tasks"]
        )

    asyncio.run(asyncio.wait_for(scenario(), 20.0))


if __name__ == "__main__":
    import traceback

    tests = [
        test_normal_probe_does_not_cancel_top_or_supervisor,
        test_track_resource_promotes_late_only_on_normal_claim,
        test_helper_completion_preserves_blocked_sibling,
        test_child_snapshot_local_root_snapshot_complete,
        test_suppressing_connector_retained_pending_in_root,
        test_tuple_closed_on_observed_wait_closed_only,
        test_run_gate_child_scope_does_not_cancel_cli_self_or_supervisor,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
