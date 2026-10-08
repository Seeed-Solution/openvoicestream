"""Ownership tests for ``_AsrSlotJobs.drain`` under repeated cancellation.

Status: AUTHORED, UNRUN (static author pass only — a root reviewer must run
these; see EVIDENCE.md). No pytest-asyncio dependency: every async case is
driven through ``asyncio.run``.

What this pins (drain ownership ONLY):

  * A live ``concurrent.futures.Future`` stays in ``jobs.futures`` until its
    ``done()`` is actually observed; ``drain`` never clears the refs before
    awaiting.
  * Repeated cancellation of the awaiting task (twice, separated by real event
    loop turns) is DEFERRED: ``drain`` keeps waiting on the SAME retained
    aggregate, keeps the slot owned, and only re-raises ``CancelledError``
    AFTER every owned future has completed.
  * A native exception raised by the executor callable after release does not
    replace the caller's cancellation; refs are still pruned.
  * A future appended WHILE the initial snapshot is held stays owned, and drain
    waits for BOTH before returning — no early release.
  * The ordinary path (no cancellation, all finished) prunes refs and leaves no
    live future.

The EXACT ``_AsrSlotJobs`` class is extracted from the pinned ``server/main.py``
via AST and executed unchanged in a controlled namespace. The class body is not
mirrored here. Exactly ONE ``ThreadPoolExecutor`` is created per case, used for
that case's futures, and shut down (``wait=True``) in the case's outer
``finally``; the scenario never builds a second executor.

Evidence for the pinned class (module path + source SHA256) is printed from the
extraction helper at import time, not asserted by a behavioural test.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "server" / "main.py"

logger = logging.getLogger("test_asr_slot_jobs_drain")

# Bounded, monotonic wait budget for "the worker has actually started" /
# "the drain is still pending" spins. No unbounded event-loop loops.
_WAIT_BUDGET_S = 5.0


def _extract_asr_slot_jobs():
    """Exec the EXACT ``_AsrSlotJobs`` class from the pinned main.py.

    Returns ``(cls, executor)`` where ``executor`` is the single test-owned
    ``ThreadPoolExecutor`` bound as ``_get_asr_executor`` for this case. The
    caller owns and must shut it down.
    """
    source = MAIN_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        (node for node in tree.body
         if isinstance(node, ast.ClassDef) and node.name == "_AsrSlotJobs"),
        None,
    )
    if cls is None:  # pragma: no cover - guards a moved/renamed class
        raise AssertionError("_AsrSlotJobs class not found in server/main.py")

    lines = source.splitlines()
    class_src = "\n".join(lines[cls.lineno - 1:cls.end_lineno])
    digest = hashlib.sha256(class_src.encode("utf-8")).hexdigest()
    print(
        f"[asr-slot-jobs] class source from {MAIN_PY} "
        f"(lines {cls.lineno}-{cls.end_lineno}) sha256={digest}"
    )

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr-test")
    namespace = {
        "asyncio": asyncio,
        "logger": logger,
        "_get_asr_executor": lambda: executor,
    }
    module = ast.Module(body=[cls], type_ignores=[])
    exec(compile(module, str(MAIN_PY), "exec"), namespace)
    return namespace["_AsrSlotJobs"], executor


async def _spin_until(predicate, budget: float = _WAIT_BUDGET_S) -> bool:
    """Yield to the loop until ``predicate()`` is true or budget elapses."""
    deadline = time.monotonic() + budget
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0)
    return True


async def _cancel_twice(task: asyncio.Task, turns: int = 3) -> None:
    """Cancel ``task``, let the loop turn, then cancel a SECOND time."""
    await asyncio.sleep(0)
    task.cancel()
    for _ in range(turns):
        await asyncio.sleep(0)
    task.cancel()
    for _ in range(turns):
        await asyncio.sleep(0)


def test_repeated_cancellation_defers_until_worker_finishes():
    """Case 1: real executor job held by an Event; two cancellations.

    The drain must stay pending and keep the actual cf owned until the worker
    is released; only then does CancelledError propagate with refs pruned.
    """
    AsrSlotJobs, executor = _extract_asr_slot_jobs()
    started = threading.Event()
    release = threading.Event()

    async def scenario():
        def held_job():
            started.set()
            # Must be released by the scenario's finally, not by timeout.
            assert release.wait(10.0) is True
            return "ok"

        jobs = AsrSlotJobs()

        async def drive():
            cf = executor.submit(held_job)
            jobs.futures.append(cf)
            await asyncio.wrap_future(cf)

        driver = asyncio.ensure_future(drive())
        try:
            assert await _spin_until(started.is_set), "worker never started"
            assert len(jobs.futures) == 1
            owned = jobs.futures[0]

            drain_task = asyncio.ensure_future(jobs.drain())
            try:
                await _cancel_twice(drain_task, turns=3)

                # Still pending: worker not released, so drain defers.
                assert not drain_task.done()
                assert not owned.done()
                assert owned in jobs.futures

                # Release the worker: only now may the deferred cancellation
                # surface.
                release.set()
                try:
                    await asyncio.wait_for(drain_task, timeout=10.0)
                except asyncio.CancelledError:
                    pass
                else:  # pragma: no cover - cancellation MUST survive
                    raise AssertionError("deferred cancellation was swallowed")

                assert owned.done()
                assert jobs.futures == []
            finally:
                # If an assertion fired before release, nothing must hang.
                release.set()
                if not drain_task.done():
                    drain_task.cancel()
                try:
                    await drain_task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            release.set()
            driver.cancel()
            try:
                await driver
            except (asyncio.CancelledError, Exception):
                pass

    try:
        asyncio.run(scenario())
    finally:
        executor.shutdown(wait=True, cancel_futures=False)


def test_native_exception_after_release_does_not_replace_cancellation():
    """Case 2: native job raises after release while drain is cancelled."""
    AsrSlotJobs, executor = _extract_asr_slot_jobs()
    started = threading.Event()
    release = threading.Event()

    async def scenario():
        def failing_job():
            started.set()
            assert release.wait(10.0) is True
            raise RuntimeError("native backend failure")

        jobs = AsrSlotJobs()
        cf = executor.submit(failing_job)
        jobs.futures.append(cf)

        try:
            assert await _spin_until(started.is_set), "worker never started"

            drain_task = asyncio.ensure_future(jobs.drain())
            try:
                await _cancel_twice(drain_task, turns=3)
                assert not drain_task.done()

                release.set()
                try:
                    await asyncio.wait_for(drain_task, timeout=10.0)
                except asyncio.CancelledError:
                    pass
                else:  # pragma: no cover
                    raise AssertionError(
                        "cancellation replaced/deferred away by native exc"
                    )

                assert cf.done()
                assert jobs.futures == []
            finally:
                release.set()
                if not drain_task.done():
                    drain_task.cancel()
                try:
                    await drain_task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            release.set()

    try:
        asyncio.run(scenario())
    finally:
        executor.shutdown(wait=True, cancel_futures=False)


def test_future_added_mid_drain_stays_owned():
    """Case 3: a cf appended while the initial snapshot is held waits too."""
    AsrSlotJobs, executor = _extract_asr_slot_jobs()
    started_a = threading.Event()
    release_a = threading.Event()
    started_b = threading.Event()
    release_b = threading.Event()

    async def scenario():
        def job_a():
            started_a.set()
            assert release_a.wait(10.0) is True
            return "a"

        def job_b():
            started_b.set()
            assert release_b.wait(10.0) is True
            return "b"

        jobs = AsrSlotJobs()
        cf_a = executor.submit(job_a)
        jobs.futures.append(cf_a)

        try:
            assert await _spin_until(started_a.is_set), "job A never started"

            drain_task = asyncio.ensure_future(jobs.drain())
            try:
                # Let drain snapshot [cf_a] and start awaiting it.
                await asyncio.sleep(0)

                cf_b = executor.submit(job_b)
                jobs.futures.append(cf_b)
                assert await _spin_until(
                    started_b.is_set
                ), "job B never started"

                # Release only the first. Observe the REAL cf_a completion
                # (not just a few loop turns) so a broken drain that ignores a
                # newly-appended cf_b cannot pass while cf_a happens to still
                # be held. Shield the wrapped future so this wait is never
                # itself cancelled by drain's bookkeeping.
                release_a.set()
                await asyncio.wait_for(
                    asyncio.shield(asyncio.wrap_future(cf_a)), timeout=5.0
                )
                # Two event-loop turns so already-registered drain aggregate
                # callbacks can run after cf_a is truly done.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert cf_a.done(), "cf_a not actually complete"
                assert not drain_task.done(), "drain returned while cf_b active"
                assert cf_b in jobs.futures
                assert not cf_b.done()

                release_b.set()
                await asyncio.wait_for(drain_task, timeout=10.0)
                assert cf_a.done() and cf_b.done()
                assert jobs.futures == []
            finally:
                release_a.set()
                release_b.set()
                if not drain_task.done():
                    drain_task.cancel()
                try:
                    await drain_task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            release_a.set()
            release_b.set()

    try:
        asyncio.run(scenario())
    finally:
        executor.shutdown(wait=True, cancel_futures=False)


def test_normal_drain_prunes_without_cancellation():
    """Case 4: no cancellation, all finished -> refs pruned, no live future."""
    AsrSlotJobs, executor = _extract_asr_slot_jobs()

    async def scenario():
        def quick(x):
            return x * 2

        jobs = AsrSlotJobs()
        results = []
        for value in (1, 2, 3):
            cf = executor.submit(quick, value)
            jobs.futures.append(cf)
            results.append(await asyncio.wrap_future(cf))

        assert results == [2, 4, 6]
        await asyncio.wait_for(jobs.drain(), timeout=10.0)
        assert jobs.futures == []

    try:
        asyncio.run(scenario())
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
