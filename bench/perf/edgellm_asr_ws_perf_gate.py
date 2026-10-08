#!/usr/bin/env python3
"""App-scope ASR performance gate for the deployed /asr/stream and /v1/asr APIs.

This is a *driver*, not a framework: standard library first, with ``websockets``
used only for the WS mode and ``numpy`` never required. It measures the SERVICE
protocol as implemented in ``server/main.py`` and never starts, configures,
kills or signals a service.

Design contract (see /tmp/slv-v011-asr-service-perf-root-repair/DIAGNOSIS.md):

  * ``--concurrency 1`` executes ONLY phase B1: the 100 ordered utterances
    once, sequentially. ``--concurrency 2`` executes ONLY phase B2: the same
    100 utterances once, as 50 adjacent pairs ``(0,1),(2,3),...`` at width 2.
    No phase runs the other's workload, and no extra unlabelled repeated
    utterance is executed.
  * The first three rows are flagged warm and are INCLUDED in every statistic.
  * RTF is ``request_runtime_s / audio_duration_s`` (lower is better).
    Throughput is ``audio / runtime`` (higher is better). They are distinct.
  * Per-request wall is measured from the monotonic stamp taken immediately
    BEFORE the first PCM ``await send`` to the final arrival. EOS is stamped
    BEFORE its ``await send``. Negative, non-finite or bool times fail the row.
  * A WS receiver task is started immediately after connect and BEFORE the
    first PCM byte is sent. No nonempty partial ⇒ first-partial UNPROVEN.
  * B1 aggregate throughput = sum(audio)/sum(valid request walls).
    B2 aggregate throughput = sum(audio)/sum(50 actual pair spans), where a
    pair span is earliest first-PCM-send → latest own final (handshake
    excluded). The complete phase elapsed INCLUDING handshake/pacing and the
    per-pair distribution are reported separately; sum-latency throughput
    (sum audio / sum of all 100 request walls) is a distinct metric. Failed or
    missing rows invalidate the aggregate gate (NOTQUALIFIED); they are never
    dropped into a truncated-success mean.
  * One absolute request deadline bounds every connect/send/pace/EOS/reset/
    recv/probe/close await; cleanup is reserved inside the overall deadline
    and cleanup failures are recorded. There are no unbounded finally awaits
    and no minimum waits after expiry. Rows with pending/failed cleanup can
    never be ok or enter qualified aggregates. F4 ASYNC LIFETIME: every
    owned awaitable runs through an AsyncLifetimeRegistry holding the ACTUAL
    Tasks/Futures and connection/writer resources (never just names or
    self-attested flags). Bounded waits are ``asyncio.wait`` against the
    absolute remaining budget; cancellation is requested EXACTLY ONCE and
    its completion is never awaited without a finite bound; a
    cancellation-suppressing child stays registered as pending (forcing the
    row/run NOTQUALIFIED) instead of blocking the caller. Resources acquired
    AFTER a timeout by a still-owned connector remain registered and are
    closed via their supported contract when they late-complete. The CLI
    owns its event loop explicitly (no ``asyncio.run`` Runner cancel-gather
    or default-executor shutdown), performs one finite top-level wait plus
    a bounded registry drain, snapshots the ledger (regenerated AFTER
    cleanup) and only closes the loop last; loop.close() itself is never
    claimed as proof of cleanup. HONEST LIMITATION (UNPROVEN): no mechanism
    here can interrupt an unsupported SYNCHRONOUS blocking call inside a
    child, and pending-at-exit objects are retained truthfully until process
    end rather than declared kernel-closed.
  * Percentiles use the frozen 0-based rule ``sorted[min(n-1, ceil(p*n))]``.
  * Failed rows are retained; there is no subsetting or exclusion.
  * Service artifact identity comes from an explicit pinned JSON file
    (``--identity-file`` + ``--identity-file-sha256``); capabilities metadata
    alone is not artifact proof, and without pinned evidence the run is
    UNPROVEN. Nothing here fabricates resource observations.
  * OWNED INPUT BOUNDARY: all corpus input (manifest + every WAV) is admitted
    by ONE explicitly owned daemon thread before any probe or request. That
    thread is the SOLE owner of the file descriptors it opens (nonblocking
    open, fstat-verified regular files, bounded chunk reads, cooperative
    cancellation checked before each open and between chunks); no raw fd is
    ever closed from another thread and the thread is never force-stopped.
    A close is accepted as CLEAN only when ``close()`` returned AND the
    wrapper's ACTUAL ``.closed`` proof is exactly ``True``; a missing
    ``.closed`` attribute means UNKNOWN (never defaulted to True) and a
    raising ``.closed`` getter is an unproven close, so both stay registered
    as dirty ownership (no raw-fd compensation). The writer thread and every
    snapshot reader publish/read admitted items, failures, wrapper and close
    ledgers, corpus, caps and running totals under the SAME lock; large-byte
    calculations happen outside it, and no live dict is iterated
    unprotected. Both the pre-done and post-done admission polls sleep at
    most the ORIGINAL remaining deadline (no floor, no renewed budget), and
    after expiry no sleep, helper or probe starts.
    The frozen WAV bytes admitted once are the SOLE basis of the SHA256, the
    decoded PCM, the duration and the HTTP payload; the request path performs
    NO file open/read/hash/stat. Any failed input, expired input budget or
    unproven helper cleanup blocks ALL probes and inference with an explicit
    ``InputAdmissionError`` report (fixed 100-item denominator retained).
    COMPATIBILITY LIMITATION (fail-closed migration): ``_run_one`` and
    ``run_http_utterance`` no longer accept a filesystem path for input;
    path-only callers get an explicit failed row/error, never a synchronous
    re-read or timed fallback. The legacy ``load_corpus``/``read_pcm16_mono``
    helpers remain for non-timed compatibility use only.

Run ``--help`` for the parameter surface. Output directory must NOT exist.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import stat as stat_module
import statistics
import string
import sys
import threading
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional, Protocol

EXIT_NOT_RUN = 98

# ──────────────────────────────────────────────────────────────────────
# Frozen method constants
# ──────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 2

# Frozen percentile rule: sorted[min(n - 1, ceil(p * n))]. 0-based.
PERCENTILE_METHOD = "sorted[min(n-1, ceil(p*n))] 0-based"

# Expected monaural 16-bit PCM at 16 kHz.
EXPECTED_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2

# Reference word count for the frozen corpus; quality cannot qualify without
# the full corpus and this denominator.
FROZEN_MANIFEST_SHA256 = (
    "4a20ee1cecb1da7264cb8d5cd59419de656dc02ff259eceab1325296905984a6"
)
FROZEN_ITEM_COUNT = 100
# OPT-IN dual pinned corpus-input contract: the frozen original manifest
# (sha 4a20…) carries the item hashes but not the item list schema; the
# frozen DETAIL wrapper (sha b144…) carries the 100 ordered items plus the
# original manifest digest. Both pins are required together and neither can
# be bypassed by a caller-supplied pin when the detail path is opted in.
FROZEN_CORPUS_DETAIL_SHA256 = (
    "b144c34538d889c93663476857c79bd5c346c91b11a356695a20cdebd1abf493"
)
FROZEN_CORPUS_DETAIL_ORDERED_SHA_FINGERPRINT = (
    "dfba0772487f0854f90d492d73aecb9d612f67478d209b1f2ea62a026cf2086b"
)
FROZEN_REF_WORDS = 787
WARM_COUNT = 3

# Server close codes that are legitimate protocol closure, not a fault.
NORMAL_CLOSE_CODES = (1000, 1001)

# Duration check tolerance (manifest duration vs actual WAV duration).
DURATION_TOLERANCE_S = 0.01

# Cleanup (receiver cancel + close) is reserved inside the overall deadline.
CLEANUP_RESERVE_S = 1.0

# CLI-owned loop: the CLI creates ONE absolute overall deadline BEFORE the
# loop/top task exists and shares it with run_gate (optional argument), so no
# layer renews the budget. A small cleanup slice is RESERVED INSIDE that one
# deadline (never added on top), carving the top-level wait shorter so the
# final registry drain still fits before expiry; the drain grace is a CAP on
# that reserved slice, never a floor granted to expired work, and a
# non-positive remaining budget starts NO new wait/task/probe.
CLI_CLEANUP_RESERVE_S = 1.0
CLI_DRAIN_GRACE_S = 5.0

# App-internal paired thresholds (frozen gate).
B2_OVER_B1_AGGREGATE_MIN = 1.25
B2_P95_VS_B1_P95_MAX = 2.0
WER_MAX = 26.0 / 787.0 + 0.005

# App-vs-baseline guard (same app metric basis only, never native).
APP_VS_BASELINE_MAX_RATIO = 1.15

# Required fields of the pinned service identity file. base_profile_sha256
# is the common base profile every variant resolves FROM; it is part of the
# explicit schema and is NEVER inferred from profile_family or from the
# resolved profile hash.
IDENTITY_REQUIRED_FIELDS = (
    "target_device",
    "sdk_version",
    "upstream_commit",
    "worker_sha256",
    "plugin_sha256",
    "profile_family",
    "base_profile_sha256",
    "profile_sha256",
    "engine_sha256",
    "config_sha256",
    "slot_variant",
)
IDENTITY_SLOT_VARIANTS = ("baseline", "candidate")

# ONE shared explicit identity contract for BOTH comparison paths
# (compare_runs app-vs-baseline and paired_threshold_document B1-vs-B2).
# These fields MUST be equal between the two sides. A field that is missing
# or empty on BOTH sides is a mismatch too (None==None is never a match).
IDENTITY_STRICT_SHARED_FIELDS = (
    "target_device",
    "sdk_version",
    "upstream_commit",
    "worker_sha256",
    "plugin_sha256",
    "profile_family",
    "base_profile_sha256",
)

# Explicit PERMITTED differences between the two sides (variant provenance).
# They are recorded per side in the comparison result, never hidden under a
# common label. They must still be present and nonempty on each side; absent
# variant provenance keeps the comparison UNPROVEN.
IDENTITY_VARIANT_FIELDS = (
    "profile_sha256",
    "engine_sha256",
    "config_sha256",
    "slot_variant",
)

# The ACTUAL driver parser accepts ONLY these service modes
# (``--mode`` choices). Any other value in a run document is not a
# protocol identity this driver produced and can never pair/compare.
SUPPORTED_SERVICE_MODES = ("ws", "http")

# Protocol/service fields that BOTH comparison paths (compare_runs and
# paired_threshold_document) must see as present, valid and EQUAL on the
# two sides. A field missing or invalid on BOTH sides is never an
# equality match — it keeps the comparison UNPROVEN (``None == None``
# and ``[] == []`` are not protocol identity).
PROTOCOL_IDENTITY_SERVICE_FIELDS = ("asr_model_id", "asr_backend")


# ──────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Iterable[float], p: float) -> float:
    """Frozen percentile rule: ``sorted[min(n-1, ceil(p*n))]``.

    ``p`` is a fraction in (0, 1]. The index is 0-based; this deliberately does
    NOT subtract one (the older ``edgellm_asr_http_gate.py`` did).
    """
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("percentile of empty sequence")
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p!r}")
    index = min(n - 1, math.ceil(p * n))
    return ordered[index]


def summary_stats(values: list[float]) -> dict[str, Any]:
    """P50/P95/min/max/mean/count under the frozen percentile rule."""
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
            "mean": None,
            "percentile_method": PERCENTILE_METHOD,
        }
    return {
        "count": len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "percentile_method": PERCENTILE_METHOD,
    }


def _strict_int(value: Any) -> bool:
    """True only for a genuine int (bool is rejected)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_time_value(value: Any) -> bool:
    """Reject bool / non-numeric / non-finite timestamps."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _time_delta(later: Any, earlier: Any) -> tuple[Optional[float], Optional[str]]:
    """Return (delta, error). Negative/nonfinite/bool inputs are rejected."""
    for v in (later, earlier):
        if not _valid_time_value(v):
            return None, f"invalid timestamp value {v!r}"
    delta = float(later) - float(earlier)
    if delta < 0:
        return None, f"negative interval {delta}"
    return delta, None


def _remaining(deadline_mono: float) -> float:
    return deadline_mono - time.monotonic()


# ──────────────────────────────────────────────────────────────────────
# F4 owned async lifetime registry
# ──────────────────────────────────────────────────────────────────────


class AsyncLifetimeRegistry:
    """Per-run ledger owning ACTUAL async objects, not just names/statuses.

    Holds the real ``asyncio.Task``/``Future`` objects and the real
    connection/writer resources until their completion/close is OBSERVED
    (a done callback records the terminal state and retrieves completed
    exceptions so they are never silently discarded). Cancellation is
    requested exactly once per object; completion of that cancellation is
    only ever awaited with a finite ``asyncio.wait`` bound, so a
    cancellation-suppressing child can never block the caller — it stays
    registered as pending and forces the row/run NOTQUALIFIED instead.

    Snapshots contain only phases/states/exception type names — never
    transcripts, payloads or credentials. Live objects stay referenced by
    this registry until the caller observes completion or the CLI process
    lifetime ends; ``loop.close()`` is never treated as cleanup proof.
    """

    def __init__(self, label: str, parent: "Optional[AsyncLifetimeRegistry]" = None) -> None:
        self.label = label
        self.parent: Optional[AsyncLifetimeRegistry] = parent
        # task -> entry {phase, added_mono, cancel_requested, state, exception}
        self.tasks: dict[asyncio.Task, dict[str, Any]] = {}
        # id(obj) -> entry {obj, kind, label, close_state, close_detail}
        self.resources: dict[int, dict[str, Any]] = {}

    def child(self, label: str) -> "AsyncLifetimeRegistry":
        """Parameterized child scope: LOCAL tasks/resources registered in
        the child, forwarded to this (root/ancestor) aggregate with the
        SAME shared mutable entry objects. A child's ``drain`` acts LOCAL
        only (never on the caller/ancestor/siblings); the root's final
        drain stays aggregate over everything, so the CLI can still cancel
        remaining child work exactly once inside the original deadline."""
        return AsyncLifetimeRegistry(label, parent=self)

    # ── tasks ──
    def track_task(self, task: asyncio.Task, phase: str) -> asyncio.Task:
        # Reuse an ALREADY registered entry (this scope, or one forwarded by
        # a descendant/duplicate registration): never reset cancel_requested,
        # terminal state or the parent's phase for an existing task.
        entry = self.tasks.get(task)
        if entry is None:
            reg = self.parent
            while reg is not None and entry is None:
                entry = reg.tasks.get(task)
                reg = reg.parent
        if entry is not None:
            self.tasks.setdefault(task, entry)
            return task
        entry = {
            "phase": phase,
            "added_mono": time.monotonic(),
            "cancel_requested": False,
            "state": "pending",
            "exception": None,
        }
        self.tasks[task] = entry
        task.add_done_callback(self._on_done)
        if task.done():
            self._refresh_task(task)
        # Forward the SAME entry object to every ancestor so the root
        # aggregate stays complete (actual strong refs included) while the
        # local scope stays the ONLY drain scope for this helper.
        reg = self.parent
        while reg is not None:
            reg.tasks.setdefault(task, entry)
            reg = reg.parent
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        entry = self.tasks.get(task)
        if entry is not None:
            self._fill_terminal(task, entry)

    @staticmethod
    def _fill_terminal(task: asyncio.Task, entry: dict[str, Any]) -> None:
        # Retrieve the completed exception exactly once here so the loop
        # never warns "exception was never retrieved" and the outcome is
        # recorded truthfully.
        try:
            if task.cancelled():
                entry["state"] = "cancelled"
                entry["exception"] = None
            else:
                exc = task.exception()
                if exc is None:
                    entry["state"] = "done"
                    entry["exception"] = None
                else:
                    entry["state"] = "failed"
                    entry["exception"] = f"{type(exc).__name__}: {exc}"
        except asyncio.CancelledError:
            entry["state"] = "cancelled"
            entry["exception"] = None
        except Exception as exc:  # noqa: BLE001 (defensive; ledger honesty)
            entry["state"] = "failed"
            entry["exception"] = f"{type(exc).__name__}: {exc}"

    def _refresh_task(self, task: asyncio.Task) -> None:
        entry = self.tasks.get(task)
        if entry is not None and entry["state"] == "pending" and task.done():
            self._fill_terminal(task, entry)

    def note_cancel_requested(self, task: asyncio.Task, phase: str) -> None:
        entry = self.tasks.get(task)
        if entry is None:
            # Unknown registration: propagate to the whole ancestor chain
            # so a late/direct note never escapes the global ledger.
            entry = {
                "phase": phase,
                "added_mono": time.monotonic(),
                "cancel_requested": False,
                "state": "pending",
                "exception": None,
            }
            self.tasks[task] = entry
            task.add_done_callback(self._on_done)
            reg = self.parent
            while reg is not None:
                reg.tasks.setdefault(task, entry)
                reg = reg.parent
        entry["cancel_requested"] = True

    def entry_for(self, task: asyncio.Task) -> Optional[dict[str, Any]]:
        self._refresh_task(task)
        return self.tasks.get(task)

    async def wait_done(self, task: asyncio.Task, timeout_s: float) -> bool:
        """Finite observation only: never cancels, never awaits past the
        bound. Returns True iff the task was observed done."""
        if task.done():
            self._refresh_task(task)
            return True
        if timeout_s <= 0:
            return False
        _done, _pending = await asyncio.wait({task}, timeout=timeout_s)
        self._refresh_task(task)
        return task.done()

    async def cancel_observe(
        self, task: asyncio.Task, timeout_s: float, phase: str
    ) -> bool:
        """Cancel EXACTLY ONCE then finitely observe; returns True iff the
        task was actually observed done within the bound."""
        entry = self.entry_for(task)
        if not task.done():
            if entry is None or not entry.get("cancel_requested"):
                task.cancel()
            self.note_cancel_requested(task, phase)
        return await self.wait_done(task, max(0.0, timeout_s))

    async def drain(self, timeout_s: float) -> None:
        """Cancel once every still-pending task in THIS SCOPE and finitely
        observe their unwinding inside ONE shared bounded budget. The scope
        is local: a child registry's drain never touches its parent's or
        siblings' tasks (ancestors/siblings can never be cancelled by a
        helper's cleanup), while the root scope remains the full aggregate.
        The CURRENTLY EXECUTING task is excluded from both the cancel set
        and the wait set: a drain must never cancel or self-wait on its own
        caller even if that task was (accidentally) registered."""
        end = time.monotonic() + max(0.0, timeout_s)
        current = asyncio.current_task()
        scope = [t for t in self.tasks if t is not current]
        for task in scope:
            entry = self.tasks[task]
            if not task.done() and not entry["cancel_requested"]:
                task.cancel()
                entry["cancel_requested"] = True
        while True:
            pending = [t for t in scope if not t.done()]
            if not pending:
                return
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.wait(pending, timeout=remaining)

    def pending_entries(self) -> list[dict[str, Any]]:
        return [
            self.tasks[t] for t in self.tasks
            if (self._refresh_task(t) or True) and not t.done()
        ]

    # ── resources ──
    def _resource_entry(self, obj: Any) -> Optional[dict[str, Any]]:
        """The ACTUAL shared entry for obj: this scope first, then any
        ancestor it was already forwarded to (same dict object)."""
        entry = self.resources.get(id(obj))
        if entry is None:
            reg = self.parent
            while reg is not None and entry is None:
                entry = reg.resources.get(id(obj))
                reg = reg.parent
        return entry

    def track_resource(self, obj: Any, kind: str, label: str) -> None:
        # Reuse an ALREADY registered entry (never reopen a closed resource
        # or overwrite its close ledger by re-tracking). Ownership
        # callback-before-claim can register the SAME entry first as the
        # provisional ``kind + "-late"`` label; when the caller then
        # SUCCESSFULLY claims the resource with the normal ``kind``, promote
        # that exact entry back to the normal kind so successful owned
        # resources are not permanently mislabeled as late-acquired. This
        # only ever promotes the matching ``kind + "-late"`` provisional
        # label: the shared entry object, its ``obj`` reference, and the
        # close_state/close_detail ledger are left untouched, a normal kind
        # is never downgraded to ``-late``, and unrelated kinds are never
        # rewritten.
        entry = self._resource_entry(obj)
        if entry is not None:
            if entry.get("kind") == f"{kind}-late" and kind != f"{kind}-late":
                # Promote provisional late label on a successful normal
                # claim. close_state/close_detail and the actual obj ref are
                # deliberately preserved (no reset, no downgrade).
                entry["kind"] = kind
            self.resources.setdefault(id(obj), entry)
            return
        entry = {
            "obj": obj,  # ACTUAL live reference retained, not just a name
            "kind": kind,
            "type": type(obj).__name__,
            "label": label,
            "close_state": "open",
            "close_detail": None,
        }
        self.resources[id(obj)] = entry
        # Same shared entry in every ancestor: root updates are immediate
        # and the root keeps the actual strong ref after the helper returns.
        reg = self.parent
        while reg is not None:
            reg.resources.setdefault(id(obj), entry)
            reg = reg.parent

    def note_resource_closed(self, obj: Any, ok: bool, detail: str) -> None:
        entry = self._resource_entry(obj)
        if entry is None:
            self.track_resource(obj, "unknown", "late-discovered")
            entry = self.resources[id(obj)]
        entry["close_state"] = "closed" if ok else "close_failed"
        entry["close_detail"] = detail

    def note_resource_close_requested(self, obj: Any, detail: str) -> None:
        entry = self._resource_entry(obj)
        if entry is not None and entry["close_state"] == "open":
            entry["close_state"] = "close_requested"
            entry["close_detail"] = detail

    def open_resources(self) -> list[dict[str, Any]]:
        return [e for e in self.resources.values() if e["close_state"] != "closed"]

    def snapshot(self) -> dict[str, Any]:
        """Regenerated on demand (AFTER cleanup): phases, terminal states and
        exception type names only — no transcripts, payloads or secrets."""
        tasks = []
        for task, entry in self.tasks.items():
            self._refresh_task(task)
            entry = self.tasks[task]
            name = getattr(task, "get_name", None)
            tasks.append(
                {
                    "phase": entry["phase"],
                    "task_name": name() if callable(name) else None,
                    "done": task.done(),
                    "state": entry["state"],
                    "cancel_requested": entry["cancel_requested"],
                    "exception": entry["exception"],
                }
            )
        resources = [
            {
                "kind": e["kind"],
                "type": e["type"],
                "label": e["label"],
                "close_state": e["close_state"],
                "close_detail": e["close_detail"],
            }
            for e in self.resources.values()
        ]
        pending = [t for t in tasks if not t["done"]]
        return {
            "label": self.label,
            "task_count": len(tasks),
            "pending_count": len(pending),
            "pending_phases": sorted({t["phase"] for t in pending}),
            "tasks": tasks,
            "resources": resources,
            "open_resource_count": sum(
                1 for r in resources if r["close_state"] != "closed"
            ),
        }


class _PollTimeout(Exception):
    """Internal: a bounded queue poll expired (distinct from row failure)."""


async def _poll_queue(
    q: "asyncio.Queue",
    timeout_s: float,
    registry: AsyncLifetimeRegistry,
) -> Any:
    """One finitely-bounded ``q.get`` poll through the registry.

    Uses ``asyncio.wait`` (never wait_for): on expiry the getter is cancelled
    EXACTLY ONCE and its unwinding is observed with a small finite bound;
    if that observation also fails the getter stays registered as pending
    instead of being awaited forever.
    """
    getter = asyncio.ensure_future(q.get())
    registry.track_task(getter, "frame poll")
    try:
        done, _pending = await asyncio.wait({getter}, timeout=max(0.0, timeout_s))
    except asyncio.CancelledError:
        # External caller cancellation: propagate, but the getter is still
        # owned — cancel once and keep it registered.
        if not getter.done():
            getter.cancel()
        registry.note_cancel_requested(getter, "frame poll")
        raise
    if getter in done:
        return getter.result()  # raises a stored exception if any
    getter.cancel()
    registry.note_cancel_requested(getter, "frame poll")
    # Bounded single-step observation of the cancellation; an unobserved
    # getter is left registered as pending (never awaited without a bound).
    await registry.wait_done(getter, min(0.05, max(0.0, timeout_s)))
    raise _PollTimeout


async def _bounded(
    awaitable: Awaitable[Any],
    deadline_mono: float,
    what: str,
    registry: Optional[AsyncLifetimeRegistry] = None,
) -> Any:
    """Await ``awaitable`` bounded by the remaining absolute deadline.

    No minimum wait is granted after expiry: a non-positive remaining budget
    raises immediately WITHOUT scheduling new work (an unscheduled coroutine
    object is closed in place so no phantom task and no warning is created).

    The awaitable is owned by ``registry`` (when given) as an ACTUAL task;
    on timeout cancellation is requested exactly once and its completion is
    NOT awaited here — a cancellation-suppressing child stays registered as
    pending and surfaces through the ledger instead of blocking the caller.
    External caller cancellation is propagated after the same single cancel
    request, so caller semantics are preserved.
    """
    remaining = _remaining(deadline_mono)
    if remaining <= 0:
        if asyncio.iscoroutine(awaitable):
            # Never scheduled: close the bare coroutine safely (this does
            # NOT start a nonexistent task).
            awaitable.close()
        raise TimeoutError(f"shared deadline exceeded before {what}")
    task = asyncio.ensure_future(awaitable)
    if registry is not None:
        registry.track_task(task, what)
    try:
        done, _pending = await asyncio.wait({task}, timeout=remaining)
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
        if registry is not None:
            registry.note_cancel_requested(task, what)
        raise
    if task not in done:
        task.cancel()  # exactly once; completion NOT awaited here
        if registry is not None:
            registry.note_cancel_requested(task, what)
        raise TimeoutError(f"shared deadline exceeded during {what}")
    # Completed: retrieving the result also surfaces a stored exception.
    return task.result()


async def _connect_owned(
    conn_factory: Callable[[], Awaitable[Any]],
    deadline_mono: float,
    what: str,
    registry: AsyncLifetimeRegistry,
    *,
    kind: str,
    late_close: Optional[Callable[[Any], Any]] = None,
    cleanup_deadline_mono: Optional[float] = None,
) -> Any:
    """Owned connector bounded by the absolute deadline.

    The ownership done-callback is registered IMMEDIATELY after the actual
    connector task is created, BEFORE any wait/cancel branch, so a late-
    acquired resource is retained on EVERY abandonment path (timeout AND
    external caller cancellation). Cancellation is requested EXACTLY ONCE
    across both branches. ``cleanup_deadline_mono`` is the caller's OWN
    absolute cleanup budget, reserved inside its one overall deadline; it
    is never renewed or extended here, and callers without a reserved
    cleanup slice fall back to the request deadline itself (no synthetic
    new grace). A connector that completes after the caller abandoned it
    keeps its ACTUAL resource registered; within the cleanup budget
    EXACTLY ONE supported close attempt is made and its outcome is
    honestly ledgered: a coroutine close must be observed complete, and a
    synchronous close return WITHOUT an observation contract stays
    UNPROVEN (close_requested), never default-closed. After the cleanup
    deadline NO new async task is created and no close coroutine is even
    instantiated: the resource stays registered pending/UNPROVEN.
    """
    remaining = _remaining(deadline_mono)
    if remaining <= 0:
        raise TimeoutError(f"shared deadline exceeded before {what}")
    if cleanup_deadline_mono is None:
        # Callers without a reserved cleanup slice: cleanup shares the
        # request deadline exactly (no synthetic new grace).
        cleanup_deadline_mono = deadline_mono
    # Shared caller/callback ownership state: the callback records the
    # resource FIRST; the caller then claims it (normal path) or has
    # already abandoned it (timeout/external cancel), in which case the
    # callback performs the single close-or-retain decision.
    state: dict[str, Any] = {
        "claimed": False,
        "abandoned": False,
        "cancel_requested": False,
        "close_attempted": False,
    }
    task = registry.track_task(asyncio.ensure_future(conn_factory()), what)

    def _cancel_once() -> None:
        if not task.done() and not state["cancel_requested"]:
            task.cancel()
        state["cancel_requested"] = True
        registry.note_cancel_requested(task, what)

    def _abandoned_close_or_retain(resource: Any) -> None:
        # Idempotent shared ownership/close decision for an abandoned
        # late-acquired resource. Called from the done-callback AND from
        # the external-cancel branch when the callback has ALREADY run
        # (callback-before-cancel race): the caller's abandonment must
        # still reenter the close decision even though the callback's own
        # abandoned check saw the not-yet-abandoned state. The guard
        # prevents re-tracking (which would reset close_state) and any
        # double close attempt / double cancel.
        if state["close_attempted"]:
            return
        state["close_attempted"] = True
        if id(resource) not in registry.resources:
            registry.track_resource(resource, f"{kind}-late", what)
        if late_close is None:
            registry.note_resource_closed(
                resource, False,
                "late-acquired resource without a supported close contract",
            )
            return
        if _remaining(cleanup_deadline_mono) <= 0:
            # No NEW async task after the cleanup deadline and no close
            # coroutine is instantiated (an unstarted coroutine would
            # need disposal): the ACTUAL resource stays registered
            # pending/UNPROVEN, never inventedly closed.
            registry.note_resource_closed(
                resource, False,
                "late close not started: cleanup deadline expired "
                "(resource retained, close UNPROVEN)",
            )
            return
        outcome = late_close(resource)
        if asyncio.iscoroutine(outcome):
            if _remaining(cleanup_deadline_mono) <= 0:
                # Boundary B: a synchronous close initializer may have
                # consumed the remaining cleanup budget before returning
                # this UNSTARTED coroutine. Re-check the caller's OWN
                # absolute deadline immediately BEFORE ensure_future: no
                # new task is created; the unstarted coroutine is safely
                # disposed and the ACTUAL resource stays retained with a
                # truthful UNPROVEN close.
                outcome.close()
                registry.note_resource_closed(
                    resource, False,
                    "late close not started: cleanup deadline expired "
                    "after synchronous close initializer "
                    "(resource retained, close UNPROVEN)",
                )
                return
            close_task = registry.track_task(
                asyncio.ensure_future(outcome), f"late {what} close"
            )
            registry.note_resource_close_requested(
                resource, "late close scheduled"
            )

            def _late_close_done(
                ct: asyncio.Task, res: Any = resource
            ) -> None:
                if ct.cancelled():
                    return  # stays close_requested/pending in the ledger
                exc = ct.exception()
                if exc is None:
                    registry.note_resource_closed(
                        res, True, "late close observed complete"
                    )
                else:
                    registry.note_resource_closed(
                        res, False,
                        f"late close failed: {type(exc).__name__}: {exc}",
                    )

            close_task.add_done_callback(_late_close_done)
        else:
            # Synchronous close initiation returned WITHOUT an
            # observation contract (e.g. an HTTP writer exposing no
            # supported wait_closed): a close was initiated but the
            # actual close is UNPROVEN — close_requested, never
            # default-closed.
            registry.note_resource_close_requested(
                resource,
                "late synchronous close returned without an observation "
                "contract (close UNPROVEN)",
            )

    def _late_complete(late: asyncio.Task) -> None:
        resource: Any = None
        acquired = False
        try:
            if late.cancelled():
                return  # no live resource acquired
            # Always retrieve the completed exception so it is never
            # silently discarded; a failed connect produced no resource.
            if late.exception() is not None:
                return
            resource = late.result()
            acquired = True
            if state["claimed"]:
                return  # the caller owns the resource on the normal path
            # Record the ACTUAL resource FIRST so callback-before-wait and
            # external-cancel races can never lose ownership. Idempotent:
            # if the external-cancel branch already ran the shared decision
            # before this scheduled callback, the id-keyed entry exists —
            # re-tracking would RESET its close ledger state
            # (close_requested/closed/close_failed) back to open, so only
            # track when genuinely absent.
            if id(resource) not in registry.resources:
                registry.track_resource(resource, f"{kind}-late", what)
            if not state["abandoned"]:
                # Completed before the caller's wait observed it; the
                # caller claims it on the normal path. Retain, never close
                # early.
                return
            # Abandoned (timeout or external cancellation): close-or-retain
            # EXACTLY ONCE inside the caller's own cleanup budget via the
            # SHARED idempotent decision.
            _abandoned_close_or_retain(resource)
        except Exception as exc:  # noqa: BLE001 (loop closed/contract error)
            # Ledger honesty: a callback-time failure is recorded against
            # the acquired resource, never silently swallowed.
            if acquired:
                registry.note_resource_closed(
                    resource, False,
                    f"late close unschedulable/failed: {type(exc).__name__}: {exc}",
                )

    task.add_done_callback(_late_complete)  # BEFORE any wait/cancel branch
    try:
        done, _pending = await asyncio.wait({task}, timeout=remaining)
    except asyncio.CancelledError:
        # External caller cancellation: abandon but KEEP ownership — the
        # done callback is already registered and retains/closes a
        # late-acquired resource.
        state["abandoned"] = True
        if (task.done() and not task.cancelled()
                and task.exception() is None):
            resource = task.result()
            # Callback-before-cancel race: the ownership callback ALREADY
            # ran while the caller was not yet abandoned, so it tracked the
            # resource but never reached the close decision. Run the SAME
            # idempotent ownership/close decision now (guard prevents
            # re-tracking, double attempts and double cancel). A raising
            # synchronous close initializer must NEVER mask the original
            # caller cancellation: ledger the failure against the ACTUAL
            # resource, then still propagate CancelledError.
            try:
                _abandoned_close_or_retain(resource)
            except Exception as exc:  # noqa: BLE001 (ledger honesty)
                if id(resource) in registry.resources:
                    registry.note_resource_closed(
                        resource, False,
                        f"late close unschedulable/failed: "
                        f"{type(exc).__name__}: {exc}",
                    )
        _cancel_once()
        raise
    if task in done:
        state["claimed"] = True
        resource = task.result()  # raises a stored connect exception
        registry.track_resource(resource, kind, what)
        return resource
    state["abandoned"] = True
    _cancel_once()  # exactly once; completion NEVER awaited here
    raise TimeoutError(f"shared deadline exceeded during {what}")


def normalize_text(text: str) -> str:
    """Frozen normalization: lowercase, strip string.punctuation, split.

    Provenance: /tmp/slv-v011-asr-baseline/wer-stats.json
    (``"lowercase + strip all string.punctuation + whitespace split"``).
    """
    lowered = str(text).lower()
    stripped = "".join(ch for ch in lowered if ch not in string.punctuation)
    return " ".join(stripped.split())


def _levenshtein(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Word-level Levenshtein returning (total, substitutions, deletions,
    insertions) using the standard DP with backtrace counts."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, subs, dels, inss)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        cost, s, d, ins = dp[i - 1][0]
        dp[i][0] = (cost + 1, s, d + 1, ins)
    for j in range(1, m + 1):
        cost, s, d, ins = dp[0][j - 1]
        dp[0][j] = (cost + 1, s, d, ins + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub = dp[i - 1][j - 1]
            dele = dp[i - 1][j]
            ins = dp[i][j - 1]
            best = min(sub[0], dele[0], ins[0])
            if sub[0] == best:
                dp[i][j] = (best + 1, sub[1] + 1, sub[2], sub[3])
            elif dele[0] == best:
                dp[i][j] = (best + 1, dele[1], dele[2] + 1, dele[3])
            else:
                dp[i][j] = (best + 1, ins[1], ins[2], ins[3] + 1)
    return dp[n][m]


def word_error_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """Frozen word-level WER with S/D/I breakdown."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    total, subs, dels, inss = _levenshtein(ref, hyp)
    ref_words = len(ref)
    wer = (total / ref_words) if ref_words else None
    return {
        "normalized_reference": " ".join(ref),
        "normalized_hypothesis": " ".join(hyp),
        "ref_words": ref_words,
        "errors": total,
        "substitutions": subs,
        "deletions": dels,
        "insertions": inss,
        "wer": wer,
    }


# ──────────────────────────────────────────────────────────────────────
# Corpus manifest
# ──────────────────────────────────────────────────────────────────────


@dataclass
class CorpusItem:
    order: int
    path: str
    file: str
    sha256: str
    rate: int
    channels: int
    bit_depth: int
    duration_s: float
    transcript: str
    lang: str
    id: str


@dataclass
class Corpus:
    manifest_sha256: str
    manifest_path: str
    corpus_dir_declared: str
    items: list[CorpusItem]
    # OPT-IN dual-pinned detail provenance (set ONLY by the owned loader when
    # --corpus-detail is used; legacy direct-item loading leaves these at
    # their defaults so existing constructors are unaffected).
    detail_sha256: Optional[str] = None
    detail_path: Optional[str] = None
    detail_ordered_fingerprint: Optional[str] = None
    source_verified: bool = False

    @property
    def ref_words(self) -> int:
        return sum(len(normalize_text(it.transcript).split()) for it in self.items)


class ManifestError(Exception):
    pass


def load_corpus(
    manifest_path: Path,
    manifest_sha256: str,
    corpus_root: Optional[Path] = None,
) -> Corpus:
    """Load the frozen manifest, verify its SHA256, and freeze item order.

    COMPATIBILITY helper only: the timed gate admits all input through the
    owned thread (``admit_corpus_inputs``) instead. The manifest hash MUST
    match ``manifest_sha256`` (the caller passes the frozen value by default).
    Item order is the manifest's declared order; it is never re-sorted by the
    driver (the baseline frozen order is authoritative and the frozen
    English100 ordering must match exactly for the comparator).
    """
    if not manifest_path.is_file():
        raise ManifestError(f"manifest not found: {manifest_path}")
    fd = os.open(str(manifest_path), os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    st = os.fstat(fd)  # fstat the ACTUAL opened descriptor
    if not stat_module.S_ISREG(st.st_mode):
        os.close(fd)  # same thread, before any wrapper is handed out
        raise ManifestError(f"manifest is not a regular file: {manifest_path}")
    chunks = []
    total = 0
    with open(fd, "rb", closefd=True) as handle:
        while True:
            block = handle.read(INPUT_READ_CHUNK_BYTES)
            if not block:
                break
            total += len(block)
            if total > INPUT_MANIFEST_MAX_BYTES:
                raise ManifestError(
                    f"manifest exceeds {INPUT_MANIFEST_MAX_BYTES} bytes"
                )
            chunks.append(block)
    payload_bytes = b"".join(chunks)
    actual = hashlib.sha256(payload_bytes).hexdigest()
    if actual != manifest_sha256:
        raise ManifestError(
            f"manifest sha256 mismatch: got {actual}, want {manifest_sha256}"
        )
    return _corpus_from_payload(payload_bytes, manifest_path, actual, corpus_root)


@dataclass
class WavCheck:
    ok: bool
    sha256: str
    rate: int
    channels: int
    bit_depth: int
    duration_s: float
    pcm_bytes: int
    error: Optional[str] = None


def read_pcm16_mono(path: Path) -> tuple[bytes, WavCheck]:
    """Read a WAV as raw little-endian PCM16, validating mono 16 kHz 16-bit.

    A validation failure is returned as a failed check (not raised) so the row
    can be retained exactly like any other failed row.
    """
    digest = sha256_file(path) if path.is_file() else ""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.getnframes()
            pcm = handle.readframes(frames)
    except Exception as exc:  # noqa: BLE001 - report, never crash a run
        return b"", WavCheck(
            ok=False,
            sha256=digest,
            rate=0,
            channels=0,
            bit_depth=0,
            duration_s=0.0,
            pcm_bytes=0,
            error=f"wav read failed: {type(exc).__name__}: {exc}",
        )
    duration = frames / float(rate) if rate else 0.0
    errors: list[str] = []
    if channels != EXPECTED_CHANNELS:
        errors.append(f"channels={channels} (want {EXPECTED_CHANNELS})")
    if width != EXPECTED_SAMPLE_WIDTH:
        errors.append(f"sample_width={width} (want {EXPECTED_SAMPLE_WIDTH})")
    if rate != EXPECTED_RATE:
        errors.append(f"rate={rate} (want {EXPECTED_RATE})")
    return pcm, WavCheck(
        ok=not errors,
        sha256=digest,
        rate=rate,
        channels=channels,
        bit_depth=width * 8,
        duration_s=duration,
        pcm_bytes=len(pcm),
        error="; ".join(errors) if errors else None,
    )


# ──────────────────────────────────────────────────────────────────────
# Owned input admission (ONE daemon thread, sole fd owner)
# ──────────────────────────────────────────────────────────────────────

INPUT_READ_CHUNK_BYTES = 1 << 20
INPUT_MANIFEST_MAX_BYTES = 8 << 20
INPUT_WAV_HEADER_MARGIN_BYTES = 64 << 10
INPUT_HARD_MAX_ITEM_BYTES = 256 << 20
INPUT_HARD_MAX_AGGREGATE_BYTES = 1 << 30
INPUT_POLL_INTERVAL_S = 0.02

# Rationale for the derived caps: they are computed from the frozen corpus
# manifest metadata actually loaded in the owning thread (duration x rate x
# channels x bytes-per-sample per item, plus a container margin), clamped
# under hard ceilings. The frozen English100 metadata (manifest detail sha256
# b144c345…, 100 items) sums to 296.44 s ≈ 9,486,080 PCM bytes with a 3.98 s
# largest item, so the derived caps admit the genuine frozen corpus with a
# wide margin. The physical WAV files live on the device host, not here; the
# actual on-disk size is therefore reported as metadata-derived, never
# claimed as a physical measurement.
INPUT_CAPS_RATIONALE = (
    "per-item and aggregate caps are DERIVED from the manifest's own "
    "duration/rate/channels/bit-depth metadata (expected PCM bytes plus a "
    "64 KiB container margin per file), clamped under hard ceilings of "
    "256 MiB/item and 1 GiB aggregate; the frozen English100 metadata sums "
    "to 296.44 s (approx 9,486,080 PCM bytes, max item 3.98 s), so genuine "
    "frozen items fit with wide margin"
)


class InputAdmissionError(Exception):
    """Raised instead of ANY probe or inference when owned input admission
    is not fully proven (failed input, pending helper, or expired budget).
    Carries the full evidence report; never a silent empty success."""

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__(str(report.get("summary") or "input admission failed"))
        self.report = report


@dataclass
class AdmittedInput:
    """Immutable per-item input admitted once by the owned thread. The frozen
    ``wav_bytes`` are the sole basis of SHA256, PCM decode, duration and the
    HTTP payload; nothing re-reads the path afterwards."""

    item: CorpusItem
    wav_bytes: bytes
    pcm: bytes
    check: WavCheck


def _corpus_from_payload(
    payload: bytes,
    manifest_path: Any,
    manifest_sha256: str,
    corpus_root: Optional[Path],
) -> Corpus:
    """Parse manifest BYTES (already hash-verified by the caller) into a
    frozen-order :class:`Corpus`. Shared by the owned thread and the legacy
    ``load_corpus`` compatibility helper."""
    doc = json.loads(payload.decode("utf-8"))
    raw_items = doc.get("items")
    if not isinstance(raw_items, list):
        raise ManifestError("manifest has no 'items' list")
    declared_dir = str(doc.get("corpus_dir") or "")
    root = Path(corpus_root) if corpus_root is not None else Path(declared_dir)
    items: list[CorpusItem] = []
    for order, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ManifestError(f"manifest item {order} is not an object")
        file_name = str(raw.get("file") or "")
        if not file_name:
            raise ManifestError(f"manifest item {order} missing 'file'")
        path = str(root / file_name) if str(root) else str(raw.get("path") or "")
        items.append(
            CorpusItem(
                order=order,
                path=path,
                file=file_name,
                sha256=str(raw.get("sha256") or ""),
                rate=int(raw.get("rate") or 0),
                channels=int(raw.get("channels") or 0),
                bit_depth=int(raw.get("bit_depth") or 0),
                duration_s=float(raw.get("duration_s") or 0.0),
                transcript=str(raw.get("transcript") or ""),
                lang=str(raw.get("lang") or ""),
                id=str(raw.get("id") or ""),
            )
        )
    return Corpus(
        manifest_sha256=manifest_sha256,
        manifest_path=str(manifest_path),
        corpus_dir_declared=declared_dir,
        items=items,
    )


def _pcm_from_wav_bytes(data: bytes, sha256_hex: str) -> tuple[bytes, WavCheck]:
    """Decode PCM16 mono 16 kHz from the ALREADY-READ frozen bytes (never
    from the path), exactly ONCE per item. Validation failures return a
    failed check (with empty PCM) so the row is retained like any other
    failed row."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.getnframes()
            pcm = handle.readframes(frames)
    except Exception as exc:  # noqa: BLE001 - report, never crash a run
        return b"", WavCheck(
            ok=False, sha256=sha256_hex, rate=0, channels=0, bit_depth=0,
            duration_s=0.0, pcm_bytes=0,
            error=f"wav read failed: {type(exc).__name__}: {exc}",
        )
    if not (_strict_int(rate) and rate > 0):
        return b"", WavCheck(
            ok=False, sha256=sha256_hex, rate=rate, channels=channels,
            bit_depth=width * 8, duration_s=0.0, pcm_bytes=len(pcm),
            error=f"invalid WAV rate: {rate!r}",
        )
    duration = frames / float(rate)
    if not math.isfinite(duration) or duration < 0:
        return b"", WavCheck(
            ok=False, sha256=sha256_hex, rate=rate, channels=channels,
            bit_depth=width * 8, duration_s=0.0, pcm_bytes=len(pcm),
            error=f"invalid WAV duration: {duration!r}",
        )
    errors: list[str] = []
    expected_pcm_bytes = frames * channels * width
    if len(pcm) != expected_pcm_bytes:
        # ``wave.readframes`` can return a SHORT payload for a truncated
        # file WITHOUT raising; the header frame count alone is not proof.
        errors.append(
            f"truncated PCM: got {len(pcm)} bytes, header declares "
            f"{expected_pcm_bytes} ({frames} frames x {channels}ch x {width}B)"
        )
    if channels != EXPECTED_CHANNELS:
        errors.append(f"channels={channels} (want {EXPECTED_CHANNELS})")
    if width != EXPECTED_SAMPLE_WIDTH:
        errors.append(f"sample_width={width} (want {EXPECTED_SAMPLE_WIDTH})")
    if rate != EXPECTED_RATE:
        errors.append(f"rate={rate} (want {EXPECTED_RATE})")
    return pcm, WavCheck(
        ok=not errors,
        sha256=sha256_hex,
        rate=rate,
        channels=channels,
        bit_depth=width * 8,
        duration_s=duration,
        pcm_bytes=len(pcm),
        error="; ".join(errors) if errors else None,
    )


def _default_owned_open(path: str):
    """Open ``path`` on the OWNING thread with O_NONBLOCK (a FIFO open can
    never hang), fstat the ACTUAL opened descriptor and require a regular
    file. Returns the wrapper whose ``close`` the owning thread alone will
    call; the raw fd is never closed from another thread. If the buffered
    wrapper cannot be created, the still-owned raw fd is closed HERE on the
    owning thread before the failure propagates (no leaked descriptor, no
    lost ownership)."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        st = os.fstat(fd)
        if not stat_module.S_ISREG(st.st_mode):
            raise ValueError(f"input is not a regular file (fstat): {path}")
    except BaseException:
        os.close(fd)  # same thread, before the wrapper is handed out
        raise
    try:
        return open(fd, "rb", closefd=True)
    except BaseException:
        os.close(fd)  # same thread: wrapper creation failed, fd still owned
        raise


def _derive_input_caps(items: list[CorpusItem]) -> dict[str, Any]:
    """Derive bounded per-item/aggregate caps from the ACTUAL manifest
    metadata, clamped under explicit hard ceilings (see
    ``INPUT_CAPS_RATIONALE``)."""
    per_item = 0
    total = 0
    total_pcm = 0
    max_dur = 0.0
    for it in items:
        expected = int(
            math.ceil(
                max(it.duration_s, 0.0)
                * max(it.rate, 0)
                * max(it.channels, 0)
                * max(it.bit_depth // 8, 0)
            )
        )
        total_pcm += expected
        max_dur = max(max_dur, it.duration_s)
        bounded = expected + INPUT_WAV_HEADER_MARGIN_BYTES
        per_item = max(per_item, bounded)
        total += bounded
    return {
        "per_item_max_bytes": min(per_item, INPUT_HARD_MAX_ITEM_BYTES),
        "aggregate_max_bytes": min(total, INPUT_HARD_MAX_AGGREGATE_BYTES),
        "hard_max_item_bytes": INPUT_HARD_MAX_ITEM_BYTES,
        "hard_max_aggregate_bytes": INPUT_HARD_MAX_AGGREGATE_BYTES,
        "derived_expected_pcm_bytes": total_pcm,
        "derived_max_item_duration_s": max_dur,
        "item_count": len(items),
        "rationale": INPUT_CAPS_RATIONALE,
    }


class OwnedInputLoader(threading.Thread):
    """The ONE explicitly owned daemon input thread.

    Sole owner of every file descriptor it opens: it opens, reads in bounded
    chunks, validates, caches the frozen bytes, and closes its OWN wrappers.
    No raw fd is ever closed from another thread and the thread is never
    force-stopped; ``cancel_event`` is cooperative only (checked before each
    open and between bounded chunks) and setting it is NEVER reported as a
    cleanup claim. ``admitted`` maps item order -> AdmittedInput so the timed
    phases never touch the filesystem.
    """

    def __init__(self, config: "RunConfig", *, opener: Any = None) -> None:
        super().__init__(name="owned-input-loader", daemon=True)
        self.config = config
        self._opener = opener or _default_owned_open
        self.cancel_event = threading.Event()
        self.done_event = threading.Event()
        # Shared publication state. EVERY reader (snapshot/report/run_gate)
        # and the writer thread publish/read these under the SAME _lock, so a
        # reader never iterates a live dict being mutated and never observes
        # a half-published corpus/caps/totals view.
        self.corpus: Optional[Corpus] = None
        self.caps: dict[str, Any] = {}
        self.admitted: dict[int, AdmittedInput] = {}
        self.failures: list[dict[str, Any]] = []
        self._open_wrappers: list[Any] = []  # still-open wrappers = pending evidence
        self._close_errors: list[str] = []  # unproven/failed close outcomes
        # Accumulated admitted totals start at their MATHEMATICAL ZERO, not
        # None: from construction onward every coherent snapshot (caps
        # included) pairs zero admitted items with zero admitted bytes/audio.
        # publish_admitted() keeps them atomic with the admitted item under
        # the SAME lock, so no reader ever sees a torn admitted/totals view.
        self.admitted_audio_s: float = 0.0
        self.admitted_bytes: int = 0
        self._lock = threading.Lock()
        self.started_mono = float("nan")
        self.finished_mono = float("nan")

    # wrapper registry (ownership ledger) --------------------------------

    def _register(self, wrapper: Any) -> None:
        with self._lock:
            self._open_wrappers.append(wrapper)

    def _unregister(self, wrapper: Any) -> None:
        with self._lock:
            try:
                self._open_wrappers.remove(wrapper)
            except ValueError:
                pass

    def publish_admitted(
        self, order: int, admitted: AdmittedInput, audio_s: float, total_bytes: int
    ) -> None:
        """Writer-side publication of one admitted item AND the running
        totals under the SAME ``_lock`` every reader uses. A concurrent
        ``snapshot()`` therefore sees either the whole prior state or the
        whole new state, never a torn admitted/totals view."""
        with self._lock:
            self.admitted[order] = admitted
            self.admitted_audio_s = audio_s
            self.admitted_bytes = total_bytes

    def publish_corpus(self, corpus: Corpus, caps: dict[str, Any]) -> None:
        """Atomically publish the parsed corpus and its derived caps so a
        concurrent reader never observes a corpus without its caps."""
        with self._lock:
            self.corpus = corpus
            self.caps = caps

    def open_wrappers(self) -> list[Any]:
        with self._lock:
            return list(self._open_wrappers)

    def close_errors(self) -> list[str]:
        with self._lock:
            return list(self._close_errors)

    def note_failure(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.failures.append(entry)

    def snapshot(self) -> dict[str, Any]:
        """Synchronized immutable snapshot of the live admission state.

        The writer thread and every reader publish/read ``admitted``,
        ``failures``, ``_open_wrappers``, ``_close_errors``, ``corpus``,
        ``caps`` and the running totals under the SAME ``_lock``. This
        method copies the REFERENCES under that lock (no bytes are copied
        while holding it), so a reader can never hit ``RuntimeError: dict
        changed size during iteration`` and never observes a half-updated
        view. ``AdmittedInput`` values and the ``bytes`` they reference are
        never mutated after construction, so the returned shallow copies stay
        coherent. The (potentially large) byte length calculation happens
        OUTSIDE the lock, in the caller/report layer.
        """
        with self._lock:
            admitted = dict(self.admitted)
            open_wrappers = list(self._open_wrappers)
            close_errors = list(self._close_errors)
            failures = list(self.failures)
            corpus = self.corpus
            caps = dict(self.caps)
            admitted_audio_s = self.admitted_audio_s
            admitted_bytes = self.admitted_bytes
        return {
            "admitted": admitted,
            "open_wrappers": open_wrappers,
            "close_errors": close_errors,
            "failures": failures,
            "corpus": corpus,
            "caps": caps,
            "admitted_audio_s": admitted_audio_s,
            "admitted_bytes": admitted_bytes,
        }

    def cleanup_proven(self) -> bool:
        """Actual cleanup proof: the helper thread has EXITED (not merely set
        done_event), EVERY wrapper it opened is unregistered, and NO close
        attempt failed or stayed unproven. Thread death alone is NOT file-
        closed proof; the close ledger must also be clean."""
        snap = self.snapshot()
        return not self.is_alive() and not snap["open_wrappers"] and not snap["close_errors"]

    def _close_owned(self, wrapper: Any) -> None:
        """Sole close path: only the owning thread calls this, exactly once
        per wrapper (no retry, no duplicate closer, no cross-thread raw-fd
        manipulation). The wrapper is UNREGISTERED only when close() returned
        without error AND the wrapper's ACTUAL ``.closed`` proof is exactly
        ``True``. A MISSING ``.closed`` attribute is UNKNOWN, never a default
        True; a ``.closed`` property that RAISES is also an unproven close.
        On failure/unproven/unknown close the ACTUAL wrapper object STAYS
        registered — ownership reference is retained, the error is ledgered,
        and admission can never claim cleanup. No raw-fd manipulation is ever
        attempted to compensate for missing proof."""
        try:
            wrapper.close()
        except BaseException as exc:  # noqa: BLE001 - ledger, never crash
            with self._lock:
                self._close_errors.append(
                    f"close error: {type(exc).__name__}: {exc} ({wrapper!r})"
                )
            return  # keep registered: dirty object, ownership retained
        try:
            closed_attr = getattr(wrapper, "closed")
        except BaseException as exc:  # noqa: BLE001 - proof getter failed
            # A raising ``.closed`` property means the close outcome is
            # UNKNOWN, not clean: treat as dirty ownership, keep registered.
            with self._lock:
                self._close_errors.append(
                    f"close not proven (closed getter raised "
                    f"{type(exc).__name__}: {exc}) ({wrapper!r})"
                )
            return
        if closed_attr is True:
            self._unregister(wrapper)
        else:
            with self._lock:
                self._close_errors.append(
                    f"close not proven (closed={closed_attr!r}) ({wrapper!r})"
                )
            # keep registered: close outcome unknown, ownership retained

    def _read_bounded(self, wrapper: Any, cap: int, label: str) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            if self.cancel_event.is_set():
                raise OSError(f"cooperative input cancel before chunk ({label})")
            block = wrapper.read(INPUT_READ_CHUNK_BYTES)
            if not block:
                break
            total += len(block)
            if total > cap:
                raise ValueError(f"input cap exceeded for {label}: >{cap} bytes")
            chunks.append(block)
        return b"".join(chunks)

    def _load_all(self) -> None:
        cfg = self.config
        # 1. Manifest: bounded read, hash of the SAME bytes that are parsed.
        wrapper = self._opener(str(cfg.manifest))
        self._register(wrapper)
        try:
            manifest_bytes = self._read_bounded(
                wrapper, INPUT_MANIFEST_MAX_BYTES, "manifest"
            )
        finally:
            self._close_owned(wrapper)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        if digest != cfg.manifest_sha256:
            raise ValueError(
                f"manifest sha256 mismatch: got {digest}, want {cfg.manifest_sha256}"
            )
        corpus: Corpus
        if cfg.corpus_detail is None:
            corpus = _corpus_from_payload(
                manifest_bytes, cfg.manifest, digest, cfg.corpus_root
            )
        else:
            # OPT-IN dual pinned contract: after the ORIGINAL manifest bytes
            # are hash-verified, the frozen DETAIL wrapper is read on THIS
            # SAME owning thread using the SAME opener/register/
            # _read_bounded/_close_owned infrastructure, the SAME cancellation
            # event and the SAME 8 MiB cap. No extra thread, no root-thread
            # read, no FIFO assumptions.
            if cfg.manifest_sha256 != FROZEN_MANIFEST_SHA256:
                raise ValueError(
                    "corpus detail opt-in requires the frozen original "
                    f"manifest pin {FROZEN_MANIFEST_SHA256}; caller pin "
                    f"{cfg.manifest_sha256!r} cannot bypass it"
                )
            if cfg.corpus_detail_sha256 != FROZEN_CORPUS_DETAIL_SHA256:
                raise ValueError(
                    "corpus detail opt-in requires the frozen detail pin "
                    f"{FROZEN_CORPUS_DETAIL_SHA256}; got "
                    f"{cfg.corpus_detail_sha256!r}"
                )
            wrapper = self._opener(str(cfg.corpus_detail))
            self._register(wrapper)
            try:
                detail_bytes = self._read_bounded(
                    wrapper, INPUT_MANIFEST_MAX_BYTES, "corpus-detail"
                )
            finally:
                self._close_owned(wrapper)
            detail_digest = hashlib.sha256(detail_bytes).hexdigest()
            if detail_digest != FROZEN_CORPUS_DETAIL_SHA256:
                raise ValueError(
                    "corpus detail sha256 mismatch: got "
                    f"{detail_digest}, want {FROZEN_CORPUS_DETAIL_SHA256}"
                )
            # Parse ONLY the hash-verified detail bytes. Its
            # manifest_json_sha256 field must exactly match the freshly
            # verified ORIGINAL manifest digest; the ordered nonempty item
            # hashes must match the frozen fingerprint (order preserved by
            # the existing shared parser).
            detail_doc = json.loads(detail_bytes.decode("utf-8"))
            linked_digest = detail_doc.get("manifest_json_sha256")
            if linked_digest != digest:
                raise ValueError(
                    "corpus detail manifest_json_sha256 does not match the "
                    f"verified original digest: got {linked_digest!r}, "
                    f"want {digest}"
                )
            raw_detail_items = detail_doc.get("items")
            if (
                not isinstance(raw_detail_items, list)
                or len(raw_detail_items) != FROZEN_ITEM_COUNT
            ):
                raise ValueError(
                    "corpus detail items count "
                    f"{len(raw_detail_items) if isinstance(raw_detail_items, list) else 'non-list'} "
                    f"!= frozen {FROZEN_ITEM_COUNT}"
                )
            ordered_shas = [
                str(raw.get("sha256") or "")
                for raw in raw_detail_items
                if isinstance(raw, dict)
            ]
            if len(ordered_shas) != FROZEN_ITEM_COUNT or not all(ordered_shas):
                raise ValueError(
                    "corpus detail ordered item sha256 fingerprint input "
                    "incomplete (missing/non-object entries)"
                )
            fingerprint = hashlib.sha256(
                "\n".join(ordered_shas).encode("utf-8")
            ).hexdigest()
            if fingerprint != FROZEN_CORPUS_DETAIL_ORDERED_SHA_FINGERPRINT:
                raise ValueError(
                    "corpus detail ordered item sha256 fingerprint mismatch: "
                    f"got {fingerprint}, want "
                    f"{FROZEN_CORPUS_DETAIL_ORDERED_SHA_FINGERPRINT}"
                )
            # Item list/order comes from the verified DETAIL bytes; the
            # corpus identity (sha256/path) stays the ACTUAL verified
            # ORIGINAL manifest — the wrapper bytes are NEVER mislabeled
            # with the original pin. source_verified is set ONLY after both
            # hashes, the digest link and the ordered fingerprint checked.
            corpus = _corpus_from_payload(
                detail_bytes, cfg.manifest, digest, cfg.corpus_root
            )
            corpus.detail_sha256 = detail_digest
            corpus.detail_path = str(cfg.corpus_detail)
            corpus.detail_ordered_fingerprint = fingerprint
            corpus.source_verified = True
        diagnostic = bool(getattr(self.config, "single_long_diagnostic", False))
        if diagnostic:
            if cfg.corpus_detail is not None:
                raise ValueError("single-long diagnostic forbids --corpus-detail")
            if len(corpus.items) != 1:
                raise ValueError(
                    "single-long diagnostic requires exactly one manifest item"
                )
            item = corpus.items[0]
            if item.transcript:
                raise ValueError(
                    "single-long diagnostic requires an empty transcript"
                )
            if not 90.0 <= item.duration_s <= 180.0:
                raise ValueError(
                    "single-long diagnostic duration must be between 90 and 180 seconds"
                )
        elif len(corpus.items) != FROZEN_ITEM_COUNT:
            raise ValueError(
                f"manifest item count {len(corpus.items)} != frozen "
                f"{FROZEN_ITEM_COUNT} (fixed denominator required)"
            )
        caps = _derive_input_caps(corpus.items)
        self.publish_corpus(corpus, caps)
        # 2. WAVs in frozen order: open/fstat/bounded-read/validate/close,
        #    all on this thread, against bytes-derived caps.
        aggregate = 0
        admitted_audio_s = 0.0
        for item in corpus.items:
            if self.cancel_event.is_set():
                raise OSError(f"cooperative input cancel before open ({item.file})")
            wrapper = self._opener(item.path)
            self._register(wrapper)
            try:
                # per-item cap is read from the ALREADY-PUBLISHED local
                # ``caps`` (the writer's own value), never from a live dict.
                data = self._read_bounded(
                    wrapper, caps["per_item_max_bytes"], item.file
                )
            finally:
                self._close_owned(wrapper)
            item_sha = hashlib.sha256(data).hexdigest()
            pcm, check = _pcm_from_wav_bytes(data, item_sha)
            errors: list[str] = []
            if item.duration_s <= 0:
                errors.append(f"manifest duration_s={item.duration_s} (want > 0)")
            if not item.sha256:
                # A per-item frozen SHA is REQUIRED for actual admission;
                # there is no invented/optional hash skip.
                errors.append("manifest sha256 missing (required for admission)")
            elif item_sha != item.sha256:
                errors.append(f"sha256 mismatch: got {item_sha}, want {item.sha256}")
            if item.rate != EXPECTED_RATE:
                errors.append(f"manifest rate={item.rate} (want {EXPECTED_RATE})")
            if item.channels != EXPECTED_CHANNELS:
                errors.append(
                    f"manifest channels={item.channels} (want {EXPECTED_CHANNELS})"
                )
            if item.bit_depth != EXPECTED_SAMPLE_WIDTH * 8:
                errors.append(
                    f"manifest bit_depth={item.bit_depth} "
                    f"(want {EXPECTED_SAMPLE_WIDTH * 8})"
                )
            if check.error:
                errors.append(check.error)
            if check.ok and item.duration_s > 0 and (
                abs(check.duration_s - item.duration_s) > DURATION_TOLERANCE_S
            ):
                errors.append(
                    f"duration mismatch: got {check.duration_s:.6f}s, "
                    f"want {item.duration_s:.6f}s"
                )
            if errors:
                raise ValueError(
                    f"item {item.order} ({item.file}) invalid: " + "; ".join(errors)
                )
            aggregate += len(data)
            if aggregate > caps["aggregate_max_bytes"]:
                raise ValueError(
                    f"aggregate input cap exceeded after {item.file}: "
                    f"{aggregate} > {caps['aggregate_max_bytes']}"
                )
            admitted_audio_s += check.duration_s
            # Publish the admitted item AND the running totals together under
            # the shared lock (no unprotected live-dict write).
            self.publish_admitted(
                item.order,
                AdmittedInput(item=item, wav_bytes=data, pcm=pcm, check=check),
                admitted_audio_s,
                aggregate,
            )

    def run(self) -> None:
        self.started_mono = time.monotonic()
        try:
            try:
                self._load_all()
            except Exception as exc:  # noqa: BLE001 - ledger, never crash
                self.note_failure(
                    {
                        "stage": "input_admission",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            except BaseException as exc:  # noqa: BLE001 - unexpected, ledgered
                self.note_failure(
                    {
                        "stage": "input_admission",
                        "error": f"unexpected {type(exc).__name__}: {exc}",
                    }
                )
        finally:
            self.finished_mono = time.monotonic()
            # done_event means "the loader body returned" ONLY. It is NOT
            # thread-finished proof and NOT file-closed proof; admission
            # requires the caller's separate cleanup_proven() observation.
            self.done_event.set()


async def admit_corpus_inputs(
    config: "RunConfig",
    deadline_mono: float,
    *,
    opener: Any = None,
    loader_factory: Any = None,
) -> dict[str, Any]:
    """Admit the full frozen corpus input via the ONE owned thread, bounded
    by the ALREADY-RUNNING absolute overall deadline. The event loop polls a
    ``threading.Event`` (never ``asyncio.to_thread``, never the default
    executor, never ``wait_for`` over a cancellation) so it stays responsive
    and returns an explicit pending/failed report on expiry WITHOUT waiting
    for the possibly-blocked helper.

    ``done_event`` alone is NOT admission proof: after it is set, a FINITE
    poll (sleep capped by the remaining budget, no minimum floor) waits for
    the helper to ACTUALLY exit, and ``admitted`` requires: no failures, ALL
    frozen items admitted against a validated corpus, a clean close ledger,
    and an empty wrapper registry (actual cleanup proof)."""
    started = time.monotonic()
    expected_count = (
        1 if bool(getattr(config, "single_long_diagnostic", False))
        else FROZEN_ITEM_COUNT
    )
    if _remaining(deadline_mono) <= 0:
        return _input_report(
            "not_started",
            None,
            input_duration_s=0.0,
            expected_count=expected_count,
            note=(
                "overall budget already expired before input work; NO helper "
                "thread started, NO file opened or read, NO probe started"
            ),
        )
    loader = (loader_factory or OwnedInputLoader)(config, opener=opener)
    loader.start()
    while not loader.done_event.is_set():
        remaining = _remaining(deadline_mono)
        if remaining <= 0:
            # Cooperative best-effort cancel; NOT a cleanup or cancellation
            # claim: the read may remain blocked in the OS and the thread is
            # never force-stopped. Pending ownership is retained and reported.
            loader.cancel_event.set()
            return _input_report(
                "pending",
                loader,
                input_duration_s=time.monotonic() - started,
                expected_count=expected_count,
                note=(
                    "overall budget expired while the owned input thread was "
                    "still working; the read may remain blocked in the OS and "
                    "was NOT canceled or force-stopped; pending helper "
                    "ownership retained; NO probe/inference started"
                ),
            )
        # Pre-done poll sleep is ALSO capped by the ORIGINAL remaining
        # deadline (same as the post-done poll): no sleep may run past
        # expiry, and after expiry NO helper/probe starts.
        await asyncio.sleep(min(INPUT_POLL_INTERVAL_S, remaining))
    # done_event set: finite poll for the helper's ACTUAL exit inside the
    # ORIGINAL deadline (each sleep is capped by the remaining budget; no
    # unbounded join, no default executor, no minimum floor).
    while loader.is_alive():
        remaining = _remaining(deadline_mono)
        if remaining <= 0:
            return _input_report(
                "pending",
                loader,
                input_duration_s=time.monotonic() - started,
                expected_count=expected_count,
                note=(
                    "helper set done_event but was still alive at budget "
                    "expiry; thread-finished proof absent; pending ownership "
                    "retained; NO probe/inference started"
                ),
            )
        await asyncio.sleep(min(INPUT_POLL_INTERVAL_S, remaining))
    input_duration_s = time.monotonic() - started
    snap = loader.snapshot()
    # Explicit terminal-state admission checks: failures, incomplete corpus,
    # dirty close ledger or unproven cleanup all CANNOT admit.
    if snap["failures"]:
        status = "failed"
    elif snap["open_wrappers"] or snap["close_errors"]:
        status = "failed"  # dirty objects / unproven closes: cleanup impossible
    elif snap["corpus"] is None or len(snap["admitted"]) != expected_count:
        loader.note_failure(
            {
                "stage": "input_admission_terminal",
                "error": (
                    f"unexpected terminal state: corpus={snap['corpus'] is not None}, "
                    f"admitted={len(snap['admitted'])}/{expected_count}, "
                    "no failures ledgered (unexpected helper exit)"
                ),
            }
        )
        status = "failed"
    else:
        status = "admitted"
    return _input_report(
        status,
        loader,
        input_duration_s=input_duration_s,
        expected_count=expected_count,
        note=(
            "input fully admitted from owned frozen bytes; helper thread "
            "actually exited and every owned wrapper close is PROVEN"
            if status == "admitted"
            else "input admission failed; fixed frozen denominator retained; "
            "NO probe/inference started"
        ),
    )


def _input_report(
    status: str,
    loader: Optional[OwnedInputLoader],
    *,
    input_duration_s: Optional[float],
    expected_count: int,
    note: str,
) -> dict[str, Any]:
    if loader is not None:
        # ONE synchronized snapshot publishes admitted/failures/wrapper state
        # AND corpus/caps/totals coherently under the SAME lock the writer
        # uses; no live-dict iteration can race the writer. The snapshot
        # copies references only; the (potentially large) byte-length totals
        # are computed here, OUTSIDE the lock.
        snap = loader.snapshot()
        admitted = snap["admitted"]
        open_wrappers = snap["open_wrappers"]
        close_errors = snap["close_errors"]
        failures = snap["failures"]
        corpus = snap["corpus"]
        thread_alive = loader.is_alive()
        caps = snap["caps"] if snap["caps"] else None
        admitted_audio_s = snap["admitted_audio_s"]
        totals = {
            "admitted_wav_bytes": sum(len(a.wav_bytes) for a in admitted.values()),
            "admitted_pcm_bytes": sum(len(a.pcm) for a in admitted.values()),
            "admitted_audio_s": admitted_audio_s,
        }
    else:
        admitted, open_wrappers, close_errors, failures = {}, [], [], []
        corpus = None
        caps = None
        thread_alive = False
        totals = {
            "admitted_wav_bytes": 0,
            "admitted_pcm_bytes": 0,
            "admitted_audio_s": None,
        }
    # Actual cleanup proof: thread exited AND no open wrapper AND no unproven
    # close. Thread death alone is NOT file-closed proof; dirty objects are
    # reported even when the thread is dead (no false cleanup ledger).
    cleanup_complete = bool(
        loader is not None
        and not thread_alive
        and not open_wrappers
        and not close_errors
    )
    return {
        "status": status,
        "summary": (
            f"input admission {status}: "
            f"{len(admitted)}/{expected_count} items admitted"
        ),
        "input_duration_s": input_duration_s,
        "manifest_path": str(loader.config.manifest) if loader is not None else None,
        "manifest_sha256": loader.config.manifest_sha256 if loader is not None else None,
        # REQUESTED opt-in detail inputs (what the caller asked the loader to
        # admit). These are REQUEST records only and never claim verified
        # provenance; the observed/verified fields are the corpus_detail_*
        # and corpus_source_verified entries below.
        "corpus_detail_path_requested": (
            str(loader.config.corpus_detail)
            if loader is not None and loader.config.corpus_detail is not None
            else None
        ),
        "corpus_detail_sha256_requested": (
            loader.config.corpus_detail_sha256 if loader is not None else None
        ),
        "manifest_sha_verified": (
            corpus.manifest_sha256 if corpus is not None else None
        ),
        "item_count": len(corpus.items) if corpus is not None else None,
        "ref_words": corpus.ref_words if corpus is not None else None,
        "corpus_detail_path": (
            corpus.detail_path if corpus is not None else None
        ),
        "corpus_detail_sha256": (
            corpus.detail_sha256 if corpus is not None else None
        ),
        "corpus_detail_ordered_fingerprint": (
            corpus.detail_ordered_fingerprint if corpus is not None else None
        ),
        "corpus_source_verified": (
            corpus.source_verified if corpus is not None else None
        ),
        "admitted_orders": sorted(admitted),
        "failure_count": len(failures),
        "failures": list(failures),
        "caps": caps,
        "totals": totals,
        "pending": {
            "thread_name": loader.name if loader is not None else None,
            "thread_alive": thread_alive,
            "daemon": loader.daemon if loader is not None else None,
            "open_wrappers": [repr(w) for w in open_wrappers],
            "close_errors": list(close_errors),
            "cleanup_complete": cleanup_complete,
        },
        "note": note,
        # Live owned-helper reference (stripped from all serialized JSON):
        # pending ownership is retained by an actual object, not a claim.
        "loader": loader,
        "boundary": (
            "ONE owned daemon input thread; sole fd owner; nonblocking open + "
            "fstat regular-file check + bounded chunk reads + cooperative "
            "cancel Event; no asyncio.to_thread, no default executor, no "
            "forceful thread stop, no cross-thread fd close"
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Pinned service identity file (explicit artifact identity evidence)
# ──────────────────────────────────────────────────────────────────────


class IdentityError(Exception):
    pass


def load_identity_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and SHA-verify the pinned service identity JSON file.

    Required fields: target_device, sdk_version, upstream_commit,
    worker_sha256, plugin_sha256, profile_family, base_profile_sha256,
    profile_sha256, engine_sha256, config_sha256, slot_variant
    (baseline|candidate). base_profile_sha256 is the common base profile the
    resolved variant profile was produced from; it is never inferred.

    This file is the artifact identity evidence: capabilities metadata
    (model_id/backend labels) alone is NOT live artifact proof.
    """
    if not path.is_file():
        raise IdentityError(f"identity file not found: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise IdentityError(
            f"identity file sha256 mismatch: got {actual}, want {expected_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise IdentityError(f"identity file is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise IdentityError("identity file must be a JSON object")
    missing = [
        name
        for name in IDENTITY_REQUIRED_FIELDS
        if not isinstance(payload.get(name), str) or not payload.get(name)
    ]
    if missing:
        raise IdentityError(f"identity file missing/empty fields: {missing}")
    if payload["slot_variant"] not in IDENTITY_SLOT_VARIANTS:
        raise IdentityError(
            f"slot_variant must be one of {IDENTITY_SLOT_VARIANTS}, "
            f"got {payload['slot_variant']!r}"
        )
    return {
        "path": str(path),
        "sha256": actual,
        "identity": {name: payload[name] for name in IDENTITY_REQUIRED_FIELDS},
    }


# ──────────────────────────────────────────────────────────────────────
# Frame / connection abstraction (lets the CPU test inject a fake)
# ──────────────────────────────────────────────────────────────────────


class ConnClosed(Exception):
    """Raised by a connection adapter when the peer closes the socket."""

    def __init__(self, code: Optional[int] = None, reason: str = "") -> None:
        super().__init__(f"closed code={code} reason={reason!r}")
        self.code = code
        self.reason = reason


def _capture_ws_close_observation(conn: Any) -> dict[str, Any]:
    """Read adapter-owned close evidence without inferring a peer frame."""

    def unproven(source: str, error: Optional[str] = None) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "UNPROVEN", "source": source}
        if error is not None:
            result["error"] = error
        return result

    try:
        observer = getattr(conn, "close_observation", None)
        if not callable(observer):
            return unproven("adapter close_observation unavailable")
        observation = observer()
    except Exception as exc:  # noqa: BLE001
        return unproven(
            "adapter close_observation raised",
            f"{type(exc).__name__}: {exc}",
        )
    if not isinstance(observation, dict):
        return unproven("adapter close_observation returned non-dict")

    status = observation.get("status")
    source = observation.get("source")
    peer_observed = observation.get("peer_close_observed")
    peer_code = observation.get("peer_code")
    peer_reason = observation.get("peer_reason")
    derived_code = observation.get("derived_close_code")
    def valid_code(value: Any) -> bool:
        return value is None or (
            isinstance(value, int) and not isinstance(value, bool)
        )
    if not isinstance(source, str) or not valid_code(derived_code):
        return unproven("adapter close_observation malformed source or derived code")
    if status == "PRESENT":
        if (
            peer_observed is not True
            or not isinstance(peer_code, int)
            or isinstance(peer_code, bool)
            or not isinstance(peer_reason, str)
            or not valid_code(derived_code)
        ):
            return unproven("adapter close_observation malformed PRESENT")
        return {
            "status": "PRESENT",
            "source": source,
            "peer_close_observed": True,
            "peer_code": peer_code,
            "peer_reason": peer_reason,
            "derived_close_code": derived_code,
        }
    if status == "ABSENT":
        if peer_observed is not False or peer_code is not None or peer_reason is not None:
            return unproven("adapter close_observation malformed ABSENT")
        return {
            "status": "ABSENT",
            "source": source,
            "peer_close_observed": False,
            "peer_code": None,
            "peer_reason": None,
            "derived_close_code": derived_code,
        }
    if status == "UNPROVEN":
        return {"status": "UNPROVEN", "source": source}
    return unproven("adapter close_observation unknown status")


class WsConn(Protocol):
    """Minimal async connection surface the WS receiver uses."""

    async def send(self, data: Any) -> None: ...
    async def recv(self) -> Any: ...
    async def close(self, code: int = 1000) -> None: ...


@dataclass
class RawFrame:
    """One frame observed by the receiver, with monotonic arrival time."""

    arrival_mono: float
    kind: str  # "text" | "bytes" | "close" | "error"
    text: Optional[str] = None
    data: Optional[str] = None  # hex for bytes frames
    code: Optional[int] = None
    error: Optional[str] = None


@dataclass
class UtteranceResult:
    """Per-utterance result. Failed rows are retained verbatim."""

    order: int
    file: str
    id: str
    warm: bool
    pair_index: Optional[int]
    ok: bool = False
    error: Optional[str] = None
    cleanup_error: Optional[str] = None
    # True when a child task/socket was still unwinding (not provably
    # finished) when its cleanup budget expired. Never hidden as success.
    cleanup_pending: bool = False
    connection_id: Optional[str] = None
    # Existing HTTP request correlator; this never proves a native SID.
    request_id: Optional[str] = None
    request_id_sent: bool = False
    # Raw timing basis (monotonic seconds relative to run start).
    first_send_mono: Optional[float] = None
    eos_send_mono: Optional[float] = None
    final_arrival_mono: Optional[float] = None
    first_partial_arrival_mono: Optional[float] = None
    connect_mono: Optional[float] = None
    handshake_mono: Optional[float] = None
    upload_end_mono: Optional[float] = None
    # Derived
    first_partial_latency_s: Optional[float] = None
    eos_to_final_s: Optional[float] = None
    request_wall_s: Optional[float] = None
    handshake_s: Optional[float] = None
    upload_paced_s: Optional[float] = None
    audio_s: float = 0.0
    # RTF = request_runtime_s / audio_duration_s (LOWER is better). Distinct
    # from throughput (audio / runtime, HIGHER is better).
    rtf: Optional[float] = None
    transcript: Optional[str] = None
    final_count: int = 0
    reset_ack_seen: bool = False
    duplicate_final: bool = False
    close_code: Optional[int] = None
    close_reason: str = ""
    ws_close_observation: Optional[dict[str, Any]] = None
    # Quality (filled later)
    wer: Optional[dict[str, Any]] = None
    raw_frames: list[RawFrame] = field(default_factory=list)
    # Pair timing basis (filled by the B2 orchestrator): earliest
    # first-PCM-send → latest own final of the pair (handshake excluded).
    pair_span_s: Optional[float] = None


class WidthTracker:
    """Live in-flight utterance counter (realized width, not metadata)."""

    def __init__(self) -> None:
        self.current = 0
        self.max = 0

    def enter(self) -> None:
        self.current += 1
        self.max = max(self.max, self.current)

    def exit(self) -> None:
        self.current -= 1


# ──────────────────────────────────────────────────────────────────────
# WS utterance protocol loop
# ──────────────────────────────────────────────────────────────────────


def _classify_frame(frame: Any) -> RawFrame:
    now = time.monotonic()
    if isinstance(frame, bytes):
        return RawFrame(arrival_mono=now, kind="bytes", data=frame.hex())
    if isinstance(frame, str):
        return RawFrame(arrival_mono=now, kind="text", text=frame)
    if isinstance(frame, dict):  # already-classified (test adapter)
        rf = RawFrame(
            arrival_mono=frame.get("arrival_mono", now),
            kind=frame.get("kind", "text"),
            text=frame.get("text"),
            data=frame.get("data"),
            code=frame.get("code"),
            error=frame.get("error"),
        )
        return rf
    return RawFrame(arrival_mono=now, kind="error", error=f"unexpected frame {frame!r}")


async def run_ws_utterance(
    conn_factory: Callable[[], Awaitable[WsConn]],
    pcm: bytes,
    *,
    order: int,
    file: str,
    item_id: str,
    warm: bool,
    pair_index: Optional[int],
    audio_s: float,
    transcript: Optional[str],
    chunk_bytes: int,
    pace_s: float,
    request_deadline_s: float,
    post_final_window_s: float,
    overall_deadline_mono: Optional[float] = None,
    registry: Optional[AsyncLifetimeRegistry] = None,
    eos: bool = True,
    send_reset: bool = False,
    reset_replay_pcm: Optional[bytes] = None,
) -> UtteranceResult:
    """Drive one WS utterance with a receiver started BEFORE the first send.

    Receiver lifecycle:
      1. connect (bounded)
      2. START receiver task immediately
      3. stamp first-send monotonic BEFORE awaiting the first PCM send
      4. paced upload (bounded sends; pacing never sleeps past the deadline)
      5. EOS (empty bytes), stamped BEFORE its await
      6. collect until: one own final observed + bounded post-final window, or
         the shared per-request deadline, whichever comes first
      7. cancel/drain receiver and close, bounded by the reserved cleanup
         budget inside the overall deadline; cleanup failures recorded

    Every connect/send/pace/EOS/reset/recv/close await shares one absolute
    request deadline. The result is ``ok`` only when exactly one own final was
    seen, no error frame/close and no duplicate final occurred, all recorded
    times are finite/non-bool/non-negative, and a final was observed.
    """
    result = UtteranceResult(
        order=order,
        file=file,
        id=item_id,
        warm=warm,
        pair_index=pair_index,
        audio_s=audio_s,
        transcript=transcript,
    )
    started = time.monotonic()
    result.connect_mono = started
    deadline = started + request_deadline_s
    if overall_deadline_mono is not None:
        # Reserve cleanup inside the overall deadline.
        deadline = min(deadline, overall_deadline_mono - CLEANUP_RESERVE_S)
    cleanup_deadline = deadline + CLEANUP_RESERVE_S

    # Owned per-utterance async lifetime ledger (F4): the ACTUAL receiver
    # task, connector task, poll getters, close task and the connection
    # resource are held here until their completion/close is OBSERVED.
    if registry is None:
        registry = AsyncLifetimeRegistry(f"ws:{item_id}")
    else:
        # Helper-local child scope: this utterance's drain must never
        # cancel its caller, an ancestor or a sibling pair member; all
        # entries are still forwarded (same shared entries) to the parent
        # aggregate so the root ledger/snapshot stays complete.
        registry = registry.child(f"ws:{item_id}")

    frames: list[RawFrame] = []
    final_seen = asyncio.Event()
    q: asyncio.Queue = asyncio.Queue()

    conn: Optional[WsConn] = None
    try:
        conn = await _connect_owned(
            conn_factory, deadline, "connect", registry,
            kind="ws-connection",
            late_close=lambda c: c.close(),
            # The EXISTING clamped cleanup budget reserved inside the one
            # overall deadline (deadline + CLEANUP_RESERVE_S); never a new
            # grace.
            cleanup_deadline_mono=cleanup_deadline,
        )
    except Exception as exc:  # noqa: BLE001
        result.error = f"connect failed: {type(exc).__name__}: {exc}"
        # A cancellation-suppressing connector may still be unwinding; drain
        # it finitely inside the reserved cleanup budget and record pending.
        await registry.drain(max(0.0, _remaining(cleanup_deadline)))
        pending = registry.pending_entries()
        if pending:
            result.cleanup_pending = True
            result.cleanup_error = (
                f"{len(pending)} owned task(s) still pending after connect "
                "failure drain"
            )
        return result
    result.connection_id = f"{id(conn):x}"
    result.request_id = getattr(conn, "request_id", None)
    result.request_id_sent = getattr(conn, "request_id_sent", False) is True
    result.handshake_mono = time.monotonic()
    delta, err = _time_delta(result.handshake_mono, started)
    if err is not None:
        result.error = f"handshake timing invalid: {err}"
    else:
        result.handshake_s = delta

    async def receiver() -> None:
        try:
            while True:
                frame = await conn.recv()  # type: ignore[union-attr]
                await q.put(_classify_frame(frame))
        except ConnClosed as exc:
            await q.put(RawFrame(arrival_mono=time.monotonic(), kind="close",
                                 code=exc.code, error=exc.reason))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await q.put(RawFrame(arrival_mono=time.monotonic(), kind="error",
                                 error=f"{type(exc).__name__}: {exc}"))

    recv_task = registry.track_task(asyncio.create_task(receiver()), "receiver")

    # ── Send path: receiver task exists BEFORE the first send. ──
    try:
        first_chunk = pcm[:chunk_bytes] if chunk_bytes > 0 else pcm
        # Stamp BEFORE the await so a fast first partial can never precede the
        # send stamp (no negative first-arrival).
        result.first_send_mono = time.monotonic()
        await _bounded(conn.send(first_chunk), deadline, "first PCM send", registry)
        offset = len(first_chunk)
        while offset < len(pcm):
            if _remaining(deadline) <= 0:
                result.error = "request deadline exceeded during upload"
                break
            chunk = pcm[offset:offset + chunk_bytes]
            await _bounded(conn.send(chunk), deadline, "PCM chunk send", registry)
            offset += len(chunk)
            if pace_s > 0 and offset < len(pcm):
                # Never sleep past the deadline; no minimum wait after expiry.
                await asyncio.sleep(min(pace_s, max(0.0, _remaining(deadline))))
        result.upload_end_mono = time.monotonic()
        delta, err = _time_delta(result.upload_end_mono, result.first_send_mono)
        if err is not None:
            result.error = result.error or f"upload timing invalid: {err}"
        else:
            result.upload_paced_s = delta

        if send_reset and conn is not None and result.error is None:
            await _bounded(
                conn.send(json.dumps({"command": "reset"})), deadline, "reset send",
                registry,
            )
            replay = reset_replay_pcm if reset_replay_pcm is not None else pcm
            offset = 0
            while offset < len(replay):
                if _remaining(deadline) <= 0:
                    result.error = "request deadline exceeded during reset replay"
                    break
                chunk = replay[offset:offset + chunk_bytes]
                await _bounded(conn.send(chunk), deadline, "reset replay send", registry)
                offset += len(chunk)
                if pace_s > 0 and offset < len(replay):
                    await asyncio.sleep(min(pace_s, max(0.0, _remaining(deadline))))

        if eos and conn is not None and result.error is None:
            # Stamp EOS BEFORE the await.
            result.eos_send_mono = time.monotonic()
            await _bounded(conn.send(b""), deadline, "EOS send", registry)
    except ConnClosed as exc:
        result.error = f"send failed: peer closed code={exc.code}"
        await q.put(RawFrame(arrival_mono=time.monotonic(), kind="close",
                             code=exc.code, error=exc.reason))
    except TimeoutError as exc:
        result.error = str(exc)
    except Exception as exc:  # noqa: BLE001
        result.error = f"send failed: {type(exc).__name__}: {exc}"

    # ── Receive path: bounded by deadline + post-final window. ──
    first_partial_state = {"seen": False}
    own_final_count = 0
    window_end: Optional[float] = None
    close_observed = False
    try:
        while True:
            now = time.monotonic()
            if now > deadline:
                if result.error is None:
                    result.error = "request deadline exceeded waiting for frames"
                break
            if window_end is not None and now >= window_end:
                break
            timeout = deadline - now
            if window_end is not None:
                timeout = min(timeout, max(0.0, window_end - now))
            if timeout <= 0:
                timeout = 0.001
            try:
                frame = await _poll_queue(q, timeout, registry)
            except _PollTimeout:
                # No frame within the current bound; loop re-checks deadline.
                continue
            frames.append(frame)
            if frame.kind == "bytes":
                # Primitive (non-JSON) frame: protocol violation for this API.
                result.error = "primitive binary frame received from server"
                break
            if frame.kind == "error":
                result.error = f"receiver error: {frame.error}"
                break
            if frame.kind == "close":
                close_observed = True
                result.close_code = frame.code
                result.close_reason = frame.error or ""
                break
            if frame.kind != "text":
                continue
            try:
                msg = json.loads(frame.text or "{}")
            except ValueError:
                result.error = "non-JSON text frame from server"
                break
            mtype = str(msg.get("type") or "")
            if mtype == "partial":
                if str(msg.get("text") or "").strip():
                    if not first_partial_state["seen"]:
                        first_partial_state["seen"] = True
                        result.first_partial_arrival_mono = frame.arrival_mono
                        if result.first_send_mono is not None:
                            delta, err = _time_delta(
                                frame.arrival_mono, result.first_send_mono
                            )
                            if err is not None:
                                result.error = f"first-partial timing invalid: {err}"
                                break
                            result.first_partial_latency_s = delta
                continue
            if mtype == "final":
                own_final_count += 1
                if own_final_count == 1:
                    result.final_arrival_mono = frame.arrival_mono
                    if result.eos_send_mono is not None:
                        delta, err = _time_delta(
                            frame.arrival_mono, result.eos_send_mono
                        )
                        if err is not None:
                            result.error = f"EOS-to-final timing invalid: {err}"
                            break
                        result.eos_to_final_s = delta
                    if result.first_send_mono is not None:
                        delta, err = _time_delta(
                            frame.arrival_mono, result.first_send_mono
                        )
                        if err is not None:
                            result.error = f"request wall timing invalid: {err}"
                            break
                        result.request_wall_s = delta
                    result.transcript = str(msg.get("text") or "")
                    # Set the bounded post-final window from the first final.
                    window_end = frame.arrival_mono + post_final_window_s
                    final_seen.set()
                else:
                    result.duplicate_final = True
                continue
            if mtype == "reset":
                # App-layer reset ACK (server/main.py reset branch). Distinct
                # from a transcript final; recorded, never counted as one.
                result.reset_ack_seen = True
                continue
            if mtype == "busy":
                result.error = f"server busy: {msg.get('endpoint')}"
                break
            if mtype == "error":
                result.error = f"server error frame: {msg.get('error')}"
                break
            # Unknown frame types are retained but do not by themselves fail
            # the row; the strict checks below catch protocol violations.
    finally:
        # Cleanup is bounded by the reserved budget inside the overall
        # deadline (F4): cancellation is requested EXACTLY ONCE and its
        # completion is only ever OBSERVED through a finite asyncio.wait —
        # never awaited without a bound. A cancellation-suppressing receiver
        # or close stays registered as pending (forcing this row NF) instead
        # of blocking the caller. No unbounded finally await, NO minimum
        # wait after expiry and no unawaited coroutine: when the budget is
        # expired the close coroutine is never even created.
        recv_observed = await registry.cancel_observe(
            recv_task, max(0.0, _remaining(cleanup_deadline)), "receiver cleanup"
        )
        if not recv_observed:
            result.cleanup_pending = True
            note = (
                "receiver cleanup pending: cancellation not observed within "
                "the reserved cleanup budget"
            )
            result.cleanup_error = (
                f"{result.cleanup_error}; {note}" if result.cleanup_error else note
            )
        else:
            entry = registry.entry_for(recv_task)
            if entry is not None and entry["state"] == "failed":
                note = f"receiver cleanup failed: {entry['exception']}"
                result.cleanup_error = (
                    f"{result.cleanup_error}; {note}" if result.cleanup_error else note
                )
        budget = _remaining(cleanup_deadline)
        if budget <= 0:
            result.cleanup_pending = True
            note = (
                "close cleanup pending: cleanup budget expired before close "
                "was scheduled (connection stays registered, no close "
                "started after expiry)"
            )
            result.cleanup_error = (
                f"{result.cleanup_error}; {note}" if result.cleanup_error else note
            )
        else:
            close_task = registry.track_task(
                asyncio.ensure_future(conn.close()), "connection close"  # type: ignore[union-attr]
            )
            registry.note_resource_close_requested(conn, "close scheduled")
            if await registry.wait_done(close_task, budget):
                close_entry = registry.entry_for(close_task)
                if close_entry is not None and close_entry["state"] == "done":
                    registry.note_resource_closed(conn, True, "close observed complete")
                    result.ws_close_observation = _capture_ws_close_observation(conn)
                else:
                    note = (
                        "close cleanup failed: "
                        f"{(close_entry or {}).get('exception') or 'cancelled'}"
                    )
                    registry.note_resource_closed(conn, False, note)
                    result.cleanup_error = (
                        f"{result.cleanup_error}; {note}" if result.cleanup_error else note
                    )
            else:
                close_task.cancel()
                registry.note_cancel_requested(close_task, "connection close")
                registry.note_resource_closed(
                    conn, False, "close not observed; cancellation requested once"
                )
                result.cleanup_pending = True
                note = (
                    "close cleanup pending: close not observed within the "
                    "reserved cleanup budget"
                )
                result.cleanup_error = (
                    f"{result.cleanup_error}; {note}" if result.cleanup_error else note
                )
        # Final per-utterance drain: late send unwinds, poll getters and the
        # close task are cancelled once and finitely observed inside the SAME
        # reserved budget; anything still pending forces cleanup_pending.
        await registry.drain(max(0.0, _remaining(cleanup_deadline)))
        still = registry.pending_entries()
        if still:
            result.cleanup_pending = True
            phases = ", ".join(sorted({e["phase"] for e in still}))
            note = (
                f"{len(still)} owned task(s) still pending after the cleanup "
                f"drain ({phases})"
            )
            result.cleanup_error = (
                f"{result.cleanup_error}; {note}" if result.cleanup_error else note
            )

    result.raw_frames = frames
    result.final_count = own_final_count

    # ── Strict positive qualification for this utterance. ──
    if result.error is None:
        if result.first_send_mono is None:
            result.error = "no PCM send recorded"
        elif not _valid_time_value(result.first_send_mono):
            result.error = f"invalid first-send timestamp {result.first_send_mono!r}"
        elif result.final_arrival_mono is None:
            result.error = "no final frame observed"
        elif own_final_count != 1:
            result.error = f"expected exactly one final, saw {own_final_count}"
        elif result.duplicate_final:
            result.error = "duplicate final frame"
        elif result.close_code is not None and result.close_code not in NORMAL_CLOSE_CODES:
            result.error = f"abnormal close code {result.close_code}"
        elif result.transcript is None:
            result.error = "final had no text field"
        elif result.request_wall_s is None:
            result.error = "no request wall computed"
    # Cleanup state is consulted EXPLICITLY: a row whose owned child
    # task/socket was still pending, or whose cleanup failed, can never be
    # ok=True and can never enter a qualified aggregate. The original error
    # (if any) and the cleanup ledger are both retained.
    if result.error is None and (result.cleanup_pending or result.cleanup_error):
        result.error = f"cleanup incomplete: {result.cleanup_error}"

    result.ok = result.error is None
    return result


# ──────────────────────────────────────────────────────────────────────
# HTTP utterance (async stdlib: owned socket, absolute-deadline bounded)
# ──────────────────────────────────────────────────────────────────────
#
# Dependency note: this repository has no top-level pyproject.toml and
# server/requirements.txt declares no async HTTP client (httpx exists only in
# the agent subproject). Per the root correction, no dependency is added: the
# HTTP path uses asyncio.open_connection with every await bounded by the
# shared absolute deadline, so B2 pairs genuinely overlap.


def build_multipart(wav_bytes: bytes, filename: str, boundary: str, *, language: Optional[str] = None) -> bytes:
    parts: list[bytes] = []
    fields = [("model", "qwen3-asr"), ("response_format", "json")]
    if language is not None:
        fields.append(("language", language))
    for name, value in fields:
        parts.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            )
        )
    parts.extend(
        (
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            b"Content-Type: audio/wav\r\n\r\n",
            wav_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return b"".join(parts)


async def _default_open_conn(
    parsed: Any,
    deadline_mono: float,
    registry: Optional[AsyncLifetimeRegistry] = None,
):
    """Open one owned TCP (or TLS) connection bounded by the deadline."""
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ssl_ctx = None
    if parsed.scheme == "https":
        import ssl

        ssl_ctx = ssl.create_default_context()
    return await _bounded(
        asyncio.open_connection(host, port, ssl=ssl_ctx), deadline_mono,
        "HTTP connect", registry,
    )


def _late_http_close(resource: Any) -> Any:
    """Supported close contract for a LATE-acquired (reader, writer) pair:
    the stdlib StreamWriter is closed synchronously (no await, no process
    signal use) and, when exposed, ``wait_closed`` is returned for a
    registry-tracked bounded observation."""
    _reader, writer = resource
    writer.close()
    wait_closed = getattr(writer, "wait_closed", None)
    if wait_closed is not None:
        return wait_closed()
    return None


async def _read_until_headers(reader: Any, deadline_mono: float,
                              registry: Optional[AsyncLifetimeRegistry] = None) -> tuple[bytes, bytes]:
    """Read bounded chunks until the header terminator; return (head, rest)."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        if len(buf) > 65536:
            raise ValueError("HTTP response headers exceed 64 KiB")
        chunk = await _bounded(reader.read(65536), deadline_mono, "HTTP header read", registry)
        if not chunk:
            raise ValueError("connection closed before response headers")
        buf += chunk
    head, rest = buf.split(b"\r\n\r\n", 1)
    return head, rest


async def _read_exactly(reader: Any, n: int, deadline_mono: float, initial: bytes = b"",
                        registry: Optional[AsyncLifetimeRegistry] = None) -> bytes:
    """Read exactly n body bytes; every read bounded by the absolute deadline
    (a slow-drip body cannot outlive the shared deadline)."""
    buf = bytearray(initial)
    while len(buf) < n:
        chunk = await _bounded(
            reader.read(min(65536, n - len(buf))), deadline_mono, "HTTP body read",
            registry,
        )
        if not chunk:
            raise ValueError(f"connection closed with {n - len(buf)} body bytes missing")
        buf += chunk
    return bytes(buf)


# Uniform response-body size cap shared by EVERY framing (content-length,
# chunked and read-to-EOF). A peer cannot drip an unbounded body past this
# cap even inside the deadline budget.
HTTP_BODY_MAX_BYTES = 64 * 1024 * 1024


def _check_body_cap(total: int, framing: str) -> None:
    if total > HTTP_BODY_MAX_BYTES:
        raise ValueError(
            f"HTTP {framing} body exceeds {HTTP_BODY_MAX_BYTES} byte cap"
        )


async def _read_chunked(reader: Any, deadline_mono: float, initial: bytes = b"",
                        registry: Optional[AsyncLifetimeRegistry] = None) -> bytes:
    """Minimal bounded chunked-transfer decoder with the shared body cap."""
    buf = bytearray(initial)
    _check_body_cap(len(buf), "chunked")
    body = bytearray()

    async def readline() -> bytes:
        while b"\r\n" not in buf:
            chunk = await _bounded(reader.read(65536), deadline_mono, "HTTP chunk read", registry)
            if not chunk:
                raise ValueError("connection closed inside chunked body")
            buf.extend(chunk)
        line, _, rest = bytes(buf).partition(b"\r\n")
        del buf[:]
        buf.extend(rest)
        return line

    while True:
        size_line = await readline()
        try:
            size = int(size_line.split(b";", 1)[0].strip(), 16)
        except ValueError as exc:
            raise ValueError(f"bad chunk size line {size_line!r}") from exc
        if size < 0:
            # int(x, 16) accepts a leading '-'; a negative chunk size is a
            # framing violation, never a length.
            raise ValueError(f"negative chunk size {size}")
        if size == 0:
            # Consume trailers up to an empty line.
            while (await readline()) != b"":
                continue
            return bytes(body)
        _check_body_cap(len(body) + size, "chunked")
        while len(buf) < size + 2:
            chunk = await _bounded(reader.read(65536), deadline_mono, "HTTP chunk read", registry)
            if not chunk:
                raise ValueError("connection closed inside chunk data")
            buf.extend(chunk)
        # Validate the mandatory chunk-data CRLF terminator instead of
        # blindly skipping two bytes (shared GET/POST parser consistency).
        if bytes(buf[size:size + 2]) != b"\r\n":
            raise ValueError("chunk data not terminated by CRLF")
        body += bytes(buf[:size])
        del buf[: size + 2]  # skip data + CRLF


async def _read_http_body(
    reader: Any,
    headers: dict[str, str],
    initial: bytes,
    deadline_mono: float,
    registry: Optional[AsyncLifetimeRegistry] = None,
) -> bytes:
    """Read one response body with the SAME caps for every framing.

    Shared by the utterance POST path and the readiness/capability GET
    probes so there is exactly one HTTP body parser: chunked (bounded by
    the absolute deadline AND the shared body cap), content-length (cap
    checked before reading) and connection-close EOF (bounded and capped).
    A slow body is bounded by the shared absolute deadline, never by a
    per-operation timeout, and no partial/fabricated fields are produced.
    """
    transfer = headers.get("transfer-encoding", "").lower()
    if "chunked" in transfer:
        return await _read_chunked(reader, deadline_mono, initial, registry)
    if "content-length" in headers:
        try:
            length = int(headers["content-length"])
        except ValueError as exc:
            raise ValueError(
                f"bad Content-Length {headers['content-length']!r}"
            ) from exc
        if length < 0 or length > HTTP_BODY_MAX_BYTES:
            raise ValueError(f"implausible Content-Length {length}")
        return await _read_exactly(reader, length, deadline_mono, initial, registry)
    # connection: close — bounded read-to-EOF under the shared cap.
    chunks = [initial]
    total = len(initial)
    while True:
        chunk = await _bounded(reader.read(65536), deadline_mono, "HTTP body read", registry)
        if not chunk:
            break
        total += len(chunk)
        _check_body_cap(total, "eof")
        chunks.append(chunk)
    return b"".join(chunks)


async def run_http_utterance(
    url: str,
    wav_bytes: bytes,
    *,
    wav_name: str,
    order: int,
    file: str,
    item_id: str,
    warm: bool,
    pair_index: Optional[int],
    audio_s: float,
    transcript: Optional[str],
    language: Optional[str] = None,
    deadline_mono: float,
    open_conn: Optional[Callable[[Any, float], Awaitable[Any]]] = None,
    registry: Optional[AsyncLifetimeRegistry] = None,
) -> UtteranceResult:
    """Run one absolute-deadline-bounded multipart HTTP request, async.

    ``wav_bytes`` are the ADMITTED frozen bytes (owned-input boundary); this
    function performs NO filesystem read. Every connect/write/drain/read/
    close await is bounded by the shared absolute deadline (NOT a
    per-operation socket timeout), so a slow-drip response or hung peer
    cannot outlive it, and two gathered utterances genuinely overlap. Only
    whole-request wall time is recorded; partial and EOS fields are NEVER
    fabricated for HTTP.
    """
    import urllib.parse
    import uuid

    if registry is None:
        registry = AsyncLifetimeRegistry(f"http:{item_id}")
    else:
        # Helper-local child scope (see run_ws_utterance): drain stays
        # strictly local to this request; entries still reach the parent
        # aggregate through the shared entry objects.
        registry = registry.child(f"http:{item_id}")

    result = UtteranceResult(
        order=order,
        file=file,
        id=item_id,
        warm=warm,
        pair_index=pair_index,
        audio_s=audio_s,
        transcript=transcript,
    )
    started = time.monotonic()
    result.connect_mono = started
    if _remaining(deadline_mono) <= 0:
        result.error = "shared deadline already exceeded before request"
        return result

    # Owned-input boundary: no path read here; the admitted frozen bytes are
    # consumed directly (previously: wav_path.read_bytes() on the event loop).
    if not isinstance(wav_bytes, (bytes, bytearray)):
        result.error = (
            "input not admitted through the owned input boundary "
            "(path-based reads removed; supply admitted frozen bytes)"
        )
        return result

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        result.error = f"unsupported URL scheme {parsed.scheme!r}"
        return result
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    boundary = f"edgellm-{uuid.uuid4().hex}"
    body = build_multipart(wav_bytes, wav_name, boundary, language=language)
    head = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()

    opener = open_conn or _default_open_conn
    writer: Any = None
    try:
        # Retain the ORIGINAL (reader, writer) tuple object tracked by
        # _connect_owned before unpacking, so the pair's ledger entry can
        # be closed exactly when the writer close is OBSERVED (never
        # faked merely because close() returned).
        pair = await _connect_owned(
            lambda: opener(parsed, deadline_mono), deadline_mono,
            "HTTP connect", registry,
            kind="http-reader-writer",
            late_close=_late_http_close,
            # The existing whole-request deadline doubles as the cleanup
            # budget (no reserved slice, no renewal).
            cleanup_deadline_mono=deadline_mono,
        )
        reader, writer = pair
        registry.track_resource(writer, "stdlib-stream-writer", "HTTP writer")
        result.connection_id = f"{id(writer):x}"
        result.handshake_mono = time.monotonic()
        writer.write(head)
        # Stamp the first-send BEFORE awaiting the drain, matching the WS
        # first-PCM-send basis.
        result.first_send_mono = time.monotonic()
        writer.write(body)
        await _bounded(writer.drain(), deadline_mono, "HTTP request send", registry)
        result.upload_end_mono = time.monotonic()

        raw_head, rest = await _read_until_headers(reader, deadline_mono, registry)
        status_line, _, header_lines = raw_head.partition(b"\r\n")
        try:
            status = int(status_line.split(b" ", 2)[1])
        except (IndexError, ValueError):
            result.error = f"malformed HTTP status line {status_line!r}"
            return result
        headers: dict[str, str] = {}
        for line in header_lines.split(b"\r\n"):
            if b":" in line:
                name, _, value = line.partition(b":")
                headers[name.strip().lower().decode("latin1")] = value.strip().decode("latin1")
        payload = await _read_http_body(reader, headers, rest, deadline_mono, registry)

        result.final_arrival_mono = time.monotonic()
        delta, err = _time_delta(result.final_arrival_mono, result.first_send_mono)
        if err is not None:
            result.error = f"request wall timing invalid: {err}"
            return result
        result.request_wall_s = delta
        # EOS / partial metrics are deliberately NOT set for HTTP.
        if status != 200:
            body_text = payload.decode("utf-8", "replace")[:2000]
            result.error = f"HTTP {status}: {body_text}"
            return result
        try:
            parsed_body = json.loads(payload)
        except ValueError:
            result.error = f"non-JSON 200 body: {payload[:500]!r}"
            return result
        text = str(parsed_body.get("text") or "").strip()
        if not text:
            result.error = "200 response had empty transcript"
            return result
        result.transcript = text
        result.final_count = 1
    except TimeoutError as exc:
        result.error = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = f"HTTP request failed: {type(exc).__name__}: {exc}"
        return result
    finally:
        if writer is not None:
            budget = _remaining(deadline_mono)
            # The owned stdlib StreamWriter is ALWAYS closed synchronously
            # (close() is sync, no await); close() returning is recorded as
            # close-REQUESTED, never as observed transport close.
            try:
                writer.close()
                registry.note_resource_close_requested(
                    writer, "StreamWriter.close() returned"
                )
            except Exception as exc:  # noqa: BLE001
                result.cleanup_error = (
                    f"{result.cleanup_error}; " if result.cleanup_error else ""
                ) + f"HTTP sync close failed: {type(exc).__name__}: {exc}"
                registry.note_resource_closed(
                    writer, False, f"sync close failed: {type(exc).__name__}"
                )
            wait_closed = getattr(writer, "wait_closed", None)
            if wait_closed is None:
                # Generic adapter without a supported close-observation
                # contract: the actual close state stays UNKNOWN, never
                # defaulted to proven. No private force-close is attempted.
                result.cleanup_pending = True
                note = (
                    "HTTP close observation UNPROVEN: writer exposes no "
                    "supported wait_closed contract"
                )
                result.cleanup_error = (
                    f"{result.cleanup_error}; {note}" if result.cleanup_error else note
                )
            elif budget <= 0:
                # Expired: the bounded close-wait is never scheduled (no
                # floor wait, no unawaited coroutine); recorded as pending.
                result.cleanup_pending = True
                note = (
                    "HTTP close cleanup pending: shared deadline expired "
                    "before close-wait"
                )
                result.cleanup_error = (
                    f"{result.cleanup_error}; {note}" if result.cleanup_error else note
                )
            else:
                wc_task = registry.track_task(
                    asyncio.ensure_future(wait_closed()), "HTTP wait_closed"
                )
                if await registry.wait_done(wc_task, budget):
                    wc_entry = registry.entry_for(wc_task)
                    if wc_entry is not None and wc_entry["state"] == "done":
                        registry.note_resource_closed(
                            writer, True, "wait_closed observed complete"
                        )
                        # OBSERVED close of the same physical connection:
                        # close the ORIGINAL tuple's entry too (shared
                        # ledger). Timeout/failure paths stay open.
                        registry.note_resource_closed(
                            pair, True,
                            "wait_closed observed complete (owned pair)"
                        )
                    else:
                        note = (
                            "HTTP close cleanup failed: "
                            f"{(wc_entry or {}).get('exception') or 'cancelled'}"
                        )
                        registry.note_resource_closed(writer, False, note)
                        result.cleanup_error = (
                            f"{result.cleanup_error}; {note}"
                            if result.cleanup_error else note
                        )
                else:
                    # Cancel exactly once; completion is NOT awaited without
                    # a bound — a stuck close-wait stays registered pending.
                    wc_task.cancel()
                    registry.note_cancel_requested(wc_task, "HTTP wait_closed")
                    registry.note_resource_closed(
                        writer, False,
                        "wait_closed not observed; cancellation requested once",
                    )
                    result.cleanup_pending = True
                    note = (
                        "HTTP close cleanup pending: wait_closed not observed "
                        "within the shared deadline"
                    )
                    result.cleanup_error = (
                        f"{result.cleanup_error}; {note}"
                        if result.cleanup_error else note
                    )
            await registry.drain(max(0.0, _remaining(deadline_mono)))
            still = registry.pending_entries()
            if still:
                result.cleanup_pending = True
                phases = ", ".join(sorted({e["phase"] for e in still}))
                note = (
                    f"{len(still)} owned task(s) still pending after the HTTP "
                    f"cleanup drain ({phases})"
                )
                result.cleanup_error = (
                    f"{result.cleanup_error}; {note}" if result.cleanup_error else note
                )

    # Cleanup state is consulted EXPLICITLY (same contract as the WS path):
    # a pending or failed close can never yield ok=True.
    if result.error is None and (result.cleanup_pending or result.cleanup_error):
        result.error = f"cleanup incomplete: {result.cleanup_error}"

    result.ok = result.error is None
    return result


# ──────────────────────────────────────────────────────────────────────
# Capabilities / readiness probes
# ──────────────────────────────────────────────────────────────────────


async def _http_get_json(
    url: str,
    deadline_mono: float,
    registry: Optional[AsyncLifetimeRegistry] = None,
) -> dict[str, Any]:
    """One absolute-deadline-bounded async GET returning {status, body} and a
    separate cleanup ledger.

    Reuses the driver's OWNED async-socket HTTP framing helpers (connect,
    bounded header read, shared single body parser with uniform caps) so no
    divergent HTTP parser exists and every await is bounded by the SHARED
    absolute deadline — never a per-operation timeout, never a synchronous
    blocking call on the event loop. No fixed/floor wait after expiry; a
    slow body is bounded only by the shared deadline and the body cap.

    The OWNED writer is ALWAYS closed, synchronously if necessary (asyncio
    ``StreamWriter.close`` is sync and involves no process signal); when the
    budget is expired the bounded ``wait_closed`` is skipped and recorded as
    pending instead of the socket being abandoned. Raw status/body and the
    cleanup ledger are kept SEPARATE — the server payload never overwrites
    the cleanup record.
    """
    import urllib.parse

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme {parsed.scheme!r}")
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    head = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Accept: application/json\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()
    if registry is None:
        registry = AsyncLifetimeRegistry("http-probe")
    else:
        # Probe-local child scope: this probe's finally-drain acts only on
        # tasks THIS probe created — never on the caller (run_gate top),
        # the supervisor or a sibling phase; shared entries still reach
        # the root aggregate.
        registry = registry.child("http-probe")
    # Retain the ORIGINAL (reader, writer) tuple object tracked by
    # _connect_owned so the pair's close can be honestly ledgered when the
    # writer's close is actually observed (no permanently-open tuple).
    pair = await _connect_owned(
        lambda: _default_open_conn(parsed, deadline_mono, registry),
        deadline_mono, "probe HTTP connect", registry,
        kind="http-reader-writer",
        late_close=_late_http_close,
        # Existing whole-probe deadline as the cleanup budget (no renewal).
        cleanup_deadline_mono=deadline_mono,
    )
    reader, writer = pair
    registry.track_resource(writer, "stdlib-stream-writer", "probe HTTP writer")
    out: dict[str, Any] = {}
    try:
        writer.write(head)
        await _bounded(writer.drain(), deadline_mono, "HTTP GET send", registry)
        raw_head, rest = await _read_until_headers(reader, deadline_mono, registry)
        status_line, _, header_lines = raw_head.partition(b"\r\n")
        status = int(status_line.split(b" ", 2)[1])
        headers: dict[str, str] = {}
        for line in header_lines.split(b"\r\n"):
            if b":" in line:
                name, _, value = line.partition(b":")
                headers[name.strip().lower().decode("latin1")] = value.strip().decode(
                    "latin1"
                )
        payload = await _read_http_body(reader, headers, rest, deadline_mono, registry)
        body = json.loads(payload) if payload else {}
        out["status"] = status
        out["body"] = body
    except Exception as exc:  # noqa: BLE001
        # The request error is recorded WITHOUT discarding the cleanup ledger
        # computed below; the caller decides readiness/IFB provenance.
        out["request_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Owned-transport close with an EXPLICIT, separately recorded
        # ledger (F4): the stdlib writer close() is synchronous (close
        # REQUESTED, not proof); wait_closed completion is only reported
        # when actually OBSERVED within a finite bound, and cancellation is
        # requested exactly once without awaiting its completion unbounded.
        budget = _remaining(deadline_mono)
        cleanup_pending = False
        cleanup_error: Optional[str] = None
        try:
            writer.close()  # sync, owned transport
            registry.note_resource_close_requested(
                writer, "StreamWriter.close() returned"
            )
            wait_closed = getattr(writer, "wait_closed", None)
            if wait_closed is None:
                # No supported close-observation contract: state UNKNOWN.
                cleanup_pending = True
                cleanup_error = (
                    "probe close observation UNPROVEN: writer exposes no "
                    "supported wait_closed contract"
                )
            elif budget > 0:
                wc_task = registry.track_task(
                    asyncio.ensure_future(wait_closed()), "probe wait_closed"
                )
                if await registry.wait_done(wc_task, budget):
                    wc_entry = registry.entry_for(wc_task)
                    if wc_entry is not None and wc_entry["state"] == "done":
                        registry.note_resource_closed(
                            writer, True, "wait_closed observed complete"
                        )
                        # The ORIGINAL tuple tracked by _connect_owned is
                        # the same physical connection: its close is now
                        # OBSERVED, so both ledger entries close together.
                        registry.note_resource_closed(
                            pair, True,
                            "wait_closed observed complete (owned pair)"
                        )
                    else:
                        cleanup_error = (
                            "probe close failed: "
                            f"{(wc_entry or {}).get('exception') or 'cancelled'}"
                        )
                        registry.note_resource_closed(writer, False, cleanup_error)
                else:
                    wc_task.cancel()
                    registry.note_cancel_requested(wc_task, "probe wait_closed")
                    registry.note_resource_closed(
                        writer, False,
                        "wait_closed not observed; cancellation requested once",
                    )
                    cleanup_pending = True
                    cleanup_error = (
                        "probe close-wait pending: bounded observation expired; "
                        "cancellation requested once, completion not awaited"
                    )
            else:
                cleanup_pending = True
                cleanup_error = (
                    "probe close-wait pending: shared deadline expired; "
                    "writer closed synchronously, wait_closed not completed"
                )
        except Exception as exc:  # noqa: BLE001
            cleanup_error = f"probe close failed: {type(exc).__name__}: {exc}"
            registry.note_resource_closed(
                writer, False, f"{type(exc).__name__}: {exc}"
            )
        await registry.drain(max(0.0, _remaining(deadline_mono)))
        still = registry.pending_entries()
        if still:
            cleanup_pending = True
            phases = ", ".join(sorted({e["phase"] for e in still}))
            note = (
                f"{len(still)} owned task(s) still pending after the probe "
                f"cleanup drain ({phases})"
            )
            cleanup_error = f"{cleanup_error}; {note}" if cleanup_error else note
        out["cleanup_pending"] = cleanup_pending
        out["cleanup_error"] = cleanup_error
    return out


async def probe_service_identity(
    base_url: str,
    deadline_mono: float,
    registry: Optional[AsyncLifetimeRegistry] = None,
) -> dict[str, Any]:
    """Capture /v1/capabilities and /readyz with the admission ceiling.

    Fully async and bounded by the ONE shared absolute overall deadline that
    also bounds pre + post probes and the phase itself (no derived timeout,
    no floor waits). The raw concurrency block (runtime-reported slots),
    readiness status/body and the runtime IFB provenance are retained
    verbatim. model_id/backend labels are provenance only: they are NOT
    artifact identity proof, and a config-derived ceiling alone is NOT
    live-slot proof.
    """
    root = base_url.rstrip("/")
    out: dict[str, Any] = {"base_url": root}
    caps = await _http_get_json(f"{root}/v1/capabilities", deadline_mono, registry)
    out["capabilities"] = caps
    out["capabilities_cleanup"] = {
        "pending": caps.get("cleanup_pending"),
        "error": caps.get("cleanup_error"),
    }
    readyz = await _http_get_json(f"{root}/readyz", deadline_mono, registry)
    out["readyz"] = readyz
    out["readyz_cleanup"] = {
        "pending": readyz.get("cleanup_pending"),
        "error": readyz.get("cleanup_error"),
    }

    ready_body = readyz.get("body") if isinstance(readyz.get("body"), dict) else None
    # Actual observed readiness: HTTP 200 AND a ready body, AND a clean owned
    # probe close (a pending/failed probe close invalidates the observation
    # for qualification rather than being silently swallowed).
    out["readyz_status"] = readyz.get("status")
    out["readyz_request_error"] = readyz.get("request_error")
    probe_clean = (
        not readyz.get("cleanup_pending") and not readyz.get("cleanup_error")
    )
    out["probe_clean"] = probe_clean
    out["ready_ok"] = bool(
        readyz.get("status") == 200
        and isinstance(ready_body, dict)
        and ready_body.get("status") == "ready"
        and not readyz.get("request_error")
        and probe_clean
    )
    body = (out.get("capabilities") or {}).get("body") or {}
    asr = body.get("asr") or {}
    out["capabilities_request_error"] = (out.get("capabilities") or {}).get(
        "request_error"
    )
    out["asr_model_id"] = asr.get("model_id")
    out["asr_backend"] = asr.get("backend")
    concurrency = asr.get("concurrency") or {}
    out["concurrency_raw"] = concurrency  # runtime-reported slots, verbatim
    out["admission_limit"] = concurrency.get("admission_limit")
    out["backend_max_concurrent"] = concurrency.get("backend_max_concurrent")
    out["identity_present"] = bool(out["asr_model_id"] and out["asr_backend"])
    # Live native-worker IFB readiness provenance, verbatim (absent stays
    # absent — never synthesized from model_id or a config ceiling).
    runtime_ifb = asr.get("runtime_ifb")
    out["runtime_ifb"] = runtime_ifb if isinstance(runtime_ifb, dict) else None
    return out


def width_evidence(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    """Admission-ceiling evidence with strict (non-bool) int checks.

    A ceiling is only evidence when it is a genuine int; bools, floats and
    strings are rejected, and absence means UNPROVEN — never assumed.
    """
    ceilings: dict[str, Any] = {}
    for phase, probe in (("pre", pre), ("post", post)):
        ceilings[phase] = {
            "admission_limit": probe.get("admission_limit"),
            "backend_max_concurrent": probe.get("backend_max_concurrent"),
            "admission_limit_strict_int": _strict_int(probe.get("admission_limit")),
            "backend_max_concurrent_strict_int": _strict_int(
                probe.get("backend_max_concurrent")
            ),
        }
    best: Optional[int] = None
    for probe in (pre, post):
        for key in ("admission_limit", "backend_max_concurrent"):
            value = probe.get(key)
            if _strict_int(value) and (best is None or value > best):
                best = value
    return {"ceilings": ceilings, "max_strict_ceiling": best}


def _readiness_evidence(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    """Tri-state observed readiness: pre+post /readyz HTTP 200 + ready body.

    UNPROVEN when the probe itself errored (no observation);
    NOTQUALIFIED on an actual observed non-ready response (503, error
    body); QUALIFIED only when BOTH phases observed 200 + ready.
    """
    per_phase: dict[str, Any] = {}
    unproven: list[str] = []
    failed: list[str] = []
    for phase, probe in (("pre", pre), ("post", post)):
        readyz = probe.get("readyz")
        entry: dict[str, Any] = {
            "status": probe.get("readyz_status"),
            "ready_ok": probe.get("ready_ok"),
            "probe_cleanup": probe.get("readyz_cleanup"),
        }
        if not isinstance(readyz, dict) or readyz.get("request_error"):
            entry["state"] = "UNPROVEN"
            unproven.append(
                f"{phase}: /readyz probe errored: "
                f"{readyz.get('request_error') if isinstance(readyz, dict) else readyz!r}"
            )
        elif probe.get("probe_clean") is False:
            # Pending/failed owned probe close: the observation is not a
            # fabricated positive — it stays UNPROVEN for qualification.
            entry["state"] = "UNPROVEN"
            unproven.append(
                f"{phase}: /readyz probe owned-close incomplete: "
                f"{probe.get('readyz_cleanup')}"
            )
        elif probe.get("ready_ok") is True:
            entry["state"] = "QUALIFIED"
        else:
            entry["state"] = "NOTQUALIFIED"
            failed.append(
                f"{phase}: /readyz observed non-ready "
                f"(status={probe.get('readyz_status')!r})"
            )
        per_phase[phase] = entry
    if failed:
        overall = "NOTQUALIFIED"
    elif unproven:
        overall = "UNPROVEN"
    else:
        overall = "QUALIFIED"
    return {
        "status": overall,
        "per_phase": per_phase,
        "unproven_reasons": unproven,
        "failure_reasons": failed,
        "basis": "actual observed pre+post HTTP /readyz 200 with a ready body",
    }


def _runtime_ifb_evidence(
    pre: dict[str, Any], post: dict[str, Any], requested_width: int
) -> dict[str, Any]:
    """Tri-state live native-worker IFB/slot contract evidence.

    UNPROVEN when actual runtime_ifb provenance (source
    native_worker_ready) is absent or malformed for either phase; a
    config-derived ceiling or model_id alone is NEVER sufficient.
    NOTQUALIFIED on an actual observed contract mismatch. QUALIFIED only
    when both phases report contract_verified, reported_ifb, matching
    strict-int reported max/gpu slots equal to the configured
    backend_max_concurrent, and slots covering the requested width.
    """
    per_phase: dict[str, Any] = {}
    unproven: list[str] = []
    failed: list[str] = []
    for phase, probe in (("pre", pre), ("post", post)):
        ifb = probe.get("runtime_ifb")
        ceiling = probe.get("backend_max_concurrent")
        ceiling_i = ceiling if _strict_int(ceiling) else None
        entry: dict[str, Any] = {
            "runtime_ifb_present": isinstance(ifb, dict),
            "configured_backend_max_concurrent": ceiling_i,
            "probe_cleanup": probe.get("capabilities_cleanup"),
        }
        if probe.get("capabilities_request_error") or probe.get(
            "capabilities_cleanup", {}
        ).get("pending") or probe.get("capabilities_cleanup", {}).get("error"):
            # Pending/failed owned probe close or request error: the IFB
            # provenance is not a fabricated positive — UNPROVEN.
            entry["state"] = "UNPROVEN"
            unproven.append(
                f"{phase}: capabilities probe incomplete (request_error="
                f"{probe.get('capabilities_request_error')!r}, cleanup="
                f"{probe.get('capabilities_cleanup')!r})"
            )
            per_phase[phase] = entry
            continue
        if not isinstance(ifb, dict) or ifb.get("source") != "native_worker_ready":
            entry["state"] = "UNPROVEN"
            unproven.append(
                f"{phase}: runtime_ifb native_worker_ready provenance absent "
                "(a config ceiling or model_id alone is not live slot proof)"
            )
            per_phase[phase] = entry
            continue
        max_slots = (
            ifb.get("reported_max_slots")
            if _strict_int(ifb.get("reported_max_slots"))
            else None
        )
        gpu_slots = (
            ifb.get("reported_gpu_slots")
            if _strict_int(ifb.get("reported_gpu_slots"))
            else None
        )
        entry.update(
            {
                "contract_verified": ifb.get("contract_verified"),
                "reported_ifb": ifb.get("reported_ifb"),
                "reported_max_slots": max_slots,
                "reported_gpu_slots": gpu_slots,
                "reported_engine_max_batch_size": ifb.get(
                    "reported_engine_max_batch_size"
                ),
            }
        )
        problems: list[str] = []
        if ifb.get("contract_verified") is not True:
            problems.append("contract_verified is not True")
        if ifb.get("reported_ifb") is not True:
            problems.append("reported_ifb is not True")
        if max_slots is None or gpu_slots is None:
            problems.append("reported slots are not strict ints")
        else:
            if max_slots != gpu_slots:
                problems.append(
                    f"slot mismatch: max_slots={max_slots} gpu_slots={gpu_slots}"
                )
            if ceiling_i is None:
                problems.append(
                    "configured backend_max_concurrent is not a strict int"
                )
            elif max_slots != ceiling_i:
                problems.append(
                    f"reported_max_slots {max_slots} != configured slots {ceiling_i}"
                )
            elif max_slots < requested_width:
                problems.append(
                    f"reported slots {max_slots} < requested width {requested_width}"
                )
        if problems:
            entry["state"] = "NOTQUALIFIED"
            entry["problems"] = problems
            failed.extend(f"{phase}: {p}" for p in problems)
        else:
            entry["state"] = "QUALIFIED"
        per_phase[phase] = entry
    if failed:
        overall = "NOTQUALIFIED"
    elif unproven:
        overall = "UNPROVEN"
    else:
        overall = "QUALIFIED"
    return {
        "status": overall,
        "per_phase": per_phase,
        "unproven_reasons": unproven,
        "failure_reasons": failed,
        "basis": (
            "live runtime_ifb native_worker_ready contract cross-checked "
            "against configured slots; not derivable from model_id or a "
            "config-derived ceiling"
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Run orchestration
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    base_url: str
    mode: str
    concurrency: int
    manifest: Path
    manifest_sha256: str
    corpus_root: Optional[Path]
    output: Path
    chunk_ms: int
    pace: bool
    request_deadline_s: float
    overall_deadline_s: float
    post_final_window_s: float
    label: str
    baseline: Optional[Path] = None
    baseline_recompute: Optional[Path] = None
    baseline_recompute_sha256: Optional[str] = None
    ws_path: str = "/asr/stream"
    http_path: str = "/v1/audio/transcriptions"
    post_warm_repeat: bool = False
    identity: Optional[dict[str, Any]] = None  # output of load_identity_file
    # OPT-IN dual pinned corpus-input contract (both or neither; see
    # --corpus-detail / --corpus-detail-sha256).
    corpus_detail: Optional[Path] = None
    corpus_detail_sha256: Optional[str] = None
    single_long_diagnostic: bool = False
    language: Optional[str] = None
    controls_only: bool = False


def chunk_bytes_for_ms(chunk_ms: int) -> int:
    # 16 kHz * 2 bytes/sample * ms/1000, rounded to an even sample boundary.
    raw = EXPECTED_RATE * EXPECTED_SAMPLE_WIDTH * chunk_ms / 1000.0
    return max(2, int(round(raw / 2)) * 2)


def pace_seconds_for_ms(chunk_ms: int) -> float:
    return chunk_ms / 1000.0


@dataclass
class PhaseInfo:
    """Whole-phase measurements kept separate from per-request metrics."""

    name: str
    elapsed_s: float  # complete phase elapsed INCLUDING handshake/pacing
    realized_width: int  # measured live in-flight maximum, not metadata
    requested_width: int


@dataclass
class RunOutcome:
    config: RunConfig
    corpus: Corpus
    pre: dict[str, Any]
    post: dict[str, Any]
    b1: list[UtteranceResult]
    b2: list[UtteranceResult]
    b1_phase: Optional[PhaseInfo]
    b2_phase: Optional[PhaseInfo]
    post_warm_repeat: Optional[UtteranceResult]
    source_hashes: dict[str, Any]
    started_wall: str
    finished_wall: str
    input_admission: dict[str, Any] = field(default_factory=dict)
    # Owned async lifetime ledger (F4). Holds ACTUAL live task/resource
    # references until the CLI process lifetime ends; the report snapshot is
    # regenerated from it AFTER cleanup.
    lifetime_registry: Optional[AsyncLifetimeRegistry] = None


def _make_ws_factory(base_url: str, ws_path: str, language: str, sample_rate: int):
    """Return a factory that connects to the ASR websocket."""

    async def factory() -> WsConn:
        import websockets

        request_id = f"asr-controls-{uuid.uuid4().hex}"
        root = base_url.rstrip("/")
        if root.startswith("https://"):
            root = "wss://" + root[len("https://"):]
        elif root.startswith("http://"):
            root = "ws://" + root[len("http://"):]
        url = f"{root}{ws_path}?language={language}&sample_rate={sample_rate}&vad=none"
        request_headers = {"X-Request-ID": request_id}
        ws = await websockets.connect(
            url, additional_headers=request_headers, open_timeout=10,
            close_timeout=5, max_size=2 ** 22, ping_interval=None,
        )

        class _Adapter:
            async def send(self, data: Any) -> None:  # noqa: D401
                await ws.send(data)

            async def recv(self) -> Any:
                from websockets.exceptions import ConnectionClosed

                try:
                    return await ws.recv()
                except ConnectionClosed as exc:  # noqa: PERF203
                    code = getattr(exc, "code", None)
                    reason = getattr(exc, "reason", "") or ""
                    raise ConnClosed(code=code, reason=str(reason)) from exc

            async def close(self, code: int = 1000) -> None:
                try:
                    await ws.close(code=code)
                except Exception:  # noqa: BLE001
                    pass

            async def abort(self) -> None:
                transport = getattr(ws, "transport", None)
                if transport is None or not callable(getattr(transport, "abort", None)):
                    raise RuntimeError("websocket transport abort is unavailable")
                transport.abort()

            def close_observation(self) -> dict[str, Any]:
                protocol = getattr(ws, "protocol", None)
                if protocol is None or not hasattr(protocol, "close_rcvd"):
                    return {
                        "status": "UNPROVEN",
                        "source": "websockets.protocol.close_rcvd unavailable",
                    }
                peer_close = getattr(protocol, "close_rcvd")
                derived_code = getattr(ws, "close_code", None)
                if peer_close is None:
                    return {
                        "status": "ABSENT",
                        "source": "websockets.protocol.close_rcvd",
                        "peer_close_observed": False,
                        "peer_code": None,
                        "peer_reason": None,
                        "derived_close_code": derived_code,
                    }
                return {
                    "status": "PRESENT",
                    "source": "websockets.protocol.close_rcvd",
                    "peer_close_observed": True,
                    "peer_code": getattr(peer_close, "code", None),
                    "peer_reason": getattr(peer_close, "reason", None),
                    "derived_close_code": derived_code,
                }

        adapter = _Adapter()
        adapter.request_id = request_id
        adapter.request_headers = dict(request_headers)
        adapter.request_id_sent = True
        return adapter

    return factory


async def run_control_only(config: RunConfig) -> dict[str, Any]:
    """Bounded opt-in control probe; never contributes performance rows.

    The native endpoint exposes the cancel receipt only in the ACK.  It does
    not expose the stream id before cancellation, so a valid ACK is recorded
    but remains UNPROVEN for identity unless an adapter supplies an explicit
    expected id.  Control probes never infer that an arbitrary non-empty id is
    the expected session id.
    """
    if config.mode != "ws":
        return {"status": "NOT_RUN", "reason": "controls-only requires --mode ws"}
    if config.output.exists():
        raise SystemExit(f"output dir already exists (refusing to overwrite): {config.output}")
    started = time.monotonic()
    deadline = started + config.overall_deadline_s
    registry = AsyncLifetimeRegistry(f"controls:{config.label}")
    admission = await admit_corpus_inputs(config, deadline)
    if admission.get("status") != "admitted" or not (admission.get("pending") or {}).get("cleanup_complete"):
        return {"scope": "CONTROL_ONLY", "status": "NOT_RUN", "performance_status": "NOT_RUN",
                "reason": "input admission did not complete", "input_admission": {k: v for k, v in admission.items() if k != "loader"}}
    snap = admission["loader"].snapshot()
    corpus = snap["corpus"]
    pcm = snap["admitted"][corpus.items[0].order].pcm
    factory = _make_ws_factory(config.base_url, config.ws_path, config.language or "en", EXPECTED_RATE)
    cases: dict[str, Any] = {}
    c1 = c2 = third = None

    async def cancel_case() -> dict[str, Any]:
        conn = await _connect_owned(
            factory, deadline, "control cancel connect", registry,
            kind="control-ws", late_close=lambda c: c.close(),
            cleanup_deadline_mono=deadline,
        )
        q: asyncio.Queue = asyncio.Queue()
        frames: list[dict[str, Any]] = []
        request_id = getattr(conn, "request_id", None)
        request_id_sent = getattr(conn, "request_id_sent", False) is True
        request_headers = getattr(conn, "request_headers", None)
        expected_id = getattr(conn, "expected_session_id", None)
        # server/main.py starts each stream at epoch=0 and increments before
        # scheduling the first cancel control job; the first valid ACK is 1.
        expected_epoch = 1
        receiver_error: Optional[str] = None

        async def receive() -> None:
            nonlocal receiver_error
            try:
                while True:
                    raw = await conn.recv()
                    if not isinstance(raw, str):
                        raise RuntimeError(f"control frame is not JSON text: {raw!r}")
                    q.put_nowait(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                receiver_error = f"{type(exc).__name__}: {exc}"
                q.put_nowait({"type": "_control_error", "error": receiver_error})

        recv_task = registry.track_task(asyncio.create_task(receive()), "control receiver")
        partial = False
        ack: Optional[dict[str, Any]] = None
        late_final = False
        hard_error: Optional[str] = None
        send_events: list[dict[str, Any]] = []
        partial_arrival_mono: Optional[float] = None
        cancel_send_mono: Optional[float] = None
        ack_arrival_mono: Optional[float] = None
        quiet_window_start_mono: Optional[float] = None
        quiet_window_end_mono: Optional[float] = None
        try:
            offset = 0
            chunk = chunk_bytes_for_ms(config.chunk_ms)
            while offset < len(pcm) and not partial:
                send_started = time.monotonic()
                chunk_sent = min(chunk, len(pcm) - offset)
                await _bounded(conn.send(pcm[offset:offset + chunk_sent]), deadline,
                               "control PCM send", registry)
                offset += chunk_sent
                send_events.append({"offset_end": offset, "sent_mono": send_started,
                                    "chunk_bytes": chunk_sent})
                pace_s = pace_seconds_for_ms(config.chunk_ms) if config.pace else 0.05
                poll_deadline = min(deadline, send_started + pace_s)
                while not partial and time.monotonic() < poll_deadline:
                    try:
                        msg = await _poll_queue(q, max(0.0, _remaining(poll_deadline)), registry)
                    except _PollTimeout:
                        break
                    frames.append(msg)
                    if msg.get("type") == "_control_error":
                        raise RuntimeError(f"_control_error: {msg.get('error')}")
                    partial = msg.get("type") == "partial" and bool(str(msg.get("text") or "").strip())
                    if partial and partial_arrival_mono is None:
                        partial_arrival_mono = time.monotonic()
                if not partial and config.pace:
                    await asyncio.sleep(max(0.0, _remaining(poll_deadline)))
            if not partial:
                raise TimeoutError("no partial observed during bounded upload")
            cancel_send_mono = time.monotonic()
            await _bounded(conn.send(json.dumps({"command": "cancel"})), deadline,
                           "cancel send", registry)
            ack_deadline = min(deadline, time.monotonic() + max(0.5, config.post_final_window_s + 0.5))
            while ack is None and time.monotonic() < ack_deadline:
                try:
                    msg = await _poll_queue(q, max(0.0, _remaining(ack_deadline)), registry)
                except _PollTimeout:
                    break
                frames.append(msg)
                mtype = msg.get("type")
                if mtype == "_control_error":
                    raise RuntimeError(f"_control_error: {msg.get('error')}")
                if mtype == "final":
                    late_final = True
                if mtype == "cancel_ack":
                    ack = msg
                    ack_arrival_mono = time.monotonic()
            if ack is None:
                raise TimeoutError("cancel ACK not observed")
            if set(ack) != {"type", "id", "epoch"} or ack.get("type") != "cancel_ack":
                raise RuntimeError(f"cancel ACK schema mismatch: {ack!r}")
            if not isinstance(ack.get("id"), str) or not ack["id"]:
                raise RuntimeError(f"cancel ACK id is not a non-empty string: {ack!r}")
            if not _strict_int(ack.get("epoch")) or ack["epoch"] <= 0:
                raise RuntimeError(f"cancel ACK epoch is not a positive strict int: {ack!r}")
            if _strict_int(expected_epoch) and ack["epoch"] != expected_epoch:
                raise RuntimeError(f"cancel ACK epoch mismatch: expected {expected_epoch}, got {ack['epoch']}")
            if expected_id is not None and ack["id"] != expected_id:
                raise RuntimeError(f"cancel ACK id mismatch: expected {expected_id!r}, got {ack['id']!r}")
            quiet_window_start_mono = time.monotonic()
            quiet_end = min(deadline, quiet_window_start_mono + config.post_final_window_s)
            quiet_window_end_mono = quiet_end
            while time.monotonic() < quiet_end:
                try:
                    msg = await _poll_queue(q, max(0.0, _remaining(quiet_end)), registry)
                except _PollTimeout:
                    break
                frames.append(msg)
                if msg.get("type") == "_control_error":
                    raise RuntimeError(f"_control_error: {msg.get('error')}")
                if msg.get("type") == "final":
                    late_final = True
                if msg.get("type") == "cancel_ack" and msg != ack:
                    raise RuntimeError(f"conflicting duplicate cancel ACK: {msg!r}")
            if late_final:
                raise RuntimeError("late final observed after cancel ACK")
            identity = "QUALIFIED" if expected_id is not None else "UNPROVEN"
            return {
                "status": identity,
                "partial_seen": partial,
                "cancel_ack": ack,
                "expected_id": expected_id,
                "expected_id_observable": expected_id is not None,
                "request_id": request_id,
                "request_id_sent": request_id_sent,
                "request_headers": request_headers,
                "expected_epoch": expected_epoch,
                "unproven_reasons": ([] if expected_id is not None else [
                    "server-generated session id is not observable before cancel ACK"
                ]),
                "late_final": late_final,
                "raw_frames": frames,
                "send_events": send_events,
                "partial_arrival_mono": partial_arrival_mono,
                "cancel_send_mono": cancel_send_mono,
                "ack_arrival_mono": ack_arrival_mono,
                "quiet_window_start_mono": quiet_window_start_mono,
                "quiet_window_end_mono": quiet_window_end_mono,
            }
        except Exception as exc:  # noqa: BLE001
            hard_error = f"{type(exc).__name__}: {exc}"
            return {"status": "HARD_FAIL", "partial_seen": partial,
                    "cancel_ack": ack, "late_final": late_final,
                    "request_id": request_id,
                    "request_id_sent": request_id_sent,
                    "request_headers": request_headers,
                    "error": hard_error, "raw_frames": frames,
                    "send_events": send_events,
                    "partial_arrival_mono": partial_arrival_mono,
                    "cancel_send_mono": cancel_send_mono,
                    "ack_arrival_mono": ack_arrival_mono,
                    "quiet_window_start_mono": quiet_window_start_mono,
                    "quiet_window_end_mono": quiet_window_end_mono}
        finally:
            await registry.cancel_observe(recv_task, max(0.0, _remaining(deadline)), "control receiver cleanup")
            budget = _remaining(deadline)
            if budget > 0:
                close_task = registry.track_task(asyncio.create_task(conn.close()), "control connection close")
                registry.note_resource_close_requested(conn, "control close scheduled")
                if await registry.wait_done(close_task, budget):
                    registry.note_resource_closed(conn, True, "control close observed complete")
                else:
                    close_task.cancel()
                    registry.note_cancel_requested(close_task, "control connection close")
                    registry.note_resource_closed(conn, False, "control close not observed")
            await registry.drain(max(0.0, _remaining(deadline)))

    try:
        cases["cancel_midupload"] = await cancel_case()
        fresh = await run_ws_utterance(factory, pcm, order=0, file=corpus.items[0].file,
                                       item_id=corpus.items[0].id, warm=False, pair_index=None,
                                       audio_s=corpus.items[0].duration_s, transcript=corpus.items[0].transcript,
                                       chunk_bytes=chunk_bytes_for_ms(config.chunk_ms),
                                       pace_s=pace_seconds_for_ms(config.chunk_ms) if config.pace else 0.0,
                                       request_deadline_s=min(config.request_deadline_s, max(1.0, _remaining(deadline))),
                                       post_final_window_s=config.post_final_window_s,
                                       overall_deadline_mono=deadline, registry=registry)
        cases["fresh_after_cancel"] = {"status": "PASS" if fresh.ok else "HARD_FAIL", "result": asdict(fresh)}
        # A transport close is intentionally recorded separately from cancel;
        # the next fresh request is the only positive recovery assertion here.
        conn = await _connect_owned(factory, deadline, "control disconnect connect", registry,
                                    kind="control-ws", late_close=lambda c: c.close(),
                                    cleanup_deadline_mono=deadline)
        try:
            await _bounded(conn.send(pcm[:chunk_bytes_for_ms(config.chunk_ms)]), deadline,
                           "disconnect PCM send", registry)
        finally:
            try:
                await _bounded(conn.abort(), deadline, "disconnect abort", registry)
            finally:
                if _remaining(deadline) > 0:
                    close_task = registry.track_task(
                        asyncio.create_task(conn.close()), "disconnect connection close"
                    )
                    registry.note_resource_close_requested(conn, "disconnect close scheduled")
                    if await registry.wait_done(close_task, max(0.0, _remaining(deadline))):
                        registry.note_resource_closed(conn, True, "disconnect close observed complete")
                    else:
                        close_task.cancel()
                        registry.note_cancel_requested(close_task, "disconnect connection close")
                        registry.note_resource_closed(conn, False, "disconnect close not observed")
        fresh2 = await run_ws_utterance(factory, pcm, order=1, file=corpus.items[0].file,
                                        item_id=corpus.items[0].id, warm=False, pair_index=None,
                                        audio_s=corpus.items[0].duration_s, transcript=corpus.items[0].transcript,
                                        chunk_bytes=chunk_bytes_for_ms(config.chunk_ms),
                                        pace_s=pace_seconds_for_ms(config.chunk_ms) if config.pace else 0.0,
                                        request_deadline_s=min(config.request_deadline_s, max(1.0, _remaining(deadline))),
                                        post_final_window_s=config.post_final_window_s,
                                        overall_deadline_mono=deadline, registry=registry)
        cases["disconnect_reconnect"] = {"status": "PASS" if fresh2.ok else "HARD_FAIL", "result": asdict(fresh2)}
        # Keep two accepted streams open while probing a third connection. A
        # busy/error close is the only positive capacity proof; a successful
        # third connection is a hard failure, never silently treated as pass.
        items = corpus.items[:2]
        if len(items) < 2:
            cases["b2_isolation"] = {"status": "NOT_RUN", "reason": "two distinct corpus references are required"}
            raise StopAsyncIteration
        c1 = await _connect_owned(factory, deadline, "capacity first", registry,
                                  kind="control-ws", late_close=lambda c: c.close(), cleanup_deadline_mono=deadline)
        c2 = await _connect_owned(factory, deadline, "capacity second", registry,
                                  kind="control-ws", late_close=lambda c: c.close(), cleanup_deadline_mono=deadline)
        pcm2 = snap["admitted"][items[1].order].pcm
        async def upload_open(conn: Any, audio: bytes, label: str) -> None:
            chunk = chunk_bytes_for_ms(config.chunk_ms)
            for offset in range(0, len(audio), chunk):
                await _bounded(conn.send(audio[offset:offset + chunk]), deadline,
                               f"{label} PCM", registry)
                if config.pace and offset + chunk < len(audio):
                    await asyncio.sleep(min(pace_seconds_for_ms(config.chunk_ms),
                                            max(0.0, _remaining(deadline))))
        # Upload both complete references while keeping both sessions open;
        # EOS is deliberately delayed until after the third admission probe.
        await asyncio.gather(
            upload_open(c1, pcm, "capacity first"),
            upload_open(c2, pcm2, "capacity second"),
        )
        async def b2_final(conn: Any, reference: str) -> dict[str, Any]:
            await _bounded(conn.send(b""), deadline, f"{reference} EOS", registry)
            while _remaining(deadline) > 0:
                raw = await _bounded(conn.recv(), deadline, f"{reference} final", registry)
                if not isinstance(raw, str):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "final":
                    return msg
                if msg.get("type") == "_control_error":
                    raise RuntimeError(f"{reference} _control_error")
            raise TimeoutError(f"{reference} final timeout")
        reject = None
        try:
            third = await _connect_owned(factory, deadline, "capacity third", registry,
                                         kind="control-ws", late_close=lambda c: c.close(), cleanup_deadline_mono=deadline)
            # /asr/stream accepts before the limiter check; handshake success
            # is not capacity evidence. Observe the typed close afterward.
            try:
                await _bounded(third.recv(), deadline, "capacity rejection observation", registry)
                cases["b2_isolation"] = {
                    "status": "HARD_FAIL",
                    "reason": "third handshake produced a normal payload instead of capacity close",
                }
            except ConnClosed as exc:
                try:
                    reason = json.loads(exc.reason or "{}")
                except (TypeError, ValueError):
                    reason = {}
                if (
                    exc.code == 4429
                    and reason.get("error") == "too_many_sessions"
                    and _strict_int(reason.get("current"))
                    and _strict_int(reason.get("limit"))
                    and reason["current"] >= reason["limit"] > 0
                ):
                    reject = {"typed": True, "code": exc.code, "reason": reason}
                else:
                    cases["b2_isolation"] = {
                        "status": "HARD_FAIL",
                        "reason": "third connection close was not typed capacity rejection",
                        "code": exc.code, "wire_reason": exc.reason,
                    }
            except TimeoutError as exc:
                cases["b2_isolation"] = {
                    "status": "HARD_FAIL", "reason": "third capacity observation timed out",
                    "error": str(exc),
                }
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            cases["b2_isolation"] = {
                "status": "HARD_FAIL",
                "reason": "third handshake failed before typed capacity observation",
                "error": text,
            }
        # Only after the third-admission probe do the two accepted sessions
        # receive EOS and produce their independent finals.
        f1_task = asyncio.create_task(b2_final(c1, "reference_1"))
        f2_task = asyncio.create_task(b2_final(c2, "reference_2"))
        registry.track_task(f1_task, "B2 reference 1 final")
        registry.track_task(f2_task, "B2 reference 2 final")
        f1, f2 = await asyncio.gather(f1_task, f2_task)
        match1 = word_error_rate(items[0].transcript, str(f1.get("text") or ""))
        match2 = word_error_rate(items[1].transcript, str(f2.get("text") or ""))
        swap1 = word_error_rate(items[0].transcript, str(f2.get("text") or ""))
        swap2 = word_error_rate(items[1].transcript, str(f1.get("text") or ""))
        if match1["errors"] != 0 or match2["errors"] != 0 or (swap1["errors"] == 0 and swap2["errors"] == 0):
            raise RuntimeError("B2 final/reference mapping failed or finals were swapped")
        if "b2_isolation" not in cases:
                cases["b2_isolation"] = {"status": "PASS" if reject is not None else "HARD_FAIL",
                                      "accepted_references": [items[0].id, items[1].id],
                                      "request_ids": [getattr(c1, "request_id", None),
                                                       getattr(c2, "request_id", None),
                                                       getattr(third, "request_id", None)],
                                      "request_ids_sent": [getattr(c1, "request_id_sent", False) is True,
                                                            getattr(c2, "request_id_sent", False) is True,
                                                            getattr(third, "request_id_sent", False) is True],
                                      "finals": [f1, f2],
                                      "reference_match": [match1, match2],
                                      "swap_check": [swap1, swap2],
                                      "third_rejection": reject}
    except Exception as exc:
        cases.setdefault("exception", {"status": "HARD_FAIL", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        admission["loader"].cancel_event.set()
        for close_conn, close_label in ((c1, "capacity first close"),
                                        (c2, "capacity second close"),
                                        (third, "capacity third close")):
            if close_conn is not None and _remaining(deadline) > 0:
                close_task = registry.track_task(
                    asyncio.create_task(close_conn.close()), close_label
                )
                registry.note_resource_close_requested(close_conn, f"{close_label} scheduled")
                if await registry.wait_done(close_task, max(0.0, _remaining(deadline))):
                    registry.note_resource_closed(close_conn, True, f"{close_label} observed complete")
                else:
                    close_task.cancel()
                    registry.note_cancel_requested(close_task, close_label)
                    registry.note_resource_closed(close_conn, False, f"{close_label} not observed")
        await registry.drain(max(0.0, _remaining(deadline)))
    required = ("cancel_midupload", "fresh_after_cancel", "disconnect_reconnect", "b2_isolation")
    hard_failed = "exception" in cases or any(cases.get(k, {}).get("status") == "HARD_FAIL" for k in required)
    complete = all(cases.get(k, {}).get("status") == "PASS" for k in required)
    overall = "PASS" if complete else ("HARD_FAIL" if hard_failed else "UNPROVEN")
    return {"schema": 1, "scope": "CONTROL_ONLY", "status": overall,
            "performance_status": "NOT_RUN", "phase": config.label, "cases": cases,
            "source_hashes": _source_hashes(), "deadline_s": config.overall_deadline_s,
            "async_lifetime": registry.snapshot()}


async def _run_one(
    conn_factory,
    corpus: Corpus,
    item: CorpusItem,
    config: RunConfig,
    mode: str,
    chunk_bytes: int,
    pace_s: float,
    deadline_mono: float,
    *,
    warm: bool,
    pair_index: Optional[int],
    tracker: Optional[WidthTracker] = None,
    http_open_conn: Optional[Callable[[Any, float], Awaitable[Any]]] = None,
    admitted: Optional[dict[int, AdmittedInput]] = None,
    registry: Optional[AsyncLifetimeRegistry] = None,
) -> UtteranceResult:
    admitted_input = (admitted or {}).get(item.order)
    if admitted_input is None:
        # Fail-closed migration (owned-input boundary): path-based synchronous
        # input reads were removed from the request path. Callers that only
        # pass a path get an EXPLICIT failed row — never a hidden re-read and
        # never a timed fallback.
        return UtteranceResult(
            order=item.order, file=item.file, id=item.id, warm=warm,
            pair_index=pair_index,
            error=(
                "input not admitted through the owned input boundary "
                "(path-based reads removed; run via run_gate so the owned "
                "thread admits the frozen corpus first)"
            ),
            audio_s=item.duration_s, transcript=item.transcript,
        )
    pcm = admitted_input.pcm
    check = admitted_input.check
    # Defensive re-check from the ADMISSION-time evidence (immutable bytes);
    # the file is NOT re-read or re-hashed here.
    if not check.ok:
        return UtteranceResult(
            order=item.order, file=item.file, id=item.id, warm=warm,
            pair_index=pair_index, error=f"input invalid: {check.error}",
            audio_s=item.duration_s, transcript=item.transcript,
        )
    if tracker is not None:
        tracker.enter()
    try:
        if mode == "http":
            # Contract fix: run_http_utterance's keyword is item_id (the old
            # call passed id=item.id, a TypeError for every HTTP row).
            return await run_http_utterance(
                config.base_url.rstrip("/") + config.http_path,
                admitted_input.wav_bytes,
                wav_name=item.file,
                order=item.order, file=item.file, item_id=item.id, warm=warm,
                pair_index=pair_index, audio_s=check.duration_s,
                transcript=item.transcript, language=config.language, deadline_mono=deadline_mono,
                open_conn=http_open_conn,
                registry=registry,
            )
        return await run_ws_utterance(
            conn_factory, pcm,
            order=item.order, file=item.file, item_id=item.id, warm=warm,
            pair_index=pair_index, audio_s=check.duration_s,
            transcript=item.transcript, chunk_bytes=chunk_bytes, pace_s=pace_s,
            request_deadline_s=config.request_deadline_s,
            post_final_window_s=config.post_final_window_s,
            overall_deadline_mono=deadline_mono,
            registry=registry,
        )
    finally:
        if tracker is not None:
            tracker.exit()


async def run_b1(
    conn_factory: Callable[[], Awaitable[WsConn]],
    corpus: Corpus,
    *,
    mode: str,
    config: RunConfig,
    deadline_mono: float,
    http_open_conn: Optional[Callable[[Any, float], Awaitable[Any]]] = None,
    admitted: Optional[dict[int, AdmittedInput]] = None,
    registry: Optional[AsyncLifetimeRegistry] = None,
) -> tuple[list[UtteranceResult], PhaseInfo]:
    """Phase B1 ONLY: the 100 ordered utterances once, sequentially (width 1).

    Realized width is measured live; it must be exactly 1.
    """
    results: list[UtteranceResult] = []
    chunk_bytes = chunk_bytes_for_ms(config.chunk_ms)
    pace_s = pace_seconds_for_ms(config.chunk_ms) if config.pace else 0.0
    tracker = WidthTracker()
    phase_start = time.monotonic()
    if registry is None:
        registry = AsyncLifetimeRegistry("b1")
    for item in corpus.items:
        if time.monotonic() > deadline_mono:
            results.append(
                UtteranceResult(
                    order=item.order, file=item.file, id=item.id, warm=item.order < WARM_COUNT,
                    pair_index=None, audio_s=item.duration_s, transcript=item.transcript,
                    error="overall deadline exceeded before utterance",
                )
            )
            continue
        results.append(
            await _run_one(
                conn_factory, corpus, item, config, mode,
                chunk_bytes, pace_s, deadline_mono, warm=item.order < WARM_COUNT,
                pair_index=None, tracker=tracker, http_open_conn=http_open_conn,
                admitted=admitted, registry=registry,
            )
        )
    phase = PhaseInfo(
        name="b1",
        elapsed_s=time.monotonic() - phase_start,
        realized_width=tracker.max,
        requested_width=1,
    )
    return results, phase


async def run_b2(
    conn_factory: Callable[[], Awaitable[WsConn]],
    corpus: Corpus,
    *,
    mode: str,
    config: RunConfig,
    deadline_mono: float,
    http_open_conn: Optional[Callable[[Any, float], Awaitable[Any]]] = None,
    admitted: Optional[dict[int, AdmittedInput]] = None,
    registry: Optional[AsyncLifetimeRegistry] = None,
) -> tuple[list[UtteranceResult], PhaseInfo]:
    """Phase B2 ONLY: the SAME 100 utterances once, as 50 ADJACENT pairs.

    Pairing rule (frozen reference run_b2_stagger.py): pair i is corpus items
    (2i, 2i+1), preserving the frozen order. Both members run concurrently at
    width 2 inside one shared bounded gather. The pair span is keyed from the
    earliest first-PCM-send to the latest own final (handshake EXCLUDED); the
    complete phase elapsed INCLUDING handshake/pacing is reported separately
    in the PhaseInfo. Realized width is measured live; it must reach 2.
    """
    chunk_bytes = chunk_bytes_for_ms(config.chunk_ms)
    pace_s = pace_seconds_for_ms(config.chunk_ms) if config.pace else 0.0
    results: list[UtteranceResult] = []
    tracker = WidthTracker()
    n = len(corpus.items)
    if registry is None:
        registry = AsyncLifetimeRegistry("b2")
    pair_count = n // 2
    phase_start = time.monotonic()
    for pair_index in range(pair_count):
        a_item = corpus.items[2 * pair_index]
        b_item = corpus.items[2 * pair_index + 1]
        if time.monotonic() > deadline_mono:
            for item in (a_item, b_item):
                results.append(
                    UtteranceResult(
                        order=item.order, file=item.file, id=item.id,
                        warm=item.order < WARM_COUNT, pair_index=pair_index,
                        audio_s=item.duration_s, transcript=item.transcript,
                        error="overall deadline exceeded before pair",
                    )
                )
            continue
        remaining = _remaining(deadline_mono)
        if remaining <= 0:
            # Expired budget: NO new tasks/coroutines are created, no probe
            # is started and no floor wait is granted. Explicit failed rows
            # keep the fixed 100-row denominator.
            for item in (a_item, b_item):
                results.append(
                    UtteranceResult(
                        order=item.order, file=item.file, id=item.id,
                        warm=item.order < WARM_COUNT, pair_index=pair_index,
                        audio_s=item.duration_s, transcript=item.transcript,
                        error="overall deadline expired before pair tasks were "
                              "created (no pair work started)",
                    )
                )
            continue
        # F4: two ACTUAL owned tasks (never a wait_for(gather) whose
        # cancellation completion is awaited unbounded). The registry
        # retains BOTH underlying request tasks until observed done.
        phase_a = f"b2 pair {pair_index} member a"
        phase_b = f"b2 pair {pair_index} member b"
        task_a = registry.track_task(
            asyncio.ensure_future(
                _run_one(conn_factory, corpus, a_item, config, mode, chunk_bytes,
                         pace_s, deadline_mono, warm=a_item.order < WARM_COUNT,
                         pair_index=pair_index, tracker=tracker,
                         http_open_conn=http_open_conn, admitted=admitted,
                         registry=registry)
            ),
            phase_a,
        )
        task_b = registry.track_task(
            asyncio.ensure_future(
                _run_one(conn_factory, corpus, b_item, config, mode, chunk_bytes,
                         pace_s, deadline_mono, warm=b_item.order < WARM_COUNT,
                         pair_index=pair_index, tracker=tracker,
                         http_open_conn=http_open_conn, admitted=admitted,
                         registry=registry)
            ),
            phase_b,
        )
        try:
            done, pending = await asyncio.wait(
                {task_a, task_b}, timeout=remaining
            )
        except asyncio.CancelledError:
            # External caller cancellation: cancel each member EXACTLY ONCE,
            # keep both registered, propagate the cancellation.
            for t, ph in ((task_a, phase_a), (task_b, phase_b)):
                if not t.done():
                    t.cancel()
                registry.note_cancel_requested(t, ph)
            raise
        pair_results: list[UtteranceResult] = []
        for t, item, ph in (
            (task_a, a_item, phase_a),
            (task_b, b_item, phase_b),
        ):
            if t in done:
                try:
                    pair_results.append(t.result())
                except Exception as exc:  # noqa: BLE001
                    # Completed-with-exception is retrieved (never silently
                    # dropped) and becomes an explicit failed row.
                    pair_results.append(
                        UtteranceResult(
                            order=item.order, file=item.file, id=item.id,
                            warm=item.order < WARM_COUNT, pair_index=pair_index,
                            audio_s=item.duration_s, transcript=item.transcript,
                            error=f"pair member raised: {type(exc).__name__}: {exc}",
                        )
                    )
            else:
                # Still pending at the absolute deadline: cancel EXACTLY
                # ONCE; the task stays OWNED by the registry (never dropped,
                # never replaced by a fabricated success row).
                t.cancel()
                registry.note_cancel_requested(t, ph)
                pair_results.append(
                    UtteranceResult(
                        order=item.order, file=item.file, id=item.id,
                        warm=item.order < WARM_COUNT, pair_index=pair_index,
                        audio_s=item.duration_s, transcript=item.transcript,
                        cleanup_pending=True,
                        cleanup_error=(
                            "pair member still pending at the shared deadline; "
                            "cancellation requested once"
                        ),
                        error=(
                            "pair bounded wait expired: member still pending "
                            "(owned task retained, cancel requested once)"
                        ),
                    )
                )
        # Finite cleanup observation for expired members INSIDE the reserved
        # cleanup budget of the same overall deadline; no renewed budget.
        grace = min(CLEANUP_RESERVE_S, max(0.0, _remaining(deadline_mono)))
        for t, item, ph, res in (
            (task_a, a_item, phase_a, pair_results[0]),
            (task_b, b_item, phase_b, pair_results[1]),
        ):
            if not t.done() and grace > 0:
                await registry.wait_done(t, grace)
            if t.done() and res.error is not None and "still pending" in res.error:
                # The member finished during the grace window; its ACTUAL
                # late outcome is recorded (never masked as success).
                entry = registry.entry_for(t) or {}
                note = f"member later completed (state={entry.get('state')})"
                if entry.get("exception"):
                    note += f" exception={entry['exception']}"
                res.cleanup_error = (
                    f"{res.cleanup_error}; {note}" if res.cleanup_error else note
                )
        pair = tuple(pair_results)
        # Pair span: earliest first-PCM-send → latest own final (handshake
        # excluded). Only computable when both rows recorded both stamps.
        sends = [r.first_send_mono for r in pair if r.first_send_mono is not None]
        finals = [r.final_arrival_mono for r in pair if r.final_arrival_mono is not None]
        span: Optional[float] = None
        if len(sends) == 2 and len(finals) == 2:
            delta, err = _time_delta(max(finals), min(sends))
            if err is None:
                span = delta
        for res in pair:
            res.pair_span_s = span
        results.extend(pair)
    phase = PhaseInfo(
        name="b2",
        elapsed_s=time.monotonic() - phase_start,
        realized_width=tracker.max,
        requested_width=2,
    )
    return results, phase


# ──────────────────────────────────────────────────────────────────────
# Metrics / qualification
# ──────────────────────────────────────────────────────────────────────


def _derive_rtf(rows: list[UtteranceResult]) -> None:
    """RTF = request_runtime_s / audio_duration_s (lower is better)."""
    for r in rows:
        if r.ok and r.rtf is None and r.request_wall_s and r.audio_s and r.audio_s > 0:
            r.rtf = r.request_wall_s / r.audio_s


def compute_metrics(
    b1: list[UtteranceResult],
    b2: list[UtteranceResult],
    corpus: Corpus,
    *,
    identity_present: bool,
    b1_phase: Optional[PhaseInfo] = None,
    b2_phase: Optional[PhaseInfo] = None,
) -> dict[str, Any]:
    """Compute app B1/B2 metrics under the frozen percentile rule.

    Denominators are strictly separated:
      * B1 aggregate = sum(audio)/sum(valid request walls);
      * B2 aggregate = sum(audio)/sum(50 actual pair spans);
      * B2 sum-latency throughput = sum(audio)/sum(all 100 request walls) —
        a distinct metric, never mixed with the aggregate.
    Failed/missing rows invalidate the aggregate gate (it becomes None and the
    qualification is NOTQUALIFIED); successful-only truncated means are never
    substituted for it. Per-row stats always include all 100 rows (warm +
    outliers); only the aggregate denominator is validity-gated.
    """
    completed = [r for r in b1 if r.ok]
    b2_completed = [r for r in b2 if r.ok]
    _derive_rtf(completed)
    _derive_rtf(b2_completed)

    request_walls = [r.request_wall_s for r in completed if r.request_wall_s is not None]
    rtfs = [r.rtf for r in completed if r.rtf is not None]
    eos_finals = [r.eos_to_final_s for r in completed if r.eos_to_final_s is not None]
    first_partials = [
        r.first_partial_latency_s
        for r in completed
        if r.first_partial_latency_s is not None
    ]

    b2_walls = [r.request_wall_s for r in b2_completed if r.request_wall_s is not None]
    b2_rtfs = [r.rtf for r in b2_completed if r.rtf is not None]

    # Per-pair distribution (audio / that pair's own span). Unequal pair
    # spans make this distribution distinct from the whole aggregate.
    pair_audio: dict[int, float] = {}
    pair_span: dict[int, float] = {}
    for r in b2:
        if r.pair_index is None:
            continue
        pair_audio[r.pair_index] = pair_audio.get(r.pair_index, 0.0) + r.audio_s
        if r.pair_span_s is not None:
            pair_span[r.pair_index] = r.pair_span_s
    per_pair_ratios = [
        pair_audio[idx] / pair_span[idx]
        for idx in sorted(pair_span)
        if pair_span[idx] > 0
    ]

    # B1 aggregate: sum audio / sum valid request walls.
    b1_total_audio = sum(r.audio_s for r in completed)
    b1_total_wall = sum(r.request_wall_s or 0.0 for r in completed)
    b1_throughput = b1_total_audio / b1_total_wall if b1_total_wall > 0 else None

    # B2 aggregate: sum audio / sum of the 50 ACTUAL pair spans. Every one of
    # the 100 rows must be ok and every one of the 50 pair spans present, else
    # the aggregate gate is INVALID (None) — never a truncated-success mean.
    expected_pairs = len(corpus.items) // 2 if corpus.items else len(b2) // 2
    b2_all_rows_ok = bool(b2) and len(b2_completed) == len(b2)
    b2_all_spans = len(pair_span) == expected_pairs and all(
        pair_span[idx] is not None and pair_span[idx] > 0 for idx in pair_span
    )
    b2_aggregate_valid = b2_all_rows_ok and b2_all_spans and bool(b2)
    if b2:
        b2_sum_audio = sum(r.audio_s for r in b2)
        b2_sum_spans = sum(pair_span.values()) if b2_all_spans else 0.0
        b2_aggregate = (
            b2_sum_audio / b2_sum_spans if b2_aggregate_valid and b2_sum_spans > 0 else None
        )
        b2_sum_walls = sum(r.request_wall_s or 0.0 for r in b2)
        b2_sum_latency = (
            b2_sum_audio / b2_sum_walls
            if b2_all_rows_ok and b2_sum_walls > 0
            else None
        )
    else:
        b2_sum_audio = 0.0
        b2_sum_spans = 0.0
        b2_aggregate = None
        b2_sum_latency = None

    first_partial_coverage = (len(first_partials) / len(b1)) if b1 else 0.0

    b1_quality = _quality_summary(b1, corpus)
    b2_quality = _quality_summary(b2, corpus)
    # Keep the historical top-level quality shape for existing consumers:
    # B1 is authoritative when present, while B2-only runs get real quality.
    quality_alias = b1_quality if b1 else b2_quality

    return {
        "b1": {
            "rows_total": len(b1),
            "rows_ok": len(completed),
            "rows_failed": len(b1) - len(completed),
            "request_wall_s": summary_stats(request_walls),
            "rtf": summary_stats(rtfs),
            "rtf_semantics": "request_runtime_s / audio_duration_s (lower is better)",
            "eos_to_final_s": summary_stats(eos_finals),
            "first_partial_latency_s": (
                summary_stats(first_partials) if first_partials else None
            ),
            "first_partial_coverage": first_partial_coverage,
            "first_partial_unproven": not first_partials,
            "throughput_audio_s_per_s": b1_throughput,
            "throughput_semantics": "sum(audio) / sum(valid request walls)",
            "phase_elapsed_s": b1_phase.elapsed_s if b1_phase else None,
            "realized_width": b1_phase.realized_width if b1_phase else None,
            "requested_width": b1_phase.requested_width if b1_phase else None,
            "realized_width_matches": (
                b1_phase.realized_width == 1 if b1_phase else None
            ),
        },
        "b2": {
            "rows_total": len(b2),
            "rows_ok": len(b2_completed),
            "rows_failed": len(b2) - len(b2_completed),
            "request_wall_s": summary_stats(b2_walls),
            "rtf": summary_stats(b2_rtfs),
            "rtf_semantics": "request_runtime_s / audio_duration_s (lower is better)",
            "pair_span_s": summary_stats(
                [pair_span[idx] for idx in sorted(pair_span)]
            ),
            "pair_span_semantics": (
                "earliest first-PCM-send to latest own final per pair; "
                "handshake excluded"
            ),
            "per_pair_throughput_audio_s_per_s": summary_stats(per_pair_ratios),
            "aggregate_throughput_audio_s_per_s": b2_aggregate,
            "aggregate_semantics": "sum(audio) / sum(50 actual pair spans)",
            "aggregate_valid": b2_aggregate_valid if b2 else None,
            "sum_latency_throughput_audio_s_per_s": b2_sum_latency,
            "sum_latency_semantics": (
                "sum(audio) / sum(all 100 request walls); distinct from aggregate"
            ),
            "sum_audio_s": b2_sum_audio if b2 else None,
            "sum_pair_span_s": b2_sum_spans if b2 else None,
            "phase_elapsed_s": b2_phase.elapsed_s if b2_phase else None,
            "phase_elapsed_semantics": "complete phase wall INCLUDING handshake/pacing",
            "realized_width": b2_phase.realized_width if b2_phase else None,
            "requested_width": b2_phase.requested_width if b2_phase else None,
            "realized_width_matches": (
                b2_phase.realized_width == 2 if b2_phase else None
            ),
        },
        "quality": quality_alias,
        "quality_by_phase": {"b1": b1_quality, "b2": b2_quality},
        "identity_present": identity_present,
        "percentile_method": PERCENTILE_METHOD,
        "note": (
            "first_partial_latency is UNPROVEN when the server never emitted a "
            "nonempty partial during upload."
        ),
    }


def _quality_summary(b1: list[UtteranceResult], corpus: Corpus) -> dict[str, Any]:
    total_errors = 0
    total_ref = 0
    rows: list[dict[str, Any]] = []
    for r in b1:
        item_ref = next((it.transcript for it in corpus.items if it.order == r.order), "")
        if r.ok and r.transcript is not None:
            wer = word_error_rate(item_ref, r.transcript)
        else:
            wer = {
                "ref_words": len(normalize_text(item_ref).split()),
                "errors": len(normalize_text(item_ref).split()),
                "wer": 1.0 if item_ref else None,
                "failed_row": True,
            }
        total_errors += wer.get("errors") or 0
        total_ref += wer.get("ref_words") or 0
        r.wer = wer
        rows.append({"order": r.order, "file": r.file, "warm": r.warm, **wer})
    corpus_ok = (
        len(corpus.items) == FROZEN_ITEM_COUNT
        and total_ref == FROZEN_REF_WORDS
    )
    return {
        "rows": rows,
        "ref_words": total_ref,
        "errors": total_errors,
        "wer": (total_errors / total_ref) if total_ref else None,
        "corpus_complete": corpus_ok,
        "expected_item_count": FROZEN_ITEM_COUNT,
        "expected_ref_words": FROZEN_REF_WORDS,
        "normalization": "lowercase + strip all string.punctuation + whitespace split",
    }


def _threshold_check(value: Optional[float], threshold: float, direction: str) -> dict[str, Any]:
    """One threshold check: PASS / FAIL, or UNPROVEN when not evaluable.

    A threshold that cannot be evaluated is NEVER a PASS.
    """
    if value is None or not _valid_time_value(value):
        return {
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "status": "UNPROVEN",
        }
    ok = value >= threshold if direction == ">=" else value <= threshold
    return {
        "value": value,
        "threshold": threshold,
        "direction": direction,
        "status": "PASS" if ok else "FAIL",
    }


def evaluate_app_thresholds(
    b1_metrics: dict[str, Any],
    b2_metrics: dict[str, Any],
    quality: dict[str, Any],
    *,
    quality_by_phase: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Frozen app-internal paired thresholds (B1 vs B2 of the SAME app).

      * B2 aggregate throughput >= 1.25 x B1 aggregate throughput;
      * B2 latency P95 <= 2 x B1 latency P95;
      * WER <= 26/787 + 0.005.

    When quality_by_phase is supplied, B1 and B2 WER are checked separately;
    this prevents a B1-only quality alias from masking B2 degradation. Status:
    FAIL if any check FAILs; UNPROVEN if any check is not evaluable; PASS only
    when every check is a proven PASS.
    """
    ratio: Optional[float] = None
    b1_agg = b1_metrics.get("throughput_audio_s_per_s")
    b2_agg = b2_metrics.get("aggregate_throughput_audio_s_per_s")
    if b1_agg and b2_agg is not None and b1_agg > 0:
        ratio = b2_agg / b1_agg
    b1_p95 = _nested(b1_metrics, ("request_wall_s", "p95"))
    b2_p95 = _nested(b2_metrics, ("request_wall_s", "p95"))
    p95_ratio: Optional[float] = None
    if b1_p95 and b2_p95 is not None and b1_p95 > 0:
        p95_ratio = b2_p95 / b1_p95
    checks = {
        "b2_over_b1_aggregate_throughput": _threshold_check(
            ratio, B2_OVER_B1_AGGREGATE_MIN, ">="
        ),
        "b2_p95_latency_vs_2x_b1_p95": _threshold_check(
            p95_ratio, B2_P95_VS_B1_P95_MAX, "<="
        ),
        "wer": _threshold_check(quality.get("wer"), WER_MAX, "<="),
    }
    if quality_by_phase is not None:
        b1_quality = quality_by_phase.get("b1") or {}
        b2_quality = quality_by_phase.get("b2") or {}
        checks["b1_wer"] = _threshold_check(b1_quality.get("wer"), WER_MAX, "<=")
        checks["b2_wer"] = _threshold_check(b2_quality.get("wer"), WER_MAX, "<=")
    statuses = [c["status"] for c in checks.values()]
    overall = (
        "FAIL" if "FAIL" in statuses
        else "UNPROVEN" if "UNPROVEN" in statuses
        else "PASS"
    )
    return {
        "status": overall,
        "checks": checks,
        "thresholds": {
            "b2_over_b1_aggregate_min": B2_OVER_B1_AGGREGATE_MIN,
            "b2_p95_vs_b1_p95_max": B2_P95_VS_B1_P95_MAX,
            "wer_max": WER_MAX,
        },
        "basis": "app-vs-app only; native measurements are never compared here",
    }


def qualification(
    b1: list[UtteranceResult],
    b2: list[UtteranceResult],
    corpus: Corpus,
    *,
    identity_present: bool,
    artifact_identity_present: Optional[bool] = None,
    width_ok: Optional[bool] = None,
    width_unproven: bool = False,
    readiness_ok: Optional[bool] = None,
    runtime_ifb_ok: Optional[bool] = None,
    live_artifact_verified: Optional[bool] = None,
    resources_proven: bool = False,
    controls_complete: bool = False,
    aggregate_valid: Optional[bool] = None,
    thresholds_status: Optional[str] = None,
    async_lifetime_pending: bool = False,
) -> dict[str, Any]:
    """Qualification status: QUALIFIED / NOTQUALIFIED / UNPROVEN.

    A failed row or an invalid aggregate keeps a run NOTQUALIFIED. Missing
    service or artifact identity, unevaluable width evidence or unevaluable
    thresholds keep it UNPROVEN. It never becomes a fabricated PASS.
    """
    reasons: list[str] = []
    status = "QUALIFIED"

    def downgrade(to: str) -> None:
        nonlocal status
        order = {"QUALIFIED": 0, "UNPROVEN": 1, "NOTQUALIFIED": 2}
        if order[to] > order[status]:
            status = to

    if not identity_present:
        reasons.append("service identity absent (capabilities model_id/backend)")
        downgrade("UNPROVEN")
    if artifact_identity_present is False:
        reasons.append(
            "no pinned artifact identity evidence (--identity-file); "
            "capabilities metadata alone is not live artifact proof"
        )
        downgrade("UNPROVEN")
    if len(corpus.items) != FROZEN_ITEM_COUNT:
        reasons.append(
            f"corpus item count {len(corpus.items)} != {FROZEN_ITEM_COUNT}"
        )
        downgrade("UNPROVEN")
    failed = [r for r in b1 + b2 if not r.ok]
    if failed:
        # A failed row keeps the run NOTQUALIFIED even when another dimension
        # is merely UNPROVEN: the failure is positive evidence of a defect and
        # must not be masked by a weaker status.
        reasons.append(f"{len(failed)} failed rows retained")
        downgrade("NOTQUALIFIED")
    # Cleanup states are consulted EXPLICITLY, never only via row.ok.
    dirty = [r for r in b1 + b2 if getattr(r, "cleanup_pending", False) or getattr(r, "cleanup_error", None)]
    if dirty:
        reasons.append(
            f"{len(dirty)} rows with pending/failed owned cleanup retained "
            "(never qualified aggregates)"
        )
        downgrade("NOTQUALIFIED")
    # Top-run owned async lifetime (F4): ANY still-pending owned task or
    # unclosed owned resource at the post-cleanup snapshot forces
    # NOTQUALIFIED even when every nominal row is ok.
    if async_lifetime_pending:
        reasons.append(
            "owned async lifetime registry has pending tasks/unclosed "
            "resources after cleanup (never qualified)"
        )
        downgrade("NOTQUALIFIED")
    total_ref = sum(
        len(normalize_text(it.transcript).split()) for it in corpus.items
    )
    if total_ref != FROZEN_REF_WORDS:
        reasons.append(f"reference word count {total_ref} != {FROZEN_REF_WORDS}")
        downgrade("UNPROVEN")
    if width_ok is False:
        reasons.append(
            "realized width / admission ceiling does not match requested "
            "concurrency (observed mismatch or failed row)"
        )
        downgrade("NOTQUALIFIED")
    if width_unproven:
        reasons.append(
            "admission-ceiling evidence missing or malformed (missing/malformed "
            "proof is UNPROVEN, not positive failure evidence)"
        )
        downgrade("UNPROVEN")
    if readiness_ok is False:
        reasons.append(
            "actual observed readiness failure: pre or post /readyz was not "
            "HTTP 200 with a ready body"
        )
        downgrade("NOTQUALIFIED")
    elif readiness_ok is None:
        reasons.append(
            "readiness liveness UNPROVEN: no actual pre+post /readyz 200 "
            "ready-body observation"
        )
        downgrade("UNPROVEN")
    if runtime_ifb_ok is False:
        reasons.append(
            "live runtime IFB slot contract observed mismatch (contract not "
            "verified, slot mismatch or slots below requested width)"
        )
        downgrade("NOTQUALIFIED")
    elif runtime_ifb_ok is None:
        reasons.append(
            "live native-worker IFB readiness UNPROVEN: no actual "
            "runtime_ifb native_worker_ready/contract_verified provenance"
        )
        downgrade("UNPROVEN")
    if live_artifact_verified is not True:
        # Runtime IFB slot-contract evidence is NOT SHA-identity proof.
        reasons.append(
            "live artifact identity UNPROVEN: no actual root-supplied "
            "matching physical preflight proof (runtime IFB proves the "
            "slot contract only; identity-file hash alone is insufficient)"
        )
        downgrade("UNPROVEN")
    if not resources_proven:
        reasons.append(
            "resource evidence UNPROVEN (no observed RSS/RAM/thermal data); "
            "full qualification cannot be QUALIFIED while unproven"
        )
        downgrade("UNPROVEN")
    if not controls_complete:
        reasons.append(
            "control gates not fully implemented/observed; full "
            "qualification cannot be QUALIFIED while unproven"
        )
        downgrade("UNPROVEN")
    if aggregate_valid is False:
        reasons.append(
            "B2 whole-phase aggregate invalid (failed/missing rows or pair "
            "spans); truncated-success means are not substituted"
        )
        downgrade("NOTQUALIFIED")
    if thresholds_status == "FAIL":
        reasons.append("paired app thresholds FAIL")
        downgrade("NOTQUALIFIED")
    elif thresholds_status == "UNPROVEN":
        reasons.append("paired app thresholds not evaluable (UNPROVEN, not PASS)")
        downgrade("UNPROVEN")
    first_partials = [
        r for r in b1 if r.ok and r.first_partial_latency_s is not None
    ]
    if not first_partials and b1:
        reasons.append("first-partial latency UNPROVEN (no nonempty partial)")
    return {
        "status": status,
        "reasons": reasons,
        "first_partial_unproven": not first_partials and bool(b1),
        "full_qualification_note": (
            "Measured metric/threshold results are reported separately in "
            "metrics and app_thresholds; full QUALIFIED requires actual "
            "live artifact proof, proven resources and complete controls, "
            "none of which this driver can fabricate."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Offline comparator
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ComparatorIdentity:
    asr_model_id: Optional[str]
    asr_backend: Optional[str]
    mode: str
    corpus_manifest_sha256: str


def _identity_of(run: dict[str, Any]) -> ComparatorIdentity:
    identity = run.get("service_identity") or {}
    return ComparatorIdentity(
        asr_model_id=identity.get("asr_model_id"),
        asr_backend=identity.get("asr_backend"),
        mode=str(run.get("mode") or ""),
        corpus_manifest_sha256=str(run.get("corpus", {}).get("manifest_sha256") or ""),
    )


def _artifact_identity_of(run: dict[str, Any]) -> dict[str, Any]:
    block = run.get("artifact_identity") or {}
    identity = block.get("identity") or {}
    return identity if isinstance(identity, dict) else {}


def _shared_identity_problems(
    art_a: dict[str, Any],
    art_b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> list[str]:
    """Enforce the shared identity contract between two comparison sides.

    Every IDENTITY_STRICT_SHARED_FIELDS entry must be a nonempty string on
    BOTH sides and equal. A field missing on BOTH sides is a mismatch too —
    ``None == None`` never matches, and absent identity can never PASS.
    """
    problems: list[str] = []
    for field_name in IDENTITY_STRICT_SHARED_FIELDS:
        val_a = art_a.get(field_name)
        val_b = art_b.get(field_name)
        ok_a = isinstance(val_a, str) and bool(val_a)
        ok_b = isinstance(val_b, str) and bool(val_b)
        if not ok_a or not ok_b:
            problems.append(
                f"shared identity field {field_name} missing/unresolved "
                f"({label_a}={val_a!r}, {label_b}={val_b!r}); comparison UNPROVEN"
            )
        elif val_a != val_b:
            problems.append(
                f"shared identity mismatch on {field_name}: "
                f"{label_a}={val_a} {label_b}={val_b}"
            )
    return problems


def _variant_identity_provenance(
    art_a: dict[str, Any],
    art_b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> tuple[list[str], dict[str, Any]]:
    """Check and record the explicit permitted variant differences.

    Each IDENTITY_VARIANT_FIELDS entry must be present and nonempty on each
    side (absence is UNPROVEN, not a silent default); their VALUES are
    allowed to differ and are recorded per side, never hidden under a common
    label. Returns (problems, provenance-doc).
    """
    problems: list[str] = []
    provenance: dict[str, Any] = {}
    for field_name in IDENTITY_VARIANT_FIELDS:
        val_a = art_a.get(field_name)
        val_b = art_b.get(field_name)
        provenance[field_name] = {label_a: val_a, label_b: val_b}
        ok_a = isinstance(val_a, str) and bool(val_a)
        ok_b = isinstance(val_b, str) and bool(val_b)
        if not ok_a or not ok_b:
            problems.append(
                f"variant provenance {field_name} missing/unresolved on a "
                f"side ({label_a}={val_a!r}, {label_b}={val_b!r}); UNPROVEN"
            )
    return problems, provenance


def _phase_provenance_problems(
    doc: dict[str, Any],
    phase: str,
    expected_width: int,
    label: str,
) -> list[str]:
    """Require ACTUAL executed-phase provenance in a run document.

    The document must prove it really executed the requested phase at the
    requested concurrency (requested concurrency, phases_executed and the
    phase's own measured realized/requested width must all agree). Missing
    or contradictory provenance — including swapped phases or a synthetic
    one-lane B1 standing in for B2 — is rejected, never defaulted.
    """
    problems: list[str] = []
    metrics = doc.get("metrics") or {}
    phase_metrics = metrics.get(phase)
    requested_conc = doc.get("concurrency")
    executed = doc.get("phases_executed")
    # Typed concurrency: a genuine int equal to the requested width (bool is
    # rejected BEFORE the equality test — True == 1 is not width evidence).
    if not _strict_int(requested_conc) or requested_conc != expected_width:
        problems.append(
            f"{label}: requested concurrency {requested_conc!r} does not "
            f"prove phase {phase} (expected {expected_width})"
        )
    if not isinstance(executed, list) or executed != [phase]:
        problems.append(
            f"{label}: phases_executed {executed!r} does not prove actual "
            f"phase {phase}"
        )
    if not isinstance(phase_metrics, dict):
        problems.append(f"{label}: no measured {phase} phase metrics")
    else:
        realized = phase_metrics.get("realized_width")
        req = phase_metrics.get("requested_width")
        # Typed widths: genuine ints only (bools rejected before equality,
        # so realized_width=True can never stand in for width 1).
        if (
            not _strict_int(realized) or not _strict_int(req)
            or realized != expected_width or req != expected_width
        ):
            problems.append(
                f"{label}: {phase} phase width provenance "
                f"(realized={realized!r}, requested={req!r}) does not prove "
                f"concurrency {expected_width}"
            )
    return problems


def _protocol_identity_problems(
    doc_a: dict[str, Any],
    doc_b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> list[str]:
    """Shared protocol-identity validation for BOTH comparison paths.

    Enforced identically by ``compare_runs`` and
    ``paired_threshold_document``:

      * the ACTUAL capabilities service identity (``asr_model_id`` and
        ``asr_backend``) must be nonempty strings on BOTH sides and equal
        (missing on both sides is UNPROVEN, never ``None == None``);
      * ``mode`` must be an actual supported driver mode (``ws``/``http``)
        on both sides and equal (a synthesized ``chunked`` or empty mode
        is not a run this driver can produce);
      * the corpus manifest sha256 must be nonempty on both sides and
        equal (an empty/absent manifest on both sides is UNPROVEN);
      * the ordered corpus hash lists must be nonempty, every hash a
        nonempty string, and the orders identical (``[] == []`` never
        proves a shared corpus; the fixed frozen denominator is enforced
        separately in the paired path and in ``qualification`` and is NOT
        altered here);
      * ``config.pace`` must be an ACTUAL bool on both sides (the driver
        parser exposes pacing only as a boolean flag) and equal; and
      * ``config.chunk_ms`` must be a positive genuine int (bools are
        rejected via ``_strict_int``) on both sides and equal.

    Any missing/invalid field on BOTH sides is an explicit problem, never
    a defaulted valid value.
    """
    problems: list[str] = []

    # Service identity from ACTUAL capabilities metadata.
    sid_a = doc_a.get("service_identity") or {}
    sid_b = doc_b.get("service_identity") or {}
    for field_name in PROTOCOL_IDENTITY_SERVICE_FIELDS:
        val_a = sid_a.get(field_name)
        val_b = sid_b.get(field_name)
        ok_a = isinstance(val_a, str) and bool(val_a)
        ok_b = isinstance(val_b, str) and bool(val_b)
        if not ok_a or not ok_b:
            problems.append(
                f"service {field_name} missing/unresolved "
                f"({label_a}={val_a!r}, {label_b}={val_b!r}); "
                f"comparison UNPROVEN"
            )
        elif val_a != val_b:
            problems.append(
                f"service {field_name} mismatch: "
                f"{label_a}={val_a!r} {label_b}={val_b!r}"
            )

    # Mode: actual supported driver modes only, same on both sides.
    mode_a = doc_a.get("mode")
    mode_b = doc_b.get("mode")
    for label, mode in ((label_a, mode_a), (label_b, mode_b)):
        if mode not in SUPPORTED_SERVICE_MODES:
            problems.append(
                f"{label}: mode {mode!r} is not an actual supported driver "
                f"mode {SUPPORTED_SERVICE_MODES}; comparison UNPROVEN"
            )
    if mode_a != mode_b:
        problems.append(f"mode mismatch: {label_a}={mode_a!r} {label_b}={mode_b!r}")

    # Corpus manifest sha256: present, nonempty and equal.
    corpus_a = doc_a.get("corpus") or {}
    corpus_b = doc_b.get("corpus") or {}
    sha_a = corpus_a.get("manifest_sha256")
    sha_b = corpus_b.get("manifest_sha256")
    if not (isinstance(sha_a, str) and sha_a) or not (
        isinstance(sha_b, str) and sha_b
    ):
        problems.append(
            f"corpus manifest sha256 missing/unresolved "
            f"({label_a}={sha_a!r}, {label_b}={sha_b!r}); comparison UNPROVEN"
        )
    elif sha_a != sha_b:
        problems.append(
            f"corpus manifest sha256 mismatch: "
            f"{label_a}={sha_a} {label_b}={sha_b}"
        )

    # Ordered corpus hashes: nonempty lists, every hash a nonempty string,
    # same order. An empty corpus on BOTH sides is not a match.
    def _hash_list(corpus: dict[str, Any]) -> list[Any]:
        return [
            item.get("sha256") if isinstance(item, dict) else None
            for item in (corpus.get("items") or [])
        ]

    hashes_a = _hash_list(corpus_a)
    hashes_b = _hash_list(corpus_b)
    all_hashes = hashes_a + hashes_b
    if (
        not hashes_a
        or not hashes_b
        or not all(isinstance(h, str) and h for h in all_hashes)
    ):
        problems.append(
            f"ordered corpus hashes missing/empty "
            f"({label_a} count={len(hashes_a)}, {label_b} count={len(hashes_b)}); "
            f"comparison UNPROVEN"
        )
    elif hashes_a != hashes_b:
        problems.append("ordered corpus hashes differ")

    # OPT-IN dual pinned detail contract: when EITHER side reports detail
    # metadata, BOTH must report the frozen detail sha256, a True
    # source_verified, matching detail pins and the frozen ordered
    # fingerprint. Missing on one side or differing detail pins is a fail-
    # closed UNPROVEN pairing (legacy synthetic direct docs are unchanged
    # when both are absent).
    detail_a = corpus_a.get("detail_sha256")
    detail_b = corpus_b.get("detail_sha256")

    # Dual-contract ACTIVATION: fail closed when ANY detail provenance is
    # populated on EITHER side — a nonempty detail sha256, detail path,
    # ordered fingerprint, a supplied detail sha, or source_verified True.
    # Both absent / default (None, None, None, False) legacy docs remain
    # valid unchanged.
    def _detail_activated(corpus: dict[str, Any]) -> bool:
        return bool(
            corpus.get("detail_sha256")
            or corpus.get("detail_path")
            or corpus.get("detail_ordered_fingerprint")
            or corpus.get("source_verified") is True
        )

    if _detail_activated(corpus_a) or _detail_activated(corpus_b):
        # When the dual contract is activated, BOTH documents must also
        # carry the frozen ORIGINAL source sha256 on their corpus identity.
        if sha_a != FROZEN_MANIFEST_SHA256 or sha_b != FROZEN_MANIFEST_SHA256:
            problems.append(
                f"activated dual corpus contract requires the frozen "
                f"original source manifest pin on BOTH sides "
                f"({label_a}={sha_a!r}, {label_b}={sha_b!r}); "
                f"comparison UNPROVEN"
            )
        if detail_a != FROZEN_CORPUS_DETAIL_SHA256 or detail_b != FROZEN_CORPUS_DETAIL_SHA256:
            problems.append(
                f"corpus detail sha256 not the frozen pin on BOTH sides "
                f"({label_a}={detail_a!r}, {label_b}={detail_b!r}); "
                f"comparison UNPROVEN"
            )
        verified_a = corpus_a.get("source_verified")
        verified_b = corpus_b.get("source_verified")
        if verified_a is not True or verified_b is not True:
            problems.append(
                f"corpus source_verified not True on BOTH sides "
                f"({label_a}={verified_a!r}, {label_b}={verified_b!r}); "
                f"comparison UNPROVEN"
            )
        fp_a = corpus_a.get("detail_ordered_fingerprint")
        fp_b = corpus_b.get("detail_ordered_fingerprint")
        if (
            fp_a != FROZEN_CORPUS_DETAIL_ORDERED_SHA_FINGERPRINT
            or fp_b != FROZEN_CORPUS_DETAIL_ORDERED_SHA_FINGERPRINT
        ):
            problems.append(
                f"corpus detail ordered fingerprint not the frozen pin on "
                f"BOTH sides ({label_a}={fp_a!r}, {label_b}={fp_b!r}); "
                f"comparison UNPROVEN"
            )
        path_a = corpus_a.get("detail_path")
        path_b = corpus_b.get("detail_path")
        if (
            corpus_a.get("detail_sha256") != corpus_b.get("detail_sha256")
            or not (isinstance(path_a, str) and path_a)
            or not (isinstance(path_b, str) and path_b)
        ):
            problems.append(
                f"corpus detail path/pins missing or different between "
                f"{label_a} and {label_b}; comparison UNPROVEN"
            )

    # Pacing: the actual parser exposes ONLY a boolean pace flag.
    pace_a = (doc_a.get("config") or {}).get("pace")
    pace_b = (doc_b.get("config") or {}).get("pace")
    if not isinstance(pace_a, bool) or not isinstance(pace_b, bool):
        problems.append(
            f"config.pace missing/invalid (actual parser supports only the "
            f"boolean pace flag) ({label_a}={pace_a!r}, {label_b}={pace_b!r}); "
            f"comparison UNPROVEN"
        )
    elif pace_a != pace_b:
        problems.append(f"pacing mismatch: {label_a}={pace_a} {label_b}={pace_b}")

    # Chunk: a positive GENUINE int on both sides (bool is not an int).
    chunk_a = (doc_a.get("config") or {}).get("chunk_ms")
    chunk_b = (doc_b.get("config") or {}).get("chunk_ms")
    if (
        not _strict_int(chunk_a) or chunk_a <= 0
        or not _strict_int(chunk_b) or chunk_b <= 0
    ):
        problems.append(
            f"config.chunk_ms missing/invalid (a positive non-bool int is "
            f"required) ({label_a}={chunk_a!r}, {label_b}={chunk_b!r}); "
            f"comparison UNPROVEN"
        )
    elif chunk_a != chunk_b:
        problems.append(f"chunk_ms mismatch: {label_a}={chunk_a} {label_b}={chunk_b}")

    return problems


def compare_runs(app_run: dict[str, Any], baseline_run: dict[str, Any]) -> dict[str, Any]:
    """Compare an app run to a baseline run, producing ratios and guards.

    Verifies the same 100 ordered hashes, service identity, the shared
    identity contract (device/SDK/upstream/worker/plugin/profile family/base
    profile), mode, chunk and pacing. Resolved profile/engine/config/slot
    variant differences are permitted and recorded per side. A native
    (non-app) baseline is rejected: app metrics are never compared to native
    measurements. A guard that cannot be evaluated is UNPROVEN, never PASS.
    """
    problems: list[str] = []

    if str(baseline_run.get("measurement_basis") or "app") != "app":
        problems.append(
            "baseline measurement_basis is not 'app': native baselines are "
            "rejected for app comparison"
        )

    # Shared protocol identity: service model/backend (actual, nonempty,
    # equal), supported mode, manifest sha, ordered nonempty hashes, boolean
    # pace and positive non-bool chunk_ms — missing on BOTH sides is never a
    # defaulted match (see _protocol_identity_problems).
    problems.extend(_protocol_identity_problems(app_run, baseline_run, "app", "baseline"))

    # Shared identity contract: device/SDK/upstream/worker/plugin/profile
    # family/base profile must match; resolved profile/engine/config/slot
    # variants are the explicit permitted differences, recorded per side.
    # Missing shared fields on both sides never match (None==None rejected);
    # missing variant provenance keeps the comparison UNPROVEN.
    app_art = _artifact_identity_of(app_run)
    base_art = _artifact_identity_of(baseline_run)
    problems.extend(_shared_identity_problems(app_art, base_art, "app", "baseline"))
    variant_problems, variant_provenance = _variant_identity_provenance(
        app_art, base_art, "app", "baseline"
    )
    problems.extend(variant_problems)
    permitted_differences = {
        name: {"app": app_art.get(name), "baseline": base_art.get(name)}
        for name in IDENTITY_VARIANT_FIELDS
        if app_art.get(name) != base_art.get(name)
    }

    app_hashes = [it.get("sha256") for it in (app_run.get("corpus", {}).get("items") or [])]
    base_hashes = [
        it.get("sha256") for it in (baseline_run.get("corpus", {}).get("items") or [])
    ]
    if app_hashes != base_hashes:
        problems.append("ordered corpus hashes differ")

    pacing_app = (app_run.get("config") or {}).get("pace")
    pacing_base = (baseline_run.get("config") or {}).get("pace")
    if pacing_app != pacing_base:
        problems.append(f"pacing mismatch: app={pacing_app} baseline={pacing_base}")
    chunk_app = (app_run.get("config") or {}).get("chunk_ms")
    chunk_base = (baseline_run.get("config") or {}).get("chunk_ms")
    if chunk_app != chunk_base:
        problems.append(f"chunk_ms mismatch: app={chunk_app} baseline={chunk_base}")

    def ratio(app_value, base_value):
        if app_value is None or base_value in (None, 0):
            return None
        return app_value / base_value

    app_metrics = app_run.get("metrics") or {}
    base_metrics = baseline_run.get("metrics") or {}
    ratios: dict[str, Any] = {}
    for metric, path in (
        ("b1_p50_request_wall", ("b1", "request_wall_s", "p50")),
        ("b1_p95_request_wall", ("b1", "request_wall_s", "p95")),
        ("b1_p95_rtf", ("b1", "rtf", "p95")),  # RTF: lower is better
        ("b2_p95_request_wall", ("b2", "request_wall_s", "p95")),
        ("b2_aggregate_throughput", ("b2", "aggregate_throughput_audio_s_per_s")),
    ):
        app_val = _nested(app_metrics, path)
        base_val = _nested(base_metrics, path)
        ratios[metric] = ratio(app_val, base_val)

    quality_guard = {
        "app_wer": (app_metrics.get("quality") or {}).get("wer"),
        "baseline_wer": (base_metrics.get("quality") or {}).get("wer"),
        "app_ref_words": (app_metrics.get("quality") or {}).get("ref_words"),
        "baseline_ref_words": (base_metrics.get("quality") or {}).get("ref_words"),
    }

    # Guards: latency / RTF <= 1.15x baseline (lower is better); throughput is
    # reported as a ratio only (the paired >=1.25x B1 guard is app-internal).
    guards = {
        "b1_p95_request_wall_vs_baseline": _threshold_check(
            ratios["b1_p95_request_wall"], APP_VS_BASELINE_MAX_RATIO, "<="
        ),
        "b1_p95_rtf_vs_baseline": _threshold_check(
            ratios["b1_p95_rtf"], APP_VS_BASELINE_MAX_RATIO, "<="
        ),
        "app_wer": _threshold_check(quality_guard["app_wer"], WER_MAX, "<="),
    }
    guard_statuses = [g["status"] for g in guards.values()]
    guard_overall = (
        "FAIL" if "FAIL" in guard_statuses
        else "UNPROVEN" if "UNPROVEN" in guard_statuses
        else "PASS"
    )
    if problems:
        guard_overall = "FAIL"

    return {
        "problems": problems,
        "identical_protocol": not problems,
        "permitted_differences": permitted_differences,
        "variant_provenance": variant_provenance,
        "ratios": ratios,
        "guards": guards,
        "guard_status": guard_overall,
        "quality_guard": quality_guard,
        "raw_basis": {
            "app": {
                "b1_request_wall": _nested(app_metrics, ("b1", "request_wall_s")),
                "b2_aggregate_throughput": _nested(
                    app_metrics, ("b2", "aggregate_throughput_audio_s_per_s")
                ),
            },
            "baseline": {
                "b1_request_wall": _nested(base_metrics, ("b1", "request_wall_s")),
                "b2_aggregate_throughput": _nested(
                    base_metrics, ("b2", "aggregate_throughput_audio_s_per_s")
                ),
            },
        },
    }


def paired_threshold_document(
    b1_doc: dict[str, Any], b2_doc: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the frozen paired B1/B2 thresholds from two single-phase runs.

    The two runs must be two ACTUAL phases of the SAME app deployment: the
    shared identity contract must hold (device/SDK/upstream/worker/plugin/
    profile family/base profile equal), the resolved profile/engine/config/
    slot variants may differ and are recorded per side, each phase document
    must prove it really executed B1 at concurrency 1 / B2 at concurrency 2
    (swapped or synthetic one-lane phases are rejected), and corpus/mode/
    chunk/pace must match. Any missing provenance, including absent artifact
    identity, keeps the pairing UNPROVEN — never a silent default PASS.
    """
    problems: list[str] = []
    # Shared protocol identity (service model/backend, supported mode,
    # manifest sha, ordered nonempty hashes, boolean pace, positive chunk).
    problems.extend(_protocol_identity_problems(b1_doc, b2_doc, "b1", "b2"))
    # Paired documents must be ACTUAL runs of the FROZEN corpus: a wrong or
    # empty manifest/denominator is not pairable even when both sides agree.
    # The fixed 100-item/hash denominator itself is NOT altered.
    for label, doc in (("b1", b1_doc), ("b2", b2_doc)):
        corpus = doc.get("corpus") or {}
        if corpus.get("manifest_sha256") != FROZEN_MANIFEST_SHA256:
            problems.append(
                f"{label}: corpus manifest is not the frozen English100 "
                f"manifest; pairing UNPROVEN"
            )
        items = corpus.get("items") or []
        if len(items) != FROZEN_ITEM_COUNT:
            problems.append(
                f"{label}: corpus item count {len(items)} != {FROZEN_ITEM_COUNT}"
            )
    art_b1 = _artifact_identity_of(b1_doc)
    art_b2 = _artifact_identity_of(b2_doc)
    problems.extend(_shared_identity_problems(art_b1, art_b2, "b1", "b2"))
    variant_problems, variant_provenance = _variant_identity_provenance(
        art_b1, art_b2, "b1", "b2"
    )
    problems.extend(variant_problems)
    problems.extend(_phase_provenance_problems(b1_doc, "b1", 1, "b1 phase"))
    problems.extend(_phase_provenance_problems(b2_doc, "b2", 2, "b2 phase"))
    # config pace/chunk equality (typed and presence-checked) is enforced in
    # _protocol_identity_problems above.

    if problems:
        return {
            "status": "UNPROVEN",
            "problems": problems,
            "checks": None,
            "reason": "phase runs are not the same app deployment; pairing rejected",
        }
    thresholds = evaluate_app_thresholds(
        (b1_doc.get("metrics") or {}).get("b1") or {},
        (b2_doc.get("metrics") or {}).get("b2") or {},
        (b2_doc.get("metrics") or {}).get("quality") or {},
        quality_by_phase={
            "b1": (b1_doc.get("metrics") or {}).get("quality") or {},
            "b2": (b2_doc.get("metrics") or {}).get("quality") or {},
        },
    )
    thresholds["problems"] = []
    thresholds["variant_provenance"] = variant_provenance
    thresholds["phase_provenance"] = {
        "b1": {
            "requested_concurrency": b1_doc.get("concurrency"),
            "phases_executed": b1_doc.get("phases_executed"),
            "realized_width": ((b1_doc.get("metrics") or {}).get("b1") or {}).get(
                "realized_width"
            ),
        },
        "b2": {
            "requested_concurrency": b2_doc.get("concurrency"),
            "phases_executed": b2_doc.get("phases_executed"),
            "realized_width": ((b2_doc.get("metrics") or {}).get("b2") or {}).get(
                "realized_width"
            ),
        },
    }
    thresholds["physical_proof_note"] = (
        "A paired PASS is a metric comparison only; live_artifact_verified "
        "stays UNPROVEN until actual target provenance integration. JSON "
        "self-attestation never becomes physical verification."
    )
    return thresholds


# ──────────────────────────────────────────────────────────────────────
# Sealed offline evidence requalification
# ──────────────────────────────────────────────────────────────────────


def _sealed_regular(path: Path, expected_sha: str | None = None,
                    expected_size: int | None = None) -> dict[str, Any]:
    """Read immutable evidence metadata without following symlinks."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence path is not a regular file: {path}")
    st = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"evidence SHA mismatch: {path}")
    if expected_size is not None and st.st_size != expected_size:
        raise ValueError(f"evidence size mismatch: {path}")
    return {"path": str(path), "sha256": digest, "size": st.st_size}


def _load_canonical_artifact_validator():
    root = Path(__file__).resolve().parents[2] / "scripts" / "edgellm_asr_validation.py"
    spec = importlib.util.spec_from_file_location("slv_canonical_asr_validation", root)
    if spec is None or spec.loader is None:
        raise ValueError(f"canonical validator unavailable: {root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, root


def _resource_evidence_report(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Validate the sealed resource envelope with the canonical validator."""
    try:
        env = json.loads(path.read_text(encoding="utf-8"))
        validator, validator_path = _load_canonical_artifact_validator()
        required = ("closure_sha256", "phase", "run_id", "boot_id")
        if any(env.get(k) != expected.get(k) for k in required):
            raise ValueError("resource envelope run binding mismatch")
        sampler = env.get("sampler") or {}
        expected_target = expected.get("target")
        expected_worker_binding = expected.get("worker_binding")
        expected_sampler_source_sha256 = expected.get("sampler_source_sha256")
        if (expected_target is None or expected_worker_binding is None
                or expected_sampler_source_sha256 is None):
            raise ValueError("sealed phase resource context is incomplete")
        if expected_sampler_source_sha256 != sampler.get("source_sha256"):
            raise ValueError("resource sampler source SHA is not sealed")
        csv_meta = env.get("csv") or {}
        csv_path = Path(str(csv_meta.get("path", "")))
        checked = validator.validate_asr_resource_evidence(
            env,
            expected_closure_sha256=expected["closure_sha256"],
            expected_phase=expected["phase"],
            expected_run_id=expected["run_id"],
            expected_boot_id=expected["boot_id"],
            expected_target=expected_target,
            expected_worker_binding=expected_worker_binding,
            expected_sampler_source_sha256=expected_sampler_source_sha256,
            csv_path=csv_path,
        )
        if checked.get("status") != "OBSERVED":
            return {"status": "UNPROVEN", "reason": checked.get("reason", "canonical resource validator rejected evidence"),
                    "cuda_status": "UNPROVEN", "canonical_validator_source": str(validator_path)}
        return {"status": "PROVEN", "cuda_status": "UNPROVEN", "csv": {
                    "path": str(csv_path), "sha256": csv_meta.get("sha256"), "size": csv_meta.get("size")},
                "coverage": checked.get("coverage"), "worker": env.get("worker"),
                "sampler": env.get("sampler"), "workload": {
                    "start_mono": env.get("workload_start_mono"),
                    "end_mono": env.get("workload_end_mono")},
                "canonical_validator_source": str(validator_path),
                "note": "CUDA allocation/memory counter was not supplied"}
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as exc:
        return {"status": "UNPROVEN", "reason": str(exc), "cuda_status": "UNPROVEN"}

def offline_evidence_report(args: argparse.Namespace) -> int:
    """Requalify sealed evidence without contacting the ASR service."""
    out = Path(args.evidence_report)
    if out.exists() or out.is_symlink():
        raise SystemExit(f"evidence report already exists: {out}")
    try:
        phase_index = Path(args.phase_evidence)
        index_meta = _sealed_regular(phase_index, args.phase_evidence_sha256)
        run_meta = _sealed_regular(Path(args.evidence_run), args.evidence_run_sha256)
        proof_meta = _sealed_regular(Path(args.live_artifact_proof), args.live_artifact_proof_sha256)
        index = json.loads(phase_index.read_text(encoding="utf-8"))
        if index.get("schema") != 1:
            raise ValueError("phase evidence schema mismatch")
        raw_links = index.get("links") or index.get("refs") or {}
        links = dict(raw_links)
        for alias, canonical in (("phase_state", "state"), ("phase_result", "result"),
                                 ("run_json", "run")):
            if alias not in links and canonical in links:
                links[alias] = links[canonical]
        if "run_json" not in links and "run" in links:
            links["run_json"] = links["run"]
        required_links = ("phase_state", "phase_result", "run_json", "artifact_proof")
        linked: dict[str, Any] = {}
        for key in required_links:
            item = links.get(key) or {}
            linked[key] = _sealed_regular(Path(item["path"]), item.get("sha256"), item.get("size"))
        if linked["run_json"] != run_meta:
            raise ValueError("sealed phase index does not bind --evidence-run")
        if linked["artifact_proof"] != proof_meta:
            raise ValueError("sealed phase index does not bind --live-artifact-proof")
        run_doc = json.loads(Path(links["run_json"]["path"]).read_text(encoding="utf-8"))
        phase = index.get("phase")
        if not isinstance(phase, str) or run_doc.get("label") != phase:
            raise ValueError("phase/run label binding mismatch")
        state = json.loads(Path(links["phase_state"]["path"]).read_text(encoding="utf-8"))
        result = json.loads(Path(links["phase_result"]["path"]).read_text(encoding="utf-8"))
        if (state.get("phase") != phase or result.get("phase") != phase
                or result.get("closure_sha256") != index.get("closure_sha256")):
            raise ValueError("phase state/result binding mismatch")
        state_meta = index.get("state") or {}
        sealed_argv = index.get("command_argv", index.get("state_command_argv", state_meta.get("command_argv")))
        if sealed_argv is None or sealed_argv != state.get("command_argv"):
            raise ValueError("sealed command argv does not match phase state")
        sealed_pins = index.get("driver_pins", state_meta.get("driver_pins"))
        if sealed_pins is None or sealed_pins != state.get("driver_pins"):
            raise ValueError("sealed driver pins do not match phase state")
        if (index.get("run_json_sha256") is not None
                and index.get("run_json_sha256") != linked["run_json"]["sha256"]):
            raise ValueError("phase index run JSON SHA mismatch")
        expected_mode = "http" if ".http." in phase else "ws"
        if run_doc.get("mode") != expected_mode:
            raise ValueError("run mode does not match sealed phase")
        expected_concurrency = 2 if phase.endswith(".b2") else 1
        if run_doc.get("concurrency") != expected_concurrency:
            raise ValueError("run concurrency does not match sealed phase")
        admission = run_doc.get("input_admission") or {}
        if admission.get("manifest_sha256") != run_doc.get("corpus", {}).get("manifest_sha256"):
            raise ValueError("run manifest admission binding mismatch")
        if "corpus_detail_sha256" in admission and admission.get("corpus_detail_sha256") != run_doc.get("corpus", {}).get("detail_sha256"):
            raise ValueError("run corpus detail admission binding mismatch")
        last_observation = state.get("last_observation") or {}
        durable_observation = last_observation.get("durable") or {}
        terminal = durable_observation.get("terminal")
        if not isinstance(terminal, dict):
            terminal = last_observation.get("terminal") or {}
        terminal_identity = terminal.get("identity") if isinstance(terminal, dict) else {}
        terminal_identity = terminal_identity if isinstance(terminal_identity, dict) else {}
        runtime_identity = index.get("runtime_identity") or {}
        resource_context = state.get("resource_collection") or index.get("resource_collection") or {}
        worker_binding = (resource_context.get("binding") or resource_context.get("worker_binding")
                           or state.get("worker_binding") or index.get("worker_binding"))
        sampler_context = resource_context.get("sampler") or {}
        collector_context = resource_context.get("collector") or {}
        sampler_source_sha256 = (resource_context.get("sampler_source_sha256")
                                 or sampler_context.get("source_sha256")
                                 or state.get("sampler_source_sha256")
                                 or index.get("sampler_source_sha256")
                                 or collector_context.get("sampler_source_sha256"))
        if not runtime_identity and isinstance(worker_binding, dict):
            runtime_identity = {"boot_id": worker_binding.get("boot_id")}
        expected = {"closure_sha256": index.get("closure_sha256"), "phase": phase,
                    "run_id": index.get("run_id"), "target": index.get("target"),
                    "boot_id": runtime_identity.get("boot_id", terminal_identity.get("boot_id")),
                    "worker_binding": worker_binding,
                    "sampler_source_sha256": sampler_source_sha256}
        artifact_expected = index.get("artifact_expected_records")
        if not isinstance(artifact_expected, list) or not artifact_expected:
            raise ValueError("sealed artifact expected records missing")
        validator, validator_path = _load_canonical_artifact_validator()
        proof = json.loads(Path(links["artifact_proof"]["path"]).read_text(encoding="utf-8"))
        artifact = validator.validate_artifact_observation(
            Path(links["artifact_proof"]["path"]), artifact_expected,
            Path(index.get("proof_root", Path(links["artifact_proof"]["path"]).parent)),
        )
        if artifact.get("status") != "PASS":
            raise ValueError(f"artifact proof rejected: {artifact.get('reason', artifact)}")
        resource_meta = _sealed_regular(Path(args.resource_evidence), args.resource_evidence_sha256)
        resource = _resource_evidence_report(Path(args.resource_evidence), expected)
        reasons = []
        if resource.get("status") != "PROVEN":
            reasons.append("resource evidence UNPROVEN: " + resource.get("reason", "invalid envelope"))
        reasons.extend(["controls_complete remains false", "paired baseline/control evidence not supplied"])
        report = {"schema": 1, "status": "UNPROVEN", "phase": phase,
                  "closure_sha256": index.get("closure_sha256"), "run_id": index.get("run_id"),
                  "qualification": {"status": "UNPROVEN", "reasons": reasons},
                  "evidence_run": linked["run_json"], "phase_evidence": index_meta,
                  "linked_evidence": linked, "artifact_proof": artifact,
                  "artifact_validator_source": {"path": str(validator_path), "sha256": hashlib.sha256(validator_path.read_bytes()).hexdigest()},
                  "resource_evidence": resource, "resource_envelope": resource_meta,
                  "raw_run_summary": {"rows": (run_doc.get("metrics") or {}).get("b1", {}).get("rows_total"),
                                      "metrics": run_doc.get("metrics"), "qualification": run_doc.get("qualification")}}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        report = {"schema": 1, "status": "UNPROVEN", "classification": "offline-evidence",
                  "reason": str(exc), "qualification": {"status": "UNPROVEN", "reasons": [str(exc)]}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", errors="strict")
    print(json.dumps({"status": report["status"], "evidence_report": str(out)}, sort_keys=True))
    return 0


def _nested(obj: Any, path: tuple[str, ...]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


# ──────────────────────────────────────────────────────────────────────
# Baseline recompute helper loading (optional, SHA-pinned)
# ──────────────────────────────────────────────────────────────────────


def load_baseline_recompute(path: Path, expected_sha256: str):
    """Load and SHA-validate an optional pinned baseline recompute helper.

    The helper must expose ``compute(items, results)`` and return a mapping
    with a ``wer`` key whose semantics match the frozen baseline recompute.
    """
    import importlib.util

    if not path.is_file():
        raise ManifestError(f"baseline recompute helper not found: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ManifestError(
            f"baseline recompute helper sha256 mismatch: got {actual}, "
            f"want {expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location("_baseline_recompute", path)
    if spec is None or spec.loader is None:
        raise ManifestError("baseline recompute helper could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "compute", None)):
        raise ManifestError("baseline recompute helper has no callable 'compute'")
    return module


# ──────────────────────────────────────────────────────────────────────
# Run entrypoint
# ──────────────────────────────────────────────────────────────────────


def _source_hashes() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    targets = {
        "bench/perf/edgellm_asr_ws_perf_gate.py": Path(__file__).resolve(),
        "server/main.py": repo / "server" / "main.py",
        "server/core/api_capabilities.py": repo / "server" / "core" / "api_capabilities.py",
    }
    out: dict[str, Any] = {}
    for name, path in targets.items():
        out[name] = sha256_file(path) if path.is_file() else None
    return out


async def run_gate(
    config: RunConfig,
    registry: Optional[AsyncLifetimeRegistry] = None,
    *,
    deadline_mono: Optional[float] = None,
) -> RunOutcome:
    """Execute the gate for the REQUESTED phase only.

    ``--concurrency 1`` runs ONLY B1; ``--concurrency 2`` runs ONLY B2. There
    is no combined two-phase run and no unlabelled extra repeated utterance;
    the optional post-warm repeat is opt-in, separate, and explicitly NOT a
    cold-start measurement. Probes share the overall deadline budget.

    When ``deadline_mono`` is provided (the CLI-owned loop creates ONE
    absolute deadline BEFORE the loop/top task and shares it here) that
    deadline is authoritative and is NEVER renewed per layer; the optional
    argument preserves the old ``run_gate(config)`` call shape used by the
    owned-input tests and other callers.
    """
    if config.output.exists():
        raise SystemExit(f"output dir already exists (refusing to overwrite): {config.output}")

    # ONE absolute overall deadline starts BEFORE any manifest/input work;
    # input admission is charged against it and its duration is reported
    # separately (source proof: the old driver ran load_corpus BEFORE the
    # deadline stamp, i.e. unbounded input work escaped the budget). When the
    # CLI already owns the deadline it is passed in and NOT recomputed.
    started = time.monotonic()
    wall = time.strftime("%FT%TZ", time.gmtime())
    if deadline_mono is None:
        deadline = started + config.overall_deadline_s
    else:
        deadline = deadline_mono

    # ONE run-level owned async lifetime ledger (F4): every owned connector,
    # receiver, request, poll, close and pair task plus every owned
    # connection/writer resource is registered here (per-utterance helpers
    # may add their own entries to THIS registry when it is passed down).
    # The run scope is a CHILD of a provided CLI/root registry: its run-level
    # drain below therefore acts ONLY on run-owned tasks and can never
    # cancel the CLI top task, its supervisor or an unrelated sibling,
    # while every registration is still forwarded (same shared entry) to
    # the root aggregate for the final CLI drain/snapshot.
    if registry is None:
        registry = AsyncLifetimeRegistry(f"run:{config.label}")
    else:
        registry = registry.child(f"run:{config.label}")

    admission = await admit_corpus_inputs(config, deadline)
    # Independent guard: the report STATUS alone is not enough — actual
    # cleanup proof (helper actually exited, every owned wrapper close
    # PROVEN, clean close ledger) is required before ANY probe/inference.
    cleanup_proven = bool(
        (admission.get("pending") or {}).get("cleanup_complete") is True
    )
    if admission["status"] != "admitted" or not cleanup_proven:
        # Any failed input, pending helper, unproven cleanup or expired input
        # budget blocks ALL probes and inference; the report retains the
        # fixed frozen denominator and every failure reason.
        raise InputAdmissionError(admission)
    # Read the corpus and admitted map through ONE synchronized snapshot so
    # the event-loop thread never touches the writer's live dicts unprotected
    # and never observes a half-published corpus/admitted view.
    _admission_snap = admission["loader"].snapshot()
    corpus: Corpus = _admission_snap["corpus"]
    admitted: dict[int, AdmittedInput] = _admission_snap["admitted"]

    pre = await probe_service_identity(config.base_url, deadline_mono=deadline,
                                       registry=registry)

    if config.mode == "ws":
        conn_factory = _make_ws_factory(
            config.base_url, config.ws_path, language=config.language or "en", sample_rate=EXPECTED_RATE
        )
    else:
        conn_factory = None  # type: ignore[assignment]

    b1: list[UtteranceResult] = []
    b2: list[UtteranceResult] = []
    b1_phase: Optional[PhaseInfo] = None
    b2_phase: Optional[PhaseInfo] = None
    try:
        if config.concurrency == 1:
            if config.single_long_diagnostic:
                phase_started = time.monotonic()
                row = await _run_one(
                    conn_factory, corpus, corpus.items[0], config, config.mode,
                    chunk_bytes_for_ms(config.chunk_ms),
                    pace_seconds_for_ms(config.chunk_ms) if config.pace else 0.0,
                    deadline, warm=False, pair_index=None, admitted=admitted,
                    registry=registry,
                )
                b1 = [row]
                b1_phase = PhaseInfo(
                    name="single-long-diagnostic",
                    elapsed_s=time.monotonic() - phase_started,
                    realized_width=1,
                    requested_width=1,
                )
            else:
                b1, b1_phase = await run_b1(conn_factory, corpus, mode=config.mode,
                                            config=config, deadline_mono=deadline,
                                            admitted=admitted, registry=registry)
        elif config.concurrency == 2:
            b2, b2_phase = await run_b2(conn_factory, corpus, mode=config.mode,
                                        config=config, deadline_mono=deadline,
                                        admitted=admitted, registry=registry)
        else:
            raise SystemExit(f"unsupported concurrency {config.concurrency}")

        repeat: Optional[UtteranceResult] = None
        if config.post_warm_repeat and time.monotonic() < deadline:
            # Opt-in SEPARATE post-warm single request. It is never mixed into
            # the 100 metrics and never qualifies as a cold-start measurement.
            repeat = await _run_one(
                conn_factory, corpus, corpus.items[0], config, config.mode,
                chunk_bytes_for_ms(config.chunk_ms),
                pace_seconds_for_ms(config.chunk_ms) if config.pace else 0.0,
                deadline, warm=False, pair_index=None, admitted=admitted,
                registry=registry,
            )
    finally:
        # Cooperative best-effort cancel of the (already finished) helper
        # bookkeeping; never a cleanup claim and never a forceful stop.
        admission["loader"].cancel_event.set()
    post = await probe_service_identity(config.base_url, deadline_mono=deadline,
                                        registry=registry)

    # Run-level drain (F4): any still-owned task is cancelled once and
    # finitely observed inside the SAME overall deadline (reserved budget);
    # whatever survives stays registered as pending and forces the run
    # NOTQUALIFIED via the regenerated snapshot in outcome_to_dict.
    await registry.drain(max(0.0, _remaining(deadline)))

    return RunOutcome(
        config=config,
        corpus=corpus,
        pre=pre,
        post=post,
        b1=b1,
        b2=b2,
        b1_phase=b1_phase,
        b2_phase=b2_phase,
        post_warm_repeat=repeat,
        source_hashes=_source_hashes(),
        started_wall=wall,
        finished_wall=time.strftime("%FT%TZ", time.gmtime()),
        input_admission=admission,
        lifetime_registry=registry,
    )


def outcome_to_dict(outcome: RunOutcome) -> dict[str, Any]:
    identity_present = bool(
        (outcome.pre.get("identity_present") or outcome.post.get("identity_present"))
    )
    metrics = compute_metrics(
        outcome.b1, outcome.b2, outcome.corpus, identity_present=identity_present,
        b1_phase=outcome.b1_phase, b2_phase=outcome.b2_phase,
    )
    artifact = outcome.config.identity
    artifact_present = artifact is not None
    evidence = width_evidence(outcome.pre, outcome.post)
    phase = outcome.b1_phase or outcome.b2_phase
    requested = outcome.config.concurrency
    if phase is None:
        width_ok: Optional[bool] = None
        width_unproven = False
    else:
        ceiling = evidence["max_strict_ceiling"]
        if ceiling is None:
            # Missing/malformed preflight ceiling proof: UNPROVEN, NOT a
            # positive failure observation.
            width_ok = None
            width_unproven = True
        else:
            width_ok = phase.realized_width == requested and ceiling >= requested
            width_unproven = False
    readiness = _readiness_evidence(outcome.pre, outcome.post)
    readiness_ok: Optional[bool] = {
        "QUALIFIED": True,
        "UNPROVEN": None,
        "NOTQUALIFIED": False,
    }[readiness["status"]]
    ifb_evidence = _runtime_ifb_evidence(outcome.pre, outcome.post, requested)
    runtime_ifb_ok: Optional[bool] = {
        "QUALIFIED": True,
        "UNPROVEN": None,
        "NOTQUALIFIED": False,
    }[ifb_evidence["status"]]
    aggregate_valid = metrics["b2"]["aggregate_valid"] if outcome.b2 else None

    # Regenerate the owned async lifetime snapshot AFTER cleanup (F4): any
    # pending task or unclosed owned resource here forces NOTQUALIFIED even
    # when every nominal row is ok. No transcripts/secrets are included.
    lifetime_snap = (
        outcome.lifetime_registry.snapshot()
        if outcome.lifetime_registry is not None
        else {
            "label": None,
            "task_count": 0,
            "pending_count": 0,
            "pending_phases": [],
            "tasks": [],
            "resources": [],
            "open_resource_count": 0,
        }
    )
    async_lifetime_pending = bool(
        lifetime_snap["pending_count"] or lifetime_snap["open_resource_count"]
    )

    both_phases = bool(outcome.b1) and bool(outcome.b2)
    if both_phases:
        thresholds = evaluate_app_thresholds(
            metrics["b1"], metrics["b2"], metrics["quality"],
            quality_by_phase=metrics["quality_by_phase"],
        )
    elif outcome.b2:
        # A B2-only run cannot prove paired throughput/latency, but its own
        # quality must still be thresholded instead of disappearing as None.
        b2_wer = _threshold_check(
            metrics["quality_by_phase"]["b2"].get("wer"), WER_MAX, "<="
        )
        thresholds = {
            "status": "UNPROVEN",
            "checks": {"wer": b2_wer, "b2_wer": b2_wer},
            "quality_status": b2_wer["status"],
            "reason": (
                f"single-phase B2 run (--concurrency {requested}); quality WER "
                "is evaluated, but pair it with B1 for throughput/latency "
                "thresholds via --pair-b1/--pair-b2"
            ),
        }
    else:
        thresholds = {
            "status": "UNPROVEN",
            "checks": None,
            "reason": (
                f"single-phase run (--concurrency {requested}); pair this run "
                "with the matching-phase run of the same deployment via "
                "--pair-b1/--pair-b2 to evaluate the paired thresholds"
            ),
        }
    qual = qualification(
        outcome.b1, outcome.b2, outcome.corpus,
        identity_present=identity_present,
        artifact_identity_present=artifact_present,
        width_ok=width_ok,
        width_unproven=width_unproven,
        readiness_ok=readiness_ok,
        runtime_ifb_ok=runtime_ifb_ok,
        live_artifact_verified=False,  # UNPROVEN: no physical preflight input
        resources_proven=False,        # resource_evidence stays UNPROVEN
        controls_complete=False,       # control gates remain partial/unimplemented
        aggregate_valid=aggregate_valid,
        thresholds_status=thresholds["status"],
        async_lifetime_pending=async_lifetime_pending,
    )
    diagnostic_contract = None
    if outcome.config.single_long_diagnostic:
        # This branch is intentionally explicit: one long input is useful for
        # transport/lifecycle observation, but it cannot satisfy the frozen
        # 100-item WER denominator or any baseline/native-batch gate.
        diagnostic_contract = {
            "mode": "single-long-diagnostic",
            "status": "UNPROVEN",
            "quality_status": "UNPROVEN",
            "wer_status": "UNPROVEN",
            "native_batch_status": "UNPROVEN",
            "formal_quality_gate": False,
            "reason": (
                "single continuous 90-180s empty-transcript input; transport "
                "and owned lifecycle evidence only"
            ),
        }
        qual["status"] = "UNPROVEN"
        qual["reasons"].append(
            "single-long diagnostic is outside the frozen 100-item quality denominator"
        )
    repeat_doc = None
    if outcome.post_warm_repeat is not None:
        repeat_doc = asdict(outcome.post_warm_repeat)
        repeat_doc["qualifies_as_cold_startup"] = False
        repeat_doc["note"] = (
            "Post-warm repeated request, measured separately AFTER the phase. "
            "It is NOT a cold-start measurement and is never mixed into the "
            "100-utterance metrics."
        )
    phases_executed = []
    if outcome.b1:
        phases_executed.append("b1")
    if outcome.b2:
        phases_executed.append("b2")
    return {
        "schema_version": SCHEMA_VERSION,
        "label": outcome.config.label,
        "base_url": outcome.config.base_url,
        "mode": outcome.config.mode,
        "measurement_basis": "app",
        "concurrency": outcome.config.concurrency,
        "phases_executed": phases_executed,
        "started_wall": outcome.started_wall,
        "finished_wall": outcome.finished_wall,
        "config": {
            "chunk_ms": outcome.config.chunk_ms,
            "pace": outcome.config.pace,
            "request_deadline_s": outcome.config.request_deadline_s,
            "overall_deadline_s": outcome.config.overall_deadline_s,
            "post_final_window_s": outcome.config.post_final_window_s,
            "concurrency": outcome.config.concurrency,
            "language_requested": outcome.config.language,
            "language_effective": outcome.config.language or "en" if outcome.config.mode == "ws" else outcome.config.language,
        },
        "service_identity": {
            "pre": outcome.pre,
            "post": outcome.post,
            "asr_model_id": outcome.pre.get("asr_model_id") or outcome.post.get("asr_model_id"),
            "asr_backend": outcome.pre.get("asr_backend") or outcome.post.get("asr_backend"),
            "admission_limit": outcome.pre.get("admission_limit")
            or outcome.post.get("admission_limit"),
            "identity_present": identity_present,
            "identity_note": (
                "model_id/backend are provenance labels only; they are NOT "
                "live artifact identity proof."
            ),
            "readiness": readiness,
            "runtime_ifb": ifb_evidence,
        },
        "artifact_identity": (
            {
                "status": "PINNED",
                "path": artifact["path"],
                "sha256": artifact["sha256"],
                "identity": artifact["identity"],
                "provenance": "root-supplied pinned preflight identity file",
                # Runtime IFB proves the live SLOT CONTRACT ONLY — it is NOT
                # proof that the actual worker/plugin/engine/config/profile
                # SHAs match the pinned identity. No invented runtime SHA and
                # no self-attested boolean can bypass that; actual root-
                # supplied matching physical preflight proof does not exist
                # as an input to this driver, so verification stays UNPROVEN.
                "live_artifact_verified": False,
                "live_artifact_status": "UNPROVEN",
                "live_artifact_note": (
                    "UNPROVEN: runtime_ifb contract evidence proves slots, "
                    "not SHA identity; actual root-supplied matching "
                    "physical preflight proof is required and absent. The "
                    "SHA identity file hash alone is insufficient."
                ),
                # Distinct, separately reported slot-contract evidence.
                "runtime_ifb_contract_verified": runtime_ifb_ok is True,
            }
            if artifact
            else {
                "status": "UNPROVEN",
                "reason": (
                    "no --identity-file supplied; capabilities metadata alone "
                    "is not live artifact proof"
                ),
                "live_artifact_verified": False,
                "live_artifact_status": "UNPROVEN",
                "runtime_ifb_contract_verified": runtime_ifb_ok is True,
            }
        ),
        "width_evidence": {
            **evidence,
            "requested_width": requested,
            "realized_width": phase.realized_width if phase else None,
            "width_ok": width_ok,
            "width_status": (
                "NOT_EVALUATED" if phase is None
                else "UNPROVEN" if width_unproven
                else "QUALIFIED" if width_ok
                else "NOTQUALIFIED"
            ),
        },
        "source_hashes": outcome.source_hashes,
        "input_admission": {
            k: v for k, v in (outcome.input_admission or {}).items() if k != "loader"
        },
        "corpus": {
            "manifest_path": outcome.corpus.manifest_path,
            "manifest_sha256": outcome.corpus.manifest_sha256,
            "corpus_dir_declared": outcome.corpus.corpus_dir_declared,
            "item_count": len(outcome.corpus.items),
            "items": [asdict(it) for it in outcome.corpus.items],
            "detail_path": outcome.corpus.detail_path,
            "detail_sha256": outcome.corpus.detail_sha256,
            "detail_ordered_fingerprint": (
                outcome.corpus.detail_ordered_fingerprint
            ),
            "source_verified": outcome.corpus.source_verified,
        },
        "b1": [asdict(r) for r in outcome.b1],
        "b2": [asdict(r) for r in outcome.b2],
        "post_warm_repeat": repeat_doc,
        "metrics": metrics,
        "async_lifetime": lifetime_snap,
        "app_thresholds": thresholds,
        "qualification": qual,
        "diagnostic": diagnostic_contract,
        "resource_evidence": {
            "status": "UNPROVEN",
            "note": (
                "This driver observes only the network protocol. No process RSS, "
                "device RAM or thermal reading is fabricated here."
            ),
        },
        "control_gates": control_gate_status(),
    }


def control_gate_status() -> dict[str, Any]:
    """Explicit NOT_IMPLEMENTED / UNPROVEN status for the control gates.

    The matched-performance task scope does not implement these; they remain
    requirements for final ASR app qualification and are listed so no fake
    PASS can be inferred. No cancel ACK is fabricated where the API lacks one.
    """
    return {
        "20_cancel_recovery": {
            "status": "NOT_IMPLEMENTED",
            "reason": (
                "The /asr/stream API has no cancel command and no cancel-ACK "
                "frame; the native worker cancel_ack_p95 in the root acceptance "
                "JSON is not app-layer evidence. No fake cancel ACK is emitted."
            ),
        },
        "100_reopen": {
            "status": "NOT_IMPLEMENTED",
            "reason": "Requires a live service lifecycle loop outside this matched-perf scope.",
        },
        "reset_full_replay_ack": {
            "status": "IMPLEMENTED_PARTIAL",
            "reason": (
                "The reset branch exists (server/main.py: cmd=='reset' -> "
                "{'type':'reset','text':'','is_final':True,'is_stable':True,"
                "'reset':True}); the driver distinguishes the reset ACK from an "
                "utterance final, but full-replay+ACK is not measured as a real "
                "gate in this matched-perf scope."
            ),
        },
        "disconnect_recovery": {
            "status": "NOT_IMPLEMENTED",
            "reason": "Requires real session-release observation on a live service.",
        },
        "occupied_caps_window_rejects": {
            "status": "NOT_IMPLEMENTED",
            "reason": "Requires a live oversubscription window measurement.",
        },
    }


def write_outputs(outcome: RunOutcome) -> dict[str, Any]:
    document = outcome_to_dict(outcome)
    out_dir: Path = outcome.config.output
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "run.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Raw per-row JSONL retained separately so a failed row cannot be dropped
    # by a later reformat.
    with (out_dir / "b1-raw.jsonl").open("w", encoding="utf-8") as handle:
        for res in outcome.b1:
            handle.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
    with (out_dir / "b2-raw.jsonl").open("w", encoding="utf-8") as handle:
        for res in outcome.b2:
            handle.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
    (out_dir / "runtime-identity.json").write_text(
        json.dumps(
            {
                "service_identity": document["service_identity"],
                "artifact_identity": document["artifact_identity"],
                "source_hashes": document["source_hashes"],
                "started_wall": document["started_wall"],
                "finished_wall": document["finished_wall"],
                "label": document["label"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=False,
                        help="e.g. http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("ws", "http"), default="ws")
    parser.add_argument("--language", default=None,
                        help="optional protocol language; omitted preserves legacy WS=en and HTTP omission")
    parser.add_argument("--concurrency", type=int, choices=(1, 2), required=False,
                        help="1 = phase B1 only (sequential 100); "
                             "2 = phase B2 only (50 adjacent pairs)")
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--manifest-sha256", default=FROZEN_MANIFEST_SHA256)
    parser.add_argument("--corpus-detail", type=Path, default=None,
                        help="opt-in frozen corpus DETAIL wrapper JSON "
                             "(items list); requires --corpus-detail-sha256")
    parser.add_argument("--corpus-detail-sha256", default=None,
                        help="expected SHA256 of --corpus-detail (required "
                             "with it; must be the frozen detail pin)")
    parser.add_argument("--corpus-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=False,
                        help="output directory; must NOT exist")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--pace", action="store_true",
                        help="real-time pacing shared by B1 and B2")
    parser.add_argument("--request-deadline-s", type=float, default=120.0)
    parser.add_argument("--overall-deadline-s", type=float, default=3600.0)
    parser.add_argument("--post-final-window-s", type=float, default=1.0)
    parser.add_argument("--label", required=False)
    parser.add_argument("--baseline", type=Path, default=None,
                        help="baseline run.json to compare against (app basis only)")
    parser.add_argument("--baseline-recompute", type=Path, default=None)
    parser.add_argument("--baseline-recompute-sha256", default=None)
    parser.add_argument("--post-warm-repeat", action="store_true",
                        help="opt-in SEPARATE post-warm single request; never "
                             "a cold-start measurement")
    parser.add_argument("--identity-file", type=Path, default=None,
                        help="pinned service identity JSON (artifact evidence)")
    parser.add_argument("--identity-file-sha256", default=None,
                        help="expected SHA256 of --identity-file (required with it)")
    parser.add_argument("--pair-b1", type=Path, default=None,
                        help="offline: B1 run.json for paired thresholds")
    parser.add_argument("--pair-b2", type=Path, default=None,
                        help="offline: B2 run.json for paired thresholds")
    parser.add_argument("--phase-evidence", type=Path, default=None,
                        help="offline sealed phase state/result/run/proof index")
    parser.add_argument("--phase-evidence-sha256", default=None)
    parser.add_argument("--evidence-run", type=Path, default=None,
                        help="offline immutable run.json to requalify")
    parser.add_argument("--evidence-run-sha256", default=None)
    parser.add_argument("--resource-evidence", type=Path, default=None,
                        help="offline sealed resource envelope")
    parser.add_argument("--resource-evidence-sha256", default=None)
    parser.add_argument("--live-artifact-proof", type=Path, default=None,
                        help="offline sealed six-role artifact proof")
    parser.add_argument("--live-artifact-proof-sha256", default=None)
    parser.add_argument("--evidence-report", type=Path, default=None,
                        help="fresh offline evidence report output")
    parser.add_argument(
        "--single-long-diagnostic", action="store_true",
        help=("diagnostic-only one-item 90-180s PCM16 mono 16kHz WS run; "
              "never a formal quality or baseline run"),
    )
    parser.add_argument("--controls-only", action="store_true",
                        help="run opt-in WS controls; performance remains NOT_RUN")
    return parser


class _CliLifetimeTimeout(Exception):
    """The finite top-level CLI wait expired; run_gate cancellation was
    requested exactly once (completion never awaited unbounded)."""


def _run_gate_owned_loop(
    config: RunConfig,
) -> tuple[Optional[RunOutcome], dict[str, Any], AsyncLifetimeRegistry]:
    """Own the CLI event loop explicitly (F4) — no ``asyncio.run``.

    ``asyncio.run``'s Runner performs an unconditional cancel-and-gather of
    all pending tasks plus a default-executor shutdown join that can wait
    on cancellation-suppressing children; neither happens here. Instead:
      1. ONE absolute overall deadline is taken BEFORE the loop/top task
         exists and is shared with run_gate (optional argument), so no layer
         renews the budget;
      2. a fresh loop is created and owned by this function;
      3. run_gate runs as ONE actual top-level task in the run registry;
      4. a small cleanup slice is RESERVED INSIDE the one deadline, so the
         finite top-level ``asyncio.wait`` bound is the remaining budget
         MINUS that slice and the final registry drain fits before expiry;
      5. on expiry/external interruption the top task is cancelled EXACTLY
         ONCE and the registry is drained finitely within the SAME deadline;
      6. the ledger snapshot is REGENERATED after the drain and the loop is
         closed LAST; loop.close() is never claimed as cleanup proof and
         unresolved objects stay referenced by the returned registry until
         process end. No process signals are used anywhere here.

    A non-positive or non-finite overall deadline is an explicit NF/error
    return WITHOUT creating a loop task or starting any work (no 1 ms floor).
    """
    registry = AsyncLifetimeRegistry(f"cli:{config.label}")
    overall = config.overall_deadline_s
    evidence: dict[str, Any] = {
        "loop_owned": True,
        "loop_closed": False,
        "top_state": "not_started",
        "overall_deadline_s": overall,
        "cleanup_reserve_s": CLI_CLEANUP_RESERVE_S,
        "drain_grace_s": CLI_DRAIN_GRACE_S,
        "note": (
            "loop.close() is disposal only, never cleanup proof; pending "
            "objects stay referenced by the registry until process end"
        ),
    }
    # Explicit rejection of an unusable overall budget: no task is created,
    # no probe/loader is started, and the caller reports NF with a reason.
    if not isinstance(overall, (int, float)) or isinstance(overall, bool):
        evidence["top_state"] = "invalid_deadline"
        evidence["error"] = f"overall deadline is not numeric: {overall!r}"
        return None, evidence, registry
    if not math.isfinite(float(overall)):
        evidence["top_state"] = "invalid_deadline"
        evidence["error"] = f"overall deadline is not finite: {overall!r}"
        return None, evidence, registry
    if float(overall) <= 0.0:
        evidence["top_state"] = "invalid_deadline"
        evidence["error"] = f"overall deadline is not positive: {overall!r}"
        return None, evidence, registry

    # ONE absolute deadline taken here, BEFORE the loop/top creation, and
    # shared with run_gate so nothing renews it.
    cli_started = time.monotonic()
    cli_deadline = cli_started + float(overall)
    evidence["cli_deadline_mono"] = cli_deadline
    # Cleanup reserved INSIDE the one overall budget (never added on top).
    # The reserve is capped by the drain grace AND by half the budget so a
    # small overall deadline still leaves the gate real time to run.
    reserve = min(CLI_CLEANUP_RESERVE_S, 0.5 * float(overall))
    top_wait_end = cli_deadline - reserve
    evidence["top_wait_bound_s"] = max(0.0, top_wait_end - cli_started)

    loop = asyncio.new_event_loop()
    supervisor: Optional[asyncio.Task] = None
    try:
        top = loop.create_task(
            run_gate(config, registry=registry, deadline_mono=cli_deadline),
            name="run_gate",
        )
        registry.track_task(top, "run_gate top-level")

        async def supervise() -> RunOutcome:
            # Top-level wait is FINITE and bounded by the reserved slice of
            # the one absolute deadline; expiry creates NO new work.
            bound = max(0.0, top_wait_end - time.monotonic())
            try:
                done, _pending = await asyncio.wait({top}, timeout=bound)
            except asyncio.CancelledError:
                if not top.done():
                    top.cancel()
                registry.note_cancel_requested(top, "run_gate top-level")
                raise
            if top in done:
                return top.result()  # surfaces stored exceptions (e.g. input)
            top.cancel()  # exactly once
            registry.note_cancel_requested(top, "run_gate top-level")
            raise _CliLifetimeTimeout(
                "finite top-level CLI wait expired; run_gate cancellation "
                "requested once (completion not awaited unbounded)"
            )

        # The supervise wrapper is an ACTUAL task held by the registry, so an
        # external interruption cannot leave it pending/untracked with an
        # unretrieved exception at loop.close().
        supervisor = registry.track_task(
            loop.create_task(supervise(), name="cli_supervise"),
            "cli supervise",
        )

        outcome: Optional[RunOutcome] = None
        try:
            outcome = loop.run_until_complete(supervisor)
            evidence["top_state"] = "done"
        except _CliLifetimeTimeout as exc:
            evidence["top_state"] = "timeout_cancel_requested"
            evidence["error"] = str(exc)
        except asyncio.CancelledError:
            # The sole cancel request came from external interruption; the
            # top task is cancelled once and reported truthfully.
            evidence["top_state"] = "interrupted_cancel_requested"
            evidence["error"] = "external cancellation during the owned loop"
            if not top.done():
                top.cancel()
                registry.note_cancel_requested(top, "run_gate top-level")
        except KeyboardInterrupt:
            # External interruption: cancel the supervise wrapper and the top
            # task EXACTLY ONCE, drain finitely, and report truthfully. The
            # KeyboardInterrupt is NOT re-raised so the NF async-lifetime
            # evidence can still be written by the caller. Ownership of the
            # supervise wrapper is completed here (cancel request recorded),
            # so no untracked pending task survives.
            evidence["top_state"] = "interrupted_cancel_requested"
            evidence["error"] = "KeyboardInterrupt during the owned loop"
            if supervisor is not None and not supervisor.done():
                supervisor.cancel()
                registry.note_cancel_requested(supervisor, "cli supervise")
            if not top.done():
                top.cancel()
                registry.note_cancel_requested(top, "run_gate top-level")
        # Finite registry drain (cancellation observation) bounded by the
        # SAME absolute deadline and capped by the drain grace; after expiry
        # the bound is 0 and NO new wait/task/probe is started.
        drain_bound = min(
            CLI_DRAIN_GRACE_S, max(0.0, cli_deadline - time.monotonic())
        )
        evidence["drain_bound_s"] = drain_bound
        try:
            loop.run_until_complete(registry.drain(drain_bound))
            evidence["drain"] = "completed"
        except asyncio.CancelledError:
            evidence["drain"] = "cancelled: external interruption"
        except Exception as exc:  # noqa: BLE001
            evidence["drain"] = f"failed: {type(exc).__name__}: {exc}"
        evidence["async_lifetime"] = registry.snapshot()
        if evidence["async_lifetime"]["pending_count"]:
            evidence["pending_unresolved"] = True
        return outcome, evidence, registry
    finally:
        loop.close()  # LAST, after the snapshot; not a cleanup claim
        evidence["loop_closed"] = True


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    offline_flags = (args.evidence_run, args.phase_evidence,
                     args.resource_evidence, args.live_artifact_proof,
                     args.evidence_report)
    if any(x is not None for x in offline_flags):
        if not all(x is not None for x in offline_flags):
            raise SystemExit(
                "offline evidence mode requires --evidence-run, --phase-evidence, "
                "--resource-evidence, --live-artifact-proof and --evidence-report"
            )
        for name in ("evidence_run_sha256", "phase_evidence_sha256",
                     "resource_evidence_sha256", "live_artifact_proof_sha256"):
            if not isinstance(getattr(args, name), str) or len(getattr(args, name)) != 64:
                raise SystemExit(f"offline evidence mode requires {name}")
        report_args = argparse.Namespace(**vars(args))
        # The evidence index is authoritative for the run/proof links. The
        # explicit pins remain required so callers cannot silently omit them.
        return offline_evidence_report(report_args)

    required_live = {
        "--base-url": args.base_url, "--concurrency": args.concurrency,
        "--manifest": args.manifest, "--output": args.output,
        "--label": args.label,
    }
    missing_live = [name for name, value in required_live.items() if value is None]
    if missing_live:
        raise SystemExit("missing required live arguments: " + ", ".join(missing_live))

    # Earliest argument-stage validation, BEFORE any mode branch, file read
    # or network path: the dual-pinned corpus detail fields are strictly
    # paired (both or neither).
    if (args.corpus_detail is None) != (args.corpus_detail_sha256 is None):
        raise SystemExit(
            "--corpus-detail and --corpus-detail-sha256 are required together "
            "(both or neither)"
        )

    if args.single_long_diagnostic:
        if args.mode != "ws":
            raise SystemExit("--single-long-diagnostic requires --mode ws")
        if args.concurrency != 1:
            raise SystemExit("--single-long-diagnostic requires --concurrency 1")
        if args.chunk_ms != 100 or not args.pace:
            raise SystemExit(
                "--single-long-diagnostic requires --chunk-ms 100 and --pace"
            )
        if args.request_deadline_s != 180.0 or args.overall_deadline_s != 240.0:
            raise SystemExit(
                "--single-long-diagnostic requires request=180s and overall=240s"
            )
        if args.corpus_detail is not None:
            raise SystemExit("--single-long-diagnostic forbids --corpus-detail")
        if args.baseline is not None or args.baseline_recompute is not None:
            raise SystemExit("--single-long-diagnostic forbids baseline comparison")
        if args.post_warm_repeat:
            raise SystemExit("--single-long-diagnostic forbids --post-warm-repeat")
        if args.manifest_sha256 == FROZEN_MANIFEST_SHA256:
            raise SystemExit(
                "--single-long-diagnostic requires a non-frozen manifest SHA"
            )

    # Offline paired-threshold mode: no service contact at all.
    if args.pair_b1 is not None or args.pair_b2 is not None:
        if args.pair_b1 is None or args.pair_b2 is None:
            raise SystemExit("--pair-b1 and --pair-b2 must be given together")
        b1_doc = json.loads(args.pair_b1.read_text(encoding="utf-8"))
        b2_doc = json.loads(args.pair_b2.read_text(encoding="utf-8"))
        document = paired_threshold_document(b1_doc, b2_doc)
        out_dir: Path = args.output
        if out_dir.exists():
            raise SystemExit(
                f"output dir already exists (refusing to overwrite): {out_dir}"
            )
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "paired-thresholds.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": document["status"]}))
        return 0

    if (args.identity_file is None) != (args.identity_file_sha256 is None):
        raise SystemExit("--identity-file and --identity-file-sha256 must be given together")
    if args.language is not None and not args.language.strip():
        raise SystemExit("--language must be a nonblank string when provided")
    identity = None
    if args.identity_file is not None:
        try:
            identity = load_identity_file(args.identity_file, args.identity_file_sha256)
        except IdentityError as exc:
            raise SystemExit(str(exc))

    config = RunConfig(
        base_url=args.base_url,
        mode=args.mode,
        concurrency=args.concurrency,
        manifest=args.manifest,
        manifest_sha256=args.manifest_sha256,
        corpus_root=args.corpus_root,
        output=args.output,
        chunk_ms=args.chunk_ms,
        pace=bool(args.pace),
        request_deadline_s=args.request_deadline_s,
        overall_deadline_s=args.overall_deadline_s,
        post_final_window_s=args.post_final_window_s,
        label=args.label,
        baseline=args.baseline,
        baseline_recompute=args.baseline_recompute,
        baseline_recompute_sha256=args.baseline_recompute_sha256,
        post_warm_repeat=bool(args.post_warm_repeat),
        identity=identity,
        corpus_detail=args.corpus_detail,
        corpus_detail_sha256=args.corpus_detail_sha256,
        single_long_diagnostic=bool(args.single_long_diagnostic),
        language=args.language,
        controls_only=bool(args.controls_only),
    )
    if config.controls_only:
        if config.mode != "ws":
            raise SystemExit("--controls-only requires --mode ws")
        report = asyncio.run(run_control_only(config))
        config.output.mkdir(parents=True, exist_ok=False)
        payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        (config.output / "run.json").write_text(payload, encoding="utf-8")
        (config.output / "control-run.json").write_text(payload, encoding="utf-8")
        print(json.dumps({"status": report.get("status"), "scope": "CONTROL_ONLY"}))
        return 0 if report.get("status") == "PASS" else EXIT_NOT_RUN
    if config.baseline_recompute is not None:
        if not config.baseline_recompute_sha256:
            raise SystemExit("--baseline-recompute requires --baseline-recompute-sha256")
        load_baseline_recompute(
            config.baseline_recompute, config.baseline_recompute_sha256
        )
    try:
        outcome, lifetime_evidence, _registry_ref = _run_gate_owned_loop(config)
    except InputAdmissionError as exc:
        # Explicit input-failure report: never a silent empty success and
        # never an uncaught crash. No probe/inference was started.
        report = {k: v for k, v in exc.report.items() if k != "loader"}
        out_dir: Path = config.output
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
        except OSError:
            pass  # keep writing the failure report even if the dir exists
        (out_dir / "input-admission-failed.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(
            {"status": "NOTQUALIFIED", "input_admission": report},
            ensure_ascii=False,
        ))
        return 3
    if outcome is None:
        # Incomplete main run (finite top-level wait expired or external
        # interruption): explicit NF async-lifetime evidence and a nonzero
        # exit. Missing rows are NEVER filled with fabricated successes and
        # input ownership reporting is preserved.
        report = {
            "status": "NOTQUALIFIED",
            "reason": (
                "async lifetime incomplete at CLI shutdown; owned tasks/"
                "resources pending are reported truthfully, never declared "
                "closed"
            ),
            "async_lifetime": lifetime_evidence,
        }
        out_dir2: Path = config.output
        try:
            out_dir2.mkdir(parents=True, exist_ok=False)
        except OSError:
            pass  # keep writing the failure report even if the dir exists
        (out_dir2 / "async-lifetime-notqualified.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "NOTQUALIFIED",
                          "async_lifetime": "incomplete"}, sort_keys=True))
        return 4
    document = write_outputs(outcome)
    if config.baseline is not None:
        baseline_doc = json.loads(config.baseline.read_text(encoding="utf-8"))
        comparison = compare_runs(document, baseline_doc)
        (config.output / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(comparison["ratios"], sort_keys=True))
    print(json.dumps({k: document["metrics"][k]["rows_ok"]
                      for k in ("b1", "b2")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
