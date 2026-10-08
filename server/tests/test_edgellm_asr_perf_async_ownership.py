"""AUTHORED-UNRUN F4 async-lifetime ownership tests for
bench/perf/edgellm_asr_ws_perf_gate.py.

Scope (root design /tmp/slv-v011-perf-async-lifetime-root-design.md): the
driver's owned async lifetime — actual Tasks/Futures and connection/writer
resources held in AsyncLifetimeRegistry until OBSERVED done/closed, finite
asyncio.wait waits against the absolute remaining budget, cancel-exactly-once
semantics without awaiting cancellation completion unbounded, B2 expired-
budget floors removed, and the CLI-owned event loop with a finite top-level
wait instead of asyncio.run.

These tests use ONLY in-process fakes and, for the CLI lifecycle, ONE owned
isolated host child process (started and terminated by these tests; TERM is
the only fallback, no KILL escalation, no GPU/device/external network and no
foreign-process interaction). They are AUTHORED and UNRUN by the author; an
independent reviewer and one frozen host run are required before any claim.

Caps disclosure: the suppressing fakes rely on asyncio scheduling order only
(deterministic events, generous CI margins); no fake clocks patch the
driver's time.monotonic basis. Timeout assertions are upper bounds against
the configured deadlines plus margin, not tight equality. The isolated CLI
child imports the REAL driver and monkeypatches ONLY its
``probe_service_identity`` and ``_make_ws_factory`` (synthetic probe + a
cancellation-suppressing connector); production ``main``/``run_gate``/
``_run_gate_owned_loop``/``_bounded``/registry/cleanup are untouched and the
suppressor is never released before shutdown. The child's elapsed upper bound
is the configured overall deadline plus a small interpreter-startup margin,
NOT the old broad 60 s surrogate.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pytest

_DRIVER_PATH = (
    Path(__file__).resolve().parents[2] / "bench" / "perf" / "edgellm_asr_ws_perf_gate.py"
)


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "edgellm_asr_ws_perf_gate_f4", _DRIVER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()

PCM = b"\x00\x01" * 16000  # 1 second of PCM16


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────


class SuppressingWSConn:
    """WS adapter whose recv/close SUPPRESS cancellation (never unwind).

    Simulates the F4 failure mode: a child that ignores cancellation must
    not block the caller; it stays registered pending instead.
    """

    def __init__(self, *, suppress_recv: bool = True, suppress_close: bool = True):
        self._suppress_recv = suppress_recv
        self._suppress_close = suppress_close
        self.closed_calls = 0
        self.sent: list[Any] = []

    async def send(self, data: Any) -> None:
        self.sent.append(data)

    async def recv(self) -> Any:
        while True:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                if not self._suppress_recv:
                    raise
                # swallow cancellation and keep waiting (suppression)
                continue

    async def close(self, code: int = 1000) -> None:
        self.closed_calls += 1
        if not self._suppress_close:
            return
        while True:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue  # suppress


class CleanWSConn:
    """Well-behaved WS adapter emitting one final then a clean close."""

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False
        self.sent: list[Any] = []
        self._script_task: Optional[asyncio.Task] = None

    async def send(self, data: Any) -> None:
        self.sent.append(data)
        if self._script_task is None:
            self._script_task = asyncio.create_task(self._script())

    async def _script(self) -> None:
        await self._inbox.put(json.dumps({"type": "final", "text": "ok", "is_final": True}))

    async def recv(self) -> Any:
        return await self._inbox.get()

    async def close(self, code: int = 1000) -> None:
        self.closed = True


class LateConnector:
    """Connector that suppresses cancellation and completes LATE with a
    live connection after the caller's deadline expired."""

    def __init__(self, conn: Any, delay_s: float = 0.05) -> None:
        self.conn = conn
        self.delay_s = delay_s
        self.calls = 0

    async def __call__(self) -> Any:
        self.calls += 1
        while True:
            try:
                await asyncio.sleep(self.delay_s)
                return self.conn
            except asyncio.CancelledError:
                continue  # suppress; completes late anyway


def _ws_params(**overrides):
    params = dict(
        order=0,
        file="x.wav",
        item_id="x",
        warm=False,
        pair_index=None,
        audio_s=len(PCM) / 2 / 16000,
        transcript="hello world",
        chunk_bytes=3200,
        pace_s=0.0,
        request_deadline_s=1.0,
        post_final_window_s=0.05,
    )
    params.update(overrides)
    return params


def _corpus(n: int = 100) -> "gate.Corpus":
    items = [
        gate.CorpusItem(
            order=i,
            path=f"u{i:03d}.wav",
            file=f"u{i:03d}.wav",
            sha256="00" * 32,
            rate=16000,
            channels=1,
            bit_depth=16,
            duration_s=1.0,
            transcript="hello world",
        )
        for i in range(n)
    ]
    return gate.Corpus(gate.FROZEN_MANIFEST_SHA256, "m", "", items)


def _admitted(corpus) -> dict[int, Any]:
    class _Check:
        ok = True
        error = None
        duration_s = 1.0
        sha256 = "00" * 32

    class _Adm:
        pcm = PCM
        wav_bytes = b"RIFFfake"
        check = _Check()

    return {it.order: _Adm() for it in corpus.items}


# ──────────────────────────────────────────────────────────────────────
# 1. Cancellation-suppressing recv/close cannot block the caller
# ──────────────────────────────────────────────────────────────────────


def test_suppressing_receiver_and_close_never_block_caller():
    conn = SuppressingWSConn(suppress_recv=True, suppress_close=True)
    registry = gate.AsyncLifetimeRegistry("t1")

    async def factory():
        return conn

    async def scenario():
        start = time.monotonic()
        result = await gate.run_ws_utterance(
            factory, PCM, registry=registry, **_ws_params(request_deadline_s=0.3)
        )
        elapsed = time.monotonic() - start
        # The caller MUST return finitely even though recv and close suppress
        # cancellation: bounded by request deadline + cleanup reserve + margin.
        assert elapsed < 0.3 + gate.CLEANUP_RESERVE_S + 2.0, elapsed
        assert not result.ok
        assert result.cleanup_pending, result.cleanup_error
        # Row is forced not-ok and error retained; close was attempted once.
        assert result.error is not None
        assert conn.closed_calls == 1
        # The registry holds the ACTUAL pending tasks (receiver + close).
        snap = registry.snapshot()
        assert snap["pending_count"] >= 1
        assert "receiver" in snap["pending_phases"] or any(
            "receiver" in t["phase"] for t in snap["tasks"]
        )
        assert conn.closed_calls == 1

    _run(scenario())


def test_suppressing_connector_times_out_and_late_resource_is_owned_and_closed():
    late_conn = CleanWSConn()
    connector = LateConnector(late_conn, delay_s=0.1)
    registry = gate.AsyncLifetimeRegistry("t2")

    async def scenario():
        result = await gate.run_ws_utterance(
            connector, PCM, registry=registry,
            **_ws_params(request_deadline_s=0.05),
        )
        assert not result.ok
        assert "connect failed" in (result.error or "")
        # Let the suppressing connector complete LATE.
        await asyncio.sleep(0.3)
        snap = registry.snapshot()
        # The late-acquired connection was retained (not leaked) and a late
        # close was scheduled/observed through the adapter contract.
        late = [r for r in snap["resources"] if "late" in r["kind"]]
        assert late, snap
        assert late[0]["close_state"] in ("close_requested", "closed")
        assert late_conn.closed is True
        # No exception was silently discarded by the connector task.
        conn_entries = [t for t in snap["tasks"] if t["phase"] == "connect"]
        assert conn_entries and conn_entries[0]["state"] == "done"
        assert conn_entries[0]["cancel_requested"] is True

    _run(scenario())


# ──────────────────────────────────────────────────────────────────────
# 2. Expired deadline starts NO new work (no task, no probe, no coroutine)
# ──────────────────────────────────────────────────────────────────────


def test_expired_deadline_creates_no_task_and_closes_bare_coroutine():
    registry = gate.AsyncLifetimeRegistry("t3")

    async def scenario():
        started = asyncio.get_running_loop()
        tasks_before = set(asyncio.all_tasks())
        expired = time.monotonic() - 1.0

        async def never_started():
            raise AssertionError("must never run")

        coro = never_started()
        with pytest.raises(TimeoutError):
            await gate._bounded(coro, expired, "never", registry)
        # No task was created for the expired awaitable.
        assert set(asyncio.all_tasks()) == tasks_before
        assert registry.snapshot()["task_count"] == 0
        assert coro.cr_frame is None  # closed in place, not scheduled

        # run_ws_utterance with an expired request deadline: the connector
        # factory is never even called.
        calls = []

        async def factory():
            calls.append(1)
            return CleanWSConn()

        result = await gate.run_ws_utterance(
            factory, PCM, registry=registry,
            **_ws_params(request_deadline_s=0.0),
        )
        assert not result.ok
        assert calls == []
        assert "connect failed" in (result.error or "")

    _run(scenario())


def test_b2_expired_budget_creates_no_pair_tasks_and_keeps_100_denominator():
    corpus = _corpus(100)
    admitted = _admitted(corpus)
    registry = gate.AsyncLifetimeRegistry("t4")

    class _Cfg:
        chunk_ms = 100
        pace = False
        request_deadline_s = 5.0
        post_final_window_s = 0.05
        base_url = "http://127.0.0.1:9"
        http_path = "/v1/asr"
        mode = "ws"

    async def factory():  # must never be reached
        raise AssertionError("no connect after expiry")

    async def scenario():
        tasks_before = set(asyncio.all_tasks())
        results, phase = await gate.run_b2(
            factory, corpus, mode="ws", config=_Cfg(),
            deadline_mono=time.monotonic() - 0.5,
            admitted=admitted, registry=registry,
        )
        # Fixed denominator: exactly 100 rows, all explicitly failed, none
        # fabricated as success; no pair tasks were ever created.
        assert len(results) == 100
        assert all(not r.ok for r in results)
        assert all(r.error for r in results)
        assert phase.requested_width == 2
        assert set(asyncio.all_tasks()) == tasks_before

    _run(scenario())


# ──────────────────────────────────────────────────────────────────────
# 3. External caller cancellation semantics are preserved
# ──────────────────────────────────────────────────────────────────────


def test_external_cancellation_propagates_and_cancels_child_once():
    conn = SuppressingWSConn(suppress_recv=True, suppress_close=False)
    registry = gate.AsyncLifetimeRegistry("t5")

    async def factory():
        return conn

    async def scenario():
        runner = asyncio.ensure_future(
            gate.run_ws_utterance(
                factory, PCM, registry=registry,
                **_ws_params(request_deadline_s=30.0),
            )
        )
        await asyncio.sleep(0.1)  # let it connect and start receiving
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
        snap = registry.snapshot()
        # The receiver got exactly ONE cancel request, tracked truthfully.
        recv = [t for t in snap["tasks"] if t["phase"] == "receiver"]
        assert recv and recv[0]["cancel_requested"] is True

    _run(scenario())


# ──────────────────────────────────────────────────────────────────────
# 4. B2 pair: no floor, real width 2, pair span basis, no extra sends
# ──────────────────────────────────────────────────────────────────────


def test_b2_pair_overlap_real_width_and_fixed_rows():
    corpus = _corpus(100)
    admitted = _admitted(corpus)
    registry = gate.AsyncLifetimeRegistry("t6")
    conns: list[CleanWSConn] = []
    in_flight = 0
    max_in_flight = 0

    class OverlapConn(CleanWSConn):
        async def send(self, data: Any) -> None:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                await asyncio.sleep(0.001)
                await super().send(data)
            finally:
                in_flight -= 1

    async def factory():
        conn = OverlapConn()
        conns.append(conn)
        return conn

    class _Cfg:
        chunk_ms = 100
        pace = False
        request_deadline_s = 5.0
        post_final_window_s = 0.01
        base_url = "http://127.0.0.1:9"
        http_path = "/v1/asr"
        mode = "ws"

    async def scenario():
        start = time.monotonic()
        results, phase = await gate.run_b2(
            factory, corpus, mode="ws", config=_Cfg(),
            deadline_mono=start + 30.0, admitted=admitted, registry=registry,
        )
        assert len(results) == 100  # fixed frozen denominator
        assert all(r.ok for r in results)
        assert phase.realized_width == 2  # measured, not assumed
        assert max_in_flight >= 2
        # Pair span present on every row and keyed from first-send basis.
        assert all(r.pair_span_s is not None and r.pair_span_s >= 0 for r in results)
        # No extra unlabelled sends: exactly 100 utterances each sent once.
        pcm_sends = sum(
            1 for c in conns for d in c.sent if isinstance(d, bytes) and d
        )
        assert pcm_sends == 100 * (len(PCM) // 3200)
        # Registry fully clean after normal completion.
        snap = registry.snapshot()
        assert snap["pending_count"] == 0, snap["pending_phases"]
        assert snap["open_resource_count"] == 0

    _run(scenario())


# ──────────────────────────────────────────────────────────────────────
# 5. Normal completion preserves the API, consumes exceptions, cleans up
# ──────────────────────────────────────────────────────────────────────


def test_normal_completion_cleans_known_resources_and_consumes_exceptions():
    conn = CleanWSConn()
    registry = gate.AsyncLifetimeRegistry("t7")

    async def factory():
        return conn

    async def scenario():
        result = await gate.run_ws_utterance(
            factory, PCM, registry=registry, **_ws_params()
        )
        assert result.ok, result.error
        assert conn.closed is True
        assert not result.cleanup_pending
        assert result.cleanup_error is None
        snap = registry.snapshot()
        assert snap["pending_count"] == 0
        assert all(r["close_state"] == "closed" for r in snap["resources"])
        # No completed exception was left unretrieved.
        assert all(
            t["state"] in ("done", "cancelled") or t["exception"] is not None
            for t in snap["tasks"]
        )
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        assert not pending

    _run(scenario())


def test_gather_free_b2_and_no_wait_for_in_driver_source():
    source = _DRIVER_PATH.read_text(encoding="utf-8")
    assert "asyncio.wait_for" not in source
    assert "asyncio.gather" not in source
    assert "asyncio.run(" not in source
    assert "max(0.001" not in source  # expired-budget floor removed
    assert "subprocess" not in source
    assert "Popen" not in source
    assert "os.kill" not in source
    assert "signal." not in source


def test_pending_registry_forces_notqualified_even_when_rows_ok():
    rows = [
        gate.UtteranceResult(
            order=i, file=f"u{i:03d}.wav", id=f"i{i}", warm=i < 3,
            pair_index=None, ok=True, audio_s=1.0, transcript="hello world",
        )
        for i in range(100)
    ]
    corpus = _corpus(100)
    qual_clean = gate.qualification(
        rows, [], corpus, identity_present=True, async_lifetime_pending=False
    )
    qual_pending = gate.qualification(
        rows, [], corpus, identity_present=True, async_lifetime_pending=True
    )
    order = {"QUALIFIED": 0, "UNPROVEN": 1, "NOTQUALIFIED": 2}
    assert order[qual_pending["status"]] == order["NOTQUALIFIED"]
    assert any("async lifetime" in r for r in qual_pending["reasons"])
    assert order[qual_clean["status"]] < order[qual_pending["status"]]


# ──────────────────────────────────────────────────────────────────────
# 6. CLI lifecycle: ONE absolute budget, owned loop, real suppressing child
# ──────────────────────────────────────────────────────────────────────

# Configured overall deadline for the isolated CLI child. The child MUST exit
# on its own inside this budget plus a small interpreter-startup margin; the
# upper assertion below uses that sum, NOT the old broad 60s ceiling that the
# previous code (overall + 5s top grace + 5s drain grace) would also satisfy.
_CLI_OVERALL_S = 2.0
_CLI_STARTUP_MARGIN_S = 2.0
_CLI_ELAPSED_UPPER_S = _CLI_OVERALL_S + _CLI_STARTUP_MARGIN_S
# Finite grace before the SOLE fallback (TERM to the one owned child).
_CLI_CHILD_FALLBACK_S = 15.0
_CLI_TERM_GRACE_S = 5.0


class _RunConfigStub:
    """Minimal config for _run_gate_owned_loop deadline tests."""

    def __init__(self, *, overall_deadline_s: float, label: str = "cli-unit"):
        self.overall_deadline_s = overall_deadline_s
        self.label = label


def test_owned_loop_rejects_nonpositive_and_nonfinite_overall_without_tasks():
    """A non-positive or non-finite overall deadline is an explicit NF/error
    return WITHOUT starting any task/probe and with NO 1ms floor."""

    calls = []

    async def spy(*_a, **k):
        calls.append(k)
        raise AssertionError("run_gate must NOT start for an invalid deadline")

    original = gate.run_gate
    gate.run_gate = spy
    try:
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            cfg = _RunConfigStub(overall_deadline_s=bad)
            outcome, evidence, registry = gate._run_gate_owned_loop(cfg)
            assert outcome is None, bad
            assert evidence["top_state"] == "invalid_deadline", (bad, evidence)
            assert "not" in evidence["error"], (bad, evidence)
            snap = registry.snapshot()
            assert snap["task_count"] == 0, (bad, snap)
            assert snap["pending_count"] == 0, (bad, snap)
        assert calls == []  # no layer ever invoked run_gate
    finally:
        gate.run_gate = original


def test_owned_loop_bounds_shutdown_inside_one_deadline_and_shares_it():
    """The top-level wait is bounded INSIDE the configured overall deadline
    (no +grace renewal), the same absolute deadline is shared with run_gate,
    and a cancellation-suppressing child stays pending while the CLI returns
    finitely instead of awaiting cancellation completion."""
    seen = {}

    async def fake_run_gate(config, registry=None, *, deadline_mono=None):
        seen["deadline_mono"] = deadline_mono

        async def suppressor():
            while True:
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    continue  # suppress cancellation indefinitely

        registry.track_task(asyncio.ensure_future(suppressor()), "receiver")
        await asyncio.sleep(3600)

    original = gate.run_gate
    gate.run_gate = fake_run_gate
    try:
        overall = 0.6
        cfg = _RunConfigStub(overall_deadline_s=overall)
        start = time.monotonic()
        outcome, evidence, registry = gate._run_gate_owned_loop(cfg)
        elapsed = time.monotonic() - start
        assert outcome is None
        assert evidence["top_state"] == "timeout_cancel_requested", evidence
        # ONE absolute deadline was shared with run_gate, not recomputed.
        assert seen["deadline_mono"] is not None
        assert 0.0 < seen["deadline_mono"] - start <= overall + 0.05
        # Shutdown stayed inside the one overall budget (small scheduling
        # margin), NOT overall + 5s + 5s of renewed grace.
        assert elapsed < overall + 0.75, elapsed
        # The suppressing receiver remains an actual pending registry task.
        snap = evidence["async_lifetime"]
        assert snap["pending_count"] >= 1, snap
        assert any("receiver" in t["phase"] for t in snap["tasks"]), snap
        assert evidence["loop_closed"] is True
    finally:
        gate.run_gate = original


def test_owned_loop_keyboardinterrupt_leaves_no_untracked_supervise_task():
    """An external KeyboardInterrupt cancels the supervise WRAPPER and the top
    task exactly once and both are registered/observed, so no untracked
    pending task with an unretrieved exception survives loop.close()."""
    real_new_loop = gate.asyncio.new_event_loop

    class _LoopWrapper:
        def __init__(self, inner):
            self._inner = inner
            self.run_calls = 0

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def run_until_complete(self, fut):
            self.run_calls += 1
            if self.run_calls == 1:
                # Simulate a KeyboardInterrupt delivered while the first
                # run_until_complete(supervisor) is executing.
                raise KeyboardInterrupt
            return self._inner.run_until_complete(fut)

    def new_loop():
        return _LoopWrapper(real_new_loop())

    async def hanging(*_a, **_k):
        await asyncio.sleep(3600)

    original_run = gate.run_gate
    gate.run_gate = hanging
    gate.asyncio.new_event_loop = new_loop
    try:
        cfg = _RunConfigStub(overall_deadline_s=5.0)
        outcome, evidence, registry = gate._run_gate_owned_loop(cfg)
        assert outcome is None
        assert evidence["top_state"] == "interrupted_cancel_requested", evidence
        snap = registry.snapshot()
        phases = {t["phase"] for t in snap["tasks"]}
        # Both the supervise wrapper and the top task are owned.
        assert "cli supervise" in phases, snap
        assert "run_gate top-level" in phases, snap
        # Every still-registered task either completed or recorded a cancel
        # request: nothing is left pending-and-untracked.
        assert all(
            t["done"] or t["cancel_requested"] for t in snap["tasks"]
        ), snap
        assert evidence["loop_closed"] is True
    finally:
        gate.run_gate = original_run
        gate.asyncio.new_event_loop = real_new_loop


def _write_cli_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    """A manifest+corpus good enough for real 100-row WAV admission; the
    isolated child never reaches a real service because its ONLY monkeypatches
    are a synthetic probe and a cancellation-suppressing connector.

    Uses the SAME manifest shape as the owned-input test (corpus_dir + full
    per-item metadata) so real admission accepts all 100 rows."""
    import hashlib
    import wave as wave_mod

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    pcm = b"\x00\x01" * 160  # 0.01 s at 16 kHz mono PCM16
    items = []
    for i in range(100):
        name = f"u{i:03d}.wav"
        path = corpus_dir / name
        with wave_mod.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        items.append(
            {
                "file": name,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rate": 16000,
                "channels": 1,
                "bit_depth": 16,
                "duration_s": 0.01,
                "transcript": "hello world",
                "lang": "en",
                "id": f"i{i}",
            }
        )
    payload = json.dumps({"corpus_dir": str(corpus_dir), "items": items}).encode("utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(payload)
    return manifest, corpus_dir, hashlib.sha256(payload).hexdigest()


# The isolated child imports the REAL driver via the same testloader pattern
# used above, then monkeypatches ONLY ``probe_service_identity`` and
# ``_make_ws_factory`` with controlled synthetic implementations. It NEVER
# replaces production ``main``/``run_gate``/``_run_gate_owned_loop``/
# ``_bounded``/the registry/cleanup, and it NEVER releases the suppressor (doing
# so would hide the bug the test exists to prove). The fake probe performs no
# network work at all; the suppressing connector's recv swallows cancellation
# indefinitely so the OLD wait_for/asyncio.run shutdown would hang here.
_CLI_CHILD_SOURCE = '''\
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

_driver_path = Path(sys.argv[1])
_marker = Path(sys.argv[2])
_argv = sys.argv[3:]

_spec = importlib.util.spec_from_file_location(
    "edgellm_asr_ws_perf_gate_child", _driver_path
)
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)

_inj = {
    "injection_entered": False,
    "probe_calls": 0,
    "factory_calls": 0,
    "recv_cancellation_suppressed": 0,
    "send_calls": 0,
}


def _flush():
    _marker.write_text(json.dumps(_inj), encoding="utf-8")


class _SuppressingConn:
    async def send(self, data):
        _inj["send_calls"] += 1

    async def recv(self):
        while True:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                # SUPPRESSES cancellation indefinitely (the F4 failure mode).
                _inj["recv_cancellation_suppressed"] += 1
                _flush()
                continue

    async def close(self, code=1000):
        while True:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue


async def _fake_probe(base_url, deadline_mono=None, registry=None):
    # Controlled synthetic service contract; NO network tasks are created.
    _inj["probe_calls"] += 1
    _flush()
    return {
        "base_url": base_url,
        "capabilities": {
            "status": 200,
            "body": {
                "asr": {
                    "model_id": "synthetic",
                    "backend": "synthetic",
                    "concurrency": {
                        "admission_limit": 2,
                        "backend_max_concurrent": 2,
                    },
                }
            },
            "request_error": None,
            "cleanup_pending": False,
            "cleanup_error": None,
        },
        "capabilities_cleanup": {"pending": False, "error": None},
        "readyz": {
            "status": 200,
            "body": {"status": "ready"},
            "request_error": None,
            "cleanup_pending": False,
            "cleanup_error": None,
        },
        "readyz_cleanup": {"pending": False, "error": None},
    }


def _fake_ws_factory(base_url, ws_path, language, sample_rate):
    async def factory():
        _inj["factory_calls"] += 1
        _flush()
        return _SuppressingConn()

    return factory


# Patch ONLY these two names on the child module; production main/run_gate/
# owned loop/_bounded/registry/cleanup are untouched.
gate.probe_service_identity = _fake_probe
gate._make_ws_factory = _fake_ws_factory

_inj["injection_entered"] = True
_flush()

_rc = gate.main(_argv)
_inj["rc"] = _rc
_flush()
sys.exit(_rc)
'''


def test_cli_finite_shutdown_suppressing_child_real_runner(tmp_path: Path):
    """Run the ACTUAL CLI runner in ONE owned isolated child process (no GPU,
    no device, no external network, no foreign processes) whose real
    production owned loop must return finitely even though its injected
    connector suppresses cancellation indefinitely.

    The child MUST exit on its own with rc 4 and write truthful
    ``async-lifetime-notqualified.json`` pending-registry evidence. TERM is
    the sole fallback and a survivor stops the whole suite (pytest.exit), not
    a continuing pytest.fail; no KILL/escalation and no foreign signals."""
    import subprocess  # test-side only; the driver itself never uses it

    manifest, _root, manifest_sha = _write_cli_fixture(tmp_path)
    out_dir = tmp_path / "out"
    marker = tmp_path / "child-evidence.json"
    child = tmp_path / "cli_suppressing_child.py"
    child.write_text(_CLI_CHILD_SOURCE, encoding="utf-8")

    cmd = [
        sys.executable, str(child), str(_DRIVER_PATH), str(marker),
        "--base-url", "http://127.0.0.1:9",  # inert; never contacted (probe faked)
        "--mode", "ws",
        "--concurrency", "1",
        "--manifest", str(manifest),
        "--manifest-sha256", manifest_sha,
        "--output", str(out_dir),
        "--overall-deadline-s", str(_CLI_OVERALL_S),
        "--request-deadline-s", "5",
        "--label", "f4-cli-suppress",
    ]
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        stdout, stderr = proc.communicate(timeout=_CLI_CHILD_FALLBACK_S)
    except subprocess.TimeoutExpired:
        proc.terminate()  # TERM only, the sole explicitly owned child
        try:
            proc.wait(timeout=_CLI_TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            # Survivor: STOP the entire suite (never continue after a leak).
            pytest.exit(
                "owned CLI child survived TERM; STOP: no KILL escalation, "
                "no further tests",
                returncode=1,
            )
        pytest.fail("owned CLI child needed TERM: finite shutdown UNPROVEN")
    elapsed = time.monotonic() - start
    assert proc.returncode is not None

    # Injection evidence: the monkeypatch really entered and the connector
    # really suppressed at least one cancellation (otherwise the test would
    # not distinguish the new owned loop from the old asyncio.run behavior).
    assert marker.exists(), (stdout, stderr)
    inj = json.loads(marker.read_text(encoding="utf-8"))
    assert inj["injection_entered"] is True, inj
    assert inj["probe_calls"] >= 1, inj
    assert inj["factory_calls"] >= 1, inj
    assert inj["recv_cancellation_suppressed"] >= 1, inj

    # The child exited on its own with the NF nonzero rc (never rc 0 success).
    assert proc.returncode == 4, (proc.returncode, stdout, stderr)
    assert inj.get("rc") == 4, inj
    assert "Traceback" not in (stderr or ""), stderr

    # Tight elapsed upper bound: configured overall + interpreter-startup
    # margin. This EXCLUDES the old overall+5s+5s renewed-budget behavior.
    assert elapsed < _CLI_ELAPSED_UPPER_S, elapsed

    # Truthful pending-cleanup evidence file (and NO success file).
    nf = out_dir / "async-lifetime-notqualified.json"
    assert nf.exists(), sorted(p.name for p in out_dir.iterdir())
    assert not (out_dir / "run.json").exists()
    report = json.loads(nf.read_text(encoding="utf-8"))
    assert report["status"] == "NOTQUALIFIED", report
    life = report["async_lifetime"]["async_lifetime"]
    assert life["pending_count"] >= 1, life
    assert any("receiver" in t["phase"] for t in life["tasks"]), life
    assert life["pending_phases"], life

