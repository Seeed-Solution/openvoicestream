"""v011 ASR ready provenance + IFB health consumption — focused CPU tests.

Scope (single focused invocation, no repo-wide suite):
  * real edited production sources: voxedge.backends.jetson.trt_edge_llm_asr
    (factory + backend) and voxedge.backends.jetson.worker_io.WorkerIO;
  * FakePopen + in-memory pipes only — NO native worker execution, NO kill
    calls (any fake cleanup is EOF/terminate, asserted kill-free);
  * opt-in ``require_ifb`` ready-provenance gate incl. legacy default path;
  * bounded no-KILL owned-worker cleanup with the SHARED-deadline budget
    (EOF phase reserves 2/3, TERM gets the remaining reserve, no minimum
    wait past the expired deadline);
  * canonical native ``type``-keyed telemetry (asr_ifb_health health_final /
    asr_ifb_trace) replayed as raw captured native JSON — never routed into
    request queues; malformed primitives never kill the reader;
  * additive runtime_diagnostics incl. live worker_alive from the owned
    Popen poll().

The voxedge repo is an external source tree; add it explicitly.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

import voxedge.backends.jetson.trt_edge_llm_asr as asr_mod
from voxedge.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRConfig,
    build_config_from_env,
)
from voxedge.backends.jetson.worker_io import WorkerIO

pytestmark = pytest.mark.timeout(60)


# ── in-memory fake process machinery (no native execution, kill is a tripwire)


class FakeOutPipe:
    """Queue-backed stdout/stderr pipe: feedable, iterable, closable."""

    def __init__(self):
        self.q: "queue.Queue[str]" = queue.Queue()
        self.closed = False

    def feed(self, obj):
        if isinstance(obj, str) and not obj.endswith("\n"):
            obj += "\n"
        self.q.put(obj)

    def close(self):
        self.closed = True
        self.q.put("")  # EOF sentinel

    def readline(self) -> str:
        try:
            return self.q.get(timeout=5)
        except queue.Empty:
            return ""

    def __iter__(self):
        while True:
            try:
                line = self.q.get(timeout=5)
            except queue.Empty:
                return
            if line == "":
                return
            yield line


class FakeInPipe:
    def __init__(self, proc):
        self._proc = proc
        self.closed = False
        self.written: list[str] = []

    def write(self, data):
        self.written.append(data)
        # A real worker acks cancels with a terminal 'cancelled' event; the
        # fake mirrors that so cancel_event-driven request() loops terminate.
        if '"type": "cancel"' in data:
            try:
                rid = json.loads(data).get("id")
            except Exception:
                rid = None
            self._proc.stdout.feed(
                json.dumps({"event": "cancelled", "id": rid}))

    def flush(self):
        pass

    def close(self):
        self.closed = True
        if self._proc.exit_on_stdin_close:
            self._proc._exit(0)


class FakePopen:
    """Records the teardown contract. ``kill()`` must NEVER be called."""

    _pid_seq = 0

    def __init__(self, args, stdout_lines=(), *, exit_on_stdin_close=True,
                 terminate_exits=True, **kwargs):
        FakePopen._pid_seq += 1
        self.pid = 42_000 + FakePopen._pid_seq
        self.args = args
        self.exit_on_stdin_close = exit_on_stdin_close
        self.terminate_exits = terminate_exits
        self.stdout = FakeOutPipe()
        self.stderr = FakeOutPipe()
        self.stdin = FakeInPipe(self)
        for line in stdout_lines:
            self.stdout.feed(line)
        self._exited = False
        self.returncode = None
        self.kill_called = False
        self.terminate_called = 0
        self.created_at = time.monotonic()

    def _exit(self, code):
        self._exited = True
        self.returncode = code

    def poll(self):
        return self.returncode if self._exited else None

    def terminate(self):
        self.terminate_called += 1
        if self.terminate_exits:
            self._exit(-15)

    def kill(self):  # tripwire: forbidden by the no-KILL contract
        self.kill_called = True
        self._exit(-9)

    def wait(self, timeout=None):
        if self._exited:
            return self.returncode
        # Model a REAL blocking wait: consume the offered timeout, then report
        # TimeoutExpired — this is what makes the shared-deadline budget
        # measurable.
        time.sleep(max(0.0, float(timeout) if timeout is not None else 0.0))
        raise subprocess.TimeoutExpired(self.args, timeout)


@pytest.fixture
def fake_proc_env(monkeypatch):
    """Route backend subprocess usage to FakePopen.

    Returns (records, queue_launch): the backend's own ``subprocess.Popen``
    call inside ``_ensure_worker`` consumes the NEXT queued (stdout_lines,
    kwargs) pair, so tests script what the spawned worker will emit.
    """
    records: list[FakePopen] = []
    pending: list[tuple[list, dict]] = []

    def make(*args, **kwargs):
        if pending:
            lines, extra = pending.pop(0)
            kwargs = {**extra, **kwargs}
            kwargs["stdout_lines"] = lines
        proc = FakePopen(*args, **kwargs)
        records.append(proc)
        return proc

    def queue_launch(stdout_lines=(), **proc_kwargs):
        pending.append((list(stdout_lines), proc_kwargs))

    fake_subprocess = SimpleNamespace(
        Popen=make,
        TimeoutExpired=subprocess.TimeoutExpired,
        PIPE=subprocess.PIPE,
    )
    monkeypatch.setattr(asr_mod, "subprocess", fake_subprocess)
    return records, queue_launch


def make_backend(**overrides) -> asr_mod.TRTEdgeLLMASRBackend:
    config = TRTEdgeLLMASRConfig(
        worker_binary="fake_worker_bin",
        max_slots=overrides.pop("max_slots", 2),
        require_ifb=overrides.pop("require_ifb", False),
        worker_warmup=False,
        **overrides,
    )
    return asr_mod.TRTEdgeLLMASRBackend(config)


def ready_line(**fields) -> str:
    payload = {"event": "ready", "init_ms": 1.0}
    payload.update(fields)
    return json.dumps(payload)


IFB2_READY = dict(
    ifb=True, max_slots=2, gpu_slots=2, engine_max_batch_size=2
)

# Raw captured native worker lines (canonical type key; source:
# qwen3_asr_worker emitHealth health_final + asr_ifb_trace emissions).
NATIVE_HEALTH_LINE = json.dumps({
    "type": "asr_ifb_health",
    "gpu_slots": 2,
    "engine_max_batch_size": 2,
    "app_sessions": 1,
    "admitted_mid_flight": 2,
    "completed": 17,
    "cancelled": 0,
    "failed": 0,
    "queued": 0,
    "stalls_founder_only": 1,
    "stalls_guided": 3,
    "stalls_incompatible": 0,
    "stalls_no_capacity": 2,
})
NATIVE_TRACE_LINE = json.dumps({
    "type": "asr_ifb_trace",
    "phase": "terminal",
    "work_id": 126,
    "engine_request_id": 126,
    "external_id": "ws-client-abc",
    "sid": "sess-9",
})


# ── 1. config + factory: EDGE_LLM_ASR_REQUIRE_IFB boolean, default false ────


def test_factory_require_ifb_default_false_and_env_boolean():
    assert build_config_from_env({}).require_ifb is False
    assert TRTEdgeLLMASRConfig().require_ifb is False
    for raw in ("1", "true", "TRUE", "yes"):
        assert build_config_from_env(
            {"EDGE_LLM_ASR_REQUIRE_IFB": raw}
        ).require_ifb is True, raw
    for raw in ("0", "false", "no"):
        assert build_config_from_env(
            {"EDGE_LLM_ASR_REQUIRE_IFB": raw}
        ).require_ifb is False, raw


# ── 2. opt-in False preserves the legacy max_slots=2 non-IFB pool path ──────


def test_legacy_require_ifb_false_accepts_non_ifb_maxslots2(fake_proc_env):
    records, queue_launch = fake_proc_env
    backend = make_backend(require_ifb=False, max_slots=2)
    queue_launch([ready_line()])  # legacy binary: no ifb fields at all
    backend._ensure_worker()

    assert backend._worker_ready_meta.get("event") == "ready"
    assert "ifb" not in backend._worker_ready_meta
    assert backend._wio is not None
    assert backend.runtime_diagnostics()["worker_alive"] is True
    assert not records[0].kill_called


# ── 3. opt-in True valid IFB2 ready is accepted ─────────────────────────────


def test_require_ifb_accepts_valid_ifb2_ready(fake_proc_env):
    records, queue_launch = fake_proc_env
    backend = make_backend(require_ifb=True, max_slots=2)
    queue_launch([ready_line(**IFB2_READY)])
    backend._ensure_worker()

    assert backend._worker_ready_meta.get("ifb") is True
    assert backend._wio is not None
    snap = backend.runtime_diagnostics()
    assert snap["require_ifb"] is True
    assert snap["validated_ready"]["max_slots"] == 2
    assert snap["latest_asr_ifb_health"] is None  # no stale engine health
    assert snap["worker_alive"] is True           # owned Popen poll() is None
    assert records[0].stdin.closed is False  # success keeps the worker alive


# ── 4. opt-in True rejects missing/false/bool/non-int/mismatch/maxbatch ─────


@pytest.mark.parametrize(
    "fields",
    [
        {},  # ifb missing
        {"ifb": False, "max_slots": 2, "gpu_slots": 2,
         "engine_max_batch_size": 2},  # ifb false
        {"ifb": True, "max_slots": True, "gpu_slots": 2,
         "engine_max_batch_size": 2},  # bool-as-int rejected
        {"ifb": True, "max_slots": 2, "gpu_slots": "2",
         "engine_max_batch_size": 2},  # non-int gpu_slots
        {"ifb": True, "max_slots": 2, "gpu_slots": 3,
         "engine_max_batch_size": 4},  # gpu mismatch
        {"ifb": True, "max_slots": 2, "gpu_slots": 2,
         "engine_max_batch_size": 1},  # engine batch < requested slots
    ],
)
def test_require_ifb_rejections_reject_startup_with_eof_cleanup(
    fake_proc_env, fields
):
    records, queue_launch = fake_proc_env
    backend = make_backend(require_ifb=True, max_slots=2)
    backend._ready = True  # a previously-ready backend must NOT stay ready
    queue_launch([ready_line(**fields)])
    with pytest.raises(RuntimeError, match="IFB provenance rejected"):
        backend._ensure_worker()

    proc = records[0]
    assert not proc.kill_called
    assert proc.stdin.closed is True          # EOF first
    assert proc.returncode == 0               # exited on EOF within deadline
    assert proc.terminate_called == 0         # no duplicate/needless TERM
    assert backend._worker is None            # clean exit → respawn allowed
    assert backend.is_ready() is False        # ready can NEVER remain true


def test_launch_ready_json_failures_also_clean_owned_worker(fake_proc_env):
    records, queue_launch = fake_proc_env
    for stdout in ([], ["not-json\n"], [ready_line()[:-2] + "\n"],
                   [json.dumps({"event": "error"}) + "\n"]):
        backend = make_backend(require_ifb=False)
        backend._ready = True
        queue_launch(stdout)
        with pytest.raises(RuntimeError):
            backend._ensure_worker()
        proc = records[-1]
        assert not proc.kill_called
        assert proc.returncode == 0
        assert backend._worker is None
        assert backend.is_ready() is False


# ── 5. survivor: ready-rejected worker that ignores EOF+TERM stays referenced


def test_survivor_kept_and_prevents_respawn(fake_proc_env):
    records, queue_launch = fake_proc_env
    backend = make_backend(require_ifb=True, max_slots=2)
    queue_launch([ready_line()], exit_on_stdin_close=False,
                 terminate_exits=False)
    with pytest.raises(RuntimeError, match="IFB provenance rejected"):
        backend._ensure_worker()

    proc = records[0]
    assert not proc.kill_called
    assert proc.stdin.closed is True
    assert proc.terminate_called == 1
    assert proc.returncode is None                    # still alive
    assert backend._worker is proc                    # reference kept
    assert backend.is_ready() is False
    assert backend.runtime_diagnostics()["worker_alive"] is True

    # A later _ensure_worker must NOT respawn a second worker over the
    # still-live survivor.
    backend._ensure_worker()
    assert len(records) == 1


# ── 6. restart_worker: EOF-exit branch (no KILL) ────────────────────────────


def test_restart_worker_eof_exit_no_kill(fake_proc_env):
    records, queue_launch = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["x"])
    records.append(proc)
    backend._worker = proc
    backend._worker_ready_meta = {"event": "ready", "ifb": True}
    backend._on_worker_telemetry({"type": "asr_ifb_health", "queued": 0})

    backend.restart_worker()

    assert not proc.kill_called
    assert proc.stdin.closed is True
    assert proc.returncode == 0
    assert proc.terminate_called == 0
    assert backend._worker is None
    # diagnostics reset on successful owned restart
    snap = backend.runtime_diagnostics()
    assert snap["latest_asr_ifb_health"] is None
    assert snap["validated_ready"] == {}
    assert snap["worker_alive"] is None  # no owned proc after clean restart


def test_restart_worker_terminate_branch_and_timeout_branch(fake_proc_env):
    # TERM branch: ignores EOF, exits on terminate.
    records, _ = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["x"], exit_on_stdin_close=False, terminate_exits=True)
    records.append(proc)
    backend._worker = proc
    backend.restart_worker()
    assert not proc.kill_called
    assert proc.stdin.closed is True
    assert proc.terminate_called == 1
    assert proc.returncode == -15
    assert backend._worker is None

    # Timeout branch: survives EOF + TERM → survivor kept, RuntimeError
    # PROPAGATES (never swallowed into a warning / clean-restart log).
    backend2 = make_backend()
    proc2 = FakePopen(["x"], exit_on_stdin_close=False, terminate_exits=False)
    records.append(proc2)
    backend2._worker = proc2
    backend2._ready = True
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="survived"):
        backend2.restart_worker()
    elapsed = time.monotonic() - t0
    assert not proc2.kill_called
    assert backend2._worker is proc2
    assert backend2.is_ready() is False
    backend2._ensure_worker()  # must not respawn over the live survivor
    assert records[-1] is proc2
    # ONE shared deadline: ~15 s total, bounded, not 15+15 and not hanging.
    assert 13.0 <= elapsed <= 25.0, elapsed


# ── 7. shared-deadline budget: EOF reserve 2/3, TERM reserve, no floor ──────


def test_shutdown_budget_eof_reserve_and_terminate_within_shared_deadline(
    fake_proc_env,
):
    records, _ = fake_proc_env
    backend = make_backend()
    # Ignores stdin EOF, exits on terminate → TERM branch must still fit in
    # the SAME shared budget (old defect: EOF wait consumed the whole budget).
    proc = FakePopen(["x"], exit_on_stdin_close=False, terminate_exits=True)
    records.append(proc)
    t0 = time.monotonic()
    exited = backend._shutdown_owned_worker(proc, deadline_s=1.5)
    elapsed = time.monotonic() - t0
    assert exited is True
    assert proc.terminate_called == 1
    assert not proc.kill_called
    # EOF true-wait consumes ~2/3 of the shared 1.5 s budget (~1.0 s); TERM then
    # exits the fake immediately from the remaining reserve.
    assert 0.8 <= elapsed <= 2.5, elapsed


def test_shutdown_budget_no_wait_past_expired_deadline(fake_proc_env):
    records, _ = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["x"], exit_on_stdin_close=False, terminate_exits=False)
    records.append(proc)
    t0 = time.monotonic()
    exited = backend._shutdown_owned_worker(proc, deadline_s=1.5)
    elapsed = time.monotonic() - t0
    assert exited is False  # survivor
    assert not proc.kill_called
    # No minimum wait beyond the expired shared deadline.
    assert elapsed <= 2.5, elapsed


# ── 8. WorkerIO telemetry routing isolation (real WorkerIO, fake pipes) ─────


def make_wio(telemetry_callback=None, concurrency=1):
    proc = FakePopen(["x"])
    wio = WorkerIO(
        proc, concurrency=concurrency, telemetry_callback=telemetry_callback
    )
    return proc, wio


def test_native_type_keyed_health_trace_never_delivered_to_requests():
    received: list[dict] = []
    proc, wio = make_wio(telemetry_callback=received.append)

    def feed():
        # RAW captured native lines: canonical ``type`` key, health even
        # carrying external_id/request_id, trace carrying internal numeric
        # ids — telemetry even so; must never enter the request queue or
        # semaphore accounting.
        proc.stdout.feed(
            json.dumps({"type": "asr_ifb_health", "request_id": "x",
                        "external_id": "ws-client-abc",
                        "admitted_mid_flight": 2}))
        proc.stdout.feed(NATIVE_HEALTH_LINE)
        proc.stdout.feed(NATIVE_TRACE_LINE)
        proc.stdout.feed(json.dumps({"event": "done", "id": "x"}))

    threading.Timer(0.05, feed).start()
    events = list(wio.request({"id": "x"}))

    assert [e["event"] for e in events] == ["done"]
    kinds = [(e.get("type") or e.get("event")) for e in received]
    assert kinds == ["asr_ifb_health", "asr_ifb_health", "asr_ifb_trace"]
    assert wio._sem._value == 1  # telemetry never consumed a slot


def test_event_alias_health_also_recognized_for_source_compat():
    received: list[dict] = []
    proc, wio = make_wio(telemetry_callback=received.append)
    proc.stdout.feed(
        json.dumps({"event": "asr_ifb_health", "id": "x", "queued": 1}))
    proc.stdout.feed(json.dumps({"event": "asr_ifb_trace", "work_id": 5}))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and len(received) < 2:
        time.sleep(0.02)
    assert [(e.get("type") or e.get("event")) for e in received] == [
        "asr_ifb_health", "asr_ifb_trace"]


def test_ordinary_events_route_by_request_id_not_work_id():
    received: list[dict] = []
    proc, wio = make_wio(telemetry_callback=received.append)
    cancel_event = threading.Event()

    def feed():
        # work_id/engine_request_id only: must NOT be delivered to request "x".
        proc.stdout.feed(json.dumps(
            {"event": "done", "work_id": "x", "engine_request_id": "x"}))
        time.sleep(0.3)
        cancel_event.set()

    threading.Thread(target=feed, daemon=True).start()
    events = list(wio.request({"id": "x"}, cancel_event=cancel_event))
    # The ONLY event the fake delivered was the cancel terminal — the
    # work_id/engine_request_id-only 'done' was never routed to request "x".
    assert not any(e["event"] == "done" for e in events)
    assert [e["event"] for e in events] == ["cancelled"]
    assert received == []


def test_malformed_primitives_and_bad_json_do_not_crash_reader():
    received: list[dict] = []
    proc, wio = make_wio(telemetry_callback=received.append)

    def feed():
        proc.stdout.feed("123\n")
        proc.stdout.feed('"a plain string"\n')
        proc.stdout.feed("[1, 2, 3]\n")
        proc.stdout.feed("null\n")
        proc.stdout.feed("not json at all\n")
        proc.stdout.feed(json.dumps({"event": "done", "id": "x"}))

    threading.Timer(0.05, feed).start()
    events = list(wio.request({"id": "x"}))
    assert [e["event"] for e in events] == ["done"]
    assert received == []  # primitives are not recognised telemetry types
    assert wio._reader_thread.is_alive()


def test_callback_exception_isolated_reader_survives():
    def boom(_event):
        raise ValueError("consumer bug")

    proc, wio = make_wio(telemetry_callback=boom)

    def feed():
        proc.stdout.feed(NATIVE_HEALTH_LINE)
        proc.stdout.feed(json.dumps({"event": "done", "id": "x"}))

    threading.Timer(0.05, feed).start()
    events = list(wio.request({"id": "x"}))
    assert [e["event"] for e in events] == ["done"]
    assert wio._reader_thread.is_alive()


def test_tts_control_api_unchanged_without_callback():
    proc, wio = make_wio()  # default None — TTS/control signatures unchanged
    proc.stdout.feed(NATIVE_HEALTH_LINE)
    time.sleep(0.1)  # dropped silently while idle (no callback)
    proc.stdout.feed(json.dumps({"event": "done", "id": "x"}))
    events = list(wio.request({"id": "x"}))
    assert [e["event"] for e in events] == ["done"]
    assert wio._reader_thread.is_alive()


# ── 9. backend diagnostics: bounded, copied, actual counter keys, reset ─────


def test_runtime_diagnostics_bounded_copied_and_logged(fake_proc_env, caplog):
    records, queue_launch = fake_proc_env
    backend = make_backend(require_ifb=True, max_slots=2)
    queue_launch([ready_line(**IFB2_READY)])
    backend._ensure_worker()

    with caplog.at_level("INFO", logger=asr_mod.logger.name):
        backend._on_worker_telemetry(dict(json.loads(NATIVE_HEALTH_LINE)))
        backend._on_worker_telemetry(
            {**dict(json.loads(NATIVE_HEALTH_LINE)), "stalls_guided": 99})
    health_logs = [r for r in caplog.records
                   if "asr_ifb_health" in r.getMessage()]
    assert len(health_logs) == 1  # rate-bounded within the 10 s window
    msg = health_logs[0].getMessage()
    # ACTUAL native counter keys — there is no single "stalls" key upstream.
    for key in ("stalls_founder_only", "stalls_guided",
                "stalls_incompatible", "stalls_no_capacity"):
        assert key in msg, key
    assert "admitted_mid_flight=2" in msg
    assert "'stalls'" not in msg  # never invent a non-native key

    for i in range(70):
        backend._on_worker_telemetry(
            {"type": "asr_ifb_trace", "phase": f"p{i}", "work_id": i})
    snap = backend.runtime_diagnostics()
    assert len(snap["asr_ifb_traces"]) == 64  # bounded ring
    assert snap["asr_ifb_traces"][-1]["phase"] == "p69"

    # snapshot is a copy: mutating it cannot corrupt backend state
    snap["latest_asr_ifb_health"]["admitted_mid_flight"] = 999
    snap["asr_ifb_traces"].clear()
    snap2 = backend.runtime_diagnostics()
    assert snap2["latest_asr_ifb_health"]["admitted_mid_flight"] == 2
    assert len(snap2["asr_ifb_traces"]) == 64
    # additive API — does not override the capability contract
    cap = backend.concurrency_capability()
    assert cap.max_concurrent == 2 and cap.supports_parallel is True


def test_worker_alive_reflects_owned_popen_snapshot(fake_proc_env):
    records, queue_launch = fake_proc_env
    backend = make_backend(require_ifb=True, max_slots=2)
    assert backend.runtime_diagnostics()["worker_alive"] is None  # no proc

    queue_launch([ready_line(**IFB2_READY)])
    backend._ensure_worker()
    backend._worker_ready_meta = {**backend._worker_ready_meta}
    backend._ready = True  # simulate the post-preload cached-ready state
    snap = backend.runtime_diagnostics()
    assert snap["ready"] is True and snap["worker_alive"] is True

    # Worker dies: cached ready stays True but worker_alive flips False, so
    # cached ready can never qualify as live IFB provenance.
    backend._worker._exit(1)
    snap = backend.runtime_diagnostics()
    assert snap["ready"] is True
    assert snap["worker_alive"] is False
    assert snap["worker_pid"] == records[0].pid


def test_end_to_end_native_health_line_reaches_backend_diagnostics(
    fake_proc_env,
):
    records, queue_launch = fake_proc_env
    backend = make_backend(require_ifb=True, max_slots=2)
    queue_launch([ready_line(**IFB2_READY)])
    backend._ensure_worker()
    proc = records[0]

    proc.stdout.feed(NATIVE_HEALTH_LINE)
    proc.stdout.feed(NATIVE_TRACE_LINE)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snap = backend.runtime_diagnostics()
        if snap["latest_asr_ifb_health"] and snap["asr_ifb_traces"]:
            break
        time.sleep(0.05)
    snap = backend.runtime_diagnostics()
    assert snap["latest_asr_ifb_health"]["admitted_mid_flight"] == 2
    assert snap["latest_asr_ifb_health"]["stalls_no_capacity"] == 2
    assert snap["asr_ifb_traces"][-1]["external_id"] == "ws-client-abc"
    proc.stdout.close()


def test_new_launch_resets_stale_engine_health(fake_proc_env):
    records, queue_launch = fake_proc_env
    backend = make_backend(require_ifb=True, max_slots=2)
    queue_launch([ready_line(**IFB2_READY)])
    backend._ensure_worker()
    backend._on_worker_telemetry({"type": "asr_ifb_health", "queued": 7})
    assert backend.runtime_diagnostics()["latest_asr_ifb_health"]["queued"] == 7

    # Old worker dies; next request path respawns a fresh owned worker.
    backend._worker._exit(1)
    queue_launch([ready_line(**IFB2_READY)])
    backend._ensure_worker()
    snap = backend.runtime_diagnostics()
    assert snap["latest_asr_ifb_health"] is None  # no stale engine health
    assert snap["asr_ifb_traces"] == []
    assert snap["validated_ready"]["event"] == "ready"
    assert snap["worker_alive"] is True


# ── 10. restart/request lifecycle race (ownership hardened) ────────────────


def test_restart_teardown_blocks_respawn_no_second_popen(fake_proc_env):
    """While restart_worker owns the lifecycle lock through teardown, a
    concurrent request path MUST NOT launch a second worker. It waits for the
    lock and then reuses the surviving/cleared ownership state."""
    records, queue_launch = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["old"], exit_on_stdin_close=False, terminate_exits=True)
    records.append(proc)
    backend._worker = proc
    backend._wio = None  # teardown will not touch a WorkerIO
    backend._ready = True

    # Script the fresh worker the post-teardown request thread may launch.
    queue_launch([ready_line()])

    teardown_entered = threading.Event()
    allow_finish = threading.Event()
    real_shutdown = backend._shutdown_owned_worker

    def slow_shutdown(worker, **kwargs):
        teardown_entered.set()
        # Hold the lifecycle lock inside the "teardown" window until the test
        # says so. Deterministic — no sleeps for correctness, only for the
        # assertion window below.
        assert allow_finish.wait(10.0)
        return real_shutdown(worker, **kwargs)

    backend._shutdown_owned_worker = slow_shutdown  # type: ignore[assignment]

    restart_errors: list[BaseException] = []

    def do_restart():
        try:
            backend.restart_worker()
        except BaseException as exc:  # pragma: no cover - test plumbing
            restart_errors.append(exc)

    rt = threading.Thread(target=do_restart, daemon=True)
    rt.start()
    assert teardown_entered.wait(5.0)

    # A concurrent request attempts to ensure/launch a worker. It must block on
    # the lifecycle lock, NOT spawn a second Popen during teardown.
    launched_during_teardown: list[int] = []

    def do_request():
        try:
            with backend._worker_lock:
                backend._ensure_worker()
        except BaseException:  # pragma: no cover - test plumbing
            pass
        launched_during_teardown.append(len(records))

    rq = threading.Thread(target=do_request, daemon=True)
    rq.start()
    time.sleep(0.3)  # bounded assertion window; the lock must remain held
    assert len(records) == 1, "second Popen launched during teardown"
    assert rq.is_alive(), "request path did not wait on the lifecycle lock"

    allow_finish.set()
    rt.join(5.0)
    rq.join(5.0)
    assert not restart_errors
    # Teardown confirmed exit → ownership cleared atomically; the request thread
    # is then free to launch exactly one fresh worker.
    assert len(records) <= 2
    if launched_during_teardown:
        assert launched_during_teardown[0] <= 2


def test_restart_survivor_keeps_consistent_pair(fake_proc_env):
    records, _ = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["old"], exit_on_stdin_close=False, terminate_exits=False)
    records.append(proc)
    backend._worker = proc
    backend._wio = None
    backend._ready = True

    # Simulate the survivor outcome deterministically (without the 15 s wait).
    backend._shutdown_owned_worker = lambda worker, **kw: False  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="survived"):
        backend.restart_worker()

    # Survivor stays referenced, ready False, and the pair stays consistent
    # (survivor Popen with NO WorkerIO — never another process's WorkerIO).
    assert backend._worker is proc
    assert backend._wio is None
    assert backend.is_ready() is False
    # No respawn over the live survivor.
    with backend._worker_lock:
        backend._ensure_worker()
    assert len(records) == 1


def test_stale_request_failure_does_not_clear_newer_ownership(fake_proc_env):
    records, _ = fake_proc_env
    backend = make_backend()
    old_proc = FakePopen(["old"])
    new_proc = FakePopen(["new"])
    records.extend([old_proc, new_proc])
    old_wio = WorkerIO(old_proc, concurrency=1)
    new_wio = WorkerIO(new_proc, concurrency=1)

    # A newer worker was installed by another thread after the old caller
    # captured its pair.
    backend._worker = new_proc
    backend._wio = new_wio

    # The OLD request's failure cleanup must NOT discard OR mark the newer pair
    # (even though old_proc is dead here) — ownership comparison fails first.
    old_proc._exit(-9)
    backend._clear_worker_if_current(old_proc, old_wio, reason="stale failure")
    assert backend._worker is new_proc
    assert backend._wio is new_wio
    assert backend._worker_failed is False


def test_clear_worker_matching_dead_worker_clears_and_resets(fake_proc_env):
    records, _ = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["x"])
    records.append(proc)
    wio = WorkerIO(proc, concurrency=1)
    backend._worker = proc
    backend._wio = wio
    backend._worker_failed = True
    backend._worker_failed_reason = "earlier"

    # PROVEN exited matching pair → clear Popen + its own WIO, reset marker.
    proc._exit(-9)
    backend._clear_worker_if_current(proc, wio, reason="died")
    assert backend._worker is None
    assert backend._wio is None
    assert backend._worker_failed is False
    assert backend._worker_failed_reason is None


def test_clear_worker_matching_live_worker_retained_and_unusable(fake_proc_env):
    records, _ = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["x"])  # NOT exited → proven live
    records.append(proc)
    wio = WorkerIO(proc, concurrency=1)
    backend._worker = proc
    backend._wio = wio
    backend._ready = True

    # A broken pipe / no response is NOT proof of exit: keep the live Popen AND
    # its WorkerIO reader ownership, and mark unusable / not-ready.
    backend._clear_worker_if_current(proc, wio, reason="stdin broken")
    assert backend._worker is proc          # live process retained
    assert backend._wio is wio              # reader ownership retained
    assert backend._worker_failed is True
    assert backend._worker_failed_reason == "stdin broken"
    assert backend._ready is False

    # Subsequent requests reject explicitly and do NOT respawn over the live
    # process (worker still alive → _ensure_worker early-returns).
    with pytest.raises(asr_mod.WorkerExitError, match="unavailable"):
        backend._worker_request({"event": "chunk", "id": "s1"})
    assert len(records) == 1
    assert not proc.kill_called


def test_ready_rejection_survivor_request_rejects_explicitly(fake_proc_env):
    records, queue_launch = fake_proc_env
    backend = make_backend(require_ifb=True, max_slots=2)
    queue_launch([ready_line()], exit_on_stdin_close=False,
                 terminate_exits=False)
    with pytest.raises(RuntimeError, match="IFB provenance rejected"):
        backend._ensure_worker()
    assert backend._worker is records[0]  # failed-ready survivor referenced
    assert backend._wio is None
    assert backend._worker_failed is True
    backend._ready = True
    # Later request must reject EXPLICITLY (not AssertionError) and must not
    # respawn a second worker over the live survivor.
    with pytest.raises(asr_mod.WorkerExitError, match="unavailable"):
        backend._worker_request({"event": "chunk", "id": "s1"})
    assert len(records) == 1
    assert not records[0].kill_called


def test_blocked_wrapper_close_no_raw_fd_close_and_eof_not_claimed():
    """Adapted from the erroneous v1 raw-fd-fallback test.

    v2: the wrapper is the ONLY closer. When its close() blocks, the EOF
    ATTEMPT is reported NOT completed (return False) and NO raw os.close(fd) is
    performed — the write fd must still be open (`fcntl` F_GETFD succeeds) until
    the wrapper close finally runs and closes it itself.
    """
    import fcntl

    backend = make_backend()
    read_fd, write_fd = os.pipe()
    release = threading.Event()

    class BlockingStdin:
        closed = False

        def fileno(self):
            return write_fd

        def close(self):
            # Blocked flush behind a full pipe / locked writer: does not finish
            # until the test releases it, mirroring a writer that unblocks only
            # when the child drains the pipe.
            release.wait(10.0)
            os.close(write_fd)   # the WRAPPER owns and performs the real close
            self.closed = True

    worker = SimpleNamespace(stdin=BlockingStdin())
    # While the real fd write end is still open, a read would block (no data).
    os.set_blocking(read_fd, False)
    t0 = time.monotonic()
    ok = backend._close_stdin_eof(worker, budget_s=0.2)
    elapsed = time.monotonic() - t0
    assert ok is False, "blocked close must report EOF attempt NOT completed"
    assert elapsed < 2.0, "bounded; did not wait for the blocked close()"
    # No raw fd close happened: the write fd is still valid/open.
    fcntl.fcntl(write_fd, fcntl.F_GETFD)
    with pytest.raises(BlockingIOError):
        os.read(read_fd, 1)  # no EOF, no data yet

    # Release the wrapper close; it (not the backend) performs the real close.
    release.set()
    deadline = time.monotonic() + 2.0
    got_eof = False
    while time.monotonic() < deadline:
        try:
            if os.read(read_fd, 1) == b"":
                got_eof = True
                break
        except BlockingIOError:
            time.sleep(0.02)
    assert got_eof  # wrapper close delivered EOF (backend never raw-closed fd)
    os.close(read_fd)


def test_repeated_survivor_restart_no_duplicate_close_thread(fake_proc_env):
    """Repeated teardown attempts against the SAME stdin must never start a
    second close thread (no duplicate close ownership). Exercised at the
    teardown helper level so the invariant is checked without a slow survivor
    deadline, then confirmed through a restart that the survivor is retained."""
    records, _ = fake_proc_env
    backend = make_backend()

    release = threading.Event()

    class BlockingStdin:
        closed = False

        def fileno(self):
            return -1  # never used in v2 (no raw fd path)

        def close(self):
            release.wait(10.0)
            self.closed = True

    proc = FakePopen(["survivor"], exit_on_stdin_close=False,
                     terminate_exits=False)
    proc.stdin = BlockingStdin()
    records.append(proc)
    backend._worker = proc
    backend._wio = None
    backend._ready = True

    import unittest.mock as _mock
    started: list[int] = []
    real_thread = threading.Thread

    def counting_thread(*args, **kwargs):
        if kwargs.get("name") == "asr-owned-stdin-close":
            started.append(1)
        return real_thread(*args, **kwargs)

    with _mock.patch.object(threading, "Thread", counting_thread):
        # Two bounded EOF attempts against the SAME stdin: the second must reuse
        # the pending close, not start another thread.
        assert backend._close_stdin_eof(proc, budget_s=0.2) is False
        assert backend._close_stdin_eof(proc, budget_s=0.2) is False
        assert sum(started) == 1
        # A restart attempt must also not start a second close thread.
        backend._shutdown_owned_worker = (  # type: ignore[assignment]
            lambda worker, **kw: False
        )
        with pytest.raises(RuntimeError, match="survived"):
            backend.restart_worker()
        assert sum(started) == 1
    release.set()
    assert backend._worker is proc
    assert backend._wio is None
    assert backend._worker_failed is True


def test_restart_lock_acquisition_timeout_bounded(fake_proc_env):
    """Both lock acquisitions are bounded by the shared deadline; a wedged
    holder yields a bounded explicit error without touching ownership."""
    records, _ = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["x"], exit_on_stdin_close=False, terminate_exits=True)
    records.append(proc)
    backend._worker = proc
    backend._ready = True

    # Hold the restart lock in another thread.
    holder_ready = threading.Event()
    release_holder = threading.Event()

    def hold_restart():
        backend._restart_lock.acquire()
        holder_ready.set()
        release_holder.wait(20.0)
        backend._restart_lock.release()

    threading.Thread(target=hold_restart, daemon=True).start()
    assert holder_ready.wait(5.0)

    import unittest.mock as _mock
    real_monotonic = asr_mod.time.monotonic
    calls = {"n": 0}

    def fast_monotonic():
        # Call 1 sets the 15 s deadline; every later call jumps past it so the
        # restart-lock acquire(timeout=0) on the HELD lock fails immediately.
        calls["n"] += 1
        return real_monotonic() + (0.0 if calls["n"] == 1 else 10_000.0)

    with _mock.patch.object(asr_mod.time, "monotonic", fast_monotonic):
        t0 = real_monotonic()
        with pytest.raises(RuntimeError, match="could not acquire"):
            backend.restart_worker()
        elapsed = real_monotonic() - t0
    release_holder.set()
    assert elapsed < 5.0
    # Ownership untouched on lock-acquisition failure.
    assert backend._worker is proc
    assert backend._worker_failed is False


def test_restart_worker_lock_acquisition_timeout_bounded(fake_proc_env):
    records, _ = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["x"], exit_on_stdin_close=False, terminate_exits=True)
    records.append(proc)
    backend._worker = proc
    backend._ready = True
    # Hold the worker lock; the restart lock is free, so `restart_worker`
    # acquires it then must time out bounded on `_worker_lock`, leaving
    # ownership unchanged.
    assert backend._worker_lock.acquire()
    try:
        import unittest.mock as _mock
        real_monotonic = asr_mod.time.monotonic
        calls = {"n": 0}

        def fast_monotonic():
            calls["n"] += 1
            return real_monotonic() + (0.0 if calls["n"] == 1 else 10_000.0)

        with _mock.patch.object(asr_mod.time, "monotonic", fast_monotonic):
            t0 = real_monotonic()
            with pytest.raises(RuntimeError, match="could not acquire"):
                backend.restart_worker()
            elapsed = real_monotonic() - t0
    finally:
        backend._worker_lock.release()
    assert elapsed < 5.0
    assert backend._worker is proc
    assert backend._worker_failed is False


def test_expired_budget_no_minimum_wait_or_threads(fake_proc_env):
    """An already-expired absolute deadline returns the CURRENT poll result
    immediately: no minimum wait, no threads, no signal."""
    records, _ = fake_proc_env
    backend = make_backend()
    alive = FakePopen(["alive"], exit_on_stdin_close=False,
                      terminate_exits=False)
    dead = FakePopen(["dead"])
    records.extend([alive, dead])
    dead._exit(0)

    import unittest.mock as _mock
    threads_started: list[int] = []
    real_thread = threading.Thread

    def counting_thread(*args, **kwargs):
        threads_started.append(1)
        return real_thread(*args, **kwargs)

    with _mock.patch.object(threading, "Thread", counting_thread):
        t0 = time.monotonic()
        assert backend._shutdown_owned_worker(
            alive, deadline_monotonic=time.monotonic() - 1.0
        ) is False
        assert backend._shutdown_owned_worker(
            dead, deadline_monotonic=time.monotonic() - 1.0
        ) is True
        elapsed = time.monotonic() - t0
    assert elapsed < 0.5                       # no minimum wait
    assert threads_started == []               # no drain/close threads
    assert alive.terminate_called == 0         # no signal
    assert not alive.kill_called


def test_close_stdin_eof_expired_budget_adds_no_thread():
    backend = make_backend()

    class Stdin:
        closed = False

        def close(self):  # pragma: no cover - must never be called
            raise AssertionError("close must not run on expired budget")

    worker = SimpleNamespace(stdin=Stdin())
    import unittest.mock as _mock
    started: list[int] = []
    real_thread = threading.Thread

    def counting_thread(*args, **kwargs):
        started.append(1)
        return real_thread(*args, **kwargs)

    with _mock.patch.object(threading, "Thread", counting_thread):
        assert backend._close_stdin_eof(worker, budget_s=0.0) is False
    assert started == []


def test_restart_shared_deadline_covers_wio_close_and_teardown(fake_proc_env):
    """The 1 s bounded WorkerIO.close() wait must draw from the SAME absolute
    budget as the owned teardown — a blocked wio close cannot add a second
    budget, and the teardown receives the absolute deadline."""
    records, _ = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["x"], exit_on_stdin_close=False, terminate_exits=True)
    records.append(proc)
    backend._worker = proc
    backend._ready = True

    release_close = threading.Event()

    class BlockingWio:
        def close(self):
            release_close.wait(10.0)

    backend._wio = BlockingWio()

    observed: list[dict] = []

    def fake_shutdown(worker, *, deadline_monotonic=None, deadline_s=15.0,
                      stdout_has_reader=False):
        observed.append({"deadline_monotonic": deadline_monotonic})
        worker.terminate()
        return True

    backend._shutdown_owned_worker = fake_shutdown  # type: ignore[assignment]
    t0 = time.monotonic()
    backend.restart_worker()
    elapsed = time.monotonic() - t0
    release_close.set()
    # wio.close() blocked for its bounded slice of the ONE shared budget, then
    # the teardown ran from the remainder — total stays under the 15 s budget
    # (it is ~1 s here), never 1 s + a second full budget.
    assert elapsed < 15.0
    assert observed and observed[0]["deadline_monotonic"] is not None
    assert backend._worker is None
    assert not proc.kill_called
    assert backend._worker_failed is False



# ── 14. survivor reader-ownership repair (v3) ───────────────────────────────


def _counting_thread_patch(names):
    """Patch threading.Thread to count constructions by thread name."""
    import unittest.mock as _mock

    started: list[str] = []
    real_thread = threading.Thread

    def counting_thread(*args, **kwargs):
        name = kwargs.get("name")
        if name in names:
            started.append(name)
        return real_thread(*args, **kwargs)

    return _mock.patch.object(threading, "Thread", counting_thread), started


def test_restart_survivor_retains_active_wio_two_restarts_one_stdout_reader(
    fake_proc_env,
):
    """Active WIO survivor: TWO real failed restart calls must keep the SAME
    WorkerIO retained with the survivor Popen and must NEVER create a teardown
    stdout reader (the WIO reader still owns stdout until EOF). Also exercises
    the actual orchestration: the pending WIO close is REUSED, so only ONE
    asr-restart-wio-close thread ever exists for the same WIO object."""
    records, _ = fake_proc_env
    backend = make_backend()
    proc = FakePopen(["old"], exit_on_stdin_close=False, terminate_exits=False)
    records.append(proc)
    backend._worker = proc
    backend._ready = True

    release = threading.Event()

    class BlockingWio:
        def close(self):
            release.wait(30.0)

    wio = BlockingWio()
    backend._wio = wio

    patch, started = _counting_thread_patch(
        {"asr-worker-teardown-drain-stdout", "asr-restart-wio-close"}
    )
    with patch:
        with pytest.raises(RuntimeError, match="survived"):
            backend.restart_worker()
        assert backend._worker is proc
        assert backend._wio is wio          # ACTUAL WIO retained, not dropped
        assert backend._worker_failed is True
        assert started.count("asr-restart-wio-close") == 1
        assert started.count("asr-worker-teardown-drain-stdout") == 0
        # SECOND real failed restart: wio still retained -> stdout_has_reader
        # stays True (no second stdout reader) and the pending WIO close is
        # reused (no second close thread for the same WIO).
        with pytest.raises(RuntimeError, match="survived"):
            backend.restart_worker()
    assert backend._worker is proc
    assert backend._wio is wio
    assert backend._worker_failed is True
    assert started.count("asr-restart-wio-close") == 1
    assert started.count("asr-worker-teardown-drain-stdout") == 0
    assert not proc.kill_called
    assert proc.terminate_called == 2  # one owned TERM per failed restart
    release.set()


def test_ready_rejection_drainer_later_restart_no_second_stdout_reader(
    monkeypatch,
):
    """Failed-ready drainer: the ready-rejection teardown owns stdout without a
    WIO; a later restart/teardown of the SAME stream must not start a second
    reader (stream-bound ownership marker).

    Fixture isolation: a REAL live survivor's stdout has no data and BLOCKS —
    it does not report EOF after an idle timeout (the generic FakeOutPipe's
    5 s empty-queue EOF would let the drainer finish before the assertion).
    This stream stays blocked until the test explicitly releases it as EOF,
    and the reader thread is joined in a ``finally`` so no drainer lingers."""
    import unittest.mock as _mock

    records: list[FakePopen] = []
    pending: list[tuple[list, dict]] = []
    release = threading.Event()
    drain_threads: list[threading.Thread] = []
    real_thread = threading.Thread

    class BlockedOutPipe(FakeOutPipe):
        """Blocked-forever stdout: lines only when fed; EOF only on release."""

        def readline(self) -> str:
            while True:
                try:
                    return self.q.get(timeout=0.1)
                except queue.Empty:
                    if release.is_set():
                        return ""  # controlled explicit EOF
                    continue

        def __iter__(self):
            while True:
                line = self.readline()
                if line == "":
                    return
                yield line

    def recording_thread(*args, **kwargs):
        t = real_thread(*args, **kwargs)
        if kwargs.get("name") == "asr-worker-teardown-drain-stdout":
            drain_threads.append(t)
        return t

    def make(*args, **kwargs):
        if pending:
            lines, extra = pending.pop(0)
            kwargs = {**extra, **kwargs}
            kwargs["stdout_lines"] = lines
        proc = FakePopen(*args, **kwargs)
        proc.stdout = BlockedOutPipe()
        for line in kwargs.pop("stdout_lines", ()):
            proc.stdout.feed(line)
        records.append(proc)
        return proc

    def queue_launch(stdout_lines=(), **proc_kwargs):
        pending.append((list(stdout_lines), proc_kwargs))

    monkeypatch.setattr(asr_mod, "subprocess", SimpleNamespace(
        Popen=make,
        TimeoutExpired=subprocess.TimeoutExpired,
        PIPE=subprocess.PIPE,
    ))

    with _mock.patch.object(threading, "Thread", recording_thread):
        backend = make_backend(require_ifb=True, max_slots=2)
        queue_launch([ready_line()], exit_on_stdin_close=False,
                     terminate_exits=False)
        try:
            with pytest.raises(RuntimeError, match="IFB provenance rejected"):
                backend._ensure_worker()
            proc = records[0]
            assert backend._wio is None
            assert backend._worker is proc  # live survivor, drainer blocked

            # Exactly ONE reader was started and it still owns the ACTUAL
            # stream (the blocked drainer has not finished).
            assert len(drain_threads) == 1
            assert backend._stdout_drain_pending.get(
                id(proc.stdout)) is proc.stdout

            # Repeated teardowns of the SAME stream must reuse the pending
            # drain ownership: no second stdout reader ever starts.
            backend._shutdown_owned_worker(proc, deadline_s=0.5,
                                           stdout_has_reader=False)
            backend._shutdown_owned_worker(proc, deadline_s=0.5,
                                           stdout_has_reader=False)
            assert len(drain_threads) == 1
            assert backend._stdout_drain_pending.get(
                id(proc.stdout)) is proc.stdout
        finally:
            # Controlled release + bounded join: the ONE reader finishes and
            # no reader thread lingers past the test.
            release.set()
            for t in drain_threads:
                t.join(5.0)
            for t in drain_threads:
                assert not t.is_alive()

    # Releasing the stream (EOF) removed the pending ownership.
    assert backend._stdout_drain_pending == {}
    assert not proc.kill_called


def test_wio_close_pending_reused_and_raise_never_claims_eof():
    """The pending WorkerIO.close operation is bound to the ACTUAL WIO object:
    a repeated call reuses it (ONE close thread). If close() RAISES, the thread
    ending is NOT proof of a delivered EOF — the actual failed state is
    returned (False), never a claimed success."""
    backend = make_backend()
    release = threading.Event()

    class BlockingWio:
        def close(self):
            release.wait(10.0)

    patch, started = _counting_thread_patch({"asr-restart-wio-close"})
    blocking = BlockingWio()
    with patch:
        assert backend._wio_close_bounded(blocking, budget_s=0.2) is False
        assert backend._wio_close_bounded(blocking, budget_s=0.2) is False
    assert started.count("asr-restart-wio-close") == 1
    release.set()

    class RaisingWio:
        def close(self):
            raise RuntimeError("close exploded")

    raising = RaisingWio()
    assert backend._wio_close_bounded(raising, budget_s=1.0) is False
    assert backend._wio_close_bounded(raising, budget_s=1.0) is False


def test_eof_wait_expiry_no_term_after_deadline(fake_proc_env):
    """If the EOF wait returns at/after the shared deadline, OWN TERM must NOT
    be sent after expiry: report current liveness, keep the Popen owned
    fail-closed, start no further helpers."""
    records, _ = fake_proc_env
    backend = make_backend()

    class SlowWaitPopen(FakePopen):
        def wait(self, timeout=None):
            # Overshoots the shared deadline regardless of the offered
            # timeout, modelling an EOF wait that returns after expiry.
            time.sleep(2.0)
            if not self._exited:
                raise subprocess.TimeoutExpired(self.args, timeout)

    proc = SlowWaitPopen(["x"], exit_on_stdin_close=False,
                         terminate_exits=True)
    records.append(proc)
    t0 = time.monotonic()
    exited = backend._shutdown_owned_worker(proc, deadline_s=1.5)
    elapsed = time.monotonic() - t0
    assert exited is False
    assert proc.terminate_called == 0  # NO TERM after the deadline expired
    assert not proc.kill_called
    assert 1.5 <= elapsed <= 4.0, elapsed


# ── 15. v2: actual stdin result + shared close outcome + shared deadline ────


def test_stdin_close_exception_not_eof_and_actual_close_eof_true():
    """Actual stdin wrapper result: close() RAISING with ``closed`` still False
    must return EOF=False (thread completion alone NEVER proves EOF); an
    actual successful wrapper close that sets ``closed`` True returns True."""
    backend = make_backend()

    class RaisingStdin:
        closed = False

        def close(self):
            raise RuntimeError("wrapper close exploded")

    worker = SimpleNamespace(stdin=RaisingStdin())
    assert backend._close_stdin_eof(worker, budget_s=1.0) is False

    class GoodStdin:
        closed = False

        def close(self):
            self.closed = True

    worker2 = SimpleNamespace(stdin=GoodStdin())
    assert backend._close_stdin_eof(worker2, budget_s=1.0) is True


def test_delayed_wio_close_raise_shares_failure_outcome_no_second_active_close():
    """A WorkerIO.close() that RAISES only AFTER an earlier caller's wait
    already timed out must expose the SAME failure outcome to a reused caller:
    the outcome belongs to the SHARED operation, and no second active close
    thread is started for the same WIO."""
    backend = make_backend()
    release = threading.Event()
    close_calls = {"n": 0}

    class DelayedRaisingWio:
        def close(self):
            release.wait(10.0)
            close_calls["n"] += 1
            raise RuntimeError("delayed close failure")

    wio = DelayedRaisingWio()
    patch, started = _counting_thread_patch({"asr-restart-wio-close"})
    with patch:
        # First caller times out while the close is still pending.
        assert backend._wio_close_bounded(wio, budget_s=0.2) is False
        assert close_calls["n"] == 0  # still blocked, nothing raised yet
        release.set()
        time.sleep(0.3)  # let the close thread finish WITH a raise
        assert close_calls["n"] == 1
        # Reused caller must consult the SAME operation outcome -> failure.
        assert backend._wio_close_bounded(wio, budget_s=2.0) is False
    assert started.count("asr-restart-wio-close") == 1  # ONE close thread total


def test_absolute_deadline_spent_no_floor_and_no_expired_helper_threads():
    """Helpers must be capped by the SHARED absolute deadline (allowed end =
    min(entry + budget_s, shared deadline)) with no floor: a partly-spent
    absolute budget yields only the remaining slice. An ALREADY-expired
    deadline starts no thread and leaves no pending marker."""
    backend = make_backend()

    # Partly spent: close blocks far longer than the remaining shared slice;
    # the call must return at the shared end, not after its own 10 s budget.
    # The block is CONTROLLED (an Event, not a blind fixed sleep): the earlier
    # helper stays pending-OWNED until the test releases it and its completion
    # is observed with bounded synchronization at the end of the test.
    release = threading.Event()

    class SlowStdin:
        closed = False

        def close(self):
            release.wait(10.0)
            self.closed = True

    slow_stdin = SlowStdin()
    deadline = time.monotonic() + 0.4
    t0 = time.monotonic()
    assert backend._close_stdin_eof(
        SimpleNamespace(stdin=slow_stdin),
        budget_s=10.0,
        deadline_monotonic=deadline,
    ) is False
    elapsed = time.monotonic() - t0
    assert elapsed <= 2.0, elapsed  # no floor past the shared deadline slice
    # The blocked helper legitimately REMAINS owned (its close is still
    # pending); capture the shared done Event for the bounded completion check
    # below, BEFORE running the expired-deadline section.
    slow_key = id(slow_stdin)
    assert set(backend._stdin_close_pending) == {slow_key}
    slow_done = backend._stdin_close_pending[slow_key][1]

    patch, started = _counting_thread_patch(
        {
            "asr-restart-wio-close",
            "asr-owned-stdin-close",
            "asr-worker-teardown-drain-stdout",
        }
    )
    with patch:
        expired = time.monotonic() - 0.01

        class NeverWio:
            def close(self):  # pragma: no cover - must never run
                raise AssertionError("close must not run on expired deadline")

        class NeverStdin:
            closed = False

            def close(self):  # pragma: no cover - must never run
                raise AssertionError("close must not run on expired deadline")

        # Already-expired shared deadline: no thread, no wait, no phantom
        # pending marker.
        assert backend._wio_close_bounded(
            NeverWio(), budget_s=1.0, deadline_monotonic=expired
        ) is False
        assert backend._close_stdin_eof(
            SimpleNamespace(stdin=NeverStdin()),
            budget_s=1.0,
            deadline_monotonic=expired,
        ) is False
        proc = FakePopen(["x"], exit_on_stdin_close=False, terminate_exits=False)
        backend._start_stdout_drain_once(proc, deadline_monotonic=expired)
        # The ALREADY-pending earlier helper remains OWNED throughout: the
        # expired calls above must not evict it nor add any new marker.
        assert set(backend._stdin_close_pending) == {slow_key}
        assert backend._stdin_close_pending[slow_key][0] is slow_stdin
    assert started == []  # expiry adds NO new thread for ANY helper
    assert backend._wio_close_pending == {}
    assert backend._stdout_drain_pending == {}
    # Release the earlier helper and observe its completion with bounded
    # synchronization: ownership is removed and the wrapper reports closed.
    release.set()
    assert slow_done.wait(5.0)
    assert slow_stdin.closed is True
    assert backend._stdin_close_pending == {}
