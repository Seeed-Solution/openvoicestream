"""Regression tests for wake-loop self-healing.

Background: the openWakeWord listen loop could die (crash) or silently stall
(``tap.get()`` blocked) with no recovery — the agent would stop responding to
"Hi Jarvis" until a container restart. Hardening:

  * TappedAudioIO.stop_capture_tap — unregister a tap (no orphaned-queue leak
    when the loop re-acquires on restart).
  * OpenWakeWordSource: a supervisor restarts _run_once on crash (backoff);
    a heartbeat (_last_chunk_ts) + request_restart let an external watchdog
    recover a silent stall the supervisor can't see.
  * BaseApp._wake_watchdog: restart the wake source when its heartbeat is
    stale WHILE the mic is fresh (and past boot / mic-restart grace).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from ovs_agent.audio.tapped_audio_io import TappedAudioIO
from ovs_agent.wake_sources.openwakeword import OpenWakeWordSource


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())
    wrapper.__name__ = coro_fn.__name__
    return wrapper


# ── 1. stop_capture_tap unregisters (no leak) ─────────────────────────

@run_async
async def test_stop_capture_tap_unregisters():
    io = TappedAudioIO.__new__(TappedAudioIO)
    io._taps = []
    io._tap_stats = {}
    q = await io.start_capture_tap()
    assert q in io._taps
    # A fanned chunk reaches a registered tap.
    io._safe_put_to_taps = None  # guard: we only touch the tap list here
    io.stop_capture_tap(q)
    assert q not in io._taps
    io.stop_capture_tap(q)  # idempotent — no raise


# ── fakes for the wake source ─────────────────────────────────────────

class _FakeModel:
    """predict() returns a score dict. ``trigger`` flips it above threshold."""
    def __init__(self):
        self.trigger = False
    def predict(self, window):
        return {"hey_jarvis_v0.1": 0.99 if self.trigger else 0.01}


class _FakeAudio:
    """Minimal TappedAudioIO surface: a single tap queue + start/stop hooks.
    ``crash_after`` makes get() raise to exercise the supervisor."""
    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()
        self.tap_count = 0
        self.stopped_taps = 0
        self.crash_after = None
        self._got = 0

    async def start_capture_tap(self, maxsize: int = 32):
        self.tap_count += 1
        return self

    def stop_capture_tap(self, q):
        self.stopped_taps += 1

    async def get(self):  # the source awaits tap.get()
        self._got += 1
        if self.crash_after is not None and self._got > self.crash_after:
            raise RuntimeError("simulated tap failure")
        return await self.q.get()


class _FakeApp:
    def __init__(self):
        self.audio = _FakeAudio()
        self.woke = 0
    async def wake(self, source: str = ""):
        self.woke += 1


def _make_source(app):
    src = OpenWakeWordSource(app)
    src._model = _FakeModel()  # skip setup()/real model load
    return src


async def _feed(app, n=1):
    # 1280 int16 samples = one openWakeWord window
    chunk = (np.ones(1280, dtype=np.int16) * 100).tobytes()
    for _ in range(n):
        await app.audio.q.put(chunk)


# ── 2. heartbeat updates as chunks flow ───────────────────────────────

@run_async
async def test_heartbeat_updates_on_chunks():
    app = _FakeApp()
    src = _make_source(app)
    assert src.last_chunk_ts() is None  # no chunk yet → grace
    await src.start()
    await _feed(app, 2)
    await asyncio.sleep(0.7)  # let the 0.5s startup delay + processing run
    assert src.last_chunk_ts() is not None, "heartbeat never stamped"
    await src.stop()


# ── 3. supervisor restarts the loop on crash ──────────────────────────

@run_async
async def test_supervisor_restarts_on_crash():
    app = _FakeApp()
    src = _make_source(app)
    app.audio.crash_after = 1  # second get() raises → _run_once crashes
    await src.start()
    await _feed(app, 1)
    try:
        # Recovery budget: 0.5s startup + crash + 0.5s backoff + 0.5s re-startup.
        await asyncio.sleep(2.2)
        assert app.audio.tap_count >= 2, f"loop not restarted (taps={app.audio.tap_count})"
        assert app.audio.stopped_taps >= 1, "crashed run did not release its tap"
    finally:
        await src.stop()


# ── 4. request_restart kicks the loop (silent-stall recovery) ─────────

@run_async
async def test_request_restart_respawns_run():
    app = _FakeApp()
    src = _make_source(app)
    await src.start()
    await asyncio.sleep(0.7)  # loop running, blocked on get()
    taps_before = app.audio.tap_count
    src.request_restart()  # simulate the watchdog kick
    try:
        await asyncio.sleep(0.9)  # kick + 0.5s re-startup
        assert app.audio.tap_count > taps_before, "request_restart did not respawn the loop"
        assert app.audio.stopped_taps >= 1, "kicked run did not release its tap"
    finally:
        await src.stop()


# ── 5. wake never re-triggers after stop ──────────────────────────────

@run_async
async def test_stop_halts_supervisor():
    app = _FakeApp()
    src = _make_source(app)
    await src.start()
    await asyncio.sleep(0.7)
    await src.stop()
    taps_after_stop = app.audio.tap_count
    await asyncio.sleep(0.5)
    assert app.audio.tap_count == taps_after_stop, "supervisor respawned after stop()"


# ── 6. wake watchdog decision ─────────────────────────────────────────

class _FakeWakeSrc:
    def __init__(self, last):
        self._last = last
        self.restarts = 0
    def last_chunk_ts(self):
        return self._last
    def request_restart(self):
        self.restarts += 1


def _watchdog_app(wake_src, *, boot_ago, mic_ago, mic_restart_ago):
    from ovs_agent.app_base import BaseApp
    app = BaseApp.__new__(BaseApp)
    now = time.monotonic()
    app.plugins = [wake_src]
    app._boot_ts = now - boot_ago
    app._last_mic_chunk_ts = now - mic_ago
    app._mic_restart_ts = (now - mic_restart_ago) if mic_restart_ago is not None else 0.0
    app._wake_restart_lock = None
    app._shutdown_evt = None
    return app


@run_async
async def test_wake_watchdog_restarts_when_stale_and_mic_fresh():
    now = time.monotonic()
    ws = _FakeWakeSrc(last=now - 30)  # wake 30s stale
    app = _watchdog_app(ws, boot_ago=120, mic_ago=1, mic_restart_ago=120)
    task = asyncio.create_task(app._wake_watchdog())
    await asyncio.sleep(2.4)  # one watchdog tick
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert ws.restarts >= 1, "watchdog should restart a stale wake loop while mic is fresh"


@run_async
async def test_wake_watchdog_skips_during_grace_and_mic_stale():
    now = time.monotonic()
    # (a) boot grace: stale wake but within 15s of boot → skip
    ws_a = _FakeWakeSrc(last=now - 30)
    app_a = _watchdog_app(ws_a, boot_ago=5, mic_ago=1, mic_restart_ago=120)
    # (b) mic stale: wake stale but mic also stale → mic watchdog's job → skip
    ws_b = _FakeWakeSrc(last=now - 30)
    app_b = _watchdog_app(ws_b, boot_ago=120, mic_ago=20, mic_restart_ago=120)
    for app in (app_a, app_b):
        t = asyncio.create_task(app._wake_watchdog())
        await asyncio.sleep(2.4)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    assert ws_a.restarts == 0, "must not restart during boot grace"
    assert ws_b.restarts == 0, "must not restart when the mic itself is stale"
