"""SLV link watchdog: the assistant must heal itself after a speech restart.

Measured on 2026-09-13 (RK3588, real device): recreating the `speech` container
— which is exactly what an image update does — left the agent disconnected
forever. `SLVClient._open_with_retry` budgets only ~4 attempts
(`_RECONNECT_BACKOFFS = (0.25, 0.5, 1.0)` + one final try), which is right for an
interactive call like `wake()` but must not be the last word for the long-lived
conversation link. Two symptoms were reproduced twice:

  * the dispatch loop stopped retrying (no further log lines for minutes), and
  * the assistant stayed mute until an operator hit the dashboard's reconnect
    button.

`_slv_link_watchdog` therefore retries forever (grace window between attempts)
and restarts the dispatch task if it ever finishes.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ovs_agent.app_base import BaseApp


def _app(*, healthy: bool, reconnecting: bool = False, dispatch_done: bool = False,
         grace_s: float = 0.0, shutdown: bool = False):
    app = BaseApp.__new__(BaseApp)
    app.config = SimpleNamespace(slv_link_grace_s=grace_s)
    evt = asyncio.Event()
    if shutdown:
        evt.set()
    app._shutdown_evt = evt
    app.slv = MagicMock()
    app.slv.is_healthy = MagicMock(return_value=healthy)
    app.slv.is_reconnecting = MagicMock(return_value=reconnecting)
    app.slv.reconnect = AsyncMock()
    app.slv._closed = False
    task = MagicMock()
    task.done = MagicMock(return_value=dispatch_done)
    task.cancelled = MagicMock(return_value=False)
    task.exception = MagicMock(return_value=RuntimeError("reader died"))
    app._dispatch_task = task
    app._slv_unhealthy_since = None
    # Never start a real dispatch loop from the watchdog inside a test: it
    # would block on the mocked client's event iterator.
    app._slv_dispatch = AsyncMock(return_value=None)
    return app


async def _tick(app: BaseApp, *, n: int = 2) -> None:
    """Run the watchdog body for a few iterations without the 2 s sleeps."""
    task = asyncio.create_task(app._slv_link_watchdog())
    # The loop sleeps 2 s per iteration; give it enough wall clock for `n`.
    await asyncio.sleep(2.0 * n + 0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_forces_reconnect_when_link_stays_unhealthy():
    app = _app(healthy=False, grace_s=0.0)
    await _tick(app)
    assert app.slv.reconnect.await_count >= 1


@pytest.mark.asyncio
async def test_no_reconnect_while_healthy():
    app = _app(healthy=True)
    await _tick(app)
    app.slv.reconnect.assert_not_awaited()
    assert app._slv_unhealthy_since is None


@pytest.mark.asyncio
async def test_skips_while_a_reconnect_is_already_in_flight():
    app = _app(healthy=False, reconnecting=True, grace_s=0.0)
    await _tick(app)
    app.slv.reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_grace_window_defers_the_first_attempt():
    app = _app(healthy=False, grace_s=60.0)
    await _tick(app, n=2)
    app.slv.reconnect.assert_not_awaited()
    assert app._slv_unhealthy_since is not None


@pytest.mark.asyncio
async def test_restarts_a_dead_dispatch_task():
    app = _app(healthy=False, dispatch_done=True, grace_s=0.0)
    dead = app._dispatch_task
    await _tick(app, n=1)
    assert app._dispatch_task is not dead, "a dead dispatch task must be replaced"


@pytest.mark.asyncio
async def test_exits_after_shutdown_or_client_close():
    app = _app(healthy=False, shutdown=True)
    task = asyncio.create_task(app._slv_link_watchdog())
    await asyncio.sleep(2.3)
    assert task.done(), "watchdog must return once shutdown is signalled"

    app2 = _app(healthy=True)
    app2.slv._closed = True
    task2 = asyncio.create_task(app2._slv_link_watchdog())
    await asyncio.sleep(2.3)
    assert task2.done(), "watchdog must return once the client is closed"
