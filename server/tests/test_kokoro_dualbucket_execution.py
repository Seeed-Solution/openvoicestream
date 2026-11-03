"""Unit tests for server-level Kokoro dual-bucket API execution and cancellation integration."""
import threading
import pytest
import anyio

from server.core.api_execution import (
    KOKORO_BACKEND_NAMES,
    _is_kokoro_convonly_cancelled,
    _execute_kokoro_call,
    _TransportDisconnected,
)
from rkvoice_stream.backends.tts.kokoro_convonly import ConvOnlyCancelled


def test_kokoro_backend_names_contains_dualbucket():
    """Verify KOKORO_BACKEND_NAMES contains all four Kokoro backend aliases."""
    expected = {"kokoro_convonly", "rk:kokoro_convonly", "kokoro_dualbucket", "rk:kokoro_dualbucket"}
    assert expected == set(KOKORO_BACKEND_NAMES)


def test_convonly_cancelled_matches_server_identity():
    """Verify that dual-bucket cancellation exception matches _is_kokoro_convonly_cancelled."""
    exc = ConvOnlyCancelled("render")
    assert _is_kokoro_convonly_cancelled(exc) is True

    # Subclasses must NOT match (exact class identity constraint)
    class SubclassCancelled(ConvOnlyCancelled):
        pass

    assert _is_kokoro_convonly_cancelled(SubclassCancelled("render")) is False

    # Foreign exception with same name must NOT match
    class FakeCancelled(RuntimeError):
        pass
    FakeCancelled.__name__ = "ConvOnlyCancelled"
    assert _is_kokoro_convonly_cancelled(FakeCancelled("render")) is False


@pytest.mark.anyio
async def test_server_execute_kokoro_call_disconnect_integration():
    """Verify integration with server.core.api_execution._execute_kokoro_call."""
    cancel_evt = threading.Event()
    entered_event = threading.Event()

    def slow_synthesis_call():
        entered_event.set()
        assert cancel_evt.wait(timeout=2.0)
        raise ConvOnlyCancelled("render")

    async def mock_disconnect():
        while not entered_event.is_set():
            await anyio.sleep(0.01)
        return {"type": "http.disconnect"}

    with pytest.raises(_TransportDisconnected):
        await _execute_kokoro_call(
            call=slow_synthesis_call,
            cancel_event=cancel_evt,
            disconnect_awaitable=mock_disconnect,
        )

    assert cancel_evt.is_set() is True
