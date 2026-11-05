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


def test_rk3588_kokoro_dualbucket_profile_config():
    """Verify rk3588-kokoro-dualbucket profile schema, root paths, digests, threads, and default voices."""
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    profile_path = repo_root / "configs" / "profiles" / "rk3588-kokoro-dualbucket.json"
    assert profile_path.is_file(), f"missing profile: {profile_path}"

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["name"] == "rk3588-kokoro-dualbucket"
    assert profile["tts_backend"] == "rk.tts"
    assert profile["asr_backend"] is None
    assert profile["execution_policy"] == {"mode": "serialized", "shared_resource": "npu"}
    assert profile["tts_model_id"] == "kokoro-convonly-v1_0-rk3588"

    env = profile["env"]
    expected_sha = "cfc6aa8d485415884e4cf5eac47b0a786889ed02d9556a1051ed66685927083d"
    assert env["TTS_BACKEND"] == "kokoro_dualbucket"
    assert env["RK_PLATFORM"] == "rk3588"
    assert env["KOKORO_DUALBUCKET_ROOT"] == "/opt/kokoro-dualbucket/rk3588"
    assert env["KOKORO_CONVONLY_ROOT"] == "/opt/kokoro-dualbucket/rk3588"
    assert env["KOKORO_DUALBUCKET_MANIFEST_SHA256"] == expected_sha
    assert env["KOKORO_CONVONLY_MANIFEST_SHA256"] == expected_sha
    assert env["KOKORO_FRONTEND_INTRA_OP_THREADS"] == "4"
    assert env["KOKORO_FRONTEND_INTER_OP_THREADS"] == "1"
    assert env["KOKORO_DEFAULT_VOICE"] == "af_heart"
    assert env["KOKORO_VOICE"] == "af_heart"
    assert env["KOKORO_JA_DICDIR"] == "/opt/resources/ja/unidic-lite-1.0.8"


def test_rk3588_kokoro_dualbucket_compose_config():
    """Verify deploy/docker-compose.kokoro-dualbucket-rk3588.yml aligns with production requirements."""
    from pathlib import Path
    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    compose_path = repo_root / "deploy" / "docker-compose.kokoro-dualbucket-rk3588.yml"
    assert compose_path.is_file(), f"missing compose file: {compose_path}"

    compose_data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    speech = compose_data["services"]["speech"]

    env = speech["environment"]
    expected_sha = "cfc6aa8d485415884e4cf5eac47b0a786889ed02d9556a1051ed66685927083d"
    assert env["OVS_PROFILE"] == "rk3588-kokoro-dualbucket"
    assert env["TTS_BACKEND"] == "kokoro_dualbucket"
    assert env["KOKORO_DUALBUCKET_ROOT"] == "/opt/kokoro-dualbucket/rk3588"
    assert env["KOKORO_CONVONLY_ROOT"] == "/opt/kokoro-dualbucket/rk3588"
    assert env["KOKORO_DUALBUCKET_MANIFEST_SHA256"] == expected_sha
    assert env["KOKORO_CONVONLY_MANIFEST_SHA256"] == expected_sha
    assert env["KOKORO_FRONTEND_INTRA_OP_THREADS"] == "4"
    assert env["KOKORO_FRONTEND_INTER_OP_THREADS"] == "1"
    assert env["KOKORO_DEFAULT_VOICE"] == "af_heart"
    assert env["KOKORO_VOICE"] == "af_heart"

    volumes = speech["volumes"]
    assert any("/opt/kokoro-dualbucket" in v for v in volumes)
    assert any("unidic-lite" in v for v in volumes)

