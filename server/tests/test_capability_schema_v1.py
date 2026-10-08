"""Phase A structured capability contract tests."""

from __future__ import annotations

import importlib
import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from server.core.api_capabilities import (
    CapabilitiesResponse,
    CloneEnrollment,
    CloningCapabilities,
    build_capabilities,
)
from server.core.tts_backend import TTSCapability


class _Limiter:
    limit = 3
    active = 1
    available = 2


class _Backend:
    name = "fake-tts"
    model_id = "qwen3-tts-0.6b-base"
    sample_rate = 24_000
    capabilities = {
        TTSCapability.BASIC_TTS,
        TTSCapability.MULTI_LANGUAGE,
        TTSCapability.STREAMING,
        TTSCapability.VOICE_CLONE,
    }
    supports_voice_enrollment = False

    def is_ready(self):
        return True

    def concurrency_capability(self):
        from server.core.concurrency_capability import ConcurrencyCapability

        return ConcurrencyCapability(supports_parallel=True, max_concurrent=2)

    def rate_pitch_caps(self):
        # Native backend controls are absent; the service's DSP fallback is
        # still a supported public control.
        return False, False


def test_base_structured_contract_is_additive_and_concurrent_snapshot_is_complete():
    body = build_capabilities(
        tts_backend=_Backend(),
        tts_ready=True,
        asr_configured=False,
        limiter=_Limiter(),
    )
    assert body["object"] == "capabilities"
    assert body["schema_version"] == "1.0"
    # The pure builder has no knowledge of application route registration;
    # callers pass the live route-derived versions explicitly.
    assert body["api_versions"] == ["legacy", "v1"]
    assert body["asr"] == {}
    tts = body["tts"]
    base_voice = tts["voices"]["items"][0]
    assert base_voice["id"] == 0
    assert base_voice["type"] == "intrinsic"
    assert base_voice["label"] == "Default reference voice"
    assert base_voice["source"] == "model"
    assert base_voice["model_id"] == "qwen3-tts-0.6b-base"
    assert tts["voices"]["default"] == {
        "speaker_id": 0,
        "source": "backend_intrinsic",
    }
    assert tts["cloning"]["modes"] == ["embedding"]
    assert tts["controls"]["speed"]["supported"] is True
    assert tts["controls"]["speed"]["implementation"] == "dsp"
    assert tts["controls"]["pitch"]["supported"] is True
    assert tts["controls"]["pitch"]["implementation"] == "dsp"
    assert tts["concurrency"] == {
        "backend_max_concurrent": 2,
        "admission_limit": 3,
        "active": 1,
        "available": 2,
    }


def test_unready_configured_component_stays_structured_200_shape():
    body = build_capabilities(
        tts_backend=None,
        tts_ready=False,
        tts_configured=True,
        asr_configured=False,
        profile={"tts_model_id": "moss-tts-nano-v1"},
        limiter=None,
    )
    assert body["api_versions"] == ["legacy", "v1"]
    assert body["tts"]["ready"] is False
    assert body["tts"]["model_id"] == "moss-tts-nano-v1"
    assert body["tts"]["voices"]["default"]["speaker_id"] is None
    assert body["tts"]["failure_class"] == "backend_unavailable"


def test_asr_model_identity_is_discoverable_without_backend_model_property():
    class ASR:
        name = "trt_edgellm"
        sample_rate = 16_000
        capabilities = set()

        def is_ready(self):
            return True

    body = build_capabilities(
        asr_backend=ASR(),
        asr_ready=True,
        asr_configured=True,
        tts_configured=False,
        profile={"asr_backend": "jetson.trt_edge_llm", "asr_model_id": "qwen3-asr-0.6b"},
        limiter=None,
    )
    assert body["asr"]["model_id"] == "qwen3-asr"


def test_component_is_either_empty_or_fully_typed():
    with pytest.raises(ValidationError):
        CapabilitiesResponse.model_validate({
            "object": "capabilities",
            "schema_version": "1.0",
            "api_versions": ["legacy", "v1"],
            "tts": {"ready": True},
            "asr": {},
        })


def test_manager_state_is_non_secret_and_stale_backend_is_not_reused(monkeypatch):
    from server import main
    from server.core import tts_service

    class FailedManager:
        state = "failed"

        def is_ready(self):
            return False

    stale = object()
    monkeypatch.setattr(main, "_get_tts_manager", lambda: FailedManager())
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: stale)
    assert main._peek_tts_backend() is None

    body = build_capabilities(
        tts_backend=None,
        tts_ready=False,
        tts_configured=True,
        tts_manager_state="failed",
        profile={"tts_model_id": "moss-tts-nano-v1"},
        asr_configured=False,
        limiter=None,
    )
    assert body["tts"]["failure_class"] == "backend_manager_failed"

    class DrainingManager:
        state = "draining"

        def is_ready(self):
            return False

    monkeypatch.setattr(main, "_get_asr_manager", lambda: DrainingManager())
    monkeypatch.setattr(main, "_get_asr_backend", lambda: stale)
    asr_body = build_capabilities(
        asr_backend=None,
        asr_ready=False,
        asr_configured=True,
        asr_manager_state="draining",
        profile={"asr_model_id": "sherpa-onnx"},
        tts_configured=False,
        limiter=None,
    )
    assert asr_body["asr"]["failure_class"] == "backend_manager_draining"


def test_clone_modes_and_enrollment_methods_reject_unknown_enums():
    with pytest.raises(ValidationError):
        CloningCapabilities.model_validate(
            {
                "supported": True,
                "modes": ["not-a-clone-mode"],
                "enrollment": {"supported": False, "methods": []},
            }
        )
    with pytest.raises(ValidationError):
        CloneEnrollment.model_validate(
            {"supported": True, "methods": ["not-an-enrollment-method"]}
        )


def test_deployed_voxedge_wheel_tts_wrapper_applies_dsp_fallback():
    """The pinned voxedge wheel's public TTS wrapper owns non-native DSP."""
    wheel = Path(__file__).parents[2] / "deploy" / "wheels" / "voxedge-0.0.6a1-py3-none-any.whl"
    if not wheel.is_file():
        pytest.skip("deployed voxedge wheel is not present")
    sys.path.insert(0, str(wheel))
    try:
        base = importlib.import_module("voxedge.backends.base")
    except Exception as exc:  # pragma: no cover - device image dependency gate
        pytest.skip(f"deployed voxedge wheel unavailable: {exc}")

    class Stub(base.TTSBackend):
        name = "stub"
        model_id = "qwen3-tts-0.6b-base"
        capabilities = {base.TTSCapability.BASIC_TTS}
        sample_rate = 24_000

        def __init__(self):
            self.calls = []

        def is_ready(self):
            return True

        def preload(self):
            return None

        def _synthesize_impl(self, text, **kwargs):
            self.calls.append(kwargs)
            samples = (np.sin(np.linspace(0.0, 100.0, 4800)) * 12000).astype(np.int16)
            out = io.BytesIO()
            with wave.open(out, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24_000)
                wf.writeframes(samples.tobytes())
            return out.getvalue(), {"sample_rate": 24_000}

    backend = Stub()
    assert backend.rate_pitch_caps() == (False, False)
    identity, _ = backend.synthesize("hello")
    changed, _ = backend.synthesize("hello", speed=2.0, pitch_shift=3.0)
    assert identity != changed
    with wave.open(io.BytesIO(identity), "rb") as wf:
        identity_frames = wf.getnframes()
    with wave.open(io.BytesIO(changed), "rb") as wf:
        changed_frames = wf.getnframes()
    assert changed_frames != identity_frames
    assert backend.calls[0]["speed"] is None
    assert backend.calls[0]["pitch_shift"] is None
    assert backend.calls[1]["speed"] is None
    assert backend.calls[1]["pitch_shift"] is None


def _backend(model_id: str, capabilities: set[TTSCapability], *, native=(False, False)):
    class Fake:
        name = f"fake-{model_id}"
        sample_rate = 16_000
        supports_voice_enrollment = False

        def is_ready(self):
            return True

        def rate_pitch_caps(self):
            return native

        def concurrency_capability(self):
            from server.core.concurrency_capability import ConcurrencyCapability

            return ConcurrencyCapability(max_concurrent=1)

    obj = Fake()
    obj.model_id = model_id
    obj.capabilities = capabilities
    return obj


def test_customvoice_golden_has_canonical_default_and_no_clone():
    from server.core.tts_speakers import available_speakers

    body = build_capabilities(
        tts_backend=_backend(
            "qwen3-tts-customvoice",
            {TTSCapability.BASIC_TTS, TTSCapability.STREAMING},
        ),
        tts_ready=True,
        tts_configured=True,
        asr_configured=False,
        limiter=None,
    )
    tts = body["tts"]
    assert tts["voices"]["default"] == {"speaker_id": 3065, "source": "preset"}
    assert len(tts["voices"]["items"]) == len(available_speakers("qwen3-tts-customvoice")) == 9
    assert tts["cloning"] == {
        "supported": False,
        "modes": [],
        "enrollment": {"supported": False, "methods": []},
    }
    assert tts["controls"]["speed"] == {
        "supported": True,
        "min": 0.25,
        "max": 4.0,
        "unit": None,
        "implementation": "dsp",
    }


def test_moss_golden_is_reference_audio_not_embedding():
    body = build_capabilities(
        tts_backend=_backend(
            "moss-tts-nano-v1",
            {
                TTSCapability.BASIC_TTS,
                TTSCapability.MULTI_LANGUAGE,
                TTSCapability.STREAMING,
                TTSCapability.VOICE_CLONE,
            },
        ),
        tts_ready=True,
        tts_configured=True,
        asr_configured=False,
        limiter=None,
    )
    tts = body["tts"]
    assert tts["voices"]["default"]["speaker_id"] is None
    assert tts["cloning"]["supported"] is True
    assert tts["cloning"]["modes"] == ["reference_audio"]
    assert tts["controls"]["speed"]["implementation"] == "dsp"


def test_base_reference_mode_requires_backend_reference_readiness():
    class ReadyBase(_Backend):
        @property
        def supports_reference_audio_cloning(self):
            return True

    ready = build_capabilities(
        tts_backend=ReadyBase(), tts_ready=True, tts_configured=True,
        asr_configured=False, limiter=None,
    )
    assert ready["tts"]["cloning"]["modes"] == ["embedding", "reference_audio"]

    not_ready = build_capabilities(
        tts_backend=_Backend(), tts_ready=True, tts_configured=True,
        asr_configured=False, limiter=None,
    )
    assert not_ready["tts"]["cloning"]["modes"] == ["embedding"]


def test_spark_golden_merges_live_profiles(monkeypatch):
    from server.core import sparktts_voices

    monkeypatch.setattr(
        sparktts_voices,
        "list_voices",
        lambda: [{
            "voice_id": "clone:alice",
            "type": "voice_profile",
            "profile_type": "voice_profile",
            "model_id": "sparktts-0p5b",
            "compatible_models": ["sparktts-0p5b"],
            "sample_rate": 16_000,
            "source_meta": {"method": "profile"},
        }],
    )
    body = build_capabilities(
        tts_backend=_backend(
            "sparktts-0p5b",
            {
                TTSCapability.BASIC_TTS,
                TTSCapability.MULTI_LANGUAGE,
                TTSCapability.STREAMING,
                TTSCapability.VOICE_CLONE,
            },
            native=(True, True),
        ),
        tts_ready=True,
        tts_configured=True,
        asr_configured=False,
        limiter=None,
    )
    tts = body["tts"]
    live = next(v for v in tts["voices"]["items"] if v["id"] == "clone:alice")
    assert live["type"] == "voice_profile"
    assert live["source"] == "live"
    assert live["model_id"] == "sparktts-0p5b"
    assert live["compatible_models"] == ["sparktts-0p5b"]
    assert tts["controls"]["speed"]["implementation"] == "backend"
    assert tts["cloning"]["modes"] == ["voice_profile"]


def test_base_golden_merges_only_model_compatible_embedding_profiles(monkeypatch):
    from server.core import sparktts_voices

    calls = []

    def list_voices(*, model_id, compatible_model):
        calls.append((model_id, compatible_model))
        return [
            {
                "voice_id": "clone:base",
                "profile_type": "speaker_embedding",
                "model_id": "qwen3-tts-0.6b-base",
                "compatible_models": ["qwen3-tts-0.6b-base"],
                "source_meta": {"method": "onnx"},
            },
            {
                "voice_id": "clone:wrong-model",
                "profile_type": "speaker_embedding",
                "model_id": "qwen3-tts-customvoice",
                "compatible_models": ["qwen3-tts-customvoice"],
            },
        ]

    monkeypatch.setattr(sparktts_voices, "list_voices", list_voices)
    body = build_capabilities(
        tts_backend=_backend(
            "qwen3-tts-0.6b-base",
            {TTSCapability.BASIC_TTS, TTSCapability.VOICE_CLONE},
        ),
        tts_ready=True,
        tts_configured=True,
        asr_configured=False,
        limiter=None,
    )
    items = body["tts"]["voices"]["items"]
    assert calls == [("qwen3-tts-0.6b-base", "qwen3-tts-0.6b-base")]
    live = next(v for v in items if v["id"] == "clone:base")
    assert live["type"] == "embedding"
    assert live["source"] == "registered_embedding"
    assert live["profile_type"] == "speaker_embedding"
    assert all(v["id"] != "clone:wrong-model" for v in items)


def test_v1_capabilities_route_passes_live_backends_explicitly(monkeypatch):
    pytest.importorskip("prometheus_client")
    from fastapi.testclient import TestClient
    from server import main
    from server.core import profile_loader

    tts = _backend(
        "qwen3-tts-customvoice",
        {TTSCapability.BASIC_TTS, TTSCapability.STREAMING},
    )
    monkeypatch.setattr(main, "_get_tts_manager", lambda: None)
    monkeypatch.setattr(main, "_get_asr_manager", lambda: None)
    monkeypatch.setattr(main, "_peek_tts_backend", lambda: tts)
    monkeypatch.setattr(main, "_get_asr_backend", lambda: None)
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {"tts_backend": "fake", "tts_model_id": tts.model_id},
    )
    response = TestClient(main.app).get("/v1/capabilities")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["api_versions"] == ["legacy", "v1", "openai-audio"]
    assert body["tts"]["ready"] is True
    assert body["tts"]["model_id"] == tts.model_id
    assert body["asr"] == {}


def test_legacy_tts_capability_clone_flags_derive_from_structured_component(monkeypatch):
    """Legacy flat flags stay aligned with the v1 cloning contract."""
    pytest.importorskip("prometheus_client")
    from fastapi.testclient import TestClient
    from server import main
    from server.core import tts_service

    backend = _backend(
        "qwen3-tts-0.6b-base",
        {TTSCapability.BASIC_TTS, TTSCapability.VOICE_CLONE},
    )
    backend.supports_voice_enrollment = True
    monkeypatch.setattr(main, "_get_tts_manager", lambda: None)
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: backend)
    monkeypatch.setattr(tts_service, "backend_name", lambda: backend.name)
    monkeypatch.setattr(tts_service, "capabilities", lambda: backend.capabilities)
    monkeypatch.setattr(tts_service, "get_sample_rate", lambda: backend.sample_rate)

    response = TestClient(main.app).get("/tts/capabilities")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["supports_voice_cloning"] is True
    assert body["supports_voice_enrollment"] is True
