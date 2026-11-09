"""Regression tests for the shared automatic audio-device selector.

These tests deliberately exercise ``AudioIO`` directly rather than a single
product app. Every app inheriting ``BaseApp`` uses the same selector and
watcher, so a reSpeaker/XVF3800 USB replug must recover in multi_mode and in
the other shared audio apps as well.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ovs_agent import app_base
from ovs_agent.audio import devices
from ovs_agent.audio_io import AudioDeviceUnavailable, AudioIO


class _FakeStream:
    def __init__(self, *, device, callback=None, **kwargs):
        self.device = device
        self.callback = callback
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class _FakeSD:
    def __init__(self):
        self.default = SimpleNamespace(device=(0, 0))
        self.input_devices: dict[str, dict] = {}
        self.output_devices: dict[str, dict] = {}
        self.input_opens: list[str | int | None] = []
        self.output_opens: list[str | int | None] = []
        self.terminate_count = 0
        self.initialize_count = 0

    def query_devices(self, device=None, kind=None):
        if device is None:
            if kind == "input":
                return next(iter(self.input_devices.values()))
            if kind == "output":
                return next(iter(self.output_devices.values()))
            return [*self.input_devices.values(), *self.output_devices.values()]
        if isinstance(device, str):
            pool = self.input_devices if kind == "input" else self.output_devices
            for item in pool.values():
                if device.lower() in item["name"].lower():
                    return item
            raise RuntimeError(f"device not found: {device}")
        return next(iter(self.input_devices.values()))

    def RawInputStream(self, **kwargs):
        self.input_opens.append(kwargs["device"])
        return _FakeStream(**kwargs)

    def RawOutputStream(self, **kwargs):
        self.output_opens.append(kwargs["device"])
        return _FakeStream(**kwargs)

    def _terminate(self):
        self.terminate_count += 1
        return None

    def _initialize(self):
        self.initialize_count += 1
        return None


def _fake_respeaker_sd() -> _FakeSD:
    sd = _FakeSD()
    sd.input_devices["xvf"] = {
        "name": "reSpeaker Flex XVF3800 C16K6Ch",
        "max_input_channels": 6,
        "max_output_channels": 0,
        "default_samplerate": 16000,
    }
    sd.output_devices["xvf"] = {
        "name": "reSpeaker Flex XVF3800 C16K6Ch",
        "max_input_channels": 6,
        "max_output_channels": 2,
        "default_samplerate": 24000,
    }
    return sd


def _wire_fake_audio(monkeypatch, fake_sd, resolver):
    import ovs_agent.audio_io as audio_io_module
    import ovs_agent.audio.profiles as profiles

    monkeypatch.setattr(audio_io_module, "sd", fake_sd)
    monkeypatch.setattr(profiles, "sd", fake_sd, raising=False)
    monkeypatch.setattr(devices, "resolve_input_index", resolver)
    monkeypatch.setattr(devices, "resolve_output_index", resolver)
    monkeypatch.setattr(
        profiles,
        "_load_yaml_profiles",
        lambda _config_dir: None,
    )


@pytest.mark.asyncio
async def test_auto_selector_waits_at_boot_then_connects_when_respeaker_is_inserted(
    monkeypatch,
):
    fake_sd = _FakeSD()
    connected = False

    def resolve(value, *, wait_s=None, require_device=False):
        return "XVF3800" if connected else None

    _wire_fake_audio(monkeypatch, fake_sd, resolve)
    audio = AudioIO(
        input_device="auto",
        output_device="auto",
        mic_channels=1,
        mic_channel_select=0,
        mic_channels_cfg="auto",
        mic_channel_select_cfg="auto",
    )
    audio._input_callback = lambda *args: None
    audio._input_capture_active = True
    audio._input_reconnect_min_s = 0.001
    audio._input_reconnect_max_s = 0.01
    audio._input_reconnect_backoff_s = 0.001

    with pytest.raises(AudioDeviceUnavailable):
        audio._open_input_stream()
    assert audio._input_stream is None

    connected = True
    fake_sd.input_devices.update(_fake_respeaker_sd().input_devices)
    fake_sd.output_devices.update(_fake_respeaker_sd().output_devices)
    task = asyncio.create_task(audio._watch_devices())
    try:
        for _ in range(50):
            if audio._input_stream is not None:
                break
            await asyncio.sleep(0.002)
        assert audio._input_stream is not None
        assert fake_sd.input_opens[-1] == "XVF3800"
        assert audio._mic_channels == 6
        assert audio._mic_channel_select == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_open_fallback_switches_when_only_linux_topology_changes(
    monkeypatch,
):
    """A live fallback stream must not hide a later Linux USB hot-plug.

    Keep the fake PortAudio enumeration unchanged for the whole test.  Only
    the Linux-native signature and resolver result change, reproducing the
    field failure where PortAudio cached its boot-time topology.
    """
    fake_sd = _FakeSD()
    fake_sd.input_devices["fallback"] = {
        "name": "Generic USB Audio",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 16000,
    }
    fake_sd.output_devices["fallback"] = {
        "name": "Generic USB Audio",
        "max_input_channels": 1,
        "max_output_channels": 2,
        "default_samplerate": 24000,
    }
    state = {"device": "Generic USB Audio", "native": ("fallback",)}

    def resolve(value, *, wait_s=None, require_device=False):
        return state["device"]

    _wire_fake_audio(monkeypatch, fake_sd, resolve)
    monkeypatch.setattr(
        devices, "linux_audio_topology_signature", lambda: state["native"]
    )
    audio = AudioIO(
        input_device="auto",
        output_device="auto",
        mic_channels=1,
        mic_channel_select=0,
        mic_channels_cfg=1,
        mic_channel_select_cfg=0,
    )
    audio._input_callback = lambda *args: None
    audio._input_capture_active = True
    audio._device_watch_interval_s = 0.001
    audio._open_input_stream()
    first = audio._input_stream
    audio._ensure_output()
    first_output = audio._output_stream
    assert first is not None
    assert first_output is not None
    assert fake_sd.input_opens == ["Generic USB Audio"]
    audio._device_signature = audio._compute_device_signature()

    # sysfs/dev now see the reSpeaker, while PortAudio's cached query_devices()
    # result intentionally stays unchanged.
    state["device"] = "XVF3800"
    state["native"] = ("fallback", "XVF3800")
    task = asyncio.create_task(audio._watch_devices())
    try:
        for _ in range(50):
            if fake_sd.input_opens[-1] == "XVF3800":
                break
            await asyncio.sleep(0.002)
        assert first.closed is True
        assert fake_sd.input_opens == ["Generic USB Audio", "XVF3800"]
        assert fake_sd.terminate_count == 1
        assert fake_sd.initialize_count == 1
        assert audio._input_stream is not None
        # Topology resets must close stale playback too.  Output remains lazy
        # and will resolve the new device on the next TTS chunk.
        assert first_output.closed is True
        assert audio._output_stream is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_auto_selector_unplug_replug_reopens_and_does_not_pin_card_index(
    monkeypatch,
):
    fake_sd = _fake_respeaker_sd()
    state = {"device": "XVF3800"}

    def resolve(value, *, wait_s=None, require_device=False):
        return state["device"]

    _wire_fake_audio(monkeypatch, fake_sd, resolve)
    audio = AudioIO(
        input_device="auto",
        output_device="auto",
        mic_channels=1,
        mic_channel_select=0,
        mic_channels_cfg="auto",
        mic_channel_select_cfg="auto",
    )
    audio._input_callback = lambda *args: None
    audio._input_capture_active = True
    audio._open_input_stream()
    first = audio._input_stream
    assert first is not None

    # The physical card disappears and later reappears at a different ALSA /
    # PortAudio index. The selector returns its stable product identity again;
    # no integer index is cached in AudioIO.
    state["device"] = None

    def resolve_after_unplug(value, *, wait_s=None, require_device=False):
        return state["device"]

    monkeypatch.setattr(devices, "resolve_input_index", resolve_after_unplug)
    await audio._reopen_streams()
    assert audio._input_stream is None

    state["device"] = "XVF3800"
    fake_sd.input_devices["xvf"]["name"] = "reSpeaker XVF3800 4-Mic Array"
    fake_sd.input_devices["xvf"]["max_input_channels"] = 2
    audio._open_input_stream()
    assert audio._input_stream is not None
    assert fake_sd.input_opens == ["XVF3800", "XVF3800"]
    assert audio._mic_channels == 2


@pytest.mark.asyncio
async def test_auto_output_unplug_replug_resolves_and_reopens_output_stream(
    monkeypatch,
):
    """Output auto-selection must recover too; it must not pin an ALSA index.

    A USB AEC array exposes both a capture and playback endpoint.  The
    watcher may reopen either side independently after a physical replug, so
    exercise the output path directly instead of relying on input recovery to
    cover it accidentally.
    """
    fake_sd = _fake_respeaker_sd()
    state = {"device": "XVF3800"}

    def resolve(value, *, wait_s=None, require_device=False):
        return state["device"]

    _wire_fake_audio(monkeypatch, fake_sd, resolve)
    audio = AudioIO(
        input_device="auto",
        output_device="auto",
        mic_channels=1,
        mic_channel_select=0,
        mic_channels_cfg="auto",
        mic_channel_select_cfg="auto",
    )
    audio._ensure_output()
    first = audio._output_stream
    assert first is not None
    assert fake_sd.output_opens == ["XVF3800"]

    state["device"] = None

    def resolve_after_unplug(value, *, wait_s=None, require_device=False):
        return state["device"]

    monkeypatch.setattr(devices, "resolve_output_index", resolve_after_unplug)
    await audio._reopen_streams()
    assert audio._output_stream is None

    state["device"] = "XVF3800"
    fake_sd.output_devices["xvf"]["name"] = "reSpeaker XVF3800 Playback"
    audio._ensure_output()
    assert audio._output_stream is not None
    assert fake_sd.output_opens == ["XVF3800", "XVF3800"]


def test_base_app_keeps_auto_selector_for_all_shared_apps(monkeypatch):
    """BaseApp, not a robot-specific subclass, owns automatic selection."""
    calls = []

    class FakeAudioIO:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(app_base.BaseApp, "AUDIO_IO_CLASS", FakeAudioIO)
    monkeypatch.setattr(
        app_base,
        "resolve_mic_setup",
        lambda *args, **kwargs: SimpleNamespace(
            channels=1,
            channel_select=0,
            makeup_gain=None,
            profile_name="unknown",
        ),
    )

    from ovs_agent.config import Config

    cfg = Config(slv_url="ws://localhost:1/ws")
    app_base.BaseApp(cfg)
    assert cfg.audio_input_device == "auto"
    assert cfg.audio_output_device == "auto"
    assert calls[0]["input_device"] == "auto"
    assert calls[0]["output_device"] == "auto"
    assert calls[0]["mic_channels_cfg"] == "auto"


def test_resolver_returns_same_product_token_after_alsa_card_index_drift(monkeypatch):
    cards = [
        (3, "Array", "USB-Audio - reSpeaker XVF3800 4-Mic Array"),
    ]
    monkeypatch.setattr(devices, "_read_sound_cards", lambda: cards)
    monkeypatch.setattr(devices.time, "sleep", lambda _seconds: None)
    first = devices.resolve_input_index("auto", wait_s=0.0, require_device=True)

    cards[:] = [
        (19, "Array", "USB-Audio - reSpeaker XVF3800 4-Mic Array"),
    ]
    second = devices.resolve_input_index("auto", wait_s=0.0, require_device=True)
    assert first == second == "XVF3800"


def test_auto_input_resolves_48k_native_sample_rate_and_resamples(monkeypatch):
    """Devices like XVF3800 only support 48000 Hz. AudioIO must resolve 48000 Hz
    and resample audio chunks to 16000 Hz for downstream KWS / ASR."""
    import numpy as np

    fake_sd = _fake_respeaker_sd()
    fake_sd.input_devices["xvf"]["default_samplerate"] = 48000

    def fake_check_input_settings(device=None, samplerate=None, channels=None, dtype=None):
        if samplerate != 48000:
            raise RuntimeError(f"Invalid sample rate: {samplerate}")

    fake_sd.check_input_settings = fake_check_input_settings

    _wire_fake_audio(monkeypatch, fake_sd, lambda *args, **kwargs: "XVF3800")
    audio = AudioIO(
        input_device="auto",
        input_sr=16000,
        mic_channels=2,
        mic_channel_select=0,
    )
    resolved_sr = audio._resolve_input_sample_rate("XVF3800", requested_sr=16000)
    assert resolved_sr == 48000

    # Test callback downmixing and downsampling 48k -> 16k
    captured_mono = []
    audio._actual_input_sr = 48000
    audio._input_callback = lambda pcm, *args: captured_mono.append(pcm)

    cb = audio._make_input_stream_callback()
    # 4800 frames of stereo (2 ch) at 48kHz = 100ms
    test_pcm = (np.ones((4800, 2), dtype=np.int16) * 1234).tobytes()
    cb(test_pcm, 4800, None, 0)

    assert len(captured_mono) == 1
    # After downsample to 16kHz, length should be 1600 frames * 2 bytes = 3200 bytes
    assert len(captured_mono[0]) == 3200
    arr = np.frombuffer(captured_mono[0], dtype=np.int16)
    assert len(arr) == 1600
    assert arr[0] == 1234

