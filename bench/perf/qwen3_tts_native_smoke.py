#!/usr/bin/env python3
"""Reusable Qwen3-TTS native API/audio smoke client.

Plan mode is default and never imports the native extension.  The caller must
provide engine, checkpoint, extension/plugin paths and native SHA pins.  The
outer harness owns engine/checkpoint hashes, foreign-container snapshots,
actual EOS, GPU qualification, and performance qualification.  This script
checks only the native API/audio contract: positive codec-frame count, stream
drain through ``None``, finished/joined state, and non-empty PCM/WAV.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import signal
import stat
import subprocess
import sys
import threading
import tempfile
import time
import wave
from pathlib import Path

_UPGRADE_ENV = ("SLV_UPGRADE_ROW_ID", "SLV_UPGRADE_PHASE", "SLV_UPGRADE_VARIANT", "SLV_UPGRADE_CLOSURE_SHA256")


def _upgrade_mode(ns: argparse.Namespace) -> bool:
    return bool(ns.upgrade_result_json)


def validate_upgrade_mode(ns: argparse.Namespace, *, child: bool = False, fresh_outputs: bool = True) -> dict[str, str] | None:
    if not ns.upgrade_result_json:
        return None
    if ns.fake_native:
        raise ValueError("--upgrade-result-json cannot be combined with --fake-native")
    if ns.child:
        raise ValueError("--upgrade-result-json cannot be combined with --child")
    if ns.repeat != 1 or ns.requests_json:
        raise ValueError("upgrade result mode requires exactly one inline request")
    identity = {key: os.environ.get(key) for key in _UPGRADE_ENV}
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise ValueError("upgrade result mode requires SLV_UPGRADE_ROW_ID/PHASE/VARIANT/CLOSURE_SHA256")
    if identity["SLV_UPGRADE_PHASE"] != "tts.customvoice.b1" or identity["SLV_UPGRADE_ROW_ID"] != identity["SLV_UPGRADE_PHASE"] or identity["SLV_UPGRADE_VARIANT"] != "customvoice":
        raise ValueError("upgrade result mode only supports tts.customvoice.b1/customvoice")
    closure = identity["SLV_UPGRADE_CLOSURE_SHA256"]
    if len(closure) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in closure):
        raise ValueError("SLV_UPGRADE_CLOSURE_SHA256 must be 64 hexadecimal characters")
    if not isinstance(ns.speaker, str) or not ns.speaker.strip() or not isinstance(ns.language, str) or not ns.language.strip():
        raise ValueError("upgrade result mode requires non-empty named speaker and language")
    for label, value in (("output-wav", ns.output_wav), ("upgrade-result-json", ns.upgrade_result_json)):
        path = Path(value)
        if not path.is_absolute() or path.is_symlink() or (fresh_outputs and path.exists()):
            qualifier = "fresh " if fresh_outputs else ""
            raise ValueError(f"{label} must be an absolute {qualifier}non-symlink path")
    return {key: value for key, value in identity.items() if value is not None}



def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true", help="import native extension and run one guarded smoke")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--fake-native", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--fake-case", choices=("ok", "queued", "negative", "zero", "mismatch", "nonzero", "empty", "unfinished", "cancelled", "exception", "odd", "allzero", "log_flood"), default="ok", help=argparse.SUPPRESS)
    p.add_argument("--output-wav", default="/tmp/qwen3tts-unified-smoke.wav")
    p.add_argument("--text", default="Hello.")
    p.add_argument("--speaker", default="serena")
    p.add_argument("--language", default="", help="optional request language name (for example english or chinese)")
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--reap-timeout-s", type=float, default=8.0)
    p.add_argument("--max-log-bytes", type=int, default=1024 * 1024, help="per-child stdout/stderr capture bound (1..67108864 bytes)")
    p.add_argument("--repeat", type=int, default=1, help="number of independent requests (default: 1)")
    p.add_argument("--upgrade-result-json", help="write the upgrade CustomVoice result envelope")
    p.add_argument("--requests-json", help="JSON file containing a non-empty array of request strings")
    p.add_argument("--requests-json-sha256", help=argparse.SUPPRESS)
    p.add_argument("--talker", help="talker engine directory (required by --execute)")
    p.add_argument("--code-predictor", help="code-predictor engine directory (required by --execute)")
    p.add_argument("--code2wav", help="code2wav engine directory (required by --execute)")
    p.add_argument("--checkpoint", help="checkpoint directory (required by --execute)")
    p.add_argument("--tokenizer", default="", help="tokenizer directory, if required by the native runtime")
    p.add_argument("--extension", help="exact native extension path (required by --execute)")
    p.add_argument("--plugin", help="exact native plugin path (required by --execute)")
    p.add_argument("--expected-extension-sha256", help="64-hex SHA256 pin for --extension")
    p.add_argument("--expected-plugin-sha256", help="64-hex SHA256 pin for --plugin")
    return p


def config(ns: argparse.Namespace) -> dict[str, object]:
    texts, request_sha256 = request_texts(ns)
    return {
        "text": ns.text,
        "speaker": ns.speaker,
        "language": ns.language,
        "talker_engine_dir": ns.talker,
        "code_predictor_engine_dir": ns.code_predictor,
        "code2wav_engine_dir": ns.code2wav,
        "tokenizer_dir": ns.tokenizer,
        "checkpoint_dir": ns.checkpoint,
        "extension": ns.extension,
        "plugin": ns.plugin,
        "timeout_s": ns.timeout_s,
        "reap_timeout_s": ns.reap_timeout_s,
        "audio_format": "pcm16le-mono",
        "mode": "native-api-audio-smoke",
        "missing_cli_paths": [name for name in ("talker", "code_predictor", "code2wav", "checkpoint", "extension", "plugin") if not getattr(ns, name)],
        "expected_pins_supplied": bool(ns.expected_extension_sha256 and ns.expected_plugin_sha256),
        "request_count": len(texts),
        "requests_json_sha256": request_sha256,
        "parent_budget_s": parent_budget_s(ns, len(texts)),
        "max_log_bytes": ns.max_log_bytes,
        "p95_qualification": "observation_only (<20 samples)" if len(texts) < 20 else "eligible_for_observation",
    }


_MAX_REQUESTS = 32
_MAX_PARENT_BUDGET_S = 1800.0


def parent_budget_s(ns: argparse.Namespace, count: int) -> float:
    return float(ns.timeout_s) * count + float(ns.reap_timeout_s) + 15.0


def request_texts(ns: argparse.Namespace) -> tuple[list[str], str | None]:
    if ns.requests_json:
        raw = Path(ns.requests_json).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if ns.requests_json_sha256 and digest != ns.requests_json_sha256.lower():
            raise ValueError(f"requests-json SHA256 mismatch: {digest} != {ns.requests_json_sha256}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"requests-json is not valid JSON: {exc}") from exc
        if not isinstance(payload, list) or not payload or any(not isinstance(item, str) for item in payload):
            raise ValueError("requests-json must be a non-empty JSON array of strings")
        if ns.repeat != 1:
            raise ValueError("--repeat conflicts with --requests-json; leave --repeat at 1")
        texts = payload
    else:
        if isinstance(ns.repeat, bool) or not isinstance(ns.repeat, int) or ns.repeat <= 0:
            raise ValueError("--repeat must be a positive integer")
        texts = [ns.text] * ns.repeat
        digest = None
    if len(texts) > _MAX_REQUESTS:
        raise ValueError(f"request count exceeds maximum {_MAX_REQUESTS}")
    budget = parent_budget_s(ns, len(texts))
    if not math.isfinite(budget) or budget > _MAX_PARENT_BUDGET_S:
        raise ValueError(f"parent request budget {budget:.3f}s exceeds maximum {_MAX_PARENT_BUDGET_S:.3f}s")
    return texts, digest


def validate_paths(ns: argparse.Namespace) -> None:
    required = {
        "talker": ns.talker,
        "code_predictor": ns.code_predictor,
        "code2wav": ns.code2wav,
        "checkpoint": ns.checkpoint,
        "extension": ns.extension,
        "plugin": ns.plugin,
    }
    missing = {name: path for name, path in required.items() if not path or not Path(path).exists()}
    if missing:
        raise FileNotFoundError(json.dumps(missing, sort_keys=True))


def validate_timeouts(ns: argparse.Namespace) -> None:
    for name in ("timeout_s", "reap_timeout_s"):
        value = float(getattr(ns, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0")
    if isinstance(ns.max_log_bytes, bool) or not isinstance(ns.max_log_bytes, int) or not 0 < ns.max_log_bytes <= 64 * 1024 * 1024:
        raise ValueError("max-log-bytes must be a positive integer no greater than 67108864")


def validate_expected_pins(ns: argparse.Namespace) -> None:
    for name in ("expected_extension_sha256", "expected_plugin_sha256"):
        value = getattr(ns, name)
        if value is None or len(value) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
            raise ValueError(f"{name} must be exactly 64 hexadecimal characters for native execution")


def _tts_metadata_path(talker: str) -> Path:
    path = Path(talker)
    return path / "config.json"


def _named_input_metadata(ns: argparse.Namespace) -> tuple[dict[str, object], Path]:
    path = _tts_metadata_path(ns.talker)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"TTS metadata unavailable at {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"TTS metadata at {path} is not a JSON object")
    return metadata, path


def validate_named_inputs(ns: argparse.Namespace) -> None:
    """Reject unavailable named inputs before loading the native GPU backend."""
    metadata, path = _named_input_metadata(ns)
    speaker_ids = metadata.get("speaker_id")
    if not isinstance(speaker_ids, dict):
        raise RuntimeError(f"TTS metadata at {path} has no usable speaker_id mapping")
    if ns.speaker not in speaker_ids:
        raise ValueError(f"speaker {ns.speaker!r} absent from metadata; available={sorted(speaker_ids)!r}")
    if ns.language:
        language_ids = metadata.get("codec_language_id")
        if not isinstance(language_ids, dict):
            raise RuntimeError(f"TTS metadata at {path} has no usable codec_language_id mapping")
        if ns.language not in language_ids:
            raise ValueError(f"language {ns.language!r} absent from metadata; available={sorted(language_ids)!r}")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_extension(path: str):
    # The init symbol is PyInit__edgellm_runtime, so the import name must end
    # in _edgellm_runtime.  This bypasses the package loader's plugin hash
    # selection and pins the reviewed native core explicitly.
    spec = importlib.util.spec_from_file_location("_edgellm_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create extension spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_edgellm_runtime"] = module
    spec.loader.exec_module(module)
    return module


def set_plugin(plugin: str) -> None:
    os.environ["EDGELLM_PLUGIN_PATH"] = plugin


class _FakeChunk:
    def __init__(self, payload: bytes, codec_frames: int) -> None:
        self.pcm16 = payload
        self.num_frames = codec_frames


class _FakeChannel:
    case = "ok"

    def __init__(self) -> None:
        payload = b"\x01\x00" * 240
        if self.case == "allzero":
            payload = b"\x00" * 480
        elif self.case == "odd":
            payload = b"\x01\x00" * 240 + b"\x01"
        if self.case == "empty":
            self._chunks = []
        elif self.case == "queued":
            self._chunks = [_FakeChunk(payload, 12), _FakeChunk(payload, 3)]
        else:
            self._chunks = [_FakeChunk(payload, 12)]
        # queued intentionally starts finished while two chunks remain pending.
        self._finished = self.case in {"cancelled", "queued"}
        self._cancelled = self.case == "cancelled"

    def wait_pop(self, _timeout_ms: int):
        if self._chunks:
            return self._chunks.pop(0)
        if self.case != "unfinished":
            self._finished = True
        return None

    def is_finished(self) -> bool:
        return self._finished

    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        self._finished = True


class _FakeParams:
    speaker_name = ""
    language_name = ""


class _FakeRuntime:
    case = "ok"

    def __init__(self, *_args) -> None:
        self._names = ["serena", "vivian"]
        self.seen_languages: list[str] = []

    def get_speaker_names(self):
        return self._names

    def handle_request_tts(self, text, _params, channel) -> int:
        self.seen_languages.append(getattr(_params, "language_name", ""))
        if text == "__RAISE__":
            raise RuntimeError("fake native request exception")
        if self.case == "exception":
            raise RuntimeError("fake native request exception")
        return {
            "ok": 12,
            "queued": 15,
            "negative": -1,
            "zero": 0,
            "mismatch": 7,
            "nonzero": 7,
            "empty": 0,
            "unfinished": 12,
            "cancelled": 12,
            "exception": 12,
            "odd": 12,
            "allzero": 12,
            "log_flood": 12,
        }[self.case]


class _FakeNative:
    TTSRuntime = _FakeRuntime
    OmniAudioParams = _FakeParams
    AudioStreamChannel = _FakeChannel


def resource_snapshot() -> dict[str, int]:
    mem_available = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1]) * 1024
            break
    shm = os.statvfs("/dev/shm")
    root = os.statvfs("/")
    values = {
        "mem_available_bytes": mem_available,
        "shm_free_bytes": shm.f_bfree * shm.f_frsize,
        "root_physical_free_bytes": root.f_bfree * root.f_frsize,
    }
    return values


def resource_guard() -> dict[str, int]:
    values = resource_snapshot()
    failures = []
    if values["mem_available_bytes"] < 2 * 1024**3:
        failures.append("MemAvailable<2GiB")
    if values["shm_free_bytes"] < 1 * 1024**3:
        failures.append("shm<1GiB")
    if values["root_physical_free_bytes"] < 1 * 1024**3:
        failures.append("root_f_bfree<1GiB")
    if failures:
        raise RuntimeError(json.dumps({"resource_guard": failures, **values}, sort_keys=True))
    return values


def _request_output_path(path: str, index: int, count: int) -> str:
    if count == 1:
        return path
    target = Path(path)
    return str(target.with_name(f"{target.stem}-{index + 1:04d}{target.suffix or '.wav'}"))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _run_request(runtime, params_cls, channel_cls, ns: argparse.Namespace, text: str, index: int, output_wav: str, mode: str) -> dict[str, object]:
    params = params_cls()
    params.speaker_name = ns.speaker
    if ns.language:
        if not hasattr(params, "language_name"):
            raise RuntimeError("language is unsupported by OmniAudioParams binding")
        params.language_name = ns.language
    channel = channel_cls()
    result: dict[str, object] = {"mode": mode, "request_index": index, "text": text, "status": "error",
                                  "requested_language": ns.language, "binding_language_name": getattr(params, "language_name", ""),
                                  "codec_frames": 0, "pcm_frames": 0, "chunks": 0, "bytes": 0}
    error: list[BaseException] = []
    request_started = time.monotonic()
    result["send_mono"] = request_started
    first_chunk_at: float | None = None

    def invoke() -> None:
        try:
            result["request_frame_count"] = int(runtime.handle_request_tts(text, params, channel))
        except BaseException as exc:
            error.append(exc)

    worker = threading.Thread(target=invoke, name=f"tts-native-request-{index}")
    worker.start()
    deadline = time.monotonic() + ns.timeout_s
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        chunk = channel.wait_pop(100)
        if chunk is not None:
            if first_chunk_at is None:
                first_chunk_at = time.monotonic()
                result["first_chunk_mono"] = first_chunk_at
            pcm = bytes(chunk.pcm16)
            chunks.append(pcm)
            result["chunks"] = int(result["chunks"]) + 1
            result["codec_frames"] = int(result["codec_frames"]) + int(chunk.num_frames)
            result["pcm_frames"] = int(result["pcm_frames"]) + len(pcm) // 2
            result["bytes"] = int(result["bytes"]) + len(pcm)
        if chunk is None and channel.is_finished() and not worker.is_alive():
            result["terminal_mono"] = time.monotonic()
            break
    if worker.is_alive():
        channel.cancel()
        worker.join(ns.reap_timeout_s)
        result["cancel_requested"] = True
        result["worker_reaped"] = not worker.is_alive()
    else:
        worker.join()
        result["cancel_requested"] = False
        result["worker_reaped"] = not worker.is_alive()
    result["request_frame_count"] = result.get("request_frame_count", None)
    ended = time.monotonic()
    result["ttfa_ms"] = None if first_chunk_at is None else (first_chunk_at - request_started) * 1000.0
    result.setdefault("first_chunk_mono", None)
    result["terminal_mono"] = ended
    result["finished"] = bool(channel.is_finished())
    result["cancelled"] = bool(channel.is_cancelled())
    result["generation_ms"] = (ended - request_started) * 1000.0
    pcm = b"".join(chunks)
    if error:
        raise error[0]
    if worker.is_alive():
        raise TimeoutError(json.dumps(result, sort_keys=True))
    request_frame_count = result["request_frame_count"]
    if not isinstance(request_frame_count, int) or isinstance(request_frame_count, bool) or request_frame_count <= 0:
        raise RuntimeError(json.dumps({"error": "invalid request_frame_count", **result}, sort_keys=True))
    if request_frame_count != result["codec_frames"]:
        raise RuntimeError(json.dumps({"error": "request_frame_count mismatch", **result}, sort_keys=True))
    if not pcm:
        raise RuntimeError(json.dumps({"error": "empty PCM", **result}, sort_keys=True))
    if len(pcm) % 2:
        raise RuntimeError(json.dumps({"error": "odd PCM byte length", **result}, sort_keys=True))
    if not result["finished"] or result["cancelled"]:
        raise RuntimeError(json.dumps({"error": "unfinished or cancelled channel", **result}, sort_keys=True))
    nonzero = sum(1 for value in pcm if value)
    if nonzero <= 0:
        raise RuntimeError(json.dumps({"error": "all-zero PCM", **result}, sort_keys=True))
    peak = max((abs(int.from_bytes(pcm[pos:pos + 2], "little", signed=True))
                for pos in range(0, len(pcm) - 1, 2)), default=0)
    duration_s = int(result["pcm_frames"]) / 24000.0
    result["nonzero_bytes"] = nonzero
    result["peak_int16"] = peak
    result["pcm_sha256"] = hashlib.sha256(pcm).hexdigest()
    result["duration_s"] = duration_s
    result["rtf"] = (float(result["generation_ms"]) / 1000.0) / duration_s if duration_s else None
    with wave.open(output_wav, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    result["wav"] = output_wav
    result["status"] = "ok"
    del channel
    return result


def _native_observation_pass(observation: dict[str, object]) -> bool:
    timing = ("send_mono", "first_chunk_mono", "terminal_mono", "ttfa_ms", "generation_ms", "rtf")
    if any(type(observation.get(key)) not in (int, float) or isinstance(observation.get(key), bool)
           or not math.isfinite(float(observation[key])) or float(observation[key]) < 0 for key in timing):
        return False
    return (observation["send_mono"] <= observation["first_chunk_mono"] <= observation["terminal_mono"]
            and observation.get("mode") == "native_api_smoke" and observation.get("status") == "ok"
            and observation.get("finished") is True and observation.get("cancelled") is False
            and isinstance(observation.get("request_frame_count"), int)
            and observation.get("request_frame_count", 0) > 0
            and isinstance(observation.get("pcm_frames"), int) and observation.get("pcm_frames", 0) > 0
            and isinstance(observation.get("bytes"), int) and observation.get("bytes", 0) > 0
            and isinstance(observation.get("wav"), str))


def _write_upgrade_result(ns: argparse.Namespace, observation: dict[str, object], stdout_path: Path, stderr_path: Path) -> None:
    identity = validate_upgrade_mode(ns, fresh_outputs=False)
    assert identity is not None
    wav_path = Path(str(observation.get("wav", "")))
    if str(wav_path) != ns.output_wav or not _native_observation_pass(observation):
        raise RuntimeError("native observation did not satisfy allchecks")
    def regular_file(path: Path) -> bool:
        try:
            return stat.S_ISREG(os.stat(path, follow_symlinks=False).st_mode)
        except OSError:
            return False

    if not regular_file(wav_path) or not regular_file(stdout_path) or not regular_file(stderr_path):
        raise RuntimeError("native result artifacts must be regular files")
    try:
        wav_data = wav_path.read_bytes()
        with wave.open(str(wav_path), "rb") as stream:
            samples = stream.getnframes()
            sample_rate, channels, sample_width = stream.getframerate(), stream.getnchannels(), stream.getsampwidth()
            pcm = stream.readframes(samples)
    except (OSError, EOFError, wave.Error) as exc:
        raise RuntimeError(f"malformed WAV: {exc}") from exc
    counters = ("request_frame_count", "codec_frames", "pcm_frames", "bytes")
    if any(type(observation.get(key)) is not int or observation[key] <= 0 for key in counters):
        raise RuntimeError("native observation counters are invalid")
    if (observation["request_frame_count"] != observation["codec_frames"]
            or observation["pcm_frames"] != samples
            or observation["bytes"] != len(pcm)
            or observation.get("pcm_sha256") != hashlib.sha256(pcm).hexdigest()
            or len(pcm) == 0 or not any(pcm)):
        raise RuntimeError("native observation PCM does not match WAV")
    if sample_rate != 24000 or channels != 1 or sample_width != 2 or len(pcm) != samples * 2:
        raise RuntimeError("WAV must be 24kHz mono PCM16")
    if observation.get("worker_reaped") is not True or observation.get("finished") is not True or observation.get("cancelled") is not False:
        raise RuntimeError("native worker lifecycle is incomplete")
    timing = ("send_mono", "first_chunk_mono", "terminal_mono", "ttfa_ms", "generation_ms", "rtf")
    if any(type(observation.get(key)) not in (int, float) or isinstance(observation.get(key), bool)
           or not math.isfinite(float(observation[key])) for key in timing):
        raise RuntimeError("native observation timing is invalid")
    if not (observation["send_mono"] <= observation["first_chunk_mono"] <= observation["terminal_mono"]):
        raise RuntimeError("native observation monotonic timing is invalid")
    def close(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    expected_ttfa = (observation["first_chunk_mono"] - observation["send_mono"]) * 1000.0
    expected_generation = (observation["terminal_mono"] - observation["send_mono"]) * 1000.0
    expected_rtf = (observation["generation_ms"] / 1000.0) / (observation["pcm_frames"] / 24000.0)
    if not close(float(observation["ttfa_ms"]), expected_ttfa) or not close(float(observation["generation_ms"]), expected_generation) or not close(float(observation["rtf"]), expected_rtf):
        raise RuntimeError("native observation derived timing does not match")
    stdout_data, stderr_data = stdout_path.read_bytes(), stderr_path.read_bytes()
    stdout_sha, stderr_sha = hashlib.sha256(stdout_data).hexdigest(), hashlib.sha256(stderr_data).hexdigest()
    envelope = {
        "schema": 1, "row_id": identity["SLV_UPGRADE_ROW_ID"],
        "phase": identity["SLV_UPGRADE_PHASE"], "variant": identity["SLV_UPGRADE_VARIANT"],
        "closure_sha256": identity["SLV_UPGRADE_CLOSURE_SHA256"],
        "upgrade_env": identity, "status": "PASS", "functional_status": "PASS",
        "threshold_status": "OPEN", "qualification_status": "UNPROVEN",
        "request": {"id": "b1-0001", "text": observation.get("text"), "stream": True,
                    "speaker": ns.speaker, "language": ns.language},
        "wav": {"path": str(wav_path), "sample_rate": sample_rate, "channels": channels,
                "sample_width": sample_width, "samples": samples,
                "sha256": hashlib.sha256(wav_data).hexdigest(), "bytes": len(wav_data)},
        "send_mono": observation.get("send_mono"), "first_chunk_mono": observation.get("first_chunk_mono"),
        "terminal_mono": observation.get("terminal_mono"),
        "ttfa_s": float(observation["ttfa_ms"]) / 1000.0, "total_s": float(observation["generation_ms"]) / 1000.0,
        "rtf": observation.get("rtf"), "raw_stdout": str(stdout_path), "raw_stderr": str(stderr_path),
        "raw_stdout_sha256": stdout_sha, "raw_stderr_sha256": stderr_sha,
        "raw_stdout_size": len(stdout_data), "raw_stderr_size": len(stderr_data),
        "wav_sha256": hashlib.sha256(wav_data).hexdigest(), "wav_bytes": len(wav_data),
        "native_observation": observation,
    }
    target = Path(ns.upgrade_result_json)
    if target.exists() or target.is_symlink():
        raise FileExistsError(str(target))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(target), flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, sort_keys=True)
            handle.write("\n")
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _metrics_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if row.get("status") == "ok"]
    generation = [float(row["generation_ms"]) for row in successful if row.get("generation_ms") is not None]
    ttfa = [float(row["ttfa_ms"]) for row in successful if row.get("ttfa_ms") is not None]
    rtf = [float(row["rtf"]) for row in successful if row.get("rtf") is not None]
    return {
        "request_count": len(rows),
        "successful_requests": len(successful),
        "p50_ttfa_ms": _percentile(ttfa, 0.50), "p95_ttfa_ms": _percentile(ttfa, 0.95),
        "p50_generation_ms": _percentile(generation, 0.50), "p95_generation_ms": _percentile(generation, 0.95),
        "p50_rtf": _percentile(rtf, 0.50), "p95_rtf": _percentile(rtf, 0.95),
        "p95_qualification": "observation_only (<20 samples)" if len(successful) < 20 else "observation_eligible",
        "percentile_method": "linear_interpolation",
    }


def child(ns: argparse.Namespace) -> int:
    validate_upgrade_mode(ns, child=True)
    validate_timeouts(ns)
    texts, request_sha256 = request_texts(ns)
    if ns.fake_native and ns.fake_case == "log_flood":
        print("L" * (2 * 1024 * 1024), flush=True)
    if ns.fake_native:
        _FakeChannel.case = ns.fake_case
        _FakeRuntime.case = ns.fake_case
        native = _FakeNative
    else:
        validate_expected_pins(ns)
        validate_paths(ns)
        validate_named_inputs(ns)
        expected = {ns.extension: ns.expected_extension_sha256.lower(), ns.plugin: ns.expected_plugin_sha256.lower()}
        for path, want in expected.items():
            got = sha256_file(path)
            if got != want:
                raise RuntimeError(f"pinned hash mismatch {path}: {got} != {want}")
        set_plugin(ns.plugin)
        plugin_mode = ctypes.RTLD_GLOBAL | getattr(ctypes, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_NODELETE", 0)
        ctypes.CDLL(ns.plugin, mode=plugin_mode)
        native = load_extension(ns.extension)
    runtime_cls = native.TTSRuntime
    params_cls = native.OmniAudioParams
    channel_cls = native.AudioStreamChannel
    if ns.language:
        try:
            language_probe = params_cls()
            if not hasattr(language_probe, "language_name"):
                raise RuntimeError("language is unsupported by OmniAudioParams binding")
            language_probe.language_name = ns.language
        except BaseException as exc:
            raise RuntimeError(f"language is unsupported by OmniAudioParams binding: {exc}") from exc
    load_started = time.monotonic()
    runtime = runtime_cls(ns.talker, ns.code_predictor, ns.code2wav, ns.tokenizer, ns.checkpoint)
    load_ms = (time.monotonic() - load_started) * 1000.0
    names = list(runtime.get_speaker_names())
    if ns.speaker not in names:
        raise ValueError(f"speaker {ns.speaker!r} absent; available={names!r}")
    mode = "cpu_fake" if ns.fake_native else "native_api_smoke"
    results: list[dict[str, object]] = []
    failed = False
    for index, text in enumerate(texts):
        output_wav = _request_output_path(ns.output_wav, index, len(texts))
        try:
            result = _run_request(runtime, params_cls, channel_cls, ns, text, index, output_wav, mode)
        except BaseException as exc:
            result = {"mode": mode, "request_index": index, "text": text, "status": "error",
                      "error": f"{type(exc).__name__}: {exc}"}
            failed = True
        result["phase"] = "first_after_load" if index == 0 else "warm"
        results.append(result)
        if failed:
            break
    all_metrics = _metrics_summary(results)
    first = results[0] if results else None
    warm_rows = results[1:]
    summary = {**all_metrics, "scope": "all_requests", "requested_requests": len(texts),
               "first_request": None if first is None else {
                   "request_index": first.get("request_index"), "phase": first.get("phase"),
                   "status": first.get("status"), "ttfa_ms": first.get("ttfa_ms"),
                   "generation_ms": first.get("generation_ms"), "rtf": first.get("rtf")},
               "warm_summary": _metrics_summary(warm_rows)}
    output: dict[str, object] = {"mode": mode, "request_count": len(texts), "requests_json_sha256": request_sha256,
                                 "requested_language": ns.language, "max_log_bytes": ns.max_log_bytes, "load_ms": load_ms, "speaker_names": names,
                                 "requests": results, "summary": summary}
    if len(texts) == 1 and results:
        output.update(results[0])
        output["load_ms"] = load_ms
        output["request_count"] = 1
        output["requests"] = results
        output["summary"] = summary
    print(json.dumps(output, sort_keys=True))
    del runtime
    return 1 if failed or len(results) != len(texts) else 0

def _capture_pipe(stream, path: Path, state: dict[str, object], overlimit: threading.Event, reader_fault: threading.Event, limit: int) -> None:
    stored = 0
    total = 0
    state["max_log_bytes"] = limit
    try:
        with path.open("wb") as handle:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if stored < limit:
                    keep = chunk[: limit - stored]
                    handle.write(keep)
                    stored += len(keep)
                if total > limit:
                    state["overlimit"] = True
                    overlimit.set()
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        reader_fault.set()
    finally:
        state["bytes_read"] = total
        state["bytes_stored"] = stored


def _join_log_readers(threads: list[threading.Thread], states: dict[str, dict[str, object]]) -> dict[str, object]:
    for thread in threads:
        thread.join(2.0)
    return {
        name: {**state, "reader_alive": any(t.is_alive() for t in threads if t.name == f"log-reader-{name}")}
        for name, state in states.items()
    }


def parent(ns: argparse.Namespace) -> int:
    validate_upgrade_mode(ns)
    validate_timeouts(ns)
    guard = resource_guard()
    command = [sys.executable, str(Path(__file__).resolve()), "--child"]
    for key, value in vars(ns).items():
        if key in {"execute", "child", "upgrade_result_json"} or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    _, parent_request_sha256 = request_texts(ns)
    if parent_request_sha256:
        command.extend(["--requests-json-sha256", parent_request_sha256])
    env = os.environ.copy()
    if ns.plugin:
        env["EDGELLM_PLUGIN_PATH"] = ns.plugin
    log_dir = Path(tempfile.mkdtemp(prefix="slv-tts-native-child-", dir="/dev/shm"))
    stdout_path, stderr_path = log_dir / "stdout", log_dir / "stderr"
    states: dict[str, dict[str, object]] = {"stdout": {}, "stderr": {}}
    overlimit = threading.Event()
    reader_fault = threading.Event()
    proc = subprocess.Popen(command, env=env, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    readers: list[threading.Thread] = []
    identity: dict[str, int] | None = None
    termination: dict[str, object] | None = None
    monitor_error: str | None = None
    first_failed: dict[str, object] | None = None
    result_rc: int | None = None
    try:
        if proc.stdout is None or proc.stderr is None:
            raise RuntimeError("Popen did not provide stdout/stderr pipes")
        readers = [
            threading.Thread(target=_capture_pipe, name="log-reader-stdout", args=(proc.stdout, stdout_path, states["stdout"], overlimit, reader_fault, ns.max_log_bytes), daemon=True),
            threading.Thread(target=_capture_pipe, name="log-reader-stderr", args=(proc.stderr, stderr_path, states["stderr"], overlimit, reader_fault, ns.max_log_bytes), daemon=True),
        ]
        for reader in readers:
            reader.start()
        identity = _proc_identity(proc.pid)
        if identity is None or identity["pgid"] != proc.pid:
            termination = _terminate_unbound(proc, proc.pid, "child identity binding failed", ns.reap_timeout_s)
            result_rc = 125
        else:
            deadline = time.monotonic() + ns.timeout_s + ns.reap_timeout_s + 15.0
            while proc.poll() is None:
                snapshot = resource_snapshot()
                try:
                    resource_guard()
                except RuntimeError as error:
                    first_failed = {"reason": str(error), "snapshot": snapshot}
                    break
                if reader_fault.is_set():
                    first_failed = {"reason": "child log reader failed", "snapshot": snapshot}
                    break
                if overlimit.is_set():
                    first_failed = {"reason": "child log limit exceeded", "snapshot": snapshot}
                    break
                if time.monotonic() >= deadline:
                    first_failed = {"reason": "child absolute deadline exceeded", "snapshot": snapshot}
                    break
                time.sleep(0.2)
            if first_failed is not None:
                termination = _terminate_owned(proc, identity, str(first_failed["reason"]), first_failed, ns.reap_timeout_s)
                result_rc = 124 if termination["worker_reaped"] and termination["empty_process_group"] else 125
            else:
                proc.wait()
                result_rc = proc.returncode
    except BaseException as exc:
        monitor_error = f"{type(exc).__name__}: {exc}"
        try:
            if identity is not None and identity.get("pgid") == proc.pid:
                termination = _terminate_owned(proc, identity, "parent exception: " + monitor_error, {"error": monitor_error}, ns.reap_timeout_s)
            else:
                termination = _terminate_unbound(proc, proc.pid, "parent exception before identity binding: " + monitor_error, ns.reap_timeout_s)
        except BaseException as cleanup_exc:
            monitor_error += f"; cleanup {type(cleanup_exc).__name__}: {cleanup_exc}"
        result_rc = 125
    finally:
        try:
            logs = _join_log_readers(readers, states)
        except BaseException as cleanup_exc:
            logs = {name: {**state, "reader_join_error": f"{type(cleanup_exc).__name__}: {cleanup_exc}"} for name, state in states.items()}
            monitor_error = (monitor_error + "; " if monitor_error else "") + logs["stdout"].get("reader_join_error", "reader join failed")
    try:
        empty_group = _group_empty(identity["pgid"] if identity is not None else proc.pid)
    except BaseException as exc:
        empty_group = False
        monitor_error = (monitor_error + "; " if monitor_error else "") + f"group check {type(exc).__name__}: {exc}"
    reader_problem = reader_fault.is_set() or any("error" in state or state.get("reader_alive") for state in logs.values())
    if monitor_error or reader_problem or overlimit.is_set() or not empty_group or (termination is not None and termination.get("unresolved_handoff")):
        print(json.dumps({"child_logs": str(log_dir), "guard": guard, "identity": identity,
                          "identity_bound": identity is not None and identity.get("pgid") == proc.pid,
                          "child_reaped": proc.poll() is not None, "empty_process_group": empty_group,
                          "termination": termination, "monitor_error": monitor_error, "logs": logs}, sort_keys=True), file=sys.stderr)
        return 125
    stdout, stderr = _read_bounded(stdout_path, ns.max_log_bytes), _read_bounded(stderr_path, ns.max_log_bytes)
    if ns.upgrade_result_json and result_rc == 0:
        observations = []
        for line in stdout.splitlines():
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                observations.append(candidate)
        if not observations:
            raise RuntimeError("native child emitted no JSON observation")
        _write_upgrade_result(ns, observations[-1], stdout_path, stderr_path)
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)
    return int(result_rc if result_rc is not None else 125)

def _terminate_unbound(proc: subprocess.Popen, expected_pgid: int, reason: str, reap_timeout_s: float) -> dict[str, object]:
    term_sent = False
    if proc.poll() is None:
        proc.terminate()
        term_sent = True
    try:
        proc.wait(timeout=reap_timeout_s)
    except subprocess.TimeoutExpired:
        pass
    reaped = proc.poll() is not None
    empty_group = _group_empty(expected_pgid)
    return {"reason": reason, "term_sent": term_sent, "worker_reaped": reaped,
            "empty_process_group": empty_group, "identity_bound": False,
            "unresolved_handoff": not (reaped and empty_group), "expected_pgid": expected_pgid}

def _proc_identity(pid: int) -> dict[str, int] | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
        closing = raw.rfind(")")
        if closing < 0:
            return None
        fields = raw[closing + 2:].split()
        # fields[0]=state, [1]=ppid, [2]=pgrp, [19]=starttime.
        return {"pid": pid, "pgid": int(fields[2]), "starttime": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def _group_empty(pgid: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            identity = _proc_identity(int(entry.name))
            if identity is not None and identity["pgid"] == pgid:
                return False
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
    return True


def _terminate_owned(proc: subprocess.Popen, identity: dict[str, int], reason: str, snapshot: dict[str, object], reap_timeout_s: float) -> dict[str, object]:
    current = _proc_identity(identity["pid"])
    if current is None or current != identity:
        worker_reaped = proc.poll() is not None
        empty_process_group = _group_empty(identity["pgid"])
        return {"reason": reason, "term_sent": False, "worker_reaped": worker_reaped,
                "empty_process_group": empty_process_group, "identity_bound": False,
                "unresolved_handoff": not (worker_reaped and empty_process_group),
                "first_failed_snapshot": snapshot}
    os.killpg(identity["pgid"], signal.SIGTERM)
    deadline = time.monotonic() + reap_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None and _group_empty(identity["pgid"]):
            return {"reason": reason, "term_sent": True, "worker_reaped": True,
                    "empty_process_group": True, "identity_bound": True,
                    "unresolved_handoff": False, "first_failed_snapshot": snapshot}
        time.sleep(0.1)
    worker_reaped = proc.poll() is not None
    empty_process_group = _group_empty(identity["pgid"])
    return {"reason": reason, "term_sent": True, "worker_reaped": worker_reaped,
            "empty_process_group": empty_process_group, "identity_bound": True,
            "unresolved_handoff": not (worker_reaped and empty_process_group),
            "first_failed_snapshot": snapshot}


def _read_bounded(path: Path, limit: int = 1024 * 1024) -> str:
    data = path.read_bytes()
    if len(data) > limit:
        raise RuntimeError(f"log exceeds {limit} bytes: {path}")
    return data.decode("utf-8", errors="replace")


def main() -> int:
    ns = parser().parse_args()
    if ns.child:
        return child(ns)
    if not ns.execute:
        print(json.dumps({"execute_required": True, "plan": config(ns)}, sort_keys=True))
        return 0
    return parent(ns)


if __name__ == "__main__":
    raise SystemExit(main())
