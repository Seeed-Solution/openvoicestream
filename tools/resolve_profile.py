#!/usr/bin/env python3
"""Resolve (LANGUAGE, DEVICE) to exactly one OpenVoiceStream profile + env.

Reads configs/matrix/language_device.yaml. Prints a shell-sourceable env block
(or JSON) and exits:

    0  cell status is `measured`
    0  cell status is `untested` — a WARN goes to stderr, the env is still emitted
    2  cell status is `unsupported` — nothing is emitted; Chinese never falls
       back to Whisper
    3  unknown language or device
    4  the resolved profile has no JSON in configs/profiles/

This selects a profile. It does not touch inference code, and it never invents
a profile name: every `ovs_profile` in the matrix must exist on disk.

Every key it emits has a named consumer. Nothing else is emitted — a variable
no code reads is a language switch that silently does nothing:

    OVS_PROFILE               server/core/profile_loader.py:129 (_select_profile_ref)
    OVS_MAX_CONCURRENT_SESSIONS
                              server/core/session_limiter.py:108
    OFFLINE_ASR_LANGUAGE      server/core/voxedge_backend_config.py:268
                              → SherpaASRConfig.offline_language ("" = auto).
                              Deployment-level pin: SenseVoice binds language at
                              recognizer construction, not per stream.
    WHISPER_LANGUAGE          server/core/voxedge_backend_config.py:944
                              → WhisperASRConfig.language, a forced decoder
                              token. Emitted only when the resolved cell's ASR
                              family IS Whisper (see cell_uses_whisper_asr) and
                              the language is one of en/zh — the shipped
                              encoders carry no other vocab and
                              WhisperASRConfig.__post_init__ raises on one.
                              A Chinese cell therefore never carries a Whisper
                              language pin: no cell routes zh to Whisper, and
                              nothing in a Qwen3/sherpa deployment's env hints
                              that Whisper is an option.
    ASR_LANGUAGE / TTS_LANGUAGE
                              read by the agent config template
                              (`asr_language`/`tts_language` in
                              agent-config.yaml, expanded by
                              agent/ovs_agent/config.py:524 `_expand_env`) and
                              sent as the per-session v2v config, which is the
                              only language knob the RK and Jetson Qwen3-ASR
                              backends have (voxedge rk/asr.py:517,
                              jetson/trt_edge_llm_asr.py:730 take `language`
                              per call, defaulting to "auto"). Forcing them is
                              what keeps a Chinese deployment out of per-
                              utterance language ID.
    OVS_MATRIX_STATUS / OVS_MATRIX_DEVICE / OVS_LANGUAGE
                              informational; read back by compose templates and
                              by operators reading the resolved env file.

Deliberately NOT emitted:

    LANGUAGE                  POSIX already owns it as the locale fallback list
                              ("en_US:en"); writing "zh" there is a malformed
                              locale, and sourcing this file must not corrupt
                              the container's locale. Compose may still use a
                              host-side `LANGUAGE` as the operator's input —
                              that is a shell variable, not container env.
    OVS_AUDIO_INPUT_CHANNELS / OVS_AUDIO_MONO
                              no consumer exists and none is needed. The single
                              mono lane is enforced where the audio actually
                              is: the agent picks one channel
                              (`mic_channel_select` in audio_profiles.yaml,
                              agent/ovs_agent/audio/profiles.py) and the server
                              downmixes anything still multi-channel
                              (server/core/asr_segmenter.py:287-289).

Usage:
    resolve_profile.py --language zh --device rk3576
    resolve_profile.py --language en --device orin_nx --format json
    resolve_profile.py --write-env /opt/voice/resolved/ovs.env   # OVS_LANGUAGE/OVS_DEVICE from env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_UNSUPPORTED = 2
EXIT_UNKNOWN = 3
EXIT_MISSING_PROFILE = 4

# voxedge/backends/whisper/asr.py:_SUPPORTED — the shipped encoders carry an
# en or zh vocab and nothing else, and WhisperASRConfig.__post_init__ raises
# rather than silently transcribing in the wrong language.
WHISPER_SUPPORTED_LANGUAGES = frozenset({"en", "zh"})

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = REPO_ROOT / "configs" / "matrix" / "language_device.yaml"
DEFAULT_PROFILES_DIR = REPO_ROOT / "configs" / "profiles"


class ResolveError(Exception):
    """Resolution failed. `code` is the process exit code to use."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def load_matrix(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ResolveError(
            "PyYAML is required to read the language/device matrix "
            f"({path}); install pyyaml in this image.",
            EXIT_UNKNOWN,
        ) from exc
    if not path.is_file():
        raise ResolveError(f"matrix file not found: {path}", EXIT_UNKNOWN)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "cells" not in data:
        raise ResolveError(f"matrix file is not a language/device matrix: {path}", EXIT_UNKNOWN)
    return data


def normalise_language(raw: str) -> str:
    """Accept `zh`, `ZH`, `zh-CN`, `zh_CN`, ` zh ` as the same language."""
    return (raw or "").strip().replace("_", "-").lower().split("-", 1)[0]


def normalise_device(raw: str) -> str:
    return (raw or "").strip().replace("-", "_").lower()


def find_cell(matrix: dict[str, Any], language: str, device: str) -> dict[str, Any]:
    languages = matrix.get("languages") or {}
    devices = matrix.get("devices") or {}

    if language not in languages:
        raise ResolveError(
            f"unknown language {language!r}; matrix knows: {', '.join(sorted(languages))}",
            EXIT_UNKNOWN,
        )
    if device not in devices:
        raise ResolveError(
            f"unknown device {device!r}; matrix knows: {', '.join(sorted(devices))}",
            EXIT_UNKNOWN,
        )

    group = languages[language]["group"]
    for cell in matrix["cells"]:
        if cell.get("device") == device and cell.get("group") == group:
            return cell
    raise ResolveError(
        f"matrix has no cell for device={device} group={group} (language={language})",
        EXIT_UNKNOWN,
    )


def cell_uses_whisper_asr(cell: dict[str, Any]) -> bool:
    """True when this cell's ASR family is Whisper.

    Prefers the cell's own `asr:` metadata (backend + model, which every
    supported cell carries); falls back to the profile name for a cell that
    declares none. A cell with neither reads as not-Whisper, which is the safe
    direction: the only consequence is that WHISPER_LANGUAGE is not written,
    and the profile's own pinned value stands.
    """
    asr = cell.get("asr") or {}
    meta = " ".join(str(asr.get(key, "")) for key in ("backend", "model"))
    if meta.strip():
        return "whisper" in meta.lower()
    return "whisper" in str(cell.get("ovs_profile") or "").lower()


def build_env(
    matrix: dict[str, Any], cell: dict[str, Any], language: str, device: str
) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in (matrix.get("common_env") or {}).items():
        env[str(key)] = str(value)
    for key, value in (cell.get("env") or {}).items():
        env[str(key)] = str(value)
    env["OVS_PROFILE"] = str(cell["ovs_profile"])
    env["OVS_MATRIX_STATUS"] = str(cell["status"])
    env["OVS_MATRIX_DEVICE"] = device
    env["OVS_LANGUAGE"] = language

    # The per-session knob the agent sends over the v2v WS. This is the one that
    # reaches the Qwen3-ASR backends on RK and Jetson; leaving it at "auto"
    # would put a Chinese deployment back on per-utterance language ID.
    env["ASR_LANGUAGE"] = language
    env["TTS_LANGUAGE"] = language

    # Deployment-level pin for the sherpa offline recognizer (SenseVoice /
    # Paraformer), which binds its language at construction time.
    env["OFFLINE_ASR_LANGUAGE"] = language

    # Only when the resolved cell actually runs a Whisper ASR, and only for a
    # language Whisper can decode. No cell in the matrix routes Chinese to
    # Whisper, so this never writes a Chinese pin.
    if cell_uses_whisper_asr(cell) and language in WHISPER_SUPPORTED_LANGUAGES:
        env["WHISPER_LANGUAGE"] = language
    return env


def resolve(
    language: str,
    device: str,
    matrix_path: Path = DEFAULT_MATRIX,
    profiles_dir: Path | None = DEFAULT_PROFILES_DIR,
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    """Return (env, cell, warnings). Raises ResolveError on any refusal."""
    matrix = load_matrix(matrix_path)
    language = normalise_language(language)
    device = normalise_device(device)
    cell = find_cell(matrix, language, device)
    status = cell.get("status")

    if status == "unsupported":
        reason = " ".join(str(cell.get("reason") or "no reason recorded").split())
        raise ResolveError(
            f"{language} on {device} is not supported: {reason}",
            EXIT_UNSUPPORTED,
        )
    if status not in ("measured", "untested"):
        raise ResolveError(
            f"cell device={device} group={cell.get('group')} has unknown status {status!r}",
            EXIT_UNKNOWN,
        )

    profile = cell.get("ovs_profile")
    if not profile:
        raise ResolveError(
            f"cell device={device} group={cell.get('group')} is {status} but names no profile",
            EXIT_UNKNOWN,
        )
    if profiles_dir is not None and not (profiles_dir / f"{profile}.json").is_file():
        raise ResolveError(
            f"profile {profile!r} is not present in {profiles_dir}",
            EXIT_MISSING_PROFILE,
        )

    warnings: list[str] = []
    if status == "untested":
        warnings.append(
            f"WARN: {language} on {device} resolves to profile {profile}, whose end-to-end "
            "combination has no measured number yet (status: untested). Deploy is allowed; "
            "do not publish latency or accuracy figures for it."
        )
        planned = cell.get("planned_alternative")
        if planned:
            warnings.append(
                f"WARN: the pairing planned for this cell is {planned.get('description')}"
                f" — blocked on: {planned.get('blocked_on')}"
            )

    return build_env(matrix, cell, language, device), cell, warnings


def format_env(env: dict[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in sorted(env.items())) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # OVS_LANGUAGE, not LANGUAGE: POSIX already defines LANGUAGE as the locale
    # fallback list (for example "en_US:en"), so defaulting to it would let an
    # operator's shell locale silently pick the profile.
    parser.add_argument("--language", default=os.environ.get("OVS_LANGUAGE", ""))
    parser.add_argument(
        "--device",
        default=os.environ.get("OVS_MATRIX_DEVICE") or os.environ.get("OVS_DEVICE", ""),
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument(
        "--no-verify-profile",
        action="store_true",
        help="skip the check that the resolved profile JSON exists on disk",
    )
    parser.add_argument("--format", choices=("env", "json"), default="env")
    parser.add_argument(
        "--write-env",
        type=Path,
        default=None,
        help="write the env block to this path instead of stdout",
    )
    args = parser.parse_args(argv)

    if not args.language or not args.device:
        print(
            "ERROR: both --language/OVS_LANGUAGE and --device/OVS_DEVICE are required",
            file=sys.stderr,
        )
        return EXIT_UNKNOWN

    try:
        env, cell, warnings = resolve(
            args.language,
            args.device,
            matrix_path=args.matrix,
            profiles_dir=None if args.no_verify_profile else args.profiles_dir,
        )
    except ResolveError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.code

    for line in warnings:
        print(line, file=sys.stderr)

    if args.format == "json":
        payload = json.dumps(
            {"env": env, "status": cell["status"], "profile": cell["ovs_profile"]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        rendered = payload + "\n"
    else:
        rendered = format_env(env)

    if args.write_env:
        args.write_env.parent.mkdir(parents=True, exist_ok=True)
        args.write_env.write_text(rendered, encoding="utf-8")
        print(f"resolved {env['OVS_LANGUAGE']}/{env['OVS_MATRIX_DEVICE']} -> "
              f"{env['OVS_PROFILE']} ({env['OVS_MATRIX_STATUS']}) -> {args.write_env}",
              file=sys.stderr)
    else:
        sys.stdout.write(rendered)

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
