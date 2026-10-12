#!/usr/bin/env python3
"""Fail-closed launcher for the v0.11 Jetson Orin (SM87) LLM candidate.

Flow:

1. Refuse to run unless the caller explicitly opts in to the unpublished
   candidate (``EDGELLM_V011_ALLOW_UNPUBLISHED_CANDIDATE=1``).
2. Validate the candidate input contract at the *runtime* gate
   (``check_edgellm_v011_candidate.py --verify-runtime``): source package set,
   real device probe, checkpoint, ready engine bundle and staged runtime
   artifacts must all be bound to actual bytes. There is deliberately NO probe
   override here: production always measures the real device.
3. Bind the launch identity to the contract: the profile batch width, the
   checkpoint and the cache directory MUST equal the contract values the
   checker validated; env values that disagree fail closed.
4. ``exec`` the **official** v0.11 server module (``python -m
   experimental.server``) with the checkpoint positional, ``--cache-dir``,
   ``--max-input-len`` / ``--max-kv-cache-capacity`` / ``--max-batch-size``
   equal to the validated engine profile (the server derives its bundle cache
   path from these flags), plus ``--enable-in-flight-batching`` for B2/B4.

No custom scheduler, no FastAPI stack, no ``sed``-patched external entrypoint,
no direct runtime KV warm-up, and no implicit download/build at launch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "deploy/artifacts/v011-candidate-llm-inputs.json"
)

SPEC_DECODE_FILES = ("spec_base.engine", "spec_draft.engine")
VANILLA_LLM_FILES = ("llm.engine", "dllm.engine")
BATCH_BY_PROFILE = {"b1": 1, "b2": 2, "b4": 4}
MIN_IFB_BATCH_SIZE = 2


def _fail(message: str) -> None:
    print(f"v0.11 candidate launcher: {message}", file=sys.stderr)
    raise SystemExit(2)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def _detect_engine_kind(bundle_dir: Path) -> str:
    if any((bundle_dir / name).is_file() for name in SPEC_DECODE_FILES):
        return "spec_decode"
    if any((bundle_dir / name).is_file() for name in VANILLA_LLM_FILES):
        return "vanilla_llm"
    return "unknown"


def _validate_launch_invariants(bundle_dir: Path, profile: str) -> None:
    if profile not in BATCH_BY_PROFILE:
        _fail(f"unknown engine profile {profile!r}; expected one of {sorted(BATCH_BY_PROFILE)}")
    if not bundle_dir.is_dir():
        _fail(f"deployment engine bundle dir is absent: {bundle_dir}")
    kind = _detect_engine_kind(bundle_dir)
    if kind == "spec_decode":
        _fail(
            "deployment engine is a speculative-decoding bundle; IFB refuses "
            "MTP/spec-decode. Build a non-MTP vanilla engine."
        )
    if kind != "vanilla_llm":
        _fail(f"engine bundle {bundle_dir} is not a vanilla llm.engine bundle")


def _validate_server_profile(
    engine: dict, *, runtime_root: Path, checkpoint: str, cache_dir: str
) -> None:
    """Require the contract profile to equal the official server's profile."""
    for name in ("load_media_engines", "components", "tp_size"):
        if name not in engine:
            _fail(f"engine.{name} is required for a launch-ready runtime contract")
    load_media_engines = engine["load_media_engines"]
    if type(load_media_engines) is not bool:
        _fail("engine.load_media_engines must be a bool")
    components = engine["components"]
    if (not isinstance(components, list) or not components
            or any(not isinstance(value, str) or not value.strip()
                   for value in components)
            or len(set(components)) != len(components)
            or set(components) - {"all", "llm"}
            or ("all" in components and components != ["all"])):
        _fail("engine.components must be exactly ['all'] or ['llm']")
    tp_size = engine["tp_size"]
    if type(tp_size) is not int or tp_size != 1:
        _fail("engine.tp_size must be the integer 1 for this SM87 candidate")
    import sys as _sys
    _sys.path.insert(0, str(runtime_root))
    try:
        from experimental.server.runtime import engine_build
        expected = (("all",) if load_media_engines else
                    engine_build.text_components(checkpoint, cache_dir))
    except Exception as exc:
        _fail(f"cannot derive the official server profile: {exc}")
    if tuple(components) != tuple(expected):
        _fail(
            "engine.components does not match the official server profile: "
            f"declared={tuple(components)!r} expected={tuple(expected)!r}"
        )


def _checker_path() -> Path:
    # The verifier is FIXED beside this launcher. There is deliberately no
    # environment override: an env-selectable checker would be a bypass of
    # the entire runtime identity gate.
    return HERE / "check_edgellm_v011_candidate.py"


def _run_runtime_check(contract: Path, runtime_root: Path) -> None:
    checker = _checker_path()
    if not checker.is_file():
        _fail(f"runtime checker is absent: {checker}")
    command = [
        sys.executable,
        str(checker),
        "--contract",
        str(contract),
        "--verify-runtime",
        "--runtime-root",
        str(runtime_root),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        _fail("candidate identity/provenance verification failed")


def build_server_command(
    checkpoint: str,
    cache_dir: str,
    batch_size: int,
    max_input_len: int,
    max_kv_cache_capacity: int,
    host: str,
    port: str,
    max_queued_requests: str,
    queue_timeout: str,
    load_media_engines: bool = True,
    served_model_name: str = "qwen3.5-4b",
) -> list[str]:
    """Assemble the official CLI argv (pure; unit-testable).

    ``--max-input-len`` / ``--max-kv-cache-capacity`` / ``--max-batch-size``
    MUST reproduce the profile the checker validated: the official server
    derives its engine bundle cache path from exactly these flags, so any
    divergence means a cache miss and an implicit GPU build at launch.
    """
    command = [
        sys.executable,
        "-m",
        "experimental.server",
        checkpoint,
        "--cache-dir",
        cache_dir,
    ]
    if batch_size >= MIN_IFB_BATCH_SIZE:
        command.append("--enable-in-flight-batching")
    if type(load_media_engines) is not bool:
        raise ValueError("load_media_engines must be a bool")
    if not load_media_engines:
        command.append("--no-load-media-engines")
    if not served_model_name.strip():
        raise ValueError("served_model_name must be non-empty")
    command += [
        "--served-model-name",
        served_model_name.strip(),
        "--max-batch-size",
        str(batch_size),
        "--max-input-len",
        str(max_input_len),
        "--max-kv-cache-capacity",
        str(max_kv_cache_capacity),
        "--max-queued-requests",
        max_queued_requests,
        "--queue-timeout",
        queue_timeout,
        "--host",
        host,
        "--port",
        port,
    ]
    return command


def main() -> int:
    if not _truthy(os.environ.get("EDGELLM_V011_ALLOW_UNPUBLISHED_CANDIDATE")):
        _fail(
            "refusing to start: set "
            "EDGELLM_V011_ALLOW_UNPUBLISHED_CANDIDATE=1 to run the "
            "unpublished candidate"
        )

    contract = Path(
        os.environ.get("EDGELLM_V011_INPUT_CONTRACT", str(DEFAULT_CONTRACT))
    )
    if not contract.is_file():
        _fail(f"candidate input contract is absent: {contract}")
    contract_data = json.loads(contract.read_text(encoding="utf-8"))

    runtime_root = Path(
        os.environ.get("EDGELLM_V011_RUNTIME_ROOT", "/opt/edgellm-v011")
    )

    # The contract is the single source of launch identity. Environment
    # overrides that disagree with the validated contract fail closed.
    engine = contract_data["engine"]
    model = contract_data.get("model") or {}
    contract_model_id = str(model.get("model_id") or "").strip()
    if not contract_model_id:
        _fail("model.model_id is not set in the contract")
    served_model_name_env = os.environ.get("EDGELLM_V011_SERVED_MODEL_NAME")
    served_model_name = (served_model_name_env or contract_model_id).strip()
    if not served_model_name:
        _fail("EDGELLM_V011_SERVED_MODEL_NAME must be non-empty")
    if served_model_name_env is not None and served_model_name != contract_model_id:
        _fail(
            "EDGELLM_V011_SERVED_MODEL_NAME "
            f"({served_model_name!r}) differs from the validated contract "
            f"model.model_id ({contract_model_id!r}); refusing to alias the "
            "candidate service to a different model"
        )

    contract_batch = engine.get("max_batch_size")
    profile_env = os.environ.get("EDGELLM_V011_ENGINE_PROFILE")
    if profile_env is None:
        # Derive the default profile from the validated contract batch so the
        # bound contract profile is selected without hardcoding one batch
        # size (which would break later qualified B1/B2 contracts).
        derived = {v: k for k, v in BATCH_BY_PROFILE.items()}.get(contract_batch)
        if derived is None:
            _fail(
                "cannot derive a default engine profile from the "
                f"contract-validated engine.max_batch_size ({contract_batch!r}); "
                f"expected one of {sorted(BATCH_BY_PROFILE.values())}; set "
                "EDGELLM_V011_ENGINE_PROFILE explicitly"
            )
        profile = derived
    else:
        profile = profile_env.strip().lower()
    if profile not in BATCH_BY_PROFILE:
        _fail(f"unknown engine profile {profile!r}; expected one of {sorted(BATCH_BY_PROFILE)}")
    if contract_batch != BATCH_BY_PROFILE[profile]:
        _fail(
            f"engine profile {profile!r} (batch {BATCH_BY_PROFILE[profile]}) does "
            f"not match the contract-validated engine.max_batch_size "
            f"({contract_batch!r}); the launch would cache-miss the validated "
            "bundle and build implicitly"
        )

    checkpoint = str(engine.get("model_dir") or "")
    cache_dir = str(engine.get("cache_dir") or "")
    env_checkpoint = os.environ.get("EDGELLM_V011_CHECKPOINT")
    if env_checkpoint is not None and env_checkpoint != checkpoint:
        _fail(
            "EDGELLM_V011_CHECKPOINT "
            f"({env_checkpoint!r}) differs from the validated contract "
            f"engine.model_dir ({checkpoint!r}); refusing to substitute "
            "identities"
        )
    env_cache_dir = os.environ.get("EDGELLM_V011_CACHE_DIR")
    if env_cache_dir is not None and env_cache_dir != cache_dir:
        _fail(
            "EDGELLM_V011_CACHE_DIR "
            f"({env_cache_dir!r}) differs from the validated contract "
            f"engine.cache_dir ({cache_dir!r}); refusing to substitute "
            "identities"
        )
    if not checkpoint:
        _fail("engine.model_dir is not set in the contract")
    if not cache_dir:
        _fail("engine.cache_dir is not set in the contract")

    max_input_len = int(engine.get("max_input_len") or 0)
    max_kv_cache_capacity = int(engine.get("max_kv_cache_capacity") or 0)
    if max_input_len <= 0 or max_kv_cache_capacity <= 0:
        _fail(
            "engine.max_input_len / engine.max_kv_cache_capacity must be "
            "positive in the contract"
        )
    bundle_dir = Path(str(engine.get("bundle_dir") or ""))
    if not str(bundle_dir):
        _fail("engine.bundle_dir is not set in the contract")

    host = os.environ.get("EDGELLM_V011_HOST", "0.0.0.0")
    port = os.environ.get("EDGELLM_V011_PORT", "8000")
    max_queued = os.environ.get("EDGELLM_V011_MAX_QUEUED_REQUESTS", "16")
    queue_timeout = os.environ.get("EDGELLM_V011_QUEUE_TIMEOUT", "600")
    load_media_engines = engine.get("load_media_engines", True)
    if type(load_media_engines) is not bool:
        _fail("engine.load_media_engines must be a bool")

    _validate_server_profile(
        engine,
        runtime_root=runtime_root,
        checkpoint=checkpoint,
        cache_dir=cache_dir,
    )

    _run_runtime_check(contract, runtime_root)
    _validate_launch_invariants(bundle_dir, profile)

    command = build_server_command(
        checkpoint,
        cache_dir,
        BATCH_BY_PROFILE[profile],
        max_input_len,
        max_kv_cache_capacity,
        host,
        port,
        max_queued,
        queue_timeout,
        load_media_engines,
        served_model_name,
    )
    print(f"v0.11 candidate launcher: exec {command!r}", file=sys.stderr)
    # Keep the staged tree byte-stable for the canonical tree hash: the
    # launched interpreter must not write __pycache__/*.pyc artifacts.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
