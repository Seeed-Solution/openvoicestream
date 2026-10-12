#!/usr/bin/env python3
"""Validate the v0.11 Spark GB10 LLM candidate input/identity contract.

Two gates, deliberately separate:

* ``--draft`` (default) lints the *input contract* schema only. A schema-valid
  draft is explicitly **not** launch-ready: build/launch identities may still be
  null.
* ``--verify-runtime`` proves the candidate is launch-ready against the actual
  staged tree: the source package set, target probe, checkpoint, engine bundle,
  external weights/metadata and runtime artifacts are all bound to real bytes.

This script validates identity and provenance only. It never publishes,
promotes, builds or writes a release lock. The canonical directory digest here
is the single hash scheme the Dockerfile and launcher also use (see
``TREE_HASH_SCOPE``); it excludes ``__pycache__``/``*.pyc`` so the identity is
stable across import and build.

Runtime facts (python ABI, CUDA/TensorRT SONAMEs, GPU SM, OS probe) come ONLY
from the official ``tensorrt_edgellm._native`` detection in production. There
is deliberately NO CLI/env probe override: a caller-supplied probe JSON could
fake the whole target gate. Hermetic CPU tests inject a probe through the
direct Python seam ``validate(..., probe_override=dict)`` which is unreachable
from the command line.

The target (Jetson Orin SM87, CUDA 12.6, TensorRT 10.3) is a CUSTOM
COMPATIBILITY candidate. SONAME probes only pin the runtime SONAME major; a
full SDK qualification marker requires independently generated build/runtime
evidence and stays ``not_qualified`` until then.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")

EXPECTED_SCHEMA_VERSION = 1
EXPECTED_ARTIFACT_SET = "orin-sm87-edgellm-v011-candidate-llm"
EXPECTED_UPSTREAM_COMMIT = "95515c2f87fba8982db5a519f9022277667b3cc9"
EXPECTED_TARGET_ID = "jetson-orin-sm87"
EXPECTED_SM = "87"
EXPECTED_CPU_ARCH = "aarch64"
SEMVER_TRIPLET = re.compile(r"(\d+)\.(\d+)\.(\d+)")
SEMVER_PAIR = re.compile(r"\d+\.\d+")
# Historical accepted checkpoint source, byte-bound per the frozen v010
# provenance (engine-overlay-v010 manifest qwen35-gdn-mtp-sm87-v010.toml and
# the measured staging bytes under /home/harve/v010-inputs/qwen35-awq-7551dd):
# the ModelOpt W4A16 g128 AWQ quant ARTIFACT repository
# harvestsu/Qwen3.5-4B-AWQ @ 7551dd66... (no zero point, pre-quant scales,
# lm_head F16). This is deliberately NOT conflated with the original
# unquantized upstream repo (Qwen/Qwen3.5-4B): the actual consumed artifact is
# the pre-quantized revision, and the original repo's revision is UNKNOWN —
# no revision may be invented for it. The frozen quant artifact is REUSED: no
# re-quantization and no new model download; the checkpoint config family
# stays the original multimodal qwen3_5 (no text-only relabel). DECLARED
# identity until the staged checkpoint content is measured at the runtime
# gate; checkpoint_content_sha256 stays null until a FULL inventory
# verification. No new engine payload repository is declared or invented;
# the engine bundle must be staged locally and hash-verified.
EXPECTED_CHECKPOINT_REPO = "harvestsu/Qwen3.5-4B-AWQ"
# Evidenced actual commit (git SHA-1, 40 hex), read from HF-generated
# sources: the hub refs/main stub (byte count 40) and all 10 .metadata first
# lines record 7551dd662b7f7b140aaa99558ac62ac9317be1b5; the model ETAG
# 4bfccfd9... matches the frozen staged bytes. The 41-hex literal recorded
# in the v010 TOML/SOURCE_REVISION carries an extra trailing 'b' — a human
# transcription error, NOT the actual commit (source file kept immutable).
EXPECTED_CHECKPOINT_REVISION = "7551dd662b7f7b140aaa99558ac62ac9317be1b5"
CANDIDATE_PROFILES = ("b1", "b2", "b4")
ALLOWED_BATCH_SIZES = (1, 2, 4)
BATCH_BY_PROFILE = {"b1": 1, "b2": 2, "b4": 4}
DRAFT_STATES = {"draft_inputs_unqualified"}
REVIEWED_NATIVE_MANIFEST_SHA256 = "75a2569823da371de5b1d4d3837d909c451ef727de858e34bc235c7b8656e61d"
REVIEWED_NATIVE_TUPLE_ALLOWLIST = {
    REVIEWED_NATIVE_MANIFEST_SHA256: {
        "variant_id": "jetson-orin-jp62-cu126-sm87-gdn-v3-diagnostic-r8-int4-enabled",
        "source_revision": "62ca2434956ae10959a41c42e55896c572d39dde",
        "source_parent": "6c3095ff0bf6b043f25432f983ff5d4e80e6979e",
        "source_patch": "773f99d865a70309b7c0265b657ed37606bdbe8966664935048076a4506eeb61",
        "extension": "_native/payloads/jetson-orin-jp62-cu126-sm87/_edgellm_runtime.diagnostic.cpython-312-aarch64-linux-gnu.so",
        "extension_sha256": "b6e05eb3315b5f56f6951f91af6f0d62208ef2cf3765cd3d9c9152cdfd8ed484",
        "plugin": "_native/payloads/jetson-orin-jp62-cu126-sm87/lib/libNvInfer_edgellm_plugin.int4-v2-branch-diagnostic.so.1.0",
        "plugin_sha256": "136c6efffdc5fd447cbe7fc35933d5bd8f4d8a94a9ddf4d8cf0fffa5aa7dd745",
    }
}
IMAGE_SELF_IDENTITY_FIELDS = (
    "ref",
    "image_id",
    "registry_digest",
    "size_bytes",
)

# Canonical directory digest scope: relative POSIX path + NUL + file bytes +
# NUL, sorted, excluding volatile interpreter caches. Keep this identical in
# the Dockerfile and launcher.
TREE_HASH_SCOPE = "relpath\\0bytes\\0 sorted; exclude __pycache__ and *.pyc"
EXCLUDED_TREE_NAMES = frozenset({"__pycache__"})
EXCLUDED_TREE_SUFFIXES = (".pyc", ".pyo")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_tree_files(root: Path, extra_excluded_names: frozenset = frozenset()):
    excluded = EXCLUDED_TREE_NAMES | extra_excluded_names
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in excluded for part in path.relative_to(root).parts):
            continue
        if path.name.endswith(EXCLUDED_TREE_SUFFIXES):
            continue
        yield path


def tree_digest(root: Path) -> str:
    """Canonical digest of a directory tree (see ``TREE_HASH_SCOPE``)."""
    digest = hashlib.sha256()
    for path in _iter_tree_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_content_digest(root: Path) -> str:
    """FULL byte-content digest of every checkpoint file (config, tokenizer,
    template and weight files), excluding only the official ``.edgellm``
    cache marker directory and interpreter caches.

    This is deliberately distinct from the official sampled
    ``engine_build._checkpoint_digest`` cache fingerprint: the fingerprint
    only samples weight shards and must never be treated as a full byte
    binding.
    """
    digest = hashlib.sha256()
    for path in _iter_tree_files(root, frozenset({".edgellm"})):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_digest(path: Path) -> str:
    """Digest a file (sha256) or a directory (canonical tree digest)."""
    if path.is_dir():
        return tree_digest(path)
    return _sha256(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        bool(SHA256.fullmatch(str(value or ""))),
        f"{label} must be an immutable sha256 (got {value!r})",
    )
    return str(value)


def _require_git_sha(value: Any, label: str) -> str:
    _require(
        bool(GIT_SHA.fullmatch(str(value or ""))),
        f"{label} must be an immutable git SHA (got {value!r})",
    )
    return str(value)


def _require_relative_safe_path(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and value and not Path(value).is_absolute(),
        f"{label} must be a safe relative path",
    )
    _require(".." not in Path(value).parts, f"{label} must be a safe relative path")
    return value


# --------------------------------------------------------------------------
# Schema / draft gate
# --------------------------------------------------------------------------


def _validate_contract_schema(contract: dict[str, Any], *, verify_runtime: bool) -> None:
    _require(
        contract.get("schema_version") == EXPECTED_SCHEMA_VERSION,
        "schema_version must be 1",
    )
    _require(
        contract.get("artifact_set") == EXPECTED_ARTIFACT_SET,
        f"artifact_set must be {EXPECTED_ARTIFACT_SET}",
    )
    _require(
        contract.get("release_state") in DRAFT_STATES,
        f"release_state must be one of {sorted(DRAFT_STATES)}",
    )
    _require(contract.get("deployable") is False, "candidate must be non-deployable")
    _require(
        contract.get("rollback_release_lock") == "deploy/artifacts/v010-release-lock.json",
        "rollback lock must remain the published v010 release",
    )

    policy = contract.get("gate_policy") or {}
    _require(
        policy.get("draft_gate") == "input-schema-only",
        "draft_gate must be input-schema-only",
    )
    _require(
        policy.get("runtime_gate") == "verify-runtime-required",
        "runtime_gate must be verify-runtime-required",
    )
    _require(
        policy.get("image_identity_gate") == "external-deployment-gate-required",
        "image identity must be gated externally (no self-referential digest)",
    )
    _require(
        policy.get("image_self_identity_fields") == list(IMAGE_SELF_IDENTITY_FIELDS),
        "image_self_identity_fields must stay explicit",
    )

    source = contract.get("source") or {}
    _require(
        source.get("upstream_commit") == EXPECTED_UPSTREAM_COMMIT,
        f"source.upstream_commit must be the fixed v0.11 commit {EXPECTED_UPSTREAM_COMMIT}",
    )
    _require(
        source.get("status") == "candidate_pinned",
        "source.status must be candidate_pinned (candidate is unpublished)",
    )
    _require(
        source.get("in_flight_batching_flag") == "--enable-in-flight-batching",
        "source.in_flight_batching_flag must be the official server flag",
    )
    _require(
        source.get("tree_hash_scope") == TREE_HASH_SCOPE,
        "source.tree_hash_scope must match the canonical tree digest scope",
    )

    # The server package set that must be present for `python -m
    # experimental.server` to import is enumerated by source.packages.
    packages = source.get("packages")
    _require(
        isinstance(packages, dict) and bool(packages),
        "source.packages must enumerate the staged official package set",
    )
    for name, entry in packages.items():
        _require_relative_safe_path(entry.get("path"), f"source.packages.{name}.path")
        _require(
            entry.get("kind") in {"package_dir", "wheel"},
            f"source.packages.{name}.kind must be package_dir or wheel",
        )

    target = contract.get("target") or {}
    _require(
        target.get("target_id") == EXPECTED_TARGET_ID,
        f"target.target_id must be {EXPECTED_TARGET_ID}",
    )
    _require(target.get("sm") == EXPECTED_SM, f"target.sm must be {EXPECTED_SM}")
    _require(
        target.get("cpu_arch") == EXPECTED_CPU_ARCH,
        f"target.cpu_arch must be {EXPECTED_CPU_ARCH}",
    )
    _require(
        "qualification_evidence" in target,
        "target.qualification_evidence must be present (null while unqualified)",
    )

    engine = contract.get("engine") or {}
    _require(
        engine.get("kind") == "vanilla_non_mtp",
        "engine.kind must be vanilla_non_mtp",
    )
    _require(
        engine.get("speculative_decoding") is False and engine.get("mtp_enabled") is False,
        "candidate engine must not use speculative decoding or MTP",
    )
    _validate_engine_profile_schema(engine, verify_runtime=verify_runtime)
    candidate_profiles = engine.get("candidate_profiles")
    _require(
        isinstance(candidate_profiles, list) and bool(candidate_profiles),
        "engine.candidate_profiles must be a non-empty list",
    )
    _require(
        all(isinstance(profile, str) for profile in candidate_profiles),
        "engine.candidate_profiles entries must be strings",
    )
    _require(
        len(set(candidate_profiles)) == len(candidate_profiles),
        "engine.candidate_profiles must not contain duplicates",
    )
    _require(
        set(candidate_profiles).issubset(CANDIDATE_PROFILES),
        f"engine.candidate_profiles must use only {list(CANDIDATE_PROFILES)}",
    )
    max_batch_size = engine.get("max_batch_size")
    _require(
        type(max_batch_size) is int and max_batch_size in ALLOWED_BATCH_SIZES,
        f"engine.max_batch_size must be an int in {list(ALLOWED_BATCH_SIZES)}",
    )
    _require(
        any(BATCH_BY_PROFILE[profile] == max_batch_size for profile in candidate_profiles),
        "engine.max_batch_size must correspond to a declared candidate profile",
    )
    if not verify_runtime:
        _require(
            engine.get("status") == "not_built",
            "draft engine.status must be not_built (no build claims)",
        )
    _require(
        "bundle_files" in engine,
        "engine.bundle_files must be present (null until the bundle is staged): "
        "the ready-bundle closure binds every component file, not just "
        "llm.engine",
    )

    model = contract.get("model") or {}
    _require(
        "checkpoint_content_sha256" in model,
        "model.checkpoint_content_sha256 must be present (null until measured): "
        "the official sampled checkpoint digest is a cache fingerprint only, "
        "not a full byte binding",
    )
    _require(
        model.get("checkpoint_repo") == EXPECTED_CHECKPOINT_REPO,
        f"model.checkpoint_repo must be the frozen byte-bound quant artifact "
        f"repository {EXPECTED_CHECKPOINT_REPO!r} (DECLARED until content is "
        "measured); do not conflate it with the original unquantized repo "
        "Qwen/Qwen3.5-4B",
    )
    _require(
        model.get("revision") == EXPECTED_CHECKPOINT_REVISION,
        f"model.revision must be the evidenced actual commit "
        f"{EXPECTED_CHECKPOINT_REVISION!r} (git SHA-1 from the HF refs/main "
        "stub and .metadata records; the 41-hex v010 TOML/SOURCE_REVISION "
        "literal carries a transcription-error trailing 'b' and must FAIL)",
    )
    _require(
        "engine_bundle_repo" not in model,
        "the engine bundle repository must be recorded under engine, not model",
    )
    _require(
        not engine.get("engine_bundle_repo"),
        "engine.engine_bundle_repo must stay null: no engine payload "
        "repository exists for this candidate; the bundle is a locally "
        "staged, hash-verified directory",
    )

    images = contract.get("runtime_images") or {}
    _require(set(images) == {"llm"}, "runtime_images must bind exactly the llm image")
    _require(
        images["llm"].get("image_identity_status") == "external_gate_pending",
        "image identity status must be external_gate_pending (unverified)",
    )
    _require(
        images["llm"].get("target_ids") == [EXPECTED_TARGET_ID],
        f"runtime_images.llm.target_ids must be exactly [{EXPECTED_TARGET_ID!r}] "
        "(the candidate target is Jetson Orin SM87, not any other platform)",
    )

    artifacts = contract.get("runtime_artifacts") or {}
    expected_keys = set(_expected_runtime_artifact_keys(contract))
    _require(
        set(artifacts) == expected_keys,
        f"runtime_artifacts must be exactly {sorted(expected_keys)}",
    )
    for key, artifact in artifacts.items():
        _require_relative_safe_path(artifact.get("path"), f"{key}.path")

    tts = contract.get("tts_isolation") or {}
    _require(
        tts.get("tts_plugin_mounted") is False
        and tts.get("tts_models_root_mounted") is False
        and tts.get("co_load_v010_plugin") is False,
        "TTS plugin/root must never be mounted or co-loaded in the v011 process",
    )


def _expected_runtime_artifact_keys(contract: dict[str, Any]) -> tuple[str, ...]:
    keys = ["llm/experimental-server"]
    for name in sorted((contract.get("source") or {}).get("packages") or {}):
        keys.append(f"source/{name}")
    return tuple(keys)


def _validate_engine_profile_schema(
    engine: dict[str, Any], *, verify_runtime: bool
) -> None:
    """Validate the profile fields derived by the official server.

    Missing fields remain accepted by the draft gate for old input contracts.
    The runtime gate requires all three fields so a legacy ``tp_size=None``
    contract cannot be called launch-ready when the server fixes tp_size=1.
    """
    names = ("load_media_engines", "components", "tp_size")
    if not verify_runtime and all(name not in engine for name in names):
        return
    if "load_media_engines" in engine:
        _require(
            type(engine["load_media_engines"]) is bool,
            "engine.load_media_engines must be a bool",
        )
    if "components" in engine:
        components = engine["components"]
        _require(
            isinstance(components, list) and bool(components)
            and all(isinstance(value, str) and value.strip() for value in components),
            "engine.components must be a non-empty list of strings",
        )
        _require(
            len(set(components)) == len(components),
            "engine.components must not contain duplicates",
        )
        _require(
            set(components).issubset({"all", "llm"})
            and ("all" not in components or components == ["all"]),
            "engine.components must be exactly ['all'] or ['llm']",
        )
    if "tp_size" in engine:
        _require(
            type(engine["tp_size"]) is int and engine["tp_size"] == 1,
            "engine.tp_size must be the integer 1 for this SM87 candidate",
        )
    if verify_runtime:
        for name in names:
            _require(name in engine, f"engine.{name} is required at runtime")


# --------------------------------------------------------------------------
# Target probe
# --------------------------------------------------------------------------


def _cuda_runtime_semantic_version() -> str:
    """Probe the installed CUDA runtime semantic version via its own API."""
    import ctypes
    import ctypes.util

    name = ctypes.util.find_library("cudart")
    _require(bool(name), "CUDA runtime library is not resolvable")
    lib = ctypes.CDLL(name)
    version = ctypes.c_int()
    result = lib.cudaRuntimeGetVersion(ctypes.byref(version))
    _require(result == 0, f"cudaRuntimeGetVersion failed with {result}")
    return f"{version.value // 1000}.{(version.value % 1000) // 10}"


def _tensorrt_semantic_version() -> str:
    """Probe the TensorRT ABI semantic version from the installed package
    metadata (never invented; build suffix is only present if the package
    reports it)."""
    try:
        import tensorrt
    except ModuleNotFoundError as exc:
        # The slim runtime image can carry the native TensorRT ABI without the
        # optional Python metadata package.  Only that exact missing package
        # takes this measured C-ABI path; unrelated import failures remain
        # fatal.  The four integer APIs are exported by libnvinfer and are
        # checked together before returning the ABI triplet required by the
        # contract.
        if exc.name != "tensorrt":
            raise
        import ctypes
        import ctypes.util

        try:
            name = ctypes.util.find_library("nvinfer")
            _require(bool(name), "TensorRT library is not resolvable")
            lib = ctypes.CDLL(name)
            values = {}
            for symbol in (
                "getInferLibMajorVersion",
                "getInferLibMinorVersion",
                "getInferLibPatchVersion",
                "getInferLibBuildVersion",
            ):
                function = getattr(lib, symbol)
                function.argtypes = []
                function.restype = ctypes.c_int32
                value = function()
                _require(isinstance(value, int) and value >= 0, f"{symbol} returned invalid version")
                values[symbol] = value
        except (OSError, AttributeError, TypeError) as error:
            raise ValueError(f"TensorRT C ABI version probe failed: {error}") from error
        _require(values["getInferLibMajorVersion"] > 0, "TensorRT major version is invalid")
        return ".".join(
            str(values[symbol])
            for symbol in (
                "getInferLibMajorVersion",
                "getInferLibMinorVersion",
                "getInferLibPatchVersion",
            )
        )

    return str(tensorrt.__version__)


def real_target_probe() -> dict[str, Any]:
    """Query the official native detector plus SDK semantic versions.
    Fatal when unavailable."""
    from tensorrt_edgellm._native.detect import detect_platform

    detected = detect_platform()
    return {
        "source": "tensorrt_edgellm._native.detect.detect_platform",
        "python_abi": detected.python_abi,
        "cpu_arch": detected.cpu_arch,
        "platform_probe_source": detected.platform_probe_source,
        "platform_probe_value": detected.platform_probe_value,
        "cuda_runtime_soname": detected.cuda_runtime_soname,
        "tensorrt_runtime_soname": detected.tensorrt_runtime_soname,
        "gpu_sm": str(detected.gpu_sm),
        "cuda_runtime_version": _cuda_runtime_semantic_version(),
        "tensorrt_version": _tensorrt_semantic_version(),
    }


def _load_probe(probe_override: dict[str, Any] | None) -> dict[str, Any]:
    """Production uses the real detector. ``probe_override`` is a direct
    Python test seam (unreachable from CLI/env) for hermetic CPU tests."""
    if probe_override is not None:
        _require(
            isinstance(probe_override, dict),
            "probe_override must be a dict (test seam)",
        )
        return probe_override
    return real_target_probe()


# --------------------------------------------------------------------------
# Staged-tree imports (runtime gate must bind the SAME code the server execs)
# --------------------------------------------------------------------------


@contextmanager
def _staged_imports(runtime_root: Path):
    """Import official modules from EXACTLY the staged runtime root.

    Any pre-imported ``experimental``/``tensorrt_edgellm`` modules (e.g. from
    an audit tree on sys.path) are masked for the duration so the runtime
    gate never validates code that differs from the staged bytes the server
    will execute. Origins are asserted by the caller via ``__file__``.
    """
    root = str(runtime_root.resolve())
    prefixes = ("experimental", "tensorrt_edgellm")
    saved_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in prefixes
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, root)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in prefixes:
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _require_module_origin(module: Any, runtime_root: Path, label: str) -> None:
    origin = Path(str(getattr(module, "__file__", ""))).resolve()
    _require(
        origin.is_relative_to(runtime_root.resolve()),
        f"{label} resolved to {origin}, outside the qualified staged tree "
        f"{runtime_root.resolve()}; the checker and the server must execute "
        "the same staged code",
    )


# --------------------------------------------------------------------------
# Runtime gate
# --------------------------------------------------------------------------


def _validate_runtime_identities(
    contract: dict[str, Any], probe: dict[str, Any]
) -> None:
    source = contract["source"]
    _require_sha256(source.get("server_tree_sha256"), "source.server_tree_sha256")
    _require_sha256(source.get("native_payload_sha256"), "source.native_payload_sha256")
    patches = source.get("patches") or {}
    _require(
        patches.get("status") == "candidate_applied",
        "source.patches.status must be candidate_applied at runtime",
    )
    _require(
        isinstance(patches.get("series"), list) and bool(patches["series"]),
        "source.patches.series must list the applied patch chain",
    )
    for entry in patches["series"]:
        _require(
            isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and bool(entry["name"].strip())
            and bool(SHA256.fullmatch(str(entry.get("sha256", "")))),
            "each applied patch must bind a name and sha256",
        )
    for name, entry in source["packages"].items():
        _require_sha256(entry.get("sha256"), f"source.packages.{name}.sha256")

    # Reconcile duplicate digest aliases: the same staged path must carry ONE
    # digest everywhere it is declared, so no pair can silently diverge.
    artifacts = contract.get("runtime_artifacts") or {}
    for name, entry in source["packages"].items():
        alias = artifacts.get(f"source/{name}") or {}
        _require(
            entry["sha256"] == alias.get("sha256"),
            f"digest alias mismatch: source.packages.{name}.sha256 != "
            f"runtime_artifacts.source/{name}.sha256 for the same staged path",
        )
    server_alias = artifacts.get("llm/experimental-server") or {}
    _require(
        source["server_tree_sha256"] == server_alias.get("sha256"),
        "digest alias mismatch: source.server_tree_sha256 != "
        "runtime_artifacts.llm/experimental-server.sha256 for the same "
        "staged path",
    )

    target = contract["target"]
    _require(
        target.get("qualification_status")
        in {"not_qualified", "candidate_baseline_measured"},
        "target.qualification_status must be an honest non-qualified candidate state",
    )
    if target["qualification_status"] != "not_qualified":
        # A measurement-claiming label must bind an evidence record (path +
        # hash + scope); a freeform 'measured' string is not proof. The
        # record bytes are hash-verified against the staged tree below.
        evidence = target.get("qualification_evidence")
        _require(
            isinstance(evidence, dict),
            "target.qualification_status="
            f"{target['qualification_status']!r} requires "
            "target.qualification_evidence {path, sha256, scope}",
        )
        _require_relative_safe_path(
            evidence.get("path"), "target.qualification_evidence.path"
        )
        _require_sha256(
            evidence.get("sha256"), "target.qualification_evidence.sha256"
        )
        scope = str(evidence.get("scope") or "").strip()
        _require(
            scope and scope.lower() != "measured",
            "target.qualification_evidence.scope must name the actual "
            "measurement scope; a freeform 'measured' label is not proof",
        )
    else:
        _require(
            target.get("qualification_evidence") is None,
            "target.qualification_evidence must stay null while "
            "qualification_status is not_qualified",
        )
    for field in ("os_sdk", "cuda", "tensorrt", "precision"):
        value = target.get(field)
        _require(
            isinstance(value, str) and bool(value.strip()),
            f"target.{field} must be measured on the actual device",
        )

    # Bind the probe to the contract's target and to the runtime ABI.
    probed_sm = str(probe.get("gpu_sm", "")).strip()
    _require(
        probed_sm != "",
        "target probe did not report gpu_sm; runtime SM cannot be verified",
    )
    _require(
        probed_sm == EXPECTED_SM,
        f"target mismatch: probe reports SM {probed_sm}, candidate pins SM {EXPECTED_SM}",
    )
    # A probe SONAME (e.g. libnvinfer.so.10) is NOT a full SDK semantic
    # version: only its major component is compared. The declared full
    # target.tensorrt/target.cuda versions remain DECLARED until an
    # independently generated build/runtime evidence exists.
    probed_tensorrt = str(probe.get("tensorrt_runtime_soname", "")).strip()
    _require(
        probed_tensorrt != "",
        "target probe did not report tensorrt_runtime_soname",
    )
    probed_cuda = str(probe.get("cuda_runtime_soname", "")).strip()
    _require(probed_cuda != "", "target probe did not report cuda_runtime_soname")
    trt_major = re.search(r"libnvinfer\.so\.(\d+)", probed_tensorrt)
    _require(
        trt_major is not None,
        f"probed tensorrt value {probed_tensorrt!r} is not a SONAME with a "
        "major version (libnvinfer.so.<major>)",
    )
    expected_trt_major = str(target["tensorrt"]).strip().split(".")[0]
    _require(
        trt_major.group(1) == expected_trt_major,
        f"target.tensorrt major {expected_trt_major!r} (DECLARED) does not "
        f"match probed TensorRT SONAME major {trt_major.group(1)!r} "
        f"({probed_tensorrt!r})",
    )
    cuda_major = re.search(r"libcudart\.so\.(\d+)", probed_cuda)
    _require(
        cuda_major is not None,
        f"probed cuda value {probed_cuda!r} is not a SONAME with a major "
        "version (libcudart.so.<major>)",
    )
    expected_cuda_major = str(target["cuda"]).strip().split(".")[0]
    _require(
        cuda_major.group(1) == expected_cuda_major,
        f"target.cuda major {expected_cuda_major!r} (DECLARED) does not "
        f"match probed CUDA runtime SONAME major {cuda_major.group(1)!r} "
        f"({probed_cuda!r})",
    )

    # Python ABI must match the interpreter and the staged extension name.
    probed_abi = str(probe.get("python_abi", "")).strip()
    _require(probed_abi != "", "target probe did not report python_abi")
    _require(
        probed_abi == target.get("python_abi"),
        f"python ABI mismatch: probe reports {probed_abi!r}, "
        f"contract pins {target.get('python_abi')!r}",
    )

    # CPU architecture and platform probe facts must be present and bound.
    probed_arch = str(probe.get("cpu_arch", "")).strip()
    _require(probed_arch != "", "target probe did not report cpu_arch")
    _require(
        probed_arch == EXPECTED_CPU_ARCH,
        f"cpu architecture mismatch: probe reports {probed_arch!r}, "
        f"candidate pins {EXPECTED_CPU_ARCH!r}",
    )
    for fact in ("platform_probe_source", "platform_probe_value"):
        _require(
            bool(str(probe.get(fact, "")).strip()),
            f"target probe did not report {fact}",
        )

    # Semantic SDK versions measured from the installed runtime/package, not
    # just SONAME majors. CUDA reports a major.minor runtime version; the
    # TensorRT package reports an ABI triplet. Build suffixes are never
    # invented: target.tensorrt pins exactly the reported ABI triplet.
    probed_cuda_sem = str(probe.get("cuda_runtime_version", "")).strip()
    _require(
        bool(SEMVER_PAIR.fullmatch(probed_cuda_sem)),
        f"target probe did not report a CUDA runtime semantic version "
        f"(major.minor), got {probed_cuda_sem!r}",
    )
    _require(
        probed_cuda_sem == str(target["cuda"]).strip(),
        f"target.cuda {target['cuda']!r} does not match the probed CUDA "
        f"runtime semantic version {probed_cuda_sem!r}",
    )
    probed_trt_sem = str(probe.get("tensorrt_version", "")).strip()
    trt_triplet = SEMVER_TRIPLET.fullmatch(probed_trt_sem)
    _require(
        trt_triplet is not None,
        f"target probe did not report a TensorRT ABI semantic version "
        f"triplet (major.minor.patch), got {probed_trt_sem!r}; if the API "
        "omits a build suffix, compare the ABI triplet only and keep any "
        "build revision separately measured",
    )
    declared_trt = str(target["tensorrt"]).strip()
    _require(
        bool(SEMVER_TRIPLET.fullmatch(declared_trt)),
        f"target.tensorrt must pin exactly the measured ABI triplet "
        f"(major.minor.patch, no invented build suffix), got {declared_trt!r}",
    )
    _require(
        declared_trt == probed_trt_sem,
        f"target.tensorrt {declared_trt!r} does not match the probed "
        f"TensorRT ABI semantic version {probed_trt_sem!r}",
    )

    model = contract["model"]
    _require_git_sha(model.get("revision"), "model.revision")

    engine = contract["engine"]
    _require(
        engine.get("status") == "built",
        "engine.status must be built for a launch-ready candidate "
        "(the official server must reuse a ready bundle, not build one)",
    )
    _require_sha256(engine.get("checkpoint_digest"), "engine.checkpoint_digest")
    _require_sha256(engine.get("bundle_sha256"), "engine.bundle_sha256")
    _require(
        isinstance(engine.get("bundle_dir"), str) and bool(engine["bundle_dir"].strip()),
        "engine.bundle_dir must be the cache bundle path the loader consumes",
    )
    _require_git_sha(engine.get("engine_bundle_revision"), "engine.engine_bundle_revision")
    _require_sha256(engine.get("payload_sha256"), "engine.payload_sha256")
    _require(
        isinstance(engine.get("payload_size"), int) and engine["payload_size"] > 0,
        "engine.payload_size must be positive",
    )
    _require(
        isinstance(engine.get("manifest_recipe"), str)
        and bool(engine["manifest_recipe"].strip()),
        "engine.manifest_recipe must name the non-MTP v011 manifest",
    )
    bundle_files = engine.get("bundle_files")
    _require(
        isinstance(bundle_files, list) and bool(bundle_files),
        "engine.bundle_files must enumerate every ready-bundle component "
        "file with per-file identities at runtime",
    )
    for entry in bundle_files:
        _require(
            isinstance(entry, dict),
            "engine.bundle_files entries must be objects",
        )
        _require_relative_safe_path(entry.get("path"), "engine.bundle_files[].path")
        _require_sha256(entry.get("sha256"), "engine.bundle_files[].sha256")
        _require(
            isinstance(entry.get("size"), int) and entry["size"] > 0,
            "engine.bundle_files[].size must be positive",
        )

    _require_sha256(
        model.get("checkpoint_content_sha256"), "model.checkpoint_content_sha256"
    )

    image = contract["runtime_images"]["llm"]
    _require_git_sha(image.get("service_revision"), "runtime_images.llm.service_revision")
    _require(
        isinstance(image.get("base_image"), str) and bool(image["base_image"].strip()),
        "runtime_images.llm.base_image must be explicit at runtime",
    )
    _require(
        isinstance(image.get("build_runtime_root"), str)
        and bool(image["build_runtime_root"].strip()),
        "runtime_images.llm.build_runtime_root must be explicit at runtime",
    )


def _validate_runtime_artifacts(
    contract: dict[str, Any], runtime_root: Path
) -> None:
    resolved_root = runtime_root.resolve()
    _require(runtime_root.is_dir(), f"runtime root is absent: {runtime_root}")
    artifacts = contract["runtime_artifacts"]
    for key in _expected_runtime_artifact_keys(contract):
        artifact = artifacts[key]
        relative = artifact["path"]
        sha = _require_sha256(artifact.get("sha256"), f"{key}.sha256")
        size = artifact.get("size")
        _require(
            isinstance(size, int) and size > 0,
            f"{key}.size must be positive at runtime",
        )
        _require(
            artifact.get("status") == "staged_qualified",
            f"{key}.status must be staged_qualified at runtime",
        )
        path = (resolved_root / relative).resolve()
        _require(
            path.is_relative_to(resolved_root),
            f"runtime artifact escapes runtime root: {relative}",
        )
        # Every required artifact must exist and match. No silent "not staged".
        _require(
            path.exists(),
            f"required runtime artifact is absent: {relative}",
        )
        _require(
            artifact_digest(path) == sha,
            f"runtime artifact hash mismatch: {relative}",
        )
        if path.is_file() and not relative.startswith("experimental/"):
            _require(
                path.stat().st_size == size,
                f"runtime artifact size mismatch: {relative}",
            )


def _validate_qualification_evidence(
    contract: dict[str, Any], runtime_root: Path
) -> None:
    """Hash-bind the qualification evidence record to staged bytes."""
    target = contract["target"]
    if target.get("qualification_status") == "not_qualified":
        return
    evidence = target["qualification_evidence"]
    resolved_root = runtime_root.resolve()
    path = (resolved_root / evidence["path"]).resolve()
    _require(
        path.is_relative_to(resolved_root),
        "qualification evidence escapes the runtime root",
    )
    _require(
        path.is_file(),
        f"qualification evidence record is absent: {evidence['path']}",
    )
    _require(
        _sha256(path) == evidence["sha256"],
        f"qualification evidence hash mismatch: {evidence['path']}",
    )


def _validate_native_payload(
    contract: dict[str, Any], runtime_root: Path, probe: dict[str, Any]
) -> None:
    """Bind the imported native module to the qualified staged tree.

    Uses the OFFICIAL manifest contract/selection/loader from the staged
    ``tensorrt_edgellm._native`` package itself: the canonical
    ``_native/variants.json`` manifest must exist (no manifestless
    bare-extension fallback), the selected variant must match the probed
    platform exactly, its ``source_revision`` must equal the fixed upstream
    commit (byte-bound manifest, not a JSON constant compared to itself),
    and the declared extension/plugin paths+hashes are verified against the
    actual selected files. The contract ``source.native_payload_sha256`` is
    checked against the actual selected extension file, not just for format.
    """
    source = contract["source"]
    resolved_root = runtime_root.resolve()
    package_root = (resolved_root / "tensorrt_edgellm").resolve()
    _require(
        package_root.is_dir(),
        f"staged tensorrt_edgellm package is absent: {package_root}",
    )
    manifest_path = package_root / "_native" / "variants.json"
    _require(
        manifest_path.is_file(),
        "official native variants manifest is absent: "
        f"{manifest_path}; a manifestless bare-extension fallback is not "
        "accepted by the runtime gate",
    )
    try:
        with _staged_imports(resolved_root):
            import tensorrt_edgellm
            from tensorrt_edgellm._native.detect import DetectedPlatform
            from tensorrt_edgellm._native.load import resolve_payload
            from tensorrt_edgellm._native.select import load_manifest, select_variant
    except Exception as exc:
        raise ValueError(
            "staged tensorrt_edgellm is not importable from the qualified "
            f"staged tree {resolved_root}: {exc}"
        ) from exc
    with _staged_imports(resolved_root):
        _require_module_origin(tensorrt_edgellm, resolved_root, "tensorrt_edgellm")
        _require(
            str(getattr(tensorrt_edgellm, "__version__", ""))
            == str(source.get("upstream_version")),
            f"staged tensorrt_edgellm.__version__ "
            f"{getattr(tensorrt_edgellm, '__version__', None)!r} != "
            f"source.upstream_version {source.get('upstream_version')!r}",
        )

        detected = DetectedPlatform(
            python_abi=str(probe["python_abi"]),
            platform_probe_source=str(probe["platform_probe_source"]),
            platform_probe_value=str(probe["platform_probe_value"]),
            cpu_arch=str(probe["cpu_arch"]),
            cuda_runtime_soname=str(probe["cuda_runtime_soname"]),
            tensorrt_runtime_soname=str(probe["tensorrt_runtime_soname"]),
            gpu_sm=int(probe["gpu_sm"]),
        )
        try:
            manifest = load_manifest(manifest_path)
            selected = select_variant(detected, manifest)
            payload = resolve_payload(
                package_root=package_root,
                manifest_path=manifest_path,
                detector=lambda: detected,
            )
        except Exception as exc:
            raise ValueError(
                f"official native payload selection/verification failed: {exc}"
            ) from exc

    reviewed = None
    if str(selected["source_revision"]) != EXPECTED_UPSTREAM_COMMIT:
        manifest_sha = _sha256(manifest_path)
        reviewed = REVIEWED_NATIVE_TUPLE_ALLOWLIST.get(manifest_sha)
        _require(
            reviewed is not None,
            f"custom native manifest SHA {manifest_sha} is not in the reviewed tuple allowlist",
        )
        for key, expected in reviewed.items():
            _require(
                selected.get(key) == expected,
                f"selected native reviewed tuple field {key} does not match the exact "
                f"manifest-SHA-bound allowlist entry: {selected.get(key)!r} != {expected!r}",
            )
        _require(
            str(selected["source_revision"]) == str(reviewed["source_revision"]),
            f"selected native variant source_revision "
            f"{selected['source_revision']!r} is neither the fixed upstream commit "
            f"{EXPECTED_UPSTREAM_COMMIT} nor the exact reviewed tuple revision "
            f"{reviewed['source_revision']}",
        )

    extension = Path(payload.extension).resolve()
    plugin = Path(payload.plugin).resolve()
    _require(
        extension.is_relative_to(package_root)
        and plugin.is_relative_to(package_root),
        "selected native extension/plugin must live inside the staged "
        "tensorrt_edgellm package",
    )
    _require(
        _sha256(extension) == source["native_payload_sha256"],
        "source.native_payload_sha256 does not match the actual selected "
        f"native extension file {extension}",
    )
    if reviewed is not None:
        _require(
            extension.relative_to(package_root).as_posix() == reviewed["extension"],
            "reviewed extension path does not match the actual selected payload path",
        )
        _require(
            plugin.relative_to(package_root).as_posix() == reviewed["plugin"],
            "reviewed plugin path does not match the actual selected payload path",
        )
        _require(
            _sha256(extension) == reviewed["extension_sha256"],
            "reviewed extension SHA does not match actual selected extension bytes",
        )
        _require(
            _sha256(plugin) == reviewed["plugin_sha256"],
            "reviewed plugin SHA does not match actual selected plugin bytes",
        )

    # Parse the actual wheel package metadata: name/version must match the
    # contract's declared upstream identity.
    dist_info_rel = (source.get("packages") or {}).get("wheel-dist-info") or {}
    dist_info = (resolved_root / str(dist_info_rel.get("path", ""))).resolve()
    metadata = dist_info / "METADATA"
    _require(
        metadata.is_file(),
        f"wheel dist-info METADATA is absent: {metadata}",
    )
    fields = {}
    for line in metadata.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, value = line.split(":", 1)
            fields.setdefault(key.strip().lower(), value.strip())
    _require(
        fields.get("name", "").lower().replace("_", "-") == "tensorrt-edgellm",
        f"wheel METADATA Name {fields.get('name')!r} != tensorrt-edgellm",
    )
    _require(
        fields.get("version") == str(source.get("upstream_version")),
        f"wheel METADATA Version {fields.get('version')!r} != "
        f"source.upstream_version {source.get('upstream_version')!r}",
    )
    entry_points = dist_info / "entry_points.txt"
    if entry_points.is_file():
        script = str(source.get("server_console_script") or "")
        _require(
            f"{script} =" in entry_points.read_text(encoding="utf-8", errors="replace"),
            f"wheel entry_points.txt does not declare the console script {script!r}",
        )


def _validate_staged_engine(
    contract: dict[str, Any], runtime_root: Path | None = None
) -> None:
    """Bind the contract engine identity to the actual cache bundle bytes."""
    engine = contract["engine"]
    bundle_dir = Path(engine["bundle_dir"])
    _require(
        bundle_dir.is_dir(),
        f"engine bundle_dir is absent: {bundle_dir}",
    )

    # The staged tree is the source of truth; import the official builder
    # contract from EXACTLY it (masked against any pre-imported copies) so
    # the readiness check is the real one on the code the server will exec.
    _require(runtime_root is not None, "staged engine validation requires runtime_root")
    try:
        with _staged_imports(runtime_root):
            from experimental.server.runtime import engine_build
    except Exception as exc:
        raise ValueError(
            "official experimental.server.runtime.engine_build is not "
            f"importable from the qualified staged tree {runtime_root}: {exc}"
        ) from exc
    with _staged_imports(runtime_root):
        from experimental.server.runtime import engine_build

        _require_module_origin(
            engine_build, runtime_root, "experimental.server.runtime.engine_build"
        )

        model_dir = engine["model_dir"]
        load_media_engines = engine["load_media_engines"]
        declared_components = tuple(engine["components"])
        expected_components = (
            ("all",)
            if load_media_engines
            else engine_build.text_components(model_dir, engine["cache_dir"])
        )
        _require(
            declared_components == expected_components,
            "engine.components does not match the official server profile: "
            f"declared={declared_components!r} expected={expected_components!r}",
        )
        try:
            options = engine_build.BuildOptions(
                spec_type="none",
                max_input_len=engine["max_input_len"],
                max_kv_cache_capacity=engine["max_kv_cache_capacity"],
                max_batch_size=engine["max_batch_size"],
                tp_size=engine["tp_size"],
                components=declared_components,
            )
        except TypeError as exc:
            raise ValueError(
                "official BuildOptions does not support the declared components profile"
            ) from exc
        expected_bundle = engine_build.bundle_cache_path(
            model_dir, engine["cache_dir"], options
        )
        _require(
            Path(expected_bundle).resolve() == bundle_dir.resolve(),
            "engine.bundle_dir does not equal the official bundle_cache_path for "
            "the declared checkpoint/profile; the launch would not reuse this bundle",
        )
        _require(
            engine_build._is_ready(model_dir, str(bundle_dir), options),
            "official engine_build._is_ready reports the bundle is not complete/ready "
            "for the declared profile; refusing (no implicit build at launch)",
        )
        # The official sampled digest is a CACHE FINGERPRINT only (it samples
        # weight shards); it keys the bundle cache path and is preserved as
        # such. The full byte binding is checkpoint_content_sha256 below.
        _require(
            engine_build._checkpoint_digest(model_dir) == engine["checkpoint_digest"],
            "engine.checkpoint_digest does not equal the official "
            "engine_build._checkpoint_digest of the staged checkpoint",
        )

    _require(
        tree_digest(bundle_dir) == engine["bundle_sha256"],
        "engine.bundle_sha256 does not match the staged bundle bytes",
    )

    # Full checkpoint inventory/content binding: every config, tokenizer,
    # template and weight file is hashed completely (not sampled).
    _require(
        checkpoint_content_digest(Path(model_dir))
        == contract["model"]["checkpoint_content_sha256"],
        "model.checkpoint_content_sha256 does not match the full staged "
        "checkpoint bytes (the sampled fingerprint is not a byte binding)",
    )

    # Full ready-bundle closure: the declared per-file identities must equal
    # EXACTLY the files present, each hash/size bound to actual bytes.
    declared = {
        entry["path"]: (entry["sha256"], entry["size"])
        for entry in engine["bundle_files"]
    }
    _require(
        len(declared) == len(engine["bundle_files"]),
        "engine.bundle_files contains duplicate paths",
    )
    actual = {
        path.relative_to(bundle_dir).as_posix(): path
        for path in _iter_tree_files(bundle_dir)
    }
    _require(
        set(declared) == set(actual),
        "engine.bundle_files does not equal the actual ready-bundle file "
        f"closure: missing={sorted(set(actual) - set(declared))} "
        f"undeclared={sorted(set(declared) - set(actual))}",
    )
    for relative, path in sorted(actual.items()):
        expected_sha, expected_size = declared[relative]
        _require(
            _sha256(path) == expected_sha,
            f"engine.bundle_files hash mismatch for {relative}",
        )
        _require(
            path.stat().st_size == expected_size,
            f"engine.bundle_files size mismatch for {relative}",
        )

    # Keep the named llm payload binding as a convenience alias, reconciled
    # with the closure above.
    payload = bundle_dir / "llm.engine"
    _require(
        payload.is_file(),
        f"engine bundle has no llm.engine payload: {payload}",
    )
    _require(
        _sha256(payload) == engine["payload_sha256"],
        "engine.payload_sha256 does not match the staged llm.engine bytes",
    )
    _require(
        payload.stat().st_size == engine["payload_size"],
        "engine.payload_size does not match the staged llm.engine size",
    )
    declared_payload = declared.get("llm.engine")
    _require(
        declared_payload == (engine["payload_sha256"], engine["payload_size"]),
        "engine.payload_sha256/payload_size aliases disagree with "
        "engine.bundle_files[llm.engine]",
    )


def validate(
    contract_path: Path,
    *,
    verify_runtime: bool = False,
    runtime_root: Path | None = None,
    probe_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(isinstance(contract, dict), "input contract must be a JSON object")
    _validate_contract_schema(contract, verify_runtime=verify_runtime)
    if verify_runtime:
        _require(runtime_root is not None, "--verify-runtime requires --runtime-root")
        probe = _load_probe(probe_override)
        _validate_runtime_identities(contract, probe)
        _validate_runtime_artifacts(contract, runtime_root)
        _validate_qualification_evidence(contract, runtime_root)
        _validate_native_payload(contract, runtime_root, probe)
        _validate_staged_engine(contract, runtime_root)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("deploy/artifacts/v011-candidate-llm-inputs.json"),
    )
    parser.add_argument(
        "--tree-digest",
        type=Path,
        help="print the canonical tree digest of DIR and exit (build helper)",
    )
    parser.add_argument(
        "--file-digest",
        type=Path,
        help="print the sha256 of FILE and exit (build helper)",
    )
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument("--draft", action="store_true", help="schema/input lint only")
    gate.add_argument(
        "--verify-runtime",
        action="store_true",
        help="prove identities, staged tree and engine bundle are launch-ready",
    )
    parser.add_argument("--runtime-root", type=Path)
    # Deliberately NO --probe-json: production must measure the real device.
    # Hermetic tests inject a probe via validate(..., probe_override=...).
    args = parser.parse_args()

    if args.tree_digest is not None:
        if not args.tree_digest.is_dir():
            parser.error(f"--tree-digest path is not a directory: {args.tree_digest}")
        print(tree_digest(args.tree_digest.resolve()))
        return 0
    if args.file_digest is not None:
        if not args.file_digest.is_file():
            parser.error(f"--file-digest path is not a file: {args.file_digest}")
        print(_sha256(args.file_digest.resolve()))
        return 0

    try:
        contract = validate(
            args.contract.resolve(),
            verify_runtime=args.verify_runtime,
            runtime_root=args.runtime_root.resolve() if args.runtime_root else None,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "artifact_set": contract["artifact_set"],
                "release_state": contract["release_state"],
                "deployable": contract["deployable"],
                "gate": "verify-runtime" if args.verify_runtime else "draft",
                "launch_ready": bool(args.verify_runtime),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
