"""Leaf-and-composition config resolver (TRACK 1 SLICE 1 — standalone).

A **leaf** is the atomic provisioning unit: everything needed to run ONE
capability with ONE backend on ONE device at ONE concurrency. A **composition**
(later: a profile) is a thin selection over leaves for a device plus env
overrides. This module loads the leaf/device/model registry from
``configs/leaves/*.yaml`` and provides three pure-functional resolvers:

  * :func:`resolve_pull`  — deterministic de-duped UNION of artifact files
    (expanding ``requires`` shared sub-leaves). ASR leaf files are invariant
    under any TTS choice — structural, not a convention.
  * :func:`resolve_env`   — merged ``runtime_env`` (leaf < overrides), with
    ``${VAR}`` expansion consistent with :mod:`server.core.profile_loader`.
  * :func:`validate_composition` — fail-fast compose-time validation (memory
    headroom, unknown/unbuilt leaf id, illegal capability pairing).

Precision is a leaf attribute: an unset leaf precision resolves via the logical
model's ``default_precision[device_class]``; an explicit leaf precision wins.
Flipping a model's per-device-class default re-resolves every leaf of that
model with no churn in compositions.

This module is STANDALONE: it is not wired into the boot sequence. The env
layer (live ``os.environ`` precedence) is applied later at integration; here
``resolve_env`` only merges ``leaf < overrides``.

See ``docs/specs/leaf-composition-config.md`` for the full design.
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CompositionError(ValueError):
    """Raised when a composition is invalid (compose-time, fail fast).

    Carries a clear human-readable message: memory over headroom, unknown or
    unbuilt leaf id (never a silent fallback), or an illegal pairing (two
    leaves contributing the same capability).
    """


class RegistryError(ValueError):
    """Raised when the on-disk leaf/device/model registry is malformed."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Artifacts:
    """Files and model-level source metadata a leaf contributes."""

    repo: str = ""
    files: tuple[str, ...] = ()
    revision: str = "main"
    canonical_model_id: str = ""
    root: str = ""
    manifest: str = ""
    cache_root: str = ""
    strict: bool = False


@dataclass(frozen=True)
class Leaf:
    """Atomic provisioning unit: (capability, backend, device, concurrency)."""

    id: str
    capability: str
    backend: str = ""
    device: str = ""
    concurrency: int = 1
    model: str = ""
    # ``None`` → resolve via ModelSpec.default_precision[device_class];
    # an explicit value wins over the model default.
    precision: str | None = None
    artifacts: Artifacts = field(default_factory=Artifacts)
    # ids of shared sub-leaves whose artifacts/env this leaf pulls in.
    requires: tuple[str, ...] = ()
    runtime_env: Mapping[str, str] = field(default_factory=dict)
    resources: Mapping[str, object] = field(default_factory=dict)

    @property
    def peak_unified_mb(self) -> int:
        raw = self.resources.get("peak_unified_mb", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0


@dataclass(frozen=True)
class DeviceSpec:
    """Device class: unified memory budget + recommended default combo."""

    id: str
    unified_mb: int = 0
    # Non-leaf resident floor (MB): idle voice-stack-absent baseline + system /
    # other-resident services that every composition pays once. Leaf
    # ``peak_unified_mb`` values are DELTAS over this baseline. Defaults to 0
    # for back-compat with registries that predate this field.
    base_reservation_mb: int = 0
    # capability -> default leaf id (e.g. {"asr": "...", "tts": "..."})
    default: Mapping[str, str] = field(default_factory=dict)
    # device class for precision resolution (e.g. "jetson"); defaults to id.
    device_class: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """Logical model: precision plus optional static catalog metadata.

    The metadata is intentionally sparse and descriptive.  Runtime backend
    state remains authoritative for readiness, sample rate, capabilities and
    concurrency; older precision-only model entries continue to parse.
    """

    id: str
    # device_class -> precision (e.g. {"jetson": "fp16"})
    default_precision: Mapping[str, str] = field(default_factory=dict)
    label: str | None = None
    aliases: tuple[str, ...] = ()
    speakers: tuple[Mapping[str, object], ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelArtifactSource:
    """One model-level repository selected by a composition."""

    model_id: str
    repo: str
    revision: str = "main"
    canonical_model_id: str = ""
    root: str = ""
    manifest: str = ""
    cache_root: str = ""
    files: tuple[str, ...] = ()
    strict: bool = False

    @property
    def canonical_id(self) -> str:
        value = (self.canonical_model_id or self.model_id).strip().replace("/", "--")
        out = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)
        return out.strip(".-") or "model"


@dataclass(frozen=True)
class ResolvedLeaf:
    """A leaf with its concrete precision resolved for a device."""

    leaf: Leaf
    precision: str | None


@dataclass(frozen=True)
class CompositionPlan:
    """Validated plan returned by :func:`validate_composition`."""

    device: str
    leaf_ids: tuple[str, ...]
    resolved: tuple[ResolvedLeaf, ...]
    peak_unified_mb: int
    headroom_mb: int


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Default headroom: only this fraction of unified memory is usable for the
# resident model footprint. Conservative; mirrors the "validate on nx, reject
# on nano" sizing intent in the design doc.
DEFAULT_HEADROOM_FRACTION = 0.85


def _leaves_dir() -> Path:
    # __file__ = <repo>/server/core/leaf_composition.py → parents[2] = <repo>
    return Path(__file__).resolve().parents[2] / "configs" / "leaves"


@dataclass
class Registry:
    """In-memory leaf/device/model registry loaded from YAML."""

    leaves: dict[str, Leaf] = field(default_factory=dict)
    devices: dict[str, DeviceSpec] = field(default_factory=dict)
    models: dict[str, ModelSpec] = field(default_factory=dict)
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION

    # -- precision resolution ------------------------------------------------

    def device_class(self, device: str) -> str:
        spec = self.devices.get(device)
        if spec is not None and spec.device_class:
            return spec.device_class
        # Heuristic fallback shared by the seed devices: orin-* → jetson.
        if device.startswith("orin"):
            return "jetson"
        return device

    def resolve_precision(self, leaf: Leaf, device: str) -> str | None:
        """Explicit leaf precision wins; else the model default for the class."""
        if leaf.precision is not None:
            return leaf.precision
        model = self.models.get(leaf.model)
        if model is None:
            return None
        return model.default_precision.get(self.device_class(device))

    # -- leaf lookup with expansion -----------------------------------------

    def get_leaf(self, leaf_id: str) -> Leaf:
        leaf = self.leaves.get(leaf_id)
        if leaf is None:
            raise CompositionError(
                f"no leaf {leaf_id!r} — not built (unknown leaf id; "
                f"compositions never silently fall back)"
            )
        return leaf

    def expand(self, leaf_ids: Iterable[str]) -> list[Leaf]:
        """Return selected leaves plus their (transitively) required sub-leaves.

        Deterministic order: a leaf's required sub-leaves are appended in
        declaration order, depth-first, de-duplicated by id. Order is stable
        across calls so resolvers are reproducible.
        """
        ordered: list[Leaf] = []
        seen: set[str] = set()

        def _visit(lid: str) -> None:
            if lid in seen:
                return
            seen.add(lid)
            leaf = self.get_leaf(lid)
            ordered.append(leaf)
            for req in leaf.requires:
                _visit(req)

        for lid in leaf_ids:
            _visit(lid)
        return ordered


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _parse_artifacts(raw: object) -> Artifacts:
    if not raw:
        return Artifacts()
    if not isinstance(raw, Mapping):
        raise RegistryError(f"leaf artifacts must be a mapping, got {type(raw)!r}")
    return Artifacts(
        repo=str(raw.get("repo", raw.get("hf_repo", "")) or ""),
        files=_as_tuple(raw.get("files")),
        revision=str(raw.get("revision", raw.get("hf_revision", "main")) or "main"),
        canonical_model_id=str(
            raw.get("canonical_model_id", raw.get("canonical_id", raw.get("model_id", ""))) or ""
        ),
        root=str(raw.get("root", raw.get("model_root", "")) or ""),
        manifest=str(raw.get("manifest", raw.get("manifest_path", "")) or ""),
        cache_root=str(raw.get("cache_root", raw.get("model_cache_root", "")) or ""),
        strict=bool(raw.get("strict", False)),
    )


def _parse_leaf(leaf_id: str, raw: Mapping) -> Leaf:
    cap = raw.get("capability")
    if not cap:
        raise RegistryError(f"leaf {leaf_id!r} missing required field 'capability'")
    runtime_env = raw.get("runtime_env") or {}
    if not isinstance(runtime_env, Mapping):
        raise RegistryError(f"leaf {leaf_id!r} runtime_env must be a mapping")
    resources = raw.get("resources") or {}
    if not isinstance(resources, Mapping):
        raise RegistryError(f"leaf {leaf_id!r} resources must be a mapping")
    return Leaf(
        id=leaf_id,
        capability=str(cap),
        backend=str(raw.get("backend", "")),
        device=str(raw.get("device", "")),
        concurrency=int(raw.get("concurrency", 1)),
        model=str(raw.get("model", "")),
        precision=(None if raw.get("precision") is None else str(raw.get("precision"))),
        artifacts=_parse_artifacts(raw.get("artifacts")),
        requires=_as_tuple(raw.get("requires")),
        runtime_env={str(k): str(v) for k, v in runtime_env.items()},
        resources={str(k): v for k, v in resources.items()},
    )


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_registry(leaves_dir: str | Path | None = None) -> Registry:
    """Load the leaf/device/model registry from ``configs/leaves/``.

    ``devices.yaml`` (``devices:`` map) and ``models.yaml`` (``models:`` map)
    are reserved filenames. Every other ``*.yaml`` is a leaf definition file
    with a top-level ``leaves:`` map of ``{id: {...leaf fields...}}``.
    """
    root = Path(leaves_dir) if leaves_dir is not None else _leaves_dir()
    if not root.is_dir():
        raise RegistryError(f"leaves dir does not exist: {root}")

    reg = Registry()

    devices_path = root / "devices.yaml"
    if devices_path.is_file():
        for dev_id, raw in (_load_yaml(devices_path).get("devices") or {}).items():
            raw = raw or {}
            reg.devices[str(dev_id)] = DeviceSpec(
                id=str(dev_id),
                unified_mb=int(raw.get("unified_mb", 0)),
                base_reservation_mb=int(raw.get("base_reservation_mb", 0)),
                default={str(k): str(v) for k, v in (raw.get("default") or {}).items()},
                device_class=str(raw.get("device_class", "")),
            )

    models_path = root / "models.yaml"
    if models_path.is_file():
        for model_id, raw in (_load_yaml(models_path).get("models") or {}).items():
            raw = raw or {}
            dp = raw.get("default_precision") or {}
            speakers_raw = raw.get("speakers") or ()
            if isinstance(speakers_raw, Mapping):
                speakers_raw = (speakers_raw,)
            if not isinstance(speakers_raw, (list, tuple)):
                raise RegistryError(
                    f"model {model_id!r} speakers must be a list or mapping"
                )
            if any(not isinstance(item, Mapping) for item in speakers_raw):
                raise RegistryError(
                    f"model {model_id!r} speakers entries must be mappings"
                )
            reg.models[str(model_id)] = ModelSpec(
                id=str(model_id),
                default_precision={str(k): str(v) for k, v in dp.items()},
                label=(None if raw.get("label") is None else str(raw.get("label"))),
                aliases=_as_tuple(raw.get("aliases")),
                speakers=tuple(dict(item) for item in speakers_raw),
                metadata={
                    str(k): v
                    for k, v in raw.items()
                    if k not in {"default_precision", "label", "aliases", "speakers"}
                },
            )

    for path in sorted(root.glob("*.yaml")):
        if path.name in ("devices.yaml", "models.yaml"):
            continue
        for leaf_id, raw in (_load_yaml(path).get("leaves") or {}).items():
            if str(leaf_id) in reg.leaves:
                raise RegistryError(f"duplicate leaf id {leaf_id!r} (in {path.name})")
            reg.leaves[str(leaf_id)] = _parse_leaf(str(leaf_id), raw or {})

    return reg


# ---------------------------------------------------------------------------
# ${VAR} expansion (consistent with profile_loader)
# ---------------------------------------------------------------------------

def _expand(value: str, env: Mapping[str, str]) -> str:
    """Expand ``$VAR`` / ``${VAR}`` against ``env``; unknown vars preserved.

    Mirrors :func:`server.core.profile_loader._expand_with_profile_env`
    semantics (``string.Template.safe_substitute`` — never raises on a
    malformed ``$``).
    """
    try:
        return string.Template(value).safe_substitute(env)
    except (ValueError, KeyError):
        return value


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------

def resolve_pull(
    selected_leaf_ids: Iterable[str],
    registry: Registry,
) -> list[str]:
    """Deterministic de-duped UNION of artifact files across selected leaves.

    Shared ``requires`` sub-leaves are expanded and contributed once. The file
    list for any one leaf is independent of which other leaves are selected, so
    ASR files are invariant under any TTS choice (structural guarantee).

    Returns files in first-seen order with duplicates removed.
    """
    files: list[str] = []
    seen: set[str] = set()
    for leaf in registry.expand(selected_leaf_ids):
        for f in leaf.artifacts.files:
            if f not in seen:
                seen.add(f)
                files.append(f)
    return files


def resolve_model_sources(
    selected_leaf_ids: Iterable[str], registry: Registry
) -> list[ModelArtifactSource]:
    """Resolve model-level source records independently of profiles."""
    grouped: dict[tuple[str, str, str, str, str, str, str, bool], list[str]] = {}
    seen: dict[tuple[str, str, str, str, str, str, str, bool], set[str]] = {}
    for leaf in registry.expand(selected_leaf_ids):
        model_id = str(leaf.model or "").strip()
        if not model_id:
            continue
        a = leaf.artifacts
        if not a.repo.strip() and not a.files:
            continue
        key = (
            model_id, a.repo.strip(), a.revision.strip() or "main",
            a.canonical_model_id.strip(), a.root.strip(), a.manifest.strip(), a.cache_root.strip(), a.strict,
        )
        grouped.setdefault(key, [])
        seen.setdefault(key, set())
        for path in a.files:
            if path not in seen[key]:
                seen[key].add(path)
                grouped[key].append(path)
    return [
        ModelArtifactSource(
            model_id=key[0], repo=key[1], revision=key[2],
            canonical_model_id=key[3], root=key[4], manifest=key[5],
            cache_root=key[6], strict=key[7], files=tuple(files),
        )
        for key, files in grouped.items()
    ]


resolve_artifacts = resolve_model_sources
resolve_model_artifacts = resolve_model_sources


def resolve_env(
    selected_leaf_ids: Iterable[str],
    registry: Registry,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge selected leaves' ``runtime_env`` then ``overrides`` on top.

    Precedence: leaf defaults < overrides. (The live-env layer — env >
    profile/leaf — is applied later at integration, not here.) ``${VAR}``
    is expanded against the merged map after layering, so a leaf may reference
    a key defined by another leaf or by an override.
    """
    merged: dict[str, str] = {}
    for leaf in registry.expand(selected_leaf_ids):
        for k, v in leaf.runtime_env.items():
            merged[k] = str(v)
    if overrides:
        for k, v in overrides.items():
            merged[k] = str(v)

    # Two passes so values can reference keys defined later in the map.
    expanded = dict(merged)
    for _ in range(2):
        changed = False
        for k, v in list(expanded.items()):
            new_v = _expand(v, expanded)
            if new_v != v:
                expanded[k] = new_v
                changed = True
        if not changed:
            break
    return expanded


def _backend_family(leaf: Leaf) -> str | None:
    """Coarse backend family for a leaf, or ``None`` when not derivable.

    Derived (in priority order) from:
      * ``LANGUAGE_MODE=rk`` in the leaf ``runtime_env`` → ``"rk"`` (an rk leaf is
        unambiguous regardless of its backend label);
      * the backend prefix before the first ``.`` (``jetson.*`` → ``"jetson"``,
        ``rk.*`` → ``"rk"``, ``cpu.*`` → ``"cpu"``).

    Returns ``None`` for backends with no ``.`` prefix (e.g. the bare
    ``sensevoice`` shared sub-leaf, or anything legacy) so the coupling check is
    OPT-IN: leaves whose family cannot be derived never trigger a rejection.
    Existing Jetson Qwen3 leaves all map to ``"jetson"`` (same family → no-op).
    """
    if str(leaf.runtime_env.get("LANGUAGE_MODE", "")).lower() == "rk":
        return "rk"
    backend = leaf.backend or ""
    if "." in backend:
        return backend.split(".", 1)[0]
    return None


def validate_composition(
    device: str,
    selected_leaf_ids: Iterable[str],
    registry: Registry,
    *,
    admission: str | None = None,
) -> CompositionPlan:
    """Validate a composition for ``device``; raise :class:`CompositionError`.

    Rejects, with a clear message:
      * unknown/unbuilt leaf id (never a silent fallback);
      * two leaves contributing the same capability (illegal pairing);
      * two leaves whose derivable backend FAMILY is incompatible (e.g. a
        ``jetson.*`` leaf paired with an ``rk.*`` leaf, or a ``LANGUAGE_MODE=rk``
        leaf with a non-rk one) — see :func:`_backend_family`;
      * two leaves on the same device that BOTH declare ``resources.exclusive:
        npu`` unless the caller passes ``admission="serial"`` (the RK/RPi NPU is
        a single shared accelerator and cannot run two models concurrently);
      * ``base_reservation_mb`` + sum of leaf DELTA ``peak_unified_mb`` over the
        device memory headroom.

    Leaf ``peak_unified_mb`` is an INCREMENTAL footprint (delta over the device
    idle baseline). The device's ``base_reservation_mb`` (idle baseline +
    non-leaf resident floor) is added ONCE so the shared baseline is not
    double-counted across leaves. Shared ``requires`` sub-leaves are included in
    the delta sum (each once) but do NOT count as a capability for the pairing
    check. Returns a :class:`CompositionPlan` on success.

    The family-coupling and NPU-exclusive checks are OPT-IN: they only fire when
    the relevant fields are present/derivable, so existing Jetson Qwen3 leaves
    (which declare neither an ``rk`` family conflict nor ``resources.exclusive``)
    are unaffected.
    """
    selected = tuple(selected_leaf_ids)

    dev = registry.devices.get(device)
    if dev is None:
        raise CompositionError(
            f"unknown device {device!r} — not in the device registry"
        )

    # Top-level (directly selected) leaves drive the capability-pairing check.
    by_capability: dict[str, str] = {}
    for lid in selected:
        leaf = registry.get_leaf(lid)  # raises on unknown id
        existing = by_capability.get(leaf.capability)
        if existing is not None:
            raise CompositionError(
                f"illegal pairing: leaves {existing!r} and {lid!r} both "
                f"provide capability {leaf.capability!r} on {device!r}"
            )
        by_capability[leaf.capability] = lid

    # Backend-family coupling: every directly-selected leaf with a derivable
    # family must agree. (Shared sub-leaves are excluded — they often have a
    # nominal/blank backend and are pulled by a concrete leaf of a known family.)
    families: dict[str, str] = {}  # family -> first leaf id with that family
    for lid in selected:
        fam = _backend_family(registry.get_leaf(lid))
        if fam is None:
            continue
        families.setdefault(fam, lid)
    if len(families) > 1:
        pairs = ", ".join(f"{lid!r}={fam}" for fam, lid in families.items())
        raise CompositionError(
            f"incompatible backend families on {device!r}: {pairs}. "
            f"A composition must not mix backend families (e.g. jetson.* with "
            f"rk.*, or a LANGUAGE_MODE=rk leaf with a non-rk leaf)."
        )

    # RK/RPi exclusive-resource: two leaves on the same device that BOTH claim
    # an exclusive accelerator (resources.exclusive: npu) cannot run concurrently
    # unless the composition opts into serial admission.
    exclusive: dict[str, list[str]] = {}
    for lid in selected:
        res = registry.get_leaf(lid).resources.get("exclusive")
        if res:
            exclusive.setdefault(str(res), []).append(lid)
    for resource, holders in exclusive.items():
        if len(holders) > 1 and admission != "serial":
            raise CompositionError(
                f"exclusive-resource conflict on {device!r}: leaves {holders} "
                f"both declare resources.exclusive={resource!r} (a single shared "
                f"accelerator). Pass admission='serial' to run them serialized, "
                f"or select only one."
            )

    # Memory: device base reservation (idle baseline + non-leaf floor, counted
    # once) plus the sum of leaf DELTA footprints over the full expansion
    # (selected + shared sub-leaves, each once). Precision is resolved per leaf
    # for the plan.
    expanded = registry.expand(selected)
    delta = sum(leaf.peak_unified_mb for leaf in expanded)
    peak = dev.base_reservation_mb + delta
    headroom = int(dev.unified_mb * registry.headroom_fraction)
    if peak > headroom:
        raise CompositionError(
            f"composition exceeds memory budget on {device!r}: "
            f"base_reservation={dev.base_reservation_mb} MB + "
            f"sum(delta peak_unified_mb)={delta} MB = {peak} MB > "
            f"headroom={headroom} MB "
            f"({int(registry.headroom_fraction * 100)}% of "
            f"{dev.unified_mb} MB unified). Leaves: {[l.id for l in expanded]}"
        )

    resolved = tuple(
        ResolvedLeaf(leaf=leaf, precision=registry.resolve_precision(leaf, device))
        for leaf in expanded
    )
    return CompositionPlan(
        device=device,
        leaf_ids=selected,
        resolved=resolved,
        peak_unified_mb=peak,
        headroom_mb=headroom,
    )
