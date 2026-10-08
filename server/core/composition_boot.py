"""Boot-time integration seam for leaf compositions (TRACK 1 SLICE 2 — gated).

This is the thin glue that wires :mod:`server.core.leaf_composition` (a pure,
standalone resolver shipped in SLICE 1) into the server startup sequence —
*gated* behind an optional ``composition`` profile key. A profile WITHOUT a
``composition`` block takes no new code path: every existing (flat) profile is
byte-for-byte unchanged and no profile is migrated in this slice.

When a profile DOES carry a ``composition`` block::

    "composition": {
        "device":   "orin-nx",
        "asr":      "asr.qwen3_asr.orin-nx.n2",
        "tts":      "tts.qwen3_tts.orin-nx.n2",
        "vad":      "...",            # optional, treated like any other leaf id
        "overrides": {"K": "V"}       # optional env overrides (leaf < override)
    }

:func:`apply_composition` then, in order:

  1. ``load_registry()`` — read the on-disk leaf/device/model registry.
  2. ``validate_composition(device, [leaf ids])`` — fail fast on a bad
     composition (unknown/unbuilt leaf, illegal capability pairing, memory over
     budget). The error propagates so boot ABORTS with a clear message.
  3. ``resolve_env(selected, overrides)`` — merge ``leaf < overrides`` and apply
     the result to ``os.environ`` with the system's env precedence: an env var
     ALREADY present in ``os.environ`` WINS over the leaf-derived value (i.e.
     we only set keys that are not already there — ``setdefault`` semantics,
     mirroring how :mod:`server.core.profile_loader` treats operator-owned env
     keys as authoritative).
  4. ``resolve_pull(selected)`` — return the de-duped UNION of artifact files
     the downloader must provision, so the caller can fold it into the existing
     model-download step additively without disturbing the flat path.

If the profile has no ``composition`` key this function is a strict no-op:
``os.environ`` is untouched, no registry is loaded, the validator never runs,
and an empty file list is returned.
"""

from __future__ import annotations

import logging
import json
import os
from typing import Mapping

from server.core import leaf_composition as lc

logger = logging.getLogger(__name__)

# Capability keys recognised inside a ``composition`` block, in a stable order
# so the selected-leaf list (and thus pull/env resolution) is deterministic.
# Any of these may be absent; only present, truthy values become leaf ids.
_CAPABILITY_KEYS: tuple[str, ...] = ("asr", "tts", "vad")

# Additive, non-disturbing env key that publishes the resolved union-pull file
# list for observability / downstream consumption. It does NOT exist on the
# flat path, so flat profiles never see it.
PULL_FILES_ENV = "OVS_COMPOSITION_PULL_FILES"
MODEL_SOURCES_ENV = "OVS_COMPOSITION_MODEL_SOURCES"


def selected_leaf_ids(composition: Mapping[str, object]) -> list[str]:
    """Extract the ordered list of selected leaf ids from a composition block.

    Reads the recognised capability keys (``asr``/``tts``/``vad``) in a stable
    order; absent or empty entries are skipped. ``overrides``/``device`` are
    not leaf ids and are ignored here.
    """
    ids: list[str] = []
    for cap in _CAPABILITY_KEYS:
        value = composition.get(cap)
        if value:
            ids.append(str(value))
    return ids


def model_sources(profile: Mapping[str, object] | None) -> list[lc.ModelArtifactSource]:
    composition = (profile or {}).get("composition")
    if not isinstance(composition, Mapping):
        return []
    selected = selected_leaf_ids(composition)
    if not selected:
        return []
    return lc.resolve_model_sources(selected, lc.load_registry())


def apply_composition(profile: Mapping[str, object] | None) -> list[str]:
    """Validate + apply a profile's ``composition`` block; return extra pulls.

    GATED: if ``profile`` has no truthy ``composition`` key this is a no-op and
    returns ``[]`` (``os.environ`` untouched, registry never loaded, validator
    never invoked).

    When composition mode IS active:
      * loads the registry, validates the composition (raising
        :class:`leaf_composition.CompositionError` to ABORT boot on a bad
        composition), applies the resolved env to ``os.environ`` with
        env-wins-over-leaf precedence, and returns the de-duped union-pull
        file list for the downloader.

    Raises:
        leaf_composition.CompositionError: invalid composition (boot must abort).
        leaf_composition.RegistryError: malformed on-disk registry.
    """
    composition = (profile or {}).get("composition")
    if not composition:
        return []
    if not isinstance(composition, Mapping):
        raise lc.CompositionError(
            f"'composition' must be a mapping, got {type(composition)!r}"
        )

    device = composition.get("device")
    if not device:
        raise lc.CompositionError("composition block missing required 'device'")
    device = str(device)

    selected = selected_leaf_ids(composition)
    if not selected:
        raise lc.CompositionError(
            f"composition on {device!r} selects no leaves "
            f"(need at least one of {_CAPABILITY_KEYS})"
        )

    overrides_raw = composition.get("overrides") or {}
    if not isinstance(overrides_raw, Mapping):
        raise lc.CompositionError("composition 'overrides' must be a mapping")
    overrides = {str(k): str(v) for k, v in overrides_raw.items()}

    registry = lc.load_registry()

    # (b) Fail fast on a bad composition — the exception propagates so boot
    # aborts with a clear message (unknown/unbuilt leaf, illegal pairing,
    # over-budget memory). No env is mutated before this point.
    plan = lc.validate_composition(device, selected, registry)

    # (c) Apply resolved env with env-wins precedence: an env var already in
    # os.environ is authoritative (operator/compose/.env/shell owns it), so we
    # only fill in keys that are NOT already present (setdefault semantics).
    #
    # resolve_env merges leaf < overrides and expands ${VAR} *within that map*
    # only (it deliberately does not read os.environ — see SLICE 1). Leaf env
    # commonly references ${QWEN3_ARTIFACT_ROOT}, which is supplied by the
    # operator/profile in os.environ, so we do a final os.environ-backed
    # expansion here — mirroring profile_loader._expand_with_profile_env, where
    # os.environ is the fallback layer for vars the profile/leaf doesn't define.
    resolved_env = lc.resolve_env(selected, registry, overrides)
    applied: list[str] = []
    skipped: list[str] = []
    for key, value in resolved_env.items():
        if key in os.environ:
            skipped.append(key)
            continue
        os.environ[key] = lc._expand(value, os.environ)
        applied.append(key)

    # (d) Union-pull file list for the downloader (de-duped, deterministic).
    pull_files = lc.resolve_pull(selected, registry)
    # Publish additively for observability / future consumption. Harmless on
    # the flat path (this key is never set there).
    if pull_files:
        os.environ[PULL_FILES_ENV] = os.pathsep.join(pull_files)
    sources = lc.resolve_model_sources(selected, registry)
    if sources:
        os.environ[MODEL_SOURCES_ENV] = json.dumps([
            {
                "model_id": source.model_id, "repo": source.repo,
                "revision": source.revision, "canonical_model_id": source.canonical_id,
                "root": source.root, "manifest": source.manifest,
                "cache_root": source.cache_root, "files": list(source.files),
                "strict": source.strict,
            }
            for source in sources
        ], separators=(",", ":"))

    logger.info(
        "composition ACTIVE on %s: leaves=%s peak=%dMB headroom=%dMB | "
        "env applied=%d skipped(env-wins)=%d | pull_files=%d",
        plan.device,
        list(plan.leaf_ids),
        plan.peak_unified_mb,
        plan.headroom_mb,
        len(applied),
        len(skipped),
        len(pull_files),
    )
    if skipped:
        logger.info(
            "composition: %d env key(s) left to pre-existing os.environ "
            "(env wins over leaf): %s",
            len(skipped),
            skipped,
        )

    return pull_files
