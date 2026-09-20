"""Tie the RK language matrix, its profiles, and the artifact manifest together.

Picking a language on an RK board walks three files that nothing else compares:
``configs/matrix/language_device.yaml`` names a profile, that profile names an
``RK_ARTIFACT_SET``, and ``deploy/artifacts/rk_manifest.json`` defines the set
whose ``runtime_contract.env`` the service validates the live env against at
startup (``server/core/rk_artifacts.py``). Each half is checked at runtime, but
only on a device, only for the set actually selected, and the contract check is
skipped entirely when artifact download is disabled -- so a mismatch introduced
here reaches hardware before anything complains.

The sets also duplicate their ASR halves: a set is everything one profile
downloads, so an ASR+Piper set repeats the ASR files of the ASR+Matcha set for
the same SoC. That duplication is deliberate and currently byte-identical; it is
also exactly the shape that drifts silently when ASR artifacts are next
republished and only one of the two sets is updated.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / "configs" / "matrix" / "language_device.yaml"
PROFILE_DIR = REPO_ROOT / "configs" / "profiles"
RK_MANIFEST = REPO_ROOT / "deploy" / "artifacts" / "rk_manifest.json"

RK_DEVICES = ("rk3576", "rk3588")


def _matrix_cells():
    cells = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))["cells"]
    return [c for c in cells if c.get("device") in RK_DEVICES and c.get("ovs_profile")]


def _manifest():
    return json.loads(RK_MANIFEST.read_text(encoding="utf-8"))


def _asr_files(file_list):
    return [f for f in file_list if "/asr/" in f["path"]]


@pytest.mark.parametrize(
    "cell", _matrix_cells(), ids=lambda c: f"{c['device']}-{c.get('group')}"
)
def test_matrix_cell_resolves_to_a_profile_for_the_same_soc(cell) -> None:
    profile_path = PROFILE_DIR / f"{cell['ovs_profile']}.json"
    assert profile_path.is_file(), (
        f"{cell['device']}/{cell.get('group')} names profile "
        f"{cell['ovs_profile']!r}, which does not exist"
    )
    env = json.loads(profile_path.read_text(encoding="utf-8")).get("env", {})
    assert env.get("RK_PLATFORM") == cell["device"], (
        f"{profile_path.name} declares RK_PLATFORM={env.get('RK_PLATFORM')!r} but "
        f"the matrix offers it for {cell['device']}"
    )


@pytest.mark.parametrize(
    "cell", _matrix_cells(), ids=lambda c: f"{c['device']}-{c.get('group')}"
)
def test_profile_artifact_set_exists_and_agrees_with_the_profile(cell) -> None:
    profile = json.loads(
        (PROFILE_DIR / f"{cell['ovs_profile']}.json").read_text(encoding="utf-8")
    )
    env = profile.get("env", {})
    set_name = env.get("RK_ARTIFACT_SET")
    assert set_name, f"{cell['ovs_profile']} declares no RK_ARTIFACT_SET"

    sets = _manifest()["artifact_sets"]
    assert set_name in sets, (
        f"{cell['ovs_profile']} wants artifact set {set_name!r}; the manifest has "
        f"{sorted(sets)}"
    )
    spec = sets[set_name]
    assert spec["soc"] == cell["device"], (
        f"set {set_name!r} is for {spec['soc']}, but {cell['ovs_profile']} runs on "
        f"{cell['device']}"
    )

    # The contract is what rk_artifacts.py checks the live env against, and it
    # fails the service closed when they disagree. Catch it here instead.
    contract = spec.get("runtime_contract", {}).get("env", {})
    mismatched = {
        k: (contract[k], env.get(k))
        for k in contract
        if str(env.get(k)) != str(contract[k])
    }
    assert not mismatched, (
        f"{cell['ovs_profile']} and artifact set {set_name!r} disagree; "
        f"RK_ARTIFACT_CONTRACT_STRICT would refuse to start. "
        f"key: (set, profile) = {mismatched}"
    )


def test_sets_sharing_a_soc_ship_the_same_asr_half() -> None:
    """ASR files repeated across sets for one SoC must stay identical."""
    sets = _manifest()["artifact_sets"]
    by_soc: dict[str, list[tuple[str, list]]] = {}
    for name, spec in sets.items():
        if spec.get("soc") not in RK_DEVICES:
            continue
        asr = _asr_files(spec.get("files") or [])
        if asr:
            by_soc.setdefault(spec["soc"], []).append((name, asr))

    for soc, entries in by_soc.items():
        # Compare each set against the first that shares its ASR path set: sets
        # built on a different ASR stack (e.g. paraformer) legitimately differ,
        # so only sets naming the SAME files have to agree on their hashes.
        by_paths: dict[tuple, list[tuple[str, list]]] = {}
        for name, asr in entries:
            by_paths.setdefault(tuple(sorted(f["path"] for f in asr)), []).append(
                (name, asr)
            )
        for paths, group in by_paths.items():
            if len(group) < 2:
                continue
            base_name, base = group[0]
            base_map = {f["path"]: (f["sha256"], f["size_bytes"]) for f in base}
            for name, asr in group[1:]:
                other = {f["path"]: (f["sha256"], f["size_bytes"]) for f in asr}
                assert other == base_map, (
                    f"{soc}: sets {base_name!r} and {name!r} list the same ASR "
                    f"files with different hashes/sizes — one was republished "
                    f"without the other"
                )
