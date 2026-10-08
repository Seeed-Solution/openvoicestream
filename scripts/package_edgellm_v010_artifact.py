#!/usr/bin/env python3
"""Package one TensorRT-Edge-LLM v0.10 model payload.

The model repositories used by the Edge-LLM runtime contain a small, explicit
manifest beside an immutable archive.  This helper creates that layout from
an *exact* payload directory.  It is intentionally fail-closed:

* the source tree must contain regular files only (no symlinks or special
  files);
* the destination must not exist before the operation starts; and
* all metadata and hashes are written only after the archive has been
  generated and verified.

The archive is deterministic.  Members are sorted by their POSIX relative
path and use fixed uid/gid, mtime, owner names, and modes.  The output is:

    <output>/payload.tar
    <output>/manifest.json
    <output>/SHA256SUMS

``manifest.json`` follows the schema-v2 model artifact contract consumed by
``server.core.qwen3_artifact_downloader``.  The archive is deliberately not
placed inside the archive itself; it is the separately locked ``payload``
entry in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
PAYLOAD_NAME = "payload.tar"
MANIFEST_NAME = "manifest.json"
SUMS_NAME = "SHA256SUMS"
_CHUNK_SIZE = 8 * 1024 * 1024


class PackageError(ValueError):
    """Raised when an artifact cannot be packaged safely."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise PackageError(f"cannot inspect {path}: {exc}") from exc


def _reject_symlink(path: Path, *, what: str) -> None:
    mode = _lstat(path).st_mode
    if stat.S_ISLNK(mode):
        raise PackageError(f"{what} contains a symbolic link: {path}")


def _validate_source_root(source: Path) -> Path:
    source = source.expanduser()
    if not source.exists():
        raise PackageError(f"payload directory does not exist: {source}")
    _reject_symlink(source, what="payload directory")
    if not source.is_dir():
        raise PackageError(f"payload path is not a directory: {source}")
    return source.resolve()


def _validate_destination(output: Path, source: Path) -> Path:
    """Validate a not-yet-created destination and return its resolved path."""

    output = output.expanduser()
    # ``Path.exists`` misses a dangling symlink.  lstat/lexists catches every
    # pre-existing path, which is important for the no-overwrite contract.
    if os.path.lexists(output):
        raise PackageError(f"output directory already exists; refusing to overwrite: {output}")

    resolved = output.resolve(strict=False)
    if resolved == source or source in resolved.parents:
        raise PackageError(
            "output directory must not be the payload directory or a child of it"
        )
    return resolved


def _relative_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive, traversal is rejected earlier
        raise PackageError(f"payload path escapes source directory: {path}") from exc
    # ``Path`` on the build hosts is POSIX, but spelling this explicitly keeps
    # manifests portable if the helper is exercised elsewhere.
    value = relative.as_posix()
    if not value or value == "." or value.startswith("../") or value == "..":
        raise PackageError(f"invalid payload-relative path: {path}")
    return value


def collect_payload_files(source: Path) -> list[tuple[str, Path]]:
    """Validate and return all regular payload files in deterministic order.

    Directories are traversed with ``os.scandir`` so every entry is inspected
    with ``lstat``.  This avoids silently following a symlinked directory and
    makes special files (FIFO/device/socket) fail before any output exists.
    Empty directories are harmless, but an entirely empty payload is not a
    valid model artifact and is rejected.
    """

    source = _validate_source_root(source)
    files: list[tuple[str, Path]] = []
    directories: list[Path] = [source]
    while directories:
        directory = directories.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise PackageError(f"cannot read payload directory {directory}: {exc}") from exc

        for entry in entries:
            path = Path(entry.path)
            mode = _lstat(path).st_mode
            if stat.S_ISLNK(mode):
                raise PackageError(f"payload contains a symbolic link: {_relative_path(path, source)}")
            if stat.S_ISDIR(mode):
                directories.append(path)
                continue
            if not stat.S_ISREG(mode):
                raise PackageError(
                    "payload contains a non-regular file: "
                    f"{_relative_path(path, source)}"
                )
            files.append((_relative_path(path, source), path))

    files.sort(key=lambda item: item[0])
    if not files:
        raise PackageError("payload directory is empty; at least one regular file is required")
    return files


def _collect_directories(source: Path) -> list[str]:
    """Return all non-root directory members in deterministic order."""

    directories: list[str] = []
    for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        mode = _lstat(path).st_mode
        if stat.S_ISLNK(mode):
            # ``collect_payload_files`` emits the user-facing error.  Keep
            # this guard for callers that use this helper independently.
            raise PackageError(f"payload contains a symbolic link: {_relative_path(path, source)}")
        if stat.S_ISDIR(mode):
            directories.append(_relative_path(path, source))
        elif not stat.S_ISREG(mode):
            raise PackageError(
                "payload contains a non-regular file: "
                f"{_relative_path(path, source)}"
            )
    return directories


def _tar_info(
    name: str, *, is_dir: bool, size: int = 0, source_mode: int | None = None
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE if is_dir else tarfile.REGTYPE
    if is_dir:
        info.mode = 0o755
    else:
        info.mode = 0o755 if source_mode is not None and source_mode & 0o111 else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.size = 0 if is_dir else size
    return info


def write_deterministic_payload_tar(
    source: Path, files: Iterable[tuple[str, Path]], archive: Path
) -> None:
    """Write a deterministic uncompressed USTAR archive for *source*."""

    file_entries = list(files)
    directory_names = _collect_directories(source)
    members: list[tuple[str, bool, Path | None]] = [
        (name, True, None) for name in directory_names
    ]
    members.extend((name, False, path) for name, path in file_entries)
    members.sort(key=lambda item: item[0])

    with tarfile.open(archive, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, is_dir, path in members:
            if is_dir:
                tar.addfile(_tar_info(name + "/", is_dir=True))
                continue
            assert path is not None
            # Re-stat immediately before reading.  A source tree changing
            # during packaging must not silently produce a stale manifest.
            before = _lstat(path)
            if not stat.S_ISREG(before.st_mode):
                raise PackageError(f"payload file changed to non-regular: {name}")
            info = _tar_info(
                name, is_dir=False, size=before.st_size, source_mode=before.st_mode
            )
            with path.open("rb") as stream:
                tar.addfile(info, stream)
            after = _lstat(path)
            if (
                after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or bool(after.st_mode & 0o111) != bool(before.st_mode & 0o111)
            ):
                raise PackageError(f"payload file changed while packaging: {name}")


def _parse_json_or_text(value: str, *, option: str) -> Any:
    """Parse a metadata value as JSON when it is clearly JSON, else a string.

    This keeps the CLI convenient for scalar source/profile values while also
    allowing structured provenance (for example ``{"upstream_sha": ...}``).
    A leading ``@`` reads JSON from a file, which avoids shell quoting for
    provenance records.
    """

    value = value.strip()
    if not value:
        raise PackageError(f"{option} must not be empty")
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        if not path.is_file() or path.is_symlink():
            raise PackageError(f"{option} JSON file is missing or not regular: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PackageError(f"cannot parse {option} JSON file {path}: {exc}") from exc
    if value[0] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise PackageError(f"{option} must be valid JSON: {exc}") from exc
    return value


def _validate_metadata(value: Any, *, option: str) -> Any:
    if isinstance(value, str):
        if not value.strip():
            raise PackageError(f"{option} must not be empty")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        raise PackageError(f"{option} must be a non-empty string or JSON object/array")
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise PackageError(f"{option} is not JSON serializable") from exc
    return value


def _validate_repo_prefix(value: str) -> str:
    """Validate a repository-relative prefix without normalizing unsafe input."""

    if not isinstance(value, str):
        raise PackageError("--repo-prefix must be a string")
    if not value:
        return ""
    if "\x00" in value or "\\" in value or value.startswith("/") or value.endswith("/"):
        raise PackageError("--repo-prefix must be a safe relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PackageError("--repo-prefix must be a safe relative path")
    return value


def _file_inventory(files: Iterable[tuple[str, Path]]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for relative, path in files:
        size = path.stat().st_size
        inventory[relative] = {"sha256": sha256_file(path), "size": size}
    return dict(sorted(inventory.items()))


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def package_artifact(
    payload_dir: Path,
    output_dir: Path,
    *,
    artifact: str,
    model_id: str,
    repo: str,
    source: Any,
    profile: Any,
    provenance: Any,
    engine_contract: Any | None = None,
    repo_prefix: str = "",
) -> dict[str, Any]:
    """Package a payload and return the generated manifest.

    The destination is committed by one final rename.  If validation or
    hashing fails, the caller gets no partially populated output directory.
    """

    source_root = _validate_source_root(Path(payload_dir))
    destination = _validate_destination(Path(output_dir), source_root)
    artifact_value = _validate_metadata(artifact, option="--artifact")
    model_value = _validate_metadata(model_id, option="--model-id")
    repo_value = _validate_metadata(repo, option="--repo")
    source_value = _validate_metadata(source, option="--source")
    profile_value = _validate_metadata(profile, option="--profile")
    provenance_value = _validate_metadata(provenance, option="--provenance")
    repo_prefix_value = _validate_repo_prefix(repo_prefix)
    contract_value = None
    if engine_contract is not None:
        contract_value = _validate_metadata(
            engine_contract, option="--engine-contract"
        )
        if not isinstance(contract_value, dict):
            raise PackageError("--engine-contract must be a JSON object")
    files = collect_payload_files(source_root)

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(parent)))
    try:
        archive = temporary / PAYLOAD_NAME
        inventory = _file_inventory(files)
        write_deterministic_payload_tar(source_root, files, archive)
        # The source tree is operator-provided and can be modified by another
        # process while packaging.  Re-hash after archive creation so the
        # manifest can never describe bytes different from the archive.
        final_inventory = _file_inventory(files)
        if final_inventory != inventory:
            raise PackageError("payload changed while packaging; refusing to publish a mixed artifact")
        payload_size = archive.stat().st_size
        payload_sha = sha256_file(archive)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": artifact_value,
            "model_id": model_value,
            "hf_repo_id": repo_value,
            "source_revision": source_value,
            "engine_profile": profile_value,
            "provenance": provenance_value,
            "files": inventory,
            "payload": {
                "path": f"{repo_prefix_value}/{PAYLOAD_NAME}" if repo_prefix_value else PAYLOAD_NAME,
                "sha256": payload_sha,
                "size": payload_size,
            },
        }
        if contract_value is not None:
            manifest["engine_contract"] = contract_value
        manifest_text = json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        _write_text(temporary / MANIFEST_NAME, manifest_text)

        # This is a repository-root checksum file.  The source files are
        # members of payload.tar and therefore do not exist at repository
        # root after upload; listing them here would make ``sha256sum -c``
        # fail for every valid artifact.  The manifest and archive are the
        # two root files that must be checked before extraction.
        checksum_entries = [
            (MANIFEST_NAME, sha256_file(temporary / MANIFEST_NAME)),
            (PAYLOAD_NAME, payload_sha),
        ]
        checksum_entries.sort(key=lambda item: item[0])
        sums_text = "".join(f"{digest}  {relative}\n" for relative, digest in checksum_entries)
        _write_text(temporary / SUMS_NAME, sums_text)

        # Refuse a race that creates the destination after the initial check.
        if os.path.lexists(destination):
            raise PackageError(
                f"output directory appeared during packaging; refusing to overwrite: {destination}"
            )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return manifest


def _add_value_option(
    parser: argparse.ArgumentParser, *flags: str, dest: str, required: bool = True
) -> None:
    parser.add_argument(*flags, dest=dest, required=required)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload_dir", nargs="?", type=Path)
    parser.add_argument("output_dir", nargs="?", type=Path)
    parser.add_argument("--payload-dir", dest="payload_dir_option", type=Path)
    parser.add_argument("--output-dir", dest="output_dir_option", type=Path)
    _add_value_option(parser, "--artifact", "--artifact-id", dest="artifact")
    _add_value_option(parser, "--model", "--model-id", dest="model_id")
    _add_value_option(parser, "--repo", "--repo-id", dest="repo")
    _add_value_option(parser, "--source", "--source-revision", dest="source")
    _add_value_option(parser, "--profile", "--engine-profile", dest="profile")
    _add_value_option(
        parser,
        "--provenance",
        "--provenance-json",
        dest="provenance",
        required=False,
    )
    parser.add_argument("--provenance-file", dest="provenance_file")
    parser.add_argument(
        "--repo-prefix",
        default="",
        help="safe repository-relative prefix for the payload archive path",
    )
    parser.add_argument(
        "--engine-contract",
        help="optional JSON object consumed by profile-aware runtimes",
    )
    parser.add_argument(
        "--engine-contract-file",
        help="read the optional engine contract JSON object from this file",
    )
    return parser


def _resolve_cli_path(
    positional: Path | None, option: Path | None, *, label: str
) -> Path:
    if positional is not None and option is not None:
        raise PackageError(f"provide {label} either positionally or with --{label}, not both")
    value = option if option is not None else positional
    if value is None:
        raise PackageError(f"missing {label}; provide it positionally or with --{label}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload_dir = _resolve_cli_path(
            args.payload_dir, args.payload_dir_option, label="payload-dir"
        )
        output_dir = _resolve_cli_path(
            args.output_dir, args.output_dir_option, label="output-dir"
        )
        if (args.provenance is None) == (args.provenance_file is None):
            raise PackageError("provide exactly one of --provenance or --provenance-file")
        if args.engine_contract is not None and args.engine_contract_file is not None:
            raise PackageError(
                "provide --engine-contract or --engine-contract-file, not both"
            )
        provenance_arg = (
            args.provenance
            if args.provenance is not None
            else "@" + str(args.provenance_file).lstrip("@")
        )
        contract_arg = None
        if args.engine_contract is not None:
            contract_arg = args.engine_contract
        elif args.engine_contract_file is not None:
            contract_arg = "@" + str(args.engine_contract_file).lstrip("@")
        manifest = package_artifact(
            payload_dir,
            output_dir,
            artifact=args.artifact,
            model_id=args.model_id,
            repo=args.repo,
            source=_parse_json_or_text(args.source, option="--source"),
            profile=_parse_json_or_text(args.profile, option="--profile"),
            provenance=_parse_json_or_text(provenance_arg, option="--provenance"),
            engine_contract=(
                _parse_json_or_text(contract_arg, option="--engine-contract")
                if contract_arg is not None
                else None
            ),
            repo_prefix=args.repo_prefix,
        )
    except PackageError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "files": len(manifest["files"]),
                "payload": manifest["payload"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
