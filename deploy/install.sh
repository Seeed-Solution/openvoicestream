#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  deploy/install.sh [--target auto|jetson|orin-nx|rk3576|rk3588|rpi] [--pull] [--build] [--verify]
  deploy/install.sh --target orin-nx --v011-llm-candidate --prepare-only
  deploy/install.sh --target orin-nx --legacy --pull --verify

Examples:
  deploy/install.sh --pull --verify
  deploy/install.sh --target jetson --pull --verify
  deploy/install.sh --target orin-nx --pull --verify
  deploy/install.sh --target rk3588 --pull --verify
  deploy/install.sh --target rpi --pull --verify

Environment overrides:
  LANGUAGE_MODE, OVS_PROFILE, RK_ARTIFACT_SET,
  QWEN3_ARTIFACT_SET, QWEN3_HF_REPO_ID, OVS_PORT,
  HF_ENDPOINT, PARAFORMER_PREROLL_MS, EDGELLM_ENGINE_PROFILE,
  EDGELLM_4K_ENGINE_REVISION, EDGELLM_8K_ENGINE_REVISION
  SLV_LLM_RUNTIME_REPO, SLV_LLM_RUNTIME_REVISION, SLV_LLM_RUNTIME_MANIFEST,
  SLV_LLM_RUNTIME_CONTRACT,
  SLV_LLM_CACHE_ROOT, SLV_LLM_PAYLOAD_ROOT, SLV_LLM_CHECKPOINT_ROOT,
  SLV_LLM_CHECKPOINT_INVENTORY, SLV_LLM_CONSUMER_OVERRIDE
EOF
}

warn() {
  echo "WARN: $*" >&2
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

detect_target() {
  if [[ -e /etc/nv_tegra_release ]]; then
    model="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
    if [[ "${model}" == *"Orin NX"* ]]; then
      echo "orin-nx"
    else
      echo "jetson"
    fi
    return
  fi
  if [[ -e /dev/rknpu ]]; then
    if grep -qi "rk3588" /proc/cpuinfo 2>/dev/null; then
      echo "rk3588"
    else
      echo "rk3576"
    fi
    return
  fi
  if [[ -r /proc/device-tree/model ]] && tr -d '\0' < /proc/device-tree/model | grep -qi "raspberry pi"; then
    echo "rpi"
    return
  fi
  case "$(uname -m)" in
    aarch64|arm64)
      warn "64-bit ARM host detected, but device family is ambiguous."
      ;;
  esac
  return 1
}

target="auto"
pull=0
build=0
verify=0
v011_llm_candidate=0
prepare_only=0
legacy=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="${2:-}"
      shift 2
      ;;
    --pull)
      pull=1
      shift
      ;;
    --build)
      build=1
      shift
      ;;
    --verify)
      verify=1
      shift
      ;;
    --v011-llm-candidate)
      v011_llm_candidate=1
      shift
      ;;
    --prepare-only)
      prepare_only=1
      shift
      ;;
    --legacy)
      legacy=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$v011_llm_candidate" -eq 1 ]]; then
  [[ "$prepare_only" -eq 1 ]] || die "v0.11 LLM candidate is not qualified for launch; use --prepare-only"
  [[ "$target" == "orin-nx" ]] || die "v0.11 LLM candidate requires --target orin-nx (Jetson Orin NX SM87)"
  [[ "$build" -eq 0 ]] || die "v0.11 candidate preparation never builds images"
  [[ "$pull" -eq 0 ]] || die "v0.11 candidate preparation does not use Compose registry pull"
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  project_root="$(cd "$script_dir/.." && pwd)"
  export SLV_LLM_PROJECT_ROOT="${SLV_LLM_PROJECT_ROOT:-$project_root}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  if [[ "$HF_ENDPOINT" != "https://hf-mirror.com" && "${SLV_LLM_ALLOW_LOCAL_FIXTURE:-0}" != "1" ]]; then
    die "v0.11 strict candidate preparation requires HF_ENDPOINT=https://hf-mirror.com"
  fi
  export SLV_LLM_RUNTIME_REPO="${SLV_LLM_RUNTIME_REPO:-harvestsu/Qwen3.5-4B-AWQ}"
  export SLV_LLM_RUNTIME_REVISION="${SLV_LLM_RUNTIME_REVISION:-41fb732368307f2a6aef3ba67d35aef027845b4a}"
  export SLV_LLM_RUNTIME_MANIFEST="${SLV_LLM_RUNTIME_MANIFEST:-v011/orin-nx-sm87/llm-ifb-b2-20261011-candidate-clean-checkpoint-consumer-path-r4/manifest.json}"
  export SLV_LLM_RUNTIME_MODEL_ID="${SLV_LLM_RUNTIME_MODEL_ID:-qwen3.5-4b}"
  export SLV_LLM_CACHE_ROOT="${SLV_LLM_CACHE_ROOT:-$project_root/.cache/slv-v011-llm}"
  export SLV_LLM_PAYLOAD_ROOT="${SLV_LLM_PAYLOAD_ROOT:-$SLV_LLM_CACHE_ROOT/payload}"
  export SLV_LLM_CHECKPOINT_ROOT="${SLV_LLM_CHECKPOINT_ROOT:-$SLV_LLM_CACHE_ROOT/checkpoint}"
  export SLV_LLM_RUNTIME_CONTRACT="${SLV_LLM_RUNTIME_CONTRACT:-$SLV_LLM_PAYLOAD_ROOT/manifest.json}"
  export SLV_LLM_CHECKPOINT_INVENTORY="${SLV_LLM_CHECKPOINT_INVENTORY:-$SLV_LLM_RUNTIME_CONTRACT}"
  export SLV_LLM_CONSUMER_OVERRIDE="${SLV_LLM_CONSUMER_OVERRIDE:-$SLV_LLM_PAYLOAD_ROOT/deploy/docker-compose.edgellm-v011-candidate-llm.consumer-override.yml}"
  export SLV_LLM_IMAGE_TAG="${SLV_LLM_IMAGE_TAG:-seeed-local-voice:edgellm-v011-cp312-b2-oct11-r3}"
  export SLV_LLM_IMAGE_ID="${SLV_LLM_IMAGE_ID:-sha256:7a9548e01db6255cb30d07ebff76ec53e0464a331ba284506483c03ea331c868}"
  export SLV_LLM_IMAGE_ARCHIVE_SHA256="${SLV_LLM_IMAGE_ARCHIVE_SHA256:-fae58ffd267f451fd06b4c2120e0b01bb9fccd0b30236c7f14c726532d364067}"
  export PYTHONPATH="${SLV_LLM_PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
  exec python3 - "$SLV_LLM_PROJECT_ROOT" <<'PY'
import hashlib, json, os, pathlib, re, stat, subprocess, sys, tarfile

def fail(msg):
    print("ERROR: v0.11 candidate preparation: " + msg, file=sys.stderr)
    raise SystemExit(1)

def required(name):
    value = os.environ.get(name, "").strip()
    if not value: fail("missing " + name)
    return value

project = pathlib.Path(sys.argv[1]).resolve()
repo = required("SLV_LLM_RUNTIME_REPO")
runtime_revision = required("SLV_LLM_RUNTIME_REVISION")
manifest = required("SLV_LLM_RUNTIME_MANIFEST")
if not re.fullmatch(r"[0-9a-f]{40}", runtime_revision): fail("runtime revision must be a pinned 40-hex commit")
cache_root = pathlib.Path(required("SLV_LLM_CACHE_ROOT")).expanduser()
payload_root = pathlib.Path(required("SLV_LLM_PAYLOAD_ROOT")).expanduser()
checkpoint_root = pathlib.Path(required("SLV_LLM_CHECKPOINT_ROOT")).expanduser()

def owned_components(path, label, create=False):
    """Check every existing component; never resolve through a symlink."""
    if not path.is_absolute(): fail(label + " must be absolute")
    missing = []
    cur = path
    while not cur.exists() and not cur.is_symlink():
        missing.append(cur); cur = cur.parent
    existing = []
    probe = cur
    while True:
        existing.append(probe)
        if probe.parent == probe: break
        probe = probe.parent
    for item in reversed(existing):
        st = item.lstat()
        if stat.S_ISLNK(st.st_mode): fail(label + " has symlinked ancestor: " + str(item))
        if not stat.S_ISDIR(st.st_mode): fail(label + " component is not a directory: " + str(item))
        if st.st_uid not in (os.getuid(), 0): fail(label + " component is not owned by current user or root: " + str(item))
        if (st.st_mode & 0o022) and not (st.st_mode & stat.S_ISVTX): fail(label + " component is writable by group/others: " + str(item))
    for item in reversed(missing):
        if create:
            item.mkdir(mode=0o700)
            item.chmod(0o700)
    if not path.is_dir() or path.is_symlink(): fail(label + " must be an owned directory")
    st = path.lstat()
    if st.st_uid != os.getuid() or (st.st_mode & 0o077): fail(label + " must be private and owned")

owned_components(cache_root, "cache root", create=True)
owned_components(checkpoint_root, "checkpoint root", create=True)
if payload_root.is_symlink():
    try: payload_root.resolve().relative_to(cache_root.resolve())
    except ValueError: fail("payload root symlink escapes owned cache root")
    owned_components(payload_root.resolve(), "payload root target", create=False)
else:
    owned_components(payload_root, "payload root", create=True)

sys.path.insert(0, str(project))
try:
    from server.core.qwen3_artifact_downloader import ensure_model_artifacts
    from server.core import hf_artifacts
except Exception as exc: fail("shared resolver imports failed: %s" % exc)
try:
    ensure_model_artifacts(os.environ["SLV_LLM_RUNTIME_MODEL_ID"], repo,
        revision=runtime_revision, manifest=manifest, root=str(payload_root),
        cache_root=str(cache_root), strict=True)
except Exception as exc: fail("runtime exact HF resolver failed: %s" % exc)

contract_path = pathlib.Path(required("SLV_LLM_RUNTIME_CONTRACT")).expanduser()
if contract_path.is_symlink() or not contract_path.is_file(): fail("runtime contract missing or symlinked: %s" % contract_path)
try: runtime_contract = json.loads(contract_path.read_text())
except Exception as exc: fail("runtime contract JSON invalid: %s" % exc)
if not isinstance(runtime_contract, dict): fail("runtime contract must be an object")
contract = runtime_contract.get("engine_contract", runtime_contract)
model = contract.get("model") if isinstance(contract, dict) else None
engine = contract.get("engine") if isinstance(contract, dict) else None
if not isinstance(model, dict) or not isinstance(engine, dict): fail("runtime contract model/engine schema missing")
if model.get("checkpoint_repo") != repo: fail("runtime contract checkpoint_repo does not match runtime repo")
contract_revision = model.get("revision")
if not isinstance(contract_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", contract_revision): fail("runtime contract checkpoint revision is not pinned")
model_dir = engine.get("model_dir")
if not isinstance(model_dir, str) or not model_dir.startswith("/"): fail("runtime contract engine.model_dir must be absolute")
inventory_path = pathlib.Path(os.environ.get("SLV_LLM_CHECKPOINT_INVENTORY", str(contract_path))).expanduser()
if inventory_path == contract_path:
    inventory = runtime_contract.get("checkpoint_inventory") or contract.get("checkpoint_inventory")
    if not isinstance(inventory, dict):
        inventory = (runtime_contract.get("provenance") or {}).get("external_checkpoint")
else:
    if inventory_path.is_symlink() or not inventory_path.is_file(): fail("checkpoint inventory with five full locks is required: %s" % inventory_path)
    try: inventory = json.loads(inventory_path.read_text())
    except Exception as exc: fail("checkpoint inventory JSON invalid: %s" % exc)
if not isinstance(inventory, dict): fail("published checkpoint inventory missing; automatic fetch is disabled")
checkpoint_revision = os.environ.get("SLV_LLM_CHECKPOINT_REVISION", contract_revision)
if checkpoint_revision != contract_revision: fail("checkpoint revision override differs from runtime contract")
if inventory.get("repo") != repo or inventory.get("revision") != checkpoint_revision: fail("checkpoint inventory repo/revision mismatch")
published_content_sha = model.get("checkpoint_content_sha256")
published_content_size = (runtime_contract.get("provenance") or {}).get("external_checkpoint", {}).get("content_size_bytes")
if inventory.get("content_sha256") != published_content_sha: fail("checkpoint inventory content SHA differs from runtime contract")
if inventory.get("content_size_bytes") != published_content_size: fail("checkpoint inventory content size differs from runtime contract")
locks = inventory.get("files")
if not isinstance(locks, dict) or not locks: fail("checkpoint inventory files are missing")
if len(locks) != 10: fail("checkpoint inventory must contain exactly ten file locks")
if inventory_path != contract_path:
    published = runtime_contract.get("checkpoint_inventory") or contract.get("checkpoint_inventory")
    if not isinstance(published, dict) or inventory != published: fail("external checkpoint inventory differs from published canonical inventory")
for rel, lock in sorted(locks.items()):
    p = pathlib.PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not isinstance(lock, dict): fail("unsafe checkpoint lock: " + rel)
    if not isinstance(lock.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", lock["sha256"]): fail("invalid checkpoint SHA: " + rel)
    if not isinstance(lock.get("size"), int) or isinstance(lock.get("size"), bool) or lock["size"] <= 0: fail("invalid checkpoint size: " + rel)
download_root = cache_root / "checkpoint-downloads"
owned_components(download_root, "checkpoint download staging", create=True)
def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def tree_digest(root):
    total = 0; digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            if path.is_dir() and path.name not in {".cache", "__pycache__"}: continue
            if path.is_symlink(): fail("checkpoint tree contains symlink: " + str(path))
            continue
        rel = path.relative_to(root).as_posix()
        if any(part in {".cache", "__pycache__"} for part in pathlib.PurePosixPath(rel).parts) or path.suffix in {".pyc", ".pyo"}: continue
        digest.update(rel.encode() + b"\0")
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                size += len(chunk)
                digest.update(chunk)
        total += size
        digest.update(b"\0")
    return digest.hexdigest(), total
for existing in checkpoint_root.rglob("*"):
    if existing.is_file() and existing.relative_to(checkpoint_root).as_posix() not in locks: fail("checkpoint root contains unknown file: " + str(existing))
for rel, lock in sorted(locks.items()):
    target = checkpoint_root / rel; staged = download_root / rel
    owned_components(target.parent, "checkpoint file parent", create=True); owned_components(staged.parent, "checkpoint staging parent", create=True)
    if target.is_symlink() or (target.exists() and not target.is_file()): fail("checkpoint destination is not a regular file: " + str(target))
    if target.exists():
        st = target.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or (st.st_mode & 0o077): fail("checkpoint completed file is not private and owned: " + rel)
        if st.st_size != lock["size"] or file_sha256(target) != lock["sha256"]: fail("checkpoint completed file identity/hash mismatch: " + rel)
    if target.exists(): continue
    try:
        hf_artifacts.download_file(rel, staged, expected_sha256=lock["sha256"], expected_size=lock["size"],
            repo=repo, revision=checkpoint_revision, endpoint=os.environ["HF_ENDPOINT"],
            allowed_redirect_hosts=__import__("server.core.qwen3_artifact_downloader", fromlist=["_strict_endpoint_hosts"])._strict_endpoint_hosts(os.environ["HF_ENDPOINT"]), resume=True,
            resume_key="%s|%s|%s|%s" % (download_root, repo, checkpoint_revision, rel))
        os.replace(staged, target)
    except Exception as exc: fail("checkpoint resume fetch failed for %s: %s" % (rel, exc))
actual_tree_sha, actual_tree_size = tree_digest(checkpoint_root)
if actual_tree_sha != inventory.get("content_sha256") or actual_tree_size != inventory.get("content_size_bytes"): fail("checkpoint tree aggregate identity mismatch")
print("CHECKPOINT_LOCKS=%d revision=%s root=%s tree_sha256=%s size=%d" % (len(locks), checkpoint_revision, checkpoint_root, actual_tree_sha, actual_tree_size))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()

meta = payload_root / "metadata/image-archive-metadata.json"
if not meta.is_file(): meta = payload_root / "source/metadata/image-archive-metadata.json"
if not meta.is_file(): fail("typed image archive metadata missing")
metadata = json.loads(meta.read_text())
image_id = os.environ["SLV_LLM_IMAGE_ID"]
archive_name = "llm-image-%s.tar" % image_id.split(":", 1)[1]
archive = payload_root / "image" / archive_name
if not archive.is_file(): archive = payload_root / "source" / "image" / archive_name
if not archive.is_file(): fail("verified image archive missing")
if archive.stat().st_size != int(metadata.get("archive_size_bytes", -1)): fail("image archive size mismatch")
if sha256(archive) != os.environ["SLV_LLM_IMAGE_ARCHIVE_SHA256"] or sha256(archive) != metadata.get("archive_sha256"): fail("image archive hash mismatch")
if metadata.get("archive_size_bytes") != archive.stat().st_size: fail("published archive size mismatch")
if metadata.get("docker_tag") != os.environ["SLV_LLM_IMAGE_TAG"] or metadata.get("docker_inspect_image_id") != image_id: fail("typed image identity mismatch")
if metadata.get("docker_inspect_repo_digest") != metadata.get("docker_tag", "").split(":", 1)[0] + "@" + image_id: fail("published repo digest mismatch")
try:
    with tarfile.open(archive, "r") as bundle:
        members = bundle.getmembers()
        if len(members) != int(metadata.get("oci_members", 29)): fail("OCI member count mismatch")
        names = [member.name for member in members]
        if len(names) != len(set(names)): fail("duplicate OCI archive member")
        for member in members:
            name = pathlib.PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk() or not (member.isdir() or member.isfile()): fail("unsafe OCI archive member: " + member.name)
            if member.name.startswith("blobs/sha256/") and member.isfile():
                expected = member.name.rsplit("/", 1)[1]
                extracted = bundle.extractfile(member)
                if extracted is None: fail("OCI blob cannot be read: " + member.name)
                digest = hashlib.sha256(extracted.read()).hexdigest()
                if digest != expected: fail("OCI blob digest mismatch: " + member.name)
        def blob(digest):
            if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest): fail("invalid OCI digest")
            path = "blobs/sha256/" + digest.split(":", 1)[1]
            member = bundle.getmember(path)
            raw = bundle.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != digest.split(":", 1)[1]: fail("OCI referenced blob digest mismatch: " + digest)
            return json.loads(raw)
        index = json.loads(bundle.extractfile(bundle.getmember("index.json")).read())
        top = index.get("manifests")
        if index.get("schemaVersion") != 2 or not isinstance(top, list) or len(top) != 1: fail("OCI index must contain one nested index")
        nested_digest = top[0].get("digest")
        nested = blob(nested_digest)
        if metadata.get("archive_index_digest") != nested_digest: fail("OCI index identity mismatch")
        choices = nested.get("manifests")
        if not isinstance(choices, list): fail("OCI nested index manifests missing")
        published_platform = metadata.get("platform")
        if not isinstance(published_platform, dict) or not published_platform.get("architecture") or not published_platform.get("os"): fail("published OCI platform metadata missing")
        target = contract.get("target", {}) if isinstance(contract, dict) else {}
        if published_platform.get("runtime_python_abi") and published_platform.get("runtime_python_abi") != target.get("python_abi", published_platform.get("runtime_python_abi")): fail("published OCI runtime ABI mismatch")
        platform = [x for x in choices if isinstance(x.get("platform"), dict) and x["platform"].get("architecture") == published_platform.get("architecture") and x["platform"].get("os") == published_platform.get("os")]
        if len(platform) != 1: fail("OCI index platform manifest is not unique")
        manifest_digest = platform[0].get("digest")
        if metadata.get("archive_manifest_digest") != manifest_digest: fail("OCI manifest identity mismatch")
        image_manifest = blob(manifest_digest)
        config_desc = image_manifest.get("config")
        config_digest = config_desc.get("digest") if isinstance(config_desc, dict) else None
        if metadata.get("archive_config_sha256") != str(config_digest).split(":", 1)[-1]: fail("OCI config identity mismatch")
        if metadata.get("archive_config_size_bytes") != config_desc.get("size"): fail("OCI config size mismatch")
        config = blob(config_digest)
        layers = image_manifest.get("layers")
        expected_layers = metadata.get("layer_digests")
        def canonical_layer(value):
            if isinstance(value, str) and value.startswith("blobs/sha256/") and len(value) == len("blobs/sha256/") + 64: return "sha256:" + value.rsplit("/", 1)[1]
            return value
        if not isinstance(layers, list) or not isinstance(expected_layers, list) or len(layers) != int(metadata.get("layers_count", -1)) or [canonical_layer(x) for x in expected_layers] != [x.get("digest") for x in layers]: fail("OCI layer identity mismatch")
        for descriptor in layers:
            digest = descriptor.get("digest")
            member = bundle.getmember("blobs/sha256/" + digest.split(":", 1)[1])
            if descriptor.get("size") != member.size: fail("OCI layer size mismatch")
        labels = config.get("config", {}).get("Labels", {})
        if not isinstance(labels, dict): fail("OCI config labels missing")
except (OSError, tarfile.TarError) as exc: fail("OCI archive validation failed: %s" % exc)
print("DOCKER_IMAGE_LOAD_LIST=1 archive=%s sha256=%s" % (archive, os.environ["SLV_LLM_IMAGE_ARCHIVE_SHA256"]))
existing = subprocess.run(["docker", "image", "inspect", os.environ["SLV_LLM_IMAGE_TAG"], "--format", "{{json .}}"], text=True, capture_output=True)
if existing.returncode == 0:
    try:
        old = json.loads(existing.stdout)
    except Exception: fail("existing candidate tag returned invalid JSON")
    if old.get("Id") != image_id: fail("candidate tag already exists with a different image ID")
    if not isinstance(old.get("RepoDigests"), list) or metadata.get("docker_inspect_repo_digest") not in old.get("RepoDigests"): fail("candidate tag already exists with a different repo digest")
    if not isinstance(old.get("RepoTags"), list) or os.environ["SLV_LLM_IMAGE_TAG"] not in old.get("RepoTags"): fail("candidate tag inspect returned a different tag")
    print("DOCKER_LOAD_REUSE=1 image_id=%s" % image_id)
else:
  try: subprocess.run(["docker", "load", "--input", str(archive)], check=True, text=True, capture_output=True)
  except FileNotFoundError: fail("docker is required for image load")
  except subprocess.CalledProcessError as exc: fail("docker load failed: %s" % exc.stderr.strip())
inspect = subprocess.run(["docker", "image", "inspect", os.environ["SLV_LLM_IMAGE_TAG"], "--format", "{{json .}}"], text=True, capture_output=True)
if inspect.returncode: fail("loaded candidate image cannot be inspected")
try: observed = json.loads(inspect.stdout)
except Exception: fail("docker inspect returned invalid JSON")
if observed.get("Id") != image_id: fail("loaded image ID mismatch")
if not isinstance(observed.get("RepoDigests"), list) or metadata.get("docker_inspect_repo_digest") not in observed.get("RepoDigests"): fail("loaded image repo digest mismatch")
if not isinstance(observed.get("RepoTags"), list) or os.environ["SLV_LLM_IMAGE_TAG"] not in observed.get("RepoTags"): fail("loaded image tag mismatch")
print("DOCKER_LOAD_RC=0 image_id=%s" % image_id)

override = pathlib.Path(required("SLV_LLM_CONSUMER_OVERRIDE")).expanduser()
if not override.is_file(): fail("consumer Compose override missing: %s" % override)
text = override.read_text()
for token in ("SLV_LLM_PAYLOAD_ROOT", "SLV_LLM_CHECKPOINT_ROOT", "EDGELLM_V011_MODEL_CONTAINER_DIR", "EDGELLM_V011_CACHE_DIR", "EDGELLM_V011_INPUT_CONTRACT", "EDGELLM_V011_CHECKPOINT"):
    if token not in text: fail("consumer override missing required binding: " + token)
def yaml_env(name):
    match = re.search(r"^\s+" + re.escape(name) + r"\s*:\s*([^#\n]+)", text, re.MULTILINE)
    return match.group(1).strip().strip("'\"") if match else None
if yaml_env("EDGELLM_V011_MODEL_CONTAINER_DIR") != model_dir: fail("consumer model container path differs from runtime contract")
if yaml_env("EDGELLM_V011_CHECKPOINT") != model_dir: fail("consumer checkpoint path differs from runtime contract")
contract_cache = engine.get("cache_dir")
if not isinstance(contract_cache, str) or not contract_cache.startswith("/"): fail("runtime contract engine.cache_dir must be absolute")
if yaml_env("EDGELLM_V011_CACHE_DIR") != contract_cache: fail("consumer cache path differs from runtime contract")
if not re.search(r"(?:\$\{)?SLV_LLM_CHECKPOINT_ROOT(?:\}|\b)", text): fail("consumer checkpoint host binding missing")
base_compose = pathlib.Path(os.environ.get("SLV_LLM_BASE_COMPOSE", str(payload_root / "contract/docker-compose.edgellm-v011-candidate-llm.yml")))
if not base_compose.is_file(): fail("candidate base Compose file missing: %s" % base_compose)
# Compose expands each file before merging it.  The consumer override cannot
# satisfy required variables in the base file, so derive the four identity
# bindings from the already-validated contract and image/checkpoint inputs.
# Reject a caller-provided disagreement instead of silently masking it.
derived_compose_env = {
    "EDGE_LLM_V011_CANDIDATE_IMAGE": "seeed-local-voice@" + image_id,
    "EDGELLM_V011_MODEL_HOST_DIR": str(checkpoint_root),
    "EDGELLM_V011_MODEL_CONTAINER_DIR": model_dir,
    "EDGELLM_V011_CHECKPOINT": model_dir,
}
compose_env = os.environ.copy()
for name, value in derived_compose_env.items():
    supplied = compose_env.get(name)
    if supplied is not None and supplied != value:
        fail("Compose identity environment conflict for %s" % name)
    compose_env[name] = value
render = subprocess.run(["docker", "compose", "-f", str(base_compose), "-f", str(override), "config", "--format", "json"], env=compose_env, text=True, capture_output=True)
if render.returncode: fail("candidate consumer Compose config rejected: %s" % render.stderr.strip())
try: rendered = json.loads(render.stdout)
except Exception as exc: fail("candidate Compose did not return rendered JSON: %s" % exc)
services = rendered.get("services") if isinstance(rendered, dict) else None
service = services.get("edge-llm-v011-candidate") if isinstance(services, dict) else None
if not isinstance(service, dict): fail("rendered Compose target service missing")
if service.get("image") != "seeed-local-voice@" + image_id: fail("rendered candidate image identity mismatch")
if service.get("runtime") != "nvidia": fail("rendered GPU runtime binding missing")
gpus = service.get("gpus")
valid_gpus = gpus in ("all", ["all"])
if isinstance(gpus, list) and len(gpus) == 1 and isinstance(gpus[0], dict):
    spec = gpus[0]
    count = spec.get("count")
    driver = spec.get("driver")
    capabilities = spec.get("capabilities")
    valid_caps = capabilities is None or capabilities == ["gpu"] or capabilities == [["gpu"]]
    valid_gpus = (set(spec).issubset({"count", "driver", "capabilities"}) and
                  isinstance(count, int) and not isinstance(count, bool) and count == -1 and
                  (driver is None or driver == "nvidia") and valid_caps)
if not valid_gpus: fail("rendered GPU runtime binding missing")
env = service.get("environment")
if not isinstance(env, dict): fail("rendered candidate environment missing")
required_env = ("EDGELLM_V011_INPUT_CONTRACT", "EDGELLM_V011_MODEL_HOST_DIR", "EDGELLM_V011_MODEL_CONTAINER_DIR", "EDGELLM_V011_CHECKPOINT", "EDGELLM_V011_CACHE_DIR")
if any(key not in env or not isinstance(env.get(key), str) or not env[key] for key in required_env): fail("rendered candidate required environment binding missing")
if env["EDGELLM_V011_MODEL_CONTAINER_DIR"] != model_dir or env["EDGELLM_V011_CHECKPOINT"] != model_dir: fail("rendered model path differs from runtime contract")
if env["EDGELLM_V011_CACHE_DIR"] != contract_cache: fail("rendered cache path differs from runtime contract")
volumes = service.get("volumes")
if not isinstance(volumes, list): fail("rendered candidate volumes missing")
canonical_contract_rel = "contract/v011-candidate-llm-inputs.json"
published_contract_lock = (runtime_contract.get("files") or {}).get(canonical_contract_rel)
if not isinstance(published_contract_lock, dict): fail("published canonical input contract file lock is missing")
contract_mounts = [v for v in volumes if isinstance(v, dict) and v.get("target") == env["EDGELLM_V011_INPUT_CONTRACT"]]
if len(contract_mounts) != 1: fail("rendered input contract mount missing or ambiguous")
raw_contract_source = contract_mounts[0].get("source")
if not isinstance(raw_contract_source, str) or not raw_contract_source.startswith("/"): fail("rendered input contract source must be absolute")
expected_contract_source = str(payload_root / canonical_contract_rel)
if os.path.normpath(raw_contract_source) != raw_contract_source or raw_contract_source != expected_contract_source: fail("rendered input contract source is not canonical path")
raw_contract_path = pathlib.Path(raw_contract_source)
if raw_contract_path.is_symlink(): fail("published canonical input contract missing or symlinked")
for component in raw_contract_path.parent.relative_to(pathlib.Path(expected_contract_source).parent.parent).parts:
    if component in ("", "."): continue
    probe = pathlib.Path(expected_contract_source).parent.parent / component
    if probe.is_symlink() or not probe.is_dir(): fail("published canonical input contract parent is not a real directory")
canonical_contract_source = pathlib.Path(raw_contract_source).resolve()
payload_root_real = payload_root.resolve()
if not (canonical_contract_source == payload_root_real or payload_root_real in canonical_contract_source.parents): fail("published canonical input contract escapes payload root")
if canonical_contract_source.relative_to(payload_root_real).as_posix() != canonical_contract_rel: fail("rendered input contract source is not canonical path")
if canonical_contract_source.is_symlink() or not canonical_contract_source.is_file(): fail("published canonical input contract missing or symlinked")
if canonical_contract_source.name != pathlib.PurePosixPath(canonical_contract_rel).name: fail("rendered input contract source is not canonical filename")
contract_stat = canonical_contract_source.stat()
if contract_stat.st_size <= 0 or contract_stat.st_size != published_contract_lock.get("size"): fail("published canonical input contract size mismatch")
if sha256(canonical_contract_source) != published_contract_lock.get("sha256"): fail("published canonical input contract hash mismatch")
try: canonical_contract_json = json.loads(canonical_contract_source.read_text())
except Exception as exc: fail("published canonical input contract JSON invalid: %s" % exc)
canonical_contract = canonical_contract_json.get("engine_contract", canonical_contract_json) if isinstance(canonical_contract_json, dict) else None
if canonical_contract != contract: fail("published canonical input contract differs from runtime engine contract")
def mount(source, target):
    for entry in volumes:
        if isinstance(entry, dict):
            if entry.get("source") == source and entry.get("target") == target and entry.get("read_only") is True: return True
        elif isinstance(entry, str):
            bits = entry.split(":")
            if len(bits) >= 3 and bits[0] == source and bits[1] == target and bits[-1] == "ro": return True
    return False
if not mount(env["EDGELLM_V011_MODEL_HOST_DIR"], model_dir): fail("rendered checkpoint bind is not read-only or path-matched")
if contract_mounts[0].get("type") not in (None, "bind"): fail("rendered input contract mount is not a bind")
if not contract_mounts[0].get("read_only") is True: fail("rendered input contract mount missing or not canonical")
if not any(isinstance(v, dict) and v.get("target") == contract_cache and v.get("read_only") is True for v in volumes): fail("rendered cache mount missing")
print("COMPOSE_OVERRIDE_READY=%s" % override)
print("CANDIDATE_PREPARED_ONLY=1; compose up/native GPU/run/foreign service stop were not executed")
PY
fi

if [[ -z "$target" || "$target" == "auto" ]]; then
  if ! target="$(detect_target)"; then
    echo "Could not auto-detect target. Re-run with --target jetson|orin-nx|rk3576|rk3588|rpi." >&2
    usage >&2
    exit 2
  fi
  echo "Auto-detected target: ${target}"
fi

project_root="$(cd "$(dirname "$0")/.." && pwd)"
export OVS_PROFILE_JSON_HOST="${OVS_PROFILE_JSON_HOST:-$project_root/configs/profiles/jetson-edgellm-v011-default-voice.json}"
compose_files=()
canonical_target=""
case "$target" in
  orin-nx)
    canonical_target="orin-nx"
    if [[ "$legacy" -eq 1 ]]; then
      case "${EDGELLM_ENGINE_PROFILE:-8k}" in
        8k) compose_files=("deploy/docker-compose.edgellm-v091-voice.yml" "deploy/docker-compose.edgellm-v091-cutover.yml") ;;
        4k) compose_files=("deploy/docker-compose.edgellm-v091-voice.yml" "deploy/docker-compose.edgellm-v091-cutover-4k.yml") ;;
        *) die "EDGELLM_ENGINE_PROFILE must be 4k or 8k for --target orin-nx" ;;
      esac
    else
      compose_files=("deploy/docker-compose.edgellm-v011-candidate-voice.yml")
    fi
    ;;
  jetson|orin|orin-nano)
    canonical_target="jetson"
    if [[ "$legacy" -eq 1 ]]; then
      compose_files=("deploy/docker-compose.edgellm-v090-legacy.yml")
    else
      compose_files=("deploy/docker-compose.edgellm-v011-candidate-voice.yml")
    fi
    ;;
  rk3576|rk)
    canonical_target="rk3576"
    compose_files=("deploy/docker-compose.rk.yml")
    ;;
  rk3588|radxa)
    canonical_target="rk3588"
    compose_files=("deploy/docker-compose.radxa.yml")
    ;;
  rpi|rpi4|rpi5|cm4|cm5)
    canonical_target="rpi"
    compose_files=("deploy/docker-compose.rpi.yml")
    ;;
  *)
    echo "Unsupported target: $target" >&2
    usage >&2
    exit 2
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  die "docker is required. Install Docker Engine first."
fi

if ! docker compose version >/dev/null 2>&1; then
  die "docker compose v2 is required."
fi

if ! docker info >/dev/null 2>&1; then
  die "docker daemon is not reachable. Start Docker, then rerun this command."
fi

compose_args=()
for compose_file in "${compose_files[@]}"; do
  [[ -f "$compose_file" ]] || die "missing compose file: $compose_file"
  compose_args+=( -f "$compose_file" )
done

case "$canonical_target" in
  jetson|orin-nx)
    if [[ ! -e /etc/nv_tegra_release ]]; then
      warn "this does not look like Jetson Linux; the container needs JetPack 6.x host libraries."
    fi
    if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
      warn "NVIDIA Container Runtime was not found in docker runtimes. Install nvidia-container-toolkit before starting the Jetson image."
    fi
    if [[ ! -x /usr/src/tensorrt/bin/trtexec ]]; then
      die "Jetson TensorRT trtexec is missing at /usr/src/tensorrt/bin/trtexec. Install the JetPack TensorRT samples/tools package or use a prebuilt engine artifact set."
    fi
    if [[ ! -e /usr/local/cuda/lib64/libcudla.so.1 ]]; then
      die "Jetson CUDA runtime library libcudla.so.1 is missing under /usr/local/cuda/lib64. Check the JetPack/CUDA installation before starting the slim image."
    fi
    profile="${OVS_PROFILE:-}"
    if [[ "$canonical_target" == "orin-nx" ]]; then
      if [[ "$legacy" -eq 1 ]]; then
        echo "Jetson legacy v0.9.1 model-level profile: ${profile:-jetson-edgellm-v091-matcha}"
      else
        echo "Jetson v0.11 combined voice profile: ${profile:-jetson-edgellm-v011-default-voice}"
      fi
      if [[ "$legacy" -eq 1 ]]; then
        echo "Speech and Qwen3.5-4B GDN/MTP ${EDGELLM_ENGINE_PROFILE:-8k} will start as independent services."
      else
        echo "v0.11 combined voice starts by default; v0.11 LLM remains an explicit opt-in service."
      fi
      if [[ "$legacy" -eq 1 && "${EDGELLM_ENGINE_PROFILE:-8k}" == "4k" ]]; then
        echo "4K payload SHA: 06273e358a579590bb8344b451aa35c89983cd99401339fb1858d61af4dbd107"
        echo "4K HF revision: ${EDGELLM_4K_ENGINE_REVISION:-9f2c2059341fd2135cc3a0ec09e05150277ea5b6}"
      elif [[ "$legacy" -eq 1 ]]; then
        echo "8K payload SHA: 9208e46d61a4f1440ac68a312e35dde3d04b88edf0e4ee12b32210e7190d3325"
        echo "8K HF revision: ${EDGELLM_8K_ENGINE_REVISION:-adb1c78fb61513e2d7d8e7f889f6196dbefb1e5e}"
      fi
      echo "Rollback: deploy/install.sh --target orin-nx --legacy --pull --verify"
    elif [[ -n "${profile}" && "${profile}" == jetson-multilang-* ]]; then
      echo "Jetson Qwen3 profile: ${profile}"
      echo "Qwen3 artifacts: ${QWEN3_HF_REPO_ID:-harvestsu/qwen3-edgellm-jetson-artifacts}@${QWEN3_HF_REVISION:-main}"
    else
      echo "Jetson zh_en artifacts: harvestsu/seeed-local-voice-artifacts"
      echo "Paraformer preroll: ${PARAFORMER_PREROLL_MS:-100} ms"
    fi
    ;;
  rk3576|rk3588)
    if [[ ! -e /dev/rknpu ]]; then
      warn "/dev/rknpu is missing. Load the Rockchip NPU driver or use a kernel image with rknpu support."
    fi
    echo "RK artifact manifest: ${RK_ARTIFACT_MANIFEST:-deploy/artifacts/rk_manifest.json}"
    echo "RK artifact set: ${RK_ARTIFACT_SET:-compose default}"
    ;;
  rpi)
    arch="$(uname -m)"
    if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
      warn "Raspberry Pi images are built for 64-bit ARM; current architecture is ${arch}."
    fi
    ;;
esac

available_mb="$(df -Pm . | awk 'NR==2 {print $4}')"
if [[ -n "${available_mb}" && "${available_mb}" -lt 5120 ]]; then
  warn "less than 5 GB free under the project filesystem; first boot may fail while downloading models."
fi

echo "Target: $canonical_target"
echo "Compose: ${compose_files[*]}"
docker compose "${compose_args[@]}" config --quiet

if [[ "$pull" -eq 1 ]]; then
  docker compose "${compose_args[@]}" pull
fi

up_args=(up -d)
if [[ "$build" -eq 1 ]]; then
  [[ "$legacy" -eq 1 ]] || die "v0.11 default uses the published digest; --build is allowed only with --legacy"
  up_args+=(--build)
fi

if ! docker compose "${compose_args[@]}" "${up_args[@]}"; then
  warn "container failed to start; recent logs follow"
  docker compose "${compose_args[@]}" ps >&2 || true
  docker compose "${compose_args[@]}" logs --tail=80 >&2 || true
  exit 1
fi

port="${OVS_PORT:-8621}"
url="http://127.0.0.1:${port}"

echo "Service URL: ${url}"

if [[ "$verify" -eq 1 ]]; then
  if ! deploy/verify.sh --url "${url}" --tts-smoke --roundtrip; then
    warn "verification failed; recent logs follow"
    docker compose "${compose_args[@]}" ps >&2 || true
    docker compose "${compose_args[@]}" logs --tail=120 >&2 || true
    exit 1
  fi
  if [[ "$canonical_target" == "orin-nx" && "$legacy" -eq 1 ]]; then
    if ! deploy/verify-llm.sh "http://127.0.0.1:8000"; then
      warn "LLM verification failed; recent logs follow"
      docker compose "${compose_args[@]}" logs --tail=120 edge-llm >&2 || true
      exit 1
    fi
  fi
fi
