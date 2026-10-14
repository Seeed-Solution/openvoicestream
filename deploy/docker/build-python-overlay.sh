#!/usr/bin/env bash
#
# Build a thin Python overlay image from a reviewed commit.
#
# The point of this script is that none of the fragile steps are left to
# memory:
#
#   * the destination path inside the image is probed from the base image,
#     twice — once plainly and once with the entrypoint's PYTHONPATH/cwd —
#     because a package can be importable from two places and the daemon may
#     import the one you did not patch;
#   * the payload is staged with `git show <commit>:<path>`, so uncommitted
#     working-tree edits can never be baked in;
#   * the build itself re-checks md5 and import resolution (see
#     deploy/docker/python-overlay/apply.sh) and fails closed.
#
# Usage:
#   build-python-overlay.sh --base <image:tag> --src <repo pkg dir> \
#       --file <path relative to --src> [--file ...] --tag <image:new-tag> \
#       [--commit <sha>] [--pkg <import name>] [--smoke <python one-liner>]
#
# Example (the RK3576 audio_io fix):
#   deploy/docker/build-python-overlay.sh \
#       --base sensecraft-missionpack.seeed.cn/solution/ovs-agent:voiceagent-20260921-bargein.4 \
#       --src agent/ovs_agent --file audio_io.py \
#       --tag ovs-agent:voiceagent-20260921-bargein.4-nativecapture \
#       --smoke "from ovs_agent.audio_io import AudioIO; assert hasattr(AudioIO, '_resolve_input_sample_rate')"
#
set -eu

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DOCKERFILE="$REPO_ROOT/deploy/docker/Dockerfile.python-overlay"
APPLY_SH="$REPO_ROOT/deploy/docker/python-overlay/apply.sh"

BASE=""; SRC=""; TAG=""; COMMIT="HEAD"; PKG=""; SMOKE=""
FILES=""

die() { echo "build-python-overlay: $*" >&2; exit 1; }

# md5sum is not on macOS; md5 -q is not on Linux.
md5_of() {
    if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
    else md5 -q "$1"; fi
}
md5_of_stdin() {
    if command -v md5sum >/dev/null 2>&1; then md5sum | cut -d' ' -f1
    else md5 -q; fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --base)   BASE=${2:?};   shift 2 ;;
        --src)    SRC=${2:?};    shift 2 ;;
        --file)   FILES="$FILES ${2:?}"; shift 2 ;;
        --tag)    TAG=${2:?};    shift 2 ;;
        --commit) COMMIT=${2:?}; shift 2 ;;
        --pkg)    PKG=${2:?};    shift 2 ;;
        --smoke)  SMOKE=${2:?};  shift 2 ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[ -n "$BASE" ] || die "--base is required"
[ -n "$SRC" ]  || die "--src is required"
[ -n "$TAG" ]  || die "--tag is required"
[ -n "$FILES" ] || die "at least one --file is required"
[ -d "$SRC" ]  || die "--src is not a directory: $SRC"

[ -n "$PKG" ] || PKG=$(basename "$SRC")
COMMIT_SHA=$(git -C "$REPO_ROOT" rev-parse --verify "$COMMIT^{commit}") \
    || die "cannot resolve --commit $COMMIT"

# ── 1. discover where the base image actually imports the package from ──
probe() {
    # "$@" = extra docker args (may be empty); prints the resolved package dir
    docker run --rm "$@" --entrypoint python3 "$BASE" -c \
        "import os, sys, $PKG as m; print(os.path.dirname(os.path.abspath(m.__file__)))" \
        2>/dev/null | tail -1
}

PLAIN_DIR=$(probe) || true
IMG_PYTHONPATH=$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$BASE" 2>/dev/null \
    | sed -n 's/^PYTHONPATH=//p' | head -1)
IMG_WORKDIR=$(docker inspect --format '{{.Config.WorkingDir}}' "$BASE" 2>/dev/null || echo "")

# Arrays, not a concatenated string: a workdir or PYTHONPATH containing a space
# would otherwise split into extra docker arguments. Plain `[ ... ] && x=`
# statements also return non-zero when the test fails, which `set -e` treats as
# a fatal error.
DAEMON_ARGS=()
if [ -n "$IMG_PYTHONPATH" ]; then DAEMON_ARGS+=(-e "PYTHONPATH=$IMG_PYTHONPATH"); fi
if [ -n "$IMG_WORKDIR" ]; then DAEMON_ARGS+=(-w "$IMG_WORKDIR"); fi

if [ ${#DAEMON_ARGS[@]} -gt 0 ]; then
    DAEMON_DIR=$(probe "${DAEMON_ARGS[@]}") || true
else
    DAEMON_DIR="$PLAIN_DIR"
fi

[ -n "$PLAIN_DIR" ] || die "could not import $PKG in $BASE — is this the right base image?"
if [ -z "$DAEMON_DIR" ]; then
    die "the daemon-shaped probe returned nothing while the plain probe found $PLAIN_DIR.
PYTHONPATH=${IMG_PYTHONPATH:-(unset)} WORKDIR=${IMG_WORKDIR:-(unset)}
Refusing to guess which copy the daemon imports."
fi
if [ "$DAEMON_DIR" != "$PLAIN_DIR" ]; then
    die "import path is ambiguous in $BASE:
  plain probe  : $PLAIN_DIR
  daemon-shaped: $DAEMON_DIR
Patch the daemon-shaped path (that is what runs), or fix the image so both agree."
fi
DEST="$PLAIN_DIR"
echo "base        : $BASE"
echo "commit      : $COMMIT_SHA"
echo "import path : $DEST   (plain and daemon-shaped probes agree)"
echo "pythonpath  : ${IMG_PYTHONPATH:-(unset)}"
echo "workdir     : ${IMG_WORKDIR:-(unset)}"

# ── 2. stage the payload from the commit, never the working tree ──
CTX=$(mktemp -d "${TMPDIR:-/tmp}/python-overlay.XXXXXX")
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/payload"
cp "$APPLY_SH" "$CTX/apply.sh"
: > "$CTX/payload.manifest"

for rel in $FILES; do
    src_path="$SRC/$rel"
    git -C "$REPO_ROOT" cat-file -e "$COMMIT_SHA:$src_path" 2>/dev/null \
        || die "$src_path does not exist at $COMMIT_SHA"
    mkdir -p "$CTX/payload/$(dirname "$rel")"
    git -C "$REPO_ROOT" show "$COMMIT_SHA:$src_path" > "$CTX/payload/$rel"
    printf '%s  %s\n' "$(md5_of "$CTX/payload/$rel")" "$rel" >> "$CTX/payload.manifest"
    echo "staged      : $src_path -> payload/$rel"
done

# working-tree drift is worth knowing about: it means the image will not match
# what a human sees in their editor.
for rel in $FILES; do
    if ! git -C "$REPO_ROOT" diff --quiet -- "$SRC/$rel" 2>/dev/null; then
        echo "note        : $SRC/$rel has uncommitted changes — image uses $COMMIT_SHA, not the working tree"
    fi
done

# ── 3. build ──
echo
echo "docker build -> $TAG"
# --provenance=false: attestation manifests are noise for an image that is
# shipped as a `docker save` tarball and loaded on the device.
docker build \
    -f "$DOCKERFILE" \
    --provenance=false \
    --sbom=false \
    --build-arg "BASE=$BASE" \
    --build-arg "OVERLAY_DEST=$DEST" \
    --build-arg "OVERLAY_SMOKE=$SMOKE" \
    -t "$TAG" "$CTX"

# ── 4. report what actually landed, straight from the built image ──
#
# This gates the script: a report that prints MISMATCH and still exits 0 is
# worse than no report, because the caller then ships the image.
echo
echo "--- verifying the built image ---"
verify_failed=0
for rel in $FILES; do
    want=$(md5_of "$CTX/payload/$rel")
    # `cut` exits 0 even when `docker run` failed, so the empty-output case has
    # to be handled explicitly rather than by the pipeline status.
    got=$(docker run --rm --entrypoint md5sum "$TAG" "$DEST/$rel" 2>/dev/null | cut -d' ' -f1 || true)
    if [ "$got" = "$want" ]; then
        printf '  %s  %s  OK\n' "$rel" "$got"
    else
        printf '  %s  %s  MISMATCH (want %s)\n' "$rel" "${got:-<no output>}" "$want"
        verify_failed=1
    fi
done
if [ "$verify_failed" -ne 0 ]; then
    die "post-build verification failed: $TAG does not match the staged payload"
fi
docker run --rm --entrypoint python3 "$TAG" -c \
    "import os, $PKG as m; print('  import path:', os.path.dirname(os.path.abspath(m.__file__)))"

DIGEST=$(docker inspect --format '{{.Id}}' "$TAG" 2>/dev/null || echo UNVERIFIED)
echo
echo "--- IMAGE-TAGS.md row ---"
echo "| \`$TAG\` | $COMMIT_SHA — $(for rel in $FILES; do printf '%s md5 %s, ' "$rel" "$(md5_of "$CTX/payload/$rel")"; done | sed 's/, $//') over \`$BASE\` | inherited | $(date -u +%Y-%m-%d) | $(hostname) ($(uname -m)) | $DIGEST |"
