#!/bin/sh
# Applied *inside* the overlay image by deploy/docker/Dockerfile.python-overlay.
#
# Job: replace a whitelisted set of Python files in the directory the daemon
# actually imports from, then fail the build if anything did not land as
# intended. Every check here exists because the corresponding mistake has
# already shipped once:
#
#   * a second copy of the package was imported instead of the patched one
#     (the entrypoint rewrites PYTHONPATH), so the image looked patched and
#     behaved unpatched;
#   * a stale __pycache__ kept serving the previous source;
#   * files from the working tree — not the reviewed commit — ended up baked in.
#
# Env:
#   OVERLAY_DEST      absolute path of the imported package dir in BASE (required)
#   OVERLAY_SMOKE     optional python one-liner that must run without raising
#   OVERLAY_PAYLOAD   default /tmp/overlay-payload
#   OVERLAY_MANIFEST  default /tmp/overlay-payload.manifest
set -eu

DEST="${OVERLAY_DEST:?OVERLAY_DEST is required}"
PAYLOAD="${OVERLAY_PAYLOAD:-/tmp/overlay-payload}"
MANIFEST="${OVERLAY_MANIFEST:-/tmp/overlay-payload.manifest}"

[ -d "$DEST" ] || { echo "OVERLAY: destination absent in BASE: $DEST" >&2; exit 1; }
[ -d "$PAYLOAD" ] || { echo "OVERLAY: payload absent: $PAYLOAD" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "OVERLAY: manifest absent: $MANIFEST" >&2; exit 1; }

# Drop bytecode before replacing sources. Python keys __pycache__ on the
# source path; a replaced file keeps whatever mtime the copy gave it, and a
# .pyc that still looks fresh will shadow the new source.
find "$DEST" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

cp -a "$PAYLOAD"/. "$DEST"/

# 1. Every staged file landed byte-identical at the destination.
while read -r want rel; do
    [ -n "${rel:-}" ] || continue
    got=$(md5sum "$DEST/$rel" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        echo "OVERLAY: md5 mismatch for $rel: got $got want $want" >&2
        exit 1
    fi
    echo "OVERLAY: md5 ok    $rel"
done < "$MANIFEST"

# 2. Importing each replaced module must resolve *to the file we just wrote*.
#    Anything else means another copy of the package wins at import time and
#    the image would ship the old behaviour with a green build.
#
#    Deliberately NO sys.path manipulation. The image's own ENV PYTHONPATH and
#    working directory are what the daemon gets at runtime, and forcing $DEST
#    onto the path would make this check pass no matter which copy wins — the
#    exact failure it exists to catch.
python3 - "$DEST" "$MANIFEST" <<'PY'
import importlib
import os
import sys

dest = os.path.realpath(sys.argv[1])
pkg = os.path.basename(dest)

checked = 0
problems = []
with open(sys.argv[2]) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        rel = line.split(None, 1)[1]
        if not rel.endswith(".py"):
            continue
        mod = pkg + "." + rel[:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001 - surface the real reason
            problems.append("%s: import failed: %s" % (mod, exc))
            continue
        resolved = os.path.realpath(getattr(sys.modules[mod], "__file__", "") or "")
        if not resolved.startswith(dest + os.sep):
            problems.append(
                "%s resolved to %s, which is outside %s — another copy shadows it"
                % (mod, resolved, dest)
            )
        checked += 1

if problems:
    raise SystemExit("OVERLAY: import check failed:\n  " + "\n  ".join(problems))
print("OVERLAY: import ok %d module(s) from %s" % (checked, dest))
PY

# 3. Optional change-specific assertion (e.g. a new method exists).
if [ -n "${OVERLAY_SMOKE:-}" ]; then
    python3 -c "${OVERLAY_SMOKE}"
    echo "OVERLAY: smoke ok"
fi

echo "OVERLAY: applied to $DEST"
