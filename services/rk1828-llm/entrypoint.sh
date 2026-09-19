#!/usr/bin/env bash
# RK1828 LLM service entrypoint: pull model artifacts if absent, then serve.
#
# Artifacts are deliberately NOT baked into the image (3.2 GB), and are pulled
# per configuration on first start into RK1828_MODEL_DIR — mount that as a
# volume so a container replacement does not re-download.
#
# Uses plain curl rather than huggingface_hub: the files are public, and this
# keeps the image free of a dependency whose only job is one download.
set -euo pipefail

MODEL_DIR="${RK1828_MODEL_DIR:-/opt/llm/models}"
MANIFEST="${RK1828_ARTIFACT_MANIFEST:-/opt/rk1828-llm/artifacts.json}"
REPO_ID="${RK1828_ARTIFACT_REPO_ID:-harvestsu/seeed-local-voice-rk-artifacts}"
PREFIX="${RK1828_ARTIFACT_PREFIX:-rk1828/opt/llm/qwen3-4b}"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
REVISION="${RK1828_ARTIFACT_REVISION:-main}"

log() { printf '[rk1828-llm] %s\n' "$*" >&2; }

# ── preflight: the card must already be initialised by the HOST ────────────
# Failing loudly here beats a confusing MODEL_SETUP failure several minutes in.
if ! compgen -G '/dev/pcie-rkep-*' >/dev/null; then
  log "FATAL: no /dev/pcie-rkep-* device visible in this container."
  log "  The RK1828 driver and firmware live on the HOST, not in this image."
  log "  On the host, check: lspci | grep 182a ; systemctl is-active rknn3.service"
  log "  and make sure the container gets the device (privileged + /dev mount)."
  exit 1
fi

# ── runtime alignment: prefer the HOST's RKNN3 client libs ────────────────
# The client lib in this image and the host's rknn3_transfer_proxy + EP
# firmware ship in ONE package, so they only speak to each other when they are
# the same generation. When they are not, the failure is silent and expensive:
# the worker connects to @transfer_proxy3, sends its init request, and blocks in
# read() forever — no error, no log line, just the supervisor's READY timeout
# three times over (observed 2026-09-17: image 1.0.4 against a host on 1.1.0,
# five attempts all timing out at exactly 179.6 s, EP memory untouched).
#
# So the host's copy wins whenever it is available: it is by construction in
# lockstep with the proxy and the firmware, which the image cannot be. The
# bundled copy stays as the fallback for hosts that do not expose /usr/lib.
# Higher runtime loads lower-version models (confirmed by Rockchip 2026-09-18),
# so moving the client up does NOT require re-exporting the 3.2 GB artifacts.
HOST_LIB_DIR="${RK1828_HOST_LIB_DIR:-/opt/rk1828/host-lib}"
LIB_DIR=/opt/rk1828/lib

# The version string is in the binary; no `strings` in this image, so grep -a.
lib_version() {
  [ -f "$1" ] || { printf 'absent'; return; }
  v=$(grep -a -o -m1 'librknn3_api version: [0-9][0-9.]*' "$1" 2>/dev/null | head -1)
  printf '%s' "${v:-unknown}"
}

log "bundled RKNN3 client: $(lib_version "${LIB_DIR}/librknn3_api_rkcp.so")" \
    "(md5 $(md5sum "${LIB_DIR}/librknn3_api_rkcp.so" 2>/dev/null | cut -c1-12))"

host_rkcp="${HOST_LIB_DIR}/librknn3_api_rkcp.so"
host_shim="${HOST_LIB_DIR}/librknn3_api.so"

if [ -f "${host_rkcp}" ] && [ -f "${host_shim}" ]; then
  log "host RKNN3 client:    $(lib_version "${host_rkcp}")" \
      "(md5 $(md5sum "${host_rkcp}" 2>/dev/null | cut -c1-12))"
  # Compare BOTH files: the shim and the backend are versioned together, and a
  # shim-only difference would otherwise be skipped here and then show up as the
  # same silent init hang this whole block exists to prevent.
  if cmp -s "${host_rkcp}" "${LIB_DIR}/librknn3_api_rkcp.so" \
     && cmp -s "${host_shim}" "${LIB_DIR}/librknn3_api.so"; then
    log "host and bundled client are identical; keeping bundled"
  elif [ "${RK1828_PREFER_HOST_RUNTIME:-1}" = "1" ]; then
    # Copy rather than symlink: the binary's rpath is $ORIGIN/lib and the shim
    # dlopen()s the rkcp backend from that same dir, so both files must land
    # there together. The mount stays read-only; only this layer is written.
    #
    # Checked explicitly, NOT as `cp && log`: under `set -e` a failing left-hand
    # side of an AND-list does not exit the shell, so a read-only layer or a
    # partial copy would leave a mismatched (or half-replaced) pair in place and
    # the service would start straight into the 180 s init hang. Refuse instead.
    if cp -f "${host_shim}" "${host_rkcp}" "${LIB_DIR}/"; then
      log "using the HOST RKNN3 client (copied over the bundled one)"
    else
      log "FATAL: could not copy the host RKNN3 client into ${LIB_DIR}."
      log "  The bundled client is a different generation than this host's"
      log "  proxy/firmware, so starting now would block in model init with no"
      log "  error. Check that ${LIB_DIR} is writable in this container, or set"
      log "  RK1828_PREFER_HOST_RUNTIME=0 to run with the bundled client anyway."
      exit 1
    fi
  else
    log "WARNING: host client differs from bundled, but RK1828_PREFER_HOST_RUNTIME=0"
    log "  If model init hangs until the READY timeout, this is the reason."
  fi
elif [ -f "${host_rkcp}" ] || [ -f "${host_shim}" ]; then
  log "FATAL: ${HOST_LIB_DIR} has only one half of the RKNN3 client pair."
  log "  librknn3_api.so (the dlopen shim) and librknn3_api_rkcp.so (the backend)"
  log "  are versioned together and must both come from the host."
  exit 1
else
  log "WARNING: no host RKNN3 client at ${HOST_LIB_DIR} — using the bundled one."
  log "  Mount the host's lib dir read-only (see deploy/docker-compose.conversation-rk3588-rk1828.yml)."
  log "  Without it, a host whose rknn3 package is a different generation makes"
  log "  model init block forever with no error — see BUILD.md 'Runtime alignment'."
fi

# ── artifact pull ─────────────────────────────────────────────────────────
if [ "${RK1828_ARTIFACT_AUTO_DOWNLOAD:-1}" = "1" ]; then
  if [ ! -f "${MANIFEST}" ]; then
    log "FATAL: artifact manifest ${MANIFEST} missing"
    exit 1
  fi
  mkdir -p "${MODEL_DIR}"
  # Read (name, size) pairs from the manifest without pulling in jq.
  python3 - "$MANIFEST" <<'PY' > /tmp/_artifacts.tsv
import json, sys
m = json.load(open(sys.argv[1]))
for f in m["files"]:
    print(f"{f['name']}\t{f.get('size_bytes', 0)}\t{f.get('sha256', '')}")
PY
  while IFS=$'\t' read -r name size sha; do
    dest="${MODEL_DIR}/${name}"
    # Size check, not just existence: a download interrupted halfway leaves a
    # short file that would otherwise be treated as present and then fail model
    # init with something far less obvious.
    if [ -f "${dest}" ]; then
      have=$(stat -c %s "${dest}")
      if [ "${size}" = "0" ] || [ "${have}" = "${size}" ]; then
        log "have ${name} (${have} bytes)"
        continue
      fi
      log "size mismatch for ${name}: have ${have}, want ${size} — refetching"
    fi
    url="${ENDPOINT}/${REPO_ID}/resolve/${REVISION}/${PREFIX}/${name}"
    log "fetching ${name} from ${url}"
    # Download to a temp name and move into place, so an interrupted transfer
    # can never be mistaken for a complete file on the next start.
    if ! curl -fL --retry 3 --retry-delay 5 -o "${dest}.part" "${url}"; then
      log "FATAL: download failed for ${name}"
      log "  If this host is behind the great firewall, set HF_ENDPOINT to a"
      log "  mirror (e.g. https://hf-mirror.com). Mirrors may lag the origin."
      rm -f "${dest}.part"
      exit 1
    fi
    mv "${dest}.part" "${dest}"
    if [ -n "${sha}" ] && command -v sha256sum >/dev/null 2>&1; then
      got=$(sha256sum "${dest}" | cut -d' ' -f1)
      if [ "${got}" != "${sha}" ]; then
        log "FATAL: sha256 mismatch for ${name}: got ${got}, want ${sha}"
        exit 1
      fi
      log "sha256 ok for ${name}"
    fi
  done < /tmp/_artifacts.tsv
  rm -f /tmp/_artifacts.tsv
else
  log "artifact auto-download disabled; expecting models already in ${MODEL_DIR}"
fi

# The four files are a MATCHED SET from one export — a mismatched .rknn/.weight
# pair does not load. Refuse to start on a partial set rather than emit a
# firmware-level ACK_FAIL that looks like a hardware problem.
missing=0
for f in $(python3 -c "
import json;print(' '.join(x['name'] for x in json.load(open('${MANIFEST}'))['files']))
"); do
  [ -f "${MODEL_DIR}/${f}" ] || { log "missing artifact: ${f}"; missing=1; }
done
[ "${missing}" = "0" ] || { log "FATAL: incomplete artifact set in ${MODEL_DIR}"; exit 1; }

log "serving ${RK1828_MODEL_ID:-Qwen3-4B} on ${RK1828_HOST:-0.0.0.0}:${RK1828_PORT:-1828}" \
    "(core_mask=${RK1828_CORE_MASK:-ff} max_context=${RK1828_MAX_CONTEXT:-8192})"

exec /opt/venv/bin/python /opt/rk1828-llm/rk1828_llm_server.py \
  --binary /opt/rk1828/rknn_qwen3_demo \
  --model-dir "${MODEL_DIR}" \
  --core-mask "${RK1828_CORE_MASK:-ff}" \
  --max-context "${RK1828_MAX_CONTEXT:-8192}" \
  --host "${RK1828_HOST:-0.0.0.0}" \
  --port "${RK1828_PORT:-1828}"
