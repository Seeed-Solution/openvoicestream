#!/usr/bin/env bash
# RK1828 host bring-up / verification.
#
# WHY THIS SCRIPT EXISTS
#   The LLM container is NOT self-contained. The RK1828 accelerator card's kernel
#   driver and firmware live on the HOST; a container can only talk to a card the
#   host has already initialised. Handing someone just an image and a compose file
#   leaves them with a service that fails at model-load time for reasons that look
#   like a hardware fault.
#
# Run with --check to verify only (safe, read-only), or with no argument to also
# load and persist the module, enable the service, and install the userspace
# runtime when it is missing.
#
#   sudo deploy/scripts/rk1828-host-bringup.sh --check
#   sudo deploy/scripts/rk1828-host-bringup.sh
#   sudo deploy/scripts/rk1828-host-bringup.sh --sdk ~/RK182X_RM182XMC0
#
# This script does NOT build the kernel module. Building it needs the RM182X SDK
# plus kernel headers; see services/rk1828-llm/BUILD.md. If the module is not
# installed this script says so and stops.
set -uo pipefail

CHECK_ONLY=0
SDK_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK_ONLY=1 ;;
    # Directory holding the vendor RM182X release (its arm64 runtime installer
    # is what puts librknn3_api*.so, rknn3_transfer_proxy, the EP firmware and
    # rknn3.service on the host). Used only when the runtime is missing AND the
    # apt package is not available.
    --sdk) shift; SDK_DIR="${1:-}" ;;
    *) echo "usage: $0 [--check] [--sdk <RM182X SDK dir>]" >&2; exit 2 ;;
  esac
  shift
done

PCI_ID="1d87:182a"
MODULE="pcie-rkep"          # hyphen for modprobe / modules-load.d
MODULE_LSMOD="pcie_rkep"    # underscore as lsmod reports it
FIRMWARE="/lib/firmware/rknn3_rk1820.img"
PERSIST="/etc/modules-load.d/pcie-rkep.conf"
# The userspace runtime the LLM container mounts read-only. These and the EP
# firmware ship together, so the container's client lib must be THIS generation
# — see services/rk1828-llm/BUILD.md 'Runtime alignment'.
HOST_LIBS="/usr/lib/librknn3_api.so /usr/lib/librknn3_api_rkcp.so"
PROXY_BIN="/bin/rknn3_transfer_proxy"
RUNTIME_PKG="rknn3-rk182x-m2"

# /sbin is not always on PATH under sudo or in a non-login shell.
PATH="/sbin:/usr/sbin:${PATH}"

fail=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; fail=1; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }

if [ "$(id -u)" != "0" ] && [ "${CHECK_ONLY}" = "0" ]; then
  echo "ERROR: run as root (or use --check for a read-only verification)" >&2
  exit 1
fi

step "1. card present on the PCIe bus"
if lspci -nn 2>/dev/null | grep -qi "${PCI_ID}"; then
  ok "$(lspci -nn | grep -i "${PCI_ID}" | head -1)"
else
  bad "no ${PCI_ID} on the PCIe bus."
  echo "       This is HARDWARE, not software. The card has its OWN 12 V supply —"
  echo "       a dark card with a still fan means it is not powered. Check that"
  echo "       the 12 V lead is on the main power pins, not the fan header (they"
  echo "       are adjacent and easy to confuse)."
fi

step "2. kernel module installed"
if modinfo "${MODULE}" >/dev/null 2>&1; then
  ok "$(modinfo "${MODULE}" | awk '/^filename:/{print $2}')"
else
  bad "module ${MODULE} not installed for kernel $(uname -r)."
  echo "       Build + install it from the RM182X SDK (pcie-rkep driver source)"
  echo "       — see services/rk1828-llm/BUILD.md. Nothing below will work."
fi

step "3. kernel module loaded"
if lsmod | grep -q "^${MODULE_LSMOD}"; then
  ok "$(lsmod | grep "^${MODULE_LSMOD}")"
elif [ "${CHECK_ONLY}" = "1" ]; then
  bad "${MODULE_LSMOD} not loaded (re-run without --check to load it)"
else
  echo "  loading ${MODULE} ..."
  if modprobe "${MODULE}"; then ok "loaded"; else bad "modprobe ${MODULE} failed"; fi
fi

step "4. module load persisted across reboot"
# The module does NOT come back by itself; without this the card disappears
# after any reboot and the LLM service fails to start with no obvious cause.
if [ -f "${PERSIST}" ] && grep -q "${MODULE}" "${PERSIST}" 2>/dev/null; then
  ok "${PERSIST} -> $(tr -d '\n' < "${PERSIST}")"
elif [ "${CHECK_ONLY}" = "1" ]; then
  bad "not persisted; re-run without --check to write ${PERSIST}"
else
  echo "${MODULE}" > "${PERSIST}"
  ok "wrote ${PERSIST}"
fi

step "5. EP firmware present"
if [ -f "${FIRMWARE}" ]; then
  ok "${FIRMWARE} ($(stat -c %s "${FIRMWARE}") bytes)"
else
  bad "${FIRMWARE} missing — installed by the RKNN3 arm64 installer from the SDK."
fi

step "6. rknn3.service (reflashes the EP firmware at boot)"
if systemctl is-active --quiet rknn3.service; then
  ok "active"
else
  bad "not active: $(systemctl is-active rknn3.service 2>&1)"
  echo "       Without it the EP has no firmware and every model init fails."
fi
if systemctl is-enabled --quiet rknn3.service 2>/dev/null; then
  ok "enabled (starts at boot)"
else
  if [ "${CHECK_ONLY}" = "1" ]; then
    bad "not enabled; the card will be dead after a reboot"
  else
    systemctl enable rknn3.service >/dev/null 2>&1 && ok "enabled" || bad "could not enable"
  fi
fi

step "7. userspace runtime (what the container's client lib must match)"
# A skew here is the most expensive failure mode we have hit: the container's
# client connects to the proxy, sends its init request and blocks in read()
# forever. No error is logged; the only symptom is the supervisor's READY
# timeout, identical to the millisecond on every attempt (2026-09-17: image
# 1.0.4 against a host on 1.1.0). The container copies these over its bundled
# pair at startup, which is why they have to be here and why the version is
# worth printing.
runtime_version() {
  [ -f "$1" ] || { printf 'absent'; return; }
  v=$(grep -a -o -m1 'librknn3_api version: [0-9][0-9.]*' "$1" 2>/dev/null | head -1)
  [ -n "$v" ] || v=$(grep -a -o -m1 'Transfer version [0-9][0-9.]*' "$1" 2>/dev/null | head -1)
  printf '%s' "${v:-unknown}"
}

runtime_missing=0
for f in ${HOST_LIBS} "${PROXY_BIN}"; do
  [ -f "$f" ] || runtime_missing=1
done

install_runtime() {
  # Preferred: the packaged runtime, which also carries the firmware and the
  # systemd unit. Installed on the Armbian/Seeed RK3588 images as of 2026-09.
  if command -v apt-get >/dev/null 2>&1 && \
     apt-cache policy "${RUNTIME_PKG}" 2>/dev/null | grep -q 'Candidate: [0-9]'; then
    echo "  installing ${RUNTIME_PKG} via apt ..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${RUNTIME_PKG}" && return 0
    return 1
  fi
  # Fallback: the vendor release's own installer. Not second-guessed here —
  # whatever it installs is what the container will then mount.
  if [ -n "${SDK_DIR}" ]; then
    inst=$(find "${SDK_DIR}" -maxdepth 4 -name 'install.sh' -path '*arm64*' 2>/dev/null | head -1)
    [ -n "${inst}" ] || inst=$(find "${SDK_DIR}" -maxdepth 4 -name 'install.sh' 2>/dev/null | head -1)
    if [ -n "${inst}" ]; then
      echo "  running vendor installer ${inst} ..."
      ( cd "$(dirname "${inst}")" && bash ./install.sh ) && return 0
      return 1
    fi
    echo "  no install.sh found under ${SDK_DIR}"
    return 1
  fi
  return 1
}

if [ "${runtime_missing}" = "0" ]; then
  ok "${PROXY_BIN} — $(runtime_version "${PROXY_BIN}")"
  for f in ${HOST_LIBS}; do
    ok "$f — $(runtime_version "$f") ($(stat -c %s "$f") bytes)"
  done
elif [ "${CHECK_ONLY}" = "1" ]; then
  bad "userspace runtime incomplete; re-run without --check to install it"
  for f in ${HOST_LIBS} "${PROXY_BIN}"; do
    [ -f "$f" ] || echo "       missing: $f"
  done
else
  warn "userspace runtime incomplete — installing"
  if install_runtime; then
    still=0
    for f in ${HOST_LIBS} "${PROXY_BIN}"; do [ -f "$f" ] || still=1; done
    if [ "${still}" = "0" ]; then
      ok "installed: $(runtime_version "${PROXY_BIN}")"
      systemctl daemon-reload >/dev/null 2>&1
      systemctl enable --now rknn3.service >/dev/null 2>&1
    else
      bad "install reported success but the files are still missing"
    fi
  else
    bad "could not install the runtime automatically."
    echo "       Either add the repo that carries ${RUNTIME_PKG} (apt-cache policy"
    echo "       ${RUNTIME_PKG} must show a Candidate), or re-run with"
    echo "       --sdk <RM182X SDK dir> to use the vendor installer."
    echo "       Without it the container falls back to its bundled client lib,"
    echo "       which only works if the host is that same generation."
  fi
fi

step "8. character device visible (what the container needs)"
if compgen -G '/dev/pcie-rkep-*' >/dev/null; then
  for d in /dev/pcie-rkep-*; do ok "$(ls -l "$d")"; done
else
  bad "no /dev/pcie-rkep-* — the container cannot reach the card."
fi

step "9. EP not already occupied by another large model"
# The card has ONE ~5 GB context. Qwen3-4B at 8192 tokens uses an estimated
# ~3.6 GB, so a second large model cannot be resident. tts-radxa is the usual
# culprit on our own boards.
if systemctl is-active --quiet tts-radxa.service 2>/dev/null; then
  warn "tts-radxa.service is ACTIVE and holds the EP with an RK1828 TTS model."
  echo "       It cannot co-reside with the LLM. Stop it before starting the LLM:"
  echo "         systemctl stop tts-radxa"
  echo "       (In this delivery TTS runs on the RK3588's own NPU, so it is not needed.)"
else
  ok "no known EP-holding service active"
fi

step "10. observability (depends on the runtime generation)"
# Measured, not assumed: rknn-smi is broken on the V1.0.4 runtime (radxa,
# 2026-07-31: fails for info / info -t memory / info -l, as root, with the EP
# idle — suspected host/EP skew, rc_cc_version=30301 vs ep_cc_version=30201)
# and WORKS on the 1.1.0 package (recomputer-rk3588-devkit, 2026-09-18: board,
# memory, health, pcie_err, temp all reported). So probe it instead of
# declaring either way.
if command -v rknn-smi >/dev/null 2>&1 && timeout 15 rknn-smi info -t memory >/dev/null 2>&1; then
  ok "rknn-smi works — EP memory/health are observable:"
  timeout 15 rknn-smi info -t memory 2>/dev/null | sed 's/^/       /'
else
  warn "rknn-smi cannot read the EP on this runtime generation."
  echo "       Consequence: EP memory and health have NO observability —"
  echo "       model-load success is the only signal."
fi
echo "       NEVER run 'rknn-smi reset': it can wedge the card into a boot state"
echo "       a host reboot may not recover, and the card does not power-cycle"
echo "       with the host. Repeated FAILED model loads also degrade the EP from"
echo "       8 cores to 4, so do not probe capacity by trial and error."

echo
if [ "${fail}" = "0" ]; then
  echo "RESULT: host is ready for the RK1828 LLM container."
  exit 0
fi
echo "RESULT: host is NOT ready — fix the FAIL items above." >&2
exit 1
