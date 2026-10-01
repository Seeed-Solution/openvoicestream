# syntax=docker/dockerfile:1.6
#
# RK1828 LLM chat service — OpenAI-compatible endpoint backed by the RK1828
# PCIe NPU accelerator card (RKNN3 / RKLLM3 toolchain).
#
# WHY THIS IMAGE EXISTS
#   The repo already ships an LLM chat service for Jetson
#   (edge-llm-chat-service), but nothing for Rockchip. The RK1828 runtime is
#   RKNN3, a completely different toolchain from the RKNN2 used by the voice
#   image, so the two cannot share a base — hence a separate service.
#
# NO MODELS IN THE IMAGE. The four Qwen3-4B artifacts (~3.2 GB) are pulled on
# first start into a volume, per RK1828_ARTIFACT_* below. The image is ~200 MB.
#
# ── HOST PREREQUISITES (the image is NOT self-contained) ──────────────────
#   The card's kernel driver and firmware live on the HOST, not here:
#     * `pcie_rkep` module loaded  (NB: does not persist across reboot — the
#       host needs this made durable, see services/rk1828-llm/BUILD.md)
#     * `rknn3.service` active — it reflashes the EP firmware at host boot
#   A container can only talk to a card the host has already initialised.
#   Acceptance: `lspci` shows [1d87:182a], rknn3.service is active, and
#   /dev/pcie-rkep-0001:11:00.0 exists.
#
# ── SINGLE-EP EXCLUSIVITY ─────────────────────────────────────────────────
#   The card has ONE ~5 GB context. Qwen3-4B at 8192 tokens occupies roughly
#   3.6 GB of it, so nothing else large can be resident at the same time — in
#   particular an RK1828-hosted TTS cannot co-exist with this. (That is why the
#   voice image runs TTS on the RK3588's own NPU instead.)
#
# Build (native arm64; the staged runtime is aarch64):
#   docker build -f services/rk1828-llm/Dockerfile -t edge-llm-rk1828:<tag> .
#   ^ build context is the REPO ROOT, because the runtime is staged under deploy/
#
# Run: see deploy/docker-compose.radxa.yml (needs privileged + the char device).

FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# libstdc++6 / libgcc-s1 : the C++ worker links against them
# curl                   : artifact download + health checks
# ca-certificates        : HTTPS to the artifact host
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libstdc++6 libgcc-s1 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir --index-url ${PIP_INDEX} --upgrade pip \
    && pip install --no-cache-dir --index-url ${PIP_INDEX} \
        "fastapi>=0.141.1" "uvicorn>=0.52.0"

# ── RK1828 runtime (staged build artifact, NOT in git) ────────────────────
# The worker binary is built on-device from the RKNN3 model zoo — see BUILD.md
# for how to stage it. Layout matters: the binary's rpath is $ORIGIN/lib, so
# librknn3_api.so must sit in a `lib/` dir NEXT TO the binary.
COPY deploy/rk1828-runtime/rknn_qwen3_demo   /opt/rk1828/rknn_qwen3_demo
COPY deploy/rk1828-runtime/lib/              /opt/rk1828/lib/
COPY deploy/rk1828-runtime/MANIFEST.json     /opt/rk1828/MANIFEST.json
RUN chmod +x /opt/rk1828/rknn_qwen3_demo

# ── service code ──────────────────────────────────────────────────────────
COPY services/rk1828-llm/rk1828_llm_server.py /opt/rk1828-llm/rk1828_llm_server.py
COPY services/rk1828-llm/artifacts/           /opt/rk1828-llm/artifacts/
COPY services/rk1828-llm/entrypoint.sh        /opt/rk1828-llm/entrypoint.sh
RUN chmod +x /opt/rk1828-llm/entrypoint.sh

# Model artifacts land here; mount a named volume so the ~3.2 GB download
# survives container replacement.
ENV RK1828_MODEL_DIR=/opt/llm/models
RUN mkdir -p /opt/llm/models

# ── runtime configuration ─────────────────────────────────────────────────
# HF_ENDPOINT defaults to the mirror rather than the origin: verified from a CN
# device, huggingface.co is unreachable (curl exit 28) while hf-mirror.com has
# these artifacts synced. Override to https://huggingface.co elsewhere.
ENV RK1828_ARTIFACT_AUTO_DOWNLOAD=1 \
    RK1828_ARTIFACT_REPO_ID=harvestsu/seeed-local-voice-rk-artifacts \
    HF_ENDPOINT=https://hf-mirror.com \
    RK1828_MODEL_ID=Qwen3-4B \
    RK1828_CORE_MASK=ff \
    RK1828_HOST=0.0.0.0 \
    RK1828_PORT=1828

EXPOSE 1828

# 8192 is the verified default: it loaded first try and cost nothing versus 2048
# (TTFT 132 ms, 81.5 tok/s either way). Context is a RUNTIME parameter, not
# compiled into the .rknn, so it can be changed without re-exporting the model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${RK1828_PORT}/health || exit 1

ENTRYPOINT ["/opt/rk1828-llm/entrypoint.sh"]

LABEL org.opencontainers.image.title="Edge LLM (RK1828)" \
      org.opencontainers.image.source="https://github.com/Seeed-Solution/openvoicestream" \
      org.opencontainers.image.description="OpenAI-compatible LLM on the RK1828 PCIe NPU (RKNN3)"
