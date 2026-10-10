# Reproduce OpenVoiceStream from Zero

A freshman-friendly, end-to-end guide. If you have never seen this repo before,
start here. It explains the moving parts, then gives three reproduction paths in
increasing depth:

- **Path A — Just run it.** Pull a prebuilt image, let it download model
  artifacts from Hugging Face on first start, hit `/readyz`. (Most people want
  this.)
- **Path B — Rebuild the engines from scratch.** Regenerate the TensorRT / RKNN
  engines on a build host and republish them. (Deep reproduction.)
- **Path C — Build the container images.** Pointer to
  [`BUILD_IMAGES.md`](BUILD_IMAGES.md).

Jargon is defined on first use. Concrete commands and paths come from the actual
manifests and scripts in this tree — where a source was missing or out of scope
it is flagged explicitly rather than invented.

---

## 1. What this is + the topology

**OpenVoiceStream (a.k.a. SLV, "seeed-local-voice")** is a local, on-device voice
stack: streaming **ASR** (automatic speech recognition — speech → text),
streaming **TTS** (text-to-speech — text → audio), VAD (voice-activity
detection), barge-in, and an optional conversational agent that wires in an LLM.
It runs fully on edge hardware (NVIDIA Jetson, Rockchip RK3576/RK3588, Raspberry
Pi) with no per-call cloud speech bill.

### The repos and what each owns

| Repo | Role | Produces |
|---|---|---|
| **`seeed-local-voice`** (this repo) | The **product**. | The two app images below. |
| └ `server/` | FastAPI voice service (ASR/TTS/LLM-loop). HTTP/WS API on container port `8000` (host `8621`). | → `seeed-voice` container |
| └ `agent/` | The conversational agent: `ovs_agent` framework + `ovs_agent/apps/voice_arm` (the SO-ARM robot-arm app). | → `voice-arm` container |
| **`voxedge`** (`../voxedge`) | The edge voice **library** — backend ABCs + the conversation engine, extracted as the open-core foundation. Installed as a wheel. | `voxedge-*.whl` |
| **`jetson-voice-engine`** (submodule `third_party/jetson-voice-engine`) | Build-time only: Jetson Qwen3 ASR/TTS ONNX export + TensorRT engine build + worker binaries. Folds in the **voxedge-engine** overlay (`engine-overlay/`, the NVIDIA TensorRT-Edge-LLM fork). | TRT engines + workers → HF |
| **`rkvoice-engine`** (`../rkvoice-engine`) | Build-time only: Rockchip RKNN/RKLLM model conversion pipeline. | `.rknn` / `.rkllm` → HF |

"Submodule" = a pinned checkout of another git repo nested inside this one. Clone
with `--recurse-submodules` (or run `git submodule update --init --recursive`).

> **voxedge today (Phase 1a).** `voxedge` currently ships clean backend ABCs, the
> ported conversation engine, and **mock** backends so the engine runs on a
> laptop with no CUDA (see `../voxedge/README.md`). The product's own backends
> still live in `server/core/`. Per [`DEVELOP.md`](../DEVELOP.md), install voxedge
> editable for local dev (`uv pip install -e ../voxedge`); the Docker image's
> `pip install`/COPY of the voxedge package is still a tracked TODO.

### The three runtime containers

| Container | Image | What it does |
|---|---|---|
| **seeed-voice** | `…/seeed-local-voice:<target>-slim` | The voice service (`uvicorn server.main:app`). |
| **voice-arm** | `…/voice-arm:<tag>` | The agent (`ovs-agent run voice_arm`); talks to seeed-voice over WS and to the LLM over HTTP. |
| **edge-llm** | `…/edge-llm-chat-service:<tag>` | The local LLM. Pulled from the registry — **not built in this repo.** |

For a voice-only deployment you only need **seeed-voice**. Add **voice-arm** +
**edge-llm** for a full local voice + LLM + robot-arm loop.

### Data flow: build host → HF → device first-start → container

```
  BUILD HOST (Path B)                 HUGGING FACE                 EDGE DEVICE (Path A)
  ┌──────────────────┐          ┌──────────────────────┐       ┌────────────────────────┐
  │ qwen3-edgellm-   │  upload  │ harvestsu/qwen3-     │       │ docker run seeed-voice │
  │ jetson (TRT)     ├─────────►│  edgellm-jetson-     │       │  (slim: NO models baked)│
  │ rkvoice-engine   │          │  artifacts           │ pull  │                        │
  │ (RKNN)           │          │ harvestsu/seeed-     │◄──────┤ first start →          │
  │ kokoro/matcha…   │          │  local-voice-arti…   │       │  engine_resolver       │
  └──────────────────┘          │ harvestsu/seeed-     │       │  downloads + SHA-256   │
                                │  local-voice-rk-arti…│       │  → /opt/models volume  │
  (official sherpa/Matcha       └──────────────────────┘       │ → GET /readyz = 200    │
   assets pulled from their                                    └────────────────────────┘
   own upstream sources)
```

The **slim** images bake **no** models or engines. On first start the runtime's
artifact resolver downloads exactly what the active profile needs from Hugging
Face (or, for the legacy `zh_en` path, official sherpa-onnx / Matcha sources +
the Seeed CDN), verifies SHA-256, and caches it in the `/opt/models` Docker
volume. Subsequent starts are offline.

---

## 2. Path A — Just run it (consume prebuilt artifacts)

This is the common case: take a published image, give it an **empty** models
volume, and let it self-provision on first boot.

### A profile is the unit of selection

A **profile** (`OVS_PROFILE`) picks the ASR + TTS backends and their engines.
`LANGUAGE_MODE` is the coarse selector (`zh_en` default, `en`, `multilanguage`);
profiles set finer env defaults. Explicit env vars always override profile
defaults.

### Quick start (auto-detect)

```bash
git clone --recurse-submodules https://github.com/Seeed-Solution/openvoicestream.git
cd openvoicestream
deploy/install.sh --pull --verify     # auto-detects Jetson / RK / RPi
```

`install.sh` validates the host, selects the right compose file, pulls the image,
starts the service, and runs health/capability/TTS-smoke/round-trip checks. After
startup the service listens on `http://<device>:8621`.

### Per-device compose + profile

For an Orin NX, the current v0.9.1 production path is:

```bash
deploy/install.sh --target orin-nx --pull --verify
```

It starts Qwen3-ASR + Matcha on `:8621` and Qwen3.5-4B GDN/MTP 8K on
`:8000`. Runtime images do not contain engines; the chosen profile downloads
each model from its own immutable revision and verifies the release lock. The
final OCI-provenance LLM image is
`sensecraft-missionpack.seeed.cn/solution/edge-llm-chat-service:v0.9.1-gdn-mtp-runtime-20260804-v13`
(`sha256:3c3e9235efb1ab5c0eac69f47e494a7d03fd381fce83320771e9328801a02116`). See
[`deploy/jetson-orin-nx-v091.md`](deploy/jetson-orin-nx-v091.md). The table
below retains the generic/legacy platform paths for rollback and other boards.

To exercise the 4K model-level payload, select the published profile:

```bash
EDGELLM_ENGINE_PROFILE=4k \
deploy/install.sh --target orin-nx --pull --verify
```

The locked payload SHA-256 values are `06273e358a579590bb8344b451aa35c89983cd99401339fb1858d61af4dbd107` (4K)
and `9208e46d61a4f1440ac68a312e35dde3d04b88edf0e4ee12b32210e7190d3325` (8K).
Their immutable revisions are `9f2c2059341fd2135cc3a0ec09e05150277ea5b6`
(4K) and `adb1c78fb61513e2d7d8e7f889f6196dbefb1e5e` (8K).

| Target | Compose file | Example profile | Validated image |
|---|---|---|---|
| **Jetson** (Orin Nano/NX/AGX) | `deploy/docker-compose.yml` | `jetson-qwen3asr-matcha-nx` (default multilang), `jetson-multilang-highperf` (heavy), `jetson-zh-en` (lightest) | `seeed-local-voice:jetson-v1.14-hotswap` |
| **RK3588** (Radxa ROCK 5T) | `deploy/docker-compose.radxa.yml` | `rk3588-default` (Qwen3 ASR W8A8 + Matcha), `rk3588-kokoro-rknn` | `seeed-local-voice:rk-qwen3asr-opt-20260610` |
| **RK3576** (BPI-M5 Pro) | `deploy/docker-compose.rk.yml` | `rk3576-default` (Qwen3 ASR W8A8 + Matcha) | `seeed-local-voice:rk-qwen3asr-opt-20260610` |
| **Raspberry Pi** (4/5) | `deploy/docker-compose.rpi.yml` | `rpi5-default` | `seeed-local-voice:rpi-v1.0-onnx` |

Images live under the registry `sensecraft-missionpack.seeed.cn/solution/`.

Manual compose, empty volume, fresh first-boot download:

```bash
# Jetson, lightweight zh_en path (Paraformer ASR + Matcha TTS):
docker compose -f deploy/docker-compose.yml up -d

# Jetson, multilingual Qwen3 ASR + Matcha TTS on Orin NX:
OVS_PROFILE=jetson-qwen3asr-matcha-nx docker compose -f deploy/docker-compose.yml up -d

# Rockchip RK3588:
OVS_PROFILE=rk3588-paraformer-matcha docker compose -f deploy/docker-compose.radxa.yml up -d
```

The RK Paraformer profile names above are stable aliases for the current
validated artifact sets:
`rk3588-paraformer-hybrid-rknn-decoder-2026-06-09` and
`rk3576-paraformer-hybrid-rknn-decoder-2026-06-09`. The older CPU-decoder
Paraformer path is deprecated.

### China-mirror note

On a network without a route to `huggingface.co` (e.g. inside the Great
Firewall), set the mirror before first start so the artifact pull resolves:

```bash
HF_ENDPOINT=https://hf-mirror.com OVS_PROFILE=jetson-qwen3asr-matcha-nx \
  docker compose -f deploy/docker-compose.yml up -d
```

`HF_ENDPOINT` is honoured by every downloader (`server/core/hf_artifacts.py`,
`qwen3_artifact_downloader.py`, `moss_artifacts.py`, `rk_artifacts.py`). The
compose file passes it through (`HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}`).

For a separate mirror or CDN serving only the HF artifact repository used by
server/core/hf_artifacts.py, set HF_ARTIFACT_ENDPOINT. It must expose the same
HF resolve layout, including the repository/revision paths and manifest.json;
downloaded files are still checked against the manifest's SHA-256 and size.
HF_ARTIFACT_ENDPOINT takes precedence only for this artifact downloader and
does not change HF_ENDPOINT for the other downloaders. The endpoint should
serve the artifact files itself; do not use a redirect back to huggingface.co
as a substitute for a domestic source.

### Validate

```bash
curl http://<device>:8621/readyz    # → 200 {"status":"ready"} once engines resolved
curl http://<device>:8621/health    # → {"asr": true, "tts": true, "streaming_asr": true}
deploy/verify.sh --url http://<device>:8621 --tts-smoke --roundtrip
```

First Jetson boot downloads ~1.16 GB of engines (~6 min; the ~1.1 GB Qwen3
`llm.engine` dominates). RK ≈ 223 MB, RPi ≈ 570 MB of artifacts.

---

## 3. Path B — Rebuild the engines from scratch (deep reproduction)

Engines are **device-specific compiled artifacts**. Path B regenerates them on a
build host and republishes to the HF artifact repos. Each model family has its
own build entry point; the engine repos own the deep detail — this section gives
the high-level sequence and points at their docs rather than copying them.

> **Where engines land on HF**
>
> | Family | HF repo | Manifest in this tree |
> |---|---|---|
> | v0.9.1 Qwen3 ASR/TTS, MOSS, Spark, Matcha | one immutable HF repository per model | `deploy/artifacts/v091-release-lock.json` |
> | Legacy Qwen3 aggregate / Kokoro / Matcha / MOSS | legacy aggregate repositories | `third_party/jetson-voice-engine/deploy/artifacts/qwen3_manifest.json`, `deploy/artifacts/{kokoro_trt_manifest,moss_manifest}.json` |
> | RKNN / RKLLM (Rockchip) | `harvestsu/seeed-local-voice-rk-artifacts` | `deploy/artifacts/rk_manifest.json` |

### 3a. Qwen3 ASR/TTS on Jetson — the one-shot path

On a Jetson Orin NX (JetPack 6, CUDA 12.6, TRT 10.3, docker `--runtime nvidia`,
~10 GB free):

```bash
git clone https://github.com/suharvest/jetson-voice-engine.git
bash qwen3-edgellm-jetson/scripts/reproduce_qwen3_highperf.sh
#   add --reference path/to/24kHz_mono.wav   to also gate voice clone
```

`reproduce_qwen3_highperf.sh` is the canonical entry point: it clones the three
repos at validated branches, builds the **TensorRT-Edge-LLM** runtime (the NVIDIA
inference engine fork), SHA-256-verifies the HF artifact set, builds the slim
docker image, starts the service, then runs `verify_reproduction.sh` (plugin
symbol check + artifact integrity + TTS→ASR loopback on three Chinese prompts +
optional voice clone). Exit 0 = the whole chain is healthy.

The step-by-step manual fallback (inspect any single layer when it breaks),
including the W8A16 plugin symbol set, the per-branch CMake flags, the engine
build, and the LD_LIBRARY_PATH shadowing gotcha, is in
**`third_party/jetson-voice-engine/docs/reproduce-from-zero.md`**.

**Re-export ONNX from official weights** (only if you need to regenerate the ONNX
intermediates, not just rebuild engines):

```bash
cd third_party/jetson-voice-engine
bash scripts/setup_trt_export_env.sh
scripts/export_qwen3_asr_onnx.sh --model-dir /models/Qwen3-ASR-0.6B --out /tmp/qwen3-asr-onnx
scripts/export_qwen3_tts_onnx.sh --model-dir /models/Qwen3-TTS-0.6B --out /tmp/qwen3-tts-onnx
```

See `docs/export-from-official-weights.md` and `HF_ARTIFACTS.md` (stage + upload
with `package_qwen3_artifacts.py` → `hf upload`) in that submodule.

Published artifact sets today: `orin-nano-highperf-2026-05-10`,
`orin-nx-highperf-2026-05-11` (and `-05-14`), `orin-nano-official-2026-05-10`.

### 3b. The voxedge-engine overlay (TensorRT-Edge-LLM fork)

The fork lives as an **overlay** in
`third_party/jetson-voice-engine/engine-overlay/`: it carries the NVIDIA
upstream pin (`UPSTREAM_PIN`, = v0.7.1) + `addon/` files + `patches/` and
reconstructs the full source tree at build time.

```bash
cd third_party/jetson-voice-engine/engine-overlay
./build.sh --apply-only                              # materialize patched tree (any host)
./build.sh manifests/qwen3-tts-highperf-sm87.toml    # full build — Jetson Orin sm_87 only
```

> **Build-verify is DEFERRED for this overlay.** Per its `README.md`, the overlay
> was extracted on a macOS box with no CUDA/TRT toolchain; only structure
> extraction was done. The compile / plugin / engine / checksum steps **must run
> on an Orin build host** — `build.sh` refuses to compile on non-aarch64.

### 3c. Other Jetson TRT families (Kokoro / Matcha / MOSS / Paraformer)

Per-model build scripts live under
`third_party/jetson-voice-engine/models/<family>/`, run on a Jetson Orin host:

| Family | Entry script | Notes / HF repo |
|---|---|---|
| Kokoro (split-generator TRT) | `models/kokoro/build_kokoro_split_generator_trt.sh` | → `seeed-local-voice-artifacts`; reproduction in [`kokoro-trt-reproduction.md`](kokoro-trt-reproduction.md), frozen record `deploy/artifacts/kokoro_trt_manifest.json`. |
| Matcha (+ Vocos) | `models/matcha/build_matcha_engines.sh` | Default bilingual TTS. |
| MOSS-TTS-Nano | `models/moss-tts-nano/build_moss_tts_engines.sh` | → `seeed-local-voice-artifacts` (`orin-nx-moss-tts-nano-2026-05-23`); manifest `deploy/artifacts/moss_manifest.json`. |
| Paraformer | `models/paraformer/build_paraformer_trt.sh` | Streaming bilingual ASR. |

### 3d. Rockchip RKNN/RKLLM — `rkvoice-engine`

The RK conversion pipeline (ONNX export → graph surgery → RKNN/RKLLM convert →
quantize → manifest) lives in `../rkvoice-engine/addon/models/`. It is
self-authored (no upstream fork, no patches).

```bash
cd ../rkvoice-engine
./build.sh --list                          # documentation / dry-run (any host)
./build.sh manifests/rk3588-kokoro-hybrid.toml   # real conversion — see status below
```

> **RK build-verify is DEFERRED.** Per `../rkvoice-engine/README.md`, RKNN/RKLLM
> conversion needs **x86 + rknn-toolkit2 / rkllm-toolkit** (e.g. `wsl2-local`) or
> an RK device — **macOS has no RKNN toolkit and `build.sh` refuses to convert
> there.** Outputs (`.rknn` / `.rkllm` / tokenizer / voice-pack) publish to
> `harvestsu/seeed-local-voice-rk-artifacts` (default set
> `rk3588-multilang-2026-05-17`), referenced by `deploy/artifacts/rk_manifest.json`.

---

## 4. Path C — Build the container images

Building the three images (server slim images per device, the voice-arm agent
image) is fully covered in **[`BUILD_IMAGES.md`](BUILD_IMAGES.md)** — the
voxedge-wheel staging, the per-device `final-slim` Dockerfile targets
(`deploy/docker/Dockerfile.{jetson.slim,rk,rpi}`), the validate-before-push step,
and the registry push relay. Not duplicated here.

---

## 5. Deploy + validate the full stack

For voice-only, Path A is the whole deployment. For the full **voice + LLM +
robot-arm** loop you bring up all three containers.

> **Where the SO-ARM compose lives.** `BUILD_IMAGES.md` references the SO-ARM
> solution compose at
> `app_collaboration/solutions/respeaker_flex_soarm/assets/docker/docker-compose.yml`.
> **That path is NOT present in this `seeed-local-voice` checkout** — it lives in
> the separate solution-packaging repo. In this repo the deployable composes are
> `deploy/docker-compose*.yml` (Jetson compose also brings up the NLLB
> `translator` service alongside `speech`). Use those here; reach for the
> solution repo's compose only when assembling the packaged SO-ARM product.

Bring up the voice service and check readiness:

```bash
OVS_PROFILE=jetson-qwen3asr-matcha-nx docker compose -f deploy/docker-compose.yml up -d
curl http://localhost:8621/readyz     # 200 {"status":"ready"}
curl http://localhost:8621/health     # {"asr":true,"tts":true,"streaming_asr":true}
```

The compose `healthcheck` polls `http://127.0.0.1:8000/readyz` (container port).
`/readyz` gates on backend READY + capacity + watchdog; `/health` is the older,
deprecated probe.

### Run the agent (voice-arm)

The agent connects to the voice service over one persistent WebSocket to
`/v2v/stream` and streams LLM tokens straight back to TTS:

```bash
# In the voice-arm container (or a checkout with the agent installed):
ovs-agent run voice_arm
```

`voice_arm` resolves to `ovs_agent.apps.voice_arm.app:App`. Its
`config.yaml` points at the voice service
(`ws://${VOICE_SERVICE_HOST}:${VOICE_SERVICE_PORT}/v2v/stream`) and the LLM
(`http://${LLM_SERVICE_HOST}:${LLM_SERVICE_PORT}/v1`).

**SO-ARM serial note.** The robot arm is a USB-serial device. The app discovers
the port via `ARM_PORT` (default `auto`); the typical device node is
`/dev/ttyACM0`. The container needs that device mapped through
(`--device /dev/ttyACM0` or compose `devices:`). The reSpeaker Flex mic array is
the audio input; the entrypoint resolves its index.

---

## 6. Gotchas

- **CN mirror (Hugging Face).** On walled devices set
  `HF_ENDPOINT=https://hf-mirror.com` before first start, or the artifact pull
  hangs/fails. Honoured by all four downloaders. Note: `hf-mirror.com` is
  read-only — it does **not** accept uploads (publish from a host with a real
  `huggingface.co` route).

- **CN mirror (Debian/pip during image build).** The RK/RPi (Debian-base)
  Dockerfiles and `agent/Dockerfile` default apt/pip to Tsinghua mirrors
  (`APT_MIRROR`, `PIP_INDEX`) because the edge build hosts have no cross-wall
  route to `deb.debian.org` / `pypi.org`. Override for non-CN builders:
  `--build-arg APT_MIRROR=deb.debian.org --build-arg PIP_INDEX=https://pypi.org/simple`.

- **Slim images bake no models.** First start needs network (or a pre-staged /
  bind-mounted models volume). After the first successful boot the `/opt/models`
  volume is populated and the service runs offline. For air-gapped deploys
  pre-stage artifacts and set `OVS_AUTO_DOWNLOAD_ARTIFACTS=0`.

- **Host-signature-keyed TRT tarballs.** Jetson engine bundles are keyed by a
  host signature `sm<NN>-trt<X.Y>-jp<X.Y>-cuda<X.Y>` (e.g.
  `sm87-trt10.3-jp6.2-cuda12.6`). The published engines cover **only sm87 /
  JetPack 6.2 / CUDA 12.6** (Orin Nano/NX/AGX). A different SM or TRT version
  will not load — you must rebuild via Path B for that host.

- **Engine overlays are not built on macOS.** Both engine overlays
  (`engine-overlay/` for Jetson, `../rkvoice-engine` for RK) have build-verify
  **DEFERRED**: they refuse to compile/convert off their target toolchain host
  (Jetson Orin sm_87 / x86 rknn-toolkit2). Do the heavy build on the right host.

- **Path mismatch in older docs.** Some docs predate the `app/`→`server/` rename
  (the service now runs `uvicorn server.main:app`, and downloader modules are
  under `server/core/`). If a command references `app/...`, read it as
  `server/...`.

---

## See also

- [`README.md`](../README.md) — feature overview, API reference, benchmarks.
- [`BUILD_IMAGES.md`](BUILD_IMAGES.md) — building & publishing the images.
- [`DEVELOP.md`](../DEVELOP.md) — local dev setup (voxedge editable install).
- [`kokoro-trt-reproduction.md`](kokoro-trt-reproduction.md) — Kokoro TRT repro.
- `third_party/jetson-voice-engine/docs/reproduce-from-zero.md` — Qwen3 manual
  fallback.
- `deploy/artifacts/MANIFEST.md` — the canonical artifact-repo map.
