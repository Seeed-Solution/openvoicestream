# Kokoro RK Deploy Runbook

## Current production path

Use the single registry image
`sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-20260903.10`
on both RK3576 and RK3588. Kokoro ConvOnly is a first-class RKVoice Stream
backend and OVS calls it directly. Combine the platform base compose with
the explicit platform overlay; it overrides inherited `ASR_BACKEND` and
`TTS_BACKEND` values so the profile starts ConvOnly.

```bash
docker compose -f deploy/docker-compose.rk.yml \
  -f deploy/docker-compose.kokoro-convonly-rk3576.yml up -d
# RK3588:
docker compose -f deploy/docker-compose.rk3588.yml \
  -f deploy/docker-compose.kokoro-convonly-rk3588.yml up -d
```

Mount the model bundle and optional Japanese dictionary read-only from
`harvestsu/seeed-local-voice-rk-artifacts`. The accepted model manifest SHA is
`24244b7054bc3626fc22f4ee9bc013ef63aaa5cf409675cafbc10e1c53957ed9` for
RK3576 and `83733c717e0ce5b76ac1295e4827cf3ad2e111955259e9d670897e100fabeb6e`
for RK3588. `KOKORO_JA_DICDIR` points the frontend at the mounted dictionary;
the image contains no dictionary data. The published HF revision is
`3f8d58c8446ec4b18891624ad4ae4ce75e0f3d3e`; the registry digest is
`sha256:fdc480da30610f46075f41a8bf95be5774427a98d3e77c69272cdec1226593c1`.
Verify `/readyz`, EN/ZH/JA, model IDs, finite and
OpenAI WAV/PCM, and finite request cancellation. The accepted `.10` image
returns one completed audio chunk. Source builds use the recorded RKVoice 0.2.0
gitlink. Rollback restores
the previous image tag while retaining the external mounts.

One-page deployment guide for the production Kokoro RK image on Seeed reComputer
(RK3588). For reproduction from-scratch (artifact build, audio
parity, R&D decisions), see
`docs/archive/specs/kokoro-rk-34pct-reproduction-guide.md`.

## TL;DR

The 2026-05-23-rebuilt image is **self-contained**: it bakes in the misaki
ZH G2P stack, the speaker_id fix (commit `6155ebe`), the 4-stage runtime +
3-bucket router (submodule `65b9a13` on
`feat/kokoro-rk-4stage-vocoder-front`), and all 14 active bucket-{8,16,32}
artifacts. No host bind-mount of model files or hot-patched `.py` files is
required.

```bash
docker pull <registry>/openvoicestream:rk-kokoro-2026-05-23-rebuilt   # or local build, see §4

docker run -d --name openvoicestream-kokoro --restart=unless-stopped \
  --privileged \
  --network host \
  -v /dev:/dev \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  -v rk-asr-models:/opt/asr/models \
  -v $(pwd)/configs:/opt/speech/configs:ro \
  -v $(pwd)/deploy/artifacts:/opt/speech/deploy/artifacts:ro \
  --group-add video \
  \
  -e OVS_PROFILE=rk3588-kokoro-rknn-34pct \
  -e RK_ARTIFACT_SET=rk3588-kokoro-hybrid-34pct-2026-05-23 \
  -e RK_ARTIFACT_AUTO_DOWNLOAD=0 \
  \
  `# Bucket-32 (default seq_len=32; baked at /opt/kokoro-rknn)` \
  -e KOKORO_RKNN_VOCODER_FRONT_PATH=kokoro-vocoder-front-half.native.fp16.rknn \
  -e KOKORO_RKNN_TAIL_REST_PATH=kokoro-vocoder-tail-rest-cpu.onnx \
  -e KOKORO_RKNN_TAIL_REST_INT8_PATH=/opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.int8.onnx \
  \
  `# Bucket-8 router (short sentences ≤ 8 phonemes)` \
  -e KOKORO_RKNN_BUCKET8_PREFIX_PATH=/opt/kokoro-bucket-8/kokoro-prefix-cpu-bucket8.onnx \
  -e KOKORO_RKNN_BUCKET8_DECODER_FRONT_PATH=/opt/kokoro-bucket-8/rk3588/kokoro-decoder-front-bucket8.fp16.rknn \
  -e KOKORO_RKNN_BUCKET8_VOCODER_FRONT_PATH=/opt/kokoro-bucket-8/rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn \
  -e KOKORO_RKNN_BUCKET8_TAIL_REST_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx \
  -e KOKORO_RKNN_BUCKET8_TAIL_REST_INT8_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8.onnx \
  -e KOKORO_RKNN_BUCKET8_TAIL_REST_INT8STATIC_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx \
  \
  `# Bucket-16 router (mid sentences 9–16 phonemes)` \
  -e KOKORO_RKNN_BUCKET16_PREFIX_PATH=/opt/kokoro-bucket-16/kokoro-prefix-cpu-bucket16.onnx \
  -e KOKORO_RKNN_BUCKET16_DECODER_FRONT_PATH=/opt/kokoro-bucket-16/kokoro-decoder-front-bucket16.fp16.rknn \
  -e KOKORO_RKNN_BUCKET16_VOCODER_FRONT_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-front-half-bucket16.native.fp16.rknn \
  -e KOKORO_RKNN_BUCKET16_TAIL_REST_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx \
  -e KOKORO_RKNN_BUCKET16_TAIL_REST_INT8_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.int8.onnx \
  \
  -e RK_ARTIFACT_MANIFEST=/opt/speech/deploy/artifacts/rk_manifest.json \
  \
  openvoicestream:rk-kokoro-2026-05-23-rebuilt \
  python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8621
```

## 1. Pre-flight

```bash
# Confirm host has the RK userspace devices.
ls /dev/dri /dev/dma_heap /dev/rga /dev/mpp_service

# Confirm Docker arm64.
docker version | grep -i arch

# Disk: image ≈ 1.1 GB, plus base ≈ 138 MB. Need ≥ 2 GB free in /var/lib/docker.
df -h /var/lib/docker

# The TL;DR `docker run` uses `$(pwd)/configs` and `$(pwd)/deploy/artifacts`
# bind-mounts — run it from a `seeed-local-voice/` checkout root (those paths
# must exist on the host). Also creates a named volume `rk-asr-models` for
# ASR model cache (auto-populated on first start). RK NPU access requires
# `--privileged` + `-v /dev:/dev` + bind of `/proc/device-tree/compatible`
# (RKNN runtime aborts with `_check_container` failure otherwise).
```

## 2. Image acquisition

Two paths:

- **Pulled from registry** (preferred when a private registry is wired
  up): `docker pull <registry>/openvoicestream:rk-kokoro-2026-05-23-rebuilt`.
- **Local build on a Rock 5B** (no cross-compile needed). See §4.

## 3. Container launch

Single `docker run` command above. After it returns, check:

```bash
docker logs openvoicestream-kokoro 2>&1 | grep -E 'Kokoro (4-stage|bucket-)|misaki' | head -10
# Expected:
#   Kokoro 4-stage path active: vocoder_front=... tail_rest=...
#   Kokoro bucket-8 router enabled:  ... (threshold n_tokens<=8)
#   Kokoro bucket-16 enabled: ... (threshold 8<n_tokens<=16)
#   Loaded Kokoro hybrid: prefix=... front=... vocoder=... tail=...
#   misaki ZH G2P (v1.1) loaded for Kokoro RKNN

curl -s http://localhost:8621/health
# {"tts":true,"tts_backend":"rk:kokoro_rknn","asr":true,...}
```

If any line is missing → `docker logs` for full error, then see §6.

### Smoke test (30 seconds)

```bash
for text in 'abc.' '你好。' 'Hello world.' '我感觉很棒。' 'hello world. how are you today?'; do
  curl -sS -o /tmp/t.wav -w "  text='$text' size=%{size_download} http=%{http_code} ttfa=%{time_starttransfer}\n" \
       -X POST http://localhost:8621/tts/stream \
       -H 'Content-Type: application/json' \
       -d "{\"text\":\"$text\"}"
done
```

Expected TTFA (single-shot, cold):

| Text                                  | Bucket | size (bytes)   | TTFA      |
| ------------------------------------- | :----: | -------------- | --------- |
| `abc.`                                | 8      | ~43 000        | ≤ 1.0 s   |
| `你好。`                              | 8      | ~30–50 000     | ≤ 1.0 s   |
| `Hello world.`                        | 16     | ~85–95 000     | ≤ 2.0 s   |
| `我感觉很棒。`                        | 16     | ~85–95 000     | ≤ 2.0 s   |
| `hello world. how are you today?`     | 32     | 251 908        | ≤ 3.5 s   |

`size < 200 B` on any case → bucket mis-routing or misaki failure; check
logs.

## 4. Build the image locally (on a Rock 5B host)

Build context lives under `seeed-local-voice/` checkout root. Bucket
artifacts are large (≈ 450 MB) and not committed to git — they must be
staged into `deploy/kokoro-artifacts/` first (see
`docs/archive/specs/kokoro-rk-34pct-reproduction-guide.md` §3.B for HF download
sources).

```bash
# 1. Clone main repo + submodule at the right pins.
git clone https://github.com/Seeed-Solution/openvoicestream seeed-local-voice
cd seeed-local-voice
git submodule update --init --recursive
( cd third_party/rkvoice-stream && git checkout feat/kokoro-rk-4stage-vocoder-front )

# 2. Stage bucket artifacts (path layout matches Dockerfile.rk COPY).
#    See reproduction-guide §3.B; bucket-32 → deploy/kokoro-artifacts/bucket-32/,
#    bucket-8 → bucket-8/ (with rk3588/ subdir), bucket-16 → bucket-16/ (flat).
mkdir -p deploy/kokoro-artifacts/{bucket-8/rk3588,bucket-16,bucket-32}
# (download from HF harvestsu/seeed-local-voice-rk-artifacts/rk3588/kokoro-hybrid-v1/ …)

# 3. Build (arm64-native on a Rock 5B; ~25–30 min including misaki pip install).
docker build -f deploy/docker/Dockerfile.rk \
  -t openvoicestream:rk-kokoro-2026-05-23-rebuilt .
```

For build-host non-CN PyPI override:

```bash
docker build --build-arg PIP_INDEX=https://pypi.org/simple ...
```

## 5. Roll back

The previous image `openvoicestream:rk-kokoro-2026-05-23` (pre-rebuild) is
retained on the RK3588 host as the rollback target. To revert:

```bash
docker stop openvoicestream-kokoro
docker rm openvoicestream-kokoro
# Re-run with the hot-patch overlay recipe from
# docs/archive/specs/kokoro-rk-34pct-reproduction-guide.md §3.D
# (bind-mounts /tmp/fixed-tts.py, /tmp/fixed-kokoro_rknn.py, and
#  /opt/models/tts/kokoro-bucket-* at the same destinations).
```

The misaki pip install in the writable layer of the old container is
*destroyed* by `docker rm` — re-run `docker exec ... pip3 install
'misaki[zh]'` after the rollback launch.

## 6. Troubleshooting

| Symptom                                        | Likely cause                                                                 | Fix                                                                                  |
| ---------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `bucket-N router enabled` line missing on boot | One of the four `KOKORO_RKNN_BUCKET{N}_*` env vars unset                     | Compare your `-e` block against §3 above.                                            |
| `misaki ZH G2P (v1.1) loaded` missing          | Running an old (pre-rebuild) image                                          | Image must be `2026-05-23-rebuilt` or newer. `docker inspect ... \| grep Image`.    |
| TTFA > 5 s on `abc.`                           | Falling through bucket-8 → bucket-32 (env or artifact missing)              | Check logs for `bucket=32 num_tokens=4` (router mis-fired).                          |
| Chinese smoke returns 4 B WAV                  | misaki not installed (legacy image)                                         | Rebuild image (§4) or fall back to hot-patch deploy + `pip3 install misaki[zh]`.    |
| `/opt/kokoro-rknn/...` not found at startup    | Image is the *baked-in* one but env var points at a non-existent file       | Either correct the env var or remove it (loader has graceful fallback).             |

## 7. Reference

- Reproduction guide (from-scratch, with R&D context): `docs/archive/specs/kokoro-rk-34pct-reproduction-guide.md`
- R&D closure: `docs/archive/specs/kokoro-rk-perf-r-and-d-closure.md`
- Bucket-8 perf: `docs/archive/specs/kokoro-rk-bucket8-ttfa.md`
- Bucket-16 perf: `docs/archive/specs/kokoro-rk-bucket16-mid-ttfa.md`
- HTTP RTF final: `docs/archive/specs/kokoro-rk-34pct-http-rtf-final.md`
- HF artifact mirror: `harvestsu/seeed-local-voice-rk-artifacts/rk3588/kokoro-hybrid-v1/`

## 8. Conv-only integration details (2026-09-03)

This section describes the production `kokoro_convonly` backend in the unified
image documented at the top of this runbook. Legacy hybrid profiles remain
available as separate rollback choices.

### Backend selection and required identity

Use `configs/profiles/rk3576-kokoro-convonly.json` or
`configs/profiles/rk3588-kokoro-convonly.json` for the matching target. Both select
OVS `tts_backend=rk.tts` and rkvoice-stream `TTS_BACKEND=kokoro_convonly`.
The old default and `kokoro_rknn` / `kokora_rknn` aliases remain unchanged.
These profiles disable ASR and automatic artifact download. They do not contain
legacy hybrid paths, fallback settings, CPU affinity, or spin/core-mask tuning.

Required operator values, set **before importing the OVS profile loader**:

- `OVS_PROFILE`: `rk3576-kokoro-convonly` or `rk3588-kokoro-convonly`.
- `RK_PLATFORM`: the same target as the profile and qualified bundle.
- `KOKORO_CONVONLY_ROOT`: the staged, read-only bundle root; the profile defaults
  to `/opt/kokoro-convonly/<platform>`.
- `KOKORO_CONVONLY_MANIFEST_SHA256`: the fixed SHA256 for the matching platform:
  RK3576 `24244b7054bc3626fc22f4ee9bc013ef63aaa5cf409675cafbc10e1c53957ed9`;
  RK3588 `83733c717e0ce5b76ac1295e4827cf3ad2e111955259e9d670897e100fabeb6e`.

The `kokoro.convonly.bundle.v1` manifest binds the platform, FP16 model lineage,
frontend and prefix manifests, 18 Conv artifacts, slopes, merge, and native
library where required, including file sizes and SHA256 values. Nested component
validation remains required. RK3576 and RK3588 RKNN artifacts are not
interchangeable. The old frozen source/binary and benchmark receipts must not
be overwritten while preparing a new integration build. Build provenance and
model qualification are separate evidence requirements.

The YAML entry point `rkvoice_stream.create_from_config` maps only the following
Conv-only fields to its environment configuration; `model_dir` and `mode` do not
select a legacy fallback:

```yaml
tts:
  backend: kokoro_convonly
  platform: rk3576
  bundle_root: /opt/kokoro-convonly/rk3576
  manifest_sha256: "24244b7054bc3626fc22f4ee9bc013ef63aaa5cf409675cafbc10e1c53957ed9"
  intra: 6
  inter: 1
```

`intra` / `inter` map to `KOKORO_FRONTEND_INTRA_OP_THREADS` /
`KOKORO_FRONTEND_INTER_OP_THREADS`. Omitted YAML fields preserve existing
operator values. This YAML bridge follows the package's existing environment
mapping behavior; explicit `KokoroConvOnlyConfig` construction is a separate
backend interface and must not mutate the process environment.

### Frozen policies and limits

| Target | CPU frontend default | Prefix/tail policy |
| --- | --- | --- |
| RK3588 | intra/inter 4/1 | Prefix AUTO; FP16 tail masks 1/2/4; merge AUTO |
| RK3576 | intra/inter 6/1 | Persistent native FP16 tail; masks 1/2/1; all-branch scheduling; query-driven merge packing |

RK3576 6/1 was validated only for the T480 CPU frontend; it is not a full-chain
performance result. RK3588's historical warm weighted text-to-WAV RTF
`0.4773507106` excludes model loading and is not a guarantee under another
workload. No affinity, governor, spinning, or mask experiment is part of this
integration.

Input duration and voice/language must match the approved platform routes.
The backend caps snap at `0.10`; unsupported duration or speaker/pitch/voice
requests are errors, not silent fallback. There is no arbitrary-length text
guarantee. Output is mono PCM16 24000-Hz WAV. The currently deployed `.10`
OVS route reports `supports_streaming=false` and returns one completed chunk.
The RKVoice 0.2.0 source gitlink is recorded for subsequent source builds; this
does not change the already-qualified `.10` image.

One inference runs per backend instance. Cancellation must drain submitted work;
it does not kill native work. Cleanup waits for active work and is idempotent.
That lifecycle contract alone does not establish safe live profile switching:
The VoxEdge 0.0.13a0 unload path forwards cleanup, and OVS can
continue after an unload exception. Require separately verified unload forwarding
and release-failure handling before claiming safe hot-swap. The profile's
serialized execution policy is not a global device lock or CPU reservation.

### Host checks and qualification handoff

From the `seeed-local-voice` repository root, these commands run host-only
registration/profile tests; they do not load models or qualify native inference:

```bash
(cd third_party/rkvoice-stream && \
  .venv/bin/python3 -m pytest tests/test_kokoro_convonly_registration.py -q)
third_party/rkvoice-stream/.venv/bin/python3 -m pytest \
  server/tests/test_kokoro_convonly_config.py -q
```

After an operator has supplied the receipt-bound environment, this read-only
profile check shows effective routing without starting a service or loading NPU
models. Use an environment that already has the OVS host dependencies installed:

```bash
: "${OVS_PROFILE:?Select the matching Conv-only profile}"
: "${KOKORO_CONVONLY_ROOT:?Set the qualified bundle root}"
: "${KOKORO_CONVONLY_MANIFEST_SHA256:?Set the approved receipt digest}"
third_party/rkvoice-stream/.venv/bin/python3 - <<'PY'
import os
from server.core.profile_loader import apply_profile
profile = apply_profile(os.environ["OVS_PROFILE"])
assert profile["tts_backend"] == "rk.tts"
assert os.environ["TTS_BACKEND"] == "kokoro_convonly"
assert os.environ["RK_PLATFORM"] == profile["env"]["RK_PLATFORM"]
print(profile["name"], os.environ["KOKORO_CONVONLY_ROOT"])
PY
```

Device qualification is a separate approved step: preflight connection, disk,
memory and existing processes/containers; retain production PIDs and health;
review the exact artifact inventory and rollback scope before transfer. Record
compiler command/version/flags and source/header/runtime hashes for the new
native library. Verify variable input lengths and failures, frozen fixture
parity, then real text frontend→prefix→tail→iSTFT/WAV, with at least two warmups
and five steady runs. Report full-chain and stage timings separately. Actual
installed VoxEdge/OVS routing must be checked in addition to mock/stub tests.

### Optional Japanese dictionary policy

The default RK image does not include `unidic_lite` (or the `unidic-lite`
distribution). The official Japanese Kokoro route constructs
`misaki.ja.JAG2P()`, which uses `pyopenjtalk`; selecting Japanese does not
require or trigger a UniDic download.

If a future language frontend explicitly requires a dictionary, provision that
dictionary through a controlled, provider-specific on-demand path. Do not
install packages during inference. This image does not claim an automatic
dictionary-download feature.
