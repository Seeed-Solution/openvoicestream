# Runtime Artifacts Manifest

This directory documents large artifacts that are not meant to be committed to
git. Deployment scripts now prefer Hugging Face manifests and official upstream
model sources over checked-in binary blobs.

## TensorRT-Edge-LLM v0.10 candidate

The production v0.10 release is frozen in `v010-release-lock.json` with
`release_state=published_and_qualified`. It pins both runtime images by registry
digest and retains `v091-release-lock.json` as the explicit rollback identity.
`v010-published-image-identities.json` records the external Registry result;
`v010-production-image-verification.json` records byte identity, NX/Nano runtime
smokes, and the no-regression gates used before publication. Production Compose
defaults to those digest-pinned images, while profile/model overrides remain
explicit.

`v010-candidate-release-lock.json` is an additive, deliberately non-deployable
identity used while Orin NX/Nano gray qualification is in progress. Unpublished
model revisions and image identities remain explicit JSON `null` values; they
must never be copied from the v0.9.1 lock. Qualified candidate runtime binaries
already have frozen size/SHA identities. Speech and LLM artifacts use separate
namespaces because their TensorRT plugins are different builds even when their
in-image destination path is the same.

`v010-publication-plan.json` covers every new target payload exactly once. It
records the verified staging location, proposed branch, payload/package hashes,
and whether the payload is ready or blocked. `external_upload_authorized` stays
false until the immediate pre-upload user confirmation; the plan is not an
upload credential or permission by itself.

Publication uses two fail-closed phases to avoid making an image contain its
own not-yet-known registry digest:

1. `--require-image-build-ready` accepts only
   `artifacts_qualified_image_pending`. Model and runtime artifact identities,
   source contracts, and both target qualifications must already be final;
   image ref/digest/ID/size must still be `null`. This pre-image lock is what
   production images embed and recheck at startup, including their owned
   runtime bytes.
2. After push, the external release job fills the digest-pinned image refs and
   runs `--require-published`. Only that final lock is deployable.

The external lock is authoritative for image identity. The embedded pre-image
lock is authoritative for inputs; neither phase can substitute for the other.
The speech image renders six NX and four Nano production profiles from that
pre-image lock. Rendering rejects candidate artifact-set names, unpublished
model revisions, unqualified targets, and target/model combinations outside the
lock's supported lanes.

The gray Compose/image files use `v010-candidate` container, volume, model, and
binary roots. Building and starting an unpublished candidate each require an
explicit opt-in. The v0.9.1 lock and Compose files remain the rollback path.

## Jetson TensorRT-EdgeLLM v0.9.1 Model Artifacts

The production contract is one Hugging Face repository per model, selected by
the active profile. Exact revisions, payload hashes, and sizes are locked in
`v091-release-lock.json`. `HF_ENDPOINT=https://hf-mirror.com` is mandatory on
edge devices. The older aggregate repository below is retained only for legacy
profiles and reproduction history.

Validated model repositories include Qwen3 ASR, Qwen3 Base/CustomVoice TTS,
MOSS, Spark, Matcha, and the model-level Qwen3.5-4B GDN/MTP payloads. The final
8K payload is 3,876,147,200 bytes with SHA-256
`9208e46d61a4f1440ac68a312e35dde3d04b88edf0e4ee12b32210e7190d3325`; the final
4K payload is 3,876,085,760 bytes with SHA-256
`06273e358a579590bb8344b451aa35c89983cd99401339fb1858d61af4dbd107`.
Their immutable Hugging Face revisions are
`adb1c78fb61513e2d7d8e7f889f6196dbefb1e5e` (8K) and
`9f2c2059341fd2135cc3a0ec09e05150277ea5b6` (4K).

## Legacy aggregate Jetson artifacts

| Item | Location | Notes |
|---|---|---|
| Qwen3 ASR/TTS exported ONNX, TensorRT engines, and worker metadata | [`harvestsu/qwen3-edgellm-jetson-artifacts`](https://huggingface.co/harvestsu/qwen3-edgellm-jetson-artifacts) | Downloaded and SHA-256 verified by the Qwen3 high-performance reproduction flow. |
| Export/build/runtime scripts | [`third_party/jetson-voice-engine`](../third_party/jetson-voice-engine) | Pinned submodule; owns Jetson Qwen3 artifact generation. |
| Slim runtime workers bundled in this repo | [`deploy/jetson-workers`](../deploy/jetson-workers) | Copied into the Jetson image so runtime does not need a full build toolchain. |

Validated image:

- `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.14-hotswap` — current
- Profile used for the 2026-05-21 release gate: `jetson-qwen3asr-matcha-nx` (default
  multilanguage preset) and `jetson-multilang-highperf` (heavy preset)
- Hot-reload support across `kokoro_trt ↔ matcha_trt ↔ trt_edge_llm` (orin-nano + orin-nx verified)
- Previously validated: `jetson-v1.12-highperf` (2026-05-17 gate, kept on registry for rollback)

The lightweight Jetson `zh_en` path still downloads official sherpa-onnx /
Matcha assets at first boot and does not require Qwen3 artifacts.

## Jetson Kokoro TensorRT Artifacts

| Item | Location | Notes |
|---|---|---|
| Kokoro split TensorRT engines and CPU ONNX sidecars | [`harvestsu/seeed-local-voice-artifacts`](https://huggingface.co/harvestsu/seeed-local-voice-artifacts) | Consumed by `OVS_PROFILE=jetson-kokoro-trt*` profiles through the standard engine resolver. |
| Frozen artifact record | `kokoro_trt_manifest.json` | Host signature, SHA-256, runtime path, compatible profiles, and Nano validation numbers. |
| Reproduction guide | [`docs/kokoro-trt-reproduction.md`](../../docs/kokoro-trt-reproduction.md) | Build/repack command, profile matrix, and TTS-to-ASR verification flow. |

Validated artifact:

- `models/kokoro-multi-lang-v1_0/engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz`
- SHA-256: `d0eceb74ecda55f3314d8fc451e7fd3d7d40e5f67100b48b4e7bf9a4684db0c8` (2026-05-21
  rebuild with `MIN_T=4` covering all short inputs; replaces earlier
  `4e6e1109…c0cb` which had `MIN_T=64` and forced CPU ORT fallback on <64-token inputs)

## Rockchip RK3576/RK3588 Artifacts

| Item | Location | Notes |
|---|---|---|
| RKNN/RKLLM generated artifacts | [`harvestsu/seeed-local-voice-rk-artifacts`](https://huggingface.co/harvestsu/seeed-local-voice-rk-artifacts) | Contains only generated files that are not already hosted by official sources. |
| Deploy manifest | `rk_manifest.json` | Source paths, sizes, SHA-256 hashes, runtime version hints, and artifact set names. |
| RK runtime sidecars | [`deploy/rk-runtime`](../deploy/rk-runtime) | Bundled into the RK image to reduce runtime drift across devices. |

Validated image:

- `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-qwen3asr-opt-20260610`

Artifact sets:

- `rk3588-multilang-2026-05-17` — Matcha TTS (closed-loop validated);
  RK3588 Qwen3 ASR defaults to the W8A8 RKLLM decoder after the 2026-06-09
  highperf short-set recheck. FP16 remains in the artifact set for rollback.
- `rk3576-paraformer-hybrid-rknn-decoder-2026-06-09` — Paraformer ASR
  hybrid encoder prefix RKNN + CPU suffix + RKNN decoder, paired with
  `rk3576-paraformer-hybrid-rknn-decoder-matcha`. `/v2v/stream` short-corpus
  validation on RK3576: client-EOS zh/en mean 548/579 ms; with
  `asr_prepare` 500 ms lead zh/en mean 105/165 ms.
- `rk3588-paraformer-hybrid-rknn-decoder-2026-06-09` — same Paraformer ASR
  layout with RK3588-specific RKNN prefix/decoder artifacts, paired with
  `rk3588-paraformer-hybrid-rknn-decoder-matcha`. `/v2v/stream` validation on
  RK3588: client-EOS zh/en mean 424/477 ms; with `asr_prepare` 500 ms lead
  zh/en mean 100/98 ms.
- `rk3588-kokoro-hybrid-2026-05-23` — Kokoro hybrid TTS (CPU prefix ONNX +
  RKNN decoder-front int8 + CPU tail ONNX), paired with `rk3588-kokoro-rknn`
  profile. Files:
  - `opt/kokoro-rknn/kokoro-prefix-cpu.onnx` sha256
    `549fece07d8696ae9ea60f4667e2680b5ca1f41735af545489a378eed3fe1f1c`
  - `opt/kokoro-rknn/kokoro-generator-tail-cpu.onnx` sha256
    `1bdbb9fdea62922e79fa29891471e7797047f16636bff9bfb8944b28ee451bcb`
  - `opt/kokoro-rknn/rk3588/kokoro-decoder-front.int8.rknn` sha256
    `a6de7a2512d83a22cfba8e1793b1f463108126dfee4eb8592a28fe7c268d6861`

## Raspberry Pi / CPU Artifacts

Raspberry Pi uses ONNX through sherpa-onnx. It does not use precompiled engine
artifacts and therefore has no device-runtime compatibility matrix.

Validated image:

- `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rpi-v1.0-onnx`

## Legacy Local Files

The older files under `deploy/artifacts/engines/` and `deploy/artifacts/onnx/` describe the
first Matcha/Paraformer TensorRT split experiments:

- Paraformer encoder/decoder TensorRT plans for Orin Nano.
- Split Matcha estimator TensorRT engines.
- Surgery-produced ONNX files for those builds.

They are retained for historical reproducibility, but new deployment should use
the manifest-driven paths above.
