# Image Tag → Commit Mapping

Reproducible record of registry image tags built for `seeed-local-voice`.

The voxedge wheel (`deploy/wheels/`) and worker binaries (`deploy/jetson-workers/`)
are **gitignored**, so reproducibility relies on the recorded voxedge commit below
plus the seeed commit. Rebuild the wheel from the recorded voxedge commit
(`uv build --wheel`) and stage the same worker binaries.

| Tag | Seeed commit | voxedge commit | Date | Build host | Registry digest |
|-----|-------------|----------------|------|-----------|-----------------|
| `prod-unified-v8` | `9bad68d99d2c20e0448c6b958f1302b35756829f` | `02e4f0bbe46e4c0cb6513c396cc83aab652ade65` | 2026-06-03 | recomputer-desktop | `sha256:910e298e9b5bf3643133c070618588164f104f9b326cc3f16de8200f5c760f5a` |
| `prod-unified-v9` | `9bad68d99d2c20e0448c6b958f1302b35756829f` | `02e4f0b+mossfix(afdef16)` | 2026-06-17 | seeed-orin-nx | `sha256:bf355e8af0e214be2c76681313f5e2ff590b0f6f692679c5f27a3b6e77c5ac22` |
| `jetson-jp62-trt103-edgellm-v091-20260804-r5` | release lock `orin-nx-edgellm-v091-jp62-trt103-sm87-20260803-r5` | `f738123` (0.0.5a0, superseded by next reproducible rebuild) | 2026-08-04 | seeed-orin-nx | `sha256:b1d9db8d0e61344dc02367bb0114fd6889335f21638cea790e3f795d8226ce5c` |
| `edge-llm-chat-service:v0.9.1-gdn-mtp-8k-20260804-v5` | `634855b` runtime artifact commits (superseded) | n/a | 2026-08-04 | seeed-orin-nx | `sha256:0ec928901a020cd9e67078d2b32837acc28137bc0c3dbfc5b08798e2133efc98` |
| `edge-llm-chat-service:v0.9.1-gdn-mtp-runtime-20260804-v13` | model-neutral v0.9.1 runtime, service `85965efe31a1b1f377a97f4e9be41405bc67737c` | `voxedge==0.0.6a1` | 2026-08-04 | orin-nx | `sha256:3c3e9235efb1ab5c0eac69f47e494a7d03fd381fce83320771e9328801a02116` (143,229,815 bytes) |
| `jetson-v1.16-symlink` | `8e3fd12` | `voxedge==0.0.7a0` | 2026-08-07 | seeed-orin-nx | `sha256:437859d2f96dc53fdefa754744c543d488daf16ba95eef64a0c7bfdf6134379b` |
| `jetson-jp62-trt103-edgellm-v091-vox070a0-slim2` | `8e3fd12` | `voxedge==0.0.7a0` | 2026-08-07 | seeed-orin-nx | `sha256:78b831af480acb81f82fa2a031b57108065e03f1d7e940bcf60e40b4282f5fa7` |
| `jetson-jp62-trt103-edgellm-v091-vox080a0` | `976c140` | `voxedge==0.0.8a0` | 2026-08-08 | seeed-orin-nx | `sha256:ba5b9f359b8a370e9fbccc5a7200dec6c0a49eeabc9f152a55a79c310b7b24d0` |
| `jetson-jp62-trt103-edgellm-v091-vox0011a0-20260818` | `13383c8` | `voxedge==0.0.11a0` | 2026-08-18 | spark | `sha256:2e3752dea4b9a7c3993229caa063e3746f48476d9b436eca6f35cdd4c3685070` |
| `rk-20260903.10` | `2a3cabbfdc8507e6058ae85e09803e1442621b20` + receipt-bound overlays | `0.0.12a0+kokoro.20260903.1` | 2026-09-03 | RK3576/RK3588 | `sha256:fdc480da30610f46075f41a8bf95be5774427a98d3e77c69272cdec1226593c1` |
| `rk-20260909` | `c34af54` + this branch's two `Dockerfile.rk` build fixes | `0.0.13a0` | 2026-09-09 | spark | `sha256:184e9336847a6a0c246c94b311b11d0379d4c90366b8ea0ad6afa0a688b91a58` |
| `rk-20260913.3` (speech, `openvoicestream:`) | UNVERIFIED here — built on an RK3588 from `deploy/docker/Dockerfile.rk` `--build-arg VOXEDGE_VERSION=0.0.15a0`, seeed commit not recorded in the solution | `0.0.15a0` | 2026-09-13 | RK3588 (per solution) | `sha256:d1677071dff68d0be3a3edd26d5959b4598a502ad05162901e509a706723794a` (from the solution's `docker manifest inspect` note, not re-verified) |
| `voiceagent-20260914-runtimekws` (`ovs-agent:`) | UNVERIFIED here — agent image built on an RK3588, commit not recorded in the solution | `0.0.15a0` | 2026-09-13/14 | RK3588 (per solution) | `sha256:a2dc17a304d7941e924d95de0863b53fd3e1869b7caba6ecf64a60e57cfd9569` (from the solution's compose comment, not re-verified) |
| `jetson-jp62-trt103-edgellm-v091-vox080a0-7330af9` (speech, `seeed-local-voice:`) | `7330af9` (per tag suffix) | `0.0.8a0` (per tag) | UNVERIFIED (tag carries no date) | UNVERIFIED | UNVERIFIED — no digest recorded in the solution; run `docker manifest inspect` |
| `rpi-hailo` (local, not pushed) | `4d66f475` + `final-hailo` stage | `0.0.12a0` baked | 2026-09-09 | harvest-pi | `sha256:f6d9bf16557a3a561968e2c942cfcc13112489faafe667a1df95bf5bc4700f65` (local image ID, 657 MB) |

`rpi-hailo` — `Dockerfile.rpi --target final-hailo`, built on `harvest-pi`
(reComputer R2000 series) and tagged locally `asrbench-rpi5-hailo-whisper:r2000`
for the bench in `bench/asr_bench/results/concurrency-harvest-pi-ceiling.md`.
Not pushed to the registry, so the digest above is the local image ID.
The bench numbers were taken on its predecessor
`sha256:2c5069e425585aa73eb7be210fd24587e2884fb401d78b16de9391c8df69726d`,
which differs only by an extra `LD_LIBRARY_PATH=/usr/lib`; that was dropped
after checking that `import hailo_platform` resolves `libhailort.so.4.21.0`
from the bind-mount without it. It is
not reproducible from a clean clone by itself: the stage needs the
operator-supplied HailoRT wheel described in `deploy/docker/wheels/README.md`
(here `hailort-4.21.0-cp311-cp311-linux_aarch64.whl`, md5
`2fde57f853ea66d670a60e68b4ca15da`), and it bind-mounts the host's matching
`libhailort.so.4.21.0` at run time.

The voxedge column is the image's baked `VOXEDGE_VERSION` default (0.0.12a0).
The bench run that produced
`bench/asr_bench/results/concurrency-harvest-pi-ceiling.md` installed
voxedge 0.0.13a0 into the running container over that wheel; the image itself
does not carry it.

`rk-20260909` — `--target final-slim`, pushed to both
`sensecraft-missionpack.seeed.cn/solution/openvoicestream:rk-20260909` and
`.../seeed-local-voice:rk-20260909` (same digest, 992 MB). First RK image
carrying `2815186`, which resolves both RK3576 and RK3588 to
`sense-voice-encoder.<soc>.fp16-scaled.rknn`; every earlier RK tag still
fetches the plain-fp16 RK3576 encoder. Verified in-container:

```
rk3576 -> sense-voice-encoder.rk3576.fp16-scaled.rknn | fp16-scaled: True
rk3588 -> sense-voice-encoder.rk3588.fp16-scaled.rknn | fp16-scaled: True
```

On-device accuracy has not been re-measured on this tag.

**当前默认**：`docker-compose.edgellm-v091-voice.yml` 的 `SPEECH_IMAGE` 缺省值是
`...-vox080a0`（在 slim2 基础上换 voxedge 0.0.8a0 + 老 profile 的
`profile_owned_env` 修复），其构建基础是 `jetson-v1.16-symlink`。两者相对 `jetson-v1.14-hotswap`
一线的差异：剔除 `transformers`、补上 `onnx`、插件由三份实体改为一份实体 + 两条软链接。
运行时镜像层级合计 1.570 → 1.151 GB（省 419 MB / 27%）。

换镜像后必须跑 `python3 scripts/regress_pipeline.py <host:port> <容器名>`，三项全 PASS
才算不衰退。第 3 项（插件软链接 dlopen）只在给了容器名时才跑，别漏。

`v0.9.1-gdn-mtp-8k-20260804-v5` is not a rollback image. Its obsolete cache
verifier rejects the final engine cache's `PROVENANCE.md`; keep it only as
build history. The qualified LLM rollback is
`edge-llm-chat-service:rollback-v080-20260724` with image ID
`sha256:af219111ef86d0c955e5795fc3e1e92c124ba920632681b83c046fd60bc88b11`.

**prod-unified-v8** — single UNIFIED image serving both conversation modes via a
runtime flag: flag-OFF = client-loop pass-through; flag-ON = server-loop
(`voxedge.engine.conversation.ConversationEngine._handle_tool_advertise`,
conversation.py:481). Built from `Dockerfile.jetson.slim` (now `deploy/docker/archive/`) target `final-slim`,
`LANGUAGE_MODE=multilanguage`. Models are HF-fetched at runtime (not baked).

**prod-unified-v9** — OVERLAY on `prod-unified-v8`: reinstalls voxedge with ONLY
the moss `channels=1` stereo→mono downmix cherry-pick (`MossTtsNanoBackend._stereo_to_mono_s16le`)
+ adds the combined `jetson-qwen3asr-moss-nx` profile (Qwen3 ASR via
`QWEN3_ARTIFACT_SET` env + MOSS TTS via `required_engines`, `OVS_TTS_CHANNELS=1`).
Built via overlay Dockerfile on seeed-orin-nx (`/home/seeed/moss-slv-build/`),
not a full rebuild. Verified: downmix present (mono_hex 9600 for stereo[100,200]),
profile parses (asr=jetson.trt_edge_llm, tts=jetson.moss_tts_nano, moss_channels=1).
