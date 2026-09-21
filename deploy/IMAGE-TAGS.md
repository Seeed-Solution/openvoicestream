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
| `voiceagent-20260914-runtimekws` (`ovs-agent:`, superseded by `voiceagent-20260921-bargein.2`) | UNVERIFIED here — agent image built on an RK3588, commit not recorded in the solution | `0.0.15a0` | 2026-09-13/14 | RK3588 (per solution) | `sha256:a2dc17a304d7941e924d95de0863b53fd3e1869b7caba6ecf64a60e57cfd9569` (from the solution's compose comment, not re-verified) |
| `jetson-jp62-trt103-edgellm-v091-vox080a0-7330af9` (speech, `seeed-local-voice:`) | `7330af9` (per tag suffix) | `0.0.8a0` (per tag) | UNVERIFIED (tag carries no date) | UNVERIFIED | UNVERIFIED — no digest recorded in the solution; run `docker manifest inspect` |
| `openvoicestream:rk-20260918-envownership` (speech, superseded by `rk-20260919-envownership`) | working tree on `a8cddf3` — one changed file (`server/core/profile_loader.py` md5 `a9a006e2…`) | `0.0.15a0` (inherited from the base, unchanged) | 2026-09-18 | macbook (arm64, thin overlay) | index `sha256:81e51b3ec91f36dc6a7262bdc0916ae1009c48d60d1f1ff3d510968c28c3eed2`, linux/arm64 `sha256:52939b54c565763474…` |
| `edge-llm-rk1828:20260919-hostruntime.2` | working tree — `entrypoint.sh` md5 changed only in the half-pair branch | n/a | 2026-09-19 | macbook (arm64, thin overlay) | index `sha256:08b4d4b8d8b0b82beb520c96657707889531b604ea60ce0fb9bcfd137c7e6113` |
| `edge-llm-rk1828:20260919-hostruntime` (superseded by `.2`) | working tree — `entrypoint.sh` md5 `9eb1c6ca…`, `rk1828_llm_server.py` md5 `a66de901…` | n/a | 2026-09-19 | macbook (arm64, thin overlay) | index `sha256:53a5ec99355755c3e66b4c399daaf700394e4d9404a56c8582c0262b3cccfa80` |
| `openvoicestream:rk-20260919-envownership` (speech) | working tree — `server/core/profile_loader.py` md5 `eb4756e7…` | `0.0.15a0` (inherited, unchanged) | 2026-09-19 | macbook (arm64, thin overlay) | index `sha256:c17693df0363a22a3e5d76d344d00b65e3e2a41ac50409f203ea4fb26da7ac7e` |
| `edge-llm-rk1828:20260918-hostruntime` (superseded by `20260919-hostruntime`) | working tree on `a8cddf3` — the two changed files only (`entrypoint.sh` md5 `c515e16d…`, `rk1828_llm_server.py` md5 `a66de901…`) | n/a | 2026-09-18 | macbook (arm64, thin overlay) | index `sha256:cc270b1ca173f9ab1e6476d14c7e256d6ce43a567eca1ffbfa9909e5d30efdbe`, linux/arm64 `sha256:4b91b9b3144936ac33ba27dd8b1e1d70515563e50303930a299228ab74824485` |
| `openvoicestream:rk-20260920-piper-en` (speech, superseded by `rk-20260920-piper-en.2`) | recorded after the fact from the image itself — four files over `rk-20260919-envownership`: `configs/profiles/rk3588-piper.json` + `rk3576-piper.json` (= `38341b3`, i.e. without the `asr_model_id` line `3c23458` added), `deploy/artifacts/rk_manifest.json` (= HEAD), `rkvoice_stream/backends/tts/piper.py` (= rkvoice-stream `3e935dd`) | `0.0.15a0` (inherited) | 2026-09-20 05:45Z (layer timestamps) | macbook (arm64, thin overlay) | index `sha256:47966ea63759bf3727d05f1645032248c0367683a0f1625a6635b772f48f0456`, linux/arm64 `sha256:3235c329d33ae563beccad85dc7340cabcfb99578869ffce78866f4856417e85` |
| `openvoicestream:rk-20260920-piper-en.2` (speech, superseded by `rk-20260921-twosentence`) | two files over `rk-20260920-piper-en`, byte-identical to rkvoice-stream main `5978d04`: `backends/asr/qwen3/streaming.py` md5 `6833d738…`, `backends/tts/piper.py` md5 `5809d480…` | `0.0.15a0` (inherited) | 2026-09-20 | macbook (arm64, thin overlay) | index `sha256:301a8105e18423abd40f83b694a1cfe9f58d44393e174814616c5c85ce429b62`, linux/arm64 `sha256:20f06d70b8006a8f2e37b84e1c45d01c95f5fdd9a1d07bc33b1a42076ee0e16b` |
| `openvoicestream:rk-20260921-twosentence` (speech, superseded by `rk-20260921-bargein`) | nine files over `rk-20260920-piper-en.2`, byte-identical to main `967d120`: the seven `configs/profiles/rk35{76,88}-*.json` Qwen3-ASR profiles, `deploy/artifacts/rk_manifest.json`, `server/core/rk_profile_contract.py` | `0.0.15a0` (inherited) | 2026-09-21 | macbook (arm64, thin overlay) | index `sha256:f05c121d6bb7128e78c47739d5300d6c12b28308c49136e0a58bf6ed1506a81d`, linux/arm64 `sha256:eb05622a4815373eceef86f5dbf6244e9749d42dbb6270e865c8146d106cbc29` |
| `openvoicestream:rk-20260921-bargein` (speech) | two files over `rk-20260921-twosentence`, byte-identical to main `c6259e0` (PR #110): `server/main.py` md5 `c7fb560a…`, `server/core/v2v.py` md5 `6cb9c8c2…` | `0.0.15a0` (inherited) | 2026-09-21 | macbook (arm64, thin overlay) | index `sha256:66c1609280eca64c2e9559f61f047e8d7f593c80d9e136787ac048cdb05fda33`, linux/arm64 `sha256:8e24d89a35215906df10779de85d0045a83fea0662d8a386a968d74dd34d041f` |
| `ovs-agent:voiceagent-20260921-bargein` (agent, superseded by `.2` before any compose pointed at it) | four files over `voiceagent-20260914-runtimekws`, byte-identical to main `c6259e0` (PR #110) | `0.0.15a0` (inherited) | 2026-09-21 | macbook (arm64, thin overlay) | index `sha256:c02ad611689b579a2df6c244ee443911d79d0d8d2384e32966bd5b6b562ed809`, linux/arm64 `sha256:afb8f4561d4be71d3f0c5f7ef3180bb8e27dcc1722b82a69912b77deaee32530` |
| `ovs-agent:voiceagent-20260921-bargein.2` (agent) | four files over `voiceagent-20260914-runtimekws`, byte-identical to main `c03e28f` (PRs #110, #111): `agent/ovs_agent/app_base.py` md5 `dde930cb…`, `slv_client.py` md5 `2df4493c…`, `config.py` md5 `ab119e8c…`, `plugins/llm_availability.py` md5 `907eefa9…` | `0.0.15a0` (inherited) | 2026-09-21 | macbook (arm64, thin overlay) | index `sha256:921f3f9092dfdff2356ee072d7eb50a48561f93e799832f6e39b231c79d57ac9`, linux/arm64 `sha256:1753a7e944e5cb7332f89e9ca0238a19618da7d432efba3798123b5cb7e251c4` |
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

`edge-llm-rk1828:20260918-hostruntime` — a **thin overlay** on
`edge-llm-rk1828:20260731-kvreuse` (base index `sha256:b4d6025fe475fc577c53a0b96e4abb5337d754d9a7c8e0b37589924a08d8b434`),
replacing exactly `/opt/rk1828-llm/entrypoint.sh` and `rk1828_llm_server.py`.
Built as an overlay on purpose: the Mac's `deploy/rk1828-runtime/rknn_qwen3_demo`
does **not** match `MANIFEST.json` (md5 `98d560df…`/859120 B vs the recorded
`600b33ee…`/859984 B), so a full rebuild there would have shipped an
unprovenanced worker binary. Verified before building: the base's copies of both
service files are byte-identical to git HEAD, and the base's four staged runtime
files match `deploy/rk1828-runtime/MANIFEST.json` exactly. Verified after:
`600b33ee…` and `79dad96c…` still in place, both service files replaced, and the
entrypoint's host-runtime alignment exercised inside the image (bundled client
reported as its real `1.0.4`, a mounted fake `1.1.0` copied over it).
Not yet run against the card — no RK1828 host was reachable on 2026-09-18.

`openvoicestream:rk-20260918-envownership` — a **thin overlay** on
`openvoicestream:rk-20260913.3`, replacing only `server/core/profile_loader.py`.
Operator env ownership is no longer a hand-maintained prefix table: it is derived
from every key the shipped `configs/profiles/*.json` and `configs/leaves/*.yaml`
declare (230 keys, 60 of which the old table did not cover — `MOSS_*`,
`PARAFORMER_*`, `SPARKTTS_*`, `SENSEVOICE_*`, `DIAR_*` …), plus the keys of the
profile actually selected, which is what covers an out-of-tree profile chosen via
`OVS_PROFILE_JSON` or bind-mounted into the image. The prefix table is kept in
full — derivation only ever widens ownership.

Two deliberate exceptions: `LD_LIBRARY_PATH` and other process variables are
denylisted (the image sets it and `configs/profiles/jetson-qwen3asr-moss-nx.json`
replaces it on purpose), and `VAD_ENDPOINT_SILENCE_MS` — declared by the four
`rk3*-default/multilang` profiles — now flips to operator-owned **if and only if
an operator actually sets it**. `deploy/docker-compose.rk3588-ha.yml:160` and
`deploy/docker-compose.radxa.yml:82` pass it as `${VAD_ENDPOINT_SILENCE_MS:-}`,
i.e. empty, and empty values were already excluded from the snapshot, so the
shipped deployments are unaffected. Anyone who has a value for it in `.env` will
see it take effect where it was previously ignored; making those four profiles
claim it via `profile_owned_env` is the follow-up that pins the old behaviour
explicitly.

Verified before building: the base's copy of the file is byte-identical to git
HEAD and the base carries `configs/profiles` (51) + `configs/leaves` (20).
Verified after, inside the image: 230 derived keys (same count as on the build
host), `LD_LIBRARY_PATH` excluded, `MOSS_ENGINE_DIR` protected, `PIPER_MODEL_DIR`
protected only when a profile declares it, and the import-time snapshot retains
the operator's original value. Tests: `server/tests -k "profile or leaf or
artifact or rk"` 714 passed / 5 skipped; the full `server/tests tests` run fails
20 tests **both with and without this change** (identical test-id sets — a
pre-existing full-suite ordering issue; those five files pass in isolation).

Building it as an overlay also surfaced that `rk-20260913.3` ships macOS
AppleDouble sidecars (`configs/leaves/._*.yaml`, one per leaf file, binary). They
are inert, but reading one raised `UnicodeDecodeError` at import in the first
build of this overlay, so the derivation skips `._*` by name and no longer
decodes strictly. Its build context was evidently copied from a Mac.

The `20260919-*` pair supersedes `20260918-*` after an independent Codex review
found two real defects in the 0918 build, both since fixed and re-verified:

* `entrypoint.sh` copied the host client as `cp … && log …`. Under `set -e` a
  failing left-hand side of an AND-list does **not** exit the shell, so a
  read-only layer or a partial copy left a mismatched pair in place and the
  service started straight into the 180 s init hang it exists to prevent. It now
  fails closed, compares the shim as well as the backend, and refuses a
  half-present host pair. All three paths exercised inside the built image:
  copy succeeds / half pair → exit 1 / read-only `lib` dir → exit 1.
* `profile_loader` kept an out-of-tree profile's operator key for one call only.
  Switching to a profile that did not declare it restored the operator value in
  step 0 and then deleted it in step 1's stale-clear, so the operator's env went
  silently unset. Reproduced, then fixed with a durable `_EXTRA_OPERATOR_KEYS`
  set; the repro now restores the operator value. The scan also shape-checks
  profile JSON (a valid-but-wrong top-level list used to raise an uncaught
  AttributeError at import) and tolerates an unreadable config dir.

Full suite after both fixes: 1579 passed, 13 skipped.

`.2` differs from `20260919-hostruntime` in one branch: a half-present host pair
no longer exits 1 when the operator has already set RK1828_PREFER_HOST_RUNTIME=0.
Failing there would have blocked a deployment that asked not to use the host copy
in the first place. Both paths were exercised in the built image (PREFER=1 →
exit 1 on the half pair; PREFER=0 → warns and continues to the artifact check).

`openvoicestream:rk-20260920-piper-en.2` — a **thin overlay** on
`rk-20260920-piper-en`, replacing two `rkvoice_stream` modules with the copies on
suharvest/rkvoice-stream main `5978d04` (PRs #7-#11).

* `backends/asr/qwen3/streaming.py` — in `true_streaming`, a continuous utterance
  longer than `QWEN3_ASR_TRUE_ROLL_SEC` (5 s) came back as its last ~5 s: encoder
  frames past the rolling buffer's cap were dropped and nothing kept their text.
  The window is now decoded and committed before it rolls, with
  `QWEN3_ASR_TRUE_ROLL_OVERLAP_SEC` (1.0 s) of frames carried over. Measured on
  radxa with old and new loaded side by side in the speech container, same audio
  at real-time pace: 2.8 s identical and no commit; 7.8 s old "And a case of edge
  computers that…" / new the whole sentence; 13.8 s three commits, whole sentence,
  no repeated, missing or clipped word at a seam. Each commit blocks `feed_audio`
  for 600-900 ms; final latency through the WS service was not measured.
* `backends/tts/piper.py` — English had no pause at "," or ".": `espeak-ng --ipa`
  prints no punctuation and the line breaks that stood in for it were flattened.
  Terminators now reach the model as tokens, and a segment gets a trailing pause
  by its final mark (`PIPER_SENTENCE_PAUSE_MS` 300, `PIPER_CLAUSE_PAUSE_MS` 150).
  Measured on radxa: commas 110-260 ms, sentence ends 300-330 ms, none inside
  "1,000" / "10:30"; "Dr. Smith paid 3.14 dollars at example.com. Thanks a lot!"
  is two segments. Also brings `7442ae7` (decoder-window probe sanity check),
  which the base predates.

Verified before building: the base's `streaming.py` is byte-identical to
rkvoice-stream `7442ae7` and its `piper.py` to `3e935dd`, so the overlay's diff is
exactly the reviewed commits. Verified after, inside the image and again after a
pull on radxa: both md5s, no `._*` sidecars, both modules compile and import.
The code went through three rounds of independent review (Codex); the first two
found real defects -- a window commit that stopped at the first sentence
terminator, an external abort accepted as a complete decode, abbreviation and
domain periods treated as sentence ends -- all fixed before this build.

Known, and not changed by this image: the final decode still runs with
`ASR_FINAL_STOP_ON_PUNCT=1`, so an utterance of two sentences loses the second
("They ship today. Do you want one?" -> "They shipped today.", identical on the
base). The Piper profiles in this lineage are the `38341b3` copies, without the
`asr_model_id` line added in `3c23458`.

`rk-20260920-piper-en` was pushed without an entry here; its row above is
reconstructed from `docker history` and file md5s, not from a build log.

`openvoicestream:rk-20260921-twosentence` — a **thin overlay** on
`rk-20260920-piper-en.2` that sets `ASR_FINAL_STOP_ON_PUNCT=0` for RK3588 and
RK3576: the seven RK Qwen3-ASR profiles, the four artifact sets that repeat the
key in `rk_manifest.json` (`RK_ARTIFACT_CONTRACT_STRICT` refuses to start when a
set and its profile disagree) and `rk_profile_contract.py`.

With the stop on, the final decode is aborted at the first sentence terminator,
so an utterance of two sentences came back as its first: "They ship today. Do you
want one?" -> "They ship today." Measured 2026-09-21, W8A8, the same audio decoded
with the stop on and off in one process (`bench/perf/corpus` short + long plus
four two-sentence clips):

| | RK3588 (radxa) | RK3576 (cat-remote) |
|---|---|---|
| two English sentences, WER | 71.4% / 60.0% -> 14.3% / 0.0% | 57.1% / 60.0% -> 0.0% / 0.0% |
| 10 single sentences | 17.9% -> 17.2% | 14.0% -> 14.0% |
| 10 long clips | 16.7% -> 16.6% | 16.5% -> 15.9% |
| garbage / prompt leaks, stop off | 0 of 24 | 0 of 24 |
| finalize cost of stop off | inside the 50-100 ms embed-cache order effect | +70-90 ms, both run orders |

Every single-sentence decode ends on EOS with the same token count (+-1 on RK3588,
identical on RK3576): the stop saved the EOS token. The redundant Chinese
continuation the 2026-06-02 RK3576 runs reported did not reproduce.

Verified in the image: the nine files match main, all seven profiles read `"0"`,
the contract verifies for both platforms and reports a mismatch when the value is
put back to `1`. Verified as a running service on radxa (a temporary container
from this image on port 8631, same volumes and env as the live one): startup logs
`RK profile contract verified … 'ASR_FINAL_STOP_ON_PUNCT': '0'` and
`RKLLM decoder loaded … final_stop_on_punct=False`; through `/asr/stream?vad=none`
both two-sentence clips come back as one whole final.

The profiles are main's copies, so this image also gains the `asr_model_id` line
from `3c23458` that the `rk-20260920-piper-en` lineage lacked.

Not measured: silence, noise and very short audio with the stop off (the decode
now ends on EOS or `ASR_MAX_NEW_TOKENS=64`), and the worst-case finalize time if
EOS is missed. `/asr/stream` in its default open-mic mode still cuts at 400 ms
pauses and emits one final per segment; that is the endpoint's documented
behaviour, not something this image changes.

`openvoicestream:rk-20260921-bargein` and `ovs-agent:voiceagent-20260921-bargein.2`
— **thin overlays**. Before overlaying, every replaced file was extracted from the
base image and compared byte for byte with main `5ed019e` (the commit before
PR #110); all six matched, so each overlay changes exactly those files.

- speech: `server/main.py` + `server/core/v2v.py` from `c6259e0`. `abort` takes
  `keep_asr`; with it the server cancels TTS only and keeps the in-flight ASR
  utterance, bounded by `OVS_V2V_ABORT_KEEP_ASR_MAX_S` (default 5 s, in the
  connection's sample rate). PR #111 did not touch these files.
- agent: `app_base.py`, `slv_client.py`, `config.py`, `plugins/llm_availability.py`
  from `c03e28f`. Barge-in sends `abort(keep_asr=True)` (V1 and V2 frames);
  "stop, <instruction>" reaches the LLM; an echo-rejected VAD segment no longer
  ends the reply; a new user turn cuts the previous reply that is still playing
  and drops its tail until the new reply starts; the availability probe uses
  `GET /health` when the server reports a `status`.
  `voiceagent-20260921-bargein` (from `c6259e0`) was superseded by `.2` before any
  compose referenced it.

Verified in the images: file md5s equal the build inputs, the modules import.
Verified on the RK3588 + RK1828 devkit (192.168.10.103) with both images and the
local Qwen3-4B: a new question during a 195-token reply cut the old reply and the
new one played; a spoken "Stop, please answer in one sentence." over a reply
reached the LLM whole; a 5.8 s question was transcribed from its first word.
Earlier, with the same files copied into the running containers: barge-in 3/3 with
a cloud LLM and 3/3 with the local LLM, and five turns 45 s apart at TTFT
0.144 / 0.161 / 0.298 / 0.168 / 0.304 s (before the probe fix: 2.07-2.60 s).

Not verified: the echo-segment path on hardware (no echo segment occurred in the
runs; unit test only); RK3576 (same server files, not run).
