# Benchmarks

Measured performance for the v0.8.0 (TensorRT-Edge-LLM) voice stack, with a
focus on **concurrent (N>1) sessions** and **zero-regression vs the v0.7.1
baseline**. Every row names the device, date, and the gate that produced it so
the numbers can be reproduced.

> Single source of truth: `docs/archive/plans/voxedge-v080-consolidation/benchmarks-dataset.md`
> (with per-row `file` provenance + `repro` metadata). This file is the
> outward-facing view of the N>1 / v0.8.0 subset of that dataset.

For the broader cross-device matrix (RTF, TTFA, CER/WER across Jetson / Rockchip
/ Raspberry Pi), see the [Performance section of the README](README.md#performance).

For **Whisper across five accelerators** (Hailo-8, RK3588, RK3576, Orin NX, Orin
Nano) see [`docs/perf/whisper-cross-device-20260827.md`](docs/perf/whisper-cross-device-20260827.md).
Its headline is a negative result worth knowing before planning any work there:
**Whisper does not belong on the TensorRT-Edge-LLM path, and Qwen3-ASR remains the
right choice for Chinese.** Whisper's own Chinese ceiling is 35-56% CER on every
board measured, against Qwen3-ASR's CER 0 on the same corpus — that gap is the
model, not the runtime, so no amount of engine work closes it. Whisper earns its
place as the **English** option on boards where Qwen3-ASR does not fit, and its
numbers are kept in a separate document precisely because the two are measured
differently and must not be read as one table.

> The voice stack has since moved to **TensorRT-Edge-LLM v0.9.0** (see the
> v0.9.0 section directly below). The v0.8.0 numbers further down are retained
> as the prior baseline for comparison.

---

## v0.9.0 upgrade — six-model on-device verification (Orin NX, 2026-07-04)

The **voice stack (ASR + TTS)** was upgraded from TensorRT-Edge-LLM v0.8.0 to
**v0.9.0**. The **LLM service** (Qwen3.5-4B GDN) deliberately **stays on v0.8.0**:
a v0.8.0-vs-v0.9.0 decode bench showed parity within ≲2% with no gain, and the
v0.9.0 `experimental/server` + GDN combination crashes. All six models were
re-verified on a real Jetson Orin NX; ASR/TTS quality is held to the v0.8.0
golden set (no regression).

**Pins:** fork `integration/v090-sparktts` (v0.9.0 tag `1ac0f2b` + our patches),
submodule overlay `repin/v090-overlay`, voxedge wheel `0.0.4a0`. Spec:
[`docs/archive/specs/edgellm-v090-tts-re-port.md`](docs/archive/specs/edgellm-v090-tts-re-port.md).

### SparkTTS-0.5B — the headline: W4A16 becomes the all-round pick

On v0.9.0 the W4A16 INT4-AWQ engine is now faster **and** lighter with **zero
quality loss**, so it is the default recommendation over bf16. Both engines ship.

| Metric | v0.8.0 baseline | **v0.9.0** |
|---|---|---|
| RTF (W4A16) | 0.74 | **0.50** |
| TTFA (W4A16) | 0.64–0.71 s (bf16) · 0.92 s (earlier baseline) | **0.41–0.46 s** |
| Quality (ZH CER / EN WER) | 0 | **0** (zero loss) |
| Engines available | bf16, W4A16 | bf16 **and** W4A16 (W4A16 preferred) |

Device: Jetson Orin NX · Date: 2026-07-04 · Path: `integration/v090-sparktts`.

### The other five models (Orin NX, 2026-07-04)

| Model | v0.9.0 result |
|---|---|
| **Qwen3-ASR 0.6B int4** | Streaming + offline transcription **CER 0** — no regression vs the v0.8.0 golden set. |
| **Qwen3-TTS CustomVoice int4** | 9-row language conditioning, cancel, and EN frame counts all correct. **RTF 0.61.** N=1 by design (session ceiling `min(asr 2, tts 1) = 1`, same as v0.8.0). |
| **Qwen3-TTS Base** | Voice-clone works — the Base embedding controls timbre (CAM++ cross-reference cos **0.366** vs same-reference **0.66–0.70**). |
| **MOSS-TTS-Nano** | TTFA **95–157 ms** (on par with the prior baseline). |
| **Qwen3.5-4B GDN (LLM)** | Stays on v0.8.0; decode ~35 tok/s. |

### N=2 concurrency on v0.9.0

The CustomVoice profile is **N=1 by design**. N=2 shared-engine was re-verified
on **Base** and **SparkTTS**: **~1284 MB VRAM saved**, PCM **byte-identical** to
solo, and **0 CUDA errors over a 50-shot burst**. Note: v0.9.0 has a larger
init-time transient, so a production N=2 needs the **lean** engines
(`code2wav optCodeLen=48` + `max_position_embeddings=4096`).

### v0.9.0 key mechanism changes

- **CuTe-DSL dependency.** TTS runs with `ENABLE_CUTE_DSL=OFF` on our in-house
  tiled FP16 GEMM kernel (no cuBLAS, production-proven); GDN rebuilds sm_87 via
  cutlass-dsl 4.5.2 on-device.
- **Mel front-end retired** → WAV-ingest (`EDGELLM_REQUEST_AUDIO_WAV=1`).
- **Native streaming API** (`streamingChunkFrames` / `onChunkReady`).
- `EDGELLM_PLUGIN_PATH` must be an **absolute** path.

---

## v0.8.0 N>1 concurrency (verified 2026-06-21)

All N=2 numbers come from real on-device burst tests gated on a **byte-identical
audio/transcript** check (concurrent output == solo output) and **zero CUDA /
race errors**. "N=2 verified" is the validated concurrency ceiling; N>2 is
untested on every device.

### ASR — N=2 streaming (Jetson Orin NX, gate v080-0023)

| Metric | Value | Notes |
|---|---|---|
| Streaming partials → final | 9 partials → 1 final | single-session, full streaming path |
| CER (streaming final) | **0.105** | offline-decode on the same clip is ~0.05 |
| N=2 concurrent zh/en isolation | **no cross-talk** | two simultaneous sessions, one zh + one en |
| Session admission (5 concurrent) | 2 admitted, 3 rejected `too_many_sessions` (4429) | `OVS_MAX_CONCURRENT_SESSIONS=2` enforced |
| CUDA errors | **0** | through-service gate (not bare worker) |

Device: Jetson Orin NX · Date: 2026-06-21 · Gate: **v080-0023** (through-service
ASR N=2 streaming).
Source: `~/project/edgellm-v080-migration/docs/plans/v080-0023-*.md` + task records.

### TTS — N=2 slot-pool, int4 talker (Jetson Orin Nano, staggered gate)

Production TTS concurrency is a **slot-pool** (independent lanes, staggered-friendly,
same model as the ASR `SessionLaneManager`), not a lockstep batch-lane.

| Gate | Result |
|---|---|
| **G1** staggered (B not blocked by in-flight A) | PASS |
| **G2** byte-identical (concurrent output == solo) | PASS |
| **G3** session admission (4429 over the limit) | PASS |

| Resource | Value |
|---|---|
| System RAM at N=2 | **~4 GB** (tegrastats peak 5718 / 7620 MB; baseline 1703 MB) |
| Worker RSS | 908 MB |
| OOM | none — fits 8 GB and 16 GB Orin Nano |
| int4 talker engine | **245.9 MB** vs **903 MB** fp16 (−73%) |

Device: Jetson Orin Nano · Date: 2026-06-21 · Gate: TTS N=2 slot-pool int4
staggered (G1/G2/G3).

### TTS — N=2 shared-engine (Jetson Orin Nano, VRAM-saving)

The shared-engine constructor lets the 2nd slot reuse the resident weights, so
only context/KV (not a second copy of the weights) is added.

| Metric | Value |
|---|---|
| N=1 peak | 3805 MB |
| N=2 peak | 5385 MB |
| 2nd slot cost | **+1.6 GB** (context/KV only, not quadratic weights) |
| vs two independent instances | **~436 MB saved** |
| byte-identical (concurrent == solo) | PASS — slot A `154f7880`, slot B `1a5324be` |
| CUDA errors | **0** |

Device: Jetson Orin Nano · Date: 2026-06-21 · Gate: TTS N=2 shared-engine.

### TTS — M5 spike (Jetson Orin NX, phase 5b)

| Check | Result |
|---|---|
| concurrent == solo (RVQ hash) | byte-exact |
| audio md5 | byte-exact |
| CUDA errors | 0 |

Device: Jetson Orin NX · Date: 2026-06-21 · Gate: phase 5b M5 spike.

---

## v0.8.0 vs v0.7.1 baseline — zero regression (Jetson Orin NX)

ASR `--check` regression gate against the v0.7.1 golden set:

| Metric | Value |
|---|---|
| Overall | **17 / 20 PASS** |
| English + clean Chinese | **all pass** |
| Improvements over v0.7.1 golden | several clips better, e.g. `zh_long_01` CER **0.080 → 0.043** |
| The 3 FAIL clips | high-baseline-CER clips where the abs-tolerance gate is brittle (hard-clip), **not a regression** |

Device: Jetson Orin NX · Date: 2026-06-21 · Baseline:
`bench/regression/baselines/v080-c2-before-20260621/`
(artifacts live in `~/project/edgellm-v080-migration`).

---

## int4 vs fp16 talker (Qwen3-TTS 0.6B Base)

| Precision | Talker engine size |
|---|---|
| fp16 | 903 MB |
| int4-AWQ + fp8 | **245.9 MB** (−73%) |

int4 is byte-stable and recommended by default for Orin Nano (fits 8 GB with
N=2). fp16 stays available for accuracy-sensitive comparison. See the
`repro` block in the source dataset for the full build recipe.

---

## Vietnamese dialogue benchmark (Jetson Orin Nano 8GB, 2026-11-10)

End-to-end Vietnamese speech synthesis and transcription verified on `orin-nano` (JetPack 6.2 / TensorRT 10.3 / SM87):

- **ASR**: Qwen3-ASR 0.6B INT4 (`/opt/jv-workers/qwen3_asr_worker`)
- **TTS**: Kokoro-Vietnamese v1.0 hybrid (Prefix TRT FP16 engine + CPU Suffix ORT)
- **Profile**: `configs/profiles/jetson-qwen3asr-kokoro-vi.json`

| Metric | Measured Value | Notes |
|---|---|---|
| Kokoro TRT Prefix Latency | **17.0 ms** | 128 dynamic token sequence, Throughput 25.5 QPS |
| Kokoro E2E TTS RTF | **0.703** | 4.15s speech generated in 2.92s wall-clock |
| Qwen3-ASR Latency | **366.3 ms** | Single utterance transcribe latency |
| First-Word Retention | **100% (0 loss)** | First word `xin` intact (fixed `Vietnamese` language tag parsing) |
| Total Artifact Memory | **~343 MB** | Prefix TRT 29.6 MB + Suffix ONNX 325 MB + Voices 1.04 MB |

Artifact on Hugging Face: `harvestsu/seeed-local-voice-artifacts:models/kokoro-vi-v1_0/engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz`.

---

## Reproduction artifacts (2026-06-21)

| Artifact | Anchor / hash |
|---|---|
| fork release tip | `port/qwen3-tts-base-v080-n1n2` @ `7142a30` |
| rebake image | `seeed-local-voice:v0.8.0-n1n2-rebake` |
| ASR worker | `5ebd436b` |
| TTS shared-engine worker | `190178f6` |
| int4 talker engine | HF `harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8` |
| asr-b2 engine | `4122dfcc` (HF) |
| talker-b2 engine | `f7339e02` |

---

## Methodology

- **N=x verified** = real on-device burst test + MD5 audio gate + zero CUDA/race.
  N=2 covers both the TTS talker batch-lane / slot-pool and the ASR streaming
  slot-pool.
- **ASR latency** = finalize (audio-end → final), VAD 400 ms wait excluded
  (constant, overlaps decode). CER/WER use greedy decode (top_k=1, temp=0) +
  force_language scaffold, replicating the production decode contract.
- **TTS latency** = TTFA (warm: prefill + first chunk). RTF = wall-clock /
  audio duration. The N=2 slow-client TTFA penalty (1.4–5×) is memory-bandwidth
  bound, not a bug.
- **Deployment scope:** these gates ran on `orin-nx` / `orin-nano` profiling
  devices. The `seeed-orin-nx` production robot-arm stack was **not** touched.

See [`docs/deploy-v080-n1n2.md`](docs/deploy-v080-n1n2.md) for the matching
deployment runbook.
