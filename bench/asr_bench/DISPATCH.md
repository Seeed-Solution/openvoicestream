# Dispatch template — retail_voice ASR bench (SenseVoice / Whisper x 5 devices)

For the main thread to hand to a per-device executor agent. Read
`docs/reports/retail-voice-asr-bench-matrix-2026-09-09.md` (hub repo) first —
it has the fact-checked support matrix, the RK3576 plain-fp16 accuracy bug,
and what was/wasn't verified. Do not re-derive that from scratch.

## Guardrails (apply to every device)

- `fleet` = `~/.rpty/bin/fleet` (full path). `--sudo` goes before the device
  name, not after.
- Never parse `devices.json` directly.
- `docker compose down` **without** `-v`, ever. Never touch the shared
  volumes `rk-asr-models`, `rk-sensevoice-rknn`, `rk-tts-models`,
  `ovs-aux-models`, `conversational-voice-data`, `seeed-local-voice-data` —
  do not delete or recreate them; other people's containers may depend on
  their contents.
- Check `docker ps -a` before touching anything. If a container is already
  running that isn't yours, don't `rm -f` it — start your own with a
  different name (`asrbench-<device>-<model>`) and, if the port is taken,
  ask before reusing it (`network_mode: host` means only one process can
  bind 8621 at a time).
- `df -h /` before pulling any image or model — RK3576 device was at 90-92%
  free space (4.8-6.1 GB) as of 2026-09-09; treat any board under ~10%
  headroom the same way (check, don't just pull).
- `export HF_ENDPOINT=https://hf-mirror.com` before any model download,
  verified with `bash -c 'echo $HF_ENDPOINT'` (non-login shell) per
  CLAUDE.md mirror policy — this applies on the device via `fleet exec`,
  not just on the Mac.
- One `--model` label does not switch backends — you must start the
  container with the right `OVS_PROFILE`/`ASR_BACKEND` env *before* running
  `bench.py`, and the `--model` flag is just what gets written into the
  output filename/JSON for the report.

## Per-device commands

### Seeed reComputer (RK3576) — SenseVoice already smoke-tested; do the real pass

The plain-fp16 encoder shipped in both cached images
(`seeed-local-voice:rk-20260803b`, `openvoicestream:rk-20260903.10`) has a
known duration-dependent overflow bug (see matrix doc). The fixed
`fp16-scaled` file is already staged at
`/opt/models/svtest-scaled/sense-voice-encoder.rk3576.fp16-scaled.rknn` (+ the
three shared decode assets in the same dir) — **use that mount**, not the
plain file, unless you specifically want to reproduce the bug for a
before/after comparison.

```bash
# 1. check state first
FLEET=~/.rpty/bin/fleet
$FLEET exec <rk3576-host> "docker ps -a; df -h /"

# 2. start SenseVoice (fixed fp16-scaled model)
$FLEET exec <rk3576-host> "docker run -d --name asrbench-rk3576-sv \
  --privileged --network host -v /dev:/dev \
  -v rk-asr-models:/opt/asr/models \
  -v /opt/models/svtest-scaled:/opt/asr/sensevoice-rknn \
  -e OVS_PROFILE=rk3576-sensevoice -e OVS_PROFILE_DEFAULT=rk3576-sensevoice \
  -e ASR_BACKEND=sensevoice_rknn -e ASR_PLATFORM=rk3576 -e RK_PLATFORM=rk3576 \
  -e LANGUAGE_MODE=rk -e ASR_NPU_CORE_MASK=NPU_CORE_0 \
  -e ASR_MODEL_DIR=/opt/asr/models -e SENSEVOICE_RKNN_MODEL_DIR=/opt/asr/sensevoice-rknn \
  -e RK_ARTIFACT_AUTO_DOWNLOAD=0 -e RK_ENSURE_MATCHA_RESOURCES=0 \
  -e OVS_VAD_BACKEND=none -e OVS_PUNCT=0 -e OVS_SPEAKER_EMB=0 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  openvoicestream:rk-20260903.10 \
  python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8621"
# wait for readyz, then from the Mac (or wherever runs bench.py):
cd bench/asr_bench
uv run bench.py --url ws://<rk3576-host>:8621 --model sensevoice --lang zh \
  --segments corpus --concurrency 1,2,4,8 --out results/rk3576-sensevoice-zh.json
# expect concurrency>1 to hard-fail with {"error":"too_many_sessions"} — that's
# server-enforced (execution_policy.mode=serialized), not a bug in bench.py.
# Report c=1 numbers as the real figure and c>=2 as "not supported by this
# profile", not as a degraded-but-working data point.

# 3. Whisper — NOT smoke-tested (disk headroom). Check df -h / again before
#    pulling; whisper-asr.yaml estimates ~1-1.4 GB peak_unified_mb for
#    encoder+decoder. If space allows:
$FLEET exec <rk3576-host> "docker rm -f asrbench-rk3576-sv"  # release NPU/port first
# then start with rk3576-whisper profile (WHISPER_VARIANT=base10 or base20,
# see configs/profiles/rk3576-whisper.json) and repeat the bench.py call with
# --model whisper --lang en (Whisper zh CER is 35-56%, not the intended path).

# cleanup when done
$FLEET exec <rk3576-host> "docker rm -f asrbench-rk3576-sv"
```

Expected time: SenseVoice pass (4 concurrency levels x ~10-20 zh items) ~10-20
min; Whisper pass similar if disk allows.

### Seeed reComputer (RK3588)

```bash
$FLEET exec <rk3588-host> "docker ps -a; df -h /"
```
Compose: `deploy/docker-compose.rk3588.yml`, image
`...seeed-local-voice:rk-qwen3asr-opt-20260610`. Both SenseVoice
(`rk3588-sensevoice`) and Whisper (`rk3588-whisper` 20s or `-10s`) profiles
exist; RK3588's `fp16-scaled` requirement is undisputed across every source
checked (unlike RK3576, no known-good/known-bad ambiguity) — download
`sense-voice-encoder.rk3588.fp16-scaled.rknn` from `harvestsu/sensevoice-rknn`
via hf-mirror and mount it the same way as the RK3576 example above
(swap `rk3576`→`rk3588` in every env var and filename).
Run `bench.py --url ws://<rk3588-host>:8621 --model sensevoice --lang zh
--concurrency 1,2,4,8`, then repeat for whisper/en.

### orin-nano (J3011) and orin-nx (J4012)

```bash
$FLEET exec orin-nano "docker ps -a; nvidia-smi 2>&1 | head -5; df -h /"
$FLEET exec orin-nx "docker ps -a; nvidia-smi 2>&1 | head -5; df -h /"
```
Compose: `deploy/docker-compose.yml`. Confirm the actual image tag on each
box before pulling anything new — README says `jetson-v1.14-hotswap`, the
runbook (possibly stale) says `jetson-v1.12-highperf`; `docker images` on
the box is the source of truth, not either doc.

SenseVoice (`jetson-sensevoice`, `jetson.sensevoice_trt`): the `.plan` TensorRT
engine is built **on first container start** from the shipped ONNX — expect
a multi-minute delay before `/readyz` on a cold model dir; do not treat that
as a hang.

Whisper (`orin-whisper`, `jetson.whisper_trt`): if the `.plan` needs
(re)building on-device, it **must** be built with `--bf16`, never `--fp16` —
the fp16 build passes silently but scores cosine 0.826 against onnxruntime
and drifts off-topic. If a prebuilt `.plan` isn't already on the box, verify
whichever build script the container runs uses `--bf16`
(`bench/perf/whisper/cmp_engine_precision.py` in this repo can confirm engine
correctness after the build — run it before trusting any WER number from a
freshly-built engine).

`execution_policy.shared_resource: gpu` for SenseVoice — same
serialize-at-1-session pattern as RK's NPU; expect the same
`too_many_sessions` behavior at concurrency>1 unless the profile says
otherwise.

### harvest-pi (Pi 5 + Hailo-8, page label: reComputer R2000)

```bash
$FLEET exec harvest-pi "docker ps -a; ls /dev/hailo0 2>&1; df -h /"
```
Compose: `deploy/docker-compose.rpi.yml`, image `...seeed-local-voice:rpi-v1.0-onnx`.

- SenseVoice (`rpi5-sensevoice`) is **CPU-only** — no Hailo acceleration
  exists for this backend. `max_concurrent_sessions: 1` is set defensively
  in the profile (one CPU decode already uses ~64% of the board's cores per
  the profile's own description) — the concurrency sweep here is measuring
  actual CPU contention, not an artificial server-side cap, so watch for
  gradual p95 degradation rather than a hard-fail like the NPU/GPU profiles.
- Whisper (`rpi5-hailo-whisper`, `hailo.whisper`) genuinely uses the Hailo-8
  NPU for the encoder (HEF file), CPU for the decoder. **`/dev/hailo0` is
  exclusive to one process** — if anything else on the box holds it
  (another container with `--device /dev/hailo0`), you'll get
  `HAILO_OUT_OF_PHYSICAL_DEVICES (74)`; check `docker ps -a` for any
  existing Hailo consumer before starting yours. English only for this path
  (zh CER 56-74%, Whisper's own ceiling) — use `--lang en` and skip the zh
  corpus on this profile.
- Resource sampling: `resource_sampler.py --accel none` (CPU/mem only) — no
  confirmed Hailo utilization sysfs hook in this pass; do not add one to the
  sampler without first verifying a real path/command on this exact device
  (`hailortcli` if present, or a `/sys/...` node — check, don't guess).

## After all 5 devices

Aggregate `results/*.json` into one markdown table (per device x model x
lang: p50/p95 latency, RTF, CER/WER, the concurrency level where errors
start) and update the matrix doc's "What this task did NOT do" section to
reflect what got covered.
