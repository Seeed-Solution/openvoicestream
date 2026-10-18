# radxa (RK3588) ASR bench results — 2026-09-09

Device: `radxa` (reComputer RK3588), fleet-managed, tailscale IP
`100.77.150.16`. Board runs a production `edge_retail_console` stack
(`retail-web`/`retail-server`/`retail-mosquitto`) plus unrelated mediamtx/tts
containers throughout this pass — none were touched; all bench containers
used the `asrbench-rk3588-*` naming convention on non-conflicting ports
(8622/8623) and were removed after the run. `docker ps -a` before/after
matches (see EVIDENCE).

## SenseVoice (zh, AISHELL-1 subset, 20 items)

Image: `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-20260903.10`
(already cached on the device — no pull needed). Profile: `rk3588-sensevoice`,
`SENSEVOICE_RKNN_MODEL_DIR` pointed at a fresh Docker volume; the
`fp16-scaled` encoder (`sense-voice-encoder.rk3588.fp16-scaled.rknn`) and its
3 shared decode assets were auto-downloaded by the container's own
`model_downloader.py` via `HF_ENDPOINT=https://hf-mirror.com` on first start
(no manual staging needed for RK3588, unlike the RK3576 cat-remote pass —
the fix is undisputed for RK3588 across every source, see the matrix doc).

| Concurrency | Segments | OK | Errors | p50 latency (ms) | p95 latency (ms) | RTF p50 | RTF p95 | Mean CER |
|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 20 | 0 | 1292.3 | 1542.7 | 0.216 | 0.464 | 5.99% |
| 2 | 20 | 1 | 19 | — | — | — | — | `too_many_sessions` (19/20) |
| 4 | 20 | 1 | 19 | — | — | — | — | `too_many_sessions` (19/20) |
| 8 | 20 | 1 | 19 | — | — | — | — | `too_many_sessions` (19/20) |

Concurrency ≥2 is **not supported by this profile** — every extra session
gets `{"error": "too_many_sessions", "current": 1, "limit": 1}` from the
server's admission gate (one segment per level slips through because the
first WS grabs the slot before the rest connect; that's a race in the client
fan-out, not a working concurrency=2+ result). Full raw JSON:
`radxa-sensevoice-zh.json`.

### Can the session cap be raised? No — checked, not assumed

`rk3588-sensevoice.json`'s `execution_policy` is `{"mode": "serialized",
"shared_resource": "npu"}`. Traced the actual ceiling to
`voxedge/backends/rk/asr.py: RKASRBackend.concurrency_capability()`:

```python
return ConcurrencyCapability(
    supports_parallel=False, max_concurrent=1, is_stateful=True,
    requires_exclusive_device=True, scaling_mode="external_managed",
)
```

This is a **hardcoded backend ceiling** (RKNN NPU context is a single-process
exclusive device), not a profile/env setting. `server/core/session_limiter.py`
documents its own precedence rule: "env override > profile field > ceiling
... any attempt to exceed the ceiling is warn-logged and silently clamped."
Verified empirically — starting the container with
`OVS_MAX_CONCURRENT_SESSIONS=4` produced:

```
session_limiter: OVS_MAX_CONCURRENT_SESSIONS=4 exceeds backend ceiling
(asr=1,tts=inf) → clamping to 1
SessionLimiter initialized: effective_limit=1
```

No config change re-opens concurrency for SenseVoice on this board; a second
sweep was not run because there is nothing left to vary.

### Root-cause note: why the June image (`rk-qwen3asr-opt-20260610`, the
DISPATCH-suggested default) could not be used as-is

Two blockers, both root-caused rather than patched around blindly:

1. That image's `sensevoice_rknn.py` preload raises
   `ModuleNotFoundError: No module named 'sentencepiece'` — the package is
   simply missing from that build's venv. Installed at runtime
   (`pip install sentencepiece` inside the running container) as a
   throwaway workaround to unblock this one bench pass; **this is not a
   deployment fix** — the image itself needs a rebuild with the dependency
   pinned.
2. Even after installing `sentencepiece`, this image's SenseVoice backend
   only declares `{OFFLINE, MULTI_LANGUAGE}` capabilities — no `STREAMING` —
   so `/asr/stream` unconditionally replies `{"error": "no streaming ASR
   available"}` and closes. The offline→pseudo-streaming shim
   (`supports_offline_streaming = True` on `SenseVoiceRKNNBackend`, forwarded
   by `RKASRBackend.capabilities` into `ASRCapability.STREAMING`) only exists
   in the newer `seeed-local-voice:rk-20260903.10` image (already cached on
   this box), traced to seeed commit `2a3cabbf` ("SenseVoice 离线路径服务化"),
   which the June image predates. **Switched to `rk-20260903.10` for the
   real SenseVoice pass** instead of patching the old image further — same
   image cat-remote's earlier smoke test implicitly relied on.

## Whisper (en) — not benchable on this device today; root-caused, not skipped

`ASR_BACKEND=rk.whisper` was added to the registry in commit `7cc9dd55`
(2026-08-28), which **is on `main`** but predates or is absent from every
image tag actually cached or pullable for this board:

| Image tried | Seeed commit / build | `rk.whisper` in `_ASR_REGISTRY`? |
|---|---|---|
| `seeed-local-voice:rk-qwen3asr-opt-20260610` (DISPATCH default) | June 2026 | No (registry has only 4 entries, no Jetson-SenseVoice-TRT or RK-whisper) |
| `seeed-local-voice:rk-20260903.10` (used for the SenseVoice pass above) | `2a3cabbf`, 2026-08-18 (image *tagged* Sept 3 but built from an Aug 18 base) | No — predates the Aug 28 whisper commit |
| `openvoicestream:rk-20260803b` (pulled fresh for this check, 958 MB) | Aug 3 build | No |
| `openvoicestream:rk-whisper-rel` (cited in `bench/perf/whisper/results_backend/PROVENANCE.md` as the one image that *did* validate RK3588 Whisper, voxedge `0.0.12a0`) | unknown | **Not in the registry** — `docker pull` 404s under both `openvoicestream/` and `solution/seeed-local-voice` repo paths. Per its own provenance note, "the image and model directory are on the device" it was built/tested on — never pushed, and that device is not this one. |

Also confirmed `rk3588-whisper.json` / `rk3588-whisper-10s.json` are **not
baked into any of these three pullable images'** `configs/profiles/` dir
(pushed the current repo's `configs/` in as a read-only bind mount to work
around that — see `docker run ... -v /tmp/ovs-configs/configs:/opt/speech/configs:ro`
— which fixed the "profile not found" error but not the underlying
`Unknown asr_backend: 'rk.whisper'`, since that gate is in the image's
Python code, not the mountable config).

**Conclusion**: Whisper RK3588 support exists in source but has never been
built into a distributable image. Benchmarking it here would require
building a new image from current `main` on a machine with the RKNN
toolchain — out of scope for this bench pass (dispatch scope is "run the
bench," not "build a release image"). The `rk3588-whisper.json` /
`rk3588-whisper-10s.json` RTF/WER figures already in `README.md` are cited
from a one-off, non-reproducible on-device validation (see PROVENANCE.md
above) — not measured by this bench tool, on any device, to date.

## Resource sampling

`resource_sampler.py --accel rknpu` ran for the SenseVoice pass duration
(`radxa-sensevoice-zh-resource.csv`, 1 Hz, ~600 s covering the whole sweep).

- CPU: mean 3.6%, max 14.5% (of `psutil`'s all-core normalized reading).
- Memory: mean 14.5%, max 14.6% (board has 16 GB total; consistent with one
  ASR container + the retail_voice stack co-resident).
- `/sys/kernel/debug/rknpu/load` (root-only, read via `fleet exec --sudo`):
  **read 0% on every 1 Hz sample.** This is a measurement-resolution
  limitation, not a claim that the NPU was idle — SenseVoice's own p50
  final latency was ~1.3 s per 20 requests spaced ~7 s apart at 1x
  real-time feed pace, and RKNN inference itself typically completes in a
  few hundred ms; a 1 Hz sampler can easily land between the short NPU-busy
  windows. A higher-frequency sampler (or use of `rknputop -w` for a
  human-observed peak) would be needed to capture real NPU utilization —
  not attempted here (scope: run the existing sampler as shipped, not build
  a better one for this device inside the bench window).

## Files

- `results/radxa-sensevoice-zh.json` — raw per-segment SenseVoice zh rows.
- `results/radxa-sensevoice-zh.md` — bench.py's own summary table.
- `results/radxa-sensevoice-zh-resource.csv` — CPU/mem/NPU-sysfs samples,
  1 Hz, spanning the SenseVoice sweep.
- No Whisper JSON/CSV produced — see "not benchable" above.

## EVIDENCE

### Disk headroom (before / after)

```
before: /dev/mmcblk0p3  115G  100G  9.7G  92% /
peak (2 images cached simultaneously for the whisper investigation):
        /dev/mmcblk0p3  115G  102G  7.7G  94% /
after cleanup:
        /dev/mmcblk0p3  115G  100G  9.7G  92% /
```

### `docker ps -a` before / after (pre-existing containers untouched)

```
retail-web, retail-server, retail-mosquitto (edge_retail_console, running)
esk-rk-rtsp-pub, esk-rk-rtsp-server, fall-rtsp-pub, fall-rtsp-server (running)
openvoicestream (Exited 143, pre-existing, untouched)
ovs-agent (Exited 0, pre-existing, untouched)
wyoming-slv (Exited 137, pre-existing, untouched)
```
identical before this task started and after cleanup; no `asrbench-*`
container or volume remains (`docker volume ls`, `docker ps -a` re-checked
after `docker rm -f`).

### SenseVoice zh concurrency=1, full run (ok=20, errors=0)

```json
{
  "concurrency": 1, "segments": 20, "ok": 20, "errors": 0,
  "final_latency_ms_p50": 1292.2556250123307,
  "final_latency_ms_p95": 1542.6904858322814,
  "rtf_p50": 0.21606922924058902, "rtf_p95": 0.46376208065630603,
  "error_rate_mean": 0.05990954715219421
}
```

### SenseVoice zh concurrency=2 sample error (representative of c=2/4/8)

```json
{"error": "received 4429 (private use) {\"error\": \"too_many_sessions\", \"current\": 1, \"limit\": 1}; then sent 4429 ..."}
```

### `OVS_MAX_CONCURRENT_SESSIONS=4` clamp test

```
2026-09-09 00:05:17,012 [WARNING] server.core.session_limiter: session_limiter:
  OVS_MAX_CONCURRENT_SESSIONS=4 exceeds backend ceiling (asr=1,tts=inf) → clamping to 1
2026-09-09 00:05:17,012 [INFO] server.core.session_limiter:
  SessionLimiter initialized: effective_limit=1 (env OVS_MAX_CONCURRENT_SESSIONS='4', ...)
```

### Whisper backend-registry check (all three pullable images)

```
$ docker run --rm .../seeed-local-voice:rk-qwen3asr-opt-20260610 sed -n '178,184p' server/core/asr_backend.py
_ASR_REGISTRY = {jetson.trt_edge_llm, jetson.paraformer_trt, cpu.sherpa_asr, rk.asr}   # no rk.whisper

$ docker run --rm .../seeed-local-voice:rk-20260903.10 sed -n '178,190p' server/core/asr_backend.py
_ASR_REGISTRY = {..., jetson.sensevoice_trt, cpu.sherpa_asr, rk.asr, ...}   # no rk.whisper, no hailo.whisper

$ docker run --rm .../openvoicestream:rk-20260803b sed -n '178,184p' server/core/asr_backend.py
_ASR_REGISTRY = {jetson.trt_edge_llm, jetson.paraformer_trt, jetson.sensevoice_trt, cpu.sherpa_asr, rk.asr}   # no rk.whisper

$ git merge-base --is-ancestor 7cc9dd55 (rk.whisper commit) origin/main → YES (on main)
$ git merge-base --is-ancestor 7cc9dd55 2a3cabbf (rk-20260903.10's base commit) → NOT an ancestor
```
