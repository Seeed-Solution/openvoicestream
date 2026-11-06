# asr_bench

Generalized latency/concurrency/accuracy benchmark for OpenVoiceStream's
`/asr/stream` WebSocket ASR endpoint. Built for the retail_voice cross-device
SenseVoice vs Whisper comparison (see
`docs/reports/retail-voice-asr-bench-matrix-2026-09-09.md` in the hub repo for
the per-device support matrix and `DISPATCH.md` in this directory for the
per-device run commands).

## Protocol target — read this before running

This tool benchmarks **OpenVoiceStream's own `/asr/stream` WebSocket**
(JSON keyed by `is_final`, query `?language=auto&sample_rate=16000`), the
same endpoint `bench/perf/asr_stream_ws_bench.py` and
`docs/perf-test-runbook.md` already use, on port `8621`.

It does **not** target the `ws://host:8080/ws` protocol documented in
`sensecraft_voice/sensecraft-asr-service/docs/ws_api.md`. That doc describes
a *different*, separately maintained Go service
(`sensecraft-asr-service` / `sensecraft-voice-client`) that sits in front of
OpenVoiceStream in the retail_voice compose stack as an app-compatibility
shim (older protocol: `type: connection|vad|final|error`, `sessionID`,
speaker diarization). See
`sensecraft-solutions/solutions/retail_voice/devices/asr_endpoint.yaml`
(citing `internal/router/router.go:41-44`, `internal/ws/websocket.go:66-72`)
for where that shim lives and why it exists. Per-model timing/CER numbers
have to come from the engine itself; the shim is a routing/auth layer on
top, so `asr_bench` talks to OVS directly.

## What it measures

For each audio segment: feed 16 kHz/16-bit/mono PCM in fixed-size chunks at
1.0x real-time pace (a "pseudo-streaming, actually non-streaming" pass — the
whole segment is sent, then an empty binary frame marks end-of-segment, and
the client waits for the final), matching how the runbook's own bench
script already drives the service.

**Endpointing.** The connection is opened with `?vad=none`: the client's EOS
frame is the only endpoint detector. `/asr/stream` supports exactly one
detector at a time (see "Streaming ASR endpointing" in `server/main.py`), and
running the server VAD alongside a client that sends EOS makes the two race —
the server splits the utterance and delivers its mid-utterance final whenever
the ASR queue lets it, which under load is after the client's EOS. Every
Whisper concurrency sweep before this change ran that way and recorded
shortened transcripts at c>=16 as a result. `--server-vad` measures open-mic
mode deliberately; the client then accumulates every final the server sends,
which is what an open-mic client has to do.

- **Final latency** = wall time from the empty EOS frame to the `is_final`
  message (ms). Not the same as "audio duration + decode time" — see the
  RTF note below.
- **RTF** = `final_latency_s / audio_duration_s`. It is a decode-only RTF
  proxy only when the profile emits nothing before EOS. A profile that
  finalizes mid-feed has already done part of the decoding by then; each
  segment record carries `pre_eos_finals`, the count of `is_final` messages
  seen before EOS, so those rows are identifiable. It uses the same
  definition everywhere, so it is comparable across devices/models.
- **CER/WER** via `jiwer` (character-level for zh, word-level for en),
  reference text from the corpus manifest. Case, punctuation and repeated
  whitespace are removed from reference and hypothesis alike before scoring,
  so an uppercase unpunctuated LibriSpeech reference is not charged an error
  per capital letter and per period.
- **Concurrency** sweep: N independent WebSocket sessions pull segments off
  a shared queue; per-run p50/p95 latency, error count, and aggregate
  throughput (segments/s and audio-seconds/wall-second) are reported so you
  can see where p95 breaks down or errors start.
- **Resource usage** (optional, run separately ON the device — see
  `resource_sampler.py`): CPU%, mem%, and a best-effort NPU/GPU probe
  (`/sys/kernel/debug/rknpu/load` on RK, `tegrastats` on Jetson). Hailo-8 has
  no confirmed sampling hook in this build — left blank rather than guessed;
  fill in only after checking `hailortcli` availability on-device.

## Setup

```bash
cd bench/asr_bench
uv sync                       # runtime deps (websockets, numpy, soundfile, jiwer)
export HF_ENDPOINT=https://hf-mirror.com
uv sync --extra corpus        # + pandas/pyarrow, only needed for the corpus builder
uv run corpus/download_public_corpus.py --limit 20   # builds corpus/manifest.json
```

The corpus builder downloads (see its docstring for exact source paths and a
license/split caveat on the zh subset):
- **zh**: AISHELL-1 (Apache-2.0) via the HF mirror `AISHELL/AISHELL-1`. That
  mirror only carries speakers S0002-S0101 (the official corpus's *train*
  range); the canonical *test* range S0764-S0916 is not present in this
  particular re-upload. The 20 zh items here are a labeled subset for CER
  measurement, not the canonical AISHELL-1 test split — say so in any report
  that cites this number.
- **en**: LibriSpeech test-clean (CC BY 4.0) via `openslr/librispeech_asr`,
  `all/test.clean/0000.parquet` — genuine test-clean split.

The existing `bench/perf/corpus/` (20 items, 5x zh/en x short/long, curated
sentences with known-good transcripts) remains the primary fixed corpus from
`docs/perf-test-runbook.md`; point `--segments ../perf/corpus` at it for a
smaller, faster smoke run.

## Running

```bash
uv run bench.py \
  --url ws://<rk3576-host>:8621 \
  --model sensevoice --lang zh \
  --segments corpus \
  --concurrency 1,2,4,8 \
  --out results/rk3576-sensevoice-zh.json
```

- `--model` is a label only (records which OVS profile you believe is
  running on `--url`) — it does not switch backends. Start the right
  compose/profile first (see `DISPATCH.md`).
- `--segments` points at a directory with a `manifest.json` in the same
  shape as `bench/perf/corpus/manifest.json` or `corpus/manifest.json`
  (built above).
- `--concurrency 1,2,4,8` runs one pass per level; each level reuses the
  same segment set (round-robined across N workers) so throughput and error
  counts are comparable across levels.
- Output: `results/<name>.json` (raw + per-segment rows) and
  `results/<name>.md` (summary table).

### Resource sampling (optional, run on-device)

```bash
# on the device itself, in parallel with the bench run from your workstation
python3 resource_sampler.py --out /tmp/res.csv --interval 1 --duration 180 --accel rknpu   # RK boards
python3 resource_sampler.py --out /tmp/res.csv --interval 1 --duration 180 --accel tegrastats  # Jetson
python3 resource_sampler.py --out /tmp/res.csv --interval 1 --duration 180                      # RPi / Hailo (CPU/mem only)
```

## Known limitations

- `--model` does not start/stop containers or switch `OVS_PROFILE` — that is
  the operator's job (see `DISPATCH.md`); mislabeling `--model` produces a
  correctly-measured but wrongly-captioned result.
- The Hailo-8 accelerator has no confirmed on-box utilization counter in
  this pass; `resource_sampler.py --accel` has no `hailo` option yet. Add
  one only after verifying a real sysfs/hailortcli hook on-device — do not
  guess a path.
- `error_rate` lowercases and strips punctuation on both sides, then scores
  word-level for `en` and character-level for `zh`; it is not tuned
  per-corpus beyond that. `jiwer` is required — without it the script exits
  rather than reporting a different metric under the same name.
