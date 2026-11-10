# Recommended models — by use case and language

One rule drives model choice: **the use case and the language pick the
models; the board then pulls its own quantized, framework-native build.**
You never hand-pick artifacts — every pairing below is a profile away.

All pairings share the same client API (`/v2v/stream`, `/asr/stream`,
`/tts`): switching models is a restart, not a rewrite.

## Dialogue (full-duplex ASR + TTS)

The TTS pick follows the language — Matcha is our best-measured Chinese
voice; English and other languages have their own best picks.

### Chinese dialogue

| Board | Profile | ASR | TTS | Measured |
|---|---|---|---|---|
| Jetson Orin NX (v0.9.1) | `jetson-edgellm-v091-matcha` | Qwen3-ASR (TRT) | Matcha (TRT) | V2V p50 162 ms; runs beside a local Qwen3.5-4B LLM |
| RK3588 | `rk3588-default` | Qwen3-ASR RKNN W8A8 | Matcha RKNN | V2V p50 528 ms; Matcha RTF 0.05 |
| RK3576 | `rk3576-default` | Qwen3-ASR RKNN W8A8 | Matcha RKNN | V2V p50 1020 ms |
| Raspberry Pi 5 | `rpi` default | sherpa (CPU) | sherpa (CPU) | real-time zh+en commands |

### English dialogue

| Board | Profile | ASR | TTS | Measured |
|---|---|---|---|---|
| Jetson | `jetson-paraformer-kokoro` | Paraformer (bilingual) | Kokoro TRT — 53 English voices | historical ~130 ms TTFT |
| RK3588 | `rk3588-kokoro-rknn` | Qwen3-ASR RKNN | Kokoro RKNN | NPU-accelerated, multilingual output |

### Vietnamese dialogue

| Board | Profile | ASR | TTS | Measured |
|---|---|---|---|---|
| Jetson Orin Nano 8GB | `jetson-qwen3asr-kokoro-vi` | Qwen3-ASR 0.6B (TRT int4) | Kokoro-VI (TRT hybrid: prefix TRT + suffix CPU) | ASR latency 366 ms, prefix TRT 17 ms, TTS RTF 0.703, first-word 100% intact |

### Multilingual / minor languages

| Board | Profile | ASR | TTS | Notes |
|---|---|---|---|---|
| Jetson Orin | `jetson-multilang-*` | Qwen3-ASR (52 langs) | Qwen3-TTS (52 langs, voice clone) | V2V p50 251 ms on Orin Nano hi-perf |
| Jetson Orin | `jetson-moss-tts-nano-trt` | — | MOSS-TTS-Nano (48 kHz stereo) | TTS-only upgrade |
| RK | Piper backend | — | Piper (de/fr/ja…) or Kokoro (ja) | small-language coverage via rkvoice-stream |

## Transcription (accuracy first, no TTS)

Two engines, different language lanes.

### Chinese / multilingual transcription — SenseVoice

| Board | Measured |
|---|---|
| RK3588 | 12-way zero-error, CER 5.13% (every NPU core busy) — [`concurrency-radxa-ceiling.md`](../bench/asr_bench/results/concurrency-radxa-ceiling.md) |
| RK3576 | 16-way zero-error — [`concurrency-cat-remote-ceiling.md`](../bench/asr_bench/results/concurrency-cat-remote-ceiling.md) |
| Raspberry Pi 5 | 8-way zero-error (#85) |

### English long-form transcription — Whisper

| Board | Measured |
|---|---|
| RK3588 / RK3576 | WER 7.50% / 8.51% on the unified 100-segment corpus |
| Jetson Orin NX / Nano | WER 7.62% (TensorRT bf16) |
| RPi5 + Hailo-8 | WER 8.39% (Hailo encoder) |

Full cross-device method and raw JSON: [`bench/asr_bench/results/accuracy-unified-corpus.md`](../bench/asr_bench/results/accuracy-unified-corpus.md).

## Why these numbers hold

Every model is quantized per target (W8A8 / W4A16 / int4 / fp16-scaled) and
runs on the accelerator's native framework — TensorRT on Jetson, RKNN/RKLLM
on Rockchip, HailoRT on Hailo-8, sherpa-onnx/ONNX Runtime on CPU. No generic
fallback sits in the hot path. See [BENCHMARKS.md](../BENCHMARKS.md) for the
measurement methodology.

Profile mechanics, environment variables, and artifact revisions:
[CONFIGURATION.md](CONFIGURATION.md).
