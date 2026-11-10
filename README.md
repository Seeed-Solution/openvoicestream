# OpenVoiceStream

> **English** | [中文](README.zh-CN.md)

**Hardware-Accelerated Streaming Voice Engine for Edge Devices** — Out-of-the-box, ultra-low latency, running 100% on-device.

<p align="center">
  <a href="https://github.com/Seeed-Solution/openvoicestream"><img src="https://img.shields.io/github/stars/Seeed-Solution/openvoicestream?style=social" alt="GitHub stars" /></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/ASR-Qwen3--ASR%20%7C%20Paraformer%20%7C%20SenseVoice%20%7C%20Whisper-2f80ed.svg" alt="ASR: Qwen3-ASR, Paraformer, SenseVoice, Whisper" /></a>
  <a href="#model-selection-guide"><img src="https://img.shields.io/badge/TTS-Matcha--TTS%20%7C%20Kokoro%20%7C%20Qwen3--TTS%20%7C%20Piper-f97316.svg" alt="TTS: Matcha-TTS, Kokoro, Qwen3-TTS, Piper" /></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/engines-TensorRT--EdgeLLM%20%7C%20RKNN%20%7C%20HailoRT%20%7C%20sherpa--onnx-16a34a.svg" alt="Engines: TensorRT-EdgeLLM, RKNN, HailoRT, sherpa-onnx" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/deploy-Docker-2563eb.svg" alt="Deploy with Docker" /></a>
  <a href="#supported-devices--requirements"><img src="https://img.shields.io/badge/ecosystems-Jetson%20%7C%20Rockchip%20%7C%20Raspberry%20Pi-65a30d.svg" alt="Supported ecosystems: Jetson, Rockchip, Raspberry Pi" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-facc15.svg" alt="MIT license" /></a>
</p>

<p align="center">
  <a href="#edge-device-recommendations--multilingual-performance-matrix">⚡ <b>Device Matrix</b></a> &nbsp;|&nbsp;
  <a href="#quick-start">🚀 <b>Quick Start</b></a> &nbsp;|&nbsp;
  <a href="#model-selection-guide">💡 <b>Model Guide</b></a> &nbsp;|&nbsp;
  <a href="#ready-made-applications--web-demo">📱 <b>Applications</b></a> &nbsp;|&nbsp;
  <a href="#api-overview">🔌 <b>API Overview</b></a> &nbsp;|&nbsp;
  <a href="#performance--benchmarks">📊 <b>Performance</b></a> &nbsp;|&nbsp;
  <a href="#architecture">🏗️ <b>Architecture</b></a>
</p>

<p align="center">
  <img src="docs/media/hero.png" alt="OpenVoiceStream - streaming ASR and TTS for edge dialogue" width="760" />
</p>

**OpenVoiceStream provides an out-of-the-box, hardware-benchmarked edge voice stack.**
Real-time Automatic Speech Recognition (ASR), Text-to-Speech synthesis (TTS), and ready-to-run interactive applications — full-duplex dialogue, smart home control, robotic voice arms, speech translation, and live captioning — run 100% locally on your own devices.

- **Zero Cloud Dependency**: No external API keys, no per-call meter billing, fully operational offline.
- **Hardware-First Acceleration**: Target-quantized (W8A8 / W4A16 / INT4 / FP16) on native frameworks — TensorRT on NVIDIA Jetson, RKNN on Rockchip, and HailoRT / CPU on Raspberry Pi.
- **Unified Client Interface**: All devices expose consistent OpenAI-compatible HTTP and WebSocket endpoints; switching hardware requires zero client changes.

The underlying streaming runtime is the standalone Python library [`voxedge`](https://github.com/suharvest/voxedge) (`pip install voxedge`). This repository delivers the complete production solution: pre-built container images, quantized model packages, operator runbooks, web management console, and turnkey applications.

---

## Solution Lineup

<p align="center">
  <img src="docs/media/solution-lineup.png" alt="OpenVoiceStream solution lineup" width="900" />
</p>

---

## Edge Device Recommendations & Multilingual Performance Matrix

> 💡 **Model Selection Principles**:
> - **English (en - Primary)**: **Kokoro** across all devices (53 native voices, natural prosody; supported across Jetson, RK3588, RK3576)
> - **Chinese (zh - Secondary)**: Qwen3-ASR + Matcha-TTS (Fast, low VRAM footprint, natural speech)
> - **Japanese (ja - APAC Focus)**: **Qwen3-ASR + Kokoro (Misaki G2P Japanese voices)**, authentic pronunciation
> - **European / 52 Languages**: **Qwen3-ASR + Qwen3-TTS** (Multilingual streaming & 3–10s clone) or **Piper TTS** (Lightweight offline)
> - **Vietnamese (vi - Global Expansion)**: Qwen3-ASR + **Kokoro-VI** (TRT prefix 17.0 ms, end-to-end 0.703 RTF, 100% word retention with 0 truncation)
> - **On-Device LM Reasoning**: Orin NX 16GB supports local offline loop with **Qwen3.5-4B** (~35 tok/s, V2V p50 162 ms)

<p align="center">
  <b>Quick Device Navigation:</b><br>
  <a href="#1-jetson-orin-nano-8gb--mainstream-edge-recomputer-j4012--j401">🟢 Jetson Orin Nano (8GB)</a> &nbsp;|&nbsp;
  <a href="#2-jetson-orin-nx-16gb--on-device-llm-flagship-recomputer-j4012--j4011--super">🔵 Jetson Orin NX (16GB)</a> &nbsp;|&nbsp;
  <a href="#3-recomputer-rk3588--high-concurrency--efficiency-recomputer-r1000--industrial-box">🟠 reComputer RK3588</a> &nbsp;|&nbsp;
  <a href="#4-recomputer-rk3576--cost-effective-iot-gateway-lubancat-3--compact-gateway">🟡 reComputer RK3576</a> &nbsp;|&nbsp;
  <a href="#5-raspberry-pi-5--pure-cpu--lightweight-edge-raspberry-pi-5--recomputer-r1000-pi">🟣 Raspberry Pi 5</a>
  <br>
  <sub>You can also open the <a href="docs/device-language-matrix.html">Interactive Tabbed Matrix in browser</a> for instant switching</sub>
</p>

### 🟢 1. Jetson Orin Nano (8GB) — Mainstream Edge (reComputer J4012 / J401)

> **Hardware Specs**: 40 TOPS compute, 8GB shared unified memory, 7–15W TDP.

| Language | Use Case | ASR Engine | On-Device LM | TTS Engine | Measured Performance (ASR / TTS / V2V) | Recommended Profile | Status |
|---|---|---|---|---|---|---|:---:|
| **🇺🇸 English (en)** | Commercial voice agent (53 voices) | Paraformer / Whisper (TRT) | Cloud API / Local model | **Kokoro-TRT (53 native voices)** | **ASR TTFT: ~50 ms**<br>**TTS TTFT: ~130 ms · RTF: 0.25** | `jetson-paraformer-kokoro` | 🟢 Verified |
| **🇨🇳 Chinese (zh)** | Kiosk / Industrial HMI / Robotics | Qwen3-ASR 0.6B (TRT int4) | Cloud / Local 1.5B | Matcha-TTS (TRT fp16) | **ASR CER: 0**<br>**TTS RTF: 0.15 · V2V: ~500 ms**<br>VRAM footprint: ~1.8 GB | `jetson-qwen3asr-matcha` | 🟢 Verified |
| **🇯🇵 Japanese (ja)** | Consumer electronics / Companion robotics | Qwen3-ASR 0.6B (TRT 52L) | Cloud API / Local 1.5B | **Kokoro-TRT (Misaki Japanese voices)** | **ASR RTF: 0.25**<br>**TTS RTF: 0.28 · V2V: ~600 ms** | `jetson-multilang-highperf` | 🟢 Verified |
| **🌐 Multilingual (52L)** | Cross-border service / Voice cloning | Qwen3-ASR 0.6B (TRT 52L) | Cloud Multilingual LM | **Qwen3-TTS CustomVoice (int4)** | **ASR RTF: 0.30**<br>**TTS RTF: 0.50 · V2V: ~1.5 s**<br>Supports 3–10s voice cloning | `jetson-multilang-highperf` | 🟢 Verified |
| **🇻🇳 Vietnamese (vi)** | Overseas AI Agent / Companion | Qwen3-ASR 0.6B (TRT int4) | Cloud API / 1.5B (Local) | Kokoro-VI (TRT Hybrid) | **ASR: 366.3 ms**<br>**TRT Prefix: 17.0 ms · RTF: 0.703**<br>**100% word retention (0 truncation)** | `jetson-qwen3asr-kokoro-vi` | 🟢 Verified |
| **🎧 Offline Transcription** | Meeting minutes / Audio to text | Whisper-base (TRT bf16) | — (Transcription only) | — | **WER: 7.62%**<br>**TTFT: 18 ms · RTF: 0.008** | `orin-whisper` | 🟢 Verified |

### 🔵 2. Jetson Orin NX (16GB) — On-Device LLM Flagship (reComputer J4012 / J4011 / Super)

> **Hardware Specs**: 100 TOPS compute, 16GB shared unified memory, 10–25W TDP.

| Language | Use Case | ASR Engine | On-Device LM | TTS Engine | Measured Performance (ASR / TTS / V2V) | Recommended Profile | Status |
|---|---|---|---|---|---|---|:---:|
| **🇺🇸 English (en)** | High-throughput English agent | Qwen3-ASR / Paraformer (TRT) | Qwen3.5-4B (Local) / Cloud | **Kokoro-TRT (53 native voices)** | **TTS RTF: 0.18**<br>**ASR TTFT: 18 ms · WER: 7.62%** | `jetson-paraformer-kokoro` | 🟢 Verified |
| **🇨🇳 Chinese Offline** | Air-gapped robotics / Healthcare / Defense | Qwen3-ASR 0.6B (TRT int4) | **Qwen3.5-4B-Instruct**<br>(TRT W4A16, ~35 tok/s, 7.8GB) | Matcha-TTS (TRT bf16) | **V2V p50: 162 ms**<br>**ASR CER: 0 · VRAM total: 10GB**<br>Full on-device concurrency on single GPU | `jetson-edgellm-v091-matcha` | 🟢 Verified |
| **🇯🇵 Japanese (ja)** | High-quality commercial dialogue / Cloning | Qwen3-ASR 0.6B (TRT 52L) | Qwen3.5-4B (Local) / Cloud | **Kokoro-TRT (Misaki Japanese voices)** | **ASR RTF: 0.15**<br>**TTS RTF: 0.20 · Natural prosody** | `jetson-multilang-highperf-nx` | 🟢 Verified |
| **🌐 Multilingual (52L)** | Multi-tenant reception kiosks / Voice clone | Qwen3-ASR 0.6B (TRT 52L) | Qwen3.5-4B (Local) / Cloud | **Qwen3-TTS CustomVoice (bf16)** | **ASR RTF: 0.205**<br>**TTS RTF: 0.33 · 52-lang streaming** | `jetson-multilang-highperf-nx` | 🟢 Verified |
| **🇻🇳 Vietnamese (vi)** | Overseas AI Agent / Companion | Qwen3-ASR 0.6B (TRT int4) | Qwen3.5-4B (Local) / Cloud | Kokoro-VI (TRT Hybrid) | **ASR: 366.3 ms**<br>**TRT Prefix: 17.0 ms · RTF: 0.703**<br>**100% word retention (0 truncation)** | `jetson-qwen3asr-kokoro-vi` | 🟢 Verified |
| **🎧 Offline Transcription** | Meeting minutes / Audio to text | Whisper-base (TRT bf16) | — (Transcription only) | — | **WER: 7.62%**<br>**TTFT: 18 ms · RTF: 0.008** | `orin-whisper` | 🟢 Verified |

### 🟠 3. reComputer RK3588 — High Concurrency & Efficiency (reComputer R1000 / Industrial Box)

> **Hardware Specs**: 6 TOPS NPU, 8GB/16GB RAM, 8–15W TDP.

| Language | Use Case | ASR Engine | On-Device LM | TTS Engine | Measured Performance (ASR / TTS / V2V) | Recommended Profile | Status |
|---|---|---|---|---|---|---|:---:|
| **🇺🇸 English (en)** | Global IoT hardware deployment | Qwen3-ASR (RKNN W8A8) | Cloud API / Lightweight local | **Kokoro-RKNN (34% NPU offload)** | **TTS RTF: 0.34**<br>**ASR RTF: 0.35** | `rk3588-kokoro-rknn-34pct` | 🟢 Verified |
| **🇨🇳 Chinese (zh)** | Building intercom / Industrial station / Inspection | Qwen3-ASR (RKNN W8A8) | RKLLM Qwen2.5-3B (~15 tok/s) / Cloud | Matcha-TTS (RKNN build) | **TTS RTF: 0.05 (Ultra-fast)**<br>**ASR RTF: 0.28 · V2V p50: 528 ms**<br>Triple-core NPU independent pipeline | `rk3588-default` | 🟢 Verified |
| **🇯🇵 Japanese (ja)** | Japanese market hardware / HMI | Qwen3-ASR (RKNN W8A8) | Cloud API / Lightweight local | **Kokoro-RKNN (34% NPU offload)** | **TTS RTF: 0.34**<br>**ASR RTF: 0.35 · Natural tone** | `rk3588-kokoro-rknn-34pct` | 🟢 Verified |
| **🇪🇺 European Languages** | German / French / Spanish terminals | Qwen3-ASR (52L RKNN) | Cloud Multilingual LM | Piper TTS (CPU) | **ASR RTF: ~0.35**<br>**TTS RTF: ~0.60** | `rk3588-piper` | 🔵 Supported |
| **📊 Concurrent Transcription** | Call center QA / Meeting records | SenseVoice-Small (RKNN) | — (Transcription only) | — | **12 parallel channels 0 error**<br>**CER: 5.13% · Tri-NPU 100% saturated** | `rk3588-sensevoice` | 🟢 Verified |

### 🟡 4. reComputer RK3576 — Cost-Effective IoT Gateway (LubanCat-3 / Compact Gateway)

> **Hardware Specs**: 6 TOPS NPU, 4GB/8GB RAM, 4–8W ultra-low TDP.

| Language | Use Case | ASR Engine | On-Device LM | TTS Engine | Measured Performance (ASR / TTS / V2V) | Recommended Profile | Status |
|---|---|---|---|---|---|---|:---:|
| **🇺🇸 English (en)** | Lightweight English voice dialogue | Qwen3-ASR 0.6B (RKNN W8A8) | Cloud API / 1.5B (Lightweight) | **Kokoro (RKNN/CPU hybrid)** | **ASR RTF: 0.44**<br>**TTS RTF: ~0.65 · Natural tone** | `rk3576-kokoro-long32-canary` | 🟢 Verified |
| **🇨🇳 Chinese (zh)** | Elderly care speaker / Warehouse picking | Qwen3-ASR 0.6B (RKNN W8A8) | Cloud API / 1.5B (Lightweight) | Matcha-TTS (RKNN) | **TTS RTF: 0.12**<br>**ASR RTF: 0.44 · V2V p50: 1020 ms** | `rk3576-default` | 🟢 Verified |
| **🇪🇺 European Languages** | Lightweight multilingual voice prompt | Qwen3-ASR (RKNN) | Cloud API | Piper TTS (CPU) | **ASR RTF: ~0.44**<br>**TTS RTF: ~0.80** | `rk3576-piper` | 🔵 Supported |
| **📊 Multi-channel Transcription** | Industrial audio monitoring | SenseVoice-Small (RKNN) | — | — | **16 parallel channels 0 error**<br>Lowest-cost multichannel pick | `rk3576-sensevoice` | 🟢 Verified |

### 🟣 5. Raspberry Pi 5 — Pure CPU & Lightweight Edge (Raspberry Pi 5 / reComputer R1000-Pi)

> **Hardware Specs**: Quad-core Cortex-A76 CPU, optional Hailo-8 M.2 module (26 TOPS).

| Language | Use Case | ASR Engine | On-Device LM | TTS Engine | Measured Performance (ASR / TTS / V2V) | Recommended Profile | Status |
|---|---|---|---|---|---|---|:---:|
| **🇺🇸 English (+Hailo-8 / CPU)** | Industrial low-power English ASR | Whisper-base (Hailo-8 hardware encoder) / Sherpa | Cloud API | Sherpa / Piper | **WER: 8.39%**<br>Saves >70% CPU usage | `rpi5-hailo-whisper` | 🟢 Verified |
| **🇨🇳 Chinese (zh)** | Maker projects / Basic assistant | Sherpa-ONNX streaming (CPU) | Cloud API (Bailian / OpenAI) | Sherpa-ONNX (CPU) | **ASR RTF: ~0.50**<br>**TTS RTF: ~0.60 · V2V: ~1.5 s** | `rpi4-default` | 🟢 Verified |
| **📊 CPU-only Transcription** | 8-way offline command transcription | SenseVoice (ORT CPU) | — | — | **8 parallel channels 0 error**<br>RTF ~0.5–0.7 | `rpi4-sensevoice` | 🟢 Verified |

Full specification available in [docs/DEVICE-LANGUAGE-MATRIX.md](docs/DEVICE-LANGUAGE-MATRIX.md) and [docs/RECOMMENDED-MODELS.md](docs/RECOMMENDED-MODELS.md).

---

## Quick Start

### 1. One-Line Install & Launch
Clone the repository and run the automated installer. It detects your hardware accelerator (Jetson, Rockchip, or Raspberry Pi), pulls optimized images, and launches the voice service:

```bash
git clone --recurse-submodules https://github.com/Seeed-Solution/openvoicestream.git
cd openvoicestream

deploy/install.sh --pull --verify
```

When auto-detection is not sufficient, target a platform explicitly:

```bash
deploy/install.sh --target orin-nx --pull --verify  # Jetson Orin defaults to v0.11+ high-perf voice stack
deploy/install.sh --target rk3588 --pull --verify   # RK3588 NPU native stack
deploy/install.sh --target rk3576 --pull --verify   # RK3576 NPU native stack
deploy/install.sh --target rpi --pull --verify      # Raspberry Pi CPU/Hailo stack
```

> **Release Notes**:
> - **Jetson**: Defaults to the unified v0.11+ production voice runtime (Qwen3-ASR IFB + Qwen3-TTS CustomVoice / Matcha). Roll back to legacy releases with `--legacy`.
> - To launch the on-device LLM (Qwen3.5-4B) on Orin NX 16GB, see [Orin NX LLM Deployment Guide](docs/deploy/jetson-orin-nx-v091.md).
> - Comprehensive compose profile catalog and environment variables are documented in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### 2. Verify Health & Roundtrip

```bash
# Automated smoke and roundtrip test
deploy/verify.sh --url http://localhost:8621 --tts-smoke --roundtrip

# Service healthcheck
curl http://localhost:8621/health
```

### 3. Send Your First API Request (OpenAI-Compatible)

The service listens on `http://<device-ip>:8621`:

```bash
# 1. Text-to-Speech (TTS) -> speech.wav
curl -X POST http://localhost:8621/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"matcha-zh-en","input":"Hello, welcome to OpenVoiceStream."}' \
  --output speech.wav

# 2. Speech-to-Text (ASR) -> transcribe speech.wav
curl -X POST http://localhost:8621/v1/audio/transcriptions \
  -F "model=qwen3-asr" -F "file=@speech.wav"
```

Python client examples are available in [`examples/`](examples/).

---

## Model Selection Guide

| Language & Scenario | Recommended ASR | Recommended TTS | Supported Hardware | Key Measured Metrics |
|---|---|---|---|---|
| **🇺🇸 English Dialogue** | Paraformer / Qwen3-ASR | **Kokoro** (TRT / RKNN) | All devices | **53 native voices**, Orin RTF 0.18, RK3588 0.34 |
| **🇨🇳 Chinese Dialogue** | Qwen3-ASR (INT4/W8A8) | **Matcha-TTS** (FP16/RKNN) | All devices | **TTS RTF 0.018–0.075**, TTFT <10ms, minimal VRAM |
| **🇯🇵 Japanese Dialogue** | Qwen3-ASR (52L) | **Kokoro** (Misaki Japanese voices) | Jetson / RK3588 | Native Japanese G2P, natural prosody, Orin RTF 0.20, RK3588 0.34 |
| **🇪🇺 European Languages** | Qwen3-ASR (52L) | **Piper** (CPU) | All devices | Lightweight offline speech for German/French/Spanish |
| **🌐 Multilingual / Clone** | Qwen3-ASR (52L) | **Qwen3-TTS** (INT4/CustomVoice) | Jetson Orin | 52-language streaming, 3–10s zero-shot cloning |
| **🇻🇳 Vietnamese Dialogue** | Qwen3-ASR (TRT INT4) | **Kokoro-VI** (TRT Hybrid) | Jetson Orin | Prefix 17ms, end-to-end RTF 0.703, 100% word retention |
| **📊 Concurrent Transcription** | SenseVoice-Small / Whisper | — | RK3588 / RPi / Jetson | RK3588 12 channels 0 error; Whisper WER 7.62% |

Full model comparisons and configs are detailed in [docs/RECOMMENDED-MODELS.md](docs/RECOMMENDED-MODELS.md).

---

## Ready-Made Applications & Web Demo

### Web Demo Gallery
```bash
docker compose -f demos/docker-compose.demos.yml --profile all up -d
# Browser access: http://<device-ip>:8700
```
Interactive portal featuring real-time device telemetry, live captioning, voice synthesis playground, voice cloning, and kiosk mode. See [`demos/README.md`](demos/README.md).

### Turnkey Applications
The repository includes nine production applications in [`agent/ovs_agent/apps/`](agent/ovs_agent/apps/):

| Application | Scenario | Pipeline | Documentation |
|---|---|---|---|
| [`conversation`](agent/ovs_agent/apps/conversation/README.md) | Full-duplex conversational voice assistant with barge-in | ASR → LLM → TTS | [README](agent/ovs_agent/apps/conversation/README.md) |
| [`home_assistant`](agent/ovs_agent/apps/home_assistant/README.md) | Smart home control ("Dim living room lights") | ASR → HA Intent | [README](agent/ovs_agent/apps/home_assistant/README.md) |
| [`companion_robot`](agent/ovs_agent/apps/companion_robot/README.md) | Embodied robot interaction brain | ASR → LLM + Tools → TTS | [README](agent/ovs_agent/apps/companion_robot/README.md) |
| [`voice_rebot_arm`](agent/ovs_agent/apps/voice_rebot_arm/README.md) | Voice-guided robotic arm with vision grasping | Wake word → ASR → LLM → Arm Control | [README](agent/ovs_agent/apps/voice_rebot_arm/README.md) |
| [`translator`](agent/ovs_agent/apps/translator/README.md) | Sentence-level voice intercom translation | ASR → MT → TTS | [README](agent/ovs_agent/apps/translator/README.md) |
| [`live_caption`](agent/ovs_agent/apps/live_caption/README.md) | Real-time meeting subtitle broadcast | ASR → Text Broadcast | [README](agent/ovs_agent/apps/live_caption/README.md) |

---

## API Overview

The gateway exposes standard REST and WebSocket endpoints:

| Endpoint | Protocol | Description |
|---|---|---|
| `POST /v1/audio/speech` | HTTP (Chunked) | OpenAI-compatible speech synthesis; returns streaming WAV/PCM |
| `POST /v1/audio/transcriptions` | HTTP Multipart | OpenAI-compatible file-based speech transcription |
| `GET /v1/models` | HTTP JSON | List active ASR/TTS models and readiness metadata |
| `GET /v1/capabilities` | HTTP JSON | Runtime discovery of voices, speed control, streaming, and voice clone flags |
| `WS /asr/stream` | WebSocket | Low-latency streaming ASR (16kHz PCM in, partial JSON out) |
| `POST /tts/stream` | HTTP Stream | Native streaming TTS with sub-10ms first audio packet |
| `GET /health` | HTTP JSON | Health status of service and loaded acceleration engines |

### Streaming ASR Python Client

```python
import asyncio, websockets, json

async def transcribe(audio_chunks):
    uri = "ws://localhost:8621/asr/stream?sample_rate=16000&language=auto"
    async with websockets.connect(uri) as ws:
        async def send_audio():
            for chunk in audio_chunks:  # int16 16kHz PCM audio frames
                await ws.send(chunk.tobytes())
                await asyncio.sleep(0.02)
            await ws.send(b"")  # send empty frame to signal EOS

        async def receive_results():
            async for message in ws:
                data = json.loads(message)
                if data.get("is_final"):
                    print("[Final]", data.get("text"))
                    break
                print("[Partial]", data.get("text"))

        await asyncio.gather(send_audio(), receive_results())
```

Detailed parameter specifications and voice management endpoints are in the [Full API Reference (docs/API.md)](docs/API.md).

---

## Architecture

```text
┌───────────────────────────────────────────────────────────┐
│  Edge Device (Jetson Orin / RK3588 / RK3576 / RPi 4–5)    │
│                                                           │
│  FastAPI Unified Voice Gateway (:8621)                    │
│  ├── WS /asr/stream    Streaming ASR (Qwen3 / Paraformer) │
│  ├── POST /asr          Batch ASR (SenseVoice / Whisper)  │
│  ├── POST /tts/stream   Streaming TTS (Matcha / Kokoro)   │
│  └── POST /v1/*         OpenAI-Compatible Audio Endpoints │
│                                                           │
│  Hardware Inference Acceleration:                         │
│  ├── Jetson: TensorRT-EdgeLLM / TensorRT C++              │
│  ├── Rockchip: RKNN NPU Runtime                           │
│  └── Raspberry Pi: HailoRT NPU / sherpa-onnx (CPU)        │
└───────────────────────────────────────────────────────────┘
         ▲ HTTP / WebSocket
         │
   Client Applications (Robots / Gateways / Industrial HMI / Web)
```

For multi-process topology and lifecycle architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Performance & Benchmarks

### 1. Unified Whisper WER across Five Hardware Accelerators
Benchmarked on a unified 100-segment test corpus with identical evaluation pipeline:

| Hardware Device | Accelerator Backend | Overall WER |
|---|---|---:|
| **Jetson Orin NX 16GB** | TensorRT bf16 encoder + CPU ONNX decoder | **7.62%** |
| **Jetson Orin Nano 8GB** | TensorRT bf16 encoder + CPU ONNX decoder | **7.62%** |
| **reComputer RK3588** | RKNN base10 encoder + CPU ONNX decoder | **7.50%** |
| **reComputer RK3576** | RKNN base10 encoder + CPU ONNX decoder | **8.51%** |
| **Raspberry Pi 5 + Hailo-8** | Hailo base encoder + CPU ONNX decoder | **8.39%** |

### 2. Cross-Device Latency & Throughput Summary

| Device Model | Recommended Pipeline | TTS RTF p50 | ASR fRTF p50 | End-to-End V2V p50 | Status |
|---|---|---:|---:|---:|:---:|
| **Orin NX (16GB)** | Qwen3-ASR + Matcha + Qwen3.5-4B (On-device) | 0.018 | 0.042 | **162 ms** | 🟢 Verified |
| **Orin NX (16GB)** | Qwen3-ASR + Kokoro (53 voices) | 0.180 | 0.042 | **~300 ms** | 🟢 Verified |
| **Orin Nano (8GB)** | Qwen3-ASR + Kokoro-VI (Vietnamese) | 0.703 | 0.076 | **~500 ms** | 🟢 Verified |
| **reComputer RK3588** | Qwen3-ASR + Matcha (RKNN) | 0.050 | 0.280 | **528 ms** | 🟢 Verified |
| **reComputer RK3588** | Qwen3-ASR + Kokoro (34% NPU) | 0.340 | 0.350 | **~650 ms** | 🟢 Verified |
| **reComputer RK3576** | Qwen3-ASR + Matcha (RKNN) | 0.120 | 0.440 | **1020 ms** | 🟢 Verified |
| **Raspberry Pi 5** | Sherpa-ONNX bilingual streaming (CPU) | 0.078 | 0.500 | **~1.5 s** | 🟢 Verified |

For complete benchmark logs, concurrency stress tests, and methodology, see [BENCHMARKS.md](BENCHMARKS.md).

---

## Supported Devices & Requirements

The stack is open-source and optimized for major edge silicon families:

| Ecosystem | Recommended Hardware | OS & Requirements | Supported Features |
|---|---|---|---|
| **NVIDIA Jetson** | reComputer J4012 / J4011 / Super | JetPack 6.x (CUDA 12.6, TensorRT 10.3) | Full stack, Qwen3 52-language streaming, voice clone, on-device LLM |
| **Rockchip RK3588** | reComputer R1000 / Industrial Host | Linux Kernel 5.10 / 6.1, `rknpu` driver | Triple-core NPU, 12-channel concurrent ASR, Matcha & Kokoro NPU offload |
| **Rockchip RK3576** | LubanCat-3 / Compact Gateway | Linux Kernel 6.1, `rknpu` driver | Ultra-low power edge gateway, 16-channel audio transcription |
| **Raspberry Pi** | Raspberry Pi 5 (8GB) / Pi 4 | 64-bit Raspberry Pi OS, Docker | Pure CPU streaming; optional Hailo-8 M.2 acceleration module |

---

## Documentation Index

- **Selection & Solutions**:
  - [Multilingual Device Matrix (Markdown Spec)](docs/DEVICE-LANGUAGE-MATRIX.md)
  - [Interactive Tabbed Matrix (Browser UI)](docs/device-language-matrix.html)
  - [Recommended Models & Scenarios](docs/RECOMMENDED-MODELS.md)
- **Deployment & Operations**:
  - [Environment Variables & Compose Profiles](docs/CONFIGURATION.md)
  - [Orin NX On-Device LLM Deployment Guide](docs/deploy/jetson-orin-nx-v091.md)
  - [End-to-End Reproduction Guide](docs/REPRODUCE.md)
- **Architecture & Development**:
  - [Full API Specification](docs/API.md)
  - [Architecture & Multi-Process Design](ARCHITECTURE.md)
  - [Local Developer Guide](DEVELOP.md)
  - [Performance Benchmarks & Concurrency](BENCHMARKS.md)

---

## Contributing & License

Contributions, issues, and pull requests are welcome. This repository is licensed under the [MIT License](LICENSE).

### Acknowledgements
- [voxedge](https://github.com/suharvest/voxedge) — Core Python/numpy real-time streaming speech runtime
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — CPU speech inference engine
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) & [Misaki](https://github.com/hexgrad/misaki) — Lightweight high-quality TTS model and G2P engine
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — Ultra-fast flow-matching speech synthesis
- [Qwen3](https://huggingface.co/Qwen) — Multilingual ASR & TTS foundation model
- [FunASR](https://github.com/modelscope/FunASR) & [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — Paraformer and SenseVoice ASR models
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) & [RKNN-Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) — Native hardware acceleration runtimes
