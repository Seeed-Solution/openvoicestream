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

| Hardware & Specs | 🇺🇸 English (en) Primary | 🇨🇳 Chinese (zh) Secondary | 🇯🇵 Japanese (ja) APAC | 🇪🇺 European / 🌐 52L | 🇻🇳 Vietnamese (vi) Expansion | 🎧 Offline Transcription |
|---|---|---|---|---|---|---|
| **Jetson Orin Nano (8GB)**<br>`40 TOPS · 7–15W`<br><sub>reComputer J4012 / J401</sub> | **ASR**: Paraformer / Whisper<br>**TTS**: Kokoro-TRT (53 voices)<br><sub>ASR 50ms · TTS RTF 0.25</sub> | **ASR**: Qwen3-ASR (INT4)<br>**TTS**: Matcha-TTS (FP16)<br><sub>CER 0 · RTF 0.15 · VRAM 1.8G</sub> | **ASR**: Qwen3-ASR (52L)<br>**TTS**: Kokoro-TRT (Misaki)<br><sub>ASR 0.25 · TTS RTF 0.28</sub> | **ASR**: Qwen3-ASR (52L)<br>**TTS**: Qwen3-TTS (INT4)<br><sub>TTS RTF 0.50 · 3–10s Clone</sub> | **ASR**: Qwen3-ASR (INT4)<br>**TTS**: Kokoro-VI (Hybrid)<br><sub>TRT 17ms · RTF 0.70 (0 drop)</sub> | **ASR**: Whisper-base (TRT)<br><sub>WER 7.62% · RTF 0.008</sub> |
| **Jetson Orin NX (16GB)**<br>`100 TOPS · 10–25W`<br><sub>reComputer J4012 / Super</sub> | **ASR**: Qwen3-ASR / Paraformer<br>**TTS**: Kokoro-TRT (53 voices)<br><sub>ASR 18ms · TTS RTF 0.18</sub> | **ASR**: Qwen3-ASR (INT4)<br>**LM**: Qwen3.5-4B (Local 35t/s)<br>**TTS**: Matcha-TTS (BF16)<br><sub>V2V 162ms · Air-gapped · 10GB</sub> | **ASR**: Qwen3-ASR (52L)<br>**TTS**: Kokoro-TRT (Misaki)<br><sub>ASR 0.15 · TTS RTF 0.20</sub> | **ASR**: Qwen3-ASR (52L)<br>**TTS**: Qwen3-TTS (BF16)<br><sub>10-stream · TTS RTF 0.33</sub> | **ASR**: Qwen3-ASR (INT4)<br>**TTS**: Kokoro-VI (Hybrid)<br><sub>TRT 17ms · RTF 0.70 (0 drop)</sub> | **ASR**: Whisper-base (TRT)<br><sub>WER 7.62% · RTF 0.008</sub> |
| **reComputer RK3588**<br>`6 TOPS NPU · 8–15W`<br><sub>reComputer R1000 / Box</sub> | **ASR**: Qwen3-ASR (W8A8)<br>**TTS**: Kokoro-RKNN (34% offload)<br><sub>ASR 0.35 · TTS RTF 0.34</sub> | **ASR**: Qwen3-ASR (W8A8)<br>**TTS**: Matcha-TTS (RKNN)<br><sub>ASR 0.28 · TTS RTF 0.05 (fast)</sub> | **ASR**: Qwen3-ASR (W8A8)<br>**TTS**: Kokoro-RKNN (Misaki)<br><sub>ASR 0.35 · TTS RTF 0.34</sub> | **ASR**: Qwen3-ASR (52L)<br>**TTS**: Piper TTS (CPU)<br><sub>ASR ~0.35 · TTS ~0.60</sub> | — | **ASR**: SenseVoice (RKNN)<br><sub>12 channels 0 error · Tri-NPU</sub> |
| **reComputer RK3576**<br>`6 TOPS NPU · 4–8W`<br><sub>LubanCat-3 / Gateway</sub> | **ASR**: Qwen3-ASR (W8A8)<br>**TTS**: Kokoro (RKNN/CPU)<br><sub>ASR 0.44 · TTS RTF ~0.65</sub> | **ASR**: Qwen3-ASR (W8A8)<br>**TTS**: Matcha-TTS (RKNN)<br><sub>ASR 0.44 · TTS RTF 0.12</sub> | — | **ASR**: Qwen3-ASR (RKNN)<br>**TTS**: Piper TTS (CPU)<br><sub>ASR ~0.44 · TTS ~0.80</sub> | — | **ASR**: SenseVoice (RKNN)<br><sub>16 channels 0 error · Low cost</sub> |
| **Raspberry Pi 5**<br>`Quad CPU · Optional Hailo-8`<br><sub>RPi 5 / R1000-Pi</sub> | **ASR**: Whisper (+Hailo-8)<br>**TTS**: Sherpa / Piper (CPU)<br><sub>WER 8.39% · Saves >70% CPU</sub> | **ASR**: Sherpa-ONNX (CPU)<br>**TTS**: Sherpa-ONNX (CPU)<br><sub>Pure CPU loop · V2V ~1.5s</sub> | — | **ASR**: Sherpa-ONNX (CPU)<br>**TTS**: Piper TTS (CPU)<br><sub>Pure CPU multilingual</sub> | — | **ASR**: SenseVoice (ORT)<br><sub>8 channels 0 error · RTF 0.5-0.7</sub> |

> 💡 **Full Per-Device Specifications & Profiles**:
> - 📄 Detailed Docker profiles, VRAM footprints, and deployment commands: **[Full Benchmark Matrix (docs/DEVICE-LANGUAGE-MATRIX.md)](docs/DEVICE-LANGUAGE-MATRIX.md)**
> - 🖥️ For instant tab switching in browser: **[Interactive Tabbed Matrix (docs/device-language-matrix.html)](docs/device-language-matrix.html)**

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
