# OpenVoiceStream

> [English](README.md) | **中文**

**面向边缘硬件的原生流式语音引擎** —— 开箱即用、超低延迟、完全在设备端运行。

<p align="center">
  <a href="https://github.com/Seeed-Solution/openvoicestream"><img src="https://img.shields.io/github/stars/Seeed-Solution/openvoicestream?style=social" alt="GitHub stars" /></a>
  <a href="#系统架构"><img src="https://img.shields.io/badge/ASR-Qwen3--ASR%20%7C%20Paraformer%20%7C%20SenseVoice%20%7C%20Whisper-2f80ed.svg" alt="ASR: Qwen3-ASR, Paraformer, SenseVoice, Whisper" /></a>
  <a href="#模型选型指南"><img src="https://img.shields.io/badge/TTS-Matcha--TTS%20%7C%20Kokoro%20%7C%20Qwen3--TTS%20%7C%20Piper-f97316.svg" alt="TTS: Matcha-TTS, Kokoro, Qwen3-TTS, Piper" /></a>
  <a href="#系统架构"><img src="https://img.shields.io/badge/engines-TensorRT--EdgeLLM%20%7C%20RKNN%20%7C%20HailoRT%20%7C%20sherpa--onnx-16a34a.svg" alt="Engines: TensorRT-EdgeLLM, RKNN, HailoRT, sherpa-onnx" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/deploy-Docker-2563eb.svg" alt="Deploy with Docker" /></a>
  <a href="#支持设备与要求"><img src="https://img.shields.io/badge/ecosystems-Jetson%20%7C%20Rockchip%20%7C%20Raspberry%20Pi-65a30d.svg" alt="Supported ecosystems: Jetson, Rockchip, Raspberry Pi" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-facc15.svg" alt="MIT license" /></a>
</p>

<p align="center">
  <a href="#边缘设备推荐搭配与多语言性能基准矩阵">⚡ <b>设备选型矩阵</b></a> &nbsp;|&nbsp;
  <a href="#quick-start">🚀 <b>快速开始</b></a> &nbsp;|&nbsp;
  <a href="#模型选型指南">💡 <b>模型推荐</b></a> &nbsp;|&nbsp;
  <a href="#现成应用-applications--web-演示">📱 <b>现成应用</b></a> &nbsp;|&nbsp;
  <a href="#api-概览">🔌 <b>API 概览</b></a> &nbsp;|&nbsp;
  <a href="#性能基准">📊 <b>性能基准</b></a> &nbsp;|&nbsp;
  <a href="#系统架构">🏗️ <b>系统架构</b></a>
</p>

<p align="center">
  <img src="docs/media/hero.png" alt="OpenVoiceStream - streaming ASR and TTS for edge dialogue" width="760" />
</p>

**OpenVoiceStream 提供一套开箱即用、经过硬件实测验证的边缘语音解决方案。** 
实时语音识别（ASR）、语音合成（TTS），以及上层开箱即用的完整交互应用 —— 全双工对话、智能家居控制、语音机械臂、跨语种翻译与实时字幕 —— 全部 100% 运行在本地设备上。

- **零云端依赖**：无需外部 API Key、无按次调用成本、断网离线稳定运行。
- **硬件原生释放**：针对每款硬件量化（W8A8 / W4A16 / INT4 / FP16），原生运行于 TensorRT（Jetson）、RKNN（Rockchip）与 HailoRT/CPU（Raspberry Pi）。
- **统一客户端契约**：所有板卡暴露一致的 OpenAI 兼容接口与 WebSocket 流式接口，切换硬件无需改动客户端代码。

底层的实时语音推理核心是独立的 Python 库 [`voxedge`](https://github.com/suharvest/voxedge)（`pip install voxedge`）。本仓库提供面向生产落地的整套交付方案：包含预构建镜像、多板卡量化模型、运维脚本、Web 控制台与现成应用。

---

## 解决方案全景 (Solution Lineup)

<p align="center">
  <img src="docs/media/solution-lineup.png" alt="OpenVoiceStream solution lineup" width="900" />
</p>

---

## 边缘设备推荐搭配与多语言性能基准矩阵

> 💡 **模型选型原则**：
> - **英文对话（首要）**：所有设备统一主推 **Kokoro**（原生 53 款音色，音质纯正自然；RK3588/RK3576 全面支持）
> - **中文对话（其次）**：Qwen3-ASR + Matcha-TTS（极快、超低显存、自然度高）
> - **日语对话（亚太重点）**：**Qwen3-ASR + Kokoro（Misaki G2P 日语音色）**，发音纯正地道
> - **欧系语言 / 52 语多语言**：统一使用 **Qwen3-ASR + Qwen3-TTS**（多语种流式与声音克隆）或 **Piper TTS**（离线轻量）
> - **越南语对话（出海拓展）**：Qwen3-ASR + **Kokoro-VI**（前缀 TRT 17ms，端到端 0.703 RTF，首词 100% 完整留存）
> - **端侧大模型思考**：Orin NX 16GB 支持端侧 **Qwen3.5-4B** 离线闭环（~35 tok/s，V2V p50 162 ms）

| 硬件平台 / 算力规格 | 🇺🇸 英语 (en) 首要 | 🇨🇳 中文 (zh) 其次 | 🇯🇵 日语 (ja) 亚太 | 🇪🇺 欧系 / 🌐 52语 | 🇻🇳 越南语 (vi) 出海 | 🎧 离线高并发转写 |
|---|---|---|---|---|---|---|
| **Jetson Orin Nano (8GB)**<br>`40 TOPS · 7–15W`<br><sub>reComputer J4012 / J401</sub> | **ASR**: Paraformer / Whisper<br>**TTS**: Kokoro-TRT (53音色)<br><sub>ASR 50ms · TTS RTF 0.25</sub> | **ASR**: Qwen3-ASR (INT4)<br>**TTS**: Matcha-TTS (FP16)<br><sub>CER 0 · RTF 0.15 · 显存 1.8G</sub> | **ASR**: Qwen3-ASR (52语)<br>**TTS**: Kokoro-TRT (Misaki)<br><sub>ASR 0.25 · TTS RTF 0.28</sub> | **ASR**: Qwen3-ASR (52语)<br>**TTS**: Qwen3-TTS (INT4)<br><sub>TTS RTF 0.50 · 支持克隆</sub> | **ASR**: Qwen3-ASR (INT4)<br>**TTS**: Kokoro-VI (Hybrid)<br><sub>TRT 17ms · RTF 0.70 (0丢词)</sub> | **ASR**: Whisper-base (TRT)<br><sub>WER 7.62% · RTF 0.008</sub> |
| **Jetson Orin NX (16GB)**<br>`100 TOPS · 10–25W`<br><sub>reComputer J4012 / Super</sub> | **ASR**: Qwen3-ASR / Paraformer<br>**TTS**: Kokoro-TRT (53音色)<br><sub>ASR 18ms · TTS RTF 0.18</sub> | **ASR**: Qwen3-ASR (INT4)<br>**LM**: Qwen3.5-4B (端侧 35t/s)<br>**TTS**: Matcha-TTS (BF16)<br><sub>V2V 162ms · 端侧闭环 · 10GB</sub> | **ASR**: Qwen3-ASR (52语)<br>**TTS**: Kokoro-TRT (Misaki)<br><sub>ASR 0.15 · TTS RTF 0.20</sub> | **ASR**: Qwen3-ASR (52语)<br>**TTS**: Qwen3-TTS (BF16)<br><sub>10路并发 · TTS RTF 0.33</sub> | **ASR**: Qwen3-ASR (INT4)<br>**TTS**: Kokoro-VI (Hybrid)<br><sub>TRT 17ms · RTF 0.70 (0丢词)</sub> | **ASR**: Whisper-base (TRT)<br><sub>WER 7.62% · RTF 0.008</sub> |
| **reComputer RK3588**<br>`6 TOPS NPU · 8–15W`<br><sub>reComputer R1000 / 工业盒</sub> | **ASR**: Qwen3-ASR (W8A8)<br>**TTS**: Kokoro-RKNN (34%卸载)<br><sub>ASR 0.35 · TTS RTF 0.34</sub> | **ASR**: Qwen3-ASR (W8A8)<br>**TTS**: Matcha-TTS (RKNN)<br><sub>ASR 0.28 · TTS RTF 0.05 (极快)</sub> | **ASR**: Qwen3-ASR (W8A8)<br>**TTS**: Kokoro-RKNN (Misaki)<br><sub>ASR 0.35 · TTS RTF 0.34</sub> | **ASR**: Qwen3-ASR (52语)<br>**TTS**: Piper TTS (CPU)<br><sub>ASR ~0.35 · TTS ~0.60</sub> | — | **ASR**: SenseVoice (RKNN)<br><sub>12路并发 0错误 · 三核满载</sub> |
| **reComputer RK3576**<br>`6 TOPS NPU · 4–8W`<br><sub>LubanCat-3 / 紧凑网关</sub> | **ASR**: Qwen3-ASR (W8A8)<br>**TTS**: Kokoro (RKNN/CPU)<br><sub>ASR 0.44 · TTS RTF ~0.65</sub> | **ASR**: Qwen3-ASR (W8A8)<br>**TTS**: Matcha-TTS (RKNN)<br><sub>ASR 0.44 · TTS RTF 0.12</sub> | — | **ASR**: Qwen3-ASR (RKNN)<br>**TTS**: Piper TTS (CPU)<br><sub>ASR ~0.44 · TTS ~0.80</sub> | — | **ASR**: SenseVoice (RKNN)<br><sub>16路并发 0错误 · 低成本</sub> |
| **Raspberry Pi 5**<br>`4核 CPU · 可选 Hailo-8`<br><sub>树莓派 5 / R1000-Pi</sub> | **ASR**: Whisper (+Hailo-8)<br>**TTS**: Sherpa / Piper (CPU)<br><sub>WER 8.39% · 节省 >70% CPU</sub> | **ASR**: Sherpa-ONNX (CPU)<br>**TTS**: Sherpa-ONNX (CPU)<br><sub>纯 CPU 全双工 · V2V ~1.5s</sub> | — | **ASR**: Sherpa-ONNX (CPU)<br>**TTS**: Piper TTS (CPU)<br><sub>纯 CPU 轻量多语种</sub> | — | **ASR**: SenseVoice (ORT)<br><sub>8路并发 0错误 · RTF 0.5-0.7</sub> |

> 📖 **完整分设备规格与 Profile 预设**：
> - 📄 查看各设备底层 Docker profile、实测分位延迟与部署命令，请移步 **[设备选型完整技术规范 (docs/DEVICE-LANGUAGE-MATRIX.md)](docs/DEVICE-LANGUAGE-MATRIX.md)**
> - 🖥️ 在浏览器中体验动态点击切换设备 Tab，请打开 **[交互式选型网页 (docs/device-language-matrix.html)](docs/device-language-matrix.html)**

---

## Quick Start

### 1. 一键安装与服务拉起
在目标设备上克隆仓库并执行安装器。安装脚本会自动探测硬件加速器（Jetson / Rockchip / Raspberry Pi）、拉取对应镜像并启动语音服务：

```bash
git clone --recurse-submodules https://github.com/Seeed-Solution/openvoicestream.git
cd openvoicestream

deploy/install.sh --pull --verify
```

当自动检测不适用时，可显式指定目标架构：

```bash
deploy/install.sh --target orin-nx --pull --verify  # Jetson Orin 默认启用 v0.11+ 高性能语音引擎
deploy/install.sh --target rk3588 --pull --verify   # RK3588 NPU 原生方案
deploy/install.sh --target rk3576 --pull --verify   # RK3576 NPU 原生方案
deploy/install.sh --target rpi --pull --verify      # Raspberry Pi CPU/Hailo 方案
```

> **版本说明**：
> - **Jetson** 默认启用统一的 v0.11+ 生产语音引擎（Qwen3-ASR IFB + Qwen3-TTS CustomVoice / Matcha）。回滚旧版请使用 `--legacy`。
> - 如需在 Orin NX 16GB 上同卡启动本地离线大模型（Qwen3.5-4B），请参考 [Orin NX 本地大模型部署指南](docs/deploy/jetson-orin-nx-v091.md)。
> - 完整 Compose Profile 清单与环境变量配置详见 [docs/CONFIGURATION.md](docs/CONFIGURATION.md)。

### 2. 状态验证与健康检查

```bash
# 运行自动化往返冒烟测试
deploy/verify.sh --url http://localhost:8621 --tts-smoke --roundtrip

# 检查服务健康状态
curl http://localhost:8621/health
```

### 3. 发送第一次语音调用 (OpenAI 兼容)

服务启动后监听在 `http://<device-ip>:8621`：

```bash
# 1. 语音合成 (TTS) -> 生成 speech.wav
curl -X POST http://localhost:8621/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"matcha-zh-en","input":"你好，欢迎使用 OpenVoiceStream。"}' \
  --output speech.wav

# 2. 语音识别 (ASR) -> 转写 speech.wav
curl -X POST http://localhost:8621/v1/audio/transcriptions \
  -F "model=qwen3-asr" -F "file=@speech.wav"
```

Python 客户端与进阶调用代码见 [`examples/`](examples/)。

---

## 模型选型指南

| 场景与语言 | 推荐 ASR 引擎 | 推荐 TTS 引擎 | 适用硬件 | 特性与实测指标 |
|---|---|---|---|---|
| **🇺🇸 英文纯正发音** | Paraformer / Qwen3-ASR | **Kokoro** (TRT / RKNN) | 全部设备 | **53 款原生纯正发音**，Orin RTF 0.18，RK3588 0.34 |
| **🇨🇳 中文极速对话** | Qwen3-ASR (INT4/W8A8) | **Matcha-TTS** (FP16/RKNN) | 全部设备 | **TTS RTF 0.018–0.075**，首包 <10ms，超低显存 |
| **🇯🇵 日语地道交互** | Qwen3-ASR (52 语) | **Kokoro** (Misaki 日语音色) | Jetson / RK3588 | 原生日文 G2P，发音地道自然，Orin RTF 0.20，RK3588 0.34 |
| **🇪🇺 欧系小语种** | Qwen3-ASR (52 语) | **Piper** (CPU) | 全部设备 | 德/法/西/俄等离线轻量级发音 |
| **🌐 多语种 / 声音克隆** | Qwen3-ASR (52 语) | **Qwen3-TTS** (INT4/CustomVoice) | Jetson Orin | 52 语流式合成，支持 3~10 秒极速克隆 |
| **🇻🇳 越南语出海对话** | Qwen3-ASR (TRT INT4) | **Kokoro-VI** (TRT Hybrid) | Jetson Orin | 前缀 17ms，端到端 RTF 0.703，首词 100% 留存 |
| **📊 高并发离线转写** | SenseVoice-Small / Whisper | — | RK3588 / RPi / Jetson | RK3588 12 路 100% 满载 0 错误；Whisper WER 7.62% |

完整对比与参数配置详见 [docs/RECOMMENDED-MODELS.md](docs/RECOMMENDED-MODELS.md)。

---

## 现成应用 (Applications) & Web 演示

### Web 演示控制台 (Demo Gallery)
```bash
docker compose -f demos/docker-compose.demos.yml --profile all up -d
# 浏览器访问：http://<device-ip>:8700
```
提供可视化设备运行状态监控、实时字幕、语音合成交互、声音克隆与展会 Kiosk 模式。详见 [`demos/README.md`](demos/README.md)。

### 内置现成产品应用
项目在 [`agent/ovs_agent/apps/`](agent/ovs_agent/apps/) 提供了九套现成的生产级应用，启动即可使用：

| 应用名称 | 业务场景 | 核心链路 | 文档链接 |
|---|---|---|---|
| [`conversation`](agent/ovs_agent/apps/conversation/README.md) | 全双工语音对话助手（支持打断） | ASR → LLM → TTS | [应用说明](agent/ovs_agent/apps/conversation/README.md) |
| [`home_assistant`](agent/ovs_agent/apps/home_assistant/README.md) | 智能家居离线控制（「把客厅灯调暗」） | ASR → HA 意图解析 | [应用说明](agent/ovs_agent/apps/home_assistant/README.md) |
| [`companion_robot`](agent/ovs_agent/apps/companion_robot/README.md) | 陪伴机器人交互大脑 | ASR → LLM + 工具调用 → TTS | [应用说明](agent/ovs_agent/apps/companion_robot/README.md) |
| [`voice_rebot_arm`](agent/ovs_agent/apps/voice_rebot_arm/README.md) | 语音控制机械臂（力控抓取 + 视觉） | 唤醒词 → ASR → LLM → 机械臂控制 | [应用说明](agent/ovs_agent/apps/voice_rebot_arm/README.md) |
| [`translator`](agent/ovs_agent/apps/translator/README.md) | 离线跨语种对讲翻译 | ASR → 机器翻译 → TTS | [应用说明](agent/ovs_agent/apps/translator/README.md) |
| [`live_caption`](agent/ovs_agent/apps/live_caption/README.md) | 实时会议与广播字幕上屏 | ASR → 文本广播 | [应用说明](agent/ovs_agent/apps/live_caption/README.md) |

---

## API 概览

统一服务提供标准 RESTful 与 WebSocket 接口：

| 接口 | 协议 | 功能描述 |
|---|---|---|
| `POST /v1/audio/speech` | HTTP (Chunked) | OpenAI 兼容语音合成，流式分块返回 WAV/PCM |
| `POST /v1/audio/transcriptions` | HTTP Multipart | OpenAI 兼容音频文件转写 |
| `GET /v1/models` | HTTP JSON | 获取当前已装载的 ASR/TTS 模型 ID 与就绪状态 |
| `GET /v1/capabilities` | HTTP JSON | 动态探测音色清单、语速控制、克隆支持与流式能力 |
| `WS /asr/stream` | WebSocket | 极低延迟实时流式 ASR（输入 16kHz PCM，输出增量 JSON） |
| `POST /tts/stream` | HTTP Stream | 原生低延迟流式 TTS（首包毫秒级直出） |
| `GET /health` | HTTP JSON | 服务及各加速引擎健康检查 |

### Python 实时流式 ASR 示例

```python
import asyncio, websockets, json

async def transcribe(audio_chunks):
    uri = "ws://localhost:8621/asr/stream?sample_rate=16000&language=auto"
    async with websockets.connect(uri) as ws:
        async def send_audio():
            for chunk in audio_chunks:  # 传入 int16 格式 16kHz 音频分块
                await ws.send(chunk.tobytes())
                await asyncio.sleep(0.02)
            await ws.send(b"")  # 发送空字节标记音频结束

        async def receive_results():
            async for message in ws:
                data = json.loads(message)
                if data.get("is_final"):
                    print("[Final]", data.get("text"))
                    break
                print("[Partial]", data.get("text"))

        await asyncio.gather(send_audio(), receive_results())
```

完整接口参数、返回格式与说话人注册请参阅 [完整 API 手册 (docs/API.md)](docs/API.md)。

---

## 系统架构

```text
┌───────────────────────────────────────────────────────────┐
│  边缘硬件设备 (Jetson Orin / RK3588 / RK3576 / RPi 4–5)   │
│                                                           │
│  FastAPI 统一语音网关 (:8621)                              │
│  ├── WS /asr/stream    流式 ASR (Qwen3 / Paraformer)      │
│  ├── POST /asr          离线 ASR (SenseVoice / Whisper)    │
│  ├── POST /tts/stream   流式 TTS (Matcha / Kokoro / Qwen3) │
│  └── POST /v1/*         OpenAI 兼容音频与模型发现接口      │
│                                                           │
│  硬件加速推理引擎：                                        │
│  ├── Jetson: TensorRT-EdgeLLM / TensorRT C++              │
│  ├── Rockchip: RKNN NPU 运行时                            │
│  └── Raspberry Pi: HailoRT NPU / sherpa-onnx (CPU)        │
└───────────────────────────────────────────────────────────┘
         ▲ HTTP / WebSocket
         │
   应用客户端 (机器人 / 智能网关 / 工业 HMI / Web 页面)
```

详细多进程拓扑与通信架构请参阅 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 性能基准

### 1. 同一标准语料下五款加速器 Whisper WER（实测）
所有设备在固定测试语料与同一打分准则下独立运行：

| 设备硬件 | 加速引擎后端 | 整体词错率 (WER) |
|---|---|---:|
| **Jetson Orin NX 16GB** | TensorRT bf16 编码器 + CPU ONNX 解码器 | **7.62%** |
| **Jetson Orin Nano 8GB** | TensorRT bf16 编码器 + CPU ONNX 解码器 | **7.62%** |
| **reComputer RK3588** | RKNN base10 编码器 + CPU ONNX 解码器 | **7.50%** |
| **reComputer RK3576** | RKNN base10 编码器 + CPU ONNX 解码器 | **8.51%** |
| **Raspberry Pi 5 + Hailo-8** | Hailo base 编码器 + CPU ONNX 解码器 | **8.39%** |

### 2. 跨设备端到端延迟与吞吐汇总

| 设备型号 | 推荐配置组合 | TTS RTF p50 | ASR fRTF p50 | V2V 端到端延迟 p50 | 状态 |
|---|---|---:|---:|---:|:---:|
| **Orin NX (16GB)** | Qwen3-ASR + Matcha + Qwen3.5-4B (端侧) | 0.018 | 0.042 | **162 ms** | 🟢 实测 |
| **Orin NX (16GB)** | Qwen3-ASR + Kokoro (53音色) | 0.180 | 0.042 | **~300 ms** | 🟢 实测 |
| **Orin Nano (8GB)** | Qwen3-ASR + Kokoro-VI (越南语) | 0.703 | 0.076 | **~500 ms** | 🟢 实测 |
| **reComputer RK3588** | Qwen3-ASR + Matcha (RKNN) | 0.050 | 0.280 | **528 ms** | 🟢 实测 |
| **reComputer RK3588** | Qwen3-ASR + Kokoro (34% NPU) | 0.340 | 0.350 | **~650 ms** | 🟢 实测 |
| **reComputer RK3576** | Qwen3-ASR + Matcha (RKNN) | 0.120 | 0.440 | **1020 ms** | 🟢 实测 |
| **Raspberry Pi 5** | Sherpa-ONNX 双语流式 (CPU) | 0.078 | 0.500 | **~1.5 s** | 🟢 实测 |

完整历史跑分、多路并发压测与原始测试数据详见 [BENCHMARKS.md](BENCHMARKS.md)。

---

## 支持设备与要求

技术栈完全开源，针对主流芯片系列优化，同架构硬件均可兼容：

| 设备系列 | 推荐产品型号 | 运行环境要求 | 支持特性 |
|---|---|---|---|
| **NVIDIA Jetson** | reComputer J4012 / J4011 / Super | JetPack 6.x (CUDA 12.6, TensorRT 10.3) | 全功能支持，Qwen3 52语流式、声音克隆、Orin NX 端侧 LLM |
| **Rockchip RK3588** | reComputer R1000 / 工业边缘主机 | Linux Kernel 5.10 / 6.1，rknpu 驱动 | 三核 NPU 并行，12 路并发 ASR，Matcha 与 Kokoro 硬件加速 |
| **Rockchip RK3576** | LubanCat-3 / 紧凑型边缘网关 | Linux Kernel 6.1，rknpu 驱动 | 超低功耗边缘网关，16 路音频转写，RKNN 原生推理 |
| **Raspberry Pi** | 树莓派 5 (8GB) / 树莓派 4 | 64 位 Raspberry Pi OS，Docker | 纯 CPU 轻量流式对讲；支持 Hailo-8 M.2 加速卡扩展 |

---

## 文档索引

- **选型与方案**：
  - [多语言设备选型矩阵 (Markdown 规范)](docs/DEVICE-LANGUAGE-MATRIX.md)
  - [多语言设备选型矩阵 (浏览器交互 Tab 网页)](docs/device-language-matrix.html)
  - [推荐模型与场景对应表](docs/RECOMMENDED-MODELS.md)
- **部署与配置**：
  - [环境变量与 Compose Profile 配置手册](docs/CONFIGURATION.md)
  - [Orin NX 本地端侧大模型部署指南](docs/deploy/jetson-orin-nx-v091.md)
  - [从零复现与构建指南 (Reproduce Guide)](docs/REPRODUCE.md)
- **开发与架构**：
  - [API 完整开发参考手册](docs/API.md)
  - [系统架构与多进程设计](ARCHITECTURE.md)
  - [开发者本地调试指南](DEVELOP.md)
  - [性能评测与并发基准详报](BENCHMARKS.md)

---

## Contributing & License

欢迎提交 Issue 和 Pull Request！本代码库遵循 [MIT License](LICENSE)。

### 致谢
- [voxedge](https://github.com/suharvest/voxedge) —— 底层纯 Python/numpy 实时流式语音推理库
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) —— 边缘 CPU 语音推理框架
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) & [Misaki](https://github.com/hexgrad/misaki) —— 高品质轻量级语音合成模型与 G2P 引擎
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) —— 超快速 Flow-matching 语音合成
- [Qwen3](https://huggingface.co/Qwen) —— 多语言 ASR 与 TTS 基础模型
- [FunASR](https://github.com/modelscope/FunASR) & [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) —— Paraformer 与 SenseVoice 语音识别模型
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) & [RKNN-Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) —— 边缘硬件原生加速推理运行时
