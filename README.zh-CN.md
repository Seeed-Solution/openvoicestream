# OpenVoiceStream

> [English](README.md) | **中文**

**面向本地语音应用的流式 ASR + TTS** —— 开箱即用、高性能、完全在设备上运行。

<p align="center">
  <a href="https://github.com/Seeed-Solution/openvoicestream"><img src="https://img.shields.io/github/stars/Seeed-Solution/openvoicestream?style=social" alt="GitHub stars" /></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/ASR-Paraformer%20%7C%20Qwen3--ASR%20%7C%20SenseVoice%20%7C%20Whisper-2f80ed.svg" alt="ASR: Paraformer, Qwen3-ASR, SenseVoice, Whisper" /></a>
  <a href="#tts-model-comparison"><img src="https://img.shields.io/badge/TTS-Matcha--TTS%20%7C%20Qwen3--TTS%20%7C%20SparkTTS%20%7C%20Kokoro%20%7C%20MOSS--TTS--Nano-f97316.svg" alt="TTS: Matcha-TTS, Qwen3-TTS, SparkTTS, Kokoro, MOSS-TTS-Nano" /></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/engines-TensorRT--EdgeLLM%20%7C%20RKNN%20%7C%20sherpa--onnx-16a34a.svg" alt="Engines: TensorRT-EdgeLLM, RKNN, sherpa-onnx" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/deploy-Docker-2563eb.svg" alt="Deploy with Docker" /></a>
  <a href="#supported-devices"><img src="https://img.shields.io/badge/ecosystems-Jetson%20%7C%20Rockchip%20%7C%20Raspberry%20Pi-65a30d.svg" alt="Supported ecosystems: Jetson, Rockchip, Raspberry Pi" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-facc15.svg" alt="MIT license" /></a>
</p>

<p align="center">
  <img src="docs/media/hero.png" alt="OpenVoiceStream - streaming ASR and TTS for edge dialogue" width="760" />
</p>

> [!TIP]
> **寻找即插即用的边缘 AI 语音硬件与开发套件？**  
> 本流式语音栈已在 Seeed Studio reComputer 边缘工控机与 reSpeaker 麦克风阵列上完成全链路基准测试与深度调优。  
> 🛒 **[获取带硬件 AEC 的语音开发套件 (reSpeaker XVF3800)](https://www.seeedstudio.com/reSpeaker-XVF3800-p-5784.html?utm_source=github&utm_medium=readme&utm_campaign=openvoicestream&utm_content=hero_banner)** | 📖 **[探索智能零售语音方案与 BOM 配置器](https://www.seeed.cc/solutions/smart-retail-store?utm_source=github&utm_medium=readme&utm_campaign=openvoicestream&utm_content=hero_banner)**

**OpenVoiceStream 提供一套开箱即用、高性能、经过我们实测的方案，可以直接
用来搭建本地语音应用。** 语音识别、语音合成，以及现成的应用 —— 对话、
智能家居控制、语音控制机械臂、翻译、实时字幕 —— 全部运行在你自己的设备
上。无云端、无语音 API 密钥、无按次计费。性能来自硬件本身：每个模型都
按加速器量化、跑在原生框架上；本 README 里的每个数字都是我们实测得出，
不是预估。

**每块板卡能干什么 —— 对话（中/英/多语言）与转录，每个数字都是公开实测**，
可溯源到 [`bench/asr_bench/results/`](bench/asr_bench/results/) 与
[BENCHMARKS.md](BENCHMARKS.md)。按场景与语言选模型：
[docs/RECOMMENDED-MODELS.md](docs/RECOMMENDED-MODELS.md)。

![每块板卡能干什么 —— 同一套栈，全部实测](docs/media/board-capability-map.svg)

**底层的语音引擎是 [`voxedge`](https://github.com/suharvest/voxedge)** —— 一个独立的、可通过 pip 安装（`pip install voxedge`）的纯 Python/numpy 库，负责实时 ASR + TTS + 对话循环。本仓库以 wheel 形式 *使用* voxedge，并在其之上补齐将其作为产品交付所需的一切。想在自己的应用里嵌入边缘语音？直接使用 voxedge。想要一套开箱即用、带预构建镜像和 agent 的设备端语音服务？那你来对地方了。

## Why This Matters

OpenVoiceStream 是硬件优先的：本地语音性能的上限由加速器决定，我们做的
事就是把这个上限释放出来。

- **性能来自硬件，所以我们从硬件出发。** 多数语音栈从软件出发，把设备当
  成通用盒子 —— 一份可移植构建、主要靠 CPU，NPU/GPU 闲置。我们反着来：
  针对每块板卡，把每个模型量化成它的加速器想要的格式（W8A8 / W4A16 /
  int4 / fp16-scaled），并跑在原生框架上 —— Jetson 用 TensorRT，Rockchip
  用 RKNN，Hailo 用 HailoRT。实测数字就是这么来的：$80 的 Raspberry Pi
  实时运行，RK3588 撑住 12 路零错误会话。
- **搭语音应用不该每次都重新解决这些。** 需要本地语音的场景越来越多 ——
  机器人、智能家居、会议室、零售终端 —— 但每个项目都在重新解决同样的
  问题：ASR 接线、TTS 服务、设备适配。本仓库把这条路径理顺：一条安装
  命令、一个稳定 API、一组现成应用。

结果就是上面的能力图：同一套栈、每块板卡、全部实测。

<p align="center">
  <img src="docs/media/solution-lineup.png" alt="OpenVoiceStream solution lineup: recommended hardware paths for real-time voice I/O, production edge voice, human-like local speech, and voice plus local LLM" width="900" />
</p>

## Quick Start

在目标设备上克隆一次即可。安装器会校验主机、选择正确的 compose 文件、拉取镜像、启动服务，并可运行健康检查、能力检查、TTS 冒烟测试以及 TTS-到-ASR 往返测试：

```bash
git clone --recurse-submodules https://github.com/Seeed-Solution/openvoicestream.git
cd openvoicestream

deploy/install.sh --pull --verify   # 自动识别 Jetson / Rockchip / Raspberry Pi
```

自动检测不适用时显式指定：

```bash
deploy/install.sh --target orin-nx --pull --verify  # v0.9.1：Qwen3-ASR + Matcha + 本地 LLM
deploy/install.sh --target jetson --pull --verify
deploy/install.sh --target rk3588 --pull --verify
deploy/install.sh --target rk3576 --pull --verify
deploy/install.sh --target rpi --pull --verify
```

### 推荐模型

模型跟随场景与语言 —— 中文对话 → Qwen3-ASR + Matcha，英文 → Kokoro，
多语言 → Qwen3-TTS，转录 → SenseVoice（中文）/ Whisper（英文）；板卡自动
拉取各自的量化构建。完整矩阵：
[docs/RECOMMENDED-MODELS.md](docs/RECOMMENDED-MODELS.md)。

> **初次接触本仓库？** [`docs/REPRODUCE.md`](docs/REPRODUCE.md) 是端到端、从零开始的复现指南：运行预构建镜像（路径 A）、从零重建引擎（路径 B），或构建镜像（路径 C）。

启动后，语音服务监听在 `http://device:8621`（Orin NX v0.9.1 还会在 `:8000`
拉起本地 LLM）。当前镜像 tag 以 `deploy/` 下的 compose 文件为准 —— 本
README 不再复制副本。（已发布镜像仍沿用先前的 registry 命名空间，现有
部署可继续拉取。）

你真正交付的是语音服务之上的一个**应用** —— 去
[Applications](#applications) 选一个。每个应用的 README 自带部署矩阵：
compose 文件、镜像 tag、模型与对应板卡的验收步骤。

Orin NX v0.9.1 的正式组合、模型级下载与回滚流程见
[`docs/deploy/jetson-orin-nx-v091.md`](docs/deploy/jetson-orin-nx-v091.md)。
从一台裸设备开始逐步搭建（宿主前置条件、拓扑选择、profile 选择、故障排查）见
[`docs/runbooks/jetson-voice-stack-setup.md`](docs/runbooks/jetson-voice-stack-setup.md)。

手动验证：

```bash
# Same default URL on Jetson, RK3576, RK3588, and Raspberry Pi.
deploy/verify.sh --url http://device:8621 --tts-smoke --roundtrip
curl http://device:8621/health
```

OpenAI 兼容客户端可先发现当前 ASR/TTS 模型 ID 及其运行时能力，再发送音频：

```bash
curl http://device:8621/v1/models
curl http://device:8621/v1/capabilities

# 使用 /v1/models 返回的当前 TTS 模型 ID。
curl -X POST http://device:8621/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"<tts-model-id>","input":"你好，边缘设备。"}' \
  --output speech.wav

# 使用 /v1/models 返回的当前 ASR 模型 ID。
curl -X POST http://device:8621/v1/audio/transcriptions \
  -F "model=<asr-model-id>" -F "file=@speech.wav"
```

当当前后端支持流式合成时，`POST /v1/audio/speech` 会在同一路由上采用
HTTP chunked 流式返回。`voice` 和 `speed` 只能依据
`GET /v1/capabilities` 返回的模型级声明选择。

客户端示例位于 [`examples/`](examples/)：

```bash
python3 examples/stream_tts_to_wav.py \
  --url http://device:8621 \
  --text "你好，欢迎使用 OpenVoiceStream。" \
  --out /tmp/ovs-tts.wav
```

**使用 compose 部署**，当你希望自己管理 profile 时。上方推荐搭配就是每个平台的首选 profile，其余都是可切换选项：

```bash
# Jetson —— 推荐：Qwen3-ASR + Matcha（v0.9.1 Orin NX 路径用上方专用 compose）。
docker compose -f deploy/docker-compose.yml up -d

# Jetson —— 最快复现的轻量档（install.sh 默认）：
OVS_PROFILE=jetson-zh-en docker compose -f deploy/docker-compose.yml up -d

# Jetson —— Qwen3 多语言 ASR/TTS，含声音克隆：
OVS_PROFILE=jetson-multilang-highperf-nx \
docker compose -f deploy/docker-compose.yml up -d

# Jetson —— TTS 升级：Kokoro TRT（英文，53 音色）、Paraformer+Kokoro 混搭、
# 或 MOSS-TTS-Nano（多语言，48kHz 立体声）。
OVS_PROFILE=jetson-kokoro-trt docker compose -f deploy/docker-compose.yml up -d
OVS_PROFILE=jetson-paraformer-kokoro docker compose -f deploy/docker-compose.yml up -d
OVS_PROFILE=jetson-moss-tts-nano-trt docker compose -f deploy/docker-compose.yml up -d

# Rockchip —— 推荐默认档（Qwen3-ASR RKNN W8A8 + Matcha RKNN）。
docker compose -f deploy/docker-compose.radxa.yml up -d   # RK3588
docker compose -f deploy/docker-compose.rk.yml up -d       # RK3576

# Rockchip —— Whisper ASR（英文长语音）或 Qwen3 ASR + Kokoro RKNN TTS，均在 RK3588。
OVS_PROFILE=rk3588-whisper-10s \
docker compose -f deploy/docker-compose.radxa.yml up -d
OVS_PROFILE=rk3588-kokoro-rknn \
docker compose -f deploy/docker-compose.radxa.yml up -d
```

`deploy/install.sh --pull --verify` 在目标设备上自动检测 Jetson/RK/RPi。以上所有 profile 共享同一客户端 API —— 切换 profile 只是重启，不是重写。

## Demo Gallery

设备本机提供的浏览器演示门户：实时设备状态、每个能力一张演示卡（实时字幕、语音合成体验、带打断的语音对话、声音克隆、说话人分离）、运行时 ASR/TTS 模型热切换，以及面向展会的 kiosk 模式（`DEMO_KIOSK=1`）。

```bash
docker compose -f demos/docker-compose.demos.yml --profile all up -d
# 打开 http://<device>:8700
```

部署与服务端前置条件见 [`demos/README.md`](demos/README.md)；全部演示资产（gallery 卡片、API 示例、agent 示例、bench 演示脚本）的总索引见 [`docs/DEMOS.md`](docs/DEMOS.md)。

## Applications

仓库内置九个应用层 —— 每个都是构建在共享 `ovs_agent` 运行时和 SLV 语音服务之上的可用语音产品，而不是代码片段。用 `uv run ovs-agent run <name> --config <config.yaml>` 启动（或按各自的部署矩阵用 `docker compose` 拉起）。

| 应用 | 你能得到什么 | 链路 | 文档 |
|---|---|---|---|
| [`conversation`](agent/ovs_agent/apps/conversation/README.md) | 最小全双工语音对话 —— 说话、得到语音回答、可打断 | ASR → LLM → TTS | [README](agent/ovs_agent/apps/conversation/README.md) 含部署矩阵 |
| [`home_assistant`](agent/ovs_agent/apps/home_assistant/README.md) | 语音控制已有的 Home Assistant：「把客厅的灯调暗一点」 | ASR → HA 意图 | [README](agent/ovs_agent/apps/home_assistant/README.md) |
| [`companion_robot`](agent/ovs_agent/apps/companion_robot/README.md) | 具身机器人（Reachy Mini 等）的语音入口 | ASR → LLM + 机器人工具 → TTS | [README](agent/ovs_agent/apps/companion_robot/README.md) |
| [`voice_rebot_arm`](agent/ovs_agent/apps/voice_rebot_arm/README.md) | 语音控制机械臂：力控夹爪 + IK + 视觉引导抓取 | 唤醒词 → ASR → LLM 工具调用 → 机械臂 | [README](agent/ovs_agent/apps/voice_rebot_arm/README.md) |
| [`voice_arm`](agent/ovs_agent/apps/voice_arm/README.md) | 语音控制 SO-ARM100 执行器 | 唤醒词 → ASR → LLM 工具 → TTS | [README](agent/ovs_agent/apps/voice_arm/README.md) |
| [`multi_mode`](agent/ovs_agent/apps/multi_mode/README.md) | 标准语音应用，运行时可切换模式（对话、命令……） | ASR → LLM → TTS | [README](agent/ovs_agent/apps/multi_mode/README.md) |
| [`translator`](agent/ovs_agent/apps/translator/README.md) | 句级语音翻译，无需 LLM | ASR → MT → TTS | [README](agent/ovs_agent/apps/translator/README.md) |
| [`simul_interpret`](agent/ovs_agent/apps/simul_interpret/README.md) | 同声传译：单调提交保证（已播出的音频永不回稿） | ASR → MT → TTS | [README](agent/ovs_agent/apps/simul_interpret/README.md) |
| [`live_caption`](agent/ovs_agent/apps/live_caption/README.md) | 实时双语字幕上屏 | ASR → MT → 广播 | [README](agent/ovs_agent/apps/live_caption/README.md) |

每个应用的文档契约（部署矩阵、推荐模型、验收步骤、实测结果规则）定义在[应用目录](agent/ovs_agent/apps/README.md)。

## Table of Contents

- [Why This Matters](#why-this-matters)
- [Applications](#applications)
- [Quick Start](#quick-start)
- [Demo Gallery](#demo-gallery)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Qwen3 Multilingual Path](#qwen3-multilingual-path)
- [Performance](#performance)
- [Configuration](#configuration)
- [Models](#models)
- [Supported Devices](#supported-devices)
- [Patched sherpa-onnx](#patched-sherpa-onnx)
- [Project Structure](#project-structure)
- [Changelog（独立文件）](CHANGELOG.md)
- [Acknowledgements](#acknowledgements)

## Key Features

- **流式优先 API** —— 带 partial/final 结果的 WebSocket ASR，以及带句级音频块的 HTTP 流式 TTS。
- **按目标量化、原生框架** —— 每个模型都按设备系列量化（W8A8 / W4A16 / int4 / fp16-scaled），并运行在各自加速器的原生运行时上：Jetson 用 TensorRT-EdgeLLM，Rockchip 用 RKNN/RKLLM，Hailo-8 用 HailoRT，CPU 路径用 sherpa-onnx 和 ONNX Runtime。热路径上没有通用兼容层。
- **可复用的边缘语音库** —— 各后端以独立的、可通过 pip 安装的 [`voxedge`](https://github.com/suharvest/voxedge) 包形式发布（`pip install --pre voxedge`）；本仓库是构建在其之上的产品服务 + 部署。
- **稳定的后端契约** —— 在 profile 切换时，客户端仍保持相同的 `/asr/stream`、`/tts`、`/tts/stream` 和 `/health` 调用。
- **实测低延迟** —— 在 Jetson Orin NX 上使用 Paraformer + Matcha 时，EOS-到-首音频为 58 ms；使用 Qwen3 ASR/TTS 声音克隆时为 157 ms。
- **已验收的 Orin NX v0.9.1 栈** —— Qwen3-ASR + Matcha-TTS 与 Qwen3.5-4B GDN/MTP 同驻，默认使用 8K 上下文，并提供已验收的可选 4K 引擎；模型级产物均锁定 revision 和 SHA。详见 [v0.9.1 部署指南](docs/deploy/jetson-orin-nx-v091.md)。
- **v0.9.0 历史并发验证** —— 上一版本验证了 2 会话 ASR 流式（中/英，无串扰）以及 N=2 Qwen3-TTS Base（int4 talker，约 4 GB RAM；或采用 shared-engine 时第二个槽仅多占用 +1.6 GB）。详见 [BENCHMARKS.md](BENCHMARKS.md)。
- **多语言选项** —— 中英双语、仅英文，以及 52 语言的 Qwen3 路径，均通过同一个服务暴露。
- **容器优先部署** —— 预构建镜像、针对目标的 compose 文件、主机检查、模型下载和验证脚本均已包含在内。
- **面向 LLM 的 agent 层** —— `agent/` 将 ASR 结果流式送入 OpenAI 兼容或 EdgeLLM 后端，再把 LLM token 直接流式回送到 TTS。
- **完全本地的经济性** —— 无语音 API key、无按次 ASR/TTS 费用、产物缓存后运行时无需联网，且语音热路径中没有 PyTorch/Transformers。

## Architecture

```text
┌───────────────────────────────────────────────────────────┐
│  Edge device (Jetson Orin / RK3576 / RK3588 / RPi 4–5)    │
│                                                           │
│  FastAPI service (container :8000; host default :8621)     │
│  ├── WS /asr/stream    Streaming ASR                      │
│  │     └─ zh_en: Paraformer  │  en: Zipformer  │  multi: Qwen3-ASR  │  rk: Paraformer RKNN · Qwen3-ASR │
│  ├── POST /asr          SenseVoice offline ASR (zh+en)    │
│  ├── POST /tts          Batch TTS                         │
│  └── POST /tts/stream   Streaming TTS                     │
│        └─ zh_en: Matcha-TTS  │  en: Kokoro v1.0  │  multi: Qwen3-TTS │
│                                                           │
│  Inference: sherpa-onnx · TRT-EdgeLLM · RKNN              │
└───────────────────────────────────────────────────────────┘
         ▲ HTTP / WebSocket
         │
   Any client (SBC, laptop, robot, kiosk, ...)
```

模型根据 `LANGUAGE_MODE` 自动选择：

| Service | Endpoint | zh_en (default) | en | multilingual | Protocol |
|---------|----------|-----------------|-----|---------------|----------|
| **流式 ASR** | `WS /asr/stream` | Paraformer 双语 | Zipformer 英文 | Qwen3-ASR（52 语言） | WebSocket：输入 int16 PCM，输出 JSON |
| **流式 TTS** | `POST /tts/stream` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS（声音克隆） | HTTP：输入 JSON，输出原始 PCM 流 |
| **批量 TTS** | `POST /tts` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS（声音克隆） | HTTP：输入 JSON，输出 WAV |
| 离线 ASR | `POST /asr` | SenseVoice（zh+en+ja+ko+yue） | SenseVoice（同上） | Qwen3-ASR（52 语言） | HTTP：上传 WAV，输出 JSON |

**各后端能力不同：**

| Backend | Speed control | Pitch shift | Voice clone | Languages | Streaming |
|---------|--------------|-------------|-------------|-----------|-----------|
| Sherpa (zh_en/en) | ✅ | ✅ | ❌ | 2 (zh+en) | ✅ |
| Paraformer RKNN (RK) | ❌ | ❌ | ❌ | 2 (zh+en) | ✅ |
| Kokoro TRT (Jetson) | ❌ | ❌ | ❌ | 1 (en) | ✅ |
| Kokoro RKNN (RK3588) | ❌ | ❌ | ❌ | multi | ✅ |
| Qwen3 (multilingual) | ❌ | ❌ | ✅ (x-vector) | 52 | ✅ |
| Qwen3-CustomVoice | ❌ | ❌ | ❌ (9 presets + instruct) | 52 | ✅ |
| MOSS-TTS-Nano (Jetson) | ❌ | ❌ | ❌ | multi | ✅ |
| RKNN (Rockchip) | ✅ | ✅ | ❌ | 2 (zh+en) | ✅ |

本服务在 API 层面与模型无关 —— 客户端发送音频/文本，得到音频/文本返回。在不改动客户端代码的情况下即可更换引擎。不受支持的参数会返回 `501` 并附带 `{"required_capability": "..."}`。

## API Reference

### OpenAI 兼容音频与发现接口

兼容接口使用 `GET /v1/models` 返回的当前模型 ID：

| 接口 | 用途 |
|---|---|
| `POST /v1/audio/speech` | 使用 JSON `model` + `input` 合成语音，默认返回 WAV；支持流式的后端在同一路由上发送 chunked 音频。 |
| `POST /v1/audio/transcriptions` | 使用 multipart `model` + `file` 转写，默认返回 `{"text":"..."}`。 |
| `GET /v1/models` | 列出已配置的 ASR/TTS 模型 ID、别名及 readiness 元数据。 |
| `GET /v1/capabilities` | 发现模型级音色、语速控制、流式、克隆及并发支持。 |

不要硬编码音色名称，也不要假设切换模型后仍支持 `speed`。应通过
`GET /v1/capabilities` 解析所选模型的 `voice` 和 `speed` 支持情况。
不支持的格式和选项会返回结构化客户端错误，不会静默修改请求。

### 流式 ASR（WebSocket）

```
WS /asr/stream?sample_rate=16000&language=auto
```

- 客户端发送：原始 **int16 PCM 字节**（音频块，例如每块 100ms）
- 客户端发送：**空字节** `b""` 以表示音频结束
- 服务端发送：JSON `{"text": "...", "is_final": bool, "is_stable": bool}`

```python
import asyncio, websockets

async def transcribe():
    async with websockets.connect("ws://device:8621/asr/stream?sample_rate=16000") as ws:
        for chunk in audio_chunks:  # np.int16 arrays
            await ws.send(chunk.tobytes())
            result = await ws.recv()  # partial results
        await ws.send(b"")  # signal end
        final = await ws.recv()  # {"text": "...", "is_final": true}
```

### 离线 ASR（HTTP）

```bash
curl -X POST http://device:8621/asr \
  -F "file=@recording.wav" -F "language=auto"
# {"text": "transcribed text"}
```

### TTS（HTTP）

```bash
curl -X POST http://device:8621/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "sid": 52, "speed": 1.0}' \
  --output output.wav
```

参数：`text`（必填）、`sid`（说话人 ID，默认 52）、`speed`（语速，默认 1.0）

**注意：** `speed` 仅在声明支持语速控制的后端（Sherpa/Matcha/RKNN）上生效。Qwen3-TTS（`multilanguage` profile）目前不支持可靠的语速或音高调整，因此客户端应将这些参数视为在 Qwen3 上不受支持。

### Speaker Management

用于列出、注册和删除 TTS 说话人的接口。说话人 ID 的作用域限定于当前激活的 TTS 模型。

```bash
# List all speakers for the active TTS model
curl http://device:8621/tts/speakers
# {"model_id": "kokoro-multi-lang-v1_0", "default_speaker_id": 52, "speakers": [...]}

# Register a voice-clone embedding (requires VOICE_CLONE capability)
curl -X POST http://device:8621/tts/speakers/register \
  -H "Content-Type: application/json" \
  -d '{"speaker_embedding_b64": "...", "label": "my-voice"}'

# Delete a registered speaker (preset speakers cannot be deleted)
curl -X DELETE http://device:8621/tts/speakers/42
```

Kokoro 暴露 53 个预设说话人（id 0-52），带有各语言的语音标签（`af_heart`、`bm_george`、`zf_xiaobei` 等）。Qwen3-TTS 通过 `/tts/clone/embedding` 暴露声音克隆能力，并支持持久化注册。

### TTS 流式（HTTP）

返回原始 PCM：前 4 字节 = 采样率（uint32 LE），随后是 int16 采样点。

```
POST /tts/stream
Content-Type: application/json
{"text": "Hello world", "sid": 52}
```

### Health Check

```
GET /health  →  {"asr": bool, "tts": bool, "streaming_asr": bool}
```

## Qwen3 Multilingual Path

v0.9.1 profile（`jetson-edgellm-v091-*`）可独立选择 Qwen3-ASR 和一个 TTS
后端。导出、引擎构建和 worker 代码维护在
[`suharvest/jetson-voice-engine`](https://github.com/suharvest/jetson-voice-engine)，
并以 `third_party/jetson-voice-engine/` submodule 固定。生成产物按模型分仓；
仓库、不可变 revision、hash 和大小以 `deploy/artifacts/v091-release-lock.json`
为准。原聚合仓 `qwen3-edgellm-jetson-artifacts` 仅保留给旧 profile。

Qwen3.5-4B GDN/MTP 的 4K 和 8K 使用同一个模型级 HF 仓库和同一个运行时
镜像；默认 compose 选择 8K，设置 `EDGELLM_ENGINE_PROFILE=4k` 可选择 4K
引擎。两者的 MTP 安全 slack 都是 `128`。最终 payload 锁定为：4K
`06273e358a579590bb8344b451aa35c89983cd99401339fb1858d61af4dbd107`，8K
`9208e46d61a4f1440ac68a312e35dde3d04b88edf0e4ee12b32210e7190d3325`。
已发布的不可变 HF revision 分别为 4K
`9f2c2059341fd2135cc3a0ec09e05150277ea5b6`、8K
`adb1c78fb61513e2d7d8e7f889f6196dbefb1e5e`。

**在全新 Orin NX 上最快的路径：**

```bash
git clone https://github.com/suharvest/jetson-voice-engine.git
bash jetson-voice-engine/scripts/reproduce_qwen3_highperf.sh \
  --reference /path/to/24kHz_mono.wav   # optional: gates the voice-clone path
```

该编排脚本会构建运行时、下载并以 SHA-256 校验 HF 产物、构建 slim docker 镜像、启动服务，并运行验证器（`scripts/verify_reproduction.sh`）。退出码 0 表示端口 18092 上的 slim 容器健康，并正在提供经过验证的整套栈服务。

在同一 API 表面下提供 **两套运行时 profile**：

| Profile | Goal | Default behavior |
|---------|------|------------------|
| `official` | 最小改动的 EdgeLLM 示例。足够贴近上游，可作为 Qwen3 ASR/TTS 示例被审阅或上游合并。 | 仅做语义/正确性修复 —— tokenizer 布局、采样、运行时契约、stream callback。常规导出的 Talker/CodePredictor/Code2Wav 目录。 |
| `highperf`（默认） | 面向 Orin 的产品级低延迟双驻留路径。 | 完整 vocab、ASR FP8 embedding、Orin NX 上的 FP16 CustomVoice Talker（1024-token Talker KV 上限）、CP BF16 I/O + `lm_head` 预转置、有状态 Code2Wav、CP decode CUDA graph、`ACTIVE_CP_GROUPS=13`。 |

在 Orin NX 上消费 NX 原生引擎集时使用 `jetson-multilang-highperf-nx`；默认的 `jetson-multilang-highperf` profile 面向 Nano 产物集。[`configs/profiles`](configs/profiles) 中的 profile 仅设置 env 默认值；显式 env 变量仍会覆盖它们。

**CustomVoice 变体。** 设置 `QWEN3_TTS_VARIANT=customvoice`（或在 `OVS_TTS_MODEL_ID` 中包含 `customvoice`）会选择 Qwen3-TTS-12Hz-0.6B-CustomVoice talker。它内置 **9 个内建说话人**（vivian、ryan、aiden、serena、dylan、eric、uncle_fu、ono_anna、sohee），由自然语言指令驱动，而非 x-vector 声音克隆 —— 因此 `VOICE_CLONE` 能力关闭，`/speakers/register` 会被拒绝。当前 CustomVoice 生产精度在 Orin NX 上为 FP16；默认 NX 引擎使用 1024-token Talker KV 上限以降低驻留内存。在不存在 no-preload 且 EOS 有效的量化版本之前，W8A16 被拒绝。

关于详细的分支归属、引擎 env 变量、冻结基线数字和产物处理，参见 Jetson
引擎仓库的
[`qwen3-current-frozen-baseline-2026-05-10.md`](https://github.com/suharvest/jetson-voice-engine/blob/main/docs/plans/qwen3-current-frozen-baseline-2026-05-10.md)。

当前发布状态、镜像 digest、产物仓库和已知缺口跟踪在 [`docs/productization-status.md`](docs/productization-status.md)。

## Performance

### 同一语料、五款加速器 —— Whisper 词错率（2026-09）

每台设备都在同一份固定语料（每语言 100 段）、同一评分器下测得；五款设备全部 100/100 段落有效。行与行之间唯一的变量就是设备/后端。

| 设备 | Whisper 后端 | 整体 WER |
|---|---|---:|
| Jetson Orin NX 16GB（J4012） | TensorRT bf16 编码器 + CPU ONNX 解码器 | **7.62%** |
| Jetson Orin Nano 8GB（J3011） | TensorRT bf16 编码器 + CPU ONNX 解码器 | **7.62%** |
| RK3588（reComputer） | RKNN base10 编码器 + CPU ONNX 解码器 | **7.50%** |
| RK3576（reComputer） | RKNN base10 编码器 + CPU ONNX 解码器 | **8.51%** |
| Raspberry Pi 5 + Hailo-8（R2000） | Hailo base 编码器 + CPU ONNX 解码器 | **8.39%** |

完整方法、逐次运行说明与被撤回的修复前数据：
[`bench/asr_bench/results/accuracy-unified-corpus.md`](bench/asr_bench/results/accuracy-unified-corpus.md)。各设备并发上限（Jetson 上流式 ASR 准入最高 16 路）见同一目录。

### 跨设备基准测试（2026-05-18 实测）

Jetson/RPi 行来自最初针对 `http://127.0.0.1:8621` 的本地 forced-EOS gate。RK 行在 true-streaming 修复后以 `QWEN3_ASR_CHUNK_CONFIRM=0`、`--eos vad` 和 `--vad-silence-ms 800` 重新运行；其 V2V 列拆分为 `/asr/stream` 加 `/tts/stream`。

| Target / profile | Image | TTS backend | ASR backend | TTS RTF p50 | ASR fRTF p50 | ASR CER p50 | V2V EOS→audio p50 |
|---|---|---|---|---:|---:|---:|---:|
| Orin Nano `jetson-multilang-highperf` | `jetson-v1.12-highperf` | `trt_edgellm` | `trt_edgellm` | 0.470 | 0.076 | 5.3% | 251 ms |
| Orin NX `jetson-multilang-highperf-nx` | `jetson-v1.12-highperf` | `trt_edgellm` | `trt_edgellm` | 0.417 | 0.042 | 5.3% | 157 ms |
| Orin Nano `jetson-qwen3asr-matcha` | `jetson-v1.12-highperf` | `matcha_trt` | `trt_edgellm` | 0.024 | 0.075 | 5.3% | 286 ms |
| Orin NX `jetson-qwen3asr-matcha-nx` | `jetson-v1.12-highperf` | `matcha_trt` | `trt_edgellm` | 0.018 | 0.042 | 5.3% | 162 ms |
| Orin Nano `jetson-zh-en` | `jetson-v1.12-highperf` | `matcha_trt` | `paraformer_trt` | 0.023 | 0.077 | 13.3% | 327 ms |
| Orin NX `jetson-zh-en` | `jetson-v1.12-highperf` | `matcha_trt` | `paraformer_trt` | 0.018 | 0.015 | 10.5% | 58 ms |
| RK3588 `rk3588-default` | `rk-qwen3asr-opt-20260610` | `rk:matcha_rknn` | `rk:qwen3_asr_rk` | 0.124 | 0.318 | 10.1% long avg | 528 ms |
| RK3576 `rk3576-default` | `rk-qwen3asr-opt-20260610` | `rk:matcha_rknn` | `rk:qwen3_asr_rk` | 0.290 | 0.265 | 9.8% long avg | 1020 ms |
| Raspberry Pi 5 `rpi5-default` | `rpi-v1.0-onnx` | `sherpa` | `sherpa_asr` | 0.078 | 0.000 | 20.0% | 3 ms |

RK 行使用 2026-06-10 的高性能 Qwen3 ASR W8A8 + Matcha 复检。强制客户端-EOS 的 V2V p50 在 RK3588 上为 528 ms，在 RK3576 上为 1020 ms；长篇听写平均错误率为 10.1% / 9.8%。真实的 `/v2v/stream` 路径仍取决于所配置的 VAD endpointing 延迟。

同一次运行得到的部署占用：

| Target | Image size | Model / engine volume | Resident memory | Startup to ready |
|---|---:|---:|---:|---:|
| Orin Nano | 2.02 GB | 5.14 GB | 2.14 GiB | 14 s |
| Orin NX | 2.02 GB | 5.45 GB | 1.02 GiB | 13 s |
| RK3588 | 767 MB | 3.31 GB ASR + 301 MB TTS | 4.09 GiB | 9 s |
| RK3576 | 767 MB | 2.21 GB ASR + 351 MB TTS | 2.71 GiB | 15 s |
| Raspberry Pi 5 | 568 MB | 2.19 GB | n/a from Docker stats | 9 s |

并发冒烟测试（`parallel=2`，`asr_tts_simul`）在 Jetson Nano/NX
Paraformer+Matcha、RK3588、RK3576 和 Raspberry Pi 5 上均通过。Jetson p=2
功能可用，但 TTS 会变为吞吐受限（RTF 约 1.3-1.4），因此当低延迟并发对话
重要时，请使用 Orin NX 或 Qwen3 ASR + Matcha 的拆分方案。完整的原始 JSON
路径和方法学见[`性能测试运行手册`](docs/perf-test-runbook.md)。

### 并发历史（v0.8.0 / v0.9.0，2026-06/07）

- **v0.8.0** —— Jetson 上验证的 2 路并发：ASR N=2 流式（中英无串音，第 3 路
  以 `4389 too_many_sessions` 拒绝），TTS N=2 走 slot-pool（int4 talker，
  245.9 MB vs 903 MB fp16）或共享引擎（第 2 路只增 +1.6 GB）；并发输出与
  单路逐字节一致，零 CUDA 错误。
- **v0.9.0** —— Orin NX 上六模型设备端验证（SparkTTS-0.5B W4A16 成为全能之选），
  N=2 在新栈复验。

完整 gate ID、逐模型表格与零回归分析见
[BENCHMARKS.md](BENCHMARKS.md)。

### TTS Model Comparison

当前发布版本在双语路径使用 Matcha/Vocos，在仅英文部署使用 Kokoro，在需要声音克隆或 52 语言 TTS 时使用 Qwen3-TTS，为轻量级多语言纯 TTS 路径使用 MOSS-TTS-Nano，以及用 SparkTTS 提供属性可控音色 + zero-shot 声音克隆。下表中的 RTF 数字在可获得处取自 2026-05-18 的基准测试运行；未使用的研究模型作为历史背景保留。

| Model | Current role | Streaming RTF p50 | First audio p50 | Notes |
|-------|--------------|------------------:|----------------:|-------|
| **Matcha-TTS + Vocos** | 默认双语 TTS | Orin NX 上 0.018，RK3588 上 0.075，RPi5 上 0.078 | 2.6-7.5 ms | 实践中最快的 TTS 路径；无声音克隆。 |
| **Qwen3-TTS** | 多语言声音克隆 | Orin NX 上 0.417，Orin Nano 上 0.470 | 4.4-7.3 ms | 质量/特性更高，但比 Matcha 重得多。x-vector 克隆，或 `customvoice` 变体（9 个指令控制的预设）。 |
| **SparkTTS** | 可控音色 + 声音克隆（Jetson） | **0.50（v0.9.0 W4A16）**，v0.8.0 上 0.74 | **0.41–0.46 s（v0.9.0 W4A16）**，v0.8.0 上克隆约 0.25 s / 可控约 0.9 s | Qwen2.5-0.5B + BiCodec 单码本。**50 种可控音色**（性别 × 5 音高 × 5 语速，无需参考音频）**且**支持 zero-shot 声音克隆（音色 cos ~0.90）。在 **v0.9.0 上 W4A16 成为全面优选** —— 更快更轻且质量零损；bf16 也发布。W4A16 INT4-AWQ 引擎 645 MB（−58%），bf16/fp16 混合精度（修复 Qwen2.5 fp16 溢出）。中文 CER 0 / 英文 WER ≤0.02；N=2 字节级一致。 |
| **MOSS-TTS-Nano** | 多语言纯 TTS（Jetson） | — | Orin NX 上约 157 ms TTFA | 0.1B 模型，通过 C++ TRT 输出 48kHz 立体声（比 ORT CPU 兜底快 19×）。无声音克隆。 |
| **Kokoro v1.0** | 仅英文 TTS | 不在本次基准运行中 | 历史值约 130 ms TTFT | 为仅英文部署保留。在 RK3588 上，hybrid CPU+NPU RKNN 路径提供多语言 TTS（`rk3588-kokoro-rknn`）。 |
| CosyVoice3 | 仅研究 | 未发布 | 历史值约 800 ms TTFT | 质量更高，但对本次发布过重。 |
| F5-TTS | 仅研究 | 未发布 | 历史值约 2.5 s TTFT | 不适合低延迟边缘对话。 |

当前的流式基准脚本位于 `bench/perf/`。

### Performance Tuning

在 Jetson 上启动后运行一次，将时钟锁定到最高：

```bash
sudo ./scripts/setup-performance.sh
```

这会设置 MAXN 功耗模式、锁定 CPU/GPU 时钟，并禁用动态频率调节。这对于一致的推理延迟至关重要。

## Configuration

### 环境变量

| Variable | Default | Description |
|----------|---------|-------------|
| `OVS_PROFILE` | unset | 首选的 OpenVoiceStream profile 选择器，例如 `jetson-zh-en`、`jetson-multilang-highperf-nx`、`rk3588-default`、`rpi5-default` |
| `LANGUAGE_MODE` | `zh_en` | `zh_en`（中文+英文）、`en`（仅英文），或 `multilanguage`（Qwen3，52 语言；profile 通常会为你设置此项） |
| `TTS_PROVIDER` | `cuda` | ONNX 执行 provider |
| `TTS_DEFAULT_SID` | `52` | 默认 TTS 说话人 ID（52=af_cute，3=af_heart）—— 仅 Sherpa |
| `TTS_DEFAULT_SPEED` | `1.0` | 支持该功能的后端的 TTS 播放语速；Qwen3-TTS 不支持 |
| `TTS_NUM_THREADS` | `4` | TTS 推理线程数 |
| `TTS_PITCH_SHIFT` | `0` | 音高偏移（半音）—— **仅 Sherpa** |
| `SENSEVOICE_LANGUAGE` | `auto` | SenseVoice 语言提示 |
| `STREAMING_ASR_PROVIDER` | `cuda` | 流式 ASR 执行 provider |
| `MODEL_DIR` | `/opt/models` | 模型存储目录 |

将 `.env.example` 复制为 `.env` 即可自定义。

### Jetson Kokoro TensorRT Profile

`OVS_PROFILE=jetson-kokoro-trt` 在 Jetson Orin 上启用经过验证的 Kokoro
split-generator 运行时（仅 TTS，英文，53 预置音色）。三个同源 profile
共享同一产物集 —— `jetson-kokoro-trt-quality`（48 token，保守长文本
 gate）、`jetson-kokoro-trt-long`（96 token，更多 256-512 bucket 覆盖）、
以及 `jetson-paraformer-kokoro`（双语 Paraformer ASR + Kokoro TTS）。

引擎布局、bucket 路由与流式 token 切分器属于引擎层细节：见冻结产物记录
[`deploy/artifacts/kokoro_trt_manifest.json`](deploy/artifacts/kokoro_trt_manifest.json)
与复现指南
[`docs/kokoro-trt-reproduction.md`](docs/kokoro-trt-reproduction.md)。
当 Kokoro TTS 和本地 ASR 服务暴露在不同端口上时，使用
`scripts/verify_tts_asr_roundtrip.py`。

## Models

九个模型家族共享同一套 API —— ASR：Qwen3-ASR、SenseVoice、Paraformer、
Whisper · TTS：Matcha、Kokoro、Qwen3-TTS、MOSS-TTS-Nano、SparkTTS。
你不需要手动挑选产物：每个设备系列在首次启动时自动拉取属于自己的、
已量化、原生框架构建。

**实测数字为什么是这样的：** 每个模型都按目标设备量化
（W8A8 / W4A16 / int4 / fp16-scaled），并运行在各自加速器原生的推理
框架上 —— Jetson 用 TensorRT，Rockchip 用 RKNN/RKLLM，Hailo-8 用
HailoRT，CPU 用 sherpa-onnx/ONNX Runtime。热路径上没有通用兼容层。
这就是为什么 $80 的 Raspberry Pi 能做到实时，RK3588 能撑住 12 路零错误
并发。

模型产物首次启动时下载并缓存在 Docker volume 中；实测 volume 占用：
Jetson 5.14-5.45 GB，RK 2.56-3.61 GB，Raspberry Pi 5 2.19 GB。产物
revision 按 profile 锁定 —— 见[配置](#配置)与 [BENCHMARKS.md](BENCHMARKS.md)。

## Supported Devices

技术栈按芯片系列划分且完全开源 —— 同系列任何板卡都应该能跑。以下是我们进行基准测试并推荐的硬件与音频外设：

| 推荐设备型号 | 计算 / 音频平台 | 硬件规格与特性 | 典型实测角色 | 采购样机 / 套件 |
|---|---|---|---|---|
| **reComputer Industrial J4012** | NVIDIA Jetson Orin NX 16GB | 100 TOPS, IP40 无风扇, -20~60°C 宽温, 双千兆网口 | 高并发 Qwen3 多语言识别与语音克隆 | [🛒 立即选购](https://www.seeedstudio.com/reComputer-Industrial-J4012-p-5586.html?utm_source=github&utm_medium=readme&utm_campaign=openvoicestream&utm_content=hw_table) |
| **reComputer J3011** | NVIDIA Jetson Orin Nano 8GB | 40 TOPS, M.2 NVMe, JetPack 6.2 | 实时中英双语对话与 Kokoro TensorRT 加速 | [🛒 立即选购](https://www.seeedstudio.com/reComputer-J3011-p-5585.html?utm_source=github&utm_medium=readme&utm_campaign=openvoicestream&utm_content=hw_table) |
| **reComputer RK3588** | Rockchip RK3588 | 6 TOPS NPU, 8核 CPU, HDMI 2.1 | 12 路高并发 SenseVoice + Matcha 本地语音服务 | [🛒 立即选购](https://www.seeedstudio.com/reComputer-RK3588-30-p-6817.html?utm_source=github&utm_medium=readme&utm_campaign=openvoicestream&utm_content=hw_table) |
| **reComputer RK3576** | Rockchip RK3576 | 6 TOPS NPU, 低功耗, 四核 A72 + 四核 A53 | 超低功耗边缘语音网关与智能家居中控 | [🛒 立即选购](https://www.seeedstudio.com/reComputer-RK3576-p-6202.html?utm_source=github&utm_medium=readme&utm_campaign=openvoicestream&utm_content=hw_table) |
| **reSpeaker XVF3800** | XMOS XVF3800 4 麦克风阵列 | 硬件级回声消除 (AEC), 波束成形, 降噪 | 5 米远场拾音与全双工语音打断 (Barge-in) 必备 | [🛒 立即选购](https://www.seeedstudio.com/reSpeaker-XVF3800-p-5784.html?utm_source=github&utm_medium=readme&utm_campaign=openvoicestream&utm_content=hw_table) |
| **Raspberry Pi 5** | Broadcom BCM2712 8GB | 4 核 Cortex-A76 @ 2.4GHz | CPU 推理, 最低硬件成本实时中英控制 (~$80 BOM) | [🛒 立即选购](https://www.seeedstudio.com/Raspberry-Pi-5-8GB-p-5810.html?utm_source=github&utm_medium=readme&utm_campaign=openvoicestream&utm_content=hw_table) |

要求：Docker 加上足以容纳镜像和模型 volume 的磁盘空间。当前实测占用约为 Jetson 总计 7.5 GB、RK 3.2-4.4 GB、Raspberry Pi 5 2.8 GB。运行时内存取决于 profile：Jetson 约 1.0-2.1 GiB，RK 2.7-4.1 GiB，Raspberry Pi 上为纯 CPU。在 Jetson 上，需要 NVIDIA Container Runtime；在 Rockchip 上，必须加载主机 NPU 驱动（`rknpu`）。

## Patched sherpa-onnx

OpenVoiceStream 附带一个打过补丁的 sherpa-onnx，修复了 Paraformer 流式尾部截断问题（原版会丢掉最后 1–3 个字符）。该补丁：

1. **IsReady()** —— 在 `InputFinished()` 后强制解码剩余帧
2. **DecodeStream()** —— 对不完整的最终块进行零填充
3. **CIF force-fire** —— 在流结束时输出残余 token

补丁细节与重建说明见 [docs/known-issues/sherpa-onnx-paraformer-eof-fix.md](docs/known-issues/sherpa-onnx-paraformer-eof-fix.md)（aarch64、Python 3.10、CUDA 12.6；预构建二进制不再提交到本仓库）。

## Project Structure

> **初次接触？** 先阅读 [ARCHITECTURE.md](ARCHITECTURE.md) —— 它梳理了三个仓库（本产品 + `voxedge` 库 + `voxedge-engine`）、两个进程，以及如何在无 GPU 的情况下在本地运行整套系统。[DEVELOP.md](DEVELOP.md) 是开发机检查清单；[docs/CONFIGURATION.md](docs/CONFIGURATION.md) 涵盖 profile 和 env 变量。

SenseCraft solution 所使用的业务层见
[Agent 应用目录](agent/ovs_agent/apps/README.md)。该目录定义每个 App 的部署说明、
推荐模型、功能验收和设备实测记录契约。
[`conversation` App 文档](agent/ovs_agent/apps/conversation/README.md)记录了
`conversational_voice_ai` 使用的业务 App、各硬件部署矩阵，以及“配置中的推荐”
与“有证据的端到端实测结果”之间的边界。

```text
openvoicestream/
├── server/                  # FastAPI voice service (the product server)
│   ├── main.py              # Endpoints and startup
│   ├── core/                # VAD, ASR/TTS contracts, streaming, HF artifact download
│   └── utils/               # numpy mel + helpers
├── agent/                   # the voice agent — a SEPARATE package + container
│   └── ovs_agent/           # framework + App 业务层
│       └── apps/            # 每个 App 的文档和实现
├── voices/                  # Custom voice embeddings (auto-patched into model)
├── bench/                   # Streaming + V2V latency benchmarks (perf harness)
├── scripts/                 # Engine build, model download, diagnostics
│   └── kokoro_experiments/  # Archived Kokoro graph-surgery investigations
├── examples/                # API usage examples (TTS streaming, V2V client)
├── tests/                   # Integration and E2E tests
├── deploy/
│   ├── docker-compose.yml   # Production deploy (pre-built image)
│   ├── artifacts/           # Deployment manifests
│   └── docker/
│       ├── Dockerfile.jetson  # Jetson Orin Nano/NX/AGX (zh_en or multilingual)
│       ├── Dockerfile.rk      # Rockchip RK3576/RK3588 NPU
│       └── Dockerfile.rpi     # Raspberry Pi 4/5 (CPU)
├── configs/                 # Device profiles (Jetson, RK, RPi)
├── third_party/             # Submodules (independently maintained)
│   ├── jetson-voice-engine  # Qwen3 export + engine build for Jetson
│   └── rkvoice-stream       # Rockchip NPU streaming voice runtime
└── docs/                    # Guides, runbooks, comparison reports
```

**各引擎的 ASR/TTS 后端位于同级的 [`voxedge`](https://github.com/suharvest/voxedge) 库中**（`pip install --pre voxedge`），而非本仓库。产品的后端注册表（`server/core/asr_backend.py` / `tts_backend.py`）指向 `voxedge.backends.*`；在 Rockchip 上安装 `voxedge[rk]` 以获得 NPU 运行时。

使用 `--recurse-submodules` 克隆以拉取 `third_party/*`，或在克隆后运行 `git submodule update --init --recursive`。

### 统一的后端结构（自助复现与发布）

每个后端 —— Jetson（TensorRT-Edge-LLM）、Rockchip（RKNN）和 Raspberry Pi（sherpa-onnx）—— 都遵循 **相同的布局**，因此其中任何一个都可以在无内部知识的情况下被复现、重建和发布：

| Per-backend asset | Purpose |
|---|---|
| `recipes/` | 引擎/模型的构建 + 导出步骤（固定上游 commit，运行导出 API） |
| `HF_ARTIFACTS` | 终端用户拉取的已发布 Hugging Face bundle（例如 `harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8`） |
| `docs/`（runbook） | 该后端的部署 + 验证步骤（例如 [docs/deploy-v080-n1n2.md](docs/deploy-v080-n1n2.md)） |
| `AGENTS` | 在该后端上工作的 agent/dispatch 护栏 |

Jetson、RK 和 RPi 是 **一等同侪** —— 没有哪个是“主”后端，且相同的 `recipes → HF_ARTIFACTS → docs → AGENTS` 契约对每个后端都成立，因此任何人都可以自助完成复现或发布。

> **差异 —— fork 与自研运行时。** 唯一的结构性差异在于运行时的 *来源*：Jetson 后端的运行时扩展位于我们 **fork 的 TensorRT-Edge-LLM** 中（上游 bug 修复 + 本地运行时扩展落在 fork 里；`jetson-voice-engine` 只承载 overlay/recipes 并从中重新生成补丁）。RK 和 RPi 运行时是 **自研的**（`rkvoice-stream`、打补丁的 sherpa-onnx）。这是有意为之的归属边界，而非不一致 —— 每个后端仍暴露上述相同的 recipes/artifacts/docs/agents 表面。

## 💼 企业级方案与商业定制 (Enterprise & Commercial Deployment)

如需在智能客服亭、商超零售语音导览、会议室转写或工业语音播报中规模化部署：

- **硬件批量集采**：支持 100+ 节点规模化部署的阶梯采购价格与供货保障。
- **语音硬件 ODM/OEM 定制**：提供专属声学结构麦克风外壳丝印、载板接口客制化、预刷出厂 Golden 模型固件及 CE/FCC 全球合规认证。
- **垂直领域模型与 ASR 定制**：提供专有声学模型微调、行业热词词库注入与本地私有化部署支持。

👉 **[联系 Seeed 企业解决方案团队](https://www.seeed.cc/customization?utm_source=github&utm_medium=readme&utm_campaign=openvoicestream&utm_content=footer_enterprise)** 或访问 [智慧门店数字化解决方案页](https://www.seeed.cc/solutions/smart-retail-store?utm_source=github&utm_medium=readme&utm_campaign=openvoicestream&utm_content=footer_solution) 提交项目需求。

## Changelog

发布历史与过往里程碑见 [CHANGELOG.md](CHANGELOG.md)（英文）。
新旧实测数据见 [BENCHMARKS.md](BENCHMARKS.md) 与
[`bench/asr_bench/results/`](bench/asr_bench/results/)。

## Contributing

欢迎提交 Issue 和 PR。最有价值的贡献：

- 新的后端集成（其他 NPU、其他推理引擎）
- 在更多硬件上的流式基准测试
- 带可复现音频样本及 `LANGUAGE_MODE` / profile 信息的 bug 报告
- 文档改进，尤其是针对新设备的部署配方

如果你在进行较大的改动，请先开 Issue 以对齐方案。子项目改动（Qwen3 导出、Rockchip 运行时）应归入它们各自的仓库：[`jetson-voice-engine`](https://github.com/suharvest/jetson-voice-engine)、[`rkvoice-stream`](https://github.com/suharvest/rkvoice-stream)。

## Acknowledgements

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) —— 驱动双语 ASR 和 TTS 路径的语音推理引擎
- [next-gen Kaldi](https://github.com/k2-fsa) —— sherpa-onnx 背后的研究基础
- [Paraformer](https://github.com/modelscope/FunASR) —— 流式双语 ASR 模型
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) —— 快速的 flow-matching TTS（zh+en 模式）
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) —— 高质量英文 TTS，带 53 个说话人（en 模式）
- [Zipformer](https://github.com/k2-fsa/icefall) —— 高效的 transducer ASR（en 模式）
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) —— 多语言离线 ASR
- [Qwen3](https://huggingface.co/Qwen) —— 多语言 ASR + TTS 基础模型（52 语言路径）
- [TensorRT-EdgeLLM](https://github.com/NVIDIA/TensorRT-LLM) —— Qwen3 路径的 Jetson 推理运行时
- [RKNN Toolkit](https://github.com/rockchip-linux/rknn-toolkit2) —— RK3576/RK3588 路径的 Rockchip NPU 运行时
