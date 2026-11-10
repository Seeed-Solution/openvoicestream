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
> - **中文对话**：Qwen3-ASR + Matcha-TTS（极快、超低显存、自然度高）
> - **英文对话**：所有设备统一主推 **Kokoro**（原生 53 款音色，音质纯正自然；RK3588/RK3576 全面支持）
> - **越南语对话**：Qwen3-ASR + **Kokoro-VI**（前缀 TRT 17ms，端到端 0.703 RTF，首词 100% 完整留存）
> - **声音克隆 / 52 语多语言**：统一使用 **Qwen3-ASR + Qwen3-TTS**（多语种流式与声音克隆）
> - **端侧大模型思考**：Orin NX 16GB 支持端侧 **Qwen3.5-4B** 离线闭环（~35 tok/s，V2V p50 162 ms）

<p align="center">
  <b>快速切换设备卡片：</b><br>
  <a href="#1-jetson-orin-nano-8gb--推荐主流边缘方案recomputer-j4012--j401">🟢 Jetson Orin Nano (8GB)</a> &nbsp;|&nbsp;
  <a href="#2-jetson-orin-nx-16gb--端侧大模型闭环旗舰recomputer-j4012--j4011--super">🔵 Jetson Orin NX (16GB)</a> &nbsp;|&nbsp;
  <a href="#3-recomputer-rk3588--高并发与性价比标杆recomputer-r1000--工业边缘主机">🟠 reComputer RK3588</a> &nbsp;|&nbsp;
  <a href="#4-recomputer-rk3576--入门级低成本边缘网关lubancat-3--紧凑型网关">🟡 reComputer RK3576</a> &nbsp;|&nbsp;
  <a href="#5-树莓派-5-raspberry-pi-5--纯-cpu-与轻量边缘树莓派-5--recomputer-r1000-pi">🟣 树莓派 5 (Raspberry Pi 5)</a>
  <br>
  <sub>也可在浏览器中打开 <a href="docs/device-language-matrix.html">交互式 Tab 选型网页</a> 体验点击切换</sub>
</p>

<details open>
<summary><b>🟢 1. Jetson Orin Nano (8GB) — 推荐主流边缘方案（reComputer J4012 / J401）</b></summary>
<br>

> **硬件规格**：40 TOPS 算力，8GB 统一内存，7-15W 功耗。

| 目标语言 | 场景定位 | ASR 语音识别 | 端侧思考 / LM | TTS 语音合成 | 实测性能 (ASR / TTS / V2V) | 推荐 Profile 预设 | 状态 |
|---|---|---|---|---|---|---|:---:|
| **🇻🇳 越南语 (vi)** | 海外智能客服 / 陪伴音箱 | Qwen3-ASR 0.6B (TRT int4) | 云端 API / 1.5B (本地) | Kokoro-VI (TRT Hybrid) | **ASR: 366.3 ms**<br>**TRT前缀: 17.0 ms · RTF: 0.703**<br>**首词保留 100%（0截断）** | `jetson-qwen3asr-kokoro-vi` | 🟢 实测 |
| **🇨🇳 中文 (zh)** | 智能终端 / 工业看板 / 机器人 | Qwen3-ASR 0.6B (TRT int4) | 云端百炼 / 本地 1.5B | Matcha-TTS (TRT fp16) | **ASR CER: 0**<br>**TTS RTF: 0.15 · V2V: ~500 ms**<br>显存占用仅 ~1.8 GB | `jetson-qwen3asr-matcha` | 🟢 实测 |
| **🇺🇸 英语 (en)** | 出海商业语音交互 (53款音色) | Paraformer / Whisper (TRT) | 云端 API / 本地模型 | **Kokoro-TRT (53 原生音色)** | **ASR TTFT: ~50 ms**<br>**TTS TTFT: ~130 ms · RTF: 0.25** | `jetson-paraformer-kokoro` | 🟢 实测 |
| **🌐 多语种 (52语)** | 跨国语音服务 / 声音克隆 | Qwen3-ASR 0.6B (TRT 52语) | 云端多语言大模型 | **Qwen3-TTS CustomVoice (int4)** | **ASR RTF: 0.30**<br>**TTS RTF: 0.50 · V2V: ~1.5 s**<br>支持 3~10s 声音克隆 | `jetson-multilang-highperf` | 🟢 实测 |
| **🎧 离线转写** | 会议记录 / 录音转文字 | Whisper-base (TRT bf16) | — (纯 ASR 场景) | — | **WER: 7.62%**<br>**TTFT: 18 ms · RTF: 0.008** | `orin-whisper` | 🟢 实测 |

</details>

<details open>
<summary><b>🔵 2. Jetson Orin NX (16GB) — 端侧大模型闭环旗舰（reComputer J4012 / J4011 / Super）</b></summary>
<br>

> **硬件规格**：100 TOPS 算力，16GB 统一内存，10-25W 功耗。

| 目标语言 | 场景定位 | ASR 语音识别 | 端侧思考 / LM | TTS 语音合成 | 实测性能 (ASR / TTS / V2V) | 推荐 Profile 预设 | 状态 |
|---|---|---|---|---|---|---|:---:|
| **🇨🇳 中文全离线** | 军工 / 政务 / 医院 / 离网机器人 | Qwen3-ASR 0.6B (TRT int4) | **Qwen3.5-4B-Instruct**<br>(TRT W4A16, ~35 tok/s, 7.8GB) | Matcha-TTS (TRT bf16) | **V2V p50: 162 ms**<br>**ASR CER: 0 · 显存占用 10GB**<br>本地端侧大模型同卡并行 | `jetson-edgellm-v091-matcha` | 🟢 实测 |
| **🎙️ 声音克隆 (多语)** | IP 形象定制 / 声音克隆 | Qwen3-ASR 0.6B (TRT int4) | Qwen3.5-4B (本地) / 云端 | **Qwen3-TTS CustomVoice / Base** | **TTS RTF: 0.33**<br>**ASR RTF: 0.205 · V2V: ~1.0 s**<br>10 路并发 GPU 98% 满载 | `jetson-multilang-highperf-nx` | 🟢 实测 |
| **🇺🇸 英语 (en) 优选** | 出海商业语音交互 (53款音色) | Qwen3-ASR / Paraformer (TRT) | Qwen3.5-4B (本地) / 云端 | **Kokoro-TRT (53 原生音色)** | **TTS RTF: 0.18**<br>**ASR TTFT: 18 ms · WER: 7.62%** | `jetson-paraformer-kokoro` | 🟢 实测 |
| **🌐 多语种 (52语)** | 高并发多语种接待大厅 | Qwen3-ASR 0.6B (TRT 52语) | Qwen3.5-4B (本地) / 云端 | **Qwen3-TTS CustomVoice (bf16)** | **ASR RTF: 0.205**<br>**TTS RTF: 0.33 · 52语流式** | `jetson-multilang-highperf-nx` | 🟢 实测 |

</details>

<details open>
<summary><b>🟠 3. reComputer RK3588 — 高并发与性价比标杆（reComputer R1000 / 工业边缘主机）</b></summary>
<br>

> **硬件规格**：6 TOPS NPU，8GB/16GB 内存，8-15W 功耗。

| 目标语言 | 场景定位 | ASR 语音识别 | 端侧思考 / LM | TTS 语音合成 | 实测性能 (ASR / TTS / V2V) | 推荐 Profile 预设 | 状态 |
|---|---|---|---|---|---|---|:---:|
| **🇨🇳 中文 (zh)** | 楼宇对讲 / 智能工位 / 巡检终端 | Qwen3-ASR (RKNN W8A8) | RKLLM Qwen2.5-3B (~15 tok/s) / 云端 | Matcha-TTS (RKNN 专版) | **TTS RTF: 0.05 (极快)**<br>**ASR RTF: 0.28 · V2V p50: 528 ms**<br>三核 NPU 独立流水线 | `rk3588-default` | 🟢 实测 |
| **🇺🇸 英语 (en) 优选** | 出海智能硬件导出 | Qwen3-ASR (RKNN W8A8) | 云端 API / 轻量本地 | **Kokoro-RKNN (34% NPU 卸载)** | **TTS RTF: 0.34**<br>**ASR RTF: 0.35** | `rk3588-kokoro-rknn-34pct` | 🟢 实测 |
| **📊 高并发转写** | 呼叫中心录音质检 / 庭审转录 | SenseVoice-Small (RKNN) | — (纯转录) | — | **12 路并发 0 错误**<br>**CER: 5.13% · 三核 NPU 100% 满载** | `rk3588-sensevoice` | 🟢 实测 |
| **🇪🇺 欧系小语种** | 德 / 法 / 西语等业务终端 | Qwen3-ASR (52语 RKNN) | 云端多语言模型 | Piper TTS (CPU) | **ASR RTF: ~0.35**<br>**TTS RTF: ~0.60** | `rk3588-piper` | 🔵 支持 |

</details>

<details open>
<summary><b>🟡 4. reComputer RK3576 — 入门级低成本边缘网关（LubanCat-3 / 紧凑型网关）</b></summary>
<br>

> **硬件规格**：6 TOPS NPU，4GB/8GB 内存，4-8W 超低功耗。

| 目标语言 | 场景定位 | ASR 语音识别 | 端侧思考 / LM | TTS 语音合成 | 实测性能 (ASR / TTS / V2V) | 推荐 Profile 预设 | 状态 |
|---|---|---|---|---|---|---|:---:|
| **🇨🇳 中文 (zh)** | 智慧养老音箱 / 仓储捡货终端 | Qwen3-ASR 0.6B (RKNN W8A8) | 云端 API / 1.5B (轻量) | Matcha-TTS (RKNN) | **TTS RTF: 0.12**<br>**ASR RTF: 0.44 · V2V p50: 1020 ms** | `rk3576-default` | 🟢 实测 |
| **🇺🇸 英语 (en) 优选** | 轻量化英语对话与提示 | Qwen3-ASR 0.6B (RKNN W8A8) | 云端 API / 1.5B (轻量) | **Kokoro (RKNN/CPU 混合)** | **ASR RTF: 0.44**<br>**TTS RTF: ~0.65 · 音质自然** | `rk3576-kokoro-long32-canary` | 🟢 实测 |
| **📊 多路音频转写** | 工业多通道音频监控 | SenseVoice-Small (RKNN) | — | — | **16 路并发 0 错误**<br>极低成本多路转录首选 | `rk3576-sensevoice` | 🟢 实测 |
| **🇪🇺 小语种覆盖** | 轻量化多语言提示 | Qwen3-ASR (RKNN) | 云端 API | Piper TTS (CPU) | **ASR RTF: ~0.44**<br>**TTS RTF: ~0.80** | `rk3576-piper` | 🔵 支持 |

</details>

<details open>
<summary><b>🟣 5. 树莓派 5 (Raspberry Pi 5) — 纯 CPU 与轻量边缘（树莓派 5 / reComputer R1000-Pi）</b></summary>
<br>

> **硬件规格**：4核 A76 CPU，可选配 Hailo-8 M.2 扩展卡 (26 TOPS)。

| 目标语言 | 场景定位 | ASR 语音识别 | 端侧思考 / LM | TTS 语音合成 | 实测性能 (ASR / TTS / V2V) | 推荐 Profile 预设 | 状态 |
|---|---|---|---|---|---|---|:---:|
| **🇨🇳 中文 / 🇺🇸 英文** | 创客制作 / 基础语音助手 | Sherpa-ONNX 流式 (CPU) | 云端百炼 / OpenAI | Sherpa-ONNX (CPU) | **ASR RTF: ~0.50**<br>**TTS RTF: ~0.60 · V2V: ~1.5 s** | `rpi4-default` | 🟢 实测 |
| **📊 纯 CPU 转写** | 8路并发离线指令转录 | SenseVoice (ORT CPU) | — | — | **8 路并发 0 错误**<br>RTF ~0.5–0.7 | `rpi4-sensevoice` | 🟢 实测 |
| **⚡ 英文 (+Hailo-8)** | 工业低功耗英文语音识别 | Whisper-base (Hailo-8 硬件编码器) | 云端 API | Sherpa / Piper | **WER: 8.39%**<br>节省 >70% CPU 占用 | `rpi5-hailo-whisper` | 🟢 实测 |

</details>

完整技术选型文档参见 [docs/DEVICE-LANGUAGE-MATRIX.md](docs/DEVICE-LANGUAGE-MATRIX.md) 与 [docs/RECOMMENDED-MODELS.md](docs/RECOMMENDED-MODELS.md)。

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
| **🇨🇳 中文极速对话** | Qwen3-ASR (INT4/W8A8) | **Matcha-TTS** (FP16/RKNN) | 全部设备 | **TTS RTF 0.018–0.075**，首包 <10ms，超低显存 |
| **🇺🇸 英文纯正发音** | Paraformer / Qwen3-ASR | **Kokoro** (TRT / RKNN) | 全部设备 | **53 款原生纯正发音**，Orin RTF 0.18，RK3588 0.34 |
| **🇻🇳 越南语出海对话** | Qwen3-ASR (TRT INT4) | **Kokoro-VI** (TRT Hybrid) | Jetson Orin | 前缀 17ms，端到端 RTF 0.703，首词 100% 留存 |
| **🌐 多语种 / 声音克隆** | Qwen3-ASR (52 语) | **Qwen3-TTS** (INT4/CustomVoice) | Jetson Orin | 52 语流式合成，支持 3~10 秒极速克隆 |
| **📊 高并发离线转写** | SenseVoice-Small / Whisper | — | RK3588 / RPi / Jetson | RK3588 12 路 100% 满载 0 错误；Whisper WER 7.62% |
| **🇪🇺 欧系小语种** | Qwen3-ASR (52 语) | **Piper** (CPU) | 全部设备 | 德/法/西/俄等离线轻量级发音 |

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

async def transcribe():
    uri = "ws://localhost:8621/asr/stream?sample_rate=16000&language=auto"
    async with websockets.connect(uri) as ws:
        for chunk in audio_chunks:  # 传入 int16 格式 16kHz 音频分块
            await ws.send(chunk.tobytes())
            result = await ws.recv()
            print("[Partial]", json.loads(result)["text"])
        await ws.send(b"")  # 发送空字节触发最终收敛
        final = await ws.recv()
        print("[Final]", json.loads(final)["text"])

asyncio.run(transcribe())
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
