# 边缘端多语言语音方案选型与性能基准矩阵

> **在线交互版本**：可直接在浏览器中打开 [`docs/device-language-matrix.html`](device-language-matrix.html) 体验 Tab 切换。
> 统一服务接口：所有预设共享相同的 API（`/v2v/stream`、`/asr/stream`、`/tts`），切换预设无需重构应用。

---

## 1. Jetson Orin Nano (8GB) — 推荐主流边缘方案

- **硬件特性**：40 TOPS 算力，8GB 统一显存，7-15W 功耗（代表产品：`reComputer J4012 / J401`）
- **核心定位**：极速响应、深度 TensorRT 加速、性价比最高的边缘交互核心。

| 目标语言 | 场景定位 | ASR 语音识别 | 端侧思考 / LLM | TTS 语音合成 | 实测性能 (ASR / TTS / V2V) | 推荐 Profile 预设 | 验证状态 |
|---|---|---|---|---|---|---|---|
| **🇺🇸 英语 (en)** | 出海商业语音交互 (53款音色) | Paraformer / Whisper-base (TRT bf16) | 云端 API / 本地模型 | Kokoro-TRT (53 原生音色) | **ASR TTFT: ~50 ms**<br>**TTS TTFT: ~130 ms · RTF: 0.25** | `jetson-paraformer-kokoro` | 🟢 实测验证 |
| **🇨🇳 中文 (zh)** | 智能终端 / 工业看板 / 机器人 | Qwen3-ASR 0.6B (TRT int4) | 云端百炼 / 本地 1.5B | Matcha-TTS (TRT fp16) | **ASR CER: 0**<br>**TTS RTF: 0.15 · V2V: ~500 ms**<br>显存占用仅 ~1.8 GB | `jetson-qwen3asr-matcha` | 🟢 实测验证 |
| **🌐 多语种 (52语)** | 跨国语音服务 / 声音克隆 | Qwen3-ASR 0.6B (TRT 52语) | 云端多语言大模型 | Qwen3-TTS CustomVoice (TRT int4) | **ASR RTF: 0.30**<br>**TTS RTF: 0.50 · V2V: ~1.5 s**<br>支持 3~10s 声音克隆 | `jetson-multilang-highperf` | 🟢 实测验证 |
| **🇻🇳 越南语 (vi)** | 海外智能客服 / 陪伴音箱 | Qwen3-ASR 0.6B (TRT int4) | 云端 API / 1.5B (本地) | Kokoro-VI (TRT Hybrid) | **ASR: 366.3 ms**<br>**TRT前缀: 17.0 ms · RTF: 0.703**<br>**首词保留: 100% (0丢词)** | `jetson-qwen3asr-kokoro-vi` | 🟢 实测验证 |
| **🎧 离线转写** | 会议记录 / 录音转文字 | Whisper-base (TRT bf16) | — (纯 ASR 场景) | — | **WER: 7.62%**<br>**TTFT: 18 ms · RTF: 0.008** | `orin-whisper` | 🟢 实测验证 |

---

## 2. Jetson Orin NX (16GB) — 端侧大模型闭环旗舰

- **硬件特性**：100 TOPS 算力，16GB 统一显存，10-25W 功耗（代表产品：`reComputer J4012 / J4011 / Super`）
- **核心定位**：完全离线端侧 4B 大模型闭环、多路高并发、极高音质声音克隆。

| 目标语言 | 场景定位 | ASR 语音识别 | 端侧思考 / LLM | TTS 语音合成 | 实测性能 (ASR / TTS / V2V) | 推荐 Profile 预设 | 验证状态 |
|---|---|---|---|---|---|---|---|
| **🇺🇸 英语 (en) 优选** | 出海商业语音交互 (53款音色) | Qwen3-ASR / Paraformer (TRT) | Qwen3.5-4B (本地) / 云端 | Kokoro-TRT (53 原生音色) | **TTS RTF: 0.18**<br>**ASR TTFT: 18 ms · WER: 7.62%** | `jetson-paraformer-kokoro` | 🟢 实测验证 |
| **🇨🇳 中文全离线** | 军工 / 政务 / 医院 / 离网机器人 | Qwen3-ASR 0.6B (TRT int4) | **Qwen3.5-4B-Instruct**<br>(TRT W4A16, ~35 tok/s, 7.8GB) | Matcha-TTS (TRT bf16) | **V2V p50: 162 ms**<br>**ASR CER: 0 · 显存占用 10GB**<br>端侧大模型同卡并行 | `jetson-edgellm-v091-matcha` | 🟢 实测验证 |
| **🌐 多语种 (52语)** | 高并发多语种接待大厅 / 声音克隆 | Qwen3-ASR 0.6B (TRT 52语) | Qwen3.5-4B (本地) / 云端 | Qwen3-TTS CustomVoice / Base (TRT) | **TTS RTF: 0.33**<br>**ASR RTF: 0.205 · V2V: ~1.0 s**<br>10 路并发 GPU 98% 满载 | `jetson-multilang-highperf-nx` | 🟢 实测验证 |
| **🇻🇳 越南语 (vi)** | 海外智能客服 / 陪伴音箱 | Qwen3-ASR 0.6B (TRT int4) | Qwen3.5-4B (本地) / 云端 | Kokoro-VI (TRT Hybrid) | **ASR: 366.3 ms**<br>**TRT前缀: 17.0 ms · RTF: 0.703**<br>**首词保留: 100% (0丢词)** | `jetson-qwen3asr-kokoro-vi` | 🟢 实测验证 |
| **🎧 离线转写** | 会议记录 / 录音转文字 | Whisper-base (TRT bf16) | — (纯 ASR 场景) | — | **WER: 7.62%**<br>**TTFT: 18 ms · RTF: 0.008** | `orin-whisper` | 🟢 实测验证 |

---

## 3. reComputer RK3588 — 高并发与性价比标杆

- **硬件特性**：6 TOPS NPU，8GB/16GB LPDDR4x，8-15W 功耗（代表产品：`reComputer R1000 / 工业边缘主机`）
- **核心定位**：三核 NPU 独立调度、工业级宽温无风扇、极致高并发转写。

| 目标语言 | 场景定位 | ASR 语音识别 | 端侧思考 / LLM | TTS 语音合成 | 实测性能 (ASR / TTS / V2V) | 推荐 Profile 预设 | 验证状态 |
|---|---|---|---|---|---|---|---|
| **🇺🇸 英文 (en) 优选** | 出海智能硬件导出 | Qwen3-ASR (RKNN W8A8) | 云端 API / 轻量本地 | Kokoro-RKNN (34% NPU 卸载) | **TTS RTF: 0.34**<br>**ASR RTF: 0.35** | `rk3588-kokoro-rknn-34pct` | 🟢 实测验证 |
| **🇨🇳 中文 (zh)** | 楼宇对讲 / 智能工位 / 巡检终端 | Qwen3-ASR (RKNN W8A8) | RKLLM Qwen2.5-3B (~15 tok/s) / 云端 | Matcha-TTS (RKNN 专版) | **TTS RTF: 0.05 (极快)**<br>**ASR RTF: 0.28 · V2V p50: 528 ms**<br>三核 NPU 独立流水线 | `rk3588-default` | 🟢 实测验证 |
| **🇪🇺 欧系小语种** | 德 / 法 / 西语等业务终端 | Qwen3-ASR (52语 RKNN) | 云端多语言模型 | Piper TTS (CPU) | **ASR RTF: ~0.35**<br>**TTS RTF: ~0.60** | `rk3588-piper` | 🔵 架构支持 |
| **📊 高并发转写** | 呼叫中心录音质检 / 庭审转录 | SenseVoice-Small (RKNN) | — (纯转录) | — | **12 路并发 0 错误**<br>**CER: 5.13% · 三核 NPU 100% 满载** | `rk3588-sensevoice` | 🟢 实测验证 |

---

## 4. reComputer RK3576 — 入门级低成本边缘网关

- **硬件特性**：6 TOPS NPU，4GB/8GB 内存，4-8W 超低功耗（代表产品：`LubanCat-3 / 紧凑型网关`）
- **核心定位**：极低功耗与 BOM 成本、单路全双工千问+Matcha 稳定交付。

| 目标语言 | 场景定位 | ASR 语音识别 | 端侧思考 / LLM | TTS 语音合成 | 实测性能 (ASR / TTS / V2V) | 推荐 Profile 预设 | 验证状态 |
|---|---|---|---|---|---|---|---|
| **🇺🇸 英语 (en) 优选** | 轻量化英语对话与提示 | Qwen3-ASR 0.6B (RKNN W8A8) | 云端 API / 1.5B (轻量) | Kokoro (RKNN/CPU 混合) | **ASR RTF: 0.44**<br>**TTS RTF: ~0.65 · 音质自然** | `rk3576-kokoro-long32-canary` | 🟢 实测验证 |
| **🇨🇳 中文 (zh)** | 智慧养老音箱 / 仓储捡货终端 | Qwen3-ASR 0.6B (RKNN W8A8) | 云端 API / 1.5B (轻量) | Matcha-TTS (RKNN) | **TTS RTF: 0.12**<br>**ASR RTF: 0.44 · V2V p50: 1020 ms** | `rk3576-default` | 🟢 实测验证 |
| **🇪🇺 小语种覆盖** | 轻量化多语言提示 | Qwen3-ASR (RKNN) | 云端 API | Piper TTS (CPU) | **ASR RTF: ~0.44**<br>**TTS RTF: ~0.80** | `rk3576-piper` | 🔵 支持 |
| **📊 多路音频转写** | 工业多通道音频监控 | SenseVoice-Small (RKNN) | — | — | **16 路并发 0 错误**<br>极低成本多路转录首选 | `rk3576-sensevoice` | 🟢 实测验证 |
| **🇪🇺 小语种覆盖** | 轻量化多语言提示 | Qwen3-ASR (RKNN) | 云端 API | Piper TTS (CPU) | **ASR RTF: ~0.44**<br>**TTS RTF: ~0.80** | `rk3576-piper` | 🔵 架构支持 |

---

## 5. Raspberry Pi 5 — 树莓派纯 CPU 与轻量边缘

- **硬件特性**：4核 A76 CPU，可选配 Hailo-8 M.2 扩展卡 (26 TOPS)（代表产品：`树莓派 5 / reComputer R1000-Pi`）
- **核心定位**：开源创客生态、零 NPU 纯 CPU 轻量化运行。

| 目标语言 | 场景定位 | ASR 语音识别 | 端侧思考 / LLM | TTS 语音合成 | 实测性能 (ASR / TTS / V2V) | 推荐 Profile 预设 | 验证状态 |
|---|---|---|---|---|---|---|---|
| **🇨🇳 中文 / 🇺🇸 英文** | 创客制作 / 基础语音助手 | Sherpa-ONNX 流式 (CPU) | 云端百炼 / OpenAI | Sherpa-ONNX (CPU) | **ASR RTF: ~0.50**<br>**TTS RTF: ~0.60 · V2V: ~1.5 s** | `rpi4-default` | 🟢 实测验证 |
| **📊 纯 CPU 转写** | 8路并发离线指令转录 | SenseVoice (ORT CPU) | — | — | **8 路并发 0 错误**<br>RTF ~0.5–0.7 | `rpi4-sensevoice` | 🟢 实测验证 |
| **⚡ 英文 (+Hailo-8)** | 工业低功耗英文语音识别 | Whisper-base (Hailo-8 硬件编码器) | 云端 API | Sherpa / Piper | **WER: 8.39%**<br>节省 >70% CPU 占用 | `rpi5-hailo-whisper` | 🟢 实测验证 |
