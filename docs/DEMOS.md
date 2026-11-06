# Demo Assets Index / 演示资产总索引

**EN** — Everything in this repo you can put in front of an audience, in one
place: the browser Demo Gallery, code-level API examples, agent framework
examples, and bench scripts that double as live demos.

**中文** — 本仓库所有可以「拿去演示」的资产索引：浏览器 Demo Gallery、代码级
API 示例、agent 框架示例，以及可当现场演示用的 bench 脚本。

## 1. Demo Gallery（浏览器演示门户, `demos/`）

One command starts the portal on `:8700`; each demo is its own thin app.
一条命令起门户（`:8700`），每个 demo 是独立薄应用。

```bash
docker compose -f demos/docker-compose.demos.yml --profile all up -d
# open http://<device>:8700        (kiosk mode for trade shows: DEMO_KIOSK=1)
```

Local dev without Docker / 免 Docker 本地开发：
`cd demos && uv sync && uv run uvicorn gallery.backend.main:app --port 8700`。

| Card / 卡片 | Port | What it demos / 演示什么 | orin-nx (Jetson) | rk3588 (Seeed reComputer RK3588) |
|---|---|---|---|---|
| **Gallery portal** 演示门户 | 8700 | Device status, demo cards, runtime ASR/TTS model hot-switch panel, kiosk attract carousel / 设备状态、演示卡片、模型热切换面板、kiosk 轮播 | ✅ | ✅ (hot-switch: RK NPU engines report `hot_reload_not_supported` / 热切换按引擎能力置灰) |
| **asr-caption** 实时字幕 | 8701 | Streaming captions word by word, first-token latency, auto language detection / 流式字幕逐字上屏、首字延迟、语言自动检测 | ✅ | ✅ |
| **tts-playground** 语音合成体验 | 8702 | Pick voice, drag speed/pitch, streaming playback with live TTFA / 选音色、拖语速/音高、流式播放 + TTFA 大数字 | ✅ | ✅ (speed/pitch depend on backend capability / 语速音高按后端能力) |
| **v2v-chat** 语音对话 | 8703 | Full voice-to-voice chat with barge-in, per-stage ASR/LLM/TTS latency bars / 完整语音对话 + 打断，逐阶段延迟条 | ✅ (needs `OVS_V2V_SERVER_LOOP=1` etc., see prerequisites / 需服务端前置配置) | ✅ (same prerequisites / 同前置) |
| **diarization** 说话人分离 | 8704 | Multi-speaker captions colored per speaker + talk-time stats / 多人字幕按说话人上色 + 占比统计 | 🔜 coming soon (lands after `feat/diarization` merges / 待分支合并) | 🔜 pending on-device CAM++ CPU validation / 待 RK 实测 |
| **voice-clone** 声音克隆 | 8705 | Record 10 s, then TTS in your own voice / 录 10 秒用自己音色念任意文本 | ✅ (SparkTTS profiles) | ❌ greyed out — SparkTTS is a Jetson path / 置灰，SparkTTS 为 Jetson 路径 |

Deployment details, mock-SLV dev loop, and per-demo env vars:
[`demos/README.md`](../demos/README.md).

### Server prerequisites / 服务端前置条件（摘要）

Full table with rationale lives in
[`demos/README.md#slv-server-prerequisites--服务端前置条件`](../demos/README.md#slv-server-prerequisites--服务端前置条件).

| SLV env | Needed by / 谁需要 |
|---|---|
| `OVS_ADMIN_KEY=<secret>`（gallery 侧配同值 `SLV_ADMIN_KEY`） | model hot-switch panel / 模型热切换面板 |
| `OVS_V2V_SERVER_LOOP=1` + `OVS_V2V_ENGINE=voxedge` | v2v-chat spoken replies / 语音对话回复 |
| `EDGE_LLM_BASE_URL=http://172.17.0.1:8000/v1` | v2v-chat (LLM on the host, reachable from the container / 容器内可达宿主 LLM) |

asr-caption / tts-playground / voice-clone only need the SLV service itself.
实时字幕 / 合成体验 / 声音克隆只需要 SLV 服务本身。

## 2. Server API examples / 服务端 API 示例（`examples/`）

| Example | One-liner / 一句话 | Run / 入口命令 |
|---|---|---|
| [`examples/stream_tts_to_wav.py`](../examples/stream_tts_to_wav.py) | Smallest `/tts/stream` client — streams PCM and writes a playable WAV, zero third-party deps / 最小流式 TTS 客户端，零三方依赖 | `python3 examples/stream_tts_to_wav.py --url http://device:8621 --text "你好" --out /tmp/tts.wav` |
| [`examples/v2v_tts_only.py`](../examples/v2v_tts_only.py) | Unified `/v2v/stream` WS protocol in TTS-only mode — the copy-paste start for feeding LLM tokens into TTS / `/v2v/stream` 协议最小起点 | `uv run --with websockets python examples/v2v_tts_only.py --url ws://device:8621/v2v/stream --text "Hello" --out /tmp/v2v.wav` |

## 3. Agent framework examples / Agent 框架示例（`examples/agent/`）

Copy → tweak → run examples for building voice apps on `ovs_agent`
(prereqs + CLI mode: [`examples/agent/README.md`](../examples/agent/README.md)).
基于 `ovs_agent` 写语音应用的「复制即改即跑」示例。

| Example | One-liner / 一句话 | Run / 入口命令（`cd agent && uv sync` 后） |
|---|---|---|
| [`minimal_echo_app.py`](../examples/agent/minimal_echo_app.py) | Minimal app: subclass `BaseApp` + one hook, echoes the user back (no LLM) / 最小语音应用，无需 LLM | `cd agent && uv run python ../examples/agent/minimal_echo_app.py --slv-url ws://device:8621/v2v/stream` |
| [`voice_tools_app.py`](../examples/agent/voice_tools_app.py) | Voice tool-calling loop: `@tool` registration + allowlist, "现在几点" → tool → spoken answer / 语音工具调用闭环（需 LLM） | `SLV_URL=ws://device:8621/v2v/stream LLM_BASE_URL=http://device:8000/v1 uv run python ../examples/agent/voice_tools_app.py` |
| [`custom_mode_app.py`](../examples/agent/custom_mode_app.py) | Custom `AppMode` (parrot mode) in `MultiModeApp` with mode switching / 自定义模式 + 模式切换 | `cd agent && uv run python ../examples/agent/custom_mode_app.py --slv-url ws://device:8621/v2v/stream` |

All three support `--check` (offline self-check, no SLV/mic needed).
三个示例都支持 `--check` 脱机自检。

## 4. Bench scripts usable as demos / 可当演示用的 bench 脚本（`bench/perf/`）

| Script | One-liner / 一句话 | Run / 入口命令 |
|---|---|---|
| [`smoke_tts_multiturn.py`](../bench/perf/smoke_tts_multiturn.py) | Multi-turn TTS lifecycle on one `/v2v/stream` WS — shows turn-after-turn stability live / 单 WS 多轮 TTS 生命周期冒烟 | `python3 bench/perf/smoke_tts_multiturn.py --host device:8621` |
| [`v2v_concurrency_probe.py`](../bench/perf/v2v_concurrency_probe.py) | N concurrent `/v2v/stream` ASR sessions with per-session timelines — the "N=2 no cross-talk" showpiece / N 路并发 ASR 演示 | `python3 bench/perf/v2v_concurrency_probe.py --url ws://device:8621 --wav clip.wav --n 2` |
| [`multi_sentence_pipeline.py`](../bench/perf/multi_sentence_pipeline.py) | Multi-sentence `/tts/stream` request showing TTFA + per-sentence pipeline overlap / 多句 TTS 流水线并行与 TTFA 演示 | `python3 bench/perf/multi_sentence_pipeline.py --host device:8621` |

## 5. Single-file legacy demo / 单文件历史演示

| Asset | Note |
|---|---|
| [`docs/asr-realtime-demo.html`](asr-realtime-demo.html) | Original single-file realtime ASR page. Superseded by the gallery's **asr-caption** card, but kept as a zero-backend, open-the-file-in-a-browser version / 已被 asr-caption 卡收编，保留为「浏览器直接打开」的零后端单文件版 |
