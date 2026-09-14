# SLV Demo Gallery / 演示门户

**EN** — Browser-based demo portal for seeed-local-voice. Opens on port `:8700`,
shows what the device can do (live device status + demo cards), and hot-swaps
ASR/TTS model profiles at runtime through the SLV admin API. Each demo is a thin
FastAPI backend + build-free static frontend; the admin key never reaches the
browser.

**中文** — seeed-local-voice 的浏览器演示门户。打开 `:8700` 即可看到设备状态与
演示卡片，并可通过 SLV admin API 运行时热切换 ASR/TTS 模型组合。每个 demo 都是
「薄 FastAPI backend + 无构建纯静态前端」，admin 密钥只存在于 demo 后端，浏览器
永远拿不到。

## Quick start / 快速启动

Compose（推荐 / recommended）:

```bash
docker compose -f demos/docker-compose.demos.yml --profile gallery up -d
# open http://<device>:8700
```

Local dev without Docker（本地开发免 Docker）:

```bash
cd demos
uv sync
uv run uvicorn gallery.backend.main:app --host 0.0.0.0 --port 8700
```

Tests / 测试:

```bash
cd demos && uv run pytest tests/ -q
```

Need a fake SLV while developing? / 开发时可起假 SLV：

```bash
uv run python tests/mock_slv.py     # 127.0.0.1:8629, admin key "test-key"
SLV_URL=http://127.0.0.1:8629 SLV_ADMIN_KEY=test-key \
  uv run uvicorn gallery.backend.main:app --port 8700
```

## SLV server prerequisites / 服务端前置条件

The demos talk to a running SLV server. For the **full** experience the SLV
container needs (真机验证得出的完整配置，orin-nx 2026-07-02)：

| SLV env | Needed by | Notes |
|---|---|---|
| `OVS_ADMIN_KEY=<secret>` | model hot-switch panel | Without it, admin routes 403 for any non-loopback caller — and with bridge networking even host-side callers are non-loopback from the container's view. Set it and give the same value to the gallery as `SLV_ADMIN_KEY`. |
| `OVS_V2V_SERVER_LOOP=1` + `OVS_V2V_ENGINE=voxedge` | v2v-chat spoken replies | Default off = ASR-only pass-through; the chat card then shows a hint instead of answering. |
| `EDGE_LLM_BASE_URL=http://172.17.0.1:8000/v1` | v2v-chat spoken replies | The code default `127.0.0.1:8000/v1` points at SLV itself inside the container. `172.17.0.1` (docker0) reaches an LLM service published on the host. |
| `QWEN3_SPEAKER_ENCODER=/path/speaker_encoder.onnx` | voice-clone live enroll | Enables the torch-less CPU-ONNX enrollment path (`/tts/voices/enroll`). Without it, `supports_voice_enrollment` is false and enroll falls back to torch → 501 on Jetson. Also requires a TTS profile that consumes external embeddings (the **Base** profile `jetson-edgellm-v090-qwen3ttsbase`, not CustomVoice's named speakers). |
| `DIAR_CAMPPLUS_ENGINE_FILE=/path/campplus_fp16.plan` (Jetson) + `OVS_DIARIZE=1` | diarization speaker colouring | Configures the CAM++ speaker-embedding backend so `/asr/stream?diarize=true` finals carry a `speaker` field. On Rockchip the CAM++ ONNX auto-downloads (sherpa CPU); on Jetson point this at a TRT `.plan`. Without it the card renders captions but no colouring. |

**Two voice-cloning capability signals** (`/tts/capabilities`): `supports_voice_cloning` = can synthesize with an *already-enrolled* voice; `supports_voice_enrollment` = can enroll on-device from a 10s recording (needs the ONNX speaker encoder above). The voice-clone card's "record → clone" flow needs the latter. asr-caption / tts-playground only need the SLV service itself.

Verified end-to-end on a v0.9.0 Base-profile SLV (orin-nx 2026-07-04): enroll → `method: onnx_speaker_encoder` → clone TTFA 65 ms → exact ASR readback. Deployment recipe is a thin overlay (`FROM <v090 image>` + force-reinstall the voxedge wheel + the Base profile + the two engines above); note the Base tokenizer dir also needs `processed_chat_template.json`, not just `tokenizer.json`.

**v2v-chat endpointing (VAD) / 断句检测器只能有一个**: the chat page sends
`vad: "none"` (via `FULL_CHAT_SESSION` in `common/frontend/v2v-client.js`) —
utterance boundaries are owned solely by the ASR backend's internal VAD (RK
Qwen3 true-streaming declares `prefer_backend_endpoint_vad`). Never point the
page at a server_vad configuration: two endpoint detectors race and silently
truncate transcripts mid-sentence, no error on either side
(docs/CONFIGURATION.md "Streaming ASR endpointing: pick exactly one detector";
same rule the production ovs-agent configs follow with `vad: "none"` +
client-side VAD). Only backends WITHOUT internal endpointing may use
`vad: "silero"` — then the server VAD is the single detector. Trade-off of
`vad: "none"`: no voice barge-in during playback; use the interrupt button or
half-duplex. Server side, `OVS_V2V_SINGLE_ENDPOINT_STRICT=1` turns this
misconfiguration into a hard session reject instead of a warning log.

## Environment / 环境变量

| Variable | Default | Description |
|---|---|---|
| `SLV_URL` | `http://127.0.0.1:8621` | SLV server base URL / SLV 服务地址 |
| `SLV_ADMIN_KEY` | *(empty)* | Forwarded as `X-Admin-Key` on admin calls; optional on loopback / 管理接口密钥，环回部署可省略 |
| `DEMO_PROFILES_DIR` | `<repo>/configs/profiles` | Profile JSONs offered in the switch panel / 切换面板可选的 profile 目录 |
| `DEMO_KIOSK` | `0` | Truthy = kiosk mode for trade shows: hides debug details **and** enables the fullscreen attract carousel on the portal / 展会 kiosk 模式：隐藏调试信息，并启用门户全屏轮播 attract 画面 |
| `PORT` | `8700` | Gallery listen port (when run as `python main.py`) |

### Kiosk attract carousel / 展会轮播画面

With `DEMO_KIOSK=1`, the gallery portal enters a fullscreen "attract" carousel
after **60 s** of no interaction: one slide per available demo capability
(big headline, e.g. "实时字幕 · 全程本机推理") plus a latency selling-point
slide, rotating every **8 s**. Any touch/click/key exits back to the portal.
The timings are frontend constants — override per browser session with URL
params for testing: `http://<device>:8700/?kiosk_idle_s=5&kiosk_slide_s=3`.

`DEMO_KIOSK=1` 时，门户空闲 **60 秒**后进入全屏轮播 attract 画面：每张可用
demo 能力一屏大字标语（如「实时字幕 · 全程本机推理」）+ 一屏延迟卖点大数字，
每 **8 秒**切换一屏；触摸/点击/按键即退出回门户。时长为前端常量，测试时可用
URL 参数覆盖：`?kiosk_idle_s=5&kiosk_slide_s=3`。非 kiosk 部署完全不受影响。

## Layout / 目录

```
demos/
  common/backend/slv_proxy.py    # shared SLV probe + admin proxy
  common/frontend/ui.css|ui.js   # design tokens + shared components
  common/frontend/slv-client.js  # browser SDK (ASR WS, TTS stream player; mic in P1)
  gallery/                       # portal app (:8700)
  registry.json                  # demo catalog (5 cards land in P1/P2)
  docker-compose.demos.yml
  tests/                         # pytest + controllable mock SLV
```
