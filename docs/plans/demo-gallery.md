# Demo Gallery 设计与实施计划

状态：待确认（2026-07-02 起草）
分支策略：独立分支 `feat/demo-gallery`，不与 `feat/diarization` 混合；diarization demo 卡在该分支合并后再接入。

## 1. 目标

面向开发者/展会观众的演示门户：浏览器打开即用，能够

- **切换 demo**：每个能力一个独立演示应用；
- **切换 backend/模型**：基于服务端既有 `/admin/backend/reload`（带 drain+回滚的运行时热切换，`server/main.py:5083`）按 profile 切 ASR/TTS；
- **看到实时效果和延迟指标**（TTFA、ASR 首字延迟、V2V 逐阶段延迟）。

## 2. 架构（已确认形态）

SLV server（`:8621`）保持为**通用 backend**，只提供稳定 API，不内置 gallery。
每个 demo 是**独立应用**（薄 backend + 纯静态 frontend），独立部署；gallery 是门户应用，负责索引、设备状态、模型切换中心。

```
浏览器
  │
  ├─ gallery 门户 (:8700)  ── /api/catalog /api/switch /api/status
  │       │                       │ (代理，admin token 不出服务端)
  │       └─ 卡片链接到各 demo 应用
  │
  ├─ asr-caption    (:8701) ─┐
  ├─ tts-playground (:8702)  │  各自薄 FastAPI backend
  ├─ v2v-chat       (:8703)  ├─→ SLV server :8621
  ├─ diarization    (:8704)  │   /asr/stream /tts/stream /v2v/stream
  └─ voice-clone    (:8705) ─┘   /diarize /tts/clone/* /admin/*
```

demo 薄 backend 的职责（三件事，保持薄）：
1. 托管本 demo 静态前端（同源，无 CORS）；
2. 代理需要 admin token 的调用（模型切换、runtime 覆盖），token 只存在于 demo 容器 env；
3. demo 特有的服务端逻辑（如 diarization 的音频缓存、clone 的 enroll 流程编排）。

前端技术栈：**无构建纯静态**（原生 JS + 共享 CSS），与现有 `docs/asr-realtime-demo.html`、debug dashboard 一致，双语 UI。

## 3. 目录结构

```
demos/
  README.md                    # 总览 + 部署说明（双语）
  docker-compose.demos.yml     # 单 compose，profiles 控制启用子集
  common/
    frontend/
      slv-client.js            # 浏览器 SLV SDK：AudioWorklet 采麦克风重采样16k、
                               #   WS ASR/V2V 客户端、流式 TTS 播放器
      ui.css / ui.js           # 共享组件：延迟条、状态 pill、模型切换面板
    backend/
      slv_proxy.py             # 共享 Python 模块：SLV 探测、admin 代理、能力聚合
  gallery/                     # 门户
    backend/main.py            # /api/catalog /api/switch /api/status
    frontend/index.html
    demo.json                  # 元数据（下述注册机制）
    Dockerfile
  asr-caption/                 # demo 1（收编 docs/asr-realtime-demo.html）
  tts-playground/              # demo 2
  v2v-chat/                    # demo 3
  diarization/                 # demo 4
  voice-clone/                 # demo 5
```

**注册机制**：每个 demo 目录放 `demo.json`（名称双语、描述、端口、依赖能力如 `needs: ["tts.clone"]`）。gallery 启动时读 compose 网络内各应用的 `demo.json` + `/healthz`，结合 SLV `/asr/capabilities`、`/tts/capabilities`、`/admin/backend/status` 生成 catalog；设备不支持的能力卡片置灰并注明原因（如"需 Jetson + SparkTTS profile"）。

**模型切换面板**是 `common/frontend/ui.js` 里的共享组件，门户和每个 demo 页头都有：
- profile 下拉只列出与当前设备匹配且模型已在盘上的（复用 `server/core/profile_selector.py` 的探测逻辑，经 gallery backend 暴露）；
- 切换中轮询 `/admin/backend/status` 展示 drain/RELOADING 进度，失败展示自动回滚结果。

## 4. Demo 卡片（V1 全做，六项）

| # | 应用 | 用到的 API | 演示什么 |
|---|------|-----------|---------|
| 1 | asr-caption | `/asr/stream` | 实时字幕：partial/final、首字延迟、语言自动检测 |
| 2 | tts-playground | `/tts/stream` `/tts/speakers` `/admin/tts/runtime` | 选音色、拖语速/音高、TTFA 显示、流式播放 |
| 3 | v2v-chat | `/v2v/stream` | 完整语音对话 + **barge-in 打断**，逐阶段延迟条（ASR/LLM/TTS） |
| 4 | diarization | `/asr/stream`（流式，final 事件带 `speaker`/`speaker_conf`，`server/main.py:825`；另有 summary 事件） | 多人实时对话，字幕按 speaker 上色 + 说话人占比统计 |
| 5 | voice-clone | `/tts/voices/enroll` `/tts/clone/stream` | 录 10 秒 → 用自己的声音念任意文本（能力不支持时置灰） |
| 6 | 模型切换面板（共享组件） | `/admin/backend/reload` `/admin/backend/status` | ASR/TTS 引擎运行时热切换，全部 demo 页可用 |

V2 备选（不在本计划内）：同传/双语字幕（依赖 NLLB 容器）、语音工具调用（server-loop）、N=2 并发演示、无麦克风脚本化 showcase（复用 `agent/tests/e2e/fake_audio.py`，供录屏/CI）。

## 4.5 演示体验设计原则（硬性要求，每张卡片逐条验收）

核心诉求：**演示醒目，市场人员/开发者打开页面一眼看懂流程、3 秒内体验到效果**。

1. **一屏一主行动**：每个 demo 打开即一个显著的主按钮（🎤 开始说话 / ▶ 试听），不需要读任何文档；次要控件（模型、参数）收进侧栏。
2. **即时可视反馈**：音量波形条、流式字幕逐字上屏、TTS 播放动效——让"流式"和"快"肉眼可见。
3. **大数字指标卡**：TTFA / 首字延迟用大号数字实时跳动展示（如 `0.42s`），这是边缘性能的核心卖点，必须抢眼。
4. **三步引导条**：页面顶部固定 ①选模型 → ②按下说话 → ③看结果 的步骤指示，当前步骤高亮。
5. **舞台化视觉**：深色底、大字号、高对比，展台 2 米外能看清正在发生什么；gallery 门户卡片带能力示意动图/图标。
6. **双语一键切换**（中/英），kiosk 模式下自动轮播 attract 画面。
7. **优雅降级**：设备不支持的能力→卡片置灰+一句话原因；运行错误→人话提示+一键重试，绝不裸抛 JSON。

### 统一视觉规范（源自 Seeed Studio 品牌 token，不使用 Seeed logo/字标）

- 配色：背景 `#000000`（Black Canvas）、面板 `#0a0a0a`（Near Black Surface）、正文 `#ffffff`、次要文字 `#737373`、分割线/描边 `#242424`（1px hairline）；
- **主强调色 Seeed Green `#8dc215`**：只用于主按钮、进度指示、激活态等高信号元素，禁止大面积铺底；次级链接/交互用 `#007aff`，从属于绿色；
- 字体：Montserrat（display+body，本地 `demos/common/frontend/fonts/montserrat-100-900.woff2`，fallback system-ui），指标数字用等宽（SFMono/Menlo）；
- 圆角统一 8px，边框统一 1px；
- 版式取向：紧凑信息密度、真实硬件/部署图像、可量化指标（deployment proof），文案务实工业风、不写空泛 hype；
- **不出现 Seeed/矽递 logo 与字标**，只沿用设计语言。

## 5. 部署模式（已确认：开发者自助 + 展会 kiosk 都要）

- **开发者模式（默认）**：`docker compose -f demos/docker-compose.demos.yml --profile all up -d`；
  demo 页全开，模型切换需要 `SLV_ADMIN_TOKEN`（compose env 注入 demo backend）。
- **kiosk 模式**：env `DEMO_KIOSK=1` —— 免鉴权、门户自动全屏轮播式布局、隐藏调试信息，给展会用。
- 每个 demo 也可单独 `--profile v2v-chat up -d`，独立部署诉求由 compose profile + 独立镜像满足。
- 镜像：每 demo 一个 `python:3.11-slim` 基础的小镜像（预期 <100MB），推 `sensecraft-missionpack.seeed.cn/solution/slv-demo-<name>`。
- 本地开发免 Docker：`uv run demos/<name>/backend/main.py`。

## 6. 关键技术风险与对策

1. **浏览器 V2V barge-in 的回声问题**：笔记本外放时 TTS 会灌回麦克风触发假打断。对策：默认半双工（TTS 播放时 mic 静音，UI 上有"打断"按钮）；页面显著提示两条全双工路径——**佩戴耳机**，或使用**带硬件 AEC 的 reSpeaker（XVF3800）**作为麦克风（注意 makeup_gain 需为 1.0，见既有踩坑记录）。UI 提供全双工开关，选了上述任一输入方案后手动开启。
2. **`/v2v/stream` 浏览器端协议适配**：需要 AudioWorklet 把 48k float 重采样成 16k PCM16。这部分做进 `slv-client.js`，一次投入全部 demo 复用。
3. **diarization 走流式（已确认）**：`/asr/stream` final 事件携带 `speaker`/`speaker_conf`（`server/main.py:825`），流末有 diarization summary（`server/main.py:2889`）。卡片 = 实时上色字幕 + 结束时的说话人占比汇总；不做离线上传形态。
4. **profile 热切换的显存约束**：切换大模型组合可能 OOM。面板只列盘上已有且 profile_selector 判定兼容的组合；reload 失败走既有回滚路径，UI 如实展示。
5. **admin 接口暴露面**：gallery/demos 都在局域网演示语境；admin 调用一律走 demo backend 代理，浏览器永远拿不到 token。
6. **服务端部署前置（orin-nx 真机验证得出）**：SLV 容器 bridge 网络下宿主调用不算 loopback，admin 必须配 `OVS_ADMIN_KEY`；v2v 对话回流必须 `OVS_V2V_SERVER_LOOP=1` + `OVS_V2V_ENGINE=voxedge` + `EDGE_LLM_BASE_URL=http://172.17.0.1:8000/v1`（代码默认 127.0.0.1 在容器内指向 SLV 自身）。已沉淀 demos/README.md「服务端前置条件」表。
7. **真机基准数字（orin-nx, prod-unified-v8, matcha_trt + trt_edgellm）**：TTS 代理首字节 21ms；ASR 回环首 partial 1.17s；v2v asr_final→首块 PCM 313ms；打断 abort 后尾帧滞后 22ms、1s 宽限外零帧。

## 7. 实施阶段与验收

**验收基准设备（已确认）**：
- `orin-nx`（Jetson Orin NX 开发机，fleet 在线）——主验收平台，六项能力全量；**不碰 seeed-orin-nx 生产栈**；
- `rk3588`（Seeed reComputer RK3588，fleet 在线）——次验收平台，asr-caption / tts-playground / v2v-chat / 模型切换四项全通；voice-clone 预期置灰（SparkTTS 为 Jetson 路径）；diarization 视 RK 上 CAM++ (sherpa-onnx CPU) 能力实测决定支持或置灰。
- 每张卡的完成定义 = 双设备实测通过（或按能力矩阵正确置灰），不以本地 mock 为准。
- 已知风险：RK3588 现有部署镜像可能不含 `/admin/backend/reload`，验证前需确认/更新镜像版本。

每阶段独立可交付、真机验收。

- **P0 — common + gallery 门户 + 模型切换**
  产出：`demos/common/`、`demos/gallery/`、compose 骨架。
  验收：浏览器打开 :8700 看到设备状态和卡片；切一次 TTS profile，drain 进度可见，`/admin/backend/status` 反映新 backend，切换期间正在播的 TTS 正常收尾。
- **P1 — 三张核心卡**：asr-caption（收编现有页）、tts-playground、v2v-chat。
  验收：Jetson 真机三卡全通；v2v-chat 完成一次带打断的多轮对话；延迟指标与 `bench/perf` 口径一致（主线程亲自复测关键数字）。
- **P2 — diarization + voice-clone 卡**
  验收：双人对话字幕两色区分；enroll→clone 闭环出声，克隆相似度主观可辨。
  依赖：`feat/diarization` 合并。
- **P3 补充事项（RK3588 构建实战发现）**：
  - demo-gallery worktree 独立构建 RK 镜像缺两个输入：`third_party/rkvoice-stream` submodule 需 `git submodule update --init`；`deploy/rk-runtime/`（librknnrt.so 2.3.2 + librkllmrt.so 1.2.3）是 git 未跟踪的staging 产物，只在主仓工作树——需在 BUILD_IMAGES.md 写明来源/获取方式；
  - RK3588 磁盘治理：docker builder cache 是大头（~16GB），`docker builder prune` 已批准；无引用 volumes（9.17GB）可能含模型数据，未经逐个 inspect 禁止删。
- **P3 — 镜像化 + kiosk + 文档收口**
  产出：五个 demo 镜像推 registry、kiosk 模式、`docs/DEMOS.md` 索引、README 双语加 gallery 入口与截图、`examples/agent/`（最小 app + @tool 两个代码级示例，面向要写代码的开发者，与 gallery 互补）。
  验收：全新设备 `install.sh --pull` + `compose --profile all up` 两步起完整 gallery。

## 8. 不做什么（明确出界）

- LLM 模型切换（edge-llm 独立进程，需容器编排，列 V2）；
- gallery 前端框架化（React/Vite 等）——保持无构建静态；
- 多设备集中管理（gallery 只管同 compose 网络里的一台 SLV）。
