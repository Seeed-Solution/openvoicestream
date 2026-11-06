# 智能家居语音助手 · 部署与开发指南（RK3588 + Home Assistant）

在 RK3588 上跑一套**全本地**的语音助手，配合已有的 Home Assistant。
ASR、TTS、LLM 全部在设备上，不依赖云端。

**实测性能**（真机，中文短句，n=5）：

| 指标 | 值 |
|---|---|
| **停止说话 → 听到第一声（嘴到耳）p50** | **840–900 ms** |
| ├ ASR 出终稿 | ~300–350 ms |
| ├ LLM 到第一个小句 | ~277 ms |
| └ 首句合成 | ~260 ms |
| LLM 吞吐（Qwen3-4B @ 8K 上下文） | TTFT 132 ms / 81.5 tok/s |

---

## 先选方案：两个独立的 app，做的是不同的事

| | **方案一：接入 HA 原生框架** | **方案二：我们自己的 V2V Agent** |
|---|---|---|
| 交付物 | `services/wyoming-adapter/` | `agent/ovs_agent/apps/home_assistant/` |
| 我们提供 | ASR + TTS（作为 HA Assist 的语音服务） | 完整对话：ASR + LLM + TTS + 工具执行 |
| 谁理解意图 | **HA 自己的意图引擎**（确定性匹配） | **我们的 LLM**（自由对话） |
| 谁持麦克风 | HA / 你现有的语音卫星 | **这个 app 自己**（`/dev/snd`） |
| 控设备靠什么 | HA 原生 intent | LLM 调用我们封装的 HA API 工具 |
| 你现有的自动化 / 语音卫星 | **全部保留** | 用不上 |
| **验证状态** | ✅ **端到端验证过**（见 §7） | ✅ **端到端验证过**（语音 → LLM 调工具 → 真设备，见 §7）<br>⚠️ 唯一未验证的是**物理麦克风** |

**两个可以同时开**，端口不冲突 —— 它们是互补的：方案一让 HA 管设备，方案二额外提供
一个带麦克风、能自由聊天的盒子。

> ### 方案二的工具调用是全本地的（要改 shim 的话先读这段）
>
> 方案二靠 **LLM 主动调用工具**来控制设备，LLM 就是自带的 RK1828 Qwen3-4B，
> **不需要任何外部端点**。四个实现要点，改 `services/rk1828-llm` 之前值得知道：
>
> * 工具交给 **runtime 自己渲染**（`rknn3_session_set_function_tools`），它用的是
>   GGUF 里模型自带的 Jinja chat template，工具说明落在**真正的 system 块**里。
>   实测 temperature=0 下 **12/12** 正确，包括无适用工具时正确弃权而不是硬编一个调用。
> * **不要在每轮之后清 KV。** `keep_history=0` 时 runtime 自带前缀缓存，而每个请求
>   重复的正是系统块+工具那一大段。清掉等于白扔：实测每次工具调用差 **约 127 ms**
>   （503–512 ms vs 375–382 ms）。怀疑复用引起怪问题时用 `RK1828_KV_REUSE=0` 退回。
> * `TOOL_MIN_MAX_TOKENS = 320` —— 一句口语回复 96 token 够用，一个 JSON tool call
>   不够。带 `tools` 的请求会被抬到这个下限；否则调用被截断，而截断的调用会被丢弃，
>   整轮静默什么也不做。
> * 流式路径**只在带 tools 时**才做缓冲切分，纯聊天路径逐字直出不变 —— 首字延迟
>   数字依赖于此。
>
> runtime 还有几处**声明了但不能用**的能力（多输入、`KEEP_SYSTEM_PROMPT` 清 KV），
> 踩过的坑连同实测数据记在 `services/rk1828-llm/BUILD.md`，动它之前值得扫一眼。
>
> 如果你**想指向外部端点**（比如换更大的模型），把 `EDGE_LLM_BASE_URL` 指过去即可。
> 注意 vLLM 必须显式启动 `--enable-auto-tool-choice` 和 `--tool-call-parser`，
> 否则它直接拒绝 `tool_choice: auto`（实测踩过）。

---

## 0. 硬件与前置条件

| | |
|---|---|
| 主板 | RK3588，**16 GB 内存**（8 GB 不够） |
| **本地 LLM** | **需要 RK1828 / RM182X M.2 加速卡**。没有这张卡，ASR 和 TTS 照常工作，但 LLM 必须指向外部（局域网另一台机器，或任何 OpenAI 兼容的云端） |
| 麦克风 / 扬声器 | 任何 ALSA 能识别的设备 |
| 磁盘 | 首次启动要拉约 4 GB 模型，**预留 10 GB 以上** |
| Home Assistant | 你已有的实例，**从这台设备网络可达** |

### 0.1 RK1828 加速卡：宿主机准备（镜像不是自包含的）

加速卡的**内核驱动和固件在宿主机上**，容器只能访问一张已被宿主机初始化好的卡。
仓库里带了核验脚本：

```bash
sudo deploy/scripts/rk1828-host-bringup.sh --check    # 只读核验
sudo deploy/scripts/rk1828-host-bringup.sh            # 顺手补上模块加载与持久化
```

9 项全绿才继续。脚本**不会**编译内核模块（那需要 RM182X SDK + 内核头文件，见
`services/rk1828-llm/BUILD.md`），缺了会明确告诉你。

三条会咬人的事，脚本里也写了：

- **模块加载必须持久化**，否则重启后卡就消失，LLM 服务启动失败且看不出原因
- 卡有**独立 12V 供电**。漆黑 + 风扇不转 = 没通电；主电源口和风扇口紧挨着，容易接错
- **绝不要跑 `rknn-smi reset`** —— 会把卡推进一个连宿主机重启都可能救不回的状态，而且卡不随宿主机断电

### 0.2 单卡独占：一次只能驻一个大模型

RK1828 只有**一个约 5 GB 的上下文**。Qwen3-4B @ 8192 token 估算占用约 3.6 GB，
所以**这张卡给了 LLM，就不能同时跑别的大模型**（比如 RK1828 版的 TTS）。
本方案的 TTS 跑在 RK3588 自己的 NPU 上，正是为此。

> ⚠️ 这张卡的**显存和健康状态没有任何观测手段** —— `rknn-smi` 在该平台整机失效
> （root 也失败、卡空闲时也失败，疑为出厂的 host/EP 固件版本不匹配）。唯一的信号
> 是"模型加载成不成"。而**失败的加载会把卡从 8 核退化到 4 核**，所以不要靠反复
> 试加载来探容量。

---

## 0.3 Home Assistant 侧需要装什么

**好消息：两个方案在 HA 侧都不需要装任何自定义组件或 HACS。**

实测于 **HA 2026.7.4**，以下全部是内置且随 `default_config` 自动加载：

| 组件 | 用途 | 状态 |
|---|---|---|
| `wyoming` | 方案一注册我们的 ASR/TTS | ✅ 内置已加载 |
| `assist_pipeline` | 语音管道（选 STT/TTS/对话代理） | ✅ 内置已加载 |
| `conversation` | 意图理解（方案一靠它控设备） | ✅ 内置已加载 |
| `stt` / `tts` | 语音服务实体框架 | ✅ 内置已加载 |

所以 HA 侧要做的只是**配置**，不是安装：

| 方案 | HA 侧要做的事 |
|---|---|
| **方案一** | ① 添加 Wyoming Protocol 集成 ×2（两个端口各一次）<br>② 建一条 Assist 管道，STT/TTS 都选 `seeed-local-voice` |
| **方案二** | 只需要一个**长效访问令牌**（见 §1）。我们的 app 直接调 HA 的 REST API，HA 侧无需配置任何集成 |

### ⚠️ 一个经典坑：实体必须暴露给 Assist（只影响方案一）

方案一的设备控制由 **HA 自己的意图引擎**做，而它只能看到**暴露给 Assist 的实体**。
HA 对受支持的域（light / switch / cover / fan / climate …）**默认是暴露的** ——
本次验证的 6 个实体查出来都是 `conversation=True`，所以开箱即用。

但如果之前有人手动关过某些实体的暴露，Assist 会回**"找不到名为 XX 的设备"**，
看起来像我们的 ASR 识别错了，实际是暴露设置的问题。检查位置：
**设置 → 语音助手 → 公开** 标签页。

> 顺带：方案二**不受**这个设置影响 —— 我们的工具走 REST API，看到的是全部实体。
> 想收窄给语音助手看的范围，用 app 配置里的 `ha_exclude_entity_ids`。

### 语音卫星 / 唤醒词（可选，两个方案都不强制）

- **方案一**：可以配合你已有的 ESPHome 语音卫星、HA 手机 App，或 HA 内置的
  openWakeWord。麦克风在**卫星侧**，我们的盒子只提供 ASR/TTS。
- **方案二**：麦克风接在**我们的盒子上**，不需要卫星。

---

## 1. 拿到 Home Assistant 的长效令牌

HA 网页 → 左下角头像 → **安全** → **长期访问令牌** → 创建。**只显示一次**，复制保存。

然后**在这台设备上**（不是你的笔记本上）验证地址可达 —— 这是最常见的失败原因：

```bash
curl -sI http://<你的HA地址>:8123/    # 期望 200 或 405，不是 timeout
```

---

## 2A. 方案一：接入 HA 原生框架（Wyoming）

把我们的 ASR/TTS 注册成 HA Assist 的语音服务，**意图理解和设备控制交给 HA 自己** ——
所以这条路**不需要 LLM 支持 function calling**，也不需要麦克风接在我们的盒子上。

### 起适配器

只需要语音服务（`speech`）+ 适配器；LLM 服务这条路用不到。

```bash
docker compose -f services/wyoming-adapter/docker-compose.wyoming.yml up -d
```

两个端口：**STT 10300**、**TTS 10200**。上游指向语音服务，用 `SLV_BASE_URL` 配置。

### 注册进 HA

UI：**设置 → 设备与服务 → 添加集成 → Wyoming Protocol**，把两个 host:port **分别**加一次。
或走 REST API（`TOKEN` 换成你的长效令牌）：

```bash
for PORT in 10300 10200; do
  FLOW=$(curl -s -X POST -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' -d '{"handler":"wyoming"}' \
    http://<HA地址>:8123/api/config/config_entries/flow | jq -r .flow_id)
  curl -s -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
    -d "{\"host\":\"<适配器地址>\",\"port\":$PORT}" \
    http://<HA地址>:8123/api/config/config_entries/flow/$FLOW
done
```

**地址要填 HA 能访问到的那个** —— 如果 HA 跑在 Docker 里而适配器在宿主机上，
填 `host.docker.internal`（先验：`docker exec homeassistant getent hosts host.docker.internal`），
**不是** `127.0.0.1`。

成功后会出现两个实体：`stt.seeed_local_voice` / `tts.seeed_local_voice`。

### 建 Assist 管道

**设置 → 语音助手 → 添加助手**，语音转文字和文字转语音都选 `seeed-local-voice`。
对话代理保持 HA 内置的即可（意图匹配由它做）。

### 验证

```bash
cd services/wyoming-adapter && uv run python verify_protocol.py
```

它会断言：两个能力标志正确、真 WAV 能拿到正确转写、只有**一次** `audio-start`
（防重复合成）、音频 RMS > 0（字节非空 ≠ 有声）。

然后在 HA 里对着语音助手说"打开客厅灯"。**实测结果**：`action_done`、
`success: [light.ke_ting_deng]`、灯真的亮。

---

## 2B. 方案二：我们自己的 V2V Agent

完整对话在我们的盒子上跑（ASR + LLM + TTS + 工具执行全本地），麦克风也接在盒子上。

```bash
cd <仓库根目录>
cat > deploy/.env <<'EOF'
HA_BASE_URL=http://192.168.1.10:8123
HA_TOKEN=<你的长效令牌>
# 想换一个更大的模型时才需要：指向任何 OpenAI 兼容且支持 function calling 的端点。
# 不设 = 用自带的 RK1828 Qwen3-4B（已验证可调工具）。
# EDGE_LLM_BASE_URL=http://192.168.1.20:8000/v1
# 没有麦克风时的自测入口，见 §3。默认关闭。
# OVS_AGENT_DEBUG_INJECT=1
EOF

docker compose -f deploy/docker-compose.rk3588-ha.yml up -d
```

三个服务：

| 服务 | 端口 | 干什么 |
|---|---|---|
| `llm` | 1828 | Qwen3-4B 在 RK1828 卡上，OpenAI 兼容接口 |
| `speech` | 8621 | Qwen3-ASR + matcha TTS 在 RK3588 NPU 上，对话编排 |
| `agent` | — | 麦克风/扬声器客户端，执行 Home Assistant 控制 |

### 起之前必须知道的四件事（都是真机踩出来的）

**① 镜像从 registry 拉，不用自己 build。** 三个镜像都已推到
`sensecraft-missionpack.seeed.cn/solution/`（**需要凭证，不是公开 registry** —— 找我们要）：

```bash
# 先登录，否则 pull 报 403
docker login sensecraft-missionpack.seeed.cn

cat >> deploy/.env <<'EOF'
REG=sensecraft-missionpack.seeed.cn/solution
VOICE_IMAGE=sensecraft-missionpack.seeed.cn/solution/openvoicestream:rk-pypi-20260730
AGENT_IMAGE=sensecraft-missionpack.seeed.cn/solution/ovs-agent:rk-pypi-20260730
LLM_IMAGE=sensecraft-missionpack.seeed.cn/solution/edge-llm-rk1828:20260731-kvreuse
EOF
```

**不设这三个变量**时，compose 用的是**不带 registry 前缀**的本地 tag，那要求你自己先按各自的
`BUILD.md` 构建过；否则 Docker 会去 Docker Hub 找一个不存在的名字，报 **403 Forbidden**
（这个报错看着像权限问题，其实是名字不存在）。

三个镜像**都是从本仓库当前源码构建的**，不含任何手工热补丁 —— 你自己 `docker build`
一次，拿到的就是本文所有数字所依据的那个产物（推到 registry 的那三个已用"删本地 tag →
重新 pull → 比对镜像 ID"验证过，与本地产物逐字节一致）。语音镜像可以自查这四条
（构建时也会断言）：`voxedge.__version__` == `0.0.6a0`（从 PyPI 装，pinned）、
`server/main.py` 含 `REALTIME_V2_SUBPROTOCOL`、`tts_sequencer.py` 含 `_to_speakable`
（Markdown 过滤）、`matcha.py` 含 `_npu_lock`（与 ASR 共享 NPU 锁）。

**② 容器 LLM 与裸机 LLM 不能共存。** 如果你之前按 `services/rk1828-llm/BUILD.md`
装过 `rk1828-llm.service`，起 compose 前必须停掉它 —— 单卡只有一个 5 GB 上下文：

```bash
sudo systemctl stop rk1828-llm
```

**③ 容器名冲突。** 本 compose 的 `speech` 用 `container_name: openvoicestream`。
如果你还在跑旧的单服务 compose，先 `docker compose -f deploy/docker-compose.rk3588.yml stop speech`。

**④ agent 会占住唯一的 session 槽。** 语音服务只允许**一个** `/v2v/stream` 会话。
所以 agent 一连上：
- **`/readyz` 会返回 503** `{"reasons":["sessions_full"]}` —— **这不是故障**。判断服务
  健康请用 **`/health`**（compose 的 healthcheck 已经用它，就是这个原因）
- **跑不了压测脚本**（或任何第二个客户端），会被 WS 4429 拒。要测先停 agent：
  `docker compose -f deploy/docker-compose.rk3588-ha.yml stop agent`

**任何镜像里都没有模型。** 每个服务首次启动时按自己的配置拉取到 named volume，
所以第一次启动会久（LLM 要拉约 3.2 GB，健康检查的宽限期设了 900 秒）。

进度观察：

```bash
docker compose -f deploy/docker-compose.rk3588-ha.yml logs -f llm      # 下载 + 加载
curl -s http://127.0.0.1:1828/health                                   # LLM 就绪
curl -s http://127.0.0.1:8621/readyz                                   # 语音服务就绪
docker logs ovs-agent --tail 30                                        # 看 HA 连上没有
```

agent 日志里应该出现这三行（真机实测）：

```
[ha] connected to http://…:8123 — 6 controllable devices
boot: … resolved server_loop_enabled=True
SLV advertise 9 tool(s)  /  server-loop mode: advertised 9 tool(s) to SLV
```

服务端侧对应：`tool_advertise: registered 9 remote tool(s) [...]`。

**没接麦克风也能启动** —— 实测容器会优雅降级：

```
mic None not available at start (Invalid sample rate [PaErrorCode -9997]);
booting without it — will auto-detect with exponential backoff
```

所以可以先不接麦克风把链路跑通，之后再插。

如果国内网络拉不动 HF，在 `.env` 里加 `HF_ENDPOINT=https://hf-mirror.com`（镜像站
可能滞后于源站）。

---

## 3. 验证

对着麦克风说：**"客厅灯打开"**。预期：约 0.9 秒后听到"好的"，灯亮。

分段排查：

```bash
# LLM 单独测（TTFT / 吞吐）
python3 - <<'EOF'
import json,urllib.request
r=urllib.request.urlopen(urllib.request.Request(
  "http://127.0.0.1:1828/v1/chat/completions",
  data=json.dumps({"messages":[{"role":"user","content":"你好"}],"max_tokens":32}).encode(),
  headers={"Content-Type":"application/json"}))
print(json.load(r)["choices"][0]["message"]["content"])
EOF

# 全链路延迟（仓库自带）。注意：agent 占着唯一的 session 槽，先 stop agent
uv run --with websocket-client --with soundfile --with numpy \
  python bench/perf/measure_v2v_unified.py --host 127.0.0.1:8621 \
  --wav bench/perf/corpus/short/zh_short_01.wav --language Chinese --runs 5 --server-loop
```

### 没有麦克风怎么验（方案二）

麦克风还没接、或想做可重复的回归时，用注入端点把一段 WAV 当成"说过的话"喂进去。
这是**唯一能忠实验证 server-loop 的办法** —— 另一个端点
`/api/control/inject_user_text` 伪造的是 ASR 终稿，走的是**客户端**对话循环，
而我们出货的 server-loop 里 LLM 和工具派发都在服务端，测它证明不了什么。

先在 `.env` 里设 `OVS_AGENT_DEBUG_INJECT=1` 并重启 agent（默认关闭 —— 开了之后
任何能访问 18000 端口的人都能操作你的设备）。然后：

```bash
# 1. 用自带的 TTS 合成一句命令音频（agent 占着 session 槽，先停）
docker compose -f deploy/docker-compose.rk3588-ha.yml stop agent
curl -s -X POST http://127.0.0.1:8621/tts -H 'Content-Type: application/json' \
  -d '{"text":"打开客厅灯","language":"zh"}' -o /tmp/cmd.wav
docker compose -f deploy/docker-compose.rk3588-ha.yml start agent && sleep 20

# 2. 注入 —— 之后灯应该真的亮
curl -X POST --data-binary @/tmp/cmd.wav \
  http://127.0.0.1:18000/api/control/inject_wav

# 3. 看证据：ASR 听成了什么、派发了哪个工具、HA 是否返回 200
docker logs ovs-agent --since 60s | grep -E 'asr_final|api/services|ok=True'
```

拿自带 TTS 合成来喂 ASR 有个已知局限：**它不等于真麦克风音频**。实测 4 句里
3 句正确，失败的那句 `"把客厅灯关掉"` 被 ASR 听成 `"把客厅厅灯调。"`，于是根本没
派发工具（换成 `"关闭客厅灯"` 就对了）。所以这个入口适合验**链路**，
不适合用来评估 ASR 准确率。

---

## 4. 加你自己的设备控制工具

这是**主要的扩展点**。完整说明在
`agent/ovs_agent/apps/home_assistant/README.md`，这里只给要点。

内置七个工具：`list_devices` / `turn_on` / `turn_off` / `set_brightness` /
`set_cover_position` / `get_state` / `call_service`（通用逃生口）。

加一个新工具 = 往 `ha_tools.py` 加一个函数：

```python
@_r.tool(
    description="启动扫地机器人清扫。room 是房间名，例如 客厅。",
    preamble_text="好的，开始打扫。",   # 工具一开始执行就先出声
    response_mode="template",
    completion_text="已经开始打扫了。",  # 跳过 LLM 第二轮，延迟最低
)
def start_vacuum(room: str) -> dict:
    try:
        ha = _ha()
        d = ha.resolve(room, domains=("vacuum",))
    except ResolveError as e:
        return _fail(str(e), e.candidates)     # 让 LLM 拿 candidates 重试
    try:
        ha.call_service("vacuum", "start", {"entity_id": d.entity_id})
    except Exception as e:
        return _fail(f"启动 {d.name} 失败: {e}")
    return {"ok": True, "device": d.name}
```

**schema 从类型标注自动生成**，不用手写 JSON Schema。改完重启 `agent` 即可：

```bash
docker compose -f deploy/docker-compose.rk3588-ha.yml up -d --build agent
```

### 关于设备名（值得单独看）

**Home Assistant 会把非拉丁实体名转写成拼音 `entity_id`**：

```
客厅灯   → light.ke_ting_deng
客厅窗帘 → cover.ke_ting_chuang_lian
```

所以**不要靠解析 `entity_id` 判断设备是什么** —— 唯一可靠的人类可读标签是
`friendly_name`。框架的匹配全部基于它，并且做了口语容错（`把客厅的灯` 能匹配到
`客厅灯`）。匹配不到或有歧义时**不会瞎猜**，而是返回 candidates 让 LLM 反问 ——
关错房间的灯比多问一句糟糕得多。

想让效果好，**在 HA 里把实体名起清楚**（"客厅主灯"而不是"Light 1"）比改代码有效得多。

---

## 5. 排查

| 症状 | 原因 |
|---|---|
| agent 日志 `[ha] cannot reach ...` | 地址从**设备上**不通，或令牌过期。用 §1 的 curl 验 |
| agent 日志 `[ha] ha_base_url / ha_token not configured` | `.env` 没被读到，或变量名写错 |
| 能对话但从不控制设备 | `speech` 的 `OVS_V2V_SERVER_LOOP` 没开。这个和 `OVS_V2V_ENGINE` **缺一不可**，少一个就退化成"只会复述的语音服务" |
| LLM 容器起不来，日志 `no /dev/pcie-rkep-*` | 宿主机没准备好，回到 §0.1 |
| LLM 容器 `MODEL_SETUP fail` / `ACK_FAIL` | 卡被别的大模型占着（检查 `systemctl is-active tts-rk3588`），或上下文设得太大。**最多重试两次**，之后干净重启宿主机 —— 反复失败会让卡退化 |
| 回复很长、把符号念出来 | `OVS_V2V_SYSTEM_PROMPT` 被覆盖成空了。不设的话模型会按聊天助手风格输出 Markdown（实测：一个短问句能合成出 **87 秒**音频） |
| 首次启动很久没反应 | 在拉 4 GB 模型。`docker logs -f ovs-llm` 看进度 |

---

## 6. 调优

| 想要 | 怎么做 |
|---|---|
| 更长的对话上下文 | `.env` 里 `RK1828_MAX_CONTEXT=16384`。**8192 和 16384 都已实测可加载**（16384 首次即成功，TTFT 132ms / 81.3 tok/s、全链路 839ms，与 8192 无差异）。默认仍是 8192，因为 16384 让卡的占用从约 71% 升到**估算 92%** —— 而这个平台**无法读取显存**，且一次失败的加载会把卡从 8 核退化到 4 核。需要长上下文再有意识地调高 |
| 更短的回复 | 降 `OVS_V2V_LLM_MAX_TOKENS`（默认 96） |
| 更快的断句 | 降 `VAD_ENDPOINT_SILENCE_MS`（默认 400）。太低会把停顿误判成说完 |
| 换 LLM | 改 `EDGE_LLM_BASE_URL` 指向任何 OpenAI 兼容端点，并相应改 `EDGE_LLM_MODEL`。不需要 RK1828 卡 |
| 换音色 | `TTS_DEFAULT_SID`；可用音色见 `GET /tts/speakers` |

**不要动的两个**：

- `MATCHA_USE_ORT=1` —— 这是 RK3588 上**唯一验证过闭环**的配方，改成 0 走全 RKNN
  路径会让 TTS→ASR 闭环失败
- `ASR_NPU_CORE_MASK=NPU_CORE_1` —— matcha 把自己的 Vocos 钉在 core 0 并给 ASR
  编码器留了 core 1。改成 `NPU_CORE_AUTO` 会**静默作废**这个隔离，让两个共驻的
  RKNN 上下文可能落到同一个核

---

## 7. 已验证到什么程度（按方案分列）

写给要复现的人：下面严格区分**已在真机验证**和**未验证**。凡是标"未验证"的，
请不要假定它能工作。

### 共用底座（两个方案都依赖）

| 项 | 状态 |
|---|---|
| 嘴到耳延迟 p50 839–900 ms，三段拆分（ASR / LLM / 合成） | ✅ 实测，跨 5 组独立测量复现 |
| 多轮：同一会话 3 句 × 4 会话 = 12 个 utterance，ASR 出终稿 p50 **320 ms**（min 305 / max 510） | ✅ 实测。**注意口径**：嘴到耳只在每个会话的**第一轮**记录一次，所以这一行是 ASR 段的多轮数据，不是"多轮嘴到耳不退化"的证据 |
| **从仓库源码构建的镜像与之前手工烘的镜像等速** | ✅ 同一 harness A/B：872 ms vs 879 ms p50 |
| ASR + TTS 在同一块 NPU 上共驻 40 分钟 | ✅ 零 RKNN 故障 |
| 说话打断真重叠 3/3（打断句仍正确识别） | ✅ 实测；打断流量**不留残留**（前 893 / 后 881 / 重启后 898 ms） |
| 回复里的 Markdown 不会变成静音碎片 | ✅ 实测，`No audio produced for text` 计数为 0 |
| RK1828 LLM：TTFT 132 ms / 81.5 tok/s @ 8K 上下文 | ✅ 实测；16K 也验证可加载（默认仍 8K，见 §6） |
| 宿主机 bring-up 脚本 9 项检查 | ✅ 全绿 |

### 方案一：接入 HA 原生框架

| 项 | 状态 |
|---|---|
| **作为容器跑在设备上**（`wyoming-slv:20260730`，248 MB） | ✅ **实测** —— 两个端口在监听，协议自检全过 |
| Wyoming 协议自检（能力标志 / 转写 / 单次 audio-start / RMS>0） | ✅ 实测（0.181 s/字符，重复合成会是 2 倍） |
| 在 HA 里注册出 `stt.seeed_local_voice` + `tts.seeed_local_voice` | ✅ 实测，指向**设备上的容器**，全程 REST/WS API 驱动，无需手点 UI |
| 完整 Assist 管道跑通（run-start → stt → intent → tts → run-end） | ✅ 实测，两个 hop 都走我们的适配器 |
| **设备控制**："打开客厅灯" → `success: [light.ke_ting_deng]` → 灯亮 | ✅ **实测** |
| HA 侧零安装（wyoming/assist_pipeline/conversation/stt/tts 全内置） | ✅ 实测于 HA 2026.7.4 |
| 经 HA 自己的 TTS 通路取回音频（4.2 s，RMS 5522） | ✅ 实测 |

### 方案二：我们自己的 V2V Agent

| 项 | 状态 |
|---|---|
| `OVS_V2V_SERVER_LOOP` + remote tool 这条码路 | ✅ **生产验证**（机械臂 app，115/115 = 100% @ temp=0，Jetson + edge-llm） |
| 7 个 HA 工具逐个对真 HA 生效（light/switch/cover，中文名 + 拼音 id） | ✅ 实测 |
| 三条错误路径（歧义名 / 不存在设备 / 超范围参数）行为正确 | ✅ 实测 |
| agent 镜像构建期断言 7 个工具全部注册 | ✅ 实测 |
| **三服务 compose 真跑起来**（llm / speech / agent 全 healthy） | ✅ **实测** |
| **agent 容器连上服务端并 advertise 9 个工具** | ✅ **实测**，服务端确认 `registered 9 remote tool(s)` |
| **没有麦克风也能启动**（容器优雅降级） | ✅ 实测 |
| 三服务形态下的延迟 p50 924 ms（LLM 进容器多一层） | ✅ 实测 |
| **LLM 自己决定调用工具** → 用自带的 RK1828 端点，**全本地** | ✅ **实测**，shim 已补齐（见开头说明） |
| ├ shim 层：`tools` 字段 / 流式 `delta.tool_calls` / `finish_reason` / 工具结果回传 | ✅ 10/10，含"content 不泄漏 sentinel"与"无工具路径不回归" |
| ├ 模型层：Qwen3-4B@RKLLM3 吐出格式正确的 `<tool_call>` | ✅ 12/12 @ temp=0，含无适用工具时正确弃权 |
| └ 切分器单测：逐字符切分 / sentinel 跨 token / 截断丢弃 / 双调用 | ✅ 8/8 |
| **语音 → LLM → 工具 → 真设备**（`turn_on` / `turn_off` / `set_brightness`） | ✅ **实测 3/3**，HA 状态真变（30% → brightness 76） |
| 无麦克风注入入口（`/api/control/inject_wav`，默认关闭） | ✅ 实测；`voice_rebot_arm` 原有 4 个测试仍通过 |
| **物理麦克风**到 HA 的整机联调 | ❌ **未验证** —— 从未接过真麦克风。上面 3/3 是自带 TTS 合成的音频 |
| 自带 TTS 合成音频喂 ASR 的准确率 | ⚠️ 4 句 3 对：`"把客厅灯关掉"` 被听成 `"把客厅厅灯调。"`（换 `"关闭客厅灯"` 正确）。**不代表真麦克风表现** |

### 首次整机联调建议

板载 ES8316 codec 有采集能力，但**增益/削顶未验证**（本仓库有过麦克风 makeup gain
爆量程导致 ASR 明显变差的先例）。第一次接麦克风时：

```bash
arecord -l                                   # 确认设备号
arecord -D hw:4,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/t.wav
python3 -c "import wave,numpy as np;w=wave.open('/tmp/t.wav');a=np.frombuffer(w.readframes(w.getnframes()),'<i2');print('rms',a.std(),'peak',abs(a).max())"
```

peak 贴近 32767 说明削顶了，要降增益 —— **字节非空不等于音频可用**。
出问题优先看 `docker logs ovs-agent`。
