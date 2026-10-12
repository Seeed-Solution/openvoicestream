# Jetson 语音栈搭建 Runbook

从一台裸 Jetson 到一套可用的本地语音服务（ASR + TTS + 声纹），可选叠加本地大模型。

**这份文档的定位**：面向"我要在一台新设备上把它跑起来"的操作流程。
产物版本号、引擎 hash、HF revision、回滚镜像 ID 这类**发布事实**不在这里维护，
看 [`docs/deploy/jetson-orin-nx-v091.md`](../deploy/jetson-orin-nx-v091.md)。
两份文档冲突时以那份为准。

---

## 0. v0.11 候选入口

本 runbook 的 Orin NX 默认语音入口绑定 v0.11 合并 voice profile；旧版安装器仍需显式使用 `--legacy` 回滚。v0.11 镜像和模型产物保持候选质量标记，不能把本节的单次 GPU smoke 当作全量性能或生产资格验收。

语音候选镜像按 digest 拉取：

```bash
export SPEECH_IMAGE=sensecraft-missionpack.seeed.cn/solution/seeed-local-voice@sha256:4a61834fdc2862ec8d61c4748898fb02694a06bfea07c4ba166bee49d811a39a
export OVS_PROFILE=jetson-edgellm-v011-default-voice
export HF_ENDPOINT=https://hf-mirror.com
docker compose -f deploy/docker-compose.edgellm-v011-candidate-voice.yml up -d
```

它使用 ASR B2 省内存配方和 CustomVoice R16；对应 HF revision 已固定在 profile。LLM 候选是独立服务，必须先执行：

```bash
HF_ENDPOINT=https://hf-mirror.com deploy/install.sh --target orin-nx --v011-llm-candidate --prepare-only
```

准备完成后按 `deploy/docker-compose.edgellm-v011-candidate-llm.yml` 或 `deploy/install.sh --target orin-nx --v011-llm-candidate` 启动显式 LLM 候选；安装器只启动该独立服务并退出。服务固定使用 IFB B2，公开 model name 为 `qwen3.5-4b`，需要本地 staging checkpoint、宿主 CUDA/TensorRT 库和 launcher 只读 bind。该服务仍保持候选质量标记，不等同于生产资格。

### v0.11 默认 voice smoke 记录（Orin NX）

已发布的 voice 镜像固定为：

```text
sensecraft-missionpack.seeed.cn/solution/seeed-local-voice@sha256:4a61834fdc2862ec8d61c4748898fb02694a06bfea07c4ba166bee49d811a39a
```

默认 profile 同时挂载 ASR IFB B2（12 个必需文件）和 TTS CustomVoice R16（25 个必需文件），使用既有本地缓存；profile 的 ASR revision 为 `862ae0db85703e894ed9ae7448ae1ce11a79cda3`，TTS revision 为 `009cd08677ec0f42378060f99c99a0a409e5ca70`。一次 Orin NX GPU smoke 使用该 digest 启动容器并完成一条 TTS→ASR 回环：音频 `172844` bytes、`3.6s`，ASR 文本为“你好，今天天气真不错。”，similarity `1.0`。

原始验收 wrapper 因按行解析格式化 JSON 而记录 aggregate `FAIL`；客户端实际 exit code 为 `0`，原始 stdout 的完整 JSON 已离线重放为 request `PASS`。保留该 aggregate FAIL 记录，不能将这次单条回环提升为候选质量标记、性能矩阵、soak 或生产资格。证据见 `/tmp/slv-v011-voice-image-final-smoke-oct12-r1/actual-run-r3-nx/reconciled-evidence.json`、`wrapper-fix.diff` 及其 `evidence/` 子目录。

## 1. 先选拓扑

| 拓扑 | 起几个容器 | 大模型在哪 | 用哪套 compose |
|---|---|---|---|
| **A. 纯语音** | 1（speech:8621） | 云端 API，或另一台机器 | `docker-compose.edgellm-v091-voice.yml` |
| **B. 语音 + 本地 LLM** | 2（speech:8621 + edge-llm:8000） | 本机 Qwen3.5-4B GDN/MTP | 上面那份 + `docker-compose.edgellm-v091-cutover.yml` |

拓扑 B 在 16GB Orin NX 上是**已验证的三模型共驻**（Qwen3-ASR + Matcha TTS + Qwen3.5-4B）。
不要在 8GB 设备（Orin Nano）上跑 B，会 OOM。

`deploy/install.sh --target orin-nx` 默认就是 B。只要 A 的话跳到 [§4.2](#42-方式-b只起语音走-compose)。

---

## 2. 硬件与系统前提

| 项 | 要求 | 怎么查 |
|---|---|---|
| 设备 | Orin NX 16GB（拓扑 B）/ Orin NX、Orin Nano 8GB（拓扑 A） | `cat /proc/device-tree/model` |
| JetPack | 6.2（L4T r36.4，TensorRT 10.3） | `cat /etc/nv_tegra_release` |
| Docker | Engine ≥ 20，Compose v2 | `docker compose version` |
| NVIDIA runtime | 必须装 | `docker info \| grep -i nvidia` |
| 磁盘 | 拓扑 A ≥ 15 GB，拓扑 B ≥ 25 GB | `df -h /var/lib/docker` |
| TensorRT 工具 | `/usr/src/tensorrt/bin/trtexec` 存在 | install.sh 会硬性检查 |
| CUDA | `/usr/local/cuda/lib64/libcudla.so.1` 存在 | install.sh 会硬性检查 |

镜像是**瘦镜像**：TensorRT、CUDA、NVIDIA 驱动库全部从宿主只读挂载进去
（`/host-cuda`、`/host-nvidia-libs`、`/host-libs`）。所以宿主的 JetPack 版本必须对得上，
不能拿 JetPack 5.x 的机器跑 v0.9.1 的镜像。

---

## 3. 宿主准备

### 3.1 装宿主库 `libsentencepiece0` ⚠️

```bash
sudo apt-get install -y libsentencepiece0
```

**这一步最容易漏，而且漏了以后的报错完全指不回原因。**

镜像本身不带 `libsentencepiece.so.0`，靠宿主 `/lib/aarch64-linux-gnu`
（挂成 `/host-libs` 并进了 `LD_LIBRARY_PATH`）提供。我们的参考机 orin-nx 是 Ubuntu jammy，
碰巧自带；**干净的 JetPack 镜像没有**。缺了以后 MOSS TTS worker 的启动预检加载不到符号，
整个 speech 容器 crash-loop，日志里是一大串 `undefined symbol`，
看不出是"宿主少装一个包"。客户现场实测踩过。

幂等检查：

```bash
ldconfig -p | grep libsentencepiece.so.0 || echo "缺失，需要装"
```

> `deploy/install.sh` 目前**不做**这项检查（`trtexec` 和 `libcudla` 检查了，这个没有）。
> SenseCraft 方案平台的部署描述（`devices/ovs_jetson_deploy.yaml`）里有一个
> `before` action 补了这一步，但走 install.sh 的人不经过它。

### 3.2 确认 HF 镜像

模型/引擎产物首次启动时从 HuggingFace 拉（拓扑 A 约 5 GB，拓扑 B 约 10 GB）。
国内设备直连 `huggingface.co` 不通，必须走镜像。

两份 compose 都写死了默认值 `HF_ENDPOINT: ${HF_ENDPOINT:-https://hf-mirror.com}`，
**即使宿主 shell 没导出也能正常走镜像**——compose 不继承未声明的变量，所以这个默认值是必要的，
不要删。

只有当你想换镜像源时才需要显式设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 或其它
```

验证镜像上有目标产物（别拿"镜像没同步"当理由走直连）：

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://hf-mirror.com/api/models/harvestsu/qwen3-asr-0.6b-jetson-artifacts
# 200 = 有
```

### 3.3 检查端口和容器名冲突

`8621` 和 `8000`，以及容器名 `seeed-voice-v091`、`edge-llm-chat-service-v091` 必须空着。

```bash
docker ps -a --filter "name=seeed-voice-v091" --filter "name=edge-llm-chat-service-v091"
ss -ltnp | grep -E ':(8621|8000)\b'
```

同名容器如果属于**另一个 compose project**（比如设备上手工装过一次），
`up -d` 会因名字冲突失败，且 `remove_orphans` 跨 project 不起作用。
处理办法是 `docker rm -f <名字>`——**数据卷不会被删，模型不用重下**。

---

## 4. 部署

### 4.1 方式 A：install.sh 一键（拓扑 B）

需要 clone 整个仓库（脚本用相对路径引用 compose 文件）。

```bash
git clone https://github.com/Seeed-Solution/openvoicestream.git
cd openvoicestream
deploy/install.sh --target orin-nx --pull --verify
```

脚本会依次做：前置检查 → `docker compose pull` → `up -d` → `deploy/verify.sh` → `deploy/verify-llm.sh`。
任何一步失败都会打印最近 80~120 行日志再退出。

想要 4K 上下文而不是默认 8K：

```bash
EDGELLM_ENGINE_PROFILE=4k deploy/install.sh --target orin-nx --pull --verify
```

想换语音 profile：

```bash
OVS_PROFILE=jetson-edgellm-v091-qwen3ttsbase deploy/install.sh --target orin-nx --pull --verify
```

### 4.2 方式 B：只起语音，走 compose

不想 clone 整个仓库的话，只需要那一个 compose 文件：

```bash
mkdir -p ~/voice && cd ~/voice
curl -fsSLO https://raw.githubusercontent.com/Seeed-Solution/openvoicestream/main/deploy/docker-compose.edgellm-v091-voice.yml
docker compose -f docker-compose.edgellm-v091-voice.yml up -d
```

> 该文件带一个 `build:` 段。只要不加 `--build`，`up -d` 用的是 registry 镜像，
> `build:` 段不会被触发，也不需要仓库上下文。

### 4.3 等它起来

**首次启动要下模型，15 分钟起步。** compose 的 `start_period` 就写了 15m，
健康检查在这之前不会判失败。别看到"还没 healthy"就重启——重启会中断下载。

```bash
docker logs -f seeed-voice-v091
# 等到 /readyz 返回 200
until curl -fsS http://127.0.0.1:8621/readyz >/dev/null; do sleep 10; done && echo READY
```

---

## 5. 验证

```bash
deploy/verify.sh --url http://127.0.0.1:8621 --tts-smoke --roundtrip
curl -fsS http://127.0.0.1:8621/v1/models
curl -fsS http://127.0.0.1:8621/v1/capabilities
```

拓扑 B 追加：

```bash
deploy/verify-llm.sh http://127.0.0.1:8000
```

`verify-llm.sh` 会等 `/v1/models` 可达（默认超时 900s，`LLM_VERIFY_TIMEOUT` 可调），
然后发一条 `temperature=0` 的 chat 请求，断言返回非空内容。

**TTS 验证的注意事项**：字节非空 ≠ 有语音。`--tts-smoke` 只走通了链路，
要确认真的出声，做一次 ASR 回环（`--roundtrip` 就是干这个的），
或者自己测一下音频能量。踩过"TTS 返回一堆全零 PCM"的坑。

手工出一个 wav：

```bash
curl --http1.1 -N http://127.0.0.1:8621/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"matcha-icefall-zh-en","input":"你好，欢迎使用本地语音服务。","voice":"0","response_format":"wav","speed":1.0}' \
  --output speech.wav
```

---

## 6. 选 profile

`OVS_PROFILE` 决定这台机器加载哪套 ASR/TTS 引擎。改 profile **只改变要下载的模型**，
不重建镜像。

| Profile | ASR | TTS | 用途 |
|---|---|---|---|
| `jetson-edgellm-v091-matcha` ⭐默认 | Qwen3-ASR | Matcha TRT | 与 GDN 共驻的验证配置，N=1 |
| `jetson-edgellm-v091-qwen3ttsbase` | Qwen3-ASR | Qwen3-TTS Base | 需要音色克隆时 |
| `jetson-edgellm-v091-customvoice` | Qwen3-ASR | Qwen3-TTS CustomVoice | 预置音色 |
| `jetson-edgellm-v091-moss` | Qwen3-ASR | MOSS-TTS-Nano | 低延迟中文 |
| `jetson-edgellm-v091-sparktts` | — | SparkTTS 0.5B | 可控合成，W4A16 |
| `jetson-edgellm-v091-qwen3ttsbase-triple` | Qwen3-ASR | Qwen3-TTS Base | 三模型共驻（GDN 保持运行），Base TTFA 约 0.9–1.0s |
| `jetson-edgellm-v091-n2` | Qwen3-ASR b2 | CustomVoice | ⚠️ N=2 资格验证，**16GB 上会 OOM**，见下 |
| `jetson-edgellm-v091-qwen3ttsbase-isolated-n2` | Qwen3-ASR | Qwen3-TTS Base | ⚠️ 隔离 N=2 验证，**需先停 GDN** |
| `jetson-edgellm-v091-asr-validation` | Qwen3-ASR | — | 只验 ASR，引擎路径由验证环境显式提供 |

后三个是**验证用 profile，不要当产品配置**：

- `n2` 的 profile 说明里写死了一条禁令——在 16GB Orin NX 上同时加载两个 N=2 worker
  会触发 **kernel OOM eviction**，不要用它做 ASR+TTS 共驻
- `isolated-n2` 假设 GDN 已经停掉独占显存
- `asr-validation` 不自带引擎路径

产品部署用默认的 `matcha`，需要克隆音色用 `qwen3ttsbase`，需要 ASR+TTS+LLM 三模型
同时在线用 `qwen3ttsbase-triple`。

各 profile 的 model repo、immutable revision、payload hash 锁在
`deploy/artifacts/v091-release-lock.json`。

---

## 7. 客户端接入

服务是 OpenAI 兼容的，`http://<设备IP>:8621/v1`。

**填地址时用设备的 LAN IP，不要填 `127.0.0.1`**——如果调用方也在容器里，
`127.0.0.1` 指的是它自己。这是接入环节最常见的错误。

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.50:8621/v1", api_key="local")
with client.audio.speech.with_streaming_response.create(
    model="matcha-icefall-zh-en",
    voice="0",
    input="你好，欢迎使用本地语音服务。",
    response_format="wav",
) as response:
    response.stream_to_file("speech.wav")
```

**不要硬编码 model id 和 voice。** 每个 profile 暴露的模型名、音色、是否支持克隆、
并发上限都不一样，运行时从 `/v1/models` 和 `/v1/capabilities` 读。
例如 matcha profile 只有 voice `0`（Default）。

需要鉴权时设 `OVS_API_KEYS`，否则客户端 API key 随便填。

更完整的 API 说明（含流式细节）见 [`docs/deploy/jetson-orin-nx-v091.md`](../deploy/jetson-orin-nx-v091.md)。

---

## 8. 故障排查

| 症状 | 大概率原因 | 处理 |
|---|---|---|
| speech 容器 crash-loop，日志一堆 `undefined symbol` | 宿主缺 `libsentencepiece0` | [§3.1](#31-装宿主库-libsentencepiece0-️) |
| `up -d` 报容器名冲突 | 设备上有别的 compose project 占了同名容器 | `docker rm -f seeed-voice-v091`，卷不受影响 |
| 起来了但 8621 不通 | 还在下模型 | `docker logs seeed-voice-v091`，等 `/readyz` 返回 200，最多 15 分钟 |
| 下载卡住 / 极慢 | `HF_ENDPOINT` 没生效 | 进容器 `docker exec seeed-voice-v091 env \| grep HF_ENDPOINT` |
| edge-llm 启动就退出 | 引擎 hash / commit 校验失败 | 看日志里 `EXPECTED_PAYLOAD_SHA256` 报的差异，别绕过校验 |
| TTS 返回了 wav 但没声音 | 全零 PCM | 做 ASR 回环验证，别只看字节数 |
| 显存不足 | 在 8GB 设备上跑了拓扑 B，或选了 isolated profile 却没停 GDN | 换拓扑 A，或按 profile 说明停 GDN |

看日志的通用姿势：

```bash
docker logs --since 10m seeed-voice-v091 2>&1 | grep -iE 'error|fail|crash|undefined'
docker logs --since 10m edge-llm-chat-service-v091 2>&1 | grep -iE 'error|fail|expected'
```

---

## 9. 回滚

语音和 LLM 是**两个独立的回滚操作**，不要一起动。

语音回到上一代通用 Jetson 版本（用它自己独立的模型卷）：

```bash
deploy/install.sh --target jetson --pull --verify
```

LLM 的回滚目标、镜像 ID、以及"哪个镜像**不能**当回滚目标"，
见 [`docs/deploy/jetson-orin-nx-v091.md`](../deploy/jetson-orin-nx-v091.md#roll-back)。

**不要在不同 runtime / context profile 之间重命名或复制引擎缓存。**
v0.9.1、v0.8、4K、8K 的缓存是刻意不共享的。

---

## 10. 和 SenseCraft 方案平台的关系

客户侧交付一般不手工执行本文档，而是走方案平台的声明式部署：平台读方案仓库里的
`devices/*.yaml`（`type: docker_deploy`）→ SSH 到设备 → 传 compose → `up -d` → 轮询 `/readyz`。

那条路径**只消费本仓库的两个 compose 文件**，不使用 `agent/ovs_agent/apps/` 下的任何应用——
对话逻辑在客户自己的控制台侧，OVS 只作为 8621 端口的语音服务存在。

所以：

- 改 `deploy/docker-compose.edgellm-v091-*.yml` 会影响客户交付，属于对外接口
- 改 `agent/` 下的应用不影响这类交付

方案侧另外补了两个我们这边没有的前置动作（宿主库检查、容器名冲突检查），
其中宿主库那条已经收进本文档 [§3.1](#31-装宿主库-libsentencepiece0-️)。

## v0.11 默认语音入口

默认 Jetson 语音使用合并的 Qwen3-ASR IFB B2 与 Qwen3-TTS CustomVoice R16 候选 profile。回滚到旧版本时使用 `deploy/install.sh --target orin-nx --legacy --pull --verify` 或 `deploy/install.sh --target jetson --legacy --pull --verify`。候选状态和 LLM 独立 opt-in 保持不变。
