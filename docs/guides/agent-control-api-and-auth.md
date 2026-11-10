# OpenVoiceStream (OVS) Agent 控制接口与 Token 鉴权指南

本文档整理了 OpenVoiceStream Agent 当前提供的所有控制接口（包含唤醒、休眠、主动播报、模式切换等），以及如何开启和调用 Token 鉴权。

---

## 一、Token 鉴权概述

为了防止设备在局域网或公网暴露时被未授权调用，Agent 控制层提供了统一的 **预共享密钥（Pre-Shared Key / 静态 API Key）** 鉴权机制：

- **受保护范围**：所有以 `/api/control/*` 开头的控制类端点（唤醒、休眠、发声、重启等）；
- **向后兼容机制**：若未配置密钥，鉴权默认处于**关闭状态**，所有接口免密直通（适用于本地开发与内网受信环境）；
- **安全保障**：底层采用 `secrets.compare_digest` 恒定时间防时序攻击比对，支持 UTF-8 字节编码转换，防恶意扫描与异常拒绝服务。

---

## 二、如何开启 Token 鉴权

由运维或部署人员在启动设备前配置自定义密钥。系统支持以下两种配置方式（任选其一，环境变量优先级更高）：

### 方式 A：环境变量配置（推荐）

在容器运行环境或 `docker-compose.yml` 的 `environment` 中添加：

```yaml
environment:
  - OVS_CONTROL_API_KEY=your_secret_token_123456
```

> **提示**：支持逗号分隔配置多个有效 Key，例如：`OVS_CONTROL_API_KEY="key1, key2, key3"`。

### 方式 B：持久化配置文件

在 Agent 配置文件 `agent-config.yaml`（例如挂载在宿主机的 `./conversation/agent-config.yaml` 或容器内的 `/opt/voice/config/agent-config.yaml`）中指定：

```yaml
metadata:
  control_api_key: "your_secret_token_123456"
```

> **生成高强度安全 Token 的命令**：
> ```bash
> openssl rand -hex 16
> # 输出示例：4a8f9c0e1b2d3f4a5c6e7b8d9a0f1c2e
> ```

---

## 三、如何调用带鉴权的接口

开启鉴权后，调用接口需在 HTTP Header 中附带凭证（任选其一）：

1. **标准 Bearer Token（推荐）**：
   ```http
   Authorization: Bearer <your_secret_token>
   ```
2. **自定义 Header**：
   ```http
   X-OVS-Control-Token: <your_secret_token>
   ```

若未携带或 Token 错误，接口将统一返回：
```json
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{
  "ok": false,
  "error": "unauthorized"
}
```

### Web 浏览器端如何访问？
控制台（Dashboard）页面默认不将 Token 输出到公开的 HTML 中以防泄露。如需在浏览器中通过图形界面操作控制按钮，可通过 URL 参数传递凭证：
```text
http://<设备IP>:18000/?control_token=<your_secret_token>
```
前端会自动将其安全保存至浏览器的 `localStorage` 中，后续点击页面上的“唤醒”、“休眠”等按钮会自动附带认证头。

---

## 四、受保护的控制接口完整清单 (`/api/control/*`)

> **默认服务端口**：`18000`（由 `metadata.dashboard_port` 配置）。

### 1. 设备唤醒与休眠（状态机控制）

#### `POST /api/control/wake` — 远程唤醒设备
* **作用**：将 Agent 状态从 `SLEEPING`（休眠）切换为 `IDLE`（收音监听），并触发唤醒流程。
* **具体行为**：
  1. 状态机切至 `IDLE`；
  2. 若配置了 `wake_tone`，扬声器播放唤醒蜂鸣音；
  3. **主动问候**：若配置了 `wake_reply_text`（例如 `"在呢"`），Agent 会通过喇叭主动朗读；
  4. 开启麦克风监听窗口，等待用户后续下达语音指令。
* **调用示例**：
  ```bash
  curl -X POST http://<设备IP>:18000/api/control/wake \
       -H "Authorization: Bearer your_secret_token_123456"
  ```
* **响应**：`{"ok": true}`

#### `POST /api/control/sleep` — 强制休眠设备
* **作用**：将 Agent 状态从 `IDLE` 切换为 `SLEEPING`。
* **具体行为**：关闭实时收音，忽略环境噪声与普通说话，直至下次被唤醒词或唤醒接口唤醒。
* **调用示例**：
  ```bash
  curl -X POST http://<设备IP>:18000/api/control/sleep \
       -H "Authorization: Bearer your_secret_token_123456"
  ```
* **响应**：`{"ok": true}`

#### `POST /api/control/ptt/start` — 按住说话（Push-to-Talk）开始
* **作用**：手动触发开启收音。

#### `POST /api/control/ptt/end` — 按住说话（Push-to-Talk）结束
* **作用**：手动结束收音，立即将已录制音频送入 ASR 识别。

---

### 2. 主动发声与对话注入（事件驱动与通知）

#### `POST /api/control/send_text` — 主动文字朗读（直接 TTS）
* **作用**：让设备直接朗读指定的文字，**不经过大模型 LLM**，不污染上下文对话历史。
* **适用场景**：传感器告警、定时提醒、门铃通知、状态播报。
* **请求体**：
  ```json
  {
    "text": "检测到门外有访客，请注意查看。"
  }
  ```
* **调用示例**：
  ```bash
  curl -X POST http://<设备IP>:18000/api/control/send_text \
       -H "Content-Type: application/json" \
       -H "Authorization: Bearer your_secret_token_123456" \
       -d '{"text": "检测到门外有访客，请注意查看。"}'
  ```
* **响应**：`{"ok": true}`

#### `POST /api/control/inject_user_text` — 模拟用户对话注入（触发 LLM）
* **作用**：绕过实体麦克风，直接向系统合成一条用户的提问，**触发完整的 LLM 思考 -> MCP 工具调用 -> TTS 流式语音回复**。
* **适用场景**：系统事件驱动的主动对话（例如主人回家时，主动让大模型组织语言打招呼）。
* **请求体**：
  ```json
  {
    "text": "系统事件：检测到主人回家，请主动向主人问好并汇报今日室温。"
  }
  ```
* **调用示例**：
  ```bash
  curl -X POST http://<设备IP>:18000/api/control/inject_user_text \
       -H "Content-Type: application/json" \
       -H "Authorization: Bearer your_secret_token_123456" \
       -d '{"text": "系统事件：检测到主人回家，请主动向主人问好。"}'
  ```

#### `POST /api/control/inject_wav` — 注入原始 WAV 音频
* **作用**：直接向系统输入一段二进制 PCM/WAV 音频（需设置环境变量 `OVS_AGENT_DEBUG_INJECT=1`）。

---

### 3. 对话打断与进程管理

#### `POST /api/control/abort` — 立即打断
* **作用**：打断当前正在进行的语音播放，或中止正在生成的 LLM 推理。
* **调用示例**：
  ```bash
  curl -X POST http://<设备IP>:18000/api/control/abort \
       -H "Authorization: Bearer your_secret_token_123456"
  ```

#### `POST /api/control/restart_mic` — 重启麦克风采集
* **作用**：当热插拔 USB 麦克风或声卡异常时，原地重连音频输入流。

#### `POST /api/control/restart_agent` — 重启 Agent 服务
* **作用**：软重启 Agent 内部各任务调度与插件。

#### `POST /api/control/reconnect` — 重连语音流底层服务
* **作用**：强制重新连接后端 WebSocket 语音流管道（SLV）。

---

### 4. 运行模式切换

#### `POST /api/control/mode` — 切换 Agent 运行模式
* **作用**：动态切换 Agent 的预设模式（如闲聊助手、工位质检模式、会议记录模式）。
* **请求体**：
  ```json
  {
    "mode": "inspection"
  }
  ```

---

## 五、附录：无需 Control Token 的接口（只读/监控）

以下端点主要用于前端看板状态展示，不直接操控硬件动作，无需携带该 Control Token：

| 接口 | 方法 | 说明 |
| :--- | :--- | :--- |
| `/` | `GET` | Dashboard Web 首页 HTML |
| `/ws` | `GET` | WebSocket 实时音频波形与状态推送 |
| `/api/session/history` | `GET` | 查看当前会话历史对话文本 |
| `/api/tts/speakers` | `GET` | 获取当前可用的 TTS 发音人列表 |
| `/api/modes` | `GET` | 获取当前设备支持的运行模式列表 |
| `/api/mcp/*` | `*` | MCP 工具管理（由单独的 `X-OVS-Admin-Token` 隔离保护） |

---

## 六、Python 自动化调用示例

```python
import requests

DEVICE_IP = "192.168.1.100"
PORT = 18000
TOKEN = "your_secret_token_123456"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# 1. 唤醒设备
resp = requests.post(f"http://{DEVICE_IP}:{PORT}/api/control/wake", headers=headers)
print("Wake response:", resp.status_code, resp.json())

# 2. 主动让设备说一句话
payload = {"text": "您好，设备已启动完成。"}
resp = requests.post(f"http://{DEVICE_IP}:{PORT}/api/control/send_text", headers=headers, json=payload)
print("Speak response:", resp.status_code, resp.json())
```
