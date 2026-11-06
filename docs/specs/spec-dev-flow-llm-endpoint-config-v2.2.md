# SenseCraft Local Voice 页面 LLM 服务端点配置规范 (v2.2)

> **目标**：
> 1. **支持在 Web 控制台直接配置 LLM 服务端点**：在设置抽屉「💬 提示词与模型」Tab 内新增独立「LLM 服务端点」配置区，支持查看、配置与持久化 `llm_backend`（引擎类型）、`llm_base_url`（服务地址）、`llm_model`（模型名称）、`llm_api_key`（API 密钥，安全脱敏）。
> 2. **提供在线连通性测试 (Test Connection)**：在保存或调试时，提供「测试连接」按钮，探针主动测试指定地址连通性与模型可用性，在 5.0 秒整体单一 deadline 内给出明确的模型匹配状态、延迟毫秒与诊断信息。
> 3. **严格凭据安全与目标绑定 (Full Origin & Scheme Binding)**：对比归一化的 `(scheme, hostname, effective_port)`，防止向不同域名或降级 HTTP 泄露密钥；禁用 HTTP 重定向；GET 接口严格掩码；所有日志脱敏；POST 强制 CSRF 校验。
> 4. **原子持久化与失败回滚契约**：严格“先落盘、后更新内存”顺序；保存失败立即回滚；缺失配置文件路径明确报错拒绝；复用 0600 权限与 fsync 机制；幂等保存准确维持既有 `pending_restart` 状态。
> 5. **视觉与无障碍全达标 (WCAG AA & SenseCraft White Theme)**：严格遵守控件边框 ≥ 3:1、焦点环 ≥ 3:1、文本 ≥ 4.5:1；100% 保持既有 169 个 DOM 核心 ID；全视口无横向溢出。
>
> **遵循准则**：`dev-flow`、`taste`（变更有界、按使用者心智模型建模、真实优先、通用优先于 Hack、零破坏性）。

---

## 1. 架构设计与用户交互体验

### 1.1 用户心智模型与界面布局
在设置抽屉「💬 提示词与模型」(`msTabPanel-prompt`) 内部进行严格的职责分离：
1. **模式对话与采样参数**（既有控件）：
   - `System prompt` 文本域 (`#msSystemPrompt`)
   - 高级折叠面板：`Temperature` (`#msTemperature`)、`Max history` (`#msMaxHistory`)、`Barge-in` (`#msBargeIn`)
   - 保持既有的 `msSave` 与 `msReset` 作用域仅限于上述参数，互不干扰。
2. **新增：LLM 服务端点与模型配置区 (`#llmEndpointSection`)**：
   - 结构独立于上面的 per-mode 表单，拥有专属保存与测试操作；
   - 标题：`LLM 服务端点与模型 (LLM Endpoint & Model)`，附带次级说明文字；
   - **引擎类型 (`#llmBackend`)**：下拉选择框，支持映射：
     - `edge_llm`：本地高性能推理（默认）
     - `openai_compat`：OpenAI 兼容协议（自动将后端已有的 `openai` 和 `openai_compat` 统一收敛至此项）
     - `noop`：空推理（仅调试测试）
   - **服务地址 (`#llmBaseUrl`)**：输入框，占位符 `http://localhost:8000/v1`；
   - **模型名称 (`#llmModel`)**：输入框，占位符 `qwen2.5-3b-instruct`；
   - **API 密钥 (`#llmApiKey`)**：带显示/隐藏切换按钮 (`#btnLlmApiKeyToggle`) 的密钥输入框；后端有非空 key 时展示为掩码 `●●●●●●`；
   - **操作与反馈条**：
     - `#btnLlmTest`：「测试连接」按钮；
     - `#btnLlmSave`：「保存端点配置」主按钮；
     - `#llmTestResult`：测试结果信息卡片（成功展示可用模型与延迟，失败展示诊断）；
     - `#llmRestartAlert`：修改成功后展示「⚠️ 端点配置已保存，需重启生效」，内嵌「立即重启」按钮 (`#btnLlmRestartNow`)。

### 1.2 SenseCraft 视觉规范与 WCAG AA 合规
- 容器背景：纯白 `#ffffff`，浅灰边框 `1px solid var(--border)`，圆角 `8px`；
- 输入框与下拉框边界：`1px solid var(--control-border)`（`#64748b`，实测 4.76:1 ≥ 3:1）；
- 占位符文本：`color: var(--placeholder)`（`#64748b`，实测 4.76:1 ≥ 4.5:1）；
- 聚焦外框：`outline: 2px solid var(--focus-ring); outline-offset: 2px;`（`#357f11`，实测 5.01:1 ≥ 3:1）；
- 主保存按钮 (`#btnLlmSave`)：背景 `#8CC020`，文本色 `#111827`（实测 8.17:1 ≥ 4.5:1），边框 `1px solid var(--accent-dark)`（`#5B7D15`，实测 4.72:1 ≥ 3:1，满足非文本控件边界标准）；
- 测试按钮 (`#btnLlmTest`)：背景 `var(--panel-2)`，文本色 `var(--text)`，边框 `1px solid var(--control-border)`（4.76:1 ≥ 3:1）；
- 状态提示色彩：
  - 成功：文本 `#166534`（实测 6.71:1 在 `#effce8` 衬底上），边框 `#cde69c`；
  - 错误：文本 `#b91c1c`（实测 5.92:1 在 `#fef2f2` 衬底上），边框 `#fca5a5`。

---

## 2. 后端 API 契约与严格安全设计

### 2.1 凭据安全决策表 (Exhaustive Credential Decision Table)

为防止将旧主机的密钥泄露给新主机，或因协议降级（HTTPS -> HTTP）发生泄露，系统定义归一化目标比较函数：
`origin_tuple(url) = (split.scheme.lower(), split.hostname.lower(), split.port or (443 if scheme == "https" else 80))`。
当且仅当目标地址的 `origin_tuple(new_url) == origin_tuple(saved_url)` 时，判定为 `dest_matched = True`。

| 输入 `llm_api_key` 取值 | 目标匹配状态 `dest_matched` | `POST /api/llm/config` 持久化行为 | `POST /api/llm/test` 探针行为 |
|---|---|---|---|
| **未传递 / `null` / `"●●●●●●"`** | `True` (相同 scheme+host+port) | **保持原密钥 (Retain)**：保留当前保存的 key | **复用原密钥 (Reuse)**：使用当前保存的 key |
| **未传递 / `null` / `"●●●●●●"`** | `False` (域名、端口或协议变更) | **自动清空 (Clear)**：重置为 `"EMPTY"`，防止越权绑定 | **无密钥 (No Key)**：严格使用 `"EMPTY"` 探测 |
| **`""` (空串) 或 `"EMPTY"`** | `True` 或 `False` | **显式清空 (Clear)**：重置为 `"EMPTY"` | **无密钥 (No Key)**：严格使用 `"EMPTY"` 探测 |
| **新字符串 `k` (`k != "●●●●●●"` 且 `k != "EMPTY"`)** | `True` 或 `False` | **更新密钥 (Replace)**：更新为 `k.strip()` | **使用传入密钥**：使用 `k.strip()` 探测 |

### 2.2 共享校验与 CSRF 契约
`POST /api/llm/config` 与 `POST /api/llm/test` 统一执行 `_validate_llm_payload(body)`：
1. **CSRF 强制校验**：校验 `Origin` 与 `Referer` 头 against `Host` 头，缺失或域名不匹配时立即抛出 `HTTP 403 Forbidden ("CSRF check failed")`；
2. **JSON 结构校验**：`body` 必须为 dict，非 dict 或畸形 JSON 抛出 `HTTP 400 Bad Request`；
3. **字段白名单**：仅允许 `["llm_backend", "llm_base_url", "llm_model", "llm_api_key"]`，包含未知键返回 `HTTP 400`；
4. **`llm_backend` 校验**：必须是字符串且位于 `("edge_llm", "openai", "openai_compat", "noop")`；
5. **`llm_base_url` 严格解析**：
   - 使用 `urllib.parse.urlsplit` 解析；
   - `scheme` 必须为 `http` 或 `https`；
   - `netloc` 必须非空且**禁止包含 userinfo（如 `http://user:pass@host`）**；
   - 端口（若指定）必须在 1..65535 范围内；
   - 自动去除末尾斜杠；
6. **`llm_model` 校验**：字符串，去除首尾空白后长度必须在 1..128 范围内；
7. **`llm_api_key` 校验**：必须是字符串或 `None`，长度 ≤ 512 字符。

### 2.3 专用端点定义与原子持久化

#### 1. `GET /api/llm/config`
- **响应**：
```json
{
  "ok": true,
  "llm_backend": "edge_llm",
  "llm_base_url": "http://localhost:8000/v1",
  "llm_model": "qwen2.5-3b-instruct",
  "has_api_key": false,
  "api_key_masked": "",
  "pending_restart": false
}
```
*注：`llm_api_key` 存在且非空/非 "EMPTY" 时，`has_api_key` 为 `true`，`api_key_masked` 严格返回 `●●●●●●`。`pending_restart` 准确反映当前是否已有未生效的端点修改。*

#### 2. `POST /api/llm/config`
更新并持久化 LLM 端点配置。
- **并发控制与持久化契约**：
  - 引入全局 `asyncio.Lock`（`_llm_config_lock`）保护并发写操作；
  - **路径可用性前置断言**：
    若当前 `cfg._source_path` 为空或不可用，**立即拒绝保存并返回 HTTP 400**：
    `{"ok": false, "error": "配置文件路径不可用，无法将端点改动持久化至磁盘以备重启"}`，绝不虚假承诺；
  - **幂等判断**：若输入值与当前配置无任何差异，直接返回：
    `{ "ok": true, "changed": [], "pending_restart": self._llm_config_pending_restart }`（准确保持既有状态）；
  - **先落盘后更新**：
    1. 按照决策表计算最终入库的各个字段值（处理凭据清空或更新）；
    2. 计算变更字段字典 `updated_fields`；
    3. 调用 `_persist_agent_settings_to_yaml(Path(src), updated_fields)` 进行原子写入（临时文件 -> 0600 权限验证 -> 原子替换 -> 目录 fsync）；
    4. 若磁盘写入发生任何异常，**立即终止，严禁更新内存中的 `cfg` 属性**，抛出错误并返回 HTTP 500 `{"ok": false, "error": f"持久化失败: {e}"}`；
    5. 只有磁盘持久化成功后，才同步更新内存运行态 `cfg` 对应字段；
    6. 设置 `self._llm_config_pending_restart = True` 并广播 `on_llm_config_change` 事件。
- **响应**：
```json
{
  "ok": true,
  "changed": ["llm_base_url", "llm_model"],
  "pending_restart": true,
  "key_cleared": false
}
```

#### 3. `POST /api/llm/test`
测试指定 LLM 端点连通性（不修改系统配置）。
- **凭据防护边界**：
  - 严格根据 2.1 决策表判定是否携带密钥；
  - HTTP 请求严格指定 `allow_redirects=False`，禁止跟随重定向；
  - 任何异常信息脱敏后再输出。
- **5.0 秒统一总体截止时间 (Single Overall Deadline)**：
  - 使用外层 `asyncio.timeout(5.0)` 包裹整个测试流程（覆盖 DNS、建连、`/models` 探测及潜在的 `/chat/completions` 回退请求全过程，整体耗时严格 ≤ 5.0s）；
  - 若 `llm_backend == "noop"`：立即返回 `{ "ok": true, "status": "noop", "latency_ms": 0, "message": "noop 后端：跳过连通性探测" }`；
  - 若为 `edge_llm` 或 `openai` / `openai_compat`：
    1. 向 `{llm_base_url}/models` 发送 GET 请求（附带 `Authorization: Bearer <key>`，若 key 非空且非 EMPTY）；
    2. 限制响应体最大读取 64KB；
    3. 若返回 HTTP 200：
       - 解析 JSON；检查目标 `llm_model` 是否在可用列表中：
         - 若存在：返回 `{ "ok": true, "status": "available", "latency_ms": X, "message": f"连接成功，模型 {llm_model} 可用" }`；
         - 若不存在：返回 `{ "ok": false, "status": "model_not_found", "latency_ms": X, "error": f"端点连接成功，但模型列表中未找到 {llm_model}" }`；
    4. 若 `/models` 返回 HTTP 404：
       - 向 `{llm_base_url}/chat/completions` 发送最小测试包：
         `{"model": llm_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}`；
       - 限制响应体最大 64KB；
       - 若返回 HTTP 200 且包含非空 `"choices"` 列表，返回 `{ "ok": true, "status": "available", "latency_ms": X, "message": f"连接成功，模型 {llm_model} 响应正常" }`；
    5. 捕获超时、连接拒绝、HTTP 401（认证失败）等并返回人类友好的错误提示。

---

## 3. 变更范围与清单 (File Manifest & Implementation Anchors)

严格限定在以下 5 个文件，严禁超出范围：

1. `agent/ovs_agent/plugins/debug_dashboard.py`：
   - 路由挂载：`debug_dashboard.py:222-225`（注册 `/api/llm/config` 与 `/api/llm/test`）；
   - 接口实现：`debug_dashboard.py:1570-1760`（实现 `_api_llm_config_get`、`_api_llm_config_post`、`_api_llm_test_post` 及校验逻辑）。
2. `agent/ovs_agent/plugins/static/dashboard.html`：
   - 表单结构：`dashboard.html:170-205`（在 `#msTabPanel-prompt` 内部 `#msAdvancedBody` 下方新增 `#llmEndpointSection`）。
3. `agent/ovs_agent/plugins/static/dashboard.css`：
   - 样式定义：`dashboard.css:385-420`（新增 `.llm-endpoint-card`、控件边框、密码切换按钮与测试反馈卡片样式）。
4. `agent/ovs_agent/plugins/static/dashboard.js`：
   - 前端逻辑：`dashboard.js:230-310`（在打开抽屉时加载 LLM 配置、绑定测试与保存事件、脱敏掩码流转及重启提示）。
5. `agent/tests/test_dashboard_api.py`：
   - 测试用例：在末尾新增 `test_llm_config_get_and_post_persistence`、`test_llm_test_probe_and_security_boundary` 回归用例。

### 3.2 现有契约与受保护基线保持
- **169 个原始 DOM ID 100% 保持**：经 `.cache/review/html_ids_inventory.json` 校验，原始 169 个 ID 一个不删，数量严格等于 1；
- **明确新增 DOM ID 白名单**：
  `llmEndpointSection`, `llmBackend`, `llmBaseUrl`, `llmModel`, `llmApiKey`, `btnLlmApiKeyToggle`, `btnLlmTest`, `btnLlmSave`, `llmTestResult`, `llmRestartAlert`, `btnLlmRestartNow`（共 11 个，全部保持唯一）；
- **单元测试收集基线事实**：
  当前测试套件收集 **38 个测试项**（`test_dashboard_api.py` 收集 17 项，包含 16 个测试函数与 1 个别名 `test_agent_settings_get_returns_three_fields = test_agent_settings_get_returns_six_fields`；`test_mcp_dashboard_api.py` 收集 21 项），全部保持 100% 通过；
- **原有交互状态机无干扰**：`msSave` / `msReset` 仅针对当前 mode 的 prompt/temperature/history/barge-in 参数，与 LLM 全局端点保存严格解耦。

---

## 4. 验证与验收矩阵 (Verification Matrix)

| 编号 | 检查项 | 验证方式与断言标准 |
|---|---|---|
| **V1** | **原始 169 个 DOM ID 完整性** | 读取 `.cache/review/html_ids_inventory.json`，断言 169 个原始 ID 每一个在 DOM 中**存在且数量严格等于 1**；断言新增 11 个 ID 存在且数量等于 1。 |
| **V2** | **GET /api/llm/config 安全脱敏** | 配置了明文 API Key 时，GET 返回 `has_api_key: true` 且 `api_key_masked: "●●●●●●"`，响应体任何字段均不含明文 Key。 |
| **V3** | **POST /api/llm/config 校验、回滚与幂等** | 1. 传入非法 backend 或非法 URL（带 `user:pass@` 或非 http/https）返回 400；<br>2. 模拟磁盘只读或 fsync 异常，断言返回 500 且内存 `cfg` 不被篡改修改；<br>3. 模拟 `_source_path` 缺失，断言返回 400 明确拒绝；<br>4. 正常修改后断言 YAML 文件被原子更新且文件权限严格为 0600；<br>5. 重复无变更保存时，断言准确保留已有 `pending_restart` 状态且不重复写盘；<br>6. 目标 host/port/scheme 改变且未提供新 key 时，断言 key 自动重置为 EMPTY（防止越权绑定与降级泄露）。 |
| **V4** | **POST /api/llm/test 凭据防护与 5s 整体单一截止时间** | 1. 传入不同 host/port 或协议降级（HTTPS -> HTTP）且不带 key 时，断言**绝不复用原有 saved key**；<br>2. 探针遇到 301/302 重定向时，断言跟随被禁止（`allow_redirects=False`）；<br>3. 针对挂起无响应地址，断言整个 probe 请求在 5.0s 内由单一 deadline 超时终止并返回结构化错误；<br>4. 探针测试前后，系统的活跃配置与 YAML 文件均未被篡改。 |
| **V5** | **CSRF 强制防御** | 对 `POST /api/llm/config` 与 `POST /api/llm/test` 发起缺失 Origin/Referer 或域名不匹配请求，断言返回 403 Forbidden。 |
| **V6** | **WCAG AA 视觉与控件边界对比度** | 1. `#btnLlmSave` 主按钮绿色背景上的文字对比度 ≥ 4.5:1，按钮边框 (`--accent-dark`) 对比度 ≥ 3:1；<br>2. 输入框边框与 `#btnLlmTest` 边框对比度 ≥ 3:1；<br>3. 焦点环可见度 ≥ 3:1；<br>4. 测试成功信息与失败错误信息文本对比度均 ≥ 4.5:1。 |
| **V7** | **全视口自适应与无横向溢出** | 在 1920/1440/1280/1024/900/899/768/375px 下展开设置抽屉，断言端点表单不挤压换行异常，页面横向溢出严格为 0。 |
| **V8** | **后端全量单元测试 100% 通过** | 在 `agent/` 目录下执行 `uv run pytest tests/test_dashboard_api.py tests/test_mcp_dashboard_api.py -v`，全部既有用例与新增用例 100% 通过。 |
