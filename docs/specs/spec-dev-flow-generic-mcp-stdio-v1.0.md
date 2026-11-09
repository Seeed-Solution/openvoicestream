# Dev-Flow Spec: Generic MCP Stdio Contract (v1.0)

## 1. 目标与背景

### 1.1 背景
此前针对客户自定义 MCP 服务（`custom_mcp.py`）挂载在 `/opt/voice/` 的情况，临时通过在前端正则与后端路径检查中硬编码 `/opt/voice/*.py` 进行了放行。
这种硬编码严重违背了通用性原则：
1. 若其他客户将脚本挂载在 `/data/mcp/`、`/app/` 或外部目录，系统将再度被拦截；
2. `-m` 模式被死硬限制在 `ALLOWED_MCP_MODULES`（仅允许 `ha_mcp` 等几个特定名称），用户通过 pip 安装的标准 MCP 服务（如 `mcp-server-sqlite`, `company_mcp` 等）无法通过 `-m` 拉起；
3. 子进程未自动将脚本所在目录加入 `PYTHONPATH`，导致脚本无法自然加载同级目录下的本地配置或依赖文件。

### 1.2 目标
对齐业界主流 MCP Client 规范（Claude Desktop / VSCode / Home Assistant），建立**通用 Executable & Script 契约**：
1. **取消路径目录黑白名单**：解除 `mcp_servers` 与 `/opt/voice` 目录限制。改为文件系统真实性与可读性校验（常规文件 + `os.access(R_OK)` + `.py` 脚本或可执行文件）；
2. **解除 `-m` 模块名硬编码**：支持任意符合 Python 语法规范的合法模块名（`^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$`）；
3. **自动同级 `PYTHONPATH` 注入**：脚本模式启动子进程时，自动将 `os.path.dirname(script_path)` 追加至 `PYTHONPATH`；
4. **前端解耦放行**：前端表单验证仅校验语法与模式（`-m <module>` 或合法路径格式），错误提示清晰友好，最终有效性交由后端权威校验；
5. **坚守安全边界**：不引入 `shell=True`，严禁 `-c` 代码求值与 Shell 元字符，保留 `X-Admin-Token` 与 CSRF 保护，保留动态链接器环境变量拦截。

---

## 2. Taste 品味准则评估矩阵

| Taste 原则 | 当前硬编码方案（现状） | 通用 Executable & Script 契约（方案 1） | 评估结论 |
|---|---|---|---|
| **真实优先 (Real-world first)** | 只覆盖 `/opt/voice` 这一种临时挂载路径 | 支持容器挂载、本地绝对路径、模块名拉起等所有真实世界形态 | **优** |
| **变更有界与最小侵入** | 每次增加路径都需要侵入修改前端正则与后端校验 | 抽象为统一的文件权限校验与模块名校验，一次到位，后续零侵入 | **优** |
| **符合使用者心智模型** | 用户必须猜测系统允许哪些目录和模块名，极易挫败 | 用户传入现有文件路径或模块名即可运行，与本地跑 python 命令心智一致 | **优** |
| **闭环由执行者负责** | 执行者假设系统知道所有合法目录 | 统一提供精准的错误码与友好中文错误提示（如文件不存在/不可读/模块格式非法） | **优** |
| **拒绝写死规则** | 写死 `/opt/voice` 和 `ALLOWED_MCP_MODULES` | 完全参数化与契约化，无业务路径死编码 | **优** |

---

## 3. 架构设计与接口契约

### 3.1 后端改动清单 (`agent/ovs_agent/plugins/mcp_manager.py`)

#### 3.1.1 模块名校验 (`validate_stdio_args` - Mode A)
- 废弃 `ALLOWED_MCP_MODULES` 固定集合；
- 引入合法 Python 模块标识符正则：
  ```python
  PYTHON_MODULE_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")
  ```
- 当 `args[0] == "-m"` 时：
  - 检查 `len(args) >= 2`；
  - 检查 `PYTHON_MODULE_REGEX.match(args[1])`；若不匹配，抛出 `SecurityError(f"Invalid Python module name: {args[1]}")`。

#### 3.1.2 脚本路径校验 (`validate_stdio_args` - Mode B)
- 当 `args[0] != "-m"` 时：
  - 禁止以 `-` 开头；
  - 允许两种有效形式：
    - 脚本模式：以 `.py` 结尾；
    - 二进制模式：文件具有可执行权限（`os.access(path, os.X_OK)`）；
  - 路径解析优先级：
    1. 绝对路径：直接 `os.path.realpath(args[0])`；
    2. 相对路径：优先基于 `project_root`（或 `CONFIG_DIR`，或当前工作目录）解析，若不存在则基于当前工作目录解析；
  - 存在性与权限校验：
    - `os.path.isfile(script_path)`：必须是真实存在的常规文件；
    - `os.access(script_path, os.R_OK)`：当前进程必须具有读权限；
    - 废弃 `rel.startswith("..")` 目录限制。

#### 3.1.3 同级 `PYTHONPATH` 自动注入 (`build_mcp_env`)
- 在 `build_mcp_env(venv_bin, extra_env=None, script_dir: str | None = None)` 中：
  - 若传入了 `script_dir` 且为有效目录：
    - 将 `script_dir` 插入到 `PYTHONPATH` 的首位；
    - 若已有 `PYTHONPATH`，以 `os.pathsep` 拼接；若无，直接创建。

#### 3.1.4 子进程启动调整 (`launch_stdio_mcp_subprocess`)
- 解析出的 `script_path` 若存在，提取 `script_dir = os.path.dirname(script_path)`；
- 传给 `build_mcp_env(..., script_dir=script_dir)`；
- 依旧使用已校验的 Python 物理路径 `proc_exe` 启动子进程，确保 virtualenv 正常识别。

---

### 3.2 前端改动清单 (`agent/ovs_agent/plugins/static/dashboard.js`)

#### 3.2.1 脚本与模块正则解耦
- 废弃 `MCP_SCRIPT_RE = /(?:^|\/)(?:mcp_servers|opt\/voice)\/[^\/]+\.py$/;`；
- 替换为通用的语法与路径检查正则：
  ```javascript
  const MCP_MODULE_RE = /^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$/;
  const MCP_SCRIPT_PATH_RE = /^([a-zA-Z]:[\\/]|~?\/|[.\w-]+[\\/])?[^<>:"|?*\n\r]+\.py$/i;
  ```
- 校验逻辑：
  ```javascript
  const a0 = p.args[0] || "";
  const a1 = p.args[1] || "";
  let argsOk = false;
  if (a0 === "-m") {
    argsOk = p.args.length >= 2 && MCP_MODULE_RE.test(a1);
    if (!argsOk) {
      return { error: "args 第一行为 -m 时，第二行需为合法的 Python 模块名（如 my_mcp_server）", field: mcpArgs };
    }
  } else {
    argsOk = MCP_SCRIPT_PATH_RE.test(a0);
    if (!argsOk) {
      return { error: "args 第一行需为 -m 模块名，或以 .py 结尾的脚本文件路径", field: mcpArgs };
    }
  }
  ```

---

## 4. 安全不变量 (Security Invariants)

1. **绝对禁止 Shell 执行**：始终使用 `asyncio.create_subprocess_exec`，不得调用 `shell=True` 或拼装 shell 字符串；
2. **严格禁止代码求值参数**：严禁 `args` 包含 `-c` 前缀或 Shell 元字符（`|`, `;`, `&`, `$`, `` ` `` 等）；
3. **危险环境变量拦截**：动态链接器劫持变量（`LD_PRELOAD`, `LD_LIBRARY_PATH`, `DYLD_*` 等）依然被严格拦截；
4. **管理端点全鉴权**：`/api/mcp/*` 维持 `X-Admin-Token`（恒定时间比较）与 CSRF（Origin/Referer）强保护；
5. **配置落盘原子性与权限**：继续保持 `0600` 临时文件 + fsync + 原子替换机制。

---

## 5. 端到端真实验证用例 (E2E Verification Gate)

1. **单元测试与回归套件**：
   - 验证 Mode A 模块模式：合法模块名（`my_tool`, `pkg.sub.mcp`）通过，非法字符拒绝；
   - 验证 Mode B 脚本模式：任意存在且可读的 `.py` 路径通过；不存在的文件拒绝；不可读文件拒绝；非 `.py` 非可执行拒绝；
   - 验证同级 `PYTHONPATH` 注入：测试脚本能够直接 `import helper`（位于同目录）；
   - 全部 40+ 项 pytest 自动化测试通过。
2. **真机运行验证 (`rk3576-devkit-key:18000`)**：
   - 部署到 `rk3576-devkit-key`；
   - Playwright 执行控制台端到端测试；
   - 验证客户挂载在 `/opt/voice/custom_mcp.py` 的平台工具一键测试成功；
   - 验证客户配置能自由编辑、保存并生效。
