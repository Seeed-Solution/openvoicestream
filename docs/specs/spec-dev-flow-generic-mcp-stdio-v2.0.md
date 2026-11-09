# Dev-Flow Spec: Generic MCP Stdio Contract (v2.0)

## 1. 目标与背景

### 1.1 背景与问题定义
当前 MCP Stdio 模式对于执行参数存在硬编码限制：
1. **路径死绑**：强制限制在 `mcp_servers` 或 `/opt/voice` 目录，客户或外部部署将脚本置于其他目录（如 `/data/mcp/`、挂载卷路径或第三方工作目录）时会被无差别拦截；
2. **模块限制**：`-m` 模式受限于固定白名单 `ALLOWED_MCP_MODULES`，无法拉起标准已安装的第三方或自定义 Python MCP 模块（如 `mcp_server_sqlite`、`company_mcp`）；
3. **工作目录与同级加载缺失**：启动子进程时未将工作目录设置到脚本所在目录，导致脚本打开同级配置文件（如 `open("config.json")`）时失败；
4. **子进程生命周期隐患**：缺少进程组级终止机制；未对 `stderr` 做持续并发读取，在输出大量诊断日志时可能填满系统 64KB 管道缓冲区导致死锁。

### 1.2 目标与范围边界
对齐标准 MCP 宿主实现规范（如 VSCode / Claude Desktop / Home Assistant）：
1. **取消目录黑白名单**：解除目录路径写死限制，基于文件系统真实可读性校验（常规文件 + `os.access(R_OK)` + `.py` 脚本）；
2. **解除 `-m` 模块名硬编码**：允许符合 PEP 8 命名规范的任意合法 Python 模块标识符（Mode A）；
3. **确定性相对路径解析与 CWD 隔离**：相对路径统一基于 `CONFIG_DIR`（回退为 `os.getcwd()`）解析为绝对路径，并将子进程 `cwd` 设置为脚本所在目录；
4. **移除冗余环境变量篡改**：依赖 Python 原生机制（执行脚本时自动将脚本所在目录插入 `sys.path[0]`），不在子进程中二次改写 `PYTHONPATH`；
5. **健壮的进程组管理与 Stderr 排水**：通过 `start_new_session=True` 隔离进程组，终止时发送 `SIGTERM` -> 优雅超时 -> `SIGKILL` 级联清理；异步持续排水 `stderr` 防止管道死锁；
6. **结构化错误码与前端语法提示解耦**：引入机器可读错误码（`SCRIPT_NOT_FOUND`, `SCRIPT_NOT_READABLE`, `INVALID_MODULE_NAME`, `PATH_TRAVERSAL` 等），前端聚焦格式提示，后端权威裁决。

### 1.3 信任模型与安全边界说明 (Admin Trust Boundary)
- **管理操作员信任边界**：`/api/mcp/*` 管理接口受 `X-Admin-Token`（恒定时间比较）和 CSRF 校验保护。拥有该 Token 的操作员代表宿主系统授权管理员，拥有与服务主体同等的配置权限；
- **防护目标**：
  - 防止 Web 层的非授权命令注入（禁止 `shell=True`，使用 `create_subprocess_exec`）；
  - 防止 Shell 参数混淆与元字符注入（禁止 `-c`、禁止包含 `;`, `&`, `|`, `$`, `` ` `` 等参数）；
  - 防止动态链接器劫持攻击（拦截 `LD_PRELOAD`, `LD_LIBRARY_PATH`, `DYLD_*` 等危险环境变量）；
  - 限制运行载体仅为 Python 进程上下文（Mode A: `-m <module>` 与 Mode B: `<script.py>`），不开放执行任意未知系统二进制程序。

---

## 2. Taste 品味准则评估矩阵

| Taste 原则 | 现状（目录硬编码 + 模块白名单） | v2.0 通用 Python Stdio 契约 | 裁决结论 |
|---|---|---|---|
| **真实优先 (Real-world first)** | 仅支持 `/opt/voice` 与 `mcp_servers` 两个特异路径 | 支持任意挂载点脚本、相对路径、已装模块，适配多样化部署 | **优** |
| **变更有界与最小侵入** | 每次用户挂载新目录均需改动代码与正则 | 一次性契约化，后续新增脚本或模块零代码改动 | **优** |
| **符合使用者心智模型** | 用户必须猜测系统允许哪些目录，报错含糊 | 与直接在终端运行 `python <script.py>` 行为保持一致 | **优** |
| **拒绝隐式魔法与冗余抽象** | 企图在环境里暴力拼接 `PYTHONPATH` | 尊重 Python 官方 `sys.path[0]` 机制，仅通过 `cwd` 确保相对文件定位 | **优** |
| **闭环与故障隔离** | 缺少僵尸进程治理，stderr 溢出风险 | 进程组级精准终止与后台无阻塞流式排水，闭环严密 | **优** |

---

## 3. 架构设计与契约细则

### 3.1 运行模式定义 (Strict Python Scope)

子进程严格限定在以下两种模式之一：

- **Mode A (Python 模块运行)**:
  - `args[0] == "-m"`；
  - `len(args) >= 2`；
  - `args[1]` 必须匹配正则 `^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$`；
  - `args[2:]` 为传递给模块的普通运行参数（禁止包含 `-c`）。

- **Mode B (Python 脚本文件运行)**:
  - `args[0]` 以 `.py`（不区分大小写）结尾，且首字符不得为 `-`；
  - 相对路径解析：若 `args[0]` 为相对路径，基于 `CONFIG_DIR`（若未配置则使用 `os.getcwd()`）解析并转换为绝对路径 `resolved_path = os.path.abspath(os.path.join(base_dir, args[0]))`；
  - 文件系统校验：
    - `os.path.isfile(resolved_path)` 为真，否则抛出 `DashboardApiError(code="SCRIPT_NOT_FOUND", ...)`；
    - `os.access(resolved_path, os.R_OK)` 为真，否则抛出 `DashboardApiError(code="SCRIPT_NOT_READABLE", ...)`；
  - 规范化：校验成功后，`args[0]` 替换为计算出的真实规范化路径 `os.path.realpath(resolved_path)`；
  - `args[1:]` 为传递给脚本的普通参数（禁止包含 `-c`）。

### 3.2 子进程执行环境与工作目录契约

1. **可执行文件确定性**：
   - 必须运行已校验的物理 Python 可执行文件路径（如 `/opt/venv/bin/python3`），直接作为 `create_subprocess_exec` 的第一个可执行参数；
   - 严禁通过 `/proc/self/fd/<fd>` 间接启动，避免剥离 virtualenv 中的 `pyvenv.cfg` 上下文。
2. **工作目录 (Working Directory - CWD)**：
   - Mode B（脚本模式）：`cwd = os.path.dirname(resolved_script_path)`。这确保脚本内执行 `open("bridge.local.json")` 或载入同级模块时能够稳定正确定位；
   - Mode A（模块模式）：`cwd = CONFIG_DIR`（或 `os.getcwd()`）。
3. **环境变量契约**：
   - 保留现有的安全环境构建：继承系统安全基线、保留 `PATH` / `VIRTUAL_ENV`、允许 `PYTHONUNBUFFERED=1`；
   - 过滤所有以 `LD_`, `DYLD_` 开头的危险动态链接变量；
   - **不主动修改 `PYTHONPATH`**：由于 Python 在执行 `python <script>` 时原生会将 `os.path.dirname(script)` 加入 `sys.path[0]`，不进行额外的环境变量污染。

### 3.3 进程组生命周期与 Stderr 排水治理

1. **进程组隔离**：
   - 启动时设置 `start_new_session=True`（或 `preexec_fn=os.setsid`），使子进程成为独立进程组组长。
2. **优雅终止策略**：
   - 停止或测试超时终止时，首先获取子进程组 ID `pgid = os.getpgid(proc.pid)`；
   - 发送 `SIGTERM` 至整个进程组：`os.killpg(pgid, signal.SIGTERM)`；
   - 等待最多 1.5 秒；若进程未退出，强制发送 `SIGKILL`：`os.killpg(pgid, signal.SIGKILL)`；
   - `await proc.wait()` 确保回收僵尸进程。
3. **Stderr 异步排水 (Drain)**：
   - 子进程启动时设置 `stderr=asyncio.subprocess.PIPE`；
   - 启动后台异步任务读取 `stderr`（保留最近 4KB 用于失败诊断），避免因 stderr 缓冲区写满导致子进程阻塞挂起。

### 3.4 错误码契约 (Error Code Mapping)

| 场景 | 错误码 (Code) | HTTP 状态码 | 返回安全中文提示 (SAFE_CLIENT_MESSAGES) |
|---|---|---|---|
| Mode A 模块名格式不合法 | `INVALID_MODULE_NAME` | 400 | Python 模块名格式不正确，必须为合法的模块标识符 |
| Mode B 脚本文件不存在 | `SCRIPT_NOT_FOUND` | 400 | 指定的 Python 脚本文件不存在，请检查路径是否正确 |
| Mode B 脚本文件不可读 | `SCRIPT_NOT_READABLE` | 400 | Python 脚本文件无读取权限，请检查文件权限 |
| 包含非法参数（如 -c） | `ILLEGAL_ARGUMENT` | 400 | 包含不支持的参数格式，严禁使用 -c 参数 |
| 包含 Shell 元字符 | `SECURITY_VIOLATION` | 400 | 参数包含非法特殊字符，已拒绝执行 |
| 动态链接器环境变量拦截 | `FORBIDDEN_ENV` | 400 | 禁止设置动态链接器相关环境变量 |

---

## 4. 前端接口与校验简化 (`dashboard.js`)

1. **废弃硬编码路径正则**：
   - 移除 `/(?:^|\/)(?:mcp_servers|opt\/voice)\/[^\/]+\.py$/`；
2. **通用语法提示正则**：
   - Mode A 格式：`a0 === "-m"` 且 `args.length >= 2`；
   - Mode B 格式：`a0.endsWith(".py")` 且不以 `-` 开头；
   - 其余参数由后端进行权威文件系统与安全性校验，前端仅拦截显而易见的格式错误并给出明确的操作指导；
3. **错误提示对接**：
   - `MCP_ERROR_TEXT` 映射表完整对齐上述错误码。

---

## 5. 验收测试矩阵 (Verification Matrix)

1. **单元测试 (`test_mcp_dashboard_api.py`)**：
   - `test_stdio_mode_a_valid_module`: 验证合法的模块名（如 `mcp_server_sqlite`, `custom.tool`）通过校验；
   - `test_stdio_mode_a_invalid_module`: 验证非法模块名（如 `-c`, `module;rm`, `123bad`）拒绝且返回 `INVALID_MODULE_NAME`；
   - `test_stdio_mode_b_valid_script`: 验证存在的 `.py` 脚本成功解析为真实绝对路径并设置正确的 `cwd`；
   - `test_stdio_mode_b_missing_script`: 验证不存在的脚本返回 `SCRIPT_NOT_FOUND`；
   - `test_stdio_mode_b_unreadable_script`: 验证权限为 `0000` 的脚本返回 `SCRIPT_NOT_READABLE`；
   - `test_stdio_rejects_dash_c`: 验证任一位置传入 `-c` 均被拒绝；
   - `test_stdio_env_ld_preload_rejected`: 验证包含 `LD_PRELOAD` 等被拒绝；
   - `test_process_group_cleanup`: 验证 `killpg` 能够级联终止子进程派生的子任务。
2. **真机运行验证 (`rk3576-devkit-key:18000`)**：
   - 重新构建轻量镜像并热载入设备；
   - Playwright 自动化校验：
     - 测试现有 `/opt/voice/custom_mcp.py` 能成功响应 `test_connection`；
     - 测试动态修改并保存生效；
     - 确保页面白底风格与所有 169 个 DOM 节点无任何破坏性回归。
