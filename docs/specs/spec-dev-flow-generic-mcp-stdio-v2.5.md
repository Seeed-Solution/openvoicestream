# Dev-Flow Spec: Generic MCP Stdio Contract (v2.5)

## 1. 目标与背景

### 1.1 背景与问题定义
当前 MCP Stdio 模式对于执行参数存在硬编码限制与生命周期治理缺口：
1. **路径死绑**：强制限制在 `mcp_servers` 或 `/opt/voice` 目录，客户或外部部署将脚本置于挂载卷、数据卷（如 `/data/mcp/`）时会被无差别拦截；
2. **模块限制**：`-m` 模式受限于固定白名单 `ALLOWED_MCP_MODULES`，无法拉起标准已安装的第三方或自定义 Python MCP 模块（如 `mcp_server_sqlite`、`company_mcp`）；
3. **测试与运行不一致 (Parity Gap)**：Dashboard 的 `test_server` 与 live agent 的 `McpBridgePlugin` / `McpConnection` 各自为政，live 连接直接使用原始配置且未设置 `cwd`，导致脚本打开同级配置（如 `open("bridge.local.json")`）在真机启动时失败；
4. **子进程生命周期治理与终态观察缺口**：
   - 依赖父进程存活状态触发退出清理，若主进程提前退出，子孙进程会失去平滑退出的优雅期；
   - 清理协程在多重或连续取消（Repeated Cancellation）下若直接使用未加屏蔽的 `wait_for`，取消信号会穿透并取消底层清理协程，导致 `SIGKILL` 无法发出；多重取消下容易覆盖首个取消异常，或在清理失败时覆盖取消语义；
   - 最终 reap 超时若直接遗弃，会导致孤儿任务无主挂起；后台任务若仅简单从集合移除而不提取 `exception()`，会引发 `Task exception was never retrieved`；
   - `proc.stderr` 未做并发流式排水，大量日志会填满 64KB 操作系统管道缓冲区导致死锁；
5. **虚拟环境隔离与符号链接解析冲突**：若将 `sys.executable` 盲目解析为物理 `realpath`（如 `/usr/bin/python3.10`），会剥离 `pyvenv.cfg` 上下文，丢失 virtualenv 中安装的 `mcp` 依赖。

### 1.2 目标与范围边界
对齐标准 MCP 宿主实现规范（VSCode / Claude Desktop / Home Assistant）：
1. **取消目录黑白名单**：解除目录路径写死限制，基于文件系统真实可读性校验（常规文件 + `os.access(R_OK)` + `.py` / `.PY` 脚本）；
2. **解除 `-m` 模块名硬编码**：允许符合 PEP 8 命名规范的任意合法 Python 模块标识符（Mode A）；
3. **测试/运行时统一规范化 (Parity)**：提取共享规范化函数 `normalize_stdio_mcp_config`，在验证、测试连接与 live 运行全链路统一执行路径规范化、解析 `config_dir` 并设置 `cwd` 为脚本所在目录；
4. **保持 Virtualenv 解释器调用契约**：安全检查可对解析后目标进行鉴权，但实际执行命令必须保留 virtualenv 解释器路径，确保 `pyvenv.cfg` 正确生效；
5. **健壮的独立进程组管理与生命周期闭环监督**：
   - 启动时标记 `pgid = proc.pid`（基于 `start_new_session=True`）；
   - 基于 `os.killpg(pgid, 0)` 探测进程组存活，无论父进程是否先退出，均保障进程组享有完整的 1.5s `SIGTERM` 优雅期，超时后无条件升级 `SIGKILL`；
   - 结构化回收跟踪与未决 Reap 任务托管：若主进程未在 1.0s 内完成回收，派生受托管的后台收割任务 `_supervise_pending_reap` 并注册到 `_PENDING_REAP_TASKS`，持续跟踪直至终态；
   - 抗重入取消与终态异常消费：保留首个捕获的原始取消异常；若清理异常发生，将其附加到首个取消异常上下文或记录；后台托管任务完成回调必须显式调用 `t.exception()` / `t.result()` 消费终态，杜绝未观测异常逃逸；
   - 异步并发读取 `stderr` 管道（环形缓冲保留最近 4KB 并脱敏），显式区分取消与读取异常，防止管道死锁；
6. **结构化错误码与前端语法提示解耦**：引入机器可读错误码并保持前后端一致映射，前端不阻止合法文件名大小写（支持 `.py` / `.PY`）。

### 1.3 信任模型与安全边界说明 (Admin Trust Boundary)
- **管理授权基线**：`/api/mcp/*` 管理接口受 `X-Admin-Token`（恒定时间比较）和 CSRF 校验保护。**拥有该 Token 的操作员代表宿主系统授权管理员，拥有与服务主体同等权限，Token 授权在当前宿主服务账户的权限范围内执行任意 Python 代码。**
- **非沙箱声明**：Python-only（Mode A 与 Mode B）是**启动契约与参数结构化规范**，而不是安全沙箱。管理员传入的脚本文件、导入模块、符号链接目标及可写目录均被视为可信输入。在已确立管理员可信边界的前提下，移除硬编码路径白名单不构成未授权提权。
- **防护目标**：
  - 防止 Web 层的非授权命令注入（禁止 `shell=True`，使用 `create_subprocess_exec`）；
  - 防止 Shell 参数混淆与元字符注入（禁止 `-c`、禁止包含 `;`, `&`, `|`, `$`, `` ` `` 等参数）；
  - 防止动态链接器劫持攻击（拦截 `LD_PRELOAD`, `LD_LIBRARY_PATH`, `DYLD_*` 等危险环境变量）；
  - 限制运行载体仅为 Python 进程上下文（Mode A: `-m <module>` 与 Mode B: `<script.py>`），不开放执行任意未知系统二进制程序。

---

## 2. Taste 品味准则评估矩阵

| Taste 原则 | 现状（目录硬编码 + 模块白名单） | v2.5 通用 Python Stdio 契约 | 裁决结论 |
|---|---|---|---|
| **真实优先 (Real-world first)** | 仅支持 `/opt/voice` 与 `mcp_servers` 两个特异路径 | 支持任意挂载点脚本、相对路径、已装模块，适配多样化部署 | **优** |
| **测试与运行时对齐 (Parity)** | Dashboard 测试与后台 Live 连接两套逻辑，易出现测试通但运行挂 | 共享相同的参数规范化、CWD 解析和环境构建逻辑 | **优** |
| **变更有界与最小侵入** | 每次用户挂载新目录均需改动代码与正则 | 一次性契约化，后续新增脚本或模块零代码改动 | **优** |
| **行为确定性与透明度** | 工作目录隐式依赖启动父进程，路径行为不可预测 | 脚本模式明确以脚本所在目录为 `cwd`，相对路径解析确定 | **优** |
| **生命周期终态强闭环** | 缺少僵尸进程治理，未观测异常逃逸 | 进程组级精准终止、后台收割托管、终态异常 100% 显式消费 | **优** |

---

## 3. 架构设计与契约细则

### 3.1 运行模式与校验优先级 (Validation Precedence)

参数校验统一按以下严格优先级执行：

1. **基本结构检查**：`args` 必须为非空字符串列表，若为空抛出 `DashboardApiError("MISSING_ARGUMENT", ...)`；
2. **禁止参数与特殊字符**：
   - 遍历所有参数，若任何参数包含 Shell 元字符（`;`, `&`, `|`, `$`, `` ` ``, `\n`, `\r`），抛出 `DashboardApiError("SECURITY_VIOLATION", ...)`；
   - 若任何参数全等于 `-c` 或以 `-c` 开头（如 `-c "..."`），抛出 `DashboardApiError("ILLEGAL_ARGUMENT", ...)`；
3. **模式匹配与区分**：
   - **Mode A (模块模式)**：`args[0] == "-m"`：
     - `len(args) >= 2`，否则抛出 `DashboardApiError("MISSING_ARGUMENT", ...)`；
     - `args[1]` 必须匹配合法 Python 模块标识符正则 `^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$`，否则抛出 `DashboardApiError("INVALID_MODULE_NAME", ...)`；
     - 规范化结果：`command = venv_python`, `args = list(args)`, `cwd = config_dir`（回退为 `os.getcwd()`）。
   - **Mode B (脚本模式)**：`args[0] != "-m"`：
     - `args[0]` 首字符不得为 `-`，且后缀必须为 `.py`（不区分大小写，如 `.py` / `.PY`），否则抛出 `DashboardApiError("INVALID_SCRIPT_PATH", ...)`；
     - 路径解析：若 `args[0]` 为相对路径，基于 `config_dir`（若未指定则使用配置文件所在目录，回退为 `os.getcwd()`）解析为绝对路径 `abs_path = os.path.abspath(os.path.join(config_dir, args[0]))`；
     - 文件存在性：`os.path.isfile(abs_path)` 必须为真，否则抛出 `DashboardApiError("SCRIPT_NOT_FOUND", ...)`；
     - 文件可读性：`os.access(abs_path, os.R_OK)` 必须为真，否则抛出 `DashboardApiError("SCRIPT_NOT_READABLE", ...)`；
     - 规范化结果：`resolved_script = os.path.realpath(abs_path)`，`command = venv_python`, `args = [resolved_script, *args[1:]]`, `cwd = os.path.dirname(resolved_script)`。
     - **CWD 与相对文件语义说明**：脚本模式下，子进程的 `cwd` 固定设置为规范化脚本所在目录（`os.path.dirname(resolved_script)`）。这意味着脚本内部的相对文件操作（如 `open("config.json")`）和传入给脚本的相对路径参数（`args[1:]` 中的相对路径）将相对于该脚本所在目录解析。这与从任意外部目录通过 CLI 调用 `python path/to/script.py`（保留外层 shell cwd）存在明确语义区别，属系统有意设计的确定性隔离契约。
4. **环境变量校验与构建**：
   - 保留既有 `validate_mcp_env` 强校验：禁止 `LD_*`、`DYLD_*` 等危险前缀；禁止 `PYTHON*` 前缀（白名单放行 `PYTHONUNBUFFERED`）；禁止非法变量名或非字符串键值；
   - 过滤系统继承环境变量中所有以 `LD_`、`DYLD_` 开头的变量；
   - 自动注入 `VIRTUAL_ENV`、将 virtualenv 的 `bin` 前置于 `PATH`，并设置 `PYTHONUNBUFFERED=1`。

### 3.2 共享规范化与运行时对齐 (`normalize_stdio_mcp_config`)

在 `agent/ovs_agent/plugins/mcp_manager.py` 中导出纯函数：

```python
def normalize_stdio_mcp_config(
    command: str | None,
    args: Sequence[str],
    env: dict[str, str] | None = None,
    config_dir: str | None = None,
) -> tuple[str, list[str], dict[str, str], str]:
    """Validate and normalize stdio config into (command, args, env, cwd)."""
    ...
```

- **统一解释器选择**：
  - 若未指定 `command` 或 `command == sys.executable`，使用当前运行环境的 virtualenv 解释器路径（`sys.executable`）；
  - 若指定了自定义 `command`，先通过 `inspect_and_open_verified_exe` 校验权限与安全性，但返回可调用的命令路径（保留其 virtualenv 符号链接入口，不剥离 `pyvenv.cfg`）；
- **Live 客户端对齐**：
  - 更新 `agent/ovs_agent/mcp/client.py` 中的 `McpConnection.connect()`：
    ```python
    server_params = StdioServerParameters(
        command=self.command,
        args=self.args,
        env=self.env,
        cwd=self.cwd,  # 传入经规范化确定的 cwd
    )
    ```
  - 更新 `agent/ovs_agent/mcp/plugin.py`：在从 `config.mcp_servers` 创建 `McpConnection` 时，统一调用 `normalize_stdio_mcp_config`，传入 `config_dir = os.path.dirname(config_file_path)`，确保 live 连接拥有与 dashboard 测试完全一致的规范化参数与 `cwd`。

### 3.3 进程组生命周期、收割托管与 Stderr 流式排水

1. **启动与进程组标记**：
   - 使用 `proc = await asyncio.create_subprocess_exec(..., start_new_session=True, stderr=asyncio.subprocess.PIPE)`；
   - 立即记录进程组 ID：`pgid = proc.pid`（`start_new_session=True` 保证子进程是进程组组长；排除了主动调用 `setsid()` 脱离组的恶意进程）。
2. **Stderr 异步排水任务 (`_drain_stderr`)**：
   - 启动后台任务：
     ```python
     stderr_tail = bytearray()
     drain_error: Exception | None = None

     async def _drain_stderr(stream: asyncio.StreamReader) -> None:
         nonlocal stderr_tail, drain_error
         try:
             while True:
                 chunk = await stream.read(1024)
                 if not chunk:
                     break
                 stderr_tail.extend(chunk)
                 if len(stderr_tail) > 4096:
                     del stderr_tail[:-4096]
         except asyncio.CancelledError:
             raise
         except Exception as e:
             drain_error = e
             logger.warning("MCP stderr drain encountered error: %s", e)
             raise

     drain_task = asyncio.create_task(_drain_stderr(proc.stderr))
     ```
   - 读取采用小块二进制流式读取（兼容无换行的大量输出）；
   - 若 `drain_task` 发生非取消的意外读取异常，记录 `drain_error`，主流程感知后终止流程并报 `CONNECTION_FAILED`；
   - 在向客户端返回错误诊断时，通过正则清洗掉其中的潜在 Token/Secret 等敏感信息。
3. **后台收割监督与任务注册表**：
   ```python
   _PENDING_CLEANUP_TASKS: set[asyncio.Task] = set()
   _PENDING_REAP_TASKS: set[asyncio.Task] = set()

   async def _supervise_pending_reap(proc: asyncio.subprocess.Process, pid: int) -> None:
       """Supervise reap of process that did not exit within initial 1.0s wait."""
       try:
           await asyncio.wait_for(proc.wait(), timeout=10.0)
           logger.info("Background reap confirmed for MCP process %d", pid)
       except asyncio.TimeoutError:
           logger.error("MCP process %d failed to reap after background timeout 10.0s", pid)
       except Exception as e:
           logger.error("Error in background reap for MCP process %d: %s", pid, e)
   ```
4. **独立进程组级联清理与回收状态机 (`_cleanup_process`)**：
   ```python
   async def _cleanup_process(
       proc: asyncio.subprocess.Process,
       pgid: int,
       drain_task: asyncio.Task | None,
   ) -> str:
       # 返回值: REAP_CONFIRMED | REAP_TIMEOUT_PENDING | REAP_FAILED
       # 1. 尝试优雅终止整个进程组 (SIGTERM)
       try:
           os.killpg(pgid, signal.SIGTERM)
       except ProcessLookupError:
           pass

       # 2. 组级存活判定：基于 os.killpg(pgid, 0) 等待最多 1.5s 优雅期
       deadline = asyncio.get_running_loop().time() + 1.5
       while asyncio.get_running_loop().time() < deadline:
           try:
               os.killpg(pgid, 0)
           except ProcessLookupError:
               break
           except PermissionError:
               pass
           await asyncio.sleep(0.05)

       # 3. 升级强制击杀整个进程组 (SIGKILL)
       try:
           os.killpg(pgid, signal.SIGKILL)
       except ProcessLookupError:
           pass

       # 4. 回收父进程并显式标记状态，超时则转交后台监督
       reap_status = "REAP_CONFIRMED"
       try:
           await asyncio.wait_for(proc.wait(), timeout=1.0)
       except asyncio.TimeoutError:
           reap_status = "REAP_TIMEOUT_PENDING"
           logger.warning("MCP parent process %d failed to reap within 1.0s after SIGKILL; launching background supervisor", proc.pid)
           reap_task = asyncio.create_task(_supervise_pending_reap(proc, proc.pid))
           _PENDING_REAP_TASKS.add(reap_task)
           reap_task.add_done_callback(_PENDING_REAP_TASKS.discard)
       except Exception as e:
           reap_status = "REAP_FAILED"
           logger.warning("Error waiting for MCP process %d to exit: %s", proc.pid, e)

       # 5. 清理 stderr 排水任务与管道
       if drain_task and not drain_task.done():
           drain_task.cancel()
           try:
               await asyncio.wait_for(drain_task, timeout=0.5)
           except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
               pass

       return reap_status
   ```
5. **抗多重取消、异常终态消费与责任转交契约**：
   ```python
   def _on_background_cleanup_done(t: asyncio.Task) -> None:
       """Explicitly observe result/exception to prevent unretrieved task exception warnings."""
       try:
           if not t.cancelled():
               exc = t.exception()
               if exc:
                   logger.error("Background MCP cleanup task failed: %s", exc)
               else:
                   res = t.result()
                   logger.info("Background MCP cleanup completed with status: %s", res)
       except Exception as e:
           logger.error("Error inspecting background cleanup task completion: %s", e)
       finally:
           _PENDING_CLEANUP_TASKS.discard(t)

   cleanup_task = asyncio.create_task(_cleanup_process(proc, pgid, drain_task))
   loop = asyncio.get_running_loop()
   cleanup_deadline = loop.time() + 3.5
   first_cancel: asyncio.CancelledError | None = None
   cleanup_exc: Exception | None = None
   cleanup_status: str | None = None

   while not cleanup_task.done():
       remaining = cleanup_deadline - loop.time()
       if remaining <= 0:
           logger.error("MCP cleanup task exceeded budget 3.5s; status=CLEANUP_PENDING_TIMEOUT; transferring to background supervisor")
           _PENDING_CLEANUP_TASKS.add(cleanup_task)
           cleanup_task.add_done_callback(_on_background_cleanup_done)
           break
       try:
           cleanup_status = await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=remaining)
       except asyncio.CancelledError as ce:
           if first_cancel is None:
               first_cancel = ce  # 锁定首个原始取消异常
       except asyncio.TimeoutError:
           logger.error("MCP cleanup task timed out against budget; status=CLEANUP_PENDING_TIMEOUT; transferring to background supervisor")
           _PENDING_CLEANUP_TASKS.add(cleanup_task)
           cleanup_task.add_done_callback(_on_background_cleanup_done)
           break
       except Exception as exc:
           cleanup_exc = exc
           logger.exception("MCP cleanup task raised exception: %s", exc)
           break

   # 若前台正常等待完成，必须显式消费其异常或返回值
   if cleanup_task.done() and not cleanup_task.cancelled():
       if cleanup_exc is None:
           cleanup_exc = cleanup_task.exception()
       if cleanup_exc is None and cleanup_status is None:
           cleanup_status = cleanup_task.result()

   if first_cancel is not None:
       if cleanup_exc is not None:
           first_cancel.__context__ = cleanup_exc
       raise first_cancel
   elif cleanup_exc is not None:
       raise cleanup_exc
   ```

### 3.4 错误码双向一致性契约 (Error Code Mapping)

在 `SAFE_CLIENT_MESSAGES` 与 `HTTP_STATUS_MAP` 中新增以下错误项：

| 错误码 (Code) | HTTP 状态码 | 返回安全中文提示 (SAFE_CLIENT_MESSAGES) |
|---|---|---|
| `MISSING_ARGUMENT` | 400 | 参数缺失，stdio 模式需要提供非空 args 列表 |
| `INVALID_MODULE_NAME` | 400 | Python 模块名格式不正确，必须为合法的模块标识符 |
| `INVALID_SCRIPT_PATH` | 400 | Python 脚本路径不合法，必须以 .py 结尾且不能以选项参数开头 |
| `SCRIPT_NOT_FOUND` | 400 | 指定的 Python 脚本文件不存在，请检查路径是否正确 |
| `SCRIPT_NOT_READABLE` | 400 | Python 脚本文件无读取权限，请检查文件权限 |
| `ILLEGAL_ARGUMENT` | 400 | 包含不支持的参数格式，严禁使用 -c 参数 |
| `FORBIDDEN_ENV` | 400 | 禁止设置动态链接器相关环境变量 |

---

## 4. 前端接口与校验对齐 (`dashboard.js`)

1. **废弃硬编码路径与模块限制**：
   - 彻底移除 `/(?:^|\/)(?:mcp_servers|opt\/voice)\/[^\/]+\.py$/` 正则；
2. **通用语法提示正则**：
   - 模块模式：`a0 === "-m"` 且 `args.length >= 2`；
   - 脚本模式：`/\.py$/i.test(a0)` 且 `!a0.startsWith("-")`（支持 `.py` / `.PY`）；
3. **结构化错误文本对接**：
   - 将上述新错误码加入前端 `MCP_ERROR_TEXT` 映射表，提供友好的中文提示。

---

## 5. 验收测试矩阵 (Verification Matrix)

1. **单元测试与回归套件 (`test_mcp_dashboard_api.py`)**：
   - `test_normalize_stdio_mode_a`: 验证合法模块名成功解析，`cwd` 设置为 `config_dir`；
   - `test_normalize_stdio_mode_b_relative`: 验证相对脚本路径基于 `config_dir` 解析，`cwd` 设置为脚本真实所在目录；
   - `test_validation_precedence`: 验证空参数 (`MISSING_ARGUMENT`) -> 元字符 (`SECURITY_VIOLATION`) -> `-c` (`ILLEGAL_ARGUMENT`) -> 模块名 (`INVALID_MODULE_NAME`) -> 文件存在 (`SCRIPT_NOT_FOUND`) 的优先级判定；
   - `test_unreadable_file_rejected`: 验证对不可读文件的权限校验（模拟 `os.access(..., R_OK) == False` 返回 `SCRIPT_NOT_READABLE`）；
   - `test_stderr_drain_under_heavy_output`: 验证向 stderr 写入超过 128KB 且无换行的大量输出时，子进程不会死锁并正常完成；
   - `test_stderr_drain_error_handling`: 验证当 stderr 读取出现异常时被捕获并转化为 `CONNECTION_FAILED`，不产生静默挂起；
   - `test_stderr_tail_redaction`: 验证错误输出中的 token/secret 等被脱敏处理；
   - `test_process_group_cleanup_parent_exited_surviving_children`: 验证父进程提前退出、子孙进程忽略 `SIGTERM` 的极端场景下，进程组级联清理仍然在 1.5s 后通过 `SIGKILL` 彻底回收，且通过 `os.killpg(pgid, 0)` 验证组完全消失；
   - `test_process_group_cleanup_under_repeated_cancellation`: 连续触发至少两次不同消息的取消信号（例如在测试任务中连续向 awaiter 发送 `cancel("cancel-1")` 和 `cancel("cancel-2")`），验证底层 `cleanup_task` 完整执行完毕，子进程树被彻底清理，且最外层精确保留并重抛首个取消异常（`"cancel-1"`）；
   - `test_process_group_cleanup_failure_with_cancellation_precedence`: 验证即使清理过程中产生内部异常，当外部存在取消信号时，最外层依然优先重抛 `CancelledError`，且将内部异常挂载至 `__context__`；
   - `test_process_group_cleanup_deadline_timeout_transfer_and_observation`: 验证在极端模拟延迟超过 3.5s 时，未竟任务正确转交给 `_PENDING_CLEANUP_TASKS` 托管，其完成回调显式调用 `exception()` / `result()`，确认无 `Task exception was never retrieved` 告警并在完成后从集合移除；
   - `test_reap_failure_status_tracking_and_supervision`: 验证当 `proc.wait()` 超过 1.0s 时正确记录 `REAP_TIMEOUT_PENDING` 结构化状态，派生并注册 `_supervise_pending_reap` 后台收割任务，持续监控直至最终回收完成；
   - `test_symlink_virtualenv_preservation`: 验证通过符号链接形式的 virtualenv 启动时，`pyvenv.cfg` 上下文得到正确保留；
   - `test_case_insensitive_script_extension`: 验证 `.py` 与 `.PY` 扩展名在前后端均顺利通过；
   - `test_mcp_bridge_plugin_parity`: 验证 `McpBridgePlugin` 使用相同的规范化函数，加载带有相对路径的配置并在正确的 `cwd` 下完成 live 工具初始化；
   - 全部 45+ 项测试执行通过。
2. **真机端到端运行验收 (`rk3576-devkit-key:18000`)**：
   - 部署更新后的镜像容器；
   - Playwright 走查：
     - 测试现有 `/opt/voice/custom_mcp.py` 连通性测试通过；
     - 测试保存后重启 Agent，Live Agent 成功加载 `/opt/voice/custom_mcp.py` 工具；
     - 验证界面布局与 169 个 DOM 节点功能正常。
