# Dev-Flow Spec v7.3: MCP 服务标准配置治理与优雅重启架构规范

## 1. 架构原则与通用设计理念

针对用户提出的 **“尽量选择通用，而不是 hack 的方向做”**，本规范遵循成熟配置治理与轻量持久化标准：

1. **“独立目录同步双日志原子替换 + 优雅重启”工业标准模型**：
   - 彻底避免在实时音频流和多轮对话中进行高风险的异步任务强制取消、连接热替换与两阶段分布式事务回滚；
   - 运行态与持久态严格分离：内存明确区分 `active_servers` 与 `pending_servers`，通过 `pending_restart: true` 契约引导用户一键优雅重启生效；
2. **多硬件后端零分歧、零脏状态**：
   - 重启后，进程从持久化配置全新加载 `McpBridgePlugin`；
   - **NVIDIA Jetson**：从零构建干净的 System Prompt，无前缀冲突，自动生成新 System KV Cache；
   - **Rockchip RKLLM**：干净加载工具集合，无需在对话中途动态计算内存碎片；
   - **Server-Loop 模式**：WebSocket 首次握手时自动向 SLV 发送最新工具集合的 `CLIENT_TOOL_ADVERTISE`，杜绝中途失步；
3. **闭环安全准入与防逃逸沙箱**：
   - 会话级 Admin-Token 强校验；写操作执行严密 CSRF 全匹配；GET 请求严格要求来源头（若存在）全合法；
   - `stdio` 参数白名单（严禁 `-c` 与 Shell 元字符）；
   - **Linux 原生 `/proc/self/fd/<fd>` 文件描述符执行模型彻底消除 TOCTOU 窗口**：
     - 以 `os.open(verified_real_exe, os.O_RDONLY | os.O_NOFOLLOW)` 打开二进制物理文件持有文件描述符 `exe_fd`（保持可继承）；
     - 对 `exe_fd` 执行 `os.fstat()` 严格校验常规文件、UID 为当前用户或 0、禁止 Group/World-Writable，且匹配记录的 `(dev, ino)`；
     - 递归校验父目录链签名通过后，执行目标指定为 `f"/proc/self/fd/{exe_fd}"` 并传入 `pass_fds=(exe_fd,)`；
     - **闭环生命周期保障**：父进程中无论参数校验、FD 传递、进程创建成功还是抛出异常，`exe_fd` 均无条件由外层 `finally` 通过 `_safe_close(exe_fd)` 释放；
     - 若任何校验不通过，直接抛出 `SecurityError("SECURITY_VIOLATION")` 拒绝启动并返回 HTTP 400；
   - `sse` 禁用环境代理 (`trust_env=False`)，单 Session 独立 Connector 自定义 Resolver 绑定固定首 IP，禁用 HTTP 重定向；
   - 平台强依赖 Linux `os.O_NOFOLLOW` 与 `os.O_DIRECTORY`（无此能力直接拒起并报错，杜绝静默降级）；
   - 配置文件权限统一规范为 `0o600`（所有临时文件、备份文件、恢复文件与目标文件统一为 Owner 读写，彻底剥离 Group/World 权限）；
   - **全生命周期绝对脱敏与跨进程双重持久化锁闭环**：
     - **零绕过契约**：所有端点（包含 `/api/control/restart_agent` 与 `/api/mcp/*`）全部装配 `@mcp_api_boundary`，所有非成功分支一律 `raise DashboardApiError`，严禁直接构造未脱敏的 `web.json_response({"ok": False, ...})`；
     - 客户端响应绝对脱敏：`SAFE_CLIENT_MESSAGES` 与 `HTTP_STATUS_MAP` 在模块加载时执行类型合法、非空、集合双射校验；每个错误响应强制注入唯一 `request_id`，内部路径及堆栈仅记录服务端日志；
     - **全链路外层捕获恢复标记事务**：`write_persistent_recovery_marker` 由最外层 `try/except` 严密包裹全部执行阶段（参数校验、创建、写、fsync、关闭、replace、目录 fsync），任何异常绝无遗漏统一经 `fatal_config_recovery_exit()` 刷新日志缓冲并触发 Exit 78 硬阻断 supervisor；前置拒绝非法 `reason` 禁止静默突变；
     - **精准扫描与类型记录**：启动精准扫描正则 `^\.config_recovery_required\.marker_tmp\.[0-9a-f]{32}$`，分类记录残留项类型（常规文件、符号链接、目录等）并硬锁死；
     - 严格隔离 FD 继承边界：所有配置 I/O FD 强制启用 `O_CLOEXEC`，仅 `exe_fd` 具备继承语义；
   - **严格区分文件原始字节散列与语义配置 ETag 域**：
     - `expected_prior_file_sha256`：磁盘原始文件未解析字节流的 SHA256，用于校验备份源及恢复副本完整性；
     - `expected_prior_config_etag`：解析后规范化配置 JSON 的 SHA256，用于检验配置语义等价性；
     - 严禁跨域比较，完美兼容 YAML 格式、格式化缩进空白与键排序差异；
   - **补偿成功后全量状态同步**：恢复后重新解析目标文件，同步更新 `persisted_mcp_servers`、`persisted_config_etag`、`persisted_file_sha256` 与重新计算 `pending_restart`，确保内存态与磁盘态绝对一致；
   - Phase 1 发生任何异常均核验 backup 完整性，无法确认完整立即置位 `CONFIG_RECOVERY_REQUIRED` 锁死；
   - 重启状态机通过带明确事件约束的 `ALLOWED_TRANSITIONS` 状态转移表和 `generation` 锁保护的 `_transition_restart_state()` 严格流转，停机状态在 `_restart_lock` 保护下阻断所有新重启请求（返回 503），并通过循环清空优雅回收所有 tracked tasks。

---

## 2. 身份认证、准入与安全防护契约

### 2.1 会话级鉴权与 CSRF 标准防御
1. **会话密钥生成**：
   - 每次 Agent 启动由 `secrets.token_hex(32)` 生成一次性内存密钥 `self._mcp_admin_token`（禁止环境变量覆写，访问日志脱敏）；
   - `GET /` 响应头必须携带 `Cache-Control: no-store, no-cache, must-revalidate, private`，并在 HTML `<head>` 中注入 `<meta name="ovs-admin-token" content="<token>">`；
2. **接口强制鉴权**：
   - `/api/mcp/*` 所有接口（包含 `GET /api/mcp/servers` 与所有写操作 `POST`, `DELETE`, `PATCH`）**一律强制校验请求头 `X-Admin-Token`**；
   - 服务端使用 `secrets.compare_digest` 常数时间比对，缺失或不匹配直接抛出 `DashboardApiError("UNAUTHORIZED", "Missing or invalid admin token")`（HTTP 401）；网络边界上仅在 loopback 环境生效，loopback 不能替代 token；
3. **CSRF / Origin & Referer 独立判定规则**：
   - **写操作 (`POST`, `DELETE`, `PATCH`)**：必须包含 `Origin` 或 `Referer` 中的至少一个；两者提取出的 Host（含端口）都必须全等于当前 HTTP 请求的 `Host` Header；任何不符直接抛出 `DashboardApiError("FORBIDDEN", "CSRF check failed")`（HTTP 403）；
   - **读操作 (`GET`)**：必须携带合法 `X-Admin-Token`；若请求携带了 `Origin`，其 Host 必须与请求 `Host` 完全一致；若请求携带了 `Referer`，其 Host 必须与请求 `Host` 完全一致；若两者均未附带（如安全同源刷新），在有效 Token 前提下放行。

### 2.2 stdio 参数白名单与 `/proc/self/fd` 原生安全执行
1. **真实执行器绑定、父路径链签名与 FD 锁定**：
   ```python
   def inspect_and_open_verified_exe() -> tuple[int, str, list[tuple[str, int, int]]]:
       real_exe = os.path.realpath(sys.executable)
       real_prefix = os.path.realpath(sys.prefix)
       try:
           if os.path.commonpath([real_exe, real_prefix]) != real_prefix:
               raise SecurityError("sys.executable is outside sys.prefix")
       except ValueError as e:
           raise SecurityError(f"commonpath validation failed: {e}") from e

       # 以 O_NOFOLLOW 打开可执行物理文件，获得内核文件句柄（不带 O_CLOEXEC，供子进程继承）
       try:
           exe_fd = os.open(real_exe, os.O_RDONLY | os.O_NOFOLLOW)
       except OSError as e:
           raise SecurityError(f"Failed to open real_exe securely: {e}") from e

       try:
           st = os.fstat(exe_fd)
           if not stat.S_ISREG(st.st_mode):
               raise SecurityError("sys.executable is not a regular file")
           if st.st_uid not in {os.getuid(), 0}:
               raise SecurityError(f"Untrusted owner {st.st_uid} on executable")
           if (st.st_mode & 0o022) != 0:
               raise SecurityError("sys.executable is group/world writable")

           # 逐级向上校验父路径链并记录签名
           expected_uids = {os.getuid(), 0}
           chain = [(real_exe, st.st_dev, st.st_ino)]
           curr = os.path.dirname(real_exe)
           while True:
               try:
                   pst = os.stat(curr)
               except OSError as e:
                   raise SecurityError(f"stat failed on parent {curr}: {e}") from e
               if pst.st_uid not in expected_uids:
                   raise SecurityError(f"Untrusted owner on parent {curr}")
               if (pst.st_mode & 0o022) != 0:
                   raise SecurityError(f"Parent directory {curr} is group/world writable")
               chain.append((curr, pst.st_dev, pst.st_ino))
               parent = os.path.dirname(curr)
               if parent == curr:
                   break
               curr = parent

           venv_bin = os.path.dirname(real_exe)
           return exe_fd, venv_bin, chain
       except Exception:
           _safe_close(exe_fd)
           raise
   ```
2. **内核级 `/proc/self/fd` 执行与外层 `finally` FD 生命周期保证**：
   - 启动任何 stdio 子进程时必须使用严格的 `try...finally` 结构：
     ```python
     async def launch_stdio_mcp_subprocess(args: list[str], env: dict[str, str]) -> asyncio.subprocess.Process:
         exe_fd = -1
         try:
             exe_fd, venv_bin, chain = inspect_and_open_verified_exe()
             proc_exe = f"/proc/self/fd/{exe_fd}"
             mcp_env = build_mcp_env(venv_bin)
             proc = await asyncio.create_subprocess_exec(
                 proc_exe,
                 *args,
                 env=mcp_env,
                 pass_fds=(exe_fd,),
                 stdin=asyncio.subprocess.PIPE,
                 stdout=asyncio.subprocess.PIPE,
                 stderr=asyncio.subprocess.PIPE,
                 start_new_session=True,
             )
             return proc
         finally:
             if exe_fd >= 0:
                 _safe_close(exe_fd)
     ```
   - **完全消除 TOCTOU**：由于子进程通过已打开验证的 FD 执行，攻击者在磁盘上修改、重命名或替换路径不会对子进程执行造成任何影响；
   - 若任何一步校验失败，直接抛出 `SecurityError` 并映射为 HTTP 400，严禁盲目执行；
3. **模块化参数白名单**：
   - `args` 必须为结构化列表，且必须满足以下标准模式之一：
     - **模式 A (标准模块)**：`args[0] == "-m"` 且 `args[1] in ALLOWED_MCP_MODULES`（默认白名单：`{"ovs_agent.mcp.servers", "ha_mcp", "device_mcp"}`）；
     - **模式 B (内置项目脚本)**：`args[0]` 解析后的绝对路径严格落在 `<project_root>/agent/mcp_servers/` 目录下；
   - 严禁 `-c` 选项；参数禁止包含 Shell 元字符（`|`, `;`, `&`, `$`, `` ` ``, `>`, `<`, `\n`, `\r`）。

### 2.3 sse 反 SSRF、DNS Pinning 与代理阻断
1. **URL Scheme 校验**：严格限定 `http` 与 `https`；
2. **DNS 解析与私网全阻断**：
   - 调用 `socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)` 解析所有 A/AAAA 记录；
   - 逐个校验：拒绝 Loopback (`127.0.0.0/8`, `::1`)、Private (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`)、Link-Local (`169.254.0.0/16`, `fe80::/10`)、Carrier-Grade NAT (`100.64.0.0/10`)、Multicast、Broadcast 及已知云厂商元数据域名；
   - 若任何一个解析地址落入黑名单，拒绝请求并抛出 `DashboardApiError("INVALID_PARAM", "Target host resolves to prohibited private or loopback address")`（HTTP 400）；
3. **底层连接 Pinning 与网络隔离**：
   - 提取首个合法 IP `pinned_ip`；
   - 每次连接使用独立的 `aiohttp.TCPConnector`，其自定义 Resolver 将目标 `hostname` 解析直接映射为 `[pinned_ip]`，禁止跨请求连接池复用；
   - 创建 `aiohttp.ClientSession(connector=connector, trust_env=False)`，强制忽略环境变量中的 `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`；保持原始 HTTP `Host` Header 与 TLS SNI；
   - 显式设置 `allow_redirects=False`；若遇到 301/302/307/308 重定向直接报错终止。

### 2.4 敏感信息脱敏与统一凭据合并 (Secret Merge)
1. **脱敏返回**：
   - `GET /api/mcp/servers` 响应中，`env` 凡是键名包含 `KEY`, `TOKEN`, `SECRET`, `AUTH`, `PASS`, `PWD` 的值强制脱敏为 `"******"`；
   - SSE 的 `headers` 明文不返回，仅返回元数据：`has_headers: true`, `configured_header_keys: ["Authorization"]`；
2. **锁内严密凭据合并语义 (Secret Retention)**：
   - 在 `_mcp_config_lock` 保护下执行逐键合并：
     - **未出现的 key**：完全保留原持久化凭据；
     - **值为 `"******"`**：完全保留原持久化凭据；
     - **值为非空字符串**：替换为新明文值；
     - **值为空字符串 `""`**：从配置字典中彻底删除该 key；
     - **非法类型**：直接抛出 `DashboardApiError("INVALID_PARAM", "Invalid secret value type")`（HTTP 400）。

---

## 3. 独立测试沙箱 (`POST /api/mcp/test`)

用于前端在添加或编辑服务表单中点击“测试连接”：
1. **独立进程组与安全存活检查回收**：
   - 测试子进程经 `launch_stdio_mcp_subprocess()` 启动，记录 `pgid`；
   - 强制 10.0 秒硬超时；
   - 清理逻辑：分别捕获每次 `killpg` 的异常并先检查进程存活状态，及时执行 `wait()` 杜绝僵尸进程，绝不因正常退出覆盖已成功发现的工具列表：
     ```python
     if proc is not None and pgid is not None:
         if proc.poll() is None:
             try:
                 os.killpg(pgid, signal.SIGTERM)
                 await asyncio.wait_for(proc.wait(), timeout=0.5)
             except (ProcessLookupError, PermissionError, asyncio.TimeoutError):
                 pass
             if proc.poll() is None:
                 try:
                     os.killpg(pgid, signal.SIGKILL)
                     await asyncio.wait_for(proc.wait(), timeout=0.5)
                 except (ProcessLookupError, PermissionError, asyncio.TimeoutError):
                     pass
     ```
   - 严禁将测试工具注入正式主 `ToolRegistry`；
2. **并发控制**：使用 `asyncio.Semaphore(2)` 控制并发，超出直接抛出 `DashboardApiError("TOO_MANY_REQUESTS", "Concurrent MCP test limit reached")`（HTTP 429）；
3. **返回结构**：
   - 成功：`{"ok": true, "tools_count": 3, "tools": [{"name": "...", "description": "..."}]}`；
   - 失败：抛出 `DashboardApiError("CONNECTION_FAILED", ...)`，由装饰器返回标准脱敏响应（HTTP 502）。

---

## 4. 运行态分离、强 ETag 与双日志原子持久化

### 4.1 运行态与持久态解耦
- `self._active_mcp_servers`: 当前正在运行的 MCP 服务列表及连接状态；
- `self._persisted_mcp_servers`: 磁盘配置文件中持久化的 MCP 服务列表；
- `self._persisted_config_etag`: 当前磁盘配置的规范化语义 ETag；
- `self._persisted_file_sha256`: 当前磁盘文件的原始二进制 SHA256；
- `self._config_recovery_required`: 故障锁死标志；
- `pending_restart`: 当持久化配置与当前运行态不一致时为 `true`；
- `GET /api/mcp/servers` 明确返回：
  ```json
  {
    "ok": true,
    "pending_restart": true,
    "active_servers": [...],
    "servers": [...],
    "etag": "<persisted_config_etag>"
  }
  ```

### 4.2 双散列域独立模型 (File Hash vs Config ETag)
为了彻底根除不同序列化格式（如 YAML、包含缩进或换行的 JSON、属性乱序等）恢复时散列失配问题，系统严格分离两个校验域：
1. **文件字节散列 (`file_sha256`)**：
   - 对磁盘上存储的文件原始未经修改的二进制字节流执行 `hashlib.sha256(raw_bytes).hexdigest()`；
   - 专门用于 Phase 1 备份写入、Phase 2 异常回滚时对 `backup_path` 物理文件的校验，确保备份文件与修改前磁盘文件完全比特一致；
2. **语义配置强 ETag (`config_etag`, RFC 7232)**：
   - 对从文件中解析出的配置对象执行规范化序列化：`json.dumps(servers, sort_keys=True, separators=(',', ':'), ensure_ascii=False)`；
   - 计算该规范化字节流的 SHA256 作为对外 API 的强 ETag；
   - 专门用于 API 并发控制 (`If-Match`)，以及在恢复完成后验证持久化配置的语义等价性；
   - **两域绝对隔离，绝不交叉比较**。

### 4.3 平台约束、双射映射硬校验与零 raw-JSON 装饰器体系
1. **平台能力强约束与权限基线**：
   - 生产环境明确限定为 Linux/POSIX 环境，`os.O_NOFOLLOW` 与 `os.O_DIRECTORY` 必须存在；若检测到缺失任一，启动或初始化直接抛出 `RuntimeError`，严禁静默降级；
   - **统一安全权限模型**：所有涉及配置的文件（临时文件、备份文件、恢复文件与目标文件）一律强制为 `0o600`（仅当前属主具备读写权限，彻底剔除 group 与 world 权限位）；
   - **单文件大小上限前置防御**：`MAX_CONFIG_SIZE = 2 * 1024 * 1024`（2MB）；

2. **严格双射错误映射字典与启动一致性强断言**：
   ```python
   SAFE_CLIENT_MESSAGES: dict[str, str] = {
       "UNAUTHORIZED": "Authentication required or invalid admin token",
       "FORBIDDEN": "Request forbidden or CSRF validation failed",
       "FILE_TOO_LARGE": "Configuration size exceeds 2MB limit",
       "CONFIG_PARSE_ERROR": "Configuration syntax is invalid",
       "SECURITY_VIOLATION": "Security validation failed",
       "INVALID_PARAM": "Invalid request parameter",
       "ETAG_MISMATCH": "Configuration was modified concurrently, please refresh",
       "CONCURRENT_MODIFICATION": "Operation conflict, please retry",
       "TOO_MANY_REQUESTS": "Too many concurrent requests, please retry later",
       "SERVICE_UNAVAILABLE": "Service is shutting down or unavailable",
       "CONNECTION_FAILED": "Failed to connect to MCP server or discovery failed",
       "TEST_TIMEOUT": "MCP server test connection timed out",
       "DISK_FULL": "Host storage space is exhausted",
       "TMP_CREATE_FAILED": "Failed to create temporary configuration file",
       "TMP_CLEANUP_FAILED": "Failed to clean up temporary configuration files",
       "FD_CLOSE_FAILED": "Failed to close file descriptor safely",
       "FCHMOD_FAILED": "Failed to set secure file permissions",
       "FSYNC_FAILED": "Failed to synchronize configuration file to storage",
       "DIR_FSYNC_FAILED": "Failed to synchronize directory metadata to storage",
       "WRITE_FAILED": "Failed to write configuration data",
       "SHORT_WRITE": "Incomplete configuration write encountered",
       "REPLACE_FAILED": "Failed to replace configuration file atomically",
       "READ_FAILED": "Failed to read configuration file",
       "CONFIG_RECOVERY_REQUIRED": "Configuration state requires manual recovery; locked in read-only mode",
       "INTERNAL_ERROR": "Internal server error occurred",
   }

   HTTP_STATUS_MAP: dict[str, int] = {
       "UNAUTHORIZED": 401,
       "FORBIDDEN": 403,
       "FILE_TOO_LARGE": 400,
       "CONFIG_PARSE_ERROR": 400,
       "SECURITY_VIOLATION": 400,
       "INVALID_PARAM": 400,
       "ETAG_MISMATCH": 409,
       "CONCURRENT_MODIFICATION": 409,
       "TOO_MANY_REQUESTS": 429,
       "SERVICE_UNAVAILABLE": 503,
       "CONNECTION_FAILED": 502,
       "TEST_TIMEOUT": 504,
       "DISK_FULL": 500,
       "TMP_CREATE_FAILED": 500,
       "TMP_CLEANUP_FAILED": 500,
       "FD_CLOSE_FAILED": 500,
       "FCHMOD_FAILED": 500,
       "FSYNC_FAILED": 500,
       "DIR_FSYNC_FAILED": 500,
       "WRITE_FAILED": 500,
       "SHORT_WRITE": 500,
       "REPLACE_FAILED": 500,
       "READ_FAILED": 500,
       "CONFIG_RECOVERY_REQUIRED": 500,
       "INTERNAL_ERROR": 500,
   }

   VALID_HTTP_STATUSES: set[int] = {400, 401, 403, 409, 429, 500, 502, 503, 504}

   def validate_error_maps_consistency() -> None:
       assert isinstance(VALID_HTTP_STATUSES, set) and len(VALID_HTTP_STATUSES) > 0, "VALID_HTTP_STATUSES must be non-empty set"
       assert all(type(s) is int for s in VALID_HTTP_STATUSES), "All status codes must be int"
       if not SAFE_CLIENT_MESSAGES or not HTTP_STATUS_MAP:
           raise RuntimeError("Error mappings must not be empty")
       if set(SAFE_CLIENT_MESSAGES.keys()) != set(HTTP_STATUS_MAP.keys()):
           raise RuntimeError(
               f"Mapping mismatch! in_msg_only={set(SAFE_CLIENT_MESSAGES) - set(HTTP_STATUS_MAP)}, "
               f"in_status_only={set(HTTP_STATUS_MAP) - set(SAFE_CLIENT_MESSAGES)}"
           )
       for code, status in HTTP_STATUS_MAP.items():
           if type(status) is not int or status not in VALID_HTTP_STATUSES:
               raise RuntimeError(f"Invalid integer status code {status} configured for {code}")

   validate_error_maps_consistency()

   class DashboardApiError(Exception):
       def __init__(self, code: str, internal_message: str, errno: int | None = None):
           if code not in HTTP_STATUS_MAP:
               raise RuntimeError(f"FATAL: Unregistered error code passed to DashboardApiError: {code}")
           super().__init__(internal_message)
           self.code = code
           self.internal_message = internal_message
           self.errno = errno

       @property
       def status(self) -> int:
           return HTTP_STATUS_MAP[self.code]

       def to_response(self, request_id: str) -> web.Response:
           client_msg = SAFE_CLIENT_MESSAGES[self.code]
           return web.json_response({
               "ok": False,
               "code": self.code,
               "error": client_msg,
               "request_id": request_id
           }, status=self.status)

   class SecurityError(DashboardApiError):
       def __init__(self, internal_message: str, code: str = "SECURITY_VIOLATION"):
           super().__init__(code=code, internal_message=internal_message)

   class ConfigPersistenceError(DashboardApiError):
       def __init__(self, code: str, internal_message: str, errno: int | None = None):
           super().__init__(code=code, internal_message=internal_message, errno=errno)

   def mcp_api_boundary(handler):
       """应用到 /api/mcp/* 及 /api/control/restart_agent 的全局统一边界装饰器"""
       @functools.wraps(handler)
       async def wrapper(request: web.Request, *args, **kwargs) -> web.Response:
           request_id = secrets.token_hex(8)
           try:
               return await handler(request, *args, **kwargs)
           except DashboardApiError as e:
               logger.warning("MCP API error [%s] %s: %s (errno=%s)", request_id, e.code, e.internal_message, e.errno)
               return e.to_response(request_id)
           except json.JSONDecodeError as e:
               logger.warning("JSON syntax error [%s]: %s", request_id, e)
               err = DashboardApiError("CONFIG_PARSE_ERROR", f"JSONDecodeError: {e}")
               return err.to_response(request_id)
           except Exception as e:
               logger.exception("Unhandled MCP API exception [%s]: %s", request_id, e)
               err = DashboardApiError("INTERNAL_ERROR", str(e))
               return err.to_response(request_id)
       return wrapper
   ```

3. **底层安全文件与 I/O 辅助函数 (完整提供全部声明函数)**：
   ```python
   def _safe_close(fd: int, raise_on_error: bool = False) -> None:
       """安全释放文件描述符；EBADF 属于已知幂等成功，其他错误在 raise_on_error=True 时抛出 FD_CLOSE_FAILED"""
       try:
           os.close(fd)
       except OSError as e:
           if e.errno != errno.EBADF:
               logger.warning("Error closing fd %d: %s", fd, e)
               if raise_on_error:
                   raise ConfigPersistenceError("FD_CLOSE_FAILED", f"Close fd {fd} failed: {e}", e.errno) from e

   def _safe_open_readonly(path: str) -> int:
       if not hasattr(os, "O_NOFOLLOW"):
           raise RuntimeError("os.O_NOFOLLOW is mandatory")
       flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
       try:
           return os.open(path, flags)
       except OSError as e:
           raise ConfigPersistenceError("READ_FAILED", f"Open {path} failed: {e}", e.errno) from e

   def _safe_fchmod(fd: int, mode: int) -> None:
       try:
           os.fchmod(fd, mode)
       except OSError as e:
           raise ConfigPersistenceError("FCHMOD_FAILED", f"fchmod failed on fd {fd}: {e}", e.errno) from e

   def _safe_fsync_dir(dirpath: str) -> None:
       if not hasattr(os, "O_DIRECTORY"):
           raise RuntimeError("os.O_DIRECTORY is mandatory")
       dfd = -1
       try:
           dfd = os.open(dirpath, os.O_DIRECTORY)
           os.fsync(dfd)
       except OSError as e:
           raise ConfigPersistenceError("DIR_FSYNC_FAILED", f"Fsync dir {dirpath} failed: {e}", e.errno) from e
       finally:
           if dfd >= 0:
               _safe_close(dfd)

   def _open_atomic_tmp(path: str) -> int:
       if not hasattr(os, "O_NOFOLLOW"):
           raise RuntimeError("os.O_NOFOLLOW is mandatory for atomic file persistence")
       flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
       fd = -1
       try:
           fd = os.open(path, flags, 0o600)
           _safe_fchmod(fd, 0o600)
           st = os.fstat(fd)
           if stat.S_IMODE(st.st_mode) != 0o600:
               raise ConfigPersistenceError("FCHMOD_FAILED", f"Exact 0600 mode verification failed: {oct(st.st_mode)}")
           return fd
       except Exception as e:
           if fd >= 0:
               _safe_close(fd)
           try:
               os.unlink(path)
           except OSError as unlink_err:
               if unlink_err.errno != errno.ENOENT:
                   logger.critical("FATAL: Failed to unlink tmp file %s: %s", path, unlink_err)
                   raise ConfigPersistenceError("TMP_CLEANUP_FAILED", f"Failed to clean up temporary file: {unlink_err}", unlink_err.errno) from unlink_err
           if isinstance(e, DashboardApiError):
               raise
           errno_val = getattr(e, "errno", None)
           raise ConfigPersistenceError("TMP_CREATE_FAILED", f"Create tmp file failed: {e}", errno_val) from e

   def _safe_fsync_file(fd: int) -> None:
       try:
           os.fsync(fd)
       except OSError as e:
           code = "DISK_FULL" if e.errno == errno.ENOSPC else "FSYNC_FAILED"
           raise ConfigPersistenceError(code, f"Fsync file failed: {e}", e.errno) from e

   def _safe_replace(src: str, dst: str) -> None:
       try:
           os.replace(src, dst)
       except OSError as e:
           raise ConfigPersistenceError("REPLACE_FAILED", f"Replace {src} -> {dst} failed: {e}", e.errno) from e

   def _write_all(fd: int, data: bytes) -> None:
       pos = 0
       while pos < len(data):
           try:
               n = os.write(fd, data[pos:])
           except InterruptedError:
               continue
           except OSError as e:
               code = "DISK_FULL" if e.errno == errno.ENOSPC else "WRITE_FAILED"
               raise ConfigPersistenceError(code, f"Write failed: {e}", errno=e.errno) from e
           if n <= 0:
               raise ConfigPersistenceError("SHORT_WRITE", "Short write encountered")
           pos += n

   def _read_all_bounded(fd: int, max_bytes: int = 2 * 1024 * 1024) -> bytes:
       buf = bytearray()
       while True:
           try:
               chunk = os.read(fd, 65536)
           except InterruptedError:
               continue
           except OSError as e:
               raise ConfigPersistenceError("READ_FAILED", f"Read failed: {e}", errno=e.errno) from e
           if not chunk:
               break
           buf.extend(chunk)
           if len(buf) > max_bytes:
               raise ConfigPersistenceError("FILE_TOO_LARGE", f"File content exceeds {max_bytes} bytes limit")
       return bytes(buf)

   def fatal_config_recovery_exit(reason: str, exit_code: int = 78) -> None:
       """统一收尾日志、刷新标准流并以 Exit 78 硬阻断 supervisor 重启"""
       logger.critical("FATAL CONFIG FAILURE: %s. Process exiting with code %d", reason, exit_code)
       try:
           logging.shutdown()
           sys.stdout.flush()
           sys.stderr.flush()
       except Exception:
           pass
       sys.exit(exit_code)

   MARKER_REGEX = re.compile(r"^CONFIG_RECOVERY_REQUIRED:v1:reason=([a-zA-Z0-9_\-\.]{1,64}):timestamp=([0-9]+\.[0-9]+)\n$")
   MARKER_TMP_REGEX = re.compile(r"^\.config_recovery_required\.marker_tmp\.[0-9a-f]{32}$")

   def write_persistent_recovery_marker(config_dir: str, reason: str) -> None:
       """通过安全原子落盘 .config_recovery_required 标记文件；外层严密封闭捕获全部异常，任何失败统一触发 Exit 78"""
       marker_path = os.path.join(config_dir, ".config_recovery_required")
       marker_tmp = f"{marker_path}.marker_tmp.{uuid.uuid4().hex}"
       try:
           if not re.match(r"^[a-zA-Z0-9_\-\.]{1,64}$", reason):
               raise ValueError(f"Invalid reason format: {reason}")
           fd = -1
           try:
               fd = _open_atomic_tmp(marker_tmp)
               payload = f"CONFIG_RECOVERY_REQUIRED:v1:reason={reason}:timestamp={time.time():.6f}\n".encode("utf-8")
               _write_all(fd, payload)
               _safe_fsync_file(fd)
           finally:
               if fd >= 0:
                   _safe_close(fd, raise_on_error=True)
           _safe_replace(marker_tmp, marker_path)
           _safe_fsync_dir(config_dir)
       except SystemExit:
           raise
       except Exception as e:
           try:
               os.unlink(marker_tmp)
           except OSError as unl_err:
               if unl_err.errno != errno.ENOENT:
                   logger.critical("FATAL: Failed to unlink marker tmp file %s: %s", marker_tmp, unl_err)
           fatal_config_recovery_exit(f"Failed to commit persistent recovery marker: {e}")

   def check_persistent_recovery_marker(config_dir: str) -> bool:
       """通过 O_NOFOLLOW、精确 0600 模式、父目录链安全校验及语义格式检查验证恢复标志"""
       marker_path = os.path.join(config_dir, ".config_recovery_required")

       # 1. 扫描检测是否存在符合命名规范的 marker_tmp 残留，分类记录类型并锁死
       try:
           with os.scandir(config_dir) as entries:
               for entry in entries:
                   if MARKER_TMP_REGEX.match(entry.name):
                       entry_type = "symlink" if entry.is_symlink() else "dir" if entry.is_dir() else "file" if entry.is_file() else "other"
                       logger.critical("FATAL: Found orphaned recovery marker tmp %s (type=%s); persistent lock enforced", entry.path, entry_type)
                       return True
       except OSError as e:
           raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", f"Failed to scan config dir for marker tmp: {e}", e.errno) from e

       # 2. 读取并严格校验正式 marker 文件
       try:
           fd = _safe_open_readonly(marker_path)
       except ConfigPersistenceError as e:
           if e.errno == errno.ENOENT:
               return False
           raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", f"Corrupted recovery marker: {e}", e.errno) from e

       try:
           st = os.fstat(fd)
           if not stat.S_ISREG(st.st_mode):
               raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Recovery marker is not a regular file")
           if st.st_uid not in {os.getuid(), 0}:
               raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Untrusted recovery marker owner")
           if stat.S_IMODE(st.st_mode) != 0o600:
               raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", f"Recovery marker mode must be exactly 0600, got {oct(st.st_mode)}")
           if st.st_size > 4096:
               raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Recovery marker size exceeds 4KB limit")
           raw_bytes = _read_all_bounded(fd, 4096)
           try:
               data = raw_bytes.decode("utf-8")
           except UnicodeDecodeError as dec_err:
               raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", f"Recovery marker decode failed: {dec_err}") from dec_err

           m = MARKER_REGEX.match(data)
           if not m:
               raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Recovery marker semantic structure corrupted")
           ts = float(m.group(2))
           if ts <= 0 or not math.isfinite(ts):
               raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Recovery marker timestamp invalid")
           return True
       finally:
           _safe_close(fd)

   def cleanup_orphaned_tmp_files(config_dir: str) -> None:
       """启动时扫描并清理历史遗留的孤立临时文件；遇删除失败或符号链接持久化写入恢复标志并抛出异常阻断启动"""
       if check_persistent_recovery_marker(config_dir):
           raise ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "Persistent recovery marker detected; configuration locked")

       try:
           with os.scandir(config_dir) as entries:
               for entry in entries:
                   # 仅匹配标准业务配置临时文件，绝不误伤 marker_tmp
                   if (entry.name.endswith(".tmp") or ".tmp." in entry.name) and not MARKER_TMP_REGEX.match(entry.name):
                       if entry.is_symlink():
                           write_persistent_recovery_marker(config_dir, "symlink_in_tmp")
                           raise ConfigPersistenceError("TMP_CLEANUP_FAILED", f"Symlink detected in tmp file: {entry.path}")
                       try:
                           os.unlink(entry.path)
                           logger.info("Cleaned orphaned tmp file: %s", entry.path)
                       except OSError as e:
                           if e.errno != errno.ENOENT:
                               logger.critical("FATAL: Failed to clean orphaned tmp file %s: %s", entry.path, e)
                               write_persistent_recovery_marker(config_dir, "unlink_tmp_failed")
                               raise ConfigPersistenceError("TMP_CLEANUP_FAILED", f"Startup tmp cleanup failed: {e}", e.errno) from e
       except OSError as scan_err:
           logger.critical("FATAL: Failed to scan config directory for tmp files: %s", scan_err)
           write_persistent_recovery_marker(config_dir, "scan_dir_failed")
           raise ConfigPersistenceError("TMP_CLEANUP_FAILED", f"Scan config dir failed: {scan_err}", scan_err.errno) from scan_err
   ```

4. **双日志提交与严密故障自愈协议（包含全链路 `try...finally` FD 释放）**：
   - 校验 `_config_recovery_required` 或 `check_persistent_recovery_marker(config_dir)`：若为 True，直接抛出 `ConfigPersistenceError("CONFIG_RECOVERY_REQUIRED", "System locked into read-only")`；
   - 定位 `target_path`；
   - **双前置大小校验**：
     - 读取当前磁盘文件字节流 `prior_raw_bytes`：若 `len(prior_raw_bytes) > MAX_CONFIG_SIZE`，抛出 `ConfigPersistenceError("FILE_TOO_LARGE", "Existing file too large")`；
     - 预备新配置写入字节流 `new_raw_bytes`：若 `len(new_raw_bytes) > MAX_CONFIG_SIZE`，抛出 `ConfigPersistenceError("FILE_TOO_LARGE", "New config payload too large")`；
   - 计算原文件原始哈希与语义 ETag：
     - `expected_prior_file_sha256 = hashlib.sha256(prior_raw_bytes).hexdigest()`；
     - `expected_prior_config_etag = self._calculate_etag(self._parse_config(prior_raw_bytes))`；
   - **Phase 1: 原子落盘物理备份 (Atomic Backup Commit，无条件 `finally` 释放 FD)**：
     - 打开临时备份文件 `backup_tmp = f"{target_path}.backup.tmp.{uuid.uuid4().hex}"`；
     - 在 `try...finally` 中确保临时文件 FD 绝对关闭：
       ```python
       backup_tmp_fd = -1
       try:
           backup_tmp_fd = _open_atomic_tmp(backup_tmp)
           _write_all(backup_tmp_fd, prior_raw_bytes)
           _safe_fsync_file(backup_tmp_fd)
       finally:
           if backup_tmp_fd >= 0:
               _safe_close(backup_tmp_fd, raise_on_error=True)
       ```
     - 原子替换备份文件：`_safe_replace(backup_tmp, backup_path)`；
     - 调用 `_safe_fsync_dir(dirpath)`；
     - *Phase 1 异常核验契约*：若 Phase 1 任一步骤抛出异常（包括目录 fsync 失败）：
       - 清理可能残留的 `backup_tmp`；
       - 打开 `backup_path` 验证完整性：以 `_safe_open_readonly(backup_path)` 打开，通过 `_read_all_bounded()` 读取并计算哈希；
       - 若 `backup_path` 不存在或哈希不等于 `expected_prior_file_sha256`：**立即标记 `self._config_recovery_required = True`，并写入持久恢复标记 `write_persistent_recovery_marker(dirpath, "phase1_backup_corrupted")`**，记录 Fatal 日志；
       - 绝不触碰目标文件 `target_path`，抛出异常；
   - **Phase 2: 原子落盘新配置 (Atomic Target Commit，无条件 `finally` 释放 FD)**：
     - 通过 `_open_atomic_tmp` 打开临时配置文件 `tmp_path = f"{target_path}.tmp.{uuid.uuid4().hex}"`；
     - 在 `try...finally` 中执行写入与 fsync：
       ```python
       tmp_fd = -1
       try:
           tmp_fd = _open_atomic_tmp(tmp_path)
           _write_all(tmp_fd, new_raw_bytes)
           _safe_fsync_file(tmp_fd)
       finally:
           if tmp_fd >= 0:
               _safe_close(tmp_fd, raise_on_error=True)
       ```
     - 原子替换目标文件：`_safe_replace(tmp_path, target_path)`；
     - 调用 `_safe_fsync_dir(dirpath)` 确保新配置元数据物理落盘；
   - **异常补偿与安全全量内存恢复 (Safe Symlink-Free Restore & State Sync)**：
     - 若在 Phase 2 的写入、替换或目录落盘中抛出异常：
       ```python
       try:
           restore_tmp = f"{target_path}.restore.tmp.{uuid.uuid4().hex}"
           dst_fd = -1
           try:
               dst_fd = _open_atomic_tmp(restore_tmp)
               src_fd = -1
               try:
                   src_fd = _safe_open_readonly(backup_path)
                   st = os.fstat(src_fd)
                   if not stat.S_ISREG(st.st_mode):
                       raise ConfigPersistenceError("READ_FAILED", "Backup file is not a regular file")
                   if st.st_uid not in {os.getuid(), 0}:
                       raise SecurityError("Backup file owner untrusted")
                   if stat.S_IMODE(st.st_mode) != 0o600:
                       raise SecurityError(f"Backup file mode must be exactly 0600, got {oct(st.st_mode)}")
                   backup_bytes = _read_all_bounded(src_fd, MAX_CONFIG_SIZE)
               finally:
                   if src_fd >= 0:
                       _safe_close(src_fd)

               # 1. 严格使用 expected_prior_file_sha256 校验备份源文件原始比特完整性
               if hashlib.sha256(backup_bytes).hexdigest() != expected_prior_file_sha256:
                   raise ConfigPersistenceError("READ_FAILED", "Backup file binary hash corrupted")
               _write_all(dst_fd, backup_bytes)
               _safe_fsync_file(dst_fd)
           finally:
               if dst_fd >= 0:
                   _safe_close(dst_fd, raise_on_error=True)

           _safe_replace(restore_tmp, target_path)
           _safe_fsync_dir(dirpath)

           # 2. 严格使用 expected_prior_config_etag 校验恢复后的语义配置等价性
           restored_data = self._read_and_verify_config(target_path)
           restored_etag = self._calculate_etag(restored_data)
           if restored_etag != expected_prior_config_etag:
               raise ConfigPersistenceError("READ_FAILED", "Restored semantic config ETag mismatch")

           # 3. 完整同步全部内存状态字段，杜绝内存态与磁盘态失步
           self._persisted_mcp_servers = restored_data
           self._persisted_config_etag = restored_etag
           self._persisted_file_sha256 = expected_prior_file_sha256
           self._pending_restart = (self._active_mcp_servers != self._persisted_mcp_servers)
       except Exception as recovery_err:
           self._config_recovery_required = True
           write_persistent_recovery_marker(dirpath, "phase2_recovery_failed")
           logger.critical("FATAL: Config recovery failed, locked into read-only")
           raise
       ```
   - 成功后更新内存配置、ETag 与 `file_sha256`，设置 `pending_restart = True`。

### 4.4 重启状态转移矩阵与代数看门狗 (`POST /api/control/restart_agent`)
1. **状态转移合法性矩阵与明确触发原因约束**：
   - 严格定义状态转移白名单，禁止任何越权或非法逆向转移：
     ```python
     # 状态流转契约：
     # idle -> restarting: 用户请求 /api/control/restart_agent 且验证通过
     # restarting -> idle: 仅在 asyncio.create_task 调度失败时安全回退
     # restarting -> restart_failed: 看门狗超时检测进程仍存活，或延迟退出抛出未捕获异常
     # restart_failed -> restarting: 用户再次发起重启重试
     ALLOWED_TRANSITIONS = {
         "idle": {"restarting"},
         "restarting": {"restart_failed", "idle"},
         "restart_failed": {"restarting"},
     }

     async def _transition_restart_state(self, gen: int, new_state: str, reason: str = "") -> bool:
         async with self._restart_lock:
             if self._restart_generation != gen:
                 logger.warning("Ignoring stale state transition to %s (gen %d != %d, reason=%s)",
                                new_state, gen, self._restart_generation, reason)
                 return False
             allowed = ALLOWED_TRANSITIONS.get(self._is_restarting, set())
             if new_state not in allowed:
                 logger.error("Illegal restart state transition: %s -> %s (gen %d, reason=%s)",
                              self._is_restarting, new_state, gen, reason)
                 return False
             logger.info("Restart state transition: %s -> %s (gen %d, reason=%s)",
                         self._is_restarting, new_state, gen, reason)
             self._is_restarting = new_state
             return True
     ```
2. **端点调度逻辑与安全生命周期管理 (零 raw-JSON 响应保证)**：
   - 在 `@mcp_api_boundary` 保护下：
     - 校验 `X-Admin-Token`（不匹配抛出 `DashboardApiError("UNAUTHORIZED", ...)`）；
     - 校验 CSRF（不匹配抛出 `DashboardApiError("FORBIDDEN", ...)`）；
     - 在 `async with self._restart_lock` 保护下：
       - 若 `self._shutting_down` 为 True，**直接抛出 `DashboardApiError("SERVICE_UNAVAILABLE", "Agent is shutting down")`**；
       - 若 `_is_restarting not in ("idle", "restart_failed")`，**直接抛出 `DashboardApiError("CONCURRENT_MODIFICATION", "Agent is already restarting")`**；
       - 自增代数 `self._restart_generation += 1`，记录本轮代数 `current_gen = self._restart_generation`；
       - 设置 `self._is_restarting = "restarting"`；
     - **绑定代数检验看门狗与退出任务**：
       ```python
       async def _do_restart(gen: int):
           try:
               await asyncio.sleep(0.5)
               app._restart_exit_fn(0)
               # 检查真实进程存活状态
               await asyncio.sleep(2.0)
               try:
                   os.kill(os.getpid(), 0)
                   is_alive = True
               except OSError:
                   is_alive = False
               if is_alive:
                   await self._transition_restart_state(gen, "restart_failed", reason="process_alive_timeout")
                   logger.error("Agent process still alive after restart signal")
           except Exception as e:
               logger.error("restart execution failed: %s", e)
               await self._transition_restart_state(gen, "restart_failed", reason=f"exception:{e}")

       try:
           shutdown_task = asyncio.create_task(_do_restart(current_gen), name="delayed-restart")
           self._tracked_tasks.add(shutdown_task)
           shutdown_task.add_done_callback(lambda t: self._tracked_tasks.discard(t) if hasattr(self, "_tracked_tasks") else None)

           def _on_shutdown_done(t: asyncio.Task):
               # 处于停机清理中直接退出，禁止衍生新任务
               if getattr(self, "_shutting_down", False):
                   return
               if t.cancelled() or t.exception():
                   fail_task = asyncio.create_task(
                       self._transition_restart_state(current_gen, "restart_failed", reason="task_done_error")
                   )
                   self._tracked_tasks.add(fail_task)
                   fail_task.add_done_callback(lambda ft: self._tracked_tasks.discard(ft) if hasattr(self, "_tracked_tasks") else None)

           shutdown_task.add_done_callback(_on_shutdown_done)
       except Exception as e:
           # 创建失败必须严格通过统一转移函数回退至 idle，并抛出 DashboardApiError 由边界统一封装
           await self._transition_restart_state(current_gen, "idle", reason=f"create_task_failed:{e}")
           raise DashboardApiError("INTERNAL_ERROR", f"Failed to schedule restart: {e}") from e

       return web.json_response({"ok": True, "message": "Agent restarting, poll /api/agent/settings for recovery"})

       async def shutdown_app(self):
           # 1. 在 _restart_lock 保护下设置停机标记，彻底阻断新请求
           async with self._restart_lock:
               self._shutting_down = True

           # 2. 执行循环清空，杜绝 callback 新增任务遗漏
           while self._tracked_tasks:
               current_batch = list(self._tracked_tasks)
               for task in current_batch:
                   if not task.done():
                       task.cancel()
               if current_batch:
                   await asyncio.gather(*current_batch, return_exceptions=True)
               for task in current_batch:
                   self._tracked_tasks.discard(task)
       ```

---

## 5. 前端实施边界与 SenseCraft-UI 规范 (Claude Code 任务定义)

### 5.1 实施文件白名单
严格限定在以下 3 个静态文件：
- `agent/ovs_agent/plugins/static/dashboard.html`
- `agent/ovs_agent/plugins/static/dashboard.css`
- `agent/ovs_agent/plugins/static/dashboard.js`
不得修改任何后端 Python 文件、构建配置或包依赖。

### 5.2 现有功能与 DOM 契约保护
严格保留现有所有交互组件与 ID，禁止破坏既有语音与重启功能：
- 语音与意图表单：`setPipelineMode`, `setSleepTimeout`, `setStopWords`, `setExitWords`, `setWakeReply`, `setExitReply`, `btnAgentSettingsSave`
- 唤醒词表单与操作：`wwPhrases`, `btnWwSave`, `btnWwValidate`, `btnWwTone`, `wwBadge`, `wwHint`
- 重启按钮：`btnRestartAgent`, `btnRestartAgentMenu`
- 模式徽标：`modeRestartTag`

### 5.3 前端组件与重启联动落地
1. **MCP 服务卡片 (`#mcpCard`)**：
   - 插入在语音设置卡片之后；
   - 顶部提供「添加服务」与「刷新」按钮；
   - 状态栏显示：若 `pending_restart === true`，顶部高亮警示条：`⚠ MCP 配置已变更，需重启生效`，附带「立即重启」快捷按钮；
   - 卡片列表展示运行中或待生效的服务信息（协议标签、状态、展开查看工具 Schema、连通性测试、编辑、删除）；
2. **添加/编辑模态框 (`#mcpModal`)**：
   - 读取 `<meta name="ovs-admin-token">` 并注入 `X-Admin-Token`；
   - 提交时附带最新 `If-Match` ETag；
   - 动态切换 `stdio` / `sse` 表单；
   - 底部「测试连接」按钮实时调用 `POST /api/mcp/test` 回显工具清单；
   - 保存成功后弹出 Toast 提示：“MCP 配置已保存”，并在界面上亮起重启提醒。
3. **SenseCraft-UI Token 变量映射与响应式**：
   - 严格采用已有的 `--bg: #0b0f19`, `--card: #131b2e`, `--primary: #00d287`, `--text-main: #f0f4fc` 等标准变量；
   - 移动端排版：在 375px 下按钮栏与参数行开启 `flex-wrap: wrap; gap: 8px;`，保证 0 横向溢出。

---

## 6. 自动化测试验收矩阵 (`agent/tests/test_mcp_dashboard_api.py`)

| 测试 ID | 场景分类 | 断言与验证行为 |
|---|---|---|
| **T1** | 会话 Token 与 CSRF | 缺失/错误 `X-Admin-Token` 抛出 `UNAUTHORIZED` 返回 401（含 GET）；GET 带伪造 Referer 抛出 `FORBIDDEN` 返回 403，无 Referer 正常放行；写操作伪造 Origin 抛出 `FORBIDDEN` 返回 403；断言所有错误响应均携带 `request_id` |
| **T2** | stdio 白名单与 `/proc/self/fd` 内核执行防 TOCTOU | 提交非白名单脚本或含 `-c`、Shell 元字符返回 400；断言 `command` 严格使用打开验证后的 `/proc/self/fd/{exe_fd}` 执行；模拟父目录权限篡改或文件非普通文件断言抛出 `SecurityError` 并映射为 HTTP 400；断言启动无论成败父进程 `exe_fd` 均被 `finally` 释放；验证子进程仅继承 `exe_fd`，不继承其他带 `O_CLOEXEC` 的配置 FD |
| **T3** | sse 反 SSRF 与代理阻断 | 目标解析到私网 IP 直接阻断并抛出 `INVALID_PARAM` (HTTP 400)；验证独立 Resolver 绑定首 IP，`trust_env=False` 忽略外部代理；HTTP 重定向直接拒绝报错 |
| **T4** | 敏感信息脱敏与凭据保留 | GET 验证 Token 脱敏为 `******`；更新时验证 `******` 保留原值，空字符串删除 key |
| **T5** | 强 ETag / If-Match 乐观锁 | 缺失 `If-Match` 或 ETag 过期调用 POST/DELETE 抛出 `ETAG_MISMATCH` 返回 409 Conflict |
| **T6** | 独立测试沙箱与进程组回收 | 验证正常退出时 ProcessLookupError 被幂等捕获且工具成功返回；超时后进程组全部被回收无僵尸进程；测试失败统一抛出 `CONNECTION_FAILED` 返回 502 |
| **T7** | 并发测试槽位限制 (Semaphore) | 同时发起 3 个测试请求，验证第 3 个请求抛出 `TOO_MANY_REQUESTS` 返回 429 Too Many Requests |
| **T8** | 双散列域独立、安全恢复与状态同步 | **针对包含 YAML 格式与非规范格式化 JSON**：断言原始字节 Hash 与语义 ETag 独立校验成功，绝无误判；新写入内容或旧文件超过 2MB 直接拦截并抛出 `FILE_TOO_LARGE` (HTTP 400)；Phase 1 backup 失败且 backup 文件损坏时触发 `CONFIG_RECOVERY_REQUIRED` (HTTP 500)；Phase 2 失败时 restore 使用 `_safe_open_readonly` 拒绝符号链接并经 `_read_all_bounded()` 杜绝长度并发竞态；**恢复成功后断言 `persisted_mcp_servers`、`persisted_config_etag`、`persisted_file_sha256` 与 `pending_restart` 均完整恢复**；验证 Phase 1/Phase 2 各步骤在异常时临时 FD 均被 `finally` 无条件关闭；验证客户端所有响应严格通过 `SAFE_CLIENT_MESSAGES` 脱敏且携带 `request_id`，绝无内部路径与系统堆栈泄漏 |
| **T9** | 运行态分离与重启代数看门狗 | 验证保存后返回 `pending_restart: true`；运行中连接保持不变；断言 `_transition_restart_state()` 严格按白名单转移，非法转移（如 restart_failed -> idle）被拒绝；`create_task` 异常通过 `_transition_restart_state(gen, "idle", reason)` 安全回退；断言在停机态下发起重启请求在锁内被直接阻断抛出 `SERVICE_UNAVAILABLE` 返回 503；所有异步任务在应用停机时经循环清空优雅取消和消费，停机中回调不衍生新任务 |
| **T10**| 平台能力与启动双重锁死防护 | 平台缺少 `os.O_NOFOLLOW` 或 `os.O_DIRECTORY` 时抛出 `RuntimeError` 拒绝启动；启动扫描发现未完成的 `.config_recovery_required.marker_tmp.<32-hex>` 孤立文件或正式标志文件时，断言经 `O_NOFOLLOW`、精确 `0600` 模式及严格正则语义格式校验检测到标志文件并维持锁死保护；断言写入持久恢复标志失败时经 `fatal_config_recovery_exit()` 触发 Exit 78 硬阻断 supervisor；断言两张映射表在模块导入时严格执行类型、非空、集合双射及合法状态码枚举断言 |
