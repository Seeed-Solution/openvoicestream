# Dev-Flow Engineering Spec: Kokoro TTS Production Solidification & Release

**Spec ID**: DEVFLOW-KOKORO-RELEASE-20261104-V1  
**Status**: APPROVED (Post-Codex Phase 1 Revision)  
**Target Hardware**: Seeed reComputer (RK3588) (4x A76 + 4x A55, 3-Core NPU) / RK3576  
**Scope**: 镜像固化构建契约、一键部署流水线集成、用户文档与 Runbook 交付、开源协议合规与 NOTICE 致谢。

---

## 阶段 1：Codex 独立审查意见与修复结论 (Codex Phase 1 Review Integration)
1. **Manifest 结构对齐**：在 `deploy/artifacts/rk_manifest.json` 中以标准 `artifact_sets` 结构完整注入 `rk3588-kokoro-dualbucket-20261104`，按 `opt/kokoro-dualbucket/rk3588/` 映射 21 个物理文件及 manifest 自身，并附带总 SHA256。
2. **部署脚本接入**：`deploy/install.sh` 增加 `--profile` 选项（或识别 `OVS_PROFILE=rk3588-kokoro-dualbucket`），当检测到该 profile 时，自动追加 `deploy/docker-compose.kokoro-dualbucket-rk3588.yml`，并预检宿主机 `/opt/kokoro-dualbucket/rk3588` 路径完整性。
3. **Runbook 与 NOTICE 补齐**：补齐 `docs/runbooks/kokoro-dualbucket-deploy.md` 与根目录 `NOTICE` 文件。
4. **镜像构建定义与 Tagging 规范**：在 `deploy/IMAGE-TAGS.md` 中记录包含该能力的正式 Tag 规格，说明离线构建与发布步骤。

---

## 一、核心目标 (Goal)
将已在真机完成算法与性能验证的 Kokoro TTS（CPU 文本前端 ONNXRuntime + NPU T50 窗口流式声学解码器）闭环交付为生产就绪状态：
1. 固化镜像定义与 `IMAGE-TAGS.md` 规格记录；
2. 补齐 `deploy/artifacts/rk_manifest.json` 与 `deploy/install.sh`，实现一键部署与自动化模型包校验；
3. 更新中英文 `README` 与新增操作 Runbook；
4. 建立根目录 `NOTICE` 文件与致谢章节，严格符合 Apache 2.0 与 MIT 开源许可规范。

---

## 二、模块拆分与改动清单 (File Manifest)

### 模块 1：镜像固化与构建规范 (Image Solidification)
- **目标文件**：
  - `deploy/docker/Dockerfile.rk`: 确认 `final-convonly` / `final-slim` 对 `rkvoice-stream` 最新 wheel 构建的闭环支撑。
  - `deploy/IMAGE-TAGS.md`: 登记最新镜像 Tag（`openvoicestream:rk-20261104-kokoro-dualbucket`）、基线 Git Commit、依赖版本及 SHA256 签名。
  - `deploy/docker-compose.kokoro-dualbucket-rk3588.yml`: 镜像名称收敛为参数化变量 `${OVS_SPEECH_IMAGE:-sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-20261104-kokoro-dualbucket}`。

### 模块 2：一键部署与 Artifact 登记 (One-Click Deployment & Manifest)
- **目标文件**：
  - `deploy/artifacts/rk_manifest.json`: 在 `artifact_sets` 中登记 `rk3588-kokoro-dualbucket-20261104` 条目，收录全部 21 个模型与前端文件、尺寸及 SHA256 签名，绑定统一的 `manifest_sha256 = cfc6aa8d485415884e4cf5eac47b0a786889ed02d9556a1051ed66685927083d`。
  - `deploy/install.sh`: 在目标与 profile 解析逻辑中接入 `rk3588-kokoro-dualbucket`，支持用户通过 `--profile=rk3588-kokoro-dualbucket` 一键创建目录、校验模型、拉起容器。

### 模块 3：用户文档与 Runbook (Documentation & Runbook)
- **目标文件**：
  - `docs/runbooks/kokoro-dualbucket-deploy.md`: 新增部署操作手册，包含硬件需求、一键部署命令、离线模型包解压路径（`/opt/kokoro-dualbucket/rk3588`）、线程环境变量与真机性能基准（TTFT 383~414ms, RTF 0.380~0.424）。
  - `README.md` 与 `README.zh-CN.md`: 在模型矩阵与快速启动章节中，补充 Kokoro Dual-Bucket（CPU 前端 + NPU T50 窗口流式）的介绍与命令。保留所有现有章节。

### 模块 4：开源协议合规与致谢 (License Compliance & Attribution)
- **目标文件**：
  - `NOTICE`: 在仓库根目录新建标准 NOTICE 文件，声明：
    - Kokoro-82M (hexgrad/kokoro, Apache License 2.0)
    - Misaki G2P (hexgrad/misaki, Apache License 2.0)
    - StyleTTS 2 (yl4579/StyleTTS2, MIT License)
  - `README.md` 与 `README.zh-CN.md`: 在 `## Acknowledgements` / `## 致谢` 章节中补充上述项目的致谢信息。

---

## 三、冻结契约 (Frozen Contracts)

1. **Manifest 与路径契约**：
   - 宿主机路径：`/opt/kokoro-dualbucket/rk3588`
   - 容器挂载：`/opt/kokoro-dualbucket:/opt/kokoro-dualbucket:ro`
   - Manifest 文件：`/opt/kokoro-dualbucket/rk3588/kokoro_dualbucket.manifest.json`
   - 固定 SHA256：`cfc6aa8d485415884e4cf5eac47b0a786889ed02d9556a1051ed66685927083d`
2. **Profile 契约**：
   - Profile 名称：`rk3588-kokoro-dualbucket`
   - `TTS_BACKEND`: `kokoro_dualbucket`
   - `KOKORO_DEFAULT_VOICE`: `af_heart`
   - `KOKORO_FRONTEND_INTRA_OP_THREADS`: `4`
3. **开源许可契约**：
   - Kokoro-82M 与 Misaki 基于 Apache 2.0，遵守 Section 4(d) 声明保留；
   - StyleTTS 2 基于 MIT，遵守版权声明保留；
   - 严禁擅自引入 GPL/AGPL 强传染性依赖。

---

## 四、验收标准 (Acceptance Criteria)

1. **配置与清单校验**：
   - `pytest server/tests/test_kokoro_dualbucket_execution.py` 全部通过；
   - `deploy/artifacts/rk_manifest.json` 语法有效，包含完整 21 个文件的 SHA256 与路径；
   - `deploy/install.sh --help` 或 profile 解析包含 `rk3588-kokoro-dualbucket` 对应规则。
2. **合规审查**：
   - `NOTICE` 文件语法规范，版权所有者、项目 URL 与 License 声明齐全；
   - `README.md` 与 `README.zh-CN.md` 包含 Acknowledgements 章节对应项目。
3. **无破坏性操作**：
   - 严禁影响 RK3588 上正在运行的 8621 生产服务（保持 `Up 4 weeks (healthy)`）。
