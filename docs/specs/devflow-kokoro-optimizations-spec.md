# Dev-Flow Engineering Spec V2: Kokoro TTS Latency Optimization

**Spec ID**: DEVFLOW-KOKORO-OPT-20261104-V2  
**Status**: APPROVED-CANDIDATE (Post-Codex Phase 1 Revision)  
**Target Hardware**: RK3588 (Seeed reComputer / Cortex-A76 + 3-Core NPU) & RK3576  
**Scope**: 
- **Scheme A**: 发音人 Style 特征预热与动态切换缓存 (Speaker Style Warm-up & Concurrency-Safe Dynamic Cache)
- **Scheme B**: 文本前端定长 Token 分桶上 NPU 架构与严密 Attention Mask 约束 (Text Frontend Fixed-Bucket RKNN Pipeline with Strict Attention Mask)

---

## 一、方案 A：发音人 Style 预热与动态切换缓存 (Scheme A)

### 1.1 核心目标 (Goal)
在固定发音人或频繁切换发音人场景下，消除声学前端由于磁盘 I/O、内存重排及 ORT Session 首次冷调用造成的首包延迟抖动（30~50ms 尖刺）。建立并发安全的发音人状态机，保障多会话并发切人时每个请求的 Style 张量严格不可变且隔离。

### 1.2 状态机与并发契约 (Concurrency State Machine & Invariants)
1. **Per-Voice 状态机**：
   每个 Route/Voice 维护明确状态：`COLD -> WARMING -> READY | FAILED`。
   - 引入线程条件变量 `threading.Condition`。
   - 当多个并发请求或 API 显式触发同一个 Voice 的预热时，仅有一个 Worker 真正执行预热逻辑，其余并发请求通过 Condition 等待其变为 `READY`，绝不重复触发预热。
   - 若预热失败，状态置为 `FAILED` 并向等待者广播抛出原始异常，后续请求可重试。
2. **请求不可变性 (Zero Shared Mutable Speaker)**：
   - 严禁在 `KokoroLong32Frontend` 内维护全局可变的“当前发音人”指针。
   - 发音人 Style 特征在 `prepare()` 时一旦由 `_voice(route, n)` 提取，立即封装进只读不可变的 `Prepared` 数据载体中，并设置 `setflags(write=False)`，彻底消除在流水线飞行中（In-flight）被外部切人修改的风险。
3. **预热前向与资源边界 (Safe Dry-Run Protocol)**：
   - `warmup_voice(route_or_voice)` 使用合法的最小探测样本（如短字符 `"a"`，3 个 Token，固定速度 1.0）。
   - 预热只用于促使 ORT 线程池就绪及分配器底层 Arena 触碰页面，**其计算产物直接丢弃**，不改变任何对外业务状态。
   - 内存预算锁定：9 个官方预置发音人（各 510×1×256 float32）常驻总内存仅为 **4.47 MiB**，直接常驻内存无须 LRU 逐出，杜绝重复从磁盘加载。

### 1.3 冻结契约 (Frozen Contract)
```python
def warmup_voice(self, route_or_voice: str) -> dict[str, float]:
    """Concurrency-safe voice pre-warmup.
    Guarantees voice blob loaded and dry-run forward executed.
    Returns: {"voice_load_ms": float, "dry_run_ms": float, "total_ms": float}
    """
```
- **超时与重试**：预热操作受单次超时约束（默认 5.0s）。

### 1.4 验收标准 (Acceptance Criteria)
1. **并发切人测试**：16 线程并发交替请求 `zf_xiaobei` 与 `af_heart`，无死锁、无竞争冲突，每个请求产出的音频音色与其请求的 voice 严格 100% 对应。
2. **预热延迟指标**：预热成功后，连续 20 次请求中 `detail["voice"]` 耗时稳定 $\le 0.5\text{ ms}$，`detail["session_init"]` 稳定 $\le 0.5\text{ ms}$。
3. **单测套件**：`third_party/rkvoice-stream/tests/test_kokoro_dualbucket.py` 补充并发预热与切换用例，全部通过。

---

## 二、方案 B：文本前端定长 Token 分桶上 NPU 架构 (Scheme B)

### 2.1 核心目标 (Goal)
将当前在 CPU 上运行的 `frontend_stem.onnx`（BERT 文本编码器 + 风格投影）按定长 32-Token 分桶编译为 RKNN 模型，迁移至 RK3588 NPU。在现有 Dual-Bucket NPU 声码器（160ms）的基础上，将前端耗时从 30ms 压缩至 **~5ms**，迈出全 NPU 化的第一步。

### 2.2 关键约束：严密 Attention Mask 与 Token 语义规范（解决 Codex Blocker）
1. **禁止无 Mask 裸补零（Prevent Attention Contamination）**：
   - Transformer 自注意力层中，若仅对 `input_ids` 补齐而不提供 `attention_mask`，未填充的有效 Token 会在所有层对 Padding Token 进行 Key/Value 注意力关联，导致有效表征被严重污染。
   - **模型输入契约调整**：
     - `input_ids: int64[1, 32]`
     - `attention_mask: int64[1, 32]`（有效位置为 1，补齐位置为 0）
     - `style: float32[1, 256]`
   - 在图导出中，如果底层 Transformer 原生接受加性或二元 mask，严格打通 mask 节点；若 RKNN 不支持动态 Softmax mask，在导出时通过固定图手术固化为常数 Mask 矩阵。
2. **边界 Token 与 Token 长度换算契约**：
   - Kokoro 编码序列带有首尾边界标记（`[0, *ids, 0]`）。
   - **容量明确界定**：32-position Bucket 能够容纳的有效音素数最大为 $N_{\text{ph}} = 30$（总模型输入长度 $N_{\text{model}} = N_{\text{ph}} + 2 = 32$）。
   - **Padding Token 选型**：避免使用 `0` 作为无意义 Padding（因为 `0` 已被分配给边界标记）；使用模型词表中合法的 `PAD` 符号 ID，且由 `attention_mask=0` 物理切断其注意力权重。
3. **输出形状与下流 ABI 对齐**：
   - RKNN 产物输出固定维度：`stem_text: [1, 32, 640]`, `stem_style: [1, 128]`。
   - **前端 Wrapper 严格截断**：`frontend_stem` RKNN 执行完毕后，Wrapper 必须以只读视图将前 $N_{\text{model}}$ 个有效特征切出：
     $$\text{stem\_text}_{\text{valid}} = \text{stem\_text}[:, :N_{\text{model}}, :]$$
     严格保证下游 `duration_head` 与 `target_tail` 接收到的 `stem_text.shape == (1, N_{\text{model}}, 640)`，严丝合缝对齐现有 ABI。

### 2.3 确定性 Fallback 与生命周期隔离
- **触发条件**：
  1. $N_{\text{ph}} > 30$（超出 32 桶容量）；
  2. RKNN 环境缺失或硬件未探测到；
  3. RKNN 加载或单次推理抛出非致命运行时异常。
- **边界隔离**：
  - RKNN Session 与 CPU ORT Session 互不干扰；
  - 发生 Fallback 时完整记录诊断日志，安全释放临时上下文，由 CPU ORT 保证 100% 兜底产出。

### 2.4 验收标准 (Acceptance Criteria)
1. **数值对齐门槛**：
   - 对 $N_{\text{ph}} \in [1, 30]$ 的所有样本，Padding 区域填入不同噪声值时，有效前缀区域的 `stem_text` 输出保持 $\text{rel\_l2} \le 0.001$（证明 Attention Mask 完全有效，零表征污染）。
   - 对比 CPU ORT 黄金基准，NPU 输出的有效前缀 `rel_l2 \le 0.005`。
2. **真机性能**：
   - Seeed reComputer (RK3588) 上 `frontend_stem_b32.rknn` 单次推理耗时 $\le 8\text{ ms}$。
3. **回退容错**：
   - 输入 31 个音素的文本自动优雅回退 CPU ORT，无异常崩溃，音频正常发出。

---

## 三、实施分工与执行流水线 (Dev-Flow Phases 2-5)

1. **阶段 2（定向分工实施）**：
   - **任务 1 (后端 Subagent)**：实施方案 A（并发安全预热状态机与切人缓存）。
   - **任务 2 (后端 Subagent)**：实施方案 B（固定 32 桶 Attention Mask 图手术与导出工具脚本）。
2. **阶段 3（双独立交叉审核）**：
   - 由 Codex CLI 独立审查代码实现（并发锁、Mask 节点与数值精度）。
3. **阶段 4 & 5（真机验收与交付）**：
   - 在 Seeed reComputer (RK3588) 上跑通端到端回归，提交标准 Commit。
