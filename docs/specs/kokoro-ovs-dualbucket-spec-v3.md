# Kokoro OVS Dual-Bucket Standard TTSBackend Specification (Spec V13 - Approved Candidate)

## 1. 核心定位与接口契约 (Core Architecture & Interface Contract)

### 1.1 精确类名与异常复用 (Zero Class Hierarchy Conflict)
- **取消异常**：严禁创建子类，直接 `from rkvoice_stream.backends.tts.kokoro_convonly import ConvOnlyCancelled` 并直接抛出原始 `ConvOnlyCancelled` 实例，确保与 `server/core/api_execution.py:100` 的精确类身份匹配完全一致。
- **服务层名称**：后端名称保留标准的 `kokoro_convonly`，并在 `server/core/api_execution.py` 与 `rkvoice_stream/engine/tts.py` 中将 `kokoro_dualbucket` 作为并列名称完全识别，打通异步取消事件注入。
- **继承与接口**：
  ```python
  from rkvoice_stream.engine.tts import TTSBackend
  from rkvoice_stream.backends.tts.kokoro_convonly import ConvOnlyCancelled

  class KokoroConvOnlyBackend(TTSBackend):
      supports_streaming = True

      @property
      def name(self) -> str:
          return "kokoro_convonly"
  ```

---

## 2. 闭环确定性分桶与路由状态机 (Deterministic Dual-Bucket Routing)

### 2.1 输入防空、前端容量拓展与边界 Token 定义 (Input Validation & Frontend Capacity)
- 空文本、纯空白字符串、或经 G2P 转换后有效音素列表为空（如仅包含不可见控制字符、无发音符号）的输入，**必须直接抛出 `ValueError("text must contain valid phonetic tokens")`**，严禁进入后续静音补齐或推理。
- **前端容量拓展**：现有 `KokoroLong32Frontend._ids` 限制为 200，为支撑桶 2 容量，必须将校验上限由 200 拓展至 238（即支持有效音素数 $N_{\text{ph}} \in [1, 238]$）。配套单测须断言 $N_{\text{ph}} = 238$ 正常放行，而 $N_{\text{ph}} = 239$ 抛出 `PhonemeLengthError`。
- **边界 Token 结构**：Kokoro 声学序列包含首尾边界 Token（`[0, *ids, 0]`），因此 duration 向量长度为 $N_{\text{dur}} = N_{\text{ph}} + 2$。每个 token 的预测时长至少为 1 帧，即 $D_{\text{nat}} = \sum d_i \ge N_{\text{dur}} = N_{\text{ph}} + 2 \ge 3$。

### 2.2 确定性路由状态机
```
             输入校验 (非空且有效音素数 1 <= N_ph <= 238, N_dur = N_ph + 2, D_nat >= N_dur)
                                │
             ┌──────────────────┴──────────────────┐
             │ 是否满足桶 1 容量?                   │
             │ (N_dur <= 120 即 N_ph <= 118 且      │
             │  D_nat <= 141)                      │
             └──────────────────┬──────────────────┘
                                │
                     ┌──────────┴──────────┐
                    YES                    NO
                     │                     │
                     ▼                     ▼
             ┌──────────────┐     ┌──────────────────┐
             │ 进入 桶 1    │     │ 是否满足桶 2 容量?│
             │ T=240 (D=120)│     │ (N_dur <= 240 即 │
             │ (细分 snap/  │     │  N_ph <= 238 且  │
             │  静音补齐)   │     │  D_nat <= 282)   │
             └──────────────┘     └────────┬─────────┘
                                           │
                                ┌──────────┴──────────┐
                               YES                    NO
                                │                     │
                                ▼                     ▼
                        ┌──────────────┐     ┌──────────────────┐
                        │ 进入 桶 2    │     │ 触发有限词法队列 │
                        │ T=480 (D=240)│     │ 递归切分子句     │
                        │ (细分 snap/  │     │ (单调缩短字符区间)│
                        │  静音补齐)   │     └──────────────────┘
                        └──────────────┘
```

### 2.3 snap_durations 算法规范与动作细分
所有分桶时长规整严格采用 `rkvoice_stream.backends.tts.kokoro_long32_frontend.snap_durations` 算法实现：
- 当 $\text{excess} \le 0$（全 1 向量）时，使用 `divmod(target_D - d.size, d.size)` 均匀商余分配；
- 当 $\text{excess} > 0$ 时，使用最大余数法（`largest remainder method` 稳定排序）分配剩余帧；
- 强断言不变量：`np.all(out >= 1) and int(out.sum()) == target_D`。

#### 静音填充精准数学定义 (Deterministic Silence Padding)
在超短句（$D_{\text{nat}} < D_{\text{snap\_min}}$）场景下，为避免语速严重失真，严禁对语音音素进行拉伸：
1. **发音区间零形变**：有效语音音素保持 100% 原始天然时长，即 $d_i = d_{\text{nat}, i}$（$\forall i \in [0, N_{\text{dur}} - 2]$，涵盖 BOS 与全部语音 token）；
2. **尾部 EOS 吸收差额**：差额帧数 $\Delta D = D_{\text{target}} - D_{\text{nat}}$ 全部追加至末尾边界 Token（EOS，Token ID 为 0）：
   $$d_{N_{\text{dur}} - 1} = d_{\text{nat}, N_{\text{dur}} - 1} + (D_{\text{target}} - D_{\text{nat}})$$
   代码实现：`durations = prepared.natural_durations.copy(); durations[-1] += (target_D - int(durations.sum()))`；
3. **渲染与截断**：`target_tail` 接收补齐后的 `durations` 生成完整特征。波形生成后按有效发音帧数截取前 $N_{\text{valid}} = 600 \times D_{\text{nat}}$ 个采样点，末尾 EOS 对应的无声尾迹直接丢弃。

#### 动作细分与有效输出帧数定义
定义**输出有效帧数 $D_{\text{out\_valid}}$**（对应波形截取长度 $N_{\text{valid}} = 600 \times D_{\text{out\_valid}}$）：
1. **桶 1 ($T=240, D_{\text{target}}=120$) 细分动作**：
   - 若 $105 \le D_{\text{nat}} \le 141$：执行 `snap_durations(durations, 120)`，整段均为发音区间，**输出有效帧数 $D_{\text{out\_valid}} = 120$**，输出采样点 $N_{\text{valid}} = 600 \times 120 = 72,000$ 点；
   - 若 $D_{\text{nat}} < 105$（超短句）：按上述静音填充规程，追加至 EOS。**输出有效帧数 $D_{\text{out\_valid}} = D_{\text{nat}}$**，截取前 $N_{\text{valid}} = 600 \times D_{\text{nat}}$ 采样点。
2. **桶 2 ($T=480, D_{\text{target}}=240$) 细分动作**：
   - 若 $209 \le D_{\text{nat}} \le 282$：执行 `snap_durations(durations, 240)`，整段均为发音区间，**输出有效帧数 $D_{\text{out\_valid}} = 240$**，输出采样点 $N_{\text{valid}} = 600 \times 240 = 144,000$ 点；
   - 若 $D_{\text{nat}} < 209$（包括 $142 \le D_{\text{nat}} < 209$ 以及当 $N_{\text{dur}} > 120$ 且 $D_{\text{nat}} < 209$ 时溢出降级入桶 2 的情况）：按上述静音填充规程，追加至 EOS。**输出有效帧数 $D_{\text{out\_valid}} = D_{\text{nat}}$**，截取前 $N_{\text{valid}} = 600 \times D_{\text{nat}}$ 采样点。
3. **超长切分兜底**：
   - $D_{\text{nat}} > 282$ 或 $N_{\text{dur}} > 240$（即 $N_{\text{ph}} > 238$）：进入词法切分队列，沿标点/空格切分，保证 `sub_len < current_len`；各子句独立执行上述决策树并流式 `yield`。不可分且超限时抛出 `ValueError("single source character exceeds bounded synthesis duration")`。

---

## 3. 0 CPU Fallback 全节点放置报告门禁 (Full-Node Placement Gate)

在真机验收时，不依赖单一字符串匹配，必须**正向解析并断言全图节点放置**：
1. **全图节点执行归属断言**：
   - 导出与运行时开启全节点 profiling（`RKNN_LOG_LEVEL=3` / `query_perf_detail`）；
   - 提取 Generator 的全部 ONNX 算子（Conv, Add, Mul, LeakyRelu, GAP 等）映射表，断言 **100% 的节点均标记为 `Device/NPU` 执行**，CPU 节点列表必须为空（`cpu_nodes == []`）；
   - 若报告缺失、日志被截断或存在未决节点，直接判定门禁失败；
2. **双板双桶全覆盖**：
   - 四份产物（`rk3588/T240`, `rk3588/T480`, `rk3576/T240`, `rk3576/T480`）分别单独执行上述全节点放置断言，必须逐一出具证据。

---

## 4. 声学客观门禁唯一确定性数学规程 (Determined Acoustic Protocol)

### 4.1 统一波形预处理与区间截取
1. **基准生成**：使用 CPU 原生 PyTorch 浮点环境，输入相同的最终 target duration 向量（与 NPU 端完全相同的 $D_{\text{target}}$ 及静音补齐帧数），使用相同的随机种子与 voice/style 向量，生成参考波形 $y_{\text{ref}}$ 与 NPU 波形 $y_{\text{npu}}$；
2. **截取有效区间**：**先截取波形，再计算频谱**。两路波形均严格截取前 $N_{\text{valid}} = 600 \times D_{\text{out\_valid}}$ 个采样点（完全对齐第 2.3 节定义）；
3. **固定尺度**：严禁采用逐条峰值归一化，两路波形均按绝对 PCM 浮点刻度 $y \in [-1.0, 1.0]$ 直接输入 STFT。

### 4.2 STFT 与 Mel 投影数学定义 (Strict Librosa Slaney-Scale Protocol)
- 采样率 24,000 Hz；
- 窗函数：`n_fft = 1024`, `win_length = 1024` (采用 `periodic=True` 的标准 Hanning 窗), `hop_length = 256`, `center = True` (两端反射填充 `reflect`)；
- 线性幅度谱：$A = |\text{STFT}(y)| \in \mathbb{R}^{513 \times T_{\text{frames}}}$（STFT 采用标准无归一化模长）；
- **Mel 滤波器组矩阵 $M \in \mathbb{R}^{80 \times 513}$ 唯一定义**：
  - 严格采用 **Slaney 频率标度与 Slaney 面积归一化**（对应 `librosa.filters.mel(sr=24000, n_fft=1024, n_mels=80, fmin=0.0, fmax=12000.0, htk=False, norm="slaney")`）；
  - 频带范围：$f_{\text{min}} = 0.0\text{ Hz}$, $f_{\text{max}} = 12000.0\text{ Hz}$；
- **Mel 投影幅度谱**：$S = M @ A \in \mathbb{R}^{80 \times T_{\text{frames}}}$；
- **对数 Mel 能量谱 (dB)**：$L(m, t) = 20 \log_{10}(\max(S(m, t), 10^{-5}))$。

### 4.3 聚合断言门禁
$$\text{Mel-L1} = \frac{1}{80 \times T_{\text{frames}}} \sum_{m=1}^{80} \sum_{t=1}^{T_{\text{frames}}} |L_{\text{npu}}(m, t) - L_{\text{ref}}(m, t)| \le 0.8\text{ dB}$$
- 伴随检查：两路波形非有限值比例严格为 0.0%，有效信号 RMS $\ge 0.01$。

### 4.4 严密可复现验收测试矩阵 (Acceptance Test Matrix)

#### 4.4.1 固定四象限测试语料 (Fixed Benchmarks)
所有用例均严格固定 `speed = 1.0`，随机种子 `seed = 42`：
1. **Case A (桶 1 静音填充分支, 断言要求 $D_{\text{nat}} < 105$)**：
   - 英文元组：`(text="Hello world.", language="en-US", voice="af_heart", speed=1.0)`
   - 中文元组：`(text="你好，世界。", language="zh", voice="zf_xiaobei", speed=1.0)`
2. **Case B (桶 1 Snap 规整分支, 断言要求 $105 \le D_{\text{nat}} \le 141$)**：
   - 英文元组：`(text="The system delivers real-time voice synthesis on edge neural processing units.", language="en-US", voice="af_heart", speed=1.0)`
   - 中文元组：`(text="边缘智能语音合成引擎在嵌入式神经网络处理器上实现了极速低延迟实时运行。", language="zh", voice="zf_xiaobei", speed=1.0)`
3. **Case C (桶 2 静音填充分支, 断言要求 $142 \le D_{\text{nat}} < 209$)**：
   - 英文元组：`(text="Artificial intelligence at the edge provides local real-time processing without network latency.", language="en-US", voice="af_heart", speed=1.0)`
   - 中文元组：`(text="矽递科技边缘计算平台能够在离线环境下提供完全自主可控的高性能自然语音交互体验。", language="zh", voice="zf_xiaobei", speed=1.0)`
4. **Case D (桶 2 Snap 规整分支, 断言要求 $209 \le D_{\text{nat}} \le 282$)**：
   - 英文元组：`(text="Seeed Studio Edge Voice Engine provides fully local high-performance speech recognition and natural sounding audio synthesis on Rockchip platforms.", language="en-US", voice="af_heart", speed=1.0)`
   - 中文元组：`(text="双分桶确定性声学路由状态机通过精确的时长规整与末端静音填充算法，全面兼顾短句低延迟与长句完整韵律，杜绝时间轴拉伸失真。", language="zh", voice="zf_xiaobei", speed=1.0)`

#### 4.4.2 分支归属前置强断言 (Preflight Branch Membership Assertion)
测试套件在进入推理与声学评估前，必须首先断言各 Case 的运行时声学特征满足其所宣称的分支区间，任何落入区间之外的情况直接判为测试失败：
- Case A：断言 `D_nat < 105` 且 `selected_bucket == "T240"` 且 `branch == "padding"`；
- Case B：断言 `105 <= D_nat <= 141` 且 `selected_bucket == "T240"` 且 `branch == "snap"`；
- Case C：断言 `142 <= D_nat < 209` 且 `selected_bucket == "T480"` 且 `branch == "padding"`；
- Case D：断言 `209 <= D_nat <= 282` 且 `selected_bucket == "T480"` 且 `branch == "snap"`。

#### 4.4.3 产物验收映射与隔离规则 (Artifact-to-Case Mapping)
测试套件严格遵循确定性路由契约，禁止将长句输入送入短桶产物：
- **端到端标准路由验收 (End-to-End Routing Acceptance)**：
  - **RK3588 平台**：
    - Case A 与 Case B 经由路由分发至 `models/tts/kokoro/rk3588/generator_T240.rknn` 执行；
    - Case C 与 Case D 经由路由分发至 `models/tts/kokoro/rk3588/generator_T480.rknn` 执行；
  - **RK3576 平台**：
    - Case A 与 Case B 经由路由分发至 `models/tts/kokoro/rk3576/generator_T240.rknn` 执行；
    - Case C 与 Case D 经由路由分发至 `models/tts/kokoro/rk3576/generator_T480.rknn` 执行；
  - 所有 8 组端到端测试（2 个平台 $\times$ 4 个 Case）必须独立达到 $\text{Mel-L1} \le 0.8\text{ dB}$，非有限值比例 $0.0\%$，$\text{RMS} \ge 0.01$。
- **强制分桶诊断测试 (Forced-Bucket Diagnostics, 独立隔离)**：
  - 单独设立单元测试，通过固定输入特征直连底层算子测试边界稳定性，明确标注为内部诊断，不混淆进入端到端路由验收报告。

---

## 5. 产物交付清单与 Manifest 强校验 (Artifact Deliverables)

- **4 份按架构严格隔离的 RKNN 产物**：
  - `models/tts/kokoro/rk3588/generator_T240.rknn`
  - `models/tts/kokoro/rk3588/generator_T480.rknn`
  - `models/tts/kokoro/rk3576/generator_T240.rknn`
  - `models/tts/kokoro/rk3576/generator_T480.rknn`
- **Manifest (`kokoro_dualbucket.manifest.json`)**：
  每份产物独立登记 SHA256、文件大小、输入输出尺寸、平台架构与编译版本 (`2.3.2`)。运行时加载前执行强校验，不匹配拒绝启动。
