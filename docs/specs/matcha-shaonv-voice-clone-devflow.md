# Dev Spec: Matcha-TTS 中英双语底座 MiniMax 少女音色微调与交付工作流 (v3.3)

- **Author**: Antigravity CTO / Architect
- **Status**: Final Specification for Implementation (Phase 1 Closure)
- **Target Voice**: MiniMax `female-shaonv` (少女亲和)
- **Dataset**: `voices/minimax_shaonv/` (102 句，总计 424.22 秒，约 7.07 分钟)
- **Dataset Split**: 
  - 训练集 (Train): 91 句 (`voices/minimax_shaonv/train.txt`)
  - 保留验证集 (Val): 11 句 (`voices/minimax_shaonv/val.txt`)，完全隔离，用于训练中早停、最佳 Checkpoint 选取及阶段 2 触发判定。
  - 真正未见终验集 (Frozen Unseen Test): 5 句独立全新文本（未在 102 句数据集及任何既有脚本中出现，仅用于阶段 5 终验）。

---

## 1. 核心目标 (Goal)

基于 `models/matcha-icefall-zh-en/pytorch_model.bin` 官方中英双语底座，通过**特征协议严格锁定、音素级 MAS 单调对齐、阶梯式 L2-SP 正则化微调与无损 ONNX 导出**，产出具备 MiniMax 少女音色特征、发音清晰、无金属毛刺/相位撕裂杂音的模型，并通过包含纯中文、纯英文、中英切换、专有名词与数字的未见测试集 ASR 闭环验收。

---

## 2. 职责切分与范围 (Boundaries)

- **前端/UI**: 无。本项目为纯后端/边缘语音管线任务。
- **阶段 0/1 (主控架构)**: 需求对齐、Spec 制定、独立审查闭环与裁决。
- **阶段 2 (实施阶段)**:
  - **任务 2.1**: 特征协议冻结与公共特征模块实现 (`tools/voice_pipeline/audio_features.py` 与 `step0_vocos_baseline.py`)。
  - **任务 2.2**: 规范音素级 MAS 对齐与阶梯式 L2-SP 正则化微调 (`tools/voice_pipeline/finetune_official_matcha_v3.py`)。
  - **任务 2.3**: 3-Step 与 6-Step ONNX 无损导出及 PyTorch/ONNX 双重数值等价门禁 (`tools/voice_pipeline/export_onnx_step3_step6.py`)。
  - **任务 2.4**: 全量未见测试集自动化 ASR 评测与 MacBook 推送 (`tools/voice_pipeline/verify_asr_and_push.py`)。
- **阶段 3 (独立代码交叉审查)**: Codex 深度代码与契约审查。
- **阶段 5 (交付验收)**: 全量未见集 ASR 门禁通过、MacBook 音质试听验收、规范 Git 提交。

---

## 3. 冻结契约 (Frozen Contract)

### 3.1 三个 Mel 域的严格数学定义与转换契约
为杜绝归一化混乱与重复缩放，严格定义以下三个域及单向转换契约：
1. **真实物理幅度谱域 (Physical Mel)**: 
   $S_{mel} = \text{MelFilterBank} \cdot |\text{STFT}(s)|$ (80 维，`fmin=0, fmax=8000`, `power=1.0`)
2. **非归一化对数 Mel 域 (Unnormalized Log-Mel)**:
   $m = \ln(\max(S_{mel}, \epsilon))$（$\epsilon$ 在 Step 0 实测对比 $10^{-10}$ 与 $10^{-5}$ 后冻结）
   - **Vocos 声码器输入契约**: 必须输入此域 $m$（人声有效值域通常在 $[-12, +5]$，静音段自然衰减至接近 $\ln(\epsilon)$，若 $\epsilon=10^{-10}$ 则可达 $\sim -23$，若 $\epsilon=10^{-5}$ 则约为 $-11.51$）。
   - **ONNX 模型输出契约**: 官方 ONNX 尾部节点通过 `mel = z * 5.446792 + (-2.9521978)` 输出此域 $m$。
   - **显式转换位置**: 反归一化 $z \to m$ 在 ONNX 内部尾部节点（或 PyTorch 模型输出出口）执行且仅执行一次，严禁外部重复反归一化。
3. **标准化流匹配域 (Normalized Flow-Matching Domain)**:
   $z = \frac{m - \text{MEL\_MEAN}}{\text{MEL\_SIGMA}}$（其中 $\text{MEL\_MEAN} = -2.9521978, \text{MEL\_SIGMA} = 5.446792$）
   - **CFM 训练与 MAS 对齐契约**: 模型内部 ODE 速度场 $v_\theta$、先验 $\mu_y$ 与真实目标 $x_1$ 严格在此标准化域 $z$ 中运算（均值 $\approx 0$，标准差 $\approx 1$）。

### 3.2 Vocos 特征协议基准锁定与实测证据 (Step 0)
- **参数配置**:
  - 音频重采样：严格通过 `librosa.resample(..., res_type="soxr_vhq")` 重采样至 16,000 Hz。
  - STFT 参数：`n_fft=1024, hop_length=256, win_length=1024, window="hann", center=True, pad_mode="reflect"`。
  - Mel 滤波组：80 维，`fmin=0.0, fmax=8000.0, norm="slaney", mel_scale="slaney"`。
  - 功率模式：锁定 `power=1.0`（幅度谱模式）。
- **实测原始对比数据**:
  - 测试音频：`voices/minimax_shaonv/wavs/sample_0001.wav`（时长 6.768 秒，参考文本：“您好！我是您的智能语音助手。Welcome to Seeed Studio! 今天有什么可以帮您的？”）。
  - 送入 `models/matcha-icefall-zh-en/vocos-16khz-univ.onnx` 分析-重建后，送入本地 TRT-EdgeLLM ASR 评测原始输出：
    - `power=1.0`: 识别结果为“您好，我是您的智能语音助手。Welcome to C的Studio，今天有什么可以帮帮您的。”（中文 CER = 4.1%，主要是多一个重叠字“帮”，主干完全清晰可懂）。
    - `power=2.0`: 识别结果为“您好，我是这个语音助手，观看《去西格的电影》你咋？”（严重错乱，CER > 70%）。确证幅度谱模式（`power=1.0`）是唯一可用的声码器特征协议。
  - 截断底噪常数 $\epsilon$：在 Step 0 运行脚本中，对 $\epsilon \in \{10^{-10}, 10^{-5}\}$ 进行重建静音底噪与人声有效段对比测试，以不产生低电平截断断崖为判定准则锁定并写入公共模块。
- **公共模块化**:
  该提取逻辑固化为公共库 `tools/voice_pipeline/audio_features.py`，全管线统一定义 `extract_mel(wav_path)`。

### 3.3 G2P 前端与 Token 映射
- 词表：`models/matcha-icefall-zh-en/tokens.txt` (2,175 Tokens)
- 字典：`models/matcha-icefall-zh-en/lexicon.txt`
- 边界强制规则：中文音节与英文单词交界处（尤其是 `shi4 (1624)` 与英文辅音），强制插入空格 Token（ID: 1），杜绝连读吞音。

### 3.4 导出的 ONNX 模型规范
导出两个完全独立的规范 ONNX：
1. **`voices/minimax_shaonv/matcha_shaonv_step3.onnx`**：基于官方 `model-steps-3.onnx` 架构，时间网格 $t \in \{0.0, \frac{1}{3}, \frac{2}{3}\}$，单步步长 $dt = \frac{1}{3}$。
2. **`voices/minimax_shaonv/matcha_shaonv_step6.onnx`**：基于 6 步 Euler 解算，时间网格 $t \in \{0.0, \frac{1}{6}, \frac{2}{6}, \frac{3}{6}, \frac{4}{6}, \frac{5}{6}\}$，单步步长 $dt = \frac{1}{6}$。
- **部署输入签名**:
  - `x`: `int64[batch, num_tokens]`
  - `x_length`: `int64[batch]`
  - `noise_scale`: `float32[1]` (固定设为 `0.0` 实现零随机白噪确定性生成)
  - `length_scale`: `float32[1]`
- **输出签名**:
  - `mel`: `float32[batch, 80, num_frames]` (输出域为非归一化对数 Mel 域 $m$)

---

## 4. 实施方案与防过拟合策略 (Implementation Plan)

### 4.1 数据集隔离划分
- **训练集 (Train)**: 91 句 (`voices/minimax_shaonv/train.txt`)
- **保留验证集 (Val)**: 11 句 (`voices/minimax_shaonv/val.txt`)，用于监控验证集 Mel L1 损失，挑选最优 Checkpoint。
- **真正未见终验集 (Frozen Unseen Test)**: 5 句独立全新文本（未在数据集及任何既有脚本中出现）：
  1. `U1_Bilingual`: "欢迎体验智能语音新时代，Welcome to Seeed intelligent voice assistant."
  2. `U2_English`: "This neural voice engine runs entirely on local edge devices with ultra low latency."
  3. `U3_Tech`: "我们通过单调对齐搜索与流匹配技术实现了高保真音色复刻。"
  4. `U4_Number`: "当前系统采样率设定为一万六千赫兹，音频缓冲延迟仅有四十五毫秒。"
  5. `U5_Daily`: "今天下午有小到中雨，外出记得带伞。"

### 4.2 严谨的音素级 MAS 单调对齐 (MAS Mathematical Formulation)
1. 提取官方已冻结 TextEncoder 在时长展开前的音素先验 $\mu_x \in \mathbb{R}^{80 \times T_{text}}$（来自 `model.encoder.proj_m`）。
2. 计算目标音频标准化 Mel $x_1 \in \mathbb{R}^{80 \times T_{mel}}$ 与 $\mu_x$ 的负对数似然代价矩阵：
   $$D_{i, j} = \frac{1}{2} \| x_{1, j} - \mu_{x, i} \|_2^2$$
3. 在动态规划中排除 padding 填充帧与填充 token，求解最优单调对齐矩阵 $A \in \{0, 1\}^{T_{text} \times T_{mel}}$：
   - 边界约束：$A(0, 0) = 1$，$A(T_{text}-1, T_{mel}-1) = 1$；
   - 步进约束：每个 Mel 帧 $j$ 仅允许转移自相同音素 $i$ 或前一音素 $i-1$（$A_{i, j} = 1 \implies A_{i, j-1} = 1 \text{ or } A_{i-1, j-1} = 1$）；
   - 唯一性：每个有效 Mel 帧 $j$ 恰好分配至一个音素 $i$。
4. 获得对齐先验：$\mu_y = \mu_x \cdot A \in \mathbb{R}^{80 \times T_{mel}}$。
5. 异常保护与拒收门槛：
   - 检查单音素分配帧数：若任何音素持续时间 $> 120$ 帧（约 1.92 秒）或有音素被跳过（$= 0$ 帧），标记为异常样本；
   - 若异常样本比例 $> 5\%$，立即中断训练并排查 G2P 前端。
6. 阶段 2 对齐更新规则：
   - 若触发阶段 2 解冻 `proj_m`，每个 epoch 使用当前候选 Checkpoint 的最新 $\mu_{x, cand}$ 重新执行 MAS 对齐；
   - 梯度隔离：对齐矩阵 $A$ 显式执行梯度截断（`A = A.detach()`），而先验展开 $\mu_y = \mu_x \cdot A$ 保持对 `proj_m` 参数的梯度传播。

### 4.3 阶梯式微调与冻结的验证集 L1 评测协议
- **损失函数设计**:
  $$\mathcal{L} = \mathcal{L}_{CFM} + \lambda_{L2SP} \frac{1}{2} \sum_{p \in \Theta_{trainable}} \| p - p_{base} \|_2^2$$
  其中 $\mathcal{L}_{CFM}$ 按有效帧 Mask $M_{mel} \in \{0, 1\}^{B \times 1 \times T}$ 归约：
  $$\mathcal{L}_{CFM} = \frac{\sum (v_\theta(x_t, t, \mu_y) - (x_1 - z_0))^2 \cdot M_{mel}}{\sum M_{mel} \cdot 80}$$
  其中 $z_0 \sim \mathcal{N}(0, I)$，$t \sim \text{Uniform}(0, 1)$，$x_t = (1-t)z_0 + tx_1$。
  设置 $\lambda_{L2SP} = 0.01$，仅针对当前可训练参数集 $\Theta_{trainable}$ 施加约束。
- **阶段 1（Decoder 速度场适配，优先主选）**:
  - 冻结全部 Text Encoder 与 Duration Predictor；
  - 仅解冻 Decoder UNet 中的 Cross-Attention 与 ResNet 卷积残差块；
  - 优化器：AdamW，学习率 $5 \times 10^{-5}$，CosineAnnealingLR 至 $10^{-6}$，训练 25 epochs。
- **冻结的验证集 Mel L1 评测计算协议**:
  - 针对 11 句保留验证集，输入验证样本的音素序列与真实目标 Mel $x_{1, val} \in \mathbb{R}^{80 \times T_{val}}$；
  - **阶段 1 先验计算**: 使用冻结官方 TextEncoder 与 baseline MAS 获得的对齐先验 $\mu_{y, val}$；
  - **阶段 2 先验计算**: 冻结 baseline 对齐矩阵 $A_{val}$，使用候选 Checkpoint 解冻微调后的 `proj_m` 计算音素先验 $\mu_{x, cand}$，展开得到候选对齐先验 $\mu_{y, val} = \mu_{x, cand} \cdot A_{val}$，确保客观评价完整模型的输出能力；
  - 固定使用 3-step Euler 解算（$t \in \{0, 1/3, 2/3\}$），起始状态设为确定性 $z_0 = 0$（`noise_scale = 0.0`），由微调后的 Decoder 生成重建 Mel $\hat{z}_{val} \in \mathbb{R}^{80 \times T_{val}}$；
  - 在标准化 Mel 域 $z$ 下，按有效帧计算平均绝对误差：
    $$\text{Val\_L1} = \frac{1}{11} \sum_{k=1}^{11} \frac{\sum_{t=1}^{T_k} \sum_{c=1}^{80} |\hat{z}_{k, c, t} - x_{1, k, c, t}|}{80 \times T_k}$$
  - **跨阶段最终最佳 Checkpoint 选优规则**: 严格比较阶段 1 与阶段 2 全部轮次，以最终 $\text{Val\_L1}$ 最低的那一轮 Checkpoint 作为全流程最终胜出模型。
- **阶段 2（先验微调）可量化触发条件与 F0 聚合协议**:
  - 仅在阶段 1 训练全部完成后进行评判；
  - 采用阶段 1 最佳 Checkpoint，在 11 句验证集上使用自由生成的 Duration Predictor 进行端到端推理（`noise_scale=0.0, length_scale=1.0`），通过 Vocos 解码为 16000 Hz 音频；
  - 对 11 条音频显式传入采样率调用 `librosa.pyin(y, sr=16000, fmin=65, fmax=800)`，提取有声帧（概率 $>0.5$）。若某音频未检测到任何有声帧，判为评测失败；
  - 计算全部 11 句有效有声帧的总体基频中位数 $F0_{val\_median}$；
  - **触发条件**: 当且仅当 $F0_{val\_median} < 330\text{ Hz}$ 且验证集 $\text{Val\_L1}$ 连续 3 轮下降幅度 $< 0.001$ 时，才触发阶段 2（解冻 `model.encoder.proj_m`，以极保守学习率 $1 \times 10^{-5}$ 联合微调 10 epochs）。

### 4.4 PyTorch 与 ONNX 数值等价性强制门禁
在导出 ONNX 之后、运行任何声学评测之前，必须执行自动化数值一致性断言：
1. **权重映射完整性检查**:
   - 验证脚本逐项核对 PyTorch `state_dict` 中的可训练参数（Decoder 各层及 Phase 2 的 `proj_m`），确保 100% 写入目标 ONNX 的 Initializer 节点。
2. **多步数与时间网格构建**:
   - 3-Step 模型：时间网格固定为 $t \in \{0.0, \frac{1}{3}, \frac{2}{3}\}$，单步步长 $dt = \frac{1}{3}$；
   - 6-Step 模型：时间网格固定为 $t \in \{0.0, \frac{1}{6}, \frac{2}{6}, \frac{3}{6}, \frac{4}{6}, \frac{5}{6}\}$，单步步长 $dt = \frac{1}{6}$。
3. **两套数值等价门禁**:
   - **门禁 A（确定性生产模型门禁）**:
     - 分别覆盖 3-Step 与 6-Step 生产导出的 4 输入 ONNX 模型 (`[x, x_length, noise_scale, length_scale]`)；
     - 设置 `noise_scale = 0.0`（内部 `RandomNormalLike` 乘零，起始状态确定性为 0）；
     - 覆盖短文本（10 tokens）、中长文本（35 tokens）与长文本（80 tokens）；
     - PyTorch 端独立使用 `IntermediateEncoder` 与候选 `proj_m` 计算音素先验 $\mu_x$ 并断言与 ONNX 投影层输出误差 $< 10^{-5}$；
     - 断言 PyTorch 原生 Euler 循环与 ONNX ORT 会话输出的最大绝对误差：
       $$\max |mel_{\text{PyTorch}} - mel_{\text{ONNX}}| < 10^{-4}$$
   - **门禁 B（同噪声诊断测试门禁）**:
     - 分别为 3-Step 与 6-Step 导出同图诊断模型（仅将内部 `RandomNormalLike` 替换为外部张量输入 `noise: float32[1, 80, num_frames]`，其余 Initializer 权重与计算节点与生产模型保持 100% SHA256 等价）；
     - **显式设置 `noise_scale = 1.0`**，使外部噪声通路完全激活；
     - 向 PyTorch 和诊断 ONNX 注入完全相同的随机高斯噪声张量 $z_0 \sim \mathcal{N}(0, I)$；
     - 同样断言 3-Step 与 6-Step 的最大绝对误差均满足 $\max |mel_{\text{PyTorch}} - mel_{\text{ONNX}}| < 10^{-4}$；
     - 将门禁测试逐例最大误差、平均误差、Checkpoint 与 ONNX 的 SHA256 哈希值持久化为 `voices/minimax_shaonv/numerical_parity_report.json`。
   - **安全检查**: 检查输出张量形状、有效长度，严格杜绝 NaN 或 Inf。

---

## 5. 版本化多维验收门槛 (Acceptance Criteria)

### 5.1 硬性安全门禁 (Hard Gates)
1. **静音段纯净度门禁**:
   - 提取 VAD 判定的全部非人声静音帧（包括句首、句末及句中自然停顿）；
   - 静音段有效时长需 $\ge 0.05$ 秒。若样本未检测到有效静音区间，该项标记为 `N/A: False`，严禁假通过；
   - 静音段 RMS 必须 $\le 0.001$（低于 $-60\text{ dBFS}$），彻底杜绝底噪沙沙声。
2. **防爆音门禁**:
   - 原始声码器输出 `audio_raw` 与最终交付音频 `audio_delivery`，二者的绝对峰值均必须 $\le 0.95$（杜绝数字削顶爆音）。
3. **数值等价门禁**:
   - 3-Step 与 6-Step 模型在门禁 A 与门禁 B 下，最大绝对误差均 $< 10^{-4}$。
4. **未见集 ASR 强制闸门**:
   - 3-Step 与 6-Step 模型针对 5 句未见测试集（U1~U5）共 10 项测试结果，必须全部满足中文 CER $\le 5\%$ 且汇总英文 WER $\le 10\%$；若有任何一项失败，直接判定未通过，严禁远端推送。
5. **底座模型只读安全门禁**:
   - 严格记录官方底座文件（`models/matcha-icefall-zh-en/pytorch_model.bin` 等）在任务前后的 SHA256 哈希，核验一致，证明只读未被篡改。

### 5.2 客观参考诊断指标 (Diagnostic Metrics)
以下指标用于辅助分布校准与异常监控（以 MiniMax 目标录音分布为基线）：
1. **零交叉率 (ZCR) 均值分布**:
   - 语音段 ZCR 统计分布均值落在 $[0.11, 0.17]$（目标参考录音实测均值约为 $0.138$），监控是否存在异常高频尖叫或模糊。
2. **基频分布 (F0)**:
   - 使用 `librosa.pyin(y, sr=16000, fmin=65, fmax=800)` 在有声帧（概率 $> 0.5$）统计基频分布；
   - 目标 F0 中位数参考区间为 $[330\text{ Hz}, 365\text{ Hz}]$（目标参考录音实测中位数 $345\text{ Hz}$）。若无有效有声帧，判为异常。

### 5.3 真正未见终验集 (Frozen Unseen Test) 本机 ASR 强制闸门
调用本机 TRT-EdgeLLM ASR (`http://127.0.0.1:8621/asr`) 严格测试 5 句全新文本，对 3-Step 与 6-Step ONNX 模型分别独立评测：
- **U1 (中英混合)**: "欢迎体验智能语音新时代，Welcome to Seeed intelligent voice assistant."
- **U2 (纯英文)**: "This neural voice engine runs entirely on local edge devices with ultra low latency."
- **U3 (算法专有名词)**: "我们通过单调对齐搜索与流匹配技术实现了高保真音色复刻。"
- **U4 (数字与量词)**: "当前系统采样率设定为一万六千赫兹，音频缓冲延迟仅有四十五毫秒。"
- **U5 (日常短句)**: "今天下午有小到中雨，外出记得带伞。"
- **文本归一化与评分规则**:
  - 预处理：剔除标点符号（`，。！？,.!?`），英文转小写；
  - 数字归一化：如 U4 "一万六千赫兹"，若 ASR 输出为阿拉伯数字 "16000赫兹" 或 "16000 Hz"，统一使用数字归一化工具将两者对齐为标准汉字发音后再计算 CER；
  - 混合语言计分：按中英文语言块拆分，中文部分计算字错误率 $\text{CER}$，英文部分计算词错误率 $\text{WER}$；
  - **英文 WER 评分与品牌词判定契约**:
    - **严格逐词评分 (Strict WER)**：不进行任何专有名词替换，逐词按编辑距离计算，原样记录并报告；
    - **专有名词诊断评分 (Tolerant WER)**：针对专有非标品牌词（如 "Seeed" 在通用 ASR 语言模型中缺乏条目，原声常转录为音似词 "see" 或 "sea"），单独列出音似诊断评分，不掩盖严格原始评分；
    - **门禁聚合判定**:
      - 逐句要求：无吞字、无重复词乱码、无未完成截断；
      - 中文逐句 $\text{CER} \le 5\%$；
      - 英文整体门禁：未见集全量汇总英文词错误率（Corpus-level Strict WER）$\le 10\%$（5 句测试集英文字词总数 19 词，允许至多 1 处专有名词音似，整体 WER $\le 5.3\%$）。
      - 两套模型（3-Step 与 6-Step）全部 10 项测试结果统一聚合，全部满足方可通过总门禁。若任一模型任一用例不满足门禁，严禁执行远端推送交付。

### 5.4 MacBook 试听对比交付
将 5 句未见集测试音频推送至 MacBook `~/Downloads/`：
- 提供与底座原声 `clean_base.wav` 及 MiniMax 目标参考录音的三方盲听比对。

---

## 6. 安全护栏与禁区 (Guardrails)

- **严禁事项**:
  - 严禁在导出的 Mel 谱后增加破坏时序局部性的后处理卷积网络或手工非线性乘幂变换；
  - 原始底座 `models/matcha-icefall-zh-en/pytorch_model.bin` 保持严格只读备份，微调产物独立保存为 `voices/minimax_shaonv/checkpoints/`；
  - 任何改动必须通过 5 句未见集 ASR 门禁后方可向用户交付；
  - 单点修复预算上限 3 轮，若 3 轮未达标立即暂停并回溯分析。
