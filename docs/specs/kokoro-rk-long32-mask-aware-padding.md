# Kokoro RK Long32 Mask-Aware Padding 设计

状态：**设计稿，尚未实施**  
目标平台：RK3588，后续兼容 RK3576  
目标模型：Kokoro 1.0 多语言  
性能门槛：完整端到端 RTF ≤ 0.30  
质量约束：不得降低现有九语言能力，不得用 legacy fallback 掩盖失败

## 1. 背景

当前 long32 路径使用静态 RKNN 时长桶。RK3588 canary 配置为：

```text
KOKORO_LONG32_ROUTE_TS=400,640
KOKORO_LONG32_MAX_SNAP_RATIO=0.10
```

桶的目标音频时长及在 10% duration snap 下的自然时长覆盖范围为：

| 桶 | 目标时长 | 自然时长覆盖范围 |
|---|---:|---:|
| T400 | 5 秒 | 4.55–5.56 秒 |
| T640 | 8 秒 | 7.27–8.89 秒 |

因此 5.56–7.27 秒存在连续覆盖空洞。旧行为可能进入 legacy hybrid fallback；fail-closed canary 会直接拒绝该请求。

增加 T480/T560 可以解决覆盖问题，但会增加模型文件、NPU context、预加载时间和内存占用。本设计研究在保留 T400/T640 两个静态桶的情况下，通过 mask-aware right padding 连续覆盖约 0–8 秒有效时长。

## 2. 当前 long32 结构

RK3588 long32 generator 当前拆为 32 个逻辑 RKNN context：

| 部分 | 数量 |
|---|---:|
| main0 | 1 |
| noise0 | 1 |
| rb0/rb1/rb2 | 3 |
| main1 | 1 |
| rb3/rb4/rb5，每个 6 段 | 18 |
| noise1-preconv + 6 个 noise1 conv | 7 |
| post | 1 |
| 合计 | **32** |

主要张量分辨率：

```text
T
L = 10 × T
F = 60 × T + 1
samples = 5 × (F - 1)
duration_seconds = T / 80
```

当前 RKNN 输入没有 `valid_length` 或 padding mask。普通 zero padding 会参与时间轴 mean/variance、AdaIN、卷积和噪声分支计算，因此不是语义无害操作。

## 3. 设计目标

1. 保留 RK3588 的 T400/T640 两个静态桶。
2. 不再把自然 duration 比例缩放到桶中心。
3. 使用最小可容纳桶，在尾部进行 mask-aware padding。
4. 所有时间轴统计忽略 padding 区。
5. 每个受影响的卷积/残差阶段在输入和输出处屏蔽 padding 区。
6. iSTFT 前裁剪到有效频谱长度。
7. 九语言均命中 `kokoro_long32`，不允许静默 fallback。
8. 完整 E2E RTF ≤ 0.30。

## 4. 新路由规则

保留自然 duration，不做 snap：

```text
natural_D
  ↓
valid_T = 2 × natural_D
  ↓
选择最小可容纳桶
```

建议规则：

```text
valid_T ≤ 400       → T400
400 < valid_T ≤ 640 → T640
valid_T > 640       → 按预测时长切句
```

示例：自然音频约 6 秒时，`valid_T≈480`，选择 T640；有效区保持 T480 的自然时长，尾部 T160 作为 masked padding，最终只输出约 6 秒音频。

不建议只保留 T640。较短句子仍会执行完整 T640 NPU 计算，增加首包延迟并恶化短句 RTF。T400/T640 是模型数量与计算量之间的折中。

## 5. Mask ABI

CPU 根据 `valid_T` 生成固定 bucket shape、动态内容的三种 mask：

```text
mask_T: [1, 1, T_bucket]
mask_L: [1, 1, 10 × T_bucket]
mask_F: [1, 1, 60 × T_bucket + 1]
```

有效长度：

```text
valid_T = 2 × natural_D
valid_L = 10 × valid_T
valid_F = 60 × valid_T + 1
valid_samples = 5 × (valid_F - 1)
```

Mask 内容：

```python
mask_T[..., :valid_T] = 1
mask_T[..., valid_T:] = 0

mask_L[..., :valid_L] = 1
mask_L[..., valid_L:] = 0

mask_F[..., :valid_F] = 1
mask_F[..., valid_F:] = 0
```

Mask 必须由 CPU 生成。不要让 RKNN 图使用动态 `Range`、动态 shape 或比较算子生成 mask，以降低 RKNN Toolkit 转换风险。

## 6. Frontend 修改

新 frontend 流程：

1. G2P 生成 phoneme IDs。
2. `duration_probe.onnx` 在 CPU 上生成自然 duration。
3. 保留自然 duration，不进行比例 snap。
4. `target_frontend.onnx` 按 `natural_D` 生成有效 tensor。
5. CPU 将有效 tensor padding 到选择的静态桶。

```text
x_valid   → right padding → [1, 512, T_bucket]
har_valid → right padding → [1, 22, 60T_bucket+1]
s         → unchanged     → [1, 128]
```

如果现有 `target_frontend.onnx` 不能输出自然动态长度，需要重新导出动态 CPU ONNX。该模型继续由 ONNX Runtime CPU 执行，不需要转成 RKNN。

## 7. Masked Normalization

所有沿时间轴的 mean/variance 必须只统计有效区域：

```python
denom = maximum(sum(mask, axis=time, keepdims=True), 1)
mean = sum(x * mask, axis=time, keepdims=True) / denom
var = sum(((x - mean) ** 2) * mask, axis=time, keepdims=True) / denom
x = (x - mean) / sqrt(var + eps)
x = x * mask
```

禁止继续使用普通 `x.mean(axis=time)` 后再清零。普通统计已经被 padding 区污染，会改变整个有效区域的归一化结果。

## 8. 卷积、AdaIN 与残差规则

每个受影响阶段遵循：

```text
masked input
  ↓
masked normalization
  ↓
AdaIN / activation
  ↓
convolution
  ↓
masked output
```

残差相加后再次应用 mask：

```python
out = (residual + branch) * mask
```

只在 32 个大子图之间清零可能不够。卷积 bias 会令 padding 区变为非零，后续卷积可能把该值传播回有效边界。需要检查每个 RKNN 子图内部的连续卷积，在必要的层间写入 mask multiplication。

## 9. 32 个子图的 Mask 分配

| 模块 | Mask |
|---|---|
| main0 输入 | mask_T |
| main0 输出 | mask_L |
| noise0 的 har 输入 | mask_F |
| noise0 输出 | mask_L |
| rb0/rb1/rb2 | mask_L |
| main1 输入 | mask_L |
| main1 输出 | mask_F |
| noise1 全部阶段 | mask_F |
| rb3/rb4/rb5 全部阶段 | mask_F |
| post | mask_F |

受影响子图增加对应 mask 输入：

```text
旧 ABI: x, style
新 ABI: x, style, valid_mask
```

如果某个子图没有时间轴统计且只有单层卷积，可评估在 CPU 子图边界清零；连续多层卷积的子图优先在 ONNX 图内部应用 mask。

## 10. iSTFT 与最终裁剪

Generator 固定输出：

```text
conv_post: [1, 22, 60T_bucket+1]
```

必须在 iSTFT 前裁剪：

```python
conv_valid = conv_post[..., :valid_F]
audio = istft(conv_valid)
assert audio.shape[-1] == valid_samples
```

不要先生成完整桶长度的波形再裁剪。iSTFT 前裁剪可减少 CPU 工作量，并避免 overlap-add 接触 padding 区。

## 11. RKNN 改动范围

不需要修改：

- Rockchip NPU 驱动；
- `librknnrt.so`；
- RKNNLite API；
- RK3588 固件。

需要修改：

- generator PyTorch/ONNX exporter；
- 受影响子图输入 ABI；
- mask-aware normalization；
- RKNN 转换脚本；
- manifest schema；
- runtime 输入绑定；
- device smoke 与性能测试。

转换链保持不变：

```text
mask-aware ONNX
  ↓ RKNN Toolkit 2.3.2
T400 / T640 RKNN
  ↓
RK3588 真机验证
```

需要先用小型 probe 验证 RKNN Toolkit 对固定轴 `ReduceSum`、broadcast multiply 和 mask input 的支持。若不支持，优先将 masked normalization 保留在 CPU，不修改 RKNN runtime。

## 12. Manifest Schema 建议

Bundle manifest 新增：

```json
{
  "padding_mode": "mask-aware-right-padding",
  "mask_schema": "kokoro-long32-mask-1",
  "valid_length_input": true,
  "mask_resolutions": ["T", "L", "F"],
  "normalization": "masked-time-axis",
  "crop_before_istft": true
}
```

每个子图记录 mask contract：

```json
{
  "stage": "rb3-convs1_0",
  "mask": "F",
  "mask_input_shape": [1, 1, 38401],
  "masks_input": true,
  "masks_output": true
}
```

Runtime 必须拒绝旧 manifest 与新 mask runtime 混用。

## 13. 开发阶段

### 阶段 A：CPU Reference

1. 在 CPU reference generator 实现 mask-aware normalization。
2. 支持自然 duration 和 right padding。
3. 建立原生短 shape 与大桶 padding 的对照输出。
4. 不修改 RKNN，不部署设备。

### 阶段 B：ONNX ABI

1. 修改 exporter，加入 T/L/F mask 输入。
2. 保持现有 32 子图边界，除非证据表明必须进一步拆分。
3. 使用 ONNX Runtime 做 shape、finite、mask 泄漏测试。
4. 验证 padding 区每个阶段输出严格为零或在规定容差内。

### 阶段 C：单桶 RKNN Probe

1. 只转换一个实验 T640 bundle。
2. 在 RK3588 对比原生参考和 T640 mask 输出。
3. 验证 RKNN Toolkit 算子支持、NPU load/init/inference/release。
4. 质量不成立时停止，不批量转换。

### 阶段 D：正式双桶

1. 转换正式 T400/T640。
2. 更新 bundle manifest 和 device smoke。
3. 接入 long32 runtime 路由。
4. 保持 `KOKORO_LONG32_FALLBACK=error`。

### 阶段 E：真机验收

1. 九语言质量和 ASR。
2. 完整 E2E RTF。
3. 并发、取消、连续请求稳定性。
4. 只在 canary 全部通过后考虑生产切换。

## 14. 验收矩阵

至少覆盖以下自然时长：

```text
4.0s, 4.8s, 5.0s, 5.2s, 5.8s,
6.5s, 7.2s, 7.8s, 8.0s
```

其中 5.2–7.2 秒是当前双桶方案的主要覆盖空洞。

每个时长覆盖九条语言路由：

```text
a: en-US
b: en-GB
e: es
f: fr
h: hi
i: it
j: ja
p: pt-BR
z: zh
```

验收项目：

- 原生 reference 与 padding 输出的有效区域波形差异；
- log-mel 差异；
- SI-SDR；
- 尾部 RMS 和 click/pop 检测；
- ASR WER/CER；
- 说话人音色相似度；
- generator RTF；
- 完整 E2E RTF；
- NPU 内存和 context load 时间；
- cancel 后下一请求是否正常；
- 连续请求是否发生 context 污染。

必须满足：

1. 九语言全部命中 `kokoro_long32`。
2. 无 legacy fallback。
3. 完整 E2E RTF ≤ 0.30。
4. ASR WER/CER 不劣于对应原生 reference。
5. 无可听尾音截断、爆音或异常静音。
6. 取消请求不触发性能门禁，且不污染下一请求。

## 15. 后续 Agent 开始前必须确认

后续实现 agent 不应直接批量转换全部模型。开始前必须：

1. 读取本文件及当前 long32 runtime/exporter 源码。
2. 记录当前 parent、rkvoice submodule 和模型 bundle SHA。
3. 保存现有 RTF < 0.3 成功样本作为性能基线。
4. 保存九语言原生 reference 音频和 ASR 结果。
5. 先完成 CPU reference A/B，再决定是否改 RKNN ABI。
6. 所有失败均 fail-closed，不得通过 fallback 生成“成功”样本。

## 16. 非目标

- 本设计不要求修改 Rockchip 驱动或 RKNN runtime。
- 本设计不把全部文本 frontend 或 iSTFT 移到 NPU。
- 本设计不以提高 snap ratio 代替 mask。
- 本设计不以 legacy hybrid fallback 作为时长覆盖方案。
- 本设计当前不授权生产部署或 8621 切换。

## 17. 预期结果

完成后，T400/T640 两个静态桶应连续覆盖最长约 8 秒的有效句段：

```text
自然 duration
  → 最小可容纳桶
  → mask-aware padding
  → RKNN generator
  → iSTFT 前裁剪
  → 自然时长音频
```

最终效果应是不改变自然音素 duration、不依赖慢 fallback，并保持九语言能力与完整 E2E RTF ≤ 0.30。
