# Matcha-TTS 专属音色定制与端侧微调流水线 (Voice Pipeline)

本工具集实现了轻量流匹配声学模型（**Matcha-TTS**）的端到端自动化微调与部署，专为 **Jetson Orin** 与 **Rockchip RK3588** 等边缘硬件打造。

通过“**教师大模型语料扩增 + 自动清洗质检 + Colab GPU 轻量微调 + ONNX 导出**”闭环，以极低成本（**约 2 ~ 3 元人民币**、**1.5 ~ 2 小时**）从极短录音（10~30 秒）或自然语言 Prompt 制作出高质量的边缘端专属发音人。

---

## 资产目录与职责

| 文件/目录 | 说明 |
| :--- | :--- |
| `config.template.yaml` | 通用配置模板，解耦 API 凭据、服务端点、语料配置与微调超参 |
| `pipeline.py` | 统一主控 CLI 驱动，按阶段或一键调度全流程 |
| `batch_synthesize.py` | 教师大模型并发生成器（独立进程防死锁，支持断点续传） |
| `prepare_dataset.py` | 数据工程与质检脚本（ASR/VAD 过滤、音素标注、训练/验证集打包） |
| `remote_colab_train.sh` | 算力调度脚本（一键启动 Colab T4、安全传输、远程训练、导出 ONNX、回收实例） |
| `colab_runner/train_and_export.py` | 云端 GPU 微调与导出脚本（OT-CFM 优化微调，导出标准 3-step ONNX） |
| `colab_runner/run_colab_finetune.sh` | 云端容器内环境初始化与微调启动封装 |

---

## 快速上手

### 1. 配置参数
复制配置模板并填入您的音色描述或参考录音：
```bash
cp tools/voice_pipeline/config.template.yaml my_voice.yaml
```
根据需求配置 `my_voice.yaml` 中的 `teacher`（如阿里百炼或 MiniMax）、`corpus` 和 `training` 参数。

### 2. 执行流水线

使用统一驱动脚本调度：

```bash
# 阶段 1: 教师大模型批量生成语料 (50~300 句)
python3 tools/voice_pipeline/pipeline.py synthesize --config my_voice.yaml

# 阶段 2: 数据质检、VAD 截断与打包
python3 tools/voice_pipeline/pipeline.py prepare --config my_voice.yaml

# 阶段 3: 一键启动 Colab GPU 训练并拉回 model-steps-3.onnx
python3 tools/voice_pipeline/pipeline.py train --config my_voice.yaml

# 或一键全自动执行:
python3 tools/voice_pipeline/pipeline.py all --config my_voice.yaml
```

---

## 边缘端编译与交付

训练完成后导出的 `model-steps-3.onnx`（约 1.86 MB）可直接转换为目标边缘硬件的高性能引擎：

### Jetson Orin (TensorRT)
```bash
trtexec --onnx=model-steps-3.onnx \
        --saveEngine=matcha_estimator_step0_bf16.engine \
        --fp16 --bf16
```
实测端侧首包推理延迟：**20 ~ 50 ms**，RTF 0.02 ~ 0.07。

### Rockchip RK3588 (RKNN)
通过 `rknn-toolkit2` 量化转换为 `matcha-s64.rknn`，交由 NPU 硬件并发加速。

---

## 详细指南
更多原理、架构图解与生产级实测数据，请参阅完整技术文档：
[docs/guides/voice_cloning_and_adaptation.md](../../docs/guides/voice_cloning_and_adaptation.md)
