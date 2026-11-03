# Matcha-TTS 专属音色定制与端侧微调流水线 (Voice Pipeline)

本工具集实现了轻量流匹配声学模型（**Matcha-TTS**）的端到端自动化微调与部署，专为 **Rockchip RK3588/RK3576** 与 **NVIDIA Jetson Orin** 等边缘硬件打造。

针对传统开源轻量模型念英语带有“中式口音/Chinglish”的痛点，通过“**通义大模型中英平衡语料扩增 + 自动清洗质检 + Colab/本地 GPU 轻量 Flow-Matching 微调 + 3 步解算 ONNX 导出**”闭环，以极低成本（**约 2 ~ 3 元人民币**、**1.5 ~ 2 小时**）从自然语言 Prompt 或预置亲和力音色快速制作出高质量的边缘端专属发音人。

---

## 资产目录与职责

| 文件/目录 | 说明 |
| :--- | :--- |
| `config.template.yaml` | 通用配置模板，解耦 API 凭据、服务端点、语料配置与微调超参 |
| `pipeline.py` | 统一主控 CLI 驱动，按阶段分步执行或一键调度全流程 |
| `batch_synthesize.py` | 教师大模型并发生成器（独立进程防死锁，支持断点续传） |
| `corpus/texts_bilingual_balanced.txt` | 内置 100 句高质量中英平衡与日常交互语料库 |
| `prepare_dataset.py` | 数据工程与质检脚本（时长过滤、采样率校验、训练/验证集打包） |
| `remote_colab_train.sh` | 算力调度脚本（一键启动 Colab T4、安全传输、远程训练、导出 ONNX、回收实例） |
| `colab_runner/train_and_export.py` | 云端/本地 GPU 微调与导出脚本（OT-CFM 优化微调，导出标准 3-step ONNX） |

---

## 快速上手

### 1. 配置参数
复制配置模板并选择发音人（如亲和力知性女声 `longxiaochun_v2`）：
```bash
cp tools/voice_pipeline/config.template.yaml my_voice.yaml
```
根据需求修改 `my_voice.yaml` 中的 `voice_name`、`voice`（如 `longxiaochun_v2` 或 `longwan_v2`）、`api_key_env`（默认为 `Aliyun_General`）。

### 2. 执行流水线

使用统一驱动脚本调度：

```bash
# 阶段 1: 教师大模型批量生成语料 (50~100 句高质量中英混读)
secret-run --env Aliyun_General -- python3 tools/voice_pipeline/pipeline.py synthesize --config my_voice.yaml

# 阶段 2: 数据质检、时长校验与打包 (生成 dataset.tar.gz)
python3 tools/voice_pipeline/pipeline.py prepare --config my_voice.yaml

# 阶段 3: 一键启动 Colab GPU 训练并拉回生产级 model-steps-3.onnx
python3 tools/voice_pipeline/pipeline.py train --config my_voice.yaml

# 或一键全自动执行:
secret-run --env Aliyun_General -- python3 tools/voice_pipeline/pipeline.py all --config my_voice.yaml
```

---

## 边缘端编译与交付

训练完成后导出的 `model-steps-3.onnx` 可直接转换为目标边缘硬件的高性能引擎：

### Rockchip RK3588 / RK3576 (RKNN)
通过 `rknn-toolkit2` 将导出模型转换为 `matcha-s64.rknn`，交由 NPU 硬件并发加速，首包延迟仅需约 **30~50 ms**。

### Jetson Orin (TensorRT)
通过工程内的 `split_matcha_trt.py` 拆分并编译为 TensorRT Engine：
```bash
python3 third_party/jetson-voice-engine/models/matcha/split_matcha_trt.py \
        --onnx=model-steps-3.onnx --output-dir=engines/
```

---

## 详细指南
更多原理、架构图解与生产级实测数据，请参阅：
[docs/guides/voice_cloning_and_adaptation.md](../../docs/guides/voice_cloning_and_adaptation.md)
