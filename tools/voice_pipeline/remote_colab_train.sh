#!/bin/bash
# ==============================================================================
# tools/voice_pipeline/remote_colab_train.sh
# 基于 colab-fleet 官方标准流水线：一键租机、推送数据、GPU 训练与导出、拉回产物
# ==============================================================================
set -euo pipefail

SESSION_NAME="${1:-matcha-voice-train}"
DATASET_TAR="${2:-}"
OUTPUT_DIR="${3:-}"
GPU_TYPE="${4:-T4}"
EPOCHS="${5:-20}"

if [[ -z "$DATASET_TAR" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 <session_name> <dataset_tar> <output_dir> [gpu_type] [epochs]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/colab_runner/train_and_export.py"
FLEET="/home/harvest/.rpty/bin/fleet"

[[ -f "$DATASET_TAR" ]] || { echo "Error: Dataset $DATASET_TAR not found!"; exit 1; }
[[ -f "$TRAIN_SCRIPT" ]] || { echo "Error: Train script $TRAIN_SCRIPT not found!"; exit 1; }

echo "=== [1/6] Starting Colab $GPU_TYPE Instance via colab-fleet: $SESSION_NAME ==="
colab-fleet up "$SESSION_NAME" --gpu "$GPU_TYPE" --keep-existing

export SSH_AUTH_SOCK="$(colab-fleet sock)"
DEVICE_NAME="colab-$SESSION_NAME"

trap 'echo "[-] Cleaning up and stopping Colab instance..."; colab-fleet down "$SESSION_NAME" || true' EXIT

fleet_run() {
    local attempts=3
    while [[ $attempts -gt 0 ]]; do
        if $FLEET "$@"; then
            return 0
        fi
        echo "[-] Fleet command hit transient failure, warming bridge and retrying..."
        colab-fleet warm "$SESSION_NAME" >/dev/null 2>&1 || true
        sleep 4
        attempts=$((attempts - 1))
    done
    echo "Error: Fleet command failed after 3 attempts: $*" >&2
    return 1
}

echo "=== [2/6] Transferring Dataset to Colab via Fleet Push ==="
fleet_run push "$DEVICE_NAME" "$DATASET_TAR" /content/dataset.tar.gz

echo "=== [3/6] Extracting Dataset & Installing Dependencies in Colab ==="
fleet_run exec "$DEVICE_NAME" -- "rm -rf /content/dataset && mkdir -p /content/dataset && tar -xzf /content/dataset.tar.gz -C /content/dataset && pip install -q onnx onnxscript torchaudio soundfile pyyaml"

echo "=== [4/6] Launching GPU Training via colab-fleet train ==="
# 传入本地训练脚本路径，colab-fleet 会自动推送到 /content 并通过 GPU Kernel 容器启动
colab-fleet train "$SESSION_NAME" "$TRAIN_SCRIPT" -- --epochs "$EPOCHS" --dataset /content/dataset --output-dir /content/output

echo "=== [5/6] Following GPU Training Progress ==="
waited=0
max_wait=1800 # 30 分钟超时

while [[ $waited -lt $max_wait ]]; do
    sleep 6
    waited=$((waited + 6))

    # 检查导出的 ONNX 是否已生成
    if $FLEET exec "$DEVICE_NAME" -- "test -f /content/output/model-steps-3.onnx" 2>/dev/null; then
        echo "[+] Detected production ONNX model in Colab: /content/output/model-steps-3.onnx"
        break
    fi

    # 打印最新训练日志
    if [[ $((waited % 18)) -eq 0 ]]; then
        echo "--- [Colab GPU Training Log (${waited}s)] ---"
        $FLEET exec "$DEVICE_NAME" -- "tail -n 6 /content/train_and_export.py.log 2>/dev/null || true" 2>/dev/null || true
    fi
done

echo "=== [6/6] Pulling Exported model-steps-3.onnx to Local ==="
mkdir -p "$OUTPUT_DIR"
fleet_run pull "$DEVICE_NAME" /content/output/model-steps-3.onnx "$OUTPUT_DIR/model-steps-3.onnx"

echo "=== [✓] Success! Production ONNX model saved to: $OUTPUT_DIR/model-steps-3.onnx ==="
