#!/bin/bash
# ==============================================================================
# tools/voice_pipeline/remote_colab_train.sh
# 一键驱动 Colab 租机、推送数据、GPU 训练与导出、回传 ONNX 并自动释放实例
# ==============================================================================
set -euo pipefail

SESSION_NAME="${1:-matcha-voice-train}"
DATASET_TAR="${2:-voices/zgl_matcha_pipeline/zgl_dataset.tar.gz}"
OUTPUT_DIR="${3:-voices/zgl_matcha_pipeline}"
GPU_TYPE="${4:-T4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/colab_runner/train_and_export.py"

[[ -f "$DATASET_TAR" ]] || { echo "Error: Dataset $DATASET_TAR not found!"; exit 1; }
[[ -f "$TRAIN_SCRIPT" ]] || { echo "Error: Train script $TRAIN_SCRIPT not found!"; exit 1; }

echo "=== [1/6] Starting Colab $GPU_TYPE Instance: $SESSION_NAME ==="
colab-fleet up "$SESSION_NAME" --gpu "$GPU_TYPE" --keep-existing

trap 'echo "[-] Cleaning up and stopping Colab instance..."; colab-fleet down "$SESSION_NAME" || true' EXIT

SSH_KEY="/home/harvest/.ssh/id_ed25519_fleet"
SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ProxyCommand=\"colab ssh --proxy-mode -s $SESSION_NAME -i $SSH_KEY\" -i $SSH_KEY root@colab"

echo "=== [2/6] Transferring Dataset and Script to Colab ==="
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ProxyCommand="colab ssh --proxy-mode -s $SESSION_NAME -i $SSH_KEY" \
    -i "$SSH_KEY" \
    "$DATASET_TAR" "$TRAIN_SCRIPT" root@colab:/content/

echo "=== [3/6] Extracting Dataset on Colab ==="
$SSH_CMD "mkdir -p /content/dataset && tar -xzf /content/$(basename "$DATASET_TAR") -C /content/dataset"

echo "=== [4/6] Installing Requirements on Colab ==="
$SSH_CMD "pip install -q onnx onnxscript"

echo "=== [5/6] Executing Flow Matching Fine-Tuning & ONNX Export ==="
$SSH_CMD "python3 /content/$(basename "$TRAIN_SCRIPT")"

echo "=== [6/6] Fetching Exported model-steps-3.onnx to Local ==="
mkdir -p "$OUTPUT_DIR"
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ProxyCommand="colab ssh --proxy-mode -s $SESSION_NAME -i $SSH_KEY" \
    -i "$SSH_KEY" \
    root@colab:/content/output/model-steps-3.onnx "$OUTPUT_DIR/model-steps-3.onnx"

echo "=== Done! ONNX model saved to: $OUTPUT_DIR/model-steps-3.onnx ==="
