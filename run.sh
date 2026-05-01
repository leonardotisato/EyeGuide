#!/bin/bash
# Unified launcher for all training/QAT/export scripts via Docker.
# Usage: bash run.sh <command> [args...]
#
# Commands:
#   train_teacher           Historical ResNet50 KD teacher stack (main.py) — produces resnet50_fp32_kd.pth
#   train_test_resnet       Canonical FP32 baseline for test_resnet (upgraded R18 teacher, sound KD pipeline)
#   train_resnet18_from_resnet50_kd  Distill resnet50_fp32_kd.pth into the upgraded full-image ResNet18 teacher
#   eval_teacher_224        Validate ResNet18 teacher at 224x224
#   eval_pipeline_checkpoint Evaluate any saved checkpoint in the canonical teacher->QAT ladder
#   qat_test_resnet         Canonical KD-QAT for test_resnet from the canonical FP32 baseline
#   export_test_resnet      QONNX export for test_resnet
#   train_custom_net        Canonical FP32 training for custom_net (m=3, strong aug, weighted CE)
#   qat_custom_net          QAT for CustomSmallNet
#   export_custom_net       QONNX export for CustomSmallNet
#   qat_resnet18            QAT for ResNet18
#   export_resnet18         QONNX export for ResNet18
#   train_mobilenetv1       Canonical FP32 KD fine-tune for MobileNetV1
#   qat_mobilenetv1         QAT for MobileNetV1
#   export_mobilenetv1      QONNX export for MobileNetV1
#
# FINN builds use a different container — use run_finn.sh instead.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_FLAGS=(--gpus all)

if [ -z "$1" ]; then
    echo "Usage: bash run.sh <command> [args...]"
    echo "Run 'bash run.sh --help' to see available commands."
    exit 1
fi

CMD="$1"
shift

case "$CMD" in
    train_teacher)          SCRIPT="src/main.py" ;;
    train_test_resnet)      SCRIPT="src/train_test_resnet.py" ;;
    train_resnet18_from_resnet50_kd) SCRIPT="src/train_resnet18_from_resnet50_kd.py" ;;
    eval_teacher_224)       SCRIPT="src/eval_teacher_224.py" ;;
    eval_pipeline_checkpoint) SCRIPT="src/eval_pipeline_checkpoint.py" ;;
    qat_test_resnet)        SCRIPT="src/qat_test_resnet.py" ;;
    export_test_resnet)     SCRIPT="src/export_test_resnet.py" ;;
    train_custom_net)       SCRIPT="src/train_custom_net.py" ;;
    qat_custom_net)         SCRIPT="src/qat_custom_net.py" ;;
    export_custom_net)      SCRIPT="src/export_custom_net.py" ;;
    qat_resnet18)           SCRIPT="src/qat_resnet18.py" ;;
    export_resnet18)        SCRIPT="src/export_resnet18.py" ;;
    train_mobilenetv1)      SCRIPT="src/train_mobilenetv1.py" ;;
    qat_mobilenetv1)        SCRIPT="src/qat_mobilenetv1.py" ;;
    export_mobilenetv1)     SCRIPT="src/export_mobilenetv1.py" ;;
    --help)
        head -18 "$0" | tail -17
        exit 0
        ;;
    *)
        echo "Unknown command: $CMD"
        echo "Run 'bash run.sh --help' to see available commands."
        exit 1
        ;;
esac

case "$CMD" in
    export_test_resnet|export_custom_net|export_resnet18|export_mobilenetv1)
        GPU_FLAGS=()
        ;;
esac

docker run --rm -it "${GPU_FLAGS[@]}" --shm-size=8g \
  -v "$REPO_DIR:/app" \
  -w /app \
  hpps_image \
  python "$SCRIPT" "$@"
