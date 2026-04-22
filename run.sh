#!/bin/bash
# Unified launcher for all training/QAT/export scripts via Docker.
# Usage: bash run.sh <command> [args...]
#
# Commands:
#   train_teacher           ResNet50 KD teacher training (main.py) — produces resnet50_fp32_kd.pth
#   train_test_resnet       Current FP32 rerun for test_resnet (R18 KD teacher, light aug, val-loss selection)
#   eval_teacher_224        Validate ResNet18 teacher at 224x224
#   qat_test_resnet         Plain QAT for test_resnet (current script uses 224 light aug)
#   qat_kd_test_resnet      QAT+KD for test_resnet (R18 teacher 512 light, student 224 light)
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
    eval_teacher_224)       SCRIPT="src/eval_teacher_224.py" ;;
    qat_test_resnet)        SCRIPT="src/qat_test_resnet.py" ;;
    qat_kd_test_resnet)     SCRIPT="src/qat_kd_test_resnet.py" ;;
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

docker run --rm -it --gpus all --shm-size=8g \
  -v /mnt/c/Users/leona/Desktop/POLIMI/Programmi/NECST/HPPS:/app \
  -w /app \
  hpps_image \
  python "$SCRIPT" "$@"
