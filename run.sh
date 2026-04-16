#!/bin/bash
# Unified launcher for all training/QAT/export scripts via Docker.
# Usage: bash run.sh <command> [args...]
#
# Commands:
#   train_test_resnet       Canonical FP32 fine-tune for test_resnet (KD + unweighted + strong aug)
#   eval_teacher_224        Validate ResNet18 teacher at 224x224
#   qat_test_resnet         QAT for test_resnet
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
    train_test_resnet)      SCRIPT="src/train_test_resnet.py" ;;
    eval_teacher_224)       SCRIPT="src/eval_teacher_224.py" ;;
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
        head -17 "$0" | tail -16
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
