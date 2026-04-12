#!/bin/bash
# Unified launcher for all training/QAT/export scripts via Docker.
# Usage: bash run.sh <command> [args...]
#
# Commands:
#   train_test_resnet       FP32 fine-tune test_resnet.r160_in1k
#   qat_test_resnet         QAT for test_resnet
#   export_test_resnet      QONNX export for test_resnet
#   train_custom_net        FP32 training of CustomSmallNet
#   train_custom_net_kd     FP32 training of CustomSmallNet with KD
#   qat_custom_net          QAT for CustomSmallNet
#   export_custom_net       QONNX export for CustomSmallNet
#   qat_resnet18            QAT for ResNet18
#   export_resnet18         QONNX export for ResNet18
#   qat_mobilenet           QAT for MobileNetV1
#   export_mobilenet        QONNX export for MobileNetV1
#   finetune_mobilenetv1    FP32 KD fine-tune of MobileNetV1
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
    qat_test_resnet)        SCRIPT="src/qat_test_resnet.py" ;;
    export_test_resnet)     SCRIPT="src/export_test_resnet.py" ;;
    train_custom_net)       SCRIPT="src/train_custom_net.py" ;;
    train_custom_net_kd)    SCRIPT="src/train_custom_net_kd.py" ;;
    qat_custom_net)         SCRIPT="src/qat_custom_net.py" ;;
    export_custom_net)      SCRIPT="src/export_custom_net.py" ;;
    qat_resnet18)           SCRIPT="src/qat_resnet18.py" ;;
    export_resnet18)        SCRIPT="src/export_resnet18.py" ;;
    qat_mobilenet)          SCRIPT="src/qat_mobilenet.py" ;;
    export_mobilenet)       SCRIPT="src/export_mobilenet.py" ;;
    finetune_mobilenetv1)   SCRIPT="src/finetune_mobilenetv1_fp32.py" ;;
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
