#!/bin/bash
docker run --rm -it --gpus all --shm-size=8g \
  -v /mnt/c/Users/leona/Desktop/POLIMI/Programmi/NECST/HPPS:/app \
  -w /app \
  hpps_image \
  python src/finetune_mobilenetv1_fp32.py "$@"
