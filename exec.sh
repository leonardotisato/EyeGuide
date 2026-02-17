#!/bin/bash
CONTAINER_NAME="vgg16_gpu1"
IMAGE_NAME="virgi_heif_5"
GPU_SPEC="device=1"
# -------------------------------
# Path mounting
# -------------------------------
HOST_PERSPECTIVE="/home/virginia/PerspectiveStudy"
HOST_EYECANCER="/home/virginia/EyeCancerDetection"

CONTAINER_PERSPECTIVE="/home/virginia/PerspectiveStudy"
CONTAINER_EYECANCER="/home/virginia/EyeCancerDetection"


CMD="python3 -m pip install wandb 'albumentations[imgaug]' && \
python3 /home/virginia/PerspectiveStudy/knowledge_distillation/src/main.py"


# -------------------------------
# Launch the container
# -------------------------------
docker run -d \
  --gpus ${GPU_SPEC} \
  -e TORCH_HOME=/tmp/.cache/torch \
  -v ${HOST_PERSPECTIVE}:${CONTAINER_PERSPECTIVE} \
  -v ${HOST_EYECANCER}:${CONTAINER_EYECANCER} \
  -w /home/virginia/PerspectiveStudy/knowledge_distillation \
  --name ${CONTAINER_NAME} \
  ${IMAGE_NAME} \
  /bin/bash -c "${CMD}"
