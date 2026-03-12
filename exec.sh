#!/bin/bash
CONTAINER_NAME="hpps_gpu_container"
IMAGE_NAME="hpps_image"
GPU_SPEC="all"
# -------------------------------
# Path mounting
# -------------------------------
HOST_WORKING_DIR="$(pwd)"

CONTAINER_WORKING_DIR="/app"

CMD="python3 src/main.py"

# -------------------------------
# Launch the container
# -------------------------------
echo "Removing old container if it exists..."
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

echo "Starting new container..."
docker run -d \
  --gpus ${GPU_SPEC} \
  --shm-size=8g \
  -e TORCH_HOME=/tmp/.cache/torch \
  -v "${HOST_WORKING_DIR}:${CONTAINER_WORKING_DIR}" \
  -w "${CONTAINER_WORKING_DIR}" \
  --name ${CONTAINER_NAME} \
  ${IMAGE_NAME} \
  /bin/bash -c "${CMD}"
