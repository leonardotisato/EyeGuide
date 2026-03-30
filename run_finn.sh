#!/bin/bash
docker run -t --rm -it --init \
  --hostname finn_dev \
  --entrypoint /bin/bash \
  -w /workspace/hpps \
  -v /mnt/c/Users/leona/Desktop/POLIMI/Programmi/NECST/HPPS:/workspace/hpps \
  -v /tmp/finn_dev_leonardotisato:/tmp/finn_dev_leonardotisato \
  -e FINN_BUILD_DIR=/tmp/finn_dev_leonardotisato \
  -e NUM_DEFAULT_WORKERS=4 \
  xilinx/finn:v0.10.1-6-g8ac41e46.xrt_202220.2.14.354_22.04-amd64-xrt \
  -c "bash /workspace/hpps/finn_setup.sh && cd /workspace/hpps && bash"
