#!/bin/bash
pip install -q \
  git+https://github.com/fastmachinelearning/qonnx.git@fd61cfeebbdaba351abf7e9d54cd785d7776fa4f \
  git+https://github.com/Xilinx/brevitas.git@84f42259ec869eb151af4cb8a8b23ad925f493db \
  git+https://github.com/Xilinx/finn-experimental.git@0724be21111a21f0d81a072fccc1c446e053f851 \
  git+https://github.com/maltanar/pyverilator.git@ce0a08c20cb8c1d1e84181d6f392390f846adbd1 \
  -e /workspace/hpps/ignore/finn-src
echo "FINN deps installed."
