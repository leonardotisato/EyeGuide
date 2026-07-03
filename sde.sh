# python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_slim128x64_trim160_6w6a.onnx --output-dir build_finn_test_resnet_slim128x64_trim160_6w6a_u250_5000fps_opt --target-fps 5000
# python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_slim128x64_trim160_6w6a.onnx --output-dir build_finn_test_resnet_slim128x64_trim160_6w6a_u250_5000fps_7.5ns --target-fps 10000 --synth-clk-ns 7.5

# python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_slim128x64_trim160_8w8a.onnx --output-dir build_finn_test_resnet_slim128x64_trim160_8w8a_u250_2000fps --target-fps 2000
# python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_slim128x64_trim160_8w8a.onnx --output-dir build_finn_test_resnet_slim128x64_trim160_8w8a_u250_5000fps --target-fps 5000

# python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_trim160_6w6a.onnx --output-dir build_finn_test_resnet_trim160_6w6a_u250_2000fps --target-fps 2000
# python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_trim160_6w6a.onnx --output-dir build_finn_test_resnet_trim160_6w6a_u250_5000fps --target-fps 5000

# python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_trim192_6w6a.onnx --output-dir build_finn_test_resnet_trim192_6w6a_u250_2000fps --target-fps 2000
# python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_trim192_6w6a.onnx --output-dir build_finn_test_resnet_trim192_6w6a_u250_5000fps --target-fps 5000

# python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_6w6a.onnx --output-dir build_finn_test_resnet_6w6a_u250_2000fps --target-fps 2000
# python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_6w6a.onnx --output-dir build_finn_test_resnet_6w6a_u250_5000fps --target-fps 5000

python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_8w8a.onnx --output-dir build_finn_test_resnet_8w8a_u250_2000fps --target-fps 2000
python3 src/finn_build/build_test_resnet.py --board U250 --onnx models/test_resnet_8w8a.onnx --output-dir build_finn_test_resnet_8w8a_u250_5000fps --target-fps 5000
