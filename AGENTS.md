# Project Brief

## Objective

Deploy a quantized model for 4-class fundus-image classification on Kria KV260
or Ultra96-v2-G via Brevitas QAT + FINN. Current best candidate: `test_resnet.r160_in1k`
(86.18% QAT test F1, ~373K params). See CHANGELOG.md for full experiment history.

## Active Workflow (test_resnet)

1. **FP32 fine-tune** (`src/train_test_resnet.py`) — DONE
   - `timm test_resnet.r160_in1k` (ImageNet pretrained), 224×224 input
   - Best FP32: exp 5 (KD + unweighted + strong aug) → 81.7% test F1
   - Checkpoint: `models/test_resnet_fp32_kd.pth`
2. **Brevitas quantized version** (`src/utils/quant_test_resnet.py`) — DONE
   - QuantTestResNet mirrors timm: TruncAvgPool2d in downsample, MaxPool in stem
   - Shared QuantReLU on residual branches; FixedPoint quantizers
3. **QAT fine-tune** (`src/qat_test_resnet.py`) — DONE
   - 8w8a, LR=1e-5, BN freeze epoch 5, patience 20
   - Best val F1: 79.77% (epoch 54) → **86.18% test F1** (exceeds FP32 baseline)
   - Checkpoint: `models/test_resnet_8w8a_qat.pth`
4. **Export** (`src/export_test_resnet.py`) — DONE
   - QONNX exported: `models/test_resnet_8w8a.onnx`
   - Numerical validation: max diff 8e-6 ✓
5. **FINN build** (`src/finn_build/build_test_resnet.py`) — ESTIMATES DONE
   - Full estimates-only FINN pipeline completes successfully
   - Current estimates: BRAM_18K 190, LUT 84,965, DSP 15, URAM 0
   - Fits Kria KV260 on estimates; Ultra96 remains a secondary/borderline target
   - Next project step: full bitstream generation and on-board inference

## Architecture Status

| Architecture | Params | Test F1 | BRAM (8w8a) | Fits Ultra96? | Status |
|---|---|---|---|---|---|
| test_resnet.r160_in1k | 373K | 86.18% (QAT) | 190 | borderline | **ACTIVE** |
| CustomSmallNet m=3 | 206K | 65.5% | 116 | YES | FINN-proven, accuracy-limited |
| MobileNetV1 | 3.2M | ~82% | 1,565 | NO | BRAM+LUT blocker |
| ResNet18 | 11M | 80.4% (4w4a) | 2,732 (4w4a) | NO | BRAM blocker |

## Key Files

| File | Role |
|------|------|
| `src/train_test_resnet.py` | FP32 fine-tune of test_resnet.r160_in1k |
| `src/qat_test_resnet.py` | QAT fine-tuning for test_resnet (8w8a) |
| `src/export_test_resnet.py` | QONNX export for test_resnet |
| `src/finn_build/build_test_resnet.py` | FINN build driver (test_resnet) |
| `src/finn_build/custom_steps_test_resnet.py` | Custom FINN steps (test_resnet) |
| `src/utils/quant_test_resnet.py` | Brevitas QuantTestResNet definition |
| `src/utils/transforms_224_strong.py` | Strong augmentation at 224×224 (for test_resnet) |
| `src/train_custom_net.py` | Canonical FP32 training of custom_net (m=3, strong aug, weighted CE) |
| `src/qat_custom_net.py` | QAT fine-tuning for CustomSmallNet |
| `src/export_custom_net.py` | QONNX export for CustomSmallNet |
| `src/finn_build/build_custom_net.py` | FINN build driver (CustomSmallNet) |
| `src/finn_build/custom_steps_custom_net.py` | Custom FINN steps (CustomSmallNet) |
| `src/utils/custom_net.py` | FP32 CustomSmallNet definition |
| `src/utils/quant_custom_net.py` | Brevitas QuantCustomNet definition |
| `src/utils/transforms_512_strong.py` | Strong augmentation at 512×512 |
| `src/train_mobilenetv1.py` | Canonical FP32 KD fine-tune for MobileNetV1 |
| `src/qat_mobilenetv1.py` | QAT fine-tuning for MobileNetV1 |
| `src/export_mobilenetv1.py` | QONNX export for MobileNetV1 |
| `src/finn_build/build_mobilenetv1.py` | FINN build for MobileNetV1 (BRAM-limited) |
| `src/finn_build/custom_steps_mobilenetv1.py` | Custom FINN steps for MobileNetV1 |
| `src/finn_build/build_resnet18.py` | FINN build for ResNet18 (archived) |
| `src/utils/dataset.py` | Dataset classes and data splitting |
| `src/utils/transforms_512_light.py` | Augmentations (input size: 512×512, light) |
| `config/config.yaml` | Hydra configuration (`results_dir: results_vgg16`) |

## Quantization Notes

### ResNet18 / test_resnet (FixedPoint quantizers)
- Power-of-2 scales → no stray Mul nodes in FINN graph
- Shared QuantReLU instance on both branches of residual add — mandatory
  (`MoveLinearPastEltwiseAdd` requires `np.array_equal` on scales)
- test_resnet has `nn.MaxPool2d` in stem (after stem_relu3) — requires
  `MoveMulPastMaxPool`, `MakeMaxPoolNHWC`, `InferStreamingMaxPool` in FINN pipeline
- ResNet18 has no MaxPool (removed for FINN compatibility; stride-2 conv instead)
- `RoundAndClipThresholds` bug → `FixThresholdDataTypes` custom transform required
- TruncAvgPool2d in downsample skip paths (FINN-compatible AvgPool replacement)

### MobileNetV1 (LOG_FP quantizers)
- Float scales, per-channel weights, unsigned activations (Uint8), signed input (Int8)
- `TruncAvgPool2d` for FINN-compatible pooling
- No residual adds → no shared QuantReLU complexity

### General
- **Do NOT recalibrate when loading a QAT checkpoint** — overwrites trained scales
- **Do NOT fold BN before export** — FINN Streamline handles it
- **Do NOT train from scratch on ~600 samples** — needs pretrained weights
- **BN stays in exported ONNX** — FINN Streamline handles folding
- `return_quant_tensor=True` everywhere; `Int32Bias` on final linear

## Working Principles

1. Always imitate official finn-examples
2. Inspect exported graphs with Netron before attempting FINN builds
3. Introduce FINN transformations incrementally
4. Files under `ignore/` are reference material, not the source of truth
