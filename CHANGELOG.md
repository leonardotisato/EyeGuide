# Changelog

## 2026-04-16 - Script and naming cleanup

### custom_net
- Canonical FP32 trainer consolidated to `src/train_custom_net.py`
- Retired `src/train_custom_net_kd.py`
- Active utility modules renamed:
  - `src/utils/custom_small_net.py` -> `src/utils/custom_net.py`
  - `src/utils/quant_custom_small_net.py` -> `src/utils/quant_custom_net.py`
- Active QAT/export path kept aligned to the canonical plain `custom_net_m3` lineage

### MobileNetV1
- Canonical active script family renamed for clarity:
  - `src/train_mobilenetv1.py`
  - `src/qat_mobilenetv1.py`
  - `src/export_mobilenetv1.py`
  - `src/finn_build/build_mobilenetv1.py`
  - `src/finn_build/custom_steps_mobilenetv1.py`
- Checkpoint / ONNX artifact names intentionally kept unchanged:
  - `models/mobilenetv1_fp32_kd.pth`
  - `models/mobilenet_8w8a_qat.pth`
  - `models/mobilenet_8w8a.onnx`

### Docs sync
- `AGENTS.md` and `CLAUDE.md` updated to reflect:
  - current `test_resnet` FINN estimate status
  - renamed `custom_net` utility modules
  - renamed MobileNetV1 script family

## 2026-04-15 — test_resnet QAT + Export + FINN pipeline

### Architecture fixes (QuantTestResNet)
- Downsample paths: replaced stride-2 `QuantConv1x1` with `TruncAvgPool2d(k=2,s=2) + QuantConv1x1(s=1)` — matches timm's `AvgPool2d + Conv1x1` layout exactly, enabling 1:1 weight transfer without index remapping
- Stem: restored `nn.MaxPool2d(k=3,s=2,pad=1)` after `stem_relu3` (reverted `stem_conv3` stride 2→1) — matches timm architecture, eliminates stem mismatch
- Removed `_DS_INDEX_MAP` from `load_fp32_weights()` — no longer needed with TruncAvgPool2d

### QAT (8w8a)
- FP32 source: `test_resnet_fp32_kd.pth` (exp 5 — best FP32, 81.7% test F1)
- Final hyperparameters: LR=1e-5, BN freeze epoch 5, patience 20, unweighted CE
- Best val F1: **79.77%** (epoch 54/74) — stable monotonic climb vs. previous spike-at-epoch-4 pattern
- Checkpoint: `models/test_resnet_8w8a_qat.pth`

### Test results (8w8a QAT)

| Class | F1 |
|---|---|
| 0 (healthy) | 92.1% |
| 1 (nevus) | 83.6% |
| 2 (melanoma) | 81.4% |
| 3 (chrpe) | 82.2% |
| **Weighted avg** | **86.18% raw / 84.82% bootstrap** |

QAT model **exceeds FP32 baseline** (81.7%) — quantization noise acted as regularization on this small dataset.

### Export
- QONNX exported to `models/test_resnet_8w8a.onnx`
- Numerical validation: max PyTorch vs QONNX diff = 8e-6 ✓

### FINN build pipeline
- `src/finn_build/build_test_resnet.py` + `custom_steps_test_resnet.py` written
- Added MaxPool-related transforms:
  - `MoveMulPastMaxPool` in streamline (×2)
  - `MakeMaxPoolNHWC` in lower step
  - `InferStreamingMaxPool` + `InferPool` in to-HW step
- Partition cycle root cause identified and fixed (2026-04-16):
  - **Brevitas version mismatch**: `AvgPoolAndTruncToQuantAvgPool` (built-in FINN) expects
    5-input Trunc with `AveragePool → Mul(k²) → Trunc`. Our Brevitas exports 6-input Trunc
    as `AveragePool → Trunc` (no k² Mul) — built-in transform silently skips → 3×(AveragePool+Trunc)
    remain non-HW → partition cycle.
  - **Stem MaxPool**: `InferStreamingMaxPool` requires k=s, rejects stem `MaxPoolNHWC(k=3,s=2)`.
    Fixed by adding `InferPool` which handles overlapping kernels via Im2Col+Pool_Batch.
  - **Fix**: `ConvertAvgPoolTruncToQuantAvgPool` custom transform added to
    `custom_steps_test_resnet.py`. Runs at start of streamline step; converts
    `AveragePool → Trunc` → `Div(out_scale) → QuantAvgPool2d → Mul(out_scale)`.
    Upstream `Mul(in_scale) → Div(out_scale)` collapses to identity during streamlining.
    `InferPool` in to-HW step then converts QuantAvgPool2d and MaxPoolNHWC(k=3,s=2) to HW.
- Full FINN estimates-only pipeline now completes successfully (2026-04-16)
- Layer resource estimates:
  - `BRAM_18K`: 190
  - `LUT`: 84,965
  - `DSP`: 15
  - `URAM`: 0
- Status:
  - Fits Kria KV260 on FINN estimates
  - Slightly above Ultra96 estimate budget, but close enough to keep as a secondary target

## 2026-04-12 — Pivot to test_resnet.r160_in1k

CustomSmallNet plateaus at ~65% test F1 regardless of training configuration. Searched timm for tiny pretrained models compatible with FINN (BN+ReLU, no special ops). Only viable candidate: `test_resnet.r160_in1k` (~470K params, ImageNet pretrained at 160×160).

### FP32 fine-tune (completed)
- Model: `timm test_resnet.r160_in1k` (pretrained), fine-tuned at 224×224
- Strong augmentation + class weights, no KD
- Best val F1: **79.8%** (epoch 82/100)
- Historical baseline checkpoint: `models/test_resnet_fp32.pth`
- Note: the final canonical FP32 source for the active QAT/export/FINN path is
  `models/test_resnet_fp32_kd.pth` from the later KD + unweighted + strong-augmentation run
- Test accuracy: **81.2%** | Test F1: **80.6%** — best result across all architectures

## 2026-04-09 — CustomSmallNet Pipeline

### Accuracy experiments (8 FP32 + 1 QAT)

Exhaustive search over multiplier (3/4), augmentation (light/strong), class weights,
and KD from ResNet18 teacher. All plateau at ~65–67% test F1. Strong augmentation
was the only clearly beneficial change. Root cause: no pretrained weights on ~600 samples.

| # | m | Aug | Weights | KD | Val F1 | Test F1 |
|---|---|---|---|---|---|---|
| 1 | 3 | light | no | yes | 59.9% | 65.1% |
| 2 | 3 | light | yes | no | 67.9% | 62.8% |
| 3 | 4 | light | yes | no | 69.0% | 61.3% |
| 4 | 3 | strong | yes | no | 69.0% | 66.3% |
| 5 | 4 | strong | yes | no | 69.1% | 62.0% |
| 6 | 3 | strong | no | no | 64.6% | 62.6% |
| 7 | 3 | strong | yes | yes | 64.3% | 64.3% |
| 8 | 4 | strong | yes | yes | 66.2% | 61.5% |
| QAT | 3 | strong | yes | — | 70.7% | 65.5% |

### FINN build (completed)
- Pipeline: `src/finn_build/build_custom_net.py` + `custom_steps_custom_net.py`
- Fixed partition cycle: missing `InferConvInpGen` in to-HW transforms, plus
  `AbsorbScalarMulIntoMatMul` custom transform for GlobalAccPool's 1/N Mul

### FINN estimates (m=3, 8w8a, PE=1, SIMD=1)

| Resource | Used | Available (Ultra96) | Fits? |
|----------|------|----------------------|-------|
| BRAM_18K | 116 | 432 | ✓ |
| LUT | 56,000 | 70K | ✓ |
| DSP | 7 | 360 | ✓ |

## 2026-04-07 — MobileNetV1 QAT + Export

### FP32 KD fine-tune (completed)
- Teacher: ResNet18 FP32 KD (`resnet18_fp32_kd.pth`, 87.28% acc)
- Student: `timm mobilenetv1_100` (ImageNet pretrained), KD with T=4.0, α=0.5
- Checkpoint: `models/mobilenetv1_fp32_kd.pth` — best val F1: **82.86%** (26 epochs)

### 8w8a QAT (completed)
- Load KD weights → calibrate (100 batches) → 60-epoch QAT with early stopping
- Converged in 5 epochs: best val F1: **82.0%**
- Checkpoint: `models/mobilenet_8w8a_qat.pth`
- Export: `models/mobilenet_8w8a.onnx`

### 4w4a QAT (completed, not yet exported)
- Checkpoint: `models/mobilenet_4w4a_qat.pth`
- Best val F1 < 70%
- Export and FINN build pending

### Cleanup
- PTQ code/results archived to `ignore/archive/` (PTQ collapsed to 24% acc on MobileNetV1)
- `results_vgg16/mobilenet/` moved to `results/mobilenet/` (archived keeper results)

## 2026-04-04 — Pivot to MobileNetV1

All ResNet18 BRAM reduction attempts exhausted. KD training from scratch (width=0.5, all stages) produced 43% test accuracy — catastrophic overfitting on ~600 samples without pretrained init. Decision: switch to MobileNetV1 with ImageNet pretrained weights.

- Full-width 4w4a MobileNetV1: ~117 BRAM → fits both KV260 and Ultra96
- FINN supports depthwise conv via `InferDepthwiseConv`

## 2026-04-03 — Architecture Reduction Experiments + 4w4a QAT

### Architecture reduction (all failed)

| Config | BRAM | Fits Ultra96? | Fits KV260? | Val F1 |
|--------|------|---------------|-------------|--------|
| 4w4a_s4 (baseline) | 2732 | ✗ | ✗ | 80.4% |
| 4w4a_s3 (no layer4) | ~670 | ✗ | ✓ | ~51% (bad init) |
| 4w4a_s2 (no layer3+4) | ~220 | ✓ | ✓ | ~45% (bad init) |
| 4w4a_s4_w0.5 | ~683 | ✗ | ✗ | 43% (overfit) |

Truncation and width reduction all fail: layers optimized for a deeper network can't be repurposed without pretrained init on this dataset size.

### 4w4a QAT results

- Best val F1: 80.4% | Test accuracy: 81.20% | Test F1: 80.36%
- Checkpoint: `models/resnet18_4w4a_qat.pth`
- Export: `models/resnet18_4w4a.onnx`

### FINN estimates at 4w4a (PE=1, SIMD=1)

| Resource | Used | Available (Ultra96) | Fits? |
|----------|------|----------------------|-------|
| LUT | 58,491 | 70K | ✓ |
| BRAM_18K | 2,732 | 432 | ✗ (6.3×) |
| DSP | 122 | 360 | ✓ |
| URAM | 0 | — | ✓ |

LUT solved by 4w4a. BRAM remains the blocker at any ResNet18 depth.

## 2026-03-30 — Clean FINN Dataflow

All residual blocks convert to HW operators. 7 non-HW nodes at graph edges only (input Transpose, classifier tail). Partitioning succeeds.

Key architecture changes required for FINN compatibility:
- Shared QuantReLU instance between main and skip paths (finn-examples pattern)
- Removed MaxPool; conv1 stride=4 instead (MaxPool dequantizes QuantTensors)
- `FixThresholdDataTypes` custom transform to work around a FINN bug in `RoundAndClipThresholds`

8w8a results: LUT 522K (7.4×), BRAM 5,464 (12.6×) — does not fit Ultra96.

## 2026-03-29 — FINN Build Pipeline

Created `src/finn_build/` with `build.py`, `custom_steps.py`, `generate_golden_io.py`.

Pipeline follows finn-examples ResNet18, extended for our model (no MaxPool, GlobalAveragePool, ImageNet normalization preprocessing).

## 2026-03-22 — Brevitas-Native QAT

Implemented `QuantResNet18` in `src/utils/quant_model.py` mirroring torchvision naming for direct weight loading. QAT fine-tune (8w8a) + export pipeline established.

| Metric    | FP32   | INT8 QAT | Delta  |
|-----------|-------:|--------:|-------:|
| Accuracy  | 87.28% | 86.51%  | -0.77% |
| F1        | 86.48% | 85.08%  | -1.40% |

## 2026-03-16 — PTQ Export

| Metric    | FP32   | INT8 PTQ | Delta  |
|-----------|-------:|--------:|-------:|
| Accuracy  | 87.28% | 81.21%  | -6.08% |
| F1        | 86.48% | 79.53%  | -6.95% |

## 2026-03-11 — Knowledge Distillation Training

ResNet18 student trained via KD from teacher model.

- Checkpoint: `models/resnet18_fp32_kd.pth`
- Test accuracy: 87.28% | F1: 86.43%
