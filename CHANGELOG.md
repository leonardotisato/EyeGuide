# Changelog

## 2026-04-21 — Stronger KD teacher (ResNet50)

Replaced ResNet18 KD teacher with a ResNet50 KD.

### Pipeline changes (`src/main.py`)
- Step-1 teacher: `ResNet18Classifier` → `ResNet50Classifier` (ImageNet pretrained)
- Step-2 student_kd: same swap (becomes the new KD teacher for test_resnet)
- Augmentation: `transforms_512_light` → `transforms_512_strong`
  (fixes mismatch — teacher was trained light but used under strong aug during testresnet KD finetuning and QAT)
- Checkpoints: `resnet50_fp32_teacher.pth`, `resnet50_fp32_kd.pth`
- `train_test_resnet.py` and `qat_kd_test_resnet.py` updated to load the R50 teacher

### Results (test F1)

| Stage | Old R18 | New R50 | Δ |
|---|---|---|---|
| Step-1 teacher | 82.09% | **87.29%** | +5.20 |
| Step-2 student_kd (test_resnet teacher) | 86.43% | **91.43%** | +5.00 |


## 2026-04-17 — Bit-width and KD experiments (test_resnet)

### QAT results

| Experiment | Test F1 | Val F1 (best) | Epoch | Checkpoint |
|---|---|---|---|---|
| 8w8a plain (baseline) | **86.18%** | 79.77% | 54 | `test_resnet_8w8a_qat.pth` |
| 8w8a + KD | 80.88% | 78.10% | 27 | `test_resnet_8w8a_kd_qat.pth` |
| 6w6a + KD | 79.21% | 76.80% | 54 | `test_resnet_6w6a_kd_qat.pth` |
| 6w6a plain | 70.82% | 79.90% | 17 | `test_resnet_6w6a_qat.pth` |
| 4w4a plain | ~45% | ~53% | 29 | — |
| 4w4a + KD | ~45% | — | — | — |

- Plain experiments: CE
- KD: ResNet18 teacher (`resnet18_fp32_kd.pth`, 512×512), T=4.0, α=0.5, DualResDataset
- KD hurts at 8-bit (quant noise already regularizes), helps at 6-bit (+8.4%)
- 4-bit collapsed from FP32 init regardless of KD or LR tuning (1e-4)


## 2026-04-16 — FINN partition cycle fix

### FINN fix
- `ConvertAvgPoolTruncToQuantAvgPool` custom transform: handles 6-input Brevitas Trunc
  format that built-in `AvgPoolAndTruncToQuantAvgPool` silently skips
- `InferPool` added for stem `MaxPoolNHWC(k=3,s=2)` — `InferStreamingMaxPool` requires k=s
- Full estimates-only pipeline completes: BRAM 190, LUT 85K, DSP 15 — fits KV260

## 2026-04-15 — test_resnet QAT + Export

### Architecture fixes (QuantTestResNet)
- Downsample: `TruncAvgPool2d(k=2,s=2) + QuantConv1x1(s=1)` — matches timm layout
- Stem: restored `nn.MaxPool2d(k=3,s=2,pad=1)` after `stem_relu3`

### 8w8a QAT
- LR=1e-5, BN freeze epoch 5, patience 20, CE
- Best val F1: **79.77%** (epoch 54) → **86.18% test F1** (exceeds FP32 81.7%)


## 2026-04-12 — Pivot to test_resnet.r160_in1k

CustomSmallNet plateaus at ~65% test F1 regardless of training configuration. Searched timm for tiny pretrained models compatible with FINN (BN+ReLU, no special ops). Only viable candidate: `test_resnet.r160_in1k` (~470K params, ImageNet pretrained at 160×160).

### FP32 fine-tune (completed)
- Model: `timm test_resnet.r160_in1k` (pretrained), fine-tuned at 224×224
- KD from ResNet18 teacher (512×512), CE, strong augmentation
- Best val F1: **79.8%** (epoch 82/100)
- Checkpoint: `models/test_resnet_fp32_kd.pth`
- Test F1: **81.7%** — best result across all architectures

## 2026-04-09 — CustomSmallNet Pipeline

### Accuracy experiments (8 FP32 + 1 QAT)

Exhaustive search over multiplier (3/4), augmentation (light/strong), class weights,
and KD from ResNet18 teacher. All plateau at ~65–67% test F1. Strong augmentation
was the only clearly beneficial change. Root cause: no pretrained weights on ~600 samples.
Best result: QAT m=3, strong aug, weighted CE (no KD) → 70.7% val F1, 65.5% test F1.

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
- Load KD weights → calibrate (100 batches) → 60-epoch QAT, CE, early stopping
- Converged in 5 epochs: best val F1: **82.0%**
- Checkpoint: `models/mobilenet_8w8a_qat.pth`
- Export: `models/mobilenet_8w8a.onnx`

### 4w4a QAT (completed)
- Checkpoint: `models/mobilenet_4w4a_qat.pth`
- Best val F1 < 70%

### FINN estimates
- Neither 8w8a nor 4w4a fits any target board (BRAM + LUT both over budget)

### Cleanup
- PTQ code/results archived to `ignore/archive/` 

## 2026-04-04 — Pivot to MobileNetV1

All ResNet18 BRAM reduction attempts exhausted. KD training from scratch (width=0.5, all stages) produced 43% test accuracy — catastrophic overfitting on ~600 samples without pretrained init. Decision: switch to MobileNetV1 with ImageNet pretrained weights.


## 2026-04-03 — Architecture Reduction Experiments + 4w4a QAT

### Architecture reduction (all failed)

| Config | BRAM | Fits Ultra96? | Fits KV260? | Val F1 |
|--------|------|---------------|-------------|--------|
| 4w4a_s4 (baseline) | 2732 | ✗ | ✗ | 80.4% |
| 4w4a_s3 (no layer4) | ~670 | ✗ | ✓ | ~51% (bad init) |
| 4w4a_s2 (no layer3+4) | ~220 | ✓ | ✓ | ~45% (bad init) |
| 4w4a_s4_w0.5 | ~683 | ✗ | ✗ | 43% (overfit) |

Truncation and width reduction all fail: layers optimized for a deeper network can't be repurposed without pretrained init on this dataset size.

### 4w4a QAT results (CE)

- Best val F1: 80.4% | Test accuracy: 81.20% | Test F1: 80.36%

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

Created `src/finn_build/` with `build_resnet18.py`, `custom_steps_resnet18.py`, `generate_golden_io.py`.

Pipeline follows finn-examples ResNet18, extended for our model (no MaxPool, GlobalAveragePool, ImageNet normalization preprocessing).

## 2026-03-22 — Brevitas-Native QAT

Implemented `QuantResNet18` in `src/utils/quant_resnet18.py` mirroring torchvision naming for direct weight loading. QAT fine-tune (8w8a, CE) + export pipeline established.

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
