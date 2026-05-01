# Changelog

## 2026-04-30 - Ultra96 6w6a hardware baseline

Recorded the first successful `test_resnet_6w6a` full FINN build targeting
`Ultra96` in `baseline-results/hw/test-resnet/ultra96_6w6a.json`.

### Hardware result

| Board | Model | LUT | FF | BRAM36 | BRAM18 | DSP | URAM |
|---|---|---:|---:|---:|---:|---:|---:|
| `Ultra96` | `test_resnet_6w6a` | 60,805 | 66,615 | 145 | 85 | 15 | 0 |

- BRAM18-equivalent usage: `375`
- Post-route setup timing met at `10.0 ns` with `WNS = 2.262 ns`
- Post-route hold timing met with `WHS = 0.010 ns`
- FINN estimated throughput from the generated report: `1.73 fps`

## 2026-04-24 - test_resnet KD-QAT sweep

First canonical QAT sweep from the current FP32 branch.

### Pipeline (`src/qat_test_resnet.py`)
- Teacher: `resnet18_from_resnet50_fp32_kd.pth` at 512, strong train, clean eval
- Student init: `test_resnet_fp32_kd.pth`
- Student architecture: `QuantTestResNet`
- Student train input: 224, strong train transform
- Student val/test/calib input: 224, clean eval transform
- Training and validation use `DualResDataset`
- Train and val loss: `alpha * CE + (1 - alpha) * KL`
- `T=3.0`, `alpha=0.25`
- `qat_lr=1e-5`, `weight_decay=1e-4`
- `epochs=200`, `patience=50`
- Quantizer calibration: all batches
- BatchNorm freeze: epoch `5`
- Selection on composite KD `val_loss`

### Results

| Run | Weighted F1 | Macro F1 | Accuracy | Best epoch | Best val loss | Val F1 |
|---|---:|---:|---:|---:|---:|---:|
| `8w8a` KD-QAT | 83.91% | 82.10% | 84.21% | 78 | 1.4777 | 80.70% |
| `6w6a` KD-QAT | **87.53%** | **84.74%** | **87.97%** | 92 | 1.7546 | 76.06% |
| `4w4a` KD-QAT | 52.45% | 43.00% | 57.14% | 178 | 3.4506 | 54.56% |

Checkpoints:
- `models/test_resnet_8w8a_qat.pth`
- `models/test_resnet_6w6a_qat.pth`
- `models/test_resnet_4w4a_qat.pth`

## 2026-04-23 - test_resnet FP32 KD with aligned hypers

Canonical FP32 baseline for the next QAT cycle.

### Pipeline (`src/train_test_resnet.py`)
- Teacher: `resnet18_from_resnet50_fp32_kd.pth` at 512, strong train, clean eval
- Student: `test_resnet.r160_in1k` at 224, strong train, clean eval
- `T=3.0`, `alpha=0.25` (loss = `alpha * CE + (1 - alpha) * KL`)
- `epochs=200`, `patience=50`
- Composite KD `val_loss` selection

### Results

| Run | Weighted F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|
| Aligned hypers | **86.16%** | **83.56%** | **86.47%** |


## 2026-04-23 - test_resnet FP32 KD with new teacher (old hypers)

Teacher swap only.

### Pipeline (`src/train_test_resnet.py`)
- Teacher: `resnet50_fp32_kd.pth` -> `resnet18_from_resnet50_fp32_kd.pth`
- Teacher at 512, strong train, clean eval
- Student: `test_resnet.r160_in1k` at 224, strong train, clean eval
- `DualResDataset` for train and validation
- Validation uses the training KD objective
- Selection on composite KD `val_loss`
- `T=4.0`, `alpha=0.5`
- `epochs=100`, `patience=30`

### Results

| Run | Weighted F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|
| New teacher + old hypers | 83.81% | 80.95% | 84.21% |


## 2026-04-23 - Intermediate R18-from-R50 teacher

New teacher distilled from `resnet50_fp32_kd.pth` into a ResNet18 at 512
(`src/train_resnet18_from_resnet50_kd.py`).

### Pipeline
- Teacher and student both at 512
- Strong train transforms, clean eval transforms
- KD weights: `soft=0.75`, `ce=0.25`, `T=3.0`

### Results

| Model | Weighted F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|
| `resnet18_from_resnet50_fp32_kd` | 90.73% | 89.38% | 90.98% |


## 2026-04-23 - FP32 reruns with audit fixes (R50 direct teacher)

Direct R50 -> test_resnet path rerun with the audit fixes applied.

### Pipeline (`src/train_test_resnet.py`)
- Teacher: `resnet50_fp32_kd.pth` at 512
- Student: `test_resnet.r160_in1k` at 224
- `DualResDataset` used for train and validation
- Validation uses the training KD objective
- Selection on composite KD `val_loss`
- Teacher transforms match the teacher's own training setup (note that this was already true when we switched to resnet50 as teacher, but it wasn't when the teacher was the resnet18 (trained with light augmentation, fed as teacher with strong augmentation))

### Results

Only the student train augmentation was varied between the two runs.

| Student train aug | Weighted F1 | Macro F1 |
|---|---:|---:|
| Light | 67.80% | 62.98% |
| Strong | **79.15%** | **74.51%** |

Cleaner pipeline lifts the R50-direct path to 79.15%, but stays below the old 81.7%. R50 is large relative to the `test_resnet` student — pursue a smaller intermediate teacher next.


## 2026-04-22 - Pipeline audit and fixes

Audit of the old `test_resnet` KD pipeline that produced the 81.7% FP32 baseline (see 2026-04-12 — Pivot to test_resnet.r160_in1k).

### Defects identified
- Selection used `val_f1` on a noisy validation split
- KD validation loss was pure CE, not the composite KD loss used in training
- Teacher inputs at validation did not match the teacher's own training setup

### Fixes applied (`src/train_test_resnet.py`)
- Selection switched to composite KD `val_loss`
- Validation loss switched to the same composite KD loss used in training
- `DualResDataset` extended to validation so teacher/student views stay aligned
- Teacher validation transforms aligned to the teacher's training setup (clean eval)

The old 81.7% FP32 / 86.18% 8w8a results are kept as historical references
but are no longer baselines.


## 2026-04-22 - Direct R50 -> test_resnet KD attempt

R50 teacher plugged directly into the existing `test_resnet` KD pipeline with no pipeline changes.

### Pipeline
- Teacher: `resnet50_fp32_kd.pth` at 512, strong train, clean eval
- Student: `test_resnet.r160_in1k` at 224, strong train, clean eval
- `DualResDataset` for training; single-view validation
- KD loss: `alpha*CE + (1-alpha)*KL`, `alpha=0.5`, `T=4.0`
- Selection by `val_f1`
- Validation loss: pure CE

### Result

| Run | Weighted F1 |
|---|---:|
| Direct R50 teacher | 71.95% |

Below the old 81.7% FP32 baseline.

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
