"""
Brevitas-native quantized CustomSmallNet for FINN-compatible QONNX export.

Architecture mirrors CustomSmallNet exactly:
  - 6 conv blocks: Conv2d + BN + ReLU + MaxPool/GlobalAvgPool
  - 2 FC layers (hidden=64 -> nr_classes)
  - No residual connections, no depthwise separable convs

Quantizer design follows finn-examples ResNet18 pattern (FixedPoint):
  - FixedPoint weight quantizers (power-of-2 scales, absorbable by FINN)
  - FixedPoint unsigned activation quantizers (post-ReLU)
  - Signed FixedPoint input quantizer (ImageNet-normalized inputs have negatives)
  - TruncAvgPool2d instead of AdaptiveAvgPool2d (FINN-compatible)
  - return_quant_tensor=True everywhere
  - BatchNorm after every conv (FINN Streamline handles folding)
  - MaxPool2d stays as nn.MaxPool2d (FINN handles via InferStreamingMaxPool)

FixedPoint quantizers produce power-of-2 scales that FINN absorbs directly
without needing Mul absorption passes. This is the standard approach for
regular CNNs (not depthwise separable) in finn-examples.

Weight loading:
  load_fp32_weights() loads from FP32 CustomSmallNet checkpoint.
  Key structure matches exactly (same nn.Sequential indices).
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.fixed_point import (
    Int8WeightPerTensorFixedPoint,
    Int8ActPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
from brevitas.quant.scaled_int import Int32Bias


# ---------------------------------------------------------------------------
# Quantizer classes — FixedPoint (power-of-2 scales), matching finn-examples
# ResNet18 pattern. Factory functions avoid the hardcoded bit_width bug.
# ---------------------------------------------------------------------------

_weight_quant_cache = {}
_act_quant_cache = {}


def _make_weight_quant(bit_width):
    """Create per-tensor FixedPoint weight quantizer with given bit width."""
    if bit_width not in _weight_quant_cache:
        _weight_quant_cache[bit_width] = type(
            f"Int{bit_width}WeightPerTensorFixedPoint",
            (Int8WeightPerTensorFixedPoint,),
            {"bit_width": bit_width},
        )
    return _weight_quant_cache[bit_width]


def _make_act_quant(bit_width):
    """Create unsigned FixedPoint activation quantizer with given bit width."""
    if bit_width not in _act_quant_cache:
        _act_quant_cache[bit_width] = type(
            f"Uint{bit_width}ActPerTensorFixedPoint",
            (Uint8ActPerTensorFixedPoint,),
            {"bit_width": bit_width},
        )
    return _act_quant_cache[bit_width]


INP_QUANT  = Int8ActPerTensorFixedPoint   # input: signed 8-bit FixedPoint
BIAS_QUANT = Int32Bias

DEFAULT_WEIGHT_BITS = 8
DEFAULT_ACT_BITS    = 8


class QuantCustomSmallNet(nn.Module):
    """Brevitas-native quantized CustomSmallNet.

    Structure mirrors FP32 CustomSmallNet exactly (same nn.Sequential indices)
    so that FP32 checkpoint weights load directly.
    """

    def __init__(self, nr_classes=4, multiplier=3,
                 weight_bit_width=DEFAULT_WEIGHT_BITS,
                 act_bit_width=DEFAULT_ACT_BITS):
        super().__init__()

        WEIGHT_QUANT = _make_weight_quant(weight_bit_width)
        ACT_QUANT    = _make_act_quant(act_bit_width)

        self.quant_inp = qnn.QuantIdentity(
            act_quant=INP_QUANT, return_quant_tensor=True,
        )

        self.features = nn.Sequential(
            # Block 1: 3 -> 3*m
            qnn.QuantConv2d(3, 3 * multiplier, kernel_size=3, stride=1, padding=1,
                            bias=False, weight_quant=WEIGHT_QUANT, return_quant_tensor=True),
            nn.BatchNorm2d(3 * multiplier),
            qnn.QuantReLU(act_quant=ACT_QUANT, return_quant_tensor=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 3*m -> 6*m
            qnn.QuantConv2d(3 * multiplier, 6 * multiplier, kernel_size=3, stride=1, padding=1,
                            bias=False, weight_quant=WEIGHT_QUANT, return_quant_tensor=True),
            nn.BatchNorm2d(6 * multiplier),
            qnn.QuantReLU(act_quant=ACT_QUANT, return_quant_tensor=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 6*m -> 12*m
            qnn.QuantConv2d(6 * multiplier, 12 * multiplier, kernel_size=3, stride=1, padding=1,
                            bias=False, weight_quant=WEIGHT_QUANT, return_quant_tensor=True),
            nn.BatchNorm2d(12 * multiplier),
            qnn.QuantReLU(act_quant=ACT_QUANT, return_quant_tensor=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 12*m -> 24*m
            qnn.QuantConv2d(12 * multiplier, 24 * multiplier, kernel_size=3, stride=1, padding=1,
                            bias=False, weight_quant=WEIGHT_QUANT, return_quant_tensor=True),
            nn.BatchNorm2d(24 * multiplier),
            qnn.QuantReLU(act_quant=ACT_QUANT, return_quant_tensor=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 24*m -> 32*m
            qnn.QuantConv2d(24 * multiplier, 32 * multiplier, kernel_size=3, stride=1, padding=1,
                            bias=False, weight_quant=WEIGHT_QUANT, return_quant_tensor=True),
            nn.BatchNorm2d(32 * multiplier),
            qnn.QuantReLU(act_quant=ACT_QUANT, return_quant_tensor=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 6: 32*m -> 40*m
            qnn.QuantConv2d(32 * multiplier, 40 * multiplier, kernel_size=3, stride=1, padding=1,
                            bias=False, weight_quant=WEIGHT_QUANT, return_quant_tensor=True),
            nn.BatchNorm2d(40 * multiplier),
            qnn.QuantReLU(act_quant=ACT_QUANT, return_quant_tensor=True),
            # AdaptiveAvgPool2d exports as GlobalAveragePool (single ONNX op).
            # TruncAvgPool2d exports as AveragePool + Trunc — the Trunc blocks
            # Mul propagation during streamlining (Mul doesn't commute with Trunc).
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            qnn.QuantLinear(40 * multiplier, 64, bias=True,
                            weight_quant=WEIGHT_QUANT, bias_quant=BIAS_QUANT,
                            return_quant_tensor=True),
            qnn.QuantReLU(act_quant=ACT_QUANT, return_quant_tensor=True),
            nn.Dropout(0.0),  # kept for key alignment with FP32 model
            qnn.QuantLinear(64, nr_classes, bias=True,
                            weight_quant=WEIGHT_QUANT, bias_quant=BIAS_QUANT),
        )

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def model_tag(weight_bit_width, act_bit_width):
    """Canonical tag string for checkpoint/ONNX naming: e.g. '8w8a'."""
    return f"{weight_bit_width}w{act_bit_width}a"


def load_fp32_weights(model, checkpoint_path):
    """Load FP32 CustomSmallNet weights into QuantCustomSmallNet.

    Key structure is identical (same nn.Sequential indices), so loading
    is direct. Conv bias=False in quant version vs bias=True in FP32
    means conv bias keys are skipped (shape mismatch or missing).

    Returns (missing_keys, unexpected_keys).
    Missing keys should only be Brevitas quantizer internals.
    """
    QUANTIZER_TAGS = (
        "scaling_impl", "int_scaling_impl", "zero_point_impl",
        "tensor_quant", "msb_clamp_bit_width_impl",
    )

    fp32_sd = torch.load(checkpoint_path, map_location="cpu")

    # Filter out keys that don't exist in our model or have shape mismatch
    model_sd = model.state_dict()
    new_sd = {}
    shape_skipped = []

    for k, v in fp32_sd.items():
        if k not in model_sd:
            continue
        if v.shape != model_sd[k].shape:
            shape_skipped.append(
                f"{k}: fp32 {tuple(v.shape)} vs quant {tuple(model_sd[k].shape)}"
            )
            continue
        new_sd[k] = v

    if shape_skipped:
        print(f"  Skipped {len(shape_skipped)} shape-mismatched keys:")
        for s in shape_skipped:
            print(f"    {s}")

    result = model.load_state_dict(new_sd, strict=False)

    if result.missing_keys:
        quant_missing = [k for k in result.missing_keys if any(t in k for t in QUANTIZER_TAGS)]
        non_quant_missing = [k for k in result.missing_keys if k not in quant_missing]
        if quant_missing:
            print(f"  Missing {len(quant_missing)} quantizer params (will init from calibration)")
        if non_quant_missing:
            print(f"  [WARNING] Non-quantizer missing keys: {non_quant_missing}")
    if result.unexpected_keys:
        print(f"  [WARNING] Unexpected keys: {result.unexpected_keys}")

    return result.missing_keys, result.unexpected_keys
