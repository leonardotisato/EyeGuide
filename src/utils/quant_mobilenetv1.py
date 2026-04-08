"""
Brevitas-native quantized MobileNetV1 for FINN-compatible QONNX export.

Architecture matches the original MobileNetV1 paper (Howard et al. 2017):
  - Stem: 3x3 conv, stride=2, 3->32 channels
  - 13 depthwise separable blocks (DWConv 3x3 + PWConv 1x1)
  - TruncAvgPool2d (16x16 for 512x512 input) -> FC(nr_classes)

Quantizer design mirrors the official Brevitas finn-examples MobileNetV1
(brevitas_examples/imagenet_classification/models/mobilenetv1.py):
  - Float (LOG_FP) weight quantizers, per-channel — same as CommonIntWeightPerChannelQuant.
  - Float (LOG_FP) activation quantizers, unsigned — same as CommonUintActQuant.
  - Per-channel activation scaling on stem and PW conv outputs (blocks 0-10).
    PW convolutions have 64-512 output channels with very different activation
    ranges; per-tensor forces one scale across all channels, wasting resolution.
    The last two blocks (flat 11-12) use per-tensor (official does the same).
  - Per-tensor activation scaling on DW conv outputs (groups=in_channels,
    each filter has 9 weights — per-channel overkill here).
  - TruncAvgPool2d instead of AdaptiveAvgPool2d — FINN InferGlobalAccPoolLayer
    requires TruncAvgPool2d to keep the data path quantized through pooling.
  - Signed input (Int8) — ImageNet-normalized inputs have negative values.
  - return_quant_tensor=True everywhere for QONNX export metadata.
  - BatchNorm after every conv — FINN Streamline handles folding.

Pretrained weights:
  load_timm_weights() loads timm mobilenetv1_100 weights (local KD checkpoint
  or ImageNet pretrained) and maps them into this model.
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.nn import TruncAvgPool2d
from brevitas.quant import (
    Int8WeightPerTensorFloat,
    Uint8ActPerTensorFloat,
    Int8ActPerTensorFloat,
    Int32Bias,
)
from brevitas.core.restrict_val import RestrictValueType


# ---------------------------------------------------------------------------
# Quantizer classes — mirrors official finn-examples common.py exactly.
# ---------------------------------------------------------------------------

class CommonIntWeightPerChannelQuant(Int8WeightPerTensorFloat):
    """Per-channel weight quantizer. bit_width set per-layer via subclass."""
    scaling_min_val = 2e-16
    scaling_per_output_channel = True


class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """Per-tensor weight quantizer (FC layer)."""
    scaling_min_val = 2e-16


class CommonUintActQuant(Uint8ActPerTensorFloat):
    """Unsigned activation quantizer. bit_width set per-layer via subclass."""
    scaling_min_val = 2e-16
    restrict_scaling_type = RestrictValueType.LOG_FP


class Int8WeightPerChannelFloat(CommonIntWeightPerChannelQuant):
    bit_width = 8


class Int8WeightPerTensorFloatQuant(CommonIntWeightPerTensorQuant):
    bit_width = 8


class Uint8ActPerTensorFloatQuant(CommonUintActQuant):
    bit_width = 8


WEIGHT_QUANT    = Int8WeightPerChannelFloat       # all convs: per-channel
FC_WEIGHT_QUANT = Int8WeightPerTensorFloatQuant   # FC: per-tensor
ACT_QUANT       = Uint8ActPerTensorFloatQuant     # all activations: unsigned
INP_QUANT       = Int8ActPerTensorFloat           # input: signed 8-bit
BIAS_QUANT      = Int32Bias

DEFAULT_WEIGHT_BITS = 8
DEFAULT_ACT_BITS    = 8

# Feature map before TruncAvgPool2d: 512 / 2^5 = 16x16 (5 stride-2 ops)
_AVGPOOL_KERNEL = 16

# MobileNetV1 block config: (in_channels, out_channels, stride) x 13
_DW_CONFIG = [
    (32,    64,  1),
    (64,   128,  2),
    (128,  128,  1),
    (128,  256,  2),
    (256,  256,  1),
    (256,  512,  2),
    (512,  512,  1),
    (512,  512,  1),
    (512,  512,  1),
    (512,  512,  1),
    (512,  512,  1),
    (512, 1024,  2),
    (1024, 1024, 1),
]


class QuantDWSepBlock(nn.Module):
    """Quantized Depthwise Separable block: DWConv + BN + ReLU + PWConv + BN + ReLU.

    DW activation: per-tensor (each DW filter has 9 weights — per-channel overkill).
    PW activation: per-channel when pw_act_per_channel=True, per-tensor otherwise.
    Official finn-examples uses per-channel for blocks 0-10, per-tensor for 11-12.
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 weight_quant=None, act_quant=None, pw_act_per_channel=True):
        super().__init__()
        # Depthwise conv
        self.conv_dw = qnn.QuantConv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False,
            weight_quant=weight_quant, return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        # DW activation: per-tensor
        self.relu1 = qnn.QuantReLU(act_quant=act_quant, return_quant_tensor=True)
        # Pointwise conv
        self.conv_pw = qnn.QuantConv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False,
            weight_quant=weight_quant, return_quant_tensor=True,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        # PW activation: per-channel or per-tensor
        if pw_act_per_channel:
            self.relu2 = qnn.QuantReLU(
                act_quant=act_quant,
                per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                scaling_stats_permute_dims=(1, 0, 2, 3),
                scaling_per_output_channel=True,
                return_quant_tensor=True,
            )
        else:
            self.relu2 = qnn.QuantReLU(act_quant=act_quant, return_quant_tensor=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class QuantMobileNetV1(nn.Module):
    """Brevitas-native quantized MobileNetV1, mirroring official finn-examples."""

    def __init__(self, nr_classes=4,
                 weight_bit_width=DEFAULT_WEIGHT_BITS,
                 act_bit_width=DEFAULT_ACT_BITS):
        super().__init__()

        self.quant_inp = qnn.QuantIdentity(
            act_quant=INP_QUANT, return_quant_tensor=True,
        )
        # Stem conv: per-channel weights, per-channel activation (out=32)
        self.conv1 = qnn.QuantConv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bias=False,
            weight_quant=WEIGHT_QUANT, return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = qnn.QuantReLU(
            act_quant=ACT_QUANT,
            per_channel_broadcastable_shape=(1, 32, 1, 1),
            scaling_stats_permute_dims=(1, 0, 2, 3),
            scaling_per_output_channel=True,
            return_quant_tensor=True,
        )

        # 13 depthwise separable blocks.
        # Per-channel PW activation for blocks 0-10, per-tensor for 11-12
        # (matches official finn-examples pw_activation_scaling_per_channel logic).
        self.blocks = nn.Sequential(*[
            QuantDWSepBlock(
                inc, outc, stride=s,
                weight_quant=WEIGHT_QUANT,
                act_quant=ACT_QUANT,
                pw_act_per_channel=(flat_idx < 11),
            )
            for flat_idx, (inc, outc, s) in enumerate(_DW_CONFIG)
        ])

        # TruncAvgPool2d: FINN-compatible, keeps data quantized through pooling.
        self.avgpool = TruncAvgPool2d(
            kernel_size=_AVGPOOL_KERNEL,
            stride=1,
            bit_width=act_bit_width,
            float_to_int_impl_type='ROUND',
        )
        self.fc = qnn.QuantLinear(
            1024, nr_classes,
            bias=True, weight_quant=FC_WEIGHT_QUANT, bias_quant=BIAS_QUANT,
        )

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def model_tag(weight_bit_width, act_bit_width):
    """Canonical tag string for checkpoint/ONNX naming: e.g. '8w8a'."""
    return f"{weight_bit_width}w{act_bit_width}a"


# Mapping from timm mobilenetv1_100 group/index pairs to flat block index.
# timm groups blocks as blocks.{g}.{i}; we use a flat nn.Sequential.
# Group sizes: [1, 2, 2, 6, 2] = 13 blocks total.
_TIMM_GROUP_TO_FLAT = [
    (0, 0),   # flat 0
    (1, 0),   # flat 1
    (1, 1),   # flat 2
    (2, 0),   # flat 3
    (2, 1),   # flat 4
    (3, 0),   # flat 5
    (3, 1),   # flat 6
    (3, 2),   # flat 7
    (3, 3),   # flat 8
    (3, 4),   # flat 9
    (3, 5),   # flat 10
    (4, 0),   # flat 11
    (4, 1),   # flat 12
]


def load_timm_weights(model, checkpoint_path=None):
    """
    Load weights from timm mobilenetv1_100 into QuantMobileNetV1.

    Key mapping:
      conv_stem.weight        -> conv1.weight
      bn1.*                   -> bn1.*
      blocks.{g}.{i}.conv_dw -> blocks.{flat}.conv_dw
      blocks.{g}.{i}.bn1.*   -> blocks.{flat}.bn1.*
      blocks.{g}.{i}.conv_pw -> blocks.{flat}.conv_pw
      blocks.{g}.{i}.bn2.*   -> blocks.{flat}.bn2.*
      classifier.*            -> fc.* (shape mismatch check handles 1000-class case)

    Returns (missing_keys, unexpected_keys) from load_state_dict.
    Missing keys should only be Brevitas quantizer internals.
    """
    import timm
    if checkpoint_path is not None:
        print(f"  Loading local FP32 checkpoint: {checkpoint_path}")
        timm_sd = torch.load(checkpoint_path, map_location="cpu")
    else:
        timm_model = timm.create_model("mobilenetv1_100", pretrained=True)
        timm_sd = timm_model.state_dict()

    QUANTIZER_TAGS = (
        "scaling_impl", "int_scaling_impl", "zero_point_impl",
        "tensor_quant", "msb_clamp_bit_width_impl",
    )

    group_to_flat = {(g, i): flat for flat, (g, i) in enumerate(_TIMM_GROUP_TO_FLAT)}

    new_sd = {}
    skipped = []

    for key, value in timm_sd.items():
        if key == "conv_stem.weight":
            new_sd["conv1.weight"] = value
            continue
        if key.startswith("bn1."):
            new_sd[key] = value
            continue
        if key.startswith("blocks."):
            parts = key.split(".")
            g, i = int(parts[1]), int(parts[2])
            flat = group_to_flat.get((g, i))
            if flat is None:
                skipped.append(key)
                continue
            rest = ".".join(parts[3:])
            new_sd[f"blocks.{flat}.{rest}"] = value
            continue
        if key.startswith("classifier."):
            rest = key[len("classifier."):]
            new_sd[f"fc.{rest}"] = value
            continue
        skipped.append(key)

    if skipped:
        print(f"  Skipped {len(skipped)} timm keys (global_pool — expected)")

    model_sd = model.state_dict()
    shape_skipped = []
    for k in list(new_sd.keys()):
        if k in model_sd and new_sd[k].shape != model_sd[k].shape:
            shape_skipped.append(
                f"{k}: timm {tuple(new_sd[k].shape)} vs model {tuple(model_sd[k].shape)}"
            )
            del new_sd[k]
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
