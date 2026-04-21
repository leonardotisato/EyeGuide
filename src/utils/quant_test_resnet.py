"""
Brevitas-native quantized test_resnet for FINN-compatible QONNX export.

Architecture mirrors timm's test_resnet.r160_in1k with one change:
  - Downsample paths use TruncAvgPool2d + Conv1x1(stride=1), matching timm's
    AvgPool2d + Conv1x1 layout for clean weight transfer.
  - Stem MaxPool2d is preserved (stem_conv3 stride=1, MaxPool after stem_relu3).

Design follows the same patterns as QuantResNet18:
  - FixedPoint quantizers (power-of-2 scales), absorbable by FINN
  - Shared QuantReLU between main and skip paths before residual add
  - Post-ReLU activations UNSIGNED, input quantizer SIGNED (Int8)
  - return_quant_tensor=True everywhere

The model has both BasicBlocks (layer1, layer2, layer4) and a Bottleneck (layer3).
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.nn import TruncAvgPool2d
from brevitas.quant.fixed_point import (
    Int8WeightPerTensorFixedPoint,
    Int8ActPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
from brevitas.quant.scaled_int import Int32Bias


# ---------------------------------------------------------------------------
# Quantizer classes
# ---------------------------------------------------------------------------
class Int4WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 4


class Int6WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 6


class Uint4ActPerTensorFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width = 4


class Uint6ActPerTensorFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width = 6


_WEIGHT_QUANT_MAP = {
    4: Int4WeightPerTensorFixedPoint,
    6: Int6WeightPerTensorFixedPoint,
    8: Int8WeightPerTensorFixedPoint,
}

_ACT_QUANT_MAP = {
    4: Uint4ActPerTensorFixedPoint,
    6: Uint6ActPerTensorFixedPoint,
    8: Uint8ActPerTensorFixedPoint,
}

INP_QUANT = Int8ActPerTensorFixedPoint   # input: always signed 8-bit
BIAS_QUANT = Int32Bias

DEFAULT_WEIGHT_BITS = 8
DEFAULT_ACT_BITS = 8


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
class QuantBasicBlock(nn.Module):
    """Quantized BasicBlock with shared QuantReLU for FINN residual handling."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 shared_quant_act=None, weight_quant=None, act_quant=None):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            weight_quant=weight_quant, return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(
            act_quant=act_quant, return_quant_tensor=True,
        )
        self.conv2 = qnn.QuantConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
            weight_quant=weight_quant, return_quant_tensor=True,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        if downsample is not None:
            shared_quant_act = self.downsample[-1]
        if shared_quant_act is None:
            shared_quant_act = qnn.QuantReLU(
                act_quant=act_quant, return_quant_tensor=True,
            )
        self.relu2 = shared_quant_act
        self.relu_out = qnn.QuantReLU(
            act_quant=act_quant, return_quant_tensor=True,
        )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = out + identity
        out = self.relu_out(out)
        return out


class QuantBottleneck(nn.Module):
    """Quantized Bottleneck with shared QuantReLU for FINN residual handling.

    Structure: conv1(1x1) -> conv2(3x3,stride) -> conv3(1x1,expand) -> add -> relu_out
    relu3 (after bn3, before add) is the shared quantizer with the downsample path.
    """

    expansion = 4

    def __init__(self, inplanes, mid_planes, outplanes, stride=1, downsample=None,
                 shared_quant_act=None, weight_quant=None, act_quant=None):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            inplanes, mid_planes, kernel_size=1, bias=False,
            weight_quant=weight_quant, return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu1 = qnn.QuantReLU(
            act_quant=act_quant, return_quant_tensor=True,
        )
        self.conv2 = qnn.QuantConv2d(
            mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False,
            weight_quant=weight_quant, return_quant_tensor=True,
        )
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.relu2 = qnn.QuantReLU(
            act_quant=act_quant, return_quant_tensor=True,
        )
        self.conv3 = qnn.QuantConv2d(
            mid_planes, outplanes, kernel_size=1, bias=False,
            weight_quant=weight_quant, return_quant_tensor=True,
        )
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.downsample = downsample

        if downsample is not None:
            shared_quant_act = self.downsample[-1]
        if shared_quant_act is None:
            shared_quant_act = qnn.QuantReLU(
                act_quant=act_quant, return_quant_tensor=True,
            )
        self.relu3 = shared_quant_act
        self.relu_out = qnn.QuantReLU(
            act_quant=act_quant, return_quant_tensor=True,
        )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = out + identity
        out = self.relu_out(out)
        return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class QuantTestResNet(nn.Module):
    """Brevitas-native quantized test_resnet.r160_in1k.

    Architecture: deep 3x3 stem (16->16->32), 4 stages with 1 block each.
    Layer3 is a Bottleneck (48->192), others are BasicBlocks.

    Changes from timm original:
      - AvgPool2d in downsample replaced with TruncAvgPool2d (FINN-compatible)
      - Stem MaxPool2d preserved after stem_relu3 (stem_conv3 stride=1)
    """

    def __init__(self, nr_classes=4,
                 weight_bit_width=DEFAULT_WEIGHT_BITS,
                 act_bit_width=DEFAULT_ACT_BITS):
        super().__init__()
        assert weight_bit_width in _WEIGHT_QUANT_MAP
        assert act_bit_width in _ACT_QUANT_MAP

        wq = _WEIGHT_QUANT_MAP[weight_bit_width]
        aq = _ACT_QUANT_MAP[act_bit_width]

        # ── Input quantizer ──────────────────────────────────────────
        self.quant_inp = qnn.QuantIdentity(
            act_quant=INP_QUANT, return_quant_tensor=True,
        )

        # ── Stem: 3x Conv3x3 (replaces timm's conv1 Sequential + MaxPool) ──
        self.stem_conv1 = qnn.QuantConv2d(
            3, 16, kernel_size=3, stride=2, padding=1, bias=False,
            weight_quant=wq, return_quant_tensor=True,
        )
        self.stem_bn1 = nn.BatchNorm2d(16)
        self.stem_relu1 = qnn.QuantReLU(act_quant=aq, return_quant_tensor=True)

        self.stem_conv2 = qnn.QuantConv2d(
            16, 16, kernel_size=3, stride=1, padding=1, bias=False,
            weight_quant=wq, return_quant_tensor=True,
        )
        self.stem_bn2 = nn.BatchNorm2d(16)
        self.stem_relu2 = qnn.QuantReLU(act_quant=aq, return_quant_tensor=True)

        self.stem_conv3 = qnn.QuantConv2d(
            16, 32, kernel_size=3, stride=1, padding=1, bias=False,
            weight_quant=wq, return_quant_tensor=True,
        )
        self.stem_bn3 = nn.BatchNorm2d(32)
        self.stem_relu3 = qnn.QuantReLU(act_quant=aq, return_quant_tensor=True)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Layer1: BasicBlock(32->32, stride=1), no downsample ──────
        shared = self.stem_relu3
        self.layer1 = nn.Sequential(
            QuantBasicBlock(32, 32, stride=1, downsample=None,
                            shared_quant_act=shared, weight_quant=wq, act_quant=aq),
        )
        shared = self.layer1[0].relu_out

        # ── Layer2: BasicBlock(32->48, stride=2), downsample ─────────
        ds2 = nn.Sequential(
            TruncAvgPool2d(kernel_size=2, stride=2, bit_width=act_bit_width,
                           float_to_int_impl_type='FLOOR'),
            qnn.QuantConv2d(32, 48, kernel_size=1, stride=1, bias=False,
                            weight_quant=wq, return_quant_tensor=True),
            nn.BatchNorm2d(48),
            qnn.QuantReLU(act_quant=aq, return_quant_tensor=True),
        )
        self.layer2 = nn.Sequential(
            QuantBasicBlock(32, 48, stride=2, downsample=ds2,
                            shared_quant_act=shared, weight_quant=wq, act_quant=aq),
        )
        shared = self.layer2[0].relu_out

        # ── Layer3: Bottleneck(48->192, stride=2), downsample ────────
        ds3 = nn.Sequential(
            TruncAvgPool2d(kernel_size=2, stride=2, bit_width=act_bit_width,
                           float_to_int_impl_type='FLOOR'),
            qnn.QuantConv2d(48, 192, kernel_size=1, stride=1, bias=False,
                            weight_quant=wq, return_quant_tensor=True),
            nn.BatchNorm2d(192),
            qnn.QuantReLU(act_quant=aq, return_quant_tensor=True),
        )
        self.layer3 = nn.Sequential(
            QuantBottleneck(48, 48, 192, stride=2, downsample=ds3,
                            shared_quant_act=shared, weight_quant=wq, act_quant=aq),
        )
        shared = self.layer3[0].relu_out

        # ── Layer4: BasicBlock(192->96, stride=2), downsample ────────
        ds4 = nn.Sequential(
            TruncAvgPool2d(kernel_size=2, stride=2, bit_width=act_bit_width,
                           float_to_int_impl_type='FLOOR'),
            qnn.QuantConv2d(192, 96, kernel_size=1, stride=1, bias=False,
                            weight_quant=wq, return_quant_tensor=True),
            nn.BatchNorm2d(96),
            qnn.QuantReLU(act_quant=aq, return_quant_tensor=True),
        )
        self.layer4 = nn.Sequential(
            QuantBasicBlock(192, 96, stride=2, downsample=ds4,
                            shared_quant_act=shared, weight_quant=wq, act_quant=aq),
        )

        # ── Head ─────────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = qnn.QuantLinear(
            96, nr_classes,
            bias=True, weight_quant=wq, bias_quant=BIAS_QUANT,
        )

    def forward(self, x):
        x = self.quant_inp(x)

        # Stem
        x = self.stem_conv1(x)
        x = self.stem_bn1(x)
        x = self.stem_relu1(x)
        x = self.stem_conv2(x)
        x = self.stem_bn2(x)
        x = self.stem_relu2(x)
        x = self.stem_conv3(x)
        x = self.stem_bn3(x)
        x = self.stem_relu3(x)
        x = self.stem_pool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def model_tag(weight_bit_width, act_bit_width):
    return f"{weight_bit_width}w{act_bit_width}a"


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
# Key mapping from timm test_resnet checkpoint to QuantTestResNet.
# Stem: timm uses nn.Sequential indices (conv1.0, conv1.3, conv1.6).
#   Weight shapes are identical — stride is metadata, not part of the tensor.
# Downsample: timm has AvgPool2d at .0, Conv at .1, BN at .2.
#   Our model has TruncAvgPool2d at .0, QuantConv at .1, BN at .2, QuantReLU at .3.
#   Conv/BN indices match timm naturally — no remapping needed.
_STEM_KEY_MAP = {
    "conv1.0.": "stem_conv1.",
    "conv1.1.": "stem_bn1.",
    "conv1.3.": "stem_conv2.",
    "conv1.4.": "stem_bn2.",
    "conv1.6.": "stem_conv3.",
    "bn1.": "stem_bn3.",
}

QUANTIZER_TAGS = (
    "scaling_impl", "int_scaling_impl", "zero_point_impl",
    "tensor_quant", "msb_clamp_bit_width_impl",
)


def _remap_key(key):
    """Map a timm checkpoint key to our model's naming."""
    # Stem keys
    for old, new in _STEM_KEY_MAP.items():
        if key.startswith(old):
            return new + key[len(old):]

    # Downsample: indices match timm (Conv at .1, BN at .2) — no remapping needed

    return key


def load_fp32_weights(model, checkpoint_path, strict=False):
    """Load FP32 fine-tuned weights into QuantTestResNet.

    Handles key remapping (stem naming) and skips Brevitas quantizer parameters.
    Downsample indices match timm naturally (Conv at .1, BN at .2).

    Returns (missing_keys, unexpected_keys) from load_state_dict.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    new_state_dict = {}
    skipped = []
    for key, value in state_dict.items():
        new_key = _remap_key(key)
        if any(tag in new_key for tag in QUANTIZER_TAGS):
            skipped.append(key)
            continue
        new_state_dict[new_key] = value

    if skipped:
        print(f"  Skipped {len(skipped)} quantizer params")

    # Drop shape-mismatched keys
    model_sd = model.state_dict()
    shape_skipped = []
    for k in list(new_state_dict.keys()):
        if k in model_sd and new_state_dict[k].shape != model_sd[k].shape:
            shape_skipped.append(f"{k}: ckpt {tuple(new_state_dict[k].shape)} vs model {tuple(model_sd[k].shape)}")
            del new_state_dict[k]
    if shape_skipped:
        print(f"  Skipped {len(shape_skipped)} shape-mismatched keys:")
        for s in shape_skipped:
            print(f"    {s}")

    result = model.load_state_dict(new_state_dict, strict=strict)

    if result.missing_keys:
        print(f"  Missing keys ({len(result.missing_keys)}):")
        non_quant = [k for k in result.missing_keys
                     if not any(tag in k for tag in QUANTIZER_TAGS)]
        quant = [k for k in result.missing_keys
                 if any(tag in k for tag in QUANTIZER_TAGS)]
        if quant:
            print(f"    {len(quant)} Brevitas quantizer params (expected)")
        if non_quant:
            print(f"    [WARNING] Non-quantizer keys missing: {non_quant}")
    if result.unexpected_keys:
        print(f"  [WARNING] Unexpected keys: {result.unexpected_keys}")

    return result.missing_keys, result.unexpected_keys
