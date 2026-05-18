"""
Brevitas-native quantized ResNet18 for FINN-compatible QONNX export.

Architecture mirrors torchvision ResNet18 layer-by-layer so that KD checkpoint
weights can be loaded directly (after stripping the 'backbone.' prefix).

Design decisions (matching finn-examples ResNet18 4w4a):
  - Post-ReLU activations UNSIGNED — FINN requires unsigned, non-narrow Quant
    nodes for ReLU activations.
  - Input quantizer (QuantIdentity) SIGNED (Int8, always 8-bit) —
    ImageNet-normalized inputs have negative values (min ≈ -2.12).
  - FixedPoint quantizers (power-of-2 scales) absorbable by FINN.
  - SHARED QuantReLU (relu2) between main and skip paths before the
    residual add, so both branches produce Mul nodes with identical
    scales. MoveLinearPastEltwiseAdd requires np.array_equal on both
    branches — sharing the quantizer instance guarantees this.
  - relu_out requantizes the sum; its instance chains to the next
    block's relu2 (same pattern as finn-examples _make_layer).
  - return_quant_tensor=True everywhere for QONNX export metadata.
  - No Dropout (no learnable params, safe to omit).
  - No MaxPool — conv1 stride=4 replaces stride-2 + maxpool stride-2.
    MaxPool dequantizes QuantTensors, breaking the shared quantizer chain.
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
# Quantizer classes — defined as subclasses with explicit bit_width.
# No Int4/Uint4 classes exist in brevitas.quant.fixed_point; the subclass
# pattern with bit_width override is the standard Brevitas approach.
# ---------------------------------------------------------------------------
class Int4WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 4


class Uint4ActPerTensorFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width = 4


_WEIGHT_QUANT_MAP = {
    4: Int4WeightPerTensorFixedPoint,
    8: Int8WeightPerTensorFixedPoint,
}

_ACT_QUANT_MAP = {
    4: Uint4ActPerTensorFixedPoint,
    8: Uint8ActPerTensorFixedPoint,
}

INP_QUANT = Int8ActPerTensorFixedPoint   # input: always signed 8-bit
BIAS_QUANT = Int32Bias

DEFAULT_WEIGHT_BITS = 4
DEFAULT_ACT_BITS = 4


class QuantBasicBlock(nn.Module):
    """Quantized BasicBlock matching finn-examples shared-quantizer pattern."""

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
        # When downsample exists, relu2 IS the same instance as
        # downsample[-1] (the QuantReLU on the skip path). This ensures
        # both branches produce identical scale Mul nodes in the ONNX
        # graph, which MoveLinearPastEltwiseAdd requires.
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


class QuantResNet18(nn.Module):
    """Brevitas-native quantized ResNet18 matching torchvision naming.

    Args:
        nr_classes: number of output classes.
        weight_bit_width: bit width for weight quantizers (4 or 8).
        act_bit_width: bit width for activation quantizers (4 or 8).
    """

    def __init__(self, nr_classes=4,
                 weight_bit_width=DEFAULT_WEIGHT_BITS,
                 act_bit_width=DEFAULT_ACT_BITS):
        super().__init__()
        assert weight_bit_width in _WEIGHT_QUANT_MAP, f"weight_bit_width must be in {list(_WEIGHT_QUANT_MAP)}"
        assert act_bit_width in _ACT_QUANT_MAP, f"act_bit_width must be in {list(_ACT_QUANT_MAP)}"

        weight_quant = _WEIGHT_QUANT_MAP[weight_bit_width]
        act_quant = _ACT_QUANT_MAP[act_bit_width]

        self.inplanes = 64

        self.quant_inp = qnn.QuantIdentity(
            act_quant=INP_QUANT, return_quant_tensor=True,
        )
        self.conv1 = qnn.QuantConv2d(
            3, 64, kernel_size=7, stride=4, padding=3, bias=False,
            weight_quant=weight_quant, return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = qnn.QuantReLU(
            act_quant=act_quant, return_quant_tensor=True,
        )

        # Chain shared_quant_act through layers (finn-examples pattern).
        shared_quant_act = self.relu

        self.layer1, shared_quant_act = self._make_layer(
            64, 2, stride=1, shared_quant_act=shared_quant_act,
            weight_quant=weight_quant, act_quant=act_quant,
        )
        self.layer2, shared_quant_act = self._make_layer(
            128, 2, stride=2, shared_quant_act=shared_quant_act,
            weight_quant=weight_quant, act_quant=act_quant,
        )
        self.layer3, shared_quant_act = self._make_layer(
            256, 2, stride=2, shared_quant_act=shared_quant_act,
            weight_quant=weight_quant, act_quant=act_quant,
        )
        self.layer4, _ = self._make_layer(
            512, 2, stride=2, shared_quant_act=shared_quant_act,
            weight_quant=weight_quant, act_quant=act_quant,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = qnn.QuantLinear(
            512, nr_classes,
            bias=True, weight_quant=weight_quant, bias_quant=BIAS_QUANT,
        )

    def _make_layer(self, planes, blocks, stride=1, shared_quant_act=None,
                    weight_quant=None, act_quant=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * QuantBasicBlock.expansion:
            downsample = nn.Sequential(
                qnn.QuantConv2d(
                    self.inplanes, planes * QuantBasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False,
                    weight_quant=weight_quant, return_quant_tensor=True,
                ),
                nn.BatchNorm2d(planes * QuantBasicBlock.expansion),
                qnn.QuantReLU(
                    act_quant=act_quant, return_quant_tensor=True,
                ),
            )

        layers = []
        layers.append(QuantBasicBlock(
            self.inplanes, planes, stride, downsample,
            shared_quant_act=shared_quant_act,
            weight_quant=weight_quant, act_quant=act_quant,
        ))
        self.inplanes = planes * QuantBasicBlock.expansion
        for _ in range(1, blocks):
            shared_quant_act = layers[-1].relu_out
            layers.append(QuantBasicBlock(
                self.inplanes, planes,
                shared_quant_act=shared_quant_act,
                weight_quant=weight_quant, act_quant=act_quant,
            ))

        return nn.Sequential(*layers), layers[-1].relu_out

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def model_tag(weight_bit_width, act_bit_width):
    """Canonical tag string for checkpoint/ONNX naming: e.g. '4w4a'."""
    return f"{weight_bit_width}w{act_bit_width}a"


def load_kd_weights(model, checkpoint_path, strict=False, skip_quantizer_params=True):
    """
    Load weights from a KD checkpoint into QuantResNet18.

    The KD checkpoint has keys prefixed with 'backbone.' (from
    ResNet18Classifier wrapping torchvision as self.backbone). This function
    strips that prefix. 'dropout.*' keys are skipped.

    skip_quantizer_params: if True (default), skip any quantizer scale/zero-point
    parameters from the checkpoint so they are initialized fresh by calibration.

    Returns (missing_keys, unexpected_keys) from load_state_dict.
    Missing keys should only be Brevitas quantizer internals.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    QUANTIZER_TAGS = (
        "scaling_impl", "int_scaling_impl", "zero_point_impl",
        "tensor_quant", "msb_clamp_bit_width_impl",
    )

    new_state_dict = {}
    skipped = []
    for key, value in state_dict.items():
        if key.startswith("dropout."):
            skipped.append(key)
            continue
        if key.startswith("backbone."):
            new_key = key[len("backbone."):]
        else:
            new_key = key
        if skip_quantizer_params and any(tag in new_key for tag in QUANTIZER_TAGS):
            skipped.append(key)
            continue
        new_state_dict[new_key] = value

    if skipped:
        quant_skipped = [k for k in skipped if any(t in k for t in QUANTIZER_TAGS)]
        other_skipped = [k for k in skipped if k not in quant_skipped]
        if quant_skipped:
            print(f"  Skipped {len(quant_skipped)} quantizer scale/zp params (will init from calibration)")
        if other_skipped:
            print(f"  Skipped keys: {other_skipped}")

    # Drop any keys whose tensor shape doesn't match the model.
    model_sd = model.state_dict()
    shape_skipped = []
    for k in list(new_state_dict.keys()):
        if k in model_sd and new_state_dict[k].shape != model_sd[k].shape:
            shape_skipped.append(f"{k}: ckpt {tuple(new_state_dict[k].shape)} vs model {tuple(model_sd[k].shape)}")
            del new_state_dict[k]
    if shape_skipped:
        print(f"  Skipped {len(shape_skipped)} shape-mismatched keys (will init randomly):")
        for s in shape_skipped:
            print(f"    {s}")

    result = model.load_state_dict(new_state_dict, strict=strict)

    if result.missing_keys:
        print(f"  Missing keys (expected — Brevitas quantizer params):")
        for k in result.missing_keys:
            print(f"    {k}")
    if result.unexpected_keys:
        print(f"  [WARNING] Unexpected keys:")
        for k in result.unexpected_keys:
            print(f"    {k}")

    return result.missing_keys, result.unexpected_keys
