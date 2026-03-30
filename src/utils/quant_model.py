"""
Brevitas-native quantized ResNet18 for FINN-compatible QONNX export.

Architecture mirrors torchvision ResNet18 layer-by-layer so that KD checkpoint
weights can be loaded directly (after stripping the 'backbone.' prefix).

Design decisions (matching finn-examples ResNet18 4w4a):
  - Post-ReLU activations UNSIGNED (Uint8) — FINN requires unsigned,
    non-narrow Quant nodes for ReLU activations.
  - Input quantizer (QuantIdentity) SIGNED (Int8) — ImageNet-normalized
    inputs have negative values (min ≈ -2.12).
  - FixedPoint quantizers (power-of-2 scales) absorbable by FINN.
  - SHARED QuantReLU (relu2) between main and skip paths before the
    residual add, so both branches produce Mul nodes with identical
    scales. MoveLinearPastEltwiseAdd requires np.array_equal on both
    branches — sharing the quantizer instance guarantees this.
  - relu_out requantizes the sum; its instance chains to the next
    block's relu2 (same pattern as finn-examples _make_layer).
  - return_quant_tensor=True everywhere for QONNX export metadata.
  - No Dropout (no learnable params, safe to omit).
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


WEIGHT_QUANT = Int8WeightPerTensorFixedPoint
ACT_QUANT = Uint8ActPerTensorFixedPoint       # post-ReLU: unsigned (FINN requirement)
INP_QUANT = Int8ActPerTensorFixedPoint         # input: signed (ImageNet-normalized, has negatives)
BIAS_QUANT = Int32Bias


class QuantBasicBlock(nn.Module):
    """Quantized BasicBlock matching finn-examples shared-quantizer pattern."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 shared_quant_act=None):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            weight_quant=WEIGHT_QUANT, return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(
            act_quant=ACT_QUANT, return_quant_tensor=True,
        )
        self.conv2 = qnn.QuantConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
            weight_quant=WEIGHT_QUANT, return_quant_tensor=True,
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
                act_quant=ACT_QUANT, return_quant_tensor=True,
            )
        self.relu2 = shared_quant_act
        self.relu_out = qnn.QuantReLU(
            act_quant=ACT_QUANT, return_quant_tensor=True,
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
    """Brevitas-native quantized ResNet18 matching torchvision naming."""

    def __init__(self, nr_classes=4):
        super().__init__()
        self.inplanes = 64

        self.quant_inp = qnn.QuantIdentity(
            act_quant=INP_QUANT, return_quant_tensor=True,
        )
        self.conv1 = qnn.QuantConv2d(
            3, 64, kernel_size=7, stride=4, padding=3, bias=False,
            weight_quant=WEIGHT_QUANT, return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = qnn.QuantReLU(
            act_quant=ACT_QUANT, return_quant_tensor=True,
        )

        # Chain shared_quant_act through layers (finn-examples pattern).
        # First block receives relu from stem; each block's relu_out
        # becomes the next block's shared_quant_act (relu2).
        shared_quant_act = self.relu
        self.layer1, shared_quant_act = self._make_layer(64, 2, stride=1, shared_quant_act=shared_quant_act)
        self.layer2, shared_quant_act = self._make_layer(128, 2, stride=2, shared_quant_act=shared_quant_act)
        self.layer3, shared_quant_act = self._make_layer(256, 2, stride=2, shared_quant_act=shared_quant_act)
        self.layer4, _ = self._make_layer(512, 2, stride=2, shared_quant_act=shared_quant_act)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = qnn.QuantLinear(
            512 * QuantBasicBlock.expansion, nr_classes,
            bias=True, weight_quant=WEIGHT_QUANT, bias_quant=BIAS_QUANT,
        )

    def _make_layer(self, planes, blocks, stride=1, shared_quant_act=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * QuantBasicBlock.expansion:
            downsample = nn.Sequential(
                qnn.QuantConv2d(
                    self.inplanes, planes * QuantBasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False,
                    weight_quant=WEIGHT_QUANT, return_quant_tensor=True,
                ),
                nn.BatchNorm2d(planes * QuantBasicBlock.expansion),
                qnn.QuantReLU(
                    act_quant=ACT_QUANT, return_quant_tensor=True,
                ),
            )

        layers = []
        layers.append(QuantBasicBlock(self.inplanes, planes, stride, downsample,
                                      shared_quant_act=shared_quant_act))
        self.inplanes = planes * QuantBasicBlock.expansion
        for _ in range(1, blocks):
            # Chain: previous block's relu_out -> this block's relu2
            shared_quant_act = layers[-1].relu_out
            layers.append(QuantBasicBlock(self.inplanes, planes,
                                          shared_quant_act=shared_quant_act))

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


def load_kd_weights(model, checkpoint_path, strict=False):
    """
    Load weights from a KD checkpoint into QuantResNet18.

    The KD checkpoint has keys prefixed with 'backbone.' (from
    ResNet18Classifier wrapping torchvision as self.backbone). This function
    strips that prefix. 'dropout.*' keys are skipped. 'fc.*' keys are kept
    as-is since both models define fc at the top level.

    Returns (missing_keys, unexpected_keys) from load_state_dict.
    Missing keys should only be Brevitas quantizer internals.
    Unexpected keys should be empty.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

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
        new_state_dict[new_key] = value

    if skipped:
        print(f"  Skipped keys: {skipped}")

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
