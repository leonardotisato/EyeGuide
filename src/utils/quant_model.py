"""
Brevitas-native quantized ResNet18 for FINN-compatible QONNX export.

Architecture mirrors torchvision ResNet18 layer-by-layer so that KD checkpoint
weights can be loaded directly (after stripping the 'backbone.' prefix).

Design decisions (matching finn-examples ResNet18 4w4a):
  - All activations SIGNED (Int8) — even after ReLU — so both branches of
    residual adds share the same signed type for FINN compatibility.
  - FixedPoint quantizers (power-of-2 scales) absorbable by FINN.
  - Plain '+' for residual adds (not QuantEltwiseAdd); subsequent QuantReLU
    requantizes the sum.
  - return_quant_tensor=True everywhere for QONNX export metadata.
  - No Dropout (no learnable params, safe to omit).
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.fixed_point import (
    Int8WeightPerTensorFixedPoint,
    Int8ActPerTensorFixedPoint,
)
from brevitas.quant.scaled_int import Int32Bias


WEIGHT_QUANT = Int8WeightPerTensorFixedPoint
ACT_QUANT = Int8ActPerTensorFixedPoint
BIAS_QUANT = Int32Bias


class QuantBasicBlock(nn.Module):
    """Quantized BasicBlock mirroring torchvision's BasicBlock."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
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
        # Plain tensor add — Brevitas dequantizes both sides, float add,
        # then relu_out requantizes. Produces clean Quant→Add→Quant for FINN.
        out = out + identity
        out = self.relu_out(out)
        return out


class QuantResNet18(nn.Module):
    """Brevitas-native quantized ResNet18 matching torchvision naming."""

    def __init__(self, nr_classes=4):
        super().__init__()
        self.inplanes = 64

        self.quant_inp = qnn.QuantIdentity(
            act_quant=ACT_QUANT, return_quant_tensor=True,
        )
        self.conv1 = qnn.QuantConv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False,
            weight_quant=WEIGHT_QUANT, return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = qnn.QuantReLU(
            act_quant=ACT_QUANT, return_quant_tensor=True,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = qnn.QuantLinear(
            512 * QuantBasicBlock.expansion, nr_classes,
            bias=True, weight_quant=WEIGHT_QUANT, bias_quant=BIAS_QUANT,
        )

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * QuantBasicBlock.expansion:
            downsample = nn.Sequential(
                qnn.QuantConv2d(
                    self.inplanes, planes * QuantBasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False,
                    weight_quant=WEIGHT_QUANT, return_quant_tensor=True,
                ),
                nn.BatchNorm2d(planes * QuantBasicBlock.expansion),
            )

        layers = []
        layers.append(QuantBasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * QuantBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(QuantBasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

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
