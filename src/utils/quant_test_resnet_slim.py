"""Quantized slim test_resnet experiment for FINN export."""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.nn import TruncAvgPool2d

from utils.quant_test_resnet import (
    BIAS_QUANT,
    DEFAULT_ACT_BITS,
    DEFAULT_WEIGHT_BITS,
    INP_QUANT,
    _ACT_QUANT_MAP,
    _WEIGHT_QUANT_MAP,
    _load_fp32_checkpoint,
    _load_quant_checkpoint,
    model_tag,
    QuantBasicBlock,
    QuantBottleneck,
)
from utils.test_resnet_slim import (
    DEFAULT_LAYER3_OUT,
    DEFAULT_LAYER4_OUT,
    slim_variant_tag,
)


class QuantTestResNetSlim(nn.Module):
    """Brevitas-native slim test_resnet.

    The default channel plan mirrors the FP32 slim experiment:
      - layer3 output: 192 -> 128
      - layer4 output: 96 -> 64
    """

    def __init__(
        self,
        nr_classes=4,
        weight_bit_width=DEFAULT_WEIGHT_BITS,
        act_bit_width=DEFAULT_ACT_BITS,
        layer3_out=DEFAULT_LAYER3_OUT,
        layer4_out=DEFAULT_LAYER4_OUT,
    ):
        super().__init__()
        assert weight_bit_width in _WEIGHT_QUANT_MAP
        assert act_bit_width in _ACT_QUANT_MAP

        self.layer3_out = int(layer3_out)
        self.layer4_out = int(layer4_out)

        wq = _WEIGHT_QUANT_MAP[weight_bit_width]
        aq = _ACT_QUANT_MAP[act_bit_width]

        self.quant_inp = qnn.QuantIdentity(
            act_quant=INP_QUANT, return_quant_tensor=True
        )

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

        shared = self.stem_relu3
        self.layer1 = nn.Sequential(
            QuantBasicBlock(
                32, 32, stride=1, downsample=None,
                shared_quant_act=shared, weight_quant=wq, act_quant=aq,
            )
        )
        shared = self.layer1[0].relu_out

        ds2 = nn.Sequential(
            TruncAvgPool2d(
                kernel_size=2, stride=2, bit_width=act_bit_width,
                float_to_int_impl_type="FLOOR",
            ),
            qnn.QuantConv2d(
                32, 48, kernel_size=1, stride=1, bias=False,
                weight_quant=wq, return_quant_tensor=True,
            ),
            nn.BatchNorm2d(48),
            qnn.QuantReLU(act_quant=aq, return_quant_tensor=True),
        )
        self.layer2 = nn.Sequential(
            QuantBasicBlock(
                32, 48, stride=2, downsample=ds2,
                shared_quant_act=shared, weight_quant=wq, act_quant=aq,
            )
        )
        shared = self.layer2[0].relu_out

        ds3 = nn.Sequential(
            TruncAvgPool2d(
                kernel_size=2, stride=2, bit_width=act_bit_width,
                float_to_int_impl_type="FLOOR",
            ),
            qnn.QuantConv2d(
                48, self.layer3_out, kernel_size=1, stride=1, bias=False,
                weight_quant=wq, return_quant_tensor=True,
            ),
            nn.BatchNorm2d(self.layer3_out),
            qnn.QuantReLU(act_quant=aq, return_quant_tensor=True),
        )
        self.layer3 = nn.Sequential(
            QuantBottleneck(
                48, 48, self.layer3_out, stride=2, downsample=ds3,
                shared_quant_act=shared, weight_quant=wq, act_quant=aq,
            )
        )
        shared = self.layer3[0].relu_out

        ds4 = nn.Sequential(
            TruncAvgPool2d(
                kernel_size=2, stride=2, bit_width=act_bit_width,
                float_to_int_impl_type="FLOOR",
            ),
            qnn.QuantConv2d(
                self.layer3_out, self.layer4_out, kernel_size=1, stride=1, bias=False,
                weight_quant=wq, return_quant_tensor=True,
            ),
            nn.BatchNorm2d(self.layer4_out),
            qnn.QuantReLU(act_quant=aq, return_quant_tensor=True),
        )
        self.layer4 = nn.Sequential(
            QuantBasicBlock(
                self.layer3_out, self.layer4_out, stride=2, downsample=ds4,
                shared_quant_act=shared, weight_quant=wq, act_quant=aq,
            )
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = qnn.QuantLinear(
            self.layer4_out, nr_classes,
            bias=True, weight_quant=wq, bias_quant=BIAS_QUANT,
        )

    def forward(self, x):
        x = self.quant_inp(x)

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def load_test_resnet_slim_quant_weights(model, checkpoint_path, strict=False):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    is_quant_checkpoint = any(
        key.startswith("quant_inp.") or key.startswith("stem_conv1.")
        for key in state_dict.keys()
    )
    if is_quant_checkpoint:
        return _load_quant_checkpoint(model, state_dict, strict=strict)
    return _load_fp32_checkpoint(model, state_dict, strict=strict)


__all__ = [
    "QuantTestResNetSlim",
    "load_test_resnet_slim_quant_weights",
    "model_tag",
    "slim_variant_tag",
]
