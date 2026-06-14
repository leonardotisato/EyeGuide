"""FP32 slim variant of timm's ``test_resnet.r160_in1k``.

The canonical test_resnet uses a late bottleneck expansion to 192 channels and
a final 96-channel stage. This experiment keeps the early ImageNet-compatible
shape intact and shrinks only the late stages by prefix-slicing compatible
weights from an existing timm/test_resnet checkpoint.
"""

from collections import defaultdict
import os

import torch
import torch.nn as nn


DEFAULT_LAYER3_OUT = 128
DEFAULT_LAYER4_OUT = 64


def slim_variant_tag(layer3_out=DEFAULT_LAYER3_OUT, layer4_out=DEFAULT_LAYER4_OUT):
    return f"slim{layer3_out}x{layer4_out}"


class SlimBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act2(out)
        return out


class SlimBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, mid_planes, outplanes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.downsample = downsample
        self.act3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + identity
        out = self.act3(out)
        return out


class TestResNetSlim(nn.Module):
    """Slim test_resnet with timm-compatible key names.

    Defaults shrink only the late stages:
      - layer3 output: 192 -> 128
      - layer4 output: 96 -> 64

    The stem/layer1/layer2 and layer3 bottleneck mid-width stay unchanged, so
    most pretrained weights can be copied exactly.
    """

    def __init__(
        self,
        nr_classes=4,
        layer3_out=DEFAULT_LAYER3_OUT,
        layer4_out=DEFAULT_LAYER4_OUT,
    ):
        super().__init__()
        self.layer3_out = int(layer3_out)
        self.layer4_out = int(layer4_out)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(SlimBasicBlock(32, 32, stride=1))

        ds2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False),
            nn.Conv2d(32, 48, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.layer2 = nn.Sequential(SlimBasicBlock(32, 48, stride=2, downsample=ds2))

        ds3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False),
            nn.Conv2d(48, self.layer3_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.layer3_out),
        )
        self.layer3 = nn.Sequential(
            SlimBottleneck(48, 48, self.layer3_out, stride=2, downsample=ds3)
        )

        ds4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False),
            nn.Conv2d(self.layer3_out, self.layer4_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.layer4_out),
        )
        self.layer4 = nn.Sequential(
            SlimBasicBlock(
                self.layer3_out,
                self.layer4_out,
                stride=2,
                downsample=ds4,
            )
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.layer4_out, nr_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _prefix_slice_tensor(source_tensor, target_shape):
    if tuple(source_tensor.shape) == tuple(target_shape):
        return source_tensor.detach().clone()
    if source_tensor.ndim != len(target_shape):
        return None
    if any(target_dim > source_dim for target_dim, source_dim in zip(target_shape, source_tensor.shape)):
        return None

    slices = tuple(slice(0, int(dim)) for dim in target_shape)
    return source_tensor[slices].detach().clone()


def adapt_test_resnet_state_dict_for_slim(model, source_state_dict):
    """Adapt a timm/test_resnet checkpoint to ``TestResNetSlim``.

    Exact matches are copied as-is. Shape-compatible tensors are copied by
    prefix slicing along each dimension, which keeps the first channels from the
    pretrained model instead of randomizing every changed late-stage layer.
    """

    model_state = model.state_dict()
    adapted = {}
    report = defaultdict(list)

    for key, target_tensor in model_state.items():
        if key not in source_state_dict:
            report["missing"].append(key)
            adapted[key] = target_tensor.detach().clone()
            continue

        source_tensor = source_state_dict[key]
        sliced = _prefix_slice_tensor(source_tensor, target_tensor.shape)
        if sliced is None:
            report["skipped"].append(key)
            adapted[key] = target_tensor.detach().clone()
            continue

        adapted[key] = sliced.to(dtype=target_tensor.dtype)
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            report["exact"].append(key)
        else:
            report["sliced"].append(key)

    for key in ("exact", "sliced", "missing", "skipped"):
        report[key] = list(report[key])
    return adapted, dict(report)


def load_test_resnet_slim_weights(model, checkpoint_or_state_dict, strict=False):
    if isinstance(checkpoint_or_state_dict, (str, bytes, os.PathLike)):
        source_state_dict = torch.load(checkpoint_or_state_dict, map_location="cpu")
    else:
        source_state_dict = checkpoint_or_state_dict

    adapted, report = adapt_test_resnet_state_dict_for_slim(model, source_state_dict)
    result = model.load_state_dict(adapted, strict=strict)
    return result.missing_keys, result.unexpected_keys, report
