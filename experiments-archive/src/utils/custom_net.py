"""Small custom CNN for 4-class fundus classification.

This is the revived "crazy recovery" branch, kept intentionally simple:
  - 6 Conv-BN-ReLU blocks
  - MaxPool after the first 5 blocks
  - Adaptive global average pool after the 6th block
  - 2 FC layers (hidden=64 -> nr_classes)
  - no residuals, no depthwise ops, no dilation

The FP32 and quantized variants are kept structurally aligned so that the FP32
checkpoint can warm-start the QAT model without shape mismatches. In particular,
convolution biases are disabled because each conv is immediately followed by BN.
"""

import torch
import torch.nn as nn


class CustomSmallNet(nn.Module):
    def __init__(self, nr_classes=4, multiplier=3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 9
            nn.Conv2d(3, 3 * multiplier, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 9 -> 18
            nn.Conv2d(
                3 * multiplier, 6 * multiplier, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(6 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 18 -> 36
            nn.Conv2d(
                6 * multiplier,
                12 * multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(12 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 36 -> 72
            nn.Conv2d(
                12 * multiplier,
                24 * multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(24 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 72 -> 96
            nn.Conv2d(
                24 * multiplier,
                32 * multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 6: 96 -> 120
            nn.Conv2d(
                32 * multiplier,
                40 * multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(40 * multiplier),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(40 * multiplier, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(64, nr_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
