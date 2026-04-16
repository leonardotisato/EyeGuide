"""
Small custom CNN for 4-class fundus classification.

Architecture from tutors, adapted for our task:
  - 6 conv layers with increasing channels (multiplier controls width)
  - BatchNorm + ReLU after each conv
  - MaxPool2d(2,2) after first 5 blocks, global avg pool after 6th
  - 2 FC layers (hidden=64 -> nr_classes)
  - No dilation, no residual connections

With multiplier=3 (default): 9->18->36->72->96->120 channels.
Very small model, should trivially fit any FPGA at any bit width.

Input: 3-channel RGB, 512x512 (after 5 maxpools: 16x16 before global avg pool).
"""

import torch
import torch.nn as nn


class CustomSmallNet(nn.Module):
    def __init__(self, nr_classes=4, multiplier=3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 9
            nn.Conv2d(3, 3 * multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 9 -> 18
            nn.Conv2d(3 * multiplier, 6 * multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 18 -> 36
            nn.Conv2d(6 * multiplier, 12 * multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 36 -> 72
            nn.Conv2d(12 * multiplier, 24 * multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 72 -> 96
            nn.Conv2d(24 * multiplier, 32 * multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 6: 96 -> 120
            nn.Conv2d(32 * multiplier, 40 * multiplier, kernel_size=3, stride=1, padding=1),
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
