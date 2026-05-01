"""Stronger augmentation pipeline for 512x512 fundus training."""

import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .transforms_512_light import SIZE

train_transform_class = A.Compose([
    A.Resize(SIZE, SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(
        rotate=(-180, 180),
        translate_percent=(0, 0.15),
        scale=(0.85, 1.15),
        p=0.7,
    ),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(SIZE // 8, SIZE // 8),
        hole_width_range=(SIZE // 8, SIZE // 8),
        fill=0,
        p=0.3,
    ),
    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

test_transform_class = A.Compose([
    A.Resize(height=SIZE, width=SIZE),
    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])
