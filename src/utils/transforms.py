import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def make_test_transform(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def make_light_train_transform(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            rotate=(-30, 30),
            translate_percent=(0, 0.1),
            p=0.5,
        ),
        A.RandomCrop(height=size, width=size, p=0.5),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def make_strong_train_transform(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
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
            hole_height_range=(size // 8, size // 8),
            hole_width_range=(size // 8, size // 8),
            fill=0,
            p=0.3,
        ),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
