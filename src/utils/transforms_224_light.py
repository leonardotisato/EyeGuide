import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

SIZE = 224

train_transform_class = A.Compose([
    A.Resize(SIZE, SIZE),
    A.HorizontalFlip(p=0.5),
    A.Affine(
        rotate=(-30, 30),
        translate_percent=(0, 0.1),
        p=0.5
    ),
    A.RandomCrop(height=SIZE, width=SIZE, p=0.5),
    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])


test_transform_class = A.Compose([
    A.Resize(height=SIZE, width=SIZE),
    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])
