"""Shared evaluation profiles and loader builders for the training ladder."""

from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvalProfile:
    """Describe the dataset and transform domain for one evaluation stage."""

    name: str
    dataset_kind: str
    transform_key: str
    description: str


FundusClsDataset = None
FundusClsDatasetZoom = None
prepare_dataframes = None
test_transform_224 = None
test_transform_512 = None


RESNET50_ZOOM_TEACHER_PROFILE = "resnet50_zoom_teacher_512"
RESNET50_FULL_KD_PROFILE = "resnet50_full_kd_512"
RESNET18_FROM_RESNET50_KD_PROFILE = "resnet18_from_resnet50_kd_512"
TEST_RESNET_224_PROFILE = "test_resnet_224"
TEST_RESNET_QAT_224_PROFILE = "test_resnet_qat_224"


EVAL_PROFILES = {
    RESNET50_ZOOM_TEACHER_PROFILE: EvalProfile(
        name=RESNET50_ZOOM_TEACHER_PROFILE,
        dataset_kind="zoom",
        transform_key="test_transform_512",
        description="Zoomed 512 eval domain for the first ResNet50 teacher.",
    ),
    RESNET50_FULL_KD_PROFILE: EvalProfile(
        name=RESNET50_FULL_KD_PROFILE,
        dataset_kind="full",
        transform_key="test_transform_512",
        description="Full-image 512 eval domain for the ResNet50 KD student.",
    ),
    RESNET18_FROM_RESNET50_KD_PROFILE: EvalProfile(
        name=RESNET18_FROM_RESNET50_KD_PROFILE,
        dataset_kind="full",
        transform_key="test_transform_512",
        description="Full-image 512 eval domain for the ResNet18 KD student.",
    ),
    TEST_RESNET_224_PROFILE: EvalProfile(
        name=TEST_RESNET_224_PROFILE,
        dataset_kind="full",
        transform_key="test_transform_224",
        description="Full-image 224 eval domain for the FP32 test_resnet student.",
    ),
    TEST_RESNET_QAT_224_PROFILE: EvalProfile(
        name=TEST_RESNET_QAT_224_PROFILE,
        dataset_kind="full",
        transform_key="test_transform_224",
        description="Full-image 224 eval domain for QAT test_resnet models.",
    ),
}


def get_eval_profile(profile_name):
    """Return a validated evaluation profile from a name or profile instance."""

    if isinstance(profile_name, EvalProfile):
        return profile_name
    if profile_name not in EVAL_PROFILES:
        available = ", ".join(sorted(EVAL_PROFILES))
        raise ValueError(f"Unknown eval profile '{profile_name}'. Available profiles: {available}")
    return EVAL_PROFILES[profile_name]


def _load_prepare_dataframes():
    global prepare_dataframes

    if prepare_dataframes is None:
        from .dataset import prepare_dataframes as _prepare_dataframes

        prepare_dataframes = _prepare_dataframes


def _load_dataset_class(dataset_kind):
    global FundusClsDataset
    global FundusClsDatasetZoom

    if dataset_kind == "full" and FundusClsDataset is None:
        from .dataset import FundusClsDataset as _FundusClsDataset

        FundusClsDataset = _FundusClsDataset
    elif dataset_kind == "zoom" and FundusClsDatasetZoom is None:
        from .dataset import FundusClsDatasetZoom as _FundusClsDatasetZoom

        FundusClsDatasetZoom = _FundusClsDatasetZoom


def _load_transform(transform_key):
    global test_transform_224
    global test_transform_512

    try:
        if transform_key == "test_transform_224" and test_transform_224 is None:
            from .transforms_224_light import test_transform_class as _test_transform_224

            test_transform_224 = _test_transform_224
        elif transform_key == "test_transform_512" and test_transform_512 is None:
            from .transforms_512_strong import test_transform_class as _test_transform_512

            test_transform_512 = _test_transform_512
    except ModuleNotFoundError:
        if transform_key == "test_transform_224" and test_transform_224 is None:
            test_transform_224 = lambda **kwargs: kwargs
        elif transform_key == "test_transform_512" and test_transform_512 is None:
            test_transform_512 = lambda **kwargs: kwargs


def _resolve_transform(profile):
    _load_transform(profile.transform_key)
    return {
        "test_transform_224": test_transform_224,
        "test_transform_512": test_transform_512,
    }[profile.transform_key]


def build_eval_dataset(cfg, test_df, profile_name):
    """Build the evaluation dataset that matches the requested profile."""

    profile = get_eval_profile(profile_name)
    transform = _resolve_transform(profile)
    _load_dataset_class(profile.dataset_kind)

    if profile.dataset_kind == "zoom":
        return FundusClsDatasetZoom(
            csv_file=test_df,
            transform=transform,
            dilation_percentage=cfg.dilation_percentage,
            is_training=False,
            random_seed=cfg.RANDOM_SEED,
        )

    if profile.dataset_kind == "full":
        return FundusClsDataset(
            test_df,
            train=False,
            transform=transform,
        )

    raise ValueError(f"Unsupported dataset kind '{profile.dataset_kind}' for profile '{profile.name}'")


def build_test_loader(
    cfg,
    profile_name,
    test_df=None,
    batch_size=None,
    num_workers=None,
    pin_memory=None,
):
    """Build a deterministic test loader for one of the shared eval profiles."""

    if test_df is None:
        _load_prepare_dataframes()
        _, _, test_df = prepare_dataframes(cfg)

    dataset = build_eval_dataset(cfg, test_df, profile_name)
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4) if num_workers is None else num_workers,
        pin_memory=getattr(cfg, "pin_memory", True) if pin_memory is None else pin_memory,
    )
