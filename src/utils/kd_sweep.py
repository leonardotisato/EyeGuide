"""Utilities for FP32 KD teacher-sweep experiments.

The training scripts keep their legacy artifact names by default. Passing an
``experiment_tag`` activates isolated model/results folders and suffixes each
artifact with teacher and seed, so multi-run sweeps do not collide.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional


TEACHER_SPECS: Dict[str, Dict[str, str]] = {
    "r18_ta": {
        "arch": "resnet18",
        "checkpoint_name": "resnet18_from_resnet50_fp32_kd.pth",
        "description": "ResNet18 teacher assistant distilled from full-field ResNet50",
    },
    "r50_direct": {
        "arch": "resnet50",
        "checkpoint_name": "resnet50_fp32_kd.pth",
        "description": "Full-field ResNet50 teacher distilled from the lesion-centered model",
    },
}


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolved root directories for one training invocation."""

    experiment_tag: Optional[str]
    models_dir: str
    results_root: str
    teacher_models_dir: str


def cfg_select(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a Hydra/OmegaConf key while staying testable with plain objects."""

    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        pass

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    return getattr(cfg, key, default)


def optional_tag(value: Any) -> Optional[str]:
    if value is None:
        return None
    tag = str(value).strip()
    return tag if tag else None


def resolve_experiment_paths(cfg: Any) -> ExperimentPaths:
    """Resolve where models/results should be written for this run."""

    base_models_dir = str(cfg_select(cfg, "models_dir", "models"))
    base_results_dir = str(cfg_select(cfg, "results_dir", "results"))
    teacher_models_dir = str(cfg_select(cfg, "teacher_models_dir", base_models_dir))
    experiment_tag = optional_tag(cfg_select(cfg, "experiment_tag", None))

    if experiment_tag is None:
        return ExperimentPaths(
            experiment_tag=None,
            models_dir=base_models_dir,
            results_root=base_results_dir,
            teacher_models_dir=teacher_models_dir,
        )

    return ExperimentPaths(
        experiment_tag=experiment_tag,
        models_dir=os.path.join(base_models_dir, experiment_tag),
        results_root=os.path.join(base_results_dir, experiment_tag),
        teacher_models_dir=teacher_models_dir,
    )


def resolve_teacher_spec(cfg: Any, paths: Optional[ExperimentPaths] = None) -> Dict[str, str]:
    """Resolve teacher architecture and checkpoint path from Hydra overrides."""

    if paths is None:
        paths = resolve_experiment_paths(cfg)

    teacher_mode = str(cfg_select(cfg, "teacher_mode", "r18_ta"))
    if teacher_mode not in TEACHER_SPECS:
        valid = ", ".join(sorted(TEACHER_SPECS))
        raise ValueError(f"Unknown teacher_mode={teacher_mode!r}. Valid choices: {valid}")

    spec = dict(TEACHER_SPECS[teacher_mode])
    checkpoint_override = optional_tag(cfg_select(cfg, "teacher_checkpoint", None))
    checkpoint_name = spec["checkpoint_name"]

    if checkpoint_override is not None:
        checkpoint_path = (
            checkpoint_override
            if os.path.isabs(checkpoint_override) or os.path.dirname(checkpoint_override)
            else os.path.join(paths.teacher_models_dir, checkpoint_override)
        )
        checkpoint_name = os.path.basename(checkpoint_override)
    else:
        checkpoint_path = os.path.join(paths.teacher_models_dir, checkpoint_name)

    spec.update(
        {
            "mode": teacher_mode,
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": checkpoint_path,
        }
    )
    return spec


def build_teacher_model(teacher_arch: str, nr_classes: int):
    """Instantiate the selected frozen teacher architecture."""

    from utils.model import ResNet18Classifier, ResNet50Classifier

    model_by_arch = {
        "resnet18": ResNet18Classifier,
        "resnet50": ResNet50Classifier,
    }
    if teacher_arch not in model_by_arch:
        valid = ", ".join(sorted(model_by_arch))
        raise ValueError(f"Unsupported teacher architecture {teacher_arch!r}. Valid: {valid}")
    return model_by_arch[teacher_arch](nr_classes=nr_classes, pretrained=False)


def _sweep_suffix(
    teacher_mode: Optional[str],
    seed: Optional[int],
    experiment_tag: Optional[str],
) -> str:
    if optional_tag(experiment_tag) is None:
        return ""
    if teacher_mode is None or seed is None:
        raise ValueError("teacher_mode and seed are required when experiment_tag is set")
    return f"_{teacher_mode}_seed{int(seed)}"


def resolve_trim_fp32_run(
    student_resolution: int,
    teacher_mode: Optional[str] = None,
    seed: Optional[int] = None,
    experiment_tag: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve stable names for one trimmed-input FP32 run."""

    run_tag = f"trim{int(student_resolution)}{_sweep_suffix(teacher_mode, seed, experiment_tag)}"
    return {
        "run_tag": run_tag,
        "checkpoint_name": f"test_resnet_fp32_kd_{run_tag}_ft.pth",
        "results_dir_name": f"test_resnet_{run_tag}",
        "report_name": f"train_test_resnet_{run_tag}_report.json",
        "log_name": f"train_test_resnet_{run_tag}_log.csv",
        "model_type": f"test_resnet_fp32_kd_{run_tag}_ft",
    }


def resolve_slim_fp32_run(
    student_resolution: int,
    layer3_out: int,
    layer4_out: int,
    teacher_mode: Optional[str] = None,
    seed: Optional[int] = None,
    experiment_tag: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve stable names for one slim FP32 run."""

    from utils.test_resnet_slim import slim_variant_tag

    trim_tag = f"trim{int(student_resolution)}"
    variant = slim_variant_tag(layer3_out, layer4_out)
    suffix = _sweep_suffix(teacher_mode, seed, experiment_tag)
    run_tag = f"{variant}_{trim_tag}{suffix}"
    return {
        "variant": variant,
        "trim_tag": trim_tag,
        "run_tag": run_tag,
        "checkpoint_name": f"test_resnet_{variant}_fp32_kd_{trim_tag}{suffix}_ft.pth",
        "results_dir_name": f"test_resnet_{run_tag}",
        "report_name": f"train_test_resnet_{run_tag}_report.json",
        "log_name": f"train_test_resnet_{run_tag}_log.csv",
        "model_type": f"test_resnet_{variant}_fp32_kd_{trim_tag}{suffix}_ft",
    }
