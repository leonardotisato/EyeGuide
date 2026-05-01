"""Offline evaluator for the full ResNet-to-QAT project ladder.

This script gives the repo one short checkpoint-evaluation entry point for the
models that make up the current canonical pipeline:

- ``resnet50_fp32_teacher``
- ``resnet50_fp32_kd``
- ``resnet18_from_resnet50_fp32_kd``
- ``test_resnet_fp32_kd``
- ``test_resnet_{8,6,4}w{8,6,4}a_qat``

Each model key resolves to the correct architecture family and the coherent
evaluation profile that matches how the checkpoint was trained.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.eval_profiles import (
    RESNET18_FROM_RESNET50_KD_PROFILE,
    RESNET50_FULL_KD_PROFILE,
    RESNET50_ZOOM_TEACHER_PROFILE,
    TEST_RESNET_224_PROFILE,
    TEST_RESNET_QAT_224_PROFILE,
    build_test_loader,
)
from utils.model import ResNet18Classifier, ResNet50Classifier
from utils.seed import set_seeds
from utils.training import test


@dataclass(frozen=True)
class EvalTarget:
    """Describe how to instantiate and evaluate one saved pipeline checkpoint."""

    key: str
    checkpoint: str
    report_path: str
    eval_profile: str
    family: str
    weight_bits: int | None = None
    act_bits: int | None = None


TARGETS = {
    "resnet50_fp32_teacher": EvalTarget(
        key="resnet50_fp32_teacher",
        checkpoint="models/resnet50_fp32_teacher.pth",
        report_path="baseline-results/resnet50/test_summary_resnet50_zoom_teacher.json",
        eval_profile=RESNET50_ZOOM_TEACHER_PROFILE,
        family="resnet50_fp32",
    ),
    "resnet50_fp32_kd": EvalTarget(
        key="resnet50_fp32_kd",
        checkpoint="models/resnet50_fp32_kd.pth",
        report_path="baseline-results/resnet50/test_summary_resnet50_full_kd.json",
        eval_profile=RESNET50_FULL_KD_PROFILE,
        family="resnet50_fp32",
    ),
    "resnet18_from_resnet50_fp32_kd": EvalTarget(
        key="resnet18_from_resnet50_fp32_kd",
        checkpoint="models/resnet18_from_resnet50_fp32_kd.pth",
        report_path="baseline-results/resnet18-from-resnet50/test_summary_resnet18_from_resnet50_kd.json",
        eval_profile=RESNET18_FROM_RESNET50_KD_PROFILE,
        family="resnet18_fp32",
    ),
    "test_resnet_fp32_kd": EvalTarget(
        key="test_resnet_fp32_kd",
        checkpoint="models/test_resnet_fp32_kd.pth",
        report_path="baseline-results/test-resnet/test_summary_test_resnet_fp32_kd.json",
        eval_profile=TEST_RESNET_224_PROFILE,
        family="test_resnet_fp32",
    ),
    "test_resnet_8w8a_qat": EvalTarget(
        key="test_resnet_8w8a_qat",
        checkpoint="models/test_resnet_8w8a_qat.pth",
        report_path="baseline-results/qat-test-resnet/8w8a/test_summary_8w8a.json",
        eval_profile=TEST_RESNET_QAT_224_PROFILE,
        family="test_resnet_qat",
        weight_bits=8,
        act_bits=8,
    ),
    "test_resnet_6w6a_qat": EvalTarget(
        key="test_resnet_6w6a_qat",
        checkpoint="models/test_resnet_6w6a_qat.pth",
        report_path="baseline-results/qat-test-resnet/6w6a/test_summary_6w6a.json",
        eval_profile=TEST_RESNET_QAT_224_PROFILE,
        family="test_resnet_qat",
        weight_bits=6,
        act_bits=6,
    ),
    "test_resnet_4w4a_qat": EvalTarget(
        key="test_resnet_4w4a_qat",
        checkpoint="models/test_resnet_4w4a_qat.pth",
        report_path="baseline-results/qat-test-resnet/4w4a/test_summary_4w4a.json",
        eval_profile=TEST_RESNET_QAT_224_PROFILE,
        family="test_resnet_qat",
        weight_bits=4,
        act_bits=4,
    ),
}


def parse_args():
    """Parse CLI arguments for pipeline checkpoint evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate one saved checkpoint from the canonical ResNet/QAT ladder."
    )
    parser.add_argument("--model", choices=sorted(TARGETS), required=True, help="Named pipeline checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    parser.add_argument("--report-path", default=None, help="Optional JSON output override.")
    parser.add_argument("--config", default="config/config.yaml", help="Base config yaml.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def load_cfg(config_path):
    """Load the project config lazily."""

    from omegaconf import OmegaConf

    return OmegaConf.load(config_path)


def load_model_for_target(target, nr_classes, checkpoint_path):
    """Instantiate and load the requested checkpoint family."""

    if target.family == "resnet50_fp32":
        model = ResNet50Classifier(nr_classes=nr_classes, dropout=0.5, pretrained=False)
    elif target.family == "resnet18_fp32":
        model = ResNet18Classifier(nr_classes=nr_classes, dropout=0.5, pretrained=False)
    elif target.family == "test_resnet_fp32":
        import timm

        model = timm.create_model(
            "test_resnet.r160_in1k",
            pretrained=False,
            num_classes=nr_classes,
        )
    elif target.family == "test_resnet_qat":
        from utils.quant_test_resnet import QuantTestResNet

        model = QuantTestResNet(
            nr_classes=nr_classes,
            weight_bit_width=target.weight_bits,
            act_bit_width=target.act_bits,
        )
    else:
        raise ValueError(f"Unsupported target family: {target.family}")

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model


def evaluate_target(cfg, target, checkpoint_path, batch_size=None, n_bootstrap=1000, savedir=None):
    """Evaluate one target using the shared ``test()`` pipeline."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_target(
        target=target,
        nr_classes=cfg.nr_classes,
        checkpoint_path=checkpoint_path,
    )
    test_loader = build_test_loader(
        cfg,
        target.eval_profile,
        batch_size=batch_size or cfg.batch_size,
    )
    return test(
        model=model,
        test_loader=test_loader,
        device=device,
        model_type=target.key,
        bootstrap=True,
        savedir=savedir,
        n_bootstrap=n_bootstrap,
    )


def build_report(target, checkpoint_path, report_path, metrics):
    """Assemble the persisted JSON report for one evaluated pipeline checkpoint."""

    return {
        "model": {
            "name": target.key,
            "family": target.family,
            "checkpoint": checkpoint_path,
            "report_path": report_path,
            "eval_profile": target.eval_profile,
            "weight_bits": target.weight_bits,
            "act_bits": target.act_bits,
        },
        "metrics": metrics,
    }


def evaluate_target_to_json(
    cfg,
    target,
    checkpoint_path,
    report_path,
    batch_size=None,
    n_bootstrap=1000,
):
    """Evaluate a target and write the normalized JSON report to disk."""

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    metrics = evaluate_target(
        cfg=cfg,
        target=target,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
        savedir=report_dir or None,
    )
    report = build_report(
        target=target,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        metrics=metrics,
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    """CLI entry point for evaluating any checkpoint in the canonical ladder."""

    args = parse_args()
    target = TARGETS[args.model]
    checkpoint_path = args.checkpoint or target.checkpoint
    report_path = args.report_path or target.report_path

    cfg = load_cfg(args.config)
    set_seeds(cfg.RANDOM_SEED)

    report = evaluate_target_to_json(
        cfg=cfg,
        target=target,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        batch_size=args.batch_size,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
