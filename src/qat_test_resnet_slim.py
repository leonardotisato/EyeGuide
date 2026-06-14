"""KD-QAT fine-tuning for the slim test_resnet experiment.

Default artifact:
  models/test_resnet_slim128x64_trim160_6w6a_qat.pth
"""

import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import torch
import torch.optim as optim
from brevitas.graph.calibrate import calibration_mode
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from qat_test_resnet_trim import (  # noqa: E402
    BN_FREEZE_EPOCH,
    CALIB_BATCHES,
    KD_ALPHA,
    KD_TEMPERATURE,
    PATIENCE,
    QAT_EPOCHS,
    QAT_LR,
    QAT_WEIGHT_DECAY,
    DualResTrimDataset,
    freeze_bn,
    qat_train_one_epoch,
    qat_validate,
    teacher_test_transform,
    teacher_train_transform,
)
from utils.dataset import FundusClsDataset, prepare_dataframes, trim_fundus_black_border  # noqa: E402
from utils.model import ResNet18Classifier  # noqa: E402
from utils.quant_test_resnet_slim import (  # noqa: E402
    QuantTestResNetSlim,
    load_test_resnet_slim_quant_weights,
    model_tag,
)
from utils.seed import set_seeds  # noqa: E402
from utils.test_resnet_slim import (  # noqa: E402
    DEFAULT_LAYER3_OUT,
    DEFAULT_LAYER4_OUT,
    slim_variant_tag,
)
from utils.training import test  # noqa: E402
from utils.transforms import make_strong_train_transform, make_test_transform  # noqa: E402


DEFAULT_STUDENT_RESOLUTION = 160
DEFAULT_QAT_ARTIFACT = "test_resnet_slim128x64_trim160_6w6a_qat.pth"


def resolve_slim_qat_run(
    student_resolution: int,
    weight_bits: int,
    act_bits: int,
    layer3_out: int = DEFAULT_LAYER3_OUT,
    layer4_out: int = DEFAULT_LAYER4_OUT,
):
    bit_tag = model_tag(weight_bits, act_bits)
    trim_tag = f"trim{student_resolution}"
    variant = slim_variant_tag(layer3_out, layer4_out)
    run_tag = f"{variant}_{trim_tag}_{bit_tag}"
    fp32_run_tag = f"{variant}_{trim_tag}"
    return {
        "bit_tag": bit_tag,
        "trim_tag": trim_tag,
        "variant": variant,
        "run_tag": run_tag,
        "fp32_checkpoint_name": f"test_resnet_{variant}_fp32_kd_{trim_tag}_ft.pth",
        "results_dir_name": f"qat_test_resnet_{run_tag}",
        "checkpoint_name": f"test_resnet_{run_tag}_qat.pth",
        "report_name": f"qat_test_resnet_{run_tag}_report.json",
        "log_name": f"qat_test_resnet_{run_tag}_log.csv",
        "model_type": f"test_resnet_{run_tag}_qat",
        "fp32_run_tag": fp32_run_tag,
    }


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weight_bits = int(OmegaConf.select(cfg, "weight_bits", default=6))
    act_bits = int(OmegaConf.select(cfg, "act_bits", default=6))
    student_resolution = int(
        OmegaConf.select(cfg, "student_resolution", default=DEFAULT_STUDENT_RESOLUTION)
    )
    layer3_out = int(OmegaConf.select(cfg, "slim_layer3_out", default=DEFAULT_LAYER3_OUT))
    layer4_out = int(OmegaConf.select(cfg, "slim_layer4_out", default=DEFAULT_LAYER4_OUT))
    student_test_transform = make_test_transform(student_resolution)
    student_train_transform = make_strong_train_transform(student_resolution)

    run_cfg = resolve_slim_qat_run(
        student_resolution=student_resolution,
        weight_bits=weight_bits,
        act_bits=act_bits,
        layer3_out=layer3_out,
        layer4_out=layer4_out,
    )

    results_dir = os.path.join(cfg.results_dir, run_cfg["results_dir_name"])
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    train_df, val_df, test_df = prepare_dataframes(cfg)

    train_dataset = DualResTrimDataset(
        train_df,
        student_transform=student_train_transform,
        teacher_transform=teacher_train_transform,
    )
    val_dataset = DualResTrimDataset(
        val_df,
        student_transform=student_test_transform,
        teacher_transform=teacher_test_transform,
    )
    test_dataset = FundusClsDataset(
        test_df,
        train=False,
        transform=student_test_transform,
        preprocess=trim_fundus_black_border,
    )
    calib_dataset = FundusClsDataset(
        train_df,
        train=False,
        transform=student_test_transform,
        preprocess=trim_fundus_black_border,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    calib_loader = DataLoader(
        calib_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    teacher_path = os.path.join(cfg.models_dir, "resnet18_from_resnet50_fp32_kd.pth")
    if not os.path.exists(teacher_path):
        print(f"[ERROR] Teacher checkpoint not found: {teacher_path}")
        return

    teacher = ResNet18Classifier(nr_classes=cfg.nr_classes, pretrained=False)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print("\n" + "=" * 50)
    print(f"Creating QuantTestResNetSlim [{run_cfg['bit_tag']}] for {run_cfg['run_tag']}")
    print("=" * 50)
    student = QuantTestResNetSlim(
        nr_classes=cfg.nr_classes,
        weight_bit_width=weight_bits,
        act_bit_width=act_bits,
        layer3_out=layer3_out,
        layer4_out=layer4_out,
    )
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {n_params:,}")

    student_init_checkpoint = OmegaConf.select(cfg, "warm_start_checkpoint", default=None)
    if student_init_checkpoint is None:
        student_init_checkpoint = os.path.join(cfg.models_dir, run_cfg["fp32_checkpoint_name"])
    if not os.path.exists(student_init_checkpoint):
        print(f"[ERROR] Student init checkpoint not found: {student_init_checkpoint}")
        print("Run train_test_resnet_slim first.")
        return

    print(f"\nLoading slim student init weights from: {student_init_checkpoint}")
    missing, unexpected = load_test_resnet_slim_quant_weights(student, student_init_checkpoint)
    non_quant_missing = [
        key for key in missing
        if not any(token in key for token in [
            "tensor_quant", "scaling_impl", "int_scaling_impl", "zero_point",
            "msb_clamp_bit_width_impl", "act_quant", "weight_quant", "bias_quant",
        ])
    ]
    if non_quant_missing:
        print(f"[WARNING] Non-quantizer keys missing: {non_quant_missing}")
    if unexpected:
        print(f"[WARNING] Unexpected keys found: {unexpected}")
    else:
        print("Weight loading OK: only Brevitas quantizer params are missing.")

    student.to(device)

    print("\n" + "=" * 50)
    print(f"Calibrating quantizer scales ({CALIB_BATCHES} batches) ...")
    print("=" * 50)
    student.eval()
    with calibration_mode(student):
        for batch_idx, (inputs, _) in enumerate(calib_loader):
            if batch_idx >= CALIB_BATCHES:
                break
            with torch.no_grad():
                student(inputs.to(device))
            if (batch_idx + 1) % 25 == 0:
                print(f"  Calibration batch {batch_idx + 1}/{CALIB_BATCHES}")
    print("Calibration complete.")

    print("\n" + "=" * 50)
    print(
        "QAT fine-tuning: "
        f"{QAT_EPOCHS} epochs, LR={QAT_LR}, patience={PATIENCE}, BN freeze={BN_FREEZE_EPOCH}"
    )
    print(
        f"Configuration: slim {run_cfg['variant']}, trim black border -> "
        f"{student_resolution} student"
    )
    print("=" * 50)

    optimizer = optim.Adam(student.parameters(), lr=QAT_LR, weight_decay=QAT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=QAT_EPOCHS)

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = -1

    logname = os.path.join(results_dir, run_cfg["log_name"])
    with open(logname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "train_f1", "val_loss",
            "val_acc", "val_f1", "val_prec", "val_rec", "best_epoch",
        ])

    for epoch in tqdm(range(QAT_EPOCHS), desc=f"KD-QAT {run_cfg['run_tag']}"):
        train_loss, train_acc, train_f1 = qat_train_one_epoch(
            student, teacher, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc, val_f1, val_prec, val_rec = qat_validate(
            student, teacher, val_loader, KD_TEMPERATURE, KD_ALPHA, device
        )
        scheduler.step()

        if epoch >= BN_FREEZE_EPOCH:
            freeze_bn(student)

        print(
            f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.2f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.2f} val_acc={val_acc:.2f}"
        )

        with open(logname, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, train_acc, train_f1, val_loss, val_acc,
                val_f1, val_prec, val_rec, best_epoch,
            ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            print(f"  -> New best val loss: {val_loss:.4f} (val F1: {val_f1:.2f}%, epoch {epoch})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    if best_state is not None:
        student.load_state_dict(best_state)
        print(
            f"\nRestored best model from epoch {best_epoch} "
            f"(val loss: {best_val_loss:.4f}, val F1: {best_val_f1:.2f}%)"
        )
    student.to(device)

    qat_ckpt_path = os.path.join(cfg.models_dir, run_cfg["checkpoint_name"])
    torch.save(student.state_dict(), qat_ckpt_path)
    print(f"QAT checkpoint saved -> {qat_ckpt_path}")

    print("\n" + "=" * 50)
    print("Evaluating on test set ...")
    print("=" * 50)
    test_metrics = test(
        model=student,
        test_loader=test_loader,
        device=device,
        model_type=run_cfg["model_type"],
        bootstrap=True,
        savedir=results_dir,
    )
    print(f"QAT test metrics: {test_metrics}")

    report = {
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        "variant": run_cfg["variant"],
        "layer3_out": layer3_out,
        "layer4_out": layer4_out,
        "n_params": n_params,
        "epochs": best_epoch + 1,
        "best_val_f1": round(best_val_f1, 4),
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": qat_ckpt_path,
        "student_init_checkpoint": student_init_checkpoint,
        "teacher": "resnet18_from_resnet50_fp32_kd.pth (512x512 full-image strong train / test eval)",
        "student_resolution": student_resolution,
        "teacher_resolution": 512,
        "input_size": [1, 3, student_resolution, student_resolution],
        "kd_temperature": KD_TEMPERATURE,
        "kd_alpha": KD_ALPHA,
        "test_metrics": test_metrics,
    }
    report_path = os.path.join(results_dir, run_cfg["report_name"])
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
