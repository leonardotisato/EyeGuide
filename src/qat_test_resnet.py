"""
Canonical KD-QAT fine-tuning for test_resnet.

This script mirrors the active FP32 baseline as closely as possible:
- teacher: upgraded ResNet18 KD teacher (`resnet18_from_resnet50_fp32_kd.pth`)
- student init: canonical FP32 `test_resnet` checkpoint (`test_resnet_fp32_kd.pth`)
- teacher train domain: full-image 512, strong train transform / clean eval transform
- student train domain: 224, strong train transform / clean eval transform
- train loss: alpha * CE + (1 - alpha) * KL
- validation loss: same composite KD objective
- checkpoint selection: best val_loss

The only planned experiment variable is quantization bit width.

Run with:
    bash run.sh qat_test_resnet
    bash run.sh qat_test_resnet ++weight_bits=6 ++act_bits=6
    bash run.sh qat_test_resnet ++weight_bits=4 ++act_bits=4
"""

import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from brevitas.graph.calibrate import calibration_mode
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from train_test_resnet import DualResDataset
from utils.dataset import FundusClsDataset, prepare_dataframes
from utils.generals import progress_bar
from utils.model import ResNet18Classifier
from utils.quant_test_resnet import QuantTestResNet, load_fp32_weights, model_tag
from utils.seed import set_seeds
from utils.training import test
from utils.transforms_224_light import (
    test_transform_class as student_test_transform,
)
from utils.transforms_224_strong import (
    train_transform_class as student_train_transform,
)
from utils.transforms_512_strong import (
    test_transform_class as teacher_test_transform,
    train_transform_class as teacher_train_transform,
)


KD_TEMPERATURE = 3.0
KD_ALPHA = 0.25

QAT_LR = 1e-5
QAT_EPOCHS = 200
QAT_WEIGHT_DECAY = 1e-4
CALIB_BATCHES = 100
BN_FREEZE_EPOCH = 5
PATIENCE = 50


def kd_loss(student_logits, teacher_logits, labels, temperature, alpha):
    ce = F.cross_entropy(student_logits, labels)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature**2)
    return alpha * ce + (1 - alpha) * kl


def freeze_bn(model):
    """Set all BatchNorm layers to eval mode to freeze running stats during QAT."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()


def qat_train_one_epoch(student, teacher, train_loader, optimizer, device, epoch):
    student.train()
    teacher.eval()
    if epoch >= BN_FREEZE_EPOCH:
        freeze_bn(student)

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for batch_idx, (inputs_s, inputs_t, labels) in enumerate(train_loader):
        inputs_s = inputs_s.to(device)
        inputs_t = inputs_t.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        student_out = student(inputs_s)
        with torch.no_grad():
            teacher_out = teacher(inputs_t)

        loss = kd_loss(student_out, teacher_out, labels, KD_TEMPERATURE, KD_ALPHA)
        loss.backward()
        clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(student_out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        progress_bar(
            batch_idx,
            len(train_loader),
            "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (running_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted") * 100.0
    return avg_loss, acc, f1


def qat_validate(student, teacher, val_loader, temperature, alpha, device):
    student.eval()
    teacher.eval()
    val_loss_sum = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (inputs_s, inputs_t, labels) in enumerate(val_loader):
            inputs_s = inputs_s.to(device)
            inputs_t = inputs_t.to(device)
            labels = labels.to(device)

            student_logits = student(inputs_s)
            teacher_logits = teacher(inputs_t)
            loss = kd_loss(student_logits, teacher_logits, labels, temperature, alpha)
            val_loss_sum += loss.item()

            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            progress_bar(
                batch_idx,
                len(val_loader),
                "Val Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (val_loss_sum / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )

    avg_loss = val_loss_sum / len(val_loader)
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted") * 100.0
    prec = precision_score(all_labels, all_preds, average="weighted") * 100.0
    rec = recall_score(all_labels, all_preds, average="weighted") * 100.0
    return avg_loss, acc, f1, prec, rec


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weight_bits = int(OmegaConf.select(cfg, "weight_bits", default=8))
    act_bits = int(OmegaConf.select(cfg, "act_bits", default=8))
    tag = model_tag(weight_bits, act_bits)

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    train_df, val_df, test_df = prepare_dataframes(cfg)

    train_dataset = DualResDataset(
        train_df,
        student_transform=student_train_transform,
        teacher_transform=teacher_train_transform,
    )
    val_dataset = DualResDataset(
        val_df,
        student_transform=student_test_transform,
        teacher_transform=teacher_test_transform,
    )
    test_dataset = FundusClsDataset(test_df, train=False, transform=student_test_transform)
    calib_dataset = FundusClsDataset(train_df, train=False, transform=student_test_transform)

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

    print(f"Loading teacher from: {teacher_path}")
    teacher = ResNet18Classifier(nr_classes=cfg.nr_classes, pretrained=False)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("Teacher loaded and frozen (512x512 full-image, strong-train/test-eval domain).")

    print("\n" + "=" * 50)
    print(f"Creating QuantTestResNet [{tag}]")
    print("=" * 50)
    student = QuantTestResNet(
        nr_classes=cfg.nr_classes,
        weight_bit_width=weight_bits,
        act_bit_width=act_bits,
    )
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {n_params:,}")

    fp32_ckpt_path = os.path.join(cfg.models_dir, "test_resnet_fp32_kd.pth")
    if not os.path.exists(fp32_ckpt_path):
        print(f"[ERROR] FP32 checkpoint not found: {fp32_ckpt_path}")
        return

    print(f"\nLoading FP32 student weights from: {fp32_ckpt_path}")
    missing, unexpected = load_fp32_weights(student, fp32_ckpt_path)
    non_quant_missing = [
        key
        for key in missing
        if not any(
            token in key
            for token in [
                "tensor_quant",
                "scaling_impl",
                "int_scaling_impl",
                "zero_point",
                "msb_clamp_bit_width_impl",
                "act_quant",
                "weight_quant",
                "bias_quant",
            ]
        )
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
        "Configuration: KD-QAT + strong student aug (224) + upgraded ResNet18 teacher "
        "on 512 strong/eval-test transforms"
    )
    print(f"KD: temperature={KD_TEMPERATURE}, alpha={KD_ALPHA}")
    print("=" * 50)

    optimizer = optim.Adam(student.parameters(), lr=QAT_LR, weight_decay=QAT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=QAT_EPOCHS)

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = -1

    logname = os.path.join(cfg.results_dir, f"qat_test_resnet_{tag}_log.csv")
    with open(logname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "train_f1",
                "val_loss",
                "val_acc",
                "val_f1",
                "val_prec",
                "val_rec",
                "best_epoch",
            ]
        )

    for epoch in tqdm(range(QAT_EPOCHS), desc="KD-QAT Epochs"):
        train_loss, train_acc, train_f1 = qat_train_one_epoch(
            student, teacher, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc, val_f1, val_prec, val_rec = qat_validate(
            student, teacher, val_loader, KD_TEMPERATURE, KD_ALPHA, device
        )
        scheduler.step()

        print(
            f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.2f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.2f} val_acc={val_acc:.2f}"
        )

        with open(logname, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_acc,
                    train_f1,
                    val_loss,
                    val_acc,
                    val_f1,
                    val_prec,
                    val_rec,
                    best_epoch,
                ]
            )

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

    qat_ckpt_path = os.path.join(cfg.models_dir, f"test_resnet_{tag}_qat.pth")
    torch.save(student.state_dict(), qat_ckpt_path)
    print(f"QAT checkpoint saved -> {qat_ckpt_path}")

    print("\n" + "=" * 50)
    print("Evaluating on test set ...")
    print("=" * 50)
    test_metrics = test(
        model=student,
        test_loader=test_loader,
        device=device,
        model_type=f"test_resnet_{tag}_qat",
        bootstrap=True,
        savedir=cfg.results_dir,
    )
    print(f"QAT test metrics: {test_metrics}")

    report = {
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        "n_params": n_params,
        "epochs": best_epoch + 1,
        "best_val_f1": round(best_val_f1, 4),
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": qat_ckpt_path,
        "fp32_checkpoint": fp32_ckpt_path,
        "teacher": "resnet18_from_resnet50_fp32_kd.pth (512x512 full-image strong train / test eval)",
        "student_resolution": 224,
        "teacher_resolution": 512,
        "input_size": [1, 3, 224, 224],
        "kd_temperature": KD_TEMPERATURE,
        "kd_alpha": KD_ALPHA,
    }
    report_path = os.path.join(cfg.results_dir, f"qat_test_resnet_{tag}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
