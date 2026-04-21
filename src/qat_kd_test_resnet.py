"""
QAT fine-tuning for test_resnet with Knowledge Distillation.

Teacher: ResNet18 FP32 (models/resnet18_fp32_kd.pth, 512x512 input).
Student: QuantTestResNet at configurable bit width (224x224 input).

This mirrors the exact KD objective used in FP32 fine-tuning (train_test_resnet.py):
  ResNet18 teacher at 512x512, test_resnet student at 224x224, T=4.0, alpha=0.5.
  DualResDataset provides separate teacher/student crops from the same image.

Checkpoint saved as models/test_resnet_{tag}_kd_qat.pth.

Run with:
    python src/qat_kd_test_resnet.py                    # 8w8a
    python src/qat_kd_test_resnet.py ++weight_bits=4 ++act_bits=4  # 4w4a
"""

import os
import sys
import json
import csv
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from brevitas.graph.calibrate import calibration_mode

sys.path.insert(0, os.path.dirname(__file__))
from utils.seed import set_seeds
from utils.quant_test_resnet import QuantTestResNet, load_fp32_weights, model_tag
from utils.dataset import FundusClsDataset, prepare_dataframes
# from utils.model import ResNet18Classifier
from utils.model import ResNet50Classifier
from utils.transforms_224_strong import (
    test_transform_class as student_test_transform,
    train_transform_class as student_train_transform,
)
from utils.transforms_512_strong import train_transform_class as teacher_train_transform
from utils.training import test
from utils.generals import progress_bar
from train_test_resnet import DualResDataset


# ─── KD hyperparameters (match FP32 training) ───────────────────────────────
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.5          # alpha * CE + (1-alpha) * KL

# ─── QAT hyperparameters ────────────────────────────────────────────────────
# Bit widths: override via Hydra: ++weight_bits=4 ++act_bits=4
QAT_LR = 1e-5
QAT_EPOCHS = 120
QAT_WEIGHT_DECAY = 1e-4
CALIB_BATCHES = 100
BN_FREEZE_EPOCH = 5
PATIENCE = 20


def kd_loss(student_logits, teacher_logits, labels, temperature, alpha):
    ce = F.cross_entropy(student_logits, labels)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
    return alpha * ce + (1 - alpha) * kl


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


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
            batch_idx, len(train_loader),
            'Train Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                running_loss / (batch_idx + 1),
                100.0 * correct / total, correct, total
            ),
        )

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100.0
    return avg_loss, acc, f1


def qat_validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            progress_bar(
                batch_idx, len(val_loader),
                'Val Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    val_loss / (batch_idx + 1),
                    100.0 * correct / total, correct, total
                ),
            )

    avg_loss = val_loss / len(val_loader)
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100.0
    prec = precision_score(all_labels, all_preds, average='weighted') * 100.0
    rec = recall_score(all_labels, all_preds, average='weighted') * 100.0
    return avg_loss, acc, f1, prec, rec


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    WEIGHT_BITS = int(OmegaConf.select(cfg, "weight_bits", default=8))
    ACT_BITS = int(OmegaConf.select(cfg, "act_bits", default=8))

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df = prepare_dataframes(cfg)

    # Training: DualResDataset returns (student_224, teacher_512, label)
    train_dataset = DualResDataset(
        train_df,
        student_transform=student_train_transform,
        teacher_transform=teacher_train_transform,
    )
    val_dataset = FundusClsDataset(val_df, train=False, transform=student_test_transform)
    test_dataset = FundusClsDataset(test_df, train=False, transform=student_test_transform)
    calib_dataset = FundusClsDataset(train_df, train=False, transform=student_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    calib_loader = DataLoader(calib_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Teacher (ResNet50 FP32 self-KD, 512x512, frozen) ─────────────────
    # --- old ResNet18 teacher (kept for reference / rollback) ---
    # teacher_path = os.path.join(cfg.models_dir, "resnet18_fp32_kd.pth")
    teacher_path = os.path.join(cfg.models_dir, "resnet50_fp32_kd.pth")
    if not os.path.exists(teacher_path):
        print(f"[ERROR] Teacher checkpoint not found: {teacher_path}")
        print("Expected: resnet50_fp32_kd.pth (ResNet50 self-KD teacher from main.py).")
        return

    print(f"Loading ResNet50 teacher from: {teacher_path}")
    # --- old ResNet18 teacher instantiation (kept for reference / rollback) ---
    # teacher = ResNet18Classifier(nr_classes=cfg.nr_classes, pretrained=False)
    teacher = ResNet50Classifier(nr_classes=cfg.nr_classes, pretrained=False)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("Teacher loaded and frozen (512x512).")

    # ── Student (QuantTestResNet) ─────────────────────────────────────────
    tag = model_tag(WEIGHT_BITS, ACT_BITS)
    print("\n" + "=" * 50)
    print(f"Creating QuantTestResNet [{tag}] with KD")
    print(f"Teacher: ResNet18 FP32 (512x512) | T={KD_TEMPERATURE}, alpha={KD_ALPHA}")
    print("=" * 50)
    model = QuantTestResNet(
        nr_classes=cfg.nr_classes,
        weight_bit_width=WEIGHT_BITS,
        act_bit_width=ACT_BITS,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Load FP32 weights into student ───────────────────────────────────
    ckpt_path = os.path.join(cfg.models_dir, "test_resnet_fp32_kd.pth")
    print(f"\nLoading FP32 weights from: {ckpt_path}")
    missing, unexpected = load_fp32_weights(model, ckpt_path)

    non_quant_missing = [k for k in missing if not any(
        tag_str in k for tag_str in [
            "tensor_quant", "scaling_impl", "int_scaling_impl",
            "zero_point", "msb_clamp_bit_width_impl",
            "act_quant", "weight_quant", "bias_quant",
        ]
    )]
    if non_quant_missing:
        print(f"\n[WARNING] Non-quantizer keys missing: {non_quant_missing}")
    if unexpected:
        print(f"\n[WARNING] Unexpected keys found — weight mapping may be wrong!")
    else:
        print("Weight loading OK — only Brevitas quantizer params missing.")

    model.to(device)

    # ── Calibration ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Calibrating quantizer scales ({CALIB_BATCHES} batches) ...")
    print("=" * 50)
    model.eval()
    with calibration_mode(model):
        for i, (imgs, _) in enumerate(calib_loader):
            if i >= CALIB_BATCHES:
                break
            with torch.no_grad():
                model(imgs.to(device))
            if (i + 1) % 25 == 0:
                print(f"  Calibration batch {i + 1}/{CALIB_BATCHES}")
    print("Calibration complete.")

    # ── QAT Training ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"QAT+KD fine-tuning: {QAT_EPOCHS} epochs, LR={QAT_LR}")
    print(f"BN freeze after epoch {BN_FREEZE_EPOCH}, patience={PATIENCE}")
    print("=" * 50)

    criterion = nn.CrossEntropyLoss()  # used for validation loss only
    optimizer = optim.Adam(model.parameters(), lr=QAT_LR, weight_decay=QAT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=QAT_EPOCHS)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = -1

    logname = os.path.join(cfg.results_dir, f"qat_kd_test_resnet_{tag}_log.csv")
    with open(logname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "train_f1",
            "val_loss", "val_acc", "val_f1", "val_prec", "val_rec", "best_epoch",
        ])

    for epoch in tqdm(range(QAT_EPOCHS), desc="QAT+KD Epochs"):
        train_loss, train_acc, train_f1 = qat_train_one_epoch(
            model, teacher, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc, val_f1, val_prec, val_rec = qat_validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(
            f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.2f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.2f} val_acc={val_acc:.2f}"
        )

        with open(logname, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, train_acc, train_f1,
                val_loss, val_acc, val_f1, val_prec, val_rec, best_epoch,
            ])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            print(f"  -> New best val F1: {val_f1:.2f}% (epoch {epoch})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model from epoch {best_epoch} (val F1: {best_val_f1:.2f}%)")
    model.to(device)

    # ── Save QAT+KD checkpoint ───────────────────────────────────────────
    qat_ckpt_path = os.path.join(cfg.models_dir, f"test_resnet_{tag}_kd_qat.pth")
    torch.save(model.state_dict(), qat_ckpt_path)
    print(f"QAT+KD checkpoint saved -> {qat_ckpt_path}")

    # ── Test evaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Evaluating QAT+KD model on test set ...")
    print("=" * 50)
    test_metrics = test(
        model=model,
        test_loader=test_loader,
        device=device,
        model_type=f"test_resnet_{tag}_kd_qat",
        bootstrap=True,
        savedir=cfg.results_dir,
    )
    print(f"QAT+KD test metrics: {test_metrics}")

    # ── Summary report ───────────────────────────────────────────────────
    report = {
        "weight_bits": WEIGHT_BITS,
        "act_bits": ACT_BITS,
        "n_params": n_params,
        "qat_epochs": best_epoch + 1,
        "best_val_f1": round(best_val_f1, 4),
        "checkpoint": qat_ckpt_path,
        "fp32_checkpoint": ckpt_path,
        "teacher": "resnet18_fp32_kd.pth (512x512)",
        "kd_temperature": KD_TEMPERATURE,
        "kd_alpha": KD_ALPHA,
        "input_size": [1, 3, 224, 224],
    }
    report_path = os.path.join(cfg.results_dir, f"qat_kd_test_resnet_{tag}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {report_path}")
    print("\nNext step: run export_test_resnet.py with matching bit widths.")


if __name__ == "__main__":
    main()
