"""
Canonical FP32 fine-tuning of MobileNetV1 with Knowledge Distillation.

Uses timm mobilenetv1_100 (ImageNet pretrained) as student backbone.
Teacher: ResNet18 FP32 KD checkpoint (resnet18_fp32_kd.pth, 87.2% acc).

Saves FP32 checkpoint in timm key format -> mobilenetv1_fp32_kd.pth.
This checkpoint is then loaded by qat_mobilenetv1.py for QAT fine-tuning.

Run with:
    python src/train_mobilenetv1.py
"""

import os
import sys
import json
import csv
import warnings

warnings.filterwarnings("ignore")

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(__file__))
from utils.seed import set_seeds
from utils.model import ResNet18Classifier
from utils.dataset import FundusClsDataset, prepare_dataframes
from utils.transforms_512_light import test_transform_class, train_transform_class
from utils.training import test
from utils.generals import progress_bar


# ─── Hyperparameters ────────────────────────────────────────────────────────
LR = 1e-4
EPOCHS = 60
WEIGHT_DECAY = 1e-4
PATIENCE = 20
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.5          # 0.5 * CE + 0.5 * KL  (same balance used in main.py)


def kd_loss(student_logits, teacher_logits, labels, temperature, alpha):
    ce = F.cross_entropy(student_logits, labels)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
    return alpha * ce + (1 - alpha) * kl


def train_one_epoch(student, teacher, train_loader, optimizer, device, epoch):
    student.train()
    teacher.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        student_out = student(inputs)
        with torch.no_grad():
            teacher_out = teacher(inputs)

        loss = kd_loss(student_out, teacher_out, labels, KD_TEMPERATURE, KD_ALPHA)
        loss.backward()
        clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(student_out, 1)
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


def validate(student, val_loader, device):
    student.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = student(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            progress_bar(
                batch_idx, len(val_loader),
                'Val Acc: %.3f%% (%d/%d)' % (100.0 * correct / total, correct, total),
            )

    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100.0
    return acc, f1


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df = prepare_dataframes(cfg)

    train_dataset = FundusClsDataset(train_df, train=True, transform=train_transform_class)
    val_dataset = FundusClsDataset(val_df, train=False, transform=test_transform_class)
    test_dataset = FundusClsDataset(test_df, train=False, transform=test_transform_class)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Teacher ───────────────────────────────────────────────────────────
    teacher_path = os.path.join(cfg.models_dir, "resnet18_fp32_kd.pth")
    print(f"\nLoading teacher from: {teacher_path}")
    teacher = ResNet18Classifier(nr_classes=cfg.nr_classes, pretrained=False)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("Teacher loaded and frozen.")

    # ── Student ───────────────────────────────────────────────────────────
    print("\nLoading timm MobileNetV1 pretrained student ...")
    student = timm.create_model("mobilenetv1_100", pretrained=True, num_classes=cfg.nr_classes)
    student.to(device)
    print("Student loaded.")

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"FP32 KD fine-tuning: {EPOCHS} epochs, LR={LR}")
    print(f"KD temperature={KD_TEMPERATURE}, alpha={KD_ALPHA}")
    print(f"{'=' * 50}")

    optimizer = optim.Adam(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = -1

    logname = os.path.join(cfg.results_dir, "train_mobilenetv1_log.csv")
    with open(logname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_f1",
                         "val_acc", "val_f1", "best_epoch"])

    for epoch in tqdm(range(EPOCHS), desc="FP32 KD Epochs"):
        train_loss, train_acc, train_f1 = train_one_epoch(
            student, teacher, train_loader, optimizer, device, epoch
        )
        val_acc, val_f1 = validate(student, val_loader, device)
        scheduler.step()

        print(
            f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.2f} | "
            f"val_f1={val_f1:.2f} val_acc={val_acc:.2f}"
        )

        with open(logname, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, train_f1,
                             val_acc, val_f1, best_epoch])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            print(f"  -> New best val F1: {val_f1:.2f}% (epoch {epoch})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # Restore best model
    student.load_state_dict(best_state)
    print(f"\nRestored best model from epoch {best_epoch} (val F1: {best_val_f1:.2f}%)")
    student.to(device)

    # ── Save ─────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(cfg.models_dir, "mobilenetv1_fp32_kd.pth")
    torch.save(student.state_dict(), ckpt_path)
    print(f"FP32 KD checkpoint saved -> {ckpt_path}")

    # ── Test ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print("Evaluating on test set ...")
    print(f"{'=' * 50}")
    test_metrics = test(
        model=student,
        test_loader=test_loader,
        device=device,
        model_type="mobilenetv1_fp32_kd",
        bootstrap=True,
        savedir=cfg.results_dir,
    )
    print(f"Test metrics: {test_metrics}")

    report = {
        "epochs": best_epoch + 1,
        "best_val_f1": round(best_val_f1, 4),
        "checkpoint": ckpt_path,
        "kd_temperature": KD_TEMPERATURE,
        "kd_alpha": KD_ALPHA,
    }
    report_path = os.path.join(cfg.results_dir, "train_mobilenetv1_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {report_path}")
    print("\nNext step: run qat_mobilenetv1.py to QAT fine-tune.")


if __name__ == "__main__":
    main()
