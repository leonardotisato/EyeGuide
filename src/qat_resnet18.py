"""
QAT fine-tuning for ResNet18 4w4a: load KD weights, calibrate, train, evaluate, save checkpoint.

Export is handled separately by export_resnet18.py.

Run with:
    bash run_qat_resnet18.sh
"""

import os
import sys
import json
import csv
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from brevitas.graph.calibrate import calibration_mode

sys.path.insert(0, os.path.dirname(__file__))
from utils.seed import set_seeds
from utils.quant_resnet18 import QuantResNet18, load_kd_weights, model_tag
from utils.dataset import FundusClsDataset, prepare_dataframes
from utils.transforms import test_transform_class, train_transform_class
from utils.training import test
from utils.generals import progress_bar


WEIGHT_BITS = 4
ACT_BITS = 4

# ─── QAT hyperparameters ────────────────────────────────────────────────────
QAT_LR = 1e-4
QAT_EPOCHS = 60
QAT_WEIGHT_DECAY = 1e-4
CALIB_BATCHES = 100
BN_FREEZE_EPOCH = 5       # freeze BN running stats after this epoch
PATIENCE = 20              # early stopping on val F1


def freeze_bn(model):
    """Set all BatchNorm layers to eval mode (freeze running stats)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


def qat_train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    if epoch >= BN_FREEZE_EPOCH:
        freeze_bn(model)

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
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

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df = prepare_dataframes(cfg)

    train_dataset = FundusClsDataset(train_df, train=True, transform=train_transform_class)
    val_dataset = FundusClsDataset(val_df, train=False, transform=test_transform_class)
    test_dataset = FundusClsDataset(test_df, train=False, transform=test_transform_class)
    calib_dataset = FundusClsDataset(train_df, train=False, transform=test_transform_class)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    calib_loader = DataLoader(calib_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    tag = model_tag(WEIGHT_BITS, ACT_BITS)
    print("\n" + "=" * 50)
    print(f"Creating QuantResNet18 [{tag}] ...")
    print("=" * 50)
    model = QuantResNet18(
        nr_classes=cfg.nr_classes,
        weight_bit_width=WEIGHT_BITS,
        act_bit_width=ACT_BITS,
    )
    print(f"QuantResNet18 instantiated: {WEIGHT_BITS}w{ACT_BITS}a")

    # ── Load KD weights ──────────────────────────────────────────────────
    ckpt_path = os.path.join(cfg.models_dir, "resnet18_fp32_kd.pth")
    print(f"\nLoading KD weights from: {ckpt_path}")
    missing, unexpected = load_kd_weights(model, ckpt_path)

    non_quant_missing = [k for k in missing if not any(
        tag in k for tag in [
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
    print(f"QAT fine-tuning: {QAT_EPOCHS} epochs, LR={QAT_LR}")
    print(f"BN freeze after epoch {BN_FREEZE_EPOCH}, patience={PATIENCE}")
    print("=" * 50)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=QAT_LR, weight_decay=QAT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=QAT_EPOCHS)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = -1

    logname = os.path.join(cfg.results_dir, "qat_resnet18_log.csv")
    with open(logname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "train_f1",
            "val_loss", "val_acc", "val_f1", "val_prec", "val_rec", "best_epoch",
        ])

    for epoch in tqdm(range(QAT_EPOCHS), desc="QAT Epochs"):
        train_loss, train_acc, train_f1 = qat_train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
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

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model from epoch {best_epoch} (val F1: {best_val_f1:.2f}%)")
    model.to(device)

    # ── Save QAT checkpoint ──────────────────────────────────────────────
    qat_ckpt_path = os.path.join(cfg.models_dir, f"resnet18_{tag}_qat.pth")
    torch.save(model.state_dict(), qat_ckpt_path)
    print(f"QAT checkpoint saved -> {qat_ckpt_path}")

    # ── Test evaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Evaluating QAT model on test set ...")
    print("=" * 50)
    test_metrics = test(
        model=model,
        test_loader=test_loader,
        device=device,
        model_type="student_kd_qat",
        bootstrap=True,
        savedir=cfg.results_dir,
    )
    print(f"QAT test metrics: {test_metrics}")

    # ── Summary report ───────────────────────────────────────────────────
    report = {
        "qat_epochs": best_epoch + 1,
        "best_val_f1": round(best_val_f1, 4),
        "checkpoint": qat_ckpt_path,
        "input_size": [1, 3, 512, 512],
    }
    report_path = os.path.join(cfg.results_dir, "qat_resnet18_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {report_path}")
    print("\nNext step: run export_resnet18.py to export QONNX.")


if __name__ == "__main__":
    main()
