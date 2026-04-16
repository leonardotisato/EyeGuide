"""
FP32 training of CustomSmallNet for 4-class fundus classification.

Canonical archived branch:
- multiplier = 3
- strong augmentation
- weighted cross-entropy
- no KD

Run with:
    python src/train_custom_net.py
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
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.custom_net import CustomSmallNet
from utils.dataset import FundusClsDataset, prepare_dataframes
from utils.generals import progress_bar
from utils.seed import set_seeds
from utils.training import test
from utils.transforms_512_strong import test_transform_class, train_transform_class


LR = 1e-4
EPOCHS = 100
WEIGHT_DECAY = 1e-4
PATIENCE = 30
MULTIPLIER = 3


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
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
            batch_idx,
            len(train_loader),
            "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (running_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted") * 100.0
    return avg_loss, acc, f1


def validate(model, val_loader, criterion, device):
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
                batch_idx,
                len(val_loader),
                "Val Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (val_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )

    avg_loss = val_loss / len(val_loader)
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

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    train_df, val_df, test_df = prepare_dataframes(cfg)

    train_dataset = FundusClsDataset(
        train_df, train=True, transform=train_transform_class
    )
    val_dataset = FundusClsDataset(
        val_df, train=False, transform=test_transform_class
    )
    test_dataset = FundusClsDataset(
        test_df, train=False, transform=test_transform_class
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    label_counts = train_df["label"].value_counts().sort_index().values.astype(float)
    class_weights = 1.0 / label_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class counts (train): {label_counts.astype(int).tolist()}")
    print(f"Class weights: {class_weights.cpu().numpy().round(4).tolist()}")

    print(f"\n{'=' * 50}")
    print(f"CustomSmallNet canonical branch (multiplier={MULTIPLIER})")
    print(f"{'=' * 50}")
    model = CustomSmallNet(nr_classes=cfg.nr_classes, multiplier=MULTIPLIER)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    print(f"\nTraining: {EPOCHS} epochs, LR={LR}, patience={PATIENCE}")
    print("Loss: weighted cross-entropy | Augmentation: strong | KD: off")
    print(f"{'=' * 50}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = -1

    logname = os.path.join(cfg.results_dir, "train_custom_net_log.csv")
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

    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1, val_prec, val_rec = validate(
            model, val_loader, criterion, device
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
        print(
            f"\nRestored best model from epoch {best_epoch} "
            f"(val F1: {best_val_f1:.2f}%)"
        )
    model.to(device)

    ckpt_path = os.path.join(cfg.models_dir, f"custom_net_m{MULTIPLIER}_fp32.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}")

    print(f"\n{'=' * 50}")
    print("Evaluating on test set ...")
    print(f"{'=' * 50}")
    test_metrics = test(
        model=model,
        test_loader=test_loader,
        device=device,
        model_type=f"custom_net_m{MULTIPLIER}_fp32",
        bootstrap=True,
        savedir=cfg.results_dir,
    )
    print(f"Test metrics: {test_metrics}")

    report = {
        "multiplier": MULTIPLIER,
        "n_params": n_params,
        "epochs": best_epoch + 1,
        "best_val_f1": round(best_val_f1, 4),
        "checkpoint": ckpt_path,
        "class_weights": class_weights.cpu().numpy().round(6).tolist(),
        "input_size": [1, 3, 512, 512],
    }
    report_path = os.path.join(cfg.results_dir, "train_custom_net_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
