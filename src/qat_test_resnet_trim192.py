"""
KD-QAT fine-tuning for the trim->192 test_resnet experiment.

This keeps the canonical teacher domain unchanged:
- teacher: upgraded ResNet18 KD teacher (`resnet18_from_resnet50_fp32_kd.pth`)
- teacher domain: full-image 512, strong-train / clean-eval

But changes the student path to the experimental trim-192 domain:
- student train domain: trim black border -> 192, strong train transform
- student eval domain: trim black border -> 192, clean eval transform
- student init: canonical trim-192 FP32 checkpoint (ImageNet-init)

Typical runs:
    python src/qat_test_resnet_trim192.py ++weight_bits=8 ++act_bits=8
    python src/qat_test_resnet_trim192.py ++weight_bits=6 ++act_bits=6
"""

import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from brevitas.graph.calibrate import calibration_mode
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tv_transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset import FundusClsDataset, prepare_dataframes, safe_pil_read, trim_fundus_black_border
from utils.generals import progress_bar
from utils.model import ResNet18Classifier
from utils.quant_test_resnet import QuantTestResNet, load_test_resnet_weights, model_tag
from utils.seed import set_seeds
from utils.training import test
from utils.transforms import make_strong_train_transform, make_test_transform

student_test_transform = make_test_transform(192)
student_train_transform = make_strong_train_transform(192)
teacher_test_transform = make_test_transform(512)
teacher_train_transform = make_strong_train_transform(512)


KD_TEMPERATURE = 3.0
KD_ALPHA = 0.25

QAT_LR = 1e-5
QAT_EPOCHS = 200
QAT_WEIGHT_DECAY = 1e-4
CALIB_BATCHES = 100
BN_FREEZE_EPOCH = 5
PATIENCE = 50
TRIM192_FP32_WARM_START = "test_resnet_fp32_kd_trim192_ft.pth"


def resolve_trim192_qat_run(weight_bits: int, act_bits: int):
    """Resolve stable naming for one canonical trim-192 QAT run."""

    bit_tag = f"{weight_bits}w{act_bits}a"
    run_tag = f"trim192_{bit_tag}"
    return {
        "bit_tag": bit_tag,
        "run_tag": run_tag,
        "results_dir_name": f"qat_test_resnet_{run_tag}",
        "checkpoint_name": f"test_resnet_{run_tag}_qat.pth",
        "report_name": f"qat_test_resnet_{run_tag}_report.json",
        "log_name": f"qat_test_resnet_{run_tag}_log.csv",
        "model_type": f"test_resnet_{run_tag}_qat",
    }


class DualResTrim192Dataset(Dataset):
    """Return (student_image, teacher_image, label) for trim->192 KD/QAT."""

    def __init__(self, data_csv, student_transform, teacher_transform):
        self.data_csv = data_csv
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        label = self.data_csv.iloc[idx]["label"]
        img_path = str(self.data_csv.iloc[idx]["image"]).strip()
        img = safe_pil_read(img_path)
        img_np = np.array(img)

        img_student = trim_fundus_black_border(img_np)

        aug_s = self.student_transform(image=img_student)
        img_s = tv_transforms.ToTensor()(np.float32(aug_s["image"]))

        aug_t = self.teacher_transform(image=img_np)
        img_t = tv_transforms.ToTensor()(np.float32(aug_t["image"]))

        label = torch.tensor(label, dtype=torch.long)
        return img_s, img_t, label


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

    run_cfg = resolve_trim192_qat_run(
        weight_bits=weight_bits,
        act_bits=act_bits,
    )

    results_dir = os.path.join(cfg.results_dir, run_cfg["results_dir_name"])
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    train_df, val_df, test_df = prepare_dataframes(cfg)

    train_dataset = DualResTrim192Dataset(
        train_df,
        student_transform=student_train_transform,
        teacher_transform=teacher_train_transform,
    )
    val_dataset = DualResTrim192Dataset(
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

    print(f"Loading teacher from: {teacher_path}")
    teacher = ResNet18Classifier(nr_classes=cfg.nr_classes, pretrained=False)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("Teacher loaded and frozen (512x512 full-image, strong-train/test-eval domain).")

    print("\n" + "=" * 50)
    print(f"Creating QuantTestResNet [{tag}] for {run_cfg['run_tag']}")
    print("=" * 50)
    student = QuantTestResNet(
        nr_classes=cfg.nr_classes,
        weight_bit_width=weight_bits,
        act_bit_width=act_bits,
    )
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {n_params:,}")

    student_init_checkpoint = OmegaConf.select(cfg, "warm_start_checkpoint", default=None)
    if student_init_checkpoint is None:
        student_init_checkpoint = os.path.join(cfg.models_dir, TRIM192_FP32_WARM_START)
    if not os.path.exists(student_init_checkpoint):
        print(f"[ERROR] Student init checkpoint not found: {student_init_checkpoint}")
        return

    print(f"\nLoading student init weights from: {student_init_checkpoint} (imagenet)")
    missing, unexpected = load_test_resnet_weights(student, student_init_checkpoint)
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
        "Configuration: KD-QAT + trim black border -> 192 student + upgraded ResNet18 teacher "
        "on 512 strong/eval-test transforms"
    )
    print(f"Warm start: {student_init_checkpoint} (imagenet)")
    print(f"KD: temperature={KD_TEMPERATURE}, alpha={KD_ALPHA}")
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

    for epoch in tqdm(range(QAT_EPOCHS), desc=f"KD-QAT Trim192 {run_cfg['run_tag']}"):
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
        "n_params": n_params,
        "epochs": best_epoch + 1,
        "best_val_f1": round(best_val_f1, 4),
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": qat_ckpt_path,
        "student_init_checkpoint": student_init_checkpoint,
        "student_init_mode": "imagenet",
        "teacher": "resnet18_from_resnet50_fp32_kd.pth (512x512 full-image strong train / test eval)",
        "student_resolution": 192,
        "teacher_resolution": 512,
        "student_preprocess": {
            "name": "trim_fundus_black_border",
            "threshold": 8,
            "pad_ratio": 0.01,
            "min_pad_px": 4,
        },
        "input_size": [1, 3, 192, 192],
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
