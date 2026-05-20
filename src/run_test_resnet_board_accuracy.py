#!/usr/bin/env python3
"""Run FINN test_resnet accuracy on an Ultra96 deployment payload."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def compute_metrics(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> dict:
    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels, preds):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            cm[true, pred] += 1

    per_class = {}
    f1_values = []
    weighted_f1_num = 0.0
    for cls in range(num_classes):
        tp = int(cm[cls, cls])
        fp = int(cm[:, cls].sum() - tp)
        fn = int(cm[cls, :].sum() - tp)
        support = int(cm[cls, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        weighted_f1_num += f1 * support
        per_class[str(cls)] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total = int(len(labels))
    accuracy = float((labels == preds).mean()) if total else 0.0
    return {
        "num_samples": total,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "weighted_f1": float(weighted_f1_num / total) if total else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "label_counts": dict(sorted(Counter(labels.tolist()).items())),
        "pred_counts": dict(sorted(Counter(preds.tolist()).items())),
    }


def apply_classifier_tail(features: np.ndarray, tail_params) -> np.ndarray:
    mul0 = np.asarray(tail_params["mul0"], dtype=np.float32)
    matmul = np.asarray(tail_params["matmul"], dtype=np.float32)
    mul1 = np.asarray(tail_params["mul1"], dtype=np.float32)
    add0 = np.asarray(tail_params["add0"], dtype=np.float32)
    flat = features.reshape(features.shape[0], -1).astype(np.float32)
    logits = (flat * mul0) @ matmul
    logits = logits * mul1 + add0
    return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-dir", default="deploy")
    parser.add_argument("--payload-dir", default="accuracy_payload_trim160")
    parser.add_argument("--bitfile", default=None)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--platform", default="zynq-iodma")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="reports/accuracy_test.json")
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--debug-npz", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    deploy_dir = Path(args.deploy_dir).resolve()
    payload_dir = Path(args.payload_dir).resolve()
    driver_dir = deploy_dir / "driver"
    bitfile = Path(args.bitfile).resolve() if args.bitfile else deploy_dir / "bitfile" / "finn-accel.bit"

    sys.path.insert(0, str(driver_dir))
    from driver import FINNExampleOverlay, io_shape_dict
    from pynq.pl_server.device import Device

    images = np.load(payload_dir / "test_images_uint8.npy")
    labels = np.load(payload_dir / "test_labels.npy")
    tail_params = np.load(payload_dir / "tail_params.npz")

    if args.max_samples is not None:
        images = images[: args.max_samples]
        labels = labels[: args.max_samples]

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB images, got shape {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"Expected uint8 images, got {images.dtype}")

    batch_size = int(args.batchsize)
    device = Device.devices[args.device]
    accel = FINNExampleOverlay(
        bitfile_name=str(bitfile),
        platform=args.platform,
        io_shape_dict=io_shape_dict,
        batch_size=batch_size,
        runtime_weight_dir=str(driver_dir / "runtime_weights"),
        device=device,
    )

    logits_chunks = []
    hw_runtime_s = 0.0
    total_runtime_start = time.time()
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = images[start:end]
        valid = end - start
        if valid < batch_size:
            padded = np.zeros((batch_size,) + images.shape[1:], dtype=np.uint8)
            padded[:valid] = batch
            batch = padded

        hw_start = time.time()
        features = accel.execute(batch)
        hw_runtime_s += time.time() - hw_start
        logits = apply_classifier_tail(features[:valid], tail_params)
        logits_chunks.append(logits)
        print(f"Processed {end}/{len(images)}", flush=True)

    logits_all = np.concatenate(logits_chunks, axis=0)
    preds = np.argmax(logits_all, axis=1).astype(np.int64)
    total_runtime_s = time.time() - total_runtime_start
    feature_sample = features[: min(valid, 8)].reshape(min(valid, 8), -1).copy()

    num_classes = int(max(labels.max(), preds.max()) + 1) if len(labels) else 0
    num_classes = max(num_classes, 4)
    metrics = compute_metrics(labels, preds, num_classes=num_classes)
    metrics.update(
        {
            "batch_size": batch_size,
            "hw_runtime_ms": hw_runtime_s * 1000.0,
            "total_runtime_ms": total_runtime_s * 1000.0,
            "hw_throughput_images_per_s": float(len(images) / hw_runtime_s) if hw_runtime_s else 0.0,
            "total_throughput_images_per_s": float(len(images) / total_runtime_s)
            if total_runtime_s
            else 0.0,
            "bitfile": str(bitfile),
            "payload_dir": str(payload_dir),
            "output_logits_shape": list(logits_all.shape),
            "logit_stats": {
                "min": float(logits_all.min()),
                "max": float(logits_all.max()),
                "mean": float(logits_all.mean()),
                "std": float(logits_all.std()),
            },
            "last_batch_feature_sample_stats": {
                "shape": list(feature_sample.shape),
                "min": float(feature_sample.min()),
                "max": float(feature_sample.max()),
                "mean": float(feature_sample.mean()),
                "std": float(feature_sample.std()),
            },
            "first_16_labels": labels[:16].astype(int).tolist(),
            "first_16_preds": preds[:16].astype(int).tolist(),
            "first_8_logits": logits_all[:8].astype(float).tolist(),
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_csv = (
        Path(args.predictions_csv)
        if args.predictions_csv is not None
        else output_path.with_name(output_path.stem + "_predictions.csv")
    )
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    with predictions_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label", "pred", "logit_0", "logit_1", "logit_2", "logit_3"])
        for idx, (label, pred, row_logits) in enumerate(zip(labels, preds, logits_all)):
            writer.writerow([idx, int(label), int(pred)] + [float(x) for x in row_logits[:4]])

    if args.debug_npz is not None:
        debug_path = Path(args.debug_npz)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            debug_path,
            labels=labels,
            preds=preds,
            logits=logits_all,
            last_feature_sample=feature_sample,
        )

    print(json.dumps(metrics, indent=2))
    print(f"Results written to {output_path}")
    print(f"Predictions written to {predictions_csv}")


if __name__ == "__main__":
    main()
