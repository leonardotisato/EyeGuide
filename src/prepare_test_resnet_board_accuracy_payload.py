#!/usr/bin/env python3
"""Prepare a minimal Ultra96 accuracy-test payload for FINN test_resnet.

The board-side FINN overlay expects resized uint8 NHWC images. The exported
FINN graph already contains the ImageNet normalization pre-processing, so this
script only reproduces the dataset-side trim + resize before saving compact
NumPy arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


WT_VARINT = 0
WT_LEN = 2
WT_32 = 5
WT_64 = 1


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7


def iter_proto_fields(buf: bytes):
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field = key >> 3
        wire_type = key & 7
        start = pos
        if wire_type == WT_VARINT:
            value, pos = read_varint(buf, pos)
            yield field, wire_type, value
        elif wire_type == WT_64:
            pos += 8
            yield field, wire_type, buf[start:pos]
        elif wire_type == WT_LEN:
            length, pos = read_varint(buf, pos)
            start = pos
            pos += length
            yield field, wire_type, buf[start:pos]
        elif wire_type == WT_32:
            pos += 4
            yield field, wire_type, buf[start:pos]
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")


def graph_from_onnx(model_bytes: bytes) -> bytes:
    for field, wire_type, value in iter_proto_fields(model_bytes):
        if field == 7 and wire_type == WT_LEN:
            return value
    raise ValueError("Could not find GraphProto in ONNX file")


def parse_tensor_proto(tensor_bytes: bytes) -> dict:
    dims: list[int] = []
    data_type = None
    name = None
    raw = None
    for field, wire_type, value in iter_proto_fields(tensor_bytes):
        if field == 1 and wire_type == WT_VARINT:
            dims.append(int(value))
        elif field == 2 and wire_type == WT_VARINT:
            data_type = int(value)
        elif field == 8 and wire_type == WT_LEN:
            name = value.decode("utf-8")
        elif field == 9 and wire_type == WT_LEN:
            raw = value
    return {"name": name, "dims": dims, "data_type": data_type, "raw": raw}


def extract_initializers(onnx_path: Path) -> dict[str, dict]:
    graph = graph_from_onnx(onnx_path.read_bytes())
    tensors: dict[str, dict] = {}
    for field, wire_type, value in iter_proto_fields(graph):
        if field == 5 and wire_type == WT_LEN:
            tensor = parse_tensor_proto(value)
            if tensor["name"] is not None:
                tensors[tensor["name"]] = tensor
    return tensors


def tensor_to_numpy(tensor: dict) -> np.ndarray:
    raw = tensor["raw"]
    if raw is None:
        raise ValueError(f"Initializer {tensor['name']} has no raw_data")
    dims = tuple(tensor["dims"])
    data_type = tensor["data_type"]
    if data_type == 1:
        dtype = np.float32
    elif data_type == 2:
        dtype = np.uint8
    elif data_type == 3:
        dtype = np.int8
    elif data_type == 6:
        dtype = np.int32
    elif data_type == 7:
        dtype = np.int64
    else:
        raise ValueError(
            f"Unsupported ONNX TensorProto data_type={data_type} for {tensor['name']}"
        )
    arr = np.frombuffer(raw, dtype=dtype).copy()
    return arr.reshape(dims) if dims else arr.reshape(())


def extract_tail_params(parent_onnx: Path, out_npz: Path) -> dict:
    initializers = extract_initializers(parent_onnx)
    required = [
        "Mul_0_param0",
        "MatMul_0_param0",
        "Mul_1_param0",
        "Add_0_param0",
        "TopK_0_param0",
    ]
    missing = [name for name in required if name not in initializers]
    if missing:
        raise ValueError(f"Missing classifier-tail initializers: {missing}")

    params = {name: tensor_to_numpy(initializers[name]) for name in required}
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        mul0=params["Mul_0_param0"].astype(np.float32),
        matmul=params["MatMul_0_param0"].astype(np.float32),
        mul1=params["Mul_1_param0"].astype(np.float32),
        add0=params["Add_0_param0"].astype(np.float32),
        topk_k=params["TopK_0_param0"].astype(np.int64),
    )
    return {
        "mul0": float(params["Mul_0_param0"]),
        "matmul_shape": list(params["MatMul_0_param0"].shape),
        "mul1": float(params["Mul_1_param0"]),
        "add0": params["Add_0_param0"].astype(float).tolist(),
        "topk_k": int(params["TopK_0_param0"].reshape(-1)[0]),
    }


def trim_fundus_black_border(
    image: np.ndarray,
    threshold: int = 8,
    pad_ratio: float = 0.01,
    min_pad_px: int = 4,
) -> np.ndarray:
    if image.ndim != 3:
        return image
    gray = image.mean(axis=2)
    ys, xs = np.where(gray > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return image
    h, w = image.shape[:2]
    pad = max(min_pad_px, int(round(min(h, w) * pad_ratio)))
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)
    return image[y0:y1, x0:x1]


def resize_rgb(image: np.ndarray, size: int, backend: str) -> np.ndarray:
    if backend == "cv2":
        import cv2

        return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    if backend == "pil":
        return np.asarray(
            Image.fromarray(image).resize((size, size), resample=Image.BILINEAR)
        )
    try:
        return resize_rgb(image, size, "cv2")
    except Exception:
        return resize_rgb(image, size, "pil")


def unique_patients_with_label(
    df: pd.DataFrame, patient_col: str, label_col: str
) -> pd.DataFrame:
    return df[[patient_col, label_col]].drop_duplicates(subset=[patient_col]).reset_index(drop=True)


def can_stratify(patients_df: pd.DataFrame, label_col: str) -> bool:
    counts = patients_df[label_col].value_counts()
    return bool((counts.min() >= 2) and (len(counts) >= 2))


def prepare_test_dataframe(
    csv_path: Path,
    seed: int,
    patient_col: str = "patient_id",
    label_col: str = "label",
    val_rel: float = 0.15,
    test_rel: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    full_df = pd.read_csv(csv_path)
    if patient_col not in full_df.columns:
        full_df[patient_col] = full_df["patient"].str.split("_").str[0].astype(str)

    patients = unique_patients_with_label(full_df, patient_col, label_col)
    if can_stratify(patients, label_col):
        splitter_test = StratifiedShuffleSplit(
            n_splits=1, test_size=test_rel, random_state=seed
        )
        idx_trainval, idx_test = next(
            splitter_test.split(patients[[patient_col]], y=patients[label_col])
        )
        test_splitter = "StratifiedShuffleSplit"
    else:
        splitter_test = GroupShuffleSplit(
            n_splits=1, test_size=test_rel, random_state=seed
        )
        idx_trainval, idx_test = next(
            splitter_test.split(patients, groups=patients[patient_col])
        )
        test_splitter = "GroupShuffleSplit"

    trainval_patients = set(patients.iloc[idx_trainval][patient_col].values)
    test_patients = set(patients.iloc[idx_test][patient_col].values)
    patients_trainval = patients[patients[patient_col].isin(trainval_patients)].reset_index(
        drop=True
    )

    val_rel_eff = val_rel / (1.0 - test_rel)
    if can_stratify(patients_trainval, label_col):
        splitter_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_rel_eff, random_state=seed
        )
        idx_train, idx_val = next(
            splitter_val.split(
                patients_trainval[[patient_col]], y=patients_trainval[label_col]
            )
        )
        val_splitter = "StratifiedShuffleSplit"
    else:
        splitter_val = GroupShuffleSplit(
            n_splits=1, test_size=val_rel_eff, random_state=seed
        )
        idx_train, idx_val = next(
            splitter_val.split(
                patients_trainval, groups=patients_trainval[patient_col]
            )
        )
        val_splitter = "GroupShuffleSplit"

    train_patients = set(patients_trainval.iloc[idx_train][patient_col].values)
    val_patients = set(patients_trainval.iloc[idx_val][patient_col].values)
    test_df = full_df[full_df[patient_col].isin(test_patients)].reset_index(drop=True)

    split_info = {
        "seed": seed,
        "val_rel": val_rel,
        "test_rel": test_rel,
        "test_splitter": test_splitter,
        "val_splitter": val_splitter,
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "test_images": len(test_df),
    }
    return test_df, split_info


def load_preprocessed_image(path: Path, resolution: int, resize_backend: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image.load()
    arr = np.asarray(image, dtype=np.uint8)
    arr = trim_fundus_black_border(arr)
    arr = resize_rgb(arr, resolution, resize_backend)
    return np.asarray(arr, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", default="Data/fundus_data_final.csv")
    parser.add_argument("--build-dir", default="build_finn_test_resnet_trim160_6w6a_ultra96")
    parser.add_argument("--output-dir", default="board_accuracy_payload_trim160")
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize-backend", choices=["auto", "cv2", "pil"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path.cwd()
    data_csv = (repo_root / args.data_csv).resolve()
    build_dir = (repo_root / args.build_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    parent_onnx = build_dir / "intermediate_models" / "dataflow_parent.onnx"

    test_df, split_info = prepare_test_dataframe(data_csv, seed=args.seed)
    if args.limit is not None:
        test_df = test_df.iloc[: args.limit].reset_index(drop=True)

    images = []
    labels = []
    manifest_rows = []
    for idx, row in test_df.iterrows():
        image_rel = str(row["image"]).strip()
        image_path = (repo_root / image_rel).resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        images.append(load_preprocessed_image(image_path, args.resolution, args.resize_backend))
        labels.append(int(row["label"]))
        manifest_rows.append(
            {
                "idx": idx,
                "image": image_rel,
                "patient": str(row.get("patient", "")),
                "patient_id": str(row.get("patient_id", "")),
                "label": int(row["label"]),
            }
        )

    x = np.stack(images).astype(np.uint8)
    y = np.asarray(labels, dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "test_images_uint8.npy", x)
    np.save(output_dir / "test_labels.npy", y)
    tail_info = extract_tail_params(parent_onnx, output_dir / "tail_params.npz")

    with (output_dir / "test_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["idx", "image", "patient", "patient_id", "label"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "resolution": args.resolution,
        "dtype": "uint8",
        "shape": list(x.shape),
        "label_counts": dict(sorted(Counter(y.tolist()).items())),
        "split": split_info,
        "resize_backend_requested": args.resize_backend,
        "payload_files": [
            "test_images_uint8.npy",
            "test_labels.npy",
            "tail_params.npz",
            "test_manifest.csv",
        ],
        "classifier_tail": tail_info,
        "note": (
            "Images are trim_fundus_black_border + resize only. "
            "ImageNet normalization is inside the FINN graph."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    size_mb = sum(p.stat().st_size for p in output_dir.iterdir() if p.is_file()) / (1024 * 1024)
    print(f"Wrote {output_dir}")
    print(f"Samples: {len(y)}  shape={x.shape}  labels={metadata['label_counts']}")
    print(f"Payload size: {size_mb:.2f} MiB")
    print(f"Classifier tail: {tail_info}")


if __name__ == "__main__":
    main()
