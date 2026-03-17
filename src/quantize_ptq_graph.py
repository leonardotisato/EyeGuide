"""
Post-Training Quantization (PTQ) of student_kd model using Brevitas graph-based quantize().

- preprocess_for_quantize() traces the model with torch.fx, merges BatchNorm, removes Dropout
- quantize() inserts INT8 quantizers on weights, activations AND between-layer requantization
  nodes (e.g. after residual additions).

Run with:
    python src/quantize_ptq_graph.py
"""

import os
import sys
import glob
import json
import copy
import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

# Brevitas graph-based PTQ APIs
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.graph.calibrate import calibration_mode
from brevitas.export import export_qonnx

# FINN-compatible fixed-point quantizers (power-of-2 scales, absorbable by FINN)
from brevitas.quant.fixed_point import (
    Int8WeightPerTensorFixedPoint,
    Int8ActPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
from brevitas.quant.scaled_int import Int32Bias

# Brevitas quantized NN layers (needed to build custom quantize() maps)
import torch.nn as nn
import brevitas.nn as qnn

# QONNX numerical validation
import numpy as np
try:
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    import qonnx.transformation.infer_shapes
    _QONNX_AVAILABLE = True
except ImportError:
    _QONNX_AVAILABLE = False
    print("[WARNING] qonnx not installed - skipping numerical validation.")
    print("          Install with: pip install qonnx")

# Project utilities
sys.path.insert(0, os.path.dirname(__file__))
from utils.seed import set_seeds
from utils.model import ResNet18Classifier
from utils.dataset import FundusClsDataset, prepare_dataframes
from utils.transforms import test_transform_class
from utils.training import test


def find_latest_checkpoint(models_dir: str, pattern: str) -> str:
    """Return the most-recently-modified checkpoint matching *pattern*."""
    paths = glob.glob(os.path.join(models_dir, pattern))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoint found matching '{pattern}' in '{models_dir}'"
        )
    return max(paths, key=os.path.getmtime)


def convert_metrics(metrics: dict) -> dict:
    """Recursively convert numpy/non-serialisable types to plain Python floats."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            out[k] = convert_metrics(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) for x in v]
        else:
            try:
                out[k] = round(float(v), 6)
            except (TypeError, ValueError):
                out[k] = v
    return out


def scalar_metrics(metrics: dict) -> dict:
    """
    Extract {accuracy, f1, precision, recall} with scalar float mean values.
    bootstrap_metrics() uses key 'f1_score' (not 'f1') and returns flat
    3-tuples (mean, ci_lower, ci_upper). non-bootstrap test() returns plain floats.
    """
    key_map = {
        "accuracy": ["accuracy"],
        "f1": ["f1", "f1_score"],
        "precision": ["precision"],
        "recall": ["recall"],
    }
    out = {}
    for canon, candidates in key_map.items():
        val = 0.0
        for k in candidates:
            if k in metrics:
                val = metrics[k]
                break
        if isinstance(val, (list, tuple)):
            val = val[0]
        out[canon] = round(float(val), 6)
    return out


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.results_dir, exist_ok=True)

    train_df, _val_df, test_df = prepare_dataframes(cfg)

    calib_dataset = FundusClsDataset(
        data_csv=train_df,
        train=False,
        transform=test_transform_class,
    )
    test_dataset = FundusClsDataset(
        data_csv=test_df,
        train=False,
        transform=test_transform_class,
    )

    calib_loader = DataLoader(
        calib_dataset,
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

    ckpt_path = find_latest_checkpoint(cfg.models_dir, "student_kd_*.pth")
    print(f"\nLoading checkpoint: {ckpt_path}")

    fp32_model = ResNet18Classifier(
        nr_classes=cfg.nr_classes,
        dropout=0.5,
        pretrained=False,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    fp32_model.load_state_dict(state_dict)
    fp32_model.eval()
    fp32_model.to(device)
    print("FP32 model loaded successfully.")

    print("\n" + "=" * 50)
    print("Evaluating FP32 model on test set ...")
    print("=" * 50)
    metrics_fp32_raw = test(
        model=fp32_model,
        test_loader=test_loader,
        device=device,
        model_type="student_kd",
        bootstrap=True,
        savedir=cfg.results_dir,
    )
    metrics_fp32 = scalar_metrics(metrics_fp32_raw)
    print(
        f"FP32  ->  Acc: {metrics_fp32['accuracy']:.4f} | "
        f"F1: {metrics_fp32['f1']:.4f} | "
        f"Prec: {metrics_fp32['precision']:.4f} | "
        f"Rec: {metrics_fp32['recall']:.4f}"
    )

    print("\n" + "=" * 50)
    print("Preprocessing model for graph quantization ...")
    print("  (symbolic_trace + BN merge + Dropout removal)")
    print("=" * 50)

    model_to_trace = copy.deepcopy(fp32_model).cpu()
    model_to_trace.eval()

    try:
        preprocessed = preprocess_for_quantize(
            model_to_trace,
            trace_model=True,
            relu6_to_relu=True,
            merge_bn=True,
            equalize_iters=0,
        )
        print("Graph tracing successful.")
    except Exception as e:
        print(f"\n[ERROR] symbolic_trace failed: {e}")
        raise

    print("\nApplying INT8 graph quantization (FINN-compatible FixedPoint) ...")
    FIXED_COMPUTE_LAYER_MAP = {
        nn.Conv2d: (qnn.QuantConv2d, {
            "weight_quant": Int8WeightPerTensorFixedPoint,
            "bias_quant": Int32Bias,
            "return_quant_tensor": True}),
        nn.Conv1d: (qnn.QuantConv1d, {
            "weight_quant": Int8WeightPerTensorFixedPoint,
            "bias_quant": Int32Bias,
            "return_quant_tensor": True}),
        nn.ConvTranspose1d: (qnn.QuantConvTranspose1d, {
            "weight_quant": Int8WeightPerTensorFixedPoint,
            "bias_quant": Int32Bias,
            "return_quant_tensor": True}),
        nn.ConvTranspose2d: (qnn.QuantConvTranspose2d, {
            "weight_quant": Int8WeightPerTensorFixedPoint,
            "bias_quant": Int32Bias,
            "return_quant_tensor": True}),
        nn.Linear: (qnn.QuantLinear, {
            "weight_quant": Int8WeightPerTensorFixedPoint,
            "bias_quant": Int32Bias,
            "return_quant_tensor": True}),
        nn.AvgPool2d: None,
    }
    # FINN requires Quant nodes following ReLU to stay unsigned, while
    # quantized identity activations must be signed. We therefore keep ReLU
    # activations UINT8 and use signed QuantIdentity nodes for residual-path
    # requantization where possible.
    FIXED_QUANT_ACT_MAP = {
        nn.ReLU: (qnn.QuantReLU, {
            "act_quant": Uint8ActPerTensorFixedPoint,
            "return_quant_tensor": True}),
    }
    FIXED_QUANT_IDENTITY_MAP = {
        "signed": (qnn.QuantIdentity, {
            "act_quant": Int8ActPerTensorFixedPoint,
            "return_quant_tensor": True}),
        "unsigned": (qnn.QuantIdentity, {
            "act_quant": Int8ActPerTensorFixedPoint,
            "return_quant_tensor": True}),
    }
    quant_model = quantize(
        preprocessed,
        compute_layer_map=FIXED_COMPUTE_LAYER_MAP,
        quant_act_map=FIXED_QUANT_ACT_MAP,
        quant_identity_map=FIXED_QUANT_IDENTITY_MAP,
    )
    quant_model.eval()
    quant_model.to(device)
    print("Quantized graph model created.")

    print("\nCalibrating on training data (50 batches) ...")
    with calibration_mode(quant_model):
        for i, (imgs, _) in enumerate(calib_loader):
            if i >= 50:
                break
            with torch.no_grad():
                quant_model(imgs.to(device))
            if (i + 1) % 10 == 0:
                print(f"  Calibration batch {i + 1}/50")
    print("Calibration complete.")

    print("\n" + "=" * 50)
    print("Evaluating INT8 graph model on test set ...")
    print("=" * 50)
    metrics_int8_raw = test(
        model=quant_model,
        test_loader=test_loader,
        device=device,
        model_type="student_kd_int8_graph",
        bootstrap=True,
        savedir=cfg.results_dir,
    )
    metrics_int8 = scalar_metrics(metrics_int8_raw)
    print(
        f"INT8  ->  Acc: {metrics_int8['accuracy']:.4f} | "
        f"F1: {metrics_int8['f1']:.4f} | "
        f"Prec: {metrics_int8['precision']:.4f} | "
        f"Rec: {metrics_int8['recall']:.4f}"
    )

    delta = {
        k: round(metrics_fp32[k] - metrics_int8[k], 6)
        for k in metrics_fp32
    }

    print("\n" + "=" * 60)
    print(f"{'Metric':<12} {'FP32':>10} {'INT8 (graph)':>14} {'Delta':>10}")
    print("-" * 60)
    for k in ["accuracy", "f1", "precision", "recall"]:
        print(
            f"{k:<12} {metrics_fp32[k]:>10.4f} "
            f"{metrics_int8[k]:>14.4f} {delta[k]:>+10.4f}"
        )
    print("=" * 60)

    report = convert_metrics({
        "fp32": metrics_fp32,
        "int8_ptq_graph": metrics_int8,
        "delta": delta,
    })
    report_path = os.path.join(cfg.results_dir, "ptq_graph_quantization_report.json")
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nReport saved -> {report_path}")

    print("\n" + "=" * 50)
    print("Exporting INT8 graph model to QONNX ...")
    print("=" * 50)

    export_path = os.path.join(cfg.models_dir, "student_kd_int8_ptq.onnx")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    quant_model.eval()
    export_qonnx(
        quant_model,
        input_t=dummy_input,
        export_path=export_path,
        opset_version=13,
    )
    print(f"QONNX model exported -> {export_path}")

    if _QONNX_AVAILABLE:
        print("Running numerical validation (PyTorch vs QONNX) ...")
        dummy_np = dummy_input.cpu().detach().numpy()
        with torch.no_grad():
            pt_out = quant_model(dummy_input).cpu().numpy()

        qonnx_model = ModelWrapper(export_path)
        qonnx_model = qonnx_model.transform(qonnx.transformation.infer_shapes.InferShapes())
        input_name = qonnx_model.graph.input[0].name
        idict = {input_name: dummy_np}
        odict = execute_onnx(qonnx_model, idict)
        qonnx_out = list(odict.values())[0]

        max_diff = float(np.abs(pt_out - qonnx_out).max())
        print(f"  Max output diff (PyTorch vs QONNX): {max_diff:.6f}")
    else:
        print("Skipping numerical validation (qonnx not available).")


if __name__ == "__main__":
    main()
