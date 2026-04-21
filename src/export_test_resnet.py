"""
Export test_resnet QAT checkpoint to QONNX.

Loads the QAT checkpoint saved by qat_test_resnet.py and exports to QONNX.
No recalibration — the checkpoint already contains trained quantizer scales.
BatchNorm nodes remain in the exported graph — FINN Streamline handles folding.

Run with:
    bash run_export_test_resnet.sh
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from brevitas.export import export_qonnx
from omegaconf import OmegaConf

try:
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    import qonnx.transformation.infer_shapes
    _QONNX_AVAILABLE = True
except ImportError:
    _QONNX_AVAILABLE = False
    print("[WARNING] qonnx not installed — skipping numerical validation.")

sys.path.insert(0, os.path.dirname(__file__))
from utils.seed import set_seeds
from utils.quant_test_resnet import QuantTestResNet, model_tag
from utils.transforms_224_strong import SIZE


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    WEIGHT_BITS = int(OmegaConf.select(cfg, "weight_bits", default=8))
    ACT_BITS = int(OmegaConf.select(cfg, "act_bits", default=8))
    use_kd = bool(OmegaConf.select(cfg, "kd", default=False))

    os.makedirs(cfg.results_dir, exist_ok=True)

    tag = model_tag(WEIGHT_BITS, ACT_BITS)
    ckpt_suffix = "kd_qat" if use_kd else "qat"

    # ── Load QAT checkpoint ──────────────────────────────────────────────
    ckpt_path = os.path.join(cfg.models_dir, f"test_resnet_{tag}_{ckpt_suffix}.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] QAT checkpoint not found: {ckpt_path}")
        print("Run qat_test_resnet.py or qat_kd_test_resnet.py first.")
        return

    model = QuantTestResNet(
        nr_classes=cfg.nr_classes,
        weight_bit_width=WEIGHT_BITS,
        act_bit_width=ACT_BITS,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded QAT checkpoint: {ckpt_path}")

    # ── Export to QONNX ──────────────────────────────────────────────────
    onnx_tag = f"{tag}_kd" if use_kd else tag
    export_path = os.path.join(cfg.models_dir, f"test_resnet_{onnx_tag}.onnx")
    dummy_input = torch.randn(1, 3, SIZE, SIZE).to(device)

    print("\nExporting to QONNX ...")
    export_qonnx(
        model,
        input_t=dummy_input,
        export_path=export_path,
        opset_version=13,
    )
    print(f"QONNX model exported -> {export_path}")

    # ── Numerical validation ─────────────────────────────────────────────
    if _QONNX_AVAILABLE:
        print("\nRunning numerical validation (PyTorch vs QONNX) ...")
        dummy_np = dummy_input.cpu().detach().numpy()
        with torch.no_grad():
            pt_out = model(dummy_input).cpu().numpy()

        qonnx_model = ModelWrapper(export_path)
        qonnx_model = qonnx_model.transform(
            qonnx.transformation.infer_shapes.InferShapes()
        )
        input_name = qonnx_model.graph.input[0].name
        idict = {input_name: dummy_np}
        odict = execute_onnx(qonnx_model, idict)
        qonnx_out = list(odict.values())[0]

        max_diff = float(np.abs(pt_out - qonnx_out).max())
        print(f"  Max output diff (PyTorch vs QONNX): {max_diff:.6f}")
    else:
        print("Skipping numerical validation (qonnx not available).")

    print(f"\nDone. Exported model: {export_path}")
    print(f"Next: python src/finn_build/build_test_resnet.py --estimates-only --onnx models/test_resnet_{onnx_tag}.onnx")


if __name__ == "__main__":
    main()
