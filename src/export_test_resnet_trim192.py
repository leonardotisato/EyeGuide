"""
Export the canonical trim-192 QAT test_resnet checkpoint to QONNX.

Loads the checkpoint saved by `src/qat_test_resnet_trim192.py` and exports it
to QONNX. No recalibration is needed: the checkpoint already contains trained
quantizer scales. BatchNorm nodes remain in the exported graph and are later
folded by FINN Streamline.

This export matches the trim-192 student domain:
- input size: 192x192
- checkpoint names: `test_resnet_trim192_{tag}_qat.pth`
- exported ONNX names: `test_resnet_trim192_{tag}.onnx`

Run with:
    bash run.sh export_test_resnet_trim192 ++weight_bits=8 ++act_bits=8
    bash run.sh export_test_resnet_trim192 ++weight_bits=6 ++act_bits=6
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
from brevitas.export import export_qonnx
from omegaconf import DictConfig, OmegaConf

try:
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    import qonnx.transformation.infer_shapes

    _QONNX_AVAILABLE = True
except ImportError:
    _QONNX_AVAILABLE = False
    print("[WARNING] qonnx not installed - skipping numerical validation.")

sys.path.insert(0, os.path.dirname(__file__))
from utils.quant_test_resnet import QuantTestResNet, model_tag
from utils.seed import set_seeds


INPUT_SIZE = 192


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weight_bits = int(OmegaConf.select(cfg, "weight_bits", default=8))
    act_bits = int(OmegaConf.select(cfg, "act_bits", default=8))
    tag = model_tag(weight_bits, act_bits)

    os.makedirs(cfg.results_dir, exist_ok=True)

    ckpt_path = os.path.join(cfg.models_dir, f"test_resnet_trim192_{tag}_qat.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] QAT checkpoint not found: {ckpt_path}")
        print("Run src/qat_test_resnet_trim192.py first.")
        return

    model = QuantTestResNet(
        nr_classes=cfg.nr_classes,
        weight_bit_width=weight_bits,
        act_bit_width=act_bits,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded trim-192 QAT checkpoint: {ckpt_path}")

    export_path = os.path.join(cfg.models_dir, f"test_resnet_trim192_{tag}.onnx")
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)

    print("\nExporting to QONNX ...")
    export_qonnx(
        model,
        input_t=dummy_input,
        export_path=export_path,
        opset_version=13,
    )
    print(f"QONNX model exported -> {export_path}")

    if _QONNX_AVAILABLE:
        print("\nRunning numerical validation (PyTorch vs QONNX) ...")
        dummy_np = dummy_input.cpu().detach().numpy()
        with torch.no_grad():
            pt_out = model(dummy_input).cpu().numpy()

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

    print(f"\nDone. Exported model: {export_path}")
    print(
        "Next: python src/finn_build/build_test_resnet.py "
        f"--estimates-only --onnx models/test_resnet_trim192_{tag}.onnx"
    )


if __name__ == "__main__":
    main()
