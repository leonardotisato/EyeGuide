"""Export a slim QAT test_resnet checkpoint to QONNX.

Default artifact:
  models/test_resnet_slim128x64_trim160_6w6a.onnx
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
from utils.quant_test_resnet_slim import QuantTestResNetSlim, model_tag  # noqa: E402
from utils.seed import set_seeds  # noqa: E402
from utils.test_resnet_slim import (  # noqa: E402
    DEFAULT_LAYER3_OUT,
    DEFAULT_LAYER4_OUT,
    slim_variant_tag,
)


DEFAULT_STUDENT_RESOLUTION = 160
DEFAULT_ONNX_ARTIFACT = "test_resnet_slim128x64_trim160_6w6a.onnx"


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weight_bits = int(OmegaConf.select(cfg, "weight_bits", default=8))
    act_bits = int(OmegaConf.select(cfg, "act_bits", default=8))
    student_resolution = int(
        OmegaConf.select(cfg, "student_resolution", default=DEFAULT_STUDENT_RESOLUTION)
    )
    layer3_out = int(OmegaConf.select(cfg, "slim_layer3_out", default=DEFAULT_LAYER3_OUT))
    layer4_out = int(OmegaConf.select(cfg, "slim_layer4_out", default=DEFAULT_LAYER4_OUT))

    bit_tag = model_tag(weight_bits, act_bits)
    trim_tag = f"trim{student_resolution}"
    variant = slim_variant_tag(layer3_out, layer4_out)
    run_tag = f"{variant}_{trim_tag}_{bit_tag}"

    os.makedirs(cfg.results_dir, exist_ok=True)

    ckpt_path = os.path.join(cfg.models_dir, f"test_resnet_{run_tag}_qat.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] QAT checkpoint not found: {ckpt_path}")
        print(
            "Run src/qat_test_resnet_slim.py first, for example: "
            f"++student_resolution={student_resolution} "
            f"++weight_bits={weight_bits} ++act_bits={act_bits}"
        )
        return

    model = QuantTestResNetSlim(
        nr_classes=cfg.nr_classes,
        weight_bit_width=weight_bits,
        act_bit_width=act_bits,
        layer3_out=layer3_out,
        layer4_out=layer4_out,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded slim QAT checkpoint: {ckpt_path}")

    export_path = os.path.join(cfg.models_dir, f"test_resnet_{run_tag}.onnx")
    dummy_input = torch.randn(1, 3, student_resolution, student_resolution).to(device)

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
        odict = execute_onnx(qonnx_model, {input_name: dummy_np})
        qonnx_out = list(odict.values())[0]

        max_diff = float(np.abs(pt_out - qonnx_out).max())
        print(f"  Max output diff (PyTorch vs QONNX): {max_diff:.6f}")
    else:
        print("Skipping numerical validation (qonnx not available).")

    print(f"\nDone. Exported model: {export_path}")
    print(
        "Next: python src/finn_build/build_test_resnet.py "
        f"--estimates-only --onnx models/test_resnet_{run_tag}.onnx"
    )


if __name__ == "__main__":
    main()
