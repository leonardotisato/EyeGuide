"""
Generate golden input/output .npy files for FINN verification.

The golden input is a raw uint8 image [1, 3, 512, 512] — the format expected
by the FINN model after preprocessing (ToTensor + ImageNet norm) is merged.

The golden output is the model prediction [1, 4] computed by running the
full pipeline: FundusPreProc -> QAT model.

Usage:
    python src/finn_build/generate_golden_io.py

Produces:
    src/finn_build/verification/golden_input.npy
    src/finn_build/verification/golden_output.npy
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.seed import set_seeds
from utils.quant_resnet18 import QuantResNet18
from utils.transforms import SIZE
from utils.dataset import prepare_dataframes

# Reuse the preprocessing module from our custom FINN steps
sys.path.insert(0, os.path.dirname(__file__))
from custom_steps_resnet18 import FundusPreProc


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cpu")  # CPU for reproducibility

    # ── Load QAT model ────────────────────────────────────────────────
    ckpt_path = os.path.join(cfg.models_dir, "resnet18_4w4a_qat.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] QAT checkpoint not found: {ckpt_path}")
        return

    model = QuantResNet18(nr_classes=cfg.nr_classes, weight_bit_width=4, act_bit_width=4)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded QAT checkpoint: {ckpt_path}")

    preproc = FundusPreProc()
    preproc.eval()

    # ── Load one test image ───────────────────────────────────────────
    _, _, test_df = prepare_dataframes(cfg)
    img_path = test_df.iloc[0]["image_path"]
    print(f"Using test image: {img_path}")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))

    # Golden input: float32 [1, 3, H, W] with integer values 0-255 (raw pixels).
    # FINN expects float32 tensors; the preprocessing merges /255 + normalization.
    img_chw = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    golden_input = img_chw[np.newaxis, ...].astype(np.float32)  # [1, 3, 512, 512]

    # Golden output: run preprocessing + model
    input_tensor = torch.from_numpy(golden_input)  # float32, values 0-255
    with torch.no_grad():
        normalized = preproc(input_tensor)
        output = model(normalized)
    golden_output = output.numpy()

    print(f"  Input shape:  {golden_input.shape} (uint8 range: {golden_input.min():.0f}-{golden_input.max():.0f})")
    print(f"  Output shape: {golden_output.shape}")
    print(f"  Predictions:  {golden_output}")

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "verification")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "golden_input.npy"), golden_input)
    np.save(os.path.join(out_dir, "golden_output.npy"), golden_output)
    print(f"\nSaved: {out_dir}/golden_input.npy")
    print(f"Saved: {out_dir}/golden_output.npy")


if __name__ == "__main__":
    main()
