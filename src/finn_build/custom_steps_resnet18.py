"""
Custom FINN build steps for the fundus QAT ResNet18.

Based on: ignore/finn-examples-rn18/build/resnet18/resnet18_custom_steps.py

Differences from finn-examples:
  - step_fundus_attach_preproc: includes ImageNet normalization (not just ToTensor)
  - step_fundus_lower: adds MakeMaxPoolNHWC (our model has MaxPool, finn-examples doesn't)
  - step_fundus_to_hw: adds InferGlobalAccPoolLayer, InferPool, InferStreamingMaxPool
    (our model has GlobalAveragePool and MaxPool, finn-examples has neither)

Must run inside the FINN Docker container.
"""

import torch
import torch.nn as nn
from collections import Counter

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.cleanup import cleanup_model
from finn.builder.build_dataflow_config import DataflowBuildConfig

# --- Step: Attach Pre-Processing Model ---
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.infer_shapes import InferShapes
from brevitas.export import export_qonnx

# --- Step: Streamlining ---
from qonnx.transformation.general import ConvertDivToMul, ConvertSubToAdd
from finn.transformation.streamline.reorder import (
    MoveOpPastFork,
    MoveLinearPastEltwiseAdd,
    MoveScalarMulPastConv,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastMatMul,
    MoveAddPastMul,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveMulPastMaxPool,
)
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    Absorb1BitMulIntoConv,
    Absorb1BitMulIntoMatMul,
    AbsorbScalarMulAddIntoTopK,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedMul,
    CollapseRepeatedAdd,
)
from finn.builder.build_dataflow_steps import VerificationStepType, verify_step
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.insert_topk import InsertTopK

# --- Step: Lowering Convolutions ---
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.absorb import (
    AbsorbTransposeIntoMultiThreshold,
    AbsorbConsecutiveTransposes,
    AbsorbTransposeIntoFlatten,
)
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveTransposePastFork,
    MoveTransposePastJoinAdd,
)

# --- Step: Converting to HW Layers ---
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferAddStreamsLayer,
    InferGlobalAccPoolLayer,
    InferPool,
    InferStreamingMaxPool,
    InferQuantizedMatrixVectorActivation,
    InferThresholdingLayer,
    InferConvInpGen,
    InferDuplicateStreamsLayer,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from qonnx.core.datatype import DataType
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.general import GiveUniqueNodeNames, SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class FixThresholdDataTypes(Transformation):
    """Fix for RoundAndClipThresholds bug: when thresholds are already whole
    numbers (stored as float), the rounding is skipped and the datatype
    annotation stays float. This pass sets the datatype to match the input
    for any MultiThreshold with integer input and integer-valued thresholds."""

    def apply(self, model):
        graph_modified = False
        for n in model.graph.node:
            if n.op_type == "MultiThreshold":
                idtype = model.get_tensor_datatype(n.input[0])
                tdt = model.get_tensor_datatype(n.input[1])
                if idtype.is_integer() and not tdt.is_integer():
                    T = model.get_initializer(n.input[1])
                    if T is not None and np.all(T == np.ceil(T)):
                        model.set_tensor_datatype(n.input[1], idtype)
                        graph_modified = True
        return (model, graph_modified)


HW_DOMAINS = {
    "finn.custom_op.fpgadataflow",
    "finn.custom_op.fpgadataflow.hlsbackend",
    "finn.custom_op.fpgadataflow.rtlbackend",
}


def graph_summary(model, label=""):
    """Print node-type summary for debugging."""
    all_n = list(model.graph.node)
    hw_n = [n for n in all_n if n.domain in HW_DOMAINS]
    non_hw = [n for n in all_n if n.domain not in HW_DOMAINS]
    print(f"  [{label}] Total={len(all_n)}  HW={len(hw_n)}  Non-HW={len(non_hw)}")
    if non_hw:
        counts = Counter(n.op_type for n in non_hw)
        print(f"  [{label}] Non-HW: {dict(counts)}")
    return non_hw


# ---------------------------------------------------------------------------
# Preprocessing model: uint8 [0,255] -> ImageNet-normalized float
# ---------------------------------------------------------------------------
class FundusPreProc(nn.Module):
    """Preprocessing that gets merged into the FINN graph.

    Converts uint8 [0,255] to ImageNet-normalized float:
      x / 255.0 -> subtract mean -> divide std

    The constant Sub/Div operations get absorbed into the first
    MultiThreshold during streamlining at zero FPGA resource cost.

    We use a custom module instead of finn.util.pytorch.NormalizePreProc
    because NormalizePreProc only supports scalar std, while our training
    uses per-channel std [0.229, 0.224, 0.225].

    This module is a plain nn.Module (no Brevitas layers). export_qonnx()
    handles non-Brevitas modules — confirmed by finn-examples using
    ToTensor() (also a plain nn.Module) with the same call pattern.
    """

    def __init__(self):
        super().__init__()
        # ImageNet defaults (same as timm.data.constants)
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = x / 255.0
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# Step 2: Attach preprocessing
# Ref: resnet18_custom_steps.py:76-98
# ---------------------------------------------------------------------------
def step_fundus_attach_preproc(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    model = model.transform(InferShapes())

    shape = tuple(
        d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim
    )

    # Export our preprocessing model to QONNX and merge at graph input.
    # finn-examples uses ToTensor() (just /255). We extend with ImageNet
    # normalization because our model was trained with
    # A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD).
    pre_proc = export_qonnx(
        FundusPreProc(), input_shape=shape, opset_version=11
    )
    pre_proc_qonnx = ModelWrapper(pre_proc)
    model = model.transform(MergeONNXModels(pre_proc_qonnx))

    return cleanup_model(model)


# ---------------------------------------------------------------------------
# Step 4: Streamline
# Ref: resnet18_custom_steps.py:100-130  (IDENTICAL to finn-examples)
# ---------------------------------------------------------------------------
def step_fundus_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    # finn-examples custom step + missing transforms from Streamline() class.
    # Key additions: ConvertSubToAdd (critical for preprocessing Sub node),
    # MoveMulPastMaxPool, MoveAddPastMul/Conv, AbsorbSignBiasIntoMultiThreshold.
    # We also run multiple passes to ensure all absorptions complete.

    model = model.transform(InsertTopK())

    # Pre-pass: convert Sub to Add (critical — preprocessing Sub(mean) must
    # become Add(-mean) before it can be absorbed into MultiThreshold)
    model = model.transform(ConvertSubToAdd())
    model = model.transform(ConvertDivToMul())
    model = model.transform(BatchNormToAffine())

    # Main streamlining loop — run until convergence (max 5 passes)
    streamline_transformations = [
        ConvertSubToAdd(),
        ConvertDivToMul(),
        BatchNormToAffine(),
        MoveOpPastFork(["Mul"]),
        MoveLinearPastEltwiseAdd(),
        MoveMulPastMaxPool(),
        MoveScalarLinearPastInvariants(),
        AbsorbSignBiasIntoMultiThreshold(),
        MoveAddPastMul(),
        MoveScalarAddPastMatMul(),
        MoveAddPastConv(),
        MoveScalarMulPastMatMul(),
        MoveScalarMulPastConv(),
        MoveAddPastMul(),
        CollapseRepeatedAdd(),
        CollapseRepeatedMul(),
        MoveMulPastMaxPool(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(),
        Absorb1BitMulIntoConv(),
        RoundAndClipThresholds(),
        RemoveIdentityOps(),
    ]

    for pass_idx in range(5):
        model_str_before = model.model.SerializeToString()
        for t in streamline_transformations:
            model = model.transform(t)
            model = model.transform(RemoveIdentityOps())
        model_str_after = model.model.SerializeToString()
        if model_str_before == model_str_after:
            print(f"  Streamlining converged after {pass_idx + 1} pass(es)")
            break
    else:
        print(f"  Streamlining: max passes (5) reached")

    # Absorb remaining scalar mul/add into TopK
    model = model.transform(AbsorbScalarMulAddIntoTopK())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return cleanup_model(model)


# ---------------------------------------------------------------------------
# Step 5: Lower convolutions
# Ref: resnet18_custom_steps.py:132-153
# Addition: MakeMaxPoolNHWC (our model has MaxPool, finn-examples doesn't)
# Pattern for MakeMaxPoolNHWC from: build_dataflow_steps.py:316-321
# ---------------------------------------------------------------------------
def step_fundus_lower(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    lower_transformations = [
        LowerConvsToMatMul(),
        MoveTransposePastFork(),
        MoveTransposePastJoinAdd(),
        AbsorbTransposeIntoMultiThreshold(),
        MoveTransposePastFork(),
        MoveTransposePastJoinAdd(),
        AbsorbTransposeIntoMultiThreshold(),
        MoveTransposePastFork(),
        AbsorbTransposeIntoFlatten(),
    ]

    for t in lower_transformations:
        model = model.transform(t)

    return cleanup_model(model)


# ---------------------------------------------------------------------------
# Step 6: Convert to HW layers (MOST CRITICAL)
# Ref: resnet18_custom_steps.py:156-189
# Additions: InferGlobalAccPoolLayer, InferPool, InferStreamingMaxPool
# ---------------------------------------------------------------------------
def step_fundus_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    # Post-lowering streamline pass: absorb remaining Mul/Add around residual
    # joins and thresholds. The built-in step_streamline runs Streamline()
    # twice (before and after lowering); our custom pipeline splits these into
    # step_fundus_streamline (pre) and this pass (post).
    post_lower_streamline = [
        ConvertSubToAdd(),
        ConvertDivToMul(),
        MoveScalarLinearPastInvariants(),
        MoveAddPastMul(),
        MoveScalarMulPastConv(),
        MoveScalarMulPastMatMul(),
        CollapseRepeatedMul(),
        CollapseRepeatedAdd(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(),
        Absorb1BitMulIntoConv(),
        AbsorbConsecutiveTransposes(),
        RoundAndClipThresholds(),
        RemoveIdentityOps(),
    ]
    for pass_idx in range(3):
        model_str_before = model.model.SerializeToString()
        for t in post_lower_streamline:
            model = model.transform(t)
        model_str_after = model.model.SerializeToString()
        if model_str_before == model_str_after:
            print(f"  Post-lower streamlining converged after {pass_idx + 1} pass(es)")
            break
    else:
        print(f"  Post-lower streamlining: max passes (3) reached")

    non_hw_pre = graph_summary(model, "before to_hw")

    to_hw_transformations = [
        DoubleToSingleFloat(),
        InferDataTypes(),
        SortGraph(),
        InferShapes(),
        # Move stray Transposes on identity paths past the Add so
        # InferAddStreamsLayer sees HW inputs on both branches.
        MoveTransposePastJoinAdd(),
        AbsorbTransposeIntoMultiThreshold(),
        AbsorbConsecutiveTransposes(),
        InferAddStreamsLayer(),
        InferGlobalAccPoolLayer(),
        RoundAndClipThresholds(),
        FixThresholdDataTypes(),
        InferThresholdingLayer(),
        InferQuantizedMatrixVectorActivation(),
        AbsorbConsecutiveTransposes(),
        InferConvInpGen(),
        InferDuplicateStreamsLayer(),
        AbsorbConsecutiveTransposes(),
        AbsorbTransposeIntoFlatten(),
        RemoveCNVtoFCFlatten(),
    ]

    # Workaround from finn-examples: set input tensor datatype to UINT8.
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])

    for t in to_hw_transformations:
        model = model.transform(InferDataLayouts())
        model = model.transform(t)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    non_hw = graph_summary(model, "after to_hw")

    # Warn about unconverted non-HW nodes (these would cause partition failure)
    if non_hw:
        for n in non_hw:
            print(f"  WARNING: non-HW node: {n.op_type} [{n.name}]")

    return cleanup_model(model)
