"""
Custom FINN build steps for the fundus QAT MobileNetV1.

Based on: ignore/finn-examples-rn18/build/mobilenet-v1/custom_steps.py

Must run inside the FINN Docker container.
"""

from custom_steps_resnet18 import (
    step_fundus_attach_preproc,  # noqa: F401 — re-exported for build_mobilenet.py
    graph_summary,
    FixThresholdDataTypes,
)

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from finn.builder.build_dataflow_config import DataflowBuildConfig

# --- Step: Streamlining ---
from finn.transformation.streamline import Streamline
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul
from qonnx.transformation.remove import RemoveIdentityOps
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from qonnx.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_datatypes import InferDataTypes

# --- Step: Lowering Convolutions ---
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul

# --- Step: Converting to HW Layers ---
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from qonnx.transformation.infer_shapes import InferShapes


# ---------------------------------------------------------------------------
# Step: Streamline
# Ref: finn-examples custom_steps.py:53-75
# ---------------------------------------------------------------------------
def step_mobilenet_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    model = model.transform(InsertTopK())
    model = model.transform(Streamline())

    additional_streamline_transformations = [
        DoubleToSingleFloat(),
        reorder.MoveMulPastDWConv(),
        absorb.AbsorbMulIntoMultiThreshold(),
        ChangeDataLayoutQuantAvgPool2d(),
        InferDataLayouts(),
        reorder.MoveTransposePastScalarMul(),
        absorb.AbsorbTransposeIntoFlatten(),
        reorder.MoveFlattenPastAffine(),
        reorder.MoveFlattenPastTopK(),
        reorder.MoveScalarMulPastMatMul(),
        CollapseRepeatedMul(),
        RemoveIdentityOps(),
        RoundAndClipThresholds(),
    ]
    for trn in additional_streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())

    return model


# ---------------------------------------------------------------------------
# Step: Lower convolutions
# Ref: finn-examples custom_steps.py:78-87
# ---------------------------------------------------------------------------
def step_mobilenet_lower_convs(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferDataLayouts())

    return model


# ---------------------------------------------------------------------------
# Step: Convert to HW layers (Zynq, separate thresholding)
# Ref: finn-examples custom_steps.py:127-138
# ---------------------------------------------------------------------------
def step_mobilenet_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    # Absorb input Transpose into MultiThreshold BEFORE converting to HW.
    # Without this, a non-HW Transpose at both edges causes a partition cycle.
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())

    graph_summary(model, "before to_hw")

    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(FixThresholdDataTypes())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(absorb.AbsorbTransposeIntoFlatten())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    graph_summary(model, "after to_hw")

    return model
