"""
Custom FINN build steps for the fundus QAT CustomSmallNet.

Based on custom_steps_resnet18.py, simplified:
  - No residual connections → no InferAddStreamsLayer, InferDuplicateStreamsLayer,
    MoveLinearPastEltwiseAdd, MoveTransposePastJoinAdd, MoveOpPastFork
  - Has MaxPool → MoveMulPastMaxPool, InferStreamingMaxPool needed
  - Has GlobalAvgPool (TruncAvgPool2d) → InferGlobalAccPoolLayer, InferPool needed

Must run inside the FINN Docker container.
"""

from custom_steps_resnet18 import (
    step_fundus_attach_preproc,  # noqa: F401 — re-exported for build_custom_net.py
    graph_summary,
    FixThresholdDataTypes,
)

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import VerificationStepType, verify_step


class AbsorbScalarMulIntoMatMul(Transformation):
    """Walk backwards from MatMul through invariant ops (GlobalAveragePool,
    Reshape, Flatten, Transpose) to find a scalar Mul, then absorb it
    into the MatMul weight matrix.

    The QuantReLU dequant scale is a 4D Mul (e.g. shape [1,1,1,1]).
    It can't move past Flatten (4D→2D shape change) via standard transforms.
    This transform handles the full chain regardless of intermediate ops.
    """

    def _find_producer(self, model, tensor_name):
        """Find the node that produces the given tensor."""
        for n in model.graph.node:
            if tensor_name in n.output:
                return n
        return None

    def apply(self, model):
        graph_modified = False
        invariant_ops = {"Reshape", "Flatten", "GlobalAveragePool", "Transpose"}

        for n in model.graph.node:
            if n.op_type != "MatMul":
                continue
            W = model.get_initializer(n.input[1])
            if W is None:
                continue

            # Walk backwards from MatMul input[0] through invariant ops
            current_input = n.input[0]
            while True:
                pred = self._find_producer(model, current_input)
                if pred is None:
                    break
                if pred.op_type == "Mul":
                    # Found Mul — check for scalar constant
                    c = model.get_initializer(pred.input[1])
                    act_inp = pred.input[0]
                    if c is None:
                        c = model.get_initializer(pred.input[0])
                        act_inp = pred.input[1]
                    if c is None:
                        break
                    if c.size != 1 and len(np.unique(c)) != 1:
                        break
                    scalar = float(c.flatten()[0])
                    # Absorb scalar into MatMul weights
                    model.set_initializer(n.input[1], (W * scalar).astype(W.dtype))
                    # Bypass Mul: redirect all consumers of Mul's output
                    mul_out = pred.output[0]
                    for consumer in model.graph.node:
                        for i, inp in enumerate(consumer.input):
                            if inp == mul_out:
                                consumer.input[i] = act_inp
                    model.graph.node.remove(pred)
                    graph_modified = True
                    print(f"  AbsorbScalarMulIntoMatMul: absorbed Mul({scalar:.6f}) into {n.name}")
                    break
                elif pred.op_type in invariant_ops:
                    current_input = pred.input[0]
                else:
                    break

        return (model, graph_modified)

# --- Step: Streamlining ---
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveUniqueNodeNames,
    SortGraph,
)
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import (
    MoveScalarMulPastConv,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastMatMul,
    MoveAddPastMul,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveMulPastMaxPool,
    MoveTransposePastScalarMul,
    MoveFlattenPastAffine,
    MoveFlattenPastTopK,
)
from qonnx.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from qonnx.transformation.general import GiveReadableTensorNames
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    Absorb1BitMulIntoConv,
    Absorb1BitMulIntoMatMul,
    AbsorbScalarMulAddIntoTopK,
    AbsorbTransposeIntoMultiThreshold,
    AbsorbConsecutiveTransposes,
    AbsorbTransposeIntoFlatten,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedMul,
    CollapseRepeatedAdd,
)
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.insert_topk import InsertTopK
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

# --- Step: Lowering Convolutions ---
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC

# --- Step: Converting to HW Layers ---
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferGlobalAccPoolLayer,
    InferPool,
    InferStreamingMaxPool,
    InferQuantizedMatrixVectorActivation,
    InferThresholdingLayer,
    InferConvInpGen,
    InferLabelSelectLayer,
    InferChannelwiseLinearLayer,
)
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_shapes import InferShapes
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from qonnx.util.cleanup import cleanup_model


# ---------------------------------------------------------------------------
# Step: Streamline
# Based on step_fundus_streamline from ResNet18, minus residual-related transforms.
# ---------------------------------------------------------------------------
def step_custom_net_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    model = model.transform(InsertTopK())
    model = model.transform(Streamline())

    # After Streamline(), the QuantReLU scale (4D Mul, e.g. [1,1,1,1]) is
    # moved past GlobalAveragePool but stuck before Flatten (can't broadcast
    # 4D constant to 2D). Absorb it directly into the FC MatMul weights.
    model = model.transform(AbsorbScalarMulIntoMatMul())

    # Second Streamline() pass to clean up patterns exposed by absorption.
    model = model.transform(Streamline())

    model = model.transform(AbsorbScalarMulAddIntoTopK())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return cleanup_model(model)


# ---------------------------------------------------------------------------
# Step: Lower convolutions
# Based on step_fundus_lower from ResNet18, minus MoveTransposePastJoinAdd
# (no residual joins). Includes MakeMaxPoolNHWC for MaxPool layers.
# ---------------------------------------------------------------------------
def step_custom_net_lower(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    lower_transformations = [
        LowerConvsToMatMul(),
        MakeMaxPoolNHWC(),
        AbsorbTransposeIntoMultiThreshold(),
        AbsorbTransposeIntoMultiThreshold(),  # second pass catches remaining
        AbsorbTransposeIntoFlatten(),
    ]

    for t in lower_transformations:
        model = model.transform(t)

    return cleanup_model(model)


# ---------------------------------------------------------------------------
# Step: Convert to HW layers
# Based on step_fundus_to_hw from ResNet18, simplified (no residual streams).
# ---------------------------------------------------------------------------
def step_custom_net_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    # Post-lowering streamline pass: absorb remaining Mul/Add around thresholds
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

    # Set input tensor datatype to UINT8 (preprocessing outputs uint8)
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    # Absorb input Transpose into MultiThreshold BEFORE converting to HW.
    # Without this, a non-HW Transpose at both edges causes a partition cycle.
    model = model.transform(AbsorbTransposeIntoMultiThreshold())

    non_hw_pre = graph_summary(model, "before to_hw")

    to_hw_transformations = [
        DoubleToSingleFloat(),
        InferDataTypes(),
        SortGraph(),
        InferShapes(),
        AbsorbConsecutiveTransposes(),
        InferGlobalAccPoolLayer(),
        # InferGlobalAccPoolLayer converts GlobalAveragePool to GlobalAccPool HW
        # (computes SUM, not average) and inserts a 1/N Mul to preserve correctness.
        # Absorb that Mul into the first FC's MatMul weights before HW inference.
        AbsorbScalarMulIntoMatMul(),
        InferPool(),
        InferStreamingMaxPool(),
        RoundAndClipThresholds(),
        FixThresholdDataTypes(),
        InferThresholdingLayer(),
        InferConvInpGen(),
        InferQuantizedMatrixVectorActivation(),
        InferChannelwiseLinearLayer(),
        InferLabelSelectLayer(),
        AbsorbConsecutiveTransposes(),
        AbsorbTransposeIntoFlatten(),
        RemoveCNVtoFCFlatten(),
    ]

    for t in to_hw_transformations:
        model = model.transform(InferDataLayouts())
        model = model.transform(t)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    non_hw = graph_summary(model, "after to_hw")

    if non_hw:
        for n in non_hw:
            print(f"  WARNING: non-HW node: {n.op_type} [{n.name}]")

    return cleanup_model(model)
