"""
Custom FINN build steps for the fundus QAT test_resnet.

Hybrid of ResNet18 pipeline (residual connections) and CustomSmallNet pipeline
(GlobalAvgPool handling with AbsorbScalarMulIntoMatMul).

Differences from ResNet18 pipeline:
  - Stem MaxPool present → MoveMulPastMaxPool in streamline, InferPool in to-HW step
  - GlobalAvgPool → needs InferGlobalAccPoolLayer + AbsorbScalarMulIntoMatMul
  - Downsample TruncAvgPool2d → ConvertAvgPoolTruncToQuantAvgPool + InferPool
  - Otherwise identical residual handling (MoveLinearPastEltwiseAdd, InferAddStreamsLayer, etc.)

Key transform added over ResNet18 pipeline:
  ConvertAvgPoolTruncToQuantAvgPool — handles the newer Brevitas 6-input Trunc format
  that the built-in AvgPoolAndTruncToQuantAvgPool does not support.

Must run inside the FINN Docker container.
"""

from onnx import TensorProto, helper
from qonnx.util.basic import get_by_name

from custom_steps_resnet18 import (
    step_fundus_attach_preproc,  # noqa: F401 — re-exported for build_test_resnet.py
    graph_summary,
    FixThresholdDataTypes,
    FundusPreProc,  # noqa: F401
)
from custom_steps_custom_net import (
    AbsorbScalarMulIntoMatMul,
)

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.base import Transformation
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import VerificationStepType, verify_step

# --- Step: Streamlining ---
from qonnx.transformation.general import ConvertDivToMul, ConvertSubToAdd
from finn.transformation.streamline.reorder import (
    MoveOpPastFork,
    MoveLinearPastEltwiseAdd,
    MoveMulPastMaxPool,
    MoveScalarMulPastConv,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastMatMul,
    MoveAddPastMul,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
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
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

# --- Step: Lowering Convolutions ---
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.absorb import (
    AbsorbTransposeIntoMultiThreshold,
    AbsorbConsecutiveTransposes,
    AbsorbTransposeIntoFlatten,
)
from finn.transformation.streamline.reorder import (
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
    InferChannelwiseLinearLayer,
    InferLabelSelectLayer,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.general import GiveUniqueNodeNames, SortGraph
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten


# ---------------------------------------------------------------------------
# Custom transform: AveragePool → Trunc → QuantAvgPool2d
#
# The built-in AvgPoolAndTruncToQuantAvgPool expects:
#   AveragePool → Mul(k²) → Trunc   (older Brevitas, 5-input Trunc)
#
# Brevitas TruncAvgPool2d in this codebase exports:
#   AveragePool → Trunc              (6-input Trunc, no k² Mul)
#
# The 6-input Trunc format (from BrevitasTruncFn.symbolic):
#   input[0] = x (value)
#   input[1] = input_scale
#   input[2] = zero_point
#   input[3] = input_bit_width
#   input[4] = output_scale
#   input[5] = output_bit_width
#   attr rounding_mode, signed, narrow
#
# This transform converts AveragePool → Trunc to:
#   Div(output_scale) → QuantAvgPool2d → Mul(output_scale)
#
# During streamlining the upstream Mul(input_scale) → Div(output_scale) collapses
# to Mul(1) → removed, leaving QuantAvgPool2d with integer input so InferPool fires.
# (input_scale == output_scale for TruncAvgPool2d: "preserves the scale of the input")
# ---------------------------------------------------------------------------
class ConvertAvgPoolTruncToQuantAvgPool(Transformation):
    """
    Convert AveragePool → Trunc (6-input Brevitas format) to QuantAvgPool2d.

    Run this at the START of the streamline step, before ConvertDivToMul consumes
    the Div node we insert.  The Div(out_scale) → QuantAvgPool2d → Mul(out_scale)
    pattern is later cleaned up by streamlining:
      Mul(in_s) → Div(out_s=in_s) → [CollapseRepeatedMul] → Mul(1) → [RemoveIdentityOps]
    leaving QuantAvgPool2d with a UINT8 integer input, which InferPool can handle.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0

        for node in graph.node:
            node_ind += 1
            if node.op_type != "AveragePool":
                continue

            # Require direct AveragePool → Trunc (no Mul between them)
            successors = model.find_direct_successors(node)
            if successors is None or len(successors) != 1:
                continue
            trunc_node = successors[0]
            if trunc_node.op_type != "Trunc":
                continue

            # --- AveragePool: square kernel/stride, no padding ---
            k_attr = get_by_name(node.attribute, "kernel_shape")
            s_attr = get_by_name(node.attribute, "strides")
            if k_attr is None or len(k_attr.ints) != 2:
                continue
            if s_attr is None or len(s_attr.ints) != 2:
                continue
            k_h, k_w = int(k_attr.ints[0]), int(k_attr.ints[1])
            s_h, s_w = int(s_attr.ints[0]), int(s_attr.ints[1])
            if k_h != k_w or s_h != s_w:
                continue
            k_s, s_s = k_h, s_h

            pads_attr = get_by_name(node.attribute, "pads")
            if pads_attr is not None and any(int(p) != 0 for p in pads_attr.ints):
                continue

            # --- Trunc: must be 6-input (new Brevitas format) ---
            if len(trunc_node.input) != 6:
                continue

            zero_pt     = model.get_initializer(trunc_node.input[2])
            in_bits_t   = model.get_initializer(trunc_node.input[3])
            out_scale_t = model.get_initializer(trunc_node.input[4])
            out_bits_t  = model.get_initializer(trunc_node.input[5])

            if any(t is None for t in [zero_pt, in_bits_t, out_scale_t, out_bits_t]):
                continue
            if float(zero_pt.flatten()[0]) != 0:
                continue

            ibits = int(in_bits_t.flatten()[0])
            obits = int(out_bits_t.flatten()[0])

            rounding_attr = get_by_name(trunc_node.attribute, "rounding_mode")
            if rounding_attr is None or rounding_attr.s.upper() != b"FLOOR":
                continue

            signed_attr = get_by_name(trunc_node.attribute, "signed")
            signed = int(signed_attr.i) if signed_attr is not None else 0

            # --- Build Div → QuantAvgPool2d → Mul ---
            running_idx = node_ind
            avg_input    = node.input[0]
            trunc_output = trunc_node.output[0]

            # Div(output_scale): normalise scaled float → integer
            div_scale_name = model.make_new_valueinfo_name()
            model.set_initializer(div_scale_name, out_scale_t)
            norm_name = model.make_new_valueinfo_name()
            graph.value_info.append(
                helper.make_tensor_value_info(norm_name, TensorProto.FLOAT, None)
            )
            div_node = helper.make_node(
                "Div", [avg_input, div_scale_name], [norm_name]
            )

            # QuantAvgPool2d
            pool_out_name = model.make_new_valueinfo_name()
            graph.value_info.append(
                helper.make_tensor_value_info(pool_out_name, TensorProto.FLOAT, None)
            )
            quant_avg_pool_node = helper.make_node(
                "QuantAvgPool2d",
                [norm_name],
                [pool_out_name],
                domain="qonnx.custom_op.general",
                stride=s_s,
                kernel=k_s,
                ibits=ibits,
                obits=obits,
                signed=signed,
                data_layout="NCHW",
            )

            # Mul(output_scale): rescale integer → scaled float
            mul_scale_name = model.make_new_valueinfo_name()
            model.set_initializer(mul_scale_name, out_scale_t)
            mul_node = helper.make_node(
                "Mul", [pool_out_name, mul_scale_name], [trunc_output]
            )

            graph.node.insert(running_idx,     div_node)
            graph.node.insert(running_idx + 1, quant_avg_pool_node)
            graph.node.insert(running_idx + 2, mul_node)

            graph.node.remove(node)
            graph.node.remove(trunc_node)

            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

            return model, True

        return model, False


# ---------------------------------------------------------------------------
# Step 4: Streamline
# ---------------------------------------------------------------------------
def step_test_resnet_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    # Convert TruncAvgPool2d export (AveragePool → Trunc) to QuantAvgPool2d
    # MUST run before ConvertDivToMul and before any Mul-reorder transforms.
    model = model.transform(ConvertAvgPoolTruncToQuantAvgPool())

    model = model.transform(InsertTopK())

    model = model.transform(ConvertSubToAdd())
    model = model.transform(ConvertDivToMul())
    model = model.transform(BatchNormToAffine())

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

    model = model.transform(AbsorbScalarMulAddIntoTopK())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return cleanup_model(model)


# ---------------------------------------------------------------------------
# Step 5: Lower convolutions
# ---------------------------------------------------------------------------
def step_test_resnet_lower(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    lower_transformations = [
        LowerConvsToMatMul(),
        # Keep stem MaxPool in place until to_hw.
        # Rewriting it here lifts the first residual fork before the pool.
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
# Step 6: Convert to HW layers
# ---------------------------------------------------------------------------
def step_test_resnet_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    # Post-lower streamline
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
        # Residual handling: move transposes past Add joins
        MoveTransposePastJoinAdd(),
        AbsorbTransposeIntoMultiThreshold(),
        AbsorbConsecutiveTransposes(),
        # Convert MaxPool BEFORE InferAddStreamsLayer.
        # InferAddStreamsLayer checks that Add inputs have integer FINN DataTypes.
        # For layer1 (identity skip), the skip tensor is the stem MaxPool output.
        # If MaxPoolNHWC(k=3,s=2) is still non-HW when InferAddStreamsLayer runs,
        # InferDataTypes cannot propagate integer dtype through it → layer1 Add stays non-HW.
        # InferStreamingMaxPool handles k=s only; InferPool handles k≠s (stem k=3,s=2)
        # via Im2Col+Pool_Batch and also converts QuantAvgPool2d from downsample paths.
        InferStreamingMaxPool(),
        InferPool(),
        # InferPool on NCHW MaxPool preserves the original layout by inserting
        # output Transpose nodes. For the stem pool, those transposes sit right
        # before the first residual fork. Push them into the branches so inverse
        # transpose pairs can collapse before partitioning.
        MoveTransposePastFork(),
        AbsorbConsecutiveTransposes(),
        # Convert residual Add to HW (now all pool outputs have integer FINN DataTypes)
        InferAddStreamsLayer(),
        # GlobalAvgPool -> GlobalAccPool HW (inserts 1/N Mul)
        InferGlobalAccPoolLayer(),
        # Absorb the 1/N Mul into FC MatMul weights
        AbsorbScalarMulIntoMatMul(),
        RoundAndClipThresholds(),
        FixThresholdDataTypes(),
        InferThresholdingLayer(),
        InferConvInpGen(),
        InferQuantizedMatrixVectorActivation(),
        # Duplicate streams for residual forks
        InferDuplicateStreamsLayer(),
        InferChannelwiseLinearLayer(),
        InferLabelSelectLayer(),
        AbsorbConsecutiveTransposes(),
        AbsorbTransposeIntoFlatten(),
        RemoveCNVtoFCFlatten(),
    ]

    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])

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
