"""
Custom FINN build steps for the fundus QAT test_resnet.

Hybrid of ResNet18 pipeline (residual connections) and CustomSmallNet pipeline
(GlobalAvgPool handling with custom QuantAvgPool conversion and HW lowering).

Differences from ResNet18 pipeline:
  - Stem MaxPool present → MoveMulPastMaxPool in streamline, InferPool in to-HW step
  - GlobalAvgPool → needs InferGlobalAccPoolLayer
  - Downsample TruncAvgPool2d → ConvertAvgPoolTruncToQuantAvgPool + InferPool
  - Otherwise identical residual handling (MoveLinearPastEltwiseAdd, InferAddStreamsLayer, etc.)

Key transform added over ResNet18 pipeline:
  ConvertAvgPoolTruncToQuantAvgPool — handles the newer Brevitas 6-input Trunc format
  that the built-in AvgPoolAndTruncToQuantAvgPool does not support.

Must run inside the FINN Docker container.
"""

import math
import numpy as np
from onnx import TensorProto, helper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import get_by_name

from custom_steps_resnet18 import (
    graph_summary,
    FixThresholdDataTypes,
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


class CollapseTransposeWrappedMul(Transformation):
    """Collapse inverse Transpose -> Mul -> Transpose patterns.

    This targets the residual branch pattern that appears *before*
    InferChannelwiseLinearLayer:

      x --Transpose(P)--> x' --Mul(const)--> y' --Transpose(P^-1)--> y

    where the Mul constant is channelwise/broadcastable. We rewrite it to:

      x --Mul(const_permuted)--> y

    by permuting the constant tensor into the pre-transpose layout. This keeps
    the branch in the same layout as its neighbors and avoids creating
    Transpose/ChannelwiseOp/Transpose wrappers later on.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        nodes = [n for n in graph.node]

        for node in nodes:
            if node.op_type != "Transpose":
                continue

            consumers = model.find_consumers(node.output[0])
            if consumers is None or len(consumers) != 1:
                continue

            mul_node = consumers[0]
            if mul_node.op_type != "Mul" or model.is_join_node(mul_node):
                continue

            if len(mul_node.input) < 2:
                continue

            const_name = mul_node.input[1]
            const_val = model.get_initializer(const_name)
            if const_val is None:
                continue

            second = model.find_consumer(mul_node.output[0])
            if second is None or second.op_type != "Transpose":
                continue

            perm0_attr = get_by_name(node.attribute, "perm")
            perm1_attr = get_by_name(second.attribute, "perm")
            if perm0_attr is None or perm1_attr is None:
                continue

            perm0 = list(perm0_attr.ints)
            perm1 = list(perm1_attr.ints)
            if len(perm0) != len(perm1):
                continue
            inv0 = [perm0.index(i) for i in range(len(perm0))]
            if perm1 != inv0:
                continue

            start_name = node.input[0]
            end_name = second.output[0]
            start_shape = model.get_tensor_shape(start_name)
            end_shape = model.get_tensor_shape(end_name)
            if start_shape is None or end_shape is None or tuple(start_shape) != tuple(end_shape):
                continue

            new_const = const_val
            if const_val.ndim == len(perm0):
                try:
                    new_const = np.transpose(const_val, perm0)
                except Exception:
                    continue
            elif const_val.ndim == 1:
                # 1-D channelwise params already broadcast on the last axis.
                new_const = const_val
            elif np.prod(const_val.shape) == 1:
                new_const = const_val
            else:
                continue

            mul_node.input[0] = start_name
            model.set_initializer(const_name, new_const)
            mul_node.output[0] = end_name
            model.set_tensor_shape(end_name, start_shape)
            start_layout = model.get_tensor_layout(start_name)
            if start_layout is not None:
                model.set_tensor_layout(end_name, start_layout)

            graph_modified = True
            graph.node.remove(node)
            graph.node.remove(second)

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataLayouts())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class MoveTransposePastDuplicateStreams(Transformation):
    """Push Transpose through DuplicateStreams with shape-aware rewiring.

    After InferDuplicateStreamsLayer we often see:

      x --Transpose(P)--> y --DuplicateStreams--> y0, y1, ...

    This pass rewrites it to:

      x --DuplicateStreams--> x0, x1, ...
      x_i --Transpose(P)--> y_i

    while updating DuplicateStreams shape attributes to match ``x``. This lets
    branch-local inverse transpose pairs collapse cleanly on consumers.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        nodes = list(graph.node)

        for node in nodes:
            if node.op_type != "Transpose":
                continue

            consumers = model.find_consumers(node.output[0])
            if consumers is None or len(consumers) != 1:
                continue

            dup_node = consumers[0]
            if dup_node.op_type != "DuplicateStreams":
                continue

            perm_attr = get_by_name(node.attribute, "perm")
            if perm_attr is None:
                continue
            perm = list(perm_attr.ints)

            start_name = node.input[0]
            trans_out = node.output[0]
            start_shape = model.get_tensor_shape(start_name)
            trans_shape = model.get_tensor_shape(trans_out)
            if start_shape is None or trans_shape is None:
                continue

            dup_outputs = list(dup_node.output)
            raw_outputs = []
            for _ in dup_outputs:
                raw_name = model.make_new_valueinfo_name()
                raw_vi = helper.make_tensor_value_info(raw_name, TensorProto.FLOAT, start_shape)
                graph.value_info.append(raw_vi)
                raw_outputs.append(raw_name)

            dup_node.input[0] = start_name
            dup_node.output[:] = raw_outputs
            dup_inst = getCustomOp(dup_node)
            dup_inst.set_nodeattr("NumChannels", int(start_shape[-1]))
            dup_inst.set_nodeattr("numInputVectors", list(start_shape[:-1]))

            start_layout = model.get_tensor_layout(start_name)
            trans_layout = model.get_tensor_layout(trans_out)

            dup_index = next(i for i, n in enumerate(graph.node) if n is dup_node)
            insert_index = dup_index + 1
            for raw_name, old_out in zip(raw_outputs, dup_outputs):
                trans_node = helper.make_node(
                    "Transpose",
                    [raw_name],
                    [old_out],
                    perm=perm,
                )
                graph.node.insert(insert_index, trans_node)
                insert_index += 1
                model.set_tensor_shape(raw_name, start_shape)
                model.set_tensor_shape(old_out, trans_shape)
                if start_layout is not None:
                    model.set_tensor_layout(raw_name, start_layout)
                if trans_layout is not None:
                    model.set_tensor_layout(old_out, trans_layout)

            graph.node.remove(node)
            graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataLayouts())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class MoveSignMulPastThresholding(Transformation):
    """Rewrite sign-only Mul -> Thresholding into Thresholding -> linear ops.

    For per-channel sign masks A in {-1, +1}, we use:

      count((-x) >= T) = B - count(x >= (-reverse(T) + 1))

    where B is the number of threshold steps. For channels with A=+1 the
    thresholding stays unchanged. For channels with A=-1 we update the
    threshold vector and then emit a post-threshold linear correction:

      y = sign * y' + bias

    with sign in {-1, +1} and bias in {0, B}. The later
    InferChannelwiseLinearLayer pass can convert those post-threshold linear
    ops into HW ChannelwiseOp nodes.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in list(graph.node):
            if n.op_type != "Mul" or model.is_fork_node(n) or model.is_join_node(n):
                continue

            A = model.get_initializer(n.input[1])
            if A is None:
                continue

            consumer = model.find_consumer(n.output[0])
            if consumer is None or consumer.op_type != "Thresholding":
                continue

            actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
            is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
            is_1d = actual_ndims == 1
            if not (is_scalar or is_1d):
                continue

            sign_vec = A.reshape(-1).astype(np.int64)
            if not np.isin(sign_vec, [-1.0, 1.0]).all():
                continue

            threshold_name = consumer.input[1]
            T = model.get_initializer(threshold_name)
            if T is None or T.ndim != 2:
                continue

            num_ch, num_steps = T.shape
            if sign_vec.size == 1:
                sign_vec = np.full((num_ch,), sign_vec.item(), dtype=np.int64)
            elif sign_vec.size != num_ch:
                continue

            Tnew = np.array(T, copy=True)
            bias_vec = np.zeros((num_ch,), dtype=np.int64)
            for ch in range(num_ch):
                if sign_vec[ch] < 0:
                    Tnew[ch] = -T[ch][::-1] + 1
                    bias_vec[ch] = int(num_steps)

            model.set_initializer(threshold_name, Tnew)
            try:
                th_inst = getCustomOp(consumer)
                th_inst.minimize_accumulator_width(model)
            except Exception:
                pass

            start_name = n.input[0]
            end_name = consumer.output[0]
            out_shape = model.get_tensor_shape(end_name)
            out_layout = model.get_tensor_layout(end_name)
            out_dt = model.get_tensor_datatype(end_name)
            if out_dt is None or not out_dt.is_integer():
                continue

            th_out = model.make_new_valueinfo_name()
            mul_out = model.make_new_valueinfo_name()
            graph.value_info.append(helper.make_tensor_value_info(th_out, TensorProto.INT64, out_shape))
            graph.value_info.append(helper.make_tensor_value_info(mul_out, TensorProto.INT64, out_shape))
            if out_shape is not None:
                model.set_tensor_shape(th_out, out_shape)
                model.set_tensor_shape(mul_out, out_shape)
            if out_layout is not None:
                model.set_tensor_layout(th_out, out_layout)
                model.set_tensor_layout(mul_out, out_layout)
            model.set_tensor_datatype(th_out, out_dt)
            model.set_tensor_datatype(mul_out, DataType["INT32"])

            sign_name = model.make_new_valueinfo_name()
            bias_name = model.make_new_valueinfo_name()
            model.set_initializer(sign_name, sign_vec)
            model.set_initializer(bias_name, bias_vec)
            model.set_tensor_datatype(sign_name, DataType["INT2"])
            model.set_tensor_datatype(bias_name, DataType["UINT32"])

            consumer.input[0] = start_name
            consumer.output[0] = th_out

            mul_node = helper.make_node("Mul", [th_out, sign_name], [mul_out], name=n.name + "_post")
            add_node = helper.make_node("Add", [mul_out, bias_name], [end_name], name=n.name + "_bias")

            consumer_index = next(i for i, node in enumerate(graph.node) if node is consumer)
            graph.node.insert(consumer_index + 1, mul_node)
            graph.node.insert(consumer_index + 2, add_node)
            graph.node.remove(n)
            graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataLayouts())
        return (model, graph_modified)


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
        MoveMulPastMaxPool(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(),
        Absorb1BitMulIntoConv(),
        RoundAndClipThresholds(),
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
def _get_test_resnet_to_hw_transformations():
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
        # Keep the post-GAP scalar separate in the test_resnet FINN flow.
        # Absorbing it into the FC MatMul weights can turn the first FC MVAU
        # weights into FLOAT32, which breaks the streamed HLS MVAU path
        # (internal_decoupled mem_mode) during IP generation.
        RoundAndClipThresholds(),
        FixThresholdDataTypes(),
        InferThresholdingLayer(),
        InferConvInpGen(),
        InferQuantizedMatrixVectorActivation(),
        # Duplicate streams for residual forks
        InferDuplicateStreamsLayer(),
        MoveTransposePastDuplicateStreams(),
        MoveTransposePastFork(),
        AbsorbConsecutiveTransposes(),
        CollapseTransposeWrappedMul(),
        MoveSignMulPastThresholding(),
        InferChannelwiseLinearLayer(),
        InferPool(),
        MoveTransposePastFork(),
        AbsorbConsecutiveTransposes(),
        InferConvInpGen(),
        InferLabelSelectLayer(),
        AbsorbConsecutiveTransposes(),
        AbsorbTransposeIntoFlatten(),
        RemoveCNVtoFCFlatten(),
    ]
    return to_hw_transformations


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

    to_hw_transformations = _get_test_resnet_to_hw_transformations()

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


def _get_raw_nodeattr(node, attr_name, default=None):
    attr = get_by_name(node.attribute, attr_name)
    if attr is None:
        return default
    value = helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode()
    return value


def _set_raw_nodeattr(node, attr_name, value):
    old_attr = get_by_name(node.attribute, attr_name)
    if old_attr is not None:
        node.attribute.remove(old_attr)
    node.attribute.append(helper.make_attribute(attr_name, value))


def _safe_fifo_width(node):
    folded_shape = list(_get_raw_nodeattr(node, "folded_shape", []) or [])
    dtype_name = _get_raw_nodeattr(node, "dataType", "")
    if len(folded_shape) == 0 or dtype_name == "":
        return 0
    width = int(folded_shape[-1]) * DataType[dtype_name].bitwidth()
    # AXI data FIFOs are configured in whole TDATA bytes in FINN's IPI backend.
    return int(math.ceil(width / 8.0) * 8) if width > 0 else 0


def _safe_fifo_depth(node):
    depth = int(_get_raw_nodeattr(node, "depth", 0) or 0)
    if _get_raw_nodeattr(node, "impl_style", "") == "vivado" and depth > 0:
        return 1 << (depth - 1).bit_length()
    return depth


def _estimate_fifo_bram18_sites(depth, width):
    if depth <= 0 or width <= 0:
        return 0
    if width == 1:
        return int(math.ceil(depth / 16384.0))
    if width == 2:
        return int(math.ceil(depth / 8192.0))
    if width <= 4:
        return int(math.ceil(depth / 4096.0) * math.ceil(width / 4.0))
    if width <= 9:
        return int(math.ceil(depth / 2048.0) * math.ceil(width / 9.0))
    if width <= 18 or depth > 512:
        return int(math.ceil(depth / 1024.0) * math.ceil(width / 18.0))
    return int(math.ceil(depth / 512.0) * math.ceil(width / 36.0))


def _estimate_fifo_lutram_luts(depth, width):
    if depth <= 0 or width <= 0:
        return 0
    return int((2 * math.ceil(math.log(depth, 2))) + (math.ceil(depth / 32.0) * math.ceil(width / 2.0)))


def step_test_resnet_apply_ultra96_fifo_lutram_config(
    model: ModelWrapper, cfg: DataflowBuildConfig
) -> ModelWrapper:
    """Move selected Ultra96 FIFOs from BRAM to LUTRAM.

    The trim-160 Ultra96 build is failing only the combined BRAM tile DRC:
      220 / 216 Block RAM tiles after opt_design.

    LUTRAM still has comfortable headroom, so this pass runs after FIFO depth
    sizing and changes selected Vivado AXI FIFOs from auto/block BRAM mapping
    to distributed memory. We avoid post-synthesis names here because FINN/Vivado
    may renumber FIFOs during stitched-IP generation.
    """

    board = getattr(cfg, "board", None)
    if board != "Ultra96":
        print(f"  Ultra96 FIFO LUTRAM relief: skipped for board={board}")
        return model

    # Moving a 32768-deep FIFO to distributed RAM fixes the BRAM DRC but creates
    # a placer-hostile LUTRAM blob. Stay with small/medium FIFOs and over-target
    # the BRAM relief instead.
    target_bram18_sites = 32
    max_fifo_depth = 4096
    max_added_lutram_luts = 8000
    candidates = []
    skipped_non_vivado = 0
    skipped_depth_monitor = 0
    skipped_already_distributed = 0
    skipped_small = 0
    skipped_too_deep = 0

    for idx, node in enumerate(model.graph.node):
        if node.op_type != "StreamingFIFO_rtl":
            continue

        impl_style = _get_raw_nodeattr(node, "impl_style", "")
        ram_style = _get_raw_nodeattr(node, "ram_style", "")
        if impl_style != "vivado":
            skipped_non_vivado += 1
            continue
        if int(_get_raw_nodeattr(node, "depth_monitor", 0) or 0) != 0:
            skipped_depth_monitor += 1
            continue
        if ram_style == "distributed":
            skipped_already_distributed += 1
            continue

        depth = _safe_fifo_depth(node)
        width = _safe_fifo_width(node)
        if depth > max_fifo_depth:
            skipped_too_deep += 1
            continue
        bram18_sites = _estimate_fifo_bram18_sites(depth, width)
        lutram_luts = _estimate_fifo_lutram_luts(depth, width)
        if bram18_sites < 1:
            skipped_small += 1
            continue

        efficiency = bram18_sites / max(lutram_luts, 1)
        candidates.append(
            {
                "idx": idx,
                "node": node,
                "depth": depth,
                "width": width,
                "bram18": bram18_sites,
                "lutram": lutram_luts,
                "efficiency": efficiency,
            }
        )

    candidates.sort(
        key=lambda x: (
            x["bram18"] > 2,
            -x["efficiency"],
            x["lutram"],
            -x["bram18"],
            x["idx"],
        )
    )

    selected = []
    recovered_bram18 = 0
    added_lutram = 0
    for cand in candidates:
        if selected and recovered_bram18 >= target_bram18_sites:
            break
        if added_lutram + cand["lutram"] > max_added_lutram_luts:
            continue

        _set_raw_nodeattr(cand["node"], "ram_style", "distributed")
        selected.append(cand)
        recovered_bram18 += cand["bram18"]
        added_lutram += cand["lutram"]

    if selected:
        selected_desc = ", ".join(
            [
                "%s:bram18_est~%d:lutram~%d:depth=%d:width=%d:idx=%d"
                % (
                    cand["node"].name,
                    cand["bram18"],
                    cand["lutram"],
                    cand["depth"],
                    cand["width"],
                    cand["idx"],
                )
                for cand in selected
            ]
        )
        print(
            "  Ultra96 FIFO LUTRAM relief applied: "
            f"FIFOs={len(selected)}, target_bram18={target_bram18_sites}, "
            f"recovered_bram18_est~{recovered_bram18}, "
            f"tile_relief_est~{recovered_bram18 / 2.0:.1f}, "
            f"added_lutram~{added_lutram}, "
            f"skipped_non_vivado={skipped_non_vivado}, "
            f"skipped_depth_monitor={skipped_depth_monitor}, "
            f"skipped_already_distributed={skipped_already_distributed}, "
            f"skipped_small={skipped_small}, "
            f"skipped_too_deep={skipped_too_deep}, "
            f"max_fifo_depth={max_fifo_depth}, selected=[{selected_desc}]"
        )
    else:
        print(
            "  Ultra96 FIFO LUTRAM relief: no eligible nodes updated "
            f"(skipped_non_vivado={skipped_non_vivado}, "
            f"skipped_depth_monitor={skipped_depth_monitor}, "
            f"skipped_already_distributed={skipped_already_distributed}, "
            f"skipped_small={skipped_small}, "
            f"skipped_too_deep={skipped_too_deep}, "
            f"max_fifo_depth={max_fifo_depth})"
        )

    return cleanup_model(model)


