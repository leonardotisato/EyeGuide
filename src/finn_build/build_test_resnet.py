"""
FINN Dataflow Build for QAT test_resnet -> Ultra96-v2 bitstream.

Usage (estimates only):
    python src/finn_build/build_test_resnet.py --estimates-only

Usage (stop after a specific step for Netron inspection):
    python src/finn_build/build_test_resnet.py --stop-after step_test_resnet_streamline

Must run inside the FINN Docker container.
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(
    description="FINN dataflow build for QAT test_resnet -> Ultra96-v2."
)
parser.add_argument("--onnx", default="models/test_resnet_8w8a.onnx")
parser.add_argument("--output-dir", default="./build_finn_test_resnet")
parser.add_argument("--estimates-only", action="store_true")
parser.add_argument("--stop-after", default=None,
                    help="Stop after this step name (for incremental debugging).")
parser.add_argument("--synth-clk-ns", type=float, default=10.0)
parser.add_argument("--target-fps", type=int, default=1)
parser.add_argument("--folding-config", default=None,
                    help="Manual folding config JSON. If None, uses target_fps auto-folding.")
parser.add_argument("--board", default="Ultra96",
                    help="Target board (default: Ultra96 = xczu3eg-sbva484-1-e)")
args = parser.parse_args()

try:
    from finn.builder.build_dataflow_config import (
        DataflowBuildConfig,
        DataflowOutputType,
        LargeFIFOMemStyle,
        ShellFlowType,
    )
    from finn.builder.build_dataflow import build_dataflow_cfg
    from finn.util.basic import pynq_part_map, alveo_part_map

    from custom_steps_resnet18 import step_fundus_attach_preproc
    from custom_steps_test_resnet import (
        step_test_resnet_streamline,
        step_test_resnet_lower,
        step_test_resnet_to_hw,
    )
except ImportError as e:
    print(f"\n[ERROR] Import failed: {e}\n"
          "Are you inside the FINN Docker container?")
    sys.exit(1)

if not os.path.exists(args.onnx):
    print(f"\n[ERROR] ONNX file not found: {args.onnx}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
estimate_steps = [
    "step_qonnx_to_finn",
    step_fundus_attach_preproc,
    "step_tidy_up",
    step_test_resnet_streamline,
    step_test_resnet_lower,
    step_test_resnet_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
]

def resolve_steps(steps, stop_after=None):
    if stop_after is None:
        return steps
    for i, step in enumerate(steps):
        name = step if isinstance(step, str) else step.__name__
        if name == stop_after:
            return steps[: i + 1]
    valid = [s if isinstance(s, str) else s.__name__ for s in steps]
    print(f"\n[ERROR] --stop-after '{stop_after}' not found in step list.")
    print(f"  Valid step names: {valid}")
    sys.exit(1)


# Insert folding step before minimize_bit_width
minimize_idx = estimate_steps.index("step_minimize_bit_width")
if args.folding_config:
    estimate_steps.insert(minimize_idx, "step_apply_folding_config")
else:
    estimate_steps.insert(minimize_idx, "step_target_fps_parallelization")

full_steps = estimate_steps + [
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

# ---------------------------------------------------------------------------
# Select steps based on mode
# ---------------------------------------------------------------------------
if args.estimates_only:
    selected_steps = estimate_steps
    mode_label = "Estimate-only"
else:
    selected_steps = full_steps
    mode_label = "Full bitstream"

if args.stop_after:
    selected_steps = resolve_steps(selected_steps, args.stop_after)
    mode_label = f"Incremental (stop after {args.stop_after})"


# ---------------------------------------------------------------------------
# Board config
# ---------------------------------------------------------------------------
board = args.board
if board in pynq_part_map:
    shell_flow_type = ShellFlowType.VIVADO_ZYNQ
elif board in alveo_part_map:
    shell_flow_type = ShellFlowType.VITIS_ALVEO
else:
    print(f"\n[ERROR] Unknown board: {board}")
    print(f"  Valid Zynq boards: {list(pynq_part_map.keys())}")
    print(f"  Valid Alveo boards: {list(alveo_part_map.keys())}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Print config
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"  FINN Dataflow Build - test_resnet")
print(f"{'=' * 60}")
print(f"  Mode:    {mode_label}")
print(f"  Steps:   {len(selected_steps)}")
print(f"  Model:   {args.onnx}")
part = pynq_part_map.get(board) or alveo_part_map.get(board, "N/A")
print(f"  Board:   {board} ({part})")
print(f"  Flow:    {shell_flow_type}")
print(f"  Clock:   {args.synth_clk_ns} ns ({1000/args.synth_clk_ns:.0f} MHz)")
print(f"  Target:  {args.target_fps} FPS")
if args.folding_config:
    print(f"  Folding: {args.folding_config}")
else:
    print(f"  Folding: auto (target_fps_parallelization)")
print(f"  Output:  {args.output_dir}")
step_names = [s if isinstance(s, str) else s.__name__ for s in selected_steps]
print(f"  Steps:   {' -> '.join(step_names)}")
print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------
if args.estimates_only or args.stop_after:
    generate_outputs = [DataflowOutputType.ESTIMATE_REPORTS]
else:
    generate_outputs = [
        DataflowOutputType.ESTIMATE_REPORTS,
        DataflowOutputType.BITFILE,
        DataflowOutputType.PYNQ_DRIVER,
        DataflowOutputType.DEPLOYMENT_PACKAGE,
    ]

cfg = DataflowBuildConfig(
    steps=selected_steps,
    output_dir=args.output_dir,
    synth_clk_period_ns=args.synth_clk_ns,
    board=board,
    shell_flow_type=shell_flow_type,
    target_fps=args.target_fps,
    folding_config_file=args.folding_config,
    split_large_fifos=True,
    default_swg_exception=True,
    generate_outputs=generate_outputs,
    save_intermediate_models=True,
)

build_dataflow_cfg(args.onnx, cfg)

print(f"\n{'=' * 60}")
print(f"  Build complete!")
print(f"{'=' * 60}")
print(f"\nOutputs:        {args.output_dir}")
print(f"Intermediates:  {args.output_dir}/intermediate_models/")
print(f"\nInspect intermediate ONNX files in Netron for debugging.")
