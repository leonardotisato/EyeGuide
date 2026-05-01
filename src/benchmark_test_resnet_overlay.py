"""Board-side measured-throughput wrapper for FINN-generated test_resnet overlays."""

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

from utils.reporting import to_jsonable


DEFAULT_BITFILE_NAME = "finn-accel.bit"
DEFAULT_METRICS_FILENAME = "nw_metrics.txt"


OVERLAY_METRIC_MAP = {
    "runtime_ms": "runtime[ms]",
    "throughput_images_per_s": "throughput[images/s]",
    "dram_in_bandwidth_mb_s": "DRAM_in_bandwidth[MB/s]",
    "dram_out_bandwidth_mb_s": "DRAM_out_bandwidth[MB/s]",
    "fclk_mhz": "fclk[mhz]",
    "batch_size": "batch_size",
}


HOST_OVERHEAD_METRIC_MAP = {
    "fold_input_ms": "fold_input[ms]",
    "pack_input_ms": "pack_input[ms]",
    "copy_input_data_to_device_ms": "copy_input_data_to_device[ms]",
    "copy_output_data_from_device_ms": "copy_output_data_from_device[ms]",
    "unpack_output_ms": "unpack_output[ms]",
    "unfold_output_ms": "unfold_output[ms]",
}


def build_parser():
    """Parse CLI arguments for benchmark execution and report output."""

    parser = argparse.ArgumentParser(
        description="Run throughput benchmarking for a FINN-generated test_resnet overlay."
    )
    parser.add_argument(
        "--build-dir",
        required=True,
        help="FINN build directory containing driver/ and bitfile/ folders.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSON file that will receive the normalized benchmark report.",
    )
    parser.add_argument(
        "--platform",
        default="zynq-iodma",
        help="Driver platform argument passed to the generated FINN driver.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Batch size passed to the generated FINN driver.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="PYNQ device index passed to the generated FINN driver.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch the generated FINN driver.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for the generated driver process.",
    )
    return parser


def resolve_benchmark_paths(build_dir, bitfile_name=DEFAULT_BITFILE_NAME):
    """Resolve and validate the generated driver, bitfile, and metrics paths."""

    build_dir = Path(build_dir).expanduser().resolve()
    driver_dir = build_dir / "driver"
    driver_script = driver_dir / "driver.py"
    bitfile_path = build_dir / "bitfile" / bitfile_name
    metrics_path = driver_dir / DEFAULT_METRICS_FILENAME

    missing = [
        str(path)
        for path in (build_dir, driver_dir, driver_script, bitfile_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing benchmark artifacts: %s" % ", ".join(missing)
        )

    return {
        "build_dir": build_dir,
        "driver_dir": driver_dir,
        "driver_script": driver_script,
        "bitfile_path": bitfile_path,
        "metrics_path": metrics_path,
    }


def load_driver_metrics(metrics_path):
    """Read the FINN driver throughput metrics file into a Python dict."""

    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError("Driver metrics file not found: %s" % metrics_path)
    raw_metrics = ast.literal_eval(metrics_path.read_text(encoding="utf-8").strip())
    if not isinstance(raw_metrics, dict):
        raise TypeError("Driver metrics payload must be a dict, got %r" % type(raw_metrics))
    return raw_metrics


def _extract_metric_group(raw_metrics, metric_map):
    """Return a renamed subset of driver metrics using the provided key map."""

    return {
        public_key: raw_metrics[source_key]
        for public_key, source_key in metric_map.items()
        if source_key in raw_metrics
    }


def _extract_external_weight_bandwidths(raw_metrics):
    """Collect measured external-weight bandwidth metrics, if any are present."""

    prefix = "DRAM_extw_"
    suffix = "_bandwidth[MB/s]"
    extw_metrics = {}
    for raw_key, value in raw_metrics.items():
        if raw_key.startswith(prefix) and raw_key.endswith(suffix):
            weight_name = raw_key[len(prefix):-len(suffix)]
            extw_metrics[weight_name] = value
    return extw_metrics


def build_benchmark_report(raw_metrics, paths, args):
    """Normalize raw driver throughput metrics into the tracked JSON schema."""

    overlay_metrics = _extract_metric_group(raw_metrics, OVERLAY_METRIC_MAP)
    host_overhead = _extract_metric_group(raw_metrics, HOST_OVERHEAD_METRIC_MAP)
    external_weight_bandwidths = _extract_external_weight_bandwidths(raw_metrics)

    return {
        "model": "test_resnet",
        "build_dir": str(paths["build_dir"]),
        "driver_script": str(paths["driver_script"]),
        "bitfile": str(paths["bitfile_path"]),
        "platform": args.platform,
        "device": args.device,
        "batch_size": args.batchsize,
        "measured_overlay_metrics": to_jsonable(overlay_metrics),
        "host_overhead_ms": to_jsonable(host_overhead),
        "external_weight_bandwidth_mb_s": to_jsonable(external_weight_bandwidths),
    }


def write_benchmark_report(report, output_path):
    """Persist the normalized benchmark report to a JSON file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_driver_throughput(paths, args, runner=subprocess.run):
    """Invoke the generated FINN driver in throughput mode and read its metrics."""

    metrics_path = paths["metrics_path"]
    if metrics_path.exists():
        metrics_path.unlink()

    command = [
        args.python_executable,
        str(paths["driver_script"]),
        "--exec_mode",
        "throughput_test",
        "--platform",
        args.platform,
        "--batchsize",
        str(args.batchsize),
        "--device",
        str(args.device),
        "--bitfile",
        str(paths["bitfile_path"]),
    ]

    runner(
        command,
        cwd=str(paths["driver_dir"]),
        check=True,
        timeout=args.timeout,
    )
    return load_driver_metrics(metrics_path)


def main(argv=None):
    """CLI entry point for board-side benchmark collection."""

    parser = build_parser()
    args = parser.parse_args(argv)
    paths = resolve_benchmark_paths(args.build_dir)
    raw_metrics = run_driver_throughput(paths, args)
    report = build_benchmark_report(raw_metrics, paths, args)
    write_benchmark_report(report, args.output)
    print("Benchmark report written to %s" % args.output)
    return report


if __name__ == "__main__":
    main()
