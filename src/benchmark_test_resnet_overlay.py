"""Board-side throughput wrapper for FINN-generated test_resnet overlays."""

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

from utils.reporting import to_jsonable


DEFAULT_BITFILE_NAME = "finn-accel.bit"
DEFAULT_METRICS_FILENAME = "nw_metrics.txt"


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


def build_benchmark_report(raw_metrics, paths, args):
    """Normalize raw driver throughput metrics into the tracked JSON schema."""

    return {
        "model": "test_resnet",
        "build_dir": str(paths["build_dir"]),
        "driver_script": str(paths["driver_script"]),
        "bitfile": str(paths["bitfile_path"]),
        "platform": args.platform,
        "device": args.device,
        "batch_size": args.batchsize,
        "throughput_metrics": to_jsonable(raw_metrics),
        "summary": {
            "runtime_ms": raw_metrics.get("runtime[ms]"),
            "throughput_images_per_s": raw_metrics.get("throughput[images/s]"),
            "fclk_mhz": raw_metrics.get("fclk[mhz]"),
            "dram_in_bandwidth_mb_s": raw_metrics.get("DRAM_in_bandwidth[MB/s]"),
            "dram_out_bandwidth_mb_s": raw_metrics.get("DRAM_out_bandwidth[MB/s]"),
        },
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
