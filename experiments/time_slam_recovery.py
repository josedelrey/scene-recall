"""Opposite-order, isolated-process raw recovery timing after scientific jobs end."""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from scene_recall.run_metadata import code_hash, sha256


def run(baseline, output):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    output.mkdir(parents=True)
    modes = ["control_no_atlas", "control", "low_pnp", "rgbd", "native"]
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    protocol = {
        "phase": "frozen before timing",
        "baseline_report_sha256": sha256(baseline / "report.json"),
        "baseline_trajectory_sha256": sha256(baseline / "trajectory.npz"),
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "recovery_driver_sha256": sha256(Path(__file__).with_name("slam_recovery.py")),
        "orders": [modes, list(reversed(modes))],
        "outage": [40, 50],
        "input": "raw images, one policy per fresh process",
        "threads": {"opencv": 1, "openblas": 1, "omp": 1},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unavailable",
        ),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = {}
    for order, modes in enumerate(protocol["orders"]):
        for mode in modes:
            policy = "control" if mode == "control_no_atlas" else mode
            phase = output / f"order{order}_{mode}"
            args = [
                sys.executable,
                str(Path(__file__).with_name("slam_recovery.py")),
                "--baseline-run",
                str(baseline),
                "--output-dir",
                str(phase),
                "--policies",
                policy,
                "--outage",
                "40",
                "50",
            ]
            if mode == "control_no_atlas":
                args.append("--no-atlas")
            subprocess.run(args, env=environment, check=True)
            report = json.loads((phase / "comparison.json").read_text())
            result = report["policies"][policy]
            results[phase.name] = {
                "wall_seconds": report["wall_seconds"],
                "peak_rss_mib": report["peak_rss_mib"],
                "process_seconds": result["process_and_atlas_seconds"],
                "atlas_seconds": result.get("atlas", {}).get("query_seconds"),
            }
            if order:
                first = output / f"order0_{mode}" / policy
                for archive in (
                    "trajectory.npz",
                    "atlas.npz",
                    "atlas_emitted.npz",
                    "atlas_links.npz",
                ):
                    if not (first / archive).exists():
                        continue
                    with (
                        np.load(first / archive) as a,
                        np.load(phase / policy / archive) as b,
                    ):
                        if a.files != b.files or any(
                            not np.array_equal(
                                a[k], b[k], equal_nan=a[k].dtype.kind in "fc"
                            )
                            for k in a.files
                        ):
                            raise ValueError(
                                "opposite-order replay changed scientific arrays"
                            )
            print("Completed timing phase", phase.name, flush=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    if protocol["package_sha256"] != code_hash() or protocol["driver_sha256"] != sha256(
        Path(__file__)
    ):
        raise RuntimeError("source changed during timing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline_run, args.output_dir)
