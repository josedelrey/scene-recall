"""Serial raw-input Sparse, frontend and SLAM timings in opposite orders.

Run after accuracy and reconstruction jobs have finished. Warm source files,
shared machine load and short-prefix scope remain explicit limitations.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from incremental_slam import run_case

from scene_recall.odometry.evaluation import error_statistics
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig
from scene_recall.run_metadata import code_hash, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    args.output_dir.mkdir(parents=True)
    baseline, _ = load_trajectory(args.baseline_run)
    protocol = {
        "orders": [["sparse", "control", "native"], ["native", "control", "sparse"]],
        "input_report_sha256": sha256(args.baseline_run / "report.json"),
        "implementation": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "sparse": asdict(SparseConfig(max_features=1500)),
        "timing": "serial raw input, one policy per process phase, warm files, no other experiment jobs",
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        if Path("/proc/cpuinfo").exists()
        else "unknown",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    results = {}
    for direction, order in zip(
        ("forward", "reverse"), protocol["orders"], strict=True
    ):
        for name in order:
            destination = args.output_dir / (direction + "_" + name)
            if name == "sparse":
                report = run_odometry(
                    Path(baseline["source"]["path"]),
                    destination,
                    dataset=baseline["source"]["dataset"],
                    backend_name="sparse",
                    config=SparseConfig(max_features=1500),
                    start=baseline["source"]["start"],
                    stop=baseline["source"]["stop"],
                )
                wall = report["timing"]["tracking_seconds"]
            else:
                result = run_case(args.baseline_run, destination, [name])
                wall = result["wall_seconds"]
                report, _ = load_trajectory(destination / name)
            diagnostics = [row["diagnostics"] for row in report["frames"]]
            results[direction + "_" + name] = {
                "frames": report["evaluation"]["frame_count"],
                "wall_tracking_seconds": wall,
                "frames_per_second": report["evaluation"]["frame_count"] / wall,
                "final40_total_seconds": error_statistics(
                    d["total_seconds"]
                    for d in diagnostics[-40:]
                    if "total_seconds" in d
                ),
                "keyframe_seconds": error_statistics(
                    d["total_seconds"]
                    for d in diagnostics
                    if "measurement_seconds" in d
                ),
                "solver_seconds": error_statistics(
                    d["solver"]["seconds"]
                    for d in diagnostics
                    if "solver" in d and "seconds" in d["solver"]
                ),
            }
            print(
                direction,
                name,
                results[direction + "_" + name]["frames_per_second"],
                flush=True,
            )
    (args.output_dir / "report.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
