"""Run a predeclared incremental SLAM matrix from verified baseline cases.

Use --case NAME BASELINE_RUN FEATURES_NPZ repeatedly, or --baseline-root for
the eight existing reliability windows. Parallel runs are accuracy experiments.
Use one worker and one variant for latency measurements.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from incremental_slam import variants

from scene_recall.run_metadata import code_hash, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "BASELINE", "FEATURES"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(variants()),
        default=["control", "native", "axial", "static"],
    )
    parser.add_argument("--evaluate-maps", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.output_dir.exists():
        parser.error("use positive workers and a fresh output directory")
    cases = list(args.case)
    if args.baseline_root:
        cases += [
            (
                n,
                str(args.baseline_root / n / "budget"),
                str(args.baseline_root / n / "features.npz"),
            )
            for n in (
                "fr2_early",
                "fr2_late",
                "fr2_middle",
                "fr3_early",
                "fr3_late",
                "fr3_middle",
                "scan1000",
                "scan3500",
            )
        ]
    if not cases or len({c[0] for c in cases}) != len(cases):
        parser.error("provide unique named cases")
    for name, baseline, features in cases:
        if Path(name).name != name:
            parser.error("case names must be directory names")
        if not (Path(baseline) / "trajectory.npz").is_file() or (
            features != "raw" and not Path(features).is_file()
        ):
            parser.error("case inputs are missing")
    args.output_dir.mkdir(parents=True)
    protocol = {
        "cases": cases,
        "variants": args.variants,
        "workers": args.workers,
        "phase": "frozen before evaluation",
        "implementation": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "population": "Previously studied windows are regression evidence. Frame counts are not independent scene replicates.",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    def run(case):
        name, baseline, features = case
        command = [
            sys.executable,
            str(Path(__file__).with_name("incremental_slam.py")),
            "--baseline-run",
            baseline,
            "--output-dir",
            str(args.output_dir / name),
            "--variants",
            *args.variants,
        ]
        if features != "raw":
            command += ["--features", features]
        if args.evaluate_maps:
            command += ["--evaluate-maps"]
        with (args.output_dir / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return name, result.returncode

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, status in pool.map(run, cases):
            print(name, status, flush=True)
            if status:
                failed.append(name)
    if failed:
        raise RuntimeError("cases failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
