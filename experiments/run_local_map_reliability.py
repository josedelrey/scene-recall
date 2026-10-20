"""Run the predeclared reliability matrix with portable source locations.

The two TUM sources must be canonical readable sequences. Prepare ambiguous
references separately. No datasets are edited or downloaded by this driver.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from scene_recall.run_metadata import sha256


def validate_baselines(cases):
    """Catch wrong historical directory links before launching any tracking."""
    for case in cases:
        if case["baseline"]:
            report = json.loads((Path(case["baseline"]) / "report.json").read_text())
            if any(
                report["source"][key] != case[key]
                for key in ("dataset", "start", "stop")
            ):
                raise ValueError("baseline source range differs for " + case["name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desk-path", type=Path, required=True)
    parser.add_argument("--office-path", type=Path, required=True)
    parser.add_argument("--sens-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--protect", type=Path, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.output_dir.exists():
        parser.error("use positive workers and a fresh output directory")
    cases = []
    for prefix, source in (("fr2", args.desk_path), ("fr3", args.office_path)):
        for tag, start, stop in (
            ("early", 0, 300),
            ("late", 1200, 1500),
            ("middle", 600, 800),
        ):
            name = f"{prefix}_{tag}"
            baseline = (
                args.baseline_root / name / "recent"
                if args.baseline_root and tag != "middle"
                else None
            )
            cases.append(
                {
                    "name": name,
                    "source": str(source.resolve()),
                    "start": start,
                    "stop": stop,
                    "dataset": "tum_rgbd",
                    "baseline": str(baseline.resolve()) if baseline else None,
                    "population": "prior independent primary"
                    if tag != "middle"
                    else "new index-selected window in independent scene",
                }
            )
    for name, start, previous in (
        ("scan1000", 1000, "regression_scan"),
        ("scan3500", 3500, "development_scan"),
    ):
        baseline = (
            args.baseline_root / previous / "recent" if args.baseline_root else None
        )
        cases.append(
            {
                "name": name,
                "source": str(args.sens_path.resolve()),
                "start": start,
                "stop": start + 200,
                "dataset": "scannet_v2",
                "baseline": str(baseline.resolve()) if baseline else None,
                "population": "same-capture regression, not an independent ScanNet scene",
            }
        )
    validate_baselines(cases)
    args.output_dir.mkdir(parents=True)
    protected = {
        str(p.resolve()): sha256(p)
        for base in args.protect
        for p in sorted(base.rglob("*"))
        if p.is_file()
        and p.suffix in (".json", ".npz", ".ply")
        and not any(x in p.parts for x in ("downloads", "sources", "prepared"))
    }
    (args.output_dir / "protected_baselines.json").write_text(
        json.dumps(protected, indent=2) + "\n"
    )
    protocol = {
        "phase": "frozen before evaluation",
        "cases": cases,
        "policies": ["budget", "legacy", "rgb", "rgbd", "family"],
        "driver_sha256": sha256(Path(__file__)),
        "protected_files": len(protected),
        "selection": "Fixed source ranges, no score-selected windows or thresholds. Surface fusion and first-chunk TSDF for every case, without replacing missing-reference chunks.",
        "runtime": f"{args.workers} concurrent cases, immutable cached features across policies. Timing is not a latency ranking.",
    }
    (args.output_dir / "frozen_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    def run(case):
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("investigate_local_map.py")),
            "--tum-path" if case["dataset"] == "tum_rgbd" else "--sens-path",
            case["source"],
            "--start",
            str(case["start"]),
            "--stop",
            str(case["stop"]),
            "--output-dir",
            str(args.output_dir / case["name"]),
            "--evaluate-maps",
        ]
        if case["baseline"]:
            cmd += ["--baseline", case["baseline"]]
        with (args.output_dir / (case["name"] + ".log")).open("w") as log:
            result = subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        return case["name"], result.returncode

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in concurrent.futures.as_completed(
            [pool.submit(run, c) for c in cases]
        ):
            name, status = future.result()
            print(name, status, flush=True)
            if status:
                failed.append(name)
    if failed:
        raise RuntimeError("cases failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
