"""Reproduce paired mapping experiments from saved odometry, without retracking.

Each child runs in a separate process for interpretable runtime and peak RSS.
The comparison checks identical keyframe inputs and segment populations before
placing backend results in a common cohort. Run again in a fresh directory to
verify deterministic cloud arrays and numerical diagnostics.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def summarize(reports):
    """Keep map populations and metric denominators visible in backend summaries."""
    rows = []
    for name, report in reports.items():
        for submap in report["submaps"]:
            for direction in ("estimated_to_reference", "reference_to_estimated"):
                metric = (
                    None
                    if submap["comparison"] is None
                    else submap["comparison"][direction]
                )
                rows.append(
                    {
                        "run": name,
                        "backend": report["odometry"]["backend"],
                        "segment_id": submap["segment_id"],
                        "submap_id": submap["submap_id"],
                        "keyframes": len(submap["keyframe_indices"]),
                        "estimated_voxels": submap["estimated"]["voxel_count"],
                        "reference_voxels": submap["reference"]["voxel_count"],
                        "direction": direction,
                        "within_cap_fraction": None
                        if metric is None
                        else metric["within_cap_fraction"],
                        "capped_distance_rmse_m": None
                        if metric is None
                        else metric["capped_distance_m"]["rmse"],
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry-run", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-keyframes-per-submap", type=int, default=64)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output already exists, choose a new directory")
    names = [path.name for path in args.odometry_run]
    if len(set(names)) != len(names):
        parser.error("odometry run directory names must be unique")
    args.output_dir.mkdir(parents=True)
    runner = Path(__file__).resolve().parents[1] / "scripts/rgbd_mapping.py"
    reports, keys = {}, {}
    for source in args.odometry_run:
        output = args.output_dir / source.name
        subprocess.run(
            [
                sys.executable,
                str(runner),
                "--odometry-run",
                str(source),
                "--output-dir",
                str(output),
                "--max-keyframes-per-submap",
                str(args.max_keyframes_per_submap),
                "--pair-keyframe-offsets",
                "1",
                "3",
                "9",
                "27",
            ],
            check=True,
        )
        reports[source.name] = json.loads((output / "report.json").read_text())
        keys[source.name] = json.loads((output / "keyframes.json").read_text())
    cohorts = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            a, b = reports[first], reports[second]
            same_source = (
                a["verified_canonical_input_sha256"]
                == b["verified_canonical_input_sha256"]
            )
            same_frames = [
                (k["source_frame_index"], k["canonical_sha256"]) for k in keys[first]
            ] == [
                (k["source_frame_index"], k["canonical_sha256"]) for k in keys[second]
            ]
            membership = lambda report: [
                (s["segment_id"], s["keyframe_indices"]) for s in report["submaps"]
            ]
            cohorts.append(
                {
                    "runs": [first, second],
                    "same_source": same_source,
                    "identical_keyframe_inputs": same_frames,
                    "identical_segment_submap_population": membership(a)
                    == membership(b),
                    "comparable": same_source
                    and same_frames
                    and membership(a) == membership(b),
                }
            )
    rows = summarize(reports)
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparison = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": {
            name: hashlib.sha256(
                (args.output_dir / name / "report.json").read_bytes()
            ).hexdigest()
            for name in names
        },
        "cohorts": cohorts,
        "submap_metrics": rows,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
