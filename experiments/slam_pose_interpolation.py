"""Offline two-sided full-rate pose interpolation between final keyframes.

Keyframe estimates, segment gauges and trailing poses stay fixed. Interior
camera centers move linearly and rotations follow the shortest SO(3) arc.
This assumes smooth motion and needs the following keyframe. It uses no raw
measurements or reference poses and supplies no extra SLAM constraints.
"""

import argparse
import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.odometry.pipeline import PairEstimate
from scene_recall.run_metadata import sha256


def interpolate_poses(poses, timestamps, segments, keyframes):
    result = poses.copy()
    for a, b in pairwise(sorted(set(keyframes))):
        if segments[a] != segments[b]:
            continue
        rotation = cv2.Rodrigues(poses[a, :3, :3].T @ poses[b, :3, :3])[0].ravel()
        for i in range(a + 1, b):
            if segments[i] != segments[a]:
                raise ValueError("interpolation interval crosses a segment boundary")
            if timestamps[a] is not None and timestamps[b] is not None:
                alpha = (timestamps[i] - timestamps[a]) / (
                    timestamps[b] - timestamps[a]
                )
            else:
                alpha = (i - a) / (b - a)
            result[i, :3, 3] = (1 - alpha) * poses[a, :3, 3] + alpha * poses[b, :3, 3]
            result[i, :3, :3] = poses[a, :3, :3] @ cv2.Rodrigues(alpha * rotation)[0]
    return result


def replay(run, output):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    report, rows = load_trajectory(run)
    keyframes = [
        i
        for i, f in enumerate(report["frames"])
        if "exposure_status" in f["diagnostics"] or rows[i].status != "tracked"
    ]
    poses = interpolate_poses(
        np.array([r.T_SC_D for r in rows]),
        [r.timestamp for r in rows],
        [r.segment_id for r in rows],
        keyframes,
    )
    revised = []
    for i, row in enumerate(rows):
        pair = row.pair
        if row.status == "tracked":
            pair = PairEstimate(np.linalg.inv(poses[i]) @ poses[i - 1])
        revised.append(replace(row, T_SC_D=poses[i], pair=pair))
    new_report = {
        **report,
        "evaluation": evaluate_trajectory(revised, (1, 5, 10, 20, 50, 100)),
        "experiment": {
            "type": "offline two-sided keyframe interpolation",
            "driver_sha256": sha256(Path(__file__)),
            "input_report_sha256": sha256(run / "report.json"),
            "input_trajectory_sha256": sha256(run / "trajectory.npz"),
            "keyframes": keyframes,
            "keyframe_poses_preserved_exactly": bool(
                np.array_equal(
                    poses[keyframes], np.array([rows[i].T_SC_D for i in keyframes])
                )
            ),
            "latency": "requires next keyframe, no online or uncertainty claim",
        },
    }
    # Emitted evaluation belongs to the source estimator, not this offline output.
    new_report.pop("emitted_evaluation", None)
    new_report.pop("measurement_consistency", None)
    save_run(output, revised, new_report, report["source"]["start"])
    result = {
        "before": report["evaluation"],
        "after": new_report["evaluation"],
        "provenance": new_report["experiment"],
    }
    (output / "interpolation_comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    cases = json.loads(args.cases.read_text())
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "cases_sha256": sha256(args.cases),
                "driver_sha256": sha256(Path(__file__)),
                "phase": "frozen before interpolation scores",
                "rule": "all final keyframes fixed, timestamp-linear center and shortest rotation arc, no score selection",
            },
            indent=2,
        )
        + "\n"
    )
    results = {
        name: replay(Path(paths["run"]), args.output_dir / name)
        for name, paths in cases.items()
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
