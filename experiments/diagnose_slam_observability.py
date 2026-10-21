"""Inspect undamped rejected windows without changing estimator decisions.

The diagnostic computes the joint normal-system spectrum only after the
unchanged solver returns. It compares rejected and accepted windows and cannot
turn an unobservable update into an accepted one.
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
from incremental_slam import run_case

from scene_recall.multiview.audit import normalized_spectrum
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256
from scene_recall.slam.graph import free_columns, linear_system, solve_window


def run(baseline, output, features, policy):
    records = []
    _, reference_rows = load_trajectory(baseline)

    def audit(poses, factors, K, cameras, root, prior, config):
        result = solve_window(poses, factors, K, cameras, root, prior, config)
        ids, _, points = landmark_layout(factors)
        _, H, _ = linear_system(poses, points, ids, factors, K, config, cameras, prior)
        free = free_columns(cameras, ids, root)
        S = H[np.ix_(free, free)]
        spectrum = normalized_spectrum(S)
        counts = [int((factors.view == i).sum()) for i in range(len(cameras))]
        exposure = {int(i): T for i, T in zip(factors.view, factors.T_RcD, strict=True)}
        indices = [
            i
            for i in sorted(exposure)
            if reference_rows[cameras[i]].reference_pose is not None
        ]
        errors = {}
        for label, fitted in (("before", poses), ("after", result[0])):
            values = []
            for a, b in pairwise(indices):
                reference = (
                    np.linalg.inv(reference_rows[cameras[a]].reference_pose)
                    @ reference_rows[cameras[b]].reference_pose
                )
                estimated = np.linalg.inv(fitted[a] @ exposure[a]) @ (
                    fitted[b] @ exposure[b]
                )
                values.append(pose_error(estimated, reference))
            errors[label] = {
                "translation_m": error_statistics(v[0] for v in values),
                "rotation_deg": error_statistics(v[1] for v in values),
            }
        records.append(
            {
                "at_frame": cameras[-1],
                "status": result[2]["status"],
                "reason": result[2]["reason"],
                "information": spectrum,
                "raw_observations_per_camera": counts,
                "prior_rank": 0 if prior is None else len(prior.b),
                "post_fit_reference_relative_errors": errors,
            }
        )
        return result

    with patch("scene_recall.slam.system.solve_window", audit):
        if policy.startswith("persistent"):
            from slam_persistent import run as persistent_run

            persistent_run(baseline, output, [policy], features=features)
        else:
            run_case(baseline, output, [policy], features=features)
    (output / "observability.json").write_text(
        json.dumps(
            {
                "driver_sha256": sha256(Path(__file__)),
                "windows": records,
                "interpretation": "Local linearized rank and conditioning, without calibrated covariance or physical correctness claims",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument(
        "--variant",
        choices=("native", "sqrt", "turnover", "persistent", "persistent_turnover"),
        default="turnover",
    )
    args = parser.parse_args()
    run(args.baseline_run, args.output_dir, args.features, args.variant)


if __name__ == "__main__":
    main()
