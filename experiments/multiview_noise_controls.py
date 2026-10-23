"""Conditional multiview controls separating fit quality from geometric accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.pose_only import retract
from scene_recall.odometry.evaluation import pose_error
from scene_recall.odometry.sparse import SparseConfig, estimate_sparse_pose
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.multiview_registration import conditional_bundle


def run(repetitions):
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    truth = retract(np.eye(4), np.array([0.08, -0.03, 0.02, 0.02, -0.05, 0.03]))
    historical = retract(np.eye(4), np.array([-0.25, 0.02, 0, -0.01, 0.02, 0]))
    results = {}
    for condition in (
        "independent_noise",
        "source_scale_bias",
        "context_pose_error",
        "shared_scale_bias",
    ):
        records = []
        for seed in range(repetitions):
            rng = np.random.default_rng(seed)
            points = rng.uniform([-1, -0.8, 1.5], [1, 0.8, 4.0], (120, 3))
            projections = [
                project_points(transform_points(points, p), K)
                + rng.normal(0, 1.0, (len(points), 2))
                for p in (np.eye(4), historical, truth)
            ]
            depth = points[:, 2] + rng.normal(0, 0.005 + 0.01 * points[:, 2])
            context_pose = historical.copy()
            if condition in ("source_scale_bias", "shared_scale_bias"):
                depth *= 1.03
            if condition == "shared_scale_bias":
                context_pose[:3, 3] *= 1.03
            if condition == "context_pose_error":
                context_pose[0, 3] += 0.02
            anchors = (
                np.column_stack((projections[0], np.ones(len(points))))
                @ np.linalg.inv(K).T
            ) * depth[:, None]
            fit = np.arange(0, len(points), 2)
            initial = estimate_sparse_pose(
                anchors[fit], projections[2][fit], K, SparseConfig()
            )
            if initial.T_C1C0 is None:
                raise RuntimeError("controlled initializer failed")
            pose, landmarks, diagnostic = conditional_bundle(
                anchors[fit],
                projections[1][fit],
                context_pose,
                K,
                initial=initial.T_C1C0,
                incoming_pixels=projections[2][fit],
            )
            records.append(
                {
                    "pnp_error": pose_error(initial.T_C1C0, truth),
                    "bundle_error": pose_error(pose, truth),
                    "source_point_rmse_m": float(
                        np.sqrt(
                            np.mean(np.sum((anchors[fit] - points[fit]) ** 2, axis=1))
                        )
                    ),
                    "fitted_point_rmse_m": float(
                        np.sqrt(np.mean(np.sum((landmarks - points[fit]) ** 2, axis=1)))
                    ),
                    "initial_cost": diagnostic.get("initial_cost"),
                    "final_cost": diagnostic.get("final_cost"),
                }
            )
        results[condition] = {
            "pnp_median": np.median([r["pnp_error"] for r in records], axis=0).tolist(),
            "bundle_median": np.median(
                [r["bundle_error"] for r in records], axis=0
            ).tolist(),
            "source_point_rmse_median": float(
                np.median([r["source_point_rmse_m"] for r in records])
            ),
            "fitted_point_rmse_median": float(
                np.median([r["fitted_point_rmse_m"] for r in records])
            ),
            "records": records,
        }
    return {
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "repetitions": repetitions,
        "scope": "synthetic model controls, not a raw SLAM benchmark",
        "conditions": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 1:
        parser.error("choose a fresh output and positive repetitions")
    args.output.write_text(json.dumps(run(args.repetitions), indent=2) + "\n")


if __name__ == "__main__":
    main()
