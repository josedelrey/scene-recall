"""Controlled point-level noise interventions without learned associations.

These synthetic controls isolate measurement-model behavior. They are not extra
independent recordings, calibrated sensor experiments or a SLAM benchmark.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.odometry.evaluation import pose_error
from scene_recall.odometry.sparse import SparseConfig, estimate_sparse_pose
from scene_recall.run_metadata import code_hash
from scene_recall.slam.metric_refinement import refine_metric_pose


def run(repetitions):
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    truth = np.eye(4)
    truth[:3, :3] = cv2.Rodrigues(np.array([0.02, -0.05, 0.03]))[0]
    truth[:3, 3] = [0.08, -0.03, 0.02]
    results = {}
    for condition in ("independent_depth", "source_scale_bias", "shared_scale_bias"):
        errors = {"pnp": [], "metric": []}
        for seed in range(repetitions):
            rng = np.random.default_rng(seed)
            a = rng.uniform([-1, -0.8, 1.5], [1, 0.8, 4.0], (120, 3))
            b = transform_points(a, truth)
            pixels = [
                project_points(p, K) + rng.normal(0, 1.0, (len(p), 2)) for p in (a, b)
            ]
            depths = [p[:, 2] + rng.normal(0, 0.005 + 0.01 * p[:, 2]) for p in (a, b)]
            if condition in ("source_scale_bias", "shared_scale_bias"):
                depths[0] *= 1.03
            if condition == "shared_scale_bias":
                depths[1] *= 1.03
            measured = [
                (np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(K).T)
                * depth[:, None]
                for uv, depth in zip(pixels, depths)
            ]
            initial = estimate_sparse_pose(measured[0], pixels[1], K, SparseConfig())
            if initial.T_C1C0 is None:
                raise RuntimeError("controlled PnP unexpectedly failed")
            refined, _ = refine_metric_pose(*measured, K, initial.T_C1C0)
            for name, pose in [("pnp", initial.T_C1C0), ("metric", refined)]:
                errors[name].append(pose_error(pose, truth))
        results[condition] = {
            name: {
                "median_translation_m": float(np.median(values, axis=0)[0]),
                "median_rotation_deg": float(np.median(values, axis=0)[1]),
                "errors": values,
            }
            for name, values in errors.items()
        }
    return {
        "code_sha256": code_hash(),
        "repetitions": repetitions,
        "scope": "synthetic conditional controls, no scene generalization claim",
        "conditions": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 1:
        parser.error("use a fresh output and positive repetitions")
    args.output.write_text(json.dumps(run(args.repetitions), indent=2) + "\n")


if __name__ == "__main__":
    main()
