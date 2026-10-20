"""Analytic first-depth bias control for persistent fixed metric anchors.

This isolates a possible mechanism. It does not attribute real-sequence errors
to sensor scale bias. Pixels and identities are exact, with known SE(3) motion.
Only the first view's depth anchors are biased, within the consensus tolerance.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from scene_recall.data import Calibration, Observation
from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.local_map import LocalMapConfig, LocalMapRGBDBackend
from scene_recall.odometry.pipeline import track_observations
from scene_recall.odometry.recent_map import RecentMapConfig, RecentMapRGBDBackend
from scene_recall.odometry.sparse import SparseConfig, SparseRGBDBackend
from scene_recall.run_metadata import code_hash


def run_control(output, include_recent=False):
    if output.exists() or output.is_symlink():
        raise ValueError("choose a fresh control output")
    cv2.setNumThreads(1)
    rng = np.random.default_rng(47)
    K = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
    calibration = Calibration(K, K, np.eye(4), (240, 320), (240, 320))
    points = rng.uniform([-0.55, -0.4, 1.8], [0.55, 0.4, 2.8], (200, 3))
    descriptors = rng.normal(size=(200, 128)).astype(np.float32)
    observations, features = [], {}
    for i in range(20):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.002 * i, 0]))[0]
        pose[:3, 3] = [0.012 * i, 0.02 * np.sin(i / 5), 0.002 * np.sin(i / 7)]
        measured = transform_points(points, np.linalg.inv(pose))
        features[i] = RGBDFeatures(
            project_points(measured, K),
            descriptors.copy(),
            measured * (1.015 if i == 0 else 1),
        )
        observations.append(
            Observation(
                i,
                np.zeros((240, 320, 3), np.uint8),
                np.ones((240, 320), np.float32),
                T_WC_D=pose,
            )
        )
    sparse = SparseConfig(max_features=1500)
    config = LocalMapConfig(sparse=sparse)
    result = {
        "code_sha256": code_hash(),
        "first_depth_scale": 1.015,
        "measurement_policy": "exact analytic pixels and identities, only first-view depth biased",
        "local_config": asdict(config),
        "variants": {},
    }
    arrays = {"reference": np.array([o.T_WC_D for o in observations])}
    backends = [
        ("sparse", SparseRGBDBackend(sparse)),
        ("local", LocalMapRGBDBackend(config)),
    ]
    if include_recent:
        recent = RecentMapConfig(sparse=sparse)
        result["recent_config"] = asdict(recent)
        backends.append(("recent", RecentMapRGBDBackend(recent)))
    for name, backend in backends:
        extract = lambda observation, calibration: features[observation.frame_index]
        if name == "sparse":
            backend._features = extract
        else:
            backend._extract = extract
        rows = list(track_observations(observations, calibration, backend))
        arrays[name] = np.array([r.T_SC_D for r in rows])
        result["variants"][name] = evaluate_trajectory(rows, (1, 5, 10))
    output.mkdir(parents=True)
    np.savez_compressed(output / "trajectories.npz", **arrays)
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-recent", action="store_true")
    args = parser.parse_args()
    run_control(args.output_dir, args.include_recent)


if __name__ == "__main__":
    main()
