"""Controlled pose-bias and anchor-depth-bias counterexamples for pose-only fitting."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from scene_recall.data import Calibration
from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.model import (
    FeatureFrame,
    Landmark,
    TrackConfig,
    TrackResult,
)
from scene_recall.multiview.pose_only import (
    PoseOnlyConfig,
    refine_training_poses,
    retract,
)
from scene_recall.run_metadata import code_hash


def probe(initial_bias_m=0.0, anchor_depth_bias_m=0.0):
    """Generate a static scene with known identities, then score after fitting.

    Truth generates the controlled observations and evaluation. The optimizer
    receives only perturbed estimated poses and measured RGB-D feature rows.
    No ground-truth pose or point is supplied through its input model.
    """
    rng = np.random.default_rng(41)
    K = np.array([[300.0, 0, 160], [0, 310, 120], [0, 0, 1]])
    calibration = Calibration(K, K, np.eye(4), (240, 320), (240, 320))
    points = rng.uniform([-0.7, -0.5, 2.0], [0.7, 0.5, 3.5], (96, 3))
    frames, truth = [], []
    for i in range(6):
        pose = retract(np.eye(4), np.array([0.03 * i, 0.005 * i, 0, 0, 0.005 * i, 0]))
        truth.append(pose)
        measured = transform_points(points, np.linalg.inv(pose))
        pixels = project_points(measured, K)
        if i == 2:
            measured = measured * (1 + anchor_depth_bias_m / measured[:, 2:3])
        initial = pose.copy()
        if i:
            noise = rng.normal(0, [0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001])
            noise[2] += initial_bias_m
            initial = retract(initial, noise)
        frames.append(
            FeatureFrame(i, 0, i * 0.1, initial, RGBDFeatures(pixels, None, measured))
        )
    landmarks = []
    for j in range(len(points)):
        median = np.median(
            [transform_points(f.features.points_R[j], f.T_SR) for f in frames], axis=0
        )
        landmarks.append(
            Landmark(
                j,
                0,
                tuple((i, j) for i in range(6)),
                median,
                "supported",
                6,
                np.zeros(6),
                np.zeros(6),
                0.0,
                0.0,
            )
        )
    tracks = TrackResult(tuple(frames), tuple(landmarks), (), (), TrackConfig(), {})
    initial = np.array([f.T_SR for f in frames])
    truth = np.array(truth)
    arrays = {"truth": truth, "initial": initial}
    summary = {
        "initial_translation_rmse_m": float(
            np.sqrt(np.mean(np.sum((initial[:, :3, 3] - truth[:, :3, 3]) ** 2, axis=1)))
        )
    }
    for name in ("anchored", "frozen"):
        poses, diagnostics = refine_training_poses(
            tracks, calibration, replace(PoseOnlyConfig(), formulation=name)
        )
        refined = np.array([poses[i] for i in range(6)])
        arrays[name] = refined
        window = diagnostics["windows"][0]
        summary[name] = {
            "translation_rmse_m": float(
                np.sqrt(
                    np.mean(np.sum((refined[:, :3, 3] - truth[:, :3, 3]) ** 2, axis=1))
                )
            ),
            "initial_cost": window.get("initial_cost"),
            "final_cost": window.get("final_cost"),
            "status": window["status"],
        }
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports = {"code_sha256": code_hash(), "cases": {}}
    for name, pose_bias, depth_bias in (
        ("shared_pose_bias", 0.04, 0.0),
        ("biased_anchor_depth", 0.0, 0.04),
    ):
        report, arrays = probe(pose_bias, depth_bias)
        reports["cases"][name] = report
        np.savez_compressed(args.output_dir / f"{name}.npz", **arrays)
    (args.output_dir / "report.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
