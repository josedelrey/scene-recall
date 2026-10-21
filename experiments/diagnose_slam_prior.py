"""Ideal shared-uncertainty control and an unidentifiable coherent-depth bias.

Linear covariance here belongs only to the stated synthetic independent-noise
model. The zero-residual scale counterexample demonstrates its physical limit.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from temporal_controls import scene

from scene_recall.multiview.batch_ba import BatchConfig
from scene_recall.multiview.fixed_landmarks import (
    direct_cost,
    direct_linearize,
    make_direct_factors,
)
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.multiview.pose_only import PoseOnlyConfig
from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.graph import free_columns, linear_system, solve_window


def diagnose():
    calibration, tracks, samples, truth = scene(8, exact=True, skew=0)
    factors, _ = make_direct_factors(
        tracks,
        samples,
        list(range(8)),
        calibration,
        PoseOnlyConfig(min_grid_coverage=0),
    )
    ids, _, points = landmark_layout(factors)
    _, H, _ = linear_system(
        truth,
        points,
        ids,
        factors,
        calibration.K_R,
        BatchConfig(loss="linear"),
        list(range(8)),
        None,
    )
    free = free_columns(list(range(8)), ids, 0)
    H = H[np.ix_(free, free)]
    dimension = 6 * 7
    marginal = np.linalg.inv(H)[:dimension, :dimension]
    conditional = np.linalg.inv(H[:dimension, :dimension])
    eig = np.linalg.eigvalsh((marginal - conditional + (marginal - conditional).T) / 2)
    changes = []
    for i in range(7):
        block = slice(6 * i, 6 * i + 3)
        changes.append(
            float(
                np.trace(marginal[block, block]) / np.trace(conditional[block, block])
            )
        )
    scale = 1.05
    biased = truth.copy()
    biased[:, :3, 3] *= scale
    packet = replace(
        factors,
        points=factors.points * scale,
        native_points=factors.native_points * scale,
    )
    residual, _ = direct_linearize(biased, packet, calibration.K_R, (), jacobian=False)
    cost, _ = direct_cost(residual, packet, BatchConfig())
    fitted, _, fit = solve_window(biased, packet, calibration.K_R, list(range(8)), 0)
    return {
        "implementation": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "shared_uncertainty": {
            "marginal_minus_conditional_minimum_eigenvalue": float(eig.min()),
            "translation_variance_trace_ratio": error_statistics(changes),
            "camera_cross_covariance_frobenius": float(
                np.linalg.norm(marginal[:6, 6:12])
            ),
        },
        "hidden_five_percent_depth_scale": {
            "robust_cost": cost,
            "maximum_whitened_residual": float(np.max(np.abs(residual))),
            "fit": fit,
            "camera_center_error_m": error_statistics(
                np.linalg.norm(fitted[:, :3, 3] - truth[:, :3, 3], axis=1)
            ),
            "endpoint_pose_error_m_deg": pose_error(fitted[-1], truth[-1]),
            "point_error_m": error_statistics(
                np.linalg.norm(points * scale - points, axis=1)
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output file")
    result = diagnose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
