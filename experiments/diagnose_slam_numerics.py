"""Compare equivalent square-root and normal steps on controlled RGB-D systems.

The same weighted Jacobian, residual, column scaling and damping enter both
solves. This isolates double-precision linear solve effects from model changes.
It does not certify long-term marginal-prior consistency or float32 behavior.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from slam_generalization import seed_from_first_depth, split_tracks
from slam_model_controls import generate

from scene_recall.multiview.batch_ba import BatchConfig, compact_linearize
from scene_recall.multiview.fixed_landmarks import direct_cost
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.run_metadata import code_hash, sha256


def compare(geometry, seed):
    poses, factors, K, _ = generate(seed, geometry, "mixed", "independent")
    train, _ = split_tracks(seed_from_first_depth(factors, poses))
    ids, inverse, points = landmark_layout(train)
    residual, JP, JL = compact_linearize(poses, points, train, K, inverse)
    _, weights = direct_cost(residual, train, BatchConfig())
    dimension = 6 * (len(poses) - 1) + 3 * len(ids)
    J = np.zeros((len(train.view), 6, dimension))
    for i, (view, point) in enumerate(zip(train.view, inverse, strict=True)):
        if view:
            J[i, :, 6 * (view - 1) : 6 * view] = JP[i].reshape(6, 6)
        column = 6 * (len(poses) - 1) + 3 * point
        J[i, :, column : column + 3] = JL[i].reshape(6, 3)
    root = np.repeat(np.sqrt(weights), 3, axis=1)
    J = (J * root[:, :, None]).reshape(-1, dimension)
    residual = (residual.reshape(-1, 6) * root).ravel()
    scale = np.linalg.norm(J, axis=0)
    A = J / scale
    singular = np.linalg.svd(A, compute_uv=False)
    records = []
    for damping in (0.0, 0.001):
        normal = (
            np.linalg.solve(A.T @ A + damping * np.eye(dimension), -A.T @ residual)
            / scale
        )
        augmented = np.vstack((A, np.sqrt(damping) * np.eye(dimension)))
        rhs = np.concatenate((-residual, np.zeros(dimension)))
        square_root = np.linalg.lstsq(augmented, rhs, rcond=1e-12)[0] / scale
        records.append(
            {
                "damping": damping,
                "step_norm": float(np.linalg.norm(square_root)),
                "absolute_step_difference": float(np.linalg.norm(normal - square_root)),
                "relative_step_difference": float(
                    np.linalg.norm(normal - square_root)
                    / max(np.linalg.norm(square_root), 1e-15)
                ),
                "maximum_camera_step_difference": float(
                    np.max(np.abs(normal[:42] - square_root[:42]))
                ),
            }
        )
    return {
        "geometry": geometry,
        "seed": seed,
        "scaled_jacobian_condition": float(singular[0] / singular[-1]),
        "steps": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output file")
    sources = {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            Path(__file__).name,
            "slam_generalization.py",
            "slam_model_controls.py",
        )
    }
    result = {
        "sources": sources,
        "package_sha256": code_hash(),
        "precision": "float64",
        "records": [
            compare(g, seed) for g in ("near", "far", "plane") for seed in range(5)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
