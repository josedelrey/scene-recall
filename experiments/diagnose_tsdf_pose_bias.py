"""Demonstrate why improved map self-consistency need not improve geometry."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scene_recall.geometry.camera import backproject_depth
from scene_recall.mapping.evaluation import heldout_metrics, render_depth
from scene_recall.mapping.mesh import render_mesh_depth
from scene_recall.mapping.tsdf import TSDFConfig, TSDFView, fuse_tsdf
from scene_recall.mapping.voxel import SurfaceSamples, VoxelMap


def probe(bias_m=0.08):
    """Keep an exact stationary scene, biasing eight of nine training poses.

    First-pose alignment stays identity. Evaluate the same observed target both
    at its true camera pose and at the biased estimate. The target is never
    integrated. Bias is deliberate synthetic input, not estimated from a scene.
    """
    K = np.array([[40, 0, 16], [0, 40, 16], [0, 0, 1]], dtype=float)
    depth = np.full((33, 33), 2.0)
    samples = SurfaceSamples(
        backproject_depth(depth, K).reshape(-1, 3),
        np.zeros((depth.size, 3), dtype=np.uint8),
        np.zeros(depth.size, dtype=bool),
        depth,
    )
    config = TSDFConfig()
    baseline = VoxelMap(config.voxel_size_m)
    views = []
    biased = np.eye(4)
    biased[2, 3] = bias_m
    for i in range(9):
        pose = np.eye(4) if i == 0 else biased
        baseline.integrate(samples, pose, i)
        views.append(TSDFView(i, pose, depth))
    volume = fuse_tsdf(views, K, config)
    mesh = volume.mesh()
    predictions = {
        f"{method}_{mode}": renderer(geometry, pose, K, depth.shape, *args)
        for method, renderer, geometry, args in (
            ("baseline", render_depth, baseline.cloud(), (0.01,)),
            ("tsdf", render_mesh_depth, mesh, ()),
        )
        for mode, pose in (("reference", np.eye(4)), ("estimated", biased))
    }
    # Retain a central region and the fixed principal ray separately. Point
    # splat coverage also affects which layer wins away from the principal ray.
    common = np.zeros(depth.shape, dtype=bool)
    common[8:25, 8:25] = True
    for prediction in predictions.values():
        common &= np.isfinite(prediction)
    center = np.zeros(depth.shape, dtype=bool)
    center[16, 16] = True
    return {
        "tsdf_config": asdict(config),
        "training_frames": 9,
        "biased_training_frames": 8,
        "bias_z_m": bias_m,
        "true_plane_z_m": 2.0,
        "K": K.tolist(),
        "depth_shape": list(depth.shape),
        "scores": {
            name: heldout_metrics(depth, value, common)
            for name, value in predictions.items()
        },
        "center_scores": {
            name: heldout_metrics(depth, value, center)
            for name, value in predictions.items()
        },
        "notes": [
            "Synthetic causal counterexample, not an estimate of real pose bias",
            "Observed target depth and true scene are unchanged across modes",
            "A lower estimated-target error does not imply better external geometry",
            "The fixed principal ray avoids point-footprint coverage confounds",
        ],
    }, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports, arrays = {}, {}
    for label, bias in (("exact_poses", 0.0), ("biased_poses", 0.08)):
        reports[label], predictions = probe(bias)
        arrays.update({f"{label}_{name}": array for name, array in predictions.items()})
    np.savez_compressed(args.output_dir / "predictions.npz", **arrays)
    (args.output_dir / "report.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False) + "\n"
    )
