"""Paired sensor-model interventions with known identities and physical truth.

Independent RGB/native Gaussian noise, frame-correlated depth error, focal
error, incorrect exposure motion and persistent identity swaps are separate
interventions. These controls establish mechanisms, not natural failure rates.
"""

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
from slam_generalization import MODELS, evaluate_block

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.batch_ba import BatchConfig
from scene_recall.multiview.fixed_landmarks import DirectFactors
from scene_recall.multiview.pose_only import retract
from scene_recall.run_metadata import code_hash, sha256

SCENARIOS = (
    "independent",
    "frame_depth_bias",
    "focal_error",
    "exposure_error",
    "identity_swaps",
)


def generate(seed, geometry, motion, scenario, *, tracks=96, initial_sigma=0.004):
    rng = np.random.default_rng(seed)
    pose_rng = np.random.default_rng(seed + 10000)
    count = 8
    K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
    world = rng.uniform([-0.8, -0.5, 1.5], [0.8, 0.5, 3.0], (tracks, 3))
    if geometry == "far":
        world *= 2
    elif geometry == "plane":
        world[:, 2] = 2.5
    truth = np.array(
        [
            retract(
                np.eye(4),
                np.array(
                    [
                        0.025 * i if motion == "mixed" else 0,
                        0.003 * i,
                        0,
                        0.003 * i,
                        0.018 * i,
                        0.002 * i,
                    ]
                ),
            )
            for i in range(count)
        ]
    )
    exposure = retract(np.eye(4), np.array([0.002, 0, 0, 0, 0.003, 0]))
    transforms = np.repeat(exposure[None], count * tracks, axis=0)
    view = np.repeat(np.arange(count), tracks)
    identities = np.tile(np.arange(tracks), count)
    native = np.concatenate(
        [transform_points(world, np.linalg.inv(p @ exposure)) for p in truth]
    )
    pixels = np.concatenate(
        [project_points(transform_points(world, np.linalg.inv(p)), K) for p in truth]
    )
    # The baseline is exactly the stated independent likelihood before intervention.
    sigma = 0.005 + 0.003 * native[:, 2] ** 2
    standard = np.column_stack((native[:, 2] / 525, native[:, 2] / 525, sigma))
    W = np.zeros((len(view), 3, 3))
    W[:, np.arange(3), np.arange(3)] = 1 / standard
    measured_native = native + rng.normal(size=native.shape) * standard
    measured_pixels = pixels + rng.normal(size=pixels.shape)
    initial = np.array(
        [
            p
            if i == 0
            else retract(
                p, pose_rng.normal(0, [initial_sigma] * 3 + [initial_sigma / 4] * 3)
            )
            for i, p in enumerate(truth)
        ]
    )
    if scenario == "frame_depth_bias":
        bias = 0.025 * np.sin(np.arange(count) * 0.8)
        measured_native += native / native[:, 2:3] * bias[view, None]
    elif scenario == "focal_error":
        K = K.copy()
        K[0, 0] *= 1.02
        K[1, 1] *= 1.02
    elif scenario == "exposure_error":
        transforms[:] = np.eye(4)
    elif scenario == "identity_swaps":
        # A persistent wrong match carries both RGB and depth from another point.
        selected = np.arange(0, tracks, 10)
        for camera in range(count // 2, count):
            rows = camera * tracks + selected
            source = camera * tracks + np.roll(selected, 1)
            measured_pixels[rows] = measured_pixels[source]
            measured_native[rows] = measured_native[source]
    elif scenario != "independent":
        raise ValueError("unknown intervention")
    packet = DirectFactors(
        view,
        identities,
        np.tile(world, (count, 1)),
        measured_pixels,
        measured_native,
        transforms,
        W,
        np.ones(len(view)),
    )
    return initial, packet, K, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists() or args.seeds < 1:
        parser.error("choose a fresh output directory and positive seed count")
    args.output_dir.mkdir(parents=True)
    config = BatchConfig(
        max_iterations=20,
        max_translation_m=0.1,
        max_rotation_deg=5,
        max_point_displacement_m=0.2,
    )
    protocol = {
        "driver_sha256": sha256(Path(__file__)),
        "diagnostic_sha256": sha256(Path(__file__).with_name("slam_generalization.py")),
        "package_sha256": code_hash(),
        "config": asdict(config),
        "models": MODELS,
        "seeds": list(range(args.seeds)),
        "geometries": ["near", "far", "plane"],
        "motions": ["mixed", "rotation"],
        "scenarios": SCENARIOS,
        "phase": "frozen before results",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    records = []
    for geometry, motion, scenario, seed in product(
        protocol["geometries"], protocol["motions"], SCENARIOS, range(args.seeds)
    ):
        poses, packet, K, truth = generate(seed, geometry, motion, scenario)
        result = evaluate_block(poses, packet, K, list(truth), config)
        records.append(
            {
                "seed": seed,
                "geometry": geometry,
                "motion": motion,
                "scenario": scenario,
                **result,
            }
        )
        if seed == args.seeds - 1:
            print(geometry, motion, scenario, flush=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )
    if (
        protocol["driver_sha256"] != sha256(Path(__file__))
        or protocol["diagnostic_sha256"]
        != sha256(Path(__file__).with_name("slam_generalization.py"))
        or protocol["package_sha256"] != code_hash()
    ):
        raise RuntimeError("source changed during evaluation")


if __name__ == "__main__":
    main()
