"""Controlled landmark freedom under random noise, coherent errors and weak geometry."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from temporal_controls import scene

from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.fixed_landmarks import (
    direct_information,
    make_direct_factors,
    solve_fixed,
)
from scene_recall.multiview.joint_ba import landmark_layout, solve_joint
from scene_recall.multiview.pose_only import PoseOnlyConfig
from scene_recall.multiview.temporal import NativeSamples, native_whitening, pair_motion
from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.run_metadata import code_hash


def case(name, seed):
    """Generate measurements first, construct factors without evaluation geometry."""
    noise = name in (
        "independent_noise",
        "correlated_depth_noise",
        "noisy_initialization",
        "independent_noise_clean_root",
        "pixel_noise_only",
        "depth_noise_only",
    )
    asynchronous = name == "hidden_latency_6ms"
    calibration, tracks, samples, truth = scene(
        6,
        exact=not noise,
        skew=0.012 if asynchronous else 0,
        bias=0.04 if name == "depth_bias_4cm" else 0,
        weak=name == "weak_collinear",
    )
    _, clean, _, _ = scene(6, exact=True, skew=0, weak=name == "weak_collinear")
    evaluation_points = {t.track_id: t.point_S for t in clean.landmarks}
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.025, (96, 1))
    initial = np.array([f.T_SR for f in tracks.frames])
    frames, native = [], {}
    for frame in tracks.frames:
        i = frame.frame_index
        raw = samples[i].points_D.copy()
        pixels = frame.features.pixels.copy()
        if name in (
            "independent_noise",
            "correlated_depth_noise",
            "independent_noise_clean_root",
            "pixel_noise_only",
            "depth_noise_only",
        ):
            pixel_noise = rng.normal(0, 0.5, pixels.shape)
            L = np.linalg.inv(samples[i].whitening_D)
            depth_noise = np.einsum("nij,nj->ni", L, rng.normal(0, 0.5, raw.shape))
            if name != "independent_noise_clean_root" or i:
                if name != "depth_noise_only":
                    pixels += pixel_noise
                if name != "pixel_noise_only":
                    raw += depth_noise
        if name == "correlated_depth_noise":
            raw *= 1 + common / raw[:, 2:3]
        if name == "depth_scale_2percent":
            raw *= 1.02
        dt = (0.012 if i % 2 else -0.012) if asynchronous else 0
        motion = pair_motion(
            initial,
            1 + 0.1 * np.arange(6),
            1 + 0.1 * i,
            1 + 0.1 * i + dt + (0.006 if asynchronous else 0),
        )
        transform = motion.T_RcRd @ calibration.T_RD
        camera_points = transform_points(raw, transform)
        native[i] = NativeSamples(
            raw,
            camera_points,
            samples[i].source_index,
            native_whitening(raw, calibration.K_D, motion),
            transform,
        )
        frames.append(
            replace(frame, features=RGBDFeatures(pixels, None, camera_points))
        )
    model = replace(
        tracks,
        frames=tuple(frames),
        landmarks=tuple(
            replace(t, point_S=np.full(3, np.nan)) for t in tracks.landmarks
        ),
    )
    config = PoseOnlyConfig(min_grid_coverage=0 if name == "weak_collinear" else 0.15)
    factors, _ = make_direct_factors(model, native, list(range(6)), calibration, config)
    if name == "depthless_scale_gauge":
        factors = replace(
            factors, native_points=np.full_like(factors.native_points, np.nan)
        )
    free = range(1, 6)
    information = direct_information(initial, factors, calibration.K_R, free, config)
    ids, _, points = landmark_layout(factors)
    truth_points = np.array([evaluation_points[int(i)] for i in ids])
    methods = {"sparse": (initial, points, {"status": "initialization"})}
    if (
        information["condition"] is not None
        and information["condition"] <= config.max_condition
    ):
        fixed_poses, diagnostic = solve_fixed(
            initial, factors, calibration.K_R, free, config
        )
        methods["fixed"] = fixed_poses, points, diagnostic
        fitted, fitted_points, diagnostic = solve_joint(
            initial, factors, calibration.K_R, free, config
        )
        methods["joint"] = fitted, fitted_points, diagnostic
    else:
        for method in ("fixed", "joint"):
            methods[method] = (
                initial,
                points,
                {"status": "rejected", "reason": "common observability gate"},
            )
    result = {}
    for method_name, (poses, fitted_points, diagnostic) in methods.items():
        errors = [pose_error(p, t) for p, t in zip(poses, truth, strict=True)]
        result[method_name] = {
            "position_m": error_statistics(e[0] for e in errors),
            "rotation_deg": error_statistics(e[1] for e in errors),
            "landmark_m": error_statistics(
                np.linalg.norm(fitted_points - truth_points, axis=1)
            ),
            "endpoint": list(pose_error(poses[-1], truth[-1])),
            "solver": diagnostic,
        }
    return {"seed": seed, "information": information, "methods": result}


def controls(seeds=(11, 23, 37, 51, 71)):
    return {
        name: [case(name, seed) for seed in seeds]
        for name in (
            "exact",
            "independent_noise",
            "independent_noise_clean_root",
            "pixel_noise_only",
            "depth_noise_only",
            "noisy_initialization",
            "correlated_depth_noise",
            "depth_bias_4cm",
            "depth_scale_2percent",
            "hidden_latency_6ms",
            "weak_collinear",
            "depthless_scale_gauge",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("synthetic output already exists")
    report = {
        "code_sha256": code_hash(
            Path(__file__), Path(__file__).with_name("temporal_controls.py")
        ),
        "cases": controls(),
    }
    repeated = controls()
    if report["cases"] != repeated:
        raise AssertionError("synthetic repeat differs")
    report["repeat_exact"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
