"""Synthetic controls for frozen within-pair motion and fixed-landmark fitting."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from scene_recall.data import Calibration
from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.fixed_landmarks import make_direct_factors, solve_fixed
from scene_recall.multiview.model import (
    FeatureFrame,
    Landmark,
    TrackConfig,
    TrackResult,
)
from scene_recall.multiview.pose_only import PoseOnlyConfig, retract
from scene_recall.multiview.temporal import (
    NativeSamples,
    PairMotion,
    native_whitening,
    pair_motion,
)
from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.run_metadata import code_hash


def scene(count=12, *, exact=False, skew=0.012, bias=0.0, extrinsic=False, weak=False):
    rng = np.random.default_rng(51)
    K = np.array([[300.0, 0, 160], [0, 310.0, 120], [0, 0, 1.0]])
    T_RD = (
        retract(np.eye(4), np.array([0.03, 0.01, 0, 0.01, -0.02, 0]))
        if extrinsic
        else np.eye(4)
    )
    calibration = Calibration(K, K, T_RD, (240, 320), (240, 320))
    world = rng.uniform([-0.6, -0.45, 2.5], [0.6, 0.45, 3.5], (96, 3))
    if weak:
        world[:, :2] = 0
    frames, samples, truth = [], {}, []
    for i in range(count):
        t = 0.1 * i
        color = retract(
            np.eye(4),
            np.array([0.12 * t, 0.02 * t * t, 0, 0, 0.12 * t + 0.04 * t * t, 0.02 * t]),
        )
        d = t + (skew if i % 2 else -skew)
        depth = retract(
            np.eye(4),
            np.array([0.12 * d, 0.02 * d * d, 0, 0, 0.12 * d + 0.04 * d * d, 0.02 * d]),
        )
        relative = np.linalg.inv(color) @ depth @ T_RD
        pD = transform_points(world, np.linalg.inv(depth @ T_RD))
        pD *= 1 + bias / pD[:, 2:3]
        pR = transform_points(pD, relative)
        pixels = project_points(transform_points(world, np.linalg.inv(color)), K)
        initial = color.copy()
        if i and not exact:
            initial = retract(initial, rng.normal(0, [0.006] * 3 + [0.002] * 3))
        frames.append(
            FeatureFrame(i, 0, 1 + t, initial, RGBDFeatures(pixels, None, pR))
        )
        samples[i] = NativeSamples(
            pD,
            pR,
            np.arange(len(pD)),
            native_whitening(
                pD,
                K,
                PairMotion(
                    relative @ np.linalg.inv(T_RD), "exact_motion_control", True
                ),
            ),
            relative,
        )
        truth.append(color)
    tracks = []
    for j in range(len(world)):
        measured = np.mean(
            [transform_points(f.features.points_R[j], f.T_SR) for f in frames], axis=0
        )
        tracks.append(
            Landmark(
                j,
                0,
                tuple((i, j) for i in range(count)),
                measured,
                "supported",
                count,
                np.zeros(count),
                np.zeros(count),
                0.0,
                0.0,
            )
        )
    return (
        calibration,
        TrackResult(tuple(frames), tuple(tracks), (), (), TrackConfig(), {}),
        samples,
        np.array(truth),
    )


def controls():
    """Score after fitting, with a separately labeled ideal known-motion control.

    Other variants build motion and landmarks from the supplied initialization
    and sensor measurements. Evaluation points never initialize a landmark.
    """
    reports = {}
    for name, options in (
        ("synchronized_exact", {"exact": True, "skew": 0.0}),
        ("asynchronous_exact", {"exact": True}),
        ("asynchronous_sparse_noise", {"exact": False}),
        ("coherent_depth_bias", {"exact": True, "bias": 0.04}),
        ("nonidentity_calibration", {"exact": True, "extrinsic": True}),
        ("weak_collinear_geometry", {"exact": True, "weak": True}),
    ):
        calibration, tracks, samples, truth = scene(6, **options)
        _, noiseless, _, _ = scene(
            6,
            exact=True,
            skew=options.get("skew", 0.012),
            extrinsic=options.get("extrinsic", False),
            weak=options.get("weak", False),
        )
        evaluation_points = {t.track_id: t.point_S for t in noiseless.landmarks}
        config = (
            PoseOnlyConfig(min_grid_coverage=0.0)
            if options.get("weak")
            else PoseOnlyConfig()
        )
        initial = np.array([f.T_SR for f in tracks.frames])
        estimated_motions = {}
        hidden_motions = {}
        for i in range(6):
            dt = options.get("skew", 0.012) * (1 if i % 2 else -1)
            estimated_motions[i] = pair_motion(
                initial, 1 + 0.1 * np.arange(6), 1 + 0.1 * i, 1 + 0.1 * i + dt
            )
            hidden_motions[i] = pair_motion(
                initial, 1 + 0.1 * np.arange(6), 1 + 0.1 * i, 1 + 0.1 * i + dt + 0.006
            )
        variants = {}
        for variant in (
            "known_motion_control",
            "static_motion",
            "interpolated_motion",
            "hidden_clock_error_6ms",
        ):
            model_frames, model_samples = [], {}
            for frame in tracks.frames:
                i = frame.frame_index
                # Raw RGB pixels and native points are generative measurements.
                # Every deployable mean transform comes from initialization only.
                estimated = (
                    hidden_motions[i]
                    if variant == "hidden_clock_error_6ms"
                    else estimated_motions[i]
                )
                if variant == "known_motion_control":
                    transform = samples[i].T_RcD
                elif variant == "static_motion":
                    transform = calibration.T_RD
                else:
                    transform = estimated.T_RcRd @ calibration.T_RD
                raw = samples[i].points_D
                camera_points = transform_points(raw, transform)
                whitening = native_whitening(raw, calibration.K_D, estimated)
                model_samples[i] = NativeSamples(
                    raw, camera_points, samples[i].source_index, whitening, transform
                )
                model_frames.append(
                    replace(
                        frame,
                        features=RGBDFeatures(
                            frame.features.pixels, None, camera_points
                        ),
                    )
                )
            model = replace(
                tracks,
                frames=tuple(model_frames),
                landmarks=tuple(
                    replace(t, point_S=np.zeros(3)) for t in tracks.landmarks
                ),
            )
            factors, _ = make_direct_factors(
                model, model_samples, list(range(6)), calibration, config
            )
            variants[variant] = factors
        report = {}
        for variant, fixed in variants.items():
            poses, diagnostic = solve_fixed(
                initial, fixed, calibration.K_R, range(1, 6), config
            )
            errors = [pose_error(p, t) for p, t in zip(poses, truth, strict=True)]
            report[variant] = {
                "initial_landmark_error_m": error_statistics(
                    np.linalg.norm(p - evaluation_points[int(t)])
                    for p, t in zip(fixed.points, fixed.track_id, strict=True)
                ),
                "position_m": error_statistics(e[0] for e in errors),
                "rotation_deg": error_statistics(e[1] for e in errors),
                "solver": diagnostic,
            }
        reports[name] = report
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("control output already exists")
    report = {"code_sha256": code_hash(Path(__file__)), "cases": controls()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
