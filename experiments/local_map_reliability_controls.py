"""Known-correspondence noise controls for conditional local-map pose fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from local_map_measurements import point_covariance, refine_measurements

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.odometry.evaluation import pose_error
from scene_recall.odometry.sparse import (
    SparseConfig,
    estimate_sparse_pose,
    reprojection_errors,
)
from scene_recall.run_metadata import sha256


def noisy_points(points, K, rng, bias=0):
    pixels = project_points(points, K) + rng.normal(size=(len(points), 2))
    z = (
        points[:, 2]
        + rng.normal(size=len(points)) * (0.005 + 0.003 * points[:, 2] ** 2)
        + bias
    )
    return np.column_stack(
        ((pixels - K[:2, 2]) / np.diag(K)[:2] * z[:, None], z)
    ), pixels


def run_controls(output, trials=50):
    if output.exists():
        raise ValueError("choose a fresh control output")
    cv2.setNumThreads(1)
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    truth = np.eye(4)
    truth[:3, :3] = cv2.Rodrigues(np.array([0.02, 0.04, -0.01]))[0]
    truth[:3, 3] = [0.08, -0.01, 0.02]
    results = {}
    for condition in ("independent", "coherent_4cm", "hidden_depth_motion"):
        rows = []
        for seed in range(trials):
            rng = np.random.default_rng(seed + 700)
            p = rng.uniform([-0.8, -0.6, 1.2], [0.8, 0.6, 5.0], (240, 3))
            target = transform_points(p, truth)
            source, _ = noisy_points(
                p, K, rng, 0.04 if condition == "coherent_4cm" else 0
            )
            measured, pixels = noisy_points(
                target, K, rng, 0.04 if condition == "coherent_4cm" else 0
            )
            if condition == "hidden_depth_motion":
                # Unmodelled 6ms at 1m/s along the optical axis changes depth,
                # while RGB coordinates remain those at the color exposure.
                measured[:, 2] += 0.006
            fit = estimate_sparse_pose(source, pixels, K, SparseConfig())
            if fit.T_C1C0 is None:
                raise ValueError("synthetic PnP control failed")
            keep = reprojection_errors(source, pixels, K, fit.T_C1C0) <= 3
            poses = {"legacy": fit.T_C1C0}
            for name in ("rgb", "rgbd"):
                poses[name], _ = refine_measurements(
                    source[keep],
                    pixels[keep],
                    measured[keep],
                    point_covariance(source[keep], K),
                    fit.T_C1C0,
                    K,
                    name == "rgbd",
                )
            rows.append(
                {
                    "seed": seed,
                    "source_point_rmse_m": float(
                        np.sqrt(np.mean(np.sum((source - p) ** 2, axis=1)))
                    ),
                    "variants": {
                        name: {
                            "translation_m": pose_error(T, truth)[0],
                            "rotation_deg": pose_error(T, truth)[1],
                        }
                        for name, T in poses.items()
                    },
                }
            )
        results[condition] = {
            "trials": rows,
            "summary": {
                name: {
                    metric: float(
                        np.sqrt(
                            np.mean([r["variants"][name][metric] ** 2 for r in rows])
                        )
                    )
                    for metric in ("translation_m", "rotation_deg")
                }
                for name in ("legacy", "rgb", "rgbd")
            },
        }
    output.mkdir(parents=True)
    report = {
        "seed_range": [700, 700 + trials],
        "driver_sha256": sha256(Path(__file__)),
        "solver_sha256": sha256(Path(__file__).with_name("local_map_measurements.py")),
        "conditions": results,
        "scope": "Perfect identities with generated RGB-ray noise. Coherent bias and hidden exposure motion deliberately violate the fitting model. No thresholds are selected from these results.",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def run_outages(source, output):
    """Reuse the existing real-input corruption protocol without retuning it."""
    from dataclasses import asdict

    from local_map_measurements import MeasurementConfig, MeasurementMapBackend
    from local_map_outages import corrupt

    from scene_recall.datasets.tum import TUMReader
    from scene_recall.features import extract_rgbd_features
    from scene_recall.odometry.evaluation import evaluate_trajectory
    from scene_recall.odometry.pipeline import track_observations
    from scene_recall.odometry.sparse import SparseRGBDBackend

    if output.exists():
        raise ValueError("choose a fresh outage output")
    cv2.setNumThreads(1)
    with TUMReader(source, stop=60) as reader:
        observations = list(reader)
        calibration = reader.calibration
    if len(observations) != 60:
        raise ValueError("outage source must contain 60 observations")
    digest = hashlib.sha256()
    for observation in observations:
        for array in (observation.rgb, observation.depth):
            digest.update(array.tobytes())
    result = {
        "source": str(source.resolve()),
        "source_rgbd_sha256": digest.hexdigest(),
        "calibration": {
            "K_R": calibration.K_R.tolist(),
            "K_D": calibration.K_D.tolist(),
            "T_RD": calibration.T_RD.tolist(),
        },
        "gap": [20, 30],
        "frames": 60,
        "driver_sha256": sha256(Path(__file__)),
        "solver_sha256": sha256(Path(__file__).with_name("local_map_measurements.py")),
        "corruption_driver_sha256": sha256(
            Path(__file__).with_name("local_map_outages.py")
        ),
        "cases": {},
    }
    arrays = {}
    for mode in ("clean", "depth_gap", "rgb_gap", "depth_bias"):
        mutated = corrupt(observations, mode, 20, 30)
        sift = cv2.SIFT_create(nfeatures=1500)
        features = {
            o.frame_index: extract_rgbd_features(o, calibration, sift) for o in mutated
        }
        backends = {
            "budget": SparseRGBDBackend(SparseConfig(max_features=1500)),
            **{
                n: MeasurementMapBackend(
                    MeasurementConfig(
                        objective=n if n in ("rgb", "rgbd") else "legacy",
                        appearance="family" if n == "family" else "latest",
                    )
                )
                for n in ("legacy", "rgb", "rgbd", "family")
            },
        }
        result["cases"][mode] = {}
        for name, backend in backends.items():
            if name == "budget":
                backend._features = lambda o, c, measured=features: measured[
                    o.frame_index
                ]
            else:
                backend._extract = lambda o, c, measured=features: measured[
                    o.frame_index
                ]
            rows = list(track_observations(mutated, calibration, backend))
            result["cases"][mode][name] = {
                "config": asdict(backend.config),
                "evaluation": evaluate_trajectory(rows, (1, 5, 10, 20)),
                "lost_frames": [r.frame_index for r in rows if r.status == "lost"],
            }
            arrays[f"{mode}_{name}_poses"] = np.array([r.T_SC_D for r in rows])
            arrays[f"{mode}_{name}_segments"] = np.array([r.segment_id for r in rows])
            print(mode, name, result["cases"][mode][name]["lost_frames"], flush=True)
    output.mkdir(parents=True)
    np.savez_compressed(output / "trajectories.npz", **arrays)
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--tum-path", type=Path)
    args = parser.parse_args()
    if args.tum_path:
        run_outages(args.tum_path, args.output_dir)
    else:
        run_controls(args.output_dir, args.trials)
