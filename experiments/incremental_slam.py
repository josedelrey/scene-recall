"""Reference-free incremental smoothing on verified canonical baseline inputs.

Baseline references enter scoring only. Optional immutable feature caches speed
controlled experiments, with raw-feature replay available by omitting --features.
No saved trajectory initializes the estimator. Existing runs are never modified.
"""

import argparse
import hashlib
import json
import os
import platform
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
from incremental_local_map import common_errors
from refinement_tsdf import evaluate_tsdf

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.features import RGBDFeatures, extract_rgbd_features
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.mapping.heldout import check_cohort, run_heldout
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.evaluation import error_statistics, evaluate_trajectory
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.odometry.pipeline import PairEstimate, TrajectoryFrame
from scene_recall.run_metadata import code_hash, peak_rss, sha256
from scene_recall.slam.system import IncrementalSLAM, RevisionArchive, SLAMConfig


def variants():
    native = SLAMConfig()
    return {
        "control": replace(native, optimize=False),
        "native": native,
        "axial": replace(native, solver=replace(native.solver, depth_model="axial")),
        "static": replace(native, exposure="static"),
        "wide": replace(native, window_size=16),
        "sqrt": replace(native, marginalization="sqrt"),
        "turnover": replace(native, marginalization="sqrt", stale_keyframes=2),
    }


def with_revised_poses(rows, poses):
    """Keep each trajectory's relative-pose diagnostics consistent with its poses."""
    result = []
    for i, (row, pose) in enumerate(zip(rows, poses, strict=True)):
        pair = row.pair
        if pair is not None and pair.T_C1C0 is not None:
            pair = replace(pair, T_C1C0=np.linalg.inv(pose) @ poses[i - 1])
        result.append(replace(row, T_SC_D=pose, pair=pair))
    return result


def run_case(baseline, output, policies, *, features=None, maps=False):
    if output.exists() or output.is_symlink():
        raise ValueError("choose a fresh output directory")
    baseline_report, baseline_rows = load_trajectory(baseline)
    cv2.setNumThreads(1)
    configurations = variants()
    systems = {name: IncrementalSLAM(configurations[name]) for name in policies}
    archives = {name: RevisionArchive() for name in policies}
    histories, revisions = {n: [] for n in policies}, {n: [] for n in policies}
    packets, points = {n: [] for n in policies}, {n: {} for n in policies}
    emitted = {n: [] for n in policies}
    identity = {"package_sha256": code_hash(), "driver_sha256": sha256(Path(__file__))}
    protocol = {
        "phase": "frozen before evaluation",
        "baseline_report_sha256": sha256(baseline / "report.json"),
        "baseline_trajectory_sha256": sha256(baseline / "trajectory.npz"),
        "features_sha256": sha256(features) if features else None,
        "policies": {n: asdict(configurations[n]) for n in policies},
        "implementation": identity,
        "selection": "All specified inputs, no score-selected thresholds or replacements",
        "reconstruction": "Reference-target voxel fusion and first-chunk TSDF, 2cm voxels, stride8, every10 training, offset5, max24 views",
        "feature_mode": "immutable cached replay" if features else "raw extraction",
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    cache = np.load(features, allow_pickle=False) if features else None

    latest_features = {}

    def extract(observation, calibration, sift, threshold):
        key = str(observation.frame_index)
        if key in latest_features:
            return latest_features[key]
        if cache is None:
            packet = extract_rgbd_features(observation, calibration, sift, threshold)
        else:
            packet = RGBDFeatures(
                cache[key + "_pixels"],
                cache.get(key + "_descriptors"),
                cache[key + "_points_R"],
            )
        for array in (packet.pixels, packet.points_R, packet.descriptors):
            if array is not None:
                array.setflags(write=False)
        latest_features.clear()
        latest_features[key] = packet
        return packet

    digest, metadata, original = hashlib.sha256(), {}, []
    reader, _ = open_source(baseline_report)
    started = perf_counter()
    with (
        reader,
        patch("scene_recall.odometry.local_map.extract_rgbd_features", extract),
    ):
        validate_source(reader, baseline_report)
        for observation in verified_observations(
            reader, baseline_report, baseline_rows, digest, metadata
        ):
            # Reuse the same stripped object within each serial stream. Reference
            # poses are retained exclusively in the runner's evaluation archive.
            clean = replace(observation, T_WC_D=None)
            original.append((observation.timestamp, observation.T_WC_D))
            for name, system in systems.items():
                pose, pair, changed, diagnostic = system.process(
                    clean, reader.calibration
                )
                emitted[name].append(pose.copy())
                points[name].update(
                    {int(k): v.copy() for k, v in system.points.items()}
                )
                if system.last_measurement is not None:
                    packets[name].append(system.last_measurement)
                archive = archives[name]
                archive.append(
                    pose,
                    keyframe=clean.frame_index in system.frontend.keyframes,
                    boundary=pair is None or pair.T_C1C0 is None,
                )
                dirty = archive.apply(changed)
                if changed:
                    revisions[name].append(
                        {
                            "at_frame": clean.frame_index,
                            "version": archive.version,
                            "keyframes": sorted(changed),
                            "dirty_frames": dirty,
                        }
                    )
                histories[name].append((system.segment_id, pair, diagnostic))
            if len(original) % 50 == 0:
                print(f"Processed {len(original)} inputs", flush=True)
    if cache is not None:
        cache.close()
    elapsed = perf_counter() - started
    result = {"variants": {}, "wall_seconds": elapsed, "peak_rss_mib": peak_rss()}
    for name, system in systems.items():
        archive = archives[name]
        rows, frame_reports = [], []
        for i, (segment, pair, diagnostic) in enumerate(histories[name]):
            status = (
                "initialized"
                if i == 0
                else "lost"
                if pair.T_C1C0 is None
                else "tracked"
            )
            # Export final relative transforms consistent with revised poses.
            if pair is not None and pair.T_C1C0 is not None:
                pair = PairEstimate(
                    np.linalg.inv(archive.poses[i]) @ archive.poses[i - 1],
                    diagnostics=pair.diagnostics,
                )
            rows.append(
                TrajectoryFrame(
                    i,
                    original[i][0],
                    segment,
                    status,
                    archive.poses[i],
                    original[i][1],
                    pair,
                )
            )
            frame_reports.append(
                {
                    **baseline_report["frames"][i],
                    "segment_id": segment,
                    "status": status,
                    "reason": None if pair is None else pair.reason,
                    "diagnostics": diagnostic,
                }
            )
        evaluation = evaluate_trajectory(rows, (1, 5, 10, 20, 50, 100))
        emitted_rows = with_revised_poses(rows, emitted[name])
        measurements = (
            {
                key: np.concatenate([p[key] for p in packets[name]])
                for key in packets[name][0]
            }
            if packets[name]
            else {}
        )
        consistency = measurement_consistency(
            measurements,
            points[name],
            archive.poses,
            np.array(baseline_report["calibration"]["K_R"]),
        )
        report = {
            **baseline_report,
            "backend": "incremental_slam",
            "initial_pose_stage": None,
            "config": asdict(system.config),
            "evaluation": evaluation,
            "emitted_evaluation": evaluate_trajectory(
                emitted_rows, (1, 5, 10, 20, 50, 100)
            ),
            "measurement_consistency": consistency,
            "frames": frame_reports,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "opencv_threads": cv2.getNumThreads(),
                "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
                "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
                "code_sha256": identity["package_sha256"],
            },
            "timing": {
                "tracking_seconds": sum(d[2]["total_seconds"] for d in histories[name]),
                "feature_mode": protocol["feature_mode"],
                "concurrent_policies": len(policies),
            },
            "experiment": {
                "protocol_sha256": sha256(output / "protocol.json"),
                "revision_version": archive.version,
                "reconstruction_requires_version": archive.version,
                "emitted_pair_consistency": True,
            },
        }
        save_run(output / name, rows, report, report["source"]["start"])
        np.savez_compressed(
            output / name / "measurements.npz",
            **measurements,
            final_track_ids=np.array(sorted(points[name]), dtype=np.int64),
            final_points=np.array(
                [points[name][k] for k in sorted(points[name])]
            ).reshape(-1, 3),
        )
        np.savez_compressed(
            output / name / "emitted_trajectory.npz", T_SC_D=np.array(emitted[name])
        )
        (output / name / "revisions.json").write_text(
            json.dumps(revisions[name], indent=2) + "\n"
        )
        diagnostics = [h[2] for h in histories[name]]
        result["variants"][name] = {
            "evaluation": evaluation,
            "common_with_baseline": {
                str(k): common_errors(baseline_rows, rows, k) for k in (1, 10, 50, 100)
            },
            "solver_statuses": dict(
                Counter(d["solver"]["status"] for d in diagnostics if "solver" in d)
            ),
            "solver_reasons": dict(
                Counter(d["solver"]["reason"] for d in diagnostics if "solver" in d)
            ),
            "exposure_statuses": dict(
                Counter(
                    d["exposure_status"] for d in diagnostics if "exposure_status" in d
                )
            ),
            "graph_restarts": [
                {"frame": i, "reason": d["graph_restart"]}
                for i, d in enumerate(diagnostics)
                if "graph_restart" in d
            ],
            "marginalizations": sum("marginalization" in d for d in diagnostics),
            "max_active_cameras": max(d.get("active_cameras", 0) for d in diagnostics),
            "max_active_points": max(d.get("active_points", 0) for d in diagnostics),
            "max_active_observations": max(
                d.get("active_observations", 0) for d in diagnostics
            ),
            "revisions": len(revisions[name]),
            "report_sha256": sha256(output / name / "report.json"),
            "trajectory_sha256": sha256(output / name / "trajectory.npz"),
        }
    if identity != {
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
    }:
        raise ValueError("implementation changed during evaluation")
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if maps:
        evaluate_maps(baseline, output, policies)
    return result


def measurement_consistency(measurements, points, poses, K):
    """Recheck all raw observations, including historical marginalized factors.

    These selected residuals diagnose fit and linearization drift. They are not
    independent landmark ground truth or posterior consistency statistics.
    """
    if not measurements:
        return {"observations": 0}
    view, ids = measurements["view"], measurements["track_id"]
    xyz = np.array([points[int(k)] for k in ids])
    rgb_poses = np.array(poses)[view] @ np.linalg.inv(measurements["T_RcD"])
    q = np.einsum(
        "nij,nj->ni", rgb_poses[:, :3, :3].transpose(0, 2, 1), xyz - rgb_poses[:, :3, 3]
    )
    pixels = np.linalg.norm(project_points(q, K) - measurements["pixels"], axis=1)
    native = np.array(
        [
            transform_points(p, np.linalg.inv(T))
            for p, T in zip(q, measurements["T_RcD"], strict=True)
        ]
    )
    valid = measurements["native_index"] >= 0
    unique_rgb = len(set(zip(view, measurements["feature_index"], strict=True))) == len(
        view
    )
    unique_depth = len(
        set(zip(view[valid], measurements["native_index"][valid], strict=True))
    ) == int(valid.sum())
    if not unique_rgb or not unique_depth:
        raise ValueError("raw measurement has multiple graph owners")
    return {
        "observations": len(view),
        "native_observations": int(valid.sum()),
        "unique_rgb_ownership": unique_rgb,
        "unique_depth_ownership": unique_depth,
        "final_rgb_error_px": error_statistics(pixels[np.isfinite(pixels)]),
        "rgb_within_3px_fraction": float(np.mean(np.isfinite(pixels) & (pixels <= 3))),
        "cheirality_failures": int((q[:, 2] <= 0.05).sum()),
        "final_native_distance_m": error_statistics(
            np.linalg.norm(native[valid] - measurements["native_points"][valid], axis=1)
        ),
        "final_native_axial_error_m": error_statistics(
            np.abs(native[valid, 2] - measurements["native_points"][valid, 2])
        ),
        "point_displacement_from_seed_m": error_statistics(
            np.linalg.norm(xyz - measurements["points"], axis=1)
        ),
    }


def evaluate_maps(baseline, output, policies):
    results = {}
    for name in policies:
        paths = {"budget": baseline, name: output / name}
        try:
            fusion_path = output / ("maps_" + name)
            fusion = (
                json.loads((fusion_path / "report.json").read_text())
                if fusion_path.exists()
                else run_heldout(
                    list(paths.values()),
                    fusion_path,
                    config=MappingConfig(pixel_stride=8, max_keyframes_per_submap=24),
                )
            )
            loaded = {path: load_trajectory(path) for path in paths.values()}
            expected = {
                report["backend"]: {
                    "report_sha256": sha256(path / "report.json"),
                    "trajectory_sha256": sha256(path / "trajectory.npz"),
                }
                for path, (report, _) in loaded.items()
            }
            config = MappingConfig(pixel_stride=8, max_keyframes_per_submap=24)
            if (
                fusion["odometry"] != expected
                or fusion["mapping_config"] != asdict(config)
                or fusion["offset"] != 5
                or fusion["render"]["radii_m"] != [0.01]
                or fusion["split"] != check_cohort(loaded, config, 5)
            ):
                raise ValueError("saved fusion inputs, settings or cohort changed")
        except ValueError as error:
            if "segmentation" not in str(error):
                raise
            results[name] = {"status": "withheld", "reason": str(error)}
            continue
        results[name] = evaluate_tsdf(
            paths, list(paths), output / ("tsdf_" + name), fusion
        )
    (output / "map_comparison.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(variants()),
        default=["control", "native", "axial", "static"],
    )
    parser.add_argument("--evaluate-maps", action="store_true")
    parser.add_argument("--maps-only", action="store_true")
    args = parser.parse_args()
    if len(args.variants) != len(set(args.variants)):
        parser.error("variants must be unique")
    if args.maps_only:
        evaluate_maps(args.baseline_run, args.output_dir, args.variants)
    else:
        run_case(
            args.baseline_run,
            args.output_dir,
            args.variants,
            features=args.features,
            maps=args.evaluate_maps,
        )


if __name__ == "__main__":
    main()
