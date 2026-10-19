"""Training-only extraction, pose refinement and independent trajectory exports."""

import copy
import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.features import extract_rgbd_features
from scene_recall.mapping.heldout import heldout_split
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.multiview.io import measurement_arrays
from scene_recall.multiview.model import FeatureFrame, TrackConfig
from scene_recall.multiview.pose_only import PoseOnlyConfig, refine_training_poses
from scene_recall.multiview.tracking import build_tracks
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.odometry.pipeline import PairEstimate, validate_trajectory
from scene_recall.run_metadata import code_hash, sha256


def prepare_training_tracks(odometry_run, mapping_config, offset, track_config=None):
    """Rebuild identities using training observations only, after frozen splitting.

    Existing all-view tracks cannot safely be filtered after association, because
    an excluded view could already have merged two feature components.
    """
    report, rows = load_trajectory(odometry_run)
    split = heldout_split(rows, mapping_config, offset)
    training = set(split["train_indices"])
    track_config = track_config or TrackConfig()
    reader, _ = open_source(report)
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=track_config.max_features)
    digest, metadata, features = hashlib.sha256(), {}, []
    started = perf_counter()
    with reader:
        validate_source(reader, report)
        calibration = reader.calibration
        for observation in verified_observations(
            reader, report, rows, digest, metadata
        ):
            if observation.frame_index in training:
                # FeatureFrame contains no reference field.
                extracted = extract_rgbd_features(
                    observation, calibration, sift, track_config.depth_edge_threshold_m
                )
                features.append(
                    FeatureFrame.from_trajectory(
                        rows[observation.frame_index], calibration, extracted
                    )
                )
    result = build_tracks(features, calibration, track_config)
    return report, rows, split, calibration, result, perf_counter() - started


def interpolate_corrections(rows, training_poses, calibration):
    """Transport keyframe corrections to intervening VO poses without image use.

    Interpolate left correction translation linearly and rotation geodesically
    in each segment. Both endpoints are training frames. No held-out measurement
    or reference pose enters this deterministic trajectory completion.
    """
    refined = []
    for row in rows:
        keys = sorted(i for i in training_poses if rows[i].segment_id == row.segment_id)
        if not keys or keys[0] > row.frame_index or keys[-1] < row.frame_index:
            raise ValueError("training poses must bracket every segment")
        if row.frame_index in training_poses:
            pose = training_poses[row.frame_index] @ calibration.T_RD
        else:
            j = int(np.searchsorted(keys, row.frame_index))
            lo, hi = keys[j - 1], keys[j]
            first = (
                training_poses[lo] @ calibration.T_RD @ np.linalg.inv(rows[lo].T_SC_D)
            )
            last = (
                training_poses[hi] @ calibration.T_RD @ np.linalg.inv(rows[hi].T_SC_D)
            )
            alpha = (row.frame_index - lo) / (hi - lo)
            correction = np.eye(4)
            vector = cv2.Rodrigues(first[:3, :3].T @ last[:3, :3])[0].ravel()
            correction[:3, :3] = first[:3, :3] @ cv2.Rodrigues(alpha * vector)[0]
            correction[:3, 3] = (1 - alpha) * first[:3, 3] + alpha * last[:3, 3]
            pose = correction @ row.T_SC_D
        if (
            row.frame_index == 0
            or rows[row.frame_index - 1].segment_id != row.segment_id
        ):
            pose = row.T_SC_D.copy()
        pair = None
        if row.frame_index:
            original = (
                None
                if row.status == "lost"
                else np.linalg.inv(row.T_SC_D) @ rows[row.frame_index - 1].T_SC_D
            )
            pair = PairEstimate(
                None
                if row.status == "lost"
                else np.linalg.inv(pose) @ refined[-1].T_SC_D,
                "preserved VO loss" if row.status == "lost" else None,
                original,
            )
        refined.append(replace(row, T_SC_D=pose, pair=pair))
    validate_trajectory(refined)
    return refined


def run_pose_experiment(
    odometry_run: Path,
    output: Path,
    *,
    config=None,
    mapping_config=None,
    offset=2,
    controls=True,
):
    """Publish a fresh experiment with baseline, variants and measurement evidence."""
    if output.exists() or output.is_symlink():
        raise ValueError("pose experiment output already exists")
    config = config or PoseOnlyConfig()
    if config.formulation != "anchored" or config.loss != "cauchy":
        raise ValueError("the primary experiment requires anchored Cauchy settings")
    mapping_config = mapping_config or MappingConfig(keyframe_stride=5)
    baseline, rows, split, calibration, tracks, preparation = prepare_training_tracks(
        odometry_run, mapping_config, offset
    )
    settings = {"anchored": config}
    if controls:
        settings["frozen"] = replace(config, formulation="frozen")
        settings["linear"] = replace(config, loss="linear")
    intervals, times = (1, 5, 10, 20, 50, 100, 200), (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    report = {
        "format_version": 1,
        "experiment": "training_only_pose_multiview",
        "baseline_run": str(odometry_run.resolve()),
        "baseline_report_sha256": sha256(odometry_run / "report.json"),
        "baseline_trajectory_sha256": sha256(odometry_run / "trajectory.npz"),
        "source": baseline["source"],
        "calibration": baseline["calibration"],
        "mapping_config": asdict(mapping_config),
        "offset": offset,
        "split": split,
        "track_config": asdict(tracks.config),
        "preparation_seconds": preparation,
        "training_track_count": len(tracks.landmarks),
        "training_supported_tracks": sum(
            t.status == "supported" for t in tracks.landmarks
        ),
        "baseline_evaluation": evaluate_trajectory(
            rows, intervals, time_intervals_s=times
        ),
        "variants": {},
        "code_sha256": code_hash(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "input_policy": "training-only tracks, anchor depth fixed, GT excluded, full-frame Sparse initializer retained",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pose-only-", dir=output.parent) as temp:
        staging = Path(temp) / "result"
        staging.mkdir()
        np.savez_compressed(
            staging / "training_measurements.npz", **measurement_arrays(tracks)
        )
        for name, settings_for_variant in settings.items():
            poses, diagnostics = refine_training_poses(
                tracks, calibration, settings_for_variant
            )
            refined = interpolate_corrections(rows, poses, calibration)
            evaluation = evaluate_trajectory(refined, intervals, time_intervals_s=times)
            variant = copy.deepcopy(baseline)
            variant.update(
                backend=f"pose_only_{name}",
                config=asdict(settings_for_variant),
                initial_pose_stage="accepted_sparse",
                evaluation=evaluation,
                pose_only={
                    "config": asdict(settings_for_variant),
                    "baseline_trajectory_sha256": report["baseline_trajectory_sha256"],
                    "initialization_config": baseline["config"],
                },
            )
            variant["environment"] = {
                **baseline["environment"],
                "code_sha256": code_hash(),
            }
            initialization = baseline["timing"]["tracking_seconds"]
            total = initialization + preparation + diagnostics["optimization_seconds"]
            variant["timing"] = {
                "initialization_seconds": initialization,
                "training_preparation_seconds": preparation,
                "pose_only_seconds": diagnostics["optimization_seconds"],
                "tracking_seconds": total,
                "frames_per_second": len(rows) / total,
            }
            for metadata, row in zip(variant["frames"], refined):
                metadata["reason"] = (
                    metadata.get("reason") if row.status == "lost" else None
                )
                metadata["diagnostics"] = {"pose_only_variant": name}
            save_run(staging / name, refined, variant, baseline["source"]["start"])
            report["variants"][name] = {
                "config": asdict(settings_for_variant),
                "trajectory_sha256": sha256(staging / name / "trajectory.npz"),
                "report_sha256": sha256(staging / name / "report.json"),
                **diagnostics,
                "evaluation": evaluation,
            }
            print(
                f"Completed {name}: {sum(w['status'] == 'refined' for w in diagnostics['windows'])}/{len(diagnostics['windows'])} refined windows",
                flush=True,
            )
        repeated, _ = refine_training_poses(tracks, calibration, config)
        with np.load(staging / "anchored/trajectory.npz") as archive:
            rebuilt = interpolate_corrections(rows, repeated, calibration)
            np.testing.assert_array_equal(
                archive["T_SC_D"], np.array([r.T_SC_D for r in rebuilt])
            )
        report["repeat_exact"] = True
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if output.exists() or output.is_symlink():
            raise ValueError("output appeared during pose experiment")
        staging.rename(output)
    return report
