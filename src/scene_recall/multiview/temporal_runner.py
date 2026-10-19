"""Offline training-only preparation and frozen-point temporal experiments."""

import copy
import hashlib
import json
import platform
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.datasets.exposure import exposure_times
from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.mapping.heldout import heldout_split
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.multiview.fixed_landmarks import refine_fixed
from scene_recall.multiview.io import measurement_arrays
from scene_recall.multiview.model import FeatureFrame, TrackConfig
from scene_recall.multiview.pose_only import PoseOnlyConfig
from scene_recall.multiview.pose_runner import interpolate_corrections
from scene_recall.multiview.temporal import (
    TimingConfig,
    bind_depth,
    pair_motion,
)
from scene_recall.multiview.tracking import build_tracks
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.run_metadata import code_hash, sha256


def motion_table(baseline, rows, train_indices, calibration, timing_config):
    """Use training pose proxies and adapter clock fields, without references."""
    times = [
        exposure_times(baseline["source"]["dataset"], f["provenance"])
        for f in baseline["frames"]
    ]
    table = {}
    for segment in sorted({r.segment_id for r in rows}):
        ids = [i for i in train_indices if rows[i].segment_id == segment]
        poses = np.array(
            [rows[i].T_SC_D @ np.linalg.inv(calibration.T_RD) for i in ids]
        )
        stamps = [np.nan if times[i][0] is None else times[i][0] for i in ids]
        for row in rows:
            if row.segment_id == segment:
                table[row.frame_index] = pair_motion(
                    poses, stamps, *times[row.frame_index], timing_config
                )
    return table, times


def prepare_temporal(baseline_run, mapping_config=None, offset=2, timing_config=None):
    """Prepare only training features and native depth bindings, then associate."""
    tick = perf_counter()
    mapping_config = mapping_config or MappingConfig(keyframe_stride=5)
    timing_config = timing_config or TimingConfig()
    baseline, rows = load_trajectory(baseline_run)
    split = heldout_split(rows, mapping_config, offset)
    reader, _ = open_source(baseline)
    calibration = reader.calibration
    motions, clocks = motion_table(
        baseline, rows, split["train_indices"], calibration, timing_config
    )
    motion_seconds = perf_counter() - tick
    track_config = TrackConfig()
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=track_config.max_features)
    samples, frames, image_seconds, binding_seconds = {}, [], 0.0, 0.0
    training = set(split["train_indices"])
    with reader:
        validate_source(reader, baseline)
        for observation in verified_observations(
            reader, baseline, rows, hashlib.sha256(), {}
        ):
            i = observation.frame_index
            if i not in training:
                continue
            started = perf_counter()
            keypoints, descriptors = sift.detectAndCompute(
                cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2GRAY), None
            )
            pixels = np.array([k.pt for k in keypoints], dtype=float).reshape(-1, 2)
            image_seconds += perf_counter() - started
            started = perf_counter()
            samples[i] = bind_depth(
                pixels, observation.depth, calibration, motions[i], timing_config
            )
            binding_seconds += perf_counter() - started
            features = RGBDFeatures(pixels, descriptors, samples[i].points_R)
            frames.append(
                FeatureFrame(
                    i,
                    rows[i].segment_id,
                    clocks[i][0],
                    rows[i].T_SC_D @ np.linalg.inv(calibration.T_RD),
                    features,
                )
            )
    started = perf_counter()
    tracks = build_tracks(frames, calibration, track_config)
    costs = {
        "motion_table_seconds": motion_seconds,
        "sift_seconds": image_seconds,
        "depth_binding_seconds": binding_seconds,
        "association_seconds": perf_counter() - started,
        "total_preparation_seconds": perf_counter() - tick,
    }
    # Gate-only availability on exactly the retained native sample population.
    shifts = []
    for sample in samples.values():
        valid = sample.source_index >= 0
        static_points = transform_points(sample.points_D[valid], calibration.T_RD)
        shift = np.linalg.norm(
            project_points(sample.points_R[valid], calibration.K_R)
            - project_points(static_points, calibration.K_R),
            axis=1,
        )
        shifts.extend(shift)
    costs["gate_only_samples_below_half_pixel"] = int(np.sum(np.array(shifts) <= 0.5))
    costs["gate_only_sample_population"] = len(shifts)
    return baseline, rows, split, calibration, tracks, samples, motions, clocks, costs


def export_color_poses(rows, poses, calibration, motions, *, static=False):
    """Complete corrections at RGB epochs and return depth-epoch segment poses.

    Training motion is frozen. Frame clocks at evaluation indices specify output
    epochs only, without using their images, depth or external poses. Re-rooting
    each segment keeps its first depth pose exactly identity.
    """
    unchanged = all(
        np.array_equal(poses[i], rows[i].T_SC_D @ np.linalg.inv(calibration.T_RD))
        for i in poses
    )
    if unchanged and (
        static or all(np.array_equal(m.T_RcRd, np.eye(4)) for m in motions.values())
    ):
        return rows.copy()
    pseudo = interpolate_corrections(rows, poses, calibration)
    converted, roots = {}, {}
    for row in pseudo:
        motion = np.eye(4) if static else motions[row.frame_index].T_RcRd
        pose = row.T_SC_D @ np.linalg.inv(calibration.T_RD) @ motion @ calibration.T_RD
        if row.segment_id not in roots:
            roots[row.segment_id] = np.linalg.inv(pose)
        depth_pose = roots[row.segment_id] @ pose
        if (
            row.frame_index == 0
            or rows[row.frame_index - 1].segment_id != row.segment_id
        ):
            depth_pose = np.eye(4)
        converted[row.frame_index] = depth_pose @ np.linalg.inv(calibration.T_RD)
    return interpolate_corrections(rows, converted, calibration)


def run_temporal_experiment(
    baseline_run,
    output,
    *,
    timing_config=None,
    config=None,
    mapping_config=None,
    offset=2,
):
    """Publish independent fixed-landmark controls with exact cached repeats."""
    if output.exists() or output.is_symlink():
        raise ValueError("temporal output already exists")
    timing_config = timing_config or TimingConfig()
    config = config or PoseOnlyConfig()
    mapping_config = mapping_config or MappingConfig(keyframe_stride=5)
    frozen_code = code_hash()
    baseline, rows, split, calibration, tracks, samples, motions, clocks, costs = (
        prepare_temporal(baseline_run, mapping_config, offset, timing_config)
    )
    initial = {f.frame_index: f.T_SR for f in tracks.frames}
    gate = {
        name: costs.pop(name)
        for name in (
            "gate_only_samples_below_half_pixel",
            "gate_only_sample_population",
        )
    }
    report = {
        "format_version": 1,
        "experiment": "fixed_landmark_two_epoch_rgbd",
        "residual_model": "rgb_2d_native_depth_3d",
        "landmark_policy": "fixed_window_geometric_median",
        "common_conditioning_policy": "joint_model_schur_at_sparse",
        "baseline_run": str(baseline_run.resolve()),
        "baseline_report_sha256": sha256(baseline_run / "report.json"),
        "baseline_trajectory_sha256": sha256(baseline_run / "trajectory.npz"),
        "source": baseline["source"],
        "calibration": baseline["calibration"],
        "mapping_config": asdict(mapping_config),
        "offset": offset,
        "split": split,
        "track_config": asdict(tracks.config),
        "timing_config": asdict(timing_config),
        "cost_seconds": costs,
        "gate_only_diagnostic": gate,
        "code_sha256": frozen_code,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
        },
        "measurement_policy": "reported frame clocks are acquisition proxies, Sparse RGB poses are proxies, train-only motion, fixed native depth bindings and covariance, references excluded",
        "training_track_count": len(tracks.landmarks),
        "training_supported_tracks": sum(
            t.status == "supported" for t in tracks.landmarks
        ),
        "training_motion_status": dict(
            Counter(motions[i].status for i in split["train_indices"])
        ),
        "all_output_motion_status": dict(Counter(m.status for m in motions.values())),
        "training_depth_bindings": sum(
            int((s.source_index >= 0).sum()) for s in samples.values()
        ),
        "baseline_evaluation": evaluate_trajectory(
            rows,
            (1, 5, 10, 20, 50, 100, 200),
            time_intervals_s=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        ),
        "variants": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".temporal-", dir=output.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        np.savez_compressed(
            stage / "training_measurements.npz", **measurement_arrays(tracks)
        )
        archive = {
            "frame_index": np.array(split["train_indices"]),
            "color_timestamp_s": np.array(
                [
                    np.nan if clocks[i][0] is None else clocks[i][0]
                    for i in split["train_indices"]
                ]
            ),
            "depth_timestamp_s": np.array(
                [
                    np.nan if clocks[i][1] is None else clocks[i][1]
                    for i in split["train_indices"]
                ]
            ),
        }
        archive["motion_status"] = np.array(
            [motions[i].status for i in split["train_indices"]]
        )
        archive["translation_sigma_m"] = np.array(
            [motions[i].translation_sigma_m for i in split["train_indices"]]
        )
        archive["rotation_sigma_rad"] = np.array(
            [motions[i].rotation_sigma_rad for i in split["train_indices"]]
        )
        archive["depth_usable"] = np.array(
            [motions[i].depth_usable for i in split["train_indices"]]
        )
        for i, sample in samples.items():
            archive.update(
                {
                    f"frame_{i}_{name}": getattr(sample, name)
                    for name in sample.__dataclass_fields__
                }
            )
        np.savez_compressed(stage / "native_measurements.npz", **archive)
        for name in ("clock", "static", "timed", "open"):
            if name == "clock":
                poses, diagnostic, factors = (
                    initial,
                    {"windows": [], "optimization_seconds": 0.0},
                    {},
                )
            else:
                poses, diagnostic, factors = refine_fixed(
                    tracks,
                    samples,
                    calibration,
                    config,
                    static=name == "static",
                    lock_overlap=name != "open",
                )
            refined = export_color_poses(
                rows, poses, calibration, motions, static=name == "static"
            )
            evaluation = evaluate_trajectory(
                refined,
                (1, 5, 10, 20, 50, 100, 200),
                time_intervals_s=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
            )
            variant = copy.deepcopy(baseline)
            variant.update(
                backend=f"fixed_landmark_{name}",
                config={
                    **asdict(config),
                    "timing": asdict(timing_config),
                    "static": name == "static",
                    "lock_overlap": name != "open",
                },
                evaluation=evaluation,
                initial_pose_stage="accepted_sparse_rgb_epoch_proxy",
            )
            variant["environment"]["code_sha256"] = frozen_code
            total = (
                baseline["timing"]["tracking_seconds"]
                + costs["total_preparation_seconds"]
                + diagnostic["optimization_seconds"]
            )
            variant["timing"] = {
                "initialization_seconds": baseline["timing"]["tracking_seconds"],
                "training_preparation_seconds": costs["total_preparation_seconds"],
                "pose_only_seconds": diagnostic["optimization_seconds"],
                "tracking_seconds": total,
                "frames_per_second": len(rows) / total,
            }
            for metadata, row in zip(variant["frames"], refined, strict=True):
                metadata["reason"] = (
                    metadata.get("reason") if row.status == "lost" else None
                )
                metadata["diagnostics"] = {
                    "fixed_landmark_variant": name,
                    "motion_status": motions[row.frame_index].status,
                }
            np.savez_compressed(stage / f"{name}_factors.npz", **factors)
            save_run(stage / name, refined, variant, baseline["source"]["start"])
            report["variants"][name] = {
                **diagnostic,
                "evaluation": evaluation,
                "report_sha256": sha256(stage / name / "report.json"),
                "trajectory_sha256": sha256(stage / name / "trajectory.npz"),
                "factor_archive_sha256": sha256(stage / f"{name}_factors.npz"),
            }
            if name != "clock":
                repeated, repeated_diagnostic, repeated_factors = refine_fixed(
                    tracks,
                    samples,
                    calibration,
                    config,
                    static=name == "static",
                    lock_overlap=name != "open",
                )
                for i in poses:
                    np.testing.assert_array_equal(poses[i], repeated[i])
                for key, value in factors.items():
                    np.testing.assert_array_equal(value, repeated_factors[key])
                if diagnostic["windows"] != repeated_diagnostic["windows"]:
                    raise AssertionError("fixed landmark diagnostics repeat differs")
            print(
                f"Completed {name}: {sum(w['status'] == 'refined' for w in diagnostic['windows'])} refined windows",
                flush=True,
            )
        report["repeat_exact"] = True
        report["training_measurements_sha256"] = sha256(
            stage / "training_measurements.npz"
        )
        report["native_measurements_sha256"] = sha256(stage / "native_measurements.npz")
        if code_hash() != frozen_code:
            raise ValueError("implementation changed during experiment")
        (stage / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if output.exists() or output.is_symlink():
            raise ValueError("output appeared during export")
        stage.rename(output)
    return report
