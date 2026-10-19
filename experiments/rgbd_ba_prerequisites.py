"""Audit frozen pose-only experiments before enabling joint RGB-D BA.

No joint solver is implemented. References are confined to labeled diagnostics.
"""

import argparse
import hashlib
import json
import platform
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.data import Calibration
from scene_recall.datasets.replay import open_source
from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.multiview.audit import epoch_displacement, landmark_schur_information
from scene_recall.multiview.io import measurement_arrays
from scene_recall.multiview.model import (
    FeatureFrame,
    Landmark,
    TrackConfig,
    TrackResult,
)
from scene_recall.multiview.pose_only import (
    PoseOnlyConfig,
    linearize,
    make_factors,
    refine_training_poses,
    retract,
    robust_cost,
    solve_window,
)
from scene_recall.multiview.pose_runner import prepare_training_tracks
from scene_recall.odometry.evaluation import (
    error_statistics,
    evaluate_trajectory,
    pose_error,
)
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def restore_tracks(arrays, settings):
    """Rehydrate reference-free fitting inputs from the existing numeric archive."""
    settings = {**settings, "pair_offsets": tuple(settings["pair_offsets"])}
    frames, tracks = [], []
    for i, index in enumerate(arrays["frame_index"]):
        lo, hi = arrays["feature_offsets"][i : i + 2]
        stamp = arrays["timestamp_s"][i]
        frames.append(
            FeatureFrame(
                int(index),
                int(arrays["segment_id"][i]),
                None if np.isnan(stamp) else float(stamp),
                arrays["T_SR"][i],
                RGBDFeatures(arrays["pixels"][lo:hi], None, arrays["points_R"][lo:hi]),
            )
        )
    for i, track_id in enumerate(arrays["track_id"]):
        lo, hi = arrays["track_offsets"][i : i + 2]
        tracks.append(
            Landmark(
                int(track_id),
                int(arrays["track_segment_id"][i]),
                tuple(tuple(map(int, k)) for k in arrays["observation_keys"][lo:hi]),
                arrays["point_S"][i],
                str(arrays["track_status"][i]),
                int(arrays["depth_views"][i]),
                arrays["reprojection_errors_px"][lo:hi],
                arrays["depth_errors_m"][lo:hi],
                float(arrays["parallax_deg"][i]),
                float(arrays["baseline_m"][i]),
            )
        )
    return TrackResult(
        tuple(frames), tuple(tracks), (), (), TrackConfig(**settings), {}
    )


def factor_hash(factors):
    digest = hashlib.sha256()
    for array in (
        factors.anchor,
        factors.target,
        factors.points_anchor,
        factors.pixels,
        factors.depth,
        factors.whitening,
        factors.weights,
        factors.track_ids,
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()


def window_observations(tracks, ids, selected, origin):
    """Use only window measurements to initialize proposed point information."""
    frames = {f.frame_index: f for f in tracks.frames}
    positions = {index: i for i, index in enumerate(ids)}
    observations = []
    for track in tracks.landmarks:
        if track.track_id not in selected:
            continue
        keys = [k for k in track.observations if k[0] in positions]
        points = [
            transform_points(frames[f].features.points_R[k], frames[f].T_SR)
            for f, k in keys
            if np.isfinite(frames[f].features.points_R[k]).all()
        ]
        median = transform_points(np.median(points, axis=0), np.linalg.inv(origin))
        observations.append(
            (
                median,
                np.array([positions[f] for f, _ in keys]),
                np.array([frames[f].features.pixels[k] for f, k in keys]),
                np.array([frames[f].features.points_R[k, 2] for f, k in keys]),
            )
        )
    return observations


def audit_windows(tracks, calibration, config, windows, refined_rows):
    """Check factor identity, local conditioning and later overlap damage."""
    frames = {f.frame_index: f for f in tracks.frames}
    reports = []
    fit_seconds = 0.0
    for saved in windows:
        ids = saved["frame_indices"]
        factors, _ = make_factors(
            tracks, ids, calibration.K_R, calibration.rgb_shape, config
        )
        if factors is None:
            if saved["status"] != "skipped":
                raise ValueError("saved window support differs")
            continue
        if factor_hash(factors) != saved["factors_sha256"]:
            raise ValueError("saved factor fingerprint differs")
        origin = frames[ids[0]].T_SR
        base = np.array([np.linalg.inv(origin) @ frames[i].T_SR for i in ids])
        final = np.array(
            [
                np.linalg.inv(origin)
                @ refined_rows[i].T_SC_D
                @ np.linalg.inv(calibration.T_RD)
                for i in ids
            ]
        )
        local_factors = replace(
            factors,
            frozen_points=transform_points(
                factors.frozen_points, np.linalg.inv(origin)
            ),
        )
        started = perf_counter()
        isolated, local = solve_window(base, local_factors, calibration.K_R, config)
        fit_seconds += perf_counter() - started
        final_r, _ = linearize(
            final, local_factors, calibration.K_R, config, jacobian=False
        )
        final_cost, _ = robust_cost(final_r, local_factors, config)
        observations = window_observations(tracks, ids, set(factors.track_ids), origin)
        information = {}
        for name, scale in (("rgbd", 1.0), ("depth_sigma_x4", 4.0), ("rgb_only", None)):
            info = landmark_schur_information(
                base, observations, calibration.K_R, config, depth_scale=scale
            )
            info["point_condition"] = error_statistics(info.pop("point_conditions"))
            information[name] = info
        reports.append(
            {
                "frame_indices": ids,
                "factors_sha256": saved["factors_sha256"],
                "status": saved["status"],
                "isolated_status": local["status"],
                "sparse_cost": local.get("initial_cost"),
                "isolated_cost": local.get("final_cost", local.get("initial_cost")),
                "cost_when_window_closed": saved.get(
                    "final_cost", saved.get("initial_cost")
                ),
                "cost_after_all_windows": final_cost
                if np.isfinite(final_cost)
                else None,
                "information_at_sparse": information,
                "star_factor_count": len(factors.target),
                "direct_observation_block_count": sum(len(o[1]) for o in observations),
                # External reference errors are appended by the caller after fitting.
                "isolated_endpoint_relative": isolated[-1].tolist(),
                "sparse_endpoint_relative": base[-1].tolist(),
                "final_endpoint_relative": (
                    np.linalg.inv(final[0]) @ final[-1]
                ).tolist(),
            }
        )
    return reports, fit_seconds


def transfer_residuals(poses, factors, K):
    """Report raw transfer errors on fixed selected factors, without re-gating."""
    world = np.einsum(
        "nij,nj->ni", poses[factors.anchor, :3, :3], factors.points_anchor
    )
    world += poses[factors.anchor, :3, 3]
    q = np.einsum(
        "nij,nj->ni",
        poses[factors.target, :3, :3].transpose(0, 2, 1),
        world - poses[factors.target, :3, 3],
    )
    valid = (q[:, 2] > 0.05) & np.isfinite(q).all(axis=1)
    depth_valid = valid & np.isfinite(factors.depth)
    pixel_error = np.linalg.norm(
        project_points(q[valid], K) - factors.pixels[valid], axis=1
    )
    depth_error = q[depth_valid, 2] - factors.depth[depth_valid]
    cholesky = np.linalg.inv(factors.whitening[depth_valid])
    depth_zscore = depth_error / np.linalg.norm(cholesky[:, 2, :], axis=1)
    return pixel_error, depth_error, depth_zscore, int((~valid).sum())


def reference_transfers(tracks, calibration, windows, config, rows, color_refs):
    """Inspect selected measurements at external poses, never estimate geometry."""
    populations = {
        epoch: {
            "pixels": [],
            "depth": [],
            "zscore": [],
            "excluded": 0,
            "view_means": [],
        }
        for epoch in ("depth", "color")
    }
    for window in windows:
        ids = window["frame_indices"]
        # Keep both epoch diagnostics on the same reference-valid window population.
        if color_refs and any(
            rows[i].reference_pose is None or color_refs.get(i) is None for i in ids
        ):
            continue
        factors, _ = make_factors(
            tracks, ids, calibration.K_R, calibration.rgb_shape, config
        )
        for epoch, population in populations.items():
            refs = [
                rows[i].reference_pose if epoch == "depth" else color_refs.get(i)
                for i in ids
            ]
            if any(p is None for p in refs):
                continue
            poses = np.array([p @ np.linalg.inv(calibration.T_RD) for p in refs])
            pixel, depth, zscore, excluded = transfer_residuals(
                poses, factors, calibration.K_R
            )
            population["pixels"].extend(pixel)
            population["depth"].extend(depth)
            population["zscore"].extend(zscore)
            population["excluded"] += excluded
            # Median signed block error exposes coherent residuals without claiming calibration.
            for target in np.unique(factors.target):
                select = np.flatnonzero(
                    (factors.target == target) & np.isfinite(factors.depth)
                )
                if len(select):
                    subset = replace(
                        factors,
                        **{
                            name: getattr(factors, name)[select]
                            for name in factors.__dataclass_fields__
                        },
                    )
                    _, errors, _, _ = transfer_residuals(poses, subset, calibration.K_R)
                    if len(errors):
                        population["view_means"].append(float(np.median(errors)))
    return {
        epoch: {
            "pixel_error": error_statistics(p["pixels"]),
            "signed_depth_error_m": error_statistics(p["depth"]),
            "signed_depth_zscore": error_statistics(p["zscore"]),
            "fraction_absolute_depth_zscore_over_three": float(
                np.mean(np.abs(p["zscore"]) > 3)
            )
            if p["zscore"]
            else None,
            "absolute_view_median_depth_error_m": error_statistics(
                np.abs(p["view_means"])
            ),
            "excluded_nonpositive_projections": p["excluded"],
        }
        for epoch, p in populations.items()
    }


def synchronization_probe(skew_s=0.012):
    """A static slanted plane with exact identities and asynchronous ray depths.

    Truth generates sensor measurements and the post-fit score. The pose-only
    API receives only single-epoch initial poses, pixels and sampled depths.
    Depth comes from intersecting the depth-time ray at the color pixel with
    the plane. Merely attaching the color timestamp cannot synchronize it.
    """
    K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]])
    calibration = Calibration(K, K.copy(), np.eye(4), (480, 640), (480, 640))
    rng = np.random.default_rng(81)
    xy = rng.uniform([-0.8, -0.6], [0.8, 0.6], (96, 2))
    points = np.column_stack((xy, 2.5 + xy @ [0.2, -0.15]))
    normal = np.array([-0.2, 0.15, 1.0])

    def motion(t):
        return retract(
            np.eye(4),
            np.array(
                [
                    0.15 * t + 0.07 * t * t,
                    0.01 * t,
                    0,
                    0,
                    0.25 * t + 0.1 * t * t,
                    0.04 * t * t,
                ]
            ),
        )

    color_poses = [motion(0.1 * i + (skew_s if i % 2 else -skew_s)) for i in range(6)]
    origin = color_poses[0]
    frames = []
    for i, color_pose in enumerate(color_poses):
        q = transform_points(points, np.linalg.inv(color_pose))
        pixels = project_points(q, K)
        rays = q / q[:, 2:3]
        depth_pose = motion(0.1 * i)
        rotated = rays @ depth_pose[:3, :3].T
        depth = (2.5 - normal @ depth_pose[:3, 3]) / (rotated @ normal)
        frames.append(
            FeatureFrame(
                i,
                0,
                i * 0.1,
                np.linalg.inv(origin) @ color_pose,
                RGBDFeatures(pixels, None, rays * depth[:, None]),
            )
        )
    landmarks = tuple(
        Landmark(
            j,
            0,
            tuple((i, j) for i in range(6)),
            np.median(
                [transform_points(f.features.points_R[j], f.T_SR) for f in frames],
                axis=0,
            ),
            "supported",
            6,
            np.zeros(6),
            np.zeros(6),
            0.0,
            0.0,
        )
        for j in range(len(points))
    )
    tracks = TrackResult(tuple(frames), landmarks, (), (), TrackConfig(), {})
    poses, diagnostic = refine_training_poses(tracks, calibration)
    errors = [pose_error(poses[f.frame_index], f.T_SR) for f in frames]
    return {
        "skew_s": skew_s,
        "initial_pose_error_is_zero": True,
        "translation_error_m": error_statistics([e[0] for e in errors]),
        "rotation_error_deg": error_statistics([e[1] for e in errors]),
        "window": diagnostic["windows"][0],
    }


def timing_audit(baseline, rows, tracks, calibration, variants):
    """Evaluate exposure sensitivity without changing official reference epochs."""
    if baseline["source"]["dataset"] != "tum_rgbd":
        return {"status": "unavailable_exposure_clocks"}, {}
    references, skews, pixels, dz, per_view = {}, [], [], [], []
    reader, _ = open_source(baseline)
    with reader:
        for row, metadata in zip(rows, baseline["frames"]):
            stamp = metadata["provenance"]["timestamp_color"]
            references[row.frame_index] = reader._reference(stamp)[0]
        for frame in tracks.frames:
            metadata = baseline["frames"][frame.frame_index]["provenance"]
            difference = metadata["timestamp_color"] - metadata["timestamp_depth"]
            skews.append(difference)
            depth_pose = rows[frame.frame_index].reference_pose
            color_pose = references[frame.frame_index]
            if depth_pose is None or color_pose is None:
                continue
            # TUM GT is the RGB optical pose at each timestamp. T_RD is identity.
            shift, zshift, excluded = epoch_displacement(
                frame.features.points_R, calibration.K_R, depth_pose, color_pose
            )
            pixels.extend(shift)
            dz.extend(zshift)
            per_view.append(
                {
                    "frame_index": frame.frame_index,
                    "absolute_skew_s": abs(difference),
                    "pixel_shift": error_statistics(shift),
                    "excluded": excluded,
                }
            )
    evaluations = {}
    # Both epoch evaluations use the exact same reference-valid source IDs.
    common = [
        r.reference_pose is not None and references[r.frame_index] is not None
        for r in rows
    ]
    for name, trajectory in {"sparse": rows, **variants}.items():
        evaluations[name] = {}
        for epoch in ("depth", "color"):
            evaluated = [
                replace(
                    r,
                    reference_pose=(
                        rows[r.frame_index].reference_pose
                        if epoch == "depth"
                        else references[r.frame_index]
                    )
                    if common[r.frame_index]
                    else None,
                )
                for r in trajectory
            ]
            evaluations[name][epoch] = evaluate_trajectory(evaluated, (1, 10, 50, 200))
    pixel_array = np.array(pixels)
    return {
        "status": "diagnostic_only",
        "training_view_count": len(tracks.frames),
        "absolute_skew_s": error_statistics(np.abs(skews)),
        "clock_only_views_within_ms": {
            str(ms): int(np.sum(np.abs(skews) <= ms / 1000)) for ms in (1, 3, 5, 10)
        },
        "predicted_static_pixel_shift": error_statistics(pixel_array),
        "fraction_shift_over_one_pixel": float(np.mean(pixel_array > 1))
        if len(pixels)
        else None,
        "predicted_static_depth_shift_m": error_statistics(dz),
        "views": per_view,
        "same_population_epoch_evaluations": evaluations,
    }, references


def audit(run, *, verify_source=False):
    started = perf_counter()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    baseline_path = Path(report["baseline_run"])
    paths = [
        baseline_path / "report.json",
        baseline_path / "trajectory.npz",
        run / "report.json",
        run / "training_measurements.npz",
    ]
    for name in report["variants"]:
        paths.extend((run / name / "report.json", run / name / "trajectory.npz"))
    if (run / "maps").exists():
        paths.extend(sorted((run / "maps").glob("*.json")))
        paths.extend(sorted((run / "maps").glob("*.npz")))
    protected = {str(path.resolve()): sha256(path) for path in paths}
    if (
        sha256(baseline_path / "report.json") != report["baseline_report_sha256"]
        or sha256(baseline_path / "trajectory.npz")
        != report["baseline_trajectory_sha256"]
    ):
        raise ValueError("baseline identity differs")
    baseline, rows = load_trajectory(baseline_path)
    variants = {}
    for name, details in report["variants"].items():
        if (
            sha256(run / name / "report.json") != details["report_sha256"]
            or sha256(run / name / "trajectory.npz") != details["trajectory_sha256"]
        ):
            raise ValueError("refined trajectory identity differs")
        _, variants[name] = load_trajectory(run / name)
    with np.load(run / "training_measurements.npz", allow_pickle=False) as data:
        arrays = dict(data)
    if arrays["frame_index"].tolist() != report["split"]["train_indices"]:
        raise ValueError("training population differs")
    tracks = restore_tracks(arrays, report["track_config"])
    values = dict(report["calibration"])
    for key in ("K_R", "K_D", "T_RD"):
        values[key] = np.array(values[key])
    for key in ("rgb_shape", "depth_shape"):
        values[key] = tuple(values[key])
    calibration = Calibration(**values)
    preparation = perf_counter() - started
    replay_seconds = None
    if verify_source:
        _, _, split, _, rebuilt, replay_seconds = prepare_training_tracks(
            baseline_path,
            MappingConfig(**report["mapping_config"]),
            report["offset"],
            tracks.config,
        )
        if split != report["split"]:
            raise ValueError("replayed split differs")
        rebuilt_arrays = measurement_arrays(rebuilt)
        if set(arrays) != set(rebuilt_arrays):
            raise ValueError("replayed measurement schema differs")
        for key, array in arrays.items():
            np.testing.assert_array_equal(array, rebuilt_arrays[key], err_msg=key)
    started = perf_counter()
    config = PoseOnlyConfig(**report["variants"]["anchored"]["config"])
    windows, fit_seconds = audit_windows(
        tracks,
        calibration,
        config,
        report["variants"]["anchored"]["windows"],
        variants["anchored"],
    )
    timing, color_refs = timing_audit(baseline, rows, tracks, calibration, variants)
    transfers = reference_transfers(
        tracks, calibration, windows, config, rows, color_refs
    )
    for window in windows:
        first, last = window["frame_indices"][0], window["frame_indices"][-1]
        window["external_endpoint_errors"] = {}
        for epoch in ("depth", "color"):
            refs = [
                rows[i].reference_pose if epoch == "depth" else color_refs.get(i)
                for i in (first, last)
            ]
            if any(r is None for r in refs):
                continue
            # Convert depth-camera estimates to RGB optical coordinates.
            truth = (
                calibration.T_RD
                @ np.linalg.inv(refs[0])
                @ refs[1]
                @ np.linalg.inv(calibration.T_RD)
            )
            window["external_endpoint_errors"][epoch] = {
                name: dict(
                    zip(
                        ("translation_m", "rotation_deg"),
                        pose_error(np.array(window[key]), truth),
                    )
                )
                for name, key in (
                    ("sparse", "sparse_endpoint_relative"),
                    ("isolated", "isolated_endpoint_relative"),
                    ("stitched", "final_endpoint_relative"),
                )
            }
    if any(sha256(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError("scientific input changed during audit")
    return {
        "run": str(run.resolve()),
        "input_report_sha256": sha256(run / "report.json"),
        "training_archive_sha256": sha256(run / "training_measurements.npz"),
        "source_replay_all_arrays_exact": verify_source,
        "training_views": len(tracks.frames),
        "initialization_diagnostics": initialization_summary(tracks, arrays, config),
        "timing": timing,
        "windows": windows,
        "reference_transfer_diagnostics": transfers,
        "protected_input_sha256": protected,
        "all_protected_inputs_unchanged": True,
        "cost_seconds": {
            "cached_preparation": preparation,
            "raw_replay_preparation": replay_seconds,
            "isolated_pose_only_solves": fit_seconds,
            "diagnostics_excluding_replay": perf_counter() - started - fit_seconds,
        },
        "policy": "training measurements only, no joint fitting, GT diagnostic only, official evaluations unchanged",
    }


def initialization_summary(tracks, arrays, config):
    supported = [t for t in tracks.landmarks if t.status == "supported"]
    points = arrays["points_R"]
    depths = points[np.isfinite(points).all(axis=1), 2]
    return {
        "supported_tracks": len(supported),
        "supported_parallax_deg": error_statistics(t.parallax_deg for t in supported),
        "supported_baseline_m": error_statistics(t.baseline_m for t in supported),
        "supported_track_length": error_statistics(
            len(t.observations) for t in supported
        ),
        "retained_sample_depth_sigma_m": error_statistics(
            config.depth_sigma_constant_m + config.depth_sigma_quadratic * depths**2
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--repeat-cached", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("audit output already exists")
    cv2.setNumThreads(1)
    frozen_code = code_hash(Path(__file__))
    reports = []
    for run in args.runs:
        report = audit(run, verify_source=args.verify_source)
        if args.repeat_cached:
            repeated = audit(run)
            scientific = {
                k: v
                for k, v in report.items()
                if k not in ("cost_seconds", "source_replay_all_arrays_exact")
            }
            repeated_scientific = {
                k: v
                for k, v in repeated.items()
                if k not in ("cost_seconds", "source_replay_all_arrays_exact")
            }
            if scientific != repeated_scientific:
                raise ValueError("cached scientific repeat differs")
        report["repeat_cached_exact"] = args.repeat_cached
        reports.append(report)
        print(f"Audited {run.name}", flush=True)
    result = {
        "format_version": 1,
        "code_sha256": frozen_code,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "cases": reports,
        "synthetic_controls": {
            "synchronized": synchronization_probe(0.0),
            "asynchronous_12ms": synchronization_probe(0.012),
        },
    }
    if code_hash(Path(__file__)) != frozen_code:
        raise ValueError("audit implementation changed during execution")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ba-audit-", dir=args.output_dir.parent
    ) as temp:
        staging = Path(temp) / "result"
        staging.mkdir()
        (staging / "report.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(args.output_dir)


if __name__ == "__main__":
    main()
