"""Source-free batch experiments from immutable training-only RGB-D packets."""

import copy
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.batch_ba import BatchConfig, solve_batch
from scene_recall.multiview.diagnostics import grid_coverage
from scene_recall.multiview.fixed_landmarks import DirectFactors, geometric_median
from scene_recall.multiview.joint_runner import calibration_from_report, verify_packets
from scene_recall.multiview.temporal import TimingConfig
from scene_recall.multiview.temporal_runner import export_color_poses, motion_table
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.run_metadata import code_hash, sha256


def make_batch_packets(graph, native, calibration, config=None):
    """Retain all eligible identities and iteratively prune unsupported cameras.

    This does not reassociate, use saved world points or inspect reference arrays.
    The actual segment root always enters the camera system. Disconnected graph
    gauges therefore remain observable as rank failures rather than new anchors.
    """
    config = config or BatchConfig()
    frame_ids = list(map(int, graph["frame_index"]))
    base = dict(zip(frame_ids, graph["T_SR"], strict=True))
    segments = dict(zip(frame_ids, map(int, graph["segment_id"]), strict=True))
    offsets = dict(zip(frame_ids, graph["feature_offsets"][:-1], strict=True))
    candidates = []

    def eligible(keys):
        return (
            len(keys) >= config.min_track_views
            and sum(
                np.isfinite(native[f"frame_{i}_points_D"][k]).all() for i, k in keys
            )
            >= 2
        )

    for track, (lo, hi) in enumerate(
        zip(graph["track_offsets"][:-1], graph["track_offsets"][1:], strict=True)
    ):
        keys = graph["observation_keys"][lo:hi]
        if len(set(map(int, keys[:, 0]))) != len(keys):
            raise ValueError("duplicate camera in batch track")
        if (
            any(int(i) not in base for i in keys[:, 0])
            or len({segments[int(i)] for i in keys[:, 0]}) != 1
        ):
            raise ValueError("batch track violates training segment boundary")
        if eligible(keys):
            candidates.append((track, keys.copy()))
    active = set(frame_ids)
    while True:
        pixels = {i: [] for i in active}
        for _, keys in candidates:
            for i, k in keys:
                pixels[int(i)].append(graph["pixels"][offsets[int(i)] + k])
        selected = {
            i
            for i in active
            if len(pixels[i]) >= config.min_observations
            and grid_coverage(pixels[i], calibration.rgb_shape)
            >= config.min_grid_coverage
        }
        if selected == active:
            break
        active = selected
        candidates = [
            (track, keys[np.array([int(i) in active for i in keys[:, 0]])])
            for track, keys in candidates
        ]
        candidates = [(track, keys) for track, keys in candidates if eligible(keys)]
    packets = []
    for segment in sorted(set(segments.values())):
        segment_ids = [i for i in frame_ids if segments[i] == segment]
        members = [
            (track, keys)
            for track, keys in candidates
            if segments[int(keys[0, 0])] == segment
        ]
        root = segment_ids[0]
        ids = sorted({root} | {int(i) for _, keys in members for i in keys[:, 0]})
        support = {
            "segment_id": segment,
            "frame_indices": ids,
            "training_camera_count": len(segment_ids),
            "unsupported_frame_indices": [i for i in segment_ids if i not in active],
            "candidate_tracks": len(members),
            "root_has_observations": root in active,
        }
        if not members or len(ids) < 2 or len(ids) > config.max_cameras:
            packets.append(
                (
                    ids,
                    None,
                    None,
                    {
                        **support,
                        "status": "skipped",
                        "reason": "insufficient support or camera budget",
                    },
                )
            )
            continue
        positions = {i: j for j, i in enumerate(ids)}
        initial = np.array([np.linalg.inv(base[root]) @ base[i] for i in ids])
        values = {field: [] for field in DirectFactors.__dataclass_fields__}
        for track, keys in members:
            measured = [
                transform_points(
                    native[f"frame_{i}_points_R"][k], initial[positions[int(i)]]
                )
                for i, k in keys
                if np.isfinite(native[f"frame_{i}_points_D"][k]).all()
            ]
            point = geometric_median(measured)
            for i, k in keys:
                for field, value in (
                    ("view", positions[int(i)]),
                    ("track_id", track),
                    ("points", point),
                    ("pixels", graph["pixels"][offsets[int(i)] + k]),
                    ("native_points", native[f"frame_{i}_points_D"][k]),
                    ("T_RcD", native[f"frame_{i}_T_RcD"]),
                    ("whitening_D", native[f"frame_{i}_whitening_D"][k]),
                    ("weights", 1 / len(keys)),
                ):
                    values[field].append(value)
        factors = DirectFactors(
            **{field: np.array(value) for field, value in values.items()}
        )
        packets.append((ids, initial, factors, support))
    return base, segments, packets


def complete_training_poses(base, segments, fitted):
    """Interpolate corrections across unsupported training cameras and carry tails."""
    poses = {i: p.copy() for i, p in base.items()}
    for segment in sorted(set(segments.values())):
        keys = sorted(i for i in fitted if segments[i] == segment)
        if not keys:
            continue
        for i in base:
            if segments[i] != segment:
                continue
            if i in fitted:
                poses[i] = fitted[i].copy()
                continue
            j = int(np.searchsorted(keys, i))
            lo, hi = keys[max(0, j - 1)], keys[min(j, len(keys) - 1)]
            first, last = (
                fitted[lo] @ np.linalg.inv(base[lo]),
                fitted[hi] @ np.linalg.inv(base[hi]),
            )
            alpha = 0 if lo == hi else (i - lo) / (hi - lo)
            correction = np.eye(4)
            vector = cv2.Rodrigues(first[:3, :3].T @ last[:3, :3])[0].ravel()
            correction[:3, :3] = first[:3, :3] @ cv2.Rodrigues(alpha * vector)[0]
            correction[:3, 3] = (1 - alpha) * first[:3, 3] + alpha * last[:3, 3]
            poses[i] = correction @ base[i]
    return poses


def fit_batch_packets(base, segments, packets, K, config):
    """Solve each independent segment without anchoring additional components."""
    fitted, reports, arrays = {}, [], {}
    for index, (ids, initial, factors, support) in enumerate(packets):
        if factors is None:
            reports.append(support)
            continue
        poses, points, diagnostic = solve_batch(initial, factors, K, config)
        reports.append(
            {**support, **diagnostic, "factors_sha256": factors.fingerprint()}
        )
        if diagnostic["status"] == "refined":
            fitted.update({i: base[ids[0]] @ poses[j] for j, i in enumerate(ids)})
        key = f"segment_{index:03d}"
        arrays.update(
            {
                f"{key}_{field}": getattr(factors, field)
                for field in DirectFactors.__dataclass_fields__
            }
        )
        arrays.update(
            {
                f"{key}_initial": initial,
                f"{key}_fitted_poses": poses,
                f"{key}_fitted_points": points,
            }
        )
    return complete_training_poses(base, segments, fitted), reports, arrays


def run_batch_experiment(run, output, *, config=None):
    """Publish a batch fit alongside unchanged clock and fixed-landmark controls."""
    run, output = Path(run), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("batch output already exists")
    config = config or BatchConfig()
    frozen_code = code_hash()
    report, protected = verify_packets(run)
    baseline, rows = load_trajectory(Path(report["baseline_run"]))
    calibration = calibration_from_report(report)
    with np.load(run / "training_measurements.npz", allow_pickle=False) as a:
        graph = {k: a[k] for k in a.files}
    with np.load(run / "native_measurements.npz", allow_pickle=False) as a:
        native = {k: a[k] for k in a.files}
    if not (
        np.isfinite(native["color_timestamp_s"])
        & np.isfinite(native["depth_timestamp_s"])
    ).all():
        raise ValueError("batch experiment requires known reported acquisition clocks")
    if set(map(int, graph["frame_index"])) != set(
        report["split"]["train_indices"]
    ) or not set(map(int, graph["observation_keys"][:, 0])) <= set(
        report["split"]["train_indices"]
    ):
        raise ValueError("batch measurements violate training population")
    started = perf_counter()
    base, segments, packets = make_batch_packets(graph, native, calibration, config)
    preparation_seconds = perf_counter() - started
    started = perf_counter()
    poses, diagnostics, arrays = fit_batch_packets(
        base, segments, packets, calibration.K_R, config
    )
    solve_seconds = perf_counter() - started
    repeat, repeated, repeated_arrays = fit_batch_packets(
        base, segments, packets, calibration.K_R, config
    )
    for i in poses:
        np.testing.assert_array_equal(poses[i], repeat[i])
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], repeated_arrays[key])
    if diagnostics != repeated:
        raise AssertionError("batch diagnostics repeat differs")
    motions, _ = motion_table(
        baseline,
        rows,
        report["split"]["train_indices"],
        calibration,
        TimingConfig(**report["timing_config"]),
    )
    refined = export_color_poses(rows, poses, calibration, motions)
    evaluation = evaluate_trajectory(
        refined,
        (1, 5, 10, 20, 50, 100, 200),
        time_intervals_s=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    child = copy.deepcopy(baseline)
    child.update(
        backend="experimental_batch_rgbd_ba",
        evaluation=evaluation,
        config=asdict(config),
    )
    child["environment"]["code_sha256"] = frozen_code
    child["timing"]["batch_solver_seconds"] = solve_seconds
    result = copy.deepcopy(report)
    result.update(
        experiment="batch_rgbd_ba",
        parent_experiment=str(run.resolve()),
        parent_report_sha256=sha256(run / "report.json"),
        config=asdict(config),
        code_sha256=frozen_code,
        batch_factor_preparation_seconds=preparation_seconds,
        batch_solver_seconds=solve_seconds,
        repeat_exact=True,
        landmark_policy="one shared point per training identity throughout segment",
        boundary_policy="original segment root only, abstain on rank failure",
    )
    result["variants"] = {name: report["variants"][name] for name in ("clock", "timed")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".batch-ba-", dir=output.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        for name in (
            "training_measurements.npz",
            "native_measurements.npz",
            "clock_factors.npz",
            "timed_factors.npz",
        ):
            shutil.copyfile(run / name, stage / name)
        for name in ("clock", "timed"):
            shutil.copytree(run / name, stage / name)
        np.savez_compressed(stage / "batch_factors.npz", **arrays)
        save_run(stage / "batch", refined, child, baseline["source"]["start"])
        result["variants"]["batch"] = {
            "segments": diagnostics,
            "evaluation": evaluation,
            "report_sha256": sha256(stage / "batch/report.json"),
            "trajectory_sha256": sha256(stage / "batch/trajectory.npz"),
            "factor_archive_sha256": sha256(stage / "batch_factors.npz"),
        }
        for path, digest in protected.items():
            if sha256(path) != digest:
                raise ValueError("input changed during batch experiment")
        if code_hash() != frozen_code:
            raise ValueError("implementation changed during batch experiment")
        (stage / "report.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if output.exists() or output.is_symlink():
            raise ValueError("batch output appeared during export")
        stage.rename(output)
    return result


def replay_batch_experiment(output):
    """Independently rebuild factors, solve and complete every saved pose."""
    output = Path(output)
    saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
    parent = Path(saved["parent_experiment"])
    if sha256(parent / "report.json") != saved["parent_report_sha256"]:
        raise ValueError("batch parent report changed")
    report, _ = verify_packets(parent)
    for name in ("clock", "timed", "batch"):
        identity = saved["variants"][name]
        for relative, field in (
            (f"{name}/trajectory.npz", "trajectory_sha256"),
            (f"{name}/report.json", "report_sha256"),
            (f"{name}_factors.npz", "factor_archive_sha256"),
        ):
            if sha256(output / relative) != identity[field]:
                raise ValueError("published batch input changed")
    for file, field in (
        ("training_measurements.npz", "training_measurements_sha256"),
        ("native_measurements.npz", "native_measurements_sha256"),
    ):
        if sha256(output / file) != saved[field]:
            raise ValueError("published batch measurements changed")
    calibration = calibration_from_report(report)
    baseline, rows = load_trajectory(Path(report["baseline_run"]))
    config = BatchConfig(**saved["config"])
    with np.load(output / "training_measurements.npz", allow_pickle=False) as a:
        graph = {key: a[key] for key in a.files}
    with np.load(output / "native_measurements.npz", allow_pickle=False) as a:
        native = {key: a[key] for key in a.files}
    base, segments, packets = make_batch_packets(graph, native, calibration, config)
    poses, diagnostics, arrays = fit_batch_packets(
        base, segments, packets, calibration.K_R, config
    )
    if diagnostics != saved["variants"]["batch"]["segments"]:
        raise AssertionError("batch diagnostic replay differs")
    with np.load(output / "batch_factors.npz", allow_pickle=False) as a:
        if set(a.files) != set(arrays):
            raise AssertionError("batch factor archive fields differ")
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, a[key])
    motions, _ = motion_table(
        baseline,
        rows,
        report["split"]["train_indices"],
        calibration,
        TimingConfig(**report["timing_config"]),
    )
    repeated = export_color_poses(rows, poses, calibration, motions)
    _, original = load_trajectory(output / "batch")
    np.testing.assert_array_equal(
        [r.T_SC_D for r in repeated], [r.T_SC_D for r in original]
    )
    return {
        "full_rate_trajectory_exact": True,
        "factors_and_points_exact": True,
        "diagnostics_exact": True,
        "segments": len(packets),
        "verified_arrays": len(arrays),
    }
