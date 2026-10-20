"""Opt-in packet-only joint BA experiment and unchanged fixed-landmark controls."""

import copy
import json
import platform
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from scene_recall.data import Calibration
from scene_recall.multiview.fixed_landmarks import (
    DirectFactors,
    direct_cost,
    direct_information,
    direct_linearize,
    solve_fixed,
)
from scene_recall.multiview.joint_ba import landmark_layout, solve_joint
from scene_recall.multiview.pose_only import PoseOnlyConfig
from scene_recall.multiview.temporal import TimingConfig
from scene_recall.multiview.temporal_runner import export_color_poses, motion_table
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.run_metadata import code_hash, sha256


def calibration_from_report(report):
    values = dict(report["calibration"])
    for key in ("K_R", "K_D", "T_RD"):
        values[key] = np.array(values[key])
    for key in ("rgb_shape", "depth_shape"):
        values[key] = tuple(values[key])
    return Calibration(**values)


def verify_packets(run):
    """Verify all inputs before any output is published, without reopening media."""
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report["experiment"] != "fixed_landmark_two_epoch_rgbd":
        raise ValueError("joint BA requires temporal fixed-landmark packets")
    files = {
        run / "training_measurements.npz": report["training_measurements_sha256"],
        run / "native_measurements.npz": report["native_measurements_sha256"],
        Path(report["baseline_run"]) / "report.json": report["baseline_report_sha256"],
        Path(report["baseline_run"]) / "trajectory.npz": report[
            "baseline_trajectory_sha256"
        ],
    }
    for name in ("timed", "clock"):
        variant = report["variants"][name]
        files.update(
            {
                run / name / "report.json": variant["report_sha256"],
                run / name / "trajectory.npz": variant["trajectory_sha256"],
                run / f"{name}_factors.npz": variant["factor_archive_sha256"],
            }
        )
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError(f"packet input changed: {path.name}")
    return report, files


def fit_packets(
    base, segments, windows, archive, K, config, *, max_point_displacement_m=0.10
):
    """Sequential joint fitting plus isolated fits from exactly saved fixed starts.

    Reuse the Sparse common gate, point initialization and committed-camera
    policy. Only sequential camera starts depend on earlier joint updates.
    """
    poses = {i: p.copy() for i, p in base.items()}
    reports, fitted_archive, objectives = [], {}, []
    joint_seconds, fixed_seconds, paired_joint_seconds = 0.0, 0.0, 0.0
    for index, window in enumerate(windows):
        report = copy.deepcopy(window)
        key = f"window_{index:03d}"
        if window["status"] == "skipped":
            reports.append(report)
            continue
        ids = window["frame_indices"]
        if any(i not in base or segments[i] != window["segment_id"] for i in ids):
            raise ValueError("window violates training segment boundary")
        factors = DirectFactors(
            **{
                field: archive[f"{key}_{field}"]
                for field in DirectFactors.__dataclass_fields__
            }
        )
        if factors.fingerprint() != window["factors_sha256"]:
            raise ValueError("factor fingerprint differs")
        free = [ids.index(i) for i in window["free_frame_indices"]]
        sparse = np.array([np.linalg.inv(base[ids[0]]) @ base[i] for i in ids])
        information = direct_information(sparse, factors, K, free, config)
        if (
            information["condition"] is None
            or information["condition"] > config.max_condition
        ):
            raise ValueError("saved supported window fails common Sparse gate")
        saved_initial = archive[f"{key}_initial"]
        started = perf_counter()
        _, fixed_diagnostic = solve_fixed(saved_initial, factors, K, free, config)
        fixed_seconds += perf_counter() - started
        for field, value in fixed_diagnostic.items():
            if window[field] != value:
                raise ValueError(f"fixed packet replay differs: {field}")
        started = perf_counter()
        _, _, paired = solve_joint(
            saved_initial,
            factors,
            K,
            free,
            config,
            max_point_displacement_m=max_point_displacement_m,
        )
        paired_joint_seconds += perf_counter() - started
        origin = poses[ids[0]].copy()
        initial = np.array([np.linalg.inv(origin) @ poses[i] for i in ids])
        started = perf_counter()
        fitted, points, diagnostic = solve_joint(
            initial,
            factors,
            K,
            free,
            config,
            max_point_displacement_m=max_point_displacement_m,
        )
        joint_seconds += perf_counter() - started
        if diagnostic["status"] == "refined":
            for j in free:
                poses[ids[j]] = origin @ fitted[j]
            correction = poses[ids[-1]] @ np.linalg.inv(base[ids[-1]])
            for i in base:
                if segments[i] == window["segment_id"] and i > ids[-1]:
                    poses[i] = correction @ base[i]
        _, inverse, _ = landmark_layout(factors)
        current_factors = replace(factors, points=points[inverse])
        current = np.array([np.linalg.inv(origin) @ poses[i] for i in ids])
        residual, _ = direct_linearize(current, current_factors, K, (), jacobian=False)
        at_close, _ = direct_cost(residual, factors, config)
        # Drop stale solver fields inherited from the fixed run.
        support = {
            k: v
            for k, v in report.items()
            if k
            not in (
                "accepted_costs",
                "condition",
                "initial_cost",
                "final_cost",
                "iterations",
                "cost_at_close",
                "cost_after_all_windows",
                "status",
                "reason",
            )
        }
        reports.append(
            {
                **support,
                **diagnostic,
                "cost_at_close": at_close,
                "paired_fixed": fixed_diagnostic,
                "paired_joint": paired,
                "sequential_start_difference_frobenius": float(
                    np.linalg.norm(initial - saved_initial)
                ),
            }
        )
        objectives.append((len(reports) - 1, ids, origin, current_factors, at_close))
        for field in DirectFactors.__dataclass_fields__:
            fitted_archive[f"{key}_{field}"] = getattr(factors, field)
        fitted_archive.update(
            {
                f"{key}_initial": initial,
                f"{key}_origin": origin,
                f"{key}_fitted_points": points,
                f"{key}_fitted_poses": fitted,
            }
        )
    for index, ids, origin, factors, closed in objectives:
        current = np.array([np.linalg.inv(origin) @ poses[i] for i in ids])
        residual, _ = direct_linearize(current, factors, K, (), jacobian=False)
        final, _ = direct_cost(residual, factors, config)
        reports[index]["cost_after_all_windows"] = final
        if not np.isclose(closed, final, rtol=1e-10, atol=1e-9):
            raise AssertionError("joint committed window changed")
    return (
        poses,
        {
            "windows": reports,
            "linear_solver_seconds": joint_seconds,
            "paired_fixed_seconds": fixed_seconds,
            "paired_joint_seconds": paired_joint_seconds,
        },
        fitted_archive,
    )


def run_joint_experiment(
    run, output, *, allow_unknown_static=False, max_point_displacement_m=0.10
):
    """Publish a fresh, reproducible packet comparison with no input mutation."""
    run, output = Path(run), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("joint output already exists")
    frozen_code = code_hash()
    report, protected = verify_packets(run)
    baseline, rows = load_trajectory(Path(report["baseline_run"]))
    calibration = calibration_from_report(report)
    child, _ = load_trajectory(run / "timed")
    config = PoseOnlyConfig(
        **{
            k: v
            for k, v in child["config"].items()
            if k in PoseOnlyConfig.__dataclass_fields__
        }
    )
    with np.load(run / "training_measurements.npz", allow_pickle=False) as a:
        base = dict(zip(map(int, a["frame_index"]), a["T_SR"], strict=True))
    if set(base) != set(report["split"]["train_indices"]):
        raise ValueError("training population differs")
    with np.load(run / "native_measurements.npz", allow_pickle=False) as a:
        known = np.isfinite(a["color_timestamp_s"]) & np.isfinite(
            a["depth_timestamp_s"]
        )
    if not known.all() and not (
        allow_unknown_static and report["timing_config"]["unknown_static"]
    ):
        raise ValueError(
            "unknown acquisition clocks require explicit assumed-static packets and opt-in"
        )
    motions, _ = motion_table(
        baseline,
        rows,
        report["split"]["train_indices"],
        calibration,
        TimingConfig(**report["timing_config"]),
    )
    segments = {i: rows[i].segment_id for i in base}
    with np.load(run / "timed_factors.npz", allow_pickle=False) as a:
        tick = perf_counter()
        poses, diagnostic, factors = fit_packets(
            base,
            segments,
            report["variants"]["timed"]["windows"],
            a,
            calibration.K_R,
            config,
            max_point_displacement_m=max_point_displacement_m,
        )
        elapsed = perf_counter() - tick
        repeated, repeated_diagnostic, repeated_factors = fit_packets(
            base,
            segments,
            report["variants"]["timed"]["windows"],
            a,
            calibration.K_R,
            config,
            max_point_displacement_m=max_point_displacement_m,
        )
    for i in poses:
        np.testing.assert_array_equal(poses[i], repeated[i])
    for key in factors:
        np.testing.assert_array_equal(factors[key], repeated_factors[key])
    if diagnostic["windows"] != repeated_diagnostic["windows"]:
        raise AssertionError("joint diagnostics repeat differs")
    refined = export_color_poses(rows, poses, calibration, motions)
    repeated_rows = export_color_poses(rows, repeated, calibration, motions)
    np.testing.assert_array_equal(
        [r.T_SC_D for r in refined], [r.T_SC_D for r in repeated_rows]
    )
    evaluation = evaluate_trajectory(
        refined,
        (1, 5, 10, 20, 50, 100, 200),
        time_intervals_s=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    variant = copy.deepcopy(child)
    variant.update(backend="experimental_local_joint_rgbd_ba", evaluation=evaluation)
    variant["config"]["max_point_displacement_m"] = max_point_displacement_m
    variant["environment"]["code_sha256"] = frozen_code
    variant["timing"] = {
        "tracking_seconds": baseline["timing"]["tracking_seconds"]
        + report["cost_seconds"]["total_preparation_seconds"]
        + diagnostic["linear_solver_seconds"],
        "joint_solver_seconds": diagnostic["linear_solver_seconds"],
    }
    for meta in variant["frames"]:
        meta["diagnostics"] = {"joint_ba": True}
    result = copy.deepcopy(report)
    result.update(
        experiment="local_joint_rgbd_ba",
        landmark_policy="free_window_local_shared_points",
        parent_experiment=str(run.resolve()),
        parent_report_sha256=sha256(run / "report.json"),
        code_sha256=frozen_code,
        environment={"python": platform.python_version(), "numpy": np.__version__},
        acquisition_policy="known_reported_clocks"
        if known.all()
        else "explicit_unknown_static_assumption",
        packet_comparison_seconds=elapsed,
        repeat_exact=True,
        point_persistence="shared within window only, reinitialized from original packets at each window",
    )
    result["variants"] = {name: result["variants"][name] for name in ("clock", "timed")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".joint-ba-", dir=output.parent) as temp:
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
        np.savez_compressed(stage / "joint_factors.npz", **factors)
        save_run(stage / "joint", refined, variant, baseline["source"]["start"])
        result["variants"]["joint"] = {
            **diagnostic,
            "evaluation": evaluation,
            "report_sha256": sha256(stage / "joint/report.json"),
            "trajectory_sha256": sha256(stage / "joint/trajectory.npz"),
            "factor_archive_sha256": sha256(stage / "joint_factors.npz"),
        }
        for path, digest in protected.items():
            if sha256(path) != digest:
                raise ValueError("input changed during joint experiment")
        if code_hash() != frozen_code:
            raise ValueError("implementation changed during joint experiment")
        (stage / "report.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if output.exists() or output.is_symlink():
            raise ValueError("joint output appeared during export")
        stage.rename(output)
    return result


def replay_joint_experiment(output):
    """Reconstruct published joint arrays from parent packets in a fresh solve."""
    output = Path(output)
    saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
    parent = Path(saved["parent_experiment"])
    if sha256(parent / "report.json") != saved["parent_report_sha256"]:
        raise ValueError("parent experiment changed")
    report, _ = verify_packets(parent)
    for name in ("clock", "timed", "joint"):
        identity = saved["variants"][name]
        for relative, field in (
            (f"{name}/trajectory.npz", "trajectory_sha256"),
            (f"{name}/report.json", "report_sha256"),
            (f"{name}_factors.npz", "factor_archive_sha256"),
        ):
            if sha256(output / relative) != identity[field]:
                raise ValueError("published joint input changed")
    for filename, field in (
        ("training_measurements.npz", "training_measurements_sha256"),
        ("native_measurements.npz", "native_measurements_sha256"),
    ):
        if sha256(output / filename) != saved[field]:
            raise ValueError("published measurement archive changed")
    baseline, rows = load_trajectory(Path(report["baseline_run"]))
    child, _ = load_trajectory(output / "joint")
    config = PoseOnlyConfig(
        **{
            k: v
            for k, v in child["config"].items()
            if k in PoseOnlyConfig.__dataclass_fields__
        }
    )
    calibration = calibration_from_report(report)
    with np.load(parent / "training_measurements.npz", allow_pickle=False) as a:
        base = dict(zip(map(int, a["frame_index"]), a["T_SR"], strict=True))
    with np.load(parent / "timed_factors.npz", allow_pickle=False) as a:
        poses, diagnostic, factors = fit_packets(
            base,
            {i: rows[i].segment_id for i in base},
            report["variants"]["timed"]["windows"],
            a,
            calibration.K_R,
            config,
            max_point_displacement_m=child["config"]["max_point_displacement_m"],
        )
    if diagnostic["windows"] != saved["variants"]["joint"]["windows"]:
        raise AssertionError("joint window replay differs")
    with np.load(output / "joint_factors.npz", allow_pickle=False) as a:
        if set(a.files) != set(factors):
            raise AssertionError("joint archive schema differs")
        for key in factors:
            np.testing.assert_array_equal(factors[key], a[key], err_msg=key)
    motions, _ = motion_table(
        baseline,
        rows,
        report["split"]["train_indices"],
        calibration,
        TimingConfig(**report["timing_config"]),
    )
    refined = export_color_poses(rows, poses, calibration, motions)
    with np.load(output / "joint/trajectory.npz", allow_pickle=False) as a:
        np.testing.assert_array_equal(a["T_SC_D"], [r.T_SC_D for r in refined])
    return {
        "run": str(output.resolve()),
        "window_diagnostics_exact": True,
        "landmarks_and_poses_exact": True,
        "full_rate_trajectory_exact": True,
    }
