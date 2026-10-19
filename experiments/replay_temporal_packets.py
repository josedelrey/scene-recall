"""Replay archived fixed-landmark windows without reopening RGB-D media."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from scene_recall.data import Calibration
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.multiview.fixed_landmarks import DirectFactors, solve_fixed
from scene_recall.multiview.io import measurement_arrays
from scene_recall.multiview.pose_only import PoseOnlyConfig
from scene_recall.multiview.temporal import TimingConfig
from scene_recall.multiview.temporal_runner import (
    export_color_poses,
    motion_table,
    prepare_temporal,
)
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def replay(run, verify_preparation=False):
    tick = perf_counter()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    baseline_run = Path(report["baseline_run"])
    if (
        sha256(baseline_run / "report.json") != report["baseline_report_sha256"]
        or sha256(baseline_run / "trajectory.npz")
        != report["baseline_trajectory_sha256"]
    ):
        raise ValueError("baseline changed")
    baseline, rows = load_trajectory(baseline_run)
    preparation_seconds = None
    if verify_preparation:
        _, _, split, _, tracks, samples, motions, clocks, costs = prepare_temporal(
            baseline_run,
            MappingConfig(**report["mapping_config"]),
            report["offset"],
            TimingConfig(**report["timing_config"]),
        )
        if split != report["split"]:
            raise ValueError("replayed source split differs")
        with np.load(run / "training_measurements.npz", allow_pickle=False) as expected:
            actual = measurement_arrays(tracks)
            if set(expected.files) != set(actual):
                raise ValueError("training schema differs")
            for key, value in actual.items():
                np.testing.assert_array_equal(value, expected[key], err_msg=key)
        with np.load(run / "native_measurements.npz", allow_pickle=False) as expected:
            ids = report["split"]["train_indices"]
            actual = {
                "frame_index": np.array(ids),
                "color_timestamp_s": np.array(
                    [np.nan if clocks[i][0] is None else clocks[i][0] for i in ids]
                ),
                "depth_timestamp_s": np.array(
                    [np.nan if clocks[i][1] is None else clocks[i][1] for i in ids]
                ),
                "motion_status": np.array([motions[i].status for i in ids]),
                "translation_sigma_m": np.array(
                    [motions[i].translation_sigma_m for i in ids]
                ),
                "rotation_sigma_rad": np.array(
                    [motions[i].rotation_sigma_rad for i in ids]
                ),
                "depth_usable": np.array([motions[i].depth_usable for i in ids]),
            }
            for i, sample in samples.items():
                actual.update(
                    {
                        f"frame_{i}_{field}": getattr(sample, field)
                        for field in sample.__dataclass_fields__
                    }
                )
            if set(expected.files) != set(actual):
                raise ValueError("native schema differs")
            for key, value in actual.items():
                np.testing.assert_array_equal(value, expected[key], err_msg=key)
        preparation_seconds = costs["total_preparation_seconds"]
    for filename, key in (
        ("training_measurements.npz", "training_measurements_sha256"),
        ("native_measurements.npz", "native_measurements_sha256"),
    ):
        if sha256(run / filename) != report[key]:
            raise ValueError("measurement archive changed")
    with np.load(run / "training_measurements.npz", allow_pickle=False) as archive:
        base = dict(zip(map(int, archive["frame_index"]), archive["T_SR"], strict=True))
    values = dict(report["calibration"])
    for key in ("K_R", "K_D", "T_RD"):
        values[key] = np.array(values[key])
    for key in ("rgb_shape", "depth_shape"):
        values[key] = tuple(values[key])
    calibration = Calibration(**values)
    motions, _ = motion_table(
        baseline,
        rows,
        report["split"]["train_indices"],
        calibration,
        TimingConfig(**report["timing_config"]),
    )
    variants = {}
    for name, variant in report["variants"].items():
        if (
            sha256(run / f"{name}_factors.npz") != variant["factor_archive_sha256"]
            or sha256(run / name / "report.json") != variant["report_sha256"]
            or sha256(run / name / "trajectory.npz") != variant["trajectory_sha256"]
        ):
            raise ValueError("variant or factor archive changed")
        child = json.loads((run / name / "report.json").read_text(encoding="utf-8"))
        config = PoseOnlyConfig(
            **{
                k: v
                for k, v in child["config"].items()
                if k in PoseOnlyConfig.__dataclass_fields__
            }
        )
        poses = {i: p.copy() for i, p in base.items()}
        replayed = 0
        with np.load(run / f"{name}_factors.npz", allow_pickle=False) as archive:
            for index, window in enumerate(variant["windows"]):
                if window["status"] == "skipped":
                    continue
                key = f"window_{index:03d}"
                factors = DirectFactors(
                    **{
                        field: archive[f"{key}_{field}"]
                        for field in DirectFactors.__dataclass_fields__
                    }
                )
                if factors.fingerprint() != window["factors_sha256"]:
                    raise ValueError("factor fingerprint differs")
                origin, initial = archive[f"{key}_origin"], archive[f"{key}_initial"]
                ids = window["frame_indices"]
                np.testing.assert_array_equal(
                    initial, np.array([np.linalg.inv(origin) @ poses[i] for i in ids])
                )
                free = [ids.index(i) for i in window["free_frame_indices"]]
                fitted, diagnostic = solve_fixed(
                    initial, factors, calibration.K_R, free, config
                )
                for field, value in diagnostic.items():
                    if window[field] != value:
                        raise AssertionError(f"solver repeat differs: {field}")
                if diagnostic["status"] == "refined":
                    for j in free:
                        poses[ids[j]] = origin @ fitted[j]
                    correction = poses[ids[-1]] @ np.linalg.inv(base[ids[-1]])
                    for i, base_pose in base.items():
                        if rows[i].segment_id == window["segment_id"] and i > ids[-1]:
                            poses[i] = correction @ base_pose
                replayed += 1
        exported = export_color_poses(
            rows, poses, calibration, motions, static=name == "static"
        )
        with np.load(run / name / "trajectory.npz", allow_pickle=False) as expected:
            np.testing.assert_array_equal(
                expected["T_SC_D"], np.array([r.T_SC_D for r in exported])
            )
        variants[name] = {
            "replayed_windows": replayed,
            "full_rate_trajectory_exact": True,
        }
    return {
        "run": str(run.resolve()),
        "variants": variants,
        "seconds": perf_counter() - tick,
        "raw_preparation_arrays_exact": verify_preparation,
        "raw_preparation_seconds": preparation_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-preparation", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("replay report already exists")
    report = {
        "code_sha256": code_hash(Path(__file__)),
        "cases": [replay(run, args.verify_preparation) for run in args.runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
