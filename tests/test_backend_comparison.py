"""Matched-population checks for the serial benchmark summarizer."""

import json
import runpy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from scene_recall.odometry import HybridConfig, ICPConfig, SparseConfig
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.pipeline import PairEstimate, TrajectoryFrame

ROOT = Path(__file__).parents[1]
save_run = runpy.run_path(ROOT / "scripts" / "scannet_odometry.py")["save_run"]
summarize = runpy.run_path(ROOT / "experiments" / "compare_odometry_backends.py")[
    "summarize_range"
]


def make_runs(tmp_path):
    runs = {}
    for name in ("sparse", "icp", "hybrid"):
        runs[name] = tmp_path / name
        rows = []
        for index in range(4):
            lost = name == "icp" and index == 2
            segment = int(name == "icp" and index >= 2)
            pose = np.eye(4)
            pose[0, 3] = index - (2 if segment else 0)
            reference = np.eye(4)
            reference[0, 3] = index
            relative = np.eye(4)
            relative[0, 3] = -1
            rows.append(
                TrajectoryFrame(
                    index,
                    None,
                    segment,
                    "initialized" if index == 0 else "lost" if lost else "tracked",
                    pose,
                    reference,
                    None if index == 0 else PairEstimate(None if lost else relative),
                )
            )
        configs = {
            "sparse": SparseConfig(),
            "icp": ICPConfig(),
            "hybrid": HybridConfig(),
        }
        report = {
            "backend": name,
            "config": asdict(configs[name]),
            "calibration": {},
            "source": {"start": 0, "stop": 4},
            "timing": {},
            "environment": dict.fromkeys(
                (
                    "python",
                    "numpy",
                    "opencv",
                    "opencv_threads",
                    "openblas_num_threads_env",
                    "omp_num_threads_env",
                    "code_sha256",
                ),
                "same",
            ),
            "evaluation": evaluate_trajectory(rows),
            "frames": [
                {
                    "source_frame_index": row.frame_index,
                    "status": row.status,
                    "reason": "failure" if row.status == "lost" else None,
                    "diagnostics": {},
                }
                for row in rows
            ],
        }
        save_run(runs[name], rows, report, 0)
    return runs


def test_comparison_excludes_cross_segment_pairs_from_every_backend(tmp_path):
    result = summarize(make_runs(tmp_path))
    for backend in ("sparse", "icp", "hybrid"):
        assert result["common_rpe"]["1"][backend]["translation_m"]["count"] == 2
        assert result["common_rpe"]["1"][backend]["translation_m"]["rmse"] == 0
        assert result["common_rpe"]["5"][backend]["translation_m"]["count"] == 0
    assert result["runs"]["icp"]["failures_by_reason"] == {"failure": 1}


@pytest.mark.parametrize("field", ["source", "calibration", "environment", "config"])
def test_comparison_rejects_uncontrolled_inputs(tmp_path, field):
    runs = make_runs(tmp_path)
    path = runs["hybrid"] / "report.json"
    report = json.loads(path.read_text())
    if field == "environment":
        report[field]["code_sha256"] = "different"
    elif field == "config":
        report[field]["icp"]["max_iterations"] = 99
    else:
        report[field]["different"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        summarize(runs)


def test_comparison_checks_timestamp_identity(tmp_path):
    runs = make_runs(tmp_path)
    path = runs["hybrid"] / "trajectory.npz"
    with np.load(path) as archive:
        arrays = dict(archive)
    arrays["timestamp_s"] = np.arange(4, dtype=float)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="timestamp"):
        summarize(runs)


def test_comparison_time_rpe_uses_common_continuous_population(tmp_path):
    from scene_recall.odometry.pipeline import TrajectoryFrame

    runs = make_runs(tmp_path)
    for run in runs.values():
        path = run / "trajectory.npz"
        with np.load(path) as archive:
            arrays = dict(archive)
        arrays["timestamp_s"] = np.array([0.0, 0.1, 0.2, 0.3])
        np.savez_compressed(path, **arrays)
        rows = [
            TrajectoryFrame(
                index,
                float(arrays["timestamp_s"][index]),
                int(arrays["segment_id"][index]),
                str(arrays["status"][index]),
                arrays["T_SC_D"][index],
                arrays["T_WC_D_reference"][index],
            )
            for index in range(4)
        ]
        report_path = run / "report.json"
        report = json.loads(report_path.read_text())
        report["evaluation"] = evaluate_trajectory(
            rows, time_intervals_s=[0.1], time_tolerance_s=0.02
        )
        report_path.write_text(json.dumps(report))
    comparison = summarize(runs)["common_time_rpe"]["0.1"]
    assert comparison["candidate_pairs"] == 3
    assert comparison["actual_interval_s"]["count"] == 2
    for values in comparison["backends"].values():
        assert values["translation_m"]["count"] == 2
        assert values["translation_m"]["rmse"] == 0


@pytest.mark.parametrize("mixed", [False, True])
def test_comparison_cli_dispatches_full_tum_and_mixed_sources(
    tmp_path, monkeypatch, mixed
):
    import sys

    namespace = runpy.run_path(ROOT / "experiments/compare_odometry_backends.py")
    globals_ = namespace["main"].__globals__
    calls = []
    monkeypatch.setattr(
        globals_["subprocess"],
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    monkeypatch.setitem(
        globals_, "summarize_range", lambda runs: {"run_names": list(runs)}
    )
    argv = [
        "compare_odometry_backends.py",
        "--tum-path",
        str(tmp_path / "tum_xyz"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    if mixed:
        argv += ["--sens-path", str(tmp_path / "scannet.sens"), "--ranges", "0:20"]
    monkeypatch.setattr(sys, "argv", argv)
    namespace["main"]()
    assert len(calls) == (6 if mixed else 3)
    for command, kwargs in calls:
        assert "rgbd_odometry.py" in command[1]
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert ("--stop" in command) == mixed
    assert any("--tum-path" in command for command, _ in calls)
    assert any("--sens-path" in command for command, _ in calls) == mixed
    assert (tmp_path / "output/comparison.json").exists()
