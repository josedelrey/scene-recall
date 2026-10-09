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
