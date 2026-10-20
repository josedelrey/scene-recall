"""Scientific comparison populations and input checks."""

import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from helpers.runs import make_run

from scene_recall.odometry.local_map import LocalMapConfig
from scene_recall.odometry.pipeline import TrajectoryFrame
from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig

EXPERIMENT = runpy.run_path(
    Path(__file__).parents[1] / "experiments/incremental_local_map.py"
)


def test_common_errors_exclude_missing_reference_and_disconnected_pairs():
    first = []
    for i in range(6):
        pose = np.eye(4)
        pose[0, 3] = 0.1 * i
        first.append(
            TrajectoryFrame(
                i,
                float(i),
                0,
                "initialized" if i == 0 else "tracked",
                pose,
                None if i == 4 else pose,
            )
        )
    second = [replace(r, segment_id=int(i >= 2)) for i, r in enumerate(first)]
    metrics = EXPERIMENT["common_errors"](first, second, 1)
    assert metrics["pairs"] == [[0, 1], [2, 3]]
    assert metrics["baseline"]["translation_m"]["count"] == 2
    assert metrics["candidate"]["rotation_deg"]["rmse"] == 0


def test_local_map_repeat_rejects_missing_starting_history(tmp_path):
    verify = runpy.run_path(
        Path(__file__).parents[1] / "experiments/verify_odometry_repeats.py"
    )["verify_clip"]
    common = {
        "backend": "local_map",
        "config": {},
        "calibration": {},
        "environment": dict.fromkeys(
            (
                "code_sha256",
                "python",
                "numpy",
                "opencv",
                "opencv_threads",
                "openblas_num_threads_env",
                "omp_num_threads_env",
            ),
            "same",
        ),
        "source": dict.fromkeys(
            ("path", "sequence_id", "association_config", "adapter"), "same"
        ),
    }
    for name, start in (("full", 0), ("clip", 5)):
        path = tmp_path / name
        path.mkdir()
        (path / "report.json").write_text(
            json.dumps({**common, "source": {**common["source"], "start": start}})
        )
    with pytest.raises(ValueError, match="starting history"):
        verify(tmp_path / "full", tmp_path / "clip")


@pytest.mark.parametrize(
    "field", ["source", "calibration", "config", "environment", "provenance"]
)
def test_summary_rejects_mismatched_comparison_inputs(tmp_path, field):
    source, _ = make_run(tmp_path)
    paths = {k: tmp_path / k for k in ("sparse", "local")}
    for name, backend, config in (
        ("sparse", "sparse", SparseConfig()),
        ("local", "local_map", LocalMapConfig()),
    ):
        run_odometry(
            source, paths[name], dataset="tum_rgbd", backend_name=backend, config=config
        )
    result = EXPERIMENT["summarize"](paths)
    assert result["variants"]["local"]["tracking_modes"] == {"lost": 2}
    path = paths["local"] / "report.json"
    report = json.loads(path.read_text())
    if field == "provenance":
        report["frames"][0]["provenance"]["injected"] = True
    else:
        report[field]["injected"] = True
        if field == "environment":
            report[field]["code_sha256"] = "changed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="comparison"):
        EXPERIMENT["summarize"](paths)


def test_resummary_reuses_verified_fusion_report_and_refuses_changed_settings(
    tmp_path, monkeypatch
):
    source, _ = make_run(tmp_path)
    output = tmp_path / "comparison"
    arguments = [
        "incremental_local_map.py",
        "--tum-path",
        str(source),
        "--output-dir",
        str(output),
        "--variants",
        "sparse",
        "local",
        "--evaluate-maps",
    ]
    monkeypatch.setattr("sys.argv", arguments)
    EXPERIMENT["main"]()
    path = output / "maps_local/report.json"
    original = path.read_bytes()
    monkeypatch.setattr("sys.argv", arguments + ["--summarize-only"])
    EXPERIMENT["main"]()
    assert path.read_bytes() == original
    report = json.loads(original)
    report["mapping_config"]["pixel_stride"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="fusion evaluation"):
        EXPERIMENT["main"]()
