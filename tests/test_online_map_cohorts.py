"""Reference availability must be explicit and independent of estimated success."""

import importlib
import json
from pathlib import Path

from helpers.runs import make_run

from scene_recall.datasets.tum import TUMConfig
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.slam.online import OnlineConfig
from scene_recall.slam.runner import run_online


def test_missing_reference_requires_explicit_available_cohort(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "experiments")
    )
    evaluate = importlib.import_module("evaluate_online_maps").evaluate
    source, _ = make_run(tmp_path)
    for stream in ("rgb", "depth"):
        (source / f"{stream}.txt").write_text(
            "".join(f"{1 + i * 0.1:.2f} {stream}.png\n" for i in range(25))
        )
    (source / "groundtruth.txt").write_text(
        "".join(f"{1 + i * 0.1:.2f} 0 0 0 0 0 0 1\n" for i in range(25) if i != 10)
    )
    runs = tmp_path / "runs"
    run_online(
        source,
        runs / "control",
        dataset="tum_rgbd",
        config=OnlineConfig(recovery=False, loops=False),
        association=TUMConfig(max_ground_truth_difference_s=0.01),
        mapping=MappingConfig(pixel_stride=8),
    )
    strict = evaluate(runs, tmp_path / "strict", ["control"])
    assert strict["status"] == "withheld"
    available = evaluate(
        runs, tmp_path / "available", ["control"], reference_cohort="available"
    )
    assert available["status"] == "evaluated"
    protocol = json.loads((tmp_path / "available" / "protocol.json").read_text())
    assert protocol["train"] == [0, 20]
    assert protocol["missing_fixed_training_reference_indices"] == [10]
    assert available["consistency"]["control"]["exact_chunks"] == 3
