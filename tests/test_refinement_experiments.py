"""Prepared references and fixed-population reconstruction controls."""

import json
import runpy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from helpers.runs import continuous_run, make_run

from scene_recall.datasets.tum import TUMReader
from scene_recall.mapping.heldout import run_heldout
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.odometry.local_map import LocalMapConfig
from scene_recall.odometry.recent_map import RecentMapConfig

ROOT = Path(__file__).parents[1]


def test_prepare_omits_both_ambiguous_poses_without_touching_original(tmp_path):
    source, _ = make_run(tmp_path)
    table = source / "groundtruth.txt"
    text = table.read_text()
    table.write_text(
        text.replace(
            "1.03 0 0 0 0 0 0 1\n", "1.03 0 0 0 0 0 0 1\n1.03 .1 0 0 0 0 0 1\n"
        )
    )
    before = table.read_bytes()
    prepare = runpy.run_path(ROOT / "experiments/prepare_tum_references.py")["prepare"]
    output = tmp_path / "prepared"
    report = prepare(source, output)
    assert table.read_bytes() == before
    assert report["removed_rows"] == 2
    assert report["removed_timestamps_s"] == [1.03]
    assert "1.03" not in (output / "groundtruth.txt").read_text()
    assert (source / "depth.png").read_bytes() == (output / "depth.png").read_bytes()
    with TUMReader(output) as reader:
        assert len(list(reader)) == 3


def test_tsdf_targets_do_not_enter_fusion_and_identical_poses_match(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    evaluate = runpy.run_path(ROOT / "experiments/refinement_tsdf.py")["evaluate_tsdf"]
    _, baseline = continuous_run(tmp_path, heldout_depth=2.3)
    report, rows = load_trajectory(baseline)
    paths = {"budget": baseline}
    for name, backend, config in (
        ("local", "local_map", LocalMapConfig()),
        ("recent", "local_map_recent", RecentMapConfig()),
    ):
        paths[name] = tmp_path / name
        save_run(
            paths[name],
            rows,
            {**report, "backend": backend, "config": asdict(config)},
            0,
        )
    group = list(paths)
    fusion = run_heldout(
        list(paths.values()),
        tmp_path / "maps",
        MappingConfig(pixel_stride=32, keyframe_stride=2),
        offset=1,
    )
    result = evaluate(paths, group, tmp_path / "tsdf", fusion)
    assert result["status"] == "evaluated"
    assert result["training_indices"] == [0, 2]
    assert result["target_indices"] == [1]
    with np.load(tmp_path / "tsdf/geometry_and_predictions.npz") as archive:
        observed = archive["observed"]
        np.testing.assert_allclose(observed[np.isfinite(observed)], 2.3)
        first = archive["budget_reference_pose"]
        assert np.isfinite(first).any()
        np.testing.assert_allclose(first[np.isfinite(first)], 2.0, atol=0.03)
        np.testing.assert_array_equal(first, archive["recent_reference_pose"])
        np.testing.assert_array_equal(first, archive["local_reference_pose"])
    assert evaluate(paths, group, tmp_path / "tsdf", fusion) == result
    saved = tmp_path / "tsdf/report.json"
    modified = json.loads(saved.read_text())
    modified["inputs"]["recent"]["report_sha256"] = "changed"
    saved.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="inputs"):
        evaluate(paths, group, tmp_path / "tsdf", fusion)


def test_refinement_resume_requires_exact_inputs_and_settings(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    run = runpy.run_path(ROOT / "experiments/refine_local_map.py")["run_case"]
    source, _ = make_run(tmp_path)
    output = tmp_path / "comparison"
    options = {
        "dataset": "tum_rgbd",
        "names": ("sparse", "budget", "local", "recent"),
        "evaluate_maps": True,
    }
    run(source, output, **options)
    repeated = run(source, output, resume=True, **options)
    assert repeated["fusion"]["status"] == "evaluated"
    summarize = runpy.run_path(ROOT / "experiments/summarize_map_refinement.py")[
        "summarize_case"
    ]
    summary = summarize(output)
    assert summary["variants"]["recent"]["lost_frames"] == [1, 2]
    assert summary["paired"]["budget"]["1"]["pairs"] == []
    path = output / "recent/report.json"
    report = json.loads(path.read_text())
    report["config"]["anchor_policy"] = "frozen"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="resume"):
        run(source, output, resume=True, **options)
    with pytest.raises(ValueError, match="changed saved input"):
        summarize(output)


def test_capped_map_score_keeps_missing_predictions_in_target_population(tmp_path):
    score = runpy.run_path(ROOT / "experiments/summarize_map_refinement.py")[
        "capped_scores"
    ]
    observed = np.array([[[1.0, 2.0, 3.0, np.nan]]])
    prediction = np.array([[[1.03, np.nan, 9.0, 4.0]]])
    np.savez(
        tmp_path / "views_000_radius_0.01.npz",
        observed=observed,
        local_map_recent_reference_pose=prediction,
        evaluated_frame_indices=[5],
    )
    result = score(tmp_path)["local_map_recent_reference_pose"]
    assert result["pooled_m"] == pytest.approx((0.03 + 0.2 + 0.2) / 3)
    assert result["per_view"] == [
        {"frame_index": 5, "capped_error_m": result["pooled_m"]}
    ]
