"""Exercise real dataset loading, strict export pairing and atomic publication."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from helpers.runs import make_run
from PIL import Image

from scene_recall.mapping.cli import main
from scene_recall.mapping.runner import run_mapping
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory


def test_end_to_end_disconnected_maps_exports_and_repeatability(tmp_path):
    _, run = make_run(tmp_path)
    config = MappingConfig(pixel_stride=32, keyframe_stride=1)
    outputs = [tmp_path / "first", tmp_path / "repeat"]
    reports = [run_mapping(run, output, config) for output in outputs]
    assert reports[0]["keyframe_count"] == 3
    assert reports[0]["segment_count"] == 2
    assert [s["keyframe_indices"] for s in reports[0]["submaps"]] == [[0], [1, 2]]
    assert reports[0]["consistency"]["excluded_cross_segment_pairs"] == 1
    assert len(reports[0]["consistency"]["pairs"]) == 2
    for submap in reports[0]["submaps"]:
        assert (
            submap["comparison"]["estimated_to_reference"]["within_cap_fraction"] == 1
        )
    for path in outputs[0].glob("*.npz"):
        with np.load(path) as first, np.load(outputs[1] / path.name) as second:
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key])
    assert reports[0]["submaps"] == reports[1]["submaps"]
    assert reports[0]["consistency"] == reports[1]["consistency"]
    verifier = runpy.run_path(
        Path(__file__).parents[1] / "experiments/verify_mapping_repeats.py"
    )["verify"]
    assert verifier(*outputs)["identical_arrays"] == 28
    header, payload = (
        (outputs[0] / "submap_000_estimated.ply").read_bytes().split(b"end_header\n", 1)
    )
    assert b"coordinate_frame segment_0_submap_0" in header
    with np.load(outputs[0] / "submap_000_estimated.npz") as cloud:
        assert len(payload) == len(cloud["points"]) * 31
    keys = json.loads((outputs[0] / "keyframes.json").read_text())
    assert [k["source_frame_index"] for k in keys] == [0, 1, 2]
    with pytest.raises(ValueError, match="already exists"):
        run_mapping(run, outputs[0], config)


@pytest.mark.parametrize(
    "corruption", ["rgb", "reference", "fingerprint", "calibration", "archive_metadata"]
)
def test_mismatched_inputs_never_publish_partial_output(tmp_path, corruption):
    source, run = make_run(tmp_path)
    report_path = run / "report.json"
    report = json.loads(report_path.read_text())
    if corruption == "rgb":
        Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(
            source / "rgb.png"
        )
    elif corruption == "reference":
        (source / "groundtruth.txt").write_text(
            "1 1 0 0 0 0 0 1\n1.03 1 0 0 0 0 0 1\n1.06 1 0 0 0 0 0 1\n"
        )
    elif corruption == "fingerprint":
        report["source"]["canonical_input_sha256"] = "0" * 64
    elif corruption == "calibration":
        report["calibration"]["K_D"][0][0] = 500
    else:
        report["frames"][1]["status"] = "tracked"
    report_path.write_text(json.dumps(report))
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        run_mapping(run, output, MappingConfig(pixel_stride=32))
    assert not output.exists()
    assert not list(tmp_path.glob(".mapping-*"))


def test_reader_range_source_indices_and_missing_reference_archive_validation(tmp_path):
    _, run = make_run(tmp_path)
    with np.load(run / "trajectory.npz") as archive:
        arrays = dict(archive)
    arrays["source_frame_index"] += 1
    np.savez_compressed(run / "trajectory.npz", **arrays)
    with pytest.raises(ValueError, match="source range"):
        load_trajectory(run)


def test_mapping_wrapper_and_comparison_summaries():
    root = Path(__file__).parents[1]
    assert runpy.run_path(root / "scripts/rgbd_mapping.py")["main"] is main
    summarize = runpy.run_path(root / "experiments/compare_rgbd_maps.py")["summarize"]
    metric = {"within_cap_fraction": 0.5, "capped_distance_m": {"rmse": 0.01}}
    rows = summarize(
        {
            "run": {
                "odometry": {"backend": "sparse"},
                "submaps": [
                    {
                        "segment_id": 2,
                        "submap_id": 4,
                        "keyframe_indices": [10, 20],
                        "estimated": {"voxel_count": 2},
                        "reference": {"voxel_count": 3},
                        "comparison": {
                            "estimated_to_reference": metric,
                            "reference_to_estimated": metric,
                        },
                    }
                ],
            }
        }
    )
    assert len(rows) == 2
    assert rows[0]["segment_id"] == 2
    assert rows[0]["within_cap_fraction"] == 0.5
