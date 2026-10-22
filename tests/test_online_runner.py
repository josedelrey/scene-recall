"""Raw-input provenance and complete publication of online scientific runs."""

import json

import numpy as np
import pytest
from helpers.runs import make_run

from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory
from scene_recall.slam.runner import run_online


def test_raw_runner_preserves_source_and_separates_controlled_inputs(tmp_path):
    source, baseline = make_run(tmp_path)
    output = tmp_path / "online"
    report = run_online(
        source,
        output,
        dataset="tum_rgbd",
        mapping=MappingConfig(pixel_stride=16, keyframe_stride=1),
        intervention=("depth", 1, 2),
    )
    saved, rows = load_trajectory(output)
    original = json.loads((baseline / "report.json").read_text())
    assert (
        saved["source"]["canonical_input_sha256"]
        == original["source"]["canonical_input_sha256"]
    )
    assert report["experiment"]["causal_online"]
    assert len(rows) == 3
    assert all(row.reference_pose is not None for row in rows)
    with np.load(output / "atlas_emitted.npz") as emitted:
        assert len(emitted["version"]) == 3
    with np.load(output / "reconstruction.npz") as reconstruction:
        frames = [
            int(frame)
            for key in reconstruction
            if key.endswith("_frames")
            for frame in reconstruction[key]
        ]
        assert frames == [0, 2]
    assert (output / "protocol.json").exists()
    assert not output.with_suffix(".protocol.json").exists()
    with pytest.raises(ValueError, match="fresh"):
        run_online(source, output, dataset="tum_rgbd")


def test_export_failure_does_not_publish_a_partial_scientific_run(
    tmp_path, monkeypatch
):
    source, _ = make_run(tmp_path)
    output = tmp_path / "online"

    def fail(*args):
        raise RuntimeError("controlled export failure")

    monkeypatch.setattr("scene_recall.slam.runner._export_online", fail)
    with pytest.raises(RuntimeError, match="controlled export"):
        run_online(source, output, dataset="tum_rgbd")
    assert not output.exists()
    assert output.with_suffix(".protocol.json").exists()
    assert not list(tmp_path.glob(".online-slam-*"))
