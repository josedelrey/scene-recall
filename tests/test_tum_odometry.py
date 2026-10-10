"""Exercise all real backends through the TUM adapter and shared exports."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scene_recall.odometry.cli import main


@pytest.mark.parametrize("backend", ["sparse", "icp", "hybrid"])
def test_tum_cli_is_reproducible_and_reference_independent(
    tmp_path, monkeypatch, backend
):
    source = tmp_path / "tum"
    source.mkdir()
    # Repeat a textured plane. Sparse motion is observable, while depth-only
    # ICP must expose the planar degeneracy through its normal failure policy.
    rgb = np.random.default_rng(5).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(source / "rgb.png")
    Image.fromarray(np.full((480, 640), 10000, dtype=np.uint16)).save(
        source / "depth.png"
    )
    (source / "rgb.txt").write_text("1.01 rgb.png\n1.04 rgb.png\n")
    (source / "depth.txt").write_text("1 depth.png\n1.03 depth.png\n")
    outputs = []
    for run in range(2):
        # Reference changes must affect evaluation without changing estimates.
        (source / "groundtruth.txt").write_text(
            f"1 0 0 0 0 0 0 1\n1.03 {run} 0 0 0 0 0 1\n"
        )
        output = tmp_path / f"run{run}"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rgbd_odometry.py",
                "--tum-path",
                str(source),
                "--backend",
                backend,
                "--output-dir",
                str(output),
            ],
        )
        main()
        outputs.append(output)
    with (
        np.load(outputs[0] / "trajectory.npz") as first,
        np.load(outputs[1] / "trajectory.npz") as second,
    ):
        for key in (
            "T_SC_D",
            "T_C1C0",
            "initial_T_C1C0",
            "candidate_T_C1C0",
            "segment_id",
        ):
            np.testing.assert_array_equal(first[key], second[key])
        assert not np.array_equal(first["T_WC_D_reference"], second["T_WC_D_reference"])
        np.testing.assert_array_equal(first["timestamp_s"], [1.0, 1.03])
    report = json.loads((outputs[0] / "report.json").read_text())
    assert report["source"]["dataset"] == "tum_rgbd"
    assert report["source"]["source_frame_count"] == 2
    assert len(report["source"]["canonical_input_sha256"]) == 64
    assert report["evaluation"]["reference_valid_frames"] == 2
    assert report["frames"][0]["provenance"]["rgb_file"] == "rgb.png"
    assert report["evaluation"]["tracked_transitions"] == (0 if backend == "icp" else 1)
    assert (outputs[0] / "frames.csv").exists()
    # Reusing an output must never overwrite a completed experiment.
    with pytest.raises(SystemExit):
        main()


def test_shared_and_legacy_cli_resolve_to_same_runner():
    root = Path(__file__).parents[1]
    assert runpy.run_path(root / "scripts/rgbd_odometry.py")["main"] is main
    assert runpy.run_path(root / "scripts/scannet_odometry.py")["main"] is main
