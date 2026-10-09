import csv
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from scene_recall.data import Calibration, Observation

script = runpy.run_path(Path(__file__).parents[1] / "scripts" / "scannet_odometry.py")


def test_cli_exports_reproducible_poses_and_evaluation(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.sens"
    source.write_bytes(b"synthetic reader input")
    rgb = np.random.default_rng(0).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    depth = np.full((192, 256), 2, dtype=np.float32)
    K = np.array([[300, 0, 127], [0, 300, 95], [0, 0, 1]], dtype=float)
    calibration = Calibration(K, K, np.eye(4), (192, 256), (192, 256))
    malformed = np.eye(4)
    malformed[0, 0] = -1
    observations = [
        Observation(index, rgb, depth, T_WC_D=malformed if index == 2 else np.eye(4))
        for index in range(4)
    ]

    class Reader:
        sequence_id = "synthetic"
        frame_count = 4

        def __init__(self, path, *, start, stop):
            assert (start, stop) == (0, 4)
            self.calibration = calibration

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter(observations)

    monkeypatch.setitem(script["main"].__globals__, "ScanNetReader", Reader)
    outputs = []
    for name in ("first", "repeat"):
        output = tmp_path / name
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "scannet_odometry.py",
                "--sens-path",
                str(source),
                "--stop",
                "4",
                "--output-dir",
                str(output),
            ],
        )
        script["main"]()
        outputs.append(output)
    with (
        np.load(outputs[0] / "trajectory.npz", allow_pickle=False) as first,
        np.load(outputs[1] / "trajectory.npz", allow_pickle=False) as second,
    ):
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first["T_WC_D_reference"][2], malformed)
        assert first["T_SC_D"].shape == (4, 4, 4)
        assert np.isnan(first["T_C1C0"][0]).all()
        assert np.isnan(first["timestamp_s"]).all()
    first_report = json.loads((outputs[0] / "report.json").read_text())
    second_report = json.loads((outputs[1] / "report.json").read_text())
    assert first_report["evaluation"] == second_report["evaluation"]
    assert first_report["evaluation"]["reference_invalid_frames"] == 1
    assert first_report["evaluation"]["tracked_transitions"] == 3
    assert first_report["config"]["iterations"] == 2000
    assert first_report["timestamp_available_frames"] == 0
    assert len(first_report["environment"]["code_sha256"]) == 64
    with (outputs[0] / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[1]["status"] == "tracked"
    assert rows[1]["refinement"] == "refined"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scannet_odometry.py",
            "--sens-path",
            str(source),
            "--output-dir",
            str(outputs[0]),
        ],
    )
    with pytest.raises(SystemExit) as error:
        script["main"]()
    assert error.value.code == 2


def test_empty_run_archive_has_consistent_shapes(tmp_path):
    script["save_run"](tmp_path, [], {}, 10)
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as archive:
        assert archive["T_SC_D"].shape == (0, 4, 4)
        assert archive["frame_index"].shape == (0,)
        assert archive["reference_present"].dtype == bool


def test_nonfinite_report_is_rejected_before_writing(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        script["save_run"](output, [], {"invalid": np.inf}, 0)
    assert not output.exists()


def test_cli_failed_transitions_export_segments_without_fabricated_motion(
    tmp_path, monkeypatch
):
    source = tmp_path / "blank.sens"
    source.write_bytes(b"blank input")
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.ones((32, 32), dtype=np.float32)
    calibration = Calibration(np.eye(3), np.eye(3), np.eye(4), (32, 32), (32, 32))

    class Reader:
        sequence_id = "blank"
        frame_count = 3

        def __init__(self, path, *, start, stop):
            self.calibration = calibration

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter(Observation(index, rgb, depth) for index in range(3))

    output = tmp_path / "run"
    monkeypatch.setitem(script["main"].__globals__, "ScanNetReader", Reader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scannet_odometry.py",
            "--sens-path",
            str(source),
            "--output-dir",
            str(output),
        ],
    )
    script["main"]()
    report = json.loads((output / "report.json").read_text())
    assert report["evaluation"]["failed_transitions"] == 2
    assert report["evaluation"]["segment_count"] == 3
    assert report["evaluation"]["tracking_coverage"] == 0
    assert report["evaluation"]["continuous_trajectory"] is None
    assert (
        report["evaluation"]["rpe_by_frame_interval"]["1"]["excluded_cross_segment"]
        == 2
    )
    with np.load(output / "trajectory.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["segment_id"], [0, 1, 2])
        np.testing.assert_array_equal(
            archive["status"], ["initialized", "lost", "lost"]
        )
        assert np.isnan(archive["T_C1C0"]).all()
        np.testing.assert_array_equal(
            archive["T_SC_D"], np.repeat(np.eye(4)[None], 3, axis=0)
        )
