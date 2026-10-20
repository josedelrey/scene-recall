"""Odometry runs publish completely or leave their destination untouched."""

import csv
from pathlib import Path

import numpy as np
import pytest

import scene_recall.odometry.io as run_io
from scene_recall.odometry.pipeline import PairEstimate, TrajectoryFrame


@pytest.fixture
def inputs():
    pose = np.eye(4)
    pose[0, 3] = 0.1
    rows = [
        TrajectoryFrame(0, 1.0, 0, "initialized", np.eye(4), np.eye(4)),
        TrajectoryFrame(
            1,
            1.03,
            0,
            "tracked",
            pose,
            None,
            PairEstimate(np.linalg.inv(pose), diagnostics={"inliers": 20}),
        ),
    ]
    report = {
        "format_version": 2,
        "source": {"start": 5, "stop": 7},
        "evaluation": {"frame_count": 2},
        "frames": [
            {
                "frame_index": row.frame_index,
                "source_frame_index": 5 + row.frame_index,
                "segment_id": row.segment_id,
                "status": row.status,
            }
            for row in rows
        ],
    }
    return rows, report


def test_atomic_export_retains_archive_report_and_csv_contracts(tmp_path, inputs):
    rows, report = inputs
    output = tmp_path / "run"
    run_io.save_run(output, rows, report, 5)
    loaded, frames = run_io.load_trajectory(output)
    assert loaded == report
    np.testing.assert_array_equal(frames[1].T_SC_D, rows[1].T_SC_D)
    assert frames[1].reference_pose is None
    with np.load(output / "trajectory.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["source_frame_index"], [5, 6])
        np.testing.assert_array_equal(archive["reference_present"], [True, False])
        assert np.isnan(archive["T_C1C0"][0]).all()
        np.testing.assert_array_equal(archive["T_C1C0"][1], rows[1].pair.T_C1C0)
    with (output / "frames.csv").open(newline="") as stream:
        entries = list(csv.DictReader(stream))
    assert [entry["source_frame_index"] for entry in entries] == ["5", "6"]
    assert entries[1]["inliers"] == "20"
    assert not list(tmp_path.glob(".odometry-*"))


@pytest.mark.parametrize("phase", ["trajectory", "csv", "report", "publication"])
def test_failed_export_cleans_staging_and_allows_retry(
    tmp_path, monkeypatch, inputs, phase
):
    rows, report = inputs
    output = tmp_path / "run"
    with monkeypatch.context() as patch:
        if phase == "trajectory":
            original = run_io.np.savez_compressed

            def fail(*args, **kwargs):
                original(*args, **kwargs)
                assert not output.exists()
                raise OSError("simulated export failure")

            patch.setattr(run_io.np, "savez_compressed", fail)
        elif phase == "csv":
            original = csv.DictWriter.writerow

            def fail(writer, entry):
                original(writer, entry)
                assert not output.exists()
                raise OSError("simulated export failure")

            patch.setattr(csv.DictWriter, "writerow", fail)
        elif phase == "report":
            original = Path.write_text

            def fail(path, *args, **kwargs):
                original(path, *args, **kwargs)
                assert not output.exists()
                raise OSError("simulated export failure")

            patch.setattr(Path, "write_text", fail)
        else:

            def fail(*args, **kwargs):
                assert not output.exists()
                raise OSError("simulated export failure")

            patch.setattr(Path, "rename", fail)
        with pytest.raises(OSError, match="simulated export failure"):
            run_io.save_run(output, rows, report, 5)
    assert not output.exists()
    assert not list(tmp_path.glob(".odometry-*"))
    run_io.save_run(output, rows, report, 5)
    assert run_io.load_trajectory(output)[0] == report


def test_existing_run_is_never_overwritten(tmp_path, inputs):
    rows, report = inputs
    output = tmp_path / "run"
    run_io.save_run(output, rows, report, 5)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="already exists"):
        run_io.save_run(output, [], {}, 0)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not list(tmp_path.glob(".odometry-*"))


@pytest.mark.parametrize("kind", ["empty_directory", "file", "symlink", "dangling"])
def test_existing_destinations_are_refused_without_modification(tmp_path, inputs, kind):
    rows, report = inputs
    output = tmp_path / "run"
    target = tmp_path / "target"
    if kind == "empty_directory":
        output.mkdir()
    elif kind == "file":
        output.write_bytes(b"preserve")
    else:
        if kind == "symlink":
            target.mkdir()
            (target / "sentinel").write_bytes(b"preserve")
        try:
            output.symlink_to(target, target_is_directory=True)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                pytest.skip("Windows symbolic-link privilege is unavailable")
            raise
        original_link_target = output.readlink()
    with pytest.raises(ValueError, match="already exists"):
        run_io.save_run(output, rows, report, 5)
    if kind == "empty_directory":
        assert list(output.iterdir()) == []
    elif kind == "file":
        assert output.read_bytes() == b"preserve"
    else:
        assert output.is_symlink()
        assert output.readlink() == original_link_target
        if kind == "symlink":
            assert (target / "sentinel").read_bytes() == b"preserve"
        else:
            assert not target.exists()
    assert not list(tmp_path.glob(".odometry-*"))


def test_destination_appearing_during_export_is_preserved(
    tmp_path, monkeypatch, inputs
):
    rows, report = inputs
    output = tmp_path / "run"
    original = Path.write_text

    def concurrent_destination(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.name == "report.json":
            output.mkdir()
            (output / "sentinel").write_bytes(b"another writer")
        return result

    monkeypatch.setattr(Path, "write_text", concurrent_destination)
    with pytest.raises(ValueError, match="appeared"):
        run_io.save_run(output, rows, report, 5)
    assert {path.name for path in output.iterdir()} == {"sentinel"}
    assert (output / "sentinel").read_bytes() == b"another writer"
    assert not list(tmp_path.glob(".odometry-*"))
