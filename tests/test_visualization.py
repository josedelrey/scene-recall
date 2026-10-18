"""Offline recordings, calibrated transforms and scientific data preservation."""

import json
import shutil
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from test_mapping import plane, pose, row, samples
from test_mapping_cli import make_run

from scene_recall.mapping.pipeline import Keyframe, build_submap
from scene_recall.mapping.runner import run_mapping
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.visualization.recording import export_recording
from scene_recall.visualization.rerun import (
    DisplayConfig,
    display_indices,
    require_rerun,
)


def test_optional_sdk_failure_is_actionable(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr("scene_recall.visualization.rerun.import_module", missing)
    with pytest.raises(RuntimeError, match="uv sync --locked --extra visualization"):
        require_rerun()


def test_core_imports_without_optional_sdk():
    code = """
import sys
sys.modules['rerun'] = None
from scene_recall.mapping.runner import run_mapping
from scene_recall.odometry.cli import main
from scene_recall.visualization.recording import export_recording
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_display_filter_is_bounded_deterministic_and_preserves_support():
    support = np.array([1, 2, 1, 4, 3, 1, 8])
    original = support.copy()
    assert display_indices(support, DisplayConfig(3, 2)).tolist() == [1, 3, 6]
    assert display_indices(support, DisplayConfig(10, 9)).tolist() == []
    np.testing.assert_array_equal(support, original)
    for kwargs in ({"min_support": 0}, {"max_points": True}):
        with pytest.raises(ValueError, match="positive integer"):
            DisplayConfig(**kwargs)


class Capture:
    """Inspect real SDK archetypes and timelines at the adapter boundary."""

    def __init__(self):
        self.time = {}
        self.logs = []

    def reset_time(self):
        self.time = {}

    def set_time(self, name, **value):
        self.time[name] = value

    def log(self, path, *values, static=False):
        self.logs.append((path, values, {} if static else self.time.copy(), static))

    def values(self, path, archetype):
        return [
            (value, time, static)
            for name, values, time, static in self.logs
            if name == path
            for value in values
            if isinstance(value, archetype)
        ]


def test_camera_geometry_depth_units_missing_references_and_time():
    rr = pytest.importorskip("rerun")
    from scene_recall.visualization.rerun import SceneLogger

    observation, calibration, _ = plane()
    calibration.T_RD[:3, 3] = [0.1, 0.2, 0.3]
    observation.depth[4, 4] = np.nan
    original = observation.depth.copy()
    frames = [
        row(0, reference=pose(x=10)),
        row(1, T=pose(x=0.1), reference=None),
        row(2, T=pose(x=0.3), reference=pose(x=10.2)),
    ]
    capture = Capture()
    logger = SceneLogger(capture, calibration)
    logger.segment(frames, 50)
    logger.set_frame(frames[0], 50)
    logger.observation(observation, frames[0])
    root = "segments/0/estimated/camera"
    rgb_transform = capture.values(root + "/rgb", rr.Transform3D)[0][0]
    np.testing.assert_allclose(
        rgb_transform.translation.as_arrow_array().to_pylist(), [[-0.1, -0.2, -0.3]]
    )
    estimated = capture.values(root, rr.Transform3D)
    np.testing.assert_allclose(
        estimated[-1][0].translation.as_arrow_array().to_pylist(), [[0.3, 0, 0]]
    )
    reference = capture.values("segments/0/reference/camera", rr.Transform3D)
    np.testing.assert_allclose(
        reference[-1][0].translation.as_arrow_array().to_pylist(), [[0.2, 0, 0]]
    )
    assert len(capture.values("segments/0/reference/camera", rr.Clear)) == 1
    assert not capture.values("segments/0/reference/trajectory", rr.LineStrips3D)
    depth, time, static = capture.values(root + "/depth/image", rr.DepthImage)[0]
    assert depth.meter.as_arrow_array().to_pylist() == [1.0]
    logged_depth = np.frombuffer(
        bytes(depth.buffer.as_arrow_array()[0].as_py()), dtype=np.float32
    ).reshape(observation.depth.shape)
    assert logged_depth[4, 4] == 0
    np.testing.assert_array_equal(logged_depth[5], original[5])
    assert time == {"frame": {"sequence": 50}, "sensor_time": {"duration": 0.0}}
    assert not static
    np.testing.assert_array_equal(observation.depth, original)
    disconnected = capture.values("segments/0", rr.Transform3D)[0][0]
    np.testing.assert_array_equal(
        disconnected.scale.as_arrow_array().to_pylist(), [[0, 0, 0]]
    )
    no_timestamp = replace(row(0), timestamp=None)
    logger.set_frame(no_timestamp, 12)
    assert capture.time == {"frame": {"sequence": 12}}


def _rrd(executable, command, path, *extra):
    result = subprocess.run(
        [executable, "rrd", command, str(path), *extra],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


@pytest.mark.parametrize("images", ["keyframes", "all", "none"])
def test_real_recording_roundtrip_and_unchanged_scientific_outputs(tmp_path, images):
    pytest.importorskip("rerun")
    executable = shutil.which("rerun") or str(Path(sys.executable).with_name("rerun"))
    _, run = make_run(tmp_path)
    mapping = tmp_path / "mapping"
    run_mapping(run, mapping, MappingConfig(pixel_stride=32, keyframe_stride=2))
    before = {p: p.read_bytes() for folder in (run, mapping) for p in folder.iterdir()}
    output = tmp_path / f"{images}.rrd"
    export_recording(
        output, mapping_run=mapping, images=images, display=DisplayConfig(25)
    )
    assert output.stat().st_size > 0
    _rrd(executable, "verify", output)
    schema = _rrd(executable, "print", output)
    assert "BlueprintActivationCommand" in schema
    assert "/segments/0/submaps/0/estimated/map/rgb" in schema
    assert "/segments/1/submaps/1/reference/map/support" in schema
    assert "/segments/1/estimated/camera" in schema
    assert "frame_support" in schema
    assert "recording/guide" in schema
    assert (
        "DepthImage:buffer" in schema
        if images != "none"
        else "DepthImage:buffer" not in schema
    )
    assert all(path.read_bytes() == contents for path, contents in before.items())
    with pytest.raises(ValueError, match="already exists"):
        export_recording(output, mapping_run=mapping)
    assert not list(tmp_path.glob(".recording-*"))


def test_maps_only_without_source_or_odometry_and_relocated_odometry(tmp_path):
    pytest.importorskip("rerun")
    source, run = make_run(tmp_path)
    mapping = tmp_path / "mapping"
    run_mapping(run, mapping, MappingConfig(pixel_stride=32))
    moved = tmp_path / "moved_odometry"
    run.rename(moved)
    shutil.rmtree(source)
    export_recording(
        tmp_path / "relocated.rrd",
        mapping_run=mapping,
        odometry_run=moved,
        images="none",
    )
    shutil.rmtree(moved)
    export_recording(tmp_path / "maps.rrd", mapping_run=mapping, maps_only=True)
    with pytest.raises(ValueError, match="--odometry-run"):
        export_recording(tmp_path / "missing.rrd", mapping_run=mapping, images="none")


def test_odometry_only_recording(tmp_path):
    pytest.importorskip("rerun")
    _, run = make_run(tmp_path)
    export_recording(tmp_path / "odometry.rrd", odometry_run=run)


def test_sparse_images_are_cleared_at_unlogged_frames(tmp_path):
    pytest.importorskip("rerun")
    executable = str(Path(sys.executable).with_name("rerun"))
    _, run = make_run(tmp_path)
    report, frames = load_trajectory(run)
    frames = [
        replace(r, segment_id=0, status="tracked" if i else "initialized")
        for i, r in enumerate(frames)
    ]
    for row_meta, frame in zip(report["frames"], frames, strict=True):
        row_meta.update(segment_id=frame.segment_id, status=frame.status)
    save_run(run, frames, report, 0)
    mapping = tmp_path / "mapping"
    run_mapping(run, mapping, MappingConfig(pixel_stride=32, keyframe_stride=2))
    output = tmp_path / "sparse.rrd"
    export_recording(output, mapping_run=mapping)
    schema = _rrd(
        executable,
        "print",
        output,
        "--entity",
        "/segments/0/estimated/camera/depth/image",
        "-v",
    )
    assert "Clear:is_recursive" in schema
    assert "DepthImage:buffer" in schema


@pytest.mark.parametrize("corruption", ["rgb", "map", "odometry"])
def test_invalid_inputs_never_publish_recording(tmp_path, corruption):
    pytest.importorskip("rerun")
    source, run = make_run(tmp_path)
    mapping = tmp_path / "mapping"
    run_mapping(run, mapping, MappingConfig(pixel_stride=32))
    if corruption == "rgb":
        from PIL import Image

        Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(
            source / "rgb.png"
        )
    elif corruption == "map":
        path = mapping / "submap_000_estimated.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        metadata = json.loads(str(arrays["metadata_json"]))
        metadata["segment_id"] += 1
        arrays["metadata_json"] = np.array(json.dumps(metadata))
        np.savez_compressed(path, **arrays)
    else:
        path = run / "report.json"
        report = json.loads(path.read_text())
        report["backend"] = "changed"
        path.write_text(json.dumps(report))
    output = tmp_path / "invalid.rrd"
    with pytest.raises(ValueError):
        export_recording(output, mapping_run=mapping)
    assert not output.exists()
    assert not list(tmp_path.glob(".recording-*"))


def test_submap_transforms_match_saved_gauges():
    rr = pytest.importorskip("rerun")
    from scene_recall.visualization.rerun import SceneLogger

    _, calibration, config = plane()
    frames = [
        row(5, segment=2, T=pose(x=2, yaw=0.2), reference=pose(x=10)),
        row(6, segment=2, T=pose(x=2.5, yaw=0.3), reference=pose(x=10.4)),
    ]
    submap = build_submap(
        [Keyframe(r, samples([[0, 0, 1]])) for r in frames], 3, config
    )
    clouds = {
        mode: {
            f.name: getattr(getattr(submap, mode), f.name)
            for f in fields(submap.estimated)
        }
        for mode in ("estimated", "reference")
    }
    capture = Capture()
    logger = SceneLogger(capture, calibration)
    logger.submap(submap.report, clouds, frames, 20)
    root = "segments/2/submaps/3"
    transform = capture.values(root, rr.Transform3D)[0][0]
    np.testing.assert_array_equal(
        transform.translation.as_arrow_array().to_pylist(), [[2, 0, 0]]
    )
    camera = capture.values(root + "/estimated/camera", rr.Transform3D)[-1][0]
    expected = np.linalg.inv(submap.T_SM) @ frames[-1].T_SC_D
    np.testing.assert_allclose(
        camera.translation.as_arrow_array().to_pylist(), expected[None, :3, 3]
    )
    reference = capture.values(root + "/reference/camera", rr.Transform3D)[-1][0]
    np.testing.assert_allclose(
        reference.translation.as_arrow_array().to_pylist(), [[0.4, 0, 0]], atol=1e-6
    )
