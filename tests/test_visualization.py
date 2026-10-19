"""Offline recordings, calibrated transforms and scientific data preservation."""

import json
import shutil
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from helpers.mapping import plane, pose, row, samples
from helpers.runs import make_run

from scene_recall.mapping.pipeline import Keyframe, build_submap
from scene_recall.mapping.runner import run_mapping
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.visualization.recording import export_recording
from scene_recall.visualization.rerun import (
    ESTIMATED,
    REFERENCE,
    DisplayConfig,
    blueprint,
    comparison_view,
    display_indices,
    require_rerun,
)


def blueprint_views(layout):
    """Collect native views without depending on container nesting."""
    import rerun.blueprint as rrb

    if isinstance(layout, rrb.Blueprint):
        yield from blueprint_views(layout.root_container)
    elif isinstance(layout, rrb.Container):
        for child in layout.contents:
            yield from blueprint_views(child)
    else:
        yield layout


@pytest.mark.parametrize("kind", ["cloud", "surface", "mesh"])
def test_comparison_uses_one_view_with_only_local_color_overrides(kind):
    rr = pytest.importorskip("rerun")
    root = "segments/2/submaps/3"
    geometries = {
        mode: (f"{root}/{mode}/map", kind) for mode in ("estimated", "reference")
    }
    view = comparison_view(root, geometries, paired=True)
    assert view.origin == root
    assert set(view.contents) == {path for path, _ in geometries.values()}
    # No separate cameras, tracking eye, transform overrides or duplicated geometry.
    assert not view.properties
    for mode, color in (("estimated", ESTIMATED), ("reference", REFERENCE)):
        visualizer = view.visualizer_overrides[geometries[mode][0]]
        assert visualizer.visualizer_type == (
            "Mesh3D" if kind == "mesh" else "Points3D"
        )
        expected = rr.components.ColorBatch([color]).as_arrow_array()
        if kind == "mesh":
            # A mesh needs a uniform material and an empty per-vertex override.
            assert len(visualizer.overrides) == 2
            assert visualizer.overrides[0].as_arrow_array().to_pylist() == []
            assert visualizer.overrides[1].as_arrow_array() == expected
        else:
            assert len(visualizer.overrides) == 1
            assert visualizer.overrides[0].as_arrow_array() == expected


def test_submap_comparison_preserves_existing_views_and_unpaired_warning():
    pytest.importorskip("rerun")
    submaps = [
        {"segment_id": 2, "submap_id": 3, "comparison_status": "paired"},
        {
            "segment_id": 9,
            "submap_id": 4,
            "comparison_status": "unavailable_unpaired_references",
        },
    ]
    workspace = blueprint([2, 9], submaps, show_images=False)
    views = list(blueprint_views(workspace))
    comparisons = [v for v in views if v.name.startswith("Compare VO / GT")]
    assert len(comparisons) == 2
    assert "unpaired" not in comparisons[0].name
    assert "unpaired keyframes" in comparisons[1].name
    for view in comparisons:
        assert all(p.startswith(view.origin + "/") for p in view.contents)
        assert not any("camera" in p or "trajectory" in p for p in view.contents)
    assert sum(v.name == "Estimated poses" for v in views) == 4
    assert workspace.blueprint_panel.state == "expanded"


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
from scene_recall.visualization.experiments import export_experiment
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
        disconnected.scale.as_arrow_array().to_pylist(), [[1, 1, 1]]
    )
    assert disconnected.parent_frame.as_arrow_array().to_pylist() == [
        "scene_recall/segments/0"
    ]
    no_timestamp = replace(row(0), timestamp=None)
    logger.set_frame(no_timestamp, 12)
    assert capture.time == {"frame": {"sequence": 12}}


def test_independent_gauges_have_distinct_named_roots_and_unit_scale():
    rr = pytest.importorskip("rerun")
    from scene_recall.visualization.rerun import SceneLogger

    _, calibration, _ = plane()
    capture = Capture()
    logger = SceneLogger(capture, calibration)
    roots = [
        "segments/0",
        "segments/1",
        "experiments/baseline/sparse/segments/0",
        "experiments/baseline/hybrid/segments/0",
        "experiments/mesh/sparse/segments/0",
        "experiments/tsdf/sparse/segments/0",
    ]
    parents = set()
    for root in roots:
        logger.independent_space(root)
        transform, _, static = capture.values(root, rr.Transform3D)[0]
        parent = transform.parent_frame.as_arrow_array().to_pylist()[0]
        assert not parent.startswith("tf#")
        assert parent.endswith(root)
        assert parent not in parents
        parents.add(parent)
        assert static
        assert transform.child_frame is None  # Keep the entity's implicit frame.
        assert transform.translation is None
        assert transform.mat3x3 is None
        assert transform.quaternion is None
        assert transform.scale.as_arrow_array().to_pylist() == [[1, 1, 1]]


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
    _, run = make_run(tmp_path, continuous=True)
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
    # Both clouds are already in M, despite nonidentity segment/world gauges.
    # Rendering them together must neither realign them nor apply T_SM twice.
    point = np.array([0, 0, 1, 1])
    for mode, transforms in (
        ("estimated", [np.linalg.inv(submap.T_SM) @ r.T_SC_D for r in frames]),
        ("reference", [submap.T_MW_reference @ r.reference_pose for r in frames]),
    ):
        logged = capture.values(f"{root}/{mode}/map/rgb", rr.Points3D)[0][0]
        positions = np.array(logged.positions.as_arrow_array().to_pylist())
        expected = np.array([(T @ point)[:3] for T in transforms])
        np.testing.assert_allclose(
            positions[np.argsort(positions[:, 0])],
            expected[np.argsort(expected[:, 0])],
            atol=1e-6,
        )
        np.testing.assert_allclose(positions, clouds[mode]["points"], atol=1e-6)
