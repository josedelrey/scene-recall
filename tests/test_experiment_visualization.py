"""Scientific experiment adapters preserve geometry, populations and pose ownership."""

import json
import runpy
import shutil
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from helpers.mapping import plane, pose, row
from helpers.runs import continuous_run
from test_visualization import Capture, _rrd, blueprint_views

from scene_recall.mapping.heldout import run_heldout
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.run_metadata import sha256
from scene_recall.visualization.experiments import (
    HELDOUT,
    TSDF,
    _view_archive,
    export_experiment,
)
from scene_recall.visualization.rerun import DisplayConfig, SceneLogger


@pytest.fixture(scope="module")
def experiments(tmp_path_factory):
    pytest.importorskip("rerun")
    root = tmp_path_factory.mktemp("experiment_visualization")
    source, original = continuous_run(root)
    metadata, frames = load_trajectory(original)
    frames = [replace(f, T_SC_D=pose(x=0.05 * f.frame_index)) for f in frames]
    odometry = root / "shifted"
    save_run(odometry, frames, metadata, 0)
    baseline = root / "baseline"
    run_heldout(
        [odometry], baseline, MappingConfig(pixel_stride=32, keyframe_stride=2), 1
    )
    outputs = {"heldout": baseline}
    for name, script in (
        ("tsdf", "compare_tsdf_fusion"),
        ("support", "support_aware_fusion"),
        ("motion", "compare_keyframe_selection"),
    ):
        output = root / name
        run = runpy.run_path(Path(__file__).parents[1] / f"experiments/{script}.py")[
            "run"
        ]
        run(baseline, [odometry], output)
        outputs[name] = output
    return root, source, odometry, baseline, outputs


@pytest.mark.parametrize("name", ["heldout", "tsdf", "support", "motion"])
def test_experiment_rrd_roundtrip_preserves_scientific_files(experiments, name):
    root, _, odometry, baseline, outputs = experiments
    run = outputs[name]
    before = {
        p: sha256(p)
        for folder in {run, baseline, odometry}
        for p in folder.iterdir()
        if p.is_file()
    }
    output = root / f"{name}.rrd"
    export_experiment(
        output,
        experiment_run=run,
        baseline_run=baseline if name != "heldout" else None,
        odometry_runs=[odometry],
        display=DisplayConfig(20),
    )
    executable = str(Path(sys.executable).with_name("rerun"))
    _rrd(executable, "verify", output)
    schema = _rrd(executable, "print", output)
    assert "BlueprintActivationCommand" in schema
    assert "DepthImage:buffer" in schema
    assert "Tensor:data" in schema
    assert "signed_error_m" in schema
    assert "recording/experiment_report" in schema
    assert "Clear:is_recursive" in schema
    assert (
        "Mesh3D:vertex_positions" in schema
        if name == "tsdf"
        else "Points3D:positions" in schema
    )
    assert all(sha256(p) == digest for p, digest in before.items())
    assert not list(root.glob(".recording-*"))


def _tensor(value):
    data = value.data.as_arrow_array().to_pylist()[0]
    return np.array(data["buffer"]).reshape(data["shape"])


def test_evaluated_lattice_error_sign_masks_and_observed_surface():
    rr = pytest.importorskip("rerun")
    _, calibration, _ = plane()
    calibration.K_D[:] = [[4, 0, 2], [0, 8, 2], [0, 0, 1]]
    observed = np.array([[1, 2, np.nan], [3, 4, 5.0]])
    predicted = np.array([[0.5, 2.3, 1], [np.nan, 4, 6.0]])
    common = np.array([[True, False, False], [False, True, False]])
    before = [a.copy() for a in (observed, predicted, common)]
    capture = Capture()
    logger = SceneLogger(capture, calibration)
    logger.set_frame(row(3), 20)
    logger.evaluation(
        "evaluation", pose(x=2, yaw=0.2), observed, predicted, {"common": common}, 2
    )
    pinhole, time, _ = capture.values("evaluation/camera/lattice", rr.Pinhole)[0]
    K = (
        np.array(pinhole.image_from_camera.as_arrow_array().to_pylist()[0])
        .reshape(3, 3)
        .T
    )
    np.testing.assert_array_equal(K, [[2, 0, 1], [0, 4, 1], [0, 0, 1]])
    assert time["frame"] == {"sequence": 23}
    tensor = capture.values("evaluation/signed_error_m", rr.Tensor)[0][0]
    np.testing.assert_allclose(_tensor(tensor), [[-0.5, 0.3, np.nan], [np.nan, 0, 1]])
    points = capture.values("evaluation/camera/residual_points", rr.Points3D)[0][0]
    positions = points.positions.as_arrow_array().to_pylist()
    np.testing.assert_allclose(
        positions,
        [[-0.5, -0.25, 1], [0, -0.5, 2], [-1.5, 0, 3], [0, 0, 4], [2.5, 0, 5]],
    )
    image = capture.values("evaluation/residual", rr.Image)[0][0]
    colors = np.frombuffer(
        bytes(image.buffer.as_arrow_array()[0].as_py()), dtype=np.uint8
    ).reshape(2, 3, 3)
    np.testing.assert_array_equal(colors[0, 0], [220, 45, 45])
    np.testing.assert_array_equal(colors[0, 1], [45, 90, 220])
    np.testing.assert_array_equal(colors[0, 2], [0, 0, 0])
    np.testing.assert_array_equal(colors[1, 0], [110, 110, 110])
    transform = capture.values("evaluation/camera", rr.Transform3D)[0][0]
    assert transform.translation.as_arrow_array().to_pylist() == [[2, 0, 0]]
    for a, b in zip((observed, predicted, common), before, strict=True):
        np.testing.assert_array_equal(a, b)


class ExperimentCapture(Capture):
    def send_blueprint(self, blueprint):
        self.blueprint = blueprint


@pytest.mark.parametrize(
    ("name", "representations"),
    [
        ("heldout", {"baseline"}),
        ("tsdf", {"baseline", "mesh", "tsdf"}),
        ("support", {"baseline", "consensus"}),
        ("motion", {"motion", "temporal"}),
    ],
)
def test_experiment_comparisons_reuse_colocated_maps(
    experiments, monkeypatch, name, representations
):
    rr = pytest.importorskip("rerun")
    root, _, odometry, baseline, outputs = experiments
    capture = ExperimentCapture()

    @contextmanager
    def sink(output):
        yield capture

    monkeypatch.setattr("scene_recall.visualization.experiments.recording_stream", sink)
    export_experiment(
        root / "comparison.rrd",
        experiment_run=outputs[name],
        baseline_run=baseline if name != "heldout" else None,
        odometry_runs=[odometry],
        display=DisplayConfig(20),
    )
    views = list(blueprint_views(capture.blueprint))
    comparisons = [v for v in views if "/ VO amber / GT teal" in v.name]
    assert {v.name.split(" / ")[0] for v in comparisons} == representations
    logged_paths = {path for path, _, _, _ in capture.logs}
    for view in comparisons:
        assert set(view.contents) <= logged_paths
        assert len(view.contents) == 2
        assert {p.split(view.origin + "/")[1].split("/")[0] for p in view.contents} == {
            "estimated",
            "reference",
        }
        assert not view.properties
    assert any(v.name == "Final reconstruction and trajectories" for v in views)
    assert any(v.name == "Signed error values (m)" for v in views)
    # Every representation's segment is metrically rooted outside the implicit
    # global tree, including geometry not selected in the active comparison tab.
    segment_roots = {
        path
        for path, _, _, _ in capture.logs
        if path.rsplit("/", 2)[-2:] == ["segments", "0"]
    }
    assert segment_roots
    for path in segment_roots:
        for transform, _, static in capture.values(path, rr.Transform3D):
            assert static
            assert transform.parent_frame.as_arrow_array().to_pylist() == [
                f"scene_recall/{path}"
            ]
            assert transform.scale.as_arrow_array().to_pylist() == [[1, 1, 1]]


def test_adapter_keeps_target_pose_modes_and_saved_scores(experiments, monkeypatch):
    rr = pytest.importorskip("rerun")
    root, _, odometry, baseline, _ = experiments
    capture = ExperimentCapture()

    @contextmanager
    def sink(output):
        yield capture

    monkeypatch.setattr("scene_recall.visualization.experiments.recording_stream", sink)
    export_experiment(
        root / "capture.rrd",
        experiment_run=baseline,
        odometry_runs=[odometry],
        images="evaluated",
        display=DisplayConfig(10),
    )
    estimated = "experiments/baseline/sparse/segments/0/submaps/0/estimated/evaluation/radius_0.01/"
    for mode, x in (("estimated_pose", 0.05), ("reference_pose", 0.0)):
        path = estimated + mode
        transforms = capture.values(path + "/camera", rr.Transform3D)
        assert len(transforms) == 1
        np.testing.assert_allclose(
            transforms[0][0].translation.as_arrow_array().to_pylist(), [[x, 0, 0]]
        )
        clears = capture.values(path, rr.Clear)
        assert [time["frame"]["sequence"] for _, time, _ in clears] == [0, 2]
        report = json.loads((baseline / "report.json").read_text())
        expected = report["views"]["0.01"][0]["scores"]["sparse_" + mode]["own"][
            "coverage"
        ]
        logged = capture.values(path + "/scores/own/coverage", rr.Scalars)[0][0]
        assert logged.scalars.as_arrow_array().to_pylist() == [expected]
    raw = capture.values(
        "experiments/baseline/sparse/segments/0/submaps/0/estimated/camera/rgb/image",
        rr.Image,
    )
    assert all(time["frame"]["sequence"] == 1 for _, time, _ in raw)


@pytest.mark.parametrize("missing", [0, 1])
def test_unavailable_references_keep_maps_without_evaluations(tmp_path, missing):
    pytest.importorskip("rerun")
    _, odometry = continuous_run(tmp_path, missing_reference=missing)
    baseline = tmp_path / "baseline"
    run_heldout(
        [odometry], baseline, MappingConfig(pixel_stride=32, keyframe_stride=2), 1
    )
    output = tmp_path / "unavailable.rrd"
    export_experiment(output, experiment_run=baseline, odometry_runs=[odometry])
    schema = _rrd(str(Path(sys.executable).with_name("rerun")), "print", output)
    assert "Points3D:positions" in schema
    assert "DepthImage:buffer" not in schema
    assert "diagnostics/evaluation" in schema


def test_source_free_export_and_wrong_baseline_are_atomic(
    experiments, tmp_path, monkeypatch
):
    _, _, odometry, baseline, outputs = experiments
    moved = tmp_path / "relocated_odometry"
    shutil.copytree(odometry, moved)

    def no_source(*args):
        raise AssertionError("source-free export opened a dataset")

    monkeypatch.setattr("scene_recall.visualization.experiments.open_source", no_source)
    output = tmp_path / "offline.rrd"
    export_experiment(
        output,
        experiment_run=outputs["tsdf"],
        baseline_run=baseline,
        odometry_runs=[moved],
    )
    wrong = tmp_path / "wrong"
    shutil.copytree(baseline, wrong)
    (wrong / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="hash differs"):
        export_experiment(
            tmp_path / "wrong.rrd",
            experiment_run=outputs["tsdf"],
            baseline_run=wrong,
            odometry_runs=[moved],
        )
    assert not (tmp_path / "wrong.rrd").exists()
    assert not list(tmp_path.glob(".recording-*"))


@pytest.mark.parametrize(
    "corruption",
    ["indices", "mask", "depth", "shape", "prediction", "training_target", "source"],
)
def test_invalid_view_archive_is_rejected(experiments, tmp_path, corruption):
    _, _, odometry, baseline, _ = experiments
    moved = tmp_path / "baseline"
    shutil.copytree(baseline, moved)
    path = moved / "views_000_radius_0.01.npz"
    with np.load(path) as archive:
        arrays = dict(archive)
    if corruption in ("training_target", "source"):
        report_path = moved / "report.json"
        report = json.loads(report_path.read_text())
        if corruption == "source":
            report["source"]["start"] += 10
        else:
            report["views"]["0.01"][0]["frame_index"] = 0
            arrays["evaluated_frame_indices"][0] = 0
        report_path.write_text(json.dumps(report))
    elif corruption == "indices":
        arrays["evaluated_frame_indices"][0] = 0
    elif corruption == "mask":
        arrays["common_all_modes"][:] = True
    elif corruption == "depth":
        arrays["observed"][0, 1, 1] = -1
    elif corruption == "shape":
        arrays["observed"] = arrays["observed"][:, :-1]
    else:
        del arrays["sparse_reference_pose"]
    np.savez_compressed(path, **arrays)
    output = tmp_path / "corrupt.rrd"
    with pytest.raises(ValueError):
        export_experiment(output, experiment_run=moved, odometry_runs=[odometry])
    assert not output.exists()
    assert not list(tmp_path.glob(".recording-*"))


def test_tsdf_primary_mask_excludes_point_disc_control(tmp_path):
    observed = np.ones((1, 1, 2))
    arrays = {
        "evaluated_frame_indices": np.array([1]),
        "observed": observed,
        "baseline_gt": observed,
        "mesh_gt": observed,
        "tsdf_gt": np.array([[[np.nan, 1.0]]]),
        "common_all_versions": np.ones_like(observed, dtype=bool),
    }
    np.savez_compressed(tmp_path / "views_000_radius_0.01.npz", **arrays)
    report = {
        "experiment": TSDF,
        "render": {"primary": "baseline_discs_vs_connected_tsdf_mesh"},
    }
    rows = [
        {
            "frame_index": 1,
            "scores": {k: {} for k in ("baseline_gt", "mesh_gt", "tsdf_gt")},
        }
    ]
    loaded = _view_archive(tmp_path, tmp_path, report, 0, "0.01", rows, (1, 2))
    assert loaded["common_all_versions"].all()
    report["experiment"] = HELDOUT
    arrays["common_all_modes"] = arrays.pop("common_all_versions")
    np.savez_compressed(tmp_path / "views_000_radius_0.01.npz", **arrays)
    with pytest.raises(ValueError, match="visibility"):
        _view_archive(tmp_path, tmp_path, report, 0, "0.01", rows, (1, 2))


def test_mesh_connectivity_and_colors_ignore_point_display_filters():
    rr = pytest.importorskip("rerun")
    _, calibration, _ = plane()
    arrays = {
        "vertices": np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1.0]]),
        "triangles": np.array([[2, 1, 0]], dtype=np.int32),
        "colors": np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8),
    }
    capture = Capture()
    logger = SceneLogger(capture, calibration, DisplayConfig(1, 20))
    logger.mesh("mesh", arrays)
    mesh, _, static = capture.values("mesh", rr.Mesh3D)[0]
    assert static
    np.testing.assert_array_equal(
        mesh.vertex_positions.as_arrow_array().to_pylist(), arrays["vertices"]
    )
    np.testing.assert_array_equal(
        mesh.triangle_indices.as_arrow_array().to_pylist(), arrays["triangles"]
    )
    colors = np.array(mesh.vertex_colors.as_arrow_array().to_pylist(), dtype=np.uint32)
    rgb = np.column_stack([(colors >> shift) & 255 for shift in (24, 16, 8)])
    np.testing.assert_array_equal(rgb, arrays["colors"])


def test_tsdf_near_view_support_is_not_voxel_frame_support():
    rr = pytest.importorskip("rerun")
    _, calibration, _ = plane()
    capture = Capture()
    logger = SceneLogger(capture, calibration, DisplayConfig(3, 20))
    arrays = {
        "points": np.array([[0, 0, 1], [1, 0, 1.0]]),
        "colors": np.zeros((2, 3), dtype=np.uint8),
        "view_support": np.array([0, 2]),
        "color_support": np.array([0, 1]),
        "signed_residual_std_m": np.array([0, 0.01]),
    }
    logger.surface("surface", arrays)
    points = capture.values("surface", rr.Points3D)[0][0]
    assert len(points.positions.as_arrow_array()) == 2
    attributes = capture.values("surface", rr.AnyValues)[0][0]
    batches = {
        str(b.component_descriptor()): b.as_arrow_array().to_pylist()
        for b in attributes.as_component_batches()
    }
    assert batches["view_support"] == [0, 2]
    assert "frame_support" not in batches


@pytest.mark.parametrize("corruption", ["indices", "colors", "count"])
def test_bad_mesh_never_publishes(experiments, tmp_path, corruption):
    _, _, odometry, baseline, outputs = experiments
    moved = tmp_path / "tsdf"
    shutil.copytree(outputs["tsdf"], moved)
    path = moved / "sparse_000_mesh.npz"
    with np.load(path) as archive:
        arrays = dict(archive)
    if corruption == "indices":
        arrays["triangles"][0, 0] = len(arrays["vertices"])
    elif corruption == "colors":
        arrays["colors"] = arrays["colors"][:-1]
    else:
        arrays["triangles"] = arrays["triangles"][:-1]
    np.savez_compressed(path, **arrays)
    output = tmp_path / "bad_mesh.rrd"
    with pytest.raises(ValueError):
        export_experiment(
            output,
            experiment_run=moved,
            baseline_run=baseline,
            odometry_runs=[odometry],
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".recording-*"))


def test_raw_image_mismatch_never_publishes(tmp_path):
    pytest.importorskip("rerun")
    from PIL import Image

    source, odometry = continuous_run(tmp_path)
    baseline = tmp_path / "baseline"
    run_heldout(
        [odometry], baseline, MappingConfig(pixel_stride=32, keyframe_stride=2), 1
    )
    Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(source / "rgb.png")
    output = tmp_path / "bad_images.rrd"
    with pytest.raises(ValueError):
        export_experiment(
            output,
            experiment_run=baseline,
            odometry_runs=[odometry],
            images="evaluated",
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".recording-*"))


def test_missing_odometry_cohort_and_baseline_hashes_are_rejected(
    experiments, tmp_path
):
    _, _, odometry, baseline, outputs = experiments
    with pytest.raises(ValueError, match="cohort"):
        export_experiment(
            tmp_path / "missing.rrd", experiment_run=baseline, odometry_runs=[]
        )
    with pytest.raises(ValueError, match="odometry input"):
        export_experiment(
            tmp_path / "duplicate.rrd",
            experiment_run=baseline,
            odometry_runs=[odometry, odometry],
        )
    moved = tmp_path / "baseline"
    shutil.copytree(baseline, moved)
    path = moved / "views_000_radius_0.01.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive hash"):
        export_experiment(
            tmp_path / "hash.rrd",
            experiment_run=outputs["tsdf"],
            baseline_run=moved,
            odometry_runs=[odometry],
        )
    assert not list(tmp_path.glob("*.rrd"))


def test_selected_submap_and_old_point_only_tsdf(experiments, tmp_path):
    pytest.importorskip("rerun")
    _, _, odometry, baseline, outputs = experiments
    moved = tmp_path / "old_tsdf"
    shutil.copytree(outputs["tsdf"], moved)
    path = moved / "report.json"
    report = json.loads(path.read_text())
    del report["render"]
    for chunk in report["chunks"]:
        for stats in chunk["tsdf"].values():
            del stats["mesh_vertices"], stats["mesh_triangles"]
    for view in report["views"]:
        view["scores"] = {
            k: v for k, v in view["scores"].items() if not k.startswith("mesh_")
        }
    archive = moved / "views_000_radius_0.01.npz"
    with np.load(archive) as data:
        arrays = {k: v for k, v in dict(data).items() if not k.startswith("mesh_")}
    common = np.isfinite(arrays["observed"])
    for key in report["views"][0]["scores"]:
        common &= np.isfinite(arrays[key])
    arrays["common_all_versions"] = common
    np.savez_compressed(archive, **arrays)
    path.write_text(json.dumps(report))
    output = tmp_path / "old.rrd"
    export_experiment(
        output,
        experiment_run=moved,
        baseline_run=baseline,
        odometry_runs=[odometry],
        submap_ids=[0],
    )
    schema = _rrd(str(Path(sys.executable).with_name("rerun")), "print", output)
    assert "Mesh3D:vertex_positions" not in schema
    assert "view_support" in schema
    with pytest.raises(ValueError, match="submap selection"):
        export_experiment(
            tmp_path / "selection.rrd",
            experiment_run=moved,
            baseline_run=baseline,
            odometry_runs=[odometry],
            submap_ids=[9],
        )
