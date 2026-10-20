"""Controlled identity checks for the independent RGB association diagnostic."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.pose_only import retract

experiment = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/rgb_geometry_multiview.py")
)
rgb_graph = experiment["rgb_graph"]


def images(count=4, features=96, outliers=16):
    rng = np.random.default_rng(913)
    points = rng.uniform([-0.7, -0.45, 2.0], [0.7, 0.45, 4.0], (features, 3))
    descriptors = rng.uniform(0, 255, (features, 128)).astype(np.float32)
    K = np.array([[300.0, 0, 160], [0, 310.0, 120], [0, 0, 1.0]])
    poses, pixels, camera_points = [], [], []
    for i in range(count):
        pose = retract(
            np.eye(4), np.array([0.04 * i, 0.006 * i * i, 0, 0, 0.02 * i, 0])
        )
        p = transform_points(points, np.linalg.inv(pose))
        xy = project_points(p, K)
        if outliers:
            xy[-outliers:] = rng.uniform([0, 0], [320, 240], (outliers, 2))
        poses.append(pose)
        camera_points.append(p)
        pixels.append(xy)
    graph = {
        "frame_index": np.arange(count),
        "segment_id": np.zeros(count, dtype=int),
        "timestamp_s": np.arange(count) * 0.1,
        "T_SR": np.array(poses),
        "feature_offsets": np.arange(count + 1) * features,
        "pixels": np.concatenate(pixels),
        "points_R": np.concatenate(camera_points),
    }
    return graph, {i: descriptors.copy() for i in range(count)}


def test_geometric_consensus_retains_true_identities_and_rejects_wrong_pixels():
    graph, descriptors = images()
    result, diagnostics = rgb_graph(graph, descriptors)
    assert diagnostics["verified_pairs"] == diagnostics["candidate_pairs"]
    good, bad = 0, 0
    for lo, hi in zip(
        result["track_offsets"][:-1], result["track_offsets"][1:], strict=True
    ):
        keys = result["observation_keys"][lo:hi]
        assert len(set(keys[:, 0])) == len(keys)
        assert len(set(keys[:, 1])) == 1
        if keys[0, 1] < 80 and len(keys) == 4:
            good += 1
        elif keys[0, 1] >= 80:
            bad += 1
    assert good >= 75
    assert bad <= 3


def test_association_is_independent_of_estimated_poses_depth_and_references():
    graph, descriptors = images()
    original, diagnostic = rgb_graph(graph, descriptors)
    poisoned = {key: value.copy() for key, value in graph.items()}
    poisoned["T_SR"][:] = np.nan
    poisoned["points_R"][:] = 1e8
    poisoned["reference_pose"] = np.full((4, 4, 4), np.nan)
    changed, other = rgb_graph(poisoned, descriptors)
    assert diagnostic == other
    for key in original:
        if key not in ("T_SR", "points_R"):
            np.testing.assert_array_equal(original[key], changed[key])


def test_repeated_descriptors_never_merge_independent_segments():
    graph, descriptors = images(count=6, outliers=0)
    graph["segment_id"][3:] = 1
    result, _ = rgb_graph(graph, descriptors)
    for lo, hi in zip(
        result["track_offsets"][:-1], result["track_offsets"][1:], strict=True
    ):
        keys = result["observation_keys"][lo:hi]
        assert len(set(graph["segment_id"][keys[:, 0]])) == 1
    assert all(
        graph["segment_id"][a] == graph["segment_id"][b]
        for a, b in result["candidate_pairs"]
    )


@pytest.mark.parametrize("empty", [False, True])
def test_insufficient_rgb_support_creates_no_identities(empty):
    graph, descriptors = images(features=8, outliers=0)
    if empty:
        descriptors = {i: None for i in descriptors}
    result, diagnostics = rgb_graph(graph, descriptors)
    assert diagnostics["verified_pairs"] == 0
    assert result["observation_keys"].shape == (0, 2)
    assert not len(result["track_id"])


def test_saved_diagnostic_replays_without_media_and_rejects_modified_inputs(
    tmp_path, monkeypatch
):
    from scene_recall.mapping.voxel import MappingConfig
    from scene_recall.multiview.temporal_runner import run_temporal_experiment

    fixture = runpy.run_path(str(Path(__file__).with_name("test_pose_experiment.py")))
    _, baseline = fixture["synthetic_source"](tmp_path)
    fixed, output = tmp_path / "fixed", tmp_path / "rgb"
    run_temporal_experiment(
        baseline, fixed, mapping_config=MappingConfig(keyframe_stride=2), offset=1
    )
    experiment["run"](fixed, output)

    def forbid(*args, **kwargs):
        raise AssertionError("RGB replay opened source media")

    with monkeypatch.context() as m:
        m.setattr("scene_recall.multiview.temporal_runner.open_source", forbid)
        assert experiment["replay"](output)["full_rate_trajectory_exact"]
    with pytest.raises(ValueError, match="already exists"):
        experiment["run"](fixed, output)
    path = output / "descriptors.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="input changed"):
        experiment["replay"](output)
