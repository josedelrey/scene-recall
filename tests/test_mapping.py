"""Metric geometry, provenance and continuity contracts for surface mapping."""

from dataclasses import replace

import numpy as np
import pytest
from helpers.mapping import plane, pose, row, samples

from scene_recall.data import Calibration, Observation
from scene_recall.geometry.camera import transform_points
from scene_recall.mapping.evaluation import (
    _nearby_distances,
    compare_clouds,
    depth_consistency,
    summarize_consistency,
)
from scene_recall.mapping.pipeline import (
    Keyframe,
    build_submap,
    reconstruct,
    select_keyframes,
)
from scene_recall.mapping.voxel import (
    MappingConfig,
    VoxelMap,
    sample_observation,
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("voxel_size_m", 0),
        ("max_depth_m", np.inf),
        ("pixel_stride", True),
        ("keyframe_stride", 0),
        ("max_voxels", -1),
        ("min_depth_m", 4),
        ("max_keyframes_per_submap", 1.5),
        ("consistency_tolerance_m", np.nan),
    ],
)
def test_config_rejects_invalid_parameters(field, value):
    with pytest.raises(ValueError):
        MappingConfig(**{field: value})


def test_equal_frame_weighting_negative_cells_and_distinct_support():
    mapping = VoxelMap(1)
    first = samples(
        [[0.1, 0, 1], [0.3, 0, 1], [-0.1, 0, 1]],
        [[0, 0, 0], [100, 100, 100], [255, 0, 0]],
    )
    mapping.integrate(first, np.eye(4), 0)
    mapping.integrate(samples([[0.8, 0, 1]], [[150, 150, 150]]), np.eye(4), 1)
    cloud = mapping.cloud()
    np.testing.assert_array_equal(cloud.keys, [[-1, 0, 1], [0, 0, 1]])
    np.testing.assert_allclose(cloud.points[:, 0], [-0.1, 0.5])
    np.testing.assert_array_equal(cloud.frame_support, [1, 2])
    np.testing.assert_array_equal(cloud.colors, [[255, 0, 0], [100, 100, 100]])
    np.testing.assert_allclose(cloud.position_std_m, [0, 0.3], atol=1e-8)
    with pytest.raises(ValueError, match="only once"):
        mapping.integrate(first, np.eye(4), 0)


def test_uncolored_geometry_does_not_dilute_color_and_capacity_is_atomic():
    mapping = VoxelMap(1, max_voxels=1)
    mapping.integrate(samples([[0.1, 0, 1]], [[20, 40, 80]]), np.eye(4), 0)
    mapping.integrate(samples([[0.2, 0, 1]], colored=[False]), np.eye(4), 1)
    with pytest.raises(ValueError, match="capacity"):
        mapping.integrate(samples([[0.1, 0, 1], [2, 0, 1]]), np.eye(4), 2)
    cloud = mapping.cloud()
    np.testing.assert_array_equal(cloud.frame_support, [2])
    np.testing.assert_array_equal(cloud.color_support, [1])
    np.testing.assert_array_equal(cloud.colors, [[20, 40, 80]])
    mapping.integrate(samples([[0.1, 0, 1]]), np.eye(4), 2)
    assert mapping.cloud().frame_support[0] == 3


def test_nonrigid_pose_and_nonfinite_points_are_rejected():
    mapping = VoxelMap(1)
    T = np.eye(4)
    T[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        mapping.integrate(samples([[0, 0, 1]]), T, 0)
    with pytest.raises(ValueError, match="finite"):
        mapping.integrate(samples([[np.nan, 0, 1]]), np.eye(4), 0)


def test_native_filtering_preserves_inputs_and_metric_bounds():
    observation, calibration, config = plane()
    depth = observation.depth.copy()
    depth[12:15, 12:15] = np.nan
    depth[20:23, 20:23] = 5
    observation = replace(observation, depth=depth)
    original = depth.copy()
    result = sample_observation(observation, calibration, config)
    assert np.isfinite(result.points).all()
    assert (result.points[:, 2] == 2).all()
    assert np.isnan(result.depth[0]).all()
    assert np.isnan(result.depth[11:16, 11:16]).all()
    assert result.colored.all()
    np.testing.assert_array_equal(
        result.colors, np.tile([30, 100, 210], (len(result.points), 1))
    )
    np.testing.assert_array_equal(observation.depth, original)


def test_rgb_zbuffer_excludes_occluded_background_and_supports_different_grids():
    K_D = np.array([[100, 0, 2], [0, 100, 2], [0, 0, 1]], dtype=float)
    K_R = np.array([[0.1, 0, 0.5], [0, 0.1, 0.5], [0, 0, 1]], dtype=float)
    calibration = Calibration(K_R, K_D, pose(x=0.1), (2, 2), (5, 5))
    depth = np.full((5, 5), 2, dtype=np.float32)
    depth[1, 1] = 1
    observation = Observation(0, np.full((2, 2, 3), 127, dtype=np.uint8), depth)
    result = sample_observation(
        observation,
        calibration,
        MappingConfig(pixel_stride=1, depth_edge_threshold_m=2),
    )
    assert len(result.points) == 9
    assert result.colored.sum() < len(result.points)
    assert result.colored[np.argmin(result.points[:, 2])]
    assert (result.colors[~result.colored] == 0).all()


def test_camera_to_map_direction_on_analytic_rotated_plane():
    observation, calibration, config = plane(pose(x=0.1, z=0.04, yaw=0.12))
    result = sample_observation(observation, calibration, config)
    mapping = VoxelMap(config.voxel_size_m)
    mapping.integrate(result, observation.T_WC_D, 0)
    np.testing.assert_allclose(mapping.cloud().points[:, 2], 2, atol=2e-7)
    wrong = transform_points(result.points, np.linalg.inv(observation.T_WC_D))
    assert np.std(wrong[:, 2]) > 0.05


def test_geometric_consistency_exposes_pose_error_and_overlap():
    observation, calibration, config = plane()
    first = sample_observation(observation, calibration, config)
    moved, _, _ = plane(pose(x=0.1, yaw=0.12))
    second = sample_observation(moved, calibration, config)
    correct = depth_consistency(
        first, second, np.linalg.inv(moved.T_WC_D), calibration.K_D, 0.05
    )
    wrong = depth_consistency(first, second, moved.T_WC_D, calibration.K_D, 0.05)
    assert correct["absolute_depth_error_m"]["rmse"] < 0.004
    assert wrong["absolute_depth_error_m"]["rmse"] > 0.08
    assert 0 < correct["valid_target_fraction"] < 1
    shifted = depth_consistency(first, first, pose(z=0.1), calibration.K_D, 0.05)
    assert shifted["absolute_depth_error_m"]["rmse"] == pytest.approx(0.1)
    assert shifted["behind_surface_samples"] == shifted["valid_target_samples"]
    assert shifted["consistent_samples"] == 0
    empty = depth_consistency(first, first, pose(x=100), calibration.K_D, 0.05)
    assert empty["valid_target_samples"] == 0
    assert empty["absolute_depth_error_m"]["rmse"] is None
    summary = summarize_consistency([correct, empty])
    assert summary["absolute_depth_rmse_m"] == correct["absolute_depth_error_m"]["rmse"]
    assert summary["valid_target_fraction"] == correct["valid_target_fraction"] / 2


def test_capped_surface_distance_matches_brute_force_including_cell_boundaries():
    rng = np.random.default_rng(3)
    source, target = VoxelMap(0.2), VoxelMap(0.2)
    source.integrate(samples(rng.uniform(-1, 1, (80, 3))), np.eye(4), 0)
    target.integrate(samples(rng.uniform(-1, 1, (90, 3))), np.eye(4), 0)
    a, b = source.cloud(), target.cloud()
    expected = np.minimum(
        0.2, np.linalg.norm(a.points[:, None] - b.points[None], axis=-1).min(axis=1)
    )
    np.testing.assert_allclose(_nearby_distances(a, b, 0.2), expected, atol=1e-15)
    compared = compare_clouds(a, a, 0.2)
    assert compared["estimated_to_reference"]["within_cap_fraction"] == 1
    assert compared["reference_to_estimated"]["capped_distance_m"]["rmse"] == 0
    empty = VoxelMap(0.2).cloud()
    assert (
        compare_clouds(empty, b, 0.2)["estimated_to_reference"]["within_cap_fraction"]
        is None
    )
    assert (
        compare_clouds(a, empty, 0.2)["estimated_to_reference"]["within_cap_fraction"]
        == 0
    )


def test_keyframes_include_segment_endpoints_and_restart_the_stride():
    frames = [row(i) for i in range(5)] + [
        row(5, 1, status="lost"),
        row(6, 1),
        row(7, 1),
    ]
    assert select_keyframes(frames, 3) == [0, 3, 4, 5, 7]
    for index, bad in [
        (5, replace(frames[5], status="tracked")),
        (5, replace(frames[5], T_SC_D=pose(x=1))),
        (6, replace(frames[6], segment_id=0)),
        (3, replace(frames[3], frame_index=8)),
    ]:
        modified = list(frames)
        modified[index] = bad
        with pytest.raises(ValueError):
            select_keyframes(modified, 3)


def test_submaps_preserve_segment_transforms_and_reference_independence():
    observation, calibration, config = plane()
    frames = [
        row(0, reference=pose(x=10)),
        row(1, T=pose(x=0.1), reference=pose(x=10.1)),
        row(2, 1, reference=pose(x=100), status="lost"),
        row(3, 1, T=pose(x=0.2), reference=pose(x=100.2)),
    ]
    observations = [
        replace(observation, frame_index=i, timestamp=float(i)) for i in range(4)
    ]
    config = replace(config, keyframe_stride=1, max_keyframes_per_submap=1)
    result = list(reconstruct(observations, frames, calibration, config))
    assert [part.segment_id for part in result] == [0, 0, 1, 1]
    np.testing.assert_array_equal(result[1].T_SM, pose(x=0.1))
    np.testing.assert_array_equal(result[2].T_SM, np.eye(4))
    original = result[1].estimated
    changed = [
        replace(frame, reference_pose=pose(x=-200 + i * 2))
        for i, frame in enumerate(frames)
    ]
    altered = list(reconstruct(observations, changed, calibration, config))
    np.testing.assert_array_equal(original.points, altered[1].estimated.points)
    np.testing.assert_array_equal(
        original.frame_support, altered[1].estimated.frame_support
    )
    with pytest.raises(ValueError, match="disconnected"):
        build_submap(
            [result[0].keyframes[0], result[2].keyframes[0]],
            0,
            replace(config, max_keyframes_per_submap=2),
        )


def test_missing_and_invalid_references_never_remove_estimated_geometry():
    observation, calibration, config = plane()
    data = sample_observation(observation, calibration, config)
    invalid = np.eye(4)
    invalid[0, 0] = 2
    frames = [
        Keyframe(row(0), data),
        Keyframe(row(1, reference=pose(x=8)), data),
        Keyframe(row(2, reference=invalid), data),
    ]
    submap = build_submap(frames, 0, config)
    assert submap.report["reference_missing"] == 1
    assert submap.report["reference_invalid"] == 1
    assert submap.report["reference_anchor_frame"] == 1
    assert submap.report["comparison"] is None
    assert submap.estimated.frame_support.max() == 3
    assert submap.reference.frame_support.max() == 1


@pytest.mark.parametrize("count", [1, 3])
def test_observation_count_mismatch_is_rejected(count):
    observation, calibration, config = plane()
    observations = [
        replace(observation, frame_index=i, timestamp=float(i)) for i in range(count)
    ]
    with pytest.raises(ValueError, match="observations"):
        list(reconstruct(observations, [row(0), row(1)], calibration, config))


def test_pose_drift_splits_a_static_surface_without_reference_leakage():
    observation, calibration, config = plane()
    data = sample_observation(observation, calibration, config)
    stable = [
        Keyframe(row(0, reference=np.eye(4)), data),
        Keyframe(row(1, reference=np.eye(4)), data),
    ]
    consistent = build_submap(stable, 0, config)
    drifted = build_submap(
        [stable[0], Keyframe(replace(stable[1].row, T_SC_D=pose(z=0.12)), data)],
        0,
        config,
    )
    assert len(drifted.estimated.points) == 2 * len(consistent.estimated.points)
    assert consistent.estimated.frame_support.min() == 2
    assert drifted.estimated.frame_support.max() == 1
    assert (
        drifted.report["comparison"]["estimated_to_reference"]["within_cap_fraction"]
        == 0.5
    )
    np.testing.assert_array_equal(drifted.reference.points, consistent.reference.points)
    changed_references = build_submap(
        [
            stable[0],
            Keyframe(replace(stable[1].row, reference_pose=pose(z=0.12)), data),
        ],
        0,
        config,
    )
    np.testing.assert_array_equal(
        changed_references.estimated.points, consistent.estimated.points
    )
    assert len(changed_references.reference.points) == 2 * len(
        consistent.reference.points
    )
