"""Causal integrated behavior, gauge safety and reconstruction revision tests."""

from dataclasses import replace

import cv2
import numpy as np
import pytest
from helpers.mapping import plane, pose
from test_local_map import scene

from scene_recall.features import RGBDFeatures, extract_rgbd_features
from scene_recall.mapping.voxel import MappingConfig, VoxelMap
from scene_recall.slam.atlas import AtlasConfig
from scene_recall.slam.linear import BlockSystem
from scene_recall.slam.online import OnlineConfig, OnlineSLAM
from scene_recall.slam.pose_graph import MetricPoseGraph, PoseGraphConfig
from scene_recall.slam.reconstruction import RevisionReconstruction


def engine_scene(monkeypatch, count=20, *, config=None, outage=None, mapping=None):
    calibration, observations, features, truth = scene(count)
    if outage is not None:
        features[outage] = RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))
    engine = OnlineSLAM(calibration, config, mapping=mapping)
    monkeypatch.setattr(
        engine.local.frontend, "_extract", lambda o, _: features[o.frame_index]
    )
    return engine, observations, truth


def test_stream_is_prefix_causal_and_reference_independent(monkeypatch):
    config = OnlineConfig(atlas=AtlasConfig(keyframe_stride=2), same_segment_gap=6)
    first, observations, _ = engine_scene(monkeypatch, config=config)
    emitted = [first.process(observation) for observation in observations]
    second, _, _ = engine_scene(monkeypatch, config=config)
    for index, observation in enumerate(observations[:11]):
        update = second.process(replace(observation, T_WC_D=pose(1000 + index)))
        np.testing.assert_array_equal(update.pose, emitted[index].pose)
        assert update.component == emitted[index].component
        assert update.status == emitted[index].status
    assert len(second.poses) == 11
    with pytest.raises(ValueError, match="consecutive"):
        second.process(observations[12])


def test_feature_outage_preserves_unverified_disconnections(monkeypatch):
    engine, observations, _ = engine_scene(monkeypatch, count=5, outage=2)
    updates = [engine.process(observation) for observation in observations]
    assert updates[2].status == "lost"
    # A loss discarded the old active map, so this observation cannot invent a
    # direct transition back to it. Historical reconnection has separate gates.
    assert updates[3].status == "lost"
    assert updates[4].status == "tracked"
    assert updates[4].component != updates[0].component


def test_checked_anchor_fallback_handles_missing_incoming_depth(monkeypatch):
    engine, observations, truth = engine_scene(monkeypatch, count=3)
    frontend = engine.local.frontend
    engine.process(observations[0])
    engine.process(observations[1])
    _, _, features, _ = scene(3)
    missing = replace(features[2], points_R=np.full_like(features[2].points_R, np.nan))
    estimated, diagnostic = frontend._recover_pose(
        features[1], missing, engine.calibration, truth[1]
    )
    np.testing.assert_allclose(estimated, truth[2], atol=1e-7)
    assert diagnostic["depth_origin"] == "native_source_observation"


def test_recovery_can_skip_previous_features_and_reject_conflicting_anchors(
    monkeypatch,
):
    engine, observations, truth = engine_scene(monkeypatch, count=3)
    engine.process(observations[0])
    engine.process(observations[1])
    _, _, features, _ = scene(3)
    empty = RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))
    frontend = engine.local.frontend
    estimate, diagnostic = frontend._recover_pose(
        empty, features[2], engine.calibration, truth[1]
    )
    np.testing.assert_allclose(estimate, truth[2], atol=1e-7)
    assert diagnostic["source"] == 0
    frontend.keyframes[0] = replace(frontend.keyframes[0], T_SR=pose(1))
    estimate, diagnostic = frontend._recover_pose(
        features[1], features[2], engine.calibration, truth[1]
    )
    assert estimate is None
    assert diagnostic["status"] == "ambiguous_anchors"


def test_graph_capacity_keeps_new_segments_disconnected(monkeypatch):
    config = OnlineConfig(graph=PoseGraphConfig(max_nodes=1))
    engine, observations, _ = engine_scene(
        monkeypatch, count=5, config=config, outage=2
    )
    updates = [engine.process(observation) for observation in observations]
    assert all(update.diagnostics["global_capacity_reached"] for update in updates)
    assert len(engine.graph.local) == 1
    assert updates[4].component != updates[0].component
    assert updates[4].status == "tracked"


def test_local_revision_refreshes_attached_global_nodes(monkeypatch):
    config = OnlineConfig(atlas=AtlasConfig(keyframe_stride=2), loops=False)
    engine, observations, _ = engine_scene(monkeypatch, count=10, config=config)
    for observation in observations[:9]:
        engine.process(observation)
    original = engine.local.process

    def revised(observation, calibration):
        local, pair, revisions, diagnostic = original(observation, calibration)
        changed = pose(0.02) @ engine.archive.poses[5]
        return local, pair, {**revisions, 5: changed}, diagnostic

    monkeypatch.setattr(engine.local, "process", revised)
    engine.process(observations[9])
    for frame in (6, 8):
        np.testing.assert_array_equal(
            engine.graph.local[frame], engine.archive.poses[frame]
        )


def test_subthreshold_global_jitter_cannot_bypass_map_invalidation(monkeypatch):
    engine, observations, _ = engine_scene(monkeypatch, count=2)
    engine.process(observations[0])
    engine.process(observations[1])
    original = np.array(engine.poses)
    engine.graph.poses[0][0, 3] += 1e-14
    assert engine._publish() == []
    np.testing.assert_array_equal(engine.poses, original)


def test_global_bridge_revocation_splits_and_invalidates_received_history(monkeypatch):
    config = OnlineConfig(atlas=AtlasConfig(keyframe_stride=1), loops=False)
    engine, observations, _ = engine_scene(
        monkeypatch, count=5, config=config, outage=2
    )
    for observation in observations:
        engine.process(observation)
    engine.graph.add_edge(0, 4, pose(-1), kind="loop", group=(0, 4))
    engine._solve()
    engine._publish()
    assert engine.components[4] == 0
    result = engine.revoke(0, 4, "controlled false bridge")
    assert engine.components[4] != 0
    assert 4 in result["dirty_frames"]
    assert engine.confirmation.confirmed == {}
    assert engine.statuses[2:4] == ["lost", "lost"]


def test_reconstruction_rebuild_matches_fresh_raw_fusion_and_gauge_change():
    observation, calibration, config = plane()
    config = replace(config, keyframe_stride=1)
    reconstruction = RevisionReconstruction(calibration, config)
    for frame in range(2):
        reconstruction.admit(replace(observation, frame_index=frame), 0)
    original = [pose(), pose(0.02)]
    reconstruction.synchronize(original, [0, 0], 1, [0, 1])
    changed = [pose(0.1), pose(0.3)]
    reconstruction.synchronize(changed, [10, 10], 2, [0, 1])
    (chunk,) = reconstruction.snapshot()
    fresh = VoxelMap(config.voxel_size_m, config.max_voxels)
    for frame in range(2):
        fresh.integrate(reconstruction.samples[frame], changed[frame], frame)
    np.testing.assert_array_equal(chunk.cloud.points, fresh.cloud().points)
    np.testing.assert_array_equal(
        chunk.cloud.frame_support, fresh.cloud().frame_support
    )
    assert chunk.component == 10
    assert chunk.version == 2


def test_failed_reconstruction_never_exposes_stale_cloud_as_current():
    observation, calibration, config = plane()
    reconstruction = RevisionReconstruction(calibration, replace(config, max_voxels=1))
    reconstruction.admit(observation, 0)
    with pytest.raises(ValueError, match="capacity"):
        reconstruction.synchronize([pose()], [0], 1, [0])
    with pytest.raises(RuntimeError, match="synchronized"):
        reconstruction.snapshot()
    reconstruction.config = replace(config, max_voxels=100_000)
    reconstruction.synchronize([pose()], [0], 2, [])
    assert reconstruction.snapshot()[0].version == 2


def test_stream_mapping_and_trajectory_share_version(monkeypatch):
    engine, observations, _ = engine_scene(
        monkeypatch, count=8, mapping=MappingConfig(pixel_stride=16, keyframe_stride=2)
    )
    for observation in observations:
        update = engine.process(observation)
        assert engine.reconstruction.version == update.version
        assert all(
            chunk.version == update.version
            for chunk in engine.reconstruction.snapshot()
        )


@pytest.mark.parametrize("robust", [False, True])
def test_sparse_graph_matches_dense_solution_and_rejection(robust):
    graphs = []
    for solver in ("dense", "block_pcg"):
        graph = MetricPoseGraph(PoseGraphConfig(linear_solver=solver))
        for i in range(24):
            graph.add_node(i, 0, pose(i * 0.11, yaw=i * 0.001))
            if i:
                graph.add_edge(
                    i - 1, i, np.linalg.inv(graph.local[i]) @ graph.local[i - 1]
                )
        graph.add_edge(0, 23, pose(-2.3), kind="loop", group=(0, 23))
        graph.add_edge(1, 22, pose(8), kind="loop", group=(1, 22))
        diagnostic = graph.solve(robust=robust)
        graphs.append((graph, diagnostic))
    assert graphs[0][1]["rejected_loops"] == graphs[1][1]["rejected_loops"]
    np.testing.assert_allclose(
        list(graphs[0][0].poses.values()), list(graphs[1][0].poses.values()), atol=1e-6
    )


def test_block_solver_verifies_equations_and_refuses_insufficient_iterations():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(30, 12))
    H = A.T @ A
    system = BlockSystem(12)
    for i in range(2):
        for j in range(2):
            system.add(
                slice(i * 6, (i + 1) * 6),
                slice(j * 6, (j + 1) * 6),
                H[i * 6 : (i + 1) * 6, j * 6 : (j + 1) * 6],
            )
    b = rng.normal(size=12)
    expected = np.linalg.solve(H + 0.01 * np.diag(np.diag(H)), b)
    np.testing.assert_allclose(system.solve(b, 0.01), expected, atol=1e-8)
    with pytest.raises(np.linalg.LinAlgError):
        system.solve(b, 0.01, max_iterations=1)


def test_contrast_changes_detector_input_without_changing_native_geometry():
    observation, calibration, _ = plane()
    rng = np.random.default_rng(2)
    observation = replace(
        observation, rgb=rng.integers(0, 40, observation.rgb.shape, dtype=np.uint8)
    )
    original_rgb, original_depth = observation.rgb.copy(), observation.depth.copy()

    class Detector:
        def detectAndCompute(self, gray, mask):
            self.gray = gray.copy()
            return [cv2.KeyPoint(10, 10, 1), cv2.KeyPoint(20, 20, 1)], np.ones(
                (2, 128), np.float32
            )

    detector = Detector()
    raw = extract_rgbd_features(observation, calibration, detector)
    gray = detector.gray
    normalized = extract_rgbd_features(
        observation, calibration, detector, contrast="clahe"
    )
    assert not np.array_equal(gray, detector.gray)
    np.testing.assert_array_equal(raw.points_R, normalized.points_R)
    np.testing.assert_array_equal(observation.rgb, original_rgb)
    np.testing.assert_array_equal(observation.depth, original_depth)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shortlist": 1},
        {"solve_interval": 0},
        {"recovery": 1},
        {"retrieval": "unknown"},
    ],
)
def test_online_config_rejects_invalid_values(kwargs):
    with pytest.raises((ValueError, TypeError)):
        OnlineConfig(**kwargs)
