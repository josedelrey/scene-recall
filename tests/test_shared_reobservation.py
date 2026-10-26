"""Native reobservation joins identities without pose-summary likelihoods."""

from dataclasses import replace

import numpy as np
import pytest
from test_observation_welding import bridge
from test_online_slam import engine_scene

from scene_recall.mapping.voxel import MappingConfig, VoxelMap
from scene_recall.slam.online import OnlineConfig
from scene_recall.slam.registration import RegistrationConfig
from scene_recall.slam.shared_estimation import SharedConfig, SharedEstimator


def fragmented():
    graph, ledger, features, _ = bridge()
    shared = SharedEstimator(ledger.calibration)
    shared.views = {key: replace(view, segment=0) for key, view in ledger.views.items()}
    shared.identity_map = {
        (0, owner + segment * 10000): track
        for (segment, owner), track in ledger.identity_map.items()
    }
    shared._next_identity = ledger._next_identity
    graph.segments = {key: 0 for key in graph.segments}
    graph.poses, graph.components = graph._initialize(list(graph.edges.values()))
    return graph, shared, features


def test_recent_identity_bridge_is_reversible_without_a_pose_edge():
    graph, shared, features = fragmented()
    fingerprints = {k: v.factors.fingerprint() for k, v in shared.views.items()}
    edges = graph.edges.copy()
    proposal = shared.reobserve(0, 4, features[0], features[4], RegistrationConfig())
    assert proposal["status"] == "admitted"
    assert proposal["verification"]["validation_pairs"] >= 6
    assert graph.edges == edges
    report = shared.refine(graph)[0]
    assert report["pose_summary_factors"] == 0
    assert report["marginal_prior_factors"] == 0
    assert len(set(shared.raw_components.values())) == 1
    assert report["recent_reobservations"] == [[0, 4]]
    arrays = shared.export_arrays(graph)
    np.testing.assert_array_equal(arrays["reobservation_edges"], [[0, 4]])
    shared.revoke(0, 4)
    shared.refine(graph)
    assert len(set(shared.raw_components.values())) == 2
    assert graph.edges == edges
    assert fingerprints == {k: v.factors.fingerprint() for k, v in shared.views.items()}


def test_registration_uses_owned_native_exposures_and_ignores_frontend_point_seeds():
    _, shared, features = fragmented()
    poison = {
        key: replace(f, points_R=f.points_R * 10 + 1000)
        for key, f in enumerate(features)
    }
    proposal = shared.reobserve(0, 4, poison[0], poison[4], RegistrationConfig())
    assert proposal["status"] == "admitted"
    _, other, _ = fragmented()
    other.reobserve(0, 4, features[0], features[4], RegistrationConfig())
    np.testing.assert_array_equal(
        shared.reobservations[0, 4], other.reobservations[0, 4]
    )
    assert shared.associations == other.associations


def test_recent_registration_cannot_replace_cross_segment_confirmation():
    _, shared, features = fragmented()
    shared.views[4] = replace(shared.views[4], segment=1)
    with pytest.raises(ValueError, match="confirmation"):
        shared.reobserve(0, 4, features[0], features[4], RegistrationConfig())
    assert not shared.reobservations


def test_recent_registration_requires_causal_camera_order():
    _, shared, features = fragmented()
    with pytest.raises(ValueError, match="earlier source"):
        shared.reobserve(4, 0, features[4], features[0], RegistrationConfig())
    assert not shared.reobservations


def test_online_revocation_preserves_odometry_and_exact_refusion(monkeypatch):
    config = OnlineConfig(
        shared_estimation=True, shared=SharedConfig(reobserve_recent=3)
    )
    config = replace(config, local=replace(config.local, exposure="static"))
    engine, observations, _ = engine_scene(
        monkeypatch, count=12, config=config, mapping=MappingConfig(pixel_stride=16)
    )
    frontend = engine.local.frontend
    original = frontend._insert

    def fragment(index, *args, **kwargs):
        original(index, *args, **kwargs)
        if index == 5:
            frontend.landmarks = {i + 10000: p for i, p in frontend.landmarks.items()}
            for owners in frontend.owners.values():
                owners[owners >= 0] += 10000
            frontend._rebuild_index()

    monkeypatch.setattr(frontend, "_insert", fragment)
    for observation in observations:
        engine.process(observation)
    assert engine.welding.reobservations
    edge = next(iter(engine.welding.reobservations))
    odometry = engine.graph.edges[edge]
    assert odometry.kind == "odometry"
    with pytest.raises(ValueError, match="reason"):
        engine.revoke(*edge, "")
    engine.revoke(*edge, "controlled identity proposal removal")
    assert engine.graph.edges[edge] is odometry
    assert edge not in engine.welding.reobservations
    reconstruction = engine.reconstruction
    for chunk in reconstruction.snapshot():
        grid = VoxelMap(
            reconstruction.config.voxel_size_m, reconstruction.config.max_voxels
        )
        for frame in chunk.frames:
            grid.integrate(reconstruction.samples[frame], engine.poses[frame], frame)
        np.testing.assert_array_equal(grid.cloud().points, chunk.cloud.points)


@pytest.mark.parametrize("value", [-1, 9, True, 1.5])
def test_bounded_recent_reobservation_budget(value):
    with pytest.raises(ValueError):
        SharedConfig(reobserve_recent=value)
