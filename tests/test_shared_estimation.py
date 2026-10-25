"""Shared likelihood, elimination, publication and reversible-history contracts."""

from dataclasses import replace

import numpy as np
import pytest
from test_observation_welding import bridge
from test_online_slam import engine_scene

from scene_recall.mapping.voxel import MappingConfig, VoxelMap
from scene_recall.multiview.batch_ba import compact_blocks, compact_linearize
from scene_recall.multiview.fixed_landmarks import direct_cost
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.slam.graph import linear_system, solve_window
from scene_recall.slam.online import OnlineConfig
from scene_recall.slam.pose_graph import PoseGraphConfig
from scene_recall.slam.shared_estimation import SharedConfig, SharedEstimator
from scene_recall.slam.shared_solver import SchurOperator, solve_shared
from scene_recall.slam.system import SLAMConfig


def problem():
    graph, ledger, features, transform = bridge()
    shared = SharedEstimator(ledger.calibration)
    shared.views = ledger.views.copy()
    shared.identity_map = ledger.identity_map.copy()
    shared._next_identity = ledger._next_identity
    shared.propose(0, 4, features[0], features[4], transform, 0.75)
    graph.add_edge(0, 4, transform, kind="loop")
    graph.poses, graph.components = graph._initialize(list(graph.edges.values()))
    return graph, shared


def objective():
    graph, shared = problem()
    # Reuse the original packet assembler only to construct a controlled dense
    # reference objective. Both solvers receive exactly these observations.
    identities, _ = shared._identities({(0, 4)})
    members = {}
    for track, identity in identities.items():
        members.setdefault(identity, []).append(track)
    factors, _ = shared._factors(
        list(range(5)),
        graph,
        identities,
        {k: tuple(sorted(v)) for k, v in members.items()},
        {(0, 4)},
    )
    poses = np.array(
        [
            graph.poses[k] @ np.linalg.inv(shared.views[k].factors.T_RcD[0])
            for k in range(5)
        ]
    )
    return poses, factors, shared.calibration.K_R, shared.shared.solver


@pytest.mark.parametrize("fixed", [(0,), (0, 3), (0, 1, 2, 3, 4)])
@pytest.mark.parametrize("damping", [0.0, 0.003])
def test_implicit_schur_matches_complete_joint_normal_equations(fixed, damping):
    poses, factors, K, config = objective()
    ids, inverse, points = landmark_layout(factors)
    r, JP, JL = compact_linearize(poses, points, factors, K, inverse)
    _, weights = direct_cost(r, factors, config)
    blocks = compact_blocks(
        r, JP, JL, weights, inverse, factors.view, len(ids), len(poses)
    )
    operator = SchurOperator(blocks, factors.view, inverse, fixed, damping)
    _, H, g = linear_system(
        poses, points, ids, factors, K, config, list(range(5)), None
    )
    H += damping * np.diag(np.maximum(np.diag(H), 1e-15))
    free = np.array(
        [
            j
            for camera in range(5)
            if camera not in fixed
            for j in range(camera * 6, camera * 6 + 6)
        ]
        + list(range(30, len(g))),
        dtype=int,
    )
    reference = np.zeros_like(g)
    reference[free] = np.linalg.solve(H[np.ix_(free, free)], -g[free])
    dp, diagnostic = operator.solve(tolerance=1e-10)
    accumulated = np.zeros_like(points)
    np.add.at(
        accumulated, inverse, np.einsum("nji,nj->ni", blocks[4], dp[factors.view])
    )
    dl = -operator.b - np.einsum("nij,nj->ni", operator.Ci, accumulated)
    np.testing.assert_allclose(
        np.r_[dp.ravel(), dl.ravel()], reference, rtol=2e-6, atol=1e-8
    )
    assert diagnostic["relative_residual"] < 2e-10


def test_shared_nonlinear_fit_matches_dense_reference():
    poses, factors, K, config = objective()
    fitted, points, diagnostic = solve_shared(poses, factors, K, (0,), config)
    dense, dense_points, dense_diagnostic = solve_window(
        poses, factors, K, list(range(5)), 0, config=config
    )
    assert diagnostic["status"] == dense_diagnostic["status"] == "refined"
    np.testing.assert_allclose(fitted, dense, atol=2e-8)
    np.testing.assert_allclose(points, dense_points, atol=2e-8)
    np.testing.assert_array_equal(fitted[0], poses[0])
    assert all(
        a > b
        for a, b in zip(
            [diagnostic["initial_cost"], *diagnostic["accepted_costs"][:-1]],
            diagnostic["accepted_costs"],
            strict=True,
        )
    )


def test_full_history_replay_has_no_pose_summary_likelihood():
    graph, shared = problem()
    first = shared.refine(graph)[0]
    assert first["status"] == "refined"
    assert first["observations"] > 300
    assert first["pose_summary_factors"] == first["marginal_prior_factors"] == 0
    fingerprint = {k: v.factors.fingerprint() for k, v in shared.views.items()}
    # Alter a loop transform after initialization. Identity claims and native
    # evidence remain identical, so it cannot change the objective or solution.
    graph.edges[0, 4] = replace(graph.edges[0, 4], T_target_source=np.eye(4))
    shared.refine(graph)
    assert fingerprint == {k: v.factors.fingerprint() for k, v in shared.views.items()}
    assert shared.last_report["raw_rebuild"] is False
    assert shared.active_point_positions(graph, 1)


def test_revocation_rebuilds_raw_partition_and_disconnected_gauges():
    graph, shared = problem()
    shared.refine(graph)
    fingerprint = {k: v.factors.fingerprint() for k, v in shared.views.items()}
    graph.revoke(0, 4, "controlled removal")
    shared.revoke(0, 4)
    result = shared.refine(graph)[0]
    assert result["raw_rebuild"]
    assert len(set(graph.components.values())) == 2
    assert (
        shared.live_identities[shared.identity_map[0, 0]]
        != shared.live_identities[shared.identity_map[1, 0]]
    )
    # A fresh estimator with the same raw evidence must reconstruct the same
    # unjoined solution. No merged historical estimate survives in a prior.
    fresh = SharedEstimator(shared.calibration)
    fresh.views, fresh.identity_map = shared.views.copy(), shared.identity_map.copy()
    fresh.refine(graph)
    for key, point in fresh.estimates.items():
        np.testing.assert_allclose(point, shared.estimates[key], atol=1e-9)
    assert fingerprint == {k: v.factors.fingerprint() for k, v in shared.views.items()}


@pytest.mark.parametrize("incremental", [False, True])
def test_shared_stream_has_one_history_and_prefix_causality(monkeypatch, incremental):
    config = OnlineConfig(
        shared_estimation=True,
        local=SLAMConfig(optimize=False, exposure="static"),
        loops=False,
        shared=SharedConfig(
            linearization="selective" if incremental else "full",
            preconditioner="coarse" if incremental else "jacobi",
            admission="covisibility" if incremental else "balanced",
            retry_linear_failure=incremental,
        ),
    )
    engine, observations, _ = engine_scene(
        monkeypatch, count=17, config=config, mapping=MappingConfig(pixel_stride=20)
    )
    updates = [engine.process(o) for o in observations]
    assert engine.local.prior is None
    assert not engine.local.packets and not engine.local.poses
    assert len(engine.welding.views) == 4
    assert all(update.status == "tracked" for update in updates[1:])
    np.testing.assert_array_equal(engine.archive.poses[0], np.eye(4))
    for frame in engine.local.frontend.keyframes:
        if frame in engine.graph.poses:
            np.testing.assert_allclose(
                engine.archive.poses[frame],
                engine.local.frontend.keyframes[frame].T_SR @ engine.calibration.T_RD,
                atol=1e-10,
            )
    prefix, _, _ = engine_scene(monkeypatch, count=17, config=config)
    reference = np.eye(4)
    reference[:3, 3] = [1000, -2000, 500]
    for i, observation in enumerate(observations[:12]):
        update = prefix.process(replace(observation, T_WC_D=reference))
        np.testing.assert_array_equal(update.pose, updates[i].pose)
        assert update.version == updates[i].version
    reconstruction = engine.reconstruction
    for chunk in reconstruction.snapshot():
        grid = VoxelMap(
            reconstruction.config.voxel_size_m, reconstruction.config.max_voxels
        )
        for frame in chunk.frames:
            grid.integrate(reconstruction.samples[frame], engine.poses[frame], frame)
        np.testing.assert_array_equal(grid.cloud().points, chunk.cloud.points)


def test_degenerate_landmarks_reject_atomically():
    poses, factors, K, config = objective()
    # No depth and identical rays cannot constrain metric range.
    factors = replace(
        factors, native_points=np.full_like(factors.native_points, np.nan)
    )
    poses[:] = poses[0]
    fitted, _, diagnostic = solve_shared(poses, factors, K, (0,), config)
    assert diagnostic["status"] == "rejected"
    np.testing.assert_array_equal(fitted, poses)


def test_partial_observability_is_reported_without_a_retained_prior():
    poses, factors, K, config = objective()
    mask = (factors.track_id == factors.track_id[0]) & (factors.view < 2)
    packet = replace(
        factors,
        **{name: getattr(factors, name)[mask] for name in factors.__dataclass_fields__},
    )
    fitted, _, diagnostic = solve_shared(poses[:2], packet, K, (0,), config)
    assert diagnostic["status"] == "refined"
    assert diagnostic["rank_check"]["weak_camera_blocks"] == 1
    assert diagnostic["rank_check"]["condition"] is None
    assert diagnostic["final_cost"] < diagnostic["initial_cost"]
    strict, _, rejected = solve_shared(
        poses[:2], packet, K, (0,), config, require_full_rank=True
    )
    assert rejected["status"] == "rejected"
    np.testing.assert_array_equal(strict, poses[:2])
    np.testing.assert_array_equal(fitted[0], poses[0])


def test_same_component_identity_edit_preserves_camera_warm_start(monkeypatch):
    graph, shared = problem()
    shared.refine(graph)
    original = {key: pose.copy() for key, pose in graph.poses.items()}
    shared._partition = None

    def rejected(poses, factors, *args, **kwargs):
        return (
            poses.copy(),
            landmark_layout(factors)[2],
            {"status": "rejected", "reason": "controlled numerical failure"},
        )

    monkeypatch.setattr("scene_recall.slam.shared_estimation.solve_shared", rejected)
    result = shared.refine(graph)[0]
    assert result["status"] == "rejected"
    for key, pose in original.items():
        np.testing.assert_array_equal(graph.poses[key], pose)
    assert not shared.estimates


def test_bridge_initialization_preserves_each_solved_component_shape():
    graph, shared = problem()
    graph.poses, graph.components = graph._initialize(
        [edge for edge in graph.edges.values() if edge.kind == "odometry"]
    )
    old = {key: pose.copy() for key, pose in graph.poses.items()}
    initialized = shared._initialize_bridges(graph, {(0, 4)})
    np.testing.assert_array_equal(initialized[0], old[0])
    np.testing.assert_allclose(
        np.linalg.inv(initialized[3]) @ initialized[4],
        np.linalg.inv(old[3]) @ old[4],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.linalg.inv(initialized[4]) @ initialized[0],
        graph.edges[0, 4].T_target_source,
        atol=1e-12,
    )
    # Initialization does not publish before the complete shared transaction.
    np.testing.assert_array_equal(graph.poses[4], old[4])


def test_reconstruction_splits_after_raw_connectivity_revision(monkeypatch):
    config = OnlineConfig(
        shared_estimation=True,
        local=SLAMConfig(optimize=False, exposure="static"),
        loops=False,
    )
    engine, observations, _ = engine_scene(
        monkeypatch, count=11, config=config, mapping=MappingConfig(pixel_stride=20)
    )
    for observation in observations:
        engine.process(observation)
    reconstruction = engine.reconstruction
    components = [0] * 5 + [5] * 6
    reconstruction.repartition(components)
    reconstruction.synchronize(
        engine.poses, components, engine.version + 1, list(range(11))
    )
    assert {chunk.component for chunk in reconstruction.snapshot()} == {0, 5}
    assert sorted(
        frame for chunk in reconstruction.snapshot() for frame in chunk.frames
    ) == sorted(reconstruction.samples)


@pytest.mark.parametrize("fixed", [(-1,), (0, 0), (True,), (5,)])
def test_invalid_gauge_indices_are_rejected(fixed):
    poses, factors, K, config = objective()
    with pytest.raises(ValueError, match="fixed cameras"):
        solve_shared(poses, factors, K, fixed, config)


def test_configuration_prevents_overlapping_objectives():
    for settings in (
        {"observation_welding": True},
        {"local": SLAMConfig()},
        {"multiview_registration": True},
    ):
        with pytest.raises(ValueError, match="only camera-landmark"):
            OnlineConfig(shared_estimation=True, **settings)
    with pytest.raises(ValueError):
        SharedConfig(rank_check_cameras=True)


def test_capacity_stops_raw_admission_without_a_second_local_graph(monkeypatch):
    config = OnlineConfig(
        shared_estimation=True,
        graph=PoseGraphConfig(max_nodes=2),
        local=SLAMConfig(optimize=False, exposure="static"),
        loops=False,
    )
    engine, observations, _ = engine_scene(monkeypatch, count=16, config=config)
    updates = [engine.process(observation) for observation in observations]
    assert len(engine.graph.local) == len(engine.welding.views) == 2
    assert not engine.local.packets and engine.local.prior is None
    assert updates[-1].diagnostics["global_capacity_reached"]
    assert updates[-1].status == "tracked"


def test_shared_raw_runner_exports_a_loadable_root_gauge(monkeypatch, tmp_path):
    from test_local_map import scene

    from scene_recall.odometry.io import load_trajectory
    from scene_recall.slam.runner import run_online

    calibration, observations, features, _ = scene(17)

    class Reader:
        frame_count = len(observations)
        sequence_id = "controlled"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter(observations)

    reader = Reader()
    reader.calibration = calibration
    reader.metadata = {}
    monkeypatch.setattr(
        "scene_recall.slam.runner.TUMReader", lambda *args, **kwargs: reader
    )
    monkeypatch.setattr(
        "scene_recall.slam.recovery.RecoveryFrontend._extract",
        lambda self, observation, calibration: features[observation.frame_index],
    )
    config = OnlineConfig(
        shared_estimation=True,
        local=SLAMConfig(optimize=False, exposure="static"),
        loops=False,
    )
    output = tmp_path / "run"
    run_online(tmp_path / "source", output, dataset="tum_rgbd", config=config)
    report, rows = load_trajectory(output)
    np.testing.assert_array_equal(rows[0].T_SC_D, np.eye(4))
    assert report["experiment"]["causal_online"]
    with np.load(output / "shared_estimation.npz") as state:
        assert len(state["view"]) > 100
        assert len(state["camera_ids"]) == 4


def test_shared_bridge_and_revocation_keep_all_local_roots_at_identity(monkeypatch):
    from test_local_map import scene

    config = OnlineConfig(
        shared_estimation=True,
        local=SLAMConfig(optimize=False, exposure="static"),
        loops=False,
    )
    engine, observations, truth = engine_scene(
        monkeypatch,
        count=23,
        config=config,
        outage=2,
        mapping=MappingConfig(pixel_stride=20),
    )
    for observation in observations:
        engine.process(observation)
    _, _, features, _ = scene(23)
    transform = np.linalg.inv(truth[18]) @ truth[0]
    proposal = engine.welding.propose(0, 18, features[0], features[18], transform, 0.75)
    assert proposal["status"] == "admitted"
    engine.graph.add_edge(0, 18, transform, kind="loop")
    engine._solve()
    for root in engine.segment_roots.values():
        np.testing.assert_array_equal(engine.archive.poses[root], np.eye(4))
    engine.revoke(0, 18, "controlled raw bridge removal")
    for root in engine.segment_roots.values():
        np.testing.assert_array_equal(engine.archive.poses[root], np.eye(4))
    assert engine.reconstruction.snapshot()
