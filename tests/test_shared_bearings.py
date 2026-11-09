"""Pending-ray ownership, initialization and complete causal publication."""

from dataclasses import replace

import numpy as np
import pytest
from test_observation_welding import bridge
from test_online_slam import engine_scene

from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.mapping.voxel import MappingConfig, VoxelMap
from scene_recall.multiview.fixed_landmarks import DirectFactors
from scene_recall.slam.bearings import (
    BearingConfig,
    RawObservations,
    triangulate_bearings,
)
from scene_recall.slam.observation_welding import WeldingConfig
from scene_recall.slam.online import OnlineConfig
from scene_recall.slam.pose_graph import MetricPoseGraph
from scene_recall.slam.registration import RegistrationConfig
from scene_recall.slam.shared_estimation import SharedConfig, SharedEstimator


def pending_scene(mode="multiview", *, baseline=0.12):
    _, old, _, _ = bridge()
    calibration = old.calibration
    rng = np.random.default_rng(19)
    points = rng.uniform([-0.65, -0.45, 1.8], [0.65, 0.45, 2.8], (64, 3))
    descriptors = rng.normal(size=(64, 128)).astype(np.float32)
    shared = SharedEstimator(
        calibration,
        WeldingConfig(max_observations_per_view=16),
        SharedConfig(bearings=BearingConfig(mode=mode)),
    )
    graph, features, measurements = MetricPoseGraph(), [], []
    for frame in range(3):
        pose = np.eye(4)
        pose[0, 3] = frame * baseline
        local = transform_points(points, np.linalg.inv(pose))
        pixels = project_points(local, calibration.K_R)
        native = local.copy()
        native[32:] = np.nan
        if frame == 2:
            native[32:48] = local[32:48]
        seeds = points.copy()
        seeds[32:] = np.nan
        owners = np.arange(64)
        owners[32:] = -1
        indices = np.arange(64)
        indices[~np.isfinite(native).all(axis=1)] = -1
        measurement = {
            "view": np.full(64, frame),
            "track_id": np.arange(64),
            "points": seeds,
            "pixels": pixels,
            "native_points": native,
            "T_RcD": np.repeat(np.eye(4)[None], 64, axis=0),
            "whitening_D": np.repeat((np.eye(3) * 50)[None], 64, axis=0),
            "weights": np.ones(64),
            "feature_index": np.arange(64),
            "native_index": indices,
            "frontend_identity": owners,
        }
        features.append(RGBDFeatures(pixels, descriptors.copy(), local))
        measurements.append(measurement)
        graph.add_node(frame, 0, pose)
        if frame:
            graph.add_edge(
                frame - 1, frame, np.linalg.inv(pose) @ graph.local[frame - 1]
            )
        shared.admit(frame, 0, measurement)
        shared.record_bearing_descriptors(frame, features[-1])
    graph.poses, graph.components = graph._initialize(list(graph.edges.values()))
    return graph, shared, features, measurements, points


def test_pending_capture_preserves_unknown_range_and_unique_owners():
    graph, shared, _, measurements, _ = pending_scene("retain")
    packet = shared.views[0].factors
    assert isinstance(packet, RawObservations)
    assert np.isnan(packet.points[packet.track_id >= 32]).any()
    with pytest.raises(ValueError, match="nonfinite"):
        DirectFactors(
            **{name: getattr(packet, name) for name in packet.__dataclass_fields__}
        )
    for view in shared.views.values():
        assert len(view.feature_index) == len(np.unique(view.feature_index)) == 64
        assert not view.factors.pixels.flags.writeable
        assert not view.factors.native_points.flags.writeable
    before = shared.views[0].factors.fingerprint()
    measurements[0]["pixels"][:] = 900
    assert shared.views[0].factors.fingerprint() == before
    with pytest.raises(ValueError, match="already admitted"):
        shared.admit(0, 0, measurements[0])
    report = shared.refine(graph)[0]
    assert report["landmarks"] == 32
    assert report["bearings"]["triangulated_classes"] == 0
    assert not shared.bearing_initializations
    assert shared.export_arrays(graph)["bearing_descriptors"].shape == (192, 128)


def test_later_native_depth_activates_historical_pending_rays_reversibly():
    graph, shared, features, _, _ = pending_scene("anchor")
    fingerprints = {k: v.factors.fingerprint() for k, v in shared.views.items()}
    result = shared.link_bearings(0, 2, features[0], features[2], RegistrationConfig())
    assert result["status"] == "admitted"
    report = shared.refine(graph)[0]
    assert report["status"] in ("refined", "unchanged")
    assert report["landmarks"] > 32
    assert report["pose_summary_factors"] == report["marginal_prior_factors"] == 0
    assert shared.active_point_positions(graph, 0)
    assert all(owner >= 0 for owner in shared.active_point_positions(graph, 0))
    shared.revoke(0, 2)
    report = shared.refine(graph)[0]
    assert report["raw_rebuild"]
    assert report["landmarks"] == 32
    assert fingerprints == {k: v.factors.fingerprint() for k, v in shared.views.items()}


def test_pure_bearings_triangulate_only_after_informative_multiview_support():
    graph, shared, features, _, truth = pending_scene()
    for a, b in [(0, 1), (1, 2)]:
        assert (
            shared.link_bearings(a, b, features[a], features[b], RegistrationConfig())[
                "status"
            ]
            == "admitted"
        )
    report = shared.refine(graph)[0]
    assert report["status"] in ("refined", "unchanged")
    assert report["bearings"]["triangulated_classes"] >= 4
    exported = shared.export_arrays(graph)
    assert len(exported["bearing_seed_ids"]) >= 4
    # No pure RGB class acquires a synthetic native measurement during activation.
    for track in shared.bearing_initializations:
        raw = [key for key, value in shared.live_identities.items() if value == track]
        for view in shared.views.values():
            mask = np.isin(view.factors.track_id, raw)
            assert np.isnan(view.factors.native_points[mask]).all()
        assert np.min(np.linalg.norm(truth - shared.estimates[track], axis=1)) < 1e-7


@pytest.mark.parametrize("baseline", [0, 1e-5])
def test_weak_parallax_stays_pending_without_fabricated_metric_information(baseline):
    graph, shared, features, _, _ = pending_scene(baseline=baseline)
    for a, b in [(0, 1), (1, 2)]:
        shared.link_bearings(a, b, features[a], features[b], RegistrationConfig())
    report = shared.refine(graph)[0]
    assert report["bearings"]["triangulated_classes"] == 0
    assert not shared.bearing_initializations


def test_triangulation_is_world_chart_equivariant_and_rejects_bad_pixels():
    graph, shared, features, _, _ = pending_scene()
    poses = np.array([graph.poses[i] for i in range(3)])
    pixels = np.array([f.pixels[55] for f in features])
    point, reason = triangulate_bearings(poses, pixels, shared.calibration.K_R)
    assert reason == "initialized"
    world = np.eye(4)
    world[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    world[:3, 3] = [30, -12, 4]
    revised, _ = triangulate_bearings(world @ poses, pixels, shared.calibration.K_R)
    np.testing.assert_allclose(revised, transform_points(point, world), atol=1e-10)
    pixels[1] += [0, 30]
    point, reason = triangulate_bearings(poses, pixels, shared.calibration.K_R)
    assert point is None and reason == "inconsistent image observations"


def test_reverse_native_anchor_keeps_validation_chart_and_causal_order():
    graph, shared, features, _, _ = pending_scene("anchor")
    source = shared.views[0]
    arrays = {
        name: getattr(source.factors, name).copy()
        for name in source.factors.__dataclass_fields__
    }
    arrays["native_points"][:] = np.nan
    shared.views[0] = replace(
        source, factors=RawObservations(**arrays), native_index=np.full(64, -1)
    )
    result = shared.link_bearings(0, 2, features[0], features[2], RegistrationConfig())
    assert result["status"] == "admitted" and result["anchor"] == 2
    report = shared.refine(graph)[0]
    assert report["validation"][0]["after"]["inlier_fraction"] == 1
    with pytest.raises(ValueError, match="earlier"):
        shared.link_bearings(2, 0, features[2], features[0], RegistrationConfig())


def test_bearing_bridges_cannot_skip_cross_segment_confirmation():
    _, shared, features, _, _ = pending_scene("anchor")
    shared.views[2] = replace(shared.views[2], segment=1)
    with pytest.raises(ValueError, match="same-segment"):
        shared.link_bearings(0, 2, features[0], features[2], RegistrationConfig())
    assert not shared.bearing_links


@pytest.mark.parametrize("mode", ["retain", "anchor", "multiview"])
def test_full_bearing_stream_preserves_prefix_publication_and_exact_refusion(
    monkeypatch, mode
):
    config = OnlineConfig(
        shared_estimation=True,
        local=replace(OnlineConfig().local, exposure="static"),
        shared=SharedConfig(bearings=BearingConfig(mode=mode)),
        loops=False,
    )
    engine, observations, _ = engine_scene(
        monkeypatch, count=17, config=config, mapping=MappingConfig(pixel_stride=20)
    )
    updates = [engine.process(o) for o in observations]
    assert len(engine.welding.views) == 4
    assert engine.local.prior is None
    assert all(u.status == "tracked" for u in updates[1:])
    prefix, _, _ = engine_scene(monkeypatch, count=17, config=config)
    altered_reference = np.eye(4)
    altered_reference[:3, 3] = [999, -500, 100]
    for i, o in enumerate(observations[:12]):
        update = prefix.process(replace(o, T_WC_D=altered_reference))
        np.testing.assert_array_equal(update.pose, updates[i].pose)
        assert (
            update.component == updates[i].component
            and update.version == updates[i].version
        )
    for chunk in engine.reconstruction.snapshot():
        grid = VoxelMap(
            engine.reconstruction.config.voxel_size_m,
            engine.reconstruction.config.max_voxels,
        )
        for frame in chunk.frames:
            grid.integrate(
                engine.reconstruction.samples[frame], engine.poses[frame], frame
            )
        np.testing.assert_array_equal(grid.cloud().points, chunk.cloud.points)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "invalid"},
        {"extra_per_view": True},
        {"recent_cameras": 0},
        {"min_views": 2},
        {"min_parallax_deg": 0},
        {"min_parallax_deg": 90},
    ],
)
def test_bearing_configuration_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        BearingConfig(**kwargs)


def test_pending_rays_do_not_join_independent_metric_gauges():
    graph, shared, _, _, _ = pending_scene()
    for frame, view in list(shared.views.items()):
        arrays = {
            name: getattr(view.factors, name).copy()
            for name in view.factors.__dataclass_fields__
        }
        arrays["native_points"][:] = np.nan
        shared.views[frame] = replace(
            view, factors=RawObservations(**arrays), native_index=np.full(64, -1)
        )
    # Even externally supplied identity claims cannot initialize metric range
    # using the arbitrary odometry placement between unmeasured components.
    for a, b in [(0, 1), (1, 2)]:
        shared.associations[a, b] = tuple(
            zip(
                shared.views[a].factors.track_id,
                shared.views[b].factors.track_id,
                strict=True,
            )
        )
        shared.reobservations[a, b] = np.eye(4)
        from scene_recall.slam.observation_welding import LoopValidation

        shared.validation[a, b] = LoopValidation(
            np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2))
        )
    report = shared.refine(graph)[0]
    assert report["observations"] == 0
    assert report["bearings"]["triangulated_classes"] == 0
    assert report["bearings"]["deferred"]["independent metric gauges"] > 0
    assert len(set(shared.raw_components.values())) == 3


def test_bearing_solver_matches_complete_dense_joint_objective():
    from scene_recall.multiview.batch_ba import BatchConfig
    from scene_recall.slam.graph import solve_window
    from scene_recall.slam.shared_solver import solve_shared

    graph, shared, features, _, _ = pending_scene()
    for a, b in [(0, 1), (1, 2)]:
        shared.link_bearings(a, b, features[a], features[b], RegistrationConfig())
    shared.refine(graph)
    data = shared.export_arrays(graph)
    live = dict(data["live_identities"])
    selected = np.array(
        [live[int(track)] in shared.estimates for track in data["track_id"]]
    )
    arrays = {name: data[name][selected] for name in DirectFactors.__dataclass_fields__}
    arrays["track_id"] = np.array([live[int(track)] for track in arrays["track_id"]])
    arrays["points"] = np.array(
        [shared.estimates[int(track)] for track in arrays["track_id"]]
    )
    factors = DirectFactors(**arrays)
    poses = data["depth_poses"].copy()
    poses[1, 1, 3] += 0.003
    poses[2, 0, 3] -= 0.004
    config = BatchConfig(max_iterations=8)
    sparse, points, report = solve_shared(
        poses, factors, shared.calibration.K_R, (0,), config
    )
    dense, dense_points, reference = solve_window(
        poses, factors, shared.calibration.K_R, list(range(3)), 0, config=config
    )
    assert report["status"] == reference["status"] == "refined"
    np.testing.assert_allclose(sparse, dense, atol=3e-8)
    np.testing.assert_allclose(points, dense_points, atol=3e-8)


def test_bearing_registration_ignores_frontend_metric_seed_poisoning():
    _, shared, features, _, _ = pending_scene("anchor")
    _, other, _, _, _ = pending_scene("anchor")
    poisoned = [
        replace(f, points_R=f.points_R * 10 + [200, 100, 500]) for f in features
    ]
    shared.link_bearings(0, 2, features[0], features[2], RegistrationConfig())
    other.link_bearings(0, 2, poisoned[0], poisoned[2], RegistrationConfig())
    assert shared.associations == other.associations
    np.testing.assert_array_equal(
        shared.reobservations[0, 2], other.reobservations[0, 2]
    )


def test_uninitialized_initial_packet_retains_explicit_output_attachment():
    graph, old, _, measurements, _ = pending_scene("retain")
    shared = SharedEstimator(old.calibration, old.config, old.shared)
    for frame, measurement in enumerate(measurements):
        measurement = {name: value.copy() for name, value in measurement.items()}
        if frame == 0:
            measurement["native_points"][:] = np.nan
            measurement["native_index"][:] = -1
            measurement["frontend_identity"][:] = -1
            measurement["points"][:] = np.nan
        shared.admit(frame, 0, measurement)
    shared.refine(graph)
    assert shared.raw_components[0] != shared.raw_components[1]
    assert graph.components[0] == graph.components[1] == graph.components[2] == 0
    assert 0 in shared.last_report["fixed_cameras"]
    assert len(shared.views[0].factors.view) > 0


@pytest.mark.parametrize("invalid", ["pixel", "owner", "native"])
def test_invalid_pending_packet_is_rejected_before_mutating_ownership(invalid):
    _, old, _, measurements, _ = pending_scene("retain")
    shared = SharedEstimator(old.calibration, old.config, old.shared)
    bad = {name: value.copy() for name, value in measurements[0].items()}
    if invalid == "pixel":
        bad["pixels"][0, 0] = np.nan
        reason = "nonfinite"
    elif invalid == "owner":
        bad["frontend_identity"][0] = bad["frontend_identity"][1]
        reason = "multiple features"
    else:
        bad["native_index"][0] = -1
        reason = "does not match"
    with pytest.raises(ValueError, match=reason):
        shared.admit(0, 0, bad)
    assert not shared.views and not shared.identity_map


def test_bearing_link_rejects_pixels_from_a_different_raw_packet():
    _, shared, features, _, _ = pending_scene("anchor")
    wrong = replace(features[2], pixels=features[2].pixels + 1)
    with pytest.raises(ValueError, match="owned observation"):
        shared.link_bearings(0, 2, features[0], wrong, RegistrationConfig())
    assert not shared.associations
