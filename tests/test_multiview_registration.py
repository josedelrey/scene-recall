"""Shared geometry, Schur elimination and disjoint incoming evidence contracts."""

from dataclasses import replace

import numpy as np

from scene_recall.data import Calibration
from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.pose_only import retract
from scene_recall.slam.multiview_registration import (
    conditional_bundle,
    register_multiview,
)
from scene_recall.slam.registration import RegistrationConfig


def scene():
    rng = np.random.default_rng(8)
    points = rng.uniform([-0.7, -0.5, 2.0], [0.7, 0.5, 4.0], (100, 3))
    K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
    context = retract(np.eye(4), np.array([-0.2, 0.01, 0, 0.01, 0, 0]))
    truth = retract(np.eye(4), np.array([0.04, -0.02, 0.01, 0, 0.02, 0]))
    descriptors = rng.normal(size=(100, 128)).astype(np.float32)
    features = [
        RGBDFeatures(
            project_points(transform_points(points, p), K),
            descriptors.copy(),
            transform_points(points, p),
        )
        for p in [np.eye(4), context, truth]
    ]
    calibration = Calibration(K, K, np.eye(4), (480, 640), (480, 640))
    return points, context, truth, features, calibration


def test_conditional_bundle_recovers_pose_and_shared_landmarks():
    points, context, truth, (_, ctx, target), calibration = scene()
    initial = retract(truth, np.array([0.005, -0.003, 0, 0.001, 0, 0]))
    pose, refined, diagnostic = conditional_bundle(
        points,
        ctx.pixels,
        context,
        calibration.K_R,
        initial=initial,
        incoming_pixels=target.pixels,
    )
    np.testing.assert_allclose(pose, truth, atol=1e-6)
    np.testing.assert_allclose(refined, points, atol=1e-6)
    assert diagnostic["final_cost"] < diagnostic["initial_cost"]


def test_registration_keeps_native_inputs_and_withheld_pixels_out_of_fit():
    _, context, truth, (first, ctx, target), calibration = scene()
    before = first.points_R.copy()
    baseline = register_multiview(
        first, target, [(7, ctx, context)], calibration, RegistrationConfig()
    )
    np.testing.assert_allclose(baseline.T_target_source, truth, atol=1e-6)
    changed = target.pixels.copy()
    changed[1::2] += 0.05
    result = register_multiview(
        first,
        replace(target, pixels=changed),
        [(7, ctx, context)],
        calibration,
        RegistrationConfig(),
    )
    np.testing.assert_array_equal(result.T_target_source, baseline.T_target_source)
    np.testing.assert_array_equal(first.points_R, before)
    assert result.diagnostics["source_depth_factors"] == 100
    assert result.diagnostics["context_depth_factors"] == 0
    assert result.diagnostics["incoming_depth_factors"] == 0
    assert result.diagnostics["fitting_track_indices"] == list(range(0, 100, 2))


def test_withheld_contradiction_abstains_without_inventing_depth():
    _, context, _, (first, ctx, target), calibration = scene()
    missing = replace(target, points_R=np.full_like(target.points_R, np.nan))
    accepted = register_multiview(
        first, missing, [(7, ctx, context)], calibration, RegistrationConfig()
    )
    assert accepted.T_target_source is not None
    assert accepted.diagnostics["validation_depth_checks"] == 0
    measured = target.points_R.copy()
    measured[1::2] += [0, 0, 1]
    rejected = register_multiview(
        first,
        replace(target, points_R=measured),
        [(7, ctx, context)],
        calibration,
        RegistrationConfig(),
    )
    assert rejected.T_target_source is None


def test_context_depth_does_not_enter_the_model_and_absence_is_explicit():
    _, context, _, (first, ctx, target), calibration = scene()
    config = RegistrationConfig()
    a = register_multiview(first, target, [(7, ctx, context)], calibration, config)
    ctx = replace(ctx, points_R=ctx.points_R * 1.2)
    b = register_multiview(first, target, [(7, ctx, context)], calibration, config)
    np.testing.assert_array_equal(a.T_target_source, b.T_target_source)
    fallback = register_multiview(first, target, [], calibration, config)
    assert fallback.diagnostics["multiview"]["status"] == "pair_fallback"


def test_scope_switch_preserves_recovery_baseline_when_only_loops_change():
    import pytest

    from scene_recall.slam.online import OnlineConfig, OnlineSLAM

    *_, calibration = scene()
    loops = OnlineSLAM(
        calibration, OnlineConfig(multiview_registration=True, multiview_scope="loops")
    )
    shared = OnlineSLAM(calibration, OnlineConfig(multiview_registration=True))
    assert not loops.local.frontend.multiview
    assert shared.local.frontend.multiview
    with pytest.raises(ValueError, match="one registration"):
        OnlineSLAM(
            calibration,
            OnlineConfig(multiview_registration=True, metric_refinement=True),
        )


def test_bundle_pose_jacobian_matches_native_projection_motion():
    from scene_recall.slam.metric_refinement import point_covariances
    from scene_recall.slam.multiview_registration import _weighted

    points, context, truth, (_, ctx, target), calibration = scene()
    K = calibration.K_R
    whitening = np.linalg.inv(np.linalg.cholesky(point_covariances(points, K)))
    *_, jacobian = _weighted(
        points, points, whitening, ctx.pixels, context, K, truth, target.pixels
    )
    for axis in range(6):
        step = np.eye(6)[axis] * 1e-6
        plus = _weighted(
            points,
            points,
            whitening,
            ctx.pixels,
            context,
            K,
            retract(truth, step),
            target.pixels,
        )[1]
        minus = _weighted(
            points,
            points,
            whitening,
            ctx.pixels,
            context,
            K,
            retract(truth, -step),
            target.pixels,
        )[1]
        np.testing.assert_allclose(
            (plus - minus) / 2e-6, jacobian[:, :, axis], atol=1e-6
        )


def test_context_subset_preserves_original_pair_split():
    _, context, _, (first, ctx, target), calibration = scene()
    keep = np.arange(3, len(ctx.pixels))
    # Dropping an odd number of context rows must not redefine test tracks as fit.
    ctx = replace(
        ctx,
        pixels=ctx.pixels[keep],
        descriptors=ctx.descriptors[keep],
        points_R=ctx.points_R[keep],
    )
    result = register_multiview(
        first, target, [(7, ctx, context)], calibration, RegistrationConfig()
    )
    assert result.T_target_source is not None
    tracks = np.array(result.diagnostics["track_rows"])
    assert (tracks[result.diagnostics["fitting_track_indices"], 0] % 2 == 0).all()
    assert (tracks[result.diagnostics["validation_track_indices"], 0] % 2 == 1).all()


def test_historical_revision_revokes_conditional_bridge_before_publication(monkeypatch):
    from helpers.mapping import pose
    from test_online_slam import engine_scene

    from scene_recall.slam.atlas import AtlasConfig
    from scene_recall.slam.online import OnlineConfig
    from scene_recall.slam.system import SLAMConfig

    config = OnlineConfig(
        local=SLAMConfig(keyframe_interval=1, optimize=False),
        atlas=AtlasConfig(keyframe_stride=1),
        loops=False,
        solve_interval=1000,
    )
    engine, observations, _ = engine_scene(
        monkeypatch, count=7, config=config, outage=2
    )
    for o in observations[:5]:
        engine.process(o)
    engine.graph.add_edge(0, 4, pose(-1), kind="loop", group=(0, 4))
    engine.conditioning[0, 4] = 1, engine._relative_context(0, 1).copy()
    engine._solve()
    engine._publish()
    assert engine.components[4] == 0
    # A common gauge change preserves the conditional relative geometry.
    saved = [engine.archive.poses[i].copy() for i in (0, 1)]
    for i in (0, 1):
        engine.archive.poses[i] = pose(0.2) @ engine.archive.poses[i]
    assert engine._invalidate_conditioning() == ([], False)
    for i, p in zip((0, 1), saved):
        engine.archive.poses[i] = p
    original = engine.local.process

    def revise_context(observation, calibration):
        p, pair, revisions, diagnostics = original(observation, calibration)
        return p, pair, {**revisions, 1: pose(0.1) @ saved[1]}, diagnostics

    monkeypatch.setattr(engine.local, "process", revise_context)
    engine.last_solve = 4
    update = engine.process(observations[5])
    assert (0, 4) not in engine.graph.edges
    assert engine.components[4] != 0
    assert update.diagnostics["graph"] is not None
    assert update.diagnostics["queries"][0]["status"] == "conditioning_invalidated"
    assert engine.conditioning == {}


def test_recovery_uses_three_views_with_nonidentity_sensor_extrinsics(monkeypatch):
    from test_local_map import scene as tracking_scene

    from scene_recall.slam.online import OnlineConfig, OnlineSLAM
    from scene_recall.slam.system import SLAMConfig

    calibration, observations, features, truth = tracking_scene(6, extrinsic=True)
    engine = OnlineSLAM(
        calibration,
        OnlineConfig(
            local=SLAMConfig(keyframe_interval=1, optimize=False),
            multiview_registration=True,
        ),
    )
    frontend = engine.local.frontend
    monkeypatch.setattr(frontend, "_extract", lambda o, _: features[o.frame_index])
    for observation in observations[:4]:
        engine.process(observation)
    pose, diagnostics = frontend._recover_pose(
        features[3], features[4], calibration, truth[3]
    )
    np.testing.assert_allclose(pose, truth[4], atol=1e-6)
    assert any(
        a.get("method") == "conditional_three_view_bundle"
        for a in diagnostics["attempts"]
    )
