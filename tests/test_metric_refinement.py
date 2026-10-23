"""Measurement symmetry, robust fitting and withheld registration contracts."""

from dataclasses import replace

import cv2
import numpy as np

from scene_recall.data import Calibration
from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.slam.metric_refinement import refine_metric_pose
from scene_recall.slam.registration import RegistrationConfig, register_anchor_pairs


def population():
    rng = np.random.default_rng(32)
    source = rng.uniform([-1, -0.7, 1.5], [1, 0.7, 3.5], (180, 3))
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.array([0.01, -0.03, 0.02]))[0]
    pose[:3, 3] = [0.03, -0.02, 0.01]
    return source, K, pose


def test_recovers_rigid_motion_with_outliers_and_preserves_inputs():
    source, K, truth = population()
    target = transform_points(source, truth)
    target[::12] += [0.2, -0.1, 0.3]
    before = source.copy(), target.copy()
    estimated, diagnostic = refine_metric_pose(source, target, K, np.eye(4))
    np.testing.assert_allclose(estimated, truth, atol=0.003)
    np.testing.assert_array_equal(source, before[0])
    np.testing.assert_array_equal(target, before[1])
    assert diagnostic["status"] == "refined"


def test_noiseless_alignment_is_symmetric_under_observation_exchange():
    source, K, truth = population()
    target = transform_points(source, truth)
    forward, _ = refine_metric_pose(source, target, K, np.eye(4))
    reverse, _ = refine_metric_pose(target, source, K, np.eye(4))
    np.testing.assert_allclose(forward, truth, atol=1e-7)
    np.testing.assert_allclose(reverse @ forward, np.eye(4), atol=1e-7)


def test_repeated_geometry_abstains_and_missing_depth_retains_pnp():
    points = np.repeat([[0.0, 0.0, 2.0]], 20, axis=0)
    initial = np.eye(4)
    pose, diagnostic = refine_metric_pose(points, points, np.eye(3), initial)
    np.testing.assert_array_equal(pose, initial)
    assert diagnostic["status"] == "degenerate"
    source, K, truth = population()
    target = transform_points(source, truth)
    first = RGBDFeatures(project_points(source, K), None, source)
    second = RGBDFeatures(project_points(target, K), None, np.full_like(target, np.nan))
    calibration = Calibration(K, K, np.eye(4), (240, 320), (240, 320))
    pairs = np.repeat(np.arange(len(source))[:, None], 2, axis=1)
    result = register_anchor_pairs(
        first, second, pairs, calibration, RegistrationConfig(metric_refinement=True)
    )
    np.testing.assert_allclose(result.T_target_source, truth, atol=1e-6)
    assert result.diagnostics["metric_refinement"]["status"] == "insufficient_depth"


def test_withheld_depth_never_changes_fitted_pose():
    source, K, truth = population()
    target = transform_points(source, truth)
    first = RGBDFeatures(project_points(source, K), None, source)
    second = RGBDFeatures(project_points(target, K), None, target)
    calibration = Calibration(K, K, np.eye(4), (240, 320), (240, 320))
    pairs = np.repeat(np.arange(len(source))[:, None], 2, axis=1)
    config = RegistrationConfig(metric_refinement=True)
    baseline = register_anchor_pairs(first, second, pairs, calibration, config)
    changed = target.copy()
    changed[1::4] += [0.001, 0.001, 0.001]
    tested = register_anchor_pairs(
        first, replace(second, points_R=changed), pairs, calibration, config
    )
    np.testing.assert_array_equal(tested.T_target_source, baseline.T_target_source)
    changed[1::2] += [0, 0, 1]
    rejected = register_anchor_pairs(
        first, replace(second, points_R=changed), pairs, calibration, config
    )
    assert rejected.T_target_source is None
    assert rejected.diagnostics["reason"] == "withheld anchor disagreement"


def test_online_switch_reaches_recovery_and_loop_registration():
    from scene_recall.slam.online import OnlineConfig, OnlineSLAM

    _, K, _ = population()
    calibration = Calibration(K, K, np.eye(4), (240, 320), (240, 320))
    control = OnlineSLAM(calibration)
    changed = OnlineSLAM(calibration, OnlineConfig(metric_refinement=True))
    assert not control.registration.metric_refinement
    assert changed.registration.metric_refinement
    assert changed.local.frontend.registration is changed.registration


def test_metric_flow_keeps_native_measurements():
    from test_flow_recovery import translated_pair

    from scene_recall.slam.flow import flow_registration

    calibration, (source, target) = translated_pair()
    result = flow_registration(
        source,
        target,
        calibration,
        registration=RegistrationConfig(metric_refinement=True),
    )
    np.testing.assert_allclose(
        result.T_target_source[:3, 3], [0.04, -0.02, 0], atol=0.002
    )
    assert result.diagnostics["metric_refinement"]["status"] == "refined"
    assert result.diagnostics["depth_origin"] == "native_source_observation"
