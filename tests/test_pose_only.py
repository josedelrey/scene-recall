"""Controlled pose-only geometry, derivatives, gauge and bias counterexamples."""

import runpy
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from scene_recall.data import Calibration
from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.model import (
    FeatureFrame,
    Landmark,
    TrackConfig,
    TrackResult,
)
from scene_recall.multiview.pose_only import (
    PoseOnlyConfig,
    linearize,
    make_factors,
    refine_training_poses,
    retract,
    robust_cost,
    solve_window,
)


def synthetic_tracks(*, bias=False, collinear=False, missing=False):
    rng = np.random.default_rng(14)
    K = np.array([[300.0, 0, 160], [0, 310, 120], [0, 0, 1]])
    calibration = Calibration(K, K, np.eye(4), (240, 320), (240, 320))
    points = rng.uniform([-0.7, -0.5, 2], [0.7, 0.5, 3.5], (96, 3))
    if collinear:
        points[:, :2] = 0
    frames, truth = [], []
    for i in range(6):
        pose = np.eye(4)
        pose[:3, 3] = [0.03 * i, 0.005 * i, 0]
        pose[:3, :3] = cv2.Rodrigues(np.array([0, 0.005 * i, 0.002 * i]))[0]
        truth.append(pose)
        local = transform_points(points, np.linalg.inv(pose))
        pixels = project_points(local, K)
        if missing and i % 2:
            local = np.full_like(local, np.nan)
        initial = pose.copy()
        if i:
            initial = retract(
                pose,
                np.array([0, 0, 0.04, 0, 0, 0])
                + rng.normal(0, [0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001])
                if bias
                else rng.normal(0, [0.006, 0.006, 0.006, 0.001, 0.001, 0.001]),
            )
        frames.append(
            FeatureFrame(i, 0, i * 0.1, initial, RGBDFeatures(pixels, None, local))
        )
    landmarks = tuple(
        Landmark(
            j,
            0,
            tuple((i, j) for i in range(6)),
            points[j] + [0, 0, 0.04],
            "supported",
            6,
            np.zeros(6),
            np.zeros(6),
            2.0,
            0.15,
        )
        for j in range(len(points))
    )
    return (
        calibration,
        TrackResult(tuple(frames), landmarks, (), (), TrackConfig(), {}),
        np.array(truth),
    )


@pytest.mark.parametrize("formulation", ["anchored", "frozen"])
def test_analytic_jacobian_matches_finite_differences(formulation):
    calibration, tracks, _ = synthetic_tracks(missing=True)
    config = PoseOnlyConfig(formulation=formulation)
    factors, _ = make_factors(
        tracks, list(range(6)), calibration.K_R, calibration.rgb_shape, config
    )
    initial = np.array([f.T_SR for f in tracks.frames])
    _, J = linearize(initial, factors, calibration.K_R, config)
    for column in range(J.shape[-1]):
        i, j = divmod(column, 6)
        delta = np.zeros(6)
        delta[j] = 1e-7
        plus, minus = initial.copy(), initial.copy()
        plus[i + 1] = retract(initial[i + 1], delta)
        minus[i + 1] = retract(initial[i + 1], -delta)
        r_plus, _ = linearize(plus, factors, calibration.K_R, config, jacobian=False)
        r_minus, _ = linearize(minus, factors, calibration.K_R, config, jacobian=False)
        np.testing.assert_allclose(
            J[:, :, column], (r_plus - r_minus) / 2e-7, atol=2e-5, rtol=2e-5
        )


def test_coupled_refinement_recovers_poses_and_preserves_root():
    calibration, tracks, truth = synthetic_tracks()
    initial = np.array([f.T_SR for f in tracks.frames])
    refined, diagnostics = refine_training_poses(tracks, calibration)
    result = np.array([refined[i] for i in range(6)])
    assert np.linalg.norm(result - truth) < np.linalg.norm(initial - truth) / 10
    np.testing.assert_array_equal(result[0], initial[0])
    for window in diagnostics["windows"]:
        if window["status"] == "refined":
            assert window["final_cost"] < window["initial_cost"]
            assert np.all(np.diff(window["accepted_costs"]) < 0)
    repeated, _ = refine_training_poses(tracks, calibration)
    np.testing.assert_array_equal(result, np.array([repeated[i] for i in range(6)]))


def test_frozen_landmarks_reinforce_shared_initialization_bias():
    calibration, tracks, truth = synthetic_tracks(bias=True)
    anchored, _ = refine_training_poses(tracks, calibration)
    frozen, report = refine_training_poses(
        tracks, calibration, PoseOnlyConfig(formulation="frozen")
    )
    anchor_error = np.linalg.norm(np.array([anchored[i] for i in range(6)]) - truth)
    frozen_error = np.linalg.norm(np.array([frozen[i] for i in range(6)]) - truth)
    assert anchor_error < frozen_error / 5
    assert report["windows"][0]["final_cost"] < report["windows"][0]["initial_cost"]
    assert frozen_error > 0.05


def test_missing_depth_remains_pixel_only():
    calibration, tracks, _ = synthetic_tracks(missing=True)
    config = PoseOnlyConfig()
    factors, _ = make_factors(
        tracks, list(range(6)), calibration.K_R, calibration.rgb_shape, config
    )
    initial = np.array([f.T_SR for f in tracks.frames])
    r, J = linearize(initial, factors, calibration.K_R, config)
    assert np.isnan(factors.depth).any()
    np.testing.assert_array_equal(r[np.isnan(factors.depth), 2], 0)
    np.testing.assert_array_equal(J[np.isnan(factors.depth), 2], 0)
    refined, _ = refine_training_poses(tracks, calibration)
    assert all(np.isfinite(p).all() for p in refined.values())


def test_world_gauge_and_saved_landmark_bias_do_not_change_anchored_fit():
    calibration, tracks, _ = synthetic_tracks()
    first, _ = refine_training_poses(tracks, calibration)
    biased = replace(
        tracks,
        landmarks=tuple(replace(t, point_S=t.point_S + 1000) for t in tracks.landmarks),
    )
    second, _ = refine_training_poses(biased, calibration)
    for i in first:
        np.testing.assert_array_equal(first[i], second[i])
    gauge = retract(np.eye(4), np.array([0.4, -0.2, 0.3, 0.2, -0.1, 0.3]))
    shifted = replace(
        tracks, frames=tuple(replace(f, T_SR=gauge @ f.T_SR) for f in tracks.frames)
    )
    third, _ = refine_training_poses(shifted, calibration)
    for i in first:
        np.testing.assert_allclose(third[i], gauge @ first[i], atol=1e-10)


def test_endpoint_correction_carries_into_an_unsupported_tail():
    calibration, tracks, _ = synthetic_tracks()
    extended = replace(
        tracks,
        frames=tracks.frames
        + tuple(replace(tracks.frames[-1], frame_index=i) for i in range(6, 12)),
    )
    poses, report = refine_training_poses(extended, calibration)
    assert report["windows"][0]["status"] == "refined"
    assert all(w["status"] == "skipped" for w in report["windows"][1:])
    correction = poses[5] @ np.linalg.inv(tracks.frames[5].T_SR)
    for i in range(6, 12):
        np.testing.assert_allclose(
            poses[i], correction @ extended.frames[i].T_SR, atol=1e-14
        )


def test_lost_segments_remain_independent():
    calibration, tracks, _ = synthetic_tracks()
    other_frames = tuple(
        replace(f, frame_index=f.frame_index + 10, segment_id=1) for f in tracks.frames
    )
    other_tracks = tuple(
        replace(
            t,
            track_id=t.track_id + 96,
            segment_id=1,
            observations=tuple((f + 10, k) for f, k in t.observations),
        )
        for t in tracks.landmarks
    )
    combined = replace(
        tracks,
        frames=tracks.frames + other_frames,
        landmarks=tracks.landmarks + other_tracks,
    )
    poses, report = refine_training_poses(combined, calibration)
    for window in report["windows"]:
        assert all(
            (i < 10) == (window["segment_id"] == 0) for i in window["frame_indices"]
        )
    for i in range(6):
        np.testing.assert_array_equal(poses[i], poses[i + 10])


def test_rank_deficiency_is_rejected_without_pose_changes():
    calibration, tracks, _ = synthetic_tracks(collinear=True)
    config = PoseOnlyConfig(min_grid_coverage=0)
    factors, _ = make_factors(
        tracks, list(range(6)), calibration.K_R, calibration.rgb_shape, config
    )
    initial = np.array([f.T_SR for f in tracks.frames])
    result, report = solve_window(initial, factors, calibration.K_R, config)
    assert report["status"] == "rejected"
    np.testing.assert_array_equal(result, initial)


def test_robust_loss_limits_a_corrupted_depth_sample():
    calibration, tracks, truth = synthetic_tracks()
    config = PoseOnlyConfig()
    factors, _ = make_factors(
        tracks, list(range(6)), calibration.K_R, calibration.rgb_shape, config
    )
    corrupted = factors.depth.copy()
    corrupted[:50] += 0.5
    factors = replace(factors, depth=corrupted)
    initial = np.array([f.T_SR for f in tracks.frames])
    robust, _ = solve_window(initial, factors, calibration.K_R, config)
    linear, _ = solve_window(
        initial, factors, calibration.K_R, replace(config, loss="linear")
    )
    assert np.linalg.norm(robust - truth) < np.linalg.norm(linear - truth)


def test_cheirality_cannot_reduce_cost_by_dropping_factors():
    calibration, tracks, _ = synthetic_tracks()
    config = PoseOnlyConfig()
    factors, _ = make_factors(
        tracks, list(range(6)), calibration.K_R, calibration.rgb_shape, config
    )
    initial = np.array([f.T_SR for f in tracks.frames])
    initial[1, 2, 3] = 10
    residual, _ = linearize(initial, factors, calibration.K_R, config)
    cost, _ = robust_cost(residual, factors, config)
    assert not np.isfinite(cost)


def test_biased_anchor_can_reduce_residuals_while_worsening_poses():
    probe = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "experiments/pose_only_bias_probe.py")
    )["probe"]
    report, _ = probe(anchor_depth_bias_m=0.04)
    fitted = report["anchored"]
    assert fitted["final_cost"] < fitted["initial_cost"]
    assert fitted["translation_rmse_m"] > 2 * report["initial_translation_rmse_m"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_step": 6},
        {"window_size": 2},
        {"loss": "unknown"},
        {"max_tracks": True},
        {"depth_sigma_constant_m": float("nan")},
        {"min_grid_coverage": 2},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        PoseOnlyConfig(**kwargs)
