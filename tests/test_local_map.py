"""Controlled incremental tracking contracts, with no dataset dependencies."""

from dataclasses import asdict, replace

import cv2
import numpy as np
import pytest
from helpers.runs import continuous_run, make_run

from scene_recall.data import Calibration, Observation
from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.mapping.heldout import run_heldout
from scene_recall.mapping.runner import run_mapping
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.odometry.local_map import (
    LocalMapConfig,
    LocalMapRGBDBackend,
    information_ratio,
)
from scene_recall.odometry.pipeline import track_observations, validate_trajectory
from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig, estimate_feature_pair


def scene(count=8, *, extrinsic=False):
    rng = np.random.default_rng(28)
    K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]])
    T_RD = np.eye(4)
    if extrinsic:
        T_RD[:3, :3] = cv2.Rodrigues(np.array([0.02, -0.04, 0.01]))[0]
        T_RD[:3, 3] = [0.08, -0.02, 0.04]
    calibration = Calibration(K, K, T_RD, (240, 320), (240, 320))
    points = rng.uniform([-0.8, -0.55, 1.6], [0.8, 0.55, 3.2], (180, 3))
    descriptors = rng.normal(size=(len(points), 128)).astype(np.float32)
    observations, features, truth = [], {}, []
    for i in range(count):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([0.002 * i, -0.003 * i, 0.001 * i]))[0]
        pose[:3, 3] = [0.015 * i, 0.002 * i, 0]
        truth.append(pose)
        measured = transform_points(points, T_RD @ np.linalg.inv(pose))
        features[i] = RGBDFeatures(
            project_points(measured, K), descriptors.copy(), measured
        )
        observations.append(
            Observation(
                i, np.zeros((240, 320, 3), np.uint8), np.full((240, 320), 2, np.float32)
            )
        )
    return calibration, observations, features, truth


def tracker(monkeypatch, features, config=None):
    backend = LocalMapRGBDBackend(config)
    monkeypatch.setattr(backend, "_extract", lambda o, c: features[o.frame_index])
    return backend


@pytest.mark.parametrize("update", ["frozen", "median"])
def test_metric_poses_extrinsics_persistent_identities_and_budgets(monkeypatch, update):
    calibration, observations, features, truth = scene(18, extrinsic=True)
    config = LocalMapConfig(
        max_keyframes=3,
        max_landmarks=200,
        min_keyframe_gap=1,
        max_keyframe_gap=2,
        landmark_update=update,
    )
    backend = tracker(monkeypatch, features, config)
    rows = list(track_observations(observations, calibration, backend))
    validate_trajectory(rows)
    assert all(row.status == "tracked" for row in rows[1:])
    np.testing.assert_allclose([r.T_SC_D for r in rows], truth, atol=1e-6)
    assert len(backend.keyframes) <= 3
    assert len(backend.landmarks) <= 200
    assert min(backend.keyframes) > 0
    assert any(len(p.observations) >= 2 for p in backend.landmarks.values())
    for landmark in backend.landmarks.values():
        assert len({i for i, _ in landmark.observations}) == len(landmark.observations)
        assert all(i in backend.keyframes for i, _ in landmark.observations)


def test_map_tracks_disjoint_consecutive_features_without_target_depth(monkeypatch):
    calibration, observations, features, truth = scene(3)
    for i, selection in ((1, slice(0, 85)), (2, slice(85, None))):
        f = features[i]
        features[i] = RGBDFeatures(
            f.pixels[selection],
            f.descriptors[selection],
            np.full_like(f.points_R[selection], np.nan),
        )
    sparse = estimate_feature_pair(
        features[1], features[2], calibration, SparseConfig()
    )
    assert sparse.T_C1C0 is None
    backend = tracker(monkeypatch, features, LocalMapConfig(pairwise_fallback=False))
    rows = list(track_observations(observations, calibration, backend))
    assert rows[-1].pair.diagnostics["tracking_mode"] == "map"
    np.testing.assert_allclose(rows[-1].T_SC_D, truth[-1], atol=1e-6)


def test_reference_independence_and_exact_repeat(monkeypatch):
    calibration, observations, features, _ = scene()
    original = list(
        track_observations(observations, calibration, tracker(monkeypatch, features))
    )
    changed = [replace(o, T_WC_D=np.diag([4.0, 3.0, 2.0, 1.0])) for o in observations]
    repeated = list(
        track_observations(changed, calibration, tracker(monkeypatch, features))
    )
    np.testing.assert_array_equal(
        [r.T_SC_D for r in original], [r.T_SC_D for r in repeated]
    )
    for a, b in zip(original[1:], repeated[1:], strict=True):
        first, second = dict(a.pair.diagnostics), dict(b.pair.diagnostics)
        first = {k: v for k, v in first.items() if not k.endswith("_seconds")}
        second = {k: v for k, v in second.items() if not k.endswith("_seconds")}
        assert first == second


def test_depth_inconsistency_cannot_enter_map_consensus(monkeypatch):
    calibration, observations, features, truth = scene(2)
    f = features[1]
    bad = f.points_R.copy()
    bad[:80] *= 1.5
    features[1] = replace(f, points_R=bad)
    backend = tracker(monkeypatch, features, LocalMapConfig(pairwise_fallback=False))
    rows = list(track_observations(observations, calibration, backend))
    assert rows[-1].pair.diagnostics["map_inliers"] == len(bad) - 80
    np.testing.assert_allclose(rows[-1].T_SC_D, truth[-1], atol=1e-6)


def test_complete_depth_disagreement_abstains_without_fallback(monkeypatch):
    calibration, observations, features, _ = scene(2)
    features[1] = replace(features[1], points_R=features[1].points_R * 2)
    rows = list(
        track_observations(
            observations,
            calibration,
            tracker(monkeypatch, features, LocalMapConfig(pairwise_fallback=False)),
        )
    )
    assert rows[-1].status == "lost"
    assert "depth-filtered" in rows[-1].pair.reason


def test_fallback_cannot_seed_rejected_measurements_as_new_landmarks(monkeypatch):
    calibration, observations, features, _ = scene(2)
    features[1] = replace(features[1], points_R=features[1].points_R * 2)
    backend = tracker(
        monkeypatch, features, LocalMapConfig(min_keyframe_gap=1, max_keyframe_gap=1)
    )
    rows = list(track_observations(observations, calibration, backend))
    assert rows[-1].pair.diagnostics["tracking_mode"] == "pairwise_fallback"
    assert len(backend.landmarks) == len(features[0].pixels)
    assert np.all(backend.owners[1] == -1)


def test_loss_resets_gauge_and_never_recovers_old_map(monkeypatch):
    calibration, observations, features, _ = scene(4)
    features[1] = RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))
    backend = tracker(monkeypatch, features)
    rows = list(track_observations(observations, calibration, backend))
    assert [r.status for r in rows] == ["initialized", "lost", "lost", "tracked"]
    assert [r.segment_id for r in rows] == [0, 1, 2, 2]
    np.testing.assert_array_equal(rows[2].T_SC_D, np.eye(4))
    assert min(backend.keyframes) == 2
    validate_trajectory(rows)


def test_conflicting_map_identities_do_not_join(monkeypatch):
    calibration, observations, features, _ = scene(2)
    backend = tracker(monkeypatch, features)
    backend._bootstrap(observations[0], features[0], calibration)
    # Duplicate keyframe anchors deliberately represent ambiguous identities.
    backend._insert(1, features[0], np.eye(4), [], calibration)
    assert backend._matches(features[1])[0] == []


def test_fallback_is_explicit_and_stream_reuse_rejected(monkeypatch):
    calibration, observations, features, _ = scene(3)
    backend = tracker(monkeypatch, features, LocalMapConfig(min_grid_coverage=1))
    pair = backend.estimate(observations[0], observations[1], calibration)
    assert pair.T_C1C0 is not None
    assert pair.diagnostics["tracking_mode"] == "pairwise_fallback"
    with pytest.raises(ValueError, match="streams"):
        backend.estimate(replace(observations[1]), observations[2], calibration)


def test_information_detects_degenerate_bearings():
    assert information_ratio(np.tile([0.0, 0.0, 2.0], (100, 1))) < 1e-12
    _, _, features, _ = scene(1)
    assert information_ratio(features[0].points_R) > 1e-4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_keyframes": 1},
        {"max_landmarks": 0},
        {"max_keyframe_gap": 1},
        {"keyframe_translation_m": np.nan},
        {"min_grid_coverage": 1.1},
        {"landmark_update": "unknown"},
        {"pairwise_fallback": 1},
        {"sparse": SparseConfig()},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LocalMapConfig(**kwargs)


def test_saved_local_map_run_replays_through_existing_mapping(tmp_path):
    source, _ = make_run(tmp_path)
    run = tmp_path / "local_map"
    report = run_odometry(
        source,
        run,
        dataset="tum_rgbd",
        backend_name="local_map",
        config=LocalMapConfig(),
    )
    saved, rows = load_trajectory(run)
    assert saved["backend"] == "local_map"
    assert report["evaluation"]["failed_transitions"] == 2
    validate_trajectory(rows)
    mapping = run_mapping(
        run, tmp_path / "mapping", MappingConfig(pixel_stride=32, keyframe_stride=1)
    )
    assert mapping["segment_count"] == 3


def test_heldout_fusion_accepts_paired_local_map_trajectories(tmp_path):
    _, baseline = continuous_run(tmp_path)
    report, rows = load_trajectory(baseline)
    candidate = tmp_path / "local_map"
    save_run(
        candidate,
        rows,
        {**report, "backend": "local_map", "config": asdict(LocalMapConfig())},
        0,
    )
    result = run_heldout(
        [baseline, candidate],
        tmp_path / "heldout",
        config=MappingConfig(pixel_stride=32, keyframe_stride=2),
        offset=1,
    )
    summary = result["summaries"]["0.01"]
    assert summary["evaluated_views"] == 1
    assert (
        summary["pooled"]["sparse_reference_pose"]
        == summary["pooled"]["local_map_reference_pose"]
    )
