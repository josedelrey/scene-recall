"""Recent map contracts under analytic measurements and controlled corruption."""

from dataclasses import replace

import numpy as np
import pytest
from test_local_map import scene

from scene_recall.features import RGBDFeatures
from scene_recall.odometry.local_map import LocalMapRGBDBackend
from scene_recall.odometry.pipeline import track_observations, validate_trajectory
from scene_recall.odometry.recent_map import (
    RecentMapConfig,
    RecentMapRGBDBackend,
    merge_proposals,
)


def track(monkeypatch, features, observations, calibration, config=None):
    backend = RecentMapRGBDBackend(config)
    monkeypatch.setattr(backend, "_extract", lambda o, c: features[o.frame_index])
    rows = list(track_observations(observations, calibration, backend))
    return backend, rows


@pytest.mark.parametrize("association", ["indexed", "exhaustive"])
@pytest.mark.parametrize("extrinsic", [False, True])
def test_poses_and_identity_lifecycle_are_metric_and_bounded(
    monkeypatch, association, extrinsic
):
    calibration, observations, features, truth = scene(18, extrinsic=extrinsic)
    backend, rows = track(
        monkeypatch,
        features,
        observations,
        calibration,
        RecentMapConfig(
            association_policy=association,
            max_keyframes=3,
            max_landmarks=200,
            max_keyframe_gap=2,
            min_keyframe_gap=1,
        ),
    )
    validate_trajectory(rows)
    assert all(r.status == "tracked" for r in rows[1:])
    np.testing.assert_allclose([r.T_SC_D for r in rows], truth, atol=1e-6)
    assert len(backend.landmarks) <= 200
    assert len(backend.keyframes) <= 3
    assert len(backend._index_ids) == len(set(backend._index_ids)) <= 200
    assert all(i in backend.landmarks for i in backend._index_ids)
    for identity, landmark in backend.landmarks.items():
        assert all(i in backend.keyframes for i, _ in landmark.observations)
        assert all(backend.owners[i][k] == identity for i, k in landmark.observations)


def test_first_depth_bias_does_not_persist_as_exact_world_geometry(monkeypatch):
    calibration, observations, features, truth = scene(20)
    features[0] = replace(features[0], points_R=features[0].points_R * 1.015)
    _, rows = track(monkeypatch, features, observations, calibration)
    assert all(r.status == "tracked" for r in rows[1:])
    assert np.linalg.norm(rows[-1].T_SC_D[:3, 3] - truth[-1][:3, 3]) < 0.001


def test_missing_depth_keeps_metric_map_and_no_reference_use(monkeypatch):
    calibration, observations, features, truth = scene(12)
    for i in range(3, 10):
        features[i] = replace(
            features[i], points_R=np.full_like(features[i].points_R, np.nan)
        )
    _, rows = track(monkeypatch, features, observations, calibration)
    _, repeated = track(
        monkeypatch,
        features,
        [replace(o, T_WC_D=np.diag([4.0, 3.0, 2.0, 1.0])) for o in observations],
        calibration,
    )
    np.testing.assert_allclose([r.T_SC_D for r in rows], truth, atol=1e-6)
    np.testing.assert_array_equal(
        [r.T_SC_D for r in rows], [r.T_SC_D for r in repeated]
    )


def test_index_recovery_tracks_reappearing_landmarks_with_disjoint_previous_features(
    monkeypatch,
):
    calibration, observations, features, truth = scene(3)
    for i, selection in ((1, slice(0, 85)), (2, slice(85, None))):
        f = features[i]
        features[i] = RGBDFeatures(
            f.pixels[selection],
            f.descriptors[selection],
            np.full_like(f.points_R[selection], np.nan),
        )
    _, rows = track(
        monkeypatch,
        features,
        observations,
        calibration,
        RecentMapConfig(pairwise_fallback=False),
    )
    assert rows[-1].pair.diagnostics["index_queries"] == 1
    np.testing.assert_allclose(rows[-1].T_SC_D, truth[-1], atol=1e-6)


def test_reanchor_rejects_corrupted_depth_and_preserves_all_active_views(monkeypatch):
    calibration, observations, features, _ = scene(6)
    f = features[3]
    corrupted = f.points_R.copy()
    corrupted[:100] *= 1.5
    features[3] = replace(f, points_R=corrupted)
    backend, rows = track(
        monkeypatch,
        features,
        observations,
        calibration,
        RecentMapConfig(min_keyframe_gap=1, max_keyframe_gap=1),
    )
    assert all(r.status == "tracked" for r in rows[1:])
    for identity, landmark in backend.landmarks.items():
        assert backend._valid_points([identity], landmark.point_S[None], calibration)[0]
    assert sum(r.pair.diagnostics["reanchored_landmarks"] for r in rows[1:]) > 0


def test_loss_discards_index_and_previous_identities(monkeypatch):
    calibration, observations, features, _ = scene(4)
    features[1] = RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))
    backend, rows = track(monkeypatch, features, observations, calibration)
    assert [r.segment_id for r in rows] == [0, 1, 2, 2]
    assert min(backend.keyframes) == 2
    assert all(i >= 2 for p in backend.landmarks.values() for i, _ in p.observations)
    validate_trajectory(rows)


def test_conflicts_are_reserved_and_never_arbitrarily_merged():
    selected, reserved = merge_proposals(
        [(1, 2), (1, 2), (2, 2), (3, 4), (3, 5), (4, 6)]
    )
    assert selected == [(4, 6)]
    assert reserved == {2, 4, 5, 6}


def test_frozen_exhaustive_control_preserves_original_frontend(monkeypatch):
    calibration, observations, features, _ = scene(18)
    legacy = LocalMapRGBDBackend()
    monkeypatch.setattr(legacy, "_extract", lambda o, c: features[o.frame_index])
    original = list(track_observations(observations, calibration, legacy))
    _, control = track(
        monkeypatch,
        features,
        observations,
        calibration,
        RecentMapConfig(anchor_policy="frozen", association_policy="exhaustive"),
    )
    np.testing.assert_array_equal(
        [r.T_SC_D for r in original], [r.T_SC_D for r in control]
    )


@pytest.mark.parametrize(
    "options",
    [
        {"anchor_policy": "invalid"},
        {"association_policy": "invalid"},
        {"landmark_update": "median"},
    ],
)
def test_invalid_policies(options):
    with pytest.raises(ValueError):
        RecentMapConfig(**options)
