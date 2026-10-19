"""Controlled geometry and identity tests, with no downloaded inputs."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from scene_recall.data import Calibration, Observation
from scene_recall.features import RGBDFeatures, extract_rgbd_features
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview import (
    FeatureFrame,
    TrackConfig,
    build_tracks,
    match_descriptors,
)
from scene_recall.multiview.diagnostics import grid_coverage, summarize_tracks
from scene_recall.multiview.io import load_measurements, measurement_arrays, save_tracks
from scene_recall.odometry.pipeline import TrajectoryFrame


@pytest.fixture
def views():
    rng = np.random.default_rng(6)
    K = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
    calibration = Calibration(K, K, np.eye(4), (240, 320), (240, 320))
    points = rng.uniform([-0.5, -0.4, 2], [0.5, 0.4, 3], (12, 3))
    descriptors = rng.normal(size=(12, 128)).astype(np.float32)
    frames = []
    for index in range(5):
        T_SR = np.eye(4)
        T_SR[:3, 3] = [0.03 * index, 0.01 * index, 0]
        T_SR[:3, :3] = cv2.Rodrigues(np.array([0.01 * index, -0.01 * index, 0]))[0]
        camera_points = transform_points(points, np.linalg.inv(T_SR))
        frames.append(
            FeatureFrame(
                index,
                0,
                index * 0.1,
                T_SR,
                RGBDFeatures(
                    project_points(camera_points, K), descriptors.copy(), camera_points
                ),
            )
        )
    return calibration, frames, points


def assert_same_science(first, second):
    a, b = measurement_arrays(first), measurement_arrays(second)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_exact_multiview_geometry_and_covisibility(views):
    calibration, frames, points = views
    result = build_tracks(frames, calibration)
    assert len(result.landmarks) == len(points)
    for track in result.landmarks:
        assert track.status == "supported"
        assert len(track.observations) == track.depth_views == 5
        assert len({key[0] for key in track.observations}) == 5
        np.testing.assert_allclose(
            track.point_S, points[track.observations[0][1]], atol=1e-14
        )
        assert max(track.reprojection_errors_px) < 1e-10
        assert max(track.depth_errors_m) < 1e-10
        assert track.baseline_m > 0.12
        assert track.parallax_deg > 1
    summary = summarize_tracks(result, calibration)
    assert summary["tracks"] == summary["supported_tracks"] == 12
    assert len(summary["covisibility"]) == 10
    assert summary["direct_cycle_checks"] > 0
    assert summary["leave_one_depth_out"]["strict_pass_fraction"] == 1
    assert all(pair["shared_supported"] == 12 for pair in summary["covisibility"])
    assert_same_science(
        result, build_tracks(frames, calibration, pairs=reversed(result.pairs))
    )


def test_gap_recovery_requires_nonconsecutive_pair(views):
    calibration, frames, _ = views
    empty = RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))
    frames = [frames[0], replace(frames[1], features=empty), frames[2]]
    adjacent = build_tracks(frames, calibration, TrackConfig(pair_offsets=(1,)))
    result = build_tracks(frames, calibration, TrackConfig(pair_offsets=(1, 2)))
    assert not adjacent.landmarks
    assert len(result.landmarks) == 12
    assert all(
        t.observations[0][0] == 0 and t.observations[-1][0] == 2
        for t in result.landmarks
    )
    summary = summarize_tracks(result, calibration)
    assert summary["tracks_with_selected_view_gaps"] == 12
    assert summary["tracks_not_connected_by_adjacent_direct_edges"] == 12


def test_missing_depth_views_are_retained_without_fabricating_depth(views):
    calibration, frames, _ = views
    frames = frames[:3]
    frames[1] = replace(
        frames[1],
        features=replace(frames[1].features, points_R=np.full((12, 3), np.nan)),
    )
    result = build_tracks(frames, calibration)
    assert len(result.landmarks) == 12
    for track in result.landmarks:
        assert track.status == "supported"
        assert track.depth_views == 2
        assert np.isnan(track.depth_errors_m[1])
        assert len(track.observations) == 3
    one_anchor = [frames[0]] + [
        replace(f, features=replace(f.features, points_R=np.full((12, 3), np.nan)))
        for f in frames[1:]
    ]
    tentative = build_tracks(one_anchor, calibration)
    assert all(
        t.status == "tentative" and t.depth_views == 1 for t in tentative.landmarks
    )
    no_anchor = [
        replace(f, features=replace(f.features, points_R=np.full((12, 3), np.nan)))
        for f in frames
    ]
    rejected = build_tracks(no_anchor, calibration)
    assert not rejected.landmarks
    assert all(e.reason == "no depth anchor" for e in rejected.associations)


def test_wrong_matches_and_occluding_depth_fail_geometry(views):
    calibration, frames, _ = views
    corrupted = replace(
        frames[1].features,
        descriptors=np.roll(frames[1].features.descriptors, 1, axis=0),
    )
    result = build_tracks(
        [frames[0], replace(frames[1], features=corrupted)], calibration
    )
    assert not result.landmarks
    assert all(e.status == "rejected" for e in result.associations)
    # Same RGB rays but a different surface must fail the metric gate.
    nearer = replace(frames[0].features, points_R=frames[0].features.points_R * 0.5)
    result = build_tracks(
        [frames[0], replace(frames[1], T_SR=frames[0].T_SR, features=nearer)],
        calibration,
    )
    assert not result.landmarks
    assert all(e.depth_m > 0.9 for e in result.associations)


def test_pose_sensitive_band_is_audited_without_merging(views):
    calibration, frames, _ = views
    first = frames[0]
    biased = first.T_SR.copy()
    biased[0, 3] = 0.035
    second = replace(first, frame_index=1, T_SR=biased)
    result = build_tracks([first, second], calibration)
    assert not result.landmarks
    assert any(e.status == "uncertain" for e in result.associations)
    assert all(e.status in ("uncertain", "rejected") for e in result.associations)
    # A behind-camera depth anchor cannot produce an accepted identity.
    biased[2, 3] = 5
    assert not build_tracks(
        [first, replace(second, T_SR=biased)], calibration
    ).landmarks


def test_transitive_drift_cannot_evade_all_view_validation():
    K = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
    calibration = Calibration(K, K, np.eye(4), (240, 320), (240, 320))
    descriptors = np.random.default_rng(1).normal(size=(2, 128)).astype(np.float32)
    frames = []
    for index in range(12):
        points = np.array([[0.012 * index, 0, 2], [0.012 * index, 0.2, 2]])
        frames.append(
            FeatureFrame(
                index,
                0,
                None,
                np.eye(4),
                RGBDFeatures(project_points(points, K), descriptors, points),
            )
        )
    result = build_tracks(frames, calibration, TrackConfig(pair_offsets=(1,)))
    assert any(e.status == "incompatible" for e in result.associations)
    assert max(len(t.observations) for t in result.landmarks) < len(frames)
    assert all(max(t.reprojection_errors_px) <= 3 for t in result.landmarks)


def test_duplicate_view_conflicts_are_never_unioned(monkeypatch, views):
    from scene_recall.multiview import tracking

    calibration, frames, _ = views
    # Two spatially nearby features, with inconsistent descriptor-cycle identity.
    points = np.array([[0.0, 0, 2], [0.001, 0, 2]])
    features = RGBDFeatures(
        project_points(points, calibration.K_R),
        frames[0].features.descriptors[:2],
        points,
    )
    frames = [replace(f, T_SR=np.eye(4), features=features) for f in frames[:3]]
    count = 0

    def inconsistent(*_):
        nonlocal count
        count += 1
        return np.array([[0, 1], [1, 0]]) if count == 3 else np.array(
            [[0, 0], [1, 1]]
        ), np.array([0.1, 0.2])

    monkeypatch.setattr(tracking, "match_descriptors", inconsistent)
    result = build_tracks(frames, calibration)
    assert any(e.status == "conflict" for e in result.associations)
    assert all(
        len({key[0] for key in t.observations}) == len(t.observations)
        for t in result.landmarks
    )


def test_segments_cannot_share_identities_even_with_identical_images(views):
    calibration, frames, _ = views
    frames = [replace(f, segment_id=int(f.frame_index >= 2)) for f in frames]
    result = build_tracks(frames, calibration)
    assert all(frames[a].segment_id == frames[b].segment_id for a, b in result.pairs)
    assert len(result.landmarks) == 24
    assert all(
        len({frames[k[0]].segment_id for k in t.observations}) == 1
        for t in result.landmarks
    )
    with pytest.raises(ValueError, match="cross"):
        build_tracks(frames, calibration, pairs=[(0, 2)])


@pytest.mark.parametrize("case", ["empty", "single", "duplicate", "nan"])
def test_ambiguous_or_absent_descriptors(case):
    descriptors = np.eye(3, 128, dtype=np.float32)
    if case == "empty":
        assert len(match_descriptors(None, descriptors)[0]) == 0
    elif case == "single":
        assert len(match_descriptors(descriptors[:1], descriptors)[0]) == 0
    elif case == "duplicate":
        assert (
            len(
                match_descriptors(
                    np.zeros((3, 128), np.float32), np.zeros((3, 128), np.float32)
                )[0]
            )
            == 0
        )
    else:
        with pytest.raises(ValueError, match="descriptors"):
            RGBDFeatures(
                np.zeros((3, 2)), np.full((3, 128), np.nan, np.float32), np.ones((3, 3))
            )


def test_mutual_ratio_checks_reject_one_way_ambiguity():
    first = np.zeros((3, 128), np.float32)
    second = first.copy()
    first[:, 0] = [0, 0.01, 10]
    second[:, 0] = [0.0049, 10, 20]
    matches, _ = match_descriptors(first, second)
    np.testing.assert_array_equal(matches, [[2, 1]])


def test_calibration_extrinsic_and_reference_independence(views):
    calibration, frames, _ = views
    T_RD = np.eye(4)
    T_RD[:3, :3] = cv2.Rodrigues(np.array([0.2, -0.1, 0.3]))[0]
    T_RD[:3, 3] = [0.2, -0.1, 0.05]
    calibration = replace(calibration, T_RD=T_RD)
    constructed, changed_references = [], []
    for f in frames:
        row = TrajectoryFrame(
            f.frame_index, f.timestamp, f.segment_id, "tracked", f.T_SR @ T_RD, None
        )
        a = FeatureFrame.from_trajectory(row, calibration, f.features)
        malicious_reference = np.eye(4)
        malicious_reference[0, 0] = -1
        malicious_reference[:3, 3] = 10000
        b = FeatureFrame.from_trajectory(
            replace(row, reference_pose=malicious_reference), calibration, f.features
        )
        np.testing.assert_array_equal(a.T_SR, b.T_SR)
        np.testing.assert_allclose(a.T_SR, f.T_SR, atol=1e-14)
        constructed.append(a)
        changed_references.append(b)
    assert_same_science(
        build_tracks(constructed, calibration),
        build_tracks(changed_references, calibration),
    )


def test_feature_extraction_uses_registered_visibility_and_grid_calibration():
    K_R = np.array([[100.0, 0, 3], [0, 100, 3], [0, 0, 1]])
    K_D = K_R.copy()
    K_D[:2] *= 2
    calibration = Calibration(K_R, K_D, np.eye(4), (7, 7), (14, 14))
    pixels = np.array([[3.0, 3.0], [4.0, 3.0]])

    class SIFT:
        def detectAndCompute(self, *_):
            return [cv2.KeyPoint(float(x), float(y), 1) for x, y in pixels], np.zeros(
                (2, 128), np.float32
            )

    depth = np.full((14, 14), 2.0, np.float32)
    depth[6, 8] = np.nan
    observation = Observation(0, np.zeros((7, 7, 3), np.uint8), depth)
    features = extract_rgbd_features(observation, calibration, SIFT(), None)
    np.testing.assert_array_equal(features.points_R[0], [0, 0, 2])
    assert np.isnan(features.points_R[1]).all()
    # Identical registered grids with a translated optical frame exercise splatting.
    T_RD = np.eye(4)
    T_RD[0, 3] = 0.02
    calibration = Calibration(K_R, K_R, T_RD, (7, 7), (7, 7))
    observation = Observation(0, observation.rgb, np.full((7, 7), 2, np.float32))
    features = extract_rgbd_features(observation, calibration, SIFT(), None)
    np.testing.assert_allclose(features.points_R, [[0, 0, 2], [0.02, 0, 2]], atol=1e-15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pair_offsets": (2, 1)},
        {"pair_offsets": (0,)},
        {"pair_offsets": (1, 1)},
        {"ratio": 1.0},
        {"max_features": True},
        {"reprojection_px": float("nan")},
        {"relaxed_factor": 0.5},
        {"min_views": 1},
        {"depth_relative": -1},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TrackConfig(**kwargs)


def test_invalid_frames_and_pairs_are_rejected(views):
    calibration, frames, _ = views
    for selection, pairs in (
        (frames[::-1], None),
        (frames + frames, None),
        (frames, [(0, 6)]),
        (frames, [(0, 1), (0, 1)]),
        (frames, [(1, 0)]),
    ):
        with pytest.raises(ValueError):
            build_tracks(selection, calibration, pairs=pairs)
    invalid = frames[0].T_SR.copy()
    invalid[0, 0] = -1
    with pytest.raises(ValueError, match="rigid"):
        replace(frames[0], T_SR=invalid)


def test_grid_coverage_does_not_inflate_clustered_support():
    assert grid_coverage([[0, 0], [1, 1]], (480, 640)) == 1 / 48
    assert grid_coverage([], (480, 640)) == 0


def test_atomic_pickle_free_export_and_existing_output(views, tmp_path, monkeypatch):
    calibration, frames, _ = views
    result = build_tracks(frames, calibration)
    output = tmp_path / "tracks"
    saved = save_tracks(output, result, calibration, {"test": "synthetic"})
    assert saved["summary"]["supported_tracks"] == 12
    loaded, arrays = load_measurements(output)
    assert loaded["summary"] == saved["summary"]
    for key, value in measurement_arrays(result).items():
        np.testing.assert_array_equal(arrays[key], value)
    with np.load(output / "measurements.npz", allow_pickle=False) as archive:
        arrays = measurement_arrays(result)
        for key in arrays:
            np.testing.assert_array_equal(archive[key], arrays[key])
        assert archive["track_offsets"][-1] == len(archive["observation_keys"])
        assert archive["feature_offsets"][-1] == len(archive["pixels"])
    with pytest.raises(ValueError, match="exists"):
        save_tracks(output, result, calibration, {})
    failing = tmp_path / "failed"
    monkeypatch.setattr(
        np,
        "savez_compressed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk failure")),
    )
    with pytest.raises(OSError, match="disk failure"):
        save_tracks(failing, result, calibration, {})
    assert not failing.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tracks"]


def test_empty_graph_diagnostics_and_export(views, tmp_path):
    calibration, _, _ = views
    result = build_tracks([], calibration)
    summary = summarize_tracks(result, calibration)
    assert summary["tracks"] == 0
    assert summary["lengths"]["median"] is None
    save_tracks(tmp_path / "empty", result, calibration, {})
    load_measurements(tmp_path / "empty")


@pytest.mark.parametrize(
    "corruption", ["duplicate_identity", "unknown_identity", "offsets", "segment"]
)
def test_archive_rejects_invalid_identity_graph(views, tmp_path, corruption):
    calibration, frames, _ = views
    result = build_tracks(frames, calibration)
    save_tracks(tmp_path / "run", result, calibration, {})
    arrays = measurement_arrays(result)
    if corruption == "duplicate_identity":
        arrays["observation_keys"][5] = arrays["observation_keys"][0]
    elif corruption == "unknown_identity":
        arrays["observation_keys"][0] = [0, 1000]
    elif corruption == "offsets":
        arrays["feature_offsets"][-1] += 1
    else:
        arrays["track_segment_id"][0] = 2
    np.savez_compressed(tmp_path / "run/measurements.npz", **arrays)
    with pytest.raises(ValueError):
        load_measurements(tmp_path / "run")


def test_runner_verifies_full_source_and_retains_segment_roots(tmp_path):
    from helpers.runs import make_run

    from scene_recall.multiview.runner import run_multiview

    source, odometry = make_run(tmp_path)
    output = tmp_path / "tracks"
    report = run_multiview(odometry, output, stride=3, controls=True)
    assert report["metadata"]["canonical_inputs_verified"] is True
    assert report["metadata"]["repeat"]["mismatches"] == []
    assert report["summary"]["frame_count"] == 2
    assert report["summary"]["segments"] == 2
    _, arrays = load_measurements(output)
    np.testing.assert_array_equal(arrays["frame_index"], [0, 1])
    assert not len(arrays["candidate_pairs"])
    # Alter an unselected view. Full replay must refuse publication regardless.
    from PIL import Image

    Image.fromarray(np.full((480, 640), 15000, dtype=np.uint16)).save(
        source / "depth.png"
    )
    with pytest.raises(ValueError, match="fingerprint"):
        run_multiview(odometry, tmp_path / "tampered", stride=3)
    assert not (tmp_path / "tampered").exists()
