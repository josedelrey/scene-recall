"""Held-out registration and explicit atlas gauge/revision contracts."""

from dataclasses import replace

import numpy as np
import pytest
from test_local_map import scene

from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import transform_points
from scene_recall.slam.atlas import AtlasConfig, SubmapAtlas
from scene_recall.slam.registration import (
    RegistrationConfig,
    register_anchor,
    register_correspondences,
    register_features,
    rigid_alignment,
)


def empty_features():
    return RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))


def translated(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


@pytest.mark.parametrize("extrinsic", [False, True])
def test_registration_recovers_metric_pose_and_handles_extrinsics(extrinsic):
    calibration, _, features, truth = scene(2, extrinsic=extrinsic)
    result = register_features(features[0], features[1], calibration)
    assert result.T_target_source is not None
    expected = (
        calibration.T_RD
        @ np.linalg.inv(truth[1])
        @ truth[0]
        @ np.linalg.inv(calibration.T_RD)
    )
    np.testing.assert_allclose(result.T_target_source, expected, atol=1e-10)
    assert result.diagnostics["validation_inliers"] == 90


def test_withheld_metric_disagreement_rejects_without_refitting():
    calibration, _, features, _ = scene(2)
    first, second = features[0], features[1]
    changed = second.points_R.copy()
    changed[1::2, 0] += 0.3
    result = register_correspondences(
        first.points_R,
        changed,
        first.pixels,
        second.pixels,
        calibration.K_R,
        calibration.rgb_shape,
    )
    assert result.T_target_source is None
    assert result.diagnostics["fit_inliers"] == 90
    assert result.diagnostics["validation_inliers"] == 0


def test_recovery_rejects_collinear_and_missing_depth():
    a = np.column_stack((np.arange(12) / 10, np.zeros(12), np.ones(12)))
    with pytest.raises(ValueError, match="collinear"):
        rigid_alignment(a, a)
    calibration, _, features, _ = scene(2)
    no_depth = replace(features[1], points_R=np.full_like(features[1].points_R, np.nan))
    assert register_features(features[0], no_depth, calibration).T_target_source is None


def test_metric_fit_does_not_absorb_scale_error():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(50, 3))
    pose = rigid_alignment(a, 1.1 * a)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)
    assert np.linalg.norm(transform_points(a, pose) - 1.1 * a) > 0.1


@pytest.mark.parametrize("register", [register_features, register_anchor])
def test_shared_depth_scale_bias_survives_disjoint_verification(register):
    calibration, _, features, truth = scene(2)
    biased = [replace(features[i], points_R=1.1 * features[i].points_R) for i in (0, 1)]
    fitted = register(*biased, calibration)
    assert fitted.T_target_source is not None
    expected = np.linalg.inv(truth[1])
    np.testing.assert_allclose(
        fitted.T_target_source[:3, 3], 1.1 * expected[:3, 3], atol=1e-8
    )
    assert np.linalg.norm(fitted.T_target_source[:3, 3] - expected[:3, 3]) > 0.001


def test_historical_pnp_verifies_missing_incoming_depth_and_rejects_wrong_pixels():
    calibration, _, features, truth = scene(2)
    target = replace(features[1], points_R=np.full_like(features[1].points_R, np.nan))
    accepted = register_anchor(features[0], target, calibration)
    assert accepted.T_target_source is not None
    np.testing.assert_allclose(
        accepted.T_target_source, np.linalg.inv(truth[1]), atol=1e-8
    )
    assert accepted.diagnostics["validation_depth_checks"] == 0
    pixels = target.pixels.copy()
    pixels[1::2] += 50
    rejected = register_anchor(features[0], replace(target, pixels=pixels), calibration)
    assert rejected.T_target_source is None
    assert rejected.diagnostics["fit_inliers"] == 90


def atlas_fixture():
    calibration, _, _, _ = scene(2)
    atlas = SubmapAtlas(calibration)
    for frame, segment, x in [(0, 0, 0), (1, 0, 0.1), (10, 1, 0), (11, 1, 0.1)]:
        atlas.observe(frame, segment, translated(x), empty_features())
    return atlas


def test_atlas_requires_distinct_views_and_preserves_source_segment_ids():
    atlas = atlas_fixture()
    diagnostic = {"status": "accepted", "fit_inliers": 20}
    assert atlas.propose(0, 10, translated(-1), diagnostic)["status"] == "tentative"
    assert atlas.components == {0: 0, 1: 1}
    assert (
        atlas.propose(0, 11, translated(-1.1), diagnostic)["status"]
        == "needs_distinct_views"
    )
    event = atlas.propose(1, 11, translated(-1), diagnostic)
    assert event["status"] == "connected"
    assert atlas.components == {0: 0, 1: 0}
    assert atlas.frames[11][0] == 1
    np.testing.assert_allclose(atlas.pose(11), translated(1.1))
    assert atlas.version == 1


def test_atlas_registers_rgb_anchors_into_metric_depth_gauges():
    calibration, _, features, truth = scene(12, extrinsic=True)
    atlas = SubmapAtlas(calibration, AtlasConfig(keyframe_stride=1))
    for frame in (0, 1, 10, 11):
        local = truth[frame] if frame < 10 else np.linalg.inv(truth[10]) @ truth[frame]
        atlas.observe(
            frame, int(frame >= 10), local, features[frame], boundary=frame in (0, 10)
        )
    assert len(atlas.forest) == 1
    np.testing.assert_allclose(atlas.pose(10), truth[10], atol=1e-7)
    np.testing.assert_allclose(atlas.pose(11), truth[11], atol=1e-7)


def test_confirmation_burst_recovers_short_segment_without_changing_bank():
    calibration, _, features, truth = scene(2)
    variants = [
        SubmapAtlas(calibration, AtlasConfig(confirmation_query_frames=n))
        for n in (0, 10)
    ]
    for atlas in variants:
        for frame, segment, view in ((0, 0, 0), (5, 0, 1), (20, 1, 0), (21, 1, 1)):
            atlas.observe(
                frame, segment, truth[view], features[view], boundary=frame in (0, 20)
            )
    assert len(variants[0].forest) == 0
    assert len(variants[1].forest) == 1
    assert list(variants[0].entries) == list(variants[1].entries) == [0, 5, 20]


def test_revocation_splits_components_and_invalidates_the_child_tree():
    atlas = atlas_fixture()
    for frame, x in ((20, 0), (21, 0.1)):
        atlas.observe(frame, 2, translated(x), empty_features())
    diagnostic = {"status": "accepted", "fit_inliers": 20}
    for source, target in ((0, 10), (1, 11), (10, 20), (11, 21)):
        atlas.propose(source, target, translated(-1), diagnostic)
    assert atlas.components == {0: 0, 1: 0, 2: 0}
    version = atlas.version
    event = atlas.revoke(0, 10, "independent evidence rejects this place identity")
    assert event["affected_segments"] == [1, 2]
    assert atlas.components == {0: 0, 1: 1, 2: 1}
    assert atlas.version > version
    np.testing.assert_allclose(atlas.pose(21), translated(1.1))
    assert len(atlas.forest) == 1
    assert atlas.frames[21][0] == 2


def test_failed_revocation_is_atomic():
    atlas = atlas_fixture()
    before = atlas.components.copy(), atlas.version
    with pytest.raises(ValueError):
        atlas.revoke(0, 10, "unknown constraint")
    with pytest.raises(ValueError):
        atlas.revoke(0, 10, " ")
    assert (atlas.components, atlas.version) == before
    with pytest.raises(ValueError):
        AtlasConfig(confirmation_query_frames=True)


def test_atlas_revisions_rebuild_links_and_invalidate_root_geometry():
    atlas = atlas_fixture()
    diagnostic = {"status": "accepted", "fit_inliers": 20}
    atlas.propose(0, 10, translated(-1), diagnostic)
    atlas.propose(1, 11, translated(-1), diagnostic)
    old = atlas.version
    assert atlas.revise({0: translated(0.02)}) == [0, 1]
    assert atlas.version > old
    np.testing.assert_allclose(
        np.linalg.inv(atlas.pose(10)) @ atlas.pose(0), translated(-1)
    )
    np.testing.assert_allclose(atlas.pose(11), translated(1.12))
    standalone = atlas_fixture()
    old = standalone.version
    assert standalone.revise({0: translated(0.01)}) == [0]
    assert standalone.version == old + 1


def test_inconsistent_confirmation_and_cycle_never_move_existing_maps():
    atlas = atlas_fixture()
    diagnostic = {"status": "accepted", "fit_inliers": 20}
    atlas.propose(0, 10, translated(-1), diagnostic)
    assert (
        atlas.propose(1, 11, translated(-2), diagnostic)["status"]
        == "inconsistent_confirmation"
    )
    assert not atlas.forest
    atlas.propose(0, 10, translated(-2), diagnostic)
    atlas.propose(1, 11, translated(-2), diagnostic)
    before = atlas.pose(11).copy()
    assert (
        atlas.propose(0, 10, translated(-4), diagnostic)["status"]
        == "inconsistent_cycle"
    )
    np.testing.assert_array_equal(atlas.pose(11), before)
    with pytest.raises(ValueError):
        atlas.propose(0, 10, translated(-2), {"status": "rejected"})


def test_ambiguous_disconnected_places_do_not_merge():
    calibration, _, features, _ = scene(2)
    atlas = SubmapAtlas(calibration, AtlasConfig(keyframe_stride=1))
    atlas.observe(0, 0, np.eye(4), features[0])
    atlas.observe(10, 1, np.eye(4), features[0], boundary=True)
    record = atlas.observe(20, 2, np.eye(4), features[0], boundary=True)
    assert len(atlas.forest) == 0
    assert all(q["status"] == "ambiguous_components" for q in record["queries"])


def test_exact_place_alias_can_pass_multiple_view_confirmation():
    # This counterexample records an identifiability limit, rather than treating
    # correspondence holdouts as evidence of independent physical place identity.
    calibration, _, features, truth = scene(2)
    atlas = SubmapAtlas(calibration, AtlasConfig(keyframe_stride=1))
    for frame, segment, view in ((0, 0, 0), (1, 0, 1), (10, 1, 0), (11, 1, 1)):
        atlas.observe(
            frame, segment, truth[view], features[view], boundary=frame in (0, 10)
        )
    assert len(atlas.forest) == 1
    assert atlas.components[0] == atlas.components[1]
    physical_second_place = translated(5) @ truth[1]
    assert np.linalg.norm(
        atlas.pose(11)[:3, 3] - physical_second_place[:3, 3]
    ) == pytest.approx(5)


def test_place_bank_is_bounded_and_replays_deterministically():
    calibration, _, features, _ = scene(2)
    config = AtlasConfig(max_entries=8, recent_entries=2, keyframe_stride=1)
    a, b = SubmapAtlas(calibration, config), SubmapAtlas(calibration, config)
    for atlas in (a, b):
        for i in range(30):
            atlas.observe(i, 0, translated(i / 100), features[0])
        assert len(atlas.entries) <= 8
        assert {0, 28, 29}.issubset(atlas.entries)
    assert list(a.entries) == list(b.entries)


def test_rejected_revision_is_atomic():
    atlas = atlas_fixture()
    original = atlas.pose(0).copy()
    with pytest.raises(ValueError):
        atlas.revise({0: translated(0.2), 100: np.eye(4)})
    np.testing.assert_array_equal(atlas.pose(0), original)


def test_place_memory_owns_raw_measurement_snapshots():
    calibration, _, features, _ = scene(2)
    atlas = SubmapAtlas(calibration)
    atlas.observe(0, 0, np.eye(4), features[0])
    expected = features[0].points_R.copy()
    features[0].points_R[:] = 0
    np.testing.assert_array_equal(atlas.entries[0][0].points_R, expected)
    assert not atlas.entries[0][0].points_R.flags.writeable


def test_causal_prefix_replay_does_not_depend_on_future_observations():
    calibration, _, features, truth = scene(2)
    first, second = SubmapAtlas(calibration), SubmapAtlas(calibration)
    snapshots = []
    for atlas, count in ((first, 26), (second, 21)):
        current = []
        for frame in range(count):
            segment = int(frame >= 15)
            record = atlas.observe(
                frame,
                segment,
                truth[frame % 2],
                features[frame % 2],
                boundary=frame in (0, 15),
            )
            current.append((atlas.pose(frame).copy(), atlas.components.copy(), record))
        snapshots.append(current)
    for a, b in zip(snapshots[0][:21], snapshots[1], strict=True):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]


@pytest.mark.parametrize(
    "kwargs",
    [{"min_pairs": 5}, {"seed": True}, {"max_trials": 0}, {"min_fraction": np.nan}],
)
def test_registration_config_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        RegistrationConfig(**kwargs)
