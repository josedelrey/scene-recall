"""Incremental geometry, marginal ownership and reconstruction revision contracts."""

import runpy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scene_recall.data import Observation
from scene_recall.features import RGBDFeatures
from scene_recall.multiview.batch_ba import BatchConfig
from scene_recall.multiview.fixed_landmarks import make_direct_factors
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.multiview.pose_only import PoseOnlyConfig, retract
from scene_recall.slam.graph import (
    linear_system,
    marginalize_oldest,
    pose_chart,
    solve_window,
    square_root_prior,
)
from scene_recall.slam.system import IncrementalSLAM, RevisionArchive, SLAMConfig

scene = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/temporal_controls.py")
)["scene"]


def problem(count=6, **kwargs):
    calibration, tracks, samples, truth = scene(count, **kwargs)
    factors, _ = make_direct_factors(
        tracks,
        samples,
        list(range(count)),
        calibration,
        PoseOnlyConfig(min_grid_coverage=0),
    )
    return calibration.K_R, np.array([f.T_SR for f in tracks.frames]), factors, truth


def subset(factors, mask):
    return replace(
        factors,
        **{name: getattr(factors, name)[mask] for name in factors.__dataclass_fields__},
    )


def test_prior_chart_transport_matches_finite_difference():
    origin = retract(np.eye(4), np.array([1.2, -0.3, 0.6, 0.2, -0.1, 0.3]))
    pose = retract(origin, np.array([0.03, 0.02, -0.01, 0.03, -0.04, 0.02]))
    x, J = pose_chart(pose, origin)
    for axis in range(6):
        step = np.eye(6)[axis] * 1e-6
        plus, _ = pose_chart(retract(pose, step), origin)
        minus, _ = pose_chart(retract(pose, -step), origin)
        np.testing.assert_allclose((plus - minus) / 2e-6, J[:, axis], atol=1e-8)
    np.testing.assert_allclose(retract(origin, x), pose, atol=1e-12)


@pytest.mark.parametrize("depth_model", ["native", "axial"])
def test_joint_window_recovers_geometry_without_references(depth_model):
    K, poses, factors, truth = problem(exact=True, extrinsic=True)
    for i in range(1, len(poses)):
        poses[i] = retract(poses[i], np.array([0.004 * i, 0, 0, 0, 0.002 * i, 0]))
    fingerprint = factors.fingerprint()
    fitted, _, diagnostic = solve_window(
        poses,
        factors,
        K,
        list(range(6)),
        0,
        config=BatchConfig(depth_model=depth_model),
    )
    assert diagnostic["status"] == "refined"
    assert np.all(
        np.diff([diagnostic["initial_cost"], *diagnostic["accepted_costs"]]) < 0
    )
    np.testing.assert_allclose(fitted, truth, atol=3e-7)
    np.testing.assert_array_equal(fitted[0], poses[0])
    assert factors.fingerprint() == fingerprint


def test_marginalization_equals_dense_elimination_without_reusing_raw_factors():
    K, poses, factors, _ = problem(exact=False)
    # Include orphan tracks and tracks born after the root, alongside persistent
    # tracks spanning every camera. The prior must preserve shared point state.
    mask = ~((factors.track_id % 3 == 0) & (factors.view > 0))
    mask &= ~((factors.track_id % 3 == 1) & (factors.view < 2))
    factors = subset(factors, mask)
    ids, _, points = landmark_layout(factors)
    config = BatchConfig(loss="linear")
    _, full_H, full_g = linear_system(
        poses, points, ids, factors, K, config, list(range(6)), None
    )
    prior, keep, retired = marginalize_oldest(
        poses, factors, K, list(range(6)), 0, None, config
    )
    assert set(retired) == set(ids[ids % 3 == 0])
    np.testing.assert_array_equal(keep, factors.view != 0)
    untouched = replace(subset(factors, keep), view=factors.view[keep] - 1)
    remaining_ids, _, remaining_points = landmark_layout(untouched)
    _, combined_H, combined_g = linear_system(
        poses[1:],
        remaining_points,
        remaining_ids,
        untouched,
        K,
        config,
        list(range(1, 6)),
        prior,
    )
    retained_columns = np.array(
        list(range(6, 36))
        + [
            36 + 3 * i + j
            for i, key in enumerate(ids)
            if key not in retired
            for j in range(3)
        ]
    )
    orphan_columns = np.array(
        [
            36 + 3 * i + j
            for i, key in enumerate(ids)
            if key in retired
            for j in range(3)
        ]
    )
    cross = full_H[np.ix_(retained_columns, orphan_columns)]
    old = full_H[np.ix_(orphan_columns, orphan_columns)]
    expected_H = full_H[
        np.ix_(retained_columns, retained_columns)
    ] - cross @ np.linalg.solve(old, cross.T)
    expected_g = full_g[retained_columns] - cross @ np.linalg.solve(
        old, full_g[orphan_columns]
    )
    np.testing.assert_allclose(combined_H, expected_H, rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(combined_g, expected_g, rtol=1e-8, atol=1e-8)
    next_prior, keep, retired = marginalize_oldest(
        poses[1:], untouched, K, list(range(1, 6)), None, prior, config
    )
    assert len(retired) == 0
    expected_H = combined_H[6:, 6:] - combined_H[6:, :6] @ np.linalg.solve(
        combined_H[:6, :6], combined_H[:6, 6:]
    )
    expected_g = combined_g[6:] - combined_H[6:, :6] @ np.linalg.solve(
        combined_H[:6, :6], combined_g[:6]
    )
    untouched = replace(subset(untouched, keep), view=untouched.view[keep] - 1)
    _, H, g = linear_system(
        poses[2:],
        remaining_points,
        remaining_ids,
        untouched,
        K,
        config,
        list(range(2, 6)),
        next_prior,
    )
    np.testing.assert_allclose(H, expected_H, rtol=1e-7, atol=2e-7)
    np.testing.assert_allclose(g, expected_g, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize(
    "failure", ["disconnected", "missing_depth", "invalid_projection"]
)
def test_window_failure_restores_all_camera_and_point_inputs(failure):
    K, poses, factors, _ = problem(exact=True)
    if failure == "disconnected":
        ids = factors.track_id.copy()
        ids[factors.view >= 3] += 1000
        factors = replace(factors, track_id=ids)
    elif failure == "missing_depth":
        factors = replace(
            factors, native_points=np.full_like(factors.native_points, np.nan)
        )
    else:
        poses[2, 2, 3] = 10
    fitted, points, diagnostic = solve_window(poses, factors, K, list(range(6)), 0)
    assert diagnostic["status"] == "rejected"
    np.testing.assert_array_equal(fitted, poses)
    np.testing.assert_array_equal(points, landmark_layout(factors)[2])


def test_prior_retains_nullspace_without_inventing_information():
    H = np.diag([1.0, 2, 0, 0, 0, 0])
    prior = square_root_prior([4], np.eye(4)[None], H, np.array([0.1, 0.2, 0, 0, 0, 0]))
    assert prior.A.shape == (2, 6)
    np.testing.assert_allclose(prior.A.T @ prior.A, H, atol=1e-12)
    with pytest.raises(ValueError, match="outside information range"):
        square_root_prior([4], np.eye(4)[None], H, np.ones(6))
    with pytest.raises(ValueError, match="indefinite"):
        square_root_prior([4], np.eye(4)[None], -np.eye(6), np.zeros(6))


def test_reconstruction_revision_updates_attachments_without_crossing_segments():
    archive = RevisionArchive()
    for i in range(6):
        pose = np.eye(4)
        pose[0, 3] = i * 0.1
        archive.append(pose, keyframe=i % 2 == 0, boundary=i == 3)
    changed = retract(archive.poses[0], np.array([0, 0.02, 0, 0, 0.1, 0]))
    dirty = archive.apply({0: changed})
    assert dirty == [0, 1]
    np.testing.assert_allclose(archive.poses[1], changed @ archive.relative[1])
    assert archive.poses[3][0, 3] == pytest.approx(0.3)
    assert archive.apply({0: changed}) == []
    assert archive.version == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 2},
        {"max_landmarks": 1},
        {"keyframe_interval": True},
        {"exposure": "guess"},
    ],
)
def test_invalid_incremental_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        SLAMConfig(**kwargs)


@pytest.mark.parametrize("outage", ["none", "depth", "rgb"])
@pytest.mark.parametrize("lifecycle", ["legacy", "turnover"])
def test_stream_crosses_lag_boundaries_with_shared_landmarks_and_explicit_losses(
    outage,
    lifecycle,
):
    calibration, tracks, samples, truth = scene(12, exact=True, skew=0)
    rng = np.random.default_rng(8)
    descriptors = rng.normal(size=(96, 128)).astype(np.float32)
    features = {
        i: RGBDFeatures(f.features.pixels, descriptors, f.features.points_R)
        for i, f in enumerate(tracks.frames)
    }
    if outage == "rgb":
        features[4] = RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))
        samples[4] = replace(
            samples[4],
            **{
                name: getattr(samples[4], name)[:0]
                for name in ("points_D", "points_R", "source_index", "whitening_D")
            },
        )
    if outage == "depth":
        for i in (4, 5):
            features[i] = replace(features[i], points_R=np.full((96, 3), np.nan))
            samples[i] = replace(
                samples[i],
                points_D=np.full((96, 3), np.nan),
                points_R=np.full((96, 3), np.nan),
                source_index=np.full(96, -1),
                whitening_D=np.zeros((96, 3, 3)),
            )
    engine = IncrementalSLAM(
        SLAMConfig(
            window_size=3,
            keyframe_interval=1,
            exposure="static",
            marginalization="sqrt" if lifecycle == "turnover" else "schur",
            stale_keyframes=2 if lifecycle == "turnover" else 0,
        )
    )
    diagnostics, losses = [], []
    with (
        patch(
            "scene_recall.odometry.local_map.extract_rgbd_features",
            side_effect=lambda observation, *_: features[observation.frame_index],
        ),
        patch(
            "scene_recall.slam.system.bind_depth",
            side_effect=lambda pixels, depth, *_: samples[int(depth[0, 0]) - 1],
        ),
    ):
        for i in range(12):
            # Deliberately unrelated references cannot affect any geometry.
            reference = np.eye(4)
            reference[0, 3] = 1000 + i
            observation = Observation(
                i,
                np.zeros((240, 320, 3), dtype=np.uint8),
                np.full((240, 320), i + 1, dtype=np.float32),
                float(i / 10),
                reference,
            )
            pose, pair, _, diagnostic = engine.process(observation, calibration)
            diagnostics.append(diagnostic)
            if pair is not None and pair.T_C1C0 is None:
                losses.append(i)
            assert len(engine.poses) <= 3
            assert len(engine.points) <= 128
            if outage != "rgb":
                np.testing.assert_allclose(pose, truth[i], atol=1e-6)
    if outage == "rgb":
        assert losses == [4, 5]
        assert engine.segment_id == 2
        assert min(engine.poses) >= 5
    else:
        assert losses == []
        assert sum("marginalization" in d for d in diagnostics) == 9
        assert not any("graph_restart" in d for d in diagnostics)
        assert len(engine.prior.track_ids) == 96
        assert set(engine.prior.track_ids) == set(engine.points)


def test_canonical_experiment_strips_references_and_exports_replayable_measurements(
    tmp_path, monkeypatch
):
    from helpers.runs import continuous_run

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    from incremental_slam import run_case

    from scene_recall.odometry.io import load_trajectory

    _, baseline = continuous_run(tmp_path)
    seen = []
    process = IncrementalSLAM.process

    def guarded(self, observation, calibration):
        seen.append(observation.T_WC_D)
        return process(self, observation, calibration)

    monkeypatch.setattr(IncrementalSLAM, "process", guarded)
    result = run_case(baseline, tmp_path / "slam", ["native"])
    assert seen == [None] * 3
    report, rows = load_trajectory(tmp_path / "slam/native")
    assert report["backend"] == "incremental_slam"
    assert report["measurement_consistency"]["observations"] == 0
    assert report["evaluation"]["failed_transitions"] == 2
    assert all(row.reference_pose is not None for row in rows)
    assert result["variants"]["native"]["max_active_points"] == 0
    with np.load(tmp_path / "slam/native/measurements.npz") as archive:
        assert archive["final_points"].shape == (0, 3)
    with pytest.raises(ValueError, match="fresh"):
        run_case(baseline, tmp_path / "slam", ["native"])


def test_revised_slam_trajectory_remains_compatible_with_surface_fusion(tmp_path):
    from helpers.runs import continuous_run

    from scene_recall.mapping.heldout import run_heldout
    from scene_recall.mapping.voxel import MappingConfig
    from scene_recall.odometry.io import load_trajectory, save_run

    _, baseline = continuous_run(tmp_path)
    report, rows = load_trajectory(baseline)
    report["backend"] = "incremental_slam"
    save_run(tmp_path / "slam", rows, report, 0)
    fusion = run_heldout(
        [baseline, tmp_path / "slam"],
        tmp_path / "maps",
        config=MappingConfig(keyframe_stride=2, pixel_stride=16),
        offset=1,
    )
    assert set(fusion["odometry"]) == {"sparse", "incremental_slam"}


def test_shared_uncertainty_and_hidden_depth_scale_counterexample(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    from diagnose_slam_prior import diagnose

    result = diagnose()
    shared = result["shared_uncertainty"]
    assert shared["marginal_minus_conditional_minimum_eigenvalue"] > -1e-12
    assert shared["translation_variance_trace_ratio"]["mean"] > 1
    bias = result["hidden_five_percent_depth_scale"]
    assert bias["robust_cost"] < 1e-20
    assert bias["fit"]["status"] == "unchanged"
    assert bias["endpoint_pose_error_m_deg"][0] > 0.004
    assert bias["point_error_m"]["mean"] > 0.1


@pytest.mark.parametrize("lifecycle", ["legacy", "turnover"])
def test_turnover_cannot_permanently_starve_a_full_landmark_budget(
    monkeypatch, lifecycle
):
    from scene_recall.odometry.pipeline import PairEstimate

    calibration, tracks, samples, _ = scene(12, exact=True, skew=0)
    features = {
        i: RGBDFeatures(f.features.pixels[:20], None, f.features.points_R[:20])
        for i, f in enumerate(tracks.frames)
    }
    samples = {
        i: replace(
            sample,
            **{
                name: getattr(sample, name)[:20]
                for name in ("points_D", "points_R", "source_index", "whitening_D")
            },
        )
        for i, sample in samples.items()
    }

    class TurnoverFrontend:
        """A controlled initializer observing disjoint identities in every view."""

        def start(self, observation, calibration):
            i = observation.frame_index
            self._pose = np.eye(4)
            self._previous = observation, features[i]
            self.keyframes = {i: True}
            self.owners = {i: np.arange(20, dtype=np.int64) + 20 * i}

        def estimate(self, source, target, calibration):
            self.start(target, calibration)
            return PairEstimate(np.eye(4))

    engine = IncrementalSLAM(
        SLAMConfig(
            window_size=3,
            keyframe_interval=1,
            max_landmarks=20,
            exposure="static",
            marginalization="sqrt" if lifecycle == "turnover" else "schur",
            stale_keyframes=2 if lifecycle == "turnover" else 0,
        )
    )
    engine.frontend = TurnoverFrontend()
    monkeypatch.setattr(
        "scene_recall.slam.system.bind_depth",
        lambda pixels, depth, *_: samples[int(depth[0, 0]) - 1],
    )
    restarts, admissions = [], []
    for i in range(12):
        observation = Observation(
            i,
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.full((240, 320), i + 1, dtype=np.float32),
        )
        _, _, _, diagnostic = engine.process(observation, calibration)
        if "graph_restart" in diagnostic:
            restarts.append(i)
        if "measurement_count" in diagnostic:
            admissions.append(i)
        assert len(engine.points) <= 20
    if lifecycle == "legacy":
        assert admissions == [0, 3, 6, 9]
        assert restarts == [3, 6, 9]
    else:
        assert admissions == [0, 2, 4, 6, 8, 10]
        assert restarts == []
        assert len(engine.last_seen) <= 20
    assert engine.segment_id == 0


def test_emitted_pairs_follow_emitted_poses(monkeypatch):
    from scene_recall.odometry.pipeline import PairEstimate, TrajectoryFrame

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    from incremental_slam import with_revised_poses

    poses = [np.eye(4) for _ in range(3)]
    poses[1][0, 3], poses[2][0, 3] = 0.1, 0.2
    rows = [
        TrajectoryFrame(
            i,
            None,
            0,
            "initialized" if i == 0 else "tracked",
            np.eye(4),
            None,
            None if i == 0 else PairEstimate(np.eye(4)),
        )
        for i in range(3)
    ]
    emitted = with_revised_poses(rows, poses)
    for row in emitted[1:]:
        assert row.pair.T_C1C0[0, 3] == pytest.approx(-0.1)
    assert rows[1].pair.T_C1C0[0, 3] == 0


def test_map_resume_rejects_changed_input_fingerprints(tmp_path, monkeypatch):
    import json

    from helpers.runs import continuous_run

    from scene_recall.mapping.heldout import run_heldout
    from scene_recall.mapping.voxel import MappingConfig
    from scene_recall.odometry.io import load_trajectory, save_run

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    from incremental_slam import evaluate_maps

    _, baseline = continuous_run(tmp_path)
    report, rows = load_trajectory(baseline)
    report["backend"] = "incremental_slam"
    output = tmp_path / "experiment"
    save_run(output / "native", rows, report, 0)
    path = output / "maps_native"
    fusion = run_heldout(
        [baseline, output / "native"],
        path,
        config=MappingConfig(pixel_stride=8, max_keyframes_per_submap=24),
    )
    fusion["odometry"]["sparse"]["trajectory_sha256"] = "changed"
    (path / "report.json").write_text(json.dumps(fusion))
    with pytest.raises(ValueError, match="saved fusion"):
        evaluate_maps(baseline, output, ["native"])


def test_tsdf_summary_reports_only_evaluated_reference_targets(tmp_path, monkeypatch):
    import json

    from helpers.runs import continuous_run

    from scene_recall.odometry.io import load_trajectory, save_run

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    from summarize_incremental_slam import maps

    _, baseline = continuous_run(tmp_path, missing_reference=1)
    report, rows = load_trajectory(baseline)
    output = tmp_path / "experiment"
    report["backend"] = "incremental_slam"
    save_run(output / "native", rows, report, 0)
    directory = output / "tsdf_native"
    directory.mkdir()
    (directory / "report.json").write_text(
        json.dumps({"status": "evaluated", "target_indices": [1, 2]})
    )
    predicted = np.full((1, 1, 1), 2.0)
    np.savez_compressed(
        directory / "geometry_and_predictions.npz",
        observed=predicted,
        budget_reference_pose=predicted,
        native_reference_pose=predicted,
        gt=predicted,
    )
    result = maps(output, ["native"], "tsdf")
    assert result["target_indices"] == [2]
    assert result["evaluated_views"] == 1
