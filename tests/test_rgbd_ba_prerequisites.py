"""Synthetic timing, gauge, information and frozen-experiment audit contracts."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.audit import (
    epoch_displacement,
    landmark_schur_information,
    normalized_spectrum,
)
from scene_recall.multiview.io import measurement_arrays
from scene_recall.multiview.pose_only import (
    PoseOnlyConfig,
    refine_training_poses,
    retract,
)
from scene_recall.odometry.pipeline import TrajectoryFrame

EXPERIMENT = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/rgbd_ba_prerequisites.py")
)
FIXTURES = runpy.run_path(str(Path(__file__).with_name("test_pose_only.py")))


def test_epoch_displacement_known_motion_and_global_gauge():
    K = np.diag([500.0, 500.0, 1.0])
    points = np.array([[0.0, 0.0, 2.0], [np.nan] * 3])
    pose = retract(np.eye(4), np.array([0.01, 0, 0, 0, 0, 0]))
    shift, depth, excluded = epoch_displacement(points, K, np.eye(4), pose)
    np.testing.assert_allclose(shift, [2.5])
    np.testing.assert_allclose(depth, [0])
    assert excluded == 1
    gauge = retract(np.eye(4), np.array([3.0, 2.0, 1.0, 0.2, 0.1, -0.3]))
    shifted = epoch_displacement(points, K, gauge, gauge @ pose)
    np.testing.assert_allclose(shifted[0], shift, atol=1e-12)
    np.testing.assert_allclose(shifted[1], depth, atol=1e-12)


def test_spectrum_reports_rank_before_damping_and_unit_scaling():
    matrix = np.array([[4.0, 1.0], [1.0, 9.0]])
    spectrum = normalized_spectrum(matrix)
    assert spectrum["nullity"] == 0
    scale = np.diag([100.0, 0.01])
    assert normalized_spectrum(scale @ matrix @ scale)["condition"] == pytest.approx(
        spectrum["condition"]
    )
    singular = normalized_spectrum(np.ones((3, 3)))
    assert singular["condition"] is None
    assert singular["nullity"] == 2


def information_fixture():
    calibration, tracks, _ = FIXTURES["synthetic_tracks"]()
    ids = list(range(6))
    observations = EXPERIMENT["window_observations"](
        tracks, ids, set(range(24)), np.eye(4)
    )
    poses = np.array([f.T_SR for f in tracks.frames])
    return tracks, calibration, poses, observations


def test_schur_matches_finite_difference_information():
    _, calibration, poses, observations = information_fixture()
    config = replace(PoseOnlyConfig(), loss="linear")
    # A few points keep the independent numerical derivative small and local.
    observations = observations[:12]
    dimension = 6 * (len(poses) - 1)
    H = np.zeros((dimension, dimension))
    epsilon = 1e-6
    for point, indices, pixels, depths in observations:
        sigma = np.column_stack(
            (
                np.ones((len(indices), 2)),
                config.depth_sigma_constant_m
                + config.depth_sigma_quadratic * depths**2,
            )
        )

        def residual(
            cameras,
            landmark,
            indices=indices,
            pixels=pixels,
            depths=depths,
            sigma=sigma,
        ):
            q = np.array(
                [transform_points(landmark, np.linalg.inv(cameras[i])) for i in indices]
            )
            raw = np.column_stack(
                (project_points(q, calibration.K_R) - pixels, q[:, 2] - depths)
            )
            return (raw / sigma / np.sqrt(len(indices))).ravel()

        base = residual(poses, point)
        columns = []
        for parameter in range(dimension):
            cameras = poses.copy()
            delta = np.zeros(6)
            delta[parameter % 6] = epsilon
            index = 1 + parameter // 6
            cameras[index] = retract(cameras[index], delta)
            columns.append((residual(cameras, point) - base) / epsilon)
        JP = np.array(columns).T
        JL = np.column_stack(
            [
                (residual(poses, point + epsilon * np.eye(3)[i]) - base) / epsilon
                for i in range(3)
            ]
        )
        cross = JP.T @ JL
        H += JP.T @ JP - cross @ np.linalg.solve(JL.T @ JL, cross.T)
    expected = normalized_spectrum(H)
    actual = landmark_schur_information(poses, observations, calibration.K_R, config)[
        "pose_schur"
    ]
    assert actual["nullity"] == expected["nullity"] == 0
    assert actual["condition"] == pytest.approx(expected["condition"], rel=2e-5)


def test_rgb_only_keeps_scale_gauge_and_depth_constrains_it():
    _, calibration, poses, observations = information_fixture()
    before = poses.copy()
    config = PoseOnlyConfig()
    rgb = landmark_schur_information(
        poses, observations, calibration.K_R, config, depth_scale=None
    )
    rgbd = landmark_schur_information(poses, observations, calibration.K_R, config)
    assert rgb["pose_schur"]["nullity"] == 1
    assert rgb["pose_schur"]["condition"] is None
    assert rgbd["pose_schur"]["nullity"] == 0
    assert rgbd["point_blocks"] == len(observations)
    np.testing.assert_array_equal(poses, before)


def test_missing_depth_has_exactly_zero_information():
    _, calibration, poses, observations = information_fixture()
    missing = [(p, i, u, np.full_like(z, np.nan)) for p, i, u, z in observations]
    config = PoseOnlyConfig()
    rgb = landmark_schur_information(
        poses, observations, calibration.K_R, config, depth_scale=None
    )
    absent = landmark_schur_information(poses, missing, calibration.K_R, config)
    assert absent == rgb


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_information_rejects_invalid_uncertainty(scale):
    _, calibration, poses, observations = information_fixture()
    with pytest.raises(ValueError, match="depth scale"):
        landmark_schur_information(
            poses, observations, calibration.K_R, PoseOnlyConfig(), depth_scale=scale
        )


def test_nonpositive_landmarks_are_excluded_explicitly():
    _, calibration, poses, observations = information_fixture()
    _, indices, pixels, depths = observations[0]
    invalid = (np.array([0.0, 0.0, -0.1]), indices, pixels, depths)
    report = landmark_schur_information(
        poses, [invalid], calibration.K_R, PoseOnlyConfig()
    )
    assert report["excluded_point_blocks"] == 1
    assert report["point_blocks"] == 0
    assert report["pose_schur"]["condition"] is None


def test_asynchrony_can_reduce_residual_while_biasing_perfect_poses():
    synchronous = EXPERIMENT["synchronization_probe"](0.0)
    asynchronous = EXPERIMENT["synchronization_probe"](0.012)
    assert synchronous["translation_error_m"]["max"] < 1e-10
    assert synchronous["rotation_error_deg"]["max"] < 1e-10
    assert asynchronous["window"]["final_cost"] < asynchronous["window"]["initial_cost"]
    assert asynchronous["translation_error_m"]["rmse"] > 0.002
    assert asynchronous["rotation_error_deg"]["rmse"] > 0.04
    repeated = EXPERIMENT["synchronization_probe"](0.012)
    assert repeated == asynchronous


def test_reference_free_archive_roundtrip_preserves_fitting_inputs():
    _, tracks, _ = FIXTURES["synthetic_tracks"]()
    from dataclasses import asdict

    arrays = measurement_arrays(tracks)
    restored = EXPERIMENT["restore_tracks"](arrays, asdict(tracks.config))
    for key, value in measurement_arrays(restored).items():
        np.testing.assert_array_equal(value, arrays[key], err_msg=key)
    assert all(not hasattr(f, "reference_pose") for f in restored.frames)


@pytest.mark.parametrize("extrinsic", [False, True])
def test_closed_single_window_cost_replays_and_reference_poison_is_ignored(extrinsic):
    calibration, tracks, _ = FIXTURES["synthetic_tracks"]()
    if extrinsic:
        T_RD = retract(np.eye(4), np.array([0.04, -0.01, 0.02, 0.02, -0.03, 0.01]))
        calibration = replace(calibration, T_RD=T_RD)
        gauge = np.linalg.inv(T_RD)
        tracks = replace(
            tracks,
            frames=tuple(replace(f, T_SR=gauge @ f.T_SR) for f in tracks.frames),
            landmarks=tuple(
                replace(t, point_S=transform_points(t.point_S, gauge))
                for t in tracks.landmarks
            ),
        )
    config = PoseOnlyConfig()
    refined, diagnostic = refine_training_poses(tracks, calibration, config)
    rows = [
        TrajectoryFrame(
            i,
            i * 0.1,
            0,
            "initialized" if i == 0 else "tracked",
            refined[i] @ calibration.T_RD,
            None,
        )
        for i in range(6)
    ]
    audited, _ = EXPERIMENT["audit_windows"](
        tracks, calibration, config, diagnostic["windows"], rows
    )
    assert audited[0]["cost_after_all_windows"] == pytest.approx(
        audited[0]["cost_when_window_closed"], abs=1e-10
    )
    poison = [replace(r, reference_pose=np.full((4, 4), np.nan)) for r in rows]
    repeated, _ = EXPERIMENT["audit_windows"](
        tracks, calibration, config, diagnostic["windows"], poison
    )
    assert repeated == audited
    corrupt = [{**diagnostic["windows"][0], "factors_sha256": "invalid"}]
    with pytest.raises(ValueError, match="fingerprint"):
        EXPERIMENT["audit_windows"](tracks, calibration, config, corrupt, rows)


def test_synthetic_saved_experiment_audit_and_tamper_rejection(tmp_path):
    from scene_recall.mapping.voxel import MappingConfig
    from scene_recall.multiview.pose_runner import run_pose_experiment

    inputs = runpy.run_path(str(Path(__file__).with_name("test_pose_experiment.py")))
    _, baseline = inputs["synthetic_source"](tmp_path)
    run = tmp_path / "experiment"
    run_pose_experiment(
        baseline,
        run,
        mapping_config=MappingConfig(keyframe_stride=2),
        offset=1,
        controls=False,
    )
    report = EXPERIMENT["audit"](run, verify_source=True)
    assert report["source_replay_all_arrays_exact"]
    assert report["all_protected_inputs_unchanged"]
    assert report["training_views"] == 4
    assert report["timing"]["predicted_static_pixel_shift"]["max"] == 0
    assert report["timing"]["clock_only_views_within_ms"]["1"] == 4
    path = run / "anchored/trajectory.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="trajectory identity"):
        EXPERIMENT["audit"](run)
