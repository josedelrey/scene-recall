"""Reference-free exposure motion, native factors and committed-window controls."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scene_recall.data import Calibration
from scene_recall.datasets.exposure import exposure_times
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.fixed_landmarks import (
    DirectFactors,
    direct_cost,
    direct_information,
    direct_linearize,
    geometric_median,
    make_direct_factors,
    refine_fixed,
    solve_fixed,
)
from scene_recall.multiview.pose_only import PoseOnlyConfig, retract
from scene_recall.multiview.temporal import (
    PairMotion,
    TimingConfig,
    bind_depth,
    native_whitening,
    pair_motion,
)

scene = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/temporal_controls.py")
)["scene"]


def test_clocks_preserve_units_and_unknowns():
    assert exposure_times(
        "tum_rgbd", {"timestamp_color": 0.0, "timestamp_depth": 0.01}
    ) == (0.0, 0.01)
    assert exposure_times(
        "scannet_v2", {"timestamp_color": 0, "timestamp_depth": 0}
    ) == (None, None)
    assert exposure_times(
        "scannet_v2", {"timestamp_color": 2_000_000, "timestamp_depth": 2_001_000}
    ) == (2.0, 2.001)
    assert exposure_times(
        "tum_rgbd", {"timestamp_color": 1.2, "timestamp_depth": 1.21}
    ) == (1.2, 1.21)
    assert not pair_motion([], [], None, None).depth_usable
    assert (
        pair_motion([], [], None, None, TimingConfig(unknown_static=True)).status
        == "assumed_static"
    )


def test_motion_uses_only_bounded_training_intervals():
    poses = [
        retract(np.eye(4), np.array([0.1 * t, 0, 0, 0, 0.2 * t, 0]))
        for t in (0.0, 0.1, 0.2)
    ]
    result = pair_motion(poses, [1.0, 1.1, 1.2], 1.1, 1.112)
    assert result.status == "estimated_interpolated"
    np.testing.assert_allclose(
        result.T_RcRd,
        np.linalg.inv(poses[1])
        @ retract(np.eye(4), np.array([0.0112, 0, 0, 0, 0.0224, 0])),
        atol=1e-12,
    )
    assert result.rotation_sigma_rad > 0
    assert not pair_motion(poses[:2], [1.0, 2.0], 1.0, 1.01).depth_usable
    assert not pair_motion(poses, [1.0, 1.1, 1.2], 1.1, 1.2).depth_usable
    assert (
        pair_motion(poses, [1.0, 1.1, 1.2], 1.0, 0.99).status
        == "estimated_extrapolated"
    )
    assert not pair_motion(poses, [1.0, 1.0, 1.2], 1.0, 1.01).depth_usable


def test_native_binding_is_unique_visible_and_keeps_actual_ray():
    K = np.array([[30.0, 0, 15.5], [0, 30.0, 15.5], [0, 0, 1.0]])
    calibration = Calibration(K, K, np.eye(4), (32, 32), (32, 32))
    pixels = np.array([[15.2, 16.1], [15.3, 16.2], [0.0, 0.0]])
    bound = bind_depth(
        pixels,
        np.full((32, 32), 2.0, np.float32),
        calibration,
        PairMotion(np.eye(4), "known", True),
    )
    assert (bound.source_index >= 0).sum() == 1
    assert bound.source_index[0] == 16 * 32 + 15
    np.testing.assert_allclose(project_points(bound.points_D[:1], K), [[15.0, 16.0]])
    assert not np.allclose(project_points(bound.points_D[:1], K), pixels[:1])
    empty = bind_depth(
        pixels,
        np.full((32, 32), np.nan, np.float32),
        calibration,
        PairMotion(np.eye(4), "known", True),
    )
    assert (empty.source_index == -1).all()


def test_native_sampling_respects_different_rgb_and_depth_grids():
    K_D = np.array([[30.0, 0, 15.5], [0, 30.0, 15.5], [0, 0, 1.0]])
    K_R = np.array([[60.0, 0, 31.5], [0, 60.0, 31.5], [0, 0, 1.0]])
    calibration = Calibration(K_R, K_D, np.eye(4), (64, 64), (32, 32))
    bound = bind_depth(
        np.array([[30.2, 32.1]]),
        np.full((32, 32), 2.0, np.float32),
        calibration,
        PairMotion(np.eye(4), "known", True),
    )
    assert bound.source_index[0] == 16 * 32 + 15
    np.testing.assert_allclose(project_points(bound.points_D, K_D), [[15.0, 16.0]])


def test_motion_footprint_rejects_uncertain_depth_boundary_bindings():
    K = np.array([[30.0, 0, 15.5], [0, 30.0, 15.5], [0, 0, 1.0]])
    calibration = Calibration(K, K, np.eye(4), (32, 32), (32, 32))
    depth = np.full((32, 32), 2.0, np.float32)
    depth[:, 16:] = 4.0
    pixels = np.array([[13.0, 16.0], [7.0, 16.0]])
    motion = PairMotion(np.eye(4), "uncertain", True, 0.001, 0.04)
    original = bind_depth(
        pixels, depth, calibration, motion, TimingConfig(motion_margin_factor=0)
    )
    guarded = bind_depth(pixels, depth, calibration, motion)
    assert (original.source_index >= 0).all()
    assert guarded.source_index[0] == -1
    assert guarded.source_index[1] >= 0


def test_joint_information_detects_a_disconnected_unanchored_component():
    calibration, tracks, samples, truth = scene(6, exact=True)
    factors, _ = make_direct_factors(tracks, samples, list(range(6)), calibration)
    connected = direct_information(truth, factors, calibration.K_R, list(range(1, 6)))
    assert connected["nullity"] == 0
    broken_ids = factors.track_id.copy()
    broken_ids[factors.view >= 3] += 1000
    broken = replace(factors, track_id=broken_ids)
    diagnostic = direct_information(truth, broken, calibration.K_R, list(range(1, 6)))
    assert diagnostic["condition"] is None
    assert diagnostic["nullity"] == 6
    _, fixed = solve_fixed(truth, broken, calibration.K_R, list(range(1, 6)))
    assert fixed["status"] != "rejected"


def test_shared_point_information_matches_numerical_schur():
    from scene_recall.multiview.audit import normalized_spectrum

    calibration, tracks, samples, _ = scene(6)
    factors, _ = make_direct_factors(tracks, samples, list(range(6)), calibration)
    selected = factors.track_id < 12
    factors = DirectFactors(
        **{
            name: getattr(factors, name)[selected]
            for name in factors.__dataclass_fields__
        }
    )
    poses = np.array([f.T_SR for f in tracks.frames])
    config = PoseOnlyConfig(loss="linear")
    free = list(range(1, 6))
    r, J = direct_linearize(poses, factors, calibration.K_R, free)
    _, weights = direct_cost(r, factors, config)
    J = J * np.sqrt(weights[:, :, None, None])
    flat = J.reshape(-1, 30)
    H = flat.T @ flat
    for track in np.unique(factors.track_id):
        mask = factors.track_id == track
        columns = []
        for axis in range(3):
            points = factors.points.copy()
            points[mask, axis] += 1e-6
            changed, _ = direct_linearize(
                poses,
                replace(factors, points=points),
                calibration.K_R,
                free,
                jacobian=False,
            )
            columns.append(
                (
                    (changed[mask] - r[mask]) / 1e-6 * np.sqrt(weights[mask, :, None])
                ).ravel()
            )
        JL = np.array(columns).T
        JP = J[mask].reshape(-1, 30)
        cross = JP.T @ JL
        H -= cross @ np.linalg.solve(JL.T @ JL, cross.T)
    expected = normalized_spectrum(H)
    actual = direct_information(poses, factors, calibration.K_R, free, config)
    assert actual["nullity"] == expected["nullity"] == 0
    assert actual["condition"] == pytest.approx(expected["condition"], rel=5e-5)


def test_geometric_initialization_is_rotation_equivariant():
    points = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [3.0, 4.0, 7.0]]
    )
    gauge = retract(np.eye(4), np.array([4.0, 2.0, 1.0, 0.2, -0.4, 0.3]))
    np.testing.assert_allclose(
        geometric_median(transform_points(points, gauge)),
        transform_points(geometric_median(points), gauge),
        atol=1e-8,
    )


@pytest.mark.parametrize("extrinsic", [False, True])
def test_direct_pose_jacobian_matches_finite_difference(extrinsic):
    calibration, tracks, samples, _ = scene(6, extrinsic=extrinsic)
    factors, _ = make_direct_factors(tracks, samples, list(range(6)), calibration)
    poses = np.array([f.T_SR for f in tracks.frames])
    free = (2, 4)
    residual, J = direct_linearize(poses, factors, calibration.K_R, free)
    for column in range(12):
        moved = poses.copy()
        change = np.zeros(6)
        change[column % 6] = 1e-7
        index = free[column // 6]
        moved[index] = retract(poses[index], change)
        perturbed, _ = direct_linearize(
            moved, factors, calibration.K_R, free, jacobian=False
        )
        np.testing.assert_allclose(
            (perturbed - residual) / 1e-7, J[..., column], rtol=2e-5, atol=3e-4
        )


def test_corrected_native_model_keeps_exact_initialization_under_skew():
    calibration, tracks, samples, truth = scene(6, exact=True)
    factors, _ = make_direct_factors(tracks, samples, list(range(6)), calibration)
    fitted, diagnostic = solve_fixed(truth, factors, calibration.K_R, range(1, 6))
    np.testing.assert_allclose(fitted, truth, atol=1e-11)
    assert diagnostic["final_cost"] < 1e-20
    static = replace(
        factors, T_RcD=np.repeat(calibration.T_RD[None], len(factors.view), axis=0)
    )
    biased, report = solve_fixed(truth, static, calibration.K_R, range(1, 6))
    assert report["final_cost"] < report["initial_cost"]
    assert np.linalg.norm(biased[:, :3, 3] - truth[:, :3, 3]) > 0.001


def test_missing_depth_is_a_zero_block_and_points_never_change():
    calibration, tracks, samples, _ = scene(6)
    factors, _ = make_direct_factors(tracks, samples, list(range(6)), calibration)
    missing = factors.native_points.copy()
    missing[::2] = np.nan
    factors = replace(factors, native_points=missing)
    fingerprint = factors.fingerprint()
    r, J = direct_linearize(
        np.array([f.T_SR for f in tracks.frames]), factors, calibration.K_R, range(1, 6)
    )
    assert not r[::2, 1].any()
    assert not J[::2, 1].any()
    solve_fixed(
        np.array([f.T_SR for f in tracks.frames]), factors, calibration.K_R, range(1, 6)
    )
    assert factors.fingerprint() == fingerprint


def test_committed_overlap_preserves_every_closed_objective():
    calibration, tracks, samples, _ = scene()
    locked, reports, _ = refine_fixed(tracks, samples, calibration)
    for report in reports["windows"]:
        assert report["cost_at_close"] == pytest.approx(
            report["cost_after_all_windows"], abs=1e-9
        )
    np.testing.assert_array_equal(locked[0], tracks.frames[0].T_SR)
    _, opened, _ = refine_fixed(tracks, samples, calibration, lock_overlap=False)
    assert any(
        abs(w["cost_at_close"] - w["cost_after_all_windows"]) > 1e-5
        for w in opened["windows"]
    )


def test_source_roundtrip_and_heldout_measurement_isolation(tmp_path, monkeypatch):
    import scene_recall.multiview.temporal_runner as runner
    from scene_recall.mapping.voxel import MappingConfig

    fixture = runpy.run_path(str(Path(__file__).with_name("test_pose_experiment.py")))
    _, baseline = fixture["synthetic_source"](tmp_path)
    original = runner.verified_observations
    calls = []
    binding = runner.bind_depth

    def poisoned(*args):
        for obs in original(*args):
            fake_reference = np.eye(4)
            fake_reference[:3, 3] = 1e6
            obs = replace(obs, T_WC_D=fake_reference)
            if obs.frame_index % 2:
                obs = replace(
                    obs,
                    rgb=np.zeros_like(obs.rgb),
                    depth=np.full_like(obs.depth, np.nan),
                )
            yield obs

    def checked(pixels, depth, *args):
        assert np.isfinite(depth).any()
        calls.append(len(pixels))
        return binding(pixels, depth, *args)

    monkeypatch.setattr(runner, "verified_observations", poisoned)
    monkeypatch.setattr(runner, "bind_depth", checked)
    report = runner.run_temporal_experiment(
        baseline,
        tmp_path / "temporal",
        mapping_config=MappingConfig(keyframe_stride=2),
        offset=1,
    )
    assert len(calls) == 4
    assert report["repeat_exact"]
    assert set(report["variants"]) == {"clock", "static", "timed", "open"}
    packet_module = runpy.run_path(
        str(Path(__file__).parents[1] / "experiments/replay_temporal_packets.py")
    )

    def forbid_media(*args, **kwargs):
        raise AssertionError("packet replay reopened media")

    with monkeypatch.context() as no_media:
        no_media.setattr(runner, "open_source", forbid_media)
        replayed = packet_module["replay"](tmp_path / "temporal")
    assert all(v["full_rate_trajectory_exact"] for v in replayed["variants"].values())
    from scene_recall.multiview.pose_evaluation import evaluate_pose_maps

    output = tmp_path / "temporal"
    maps = evaluate_pose_maps(
        output,
        output / "maps",
        variants=("clock", "static", "timed", "open"),
        chunk_ids=[0],
    )
    assert maps["evaluated_chunk_ids"] == [0]
    archive = output / "native_measurements.npz"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="measurement archive"):
        evaluate_pose_maps(
            output, output / "tampered_maps", variants=("timed",), chunk_ids=[0]
        )
    assert not (output / "tampered_maps").exists()
    with pytest.raises(ValueError, match="already exists"):
        runner.run_temporal_experiment(baseline, output)


def test_noise_allowance_increases_covariance_without_changing_measurements():
    points = np.array([[0.2, 0.1, 2.0], [-0.3, 0.1, 3.0]])
    K = np.diag([300.0, 310.0, 1.0])
    exact = PairMotion(np.eye(4), "test", True)
    uncertain = replace(exact, translation_sigma_m=0.003, rotation_sigma_rad=0.002)

    def covariance(motion):
        L = np.linalg.inv(native_whitening(points, K, motion))
        return L @ L.transpose(0, 2, 1)

    eigenvalues = np.linalg.eigvalsh(covariance(uncertain) - covariance(exact))
    assert (eigenvalues > 0).all()


def test_weak_configuration_rejects_updates_before_damping():
    calibration, tracks, samples, truth = scene(6, exact=True, weak=True)
    config = PoseOnlyConfig(min_grid_coverage=0.0)
    factors, _ = make_direct_factors(
        tracks, samples, list(range(6)), calibration, config
    )
    fitted, report = solve_fixed(truth, factors, calibration.K_R, range(1, 6), config)
    assert report["status"] == "rejected"
    assert report["condition"] is None
    np.testing.assert_array_equal(fitted, truth)


def test_synthetic_variation_hidden_clock_and_depth_bias_controls():
    control = runpy.run_path(
        str(Path(__file__).parents[1] / "experiments/temporal_controls.py")
    )
    report = control["controls"]()
    asynchronous = report["asynchronous_exact"]
    assert asynchronous["known_motion_control"]["position_m"]["rmse"] < 1e-10
    assert (
        asynchronous["interpolated_motion"]["position_m"]["rmse"]
        < asynchronous["static_motion"]["position_m"]["rmse"]
    )
    assert (
        asynchronous["hidden_clock_error_6ms"]["rotation_deg"]["rmse"]
        > asynchronous["interpolated_motion"]["rotation_deg"]["rmse"]
    )
    assert (
        report["coherent_depth_bias"]["known_motion_control"]["position_m"]["rmse"]
        > 1e-4
    )
    assert all(
        v["solver"]["status"] == "rejected"
        for v in report["weak_collinear_geometry"].values()
    )


def test_motion_and_output_bridge_ignore_references_and_other_segments():
    from scene_recall.multiview.temporal_runner import export_color_poses, motion_table
    from scene_recall.odometry.pipeline import TrajectoryFrame

    calibration, tracks, _, _ = scene(6, exact=True, extrinsic=True)
    rows = []
    for i in range(12):
        local = i % 6
        depth_pose = tracks.frames[local].T_SR @ calibration.T_RD
        # Each segment is rooted in its first depth camera.
        depth_pose = np.linalg.inv(calibration.T_RD) @ depth_pose
        if local == 0:
            depth_pose = np.eye(4)
        rows.append(
            TrajectoryFrame(
                i,
                1 + 0.1 * i,
                i // 6,
                "initialized" if i == 0 else "lost" if i == 6 else "tracked",
                depth_pose,
                None,
            )
        )
    baseline = {
        "source": {"dataset": "tum_rgbd"},
        "frames": [
            {
                "provenance": {
                    "timestamp_color": 1 + 0.1 * i,
                    "timestamp_depth": 1 + 0.1 * i + 0.01,
                }
            }
            for i in range(12)
        ],
    }
    training = [0, 2, 4, 5, 6, 8, 10, 11]
    first, _ = motion_table(baseline, rows, training, calibration, TimingConfig())
    poison = []
    for row in rows:
        pose = row.T_SC_D.copy()
        if row.frame_index not in training:
            pose[:3, 3] += 10
        fake = np.eye(4)
        fake[:3, 3] = 1e6
        poison.append(replace(row, T_SC_D=pose, reference_pose=fake))
    repeated, _ = motion_table(baseline, poison, training, calibration, TimingConfig())
    for i in training:
        np.testing.assert_array_equal(first[i].T_RcRd, repeated[i].T_RcRd)
    poses = {i: rows[i].T_SC_D @ np.linalg.inv(calibration.T_RD) for i in training}
    exported = export_color_poses(rows, poses, calibration, first)
    for i in (0, 6):
        np.testing.assert_array_equal(exported[i].T_SC_D, np.eye(4))
    assert [r.segment_id for r in exported] == [r.segment_id for r in rows]
