"""Joint derivatives, Schur elimination, gauges and scientific packet boundaries."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scene_recall.multiview.fixed_landmarks import (
    direct_cost,
    direct_linearize,
    make_direct_factors,
)
from scene_recall.multiview.joint_ba import (
    joint_linearize,
    landmark_layout,
    normal_blocks,
    schur_step,
    solve_joint,
)
from scene_recall.multiview.pose_only import PoseOnlyConfig

scene = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/temporal_controls.py")
)["scene"]


def problem(**kwargs):
    calibration, tracks, samples, truth = scene(6, **kwargs)
    factors, _ = make_direct_factors(
        tracks,
        samples,
        list(range(6)),
        calibration,
        PoseOnlyConfig(min_grid_coverage=0),
    )
    return calibration.K_R, np.array([f.T_SR for f in tracks.frames]), factors, truth


def test_point_jacobian_matches_finite_differences_with_native_extrinsic():
    K, initial, factors, _ = problem(extrinsic=True)
    ids, inverse, points = landmark_layout(factors)
    residual, _, JL = joint_linearize(initial, points, factors, K, (1, 3, 5))
    for axis in range(3):
        perturbed = points.copy()
        perturbed[:, axis] += 1e-7
        r, _, _ = joint_linearize(initial, perturbed, factors, K, (1, 3, 5))
        np.testing.assert_allclose(
            (r - residual) / 1e-7, JL[..., axis], rtol=2e-5, atol=3e-4
        )
    assert len(ids) < len(inverse)


def test_schur_step_equals_full_damped_joint_solve():
    K, initial, factors, _ = problem()
    ids, inverse, points = landmark_layout(factors)
    r, JP, JL = joint_linearize(initial, points, factors, K, (1, 3))
    _, weights = direct_cost(r, factors, PoseOnlyConfig())
    blocks = normal_blocks(r, JP, JL, weights, inverse, len(ids))
    Hpp, Hpl, Hll, gp, gl = blocks
    dimension = len(gp)
    H = np.zeros((dimension + 3 * len(ids),) * 2)
    H[:dimension, :dimension] = Hpp
    for i in range(len(ids)):
        s = slice(dimension + i * 3, dimension + (i + 1) * 3)
        H[s, s] = Hll[i]
        H[:dimension, s] = Hpl[i]
        H[s, :dimension] = Hpl[i].T
    damping = 0.03
    H += damping * np.diag(np.maximum(np.diag(H), 1e-15))
    expected = np.linalg.solve(H, -np.r_[gp, gl.ravel()])
    dp, dl = schur_step(blocks, damping)
    np.testing.assert_allclose(np.r_[dp, dl.ravel()], expected, rtol=1e-9, atol=1e-11)


def test_shared_points_reduce_cost_without_mutation_and_preserve_fixed_cameras():
    K, initial, factors, _ = problem()
    original = factors.fingerprint()
    poses, points, report = solve_joint(initial, factors, K, (2, 3, 4, 5))
    assert report["status"] == "refined"
    assert report["final_cost"] < report["initial_cost"]
    assert np.all(np.diff(report["accepted_costs"]) < 0)
    np.testing.assert_array_equal(poses[:2], initial[:2])
    assert factors.fingerprint() == original
    _, inverse, start = landmark_layout(factors)
    assert np.linalg.norm(points - start) > 0
    r, _ = direct_linearize(
        poses, replace(factors, points=points[inverse]), K, (), jacobian=False
    )
    assert direct_cost(r, factors, PoseOnlyConfig())[0] == report["final_cost"]


def test_exact_geometry_and_missing_depth_blocks():
    K, initial, factors, truth = problem(exact=True, extrinsic=True)
    native = factors.native_points.copy()
    native[::2] = np.nan
    factors = replace(factors, native_points=native)
    _, _, points = landmark_layout(factors)
    r, _, JL = joint_linearize(initial, points, factors, K, range(1, 6))
    assert not r[::2, 1].any() and not JL[::2, 1].any()
    poses, _, report = solve_joint(initial, factors, K, range(1, 6))
    np.testing.assert_allclose(poses, truth, atol=1e-11)
    assert report["final_cost"] < 1e-20


@pytest.mark.parametrize("failure", ["weak", "depthless", "disconnected"])
def test_undamped_observability_failure_rolls_back(failure):
    K, initial, factors, _ = problem(exact=True, weak=failure == "weak")
    if failure == "depthless":
        factors = replace(
            factors, native_points=np.full_like(factors.native_points, np.nan)
        )
    if failure == "disconnected":
        ids = factors.track_id.copy()
        ids[factors.view >= 3] += 1000
        factors = replace(factors, track_id=ids)
    poses, points, report = solve_joint(initial, factors, K, range(1, 6))
    assert report["status"] == "rejected"
    np.testing.assert_array_equal(poses, initial)
    np.testing.assert_array_equal(points, landmark_layout(factors)[2])


def test_joint_is_equivariant_to_rigid_world_coordinates():
    from scene_recall.geometry.camera import transform_points
    from scene_recall.multiview.pose_only import retract

    K, initial, factors, _ = problem()
    gauge = retract(np.eye(4), np.array([0.5, -0.1, 0.2, 0.1, -0.1, 0.1]))
    poses, points, _ = solve_joint(initial, factors, K, range(1, 6))
    changed = replace(factors, points=transform_points(factors.points, gauge))
    other, other_points, _ = solve_joint(gauge @ initial, changed, K, range(1, 6))
    np.testing.assert_allclose(other, gauge @ poses, atol=2e-7)
    np.testing.assert_allclose(other_points, transform_points(points, gauge), atol=2e-7)


def test_joint_validation_and_bounds():
    K, initial, factors, _ = problem()
    with pytest.raises(ValueError, match="non-root"):
        solve_joint(initial, factors, K, (0, 1))
    with pytest.raises(ValueError, match="positive"):
        solve_joint(initial, factors, K, (1,), max_point_displacement_m=0)
    points = factors.points.copy()
    points[0, 0] += 0.01
    with pytest.raises(ValueError, match="inconsistent"):
        solve_joint(initial, replace(factors, points=points), K, (1,))
    _, fitted, report = solve_joint(
        initial, factors, K, range(1, 6), max_point_displacement_m=1e-5
    )
    assert np.linalg.norm(fitted - landmark_layout(factors)[2], axis=1).max() <= 1e-5
    assert "update bound" in report["rejected_trials"]


def test_synthetic_noise_bias_timing_and_scale_controls(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    from joint_ba_controls import controls

    reports = controls(seeds=(51,))
    assert reports == controls(seeds=(51,))
    for name in ("weak_collinear", "depthless_scale_gauge"):
        assert reports[name][0]["methods"]["joint"]["solver"]["status"] == "rejected"
    exact = reports["exact"][0]["methods"]["joint"]
    assert exact["position_m"]["rmse"] < 1e-10
    biased = reports["depth_bias_4cm"][0]["methods"]["joint"]
    assert biased["landmark_m"]["rmse"] > 0.03
    assert (
        reports["hidden_latency_6ms"][0]["methods"]["joint"]["position_m"]["rmse"]
        > 1e-5
    )


def test_packet_roundtrip_map_cohort_and_tamper_rejection(tmp_path, monkeypatch):
    import scene_recall.multiview.temporal_runner as temporal
    from scene_recall.mapping.voxel import MappingConfig
    from scene_recall.multiview.joint_runner import run_joint_experiment
    from scene_recall.multiview.pose_evaluation import evaluate_pose_maps
    from scene_recall.multiview.temporal_runner import run_temporal_experiment

    fixture = runpy.run_path(str(Path(__file__).with_name("test_pose_experiment.py")))
    _, baseline = fixture["synthetic_source"](tmp_path)
    fixed = tmp_path / "fixed"
    run_temporal_experiment(
        baseline, fixed, mapping_config=MappingConfig(keyframe_stride=2), offset=1
    )

    def forbid(*args, **kwargs):
        raise AssertionError("joint optimizer opened media")

    with monkeypatch.context() as m:
        m.setattr(temporal, "open_source", forbid)
        result = run_joint_experiment(fixed, tmp_path / "joint")
    assert result["repeat_exact"]
    assert result["acquisition_policy"] == "known_reported_clocks"
    maps = evaluate_pose_maps(
        tmp_path / "joint",
        tmp_path / "maps",
        variants=("timed", "joint"),
        chunk_ids=[0],
    )
    assert maps["split"] == result["split"]
    archive = fixed / "timed_factors.npz"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="input changed"):
        run_joint_experiment(fixed, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_packet_windows_preserve_segments_and_closed_geometry():
    from scene_recall.multiview.fixed_landmarks import refine_fixed
    from scene_recall.multiview.joint_runner import fit_packets

    calibration, tracks, samples, _ = scene(12)
    frames = tuple(
        replace(f, segment_id=int(f.frame_index >= 8)) for f in tracks.frames
    )
    landmarks = []
    for track in tracks.landmarks:
        for segment, selected in ((0, range(8)), (1, range(8, 12))):
            keys = tuple(k for k in track.observations if k[0] in selected)
            landmarks.append(
                replace(
                    track,
                    track_id=track.track_id + 1000 * segment,
                    segment_id=segment,
                    observations=keys,
                )
            )
    tracks = replace(tracks, frames=frames, landmarks=tuple(landmarks))
    _, fixed, archive = refine_fixed(tracks, samples, calibration)
    base = {f.frame_index: f.T_SR for f in frames}
    segments = {f.frame_index: f.segment_id for f in frames}
    poses, joint, _ = fit_packets(
        base, segments, fixed["windows"], archive, calibration.K_R, PoseOnlyConfig()
    )
    assert any("paired_joint" in w for w in joint["windows"])
    for w in joint["windows"]:
        if "cost_at_close" in w:
            assert w["cost_at_close"] == pytest.approx(
                w["cost_after_all_windows"], abs=1e-9
            )
    for root in (0, 8):
        np.testing.assert_array_equal(poses[root], base[root])


def test_unknown_clocks_need_both_packet_assumption_and_runner_opt_in(tmp_path):
    import json

    from scene_recall.mapping.voxel import MappingConfig
    from scene_recall.multiview.joint_runner import (
        replay_joint_experiment,
        run_joint_experiment,
    )
    from scene_recall.multiview.temporal_runner import run_temporal_experiment
    from scene_recall.run_metadata import sha256

    fixture = runpy.run_path(str(Path(__file__).with_name("test_pose_experiment.py")))
    _, baseline = fixture["synthetic_source"](tmp_path)
    fixed = tmp_path / "fixed"
    run_temporal_experiment(
        baseline, fixed, mapping_config=MappingConfig(keyframe_stride=2), offset=1
    )
    path = fixed / "native_measurements.npz"
    with np.load(path, allow_pickle=False) as a:
        arrays = {key: a[key] for key in a.files}
    arrays["color_timestamp_s"][:] = np.nan
    arrays["depth_timestamp_s"][:] = np.nan
    np.savez_compressed(path, **arrays)
    report_path = fixed / "report.json"
    report = json.loads(report_path.read_text())
    report["native_measurements_sha256"] = sha256(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="unknown acquisition"):
        run_joint_experiment(fixed, tmp_path / "bad")
    with pytest.raises(ValueError, match="unknown acquisition"):
        run_joint_experiment(fixed, tmp_path / "bad", allow_unknown_static=True)
    report["timing_config"]["unknown_static"] = True
    report_path.write_text(json.dumps(report))
    output = tmp_path / "assumed"
    result = run_joint_experiment(fixed, output, allow_unknown_static=True)
    assert result["acquisition_policy"] == "explicit_unknown_static_assumption"
    assert replay_joint_experiment(output)["full_rate_trajectory_exact"]
    published = output / "joint_factors.npz"
    published.write_bytes(published.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="published joint input"):
        replay_joint_experiment(output)


def test_a_singular_landmark_rejects_the_window_without_dropping_factors():
    K, initial, factors, _ = problem()
    track = factors.track_id[0]
    keep = (factors.track_id != track) | (factors.view == 0)
    fields = {
        name: getattr(factors, name)[keep].copy()
        for name in factors.__dataclass_fields__
    }
    fields["native_points"][fields["track_id"] == track] = np.nan
    broken = type(factors)(**fields)
    fingerprint = broken.fingerprint()
    poses, _, report = solve_joint(initial, broken, K, range(1, 6))
    assert report["status"] == "rejected"
    assert report["information"]["excluded_point_blocks"] == 1
    np.testing.assert_array_equal(poses, initial)
    assert broken.fingerprint() == fingerprint
