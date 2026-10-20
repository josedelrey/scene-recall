"""Batch geometry, exact algebra, observability and training-only packet contracts."""

import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.batch_ba import (
    BatchConfig,
    back_substitute,
    compact_blocks,
    compact_linearize,
    compact_schur,
    solve_batch,
)
from scene_recall.multiview.batch_runner import (
    complete_training_poses,
    make_batch_packets,
    replay_batch_experiment,
    run_batch_experiment,
)
from scene_recall.multiview.fixed_landmarks import direct_cost, make_direct_factors
from scene_recall.multiview.io import measurement_arrays
from scene_recall.multiview.joint_ba import (
    joint_linearize,
    landmark_layout,
    normal_blocks,
    schur_step,
)
from scene_recall.multiview.pose_only import PoseOnlyConfig, retract

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
    return calibration, tracks, samples, factors, truth


def native_table(samples):
    return {
        f"frame_{i}_{field}": getattr(sample, field)
        for i, sample in samples.items()
        for field in sample.__dataclass_fields__
    }


@pytest.mark.parametrize("damping", [0.0, 0.03])
def test_compact_schur_equals_existing_joint_system(damping):
    calibration, tracks, _, factors, _ = problem(extrinsic=True)
    poses = np.array([f.T_SR for f in tracks.frames])
    ids, inverse, points = landmark_layout(factors)
    r, A, B = compact_linearize(poses, points, factors, calibration.K_R, inverse)
    dense_r, dense_A, dense_B = joint_linearize(
        poses, points, factors, calibration.K_R, range(1, 6)
    )
    np.testing.assert_array_equal(r, dense_r)
    np.testing.assert_array_equal(B, dense_B)
    for camera in range(1, 6):
        np.testing.assert_array_equal(
            A[factors.view == camera],
            dense_A[factors.view == camera, :, :, 6 * (camera - 1) : 6 * camera],
        )
    _, weights = direct_cost(r, factors, BatchConfig())
    dense = normal_blocks(r, dense_A, dense_B, weights, inverse, len(ids))
    blocks = compact_blocks(r, A, B, weights, inverse, factors.view, len(ids), 6)
    groups = [np.flatnonzero(inverse == l) for l in range(len(ids))]
    S, rhs, Ci, b = compact_schur(blocks, factors.view, groups, damping)
    dp = np.zeros((6, 6))
    dp[1:] = np.linalg.solve(S, rhs).reshape(-1, 6)
    dl = back_substitute(dp, blocks, inverse, factors.view, Ci, b)
    expected_dp, expected_dl = schur_step(dense, damping)
    np.testing.assert_allclose(dp[1:].ravel(), expected_dp, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(dl, expected_dl, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("model", ["native", "axial"])
def test_pose_and_point_derivatives_with_missing_depth(model):
    calibration, tracks, _, factors, _ = problem(extrinsic=True)
    poses = np.array([f.T_SR for f in tracks.frames])
    native = factors.native_points.copy()
    native[::3] = np.nan
    factors = replace(factors, native_points=native)
    _, inverse, points = landmark_layout(factors)
    r, A, B = compact_linearize(poses, points, factors, calibration.K_R, inverse, model)
    assert not r[::3, 1].any() and not A[::3, 1].any() and not B[::3, 1].any()
    for axis in range(6):
        trial = poses.copy()
        step = np.zeros(6)
        step[axis] = 1e-7
        trial[2] = retract(trial[2], step)
        other, _, _ = compact_linearize(
            trial, points, factors, calibration.K_R, inverse, model
        )
        expected = np.zeros_like(r)
        expected[factors.view == 2] = A[factors.view == 2, :, :, axis]
        np.testing.assert_allclose((other - r) / 1e-7, expected, rtol=1e-4, atol=1e-3)
    for axis in range(3):
        trial = points.copy()
        trial[:, axis] += 1e-7
        other, _, _ = compact_linearize(
            poses, trial, factors, calibration.K_R, inverse, model
        )
        np.testing.assert_allclose(
            (other - r) / 1e-7, B[..., axis], rtol=1e-4, atol=1e-3
        )


def test_persistent_points_recover_exact_geometry_from_pose_drift():
    calibration, tracks, _, factors, truth = problem(exact=True)
    initial = np.array([f.T_SR for f in tracks.frames])
    for i in range(1, 6):
        initial[i] = retract(initial[i], np.array([0.005 * i, 0, 0, 0, 0.002 * i, 0]))
    fingerprint = factors.fingerprint()
    poses, _, report = solve_batch(initial, factors, calibration.K_R)
    assert report["status"] == "refined"
    assert np.all(np.diff(report["accepted_costs"]) < 0)
    np.testing.assert_allclose(poses, truth, atol=2e-7)
    np.testing.assert_array_equal(poses[0], initial[0])
    assert factors.fingerprint() == fingerprint


@pytest.mark.parametrize(
    "failure", ["disconnected", "depthless", "collinear", "root_absent"]
)
def test_unobservable_segment_abstains_without_damping_or_extra_roots(failure):
    calibration, tracks, _, factors, _ = problem(
        exact=True, weak=failure == "collinear"
    )
    initial = np.array([f.T_SR for f in tracks.frames])
    if failure == "disconnected":
        ids = factors.track_id.copy()
        ids[factors.view >= 3] += 1000
        factors = replace(factors, track_id=ids)
    if failure == "depthless":
        factors = replace(
            factors, native_points=np.full_like(factors.native_points, np.nan)
        )
    if failure == "root_absent":
        mask = factors.view != 0
        factors = replace(
            factors,
            **{
                name: getattr(factors, name)[mask]
                for name in factors.__dataclass_fields__
            },
        )
    poses, points, report = solve_batch(initial, factors, calibration.K_R)
    assert report["status"] == "rejected"
    np.testing.assert_array_equal(poses, initial)
    np.testing.assert_array_equal(points, landmark_layout(factors)[2])


def test_packet_selection_ignores_world_seeds_and_prunes_support_to_fixed_point():
    calibration, tracks, samples, _, _ = problem()
    graph = measurement_arrays(tracks)
    graph["point_S"][:] = np.nan
    graph["T_reference"] = np.full((6, 4, 4), np.nan)
    base, segments, packets = make_batch_packets(
        graph, native_table(samples), calibration
    )
    assert len(packets) == 1 and len(base) == 6 and set(segments.values()) == {0}
    ids, _, factors, _ = packets[0]
    assert ids == list(range(6))
    assert np.isfinite(factors.points).all()
    assert len(set(factors.track_id)) == 96
    # Iterative support pruning cannot leave stale supported cameras or orphaned tracks.
    offsets = graph["feature_offsets"]
    graph["pixels"][offsets[2] : offsets[3]] = 1
    _, _, packets = make_batch_packets(graph, native_table(samples), calibration)
    assert 2 in packets[0][3]["unsupported_frame_indices"]
    assert 2 not in packets[0][0]


def test_correction_completion_interpolates_missing_cameras_and_carries_tail():
    base = {i: np.eye(4) for i in range(8)}
    segments = {i: int(i >= 6) for i in base}
    endpoint = np.eye(4)
    endpoint[0, 3] = 0.04
    poses = complete_training_poses(base, segments, {0: base[0], 4: endpoint})
    assert poses[2][0, 3] == pytest.approx(0.02)
    assert poses[5][0, 3] == pytest.approx(0.04)
    for i in (6, 7):
        np.testing.assert_array_equal(poses[i], base[i])


def test_batch_respects_center_point_bounds_and_rigid_gauge():
    calibration, tracks, _, factors, _ = problem()
    initial = np.array([f.T_SR for f in tracks.frames])
    config = BatchConfig(max_translation_m=1e-5, max_point_displacement_m=1e-5)
    poses, points, report = solve_batch(initial, factors, calibration.K_R, config)
    assert np.linalg.norm(poses[:, :3, 3] - initial[:, :3, 3], axis=1).max() <= 1e-5
    assert np.linalg.norm(points - landmark_layout(factors)[2], axis=1).max() <= 1e-5
    assert "update bound" in report["rejected_trials"]
    expected, expected_points, _ = solve_batch(initial, factors, calibration.K_R)
    gauge = retract(np.eye(4), np.array([0.5, -0.1, 0.2, 0.1, -0.1, 0.1]))
    changed = replace(factors, points=transform_points(factors.points, gauge))
    other, other_points, _ = solve_batch(gauge @ initial, changed, calibration.K_R)
    np.testing.assert_allclose(other, gauge @ expected, atol=2e-7)
    np.testing.assert_allclose(
        other_points, transform_points(expected_points, gauge), atol=2e-7
    )


def test_packet_only_run_and_maps_preserve_cohort_and_reject_tampering(
    tmp_path, monkeypatch
):
    from scene_recall.mapping.voxel import MappingConfig
    from scene_recall.multiview.pose_evaluation import evaluate_pose_maps
    from scene_recall.multiview.temporal_runner import run_temporal_experiment

    fixture = runpy.run_path(str(Path(__file__).with_name("test_pose_experiment.py")))
    _, baseline = fixture["synthetic_source"](tmp_path)
    fixed, output = tmp_path / "fixed", tmp_path / "batch"
    run_temporal_experiment(
        baseline, fixed, mapping_config=MappingConfig(keyframe_stride=2), offset=1
    )

    def forbid(*args, **kwargs):
        raise AssertionError("batch optimizer opened media")

    with monkeypatch.context() as m:
        m.setattr("scene_recall.multiview.temporal_runner.open_source", forbid)
        report = run_batch_experiment(fixed, output)
    assert report["repeat_exact"]
    assert replay_batch_experiment(output)["full_rate_trajectory_exact"]
    maps = evaluate_pose_maps(
        output, tmp_path / "maps", variants=("clock", "batch"), chunk_ids=[0]
    )
    assert maps["split"] == report["split"]
    with pytest.raises(ValueError, match="already exists"):
        run_batch_experiment(fixed, output)
    published = output / "batch_factors.npz"
    original = published.read_bytes()
    published.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="published batch input"):
        replay_batch_experiment(output)
    published.write_bytes(original)
    archive = fixed / "training_measurements.npz"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="input changed"):
        run_batch_experiment(fixed, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_unknown_clocks_are_not_silently_assumed_static(tmp_path):
    from scene_recall.mapping.voxel import MappingConfig
    from scene_recall.multiview.temporal_runner import run_temporal_experiment
    from scene_recall.run_metadata import sha256

    fixture = runpy.run_path(str(Path(__file__).with_name("test_pose_experiment.py")))
    _, baseline = fixture["synthetic_source"](tmp_path)
    fixed = tmp_path / "fixed"
    run_temporal_experiment(
        baseline, fixed, mapping_config=MappingConfig(keyframe_stride=2), offset=1
    )
    path = fixed / "native_measurements.npz"
    with np.load(path) as a:
        arrays = {k: a[k] for k in a.files}
    arrays["color_timestamp_s"][:] = np.nan
    np.savez_compressed(path, **arrays)
    path = fixed / "report.json"
    report = json.loads(path.read_text())
    report["native_measurements_sha256"] = sha256(fixed / "native_measurements.npz")
    report["timing_config"]["unknown_static"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="known reported"):
        run_batch_experiment(fixed, tmp_path / "bad")


@pytest.mark.parametrize("missing_anchor", [False, True])
def test_partial_reference_rendering_needs_anchor_and_keeps_default_policy(
    tmp_path, missing_anchor
):
    from scene_recall.mapping.voxel import MappingConfig
    from scene_recall.multiview.pose_evaluation import evaluate_pose_maps
    from scene_recall.multiview.temporal_runner import run_temporal_experiment
    from scene_recall.odometry.runner import run_odometry
    from scene_recall.odometry.sparse import SparseConfig

    fixture = runpy.run_path(str(Path(__file__).with_name("test_pose_experiment.py")))
    source, _ = fixture["synthetic_source"](tmp_path)
    groundtruth = source / "groundtruth.txt"
    lines = groundtruth.read_text().splitlines()
    groundtruth.write_text(
        "\n".join(
            line
            for i, line in enumerate(lines)
            if i != 2 and (i != 0 or not missing_anchor)
        )
        + "\n"
    )
    baseline = tmp_path / "missing_reference_baseline"
    run_odometry(
        source,
        baseline,
        dataset="tum_rgbd",
        backend_name="sparse",
        config=SparseConfig(max_features=300),
    )
    fixed, batch = tmp_path / "fixed", tmp_path / "batch"
    run_temporal_experiment(
        baseline, fixed, mapping_config=MappingConfig(keyframe_stride=2), offset=1
    )
    run_batch_experiment(fixed, batch)
    strict = evaluate_pose_maps(batch, tmp_path / "strict", variants=("batch",))
    assert not strict["pooled"]
    assert "reference_policy" not in strict
    partial = evaluate_pose_maps(
        batch,
        tmp_path / "partial",
        variants=("batch",),
        allow_partial_training_reference=True,
    )
    assert (
        partial["reference_policy"] == "valid_first_training_anchor_and_scored_target"
    )
    assert bool(partial["pooled"]) is not missing_anchor
    for chunk in partial["chunks"]:
        for submap in chunk["maps"].values():
            assert submap["comparison_status"] == "unavailable_unpaired_references"
