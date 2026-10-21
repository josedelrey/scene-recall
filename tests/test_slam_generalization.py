"""Scientific contracts of the frozen-objective prediction experiment."""

import importlib.util
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scene_recall.multiview.batch_ba import BatchConfig
from scene_recall.multiview.fixed_landmarks import make_direct_factors
from scene_recall.multiview.pose_only import PoseOnlyConfig

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "generalization", ROOT / "experiments/slam_generalization.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
scene = runpy.run_path(str(ROOT / "experiments/temporal_controls.py"))["scene"]


def fixture():
    calibration, tracks, samples, truth = scene(4, exact=True, skew=0)
    factors, _ = make_direct_factors(
        tracks,
        samples,
        list(range(4)),
        calibration,
        PoseOnlyConfig(min_grid_coverage=0),
    )
    return calibration, factors, truth


def test_metric_seed_ignores_saved_points_and_later_samples():
    _, factors, poses = fixture()
    seeded = study.seed_from_first_depth(factors, poses)
    changed = factors.native_points.copy()
    changed[factors.view > 0] *= 1.5
    poisoned = replace(factors, points=factors.points + 100, native_points=changed)
    np.testing.assert_array_equal(
        study.seed_from_first_depth(poisoned, poses).points, seeded.points
    )


def test_whole_identity_holdout_does_not_enter_training():
    calibration, factors, poses = fixture()
    train, test = study.split_tracks(factors)
    assert set(train.track_id).isdisjoint(test.track_id)
    assert len(train.view) + len(test.view) == len(factors.view)
    original = train.fingerprint()
    changed = factors.pixels.copy()
    changed[factors.track_id % 4 == 0] += 50
    poisoned = replace(factors, pixels=changed)
    assert study.split_tracks(poisoned)[0].fingerprint() == original
    before = study.prediction_errors(poses, test, calibration.K_R)
    after = study.prediction_errors(
        poses, study.split_tracks(poisoned)[1], calibration.K_R
    )
    assert before["rgb_px"]["p95"] < 1e-10
    assert after["rgb_px"]["median"] > 50


@pytest.mark.parametrize("model", study.MODELS)
def test_exact_geometry_is_stationary_for_all_models(model):
    from scene_recall.multiview.batch_ba import solve_batch

    calibration, factors, poses = fixture()
    factors = study.seed_from_first_depth(factors, poses)
    modified = study.model_factors(factors, model)
    fitted, _, result = solve_batch(
        poses,
        modified,
        calibration.K_R,
        BatchConfig(
            max_iterations=5, depth_model="axial" if model == "axial" else "native"
        ),
    )
    assert result["status"] != "rejected"
    np.testing.assert_allclose(fitted, poses, atol=1e-10)
    assert (
        study.prediction_errors(fitted, factors, calibration.K_R)["axial_m"]["p95"]
        < 1e-10
    )


def test_prediction_is_gauge_invariant_and_excludes_seed():
    from scene_recall.multiview.pose_only import retract

    calibration, factors, poses = fixture()
    changed = factors.pixels.copy()
    changed[factors.view == 0] += 50
    factors = replace(factors, pixels=changed)
    before = study.prediction_errors(poses, factors, calibration.K_R)
    transform = retract(np.eye(4), np.array([1, 2, -1, 0.1, 0.2, -0.1]))
    after = study.prediction_errors(transform @ poses, factors, calibration.K_R)
    assert before["rgb_px"]["p95"] < 1e-10
    assert after["rgb_px"]["p95"] < 1e-10
    assert before["rgb_px"]["count"] == 3 * len(np.unique(factors.track_id))


def test_connectivity_reports_independent_gauges():
    _, factors, _ = fixture()
    assert study.camera_components(factors, 4) == [[0, 1, 2, 3]]
    changed = replace(factors, track_id=factors.track_id + 1000 * (factors.view >= 2))
    assert study.camera_components(changed, 5) == [[0, 1], [2, 3], [4]]


def test_first_depth_ablation_preserves_rgb_and_source():
    _, factors, _ = fixture()
    fingerprint = factors.fingerprint()
    modified = study.model_factors(factors, "first_depth")
    assert np.isfinite(modified.native_points).all(axis=1).sum() == len(
        np.unique(factors.track_id)
    )
    np.testing.assert_array_equal(modified.pixels, factors.pixels)
    assert factors.fingerprint() == fingerprint


def test_component_solver_recovers_relative_geometry_without_joining_gauges(
    monkeypatch,
):
    from scene_recall.multiview.batch_ba import solve_batch
    from scene_recall.multiview.pose_only import retract

    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    from slam_component_replay import solve_components

    calibration, factors, truth = fixture()
    factors = replace(factors, track_id=factors.track_id + 1000 * (factors.view >= 2))
    initial = truth.copy()
    for camera in (1, 3):
        initial[camera] = retract(
            initial[camera], np.array([0.005, -0.003, 0.002, 0.001, -0.001, 0])
        )
    factors = study.seed_from_first_depth(factors, initial)
    _, _, rejected = solve_batch(initial, factors, calibration.K_R)
    assert rejected["status"] == "rejected"
    fitted, _, result = solve_components(initial, factors, calibration.K_R)
    assert result["status"] == "refined"
    assert result["gauge_roots"] == [0, 2]
    np.testing.assert_array_equal(fitted[[0, 2]], initial[[0, 2]])
    np.testing.assert_allclose(fitted, truth, atol=1e-5)


def test_component_solver_equals_connected_batch(monkeypatch):
    from scene_recall.multiview.batch_ba import solve_batch

    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    from slam_component_replay import solve_components

    calibration, factors, truth = fixture()
    full = solve_batch(truth, factors, calibration.K_R)
    component = solve_components(truth, factors, calibration.K_R)
    np.testing.assert_array_equal(full[0], component[0])
    np.testing.assert_array_equal(full[1], component[1])


def test_interpolation_preserves_keyframes_segments_and_trailing_poses(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    from slam_pose_interpolation import interpolate_poses

    poses = np.repeat(np.eye(4)[None], 9, axis=0)
    poses[:, 0, 3] = [0, 0.1, 0.1, 0.1, 1, 10, 10.5, 11, 20]
    segments = [0] * 5 + [1] * 4
    keys = [0, 4, 5, 7]
    fitted = interpolate_poses(poses, list(range(9)), segments, keys)
    np.testing.assert_array_equal(fitted[keys], poses[keys])
    np.testing.assert_array_equal(fitted[-1], poses[-1])
    np.testing.assert_allclose(fitted[:5, 0, 3], [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(fitted[5:8, 0, 3], [10, 10.5, 11])


def test_synthetic_interventions_have_paired_initialization(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    from slam_model_controls import SCENARIOS, generate

    initial, _, _, truth = generate(3, "near", "mixed", "independent")
    for scenario in SCENARIOS:
        other, _, _, reference = generate(3, "near", "mixed", scenario, tracks=384)
        np.testing.assert_array_equal(other, initial)
        np.testing.assert_array_equal(reference, truth)


def test_replay_rejects_mismatched_source_before_reading_factors(monkeypatch, tmp_path):
    reports = [
        ({"source": {"identity": 1}, "calibration": {}}, []),
        ({"source": {"identity": 2}, "calibration": {}}, []),
    ]
    monkeypatch.setattr(study, "load_trajectory", lambda _: reports.pop(0))
    with pytest.raises(ValueError, match="canonical source and calibration"):
        study.run_case(tmp_path / "run", tmp_path / "initial", BatchConfig(), 8)
