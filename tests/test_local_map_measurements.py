"""Conditional RGB-D factors and descriptor identity contracts."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_local_map import scene

from scene_recall.features import RGBDFeatures
from scene_recall.multiview.pose_only import retract
from scene_recall.odometry.pipeline import track_observations

M = runpy.run_path(Path(__file__).parents[1] / "experiments/local_map_measurements.py")


@pytest.mark.parametrize("depth", [False, True])
def test_joint_residual_jacobian_and_missing_depth(depth):
    _, _, features, _ = scene(2)
    f = features[0]
    measured = f.points_R.copy()
    measured[:10] = np.nan
    C = M["point_covariance"](f.points_R, np.diag([300, 300, 1.0]))
    K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]])
    args = (f.points_R, f.pixels, measured, C)
    pose = retract(np.eye(4), np.array([0.01, -0.02, 0.03, 0.01, 0.02, -0.01]))
    r, J, W = M["measurement_system"](*args, pose, K, depth)
    for j in range(6):
        delta = np.eye(6)[j] * 1e-7
        plus = M["measurement_system"](*args, retract(pose, delta), K, depth)[0]
        minus = M["measurement_system"](*args, retract(pose, -delta), K, depth)[0]
        np.testing.assert_allclose((plus - minus) / 2e-7, J[..., j], atol=1e-6)
    np.testing.assert_array_equal(J[:10, 2], 0)
    np.testing.assert_array_equal(r[:10, 2], 0)
    assert np.isfinite(W).all()


def test_depth_uncertainty_cancels_along_identical_projection_ray():
    K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]])
    p = np.array([[0.4, 0.2, 2.0]])
    C = M["point_covariance"](p, K)
    _, _, W = M["measurement_system"](
        p, np.array([[220, 150]]), p, C, np.eye(4), K, False
    )
    S = np.linalg.inv(W) @ np.linalg.inv(W).transpose(0, 2, 1)
    np.testing.assert_allclose(S[0, :2, :2], 2 * np.eye(2), atol=1e-12)


def test_family_ratio_uses_distinct_identities_and_preserves_old_appearance():
    target = np.zeros((3, 128), np.float32)
    target[:, 0] = [0, 10, 20]
    descriptors = np.zeros((4, 128), np.float32)
    descriptors[:, 0] = [0.01, 0.02, 7, 20.01]
    matches = M["family_matches"](np.array([1, 1, 1, 2]), descriptors, target)
    assert matches == [(1, 0), (2, 2)]
    # Distinct identities compete, and equal distances fail the ratio test.
    assert M["family_matches"](np.array([1, 2]), descriptors[:2], target) == [(1, 0)]
    assert (
        M["family_matches"](np.array([1, 2]), np.zeros((2, 128), np.float32), target)
        == []
    )


@pytest.mark.parametrize("objective", ["legacy", "rgb", "rgbd"])
@pytest.mark.parametrize("appearance", ["latest", "family"])
def test_online_exact_motion_eviction_outage_and_reference_isolation(
    objective, appearance
):
    calibration, observations, features, truth = scene(18, extrinsic=True)
    f = features[8]
    features[8] = RGBDFeatures(
        f.pixels, f.descriptors, np.full_like(f.points_R, np.nan)
    )
    config = M["MeasurementConfig"](
        objective=objective,
        appearance=appearance,
        max_keyframes=3,
        min_keyframe_gap=1,
        max_keyframe_gap=2,
    )

    def run(obs):
        backend = M["MeasurementMapBackend"](config)
        backend._extract = lambda o, c: features[o.frame_index]
        rows = list(track_observations(obs, calibration, backend))
        return rows, backend

    rows, backend = run(observations)
    repeated, _ = run(
        [replace(o, T_WC_D=np.diag([4.0, 3.0, 2.0, 1.0])) for o in observations]
    )
    np.testing.assert_allclose([r.T_SC_D for r in rows], truth, atol=1e-6)
    np.testing.assert_array_equal(
        [r.T_SC_D for r in rows], [r.T_SC_D for r in repeated]
    )
    assert set(backend._covariances) == set(backend.landmarks)
    assert backend.packets


def test_reconstruction_scores_charge_missing_predictions_and_share_masks(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "experiments"))
    summary = runpy.run_path(root / "experiments/summarize_local_map_reliability.py")
    observed = np.array([[[1.0, 1.0, 1.0]]])
    predictions = {
        "first": np.array([[[1.01, np.nan, 1.03]]]),
        "second": np.array([[[1.02, 1.01, np.nan]]]),
    }
    scores = summary["geometry_scores"](observed, predictions)
    assert scores["first"]["common_samples"] == scores["second"]["common_samples"] == 1
    assert scores["first"]["capped_missing_20cm_m"] == pytest.approx(0.08)
    assert scores["second"]["capped_missing_20cm_m"] == pytest.approx(0.23 / 3)
    assert scores["first"]["coverage"] == pytest.approx(2 / 3)


def test_baseline_links_are_checked_against_recorded_source_range(tmp_path):
    import json

    runner = runpy.run_path(
        Path(__file__).parents[1] / "experiments/run_local_map_reliability.py"
    )
    (tmp_path / "report.json").write_text(
        json.dumps({"source": {"dataset": "scannet_v2", "start": 3500, "stop": 3700}})
    )
    case = {
        "name": "clip",
        "baseline": str(tmp_path),
        "dataset": "scannet_v2",
        "start": 1000,
        "stop": 1200,
    }
    with pytest.raises(ValueError, match="baseline source range differs"):
        runner["validate_baselines"]([case])
    case.update(start=3500, stop=3700)
    runner["validate_baselines"]([case])
