"""Protect controlled diagnostic populations and analytic plane geometry."""

import runpy
from pathlib import Path

import numpy as np
import pytest

SCRIPT = runpy.run_path(
    Path(__file__).parents[1] / "experiments/diagnose_heldout_residuals.py"
)


def test_probe_comparison_uses_fixed_intersection_without_residual_trimming():
    observed = np.ones(4)
    predictions = {
        "a": np.array([1.1, 2.4, np.nan, 2]),
        "b": np.array([1.3, 2.5, 1.5, 3]),
    }
    anchor = np.array([True, True, True, False])
    result = SCRIPT["paired_summary"](observed, predictions, anchor)
    assert result["a"]["target_samples"] == result["b"]["target_samples"] == 2
    assert result["a"]["absolute_depth_error_m"]["mean"] == pytest.approx(0.75)
    assert result["b"]["absolute_depth_error_m"]["mean"] == pytest.approx(0.9)
    own = SCRIPT["summarize"](observed, predictions["a"], anchor)["saved_common"]
    assert own["missing_samples"] == 1
    assert own["coverage"] == 2 / 3
    assert own["thresholds"]["0.05"]["explained_fraction_of_target"] == 0


def test_empty_probe_intersection_is_unavailable():
    result = SCRIPT["paired_summary"](
        np.ones(2),
        {"a": np.array([1.0, np.nan]), "b": np.array([np.nan, 1.0])},
        np.ones(2, dtype=bool),
    )
    for metric in result.values():
        assert metric["target_samples"] == 0
        assert metric["absolute_depth_error_m"]["mean"] is None


def test_synthetic_plane_uses_camera_z_depth_under_rotation_and_translation():
    angle = np.pi / 4
    pose = np.array(
        [
            [np.cos(angle), 0, np.sin(angle), 0.1],
            [0, 1, 0, 0.2],
            [-np.sin(angle), 0, np.cos(angle), 0.5],
            [0, 0, 0, 1],
        ]
    )
    K = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=float)
    depth = SCRIPT["plane_depth"](pose, K, (3, 3))
    assert depth[1, 1] == pytest.approx(1.5 / np.cos(angle))
    from scene_recall.geometry.camera import backproject_depth, transform_points

    points = transform_points(backproject_depth(depth, K), pose)
    np.testing.assert_allclose(points[..., 2], 2, atol=1e-14)
    pose[2, 3] = 3
    assert np.isnan(SCRIPT["plane_depth"](pose, K, (3, 3))).all()
