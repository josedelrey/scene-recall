"""Reverse reconstruction checks distinguish free space and true occlusion."""

import runpy
from pathlib import Path

import numpy as np
import pytest

surface_support = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/shared_map_support.py")
)["surface_support"]


def test_unique_voxels_keep_support_contradiction_and_unassessed_populations():
    points = np.array([[0, 0, 2], [0, 0, 1.5], [0, 0, 2.5], [100, 0, 2]])
    K = np.array([[1, 0, 2], [0, 1, 2], [0, 0, 1]])
    result = surface_support(
        points, [(np.eye(4), np.full((5, 5), depth)) for depth in (2.0, 2.2)], K
    )
    assert result["map_points"] == 4
    assert result["target_views"] == 2
    assert result["supported_in_any_view"]["count"] == 1
    assert result["supported_and_free_space"]["count"] == 1
    assert result["free_space_without_support"]["count"] == 1
    assert result["only_behind_measured_surfaces"]["count"] == 1
    assert result["unassessed"]["count"] == 1
    assert result["supported_in_any_view"]["fraction_of_all_voxels"] == 0.25


def test_invalid_depth_and_negative_camera_range_are_unassessed():
    points = np.array([[0, 0, 2], [0, 0, -2]])
    result = surface_support(points, [(np.eye(4), np.full((5, 5), np.nan))], np.eye(3))
    assert result["unassessed"]["count"] == 2
    assert result["projected_to_valid_depth"]["count"] == 0
    empty = surface_support(np.empty((0, 3)), [], np.eye(3))
    assert empty["supported_in_any_view"]["fraction_of_all_voxels"] is None


@pytest.mark.parametrize("tolerance", [0, -1, np.nan])
def test_invalid_surface_tolerance(tolerance):
    with pytest.raises(ValueError):
        surface_support(np.empty((0, 3)), [], np.eye(3), tolerance)
