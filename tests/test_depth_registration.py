"""Numerical checks for the isolated depth registration experiment."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from scene_recall.geometry.camera import backproject_depth, transform_points
from scene_recall.odometry.evaluation import pose_error

experiment = runpy.run_path(
    Path(__file__).parents[1] / "experiments" / "depth_registration.py"
)


def render_room(K, shape, camera_to_world):
    """Intersect camera rays with five analytic room planes."""
    rays = backproject_depth(np.ones(shape), K)
    rays_world = rays @ camera_to_world[:3, :3].T
    origin = camera_to_world[:3, 3]
    depth = np.full(shape, np.inf)
    planes = [
        (np.array([0, 0, 1]), 2.5),
        (np.array([1, 0, 0]), 0.65),
        (np.array([-1, 0, 0]), 0.65),
        (np.array([0, 1, 0]), 0.5),
        (np.array([0, -1, 0]), 0.5),
    ]
    for normal, offset in planes:
        denominator = rays_world @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            intersection = (offset - origin @ normal) / denominator
        intersection[intersection <= 0] = np.inf
        depth = np.minimum(depth, intersection)
    depth[~np.isfinite(depth)] = np.nan
    return depth.astype(np.float32)


def test_point_to_plane_jacobian_matches_independent_finite_difference():
    rng = np.random.default_rng(3)
    moved = rng.normal(size=(20, 3)) + [0, 0, 2]
    normals = rng.normal(size=(20, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    fixed = moved + rng.normal(0, 0.01, moved.shape)
    residual = np.sum((moved - fixed) * normals, axis=1)
    J = experiment["linearization"](
        {"moved": moved, "normals": normals, "residual": residual}, 2
    )[0]
    measured = []
    for k in range(6):
        step = np.zeros(6)
        step[k] = 1e-6
        plus = transform_points(moved, experiment["increment"](step, 2))
        minus = transform_points(moved, experiment["increment"](-step, 2))
        measured.append(np.sum((plus - minus) * normals, axis=1) / 2e-6)
    np.testing.assert_allclose(J, np.array(measured).T, atol=1e-9)


def test_subsampling_preserves_native_rays_and_disjoint_spatial_split():
    K = np.array([[180, 0, 79.5], [0, 180, 63.5], [0, 0, 1.0]])
    depth = np.full((128, 160), 2, dtype=np.float32)
    cloud = experiment["make_cloud"](depth, K, 4)
    expected = backproject_depth(depth, K)[::4, ::4].reshape(-1, 3)
    valid = np.isfinite(cloud["points"]).all(axis=1)
    np.testing.assert_allclose(cloud["points"][valid], expected[valid])
    assert not len(np.intersect1d(cloud["train"], cloud["test"]))
    np.testing.assert_array_equal(
        np.union1d(cloud["train"], cloud["test"]), cloud["all"]
    )
    np.testing.assert_allclose(
        cloud["normals"][cloud["all"]], np.tile([0, 0, 1], (len(cloud["all"]), 1))
    )


def test_single_plane_is_rank_deficient():
    K = np.array([[180, 0, 79.5], [0, 180, 63.5], [0, 0, 1.0]])
    cloud = experiment["make_cloud"](np.full((128, 160), 2, dtype=np.float32), K)
    obs = experiment["observability"](cloud, cloud, np.eye(4))
    assert obs["rank"] == 3
    fit = experiment["register"]([cloud], [cloud], np.eye(4))
    assert fit["levels"][0]["status"] == "rank_deficient"


def test_missing_depth_cannot_produce_a_registration():
    K = np.array([[180, 0, 79.5], [0, 180, 63.5], [0, 0, 1.0]])
    cloud = experiment["make_cloud"](np.full((128, 160), np.nan, dtype=np.float32), K)
    fit = experiment["register"]([cloud], [cloud], np.eye(4))
    assert fit["levels"][0]["status"] == "insufficient_support"
    assert not fit["trace"]
    np.testing.assert_array_equal(fit["pose"], np.eye(4))


@pytest.mark.parametrize("initial_offset", [False, True])
def test_registration_recovers_analytic_motion_in_both_directions(initial_offset):
    K = np.array([[360, 0, 159.5], [0, 360, 127.5], [0, 0, 1.0]])
    shape = (256, 320)
    truth = experiment["increment"](
        np.array([0.018, -0.012, 0.008, 0.007, -0.004, 0.003])
    )
    source_depth = render_room(K, shape, np.eye(4))
    target_depth = render_room(K, shape, np.linalg.inv(truth))
    source = [
        experiment["make_cloud"](source_depth, K, f) for f in experiment["FACTORS"]
    ]
    target = [
        experiment["make_cloud"](target_depth, K, f) for f in experiment["FACTORS"]
    ]
    initial = np.eye(4)
    if initial_offset:
        initial = experiment["increment"](
            np.array([-0.012, 0.009, -0.006, -0.006, 0.003, -0.002])
        )
    forward = experiment["register"](source, target, initial)
    backward = experiment["register"](target, source, np.linalg.inv(initial))
    for estimated, expected in [
        (forward["pose"], truth),
        (backward["pose"], np.linalg.inv(truth)),
    ]:
        translation, rotation = pose_error(estimated, expected)
        assert translation < 0.001
        assert rotation < 0.04
    assert pose_error(backward["pose"] @ forward["pose"], np.eye(4))[0] < 0.001
    assert forward["levels"][-1]["status"] == "step_tolerance"
    assert (
        experiment["observability"](source[-1], target[-1], forward["pose"])["rank"]
        == 6
    )


def test_frozen_correspondence_objective_does_not_increase():
    K = np.array([[360, 0, 159.5], [0, 360, 127.5], [0, 0, 1.0]])
    depth = render_room(K, (256, 320), np.eye(4))
    clouds = [experiment["make_cloud"](depth, K, f) for f in experiment["FACTORS"]]
    fit = experiment["register"](
        clouds,
        clouds,
        experiment["increment"](np.array([0.01, 0.01, -0.01, 0.004, -0.005, 0.002])),
    )
    assert fit["trace"]
    for step in fit["trace"]:
        if step["accepted"]:
            assert (
                step["frozen_huber_after_m2"] <= step["frozen_huber_before_m2"] + 1e-15
            )
    assert pose_error(fit["pose"], np.eye(4))[0] < 0.001
