from dataclasses import replace

import cv2
import numpy as np
import pytest

from scene_recall.data import Calibration, Observation
from scene_recall.geometry.camera import (
    project_points,
    register_depth_to_rgb,
    reject_depth_edges,
    transform_points,
)
from scene_recall.odometry.evaluation import pose_error
from scene_recall.odometry.pipeline import PairEstimate, track_observations
from scene_recall.odometry.sparse import (
    SparseConfig,
    SparseRGBDBackend,
    _Features,
    estimate_sparse_pose,
    reprojection_errors,
)


@pytest.fixture
def correspondences():
    rng = np.random.default_rng(23)
    points = rng.uniform([-0.8, -0.6, 1.5], [0.8, 0.6, 4], (100, 3))
    K = np.array([[300, 0, 160], [0, 310, 120], [0, 0, 1]], dtype=float)
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.array([0.03, -0.04, 0.01]))[0]
    pose[:3, 3] = [0.05, -0.01, 0.03]
    pixels = project_points(transform_points(points, pose), K)
    pixels += rng.normal(0, 0.25, pixels.shape)
    pixels[-25:] = rng.uniform([0, 0], [320, 240], (25, 2))
    return points, pixels, K, pose


def test_seeded_pose_fit_refines_and_reclassifies_outliers(correspondences):
    points, pixels, K, expected = correspondences
    config = SparseConfig()
    result = estimate_sparse_pose(points, pixels, K, config)
    repeated = estimate_sparse_pose(points, pixels, K, config)
    assert result.reason is None
    assert result.diagnostics["refinement"] == "refined"
    np.testing.assert_array_equal(result.T_C1C0, repeated.T_C1C0)
    translation, angle = pose_error(result.T_C1C0, expected)
    assert translation < 0.005
    assert angle < 0.15
    errors = reprojection_errors(points, pixels, K, result.T_C1C0)
    assert (errors[:75] < 3).all()
    assert (errors[75:] > 3).all()
    assert result.diagnostics["inliers"] == 75
    assert result.diagnostics["inlier_max_px"] <= config.reprojection_threshold_px
    assert (
        result.diagnostics["final_original_consensus_rmse_px"]
        <= result.diagnostics["initial_consensus_rmse_px"]
    )


def test_refinement_can_be_disabled(correspondences):
    points, pixels, K, _ = correspondences
    result = estimate_sparse_pose(points, pixels, K, SparseConfig(refine=False))
    assert result.reason is None
    assert result.diagnostics["refinement"] == "disabled"
    np.testing.assert_array_equal(result.T_C1C0, result.initial_T_C1C0)


@pytest.mark.parametrize("failure", ["opencv", "nonfinite", "worse"])
def test_lm_failure_is_explicit_without_mutating_initial_pose(
    monkeypatch, correspondences, failure
):
    points, pixels, K, _ = correspondences
    unrefined = estimate_sparse_pose(points, pixels, K, SparseConfig(refine=False))

    def refine(**kwargs):
        if failure == "opencv":
            raise cv2.error("synthetic failure")
        kwargs["tvec"][:] = np.nan if failure == "nonfinite" else 100
        return kwargs["rvec"], kwargs["tvec"]

    monkeypatch.setattr(cv2, "solvePnPRefineLM", refine)
    result = estimate_sparse_pose(points, pixels, K, SparseConfig())
    assert result.T_C1C0 is None
    assert result.reason == "LM refinement failed"
    assert result.diagnostics["refinement"] == "failed"
    np.testing.assert_array_equal(result.initial_T_C1C0, unrefined.initial_T_C1C0)


def test_inlier_gate_rejects_a_finite_low_support_pose(correspondences):
    points, pixels, K, _ = correspondences
    result = estimate_sparse_pose(points, pixels, K, SparseConfig(min_inlier_ratio=0.9))
    assert result.T_C1C0 is None
    assert result.reason == "low final inlier ratio"
    assert result.initial_T_C1C0 is not None


def test_invalid_projections_are_not_inliers():
    points = np.array([[0, 0, 2], [0, 0, -2], [0, 0, 0], [np.nan, 0, 1]])
    errors = reprojection_errors(points, np.zeros((4, 2)), np.eye(3), np.eye(4))
    assert errors[0] == 0
    assert np.isinf(errors[1:]).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"confidence": 1},
        {"ratio": np.nan},
        {"min_inliers": 4},
        {"seed": 2**31},
        {"depth_edge_threshold_m": -1},
        {"min_coverage": 1.1},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        SparseConfig(**kwargs)


def test_depth_filter_rejects_mixed_surfaces_and_holes_without_mutation():
    depth = np.full((9, 9), 2, dtype=np.float32)
    depth[:, 5:] = 3
    depth[2, 2] = np.nan
    original = depth.copy()
    filtered = reject_depth_edges(depth, 0.05)
    assert np.isnan(filtered[[0, -1]]).all()
    assert np.isnan(filtered[:, [0, -1, 4, 5]]).all()
    assert np.isnan(filtered[1:4, 1:4]).all()
    assert filtered[6, 2] == 2
    assert filtered[6, 7] == 3
    np.testing.assert_array_equal(depth, original)


def test_depth_registration_uses_rgb_z_and_nearest_surface():
    depth = np.array([[2, 4]], dtype=np.float32)
    T = np.eye(4)
    T[:3, 3] = [2, 0, 1]
    K_R = np.diag([0.2, 1, 1])
    registered = register_depth_to_rgb(depth, np.eye(3), K_R, T, (2, 3))
    assert registered[0, 0] == 3
    assert np.isfinite(registered).sum() == 1
    T[2, 3] = -3
    registered = register_depth_to_rgb(depth, np.eye(3), K_R, T, (2, 3))
    assert np.isfinite(registered).sum() == 1
    assert registered[0, 1] == 1


def test_rejected_foreground_still_occludes_registered_background():
    depth = np.array([[2, 4]], dtype=np.float32)
    K_R = np.diag([0.2, 1, 1])
    registered = register_depth_to_rgb(
        depth,
        np.eye(3),
        K_R,
        np.eye(4),
        (2, 3),
        valid_mask=np.array([[False, True]]),
    )
    assert np.isnan(registered).all()


def test_rgb_motion_is_conjugated_into_depth_frame(monkeypatch, correspondences):
    points, pixels, K, expected_R = correspondences
    T_RD = np.eye(4)
    T_RD[:3, :3] = cv2.Rodrigues(np.array([0.4, -0.3, 0.2]))[0]
    T_RD[:3, 3] = [0.1, -0.06, 0.03]
    calibration = Calibration(K, K, T_RD, (240, 320), (240, 320))
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 2, dtype=np.float32)
    source, target = Observation(0, rgb, depth), Observation(1, rgb, depth)
    descriptors = np.random.default_rng(3).normal(size=(100, 128)).astype(np.float32)
    features = {
        0: _Features(project_points(points, K), descriptors, points),
        1: _Features(pixels, descriptors.copy(), points),
    }
    backend = SparseRGBDBackend(SparseConfig(min_coverage=0))
    monkeypatch.setattr(
        backend, "_features", lambda observation, _: features[observation.frame_index]
    )
    result = backend.estimate(source, target, calibration)
    expected_D = np.linalg.inv(T_RD) @ expected_R @ T_RD
    translation, angle = pose_error(result.T_C1C0, expected_D)
    assert translation < 0.005
    assert angle < 0.15
    assert np.linalg.norm(result.T_C1C0 - expected_R) > 0.005
    rejected = SparseRGBDBackend(SparseConfig(min_coverage=1))
    monkeypatch.setattr(
        rejected, "_features", lambda observation, _: features[observation.frame_index]
    )
    assert (
        rejected.estimate(source, target, calibration).reason == "low spatial coverage"
    )


def test_real_sift_tracking_caches_frames_and_ignores_reference_poses(monkeypatch):
    rgb = np.random.default_rng(0).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    depth = np.full((192, 256), 2, dtype=np.float32)
    K = np.array([[300, 0, 127], [0, 300, 95], [0, 0, 1]], dtype=float)
    calibration = Calibration(K, K, np.eye(4), (192, 256), (192, 256))
    observations = []
    for index in range(3):
        shifted = cv2.warpAffine(
            rgb, np.float32([[1, 0, 3 * index], [0, 1, 2 * index]]), (256, 192)
        )
        observations.append(Observation(index, shifted, depth))
    backend = SparseRGBDBackend()
    sift = backend._sift
    calls = []

    class CountedSIFT:
        def detectAndCompute(self, image, mask):
            calls.append(1)
            return sift.detectAndCompute(image, mask)

    monkeypatch.setattr(backend, "_sift", CountedSIFT())
    tracked = list(track_observations(observations, calibration, backend))
    assert len(calls) == 3
    assert [row.status for row in tracked] == ["initialized", "tracked", "tracked"]
    np.testing.assert_allclose(
        tracked[-1].T_SC_D[:3, 3], [-0.04, -4 / 150, 0], atol=0.002
    )
    malformed_reference = np.eye(4)
    malformed_reference[0, 0] = -1
    changed = [
        replace(observation, T_WC_D=malformed_reference) for observation in observations
    ]
    rerun = list(track_observations(changed, calibration, SparseRGBDBackend()))
    for before, after in zip(tracked, rerun):
        np.testing.assert_array_equal(before.T_SC_D, after.T_SC_D)


def test_blank_frames_fail_without_a_pose():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.full((32, 32), 2, dtype=np.float32)
    cal = Calibration(np.eye(3), np.eye(3), np.eye(4), (32, 32), (32, 32))
    estimate = SparseRGBDBackend().estimate(
        Observation(0, rgb, depth), Observation(1, rgb, depth), cal
    )
    assert estimate.T_C1C0 is None
    assert estimate.reason == "insufficient correspondences"
    assert estimate.diagnostics["depth_valid_matches"] == 0


def test_unregistered_depth_uses_rgb_frame_z_for_real_sift():
    rgb = np.random.default_rng(4).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    moved = cv2.warpAffine(rgb, np.float32([[1, 0, 5], [0, 1, 3]]), (256, 192))
    depth = np.full((192, 256), 2, dtype=np.float32)
    K = np.array([[300, 0, 127], [0, 300, 95], [0, 0, 1]], dtype=float)
    T_RD = np.eye(4)
    T_RD[:3, 3] = [0.04, 0.02, 0.1]
    calibration = Calibration(K, K, T_RD, (192, 256), (192, 256))
    result = SparseRGBDBackend().estimate(
        Observation(0, rgb, depth), Observation(1, moved, depth), calibration
    )
    assert result.reason is None
    np.testing.assert_allclose(
        result.T_C1C0[:3, 3], [5 * 2.1 / 300, 3 * 2.1 / 300, 0], atol=0.0007
    )


def test_pipeline_rejects_noncanonical_indices_before_estimation():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    calibration = Calibration(np.eye(3), np.eye(3), np.eye(4), (2, 2), (2, 2))

    class Backend:
        def estimate(self, *args):
            pytest.fail("Invalid indices must be rejected before estimation")

    with pytest.raises(ValueError, match="contiguous"):
        list(track_observations([Observation(3, rgb, depth)], calibration, Backend()))


def test_pipeline_inverts_forward_motion_and_keeps_failure_segments_separate():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    cal = Calibration(np.eye(3), np.eye(3), np.eye(4), (2, 2), (2, 2))
    world = []
    for index in range(5):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(
            np.array([0.1 * index, -0.05 * index, 0.08 * index])
        )[0]
        pose[:3, 3] = [index, index**2, 2]
        world.append(pose)
    observations = [
        Observation(index, rgb, depth, T_WC_D=pose) for index, pose in enumerate(world)
    ]

    class Backend:
        def estimate(self, source, target, calibration):
            if target.frame_index == 2:
                return PairEstimate(None, "synthetic tracking failure")
            return PairEstimate(
                np.linalg.inv(world[target.frame_index]) @ world[source.frame_index]
            )

    rows = list(track_observations(iter(observations), cal, Backend()))
    assert [row.segment_id for row in rows] == [0, 0, 1, 1, 1]
    assert rows[2].status == "lost"
    for index, row in enumerate(rows):
        anchor = world[0 if index < 2 else 2]
        np.testing.assert_allclose(
            row.T_SC_D, np.linalg.inv(anchor) @ world[index], atol=1e-14
        )
    assert rows[2].pair.reason == "synthetic tracking failure"


def test_nonrigid_backend_result_breaks_tracking():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    cal = Calibration(np.eye(3), np.eye(3), np.eye(4), (2, 2), (2, 2))
    malformed = np.eye(4)
    malformed[0, 0] = -1

    class Backend:
        def estimate(self, *args):
            return PairEstimate(malformed)

    rows = list(
        track_observations(
            [Observation(index, rgb, depth) for index in range(2)], cal, Backend()
        )
    )
    assert rows[-1].status == "lost"
    assert rows[-1].pair.T_C1C0 is None
    assert rows[-1].pair.reason == "backend returned a nonrigid pose"
