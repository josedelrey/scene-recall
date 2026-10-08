import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scene_recall.data import Calibration
from scene_recall.geometry.camera import project_points

experiment = runpy.run_path(
    Path(__file__).parents[1] / "experiments" / "scannet_sift.py"
)


def run_experiment(
    tmp_path,
    monkeypatch,
    rgb_0,
    rgb_1,
    *,
    depth=None,
    calibration=None,
    reference_poses=(None, None),
    frame_indices=None,
):
    frame0, frame1 = (0, 1) if frame_indices is None else frame_indices
    if depth is None:
        depth = np.full(rgb_0.shape[:2], 2.0, dtype=np.float32)
    if calibration is None:
        calibration = Calibration(
            K_R=np.eye(3),
            K_D=np.eye(3),
            T_RD=np.eye(4),
            rgb_shape=rgb_0.shape[:2],
            depth_shape=depth.shape,
        )

    class RGBDReader:
        sequence_id = "synthetic"

        def __init__(self, sens_path, *, start, stop):
            assert (start, stop) == (min(frame0, frame1), max(frame0, frame1) + 1)
            self.calibration = calibration
            self.frames = iter(
                SimpleNamespace(
                    rgb=(
                        rgb_0
                        if index == frame0
                        else rgb_1
                        if index == frame1
                        else np.zeros_like(rgb_0)
                    ),
                    depth=depth if index == frame0 else np.full_like(depth, np.nan),
                    T_WC_D=(
                        reference_poses[0]
                        if index == frame0
                        else reference_poses[1]
                        if index == frame1
                        else None
                    ),
                    frame_index=index - start,
                )
                for index in range(start, stop)
            )
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def __next__(self):
            return next(self.frames)

        def __iter__(self):
            return self

    main = experiment["main"]
    monkeypatch.setitem(main.__globals__, "ScanNetReader", RGBDReader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scannet_sift.py",
            "--sens-path",
            "unused.sens",
            "--output-dir",
            str(tmp_path),
            *(
                ["--frame0", str(frame0), "--frame1", str(frame1)]
                if frame_indices is not None
                else []
            ),
        ],
    )
    main()
    stem = f"synthetic_frames{frame0}_{frame1}_sift"
    with np.load(tmp_path / f"{stem}.npz") as matches:
        points_C0 = matches["points_C0"]
        pixels_0 = matches["pixels_0"]
        pixels_1 = matches["pixels_1"]
    visualization = cv2.imread(str(tmp_path / f"{stem}.png"))
    return points_C0, pixels_0, pixels_1, visualization


@pytest.mark.parametrize("frame_indices", [(0, 2), (0, 5), (0, 10), (4, 4)])
def test_selected_source_frames_have_distinct_outputs(
    tmp_path, monkeypatch, capsys, frame_indices
) -> None:
    rgb_0 = np.full((64, 64, 3), [210, 60, 20], dtype=np.uint8)
    rgb_1 = np.full_like(rgb_0, [30, 90, 180])
    if frame_indices[0] == frame_indices[1]:
        rgb_1 = rgb_0
    run_experiment(tmp_path, monkeypatch, rgb_0, rgb_1)
    _, _, _, visualization = run_experiment(
        tmp_path, monkeypatch, rgb_0, rgb_1, frame_indices=frame_indices
    )

    frame0, frame1 = frame_indices
    for pair in ((0, 1), frame_indices):
        for extension in ("npz", "png"):
            assert (
                tmp_path / f"synthetic_frames{pair[0]}_{pair[1]}_sift.{extension}"
            ).is_file()
    np.testing.assert_array_equal(visualization[:, :64], rgb_0[..., ::-1])
    np.testing.assert_array_equal(visualization[:, 64:], rgb_1[..., ::-1])
    assert f"Source RGB frames: {frame0} -> {frame1}" in capsys.readouterr().out


@pytest.mark.parametrize("argument", ["--frame0", "--frame1"])
def test_negative_source_indices_are_rejected(monkeypatch, capsys, argument) -> None:
    monkeypatch.setattr(
        sys, "argv", ["scannet_sift.py", "--sens-path", "unused.sens", argument, "-1"]
    )

    with pytest.raises(SystemExit) as error:
        experiment["main"]()

    assert error.value.code == 2
    assert "must be nonnegative source indices" in capsys.readouterr().err


def test_translated_texture_preserves_pixel_correspondence(
    tmp_path, monkeypatch, capsys
) -> None:
    rng = np.random.default_rng(0)
    rgb_0 = rng.integers(0, 256, size=(192, 256, 3), dtype=np.uint8)
    rgb_1 = np.zeros_like(rgb_0)
    rgb_1[5:, 7:] = rgb_0[:-5, :-7]

    points_C0, pixels_0, pixels_1, visualization = run_experiment(
        tmp_path, monkeypatch, rgb_0, rgb_1
    )

    assert pixels_0.shape == pixels_1.shape
    assert pixels_0.dtype == pixels_1.dtype == np.float32
    assert len(pixels_0) > 20
    assert points_C0.shape == (len(pixels_0), 3)
    np.testing.assert_allclose(points_C0[:, 2], 2.0)
    np.testing.assert_allclose(project_points(points_C0, np.eye(3)), pixels_0)
    offsets = pixels_1 - pixels_0
    assert (np.linalg.norm(offsets - [7, 5], axis=1) < 0.5).mean() > 0.95
    assert visualization.shape == (192, 512, 3)
    diagnostics = capsys.readouterr().out
    assert "KNN matches (query rows, k=2):" in diagnostics
    assert f"Matches surviving ratio test (0.75): {len(pixels_0)}" in diagnostics
    assert "Retention ratio (accepted / KNN rows):" in diagnostics
    assert f"Matches with valid frame-0 depth: {len(points_C0)}" in diagnostics
    assert "Depth-valid retention ratio (valid / accepted): 1.0000" in diagnostics
    assert "Frame-0 RGB reprojection check: passed" in diagnostics


@pytest.mark.parametrize("blank_frame", [0, 1])
def test_missing_descriptors_save_empty_pairs_and_preserve_rgb_colors(
    tmp_path, monkeypatch, capsys, blank_frame
) -> None:
    texture = np.random.default_rng(0).integers(
        0, 256, size=(192, 256, 3), dtype=np.uint8
    )
    blank = np.full_like(texture, [210, 60, 20])
    frames = [texture, texture]
    frames[blank_frame] = blank

    points_C0, pixels_0, pixels_1, visualization = run_experiment(
        tmp_path, monkeypatch, *frames
    )

    assert pixels_0.shape == pixels_1.shape == (0, 2)
    assert points_C0.shape == (0, 3)
    np.testing.assert_array_equal(
        visualization[:, blank_frame * 256 : (blank_frame + 1) * 256],
        blank[..., ::-1],
    )
    diagnostics = capsys.readouterr().out
    assert f"Keypoints in frame {blank_frame}: 0" in diagnostics
    assert "KNN matches (query rows, k=2): 0" in diagnostics
    assert "Retention ratio (accepted / KNN rows): 0.0000" in diagnostics
    assert "Matches with valid frame-0 depth: 0" in diagnostics
    assert "Depth-valid retention ratio (valid / accepted): 0.0000" in diagnostics


def test_single_train_descriptor_cannot_pass_ratio_test(
    tmp_path, monkeypatch, capsys
) -> None:
    detections = iter(
        [
            ([cv2.KeyPoint(10, 20, 1)], np.zeros((1, 128), dtype=np.float32)),
            ([cv2.KeyPoint(30, 40, 1)], np.ones((1, 128), dtype=np.float32)),
        ]
    )
    monkeypatch.setattr(
        cv2,
        "SIFT_create",
        lambda: SimpleNamespace(detectAndCompute=lambda *args: next(detections)),
    )
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    points_C0, pixels_0, pixels_1, _ = run_experiment(tmp_path, monkeypatch, rgb, rgb)

    assert pixels_0.shape == pixels_1.shape == (0, 2)
    assert points_C0.shape == (0, 3)
    diagnostics = capsys.readouterr().out
    assert "KNN matches (query rows, k=2): 1" in diagnostics
    assert "Matches surviving ratio test (0.75): 0" in diagnostics


def mock_sift_matches(monkeypatch, pixels_0, pixels_1, query_order):
    detections = iter(
        (
            [cv2.KeyPoint(float(x), float(y), 1) for x, y in pixels],
            np.zeros((len(pixels), 128), dtype=np.float32),
        )
        for pixels in (pixels_0, pixels_1)
    )
    monkeypatch.setattr(
        cv2,
        "SIFT_create",
        lambda: SimpleNamespace(detectAndCompute=lambda *args: next(detections)),
    )
    neighbors = [
        [
            cv2.DMatch(index, len(pixels_1) - 1 - index, 1),
            cv2.DMatch(index, index, 4),
        ]
        for index in query_order
    ]
    # Equality at the ratio threshold still rejects a match.
    neighbors.append([cv2.DMatch(0, 0, 3), cv2.DMatch(0, 1, 4)])

    def matcher(norm, *, crossCheck):
        assert norm == cv2.NORM_L2
        assert crossCheck is False

        def knn_match(descriptors_0, descriptors_1, *, k):
            assert k == 2
            return neighbors

        return SimpleNamespace(knnMatch=knn_match)

    monkeypatch.setattr(cv2, "BFMatcher", matcher)


@pytest.mark.parametrize("all_invalid", [False, True])
def test_depth_filter_preserves_match_order_across_registered_grids(
    tmp_path, monkeypatch, capsys, all_invalid
) -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((16, 16), np.nan, dtype=np.float32)
    calibration = Calibration(
        K_R=np.array([[8, 0, 31.5], [0, 16, 31.5], [0, 0, 1]], dtype=np.float64),
        K_D=np.array([[2, 0, 7.5], [0, 4, 7.5], [0, 0, 1]], dtype=np.float64),
        T_RD=np.eye(4),
        rgb_shape=rgb.shape[:2],
        depth_shape=depth.shape,
    )
    source_pixels = np.array(
        [
            [5.5, 9.5],
            [9.5, 5.5],
            [13.5, 9.5],
            [17.5, 5.5],
            [21.5, 9.5],
            [25.5, 5.5],
            [29.5, 9.5],
            [0, 0],
        ],
        dtype=np.float32,
    )
    target_pixels = source_pixels + [7, 5]
    query_order = [4, 0, 6, 1, 2, 5, 3, 7]
    mock_sift_matches(monkeypatch, source_pixels, target_pixels, query_order)
    if not all_invalid:
        depth[[2, 1, 2, 1, 2, 1, 2], [1, 2, 3, 4, 5, 6, 7]] = [
            np.nan,
            2,
            np.inf,
            0,
            3,
            -1,
            -np.inf,
        ]
    drawn_matches = []
    original_draw_matches = cv2.drawMatches

    def draw_matches(*args, **kwargs):
        drawn_matches.extend(args[4])
        return original_draw_matches(*args, **kwargs)

    monkeypatch.setattr(cv2, "drawMatches", draw_matches)

    points_C0, pixels_0, pixels_1, _ = run_experiment(
        tmp_path, monkeypatch, rgb, rgb, depth=depth, calibration=calibration
    )

    retained_queries = [] if all_invalid else [4, 1]
    retained_targets = [7 - index for index in retained_queries]
    assert points_C0.shape == (len(retained_queries), 3)
    assert pixels_0.shape == pixels_1.shape == (len(retained_queries), 2)
    assert np.isfinite(points_C0).all()
    np.testing.assert_array_equal(pixels_0, source_pixels[retained_queries])
    np.testing.assert_array_equal(pixels_1, target_pixels[retained_targets])
    np.testing.assert_allclose(
        project_points(points_C0, calibration.K_R), pixels_0, rtol=1e-10, atol=1e-8
    )
    if not all_invalid:
        np.testing.assert_allclose(points_C0, [[-3.75, -4.125, 3], [-5.5, -3.25, 2]])
    assert [match.queryIdx for match in drawn_matches] == query_order
    diagnostics = capsys.readouterr().out
    assert "Matches surviving ratio test (0.75): 8" in diagnostics
    assert f"Matches with valid frame-0 depth: {len(retained_queries)}" in diagnostics
    ratio = "0.0000" if all_invalid else "0.2500"
    assert f"Depth-valid retention ratio (valid / accepted): {ratio}" in diagnostics


def test_experiment_checks_frame_0_rgb_reprojection(tmp_path, monkeypatch) -> None:
    pixels = np.array([[10.5, 20.5], [30.5, 40.5]], dtype=np.float32)
    mock_sift_matches(monkeypatch, pixels, pixels, [0, 1])
    monkeypatch.setitem(
        experiment["main"].__globals__,
        "backproject_rgb_pixels",
        lambda *args: np.array([[0, 0, 2], [0, 0, 2]], dtype=np.float64),
    )
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Frame-0 RGB reprojection check failed"):
        run_experiment(tmp_path, monkeypatch, rgb, rgb)


@pytest.fixture
def pose_correspondences():
    rng = np.random.default_rng(7)
    K_R = np.array([[180, 0, 127.5], [0, 175, 95.5], [0, 0, 1]], dtype=np.float64)
    u, v = np.meshgrid(np.arange(30, 230, 25), np.arange(30, 170, 20))
    pixels_0 = np.column_stack((u.ravel(), v.ravel())).astype(np.float32)
    depths = rng.uniform(1.5, 4, len(pixels_0)).astype(np.float32)
    points_C0 = (
        np.column_stack((pixels_0, np.ones(len(pixels_0)))) @ np.linalg.inv(K_R).T
    )
    points_C0 *= depths[:, None]
    T_C1C0 = np.eye(4)
    T_C1C0[:3, :3] = cv2.Rodrigues(np.array([0.025, -0.04, 0.012]))[0]
    T_C1C0[:3, 3] = [0.08, -0.025, 0.05]
    points_C1 = points_C0 @ T_C1C0[:3, :3].T + T_C1C0[:3, 3]
    pixels_1 = project_points(points_C1, K_R)
    pixels_1 += rng.normal(0, 0.08, pixels_1.shape)
    pixels_1[-12:] += [35, -25]
    return points_C0, pixels_0, pixels_1.astype(np.float32), K_R, T_C1C0


def test_epnp_recovers_forward_pose_with_outliers(pose_correspondences) -> None:
    points_C0, _, pixels_1, K_R, expected = pose_correspondences
    cv2.setRNGSeed(0)

    estimated, indices, errors = experiment["estimate_relative_pose"](
        points_C0, pixels_1, K_R
    )

    assert estimated.shape == (4, 4)
    assert estimated.dtype == np.float64
    np.testing.assert_array_equal(estimated[3], [0, 0, 0, 1])
    np.testing.assert_allclose(estimated, expected, atol=0.002)
    np.testing.assert_array_equal(np.sort(indices), np.arange(len(points_C0) - 12))
    projected = project_points(
        points_C0[indices] @ estimated[:3, :3].T + estimated[:3, 3], K_R
    )
    np.testing.assert_allclose(
        errors, np.linalg.norm(projected - pixels_1[indices], axis=1)
    )
    assert errors.mean() < 0.2
    # The inverse has a measurably different reprojection on this fixture.
    inverse = np.linalg.inv(estimated)
    wrong_pixels = project_points(
        points_C0[indices] @ inverse[:3, :3].T + inverse[:3, 3], K_R
    )
    assert np.linalg.norm(wrong_pixels - pixels_1[indices], axis=1).mean() > 5


@pytest.mark.parametrize("count", range(5))
def test_epnp_requires_five_points_without_calling_opencv(monkeypatch, count) -> None:
    def unexpected_call(*args, **kwargs):
        pytest.fail("Insufficient correspondences must skip solvePnPRansac")

    monkeypatch.setattr(cv2, "solvePnPRansac", unexpected_call)
    with pytest.raises(ValueError, match="at least 5 correspondences"):
        experiment["estimate_relative_pose"](
            np.zeros((count, 3)), np.zeros((count, 2)), np.eye(3)
        )


@pytest.mark.parametrize(
    "result",
    [
        (False, None, None, None),
        (True, np.zeros((3, 1)), np.zeros((3, 1)), None),
        (True, np.zeros((3, 1)), np.zeros((3, 1)), np.empty((0, 1), dtype=np.int32)),
        (True, np.full((3, 1), np.nan), np.zeros((3, 1)), np.array([[0]])),
    ],
)
def test_unusable_pnp_results_are_rejected(monkeypatch, result) -> None:
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda **kwargs: result)
    with pytest.raises(RuntimeError, match="pose"):
        experiment["estimate_relative_pose"](
            np.ones((6, 3)), np.ones((6, 2)), np.eye(3)
        )


@pytest.mark.parametrize(
    "gt_case", ["rigid", "missing", "nonrigid", "reflection", "near_rigid", "singular"]
)
@pytest.mark.parametrize("frame_indices", [(0, 1), (4, 10), (10, 4)])
def test_experiment_pose_outputs_and_local_gt_comparison(
    tmp_path, monkeypatch, capsys, pose_correspondences, gt_case, frame_indices
) -> None:
    points, pixels_0, pixels_1, K_R, expected = pose_correspondences
    rgb = np.zeros((192, 256, 3), dtype=np.uint8)
    depth = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
    depth[pixels_0[:, 1].astype(int), pixels_0[:, 0].astype(int)] = points[:, 2]
    # Remove two matches before PnP and permute the accepted-match order.
    depth[pixels_0[:2, 1].astype(int), pixels_0[:2, 0].astype(int)] = np.nan
    order = np.random.default_rng(8).permutation(len(points))
    valid_order = order[order >= 2]
    mock_sift_matches(monkeypatch, pixels_0, pixels_1[::-1], order)
    calibration = Calibration(
        K_R=K_R,
        K_D=K_R.copy(),
        T_RD=np.eye(4),
        rgb_shape=rgb.shape[:2],
        depth_shape=depth.shape,
    )
    T_WC0 = np.eye(4)
    T_WC0[:3, :3] = cv2.Rodrigues(np.array([0.1, 0.2, -0.3]))[0]
    T_WC0[:3, 3] = [1, 2, 3]
    # Deliberately differ from the estimated motion to exercise error reporting.
    gt = expected.copy()
    gt[:3, :3] = cv2.Rodrigues(np.array([0.04, 0.01, -0.02]))[0]
    gt[:3, 3] += [0.01, -0.02, 0.03]
    if gt_case == "nonrigid":
        gt[:3, :3] *= 1.1
    elif gt_case == "reflection":
        gt[:3, 0] *= -1
    elif gt_case == "near_rigid":
        gt[:3, :3] *= 1 + 1e-6
    T_WC1 = T_WC0 @ np.linalg.inv(gt)
    if gt_case == "missing":
        T_WC1 = None
    elif gt_case == "singular":
        T_WC1[:3, :3] = 0
    original_T_WC0 = T_WC0.copy()
    original_T_WC1 = None if T_WC1 is None else T_WC1.copy()
    original_solver = cv2.solvePnPRansac

    def solve_pnp(**kwargs):
        np.testing.assert_allclose(kwargs["objectPoints"], points[valid_order])
        np.testing.assert_array_equal(kwargs["imagePoints"], pixels_1[valid_order])
        np.testing.assert_array_equal(kwargs["cameraMatrix"], K_R)
        assert kwargs["distCoeffs"] is None
        assert kwargs["flags"] == cv2.SOLVEPNP_EPNP
        assert kwargs["reprojectionError"] == 3.0
        assert kwargs["iterationsCount"] == 100
        assert kwargs["confidence"] == 0.99
        return original_solver(**kwargs)

    monkeypatch.setattr(cv2, "solvePnPRansac", solve_pnp)
    cv2.setRNGSeed(0)
    run_experiment(
        tmp_path,
        monkeypatch,
        rgb,
        rgb,
        depth=depth,
        calibration=calibration,
        reference_poses=(T_WC0, T_WC1),
        frame_indices=frame_indices,
    )

    frame0, frame1 = frame_indices
    with np.load(tmp_path / f"synthetic_frames{frame0}_{frame1}_sift.npz") as saved:
        assert saved["pose_status"].item() == "estimated"
        estimated = saved["T_C1C0"]
        np.testing.assert_allclose(estimated, expected, atol=0.002)
        indices = saved["inlier_indices"]
        np.testing.assert_array_equal(
            np.sort(valid_order[indices]), np.arange(2, len(points) - 12)
        )
        projected = project_points(
            saved["points_C0"][indices] @ estimated[:3, :3].T + estimated[:3, 3], K_R
        )
        errors = np.linalg.norm(projected - saved["pixels_1"][indices], axis=1)
        np.testing.assert_allclose(saved["inlier_reprojection_errors_px"], errors)
        if gt_case in ("missing", "singular"):
            assert "T_C1C0_gt" not in saved
        else:
            np.testing.assert_allclose(saved["T_C1C0_gt"], gt, atol=1e-14)

    np.testing.assert_array_equal(T_WC0, original_T_WC0)
    if T_WC1 is not None:
        np.testing.assert_array_equal(T_WC1, original_T_WC1)
    diagnostics = capsys.readouterr().out
    assert f"Source RGB frames: {frame0} -> {frame1}" in diagnostics
    assert f"3D-2D correspondences: {len(valid_order)}" in diagnostics
    assert f"RANSAC inliers: {len(indices)}" in diagnostics
    assert (
        f"(inliers / correspondences): {len(indices) / len(valid_order):.4f}"
        in diagnostics
    )
    assert f"mean={errors.mean():.4f}, median={np.median(errors):.4f}" in diagnostics
    assert (
        f"RMSE={np.sqrt(np.mean(errors**2)):.4f}, max={errors.max():.4f}" in diagnostics
    )
    assert f"Estimated translation (m): {estimated[:3, 3]}" in diagnostics
    assert (
        f"Estimated translation magnitude (m): {np.linalg.norm(estimated[:3, 3]):.6f}"
        in diagnostics
    )
    if gt_case in ("missing", "singular"):
        assert "GT comparison unavailable:" in diagnostics
        assert "Translation-vector error norm" not in diagnostics
        assert "GT translation magnitude" not in diagnostics
    else:
        assert f"GT translation (m): {gt[:3, 3]}" in diagnostics
        assert (
            f"GT translation magnitude (m): {np.linalg.norm(gt[:3, 3]):.6f}"
            in diagnostics
        )
        error = np.linalg.norm(estimated[:3, 3] - gt[:3, 3])
        assert f"Translation-vector error norm (m): {error:.6f}" in diagnostics
        if gt_case in ("nonrigid", "reflection"):
            assert "Rotation angle unavailable:" in diagnostics
            assert "Rotation error (deg" not in diagnostics
        else:
            cosine = (np.trace(estimated[:3, :3] @ gt[:3, :3].T) - 1) / 2
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            assert (
                f"Rotation error (deg, raw GT accepted at atol=1e-4): {angle:.6f}"
                in diagnostics
            )


@pytest.mark.parametrize("raises", [False, True])
def test_failed_solver_preserves_correspondences_and_visualization(
    tmp_path, monkeypatch, capsys, raises
) -> None:
    def failed_solver(**kwargs):
        if raises:
            raise cv2.error("Synthetic solver error")
        return False, None, None, None

    monkeypatch.setattr(cv2, "solvePnPRansac", failed_solver)
    rgb = np.random.default_rng(0).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    points, _, _, visualization = run_experiment(tmp_path, monkeypatch, rgb, rgb)

    assert len(points) > 5
    assert visualization.shape == (192, 512, 3)
    with np.load(tmp_path / "synthetic_frames0_1_sift.npz") as saved:
        assert saved["pose_status"].item() != "estimated"
        assert "T_C1C0" not in saved
        assert "T_C1C0_gt" not in saved
        assert saved["inlier_indices"].shape == (0,)
        assert saved["inlier_reprojection_errors_px"].shape == (0,)
    diagnostics = capsys.readouterr().out
    assert "Pose estimation unavailable:" in diagnostics
    assert "RANSAC inliers: 0" in diagnostics
    assert "GT comparison unavailable: no estimated pose" in diagnostics
