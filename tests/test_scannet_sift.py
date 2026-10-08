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
    diagnose_gt_reprojection=False,
    diagnose_pnp_refinement=False,
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
            *(["--diagnose-gt-reprojection"] if diagnose_gt_reprojection else []),
            *(["--diagnose-pnp-refinement"] if diagnose_pnp_refinement else []),
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


def test_reprojection_diagnostic_uses_forward_pinhole_and_signed_residuals(capsys):
    points = np.array([[1, 2, 2], [2, 0, 4], [0, -1, 1], [1, 1, 3]], dtype=float)
    K_R = np.array([[100, 0, 32], [0, 80, 24], [0, 0, 1]], dtype=float)
    rvec = np.array([0.1, -0.2, 0.3])
    tvec = np.array([0.2, -0.1, 0.5])
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec
    projected, _ = cv2.projectPoints(points, rvec, tvec, K_R, None)
    residuals = np.array([[3, 4], [-1, 0], [0, -2], [3, 0]])
    observed = projected.reshape(-1, 2) + residuals

    experiment["print_reprojection_diagnostics"](
        "Estimated", points, observed, K_R, pose
    )

    diagnostics = capsys.readouterr().out
    assert "Estimated inlier reprojection: selected=4, invalid=0" in diagnostics
    assert "mean=2.7500, median=2.5000, RMSE=3.1225" in diagnostics
    assert "du=1.2500, dv=0.5000" in diagnostics
    assert "below 3 px (all selected): 50.00%" in diagnostics


@pytest.mark.parametrize(
    "invalid_point",
    [[0, 0, -1], [0, 0, 0], [np.nan, 0, 1], [0, np.inf, 1], [1e308, 0, 1e-308]],
)
def test_invalid_reprojections_remain_in_statistics(capsys, invalid_point):
    points = np.array([[0, 0, 1], [0, 0, 1], invalid_point], dtype=float)
    observed = np.array([[0, 0], [3, 4], [0, 0]], dtype=float)

    with np.errstate(all="raise"):
        experiment["print_reprojection_diagnostics"](
            "GT", points, observed, np.eye(3), np.eye(4)
        )

    diagnostics = capsys.readouterr().out
    assert "GT inlier reprojection: selected=3, invalid=1" in diagnostics
    assert "invalid reprojection errors count as +inf in statistics" in diagnostics
    assert "mean=inf, median=5.0000, RMSE=inf" in diagnostics
    assert "mean signed residual (observed - projected, px): unavailable" in diagnostics
    assert "below 3 px (all selected): 33.33%" in diagnostics


def test_all_invalid_reprojections_are_reported(capsys):
    with np.errstate(all="raise"):
        experiment["print_reprojection_diagnostics"](
            "GT",
            np.array([[0, 0, 0], [1, 1, -1]]),
            np.zeros((2, 2)),
            np.eye(3),
            np.eye(4),
        )

    diagnostics = capsys.readouterr().out
    assert "GT inlier reprojection: selected=2, invalid=2" in diagnostics
    assert "mean=inf, median=inf, RMSE=inf" in diagnostics
    assert "mean signed residual (observed - projected, px): unavailable" in diagnostics
    assert "below 3 px (all selected): 0.00%" in diagnostics


def test_lm_refinement_improves_inlier_fit_without_mutating_originals(
    monkeypatch, capsys, pose_correspondences
):
    points, _, pixels, intrinsic, gt = pose_correspondences
    cv2.setRNGSeed(0)
    original, indices, _ = experiment["estimate_relative_pose"](
        points, pixels, intrinsic
    )
    selected_points, selected_pixels = points[indices], pixels[indices]
    snapshots = [
        array.copy()
        for array in (selected_points, selected_pixels, intrinsic, original, gt)
    ]
    original_refiner = cv2.solvePnPRefineLM
    refined_poses = []

    def refine(**kwargs):
        np.testing.assert_array_equal(kwargs["objectPoints"], selected_points)
        np.testing.assert_array_equal(kwargs["imagePoints"], selected_pixels)
        np.testing.assert_array_equal(kwargs["cameraMatrix"], intrinsic)
        np.testing.assert_allclose(cv2.Rodrigues(kwargs["rvec"])[0], original[:3, :3])
        np.testing.assert_array_equal(kwargs["tvec"].ravel(), original[:3, 3])
        assert kwargs["distCoeffs"] is None
        assert "criteria" not in kwargs
        rvec, tvec = original_refiner(**kwargs)
        refined = np.eye(4)
        refined[:3, :3] = cv2.Rodrigues(rvec)[0]
        refined[:3, 3] = tvec.ravel()
        refined_poses.append(refined)
        return rvec, tvec

    monkeypatch.setattr(cv2, "solvePnPRefineLM", refine)
    experiment["diagnose_pnp_refinement"](
        selected_points, selected_pixels, intrinsic, original, gt
    )

    for array, snapshot in zip(
        (selected_points, selected_pixels, intrinsic, original, gt),
        snapshots,
        strict=True,
    ):
        np.testing.assert_array_equal(array, snapshot)
    diagnostics = capsys.readouterr().out
    rmses = []
    for label, pose in zip(
        ("Original EPnP+RANSAC", "LM-refined", "GT"),
        (original, refined_poses[0], gt),
        strict=True,
    ):
        residuals = selected_pixels - project_points(
            selected_points @ pose[:3, :3].T + pose[:3, 3], intrinsic
        )
        rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
        rmses.append(rmse)
        assert (
            f"{label} inlier reprojection: selected={len(indices)}, invalid=0"
            in diagnostics
        )
        line = next(
            line
            for line in diagnostics.splitlines()
            if line.startswith(f"{label} inlier reprojection errors")
        )
        assert f"RMSE={rmse:.4f}" in line
        signed = residuals.mean(axis=0)
        assert (
            f"{label} mean signed residual (observed - projected, px): du={signed[0]:.4f}, dv={signed[1]:.4f}"
            in diagnostics
        )
        error_mm = 1000 * np.linalg.norm(pose[:3, 3] - gt[:3, 3])
        assert (
            f"{label} translation-vector error against GT (mm): {error_mm:.6f}"
            in diagnostics
        )
        cosine = (np.trace(pose[:3, :3] @ gt[:3, :3].T) - 1) / 2
        angle = 0 if label == "GT" else np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        assert f"{label} rotation error against GT (deg): {angle:.6f}" in diagnostics
    assert rmses[1] < rmses[0]


@pytest.mark.parametrize("failure", ["opencv", "nonfinite"])
def test_refinement_failure_reports_original_and_gt(monkeypatch, capsys, failure):
    pose = np.eye(4)
    pose[:3, 3] = [0.003, 0.004, 0]
    original = pose.copy()

    def refine(**kwargs):
        kwargs["tvec"][:] = np.nan
        if failure == "opencv":
            raise cv2.error("Synthetic refinement failure")
        return kwargs["rvec"], kwargs["tvec"]

    monkeypatch.setattr(cv2, "solvePnPRefineLM", refine)
    experiment["diagnose_pnp_refinement"](
        np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2]], dtype=float),
        np.array([[0, 0], [0.5, 0], [0, 0.5]]),
        np.eye(3),
        pose,
        np.eye(4),
    )

    np.testing.assert_array_equal(pose, original)
    diagnostics = capsys.readouterr().out
    assert "PnP refinement failed:" in diagnostics
    assert "LM-refined" not in diagnostics
    assert (
        "Original EPnP+RANSAC inlier reprojection: selected=3, invalid=0" in diagnostics
    )
    assert "GT inlier reprojection: selected=3, invalid=0" in diagnostics
    assert (
        "Original EPnP+RANSAC translation-vector error against GT (mm): 5.000000"
        in diagnostics
    )


@pytest.mark.parametrize("gt_case", ["missing", "nonrigid", "reflection"])
def test_refinement_reports_unavailable_gt_metrics(monkeypatch, capsys, gt_case):
    gt = None if gt_case == "missing" else np.eye(4)
    if gt_case == "nonrigid":
        gt[:3, :3] *= 1.1
    elif gt_case == "reflection":
        gt[0, 0] = -1
    monkeypatch.setattr(
        cv2, "solvePnPRefineLM", lambda **kwargs: (kwargs["rvec"], kwargs["tvec"])
    )

    experiment["diagnose_pnp_refinement"](
        np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2]], dtype=float),
        np.zeros((3, 2)),
        np.eye(3),
        np.eye(4),
        gt,
    )

    diagnostics = capsys.readouterr().out
    for label in ("Original EPnP+RANSAC", "LM-refined"):
        assert f"{label} rotation error against GT (deg): unavailable" in diagnostics
        if gt_case == "missing":
            assert (
                f"{label} translation-vector error against GT (mm): unavailable"
                in diagnostics
            )
        else:
            assert (
                f"{label} translation-vector error against GT (mm): 0.000000"
                in diagnostics
            )
    if gt_case == "missing":
        assert (
            "PnP refinement GT comparison unavailable: no relative GT pose"
            in diagnostics
        )
    else:
        assert "GT rotation error against GT (deg): unavailable" in diagnostics


@pytest.mark.parametrize("refinement_result", ["success", "opencv", "nonfinite"])
@pytest.mark.parametrize("diagnose_gt_reprojection", [False, True])
def test_refinement_flag_preserves_baseline_outputs_and_inlier_selection(
    tmp_path,
    monkeypatch,
    capsys,
    pose_correspondences,
    refinement_result,
    diagnose_gt_reprojection,
):
    points, pixels_0, pixels_1, K_R, gt = pose_correspondences
    rgb = np.zeros((192, 256, 3), dtype=np.uint8)
    depth = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
    depth[pixels_0[:, 1].astype(int), pixels_0[:, 0].astype(int)] = points[:, 2]
    calibration = Calibration(
        K_R=K_R,
        K_D=K_R.copy(),
        T_RD=np.eye(4),
        rgb_shape=rgb.shape[:2],
        depth_shape=depth.shape,
    )
    original = gt.copy()
    original[:3, 3] += [0.01, -0.02, 0.03]
    original_snapshot = original.copy()
    indices = np.array([9, 3, 7, 0, 6, 2], dtype=np.int32)
    indices_snapshot = indices.copy()
    errors = np.linalg.norm(
        project_points(points[indices] @ original[:3, :3].T + original[:3, 3], K_R)
        - pixels_1[indices],
        axis=1,
    )
    monkeypatch.setitem(
        experiment["main"].__globals__,
        "estimate_relative_pose",
        lambda *args: (original, indices, errors),
    )
    refiner_calls = []

    def refine(**kwargs):
        refiner_calls.append(kwargs)
        np.testing.assert_allclose(kwargs["objectPoints"], points[indices])
        np.testing.assert_array_equal(kwargs["imagePoints"], pixels_1[indices])
        np.testing.assert_array_equal(kwargs["cameraMatrix"], K_R)
        np.testing.assert_array_equal(kwargs["tvec"].ravel(), original[:3, 3])
        np.testing.assert_allclose(cv2.Rodrigues(kwargs["rvec"])[0], original[:3, :3])
        assert kwargs["distCoeffs"] is None
        kwargs["objectPoints"][:] = 0
        kwargs["imagePoints"][:] = 0
        kwargs["cameraMatrix"][:] = 0
        kwargs["rvec"][:] = cv2.Rodrigues(gt[:3, :3])[0]
        kwargs["tvec"][:] = gt[:3, 3].reshape(3, 1)
        if refinement_result == "opencv":
            raise cv2.error("Synthetic refinement failure")
        if refinement_result == "nonfinite":
            kwargs["tvec"][:] = np.nan
        return kwargs["rvec"], kwargs["tvec"]

    monkeypatch.setattr(cv2, "solvePnPRefineLM", refine)
    diagnostic_calls = []
    original_diagnostic = experiment["print_reprojection_diagnostics"]

    def diagnostic(label, selected_points, selected_pixels, intrinsic, pose):
        diagnostic_calls.append(
            (label, selected_points.copy(), selected_pixels.copy(), pose.copy())
        )
        original_diagnostic(label, selected_points, selected_pixels, intrinsic, pose)

    monkeypatch.setitem(
        experiment["main"].__globals__,
        "print_reprojection_diagnostics",
        diagnostic,
    )
    run_options = {
        "depth": depth,
        "calibration": calibration,
        "reference_poses": (gt, np.eye(4)),
        "frame_indices": (4, 10),
        "diagnose_gt_reprojection": diagnose_gt_reprojection,
    }
    archive_path = tmp_path / "synthetic_frames4_10_sift.npz"
    image_path = tmp_path / "synthetic_frames4_10_sift.png"
    mock_sift_matches(monkeypatch, pixels_0, pixels_1[::-1], range(len(points)))
    run_experiment(tmp_path, monkeypatch, rgb, rgb, **run_options)
    with np.load(archive_path) as saved:
        baseline = {key: saved[key].copy() for key in saved.files}
    baseline_image = image_path.read_bytes()
    assert refiner_calls == []
    assert "PnP refinement" not in capsys.readouterr().out
    diagnostic_calls.clear()

    mock_sift_matches(monkeypatch, pixels_0, pixels_1[::-1], range(len(points)))
    run_experiment(
        tmp_path,
        monkeypatch,
        rgb,
        rgb,
        **run_options,
        diagnose_pnp_refinement=True,
    )

    assert len(refiner_calls) == 1
    with np.load(archive_path) as saved:
        assert set(saved.files) == set(baseline)
        for key, value in baseline.items():
            np.testing.assert_array_equal(saved[key], value)
    assert image_path.read_bytes() == baseline_image
    np.testing.assert_array_equal(original, original_snapshot)
    np.testing.assert_array_equal(indices, indices_snapshot)
    expected_labels = (
        ["Original EPnP+RANSAC", "LM-refined", "GT"]
        if refinement_result == "success"
        else ["Original EPnP+RANSAC", "GT"]
    )
    comparison_calls = diagnostic_calls[-len(expected_labels) :]
    assert [call[0] for call in comparison_calls] == expected_labels
    for label, selected_points, selected_pixels, pose in comparison_calls:
        np.testing.assert_allclose(selected_points, points[indices])
        np.testing.assert_array_equal(selected_pixels, pixels_1[indices])
        np.testing.assert_allclose(
            pose, original if label == "Original EPnP+RANSAC" else gt
        )
    diagnostics = capsys.readouterr().out
    assert ("PnP refinement failed:" in diagnostics) == (refinement_result != "success")


@pytest.mark.parametrize(
    "gt_case", ["rigid", "missing", "nonrigid", "reflection", "near_rigid", "singular"]
)
@pytest.mark.parametrize("frame_indices", [(0, 1), (4, 10), (10, 4)])
@pytest.mark.parametrize("diagnose_gt_reprojection", [False, True])
def test_experiment_pose_outputs_and_local_gt_comparison(
    tmp_path,
    monkeypatch,
    capsys,
    pose_correspondences,
    gt_case,
    frame_indices,
    diagnose_gt_reprojection,
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
    diagnostic_calls = []
    original_diagnostic = experiment["print_reprojection_diagnostics"]

    def diagnostic(label, selected_points, selected_pixels, intrinsic, pose):
        diagnostic_calls.append(
            (label, selected_points.copy(), selected_pixels.copy(), intrinsic, pose)
        )
        original_diagnostic(label, selected_points, selected_pixels, intrinsic, pose)

    monkeypatch.setitem(
        experiment["main"].__globals__, "print_reprojection_diagnostics", diagnostic
    )
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
        diagnose_gt_reprojection=diagnose_gt_reprojection,
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
        expected_keys = {
            "points_C0",
            "pixels_0",
            "pixels_1",
            "pose_status",
            "T_C1C0",
            "inlier_indices",
            "inlier_reprojection_errors_px",
        }
        if gt_case not in ("missing", "singular"):
            expected_keys.add("T_C1C0_gt")
        assert set(saved.files) == expected_keys

    np.testing.assert_array_equal(T_WC0, original_T_WC0)
    if T_WC1 is not None:
        np.testing.assert_array_equal(T_WC1, original_T_WC1)
    diagnostics = capsys.readouterr().out
    if diagnose_gt_reprojection:
        expected_labels = ["Estimated"]
        if gt_case not in ("missing", "singular"):
            expected_labels.append("GT")
        else:
            assert (
                "GT reprojection diagnostic unavailable: no relative GT pose"
                in diagnostics
            )
        assert [call[0] for call in diagnostic_calls] == expected_labels
        for (
            label,
            selected_points,
            selected_pixels,
            intrinsic,
            pose,
        ) in diagnostic_calls:
            np.testing.assert_allclose(selected_points, points[valid_order[indices]])
            np.testing.assert_array_equal(
                selected_pixels, pixels_1[valid_order[indices]]
            )
            np.testing.assert_array_equal(intrinsic, K_R)
            np.testing.assert_allclose(pose, estimated if label == "Estimated" else gt)
            residuals = selected_pixels - project_points(
                selected_points @ pose[:3, :3].T + pose[:3, 3], K_R
            )
            diagnostic_errors = np.linalg.norm(residuals, axis=1)
            assert (
                f"{label} inlier reprojection: selected={len(indices)}, invalid=0"
                in diagnostics
            )
            assert (
                f"{label} inlier reprojection errors (px): "
                f"mean={diagnostic_errors.mean():.4f}, "
                f"median={np.median(diagnostic_errors):.4f}, "
                f"RMSE={np.sqrt(np.mean(diagnostic_errors**2)):.4f}"
            ) in diagnostics
            signed = residuals.mean(axis=0)
            assert (
                f"{label} mean signed residual (observed - projected, px): "
                f"du={signed[0]:.4f}, dv={signed[1]:.4f}"
            ) in diagnostics
            assert (
                f"{label} reprojection error below 3 px (all selected): "
                f"{100 * np.count_nonzero(diagnostic_errors < 3) / len(indices):.2f}%"
            ) in diagnostics
    else:
        assert diagnostic_calls == []
        assert "mean signed residual" not in diagnostics
        assert "reprojection error below 3 px" not in diagnostics
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
@pytest.mark.parametrize("diagnose_gt_reprojection", [False, True])
@pytest.mark.parametrize("diagnose_pnp_refinement", [False, True])
def test_failed_solver_preserves_correspondences_and_visualization(
    tmp_path,
    monkeypatch,
    capsys,
    raises,
    diagnose_gt_reprojection,
    diagnose_pnp_refinement,
) -> None:
    def failed_solver(**kwargs):
        if raises:
            raise cv2.error("Synthetic solver error")
        return False, None, None, None

    monkeypatch.setattr(cv2, "solvePnPRansac", failed_solver)
    rgb = np.random.default_rng(0).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    points, _, _, visualization = run_experiment(
        tmp_path,
        monkeypatch,
        rgb,
        rgb,
        diagnose_gt_reprojection=diagnose_gt_reprojection,
        diagnose_pnp_refinement=diagnose_pnp_refinement,
    )

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
    if diagnose_gt_reprojection:
        assert (
            "GT reprojection diagnostic unavailable: no estimated pose" in diagnostics
        )
    assert "mean signed residual" not in diagnostics
    if diagnose_pnp_refinement:
        assert "PnP refinement unavailable: no estimated pose" in diagnostics
