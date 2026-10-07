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
    tmp_path, monkeypatch, rgb_0, rgb_1, *, depth=None, calibration=None
):
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
            assert (start, stop) == (0, 2)
            self.calibration = calibration
            self.frames = iter(
                [SimpleNamespace(rgb=rgb_0, depth=depth), SimpleNamespace(rgb=rgb_1)]
            )
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def __next__(self):
            return next(self.frames)

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
        ],
    )
    main()
    with np.load(tmp_path / "synthetic_frames0_1_sift.npz") as matches:
        points_C0 = matches["points_C0"]
        pixels_0 = matches["pixels_0"]
        pixels_1 = matches["pixels_1"]
    visualization = cv2.imread(str(tmp_path / "synthetic_frames0_1_sift.png"))
    return points_C0, pixels_0, pixels_1, visualization


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
