import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

experiment = runpy.run_path(
    Path(__file__).parents[1] / "experiments" / "scannet_sift.py"
)


def run_experiment(tmp_path, monkeypatch, rgb_0, rgb_1):
    class RGBReader:
        sequence_id = "synthetic"

        def __init__(self, sens_path, *, start, stop):
            assert (start, stop) == (0, 2)
            self.frames = iter([SimpleNamespace(rgb=rgb_0), SimpleNamespace(rgb=rgb_1)])
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def __next__(self):
            return next(self.frames)

    main = experiment["main"]
    monkeypatch.setitem(main.__globals__, "ScanNetReader", RGBReader)
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
        pixels_0 = matches["pixels_0"]
        pixels_1 = matches["pixels_1"]
    visualization = cv2.imread(str(tmp_path / "synthetic_frames0_1_sift.png"))
    return pixels_0, pixels_1, visualization


def test_translated_texture_preserves_pixel_correspondence(
    tmp_path, monkeypatch, capsys
) -> None:
    rng = np.random.default_rng(0)
    rgb_0 = rng.integers(0, 256, size=(192, 256, 3), dtype=np.uint8)
    rgb_1 = np.zeros_like(rgb_0)
    rgb_1[5:, 7:] = rgb_0[:-5, :-7]

    pixels_0, pixels_1, visualization = run_experiment(
        tmp_path, monkeypatch, rgb_0, rgb_1
    )

    assert pixels_0.shape == pixels_1.shape
    assert pixels_0.dtype == pixels_1.dtype == np.float32
    assert len(pixels_0) > 20
    offsets = pixels_1 - pixels_0
    assert (np.linalg.norm(offsets - [7, 5], axis=1) < 0.5).mean() > 0.95
    assert visualization.shape == (192, 512, 3)
    diagnostics = capsys.readouterr().out
    assert "KNN matches (query rows, k=2):" in diagnostics
    assert f"Matches surviving ratio test (0.75): {len(pixels_0)}" in diagnostics
    assert "Retention ratio (accepted / KNN rows):" in diagnostics


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

    pixels_0, pixels_1, visualization = run_experiment(tmp_path, monkeypatch, *frames)

    assert pixels_0.shape == pixels_1.shape == (0, 2)
    np.testing.assert_array_equal(
        visualization[:, blank_frame * 256 : (blank_frame + 1) * 256],
        blank[..., ::-1],
    )
    diagnostics = capsys.readouterr().out
    assert f"Keypoints in frame {blank_frame}: 0" in diagnostics
    assert "KNN matches (query rows, k=2): 0" in diagnostics
    assert "Retention ratio (accepted / KNN rows): 0.0000" in diagnostics


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

    pixels_0, pixels_1, _ = run_experiment(tmp_path, monkeypatch, rgb, rgb)

    assert pixels_0.shape == pixels_1.shape == (0, 2)
    diagnostics = capsys.readouterr().out
    assert "KNN matches (query rows, k=2): 1" in diagnostics
    assert "Matches surviving ratio test (0.75): 0" in diagnostics
