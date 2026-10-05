import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from scene_recall.data import Calibration, Observation, Sequence
from scene_recall.geometry.camera import backproject_depth

exporter = runpy.run_path(
    Path(__file__).parents[1] / "scripts" / "scannet_point_cloud.py"
)
sample_rgb_colors = exporter["sample_rgb_colors"]


@pytest.mark.parametrize("color", [False, True])
def test_export_frame_with_only_valid_center_pixel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, color: bool
) -> None:
    depth = np.full((3, 3), np.nan, dtype=np.float32)
    depth[1, 1] = 2.0
    K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    calibration = Calibration(K, K, np.eye(4), (3, 3), (3, 3))
    rgb = np.full((3, 3, 3), [10, 20, 30], dtype=np.uint8)
    sequence = Sequence("center", calibration, (Observation(0, rgb, depth),))
    output = tmp_path / "center.ply"
    argv = [
        "scannet_point_cloud.py",
        "--sens-path",
        "unused.sens",
        "--output",
        str(output),
    ]
    if color:
        argv.append("--color")
    main = exporter["main"]
    monkeypatch.setitem(
        main.__globals__, "load_scannet_frame", lambda *args, **kwargs: sequence
    )
    monkeypatch.setattr(sys, "argv", argv)

    main()

    header, payload = output.read_bytes().split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0\n" in header
    assert b"element vertex 1\n" in header
    assert len(payload) == (27 if color else 24)
    np.testing.assert_array_equal(np.frombuffer(payload[:24], dtype="<f8"), [0, 0, 2])
    if color:
        assert (
            b"property uchar red\nproperty uchar green\nproperty uchar blue\n" in header
        )
        assert payload[24:] == bytes([10, 20, 30])


def test_color_association_with_different_grids_and_half_pixel_ties() -> None:
    depth = np.array([[2, 2, np.nan], [2, 2, 2]], dtype=np.float32)
    K_D = np.array([[2, 0, 1], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
    K_R = np.array([[4, 0, 2], [0, 4, 0], [0, 0, 1]], dtype=np.float64)
    T_RD = np.eye(4)
    T_RD[:3, 3] = [0.25, 0.125, 0]
    v, u = np.indices((4, 8))
    rgb = np.stack((20 * u, 50 * v, u + v), axis=-1).astype(np.uint8)
    points_D = backproject_depth(depth, K_D)[np.isfinite(depth)]
    original_points = points_D.copy()

    colors, sampled = sample_rgb_colors(points_D, rgb, K_R, T_RD)

    # C_D points project to (0.5, 0.25), (2.5, 0.25), (0.5, 2.25),
    # (2.5, 2.25), and (4.5, 2.25) in C_R, using the nonidentity T_RD.
    np.testing.assert_array_equal(
        colors, [[20, 0, 1], [60, 0, 3], [20, 100, 3], [60, 100, 5], [100, 100, 7]]
    )
    assert colors.dtype == np.uint8
    assert sampled.all()
    np.testing.assert_array_equal(points_D, original_points)


def test_invalid_and_outside_projections_keep_black_colors() -> None:
    rgb = np.full((4, 8, 3), [10, 20, 30], dtype=np.uint8)
    points_D = np.array(
        [
            [0, 0, 1],
            [7, 3, 1],
            [0.5, 0.5, 1],
            [-0.1, 0, 1],
            [7.1, 3, 1],
            [0, 3.1, 1],
            [0, 0, -1],
            [np.nan, 0, 1],
            [0, np.inf, 1],
        ]
    )
    # A nonfinite point is intentionally included to exercise rejection.
    with np.errstate(invalid="ignore"):
        colors, sampled = sample_rgb_colors(points_D, rgb, np.eye(3), np.eye(4))

    np.testing.assert_array_equal(
        sampled, [True, True, True, False, False, False, False, False, False]
    )
    np.testing.assert_array_equal(colors[:3], np.full((3, 3), [10, 20, 30]))
    np.testing.assert_array_equal(colors[3:], np.zeros((6, 3), dtype=np.uint8))
    assert int((~sampled).sum()) == 6
