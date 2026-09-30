import numpy as np

from scene_recall.geometry.camera import backproject_depth


def test_backproject_depth_known_pixels() -> None:
    depth = np.ones((3, 4), dtype=np.float32)
    depth[1, 1] = 2.0
    depth[0, 3] = 4.0
    depth[2, 0] = 8.0
    K = np.array([[2.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]])

    points = backproject_depth(depth, K)

    assert points.shape == (3, 4, 3)
    np.testing.assert_allclose(points[1, 1], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(points[0, 3], [4.0, -1.0, 4.0])
    np.testing.assert_allclose(points[2, 0], [-4.0, 2.0, 8.0])


def test_backproject_depth_propagates_nan() -> None:
    depth = np.array([[np.nan, 1.0]], dtype=np.float32)
    K = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])

    points = backproject_depth(depth, K)

    assert np.isnan(points[0, 0]).all()
    np.testing.assert_allclose(points[0, 1], [0.5, 0.0, 1.0])
