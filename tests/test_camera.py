import numpy as np

from scene_recall.geometry.camera import (
    backproject_depth,
    project_points,
    transform_points,
)


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


def test_transform_points_identity() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])

    transformed = transform_points(points, np.eye(4))

    np.testing.assert_array_equal(transformed, points)


def test_transform_points_translation_does_not_modify_input() -> None:
    points = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]])
    T_AB = np.eye(4)
    T_AB[:3, 3] = [2.0, -3.0, 4.0]

    transformed = transform_points(points, T_AB)

    np.testing.assert_array_equal(transformed, [[3.0, -1.0, 7.0], [2.0, -4.0, 6.0]])
    np.testing.assert_array_equal(points, [[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]])


def test_transform_points_rotation_and_translation() -> None:
    points = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    T_AB = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    transformed = transform_points(points, T_AB)

    np.testing.assert_array_equal(transformed, [[1.0, 3.0, 5.0], [0.0, 2.0, 3.0]])


def test_transform_points_preserves_dense_shape() -> None:
    points = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    T_AB = np.eye(4)
    T_AB[:3, 3] = [1.0, 2.0, 3.0]

    transformed = transform_points(points, T_AB)

    assert transformed.shape == (2, 2, 3)
    np.testing.assert_array_equal(transformed[1, 0], [1.0, 3.0, 3.0])


def test_transform_points_propagates_nan() -> None:
    points = np.array([[np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]])
    T_AB = np.eye(4)
    T_AB[:3, 3] = [2.0, -3.0, 4.0]

    transformed = transform_points(points, T_AB)

    assert np.isnan(transformed[0]).all()
    np.testing.assert_array_equal(transformed[1], [3.0, -1.0, 7.0])


def test_project_points_known_pixels() -> None:
    K = np.array([[4.0, 0.0, 2.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
    points = np.array([[0.0, 0.0, 5.0], [2.0, -3.0, 2.0]])

    pixels = project_points(points, K)

    np.testing.assert_array_equal(pixels, [[2.0, 3.0], [6.0, 0.0]])


def test_project_points_preserves_leading_dimensions() -> None:
    K = np.eye(3)
    points = np.array(
        [
            [[0.0, 0.0, 1.0], [2.0, 0.0, 2.0]],
            [[0.0, 3.0, 3.0], [4.0, 6.0, 2.0]],
        ]
    )

    pixels = project_points(points, K)

    assert pixels.shape == (2, 2, 2)
    np.testing.assert_array_equal(
        pixels, [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [2.0, 3.0]]]
    )


def test_project_points_non_projectable_points() -> None:
    points = np.array(
        [
            [1.0, 2.0, 0.0],
            [1.0, 2.0, -1.0],
            [np.nan, 0.0, 2.0],
            [0.0, np.inf, 2.0],
            [0.0, 0.0, np.inf],
            [0.0, 0.0, 1.0],
        ]
    )

    pixels = project_points(points, np.eye(3))

    assert np.isnan(pixels[:5]).all()
    np.testing.assert_array_equal(pixels[5], [0.0, 0.0])


def test_backproject_project_round_trip() -> None:
    depth = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    K = np.array([[2.0, 0.0, 0.5], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]])

    pixels = project_points(backproject_depth(depth, K), K)

    np.testing.assert_allclose(
        pixels, [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]
    )


def test_depth_pixel_to_rgb_pixel() -> None:
    depth = np.full((2, 3), np.nan, dtype=np.float32)
    depth[1, 2] = 4.0
    K_D = np.array([[4.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    T_RD = np.array(
        [
            [0.0, -1.0, 0.0, 3.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    K_R = np.array([[6.0, 0.0, 2.0], [0.0, 3.0, 4.0], [0.0, 0.0, 1.0]])

    points_D = backproject_depth(depth, K_D)
    points_R = transform_points(points_D, T_RD)
    pixels_R = project_points(points_R, K_R)

    np.testing.assert_array_equal(points_D[1, 2], [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(points_R[1, 2], [1.0, 2.0, 6.0])
    np.testing.assert_array_equal(pixels_R[1, 2], [3.0, 5.0])
