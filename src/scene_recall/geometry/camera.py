"""Camera geometry for canonical RGB-D views."""

import numpy as np


def backproject_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Map z-depth pixels to 3D points in the depth optical frame."""
    v, u = np.indices(depth.shape)
    x = depth * (u - K[0, 2]) / K[0, 0]
    y = depth * (v - K[1, 2]) / K[1, 1]
    return np.stack((x, y, depth), axis=-1)


def backproject_rgb_pixels(
    pixels: np.ndarray, depth: np.ndarray, K_R: np.ndarray, K_D: np.ndarray
) -> np.ndarray:
    """Backproject RGB pixels using z-depth in a shared optical camera frame.

    RGB and depth must share the optical frame, with canonical pinhole intrinsics
    for their respective grids. Map RGB rays to the depth grid via K_D @ inv(K_R).
    Accept finite depth coordinates within [0, W_D - 1] x [0, H_D - 1], inclusive.
    Nearest-neighbor sampling uses floor(coordinate + 0.5), rounding ties up.
    Backproject the original RGB ray using the sampled z-depth in meters.

    Input pixels have shape (..., 2) in (u, v) order. Output points have shape
    (..., 3), preserving correspondence. Invalid pixels, outside coordinates,
    and nonfinite or nonpositive depth produce all-NaN points.
    """
    homogeneous = np.concatenate((pixels, np.ones(pixels.shape[:-1] + (1,))), axis=-1)
    with np.errstate(invalid="ignore"):
        rays = homogeneous @ np.linalg.inv(K_R).T
        depth_pixels = (rays @ K_D.T)[..., :2]
    height, width = depth.shape
    sampled = (
        np.isfinite(depth_pixels).all(axis=-1)
        & (depth_pixels[..., 0] >= 0)
        & (depth_pixels[..., 0] <= width - 1)
        & (depth_pixels[..., 1] >= 0)
        & (depth_pixels[..., 1] <= height - 1)
    )
    indices = np.floor(depth_pixels[sampled] + 0.5).astype(np.intp)
    z = np.full(pixels.shape[:-1], np.nan)
    z[sampled] = depth[indices[:, 1], indices[:, 0]]
    z[~np.isfinite(z) | (z <= 0)] = np.nan
    return rays * z[..., None]


def transform_points(points: np.ndarray, T_AB: np.ndarray) -> np.ndarray:
    """Transform points from frame B to frame A."""
    return points @ T_AB[:3, :3].T + T_AB[:3, 3]


def project_points(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame points to (u, v) pixel coordinates."""
    valid = np.isfinite(points).all(axis=-1, keepdims=True) & (points[..., 2:3] > 0)
    xy = np.full(points.shape[:-1] + (2,), np.nan)
    np.divide(points[..., :2], points[..., 2:3], out=xy, where=valid)
    return xy * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
