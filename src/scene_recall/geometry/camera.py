"""Camera geometry for canonical RGB-D views."""

import numpy as np


def backproject_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Map z-depth pixels to 3D points in the depth optical frame."""
    v, u = np.indices(depth.shape)
    x = depth * (u - K[0, 2]) / K[0, 0]
    y = depth * (v - K[1, 2]) / K[1, 1]
    return np.stack((x, y, depth), axis=-1)


def transform_points(points: np.ndarray, T_AB: np.ndarray) -> np.ndarray:
    """Transform points from frame B to frame A."""
    return points @ T_AB[:3, :3].T + T_AB[:3, 3]


def project_points(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame points to (u, v) pixel coordinates."""
    valid = np.isfinite(points).all(axis=-1, keepdims=True) & (points[..., 2:3] > 0)
    xy = np.full(points.shape[:-1] + (2,), np.nan)
    np.divide(points[..., :2], points[..., 2:3], out=xy, where=valid)
    return xy * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
