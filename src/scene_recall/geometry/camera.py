"""Pinhole camera geometry for canonical depth views."""

import numpy as np


def backproject_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Map z-depth pixels to 3D points in the depth optical frame."""
    v, u = np.indices(depth.shape)
    x = depth * (u - K[0, 2]) / K[0, 0]
    y = depth * (v - K[1, 2]) / K[1, 1]
    return np.stack((x, y, depth), axis=-1)
