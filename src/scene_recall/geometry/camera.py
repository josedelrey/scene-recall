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


def reject_depth_edges(depth: np.ndarray, threshold_m: float) -> np.ndarray:
    """Reject borders, holes, and 3x3 depth ranges above threshold_m.

    This conservative surface test preserves the input and does not interpolate
    across discontinuities. The threshold is an absolute z-depth range in meters.
    """
    if not np.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("depth edge threshold must be finite and positive")
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(depth, 1, constant_values=np.nan), (3, 3)
    )
    valid = np.isfinite(windows).all(axis=(-2, -1)) & (windows > 0).all(axis=(-2, -1))
    continuous = (
        windows.max(axis=(-2, -1)) - windows.min(axis=(-2, -1))
    ) <= threshold_m
    return np.where(valid & continuous, depth, np.nan).astype(depth.dtype)


def register_depth_to_rgb(
    depth: np.ndarray,
    K_D: np.ndarray,
    K_R: np.ndarray,
    T_RD: np.ndarray,
    rgb_shape: tuple[int, int],
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Splat depth into the RGB grid with a nearest-pixel z-buffer.

    Output is RGB-frame z-depth. The nearest positive surface wins collisions.
    Unobserved pixels remain NaN. Optional native-grid validity is applied after
    visibility selection, so rejected foreground cannot expose background points.
    This registration neither fills holes nor models subpixel surface extent.
    """
    points_R = transform_points(backproject_depth(depth, K_D), T_RD).reshape(-1, 3)
    pixels = project_points(points_R, K_R)
    height, width = rgb_shape
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= height - 1)
    )
    indices = np.floor(pixels[valid] + 0.5).astype(np.intp)
    registered = np.full(height * width, np.inf)
    np.minimum.at(registered, indices[:, 1] * width + indices[:, 0], points_R[valid, 2])
    if valid_mask is not None:
        if valid_mask.shape != depth.shape or valid_mask.dtype != np.dtype(bool):
            raise ValueError("valid mask must be boolean and match the depth grid")
        flat_indices = indices[:, 1] * width + indices[:, 0]
        foreground = points_R[valid, 2] <= registered[flat_indices] + 1e-6
        rejected = foreground & ~valid_mask.reshape(-1)[valid]
        registered[flat_indices[rejected]] = np.nan
    registered[~np.isfinite(registered)] = np.nan
    return registered.reshape(rgb_shape)
