"""Reference-free, frozen within-pair motion and native depth associations."""

from dataclasses import dataclass

import cv2
import numpy as np

from scene_recall.geometry.camera import (
    backproject_depth,
    project_points,
    reject_depth_edges,
    transform_points,
)
from scene_recall.geometry.transforms import is_rigid


@dataclass(frozen=True)
class TimingConfig:
    max_skew_s: float = 0.0201
    max_interval_s: float = 0.25
    clock_sigma_s: float = 0.002
    motion_fraction: float = 0.5
    max_translation_m: float = 0.03
    max_rotation_deg: float = 2.0
    association_radius_px: float = 1.0
    unknown_static: bool = False
    motion_margin_factor: float = 2.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "unknown_static":
                if not isinstance(value, bool):
                    raise ValueError("unknown_static must be boolean")
            elif name == "motion_margin_factor":
                if not np.isfinite(value) or value < 0:
                    raise ValueError(
                        "motion_margin_factor must be finite and nonnegative"
                    )
            elif not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, eq=False)
class PairMotion:
    T_RcRd: np.ndarray
    status: str
    depth_usable: bool
    translation_sigma_m: float = 0.0
    rotation_sigma_rad: float = 0.0
    skew_s: float | None = None


def pair_motion(poses, stamps, color_time, depth_time, config=None):
    """Interpolate only a single segment's training-view RGB pose proxies.

    Piecewise linear translation and shortest rotation interpolation approximate
    short exposure motion. One-sided continuation is limited to max_skew_s.
    Local velocity variation and an explicit clock allowance inflate uncertainty.
    No timestamp offset, calibration or velocity is fitted.
    """
    config = config or TimingConfig()
    identity = np.eye(4)
    if color_time is None or depth_time is None:
        return PairMotion(
            identity,
            "assumed_static" if config.unknown_static else "unknown_clock",
            config.unknown_static,
        )
    skew = depth_time - color_time
    if abs(skew) > config.max_skew_s:
        return PairMotion(identity, "skew_exceeds_limit", False, skew_s=skew)
    stamps = np.asarray(stamps, dtype=float)
    poses = np.asarray(poses)
    if len(stamps) < 2 or not np.isfinite(stamps).all() or (np.diff(stamps) <= 0).any():
        return PairMotion(identity, "unusable_training_clock", False, skew_s=skew)
    if any(not is_rigid(p) for p in poses):
        raise ValueError("motion interpolation requires rigid pose proxies")

    def interpolate(stamp):
        j = int(
            np.clip(
                np.searchsorted(stamps, stamp, side="right") - 1, 0, len(stamps) - 2
            )
        )
        interval = stamps[j + 1] - stamps[j]
        outside = max(stamps[0] - stamp, stamp - stamps[-1], 0)
        if interval > config.max_interval_s or outside > config.max_skew_s:
            return None, j, outside > 0
        alpha = (stamp - stamps[j]) / interval
        pose = np.eye(4)
        pose[:3, 3] = (1 - alpha) * poses[j, :3, 3] + alpha * poses[j + 1, :3, 3]
        rotation = cv2.Rodrigues(poses[j, :3, :3].T @ poses[j + 1, :3, :3])[0].ravel()
        pose[:3, :3] = poses[j, :3, :3] @ cv2.Rodrigues(alpha * rotation)[0]
        return pose, j, outside > 0

    color, j, extrap_color = interpolate(color_time)
    depth, k, extrap_depth = interpolate(depth_time)
    if color is None or depth is None:
        return PairMotion(identity, "unsupported_motion_interval", False, skew_s=skew)
    relative = np.linalg.inv(color) @ depth
    angle = float(np.linalg.norm(cv2.Rodrigues(relative[:3, :3])[0]))
    if (
        np.linalg.norm(relative[:3, 3]) > config.max_translation_m
        or np.degrees(angle) > config.max_rotation_deg
    ):
        return PairMotion(relative, "exposure_motion_exceeds_limit", False, skew_s=skew)
    velocities, angular = [], []
    for index in range(max(0, min(j, k) - 1), min(len(stamps) - 1, max(j, k) + 2)):
        interval = stamps[index + 1] - stamps[index]
        if interval > config.max_interval_s:
            continue
        velocities.append((poses[index + 1, :3, 3] - poses[index, :3, 3]) / interval)
        local = (
            cv2.Rodrigues(poses[index, :3, :3].T @ poses[index + 1, :3, :3])[0].ravel()
            / interval
        )
        angular.append(poses[index, :3, :3] @ local)
    velocities, angular = np.array(velocities), np.array(angular)
    speed = np.linalg.norm(velocities, axis=1).max()
    omega = np.linalg.norm(angular, axis=1).max()
    variation_t = np.linalg.norm(velocities - velocities.mean(axis=0), axis=1).max()
    variation_r = np.linalg.norm(angular - angular.mean(axis=0), axis=1).max()
    sigma_t = (config.motion_fraction * speed + variation_t) * abs(
        skew
    ) + speed * config.clock_sigma_s
    sigma_r = (config.motion_fraction * omega + variation_r) * abs(
        skew
    ) + omega * config.clock_sigma_s
    return PairMotion(
        relative,
        "estimated_extrapolated"
        if extrap_color or extrap_depth
        else "estimated_interpolated",
        True,
        float(sigma_t),
        float(sigma_r),
        float(skew),
    )


@dataclass(frozen=True, eq=False)
class NativeSamples:
    points_D: np.ndarray
    points_R: np.ndarray
    source_index: np.ndarray
    whitening_D: np.ndarray
    T_RcD: np.ndarray


def motion_radius(points_D, K_D, motion, factor):
    """Native-pixel safety radius from the stated engineering motion allowance."""
    p = points_D
    covariance = motion.translation_sigma_m**2 * np.eye(3)
    covariance = covariance + motion.rotation_sigma_rad**2 * (
        np.sum(p * p, axis=1)[:, None, None] * np.eye(3) - p[:, :, None] * p[:, None, :]
    )
    J = np.zeros((len(p), 2, 3))
    J[:, 0, 0], J[:, 1, 1] = K_D[0, 0] / p[:, 2], K_D[1, 1] / p[:, 2]
    J[:, 0, 2], J[:, 1, 2] = (
        -K_D[0, 0] * p[:, 0] / p[:, 2] ** 2,
        -K_D[1, 1] * p[:, 1] / p[:, 2] ** 2,
    )
    variance = np.trace(J @ covariance @ J.transpose(0, 2, 1), axis1=1, axis2=2)
    return np.ceil(factor * np.sqrt(np.maximum(variance, 0)))


def bind_depth(pixels, depth, calibration, motion, timing_config=None):
    """Bind unique native samples within a calibrated pixel footprint.

    Search neighboring RGB raster cells using native-equivalent ray distance.
    Keep actual native points and reject mixed surfaces before validity checks.
    A motion-dependent native margin protects depth holes and boundaries.
    """
    config = timing_config or TimingConfig()
    n = len(pixels)
    native = np.full((n, 3), np.nan)
    source = np.full(n, -1, dtype=np.int64)
    whitening = np.zeros((n, 3, 3))
    transform = motion.T_RcRd @ calibration.T_RD
    if not motion.depth_usable:
        return NativeSamples(native, native.copy(), source, whitening, transform)
    points = backproject_depth(depth, calibration.K_D).reshape(-1, 3)
    moved = transform_points(points, transform)
    projected = project_points(moved, calibration.K_R)
    height, width = calibration.rgb_shape
    valid = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] <= width - 1)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] <= height - 1)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return NativeSamples(native, native.copy(), source, whitening, transform)
    rounded = np.floor(projected[valid] + 0.5).astype(int)
    flat = rounded[:, 1] * width + rounded[:, 0]
    order = np.lexsort((indices, moved[valid, 2], flat))
    flat, indices = flat[order], indices[order]
    unique = np.r_[True, np.diff(flat) != 0]
    lookup = np.full(height * width, -1, dtype=np.int64)
    lookup[flat[unique]] = indices[unique]
    ratios = np.array(
        [
            calibration.K_R[0, 0] / calibration.K_D[0, 0],
            calibration.K_R[1, 1] / calibration.K_D[1, 1],
        ]
    )
    radius = np.ceil(ratios * config.association_radius_px + 0.5).astype(int)
    if radius.max() > 16:
        raise ValueError(
            "calibrated native footprint exceeds bounded association search"
        )
    grid = np.floor(pixels + 0.5).astype(int)
    best = np.full(n, np.inf)
    minimum_z, maximum_z = np.full(n, np.inf), np.full(n, -np.inf)
    for dy in range(-radius[1], radius[1] + 1):
        for dx in range(-radius[0], radius[0] + 1):
            x, y = grid[:, 0] + dx, grid[:, 1] + dy
            rows = np.flatnonzero((x >= 0) & (x < width) & (y >= 0) & (y < height))
            ids = lookup[y[rows] * width + x[rows]]
            keep = ids >= 0
            rows, ids = rows[keep], ids[keep]
            distances = np.linalg.norm((projected[ids] - pixels[rows]) / ratios, axis=1)
            keep = distances <= config.association_radius_px
            rows, ids, distances = rows[keep], ids[keep], distances[keep]
            minimum_z[rows] = np.minimum(minimum_z[rows], moved[ids, 2])
            maximum_z[rows] = np.maximum(maximum_z[rows], moved[ids, 2])
            better = (distances < best[rows]) | (
                (distances == best[rows]) & (ids < source[rows])
            )
            rows, ids, distances = rows[better], ids[better], distances[better]
            source[rows], best[rows] = ids, distances
    # A nearby raw foreground cannot silently expose a valid background sample.
    source[maximum_z - minimum_z > 0.05] = -1
    edge_valid = np.isfinite(reject_depth_edges(depth, 0.05))
    accepted = np.flatnonzero(source >= 0)
    accepted = accepted[edge_valid.ravel()[source[accepted]]]
    if len(accepted):
        radius = motion_radius(
            points[source[accepted]],
            calibration.K_D,
            motion,
            config.motion_margin_factor,
        )
        clearance = cv2.distanceTransform(
            edge_valid.astype(np.uint8), cv2.DIST_C, 3
        ).ravel()
        accepted = accepted[clearance[source[accepted]] > radius]
    ranking = sorted(accepted, key=lambda i: (float(best[i]), int(i)))
    seen, retained = set(), []
    for i in ranking:
        if int(source[i]) not in seen:
            seen.add(int(source[i]))
            retained.append(i)
    keep = np.array(retained, dtype=int)
    rejected = np.ones(n, dtype=bool)
    rejected[keep] = False
    source[rejected] = -1
    native[keep] = points[source[keep]]
    whitening[keep] = native_whitening(native[keep], calibration.K_D, motion)
    return NativeSamples(
        native, transform_points(native, transform), source, whitening, transform
    )


def native_whitening(p, K_D, motion):
    """Freeze native pixel/axial noise and engineering motion allowances."""
    B = np.zeros((len(p), 3, 3))
    B[:, 0, 0], B[:, 1, 1] = (
        p[:, 2] / K_D[0, 0],
        p[:, 2] / K_D[1, 1],
    )
    B[:, :, 2] = p / p[:, 2:3]
    variance = np.column_stack(
        (np.ones((len(p), 2)), (0.005 + 0.003 * p[:, 2] ** 2) ** 2)
    )
    covariance = (B * variance[:, None, :]) @ B.transpose(0, 2, 1)
    outer = p[:, :, None] * p[:, None, :]
    covariance += motion.translation_sigma_m**2 * np.eye(3)
    covariance += motion.rotation_sigma_rad**2 * (
        np.sum(p * p, axis=1)[:, None, None] * np.eye(3) - outer
    )
    return np.linalg.inv(np.linalg.cholesky(covariance))
