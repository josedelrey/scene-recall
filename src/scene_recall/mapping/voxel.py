"""Deterministic surface samples in a bounded, segment-local voxel grid."""

from dataclasses import dataclass

import numpy as np

from scene_recall.data import Calibration, Observation
from scene_recall.geometry.camera import (
    backproject_depth,
    project_points,
    register_depth_to_rgb,
    reject_depth_edges,
    transform_points,
)
from scene_recall.odometry.pipeline import is_rigid


@dataclass(frozen=True)
class MappingConfig:
    voxel_size_m: float = 0.02
    pixel_stride: int = 4
    min_depth_m: float = 0.3
    max_depth_m: float = 4.0
    depth_edge_threshold_m: float = 0.05
    keyframe_stride: int = 10
    max_keyframes_per_submap: int = 24
    max_voxels: int = 500_000
    consistency_tolerance_m: float = 0.05

    def __post_init__(self):
        for name in (
            "voxel_size_m",
            "min_depth_m",
            "max_depth_m",
            "depth_edge_threshold_m",
            "consistency_tolerance_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError("require min_depth_m < max_depth_m")
        for name in (
            "pixel_stride",
            "keyframe_stride",
            "max_keyframes_per_submap",
            "max_voxels",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SurfaceSamples:
    points: np.ndarray
    colors: np.ndarray
    colored: np.ndarray
    depth: np.ndarray


def sample_observation(
    observation: Observation, calibration: Calibration, config: MappingConfig
) -> SurfaceSamples:
    """Filter metric depth, sample its native grid and project visible RGB.

    Geometry never depends on RGB availability. A native-depth z-buffer in C_R
    prevents a background depth sample from taking foreground color. This is a
    nearest-pixel visibility approximation, without hole filling or distortion.
    """
    if observation.depth.shape != calibration.depth_shape:
        raise ValueError("depth shape must match calibration")
    if observation.rgb.shape[:2] != calibration.rgb_shape:
        raise ValueError("RGB shape must match calibration")
    if not is_rigid(calibration.T_RD):
        raise ValueError("mapping requires rigid RGB/depth extrinsics")
    depth = reject_depth_edges(observation.depth, config.depth_edge_threshold_m)
    depth = np.where(
        (depth >= config.min_depth_m) & (depth <= config.max_depth_m), depth, np.nan
    ).astype(np.float32)
    stride = config.pixel_stride
    grid = backproject_depth(depth, calibration.K_D)[::stride, ::stride]
    points = grid[np.isfinite(grid).all(axis=-1)]
    points_R = transform_points(points, calibration.T_RD)
    pixels = project_points(points_R, calibration.K_R)
    height, width = calibration.rgb_shape
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= height - 1)
    )
    indices = np.floor(pixels[inside] + 0.5).astype(np.intp)
    zbuffer = register_depth_to_rgb(
        observation.depth,
        calibration.K_D,
        calibration.K_R,
        calibration.T_RD,
        calibration.rgb_shape,
    )
    visible = points_R[inside, 2] <= zbuffer[indices[:, 1], indices[:, 0]] + 1e-6
    colored = inside.copy()
    colored[inside] = visible
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    colors[colored] = observation.rgb[indices[visible, 1], indices[visible, 0]]
    return SurfaceSamples(points, colors, colored, depth)


@dataclass(frozen=True)
class VoxelCloud:
    keys: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    frame_support: np.ndarray
    color_support: np.ndarray
    position_std_m: np.ndarray


class VoxelMap:
    """Average one centroid per voxel per keyframe, with distinct-frame support.

    Equal frame weighting limits dependence on pixel density. Moments describe
    the scatter of frame centroids, not calibrated uncertainty or surface error.
    No free-space integration, meshing or normal estimation is performed.
    Capacity overflow raises before mutation instead of silently pruning data.
    """

    def __init__(self, voxel_size_m: float, max_voxels: int = 500_000):
        if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise ValueError("voxel size must be finite and positive")
        if (
            not isinstance(max_voxels, int)
            or isinstance(max_voxels, bool)
            or max_voxels <= 0
        ):
            raise ValueError("max_voxels must be a positive integer")
        self.voxel_size_m = voxel_size_m
        self.max_voxels = max_voxels
        self._cells = {}
        self._frames = set()

    def integrate(
        self, samples: SurfaceSamples, T_MC_D: np.ndarray, frame_index: int
    ) -> None:
        if not is_rigid(T_MC_D):
            raise ValueError("integration requires a rigid camera-to-map pose")
        if frame_index in self._frames:
            raise ValueError("a keyframe may be integrated only once")
        points = transform_points(samples.points, T_MC_D)
        if not np.isfinite(points).all():
            raise ValueError("surface points must be finite")
        scaled = np.floor(points / self.voxel_size_m)
        if np.any(np.abs(scaled) >= 2**62):
            raise ValueError("voxel coordinates exceed supported range")
        keys, inverse, counts = np.unique(
            scaled.astype(np.int64), axis=0, return_inverse=True, return_counts=True
        )
        tuples = [tuple(key) for key in keys]
        if (
            len(self._cells) + sum(key not in self._cells for key in tuples)
            > self.max_voxels
        ):
            raise ValueError(
                "submap voxel capacity exceeded, reduce submap size or density"
            )
        centroids = np.zeros((len(keys), 3))
        np.add.at(centroids, inverse, points)
        centroids /= counts[:, None]
        color_counts = np.bincount(inverse[samples.colored], minlength=len(keys))
        color_means = np.zeros((len(keys), 3))
        np.add.at(
            color_means, inverse[samples.colored], samples.colors[samples.colored]
        )
        np.divide(
            color_means,
            color_counts[:, None],
            out=color_means,
            where=color_counts[:, None] > 0,
        )
        for key, point, color, colored in zip(
            tuples, centroids, color_means, color_counts > 0
        ):
            if key not in self._cells:
                self._cells[key] = [np.zeros(3), np.zeros(3), 0, np.zeros(3), 0]
            cell = self._cells[key]
            cell[0] += point
            cell[1] += point**2
            cell[2] += 1
            if colored:
                cell[3] += color
                cell[4] += 1
        self._frames.add(frame_index)

    def cloud(self) -> VoxelCloud:
        keys = sorted(self._cells)
        points, colors, support, color_support, scatter = [], [], [], [], []
        for key in keys:
            total, square, count, color, colored = self._cells[key]
            point = total / count
            points.append(point)
            scatter.append(np.sqrt(np.maximum(0, square / count - point**2).sum()))
            colors.append(np.rint(color / colored) if colored else np.zeros(3))
            support.append(count)
            color_support.append(colored)
        return VoxelCloud(
            np.array(keys, dtype=np.int64).reshape(-1, 3),
            np.array(points, dtype=np.float64).reshape(-1, 3),
            np.array(colors, dtype=np.uint8).reshape(-1, 3),
            np.array(support, dtype=np.int64),
            np.array(color_support, dtype=np.int64),
            np.array(scatter, dtype=np.float64),
        )
