"""Bounded offline projective TSDF fusion with observed-cell surface extraction."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from scene_recall.geometry.camera import (
    backproject_depth,
    project_points,
    transform_points,
)
from scene_recall.geometry.transforms import is_rigid
from scene_recall.mapping.mesh import TriangleMesh


@dataclass(frozen=True)
class TSDFConfig:
    voxel_size_m: float = 0.02
    truncation_m: float = 0.06
    max_grid_voxels: int = 20_000_000
    max_active_voxels: int = 2_000_000

    def __post_init__(self):
        for name in ("voxel_size_m", "truncation_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.truncation_m < self.voxel_size_m:
            raise ValueError("truncation must span at least one voxel")
        for name in ("max_grid_voxels", "max_active_voxels"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TSDFView:
    """One already-filtered depth lattice and its camera-to-map pose.

    The caller supplies training views from one connected map only. Use the
    same sampled lattice as the baseline to avoid changing depth information.
    Optional RGB and validity are on that lattice and never influence geometry.
    """

    frame_index: int
    pose: np.ndarray
    depth: np.ndarray
    colors: np.ndarray | None = None
    colored: np.ndarray | None = None


@dataclass(frozen=True)
class TSDFSurface:
    """One point per metric cell, with distinct near-surface view counts.

    Counts and signed residual spread describe projective depth agreement at
    the extracted point. They differ from baseline within-voxel frame support.
    """

    keys: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    view_support: np.ndarray
    color_support: np.ndarray
    signed_residual_std_m: np.ndarray


def _project(points, view, K):
    camera = transform_points(points, np.linalg.inv(view.pose))
    pixels = project_points(camera, K)
    h, w = view.depth.shape
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= w - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= h - 1)
    )
    ids = np.flatnonzero(inside)
    uv = np.floor(pixels[inside] + 0.5).astype(np.intp)
    depth = view.depth[uv[:, 1], uv[:, 0]]
    valid = np.isfinite(depth) & (depth > 0)
    return ids[valid], depth[valid] - camera[ids[valid], 2], uv[valid]


@dataclass(frozen=True)
class TSDFVolume:
    config: TSDFConfig
    origin_key: np.ndarray
    grid_shape: tuple[int, int, int]
    keys: np.ndarray
    signed_distance_m: np.ndarray
    frame_support: np.ndarray

    def _surface_cells(self):
        size = self.config.voxel_size_m
        field = np.full(self.grid_shape, np.nan)
        local = self.keys - self.origin_key
        field[tuple(local.T)] = np.where(
            self.frame_support > 0, self.signed_distance_m, np.nan
        )
        shape = tuple(np.maximum(np.array(self.grid_shape) - 1, 0))
        low, high = np.full(shape, np.inf), np.full(shape, -np.inf)
        known = np.ones(shape, dtype=bool)
        corners = np.array(list(product((0, 1), repeat=3)))
        for corner in corners:
            values = field[
                tuple(slice(int(c), int(c + n)) for c, n in zip(corner, shape))
            ]
            known &= np.isfinite(values)
            np.minimum(low, values, out=low)
            np.maximum(high, values, out=high)
        cubes = np.argwhere(known & (low <= 0) & (high > 0))
        values = field[tuple((cubes[:, None] + corners).transpose(2, 0, 1))]
        sums, counts = np.zeros((len(cubes), 3)), np.zeros(len(cubes), dtype=np.int64)
        for a, first in enumerate(corners):
            for axis in range(3):
                second = first.copy()
                second[axis] += 1
                if second[axis] != 1:
                    continue
                b = int(np.flatnonzero((corners == second).all(axis=1))[0])
                x, y = values[:, a], values[:, b]
                crossing = ((x <= 0) & (y > 0)) | ((y <= 0) & (x > 0))
                t = x[crossing] / (x[crossing] - y[crossing])
                sums[crossing] += first + t[:, None] * (second - first)
                counts[crossing] += 1
        points = (
            cubes + self.origin_key + 0.5 + sums / np.maximum(counts[:, None], 1)
        ) * size
        return field, cubes, points

    def mesh(self) -> TriangleMesh:
        """Connect observed surface cells around sign-changing grid edges.

        Each quad requires four fully observed surface cells. Unknown regions
        and missing neighboring cells stay open, without extrapolation. Split
        every quad along a fixed diagonal. This surface-nets extraction does
        not promise watertightness or resolve ambiguous multi-surface cells.
        """
        field, cubes, points = self._surface_cells()
        cube_shape = tuple(n - 1 for n in self.grid_shape)
        index = np.full(cube_shape, -1, dtype=np.int32)
        index[tuple(cubes.T)] = np.arange(len(cubes))
        faces = []
        for axis in range(3):
            first, second = [slice(None)] * 3, [slice(None)] * 3
            first[axis], second[axis] = slice(None, -1), slice(1, None)
            a, b = field[tuple(first)], field[tuple(second)]
            crossings = np.argwhere(((a <= 0) & (b > 0)) | ((b <= 0) & (a > 0)))
            offsets = np.zeros((4, 3), dtype=np.int64)
            offsets[1:3, (axis + 1) % 3] = -1
            offsets[2:4, (axis + 2) % 3] = -1
            neighbors = crossings[:, None] + offsets
            inside = ((neighbors >= 0) & (neighbors < np.array(cube_shape))).all(
                axis=(1, 2)
            )
            crossings, neighbors = crossings[inside], neighbors[inside]
            quads = index[tuple(neighbors.transpose(2, 0, 1))]
            complete = (quads >= 0).all(axis=1)
            quads, crossings = quads[complete], crossings[complete]
            outward = a[tuple(crossings.T)] <= 0
            for indices in ([0, 1, 2], [0, 2, 3]):
                triangles = quads[:, indices].copy()
                triangles[~outward] = triangles[~outward, ::-1]
                faces.append(triangles)
        return TriangleMesh(points, np.concatenate(faces))

    def surface(self, views: list[TSDFView], K: np.ndarray) -> TSDFSurface:
        """Return revoxelized zero crossings for the legacy disc control.

        Unknown corners never acquire an invented sign. Counts are measured
        at the extracted points, independently of the TSDF interpolation.
        """
        field, _, vertices = self._surface_cells()
        del field
        size = self.config.voxel_size_m
        keys, inverse, count = np.unique(
            np.floor(vertices / size).astype(np.int64),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        points = np.zeros((len(keys), 3))
        np.add.at(points, inverse, vertices)
        points /= count[:, None]
        support, colored = (
            np.zeros(len(points), dtype=np.int64),
            np.zeros(len(points), dtype=np.int64),
        )
        total, square, colors = (
            np.zeros(len(points)),
            np.zeros(len(points)),
            np.zeros((len(points), 3)),
        )
        for view in sorted(views, key=lambda v: v.frame_index):
            ids, residual, uv = _project(points, view, K)
            near = np.abs(residual) <= self.config.truncation_m
            ids, residual, uv = ids[near], residual[near], uv[near]
            support[ids] += 1
            total[ids] += residual
            square[ids] += residual**2
            if view.colors is not None:
                valid = view.colored[uv[:, 1], uv[:, 0]]
                colors[ids[valid]] += view.colors[uv[valid, 1], uv[valid, 0]]
                colored[ids[valid]] += 1
        mean = total / np.maximum(support, 1)
        std = np.sqrt(np.maximum(0, square / np.maximum(support, 1) - mean**2))
        return TSDFSurface(
            keys,
            points,
            np.rint(colors / np.maximum(colored[:, None], 1)).astype(np.uint8),
            support,
            colored,
            std,
        )


def fuse_tsdf(
    views: list[TSDFView], K: np.ndarray, config: TSDFConfig | None = None
) -> TSDFVolume:
    """Fuse equal votes per view with positive free space and truncated negatives.

    Allocate the union of training depth bands, with one-cell spatial dilation.
    Evaluate all training views on that fixed active set, so earlier free-space
    evidence also applies to later-allocated surfaces. This is deliberately an
    offline algorithm. Dense scratch indexing is bounded separately from the
    sparse persisted field. Capacity failure never silently prunes geometry.
    """
    config = config or TSDFConfig()
    views = sorted(views, key=lambda v: v.frame_index)
    if not views or len({v.frame_index for v in views}) != len(views):
        raise ValueError("supply nonempty distinct training views")
    if len(views) > np.iinfo(np.uint16).max:
        raise ValueError("too many training views")
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("invalid pinhole calibration")
    size, mu = config.voxel_size_m, config.truncation_m
    samples, low, high = [], np.full(3, np.inf), np.full(3, -np.inf)
    for view in views:
        if not is_rigid(view.pose) or view.depth.ndim != 2:
            raise ValueError("views require rigid poses and 2D depth")
        if view.colors is not None and (
            view.colors.shape != view.depth.shape + (3,)
            or view.colored is None
            or view.colored.shape != view.depth.shape
        ):
            raise ValueError("color arrays must share the depth lattice")
        points = backproject_depth(view.depth, K).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
        rays = points / points[:, 2:3]
        samples.append((points, rays))
        for offset in (-mu, mu):
            band = transform_points(points + offset * rays, view.pose)
            if len(band):
                low = np.minimum(low, band.min(axis=0))
                high = np.maximum(high, band.max(axis=0))
    if not np.isfinite(low).all():
        return TSDFVolume(
            config,
            np.zeros(3, dtype=np.int64),
            (2, 2, 2),
            np.empty((0, 3), dtype=np.int64),
            np.empty(0),
            np.empty(0, dtype=np.uint16),
        )
    bounds = np.stack((np.floor(low / size) - 2, np.floor(high / size) + 2))
    if np.any(np.abs(bounds) >= 2**62):
        raise ValueError("TSDF coordinates exceed supported range")
    origin = bounds[0].astype(np.int64)
    shape = tuple(int(n) for n in bounds[1] - bounds[0] + 1)
    if np.prod(shape, dtype=object) > config.max_grid_voxels:
        raise ValueError("TSDF scratch grid capacity exceeded")
    active = np.zeros(shape, dtype=bool)
    offsets = np.linspace(-mu, mu, int(np.ceil(2 * mu / size)) + 1)
    for view, (points, rays) in zip(views, samples, strict=True):
        for offset in offsets:
            p = transform_points(points + offset * rays, view.pose)
            keys = np.floor(p / size).astype(np.int64) - origin
            active[tuple(keys.T)] = True
    for axis in range(3):
        before, after = [slice(None)] * 3, [slice(None)] * 3
        before[axis], after[axis] = slice(None, -1), slice(1, None)
        dilated = active.copy()
        dilated[tuple(before)] |= active[tuple(after)]
        dilated[tuple(after)] |= active[tuple(before)]
        active = dilated
    if int(active.sum()) > config.max_active_voxels:
        raise ValueError("TSDF active voxel capacity exceeded")
    keys = np.argwhere(active) + origin
    total, support = np.zeros(len(keys)), np.zeros(len(keys), dtype=np.uint16)
    for start in range(0, len(keys), 100_000):
        points = (keys[start : start + 100_000] + 0.5) * size
        for view in views:
            ids, residual, _ = _project(points, view, K)
            visible = residual >= -mu
            ids, residual = ids[visible] + start, residual[visible]
            total[ids] += np.minimum(residual, mu)
            support[ids] += 1
    return TSDFVolume(
        config, origin, shape, keys, total / np.maximum(support, 1), support
    )
