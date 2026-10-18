"""Target-independent triangle depth rasterization and portable mesh export."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.odometry.pipeline import is_rigid


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    triangles: np.ndarray


def _clip_near(triangles, near):
    inside = triangles[..., 2] >= near
    accepted = [triangles[inside.all(axis=1)]]
    for triangle in triangles[inside.any(axis=1) & ~inside.all(axis=1)]:
        polygon = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            if a[2] >= near:
                polygon.append(a)
            if (a[2] >= near) != (b[2] >= near):
                polygon.append(a + (near - a[2]) / (b[2] - a[2]) * (b - a))
        accepted.extend(
            np.array([[polygon[0], polygon[i], polygon[i + 1]]])
            for i in range(1, len(polygon) - 1)
        )
    return np.concatenate(accepted)


def render_mesh_depth(mesh, T_MC_D, K_D, shape, near_m=0.05):
    """Rasterize both sides at pixel centers with perspective-correct z.

    Visibility uses only geometry. No target depth, image dilation, footprint
    parameter or residual gating enters this renderer. Clip crossing triangles
    at the near plane before projection and keep the nearest positive hit.
    """
    if not is_rigid(T_MC_D) or not np.isfinite(near_m) or near_m <= 0:
        raise ValueError("require a rigid camera pose and positive near plane")
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("render shape must be positive")
    points = transform_points(mesh.vertices, np.linalg.inv(T_MC_D))
    triangles = points[mesh.triangles]
    if not np.isfinite(triangles).all():
        raise ValueError("mesh coordinates must be finite")
    triangles = _clip_near(triangles, near_m)
    depth = np.full(h * w, np.inf)
    for start in range(0, len(triangles), 128):
        batch = triangles[start : start + 128]
        xy = project_points(batch, K_D)
        low = np.maximum(np.ceil(xy.min(axis=1)), 0).astype(np.int64)
        high = np.minimum(np.floor(xy.max(axis=1)), [w - 1, h - 1]).astype(np.int64)
        sizes = np.maximum(high - low + 1, 0)
        counts = sizes.prod(axis=1)
        ids = np.repeat(np.arange(len(batch)), counts)
        if not len(ids):
            continue
        offsets = np.arange(len(ids)) - np.repeat(np.cumsum(counts) - counts, counts)
        u = low[ids, 0] + offsets % sizes[ids, 0]
        v = low[ids, 1] + offsets // sizes[ids, 0]
        a, b, c = xy[ids, 0], xy[ids, 1], xy[ids, 2]
        denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (
            c[:, 0] - b[:, 0]
        ) * (a[:, 1] - c[:, 1])
        nondegenerate = np.abs(denominator) > 1e-15
        x, y = np.zeros(len(ids)), np.zeros(len(ids))
        np.divide(
            (b[:, 1] - c[:, 1]) * (u - c[:, 0]) + (c[:, 0] - b[:, 0]) * (v - c[:, 1]),
            denominator,
            out=x,
            where=nondegenerate,
        )
        np.divide(
            (c[:, 1] - a[:, 1]) * (u - c[:, 0]) + (a[:, 0] - c[:, 0]) * (v - c[:, 1]),
            denominator,
            out=y,
            where=nondegenerate,
        )
        z = 1 - x - y
        valid = nondegenerate & (x >= -1e-10) & (y >= -1e-10) & (z >= -1e-10)
        inverse_depth = (
            x / batch[ids, 0, 2] + y / batch[ids, 1, 2] + z / batch[ids, 2, 2]
        )
        valid &= inverse_depth > 0
        np.minimum.at(depth, v[valid] * w + u[valid], 1 / inverse_depth[valid])
    depth[~np.isfinite(depth)] = np.nan
    return depth.reshape(shape)


def save_mesh(path: Path, mesh: TriangleMesh, colors: np.ndarray):
    """Write binary PLY in the same map gauge, without display reduction."""
    vertices = np.empty(
        len(mesh.vertices),
        dtype=[
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = mesh.vertices[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, axis]
    faces = np.empty(
        len(mesh.triangles), dtype=[("count", "u1"), ("indices", "<i4", (3,))]
    )
    faces["count"], faces["indices"] = 3, mesh.triangles
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment units meters\n"
        f"element vertex {len(vertices)}\nproperty double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)
        faces.tofile(stream)
