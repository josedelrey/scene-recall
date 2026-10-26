"""Reverse map-to-held-out-depth diagnostics with explicit unassessed voxels."""

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points


def surface_support(points, views, K, tolerance_m=0.05):
    """Count each map point once across fixed reference-camera depth views.

    A point behind measured depth may be occluded. A point in front contradicts
    observed free space under the assumed reference pose and sensor model.
    Support in one view and contradiction in another are reported together.
    Neither is independent physical surface truth. Unseen points stay explicit.
    """
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("surface support tolerance must be finite and positive")
    tested = np.zeros(len(points), dtype=bool)
    supported, free = tested.copy(), tested.copy()
    count = 0
    for camera, depth in views:
        local = transform_points(points, np.linalg.inv(camera))
        pixels = project_points(local, K)
        height, width = depth.shape
        valid = (
            np.isfinite(pixels).all(axis=1)
            & (local[:, 2] > 0)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= width - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= height - 1)
        )
        indices = np.flatnonzero(valid)
        uv = np.floor(pixels[valid] + 0.5).astype(int)
        measured = depth[uv[:, 1], uv[:, 0]]
        finite = np.isfinite(measured) & (measured > 0)
        indices = indices[finite]
        residual = local[indices, 2] - measured[finite]
        tested[indices] = True
        supported[indices[np.abs(residual) <= tolerance_m]] = True
        free[indices[residual < -tolerance_m]] = True
        count += 1
    counts = {
        "projected_to_valid_depth": tested,
        "supported_in_any_view": supported,
        "free_space_in_any_view": free,
        "supported_and_free_space": supported & free,
        "free_space_without_support": free & ~supported,
        "only_behind_measured_surfaces": tested & ~supported & ~free,
        "unassessed": ~tested,
    }
    return {
        "map_points": len(points),
        "target_views": count,
        "tolerance_m": tolerance_m,
        "scope": "unique map voxels projected into fixed held-out native depth at reference poses, all voxels retained in denominators, behind-surface observations may be occluded",
        **{
            name: {
                "count": int(mask.sum()),
                "fraction_of_all_voxels": float(mask.mean()) if len(mask) else None,
            }
            for name, mask in counts.items()
        },
    }
