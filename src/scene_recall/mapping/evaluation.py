"""Geometry diagnostics with explicit overlap and independent segment gauges."""

from itertools import product

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.geometry.transforms import is_rigid
from scene_recall.mapping.voxel import SurfaceSamples, VoxelCloud
from scene_recall.odometry.evaluation import error_statistics


def depth_consistency(
    source: SurfaceSamples,
    target: SurfaceSamples,
    T_Ctarget_Csource: np.ndarray,
    K_D: np.ndarray,
    tolerance_m: float,
) -> dict:
    """Compare source geometry with target z-depth, without residual trimming.

    Nearest-pixel projection compares only valid filtered target depth. Positive
    residuals lie behind the observed surface and may include true occlusion.
    Negative residuals lie in observed free space and may include disocclusion.
    Both remain in statistics, so changing overlap cannot silently improve error.
    This is a cross-view consistency diagnostic, not ground-truth surface error.
    """
    if not is_rigid(T_Ctarget_Csource):
        raise ValueError("consistency requires a rigid relative pose")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("consistency tolerance must be finite and positive")
    points = transform_points(source.points, T_Ctarget_Csource)
    pixels = project_points(points, K_D)
    height, width = target.depth.shape
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= height - 1)
    )
    indices = np.floor(pixels[inside] + 0.5).astype(np.intp)
    measured = target.depth[indices[:, 1], indices[:, 0]]
    valid = np.isfinite(measured)
    residual = points[inside, 2][valid] - measured[valid]
    consistent = np.abs(residual) <= tolerance_m
    return {
        "source_samples": len(points),
        "projected_in_bounds": int(inside.sum()),
        "valid_target_samples": len(residual),
        "valid_target_fraction": len(residual) / len(points) if len(points) else None,
        "consistent_samples": int(consistent.sum()),
        "consistent_fraction_of_valid": float(consistent.mean())
        if len(residual)
        else None,
        "behind_surface_samples": int((residual > tolerance_m).sum()),
        "in_front_of_surface_samples": int((residual < -tolerance_m).sum()),
        "absolute_depth_error_m": error_statistics(np.abs(residual)),
    }


def summarize_consistency(rows: list[dict]) -> dict:
    """Pool counts and squared errors without retaining all pixel residuals."""
    count = sum(row["valid_target_samples"] for row in rows)
    samples = sum(row["source_samples"] for row in rows)
    consistent = sum(row["consistent_samples"] for row in rows)
    return {
        "directed_pairs": len(rows),
        "source_samples": samples,
        "valid_target_samples": count,
        "valid_target_fraction": count / samples if samples else None,
        "consistent_fraction_of_valid": consistent / count if count else None,
        "absolute_depth_rmse_m": (
            float(
                np.sqrt(
                    sum(
                        row["absolute_depth_error_m"]["rmse"] ** 2
                        * row["valid_target_samples"]
                        for row in rows
                        if row["valid_target_samples"]
                    )
                    / count
                )
            )
            if count
            else None
        ),
        "behind_surface_samples": sum(row["behind_surface_samples"] for row in rows),
        "in_front_of_surface_samples": sum(
            row["in_front_of_surface_samples"] for row in rows
        ),
    }


def cloud_statistics(cloud: VoxelCloud) -> dict:
    multiple = cloud.frame_support >= 2
    return {
        "voxel_count": len(cloud.points),
        "bbox_min_m": cloud.points.min(axis=0).tolist() if len(cloud.points) else None,
        "bbox_max_m": cloud.points.max(axis=0).tolist() if len(cloud.points) else None,
        "multiframe_voxels": int(multiple.sum()),
        "multiframe_fraction": float(multiple.mean()) if len(multiple) else None,
        "frame_support": error_statistics(cloud.frame_support),
        "frame_centroid_scatter_m": error_statistics(cloud.position_std_m[multiple]),
        "uncolored_voxels": int((cloud.color_support == 0).sum()),
    }


def _nearby_distances(
    source: VoxelCloud, target: VoxelCloud, size: float
) -> np.ndarray:
    """Exact nearest distance capped at one voxel width, using 27 cells.

    Every target within the open radius lies in the source cell or a neighbor.
    Unmatched samples retain the cap. No unbounded nearest-neighbor claim is made.
    """
    distances = np.full(len(source.points), size)
    if not len(target.points):
        return distances
    dtype = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
    target_keys = np.ascontiguousarray(target.keys, dtype="<i8").view(dtype).ravel()
    # VoxelCloud produced by VoxelMap is lexicographically ordered.
    order = np.argsort(target_keys)
    target_keys = target_keys[order]
    keys = np.floor(source.points / size).astype(np.int64)
    for offset in product((-1, 0, 1), repeat=3):
        query = np.ascontiguousarray(keys + offset, dtype="<i8").view(dtype).ravel()
        positions = np.searchsorted(target_keys, query)
        positions = np.minimum(positions, len(target_keys) - 1)
        matched = target_keys[positions] == query
        distance = np.linalg.norm(
            source.points[matched] - target.points[order[positions[matched]]], axis=1
        )
        distances[matched] = np.minimum(distances[matched], distance)
    return distances


def compare_clouds(estimated: VoxelCloud, reference: VoxelCloud, size: float) -> dict:
    """Symmetric capped surface distance on a paired, first-pose-aligned cohort."""
    result = {"distance_cap_m": size}
    for name, source, target in (
        ("estimated_to_reference", estimated, reference),
        ("reference_to_estimated", reference, estimated),
    ):
        distances = _nearby_distances(source, target, size)
        matched = distances < size
        result[name] = {
            "samples": len(distances),
            "within_cap_fraction": float(matched.mean()) if len(matched) else None,
            "capped_distance_m": error_statistics(distances),
            "matched_distance_m": error_statistics(distances[matched]),
        }
    return result


def render_depth(
    cloud: VoxelCloud,
    T_MC_D: np.ndarray,
    K_D: np.ndarray,
    shape: tuple[int, int],
    radius_m: float,
    near_m: float = 0.05,
) -> np.ndarray:
    """Z-buffer camera-facing metric discs centered on surface voxel centroids.

    Each disc has constant camera z and a fixed metric radius. This explicit
    footprint approximates a point surface without inventing mesh connectivity.
    It can bridge small holes and bias silhouettes, so report radius sensitivity.
    K_D and shape describe the evaluation lattice, including its pixel origin.
    No observed depth participates in rendering or visibility selection.
    """
    if not is_rigid(T_MC_D):
        raise ValueError("rendering requires a rigid camera-to-map pose")
    if any(not np.isfinite(x) or x <= 0 for x in (radius_m, near_m)):
        raise ValueError("render radius and near plane must be finite and positive")
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("render shape must be positive")
    points = transform_points(cloud.points, np.linalg.inv(T_MC_D))
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] >= near_m)]
    pixels = project_points(points, K_D)
    depth = np.full(height * width, np.inf)
    # Small chunks bound temporary rasterization storage even near the camera.
    for start in range(0, len(points), 256):
        z = points[start : start + 256, 2]
        xy = pixels[start : start + 256]
        radii = radius_m * np.array([K_D[0, 0], K_D[1, 1]]) / z[:, None]
        low = np.maximum(np.ceil(xy - radii), 0).astype(np.int64)
        high = np.minimum(np.floor(xy + radii), [width - 1, height - 1]).astype(
            np.int64
        )
        sizes = np.maximum(high - low + 1, 0)
        counts = sizes.prod(axis=1)
        index = np.repeat(np.arange(len(z)), counts)
        if not len(index):
            continue
        offset = np.arange(len(index)) - np.repeat(np.cumsum(counts) - counts, counts)
        u = low[index, 0] + offset % sizes[index, 0]
        v = low[index, 1] + offset // sizes[index, 0]
        distance2 = (
            ((u - xy[index, 0]) / K_D[0, 0]) ** 2
            + ((v - xy[index, 1]) / K_D[1, 1]) ** 2
        ) * z[index] ** 2
        inside = distance2 <= radius_m**2
        np.minimum.at(depth, v[inside] * width + u[inside], z[index[inside]])
    depth[~np.isfinite(depth)] = np.nan
    return depth.reshape(shape)


def heldout_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    population: np.ndarray,
    tolerances_m: tuple[float, ...] = (0.02, 0.05, 0.10),
) -> dict:
    """Separate coverage from untrimmed error on an explicitly fixed population.

    Positive residual means the visible map lies behind the measured surface.
    Negative residual means map geometry occupies measured free space. Either
    can reflect pose error, scene change or sensor error and neither is trimmed.
    Missing predictions fail explained fractions but have no invented depth error.
    """
    if observed.shape != predicted.shape or population.shape != observed.shape:
        raise ValueError("held-out arrays must share a pixel lattice")
    if population.dtype.kind != "b":
        raise ValueError("evaluation population must be a boolean mask")
    if np.any(population & (~np.isfinite(observed) | (observed <= 0))):
        raise ValueError("evaluation population requires valid observed depth")
    if not tolerances_m or any(not np.isfinite(t) or t <= 0 for t in tolerances_m):
        raise ValueError("depth tolerances must be finite and positive")
    valid = population & np.isfinite(predicted) & (predicted > 0)
    residual = predicted[valid] - observed[valid]
    count, covered = int(population.sum()), int(valid.sum())
    return {
        "target_samples": count,
        "predicted_samples": covered,
        "missing_samples": count - covered,
        "coverage": covered / count if count else None,
        "absolute_depth_error_m": error_statistics(np.abs(residual)),
        "signed_depth_error_m": error_statistics(residual),
        "thresholds": {
            str(t): {
                "agree_samples": int((np.abs(residual) <= t).sum()),
                "front_samples": int((residual < -t).sum()),
                "behind_samples": int((residual > t).sum()),
                "agreement_of_predicted": float((np.abs(residual) <= t).mean())
                if covered
                else None,
                "explained_fraction_of_target": int((np.abs(residual) <= t).sum())
                / count
                if count
                else None,
            }
            for t in tolerances_m
        },
    }
