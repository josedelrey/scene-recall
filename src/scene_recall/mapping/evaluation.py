"""Geometry diagnostics with explicit overlap and independent segment gauges."""

from itertools import product

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.mapping.voxel import SurfaceSamples, VoxelCloud
from scene_recall.odometry.evaluation import error_statistics
from scene_recall.odometry.pipeline import is_rigid


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
