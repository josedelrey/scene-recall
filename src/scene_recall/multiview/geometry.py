"""Depth-anchored landmark construction and all-observation validation."""

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.model import FeatureKey, TrackConfig


def landmark_geometry(
    keys: tuple[FeatureKey, ...],
    frames: dict,
    K: np.ndarray,
    config: TrackConfig,
    cache: dict | None = None,
) -> dict | None:
    """Construct a robust depth median, then assess every member without pruning.

    No triangulation-only point is created. Missing-depth views contribute pixel
    constraints once another member anchors the point. A coordinate median limits
    single-sample influence, but inconsistent depth is still rejected explicitly.
    """
    samples, centers, depths, poses, pixels = [], [], [], [], []
    for frame_id, feature_id in keys:
        frame = frames[frame_id]
        point = frame.features.points_R[feature_id]
        if np.isfinite(point).all():
            samples.append(
                cache[frame_id][1][feature_id]
                if cache is not None
                else transform_points(point, frame.T_SR)
            )
        depths.append(point)
        centers.append(frame.T_SR[:3, 3])
        poses.append(
            cache[frame_id][0] if cache is not None else np.linalg.inv(frame.T_SR)
        )
        pixels.append(frame.features.pixels[feature_id])
    if not samples:
        return None
    point_S = np.median(samples, axis=0)
    poses = np.array(poses)
    projected_R = np.einsum("nij,j->ni", poses[:, :3, :3], point_S) + poses[:, :3, 3]
    errors = np.linalg.norm(project_points(projected_R, K) - pixels, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    depths = np.array(depths)
    valid = np.isfinite(depths).all(axis=1)
    depth_errors = np.full(len(keys), np.nan)
    depth_errors[valid] = np.linalg.norm(projected_R[valid] - depths[valid], axis=1)
    limits = config.depth_absolute_m + config.depth_relative * depths[valid, 2]
    score = max(
        float(np.max(errors) / config.reprojection_px),
        float(np.max(depth_errors[valid] / limits)),
    )
    centers = np.array(centers)
    directions = point_S - centers
    norms = np.linalg.norm(directions, axis=1)
    directions = directions / np.maximum(norms[:, None], 1e-12)
    # Compute extent in O(N^2) only when materializing final landmarks.
    return {
        "point_S": point_S,
        "depth_views": int(valid.sum()),
        "reprojection_errors_px": errors,
        "depth_errors_m": depth_errors,
        "score": score,
        "centers": centers,
        "directions": directions,
    }


def pair_geometry(first, second, queries, trains, K, config):
    """Vectorized symmetric endpoint gates using each available depth anchor.

    Both depths must agree when available. At least one depth is required.
    Each anchor must project in front of the other camera. Missing depth is
    never represented by zero or inferred from reference poses.
    """
    p = first.features.points_R[queries]
    q = second.features.points_R[trains]
    valid_p = np.isfinite(p).all(axis=1)
    valid_q = np.isfinite(q).all(axis=1)
    T_QP = np.linalg.inv(second.T_SR) @ first.T_SR
    pq = transform_points(p, T_QP)
    qp = transform_points(q, np.linalg.inv(T_QP))
    ep = np.linalg.norm(project_points(pq, K) - second.features.pixels[trains], axis=1)
    eq = np.linalg.norm(project_points(qp, K) - first.features.pixels[queries], axis=1)
    ep = np.where(valid_p, np.where(np.isfinite(ep), ep, np.inf), 0.0)
    eq = np.where(valid_q, np.where(np.isfinite(eq), eq, np.inf), 0.0)
    pixel_error = np.maximum(ep, eq)
    both = valid_p & valid_q
    depth_error = np.linalg.norm(pq - q, axis=1)
    # Use the tighter of the two anchor tolerances for symmetric acceptance.
    limit = config.depth_absolute_m + config.depth_relative * np.minimum(
        p[:, 2], q[:, 2]
    )
    depth_score = np.where(both, depth_error / limit, 0.0)
    score = np.maximum(pixel_error / config.reprojection_px, depth_score)
    score[~(valid_p | valid_q)] = np.inf
    return score, pixel_error, np.where(both, depth_error, np.nan), valid_p | valid_q
