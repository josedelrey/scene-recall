"""Read-only measurement and observability checks, without trajectory fitting."""

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.pose_only import projection_jacobian, skew


def epoch_displacement(points_R, K, pose_at_depth, pose_at_color):
    """Predict motion between two exposures for static depth-supported samples.

    Both poses map the virtual RGB optical frame to the same external frame.
    This is a timing sensitivity diagnostic. It does not establish that a depth
    sampled at a color keypoint belongs to the same surface at both exposures.
    External poses must never feed association, initialization or fitting.
    """
    points = np.asarray(points_R)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05)
    first = points[valid]
    second = transform_points(first, np.linalg.inv(pose_at_color) @ pose_at_depth)
    front = second[:, 2] > 0.05
    first, second = first[front], second[front]
    return (
        np.linalg.norm(project_points(second, K) - project_points(first, K), axis=1),
        second[:, 2] - first[:, 2],
        int(len(points) - len(first)),
    )


def normalized_spectrum(matrix):
    """Report diagonal-normalized information before any damping or prior."""
    matrix = (matrix + matrix.T) / 2
    diagonal = np.diag(matrix)
    if not len(diagonal) or (diagonal <= 1e-15).any():
        return {
            "dimension": len(diagonal),
            "condition": None,
            "minimum": None,
            "nullity": None,
            "observable_condition": None,
        }
    scaled = matrix / np.sqrt(diagonal[:, None] * diagonal[None, :])
    eig = np.linalg.eigvalsh(scaled)
    return {
        "dimension": len(diagonal),
        "minimum": float(eig[0]),
        "condition": float(eig[-1] / eig[0]) if eig[0] > 1e-12 else None,
        "nullity": int(np.sum(eig <= 1e-12)),
        "observable_condition": float(eig[-1] / eig[eig > 1e-12][0])
        if (eig > 1e-12).any()
        else None,
    }


def landmark_schur_information(poses, observations, K, config, *, depth_scale=1.0):
    """Inspect a proposed independent-observation RGB-D model without solving it.

    observations contains one (point_S, view_indices, pixels, depths) per track.
    Points are measured Sparse initializations, never external reference points.
    Fix the first camera and eliminate independent 3D point blocks analytically.
    None for depth_scale removes depth. A positive scale multiplies depth sigma.
    Return local linearized conditioning, without a covariance or accuracy claim.
    """
    if depth_scale is not None and (not np.isfinite(depth_scale) or depth_scale <= 0):
        raise ValueError("depth scale must be positive or None")
    dimension = 6 * (len(poses) - 1)
    information = np.zeros((dimension, dimension))
    point_conditions, rejected = [], 0
    for point, indices, pixels, depths in observations:
        rotation = poses[indices, :3, :3].transpose(0, 2, 1)
        q = np.einsum("nij,nj->ni", rotation, point - poses[indices, :3, 3])
        if not np.isfinite(q).all() or (q[:, 2] <= 0.05).any():
            rejected += 1
            continue
        valid = np.isfinite(depths)
        residual = np.column_stack(
            (project_points(q, K) - pixels, q[:, 2] - np.nan_to_num(depths))
        )
        inverse_sigma = np.ones_like(residual) / config.pixel_sigma
        inverse_sigma[:, 2] = 0
        if depth_scale is not None:
            inverse_sigma[valid, 2] = 1 / (
                depth_scale
                * (
                    config.depth_sigma_constant_m
                    + config.depth_sigma_quadratic * depths[valid] ** 2
                )
            )
        residual *= inverse_sigma
        weight = np.full(len(indices), 1 / len(indices))
        if config.loss == "cauchy":
            weight /= 1 + np.sum(residual**2, axis=1) / config.loss_scale**2
        A = (projection_jacobian(q, K) * inverse_sigma[:, :, None]) @ rotation
        A *= np.sqrt(weight[:, None, None])
        JL = A.reshape(-1, 3)
        JP = np.zeros((len(indices), 3, dimension))
        for row, index in enumerate(indices):
            if index:
                JP[row, :, 6 * (index - 1) : 6 * index] = -np.concatenate(
                    (A[row], -A[row] @ skew(point)), axis=1
                )
        JP = JP.reshape(-1, dimension)
        HLL = JL.T @ JL
        eig = np.linalg.eigvalsh(HLL)
        if eig[0] <= max(eig[-1] * 1e-12, 1e-15):
            rejected += 1
            continue
        point_conditions.append(float(eig[-1] / eig[0]))
        cross = JP.T @ JL
        information += JP.T @ JP - cross @ np.linalg.solve(HLL, cross.T)
    return {
        "pose_schur": normalized_spectrum(information),
        "point_conditions": point_conditions,
        "excluded_point_blocks": rejected,
        "point_blocks": len(point_conditions),
    }
