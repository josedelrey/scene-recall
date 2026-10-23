"""Robust two-observation metric alignment with conditional sensor weights.

Each matched point contributes one 3D difference. Image coordinates and depth
define its covariance proxy, without adding duplicate reprojection factors.
Shared calibration, temporal depth bias and reused anchors remain correlated.
The normal matrix is a conditioning diagnostic, never a graph information prior.
"""

import cv2
import numpy as np

from scene_recall.geometry.camera import transform_points
from scene_recall.geometry.transforms import is_rigid


def point_covariances(points, K):
    """Propagate a 1 px lateral and (5 mm + 1% z) axial noise proxy."""
    z = points[:, 2]
    jacobian = np.zeros((len(points), 3, 3))
    jacobian[:, 0, 0] = z / K[0, 0]
    jacobian[:, 1, 1] = z / K[1, 1]
    jacobian[:, :, 2] = points / z[:, None]
    jacobian[:, :, 2] *= (0.005 + 0.01 * z)[:, None]
    return jacobian @ jacobian.transpose(0, 2, 1)


def _skew(points):
    result = np.zeros((len(points), 3, 3))
    x, y, z = points.T
    result[:, 0, 1], result[:, 0, 2] = -z, y
    result[:, 1, 0], result[:, 1, 2] = z, -x
    result[:, 2, 0], result[:, 2, 1] = -y, x
    return result


def _cost(norms):
    return float(np.where(norms <= 2.5, norms**2 / 2, 2.5 * (norms - 1.25)).sum())


def refine_metric_pose(source, target, K, initial):
    """Refine fixed fitting rows only, using bounded IRLS and six pose variables.

    Covariances are frozen within each line search and propagated again after an
    accepted update. The objective is an engineering generalized least squares
    surrogate, without a calibrated Gaussian likelihood claim.
    """
    source, target = np.asarray(source), np.asarray(target)
    K = np.asarray(K)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("refinement requires finite positive camera intrinsics")
    if not is_rigid(initial):
        raise ValueError("refinement requires a rigid initial pose")
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or target.shape != source.shape
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or (source[:, 2] <= 0).any()
        or (target[:, 2] <= 0).any()
    ):
        raise ValueError("metric refinement requires aligned positive finite points")
    pose = initial.copy()
    diagnostic = {"method": "two_observation_metric_irls", "pairs": len(source)}
    if len(source) < 6:
        return pose, {**diagnostic, "status": "insufficient_depth"}
    ca, cb = point_covariances(source, K), point_covariances(target, K)
    for iteration in range(8):
        predicted = transform_points(source, pose)
        covariance = pose[:3, :3] @ ca @ pose[:3, :3].T + cb
        whitening = np.linalg.inv(np.linalg.cholesky(covariance))
        residual = np.einsum("nij,nj->ni", whitening, predicted - target)
        norms = np.linalg.norm(residual, axis=1)
        jacobian = np.concatenate(
            (-_skew(predicted), np.broadcast_to(np.eye(3), (len(source), 3, 3))),
            axis=2,
        )
        jacobian = whitening @ jacobian
        weights = np.minimum(1.0, 2.5 / np.maximum(norms, 1e-12))
        normal = np.einsum("nji,njk,n->ik", jacobian, jacobian, weights)
        gradient = np.einsum("nji,nj,n->i", jacobian, residual, weights)
        # Normalize translation by scene depth before assessing pose rank.
        scale = np.diag([1.0] * 3 + [float(np.median(source[:, 2]))] * 3)
        spectrum = np.linalg.eigvalsh(scale @ normal @ scale)
        ratio = float(spectrum[0] / spectrum[-1])
        diagnostic.update(iterations=iteration + 1, conditional_rank_ratio=ratio)
        if ratio < 1e-7:
            return initial.copy(), {**diagnostic, "status": "degenerate"}
        delta = -np.linalg.solve(normal, gradient)
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            step = delta * fraction
            change = np.eye(4)
            change[:3, :3] = cv2.Rodrigues(step[:3])[0]
            change[:3, 3] = step[3:]
            candidate = change @ pose
            points = transform_points(source, candidate)
            errors = np.einsum("nij,nj->ni", whitening, points - target)
            if (points[:, 2] > 0.05).all() and _cost(
                np.linalg.norm(errors, axis=1)
            ) <= _cost(norms):
                pose, accepted = candidate, True
                break
        if not accepted or np.linalg.norm(delta) < 1e-7:
            break
    return pose, {**diagnostic, "status": "refined"}
