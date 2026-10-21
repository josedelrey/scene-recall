"""Rank-aware square-root elimination for bounded RGB-D graphs.

Only incident raw factors and the old prior are consumed. Orthogonal projection
eliminates observable directions without damping unobservable state directions.
The remaining factor preserves camera, point and cross correlations.
"""

from dataclasses import replace

import numpy as np

from scene_recall.multiview.batch_ba import compact_linearize
from scene_recall.multiview.fixed_landmarks import direct_cost
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.slam.graph import StatePrior


def eliminate_columns(J, residual, keep, eliminate):
    """Project a linear least-squares factor and compress its nonconstant rows.

    Column equilibration makes numerical rank independent of state units.
    Discarded residual-only directions contribute a constant to the objective.
    No normal equations or positive information floor enter the calculation.
    """
    retained = J[:, keep].copy()
    retained_scale = np.maximum(np.linalg.norm(retained, axis=0), 1e-15)
    residual = residual.copy()
    eliminated_rank = 0
    if len(eliminate):
        old = J[:, eliminate]
        scale = np.maximum(np.linalg.norm(old, axis=0), 1e-15)
        U, s, _ = np.linalg.svd(old / scale, full_matrices=False)
        eliminated_rank = int(np.sum(s > max(1e-12, s[0] * 1e-8))) if len(s) else 0
        U = U[:, :eliminated_rank]
        retained -= U @ (U.T @ retained)
        residual -= U @ (U.T @ residual)
    if not len(keep):
        return np.empty((0, 0)), np.empty(0), eliminated_rank
    scale = retained_scale
    U, s, V = np.linalg.svd(retained / scale, full_matrices=False)
    rank = int(np.sum(s > max(1e-12, s[0] * 1e-8))) if len(s) else 0
    return s[:rank, None] * V[:rank] * scale, U[:, :rank].T @ residual, eliminated_rank


def marginalize(
    poses, factors, K, camera_ids, root_id, prior, config, *, cameras=(), tracks=()
):
    """Retire cameras or points, consuming each incident measurement exactly once.

    Orphan points are retired with their final camera. A point can be retired
    while its observing cameras remain active. That operation creates camera
    correlations instead of discarding its multiview information.
    """
    ids, inverse, points = (
        landmark_layout(factors)
        if factors is not None
        else (np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty((0, 3)))
    )
    camera_mask = np.isin(camera_ids, cameras)
    consumed = (
        camera_mask[factors.view] | np.isin(factors.track_id, tracks)
        if factors is not None
        else np.empty(0, dtype=bool)
    )
    live = np.unique(factors.track_id[~consumed]) if factors is not None else ids
    retired = np.setdiff1d(ids, live)
    if not consumed.any() and prior is None:
        raise ValueError("retirement has no incident information")
    dimension = 6 * len(poses) + 3 * len(points)
    J, residual = np.zeros((0, dimension)), np.empty(0)
    if consumed.any():
        J, residual = incident_linearization(
            poses, points, factors, K, inverse, config, consumed
        )
    if prior is not None:
        pr, PJ = prior.linearize(camera_ids, poses, ids, points)
        J, residual = np.vstack((J, PJ)), np.r_[residual, pr]
    return finish_marginalization(
        J,
        residual,
        poses,
        points,
        ids,
        camera_ids,
        camera_mask,
        root_id,
        cameras,
        live,
        retired,
        consumed,
    )


def incident_linearization(poses, points, factors, K, inverse, config, consumed):
    """Build only consumed raw residual rows, including frozen IRLS weights."""
    packet = replace(
        factors,
        **{
            name: getattr(factors, name)[consumed]
            for name in factors.__dataclass_fields__
        },
    )
    r, JP, JL = compact_linearize(
        poses, points, packet, K, inverse[consumed], config.depth_model
    )
    cost, weights = direct_cost(r, packet, config)
    if not np.isfinite(cost):
        raise ValueError("invalid projections at marginalization")
    dimension = 6 * len(poses) + 3 * len(points)
    J = np.zeros((len(packet.view), 2, 3, dimension))
    for i, (camera, point) in enumerate(
        zip(packet.view, inverse[consumed], strict=True)
    ):
        J[i, :, :, 6 * camera : 6 * camera + 6] = JP[i]
        column = 6 * len(poses) + 3 * point
        J[i, :, :, column : column + 3] = JL[i]
    weight = np.sqrt(weights[:, :, None])
    J, residual = (J * weight[..., None]).reshape(-1, dimension), (r * weight).ravel()
    return J, residual


def finish_marginalization(
    J,
    residual,
    poses,
    points,
    ids,
    camera_ids,
    camera_mask,
    root_id,
    cameras,
    live,
    retired,
    consumed,
):
    """Project retired columns and record the surviving state chart."""
    keep = np.array(
        [
            j
            for i, key in enumerate(camera_ids)
            if key not in cameras
            for j in range(6 * i, 6 * i + 6)
        ]
        + [
            6 * len(poses) + 3 * i + j
            for i, key in enumerate(ids)
            if key in live
            for j in range(3)
        ],
        dtype=int,
    )
    eliminate = np.array(
        [
            j
            for i, key in enumerate(camera_ids)
            if key in cameras and key != root_id
            for j in range(6 * i, 6 * i + 6)
        ]
        + [
            6 * len(poses) + 3 * i + j
            for i, key in enumerate(ids)
            if key in retired
            for j in range(3)
        ],
        dtype=int,
    )
    A, b, rank = eliminate_columns(J, residual, keep, eliminate)
    retained_cameras = tuple(k for k in camera_ids if k not in cameras)
    result = StatePrior(
        retained_cameras,
        poses[~camera_mask].copy(),
        tuple(live),
        points[np.isin(ids, live)].copy(),
        A,
        b,
    )
    return (
        result,
        ~consumed,
        retired,
        {
            "consumed_observations": int(consumed.sum()),
            "retired_landmarks": len(retired),
            "eliminated_dimension": len(eliminate),
            "eliminated_rank": rank,
            "prior_rank": len(b),
        },
    )


def prior_supports_camera(prior, key):
    """Check local six-direction pose support, with global rank checked by solve."""
    if prior is None or key not in prior.camera_ids:
        return False
    i = prior.camera_ids.index(key)
    block = prior.A[:, 6 * i : 6 * i + 6]
    scale = np.maximum(np.linalg.norm(block, axis=0), 1e-15)
    s = np.linalg.svd(block / scale, compute_uv=False)
    return len(s) >= 6 and s[-1] > max(1e-12, s[0] * 1e-8)
