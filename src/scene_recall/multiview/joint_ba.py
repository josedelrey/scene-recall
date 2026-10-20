"""Experimental bounded RGB-D bundle adjustment on frozen direct factors.

The API accepts no references, images or held-out measurements. Landmarks are
window-local shared variables with no point prior or cross-window persistence.
"""

from dataclasses import replace

import numpy as np

from scene_recall.geometry.transforms import is_rigid
from scene_recall.multiview.audit import normalized_spectrum
from scene_recall.multiview.fixed_landmarks import direct_cost, direct_linearize
from scene_recall.multiview.pose_only import (
    PoseOnlyConfig,
    projection_jacobian,
    retract,
)


def landmark_layout(factors):
    """Return stable IDs, observation-to-point indices and measured initial points."""
    ids, first, inverse = np.unique(
        factors.track_id, return_index=True, return_inverse=True
    )
    points = factors.points[first].copy()
    if not np.array_equal(points[inverse], factors.points):
        raise ValueError("shared landmark has inconsistent initial coordinates")
    return ids, inverse, points


def joint_linearize(poses, points, factors, K, free):
    """Reuse exact direct residuals and append analytic shared-point derivatives."""
    _, inverse, _ = landmark_layout(factors)
    current = replace(factors, points=points[inverse])
    residual, JP = direct_linearize(poses, current, K, free)
    if residual is None:
        return None, None, None
    rotation = poses[factors.view, :3, :3].transpose(0, 2, 1)
    qR = np.einsum("nij,nj->ni", rotation, current.points - poses[factors.view, :3, 3])
    JL = np.zeros((len(factors.view), 2, 3, 3))
    JL[:, 0] = projection_jacobian(qR, K) @ rotation
    JL[:, 0, 2] = 0
    valid = np.isfinite(factors.native_points).all(axis=1)
    JL[valid, 1] = (
        factors.whitening_D[valid]
        @ np.linalg.inv(factors.T_RcD[valid])[:, :3, :3]
        @ rotation[valid]
    )
    return residual, JP, JL


def normal_blocks(residual, JP, JL, weights, inverse, count):
    """Accumulate independent 3x3 point blocks without a dense joint Hessian."""
    root = np.sqrt(weights[:, :, None])
    A, B, r = JP * root[..., None], JL * root[..., None], residual * root
    dimension = JP.shape[-1]
    flat = A.reshape(-1, dimension)
    Hpp, gp = flat.T @ flat, flat.T @ r.ravel()
    Hll = np.zeros((count, 3, 3))
    Hpl = np.zeros((count, dimension, 3))
    gl = np.zeros((count, 3))
    np.add.at(Hll, inverse, np.einsum("nbri,nbrj->nij", B, B))
    np.add.at(Hpl, inverse, np.einsum("nbri,nbrj->nij", A, B))
    np.add.at(gl, inverse, np.einsum("nbri,nbr->ni", B, r))
    return Hpp, Hpl, Hll, gp, gl


def schur_step(blocks, damping):
    """Damp both variable families before elimination and back-substitution."""
    Hpp, Hpl, Hll, gp, gl = blocks
    C = Hll.copy()
    C[:, np.arange(3), np.arange(3)] += damping * np.maximum(
        np.diagonal(Hll, axis1=1, axis2=2), 1e-15
    )
    B = np.linalg.solve(C, Hpl.transpose(0, 2, 1))
    b = np.linalg.solve(C, gl[..., None])[..., 0]
    S = Hpp + damping * np.diag(np.maximum(np.diag(Hpp), 1e-15))
    S -= np.einsum("lpi,liq->pq", Hpl, B)
    rhs = -gp + np.einsum("lpi,li->p", Hpl, b)
    dp = np.linalg.solve((S + S.T) / 2, rhs)
    dl = -b - np.einsum("lip,p->li", B, dp)
    return dp, dl


def block_information(blocks):
    """Inspect undamped point blocks and camera Schur curvature."""
    Hpp, Hpl, Hll, _, _ = blocks
    eigenvalues = np.linalg.eigvalsh(Hll)
    bad = eigenvalues[:, 0] <= np.maximum(1e-15, eigenvalues[:, -1] * 1e-12)
    if bad.any():
        return {"condition": None, "excluded_point_blocks": int(bad.sum())}
    S = Hpp - np.einsum(
        "lpi,liq->pq", Hpl, np.linalg.solve(Hll, Hpl.transpose(0, 2, 1))
    )
    return {
        **normalized_spectrum(S),
        "excluded_point_blocks": 0,
        "maximum_point_condition": float(
            (eigenvalues[:, -1] / eigenvalues[:, 0]).max()
        ),
    }


def solve_joint(
    initial, factors, K, free, config=None, *, max_point_displacement_m=0.10
):
    """Bounded Cauchy IRLS/LM with exact 3x3 landmark Schur elimination.

    Every camera outside free is fixed exactly. Reject rank failures before
    damping. A failed later condition check rolls back the whole window.
    """
    config = config or PoseOnlyConfig()
    poses = np.array(initial, dtype=float, copy=True)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or any(not is_rigid(p) for p in poses)
    ):
        raise ValueError("joint initialization must contain rigid poses")
    free = tuple(free)
    if len(set(free)) != len(free) or any(i <= 0 or i >= len(poses) for i in free):
        raise ValueError("free poses must be unique non-root camera indices")
    if (factors.view >= len(poses)).any():
        raise ValueError("factor references missing camera")
    if not np.isfinite(max_point_displacement_m) or max_point_displacement_m <= 0:
        raise ValueError("point displacement limit must be finite and positive")
    ids, inverse, points = landmark_layout(factors)
    initial_points = points.copy()
    residual, _ = direct_linearize(poses, factors, K, free, jacobian=False)
    cost, _ = direct_cost(residual, factors, config)
    initial_cost = cost
    if not np.isfinite(cost):
        return (
            poses,
            points,
            {"status": "rejected", "reason": "invalid initial projections"},
        )
    if not free:
        return (
            poses,
            points,
            {"status": "unchanged", "initial_cost": cost, "final_cost": cost},
        )
    damping, accepted_costs, trials = 0.001, [], []
    reason, condition, information = "iteration limit", None, {}
    for iteration in range(config.max_iterations):
        residual, JP, JL = joint_linearize(poses, points, factors, K, free)
        _, weights = direct_cost(residual, factors, config)
        blocks = normal_blocks(residual, JP, JL, weights, inverse, len(ids))
        information = block_information(blocks)
        condition = information["condition"]
        if condition is None or condition > config.max_condition:
            return (
                np.array(initial, copy=True),
                initial_points,
                {
                    "status": "rejected",
                    "reason": "unobservable or ill-conditioned joint geometry",
                    "information": information,
                },
            )
        if max(np.max(np.abs(blocks[3])), np.max(np.abs(blocks[4]))) < 1e-8:
            reason = "stationary gradient"
            break
        accepted = False
        for _ in range(10):
            try:
                dp, dl = schur_step(blocks, damping)
            except np.linalg.LinAlgError:
                trials.append("linear solve failure")
                damping *= 10
                continue
            trial = poses.copy()
            for index, change in zip(free, dp.reshape(-1, 6), strict=True):
                trial[index] = retract(poses[index], change)
            candidate_points = points + dl
            correction = trial @ np.linalg.inv(initial)
            displacement = np.linalg.norm(correction[:, :3, 3], axis=1).max()
            angles = np.degrees(
                np.arccos(
                    np.clip(
                        (np.trace(correction[:, :3, :3], axis1=1, axis2=2) - 1) / 2,
                        -1,
                        1,
                    )
                )
            )
            point_shift = np.linalg.norm(
                candidate_points - initial_points, axis=1
            ).max()
            if (
                displacement > config.max_translation_m
                or angles.max() > config.max_rotation_deg
                or point_shift > max_point_displacement_m
            ):
                trials.append("update bound")
                damping *= 10
                continue
            current = replace(factors, points=candidate_points[inverse])
            r, _ = direct_linearize(trial, current, K, free, jacobian=False)
            candidate, _ = direct_cost(r, factors, config)
            if np.isfinite(candidate) and candidate < cost:
                poses, points, cost = trial, candidate_points, candidate
                accepted_costs.append(cost)
                damping = max(damping / 3, 1e-9)
                accepted = True
                break
            trials.append("cheirality" if r is None else "no decrease")
            damping *= 10
        if not accepted:
            reason = "no acceptable step"
            break
        if np.linalg.norm(dp) + np.linalg.norm(dl) < 1e-6:
            reason = "small step"
            break
    shifts = np.linalg.norm(points - initial_points, axis=1)
    return (
        poses,
        points,
        {
            "status": "refined" if accepted_costs else "unchanged",
            "reason": reason,
            "initial_cost": initial_cost,
            "final_cost": cost,
            "condition": condition,
            "information": information,
            "iterations": iteration + 1,
            "accepted_costs": accepted_costs,
            "rejected_trials": trials,
            "final_damping": damping,
            "landmark_count": len(ids),
            "landmark_displacement_rms_m": float(np.sqrt(np.mean(shifts**2))),
            "landmark_displacement_max_m": float(shifts.max()),
            "max_point_displacement_m": max_point_displacement_m,
        },
    )
