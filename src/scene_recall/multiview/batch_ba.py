"""Offline batch RGB-D adjustment with persistent training-track identities.

Compact observation Jacobians avoid a dense observation by camera array. The
camera Schur system is dense and deliberately bounded to short offline segments.
No reference poses, odometry priors or marginalization enter the objective.
"""

from dataclasses import dataclass, replace

import numpy as np

from scene_recall.geometry.transforms import is_rigid
from scene_recall.multiview.audit import normalized_spectrum
from scene_recall.multiview.fixed_landmarks import direct_cost, direct_linearize
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.multiview.pose_only import projection_jacobian, retract, skew


@dataclass(frozen=True)
class BatchConfig:
    min_track_views: int = 3
    min_observations: int = 12
    min_grid_coverage: float = 0.15
    max_cameras: int = 200
    max_iterations: int = 40
    max_condition: float = 1e8
    max_translation_m: float = 0.5
    max_rotation_deg: float = 30.0
    max_point_displacement_m: float = 0.5
    loss: str = "cauchy"
    loss_scale: float = 2.5
    depth_model: str = "native"

    def __post_init__(self):
        for name, lower in (
            ("min_track_views", 2),
            ("min_observations", 3),
            ("max_cameras", 2),
            ("max_iterations", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        for name in (
            "max_condition",
            "max_translation_m",
            "max_rotation_deg",
            "max_point_displacement_m",
            "loss_scale",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.min_grid_coverage)
            or not 0 <= self.min_grid_coverage <= 1
        ):
            raise ValueError("min_grid_coverage must lie between zero and one")
        if self.loss not in ("cauchy", "linear"):
            raise ValueError("unsupported batch loss")
        if self.depth_model not in ("native", "axial"):
            raise ValueError("unsupported batch depth model")


def compact_linearize(poses, points, factors, K, inverse, depth_model="native"):
    """Return residuals and one 6-column camera block per observation.

    Axial is a measurement ablation using the marginal native z variance. Native
    exactly reproduces the existing RGB 2D / native depth 3D joint objective.
    """
    current = replace(factors, points=points[inverse])
    residual, _ = direct_linearize(poses, current, K, (), jacobian=False)
    if residual is None:
        return None, None, None
    R = poses[factors.view, :3, :3].transpose(0, 2, 1)
    q = np.einsum("nij,nj->ni", R, current.points - poses[factors.view, :3, 3])
    D = np.concatenate((-R, R @ skew(current.points)), axis=2)
    rgb = projection_jacobian(q, K)
    rgb[:, 2] = 0
    valid = np.isfinite(factors.native_points).all(axis=1)
    transform = np.linalg.inv(factors.T_RcD)
    W = factors.whitening_D.copy()
    if depth_model == "axial":
        L = np.linalg.inv(W[valid])
        sigma = np.linalg.norm(L[:, 2], axis=1)
        W[:] = 0
        W[valid, 2, 2] = 1 / sigma
        qD = np.einsum("nij,nj->ni", transform[:, :3, :3], q) + transform[:, :3, 3]
        residual[:, 1] = 0
        residual[valid, 1, 2] = (qD[valid, 2] - factors.native_points[valid, 2]) / sigma
    elif depth_model != "native":
        raise ValueError("unsupported batch depth model")
    W[~valid] = 0
    native = W @ transform[:, :3, :3]
    JP = np.stack((rgb @ D, native @ D), axis=1)
    JL = np.stack((rgb @ R, native @ R), axis=1)
    return residual, JP, JL


def compact_blocks(residual, JP, JL, weights, inverse, views, count, cameras):
    """Accumulate camera diagonals, 3x3 points and observation-local cross blocks."""
    root = np.sqrt(weights[:, :, None])
    A, B, r = JP * root[..., None], JL * root[..., None], residual * root
    Hpp, gp = np.zeros((cameras, 6, 6)), np.zeros((cameras, 6))
    Hll, gl = np.zeros((count, 3, 3)), np.zeros((count, 3))
    np.add.at(Hpp, views, np.einsum("nbri,nbrj->nij", A, A))
    np.add.at(gp, views, np.einsum("nbri,nbr->ni", A, r))
    np.add.at(Hll, inverse, np.einsum("nbri,nbrj->nij", B, B))
    np.add.at(gl, inverse, np.einsum("nbri,nbr->ni", B, r))
    cross = np.einsum("nbri,nbrj->nij", A, B)
    return Hpp, gp, Hll, gl, cross


def compact_schur(blocks, views, groups, damping):
    """Eliminate points while fixing camera zero exactly, before or after damping."""
    Hpp, gp, Hll, gl, cross = blocks
    C = Hll.copy()
    C[:, np.arange(3), np.arange(3)] += damping * np.maximum(
        np.diagonal(C, axis1=1, axis2=2), 1e-15
    )
    Ci = np.linalg.inv(C)
    b = np.einsum("nij,nj->ni", Ci, gl)
    dimension = 6 * (len(Hpp) - 1)
    S, rhs = np.zeros((dimension, dimension)), -gp[1:].ravel().copy()
    for camera in range(1, len(Hpp)):
        columns = slice(6 * (camera - 1), 6 * camera)
        S[columns, columns] = Hpp[camera] + damping * np.diag(
            np.maximum(np.diag(Hpp[camera]), 1e-15)
        )
    for point, observations in enumerate(groups):
        observations = observations[views[observations] > 0]
        columns = (6 * (views[observations, None] - 1) + np.arange(6)).ravel()
        X = cross[observations].reshape(-1, 3)
        S[np.ix_(columns, columns)] -= X @ Ci[point] @ X.T
        rhs[columns] += X @ b[point]
    return (S + S.T) / 2, rhs, Ci, b


def back_substitute(dp, blocks, inverse, views, Ci, b):
    """Recover point steps from the same damped system as the camera step."""
    contribution = np.einsum("nij,nj->ni", blocks[4].transpose(0, 2, 1), dp[views])
    accumulated = np.zeros_like(b)
    np.add.at(accumulated, inverse, contribution)
    return -b - np.einsum("nij,nj->ni", Ci, accumulated)


def solve_batch(initial, factors, K, config=None):
    """Fit a whole candidate segment or abstain on undamped rank failure.

    Damping never supplies missing observability. Any later rank failure or
    invalid starting geometry restores the complete input, including its points.
    Bounds measure actual camera-center and orientation changes in the root gauge.
    """
    config = config or BatchConfig()
    initial = np.asarray(initial, dtype=float)
    if (
        initial.ndim != 3
        or initial.shape[1:] != (4, 4)
        or len(initial) < 2
        or any(not is_rigid(p) for p in initial)
    ):
        raise ValueError("batch initialization must contain at least two rigid poses")
    if (factors.view >= len(initial)).any():
        raise ValueError("factor references missing camera")
    ids, inverse, initial_points = landmark_layout(factors)
    poses, points = initial.copy(), initial_points.copy()
    groups = np.split(
        np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1]
    )
    diagnostic = {
        "camera_count": len(initial),
        "landmark_count": len(ids),
        "observation_count": len(factors.view),
        "depth_model": config.depth_model,
    }

    def rejected(reason, **extra):
        return (
            initial.copy(),
            initial_points.copy(),
            {**diagnostic, "status": "rejected", "reason": reason, **extra},
        )

    if len(initial) > config.max_cameras:
        return rejected("camera budget exceeded")
    damping, costs, trials, information = 0.001, [], [], {}
    initial_cost = None
    reason = "iteration limit"
    for iteration in range(config.max_iterations):
        r, JP, JL = compact_linearize(
            poses, points, factors, K, inverse, config.depth_model
        )
        cost, weights = direct_cost(r, factors, config)
        if not np.isfinite(cost):
            return rejected("invalid initial projections")
        if initial_cost is None:
            initial_cost = cost
        blocks = compact_blocks(
            r, JP, JL, weights, inverse, factors.view, len(ids), len(poses)
        )
        eig = np.linalg.eigvalsh(blocks[2])
        if (eig[:, 0] <= np.maximum(1e-15, eig[:, -1] * 1e-12)).any():
            return rejected("singular point information")
        S, _, _, _ = compact_schur(blocks, factors.view, groups, 0)
        information = normalized_spectrum(S)
        if (
            information["condition"] is None
            or information["condition"] > config.max_condition
        ):
            return rejected(
                "unobservable or ill-conditioned segment", information=information
            )
        if max(np.max(np.abs(blocks[1][1:])), np.max(np.abs(blocks[3]))) < 1e-8:
            reason = "stationary gradient"
            break
        accepted = False
        for _ in range(10):
            try:
                S, rhs, Ci, b = compact_schur(blocks, factors.view, groups, damping)
                dp = np.zeros((len(poses), 6))
                dp[1:] = np.linalg.solve(S, rhs).reshape(-1, 6)
                dl = back_substitute(dp, blocks, inverse, factors.view, Ci, b)
            except np.linalg.LinAlgError:
                trials.append("linear solve failure")
                damping *= 10
                continue
            trial = poses.copy()
            for i in range(1, len(poses)):
                trial[i] = retract(poses[i], dp[i])
            candidate_points = points + dl
            movement = np.linalg.norm(trial[:, :3, 3] - initial[:, :3, 3], axis=1).max()
            rotation = trial[:, :3, :3] @ initial[:, :3, :3].transpose(0, 2, 1)
            angle = np.degrees(
                np.arccos(
                    np.clip((np.trace(rotation, axis1=1, axis2=2) - 1) / 2, -1, 1)
                )
            ).max()
            shift = np.linalg.norm(candidate_points - initial_points, axis=1).max()
            if (
                movement > config.max_translation_m
                or angle > config.max_rotation_deg
                or shift > config.max_point_displacement_m
            ):
                trials.append("update bound")
                damping *= 10
                continue
            residual, _, _ = compact_linearize(
                trial, candidate_points, factors, K, inverse, config.depth_model
            )
            candidate, _ = direct_cost(residual, factors, config)
            if np.isfinite(candidate) and candidate < cost:
                poses, points = trial, candidate_points
                costs.append(candidate)
                damping = max(damping / 3, 1e-9)
                accepted = True
                break
            trials.append("cheirality" if residual is None else "no decrease")
            damping *= 10
        if not accepted:
            reason = "no acceptable step"
            break
        if (
            np.linalg.norm(dp) + np.linalg.norm(dl) < 1e-6
            or (cost - candidate) / max(1, cost) < 1e-8
        ):
            reason = "small step or relative decrease"
            break
    return (
        poses,
        points,
        {
            **diagnostic,
            "status": "refined" if costs else "unchanged",
            "reason": reason,
            "initial_cost": initial_cost,
            "final_cost": costs[-1] if costs else initial_cost,
            "iterations": iteration + 1,
            "accepted_costs": costs,
            "rejected_trials": trials,
            "information": information,
            "final_damping": damping,
            "landmark_displacement_max_m": float(
                np.linalg.norm(points - initial_points, axis=1).max()
            ),
            "camera_center_displacement_max_m": float(
                np.linalg.norm(poses[:, :3, 3] - initial[:, :3, 3], axis=1).max()
            ),
        },
    )
