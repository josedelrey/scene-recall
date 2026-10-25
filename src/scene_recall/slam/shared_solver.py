"""Joint raw RGB-D IRLS with exact, matrix-free landmark elimination.

Every solve relinearizes the supplied observations. Elimination is temporary,
so no frozen marginal prior survives a historical correction or identity edit.
Memory scales with observations, cameras and landmarks, not camera pairs.
"""

from dataclasses import replace

import numpy as np

from scene_recall.geometry.transforms import is_rigid
from scene_recall.multiview.audit import normalized_spectrum
from scene_recall.multiview.batch_ba import (
    back_substitute,
    compact_blocks,
    compact_linearize,
)
from scene_recall.multiview.fixed_landmarks import direct_cost
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.multiview.pose_only import retract


class SchurOperator:
    """Implicit B - E C^-1 E.T with exact fixed-camera conditioning."""

    def __init__(self, blocks, views, inverse, fixed, damping):
        Hpp, gp, Hll, gl, cross = blocks
        self.views, self.inverse, self.cross = views, inverse, cross
        self.fixed = np.asarray(fixed, dtype=int)
        self.free = np.setdiff1d(np.arange(len(Hpp)), self.fixed)
        self.B = Hpp.copy()
        self.B[:, np.arange(6), np.arange(6)] += damping * np.maximum(
            np.diagonal(Hpp, axis1=1, axis2=2), 1e-15
        )
        C = Hll.copy()
        C[:, np.arange(3), np.arange(3)] += damping * np.maximum(
            np.diagonal(Hll, axis1=1, axis2=2), 1e-15
        )
        self.Ci = np.linalg.inv(C)
        self.b = np.einsum("nij,nj->ni", self.Ci, gl)
        rhs = -gp.copy()
        np.add.at(rhs, views, np.einsum("nij,nj->ni", cross, self.b[inverse]))
        self.rhs = rhs[self.free].ravel()
        diagonal = self.B.copy()
        np.add.at(
            diagonal,
            views,
            -cross @ self.Ci[inverse] @ cross.transpose(0, 2, 1),
        )
        self.diagonal = diagonal[self.free]

    def multiply(self, x):
        dp = np.zeros((len(self.B), 6))
        dp[self.free] = x.reshape(-1, 6)
        accumulated = np.zeros_like(self.b)
        np.add.at(
            accumulated,
            self.inverse,
            np.einsum("nji,nj->ni", self.cross, dp[self.views]),
        )
        result = np.einsum("nij,nj->ni", self.B, dp)
        np.add.at(
            result,
            self.views,
            -np.einsum(
                "nij,nj->ni",
                self.cross,
                np.einsum("nij,nj->ni", self.Ci, accumulated)[self.inverse],
            ),
        )
        return result[self.free].ravel()

    def solve(self, tolerance=1e-8, max_iterations=1000):
        """Checked block-Jacobi PCG with a true residual convergence test."""
        if not len(self.rhs):
            return np.zeros((len(self.B), 6)), {
                "iterations": 0,
                "relative_residual": 0.0,
            }
        preconditioner = np.linalg.inv(self.diagonal)

        def precondition(x):
            return np.einsum("nij,nj->ni", preconditioner, x.reshape(-1, 6)).ravel()

        x = np.zeros_like(self.rhs)
        r = self.rhs.copy()
        threshold = tolerance * max(np.linalg.norm(self.rhs), 1e-12)
        z = precondition(r)
        p, rz = z.copy(), float(r @ z)
        for iteration in range(max_iterations + 1):
            if np.linalg.norm(r) <= threshold:
                actual = self.rhs - self.multiply(x)
                if np.linalg.norm(actual) <= 2 * threshold:
                    dp = np.zeros((len(self.B), 6))
                    dp[self.free] = x.reshape(-1, 6)
                    return dp, {
                        "iterations": iteration,
                        "relative_residual": float(
                            np.linalg.norm(actual)
                            / max(np.linalg.norm(self.rhs), 1e-12)
                        ),
                    }
                r = actual
                z = precondition(r)
                p, rz = z.copy(), float(r @ z)
            Ap = self.multiply(p)
            denominator = float(p @ Ap)
            if not np.isfinite(denominator) or denominator <= 0 or rz <= 0:
                break
            alpha = rz / denominator
            x += alpha * p
            r -= alpha * Ap
            z = precondition(r)
            updated = float(r @ z)
            p = z + updated / rz * p
            rz = updated
        raise np.linalg.LinAlgError("shared Schur PCG residual tolerance not reached")


def solve_shared(
    poses, factors, K, fixed, config, *, rank_check_cameras=24, require_full_rank=False
):
    """Refine every supplied state together, rolling back on numerical failure.

    Small systems receive an undamped global rank diagnostic. Larger systems
    inspect conditional blocks and explicitly lack a global rank certificate.
    The optional strict policy rejects a weak camera. Otherwise LM can update a
    partially observable system, selecting an initialization-dependent solution.
    Damping is algorithmic and is never retained as measurement information.
    """
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or any(not is_rigid(p) for p in poses)
    ):
        raise ValueError("shared cameras must be rigid poses")
    if len(set(fixed)) != len(fixed) or any(
        isinstance(i, bool)
        or not isinstance(i, (int, np.integer))
        or i < 0
        or i >= len(poses)
        for i in fixed
    ):
        raise ValueError("fixed cameras must be unique valid indices")
    if (factors.view >= len(poses)).any():
        raise ValueError("observation camera is outside the shared state")
    ids, inverse, seeds = landmark_layout(factors)
    initial, poses, points = poses.copy(), poses.copy(), seeds.copy()
    accepted, linear_reports = [], []
    damping, first, reason = 0.001, None, "iteration limit"
    rank = {"scope": "global" if len(poses) <= rank_check_cameras else "blocks"}

    def rejected(message):
        return initial.copy(), seeds.copy(), {"status": "rejected", "reason": message}

    for iteration in range(config.max_iterations):
        r, JP, JL = compact_linearize(
            poses, points, factors, K, inverse, config.depth_model
        )
        cost, weights = direct_cost(r, factors, config)
        if not np.isfinite(cost):
            return rejected("invalid projections")
        first = cost if first is None else first
        blocks = compact_blocks(
            r, JP, JL, weights, inverse, factors.view, len(ids), len(poses)
        )
        try:
            point_values = np.linalg.eigvalsh(blocks[2])
            if (
                point_values[:, 0] <= np.maximum(1e-12, point_values[:, -1] * 1e-12)
            ).any():
                return rejected("unobservable landmark block")
            undamped = SchurOperator(blocks, factors.view, inverse, fixed, 0)
            if len(undamped.free):
                diagonal = undamped.diagonal
                scale = np.sqrt(
                    np.maximum(np.diagonal(diagonal, axis1=1, axis2=2), 1e-15)
                )
                values = np.linalg.eigvalsh(
                    diagonal / scale[:, :, None] / scale[:, None, :]
                )
                weak = values[:, 0] <= np.maximum(
                    1e-10, values[:, -1] / config.max_condition
                )
                rank["weak_camera_blocks"] = int(weak.sum())
                if require_full_rank and weak.any():
                    return rejected("unobservable conditional camera block")
                if len(poses) <= rank_check_cameras:
                    identity = np.eye(len(undamped.rhs))
                    S = np.column_stack([undamped.multiply(x) for x in identity])
                    rank.update(normalized_spectrum((S + S.T) / 2))
                    if require_full_rank and (
                        rank["condition"] is None
                        or rank["condition"] > config.max_condition
                    ):
                        return rejected("unobservable shared camera system")
            if (
                max(np.max(np.abs(blocks[3])), np.max(np.abs(undamped.rhs), initial=0))
                < 1e-8
            ):
                reason = "stationary gradient"
                break
            found = False
            for _ in range(10):
                operator = SchurOperator(blocks, factors.view, inverse, fixed, damping)
                dp, linear = operator.solve()
                dl = back_substitute(
                    dp, blocks, inverse, factors.view, operator.Ci, operator.b
                )
                trial = np.array(
                    [retract(p, d) for p, d in zip(poses, dp, strict=True)]
                )
                trial[list(fixed)] = initial[list(fixed)]
                movement = np.linalg.norm(
                    trial[:, :3, 3] - initial[:, :3, 3], axis=1
                ).max()
                rotations = trial[:, :3, :3] @ initial[:, :3, :3].transpose(0, 2, 1)
                angle = np.degrees(
                    np.arccos(
                        np.clip((np.trace(rotations, axis1=1, axis2=2) - 1) / 2, -1, 1)
                    )
                ).max()
                if (
                    movement > config.max_translation_m
                    or angle > config.max_rotation_deg
                    or np.linalg.norm(points + dl - seeds, axis=1).max()
                    > config.max_point_displacement_m
                ):
                    damping *= 10
                    continue
                current = replace(factors, points=(points + dl)[inverse])
                residual, _, _ = compact_linearize(
                    trial, points + dl, current, K, inverse, config.depth_model
                )
                candidate, _ = direct_cost(residual, factors, config)
                if np.isfinite(candidate) and candidate < cost:
                    poses, points = trial, points + dl
                    accepted.append(candidate)
                    linear_reports.append(linear)
                    damping = max(damping / 3, 1e-9)
                    found = True
                    break
                damping *= 10
            if not found:
                reason = "no acceptable step"
                break
            if (
                np.linalg.norm(dp) + np.linalg.norm(dl) < 1e-6
                or (cost - candidate) / max(1, cost) < 1e-8
            ):
                reason = "small step or relative decrease"
                break
        except np.linalg.LinAlgError as error:
            return rejected(str(error))
    return (
        poses,
        points,
        {
            "status": "refined" if accepted else "unchanged",
            "reason": reason,
            "iterations": iteration + 1,
            "initial_cost": first,
            "final_cost": accepted[-1] if accepted else first,
            "accepted_costs": accepted,
            "linear_solves": linear_reports,
            "rank_check": rank,
            "max_camera_change_m": float(
                np.linalg.norm(poses[:, :3, 3] - initial[:, :3, 3], axis=1).max()
            ),
            "max_point_change_m": float(np.linalg.norm(points - seeds, axis=1).max()),
        },
    )
