"""Bounded joint smoothing with coupled camera and landmark marginal priors.

Only factors touching a retiring camera and the previous prior are consumed.
Landmarks still observed by retained cameras stay variable, including historical
cross correlations. Remaining raw factors never enter the marginal prior twice.
The dense reference implementation deliberately uses a small landmark budget.
"""

from dataclasses import dataclass, replace

import cv2
import numpy as np

from scene_recall.geometry.transforms import is_rigid
from scene_recall.multiview.audit import normalized_spectrum
from scene_recall.multiview.batch_ba import (
    BatchConfig,
    compact_blocks,
    compact_linearize,
)
from scene_recall.multiview.fixed_landmarks import direct_cost
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.multiview.pose_only import retract, skew


def pose_chart(pose, origin):
    """Left retraction coordinates and their derivative at the current pose."""
    change = pose @ np.linalg.inv(origin)
    phi = cv2.Rodrigues(change[:3, :3])[0].ravel()
    angle = np.linalg.norm(phi)
    W = skew(phi)
    coefficient = (
        1 / 12 + angle**2 / 720
        if angle < 1e-4
        else (1 - angle / (2 * np.tan(angle / 2))) / angle**2
    )
    J = np.eye(6)
    J[:3, 3:] = -skew(change[:3, 3])
    J[3:, 3:] = np.eye(3) - W / 2 + coefficient * W @ W
    return np.r_[change[:3, 3], phi], J


@dataclass(frozen=True, eq=False)
class StatePrior:
    """Frozen square-root factor with explicit pose charts and point origins.

    Historical robust weights and Jacobians stay frozen. Pose chart transport is
    differentiated. This local Gaussian approximation does not imply calibrated
    sensor likelihoods or resolve hidden depth, calibration and clock biases.
    """

    camera_ids: tuple[int, ...]
    origins: np.ndarray
    track_ids: tuple[int, ...]
    point_origins: np.ndarray
    A: np.ndarray
    b: np.ndarray

    def linearize(self, camera_ids, poses, track_ids, points):
        cameras = {key: i for i, key in enumerate(camera_ids)}
        tracks = {key: i for i, key in enumerate(track_ids)}
        x = np.zeros(6 * len(self.camera_ids) + 3 * len(self.track_ids))
        J = np.zeros((len(x), 6 * len(camera_ids) + 3 * len(track_ids)))
        for i, (key, origin) in enumerate(
            zip(self.camera_ids, self.origins, strict=True)
        ):
            j = cameras[key]
            block = slice(6 * i, 6 * i + 6)
            x[block], J[block, 6 * j : 6 * j + 6] = pose_chart(poses[j], origin)
        for i, (key, origin) in enumerate(
            zip(self.track_ids, self.point_origins, strict=True)
        ):
            j = tracks[key]
            block = slice(
                6 * len(self.camera_ids) + 3 * i, 6 * len(self.camera_ids) + 3 * i + 3
            )
            column = 6 * len(camera_ids) + 3 * j
            x[block] = points[j] - origin
            J[block, column : column + 3] = np.eye(3)
        return self.A @ x + self.b, self.A @ J


def square_root_prior(camera_ids, poses, H, gradient, track_ids=(), points=None):
    """Factor a PSD quadratic with diagonal scaling and no invented information."""
    H = (H + H.T) / 2
    scale = np.sqrt(np.maximum(np.diag(H), 1e-15))
    values, vectors = np.linalg.eigh(H / scale[:, None] / scale[None, :])
    tolerance = max(1e-12, float(values[-1]) * 1e-10)
    if values[0] < -tolerance:
        raise ValueError("indefinite marginal information")
    keep = values > tolerance
    projected = vectors.T @ (gradient / scale)
    if np.linalg.norm(projected[~keep]) > 1e-6 * max(
        1, np.linalg.norm(gradient / scale)
    ):
        raise ValueError("marginal gradient outside information range")
    A = np.sqrt(values[keep, None]) * vectors[:, keep].T * scale[None, :]
    b = projected[keep] / np.sqrt(values[keep])
    return StatePrior(
        tuple(camera_ids),
        poses.copy(),
        tuple(track_ids),
        np.empty((0, 3)) if points is None else points.copy(),
        A,
        b,
    )


def linear_system(poses, points, track_ids, factors, K, config, camera_ids, prior):
    """Accumulate compact raw blocks and the complete coupled historical factor."""
    dimension = 6 * len(poses) + 3 * len(points)
    H, g = np.zeros((dimension, dimension)), np.zeros(dimension)
    cost = 0.0
    if factors is not None:
        positions = {key: i for i, key in enumerate(track_ids)}
        inverse = np.array([positions[int(key)] for key in factors.track_id])
        current = replace(factors, points=points[inverse])
        r, JP, JL = compact_linearize(
            poses, points, current, K, inverse, config.depth_model
        )
        cost, weights = direct_cost(r, factors, config)
        if not np.isfinite(cost):
            raise ValueError("invalid projections")
        Hpp, gp, Hll, gl, cross = compact_blocks(
            r, JP, JL, weights, inverse, factors.view, len(points), len(poses)
        )
        for camera in range(len(poses)):
            block = slice(6 * camera, 6 * camera + 6)
            H[block, block], g[block] = Hpp[camera], gp[camera]
        for point in range(len(points)):
            block = slice(6 * len(poses) + 3 * point, 6 * len(poses) + 3 * point + 3)
            H[block, block], g[block] = Hll[point], gl[point]
        for observation, (view, point) in enumerate(
            zip(factors.view, inverse, strict=True)
        ):
            a = slice(6 * view, 6 * view + 6)
            b = slice(6 * len(poses) + 3 * point, 6 * len(poses) + 3 * point + 3)
            H[a, b] += cross[observation]
            H[b, a] += cross[observation].T
    if prior is not None:
        residual, J = prior.linearize(camera_ids, poses, track_ids, points)
        H += J.T @ J
        g += J.T @ residual
        cost += float(residual @ residual) / 2
    return cost, (H + H.T) / 2, g


def free_columns(camera_ids, track_ids, root_id):
    return np.array(
        [
            j
            for i, key in enumerate(camera_ids)
            if key != root_id
            for j in range(6 * i, 6 * i + 6)
        ]
        + list(range(6 * len(camera_ids), 6 * len(camera_ids) + 3 * len(track_ids))),
        dtype=int,
    )


def solve_window(poses, factors, K, camera_ids, root_id, prior=None, config=None):
    """Bounded joint IRLS with atomic rollback and undamped observability checks."""
    config = config or BatchConfig(max_iterations=10)
    if (
        len(set(camera_ids)) != len(camera_ids)
        or len(camera_ids) != len(poses)
        or any(not is_rigid(p) for p in poses)
    ):
        raise ValueError("window cameras must be unique rigid poses")
    ids, inverse, seeds = landmark_layout(factors)
    initial, poses, points = poses.copy(), poses.copy(), seeds.copy()
    free = free_columns(camera_ids, ids, root_id)
    if len(poses) < 2:
        return poses, points, {"status": "unchanged", "reason": "root only"}
    damping, accepted, first, reason = 0.001, [], None, "iteration limit"

    def rejected(reason):
        return initial.copy(), seeds.copy(), {"status": "rejected", "reason": reason}

    for iteration in range(config.max_iterations):
        try:
            cost, H, g = linear_system(
                poses, points, ids, factors, K, config, camera_ids, prior
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            return rejected(str(error))
        first = cost if first is None else first
        S, gradient = H[np.ix_(free, free)], g[free]
        information = normalized_spectrum(S)
        if (
            information["condition"] is None
            or information["condition"] > config.max_condition
        ):
            return rejected("unobservable or ill-conditioned window")
        if np.max(np.abs(gradient)) < 1e-8:
            reason = "stationary gradient"
            break
        found = False
        for _ in range(10):
            try:
                delta = np.zeros(len(g))
                delta[free] = np.linalg.solve(
                    S + damping * np.diag(np.maximum(np.diag(S), 1e-15)), -gradient
                )
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            dp, dl = (
                delta[: 6 * len(poses)].reshape(-1, 6),
                delta[6 * len(poses) :].reshape(-1, 3),
            )
            trial = np.array(
                [
                    retract(p, d) if key != root_id else p
                    for key, p, d in zip(camera_ids, poses, dp, strict=True)
                ]
            )
            movement = np.linalg.norm(trial[:, :3, 3] - initial[:, :3, 3], axis=1).max()
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
            if prior is not None:
                pr, _ = prior.linearize(camera_ids, trial, ids, points + dl)
                candidate += float(pr @ pr) / 2
            if np.isfinite(candidate) and candidate < cost:
                poses, points = trial, points + dl
                accepted.append(candidate)
                damping = max(damping / 3, 1e-9)
                found = True
                break
            damping *= 10
        if not found:
            reason = "no acceptable step"
            break
        if np.linalg.norm(delta) < 1e-6 or (cost - candidate) / max(1, cost) < 1e-8:
            reason = "small step or relative decrease"
            break
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
            "information": information,
            "max_camera_change_m": float(
                np.linalg.norm(poses[:, :3, 3] - initial[:, :3, 3], axis=1).max()
            ),
            "max_point_change_m": float(np.linalg.norm(points - seeds, axis=1).max()),
        },
    )


def marginalize_oldest(poses, factors, K, camera_ids, root_id, prior, config):
    """Eliminate the oldest camera and orphan points from incident factors only.

    Shared landmarks remain in the prior. Every later raw observation remains
    outside it. The root is conditioned exactly before its first retirement.
    """
    ids, _, points = landmark_layout(factors)
    consumed = factors.view == 0
    packet = (
        replace(
            factors,
            **{
                name: getattr(factors, name)[consumed]
                for name in factors.__dataclass_fields__
            },
        )
        if consumed.any()
        else None
    )
    if packet is None and prior is None:
        raise ValueError("oldest camera has no marginalizable information")
    _, H, g = linear_system(poses, points, ids, packet, K, config, camera_ids, prior)
    live = np.unique(factors.track_id[~consumed])
    retired = np.setdiff1d(ids, live)
    keep = np.array(
        list(range(6, 6 * len(poses)))
        + [
            6 * len(poses) + 3 * i + j
            for i, key in enumerate(ids)
            if key in live
            for j in range(3)
        ],
        dtype=int,
    )
    eliminate = np.array(
        (list(range(6)) if camera_ids[0] != root_id else [])
        + [
            6 * len(poses) + 3 * i + j
            for i, key in enumerate(ids)
            if key in retired
            for j in range(3)
        ],
        dtype=int,
    )
    S, gradient = H[np.ix_(keep, keep)].copy(), g[keep].copy()
    if len(eliminate):
        old, cross = H[np.ix_(eliminate, eliminate)], H[np.ix_(eliminate, keep)]
        spectrum = normalized_spectrum(old)
        if (
            spectrum["condition"] is None
            or spectrum["condition"] > config.max_condition
        ):
            raise ValueError("unobservable retiring state")
        S -= cross.T @ np.linalg.solve(old, cross)
        gradient -= cross.T @ np.linalg.solve(old, g[eliminate])
    selected_points = np.array(
        [p for key, p in zip(ids, points, strict=True) if key in live]
    ).reshape(-1, 3)
    result = square_root_prior(
        camera_ids[1:], poses[1:], S, gradient, live, selected_points
    )
    return result, ~consumed, retired
