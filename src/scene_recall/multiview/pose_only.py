"""Bounded offline pose-only refinement from fixed RGB-D anchor measurements.

Only camera poses are variables. A star of observations shares one fixed
camera-frame anchor sample per track. References and evaluation images are absent
from this API. This is an experimental solver, without landmark optimization.
"""

import hashlib
from dataclasses import dataclass, replace
from itertools import pairwise
from time import perf_counter

import cv2
import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.geometry.transforms import is_rigid
from scene_recall.multiview.diagnostics import grid_coverage


@dataclass(frozen=True)
class PoseOnlyConfig:
    window_size: int = 6
    window_step: int = 4
    max_tracks: int = 256
    min_factors_per_view: int = 20
    min_grid_coverage: float = 0.15
    pixel_sigma: float = 1.0
    depth_sigma_constant_m: float = 0.005
    depth_sigma_quadratic: float = 0.003
    loss_scale: float = 2.5
    loss: str = "cauchy"
    formulation: str = "anchored"
    max_iterations: int = 20
    max_condition: float = 1e8
    max_translation_m: float = 0.10
    max_rotation_deg: float = 5.0

    def __post_init__(self):
        for name, lower in (
            ("window_size", 3),
            ("window_step", 1),
            ("max_tracks", 1),
            ("min_factors_per_view", 3),
            ("max_iterations", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.window_step >= self.window_size:
            raise ValueError("windows must overlap")
        for name in (
            "pixel_sigma",
            "depth_sigma_constant_m",
            "depth_sigma_quadratic",
            "loss_scale",
            "max_condition",
            "max_translation_m",
            "max_rotation_deg",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.min_grid_coverage)
            or not 0 <= self.min_grid_coverage <= 1
        ):
            raise ValueError("min_grid_coverage must lie between zero and one")
        if self.loss not in ("cauchy", "linear"):
            raise ValueError("unsupported robust loss")
        if self.formulation not in ("anchored", "frozen"):
            raise ValueError("unsupported pose-only formulation")


def skew(points):
    points = np.asarray(points)
    result = np.zeros(points.shape[:-1] + (3, 3))
    result[..., 0, 1], result[..., 0, 2] = -points[..., 2], points[..., 1]
    result[..., 1, 0], result[..., 1, 2] = points[..., 2], -points[..., 0]
    result[..., 2, 0], result[..., 2, 1] = -points[..., 1], points[..., 0]
    return result


def retract(pose, delta):
    """Left SE(3) retraction with translation then rotation tangent coordinates."""
    update = np.eye(4)
    update[:3, :3] = cv2.Rodrigues(np.asarray(delta[3:], dtype=float))[0]
    update[:3, 3] = delta[:3]
    return update @ pose


def projection_jacobian(points, K):
    x, y, z = points.T
    J = np.zeros((len(points), 3, 3))
    J[:, 0, 0], J[:, 1, 1] = K[0, 0] / z, K[1, 1] / z
    J[:, 0, 2], J[:, 1, 2] = -K[0, 0] * x / z**2, -K[1, 1] * y / z**2
    J[:, 2, 2] = 1
    return J


@dataclass(frozen=True, eq=False)
class PoseFactors:
    anchor: np.ndarray
    target: np.ndarray
    points_anchor: np.ndarray
    pixels: np.ndarray
    depth: np.ndarray
    whitening: np.ndarray
    weights: np.ndarray
    frozen_points: np.ndarray
    track_ids: np.ndarray


def make_factors(result, frame_ids, K, shape, config):
    """Select supported tracks and freeze measurement uncertainty before fitting.

    Reassess support inside this window. The anchor is the depth-supported view
    nearest its temporal center. Round-robin RGB cells limit feature clustering.
    No original median landmark is used by the anchored formulation.
    """
    frames = {frame.frame_index: frame for frame in result.frames}
    positions = {frame_id: i for i, frame_id in enumerate(frame_ids)}
    candidates = []
    for track in result.landmarks:
        keys = tuple(k for k in track.observations if k[0] in positions)
        valid = [
            k for k in keys if np.isfinite(frames[k[0]].features.points_R[k[1]]).all()
        ]
        if len(keys) < 3 or len(valid) < 2:
            continue
        anchor = min(
            valid, key=lambda k: (abs(positions[k[0]] - (len(frame_ids) - 1) / 2), k)
        )
        pixel = frames[anchor[0]].features.pixels[anchor[1]]
        cell = tuple(
            np.minimum((pixel / [shape[1], shape[0]] * [8, 6]).astype(int), [7, 5])
        )
        candidates.append((cell, -len(keys), track.track_id, anchor, keys))
    cells = {}
    for item in sorted(candidates):
        cells.setdefault(item[0], []).append(item)
    chosen = []
    for depth in range(max(map(len, cells.values()), default=0)):
        for cell in sorted(cells):
            if depth < len(cells[cell]):
                chosen.append(cells[cell][depth])
                if len(chosen) == config.max_tracks:
                    break
        if len(chosen) == config.max_tracks:
            break
    anchors, targets, points, pixels, depths, weights, fixed, ids = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for _, _, track_id, anchor, keys in chosen:
        samples = [
            transform_points(frames[f].features.points_R[k], frames[f].T_SR)
            for f, k in keys
            if np.isfinite(frames[f].features.points_R[k]).all()
        ]
        frozen = np.median(samples, axis=0)
        for frame_id, feature_id in keys:
            if (frame_id, feature_id) == anchor and config.formulation == "anchored":
                continue
            anchors.append(positions[anchor[0]])
            targets.append(positions[frame_id])
            points.append(frames[anchor[0]].features.points_R[anchor[1]])
            pixels.append(frames[frame_id].features.pixels[feature_id])
            depths.append(frames[frame_id].features.points_R[feature_id, 2])
            weights.append(1 / (len(keys) - (config.formulation == "anchored")))
            fixed.append(frozen)
            ids.append(track_id)
    n = len(anchors)
    anchors, targets = np.array(anchors, dtype=int), np.array(targets, dtype=int)
    points = np.array(points).reshape(-1, 3)
    pixels, depths = np.array(pixels).reshape(-1, 2), np.array(depths)
    poses = np.array([frames[i].T_SR for i in frame_ids])
    if not n:
        return None, {"reason": "insufficient supported tracks", "tracks": 0}
    rotation = poses[targets, :3, :3].transpose(0, 2, 1) @ poses[anchors, :3, :3]
    world = (
        np.einsum("nij,nj->ni", poses[anchors, :3, :3], points) + poses[anchors, :3, 3]
    )
    q = np.einsum(
        "nij,nj->ni",
        poses[targets, :3, :3].transpose(0, 2, 1),
        world - poses[targets, :3, 3],
    )
    B = np.zeros((n, 3, 3))
    B[:, 0, 0], B[:, 1, 1] = points[:, 2] / K[0, 0], points[:, 2] / K[1, 1]
    B[:, :, 2] = points / points[:, 2:3]
    sigma_a = (
        config.depth_sigma_constant_m + config.depth_sigma_quadratic * points[:, 2] ** 2
    )
    sigma_t = (
        config.depth_sigma_constant_m
        + config.depth_sigma_quadratic * np.nan_to_num(depths) ** 2
    )
    variances = np.stack(
        [
            np.full(n, config.pixel_sigma**2),
            np.full(n, config.pixel_sigma**2),
            sigma_a**2,
        ],
        axis=1,
    )
    G = projection_jacobian(q, K) @ rotation @ B
    covariance = (G * variances[:, None, :]) @ G.transpose(0, 2, 1)
    covariance[:, 0, 0] += config.pixel_sigma**2
    covariance[:, 1, 1] += config.pixel_sigma**2
    covariance[:, 2, 2] += sigma_t**2
    missing = ~np.isfinite(depths)
    covariance[missing, 2, :], covariance[missing, :, 2] = 0, 0
    covariance[missing, 2, 2] = 1
    whitening = np.linalg.inv(np.linalg.cholesky(covariance))
    coverage, counts = [], []
    for i, frame_id in enumerate(frame_ids):
        # Count independent feature identities, including features serving as anchors.
        keys = {(f, k) for *_, members in chosen for f, k in members if f == frame_id}
        counts.append(len(keys))
        coverage.append(
            grid_coverage([frames[f].features.pixels[k] for f, k in keys], shape)
        )
    diagnostics = {
        "tracks": len(chosen),
        "factors": n,
        "counts_by_view": counts,
        "coverage_by_view": coverage,
    }
    if (
        min(counts) < config.min_factors_per_view
        or min(coverage) < config.min_grid_coverage
    ):
        return None, {**diagnostics, "reason": "insufficient view support or coverage"}
    factors = PoseFactors(
        anchors,
        targets,
        points,
        pixels,
        depths,
        whitening,
        np.array(weights),
        np.array(fixed),
        np.array(ids, dtype=int),
    )
    return factors, diagnostics


def linearize(poses, factors, K, config, *, jacobian=True):
    """Whitened residual blocks and analytic derivatives for left perturbations."""
    a, b = factors.anchor, factors.target
    if config.formulation == "anchored":
        world = (
            np.einsum("nij,nj->ni", poses[a, :3, :3], factors.points_anchor)
            + poses[a, :3, 3]
        )
    else:
        world = factors.frozen_points
    R = poses[b, :3, :3].transpose(0, 2, 1)
    q = np.einsum("nij,nj->ni", R, world - poses[b, :3, 3])
    if not np.isfinite(q).all() or (q[:, 2] <= 0.05).any():
        return None, None
    residual = np.column_stack(
        (project_points(q, K) - factors.pixels, q[:, 2] - np.nan_to_num(factors.depth))
    )
    missing = ~np.isfinite(factors.depth)
    residual[missing, 2] = 0
    residual = np.einsum("nij,nj->ni", factors.whitening, residual)
    if not jacobian:
        return residual, None
    A = projection_jacobian(q, K)
    A[missing, 2] = 0
    A = factors.whitening @ A @ R
    Ja = np.concatenate((A, -A @ skew(world)), axis=2)
    J = np.zeros((len(a), 3, 6 * (len(poses) - 1)))
    for i in range(1, len(poses)):
        selected = b == i
        J[selected, :, 6 * (i - 1) : 6 * i] -= Ja[selected]
        if config.formulation == "anchored":
            selected = a == i
            J[selected, :, 6 * (i - 1) : 6 * i] += Ja[selected]
    return residual, J


def robust_cost(residual, factors, config):
    if residual is None:
        return float("inf"), None
    s = np.sum(residual**2, axis=1)
    if config.loss == "linear":
        return float(0.5 * np.sum(factors.weights * s)), factors.weights
    c2 = config.loss_scale**2
    return float(
        0.5 * np.sum(factors.weights * c2 * np.log1p(s / c2))
    ), factors.weights / (1 + s / c2)


def solve_window(initial, factors, K, config):
    """Dense damped IRLS with a fixed root, fixed factors and guarded proposals."""
    poses = np.array(initial, dtype=float, copy=True)
    if any(not is_rigid(pose, atol=1e-6) for pose in poses):
        raise ValueError("pose-only initialization must be rigid")
    r, J = linearize(poses, factors, K, config)
    initial_cost, _ = robust_cost(r, factors, config)
    if not np.isfinite(initial_cost):
        return poses, {"status": "rejected", "reason": "invalid initial projections"}
    cost, damping, steps, condition = initial_cost, 1e-3, [], None
    for iteration in range(config.max_iterations):
        r, J = linearize(poses, factors, K, config)
        _, weights = robust_cost(r, factors, config)
        flat_J = (J * np.sqrt(weights[:, None, None])).reshape(-1, J.shape[-1])
        flat_r = (r * np.sqrt(weights[:, None])).ravel()
        H, gradient = flat_J.T @ flat_J, flat_J.T @ flat_r
        diagonal = np.maximum(np.diag(H), 1e-15)
        scaled = H / np.sqrt(diagonal[:, None] * diagonal[None, :])
        eigenvalues = np.linalg.eigvalsh(scaled)
        condition = (
            float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 1e-12 else None
        )
        if condition is None or condition > config.max_condition:
            return np.array(initial, copy=True), {
                "status": "rejected",
                "reason": "unobservable or ill-conditioned",
                "condition": condition,
                "initial_cost": initial_cost,
            }
        accepted = False
        for _ in range(10):
            delta = np.linalg.solve(H + damping * np.diag(diagonal), -gradient).reshape(
                -1, 6
            )
            trial = poses.copy()
            for i, change in enumerate(delta, 1):
                trial[i] = retract(poses[i], change)
            correction = trial @ np.linalg.inv(initial)
            translations = np.linalg.norm(correction[:, :3, 3], axis=1)
            rotations = np.degrees(
                np.arccos(
                    np.clip(
                        (np.trace(correction[:, :3, :3], axis1=1, axis2=2) - 1) / 2,
                        -1,
                        1,
                    )
                )
            )
            admissible = (
                translations.max() <= config.max_translation_m
                and rotations.max() <= config.max_rotation_deg
            )
            trial_r, _ = linearize(trial, factors, K, config, jacobian=False)
            trial_cost, _ = robust_cost(trial_r, factors, config)
            if admissible and trial_cost < cost:
                poses, cost = trial, trial_cost
                damping = max(damping / 3, 1e-9)
                accepted = True
                steps.append(cost)
                break
            damping *= 10
        if not accepted or np.linalg.norm(delta) < 1e-6:
            break
    return poses, {
        "status": "refined" if steps else "unchanged",
        "initial_cost": initial_cost,
        "final_cost": cost,
        "accepted_costs": steps,
        "iterations": iteration + 1,
        "condition": condition,
        "factors": len(factors.anchor),
    }


def refine_training_poses(result, calibration, config=None):
    """Chronological overlapping windows, preserving every segment's gauge."""
    config = config or PoseOnlyConfig()
    indices = [f.frame_index for f in result.frames]
    if indices != sorted(set(indices)) or any(
        a.segment_id > b.segment_id for a, b in pairwise(result.frames)
    ):
        raise ValueError(
            "feature frames must have unique increasing indices and segments"
        )
    started = perf_counter()
    frame_map = {f.frame_index: f for f in result.frames}
    poses = {f.frame_index: f.T_SR.copy() for f in result.frames}
    diagnostics = []
    for segment in sorted({f.segment_id for f in result.frames}):
        ids = [f.frame_index for f in result.frames if f.segment_id == segment]
        for start in range(0, len(ids) - 2, config.window_step):
            window = ids[start : start + config.window_size]
            factors, support = make_factors(
                result, window, calibration.K_R, calibration.rgb_shape, config
            )
            diagnostic = {"frame_indices": window, "segment_id": segment, **support}
            if factors is None:
                diagnostics.append({**diagnostic, "status": "skipped"})
                continue
            digest = hashlib.sha256()
            for array in (
                factors.anchor,
                factors.target,
                factors.points_anchor,
                factors.pixels,
                factors.depth,
                factors.whitening,
                factors.weights,
                factors.track_ids,
            ):
                digest.update(array.tobytes())
            diagnostic.update(
                factors_sha256=digest.hexdigest(),
                selected_track_ids=sorted(set(map(int, factors.track_ids))),
                missing_depth_factors=int(np.isnan(factors.depth).sum()),
            )
            # Express this window near its root for numerical conditioning.
            origin = poses[window[0]].copy()
            initial = np.array([np.linalg.inv(origin) @ poses[i] for i in window])
            frozen = transform_points(factors.frozen_points, np.linalg.inv(origin))
            factors = replace(factors, frozen_points=frozen)
            refined, report = solve_window(initial, factors, calibration.K_R, config)
            if report["status"] == "refined":
                for i, pose in zip(window, refined):
                    poses[i] = origin @ pose
                # Carry the endpoint correction into the unseen VO tail while
                # preserving its original relative camera increments.
                tail = poses[window[-1]] @ np.linalg.inv(frame_map[window[-1]].T_SR)
                for i in ids:
                    if i > window[-1]:
                        poses[i] = tail @ frame_map[i].T_SR
                poses[window[0]] = origin
            diagnostics.append({**diagnostic, **report})
    for segment in sorted({f.segment_id for f in result.frames}):
        root = next(f for f in result.frames if f.segment_id == segment)
        poses[root.frame_index] = frame_map[root.frame_index].T_SR.copy()
    return poses, {
        "windows": diagnostics,
        "optimization_seconds": perf_counter() - started,
    }
