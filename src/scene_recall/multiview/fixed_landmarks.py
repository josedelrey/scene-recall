"""Fixed-landmark RGB and native-depth factors for a future matched BA control.

Camera poses alone are updated. Point geometry, within-pair motion, depth
bindings, whitening and robust block definitions stay fixed during each window.
"""

import hashlib
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.geometry.transforms import is_rigid
from scene_recall.multiview.audit import normalized_spectrum
from scene_recall.multiview.pose_only import (
    PoseOnlyConfig,
    make_factors,
    projection_jacobian,
    retract,
    skew,
)


def geometric_median(samples):
    """Bounded, rotation-equivariant initialization from measured samples only."""
    samples = np.asarray(samples)
    point = samples.mean(axis=0)
    for _ in range(50):
        weights = 1 / np.maximum(np.linalg.norm(samples - point, axis=1), 1e-10)
        candidate = np.sum(samples * weights[:, None], axis=0) / weights.sum()
        if np.linalg.norm(candidate - point) < 1e-9:
            return candidate
        point = candidate
    return point


@dataclass(frozen=True, eq=False)
class DirectFactors:
    view: np.ndarray
    track_id: np.ndarray
    points: np.ndarray
    pixels: np.ndarray
    native_points: np.ndarray
    T_RcD: np.ndarray
    whitening_D: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        n = len(self.view)
        shapes = {
            "view": (n,),
            "track_id": (n,),
            "points": (n, 3),
            "pixels": (n, 2),
            "native_points": (n, 3),
            "T_RcD": (n, 4, 4),
            "whitening_D": (n, 3, 3),
            "weights": (n,),
        }
        if not n or any(
            getattr(self, name).shape != shape for name, shape in shapes.items()
        ):
            raise ValueError("invalid direct factor shape")
        if (
            self.view.dtype.kind not in "iu"
            or self.track_id.dtype.kind not in "iu"
            or (self.view < 0).any()
        ):
            raise ValueError("direct factor identities must be nonnegative integers")
        if len(set(zip(self.view, self.track_id, strict=True))) != n:
            raise ValueError("duplicate direct observation")
        for name in ("points", "pixels", "weights", "whitening_D"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError("nonfinite direct factor")
        valid = np.isfinite(self.native_points).all(axis=1) & (
            self.native_points[:, 2] > 0
        )
        if (
            not (valid | np.isnan(self.native_points).all(axis=1)).all()
            or (self.weights <= 0).any()
        ):
            raise ValueError("invalid native sample or factor weight")
        if (np.linalg.det(self.whitening_D[valid]) <= 0).any():
            raise ValueError("invalid native whitening")
        if any(not is_rigid(m) for m in np.unique(self.T_RcD, axis=0)):
            raise ValueError("native camera transforms must be rigid")

    def fingerprint(self):
        digest = hashlib.sha256()
        for name in self.__dataclass_fields__:
            array = getattr(self, name)
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


def make_direct_factors(
    tracks, samples, ids, calibration, config=None, *, static=False
):
    """Select one frozen population for fixed and future free-landmark fitting."""
    config = config or PoseOnlyConfig()
    if (
        config.pixel_sigma != 1.0
        or config.depth_sigma_constant_m != 0.005
        or config.depth_sigma_quadratic != 0.003
    ):
        raise ValueError(
            "direct factors use the frozen one-pixel and native-depth uncertainty model"
        )
    selection, support = make_factors(
        tracks, ids, calibration.K_R, calibration.rgb_shape, config
    )
    if selection is None:
        return None, support
    selected = set(map(int, selection.track_ids))
    frames = {f.frame_index: f for f in tracks.frames}
    positions = {index: i for i, index in enumerate(ids)}
    origin = frames[ids[0]].T_SR
    views, track_ids, points, pixels, native, transforms, whitening, weights = (
        [] for _ in range(8)
    )
    for track in tracks.landmarks:
        if track.track_id not in selected:
            continue
        keys = [k for k in track.observations if k[0] in positions]
        measured = []
        for f, k in keys:
            sample = samples[f]
            if np.isfinite(sample.points_D[k]).all():
                transform = calibration.T_RD if static else sample.T_RcD
                local = transform_points(sample.points_D[k], transform)
                measured.append(
                    transform_points(local, np.linalg.inv(origin) @ frames[f].T_SR)
                )
        point = geometric_median(measured)
        for f, k in keys:
            views.append(positions[f])
            track_ids.append(track.track_id)
            points.append(point)
            pixels.append(frames[f].features.pixels[k])
            native.append(samples[f].points_D[k])
            transforms.append(calibration.T_RD if static else samples[f].T_RcD)
            whitening.append(samples[f].whitening_D[k])
            weights.append(1 / len(keys))
    factors = DirectFactors(
        np.array(views, dtype=int),
        np.array(track_ids, dtype=int),
        np.array(points),
        np.array(pixels),
        np.array(native),
        np.array(transforms),
        np.array(whitening),
        np.array(weights),
    )
    return factors, {
        **support,
        "selection_star_factors": support.get("factors", 0),
        "factors": len(views)
        + int(np.isfinite(factors.native_points).all(axis=1).sum()),
        "rgb_blocks": len(views),
        "depth_blocks": int(np.isfinite(factors.native_points).all(axis=1).sum()),
        "factors_sha256": factors.fingerprint(),
    }


def direct_linearize(poses, factors, K, free, *, jacobian=True):
    """Padded RGB/depth blocks and left-pose Jacobians, without point updates."""
    views = factors.view
    rotation = poses[views, :3, :3].transpose(0, 2, 1)
    qR = np.einsum("nij,nj->ni", rotation, factors.points - poses[views, :3, 3])
    inverse = np.linalg.inv(factors.T_RcD)
    qD = np.einsum("nij,nj->ni", inverse[:, :3, :3], qR) + inverse[:, :3, 3]
    valid = np.isfinite(factors.native_points).all(axis=1)
    if (
        not np.isfinite(qR).all()
        or (qR[:, 2] <= 0.05).any()
        or (qD[valid, 2] <= 0.05).any()
    ):
        return None, None
    residual = np.zeros((len(views), 2, 3))
    residual[:, 0, :2] = project_points(qR, K) - factors.pixels
    residual[valid, 1] = np.einsum(
        "nij,nj->ni",
        factors.whitening_D[valid],
        qD[valid] - factors.native_points[valid],
    )
    if not jacobian:
        return residual, None
    derivative = np.concatenate((-rotation, rotation @ skew(factors.points)), axis=2)
    rgb = projection_jacobian(qR, K) @ derivative
    rgb[:, 2] = 0
    depth = factors.whitening_D @ inverse[:, :3, :3] @ derivative
    J = np.zeros((len(views), 2, 3, 6 * len(free)))
    for column, index in enumerate(free):
        mask = views == index
        J[mask, 0, :, 6 * column : 6 * (column + 1)] = rgb[mask]
        mask &= valid
        J[mask, 1, :, 6 * column : 6 * (column + 1)] = depth[mask]
    return residual, J


def direct_cost(residual, factors, config):
    if residual is None:
        return float("inf"), None
    s = np.sum(residual**2, axis=2)
    weights = np.repeat(factors.weights[:, None], 2, axis=1)
    weights[:, 1] *= np.isfinite(factors.native_points).all(axis=1)
    if config.loss == "linear":
        return float(0.5 * np.sum(weights * s)), weights
    c2 = config.loss_scale**2
    return float(0.5 * np.sum(weights * c2 * np.log1p(s / c2))), weights / (1 + s / c2)


def direct_information(poses, factors, K, free, config=None):
    """Inspect free-point observability without computing any point update.

    A future joint backend must reuse this frozen Sparse gate. Fixed-landmark
    pose information alone can conceal disconnected multiview components.
    """
    config = config or PoseOnlyConfig()
    r, JP = direct_linearize(poses, factors, K, free)
    if r is None or not free:
        return {
            "condition": None,
            "nullity": None,
            "reason": "no valid free-pose information",
        }
    _, weights = direct_cost(r, factors, config)
    R = poses[factors.view, :3, :3].transpose(0, 2, 1)
    qR = np.einsum("nij,nj->ni", R, factors.points - poses[factors.view, :3, 3])
    JL = np.zeros((len(factors.view), 2, 3, 3))
    JL[:, 0] = projection_jacobian(qR, K) @ R
    JL[:, 0, 2] = 0
    JL[:, 1] = factors.whitening_D @ np.linalg.inv(factors.T_RcD)[:, :3, :3] @ R
    JP = JP * np.sqrt(weights[:, :, None, None])
    JL = JL * np.sqrt(weights[:, :, None, None])
    flat = JP.reshape(-1, JP.shape[-1])
    information = flat.T @ flat
    excluded = 0
    for track in np.unique(factors.track_id):
        selected = factors.track_id == track
        point_J = JL[selected].reshape(-1, 3)
        pose_J = JP[selected].reshape(-1, JP.shape[-1])
        HLL = point_J.T @ point_J
        eig = np.linalg.eigvalsh(HLL)
        if eig[0] <= max(1e-15, eig[-1] * 1e-12):
            excluded += 1
            continue
        cross = pose_J.T @ point_J
        information -= cross @ np.linalg.solve(HLL, cross.T)
    spectrum = normalized_spectrum(information)
    if excluded:
        spectrum["condition"] = None
    return {**spectrum, "excluded_point_blocks": excluded}


def solve_fixed(initial, factors, K, free, config=None):
    """Bounded damped IRLS, fixing every camera outside free exactly."""
    config = config or PoseOnlyConfig()
    poses = np.array(initial, dtype=float, copy=True)
    if any(not is_rigid(p) for p in poses):
        raise ValueError("fixed-landmark initialization must be rigid")
    free = tuple(free)
    if len(set(free)) != len(free) or any(i <= 0 or i >= len(poses) for i in free):
        raise ValueError("free poses must be unique non-root camera indices")
    r, _ = direct_linearize(poses, factors, K, free)
    cost, _ = direct_cost(r, factors, config)
    initial_cost = cost
    if not np.isfinite(cost):
        return poses, {"status": "rejected", "reason": "invalid initial projections"}
    if not free:
        return poses, {"status": "unchanged", "initial_cost": cost, "final_cost": cost}
    damping, accepted_costs, condition = 0.001, [], None
    for iteration in range(config.max_iterations):
        r, J = direct_linearize(poses, factors, K, free)
        _, weights = direct_cost(r, factors, config)
        flat_J = (J * np.sqrt(weights[:, :, None, None])).reshape(-1, J.shape[-1])
        flat_r = (r * np.sqrt(weights[:, :, None])).ravel()
        H, gradient = flat_J.T @ flat_J, flat_J.T @ flat_r
        spectrum = normalized_spectrum(H)
        condition = spectrum["condition"]
        if condition is None or condition > config.max_condition:
            return np.array(initial, copy=True), {
                "status": "rejected",
                "reason": "unobservable or ill-conditioned",
                "condition": condition,
            }
        diagonal = np.maximum(np.diag(H), 1e-15)
        accepted = False
        for _ in range(10):
            delta = np.linalg.solve(H + damping * np.diag(diagonal), -gradient).reshape(
                -1, 6
            )
            trial = poses.copy()
            for index, change in zip(free, delta, strict=True):
                trial[index] = retract(poses[index], change)
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
            residual, _ = direct_linearize(trial, factors, K, free, jacobian=False)
            candidate, _ = direct_cost(residual, factors, config)
            if (
                displacement <= config.max_translation_m
                and angles.max() <= config.max_rotation_deg
                and candidate < cost
            ):
                poses, cost = trial, candidate
                accepted_costs.append(cost)
                damping = max(damping / 3, 1e-9)
                accepted = True
                break
            damping *= 10
        if not accepted or np.linalg.norm(delta) < 1e-6:
            break
    return poses, {
        "status": "refined" if accepted_costs else "unchanged",
        "initial_cost": initial_cost,
        "final_cost": cost,
        "condition": condition,
        "iterations": iteration + 1,
        "accepted_costs": accepted_costs,
    }


def refine_fixed(
    tracks, samples, calibration, config=None, *, static=False, lock_overlap=True
):
    """Local frozen-point fitting with optional committed overlap boundaries."""
    config = config or PoseOnlyConfig()
    tick = perf_counter()
    base = {f.frame_index: f.T_SR for f in tracks.frames}
    poses = {i: p.copy() for i, p in base.items()}
    reports, objectives, factor_archive = [], [], {}
    factor_seconds, solve_seconds, information_seconds = 0.0, 0.0, 0.0
    for segment in sorted({f.segment_id for f in tracks.frames}):
        ids = [f.frame_index for f in tracks.frames if f.segment_id == segment]
        closed_until = ids[0]
        for start in range(0, len(ids) - 2, config.window_step):
            window = ids[start : start + config.window_size]
            started = perf_counter()
            factors, support = make_direct_factors(
                tracks, samples, window, calibration, config, static=static
            )
            factor_seconds += perf_counter() - started
            report = {"segment_id": segment, "frame_indices": window, **support}
            if factors is None:
                reports.append({**report, "status": "skipped"})
                closed_until = max(closed_until, window[-1])
                continue
            origin = poses[window[0]].copy()
            initial = np.array([np.linalg.inv(origin) @ poses[i] for i in window])
            free = [
                j
                for j, i in enumerate(window)
                if j and (not lock_overlap or i > closed_until)
            ]
            base_initial = np.array(
                [np.linalg.inv(base[window[0]]) @ base[i] for i in window]
            )
            started = perf_counter()
            information = direct_information(
                base_initial, factors, calibration.K_R, free, config
            )
            information_seconds += perf_counter() - started
            report["joint_model_information_at_sparse"] = information
            key = f"window_{len(reports):03d}"
            factor_archive.update(
                {
                    f"{key}_{name}": getattr(factors, name)
                    for name in factors.__dataclass_fields__
                }
            )
            factor_archive[f"{key}_origin"] = origin
            factor_archive[f"{key}_initial"] = initial
            if (
                information["condition"] is None
                or information["condition"] > config.max_condition
            ):
                reports.append(
                    {
                        **report,
                        "status": "skipped",
                        "reason": "joint model unobservable or ill-conditioned",
                    }
                )
                closed_until = max(closed_until, window[-1])
                continue
            started = perf_counter()
            fitted, diagnostic = solve_fixed(
                initial, factors, calibration.K_R, free, config
            )
            solve_seconds += perf_counter() - started
            if diagnostic["status"] == "refined":
                for j in free:
                    poses[window[j]] = origin @ fitted[j]
                correction = poses[window[-1]] @ np.linalg.inv(base[window[-1]])
                for i in ids:
                    if i > window[-1]:
                        poses[i] = correction @ base[i]
            current = np.array([np.linalg.inv(origin) @ poses[i] for i in window])
            r, _ = direct_linearize(
                current, factors, calibration.K_R, (), jacobian=False
            )
            at_close, _ = direct_cost(r, factors, config)
            objectives.append((len(reports), window, origin, factors, at_close))
            reports.append(
                {
                    **report,
                    **diagnostic,
                    "free_frame_indices": [window[j] for j in free],
                    "cost_at_close": at_close,
                }
            )
            closed_until = max(closed_until, window[-1])
    for index, window, origin, factors, closed in objectives:
        final = np.array([np.linalg.inv(origin) @ poses[i] for i in window])
        residual, _ = direct_linearize(
            final, factors, calibration.K_R, (), jacobian=False
        )
        cost, _ = direct_cost(residual, factors, config)
        reports[index]["cost_after_all_windows"] = cost if np.isfinite(cost) else None
        if lock_overlap and not np.isclose(cost, closed, rtol=1e-10, atol=1e-9):
            raise AssertionError("committed overlap constraint changed")
    return (
        poses,
        {
            "windows": reports,
            "optimization_seconds": perf_counter() - tick,
            "factor_preparation_seconds": factor_seconds,
            "linear_solver_seconds": solve_seconds,
            "information_seconds": information_seconds,
            "lock_overlap": lock_overlap,
        },
        factor_archive,
    )
