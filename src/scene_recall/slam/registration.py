"""Reference-free RGB-D registration with a disjoint verification population.

One half of mutual descriptor correspondences estimates a rigid transform.
The other half never selects a hypothesis or refits its parameters. Native
metric distance and bidirectional image projection verify the frozen result.
These engineering gates do not imply calibrated match probabilities.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.diagnostics import grid_coverage
from scene_recall.multiview.tracking import match_descriptors
from scene_recall.odometry.local_map import information_ratio
from scene_recall.odometry.sparse import SparseConfig, estimate_sparse_pose


@dataclass(frozen=True)
class RegistrationConfig:
    ratio: float = 0.75
    min_pairs: int = 12
    min_fit: int = 6
    min_validation: int = 6
    min_fraction: float = 0.5
    min_grid_coverage: float = 0.08
    min_information_ratio: float = 1e-5
    depth_absolute_m: float = 0.03
    depth_relative: float = 0.01
    reprojection_px: float = 3.0
    max_trials: int = 256
    seed: int = 0

    def __post_init__(self):
        for name, lower in (
            ("min_pairs", 12),
            ("min_fit", 3),
            ("min_validation", 3),
            ("max_trials", 1),
            ("seed", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        for name in ("ratio", "min_fraction", "min_grid_coverage"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be strictly between zero and one")
        for name in (
            "min_information_ratio",
            "depth_absolute_m",
            "depth_relative",
            "reprojection_px",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Registration:
    T_target_source: np.ndarray | None
    diagnostics: dict = field(default_factory=dict)


def rigid_alignment(source, target):
    """Unit-scale Kabsch alignment, rejecting collinear and repeated samples."""
    a, b = source - source.mean(axis=0), target - target.mean(axis=0)
    for centered in (a, b):
        singular = np.linalg.svd(centered, compute_uv=False)
        if len(singular) < 2 or singular[1] <= max(1e-9, singular[0] * 1e-3):
            raise ValueError("collinear metric correspondences")
    U, _, Vt = np.linalg.svd(a.T @ b)
    correction = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)])
    pose = np.eye(4)
    pose[:3, :3] = Vt.T @ correction @ U.T
    pose[:3, 3] = target.mean(axis=0) - pose[:3, :3] @ source.mean(axis=0)
    return pose


def _errors(pose, a, b, pixels_a, pixels_b, K, config):
    forward = transform_points(a, pose)
    backward = transform_points(b, np.linalg.inv(pose))
    metric = np.linalg.norm(forward - b, axis=1)
    pixel = np.maximum(
        np.linalg.norm(project_points(forward, K) - pixels_b, axis=1),
        np.linalg.norm(project_points(backward, K) - pixels_a, axis=1),
    )
    threshold = (
        config.depth_absolute_m + config.depth_relative * (a[:, 2] + b[:, 2]) / 2
    )
    valid = (
        (forward[:, 2] > 0.05)
        & (backward[:, 2] > 0.05)
        & (metric <= threshold)
        & (pixel <= config.reprojection_px)
    )
    return valid, metric, pixel


def register_correspondences(a, b, pixels_a, pixels_b, K, shape, config=None):
    """Fit even rows only, then validate odd rows without fitting them."""
    config = config or RegistrationConfig()
    n = len(a)
    if (
        a.shape != (n, 3)
        or b.shape != (n, 3)
        or pixels_a.shape != (n, 2)
        or pixels_b.shape != (n, 2)
    ):
        raise ValueError("invalid correspondence shapes")
    if (
        not all(np.isfinite(v).all() for v in (a, b, pixels_a, pixels_b))
        or (a[:, 2] <= 0).any()
        or (b[:, 2] <= 0).any()
    ):
        raise ValueError("metric correspondences must be finite and in front")
    fit, validation = np.arange(0, n, 2), np.arange(1, n, 2)
    diagnostic = {
        "metric_pairs": n,
        "fit_pairs": len(fit),
        "validation_pairs": len(validation),
    }

    def rejected(reason, **extra):
        return Registration(
            None, {**diagnostic, "status": "rejected", "reason": reason, **extra}
        )

    if (
        n < config.min_pairs
        or len(fit) < config.min_fit
        or len(validation) < config.min_validation
    ):
        return rejected("insufficient disjoint metric support")
    if len(fit) <= 12:
        trials = list(combinations(fit, 3))[: config.max_trials]
    else:
        rng = np.random.default_rng(config.seed)
        trials = [rng.choice(fit, 3, replace=False) for _ in range(config.max_trials)]
    best, best_score = None, (-1, -float("inf"))
    for rows in trials:
        rows = np.asarray(rows)
        try:
            pose = rigid_alignment(a[rows], b[rows])
        except np.linalg.LinAlgError:
            continue
        except ValueError:
            continue
        valid, metric, _ = _errors(
            pose, a[fit], b[fit], pixels_a[fit], pixels_b[fit], K, config
        )
        score = (
            int(valid.sum()),
            -float(np.median(metric[valid])) if valid.any() else -float("inf"),
        )
        if score > best_score:
            best, best_score = pose, score
    if best is None or best_score[0] < config.min_fit:
        return rejected("insufficient fitting consensus")
    for _ in range(2):
        valid, _, _ = _errors(
            best, a[fit], b[fit], pixels_a[fit], pixels_b[fit], K, config
        )
        if valid.sum() < config.min_fit:
            return rejected("insufficient refined fitting consensus")
        try:
            best = rigid_alignment(a[fit[valid]], b[fit[valid]])
        except (ValueError, np.linalg.LinAlgError):
            return rejected("degenerate refined geometry")
    valid, metric, pixel = _errors(best, a, b, pixels_a, pixels_b, K, config)
    train_ok, test_ok = valid[fit], valid[validation]
    diagnostic.update(
        fit_inliers=int(train_ok.sum()),
        validation_inliers=int(test_ok.sum()),
        validation_fraction=float(test_ok.mean()),
    )
    if train_ok.sum() < config.min_fit or train_ok.mean() < config.min_fraction:
        return rejected("insufficient refined fitting consensus", **diagnostic)
    if test_ok.sum() < config.min_validation or test_ok.mean() < config.min_fraction:
        return rejected("withheld correspondence disagreement", **diagnostic)
    rows = validation[test_ok]
    coverage = min(
        grid_coverage(pixels_a[rows], shape), grid_coverage(pixels_b[rows], shape)
    )
    information = min(information_ratio(a[rows]), information_ratio(b[rows]))
    diagnostic.update(
        validation_grid_coverage=coverage,
        validation_information_ratio=information,
        validation_metric_rmse_m=float(np.sqrt(np.mean(metric[rows] ** 2))),
        validation_pixel_rmse=float(np.sqrt(np.mean(pixel[rows] ** 2))),
    )
    if (
        coverage < config.min_grid_coverage
        or information < config.min_information_ratio
    ):
        return rejected("weak withheld geometry", **diagnostic)
    return Registration(
        best,
        {
            **diagnostic,
            "status": "accepted",
            "reason": "disjoint metric and image verification",
        },
    )


def register_features(first, second, calibration, config=None):
    config = config or RegistrationConfig()
    pairs, _ = match_descriptors(first.descriptors, second.descriptors, config.ratio)
    if len(pairs):
        valid = np.isfinite(first.points_R[pairs[:, 0]]).all(axis=1) & np.isfinite(
            second.points_R[pairs[:, 1]]
        ).all(axis=1)
        pairs = pairs[valid]
    q, t = pairs.T
    result = register_correspondences(
        first.points_R[q],
        second.points_R[t],
        first.pixels[q],
        second.pixels[t],
        calibration.K_R,
        calibration.rgb_shape,
        config,
    )
    return result


def register_anchor(first, second, calibration, config=None):
    """Historical metric anchors verified by new, withheld image measurements.

    Incoming depth is checked wherever it exists, without requiring it to exist.
    Only even correspondences enter PnP or its refinement. This still conditions
    on noisy source depth and does not identify coherent calibration bias.
    """
    config = config or RegistrationConfig()
    pairs, _ = match_descriptors(first.descriptors, second.descriptors, config.ratio)
    return register_anchor_pairs(first, second, pairs, calibration, config)


def register_anchor_pairs(first, second, pairs, calibration, config=None):
    """Verify explicit one-to-one associations, keeping fitting and test rows apart.

    Association provenance belongs to the caller. No descriptor or flow score is
    interpreted as extra metric evidence or independent information.
    """
    config = config or RegistrationConfig()
    pairs = np.asarray(pairs)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or not np.issubdtype(pairs.dtype, np.integer)
    ):
        raise ValueError("pairs must be integer (N, 2)")
    if len(pairs) and (
        (pairs < 0).any()
        or (pairs[:, 0] >= len(first.pixels)).any()
        or (pairs[:, 1] >= len(second.pixels)).any()
        or len(np.unique(pairs[:, 0])) != len(pairs)
        or len(np.unique(pairs[:, 1])) != len(pairs)
    ):
        raise ValueError("pairs must reference unique source and target rows")
    if len(pairs):
        pairs = pairs[np.isfinite(first.points_R[pairs[:, 0]]).all(axis=1)]
    fit, test = np.arange(0, len(pairs), 2), np.arange(1, len(pairs), 2)
    diagnostic = {
        "metric_pairs": len(pairs),
        "fit_pairs": len(fit),
        "validation_pairs": len(test),
        "method": "historical_pnp",
    }

    def rejected(reason):
        return Registration(
            None, {**diagnostic, "status": "rejected", "reason": reason}
        )

    if (
        len(pairs) < config.min_pairs
        or len(fit) < config.min_fit
        or len(test) < config.min_validation
    ):
        return rejected("insufficient disjoint anchor support")
    q, t = pairs.T
    a, b = first.points_R[q], second.points_R[t]
    minimum = max(5, config.min_fit)
    if len(fit) < minimum:
        return rejected("insufficient PnP fit support")
    result = estimate_sparse_pose(
        a[fit],
        second.pixels[t[fit]],
        calibration.K_R,
        SparseConfig(
            min_correspondences=minimum,
            min_inliers=minimum,
            min_inlier_ratio=config.min_fraction,
            reprojection_threshold_px=config.reprojection_px,
            seed=config.seed,
        ),
    )
    if result.T_C1C0 is None:
        return rejected("anchor PnP " + result.reason)
    pose = result.T_C1C0
    projected = transform_points(a, pose)
    errors = np.linalg.norm(
        project_points(projected, calibration.K_R) - second.pixels[t], axis=1
    )
    has_depth = np.isfinite(b).all(axis=1)
    limits = config.depth_absolute_m + config.depth_relative * np.where(
        has_depth, b[:, 2], 0
    )
    valid = (projected[:, 2] > 0.05) & (errors <= config.reprojection_px)
    valid &= ~has_depth | (np.linalg.norm(projected - b, axis=1) <= limits)
    selected = test[valid[test]]
    diagnostic.update(
        fit_inliers=int(valid[fit].sum()),
        validation_inliers=len(selected),
        validation_depth_checks=int(has_depth[selected].sum()),
        validation_fraction=float(valid[test].mean()),
    )
    if (
        valid[fit].sum() < config.min_fit
        or valid[fit].mean() < config.min_fraction
        or len(selected) < config.min_validation
        or valid[test].mean() < config.min_fraction
    ):
        return rejected("withheld anchor disagreement")
    coverage = min(
        grid_coverage(first.pixels[q[selected]], calibration.rgb_shape),
        grid_coverage(second.pixels[t[selected]], calibration.rgb_shape),
    )
    information = information_ratio(projected[selected])
    diagnostic.update(
        validation_grid_coverage=coverage,
        validation_information_ratio=information,
        validation_pixel_rmse=float(np.sqrt(np.mean(errors[selected] ** 2))),
    )
    if (
        coverage < config.min_grid_coverage
        or information < config.min_information_ratio
    ):
        return rejected("weak withheld anchor geometry")
    return Registration(
        pose,
        {
            **diagnostic,
            "status": "accepted",
            "reason": "disjoint historical anchor verification",
        },
    )
