"""Multiscale projective point-to-plane ICP in canonical depth optical frames."""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import cv2
import numpy as np

from scene_recall.data import Calibration, Observation
from scene_recall.geometry.camera import (
    backproject_depth,
    project_points,
    reject_depth_edges,
    transform_points,
)
from scene_recall.odometry.pipeline import PairEstimate, is_rigid

FACTORS = (4, 2, 1)
DISTANCES_M = (0.12, 0.08, 0.05)
HUBER_M = 0.01


@dataclass(frozen=True)
class ICPConfig:
    """Validated experiment defaults plus explicit acceptance gates.

    Finest-scale convergence is required. Coarse iteration caps allow progress
    to the next scale. Rank and support gates apply at every scale and again
    after the last update. Residuals describe measured geometry, not pose truth.
    """

    max_iterations: int = 35
    min_support: int = 30
    min_support_fraction: float = 0.2
    depth_edge_threshold_m: float = 0.05
    min_depth_m: float = 0.3
    max_depth_m: float = 6.0
    huber_m: float = HUBER_M
    normal_gate_deg: float = 45.0
    translation_tolerance_m: float = 1e-5
    rotation_tolerance_deg: float = 0.001

    def __post_init__(self):
        for name, lower in (("max_iterations", 1), ("min_support", 6)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        for name in (
            "depth_edge_threshold_m",
            "min_depth_m",
            "max_depth_m",
            "huber_m",
            "translation_tolerance_m",
            "rotation_tolerance_deg",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must exceed min_depth_m")
        if (
            not isfinite(self.min_support_fraction)
            or not 0 <= self.min_support_fraction <= 1
        ):
            raise ValueError("min_support_fraction must be between zero and one")
        if not isfinite(self.normal_gate_deg) or not 0 < self.normal_gate_deg < 90:
            raise ValueError("normal_gate_deg must be between zero and 90")


def huber(residual, delta=HUBER_M):
    absolute = np.abs(residual)
    return np.where(
        absolute <= delta, 0.5 * residual**2, delta * (absolute - 0.5 * delta)
    )


def increment(update, scale=1.0):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(update[3:], float))[0]
    result[:3, 3] = np.asarray(update[:3]) * scale
    return result


def make_cloud(depth, K, factor=1, normal_radius_native=3, config=None):
    """Retain original z-depth samples and use smoothed depth only for normals.

    Subsampling starts at native pixel zero, so every entry of K's first two
    rows is divided by factor. Missing depth remains unavailable. Normals use
    central differences of a bilateral normal-estimation image.
    """
    config = config or ICPConfig()
    depth = reject_depth_edges(depth, config.depth_edge_threshold_m)[
        ::factor, ::factor
    ].copy()
    depth[(depth < config.min_depth_m) | (depth > config.max_depth_m)] = np.nan
    intrinsic = K.copy()
    intrinsic[:2] /= factor
    points = backproject_depth(depth, intrinsic).astype(np.float64)
    smooth = cv2.bilateralFilter(np.nan_to_num(depth).astype(np.float32), 5, 0.02, 2)
    smooth[~np.isfinite(depth)] = np.nan
    normal_points = backproject_depth(smooth, intrinsic)
    radius = max(1, normal_radius_native // factor)
    normals = np.full(points.shape, np.nan)
    du = (
        normal_points[radius:-radius, 2 * radius :]
        - normal_points[radius:-radius, : -2 * radius]
    )
    dv = (
        normal_points[2 * radius :, radius:-radius]
        - normal_points[: -2 * radius, radius:-radius]
    )
    cross = np.cross(du, dv)
    length = np.linalg.norm(cross, axis=2, keepdims=True)
    np.divide(cross, length, out=cross, where=length > 1e-12)
    cross[length[..., 0] <= 1e-12] = np.nan
    normals[radius:-radius, radius:-radius] = cross
    valid = np.isfinite(points).all(axis=2) & np.isfinite(normals).all(axis=2)
    normals[~valid] = np.nan
    v, u = np.indices(depth.shape)
    sample = valid & (u % 2 == 0) & (v % 2 == 0)
    # Native 16x16-pixel tiles are split without referring to any pose.
    heldout = ((u * factor // 16 + v * factor // 16) % 2) == 1
    return {
        "points": points.reshape(-1, 3),
        "normals": normals.reshape(-1, 3),
        "K": intrinsic,
        "shape": depth.shape,
        "factor": factor,
        "train": np.flatnonzero(sample & ~heldout),
        "test": np.flatnonzero(sample & heldout),
        "all": np.flatnonzero(sample),
        "valid_normal_count": int(valid.sum()),
    }


def correspondences(source, target, pose, indices, distance=0.05, normal_gate_deg=45):
    """Project each selected source point onto the nearest target pixel.

    Support requires positive visible projection, valid source/target normals,
    point distance within the gate, and normals agreeing within 45 degrees.
    Returned source indices always refer to the same source depth array.
    """
    moved = transform_points(source["points"][indices], pose)
    pixels = project_points(moved, target["K"])
    h, w = target["shape"]
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= w - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= h - 1)
    )
    selected = indices[valid]
    moved = moved[valid]
    uv = np.floor(pixels[valid] + 0.5).astype(int)
    target_indices = uv[:, 1] * w + uv[:, 0]
    fixed = target["points"][target_indices]
    normal = target["normals"][target_indices]
    source_normal = source["normals"][selected] @ pose[:3, :3].T
    valid = (
        np.isfinite(fixed).all(axis=1)
        & np.isfinite(normal).all(axis=1)
        & (np.linalg.norm(moved - fixed, axis=1) <= distance)
        & (
            np.sum(source_normal * normal, axis=1)
            >= np.cos(np.radians(normal_gate_deg))
        )
    )
    selected, moved, fixed, normal, target_indices = (
        x[valid] for x in (selected, moved, fixed, normal, target_indices)
    )
    residual = np.sum((moved - fixed) * normal, axis=1)
    return {
        "source_ids": selected,
        "target_ids": target_indices,
        "moved": moved,
        "fixed": fixed,
        "normals": normal,
        "residual": residual,
    }


def linearization(correspondence, scale, huber_m=HUBER_M):
    normals = correspondence["normals"]
    J = np.column_stack([normals * scale, np.cross(correspondence["moved"], normals)])
    residual = correspondence["residual"]
    weights = np.minimum(1, huber_m / np.maximum(np.abs(residual), 1e-12))
    H = J.T @ (weights[:, None] * J)
    gradient = J.T @ (weights * residual)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    rank = int(np.count_nonzero(eigenvalues > max(eigenvalues[-1] * 1e-8, 1e-12)))
    return J, H, gradient, eigenvalues, eigenvectors, rank


def register_depth(source_pyramid, target_pyramid, initial, config):
    """Return an accepted pose or explicit failure, retaining the candidate.

    Native alternating tiles reserve depth samples for common-support scoring.
    Target neighborhoods can overlap, so these scores are diagnostics only.
    """
    if not is_rigid(initial, atol=1e-6):
        raise ValueError("registration initialization must be rigid")
    if len(source_pyramid) != len(FACTORS) or len(target_pyramid) != len(FACTORS):
        raise ValueError("registration requires all three pyramid levels")
    pose = initial.copy()
    diagnostics = {}
    failure = None
    for source, target, distance in zip(source_pyramid, target_pyramid, DISTANCES_M):
        factor = source["factor"]
        indices = source["train"]
        status = "iteration_limit"
        iterations = 0
        if len(indices) < config.min_support:
            status = "insufficient_support"
        else:
            scale = float(np.median(source["points"][indices, 2]))
            for iterations in range(1, config.max_iterations + 1):
                c = correspondences(
                    source, target, pose, indices, distance, config.normal_gate_deg
                )
                if (
                    len(c["residual"]) < config.min_support
                    or len(c["residual"]) / len(indices) < config.min_support_fraction
                ):
                    status = "insufficient_support"
                    break
                _, H, gradient, eigenvalues, _, rank = linearization(
                    c, scale, config.huber_m
                )
                if rank < 6:
                    status = "rank_deficient"
                    break
                try:
                    update = -np.linalg.solve(
                        H + np.eye(6) * eigenvalues[-1] * 1e-8, gradient
                    )
                except np.linalg.LinAlgError:
                    status = "linear_solver_failed"
                    break
                if not np.isfinite(update).all():
                    status = "nonfinite_update"
                    break
                before = float(huber(c["residual"], config.huber_m).mean())
                accepted = False
                for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    delta = increment(update * alpha, scale)
                    moved = transform_points(c["moved"], delta)
                    residual = np.sum((moved - c["fixed"]) * c["normals"], axis=1)
                    after = float(huber(residual, config.huber_m).mean())
                    if isfinite(after) and after <= before + 1e-15:
                        pose = delta @ pose
                        accepted = True
                        break
                if not accepted:
                    status = "line_search_stalled"
                    break
                if (
                    np.linalg.norm(update[:3] * scale) * alpha
                    < config.translation_tolerance_m
                    and np.degrees(np.linalg.norm(update[3:])) * alpha
                    < config.rotation_tolerance_deg
                ):
                    status = "step_tolerance"
                    break
        diagnostics[f"icp_level_{factor}_status"] = status
        diagnostics[f"icp_level_{factor}_iterations"] = iterations
        if status != "step_tolerance" and (factor == 1 or status != "iteration_limit"):
            failure = status
            break
    source, target = source_pyramid[-1], target_pyramid[-1]
    c = correspondences(
        source, target, pose, source["train"], DISTANCES_M[-1], config.normal_gate_deg
    )
    count = len(c["residual"])
    fraction = count / len(source["train"]) if len(source["train"]) else 0.0
    diagnostics.update(
        icp_support=count,
        icp_support_fraction=fraction,
        icp_rmse_m=float(np.sqrt(np.mean(c["residual"] ** 2))) if count else None,
        icp_rank=None,
        icp_condition=None,
    )
    if count >= config.min_support:
        scale = float(np.median(source["points"][source["train"], 2]))
        _, _, _, eigenvalues, _, rank = linearization(c, scale, config.huber_m)
        diagnostics["icp_rank"] = rank
        diagnostics["icp_condition"] = (
            float(np.sqrt(eigenvalues[-1] / eigenvalues[0])) if rank == 6 else None
        )
        if rank < 6:
            failure = failure or "rank_deficient"
    if count < config.min_support or fraction < config.min_support_fraction:
        failure = failure or "insufficient_support"
    if not is_rigid(pose, atol=1e-6):
        failure = failure or "nonrigid_pose"
    # Both poses are scored on identical held-out source IDs with their own associations.
    before = correspondences(
        source, target, initial, source["test"], DISTANCES_M[-1], config.normal_gate_deg
    )
    after = correspondences(
        source, target, pose, source["test"], DISTANCES_M[-1], config.normal_gate_deg
    )
    common = np.intersect1d(before["source_ids"], after["source_ids"])
    diagnostics["icp_common_test_support"] = len(common)
    for name, row in (("initial", before), ("final", after)):
        residual = row["residual"][np.isin(row["source_ids"], common)]
        diagnostics[f"icp_common_test_{name}_rmse_m"] = (
            float(np.sqrt(np.mean(residual**2))) if len(residual) else None
        )
    diagnostics["icp_status"] = failure or "converged"
    return PairEstimate(
        None if failure else pose,
        failure,
        diagnostics=diagnostics,
        candidate_T_C1C0=pose,
    )


class DepthICPBackend:
    """Independent depth estimator initialized at identity for every pair.

    A single target pyramid is cached for sequential use. Neither RGB values,
    T_RD, reference poses, nor prior estimated motion enter registration.
    """

    def __init__(self, config: ICPConfig | None = None):
        self.config = config or ICPConfig()
        self._cached = None

    def _pyramid(self, observation, calibration):
        if observation.depth.shape != calibration.depth_shape:
            raise ValueError("depth shape must match calibration")
        if self._cached is not None:
            cached_observation, cached_calibration, pyramid = self._cached
            if cached_observation is observation and cached_calibration is calibration:
                return pyramid
        return [
            make_cloud(observation.depth, calibration.K_D, f, config=self.config)
            for f in FACTORS
        ]

    def estimate(
        self, source: Observation, target: Observation, calibration: Calibration
    ) -> PairEstimate:
        return self.refine(source, target, calibration, np.eye(4))

    def refine(self, source, target, calibration, initial):
        started = perf_counter()
        source_pyramid = self._pyramid(source, calibration)
        target_pyramid = self._pyramid(target, calibration)
        self._cached = (target, calibration, target_pyramid)
        result = register_depth(source_pyramid, target_pyramid, initial, self.config)
        result.diagnostics["icp_seconds"] = perf_counter() - started
        return result
