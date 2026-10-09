"""SIFT correspondences with metric PnP, RANSAC, and optional LM refinement."""

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np

from scene_recall.data import Calibration, Observation
from scene_recall.geometry.camera import (
    backproject_rgb_pixels,
    project_points,
    register_depth_to_rgb,
    reject_depth_edges,
    transform_points,
)
from scene_recall.odometry.pipeline import PairEstimate, is_rigid


@dataclass(frozen=True)
class SparseConfig:
    """Explicit, serializable settings for the sparse RGB-D baseline."""

    ratio: float = 0.75
    mutual_matching: bool = True
    max_features: int = 0
    depth_edge_threshold_m: float | None = 0.05
    iterations: int = 2000
    confidence: float = 0.999
    reprojection_threshold_px: float = 3.0
    refine: bool = True
    min_correspondences: int = 20
    min_inliers: int = 15
    min_inlier_ratio: float = 0.25
    min_coverage: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        for name, lower in (
            ("max_features", 0),
            ("iterations", 1),
            ("min_correspondences", 5),
            ("min_inliers", 5),
            ("seed", 0),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.seed > 2**31 - 1:
            raise ValueError("seed must fit a signed 32-bit integer")
        for name in ("ratio", "confidence"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and strictly between 0 and 1")
        for name in ("min_inlier_ratio", "min_coverage"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        for name in ("reprojection_threshold_px", "depth_edge_threshold_m"):
            value = getattr(self, name)
            if value is None and name == "depth_edge_threshold_m":
                continue
            if value is None or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.refine, bool) or not isinstance(
            self.mutual_matching, bool
        ):
            raise TypeError("refine and mutual_matching must be booleans")


def ransac_pnp(
    points_C0: np.ndarray,
    pixels_1: np.ndarray,
    K_R: np.ndarray,
    *,
    iterations: int = 100,
    confidence: float = 0.99,
    threshold_px: float = 3.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw EPnP RANSAC pose and original consensus for legacy diagnostics.

    C0 is the source RGB optical frame. No inversion is needed for the forward
    C0 -> C1 pose. Returned errors use OpenCV's final EPnP fit and original mask,
    so some can exceed the RANSAC threshold. No refinement occurs here.
    """
    if len(points_C0) < 5:
        raise ValueError("EPNP RANSAC requires at least 5 correspondences")
    if seed is not None:
        cv2.setRNGSeed(seed)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=np.ascontiguousarray(points_C0, dtype=np.float64),
        imagePoints=np.ascontiguousarray(pixels_1, dtype=np.float64),
        cameraMatrix=K_R,
        distCoeffs=None,
        iterationsCount=iterations,
        reprojectionError=threshold_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or inliers.size == 0:
        raise RuntimeError("solvePnPRansac did not return a pose with inliers")
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        raise RuntimeError("solvePnPRansac returned a nonfinite pose")
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(3)
    indices = inliers.reshape(-1)
    projected, _ = cv2.projectPoints(points_C0[indices], rvec, tvec, K_R, None)
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels_1[indices], axis=1)
    if not np.isfinite(errors).all():
        raise RuntimeError("PnP inlier reprojection errors are nonfinite")
    return T, indices, errors


def reprojection_errors(
    points: np.ndarray, pixels: np.ndarray, K: np.ndarray, T: np.ndarray
) -> np.ndarray:
    """Count nonfinite and behind-camera projections as infinite errors."""
    projected = project_points(transform_points(points, T), K)
    errors = np.linalg.norm(projected - pixels, axis=1)
    return np.where(np.isfinite(errors), errors, np.inf)


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors**2)))


def estimate_sparse_pose(
    points: np.ndarray, pixels: np.ndarray, K: np.ndarray, config: SparseConfig
) -> PairEstimate:
    """Fit an RGB-frame pose and assess its final, recomputed consensus.

    LM optimizes ordinary squared reprojection error on the RANSAC consensus.
    A second fit uses reclassified inliers. This is not a robust-loss optimizer.
    Failures are explicit and never silently fall back to an unrefined pose.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points must have shape (N, 3)")
    if pixels.shape != (len(points), 2):
        raise ValueError("pixels must have shape (N, 2) aligned with points")
    if not np.isfinite(points).all() or not np.isfinite(pixels).all():
        raise ValueError("correspondences must be finite")
    diagnostics = {
        "correspondences": len(points),
        "ransac_inliers": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "refinement": "disabled" if not config.refine else "not_run",
    }
    if len(points) < config.min_correspondences:
        return PairEstimate(
            None, "insufficient correspondences", diagnostics=diagnostics
        )
    try:
        initial, indices, _ = ransac_pnp(
            points,
            pixels,
            K,
            iterations=config.iterations,
            confidence=config.confidence,
            threshold_px=config.reprojection_threshold_px,
            seed=config.seed,
        )
    except (ValueError, RuntimeError, cv2.error) as error:
        diagnostics["solver_error"] = str(error)
        return PairEstimate(None, "RANSAC failed", diagnostics=diagnostics)
    diagnostics["ransac_inliers"] = len(indices)
    initial_errors = reprojection_errors(points, pixels, K, initial)
    if not np.isfinite(initial_errors[indices]).all():
        return PairEstimate(None, "invalid RANSAC projections", initial, diagnostics)
    diagnostics["initial_consensus_rmse_px"] = _rmse(initial_errors[indices])
    pose = initial.copy()
    if config.refine:
        try:
            selected = indices
            for _ in range(2):
                if len(selected) < 5:
                    raise RuntimeError("insufficient refinement inliers")
                rvec, tvec = cv2.solvePnPRefineLM(
                    objectPoints=points[selected].copy(),
                    imagePoints=pixels[selected].copy(),
                    cameraMatrix=K.copy(),
                    distCoeffs=None,
                    rvec=cv2.Rodrigues(pose[:3, :3])[0],
                    tvec=pose[:3, 3].reshape(3, 1).copy(),
                )
                if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                    raise RuntimeError("LM returned a nonfinite pose")
                candidate = np.eye(4)
                candidate[:3, :3] = cv2.Rodrigues(rvec)[0]
                candidate[:3, 3] = tvec.ravel()
                before = reprojection_errors(
                    points[selected], pixels[selected], K, pose
                )
                after = reprojection_errors(
                    points[selected], pixels[selected], K, candidate
                )
                if not np.isfinite(after).all() or _rmse(after) > _rmse(before) + 1e-6:
                    raise RuntimeError("LM increased the selected reprojection error")
                pose = candidate
                new_selected = np.flatnonzero(
                    reprojection_errors(points, pixels, K, pose)
                    <= config.reprojection_threshold_px
                )
                if np.array_equal(np.sort(selected), new_selected):
                    break
                selected = new_selected
            diagnostics["refinement"] = "refined"
        except (ValueError, RuntimeError, cv2.error) as error:
            diagnostics["refinement"] = "failed"
            diagnostics["solver_error"] = str(error)
            return PairEstimate(None, "LM refinement failed", initial, diagnostics)
    errors = reprojection_errors(points, pixels, K, pose)
    inliers = np.flatnonzero(errors <= config.reprojection_threshold_px)
    diagnostics.update(
        inliers=len(inliers),
        inlier_ratio=len(inliers) / len(points),
        final_original_consensus_rmse_px=(
            _rmse(errors[indices]) if np.isfinite(errors[indices]).all() else None
        ),
        inlier_rmse_px=_rmse(errors[inliers]) if len(inliers) else None,
        inlier_median_px=float(np.median(errors[inliers])) if len(inliers) else None,
        inlier_max_px=float(errors[inliers].max()) if len(inliers) else None,
    )
    if not is_rigid(pose, atol=1e-6):
        return PairEstimate(None, "nonrigid estimated pose", initial, diagnostics)
    if len(inliers) < config.min_inliers:
        return PairEstimate(None, "insufficient final inliers", initial, diagnostics)
    if diagnostics["inlier_ratio"] < config.min_inlier_ratio:
        return PairEstimate(None, "low final inlier ratio", initial, diagnostics)
    return PairEstimate(pose, initial_T_C1C0=initial, diagnostics=diagnostics)


@dataclass(frozen=True, eq=False)
class _Features:
    pixels: np.ndarray
    descriptors: np.ndarray | None
    points_R: np.ndarray


class SparseRGBDBackend:
    """SIFT backend with a one-observation feature cache and no reference-pose use.

    RGB-frame PnP estimates are conjugated into the canonical depth frame.
    Seeded OpenCV RANSAC is reproducible within a fixed software environment.
    The stateful cache and OpenCV RNG calls are intended for serial use.
    """

    def __init__(self, config: SparseConfig | None = None) -> None:
        self.config = config or SparseConfig()
        self._sift = cv2.SIFT_create(nfeatures=self.config.max_features)
        self._cached: tuple[Observation, Calibration, _Features] | None = None

    def _features(
        self, observation: Observation, calibration: Calibration
    ) -> _Features:
        if self._cached is not None:
            previous, previous_calibration, features = self._cached
            if previous is observation and previous_calibration is calibration:
                return features
        keypoints, descriptors = self._sift.detectAndCompute(
            cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2GRAY), None
        )
        pixels = np.array(
            [keypoint.pt for keypoint in keypoints], dtype=np.float64
        ).reshape(-1, 2)
        depth = observation.depth
        if self.config.depth_edge_threshold_m is not None:
            depth = reject_depth_edges(depth, self.config.depth_edge_threshold_m)
        if np.array_equal(calibration.T_RD, np.eye(4)):
            points = backproject_rgb_pixels(
                pixels, depth, calibration.K_R, calibration.K_D
            )
        else:
            registered = register_depth_to_rgb(
                observation.depth,
                calibration.K_D,
                calibration.K_R,
                calibration.T_RD,
                calibration.rgb_shape,
                valid_mask=np.isfinite(depth),
            )
            points = backproject_rgb_pixels(
                pixels, registered, calibration.K_R, calibration.K_R
            )
        return _Features(pixels, descriptors, points)

    def estimate(
        self, source: Observation, target: Observation, calibration: Calibration
    ) -> PairEstimate:
        if not is_rigid(calibration.T_RD, atol=1e-6):
            raise ValueError("odometry requires rigid T_RD calibration")
        for observation in (source, target):
            if observation.rgb.shape[:2] != calibration.rgb_shape:
                raise ValueError("RGB shape must match calibration")
            if observation.depth.shape != calibration.depth_shape:
                raise ValueError("depth shape must match calibration")
        features_0 = self._features(source, calibration)
        features_1 = self._features(target, calibration)
        self._cached = (target, calibration, features_1)
        descriptors_0, descriptors_1 = features_0.descriptors, features_1.descriptors
        matches = []
        ratio_count = 0
        if descriptors_0 is not None and descriptors_1 is not None:
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            neighbors = matcher.knnMatch(descriptors_0, descriptors_1, k=2)
            matches = [
                row[0]
                for row in neighbors
                if len(row) == 2
                and row[0].distance < self.config.ratio * row[1].distance
            ]
            ratio_count = len(matches)
            if self.config.mutual_matching:
                reverse = {
                    match.queryIdx: match.trainIdx
                    for match in matcher.match(descriptors_1, descriptors_0)
                }
                matches = [
                    match
                    for match in matches
                    if reverse.get(match.trainIdx) == match.queryIdx
                ]
            # Keep one source per target even when mutual checking is disabled.
            unique = {}
            for match in sorted(
                matches, key=lambda match: (match.distance, match.queryIdx)
            ):
                unique.setdefault(match.trainIdx, match)
            matches = sorted(unique.values(), key=lambda match: match.queryIdx)
        queries = np.array([match.queryIdx for match in matches], dtype=np.intp)
        trains = np.array([match.trainIdx for match in matches], dtype=np.intp)
        valid = np.isfinite(features_0.points_R[queries]).all(axis=1)
        queries, trains = queries[valid], trains[valid]
        points = features_0.points_R[queries]
        pixels_1 = features_1.pixels[trains]
        estimate = estimate_sparse_pose(points, pixels_1, calibration.K_R, self.config)
        diagnostics = dict(estimate.diagnostics)
        diagnostics.update(
            source_keypoints=len(features_0.pixels),
            target_keypoints=len(features_1.pixels),
            ratio_matches=ratio_count,
            unique_matches=len(matches),
            depth_valid_matches=len(points),
        )
        pose_R = estimate.T_C1C0
        reason = estimate.reason
        if pose_R is not None:
            inliers = (
                reprojection_errors(points, pixels_1, calibration.K_R, pose_R)
                <= self.config.reprojection_threshold_px
            )
            height, width = calibration.rgb_shape
            coverage = []
            for pixels in (features_0.pixels[queries][inliers], pixels_1[inliers]):
                coverage.append(
                    float(np.prod(np.ptp(pixels, axis=0)) / (height * width))
                )
            diagnostics.update(source_coverage=coverage[0], target_coverage=coverage[1])
            if min(coverage) < self.config.min_coverage:
                pose_R = None
                reason = "low spatial coverage"
        T_DR = np.linalg.inv(calibration.T_RD)
        return PairEstimate(
            None if pose_R is None else T_DR @ pose_R @ calibration.T_RD,
            reason,
            None
            if estimate.initial_T_C1C0 is None
            else T_DR @ estimate.initial_T_C1C0 @ calibration.T_RD,
            diagnostics,
        )
