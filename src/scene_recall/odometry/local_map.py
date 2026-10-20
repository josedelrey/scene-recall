"""Experimental bounded keyframe frontend with persistent metric landmarks.

All poses and landmarks live in the current independent depth-camera segment.
Only accepted observations enter the map. Reference poses are never accessed.
The serial backend owns its stream state and must not be shared across streams.
"""

from dataclasses import dataclass, field
from time import perf_counter

import cv2
import numpy as np

from scene_recall.features import extract_rgbd_features
from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.diagnostics import grid_coverage
from scene_recall.multiview.geometry import landmark_geometry
from scene_recall.multiview.model import FeatureFrame, TrackConfig
from scene_recall.multiview.tracking import match_descriptors
from scene_recall.odometry.evaluation import pose_error
from scene_recall.odometry.pipeline import PairEstimate
from scene_recall.odometry.sparse import (
    SparseConfig,
    estimate_feature_pair,
    estimate_sparse_pose,
    reprojection_errors,
)


@dataclass(frozen=True)
class LocalMapConfig:
    """Fixed engineering gates, independent of source reference trajectories."""

    sparse: SparseConfig = field(
        default_factory=lambda: SparseConfig(max_features=1500)
    )
    max_keyframes: int = 6
    max_landmarks: int = 4000
    min_keyframe_gap: int = 3
    max_keyframe_gap: int = 20
    keyframe_translation_m: float = 0.08
    keyframe_rotation_deg: float = 8.0
    keyframe_overlap: float = 0.65
    depth_absolute_m: float = 0.03
    depth_relative: float = 0.01
    min_grid_coverage: float = 0.15
    min_information_ratio: float = 1e-5
    landmark_update: str = "frozen"
    pairwise_fallback: bool = True

    def __post_init__(self):
        if not isinstance(self.sparse, SparseConfig):
            raise TypeError("sparse must be a SparseConfig")
        if self.sparse.max_features < 1:
            raise ValueError("local-map tracking requires a positive feature budget")
        for name, lower in (
            ("max_keyframes", 2),
            ("max_landmarks", 20),
            ("min_keyframe_gap", 1),
            ("max_keyframe_gap", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.max_keyframe_gap < self.min_keyframe_gap:
            raise ValueError("maximum keyframe gap must cover the minimum gap")
        for name in (
            "keyframe_translation_m",
            "keyframe_rotation_deg",
            "depth_absolute_m",
            "depth_relative",
            "min_information_ratio",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("keyframe_overlap", "min_grid_coverage"):
            if (
                not np.isfinite(getattr(self, name))
                or not 0 <= getattr(self, name) <= 1
            ):
                raise ValueError(f"{name} must lie between zero and one")
        if self.landmark_update not in ("frozen", "median"):
            raise ValueError("landmark_update must be frozen or median")
        if not isinstance(self.pairwise_fallback, bool):
            raise TypeError("pairwise_fallback must be a boolean")


@dataclass
class MapLandmark:
    """A stable identity with bounded active keyframe measurement support."""

    point_S: np.ndarray
    observations: tuple[tuple[int, int], ...]


def information_ratio(points_R):
    """Smallest/largest normalized bearing information eigenvalue.

    Translation columns use median scene depth to remove metric scale dependence.
    This is a degeneracy diagnostic, without a calibrated uncertainty claim.
    """
    if not len(points_R) or np.any(points_R[:, 2] <= 0):
        return 0.0
    x, y, z = points_R.T
    u, v = x / z, y / z
    depth = np.median(z)
    J = np.zeros((len(z), 2, 6))
    J[:, 0, :] = np.column_stack(
        (depth / z, z * 0, -u * depth / z, -u * v, 1 + u * u, -v)
    )
    J[:, 1, :] = np.column_stack(
        (z * 0, depth / z, -v * depth / z, -1 - v * v, u * v, u)
    )
    values = np.linalg.eigvalsh(J.reshape(-1, 6).T @ J.reshape(-1, 6))
    return float(max(0, values[0]) / max(values[-1], 1e-15))


class LocalMapRGBDBackend:
    """Fit incoming RGB pixels to a bounded persistent keyframe map.

    Map PnP is attempted before consecutive-frame fallback. Successful map poses
    are absolute in the segment gauge, so chain drift is not reaccumulated.
    A failed transition discards the old map and bootstraps the target at identity.
    No recovery or association crosses that boundary.
    """

    def __init__(self, config=None):
        self.config = config or LocalMapConfig()
        self._sift = cv2.SIFT_create(nfeatures=self.config.sparse.max_features)
        self.keyframes = {}
        self.owners = {}
        self.landmarks = {}
        self._next_landmark = 0
        self._previous = None
        self._calibration = None
        self._pose = np.eye(4)

    def _extract(self, observation, calibration):
        return extract_rgbd_features(
            observation,
            calibration,
            self._sift,
            self.config.sparse.depth_edge_threshold_m,
        )

    def _bootstrap(self, observation, features, calibration):
        self.keyframes.clear()
        self.owners.clear()
        self.landmarks.clear()
        self._pose = np.eye(4)
        self._insert(observation.frame_index, features, self._pose, [], calibration)

    def _matches(self, features):
        # Match each view separately to avoid ratio-test ambiguity between multiple
        # descriptors of one identity. Resolve conflicts before geometric fitting.
        proposals = []
        for frame_id, frame in self.keyframes.items():
            pairs, ratios = match_descriptors(
                frame.features.descriptors,
                features.descriptors,
                self.config.sparse.ratio,
            )
            for (query, train), ratio in zip(pairs, ratios, strict=True):
                landmark = int(self.owners[frame_id][query])
                if landmark in self.landmarks:
                    proposals.append((float(ratio), landmark, int(train)))
        by_feature, by_landmark = {}, {}
        for ratio, landmark, feature in proposals:
            by_feature.setdefault(feature, set()).add(landmark)
            by_landmark.setdefault(landmark, set()).add(feature)
        unique = {
            (landmark, feature)
            for _, landmark, feature in proposals
            if len(by_feature[feature]) == len(by_landmark[landmark]) == 1
        }
        return sorted(unique), set(by_feature)

    def _consensus(self, points, pixels, measured, pose, K):
        projected = transform_points(points, pose)
        valid_depth = np.isfinite(measured).all(axis=1)
        limits = (
            self.config.depth_absolute_m + self.config.depth_relative * measured[:, 2]
        )
        depth_ok = ~valid_depth | (
            np.linalg.norm(projected - measured, axis=1) <= limits
        )
        return (
            reprojection_errors(points, pixels, K, pose)
            <= self.config.sparse.reprojection_threshold_px
        ) & depth_ok

    def _estimate_map(self, features, calibration, matches):
        diagnostics = {"map_matches": len(matches)}
        if not matches:
            return None, [], {**diagnostics, "map_reason": "no map matches"}
        ids, indices = np.array(matches).T
        points = np.array([self.landmarks[int(i)].point_S for i in ids])
        pixels = features.pixels[indices]
        measured = features.points_R[indices]
        fit = estimate_sparse_pose(points, pixels, calibration.K_R, self.config.sparse)
        if fit.T_C1C0 is None:
            return None, [], {**diagnostics, "map_reason": fit.reason}
        # Depth gates precede the final fit, then the final consensus is rechecked.
        selected = self._consensus(
            points, pixels, measured, fit.T_C1C0, calibration.K_R
        )
        fit = estimate_sparse_pose(
            points[selected], pixels[selected], calibration.K_R, self.config.sparse
        )
        if fit.T_C1C0 is None:
            return (
                None,
                [],
                {**diagnostics, "map_reason": "depth-filtered " + fit.reason},
            )
        selected = self._consensus(
            points, pixels, measured, fit.T_C1C0, calibration.K_R
        )
        count = int(selected.sum())
        coverage = grid_coverage(pixels[selected], calibration.rgb_shape)
        ratio = information_ratio(transform_points(points[selected], fit.T_C1C0))
        diagnostics.update(
            map_inliers=count,
            map_inlier_ratio=count / len(matches),
            map_grid_coverage=coverage,
            map_information_ratio=ratio,
            map_confirmed_inliers=sum(
                len(self.landmarks[int(i)].observations) >= 2 for i in ids[selected]
            ),
            map_latest_keyframe_inliers=sum(
                any(
                    k[0] == next(reversed(self.keyframes))
                    for k in self.landmarks[int(i)].observations
                )
                for i in ids[selected]
            ),
            map_depth_checked_inliers=int(
                np.isfinite(measured[selected]).all(axis=1).sum()
            ),
            map_inlier_rmse_px=float(
                np.sqrt(
                    np.mean(
                        reprojection_errors(
                            points[selected],
                            pixels[selected],
                            calibration.K_R,
                            fit.T_C1C0,
                        )
                        ** 2
                    )
                )
            )
            if count
            else None,
        )
        reason = None
        if (
            count < self.config.sparse.min_inliers
            or count / len(matches) < self.config.sparse.min_inlier_ratio
        ):
            reason = "low RGB-D consensus"
        elif coverage < self.config.min_grid_coverage:
            reason = "low map grid coverage"
        elif ratio < self.config.min_information_ratio:
            reason = "degenerate map geometry"
        diagnostics["map_reason"] = reason
        if reason:
            return None, [], diagnostics
        pose = np.linalg.inv(fit.T_C1C0) @ calibration.T_RD
        return pose, [matches[i] for i in np.flatnonzero(selected)], diagnostics

    def _insert(self, index, features, pose, matches, calibration, reserved=()):
        frame = FeatureFrame(
            index, 0, None, pose @ np.linalg.inv(calibration.T_RD), features
        )
        frames = {**self.keyframes, index: frame}
        owners = np.full(len(features.pixels), -1, dtype=np.int64)
        geometry_config = TrackConfig(
            reprojection_px=self.config.sparse.reprojection_threshold_px,
            depth_absolute_m=self.config.depth_absolute_m,
            depth_relative=self.config.depth_relative,
        )
        # A descriptor-associated feature that fails geometry or identity checks
        # must not be reintroduced as a fresh depth anchor under a new identity.
        rejected = set(reserved)
        for landmark_id, feature in matches:
            landmark = self.landmarks[landmark_id]
            keys = landmark.observations + ((index, feature),)
            if self.config.landmark_update == "median":
                geometry = landmark_geometry(
                    keys, frames, calibration.K_R, geometry_config
                )
                if geometry is None or geometry["score"] > 1:
                    rejected.add(feature)
                    continue
                landmark.point_S = geometry["point_S"]
            # Frozen points have already passed current pixel and depth gates.
            landmark.observations = keys
            owners[feature] = landmark_id
        self.keyframes[index] = frame
        self.owners[index] = owners
        while len(self.keyframes) > self.config.max_keyframes:
            oldest = next(iter(self.keyframes))
            del self.keyframes[oldest], self.owners[oldest]
            for identity, landmark in list(self.landmarks.items()):
                landmark.observations = tuple(
                    k for k in landmark.observations if k[0] != oldest
                )
                if not landmark.observations:
                    del self.landmarks[identity]
        # Round-robin occupied cells retain spatially distributed new anchors.
        valid = np.flatnonzero(
            np.isfinite(features.points_R).all(axis=1) & (owners < 0)
        )
        cells = np.floor(
            features.pixels[valid]
            / [calibration.rgb_shape[1], calibration.rgb_shape[0]]
            * [8, 6]
        ).astype(int)
        bins = {}
        for feature, cell in zip(valid, cells, strict=True):
            if int(feature) not in rejected:
                bins.setdefault(tuple(cell), []).append(int(feature))
        ordered = [
            values[j]
            for j in range(max(map(len, bins.values()), default=0))
            for _, values in sorted(bins.items())
            if j < len(values)
        ]
        for feature in ordered[
            : max(0, self.config.max_landmarks - len(self.landmarks))
        ]:
            identity = self._next_landmark
            self._next_landmark += 1
            self.landmarks[identity] = MapLandmark(
                transform_points(features.points_R[feature], frame.T_SR),
                ((index, feature),),
            )
            owners[feature] = identity

    def estimate(self, source, target, calibration):
        started = perf_counter()
        if target.frame_index != source.frame_index + 1:
            raise ValueError("local-map tracking requires consecutive input indices")
        if self._previous is None:
            first = self._extract(source, calibration)
            self._bootstrap(source, first, calibration)
            self._previous = (source, first)
            self._calibration = calibration
        elif self._previous[0] is not source or self._calibration is not calibration:
            raise ValueError("local-map backend cannot be reused across streams")
        features = self._extract(target, calibration)
        extracted = perf_counter()
        previous_pose = self._pose.copy()
        proposals, reserved = self._matches(features)
        pose, matches, diagnostics = self._estimate_map(
            features, calibration, proposals
        )
        mode = "map"
        if pose is None and self.config.pairwise_fallback:
            fallback = estimate_feature_pair(
                self._previous[1], features, calibration, self.config.sparse
            )
            diagnostics["fallback_reason"] = fallback.reason
            if fallback.T_C1C0 is not None:
                pose = previous_pose @ np.linalg.inv(fallback.T_C1C0)
                mode = "pairwise_fallback"
                if proposals:
                    ids, indices = np.array(proposals).T
                    selected = self._consensus(
                        np.array([self.landmarks[int(i)].point_S for i in ids]),
                        features.pixels[indices],
                        features.points_R[indices],
                        calibration.T_RD @ np.linalg.inv(pose),
                        calibration.K_R,
                    )
                    matches = [proposals[i] for i in np.flatnonzero(selected)]
        inserted = False
        tracked = perf_counter()
        if pose is None:
            mode = "lost"
            self._bootstrap(target, features, calibration)
        else:
            self._pose = pose
            last = next(reversed(self.keyframes.values()))
            gap = target.frame_index - last.frame_index
            translation, rotation = pose_error(
                pose @ np.linalg.inv(calibration.T_RD), last.T_SR
            )
            overlap = len(matches) / max(
                1, int((self.owners[last.frame_index] >= 0).sum())
            )
            inserted = gap >= self.config.min_keyframe_gap and (
                gap >= self.config.max_keyframe_gap
                or translation >= self.config.keyframe_translation_m
                or rotation >= self.config.keyframe_rotation_deg
                or overlap < self.config.keyframe_overlap
            )
            if inserted:
                self._insert(
                    target.frame_index, features, pose, matches, calibration, reserved
                )
        self._previous = (target, features)
        diagnostics.update(
            tracking_mode=mode,
            keyframe_inserted=inserted,
            active_keyframes=len(self.keyframes),
            active_landmarks=len(self.landmarks),
            oldest_keyframe=next(iter(self.keyframes)),
            feature_seconds=extracted - started,
            map_tracking_seconds=tracked - extracted,
            map_update_seconds=perf_counter() - tracked,
            frontend_seconds=perf_counter() - started,
        )
        return PairEstimate(
            None if pose is None else np.linalg.inv(pose) @ previous_pose,
            diagnostics.get("map_reason") if pose is None else None,
            diagnostics=diagnostics,
        )
