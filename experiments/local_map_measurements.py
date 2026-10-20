"""Conditional landmark-noise and appearance controls for the recent local map.

This module is experimental. Camera poses are conditioned on, so covariances
exclude shared pose errors and must never be interpreted as calibrated map
uncertainty. The RGB-ray depth approximation and static exposure model remain.
"""

from dataclasses import dataclass

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.diagnostics import grid_coverage
from scene_recall.multiview.pose_only import retract, skew
from scene_recall.odometry.local_map import information_ratio
from scene_recall.odometry.recent_map import RecentMapConfig, RecentMapRGBDBackend


@dataclass(frozen=True)
class MeasurementConfig(RecentMapConfig):
    objective: str = "legacy"
    appearance: str = "latest"

    def __post_init__(self):
        super().__post_init__()
        if self.objective not in ("legacy", "rgb", "rgbd"):
            raise ValueError("unsupported objective")
        if self.appearance not in ("latest", "family"):
            raise ValueError("unsupported appearance")


def point_covariance(points, K):
    """Propagate one-pixel RGB-ray and existing .005 + .003 z^2 axial noise."""
    B = np.zeros((len(points), 3, 3))
    B[:, 0, 0] = points[:, 2] / K[0, 0]
    B[:, 1, 1] = points[:, 2] / K[1, 1]
    B[:, :, 2] = points / points[:, 2:3]
    variance = np.ones((len(points), 3))
    variance[:, 2] = (0.005 + 0.003 * points[:, 2] ** 2) ** 2
    return (B * variance[:, None, :]) @ B.transpose(0, 2, 1)


def measurement_system(points, pixels, measured, covariance, pose, K, use_depth):
    """Build joint (u,v,z) residuals without counting target RGB rays twice.

    Missing depth contributes only two RGB rows. Source covariance is propagated
    to the target. Whitening is conditional and frozen during each line search.
    """
    p = transform_points(points, pose)
    if np.any(p[:, 2] <= 0):
        raise ValueError("points behind camera")
    n = len(p)
    A = np.zeros((n, 3, 3))
    A[:, 0, 0], A[:, 1, 1] = K[0, 0] / p[:, 2], K[1, 1] / p[:, 2]
    A[:, 0, 2] = -K[0, 0] * p[:, 0] / p[:, 2] ** 2
    A[:, 1, 2] = -K[1, 1] * p[:, 1] / p[:, 2] ** 2
    depth = np.isfinite(measured).all(axis=1) & use_depth
    A[depth, 2, 2] = 1
    residual = np.zeros((n, 3))
    residual[:, :2] = project_points(p, K) - pixels
    residual[depth, 2] = p[depth, 2] - measured[depth, 2]
    C = pose[:3, :3] @ covariance @ pose[:3, :3].T
    S = A @ C @ A.transpose(0, 2, 1)
    S[:, 0, 0] += 1
    S[:, 1, 1] += 1
    S[:, 2, 2] += np.where(depth, (0.005 + 0.003 * measured[:, 2] ** 2) ** 2, 1)
    W = np.linalg.inv(np.linalg.cholesky(S))
    J = A @ np.concatenate((np.broadcast_to(np.eye(3), (n, 3, 3)), -skew(p)), axis=2)
    return residual, J, W


def refine_measurements(points, pixels, measured, covariance, pose, K, use_depth):
    """Bounded Cauchy IRLS with per-step frozen covariance and common support."""
    pose = pose.copy()
    initial = pose.copy()
    accepted = 0
    for _ in range(10):
        r, J, W = measurement_system(
            points, pixels, measured, covariance, pose, K, use_depth
        )
        e = (W @ r[..., None])[..., 0]
        B = W @ J
        weights = 1 / np.sqrt(1 + np.sum(e * e, axis=1) / 2.5**2)
        B = (B * weights[:, None, None]).reshape(-1, 6)
        e = (e * weights[:, None]).ravel()
        delta, _, rank, _ = np.linalg.lstsq(B, -e, rcond=1e-10)
        if rank < 6 or not np.isfinite(delta).all():
            break
        cost = np.log1p(np.sum((W @ r[..., None])[..., 0] ** 2, axis=1) / 2.5**2).sum()
        moved = False
        for scale in (1.0, 0.5, 0.25, 0.125):
            candidate = retract(pose, scale * delta)
            relative = candidate @ np.linalg.inv(initial)
            if np.linalg.norm(relative[:3, 3]) > 0.1 or np.trace(
                relative[:3, :3]
            ) < 1 + 2 * np.cos(np.radians(5)):
                continue
            try:
                trial, _, _ = measurement_system(
                    points, pixels, measured, covariance, candidate, K, use_depth
                )
            except ValueError:
                continue
            trial = (W @ trial[..., None])[..., 0]
            value = np.log1p(np.sum(trial * trial, axis=1) / 2.5**2).sum()
            if value < cost - 1e-12:
                pose, moved = candidate, True
                accepted += 1
                break
        if not moved or np.linalg.norm(scale * delta) < 1e-7:
            break
    return pose, accepted


def family_matches(ids, descriptors, target, ratio=0.75):
    """Mutual Lowe tests over distinct landmark families and target features.

    Distance to an identity is the minimum over its retained descriptors. Two
    views of the same landmark cannot become each other's ratio competitor.
    """
    if descriptors is None or target is None or len(target) < 2:
        return []
    unique = np.unique(ids)
    if len(unique) < 2:
        return []
    distances = np.full((len(unique), len(target)), np.inf, dtype=np.float32)
    for begin in range(0, len(ids), 512):
        a = descriptors[begin : begin + 512]
        d = np.maximum(
            np.sum(a * a, axis=1)[:, None]
            + np.sum(target * target, axis=1)[None]
            - 2 * (a @ target.T),
            0,
        )
        np.minimum.at(distances, np.searchsorted(unique, ids[begin : begin + 512]), d)
    forward = np.argsort(distances, axis=1, kind="stable")[:, :2]
    reverse = np.argsort(distances, axis=0, kind="stable")[:2]
    result = []
    for i, (j, k) in enumerate(forward):
        a, b = reverse[:, j]
        if (
            a == i
            and distances[i, j] < ratio**2 * distances[i, k]
            and distances[i, j] < ratio**2 * distances[b, j]
        ):
            result.append((int(unique[i]), int(j)))
    return result


class MeasurementMapBackend(RecentMapRGBDBackend):
    """Keep original consensus and insertion gates while changing one factor model."""

    def __init__(self, config=None):
        self._covariances = {}
        self._anchors = {}
        self.packets = []
        super().__init__(config or MeasurementConfig())

    def _insert(self, index, features, pose, matches, calibration, reserved=()):
        super()._insert(index, features, pose, matches, calibration, reserved)
        R = (pose @ np.linalg.inv(calibration.T_RD))[:3, :3]
        for feature in np.flatnonzero(self.owners[index] >= 0):
            identity = int(self.owners[index][feature])
            point = features.points_R[feature]
            if np.isfinite(point).all():
                candidate = transform_points(
                    point, pose @ np.linalg.inv(calibration.T_RD)
                )
                if np.allclose(
                    candidate, self.landmarks[identity].point_S, atol=1e-12, rtol=0
                ):
                    self._anchors[identity] = (index, point.copy())
                    self._covariances[identity] = (
                        R @ point_covariance(point[None], calibration.K_R)[0] @ R.T
                    )
        self._covariances = {
            i: c for i, c in self._covariances.items() if i in self.landmarks
        }
        self._anchors = {i: a for i, a in self._anchors.items() if i in self.landmarks}

    def _search(self, features):
        if self.config.appearance == "latest":
            return super()._search(features)
        if not self._searched:
            ids, descriptors = [], []
            for frame_id, frame in self.keyframes.items():
                for feature in np.flatnonzero(self.owners[frame_id] >= 0):
                    identity = int(self.owners[frame_id][feature])
                    if identity in self.landmarks:
                        ids.append(identity)
                        descriptors.append(frame.features.descriptors[feature])
            self._search_pairs = family_matches(
                np.array(ids),
                np.array(descriptors, dtype=np.float32) if ids else None,
                features.descriptors,
                self.config.sparse.ratio,
            )
            self._searched = True
            self._stats.update(index_queries=1, index_matches=len(self._search_pairs))
        return self._search_pairs

    def _point_noise(self, matches, points, calibration):
        covariance = np.array([self._covariances[i] for i, _ in matches])
        anchors = [self._anchors[i] for i, _ in matches]
        lookup = {
            int(i): j for j, i in enumerate(self._previous_ids) if i in self.landmarks
        }
        R = (self._source_pose @ np.linalg.inv(calibration.T_RD))[:3, :3]
        for slot, (identity, _) in enumerate(matches):
            j = lookup.get(identity)
            if j is None:
                continue
            point = self._previous[1].points_R[j]
            candidate = transform_points(
                point, self._source_pose @ np.linalg.inv(calibration.T_RD)
            )
            if np.isfinite(point).all() and np.allclose(
                candidate, points[slot], atol=1e-12, rtol=0
            ):
                anchors[slot] = (self._previous[0].frame_index, point.copy())
                covariance[slot] = (
                    R @ point_covariance(point[None], calibration.K_R)[0] @ R.T
                )
        self._selected_anchors = anchors
        return covariance

    def _estimate_map(self, features, calibration, matches):
        # The parent may recover using an augmented correspondence set. Returned
        # support is the authoritative common population for all factor controls.
        pose, support, diagnostics = super()._estimate_map(
            features, calibration, matches
        )
        if pose is None:
            return pose, support, diagnostics
        saved_stats = self._stats.copy()
        points = self._tracking_points(support, calibration)
        self._stats = saved_stats
        indices = np.array([j for _, j in support])
        pixels, measured = features.pixels[indices], features.points_R[indices]
        covariance = self._point_noise(support, points, calibration)
        T = calibration.T_RD @ np.linalg.inv(pose)
        if (self._previous[0].frame_index + 1) % 7 == 0:
            self.packets.append(
                {
                    "target": np.array(self._previous[0].frame_index + 1),
                    "points": points.copy(),
                    "pixels": pixels.copy(),
                    "measured": measured.copy(),
                    "covariance": covariance.copy(),
                    "pose": T.copy(),
                    "previous_pose": self._source_pose.copy(),
                    "K": calibration.K_R.copy(),
                    "anchor_indices": np.array([a[0] for a in self._selected_anchors]),
                    "anchor_points": np.array([a[1] for a in self._selected_anchors]),
                }
            )
        r, _, W = measurement_system(
            points, pixels, measured, covariance, T, calibration.K_R, True
        )
        whitened = (W @ r[..., None])[..., 0]
        has_depth = np.isfinite(measured).all(axis=1)
        diagnostics.update(
            conditional_rgb_chi2_mean=float(
                np.mean(np.sum(whitened[:, :2] ** 2, axis=1))
            ),
            conditional_depth_chi2_mean=float(np.mean(whitened[has_depth, 2] ** 2))
            if has_depth.any()
            else None,
            signed_depth_median_m=float(np.median(r[has_depth, 2]))
            if has_depth.any()
            else None,
            conditional_depth_over_3sigma=int(
                (np.abs(whitened[has_depth, 2]) > 3).sum()
            ),
        )
        if self.config.objective == "legacy":
            return pose, support, diagnostics
        T, steps = refine_measurements(
            points,
            pixels,
            measured,
            covariance,
            T,
            calibration.K_R,
            self.config.objective == "rgbd",
        )
        selected = self._consensus(points, pixels, measured, T, calibration.K_R)
        valid = (
            selected.sum() >= self.config.sparse.min_inliers
            and selected.sum() / diagnostics["map_matches"]
            >= self.config.sparse.min_inlier_ratio
        )
        valid &= (
            grid_coverage(pixels[selected], calibration.rgb_shape)
            >= self.config.min_grid_coverage
        )
        valid &= (
            information_ratio(transform_points(points[selected], T))
            >= self.config.min_information_ratio
        )
        diagnostics.update(
            measurement_refinement_steps=steps,
            measurement_refinement_accepted=bool(valid),
            measurement_retained_inliers=int(selected.sum()),
        )
        if valid:
            pose = np.linalg.inv(T) @ calibration.T_RD
            support = [support[i] for i in np.flatnonzero(selected)]
        return pose, support, diagnostics
