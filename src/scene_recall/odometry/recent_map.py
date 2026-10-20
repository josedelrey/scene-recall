"""Recent-depth local-map tracking with bounded temporal identity propagation.

Retained keyframes define the segment gauge. Fresh depth supplies local motion
without treating changing sets of immutable depth anchors as exact world points.
Every proposed point refresh must agree with all active keyframe observations.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.diagnostics import grid_coverage
from scene_recall.multiview.tracking import match_descriptors
from scene_recall.odometry.local_map import LocalMapConfig, LocalMapRGBDBackend
from scene_recall.odometry.sparse import reprojection_errors


@dataclass(frozen=True)
class RecentMapConfig(LocalMapConfig):
    """Policies are explicit ablations, with the original geometric gates."""

    anchor_policy: str = "recent"
    association_policy: str = "indexed"

    def __post_init__(self):
        super().__post_init__()
        if self.anchor_policy not in ("recent", "frozen"):
            raise ValueError("anchor_policy must be recent or frozen")
        if self.association_policy not in ("indexed", "exhaustive"):
            raise ValueError("association_policy must be indexed or exhaustive")
        if self.landmark_update != "frozen":
            raise ValueError("recent-map policies require the frozen base update")


def merge_proposals(proposals):
    """Exclude every conflicting identity, without choosing by estimated pose."""
    by_feature, by_landmark = {}, {}
    for identity, feature in proposals:
        by_feature.setdefault(feature, set()).add(identity)
        by_landmark.setdefault(identity, set()).add(feature)
    return sorted(
        {
            (identity, feature)
            for identity, feature in proposals
            if len(by_feature[feature]) == len(by_landmark[identity]) == 1
        }
    ), set(by_feature)


class RecentMapRGBDBackend(LocalMapRGBDBackend):
    """Track recent identities first and search a unique-landmark descriptor index.

    Search recovery on insufficient temporal support or failed map fitting, and
    search before inserting a keyframe so reappearing features do not acquire
    duplicate identities. Map fallback and segment resets retain legacy policy.
    """

    def __init__(self, config=None):
        super().__init__(config or RecentMapConfig())
        self._previous_ids = np.empty(0, dtype=np.int64)
        self._index_ids = np.empty(0, dtype=np.int64)
        self._index_descriptors = None
        self._searched = False
        self._search_pairs = []
        self._stats = {}
        self._source_pose = np.eye(4)

    def _bootstrap(self, observation, features, calibration):
        super()._bootstrap(observation, features, calibration)
        self._previous_ids = self.owners[observation.frame_index].copy()

    def _rebuild_index(self):
        # Latest retained descriptor per identity avoids ratio-test competition
        # between six copies of the same physical landmark.
        representatives = {}
        for frame_id, frame in self.keyframes.items():
            if frame.features.descriptors is not None:
                for feature in np.flatnonzero(self.owners[frame_id] >= 0):
                    identity = int(self.owners[frame_id][feature])
                    if identity in self.landmarks:
                        representatives[identity] = frame.features.descriptors[feature]
        self._index_ids = np.array(sorted(representatives), dtype=np.int64)
        self._index_descriptors = (
            np.array(
                [representatives[int(i)] for i in self._index_ids], dtype=np.float32
            )
            if representatives
            else None
        )

    def _search(self, features):
        if not self._searched:
            started = perf_counter()
            pairs, _ = match_descriptors(
                self._index_descriptors, features.descriptors, self.config.sparse.ratio
            )
            self._search_pairs = [(int(self._index_ids[i]), int(j)) for i, j in pairs]
            self._searched = True
            self._stats.update(
                index_queries=1,
                index_matches=len(pairs),
                index_search_seconds=perf_counter() - started,
            )
        return self._search_pairs

    def _matches(self, features):
        self._searched = False
        self._search_pairs = []
        self._stats = {
            "index_queries": 0,
            "carry_matches": 0,
            "recent_depth_anchors": 0,
            "reanchored_landmarks": 0,
        }
        if self.config.association_policy == "exhaustive":
            return super()._matches(features)
        pairs, _ = match_descriptors(
            self._previous[1].descriptors,
            features.descriptors,
            self.config.sparse.ratio,
        )
        proposals = [
            (int(self._previous_ids[i]), int(j))
            for i, j in pairs
            if int(self._previous_ids[i]) in self.landmarks
        ]
        self._stats["carry_matches"] = len(proposals)
        coverage = grid_coverage(
            features.pixels[[j for _, j in proposals]], self._calibration.rgb_shape
        )
        if (
            len(proposals) < self.config.sparse.min_correspondences
            or coverage < self.config.min_grid_coverage
        ):
            proposals += self._search(features)
        return merge_proposals(proposals)

    def _valid_points(self, ids, points, calibration):
        """Vectorized all-active-view check for candidate segment-frame points."""
        valid = np.isfinite(points).all(axis=1)
        if not len(ids):
            return valid
        order = np.argsort(ids)
        ordered = np.asarray(ids)[order]
        for frame_id, frame in self.keyframes.items():
            owners = self.owners[frame_id]
            slots = np.searchsorted(ordered, owners)
            inside = slots < len(ordered)
            inside &= ordered[np.minimum(slots, len(ordered) - 1)] == owners
            features = np.flatnonzero(inside)
            indices = order[slots[inside]]
            if not len(indices):
                continue
            local = transform_points(points[indices], np.linalg.inv(frame.T_SR))
            errors = reprojection_errors(
                points[indices],
                frame.features.pixels[features],
                calibration.K_R,
                np.linalg.inv(frame.T_SR),
            )
            measured = frame.features.points_R[features]
            has_depth = np.isfinite(measured).all(axis=1)
            limits = (
                self.config.depth_absolute_m
                + self.config.depth_relative * measured[:, 2]
            )
            consistent = (errors <= self.config.sparse.reprojection_threshold_px) & (
                ~has_depth | (np.linalg.norm(local - measured, axis=1) <= limits)
            )
            np.logical_and.at(valid, indices, consistent)
        return valid

    def _tracking_points(self, matches, calibration):
        points = super()._tracking_points(matches, calibration)
        if (
            self.config.anchor_policy == "frozen"
            or self._previous is None
            or not matches
        ):
            return points
        lookup = {
            int(identity): i
            for i, identity in enumerate(self._previous_ids)
            if identity in self.landmarks
        }
        proposals, slots, ids = [], [], []
        for slot, (identity, _) in enumerate(matches):
            feature = lookup.get(identity)
            if feature is not None:
                point = self._previous[1].points_R[feature]
                if np.isfinite(point).all():
                    proposals.append(point)
                    slots.append(slot)
                    ids.append(identity)
        if proposals:
            T_SR = self._source_pose @ np.linalg.inv(calibration.T_RD)
            candidates = transform_points(np.array(proposals), T_SR)
            valid = self._valid_points(ids, candidates, calibration)
            points[np.asarray(slots)[valid]] = candidates[valid]
            self._stats["recent_depth_anchors"] = int(valid.sum())
        return points

    def _estimate_map(self, features, calibration, matches):
        result = super()._estimate_map(features, calibration, matches)
        if (
            result[0] is None
            and self.config.association_policy == "indexed"
            and not self._searched
        ):
            combined, _ = merge_proposals(matches + self._search(features))
            result = super()._estimate_map(features, calibration, combined)
            self._stats["recovery_matches"] = len(combined)
        return result

    def _insert(self, index, features, pose, matches, calibration, reserved=()):
        if self.keyframes and self.config.association_policy == "indexed":
            combined, blocked = merge_proposals(matches + self._search(features))
            if combined:
                _, indices = np.array(combined).T
                selected = self._consensus(
                    self._tracking_points(combined, calibration),
                    features.pixels[indices],
                    features.points_R[indices],
                    calibration.T_RD @ np.linalg.inv(pose),
                    calibration.K_R,
                )
                matches = [combined[i] for i in np.flatnonzero(selected)]
            else:
                matches = []
            reserved = set(reserved) | blocked
        updates = {}
        if matches:
            ids, indices = np.array(matches).T
            points = np.array([self.landmarks[int(i)].point_S for i in ids])
            measured = features.points_R[indices]
            T_RS = calibration.T_RD @ np.linalg.inv(pose)
            if self.config.anchor_policy == "recent":
                candidates = transform_points(measured, np.linalg.inv(T_RS))
                refresh = self._valid_points(ids, candidates, calibration)
                refresh &= self._consensus(
                    candidates,
                    features.pixels[indices],
                    measured,
                    T_RS,
                    calibration.K_R,
                )
                points[refresh] = candidates[refresh]
                updates = {
                    int(i): p
                    for i, p in zip(ids[refresh], candidates[refresh], strict=True)
                }
            # Inductive consistency includes the incoming observation. A failed
            # refresh never appends a view incompatible with the retained point.
            selected = self._consensus(
                points, features.pixels[indices], measured, T_RS, calibration.K_R
            )
            rejected = {int(indices[i]) for i in np.flatnonzero(~selected)}
            reserved = set(reserved) | rejected
            self._stats["geometry_rejected_associations"] = len(rejected)
            matches = [matches[i] for i in np.flatnonzero(selected)]
        super()._insert(index, features, pose, matches, calibration, reserved)
        for identity, point in updates.items():
            if identity in self.landmarks:
                self.landmarks[identity].point_S = point
        self._stats["reanchored_landmarks"] = len(updates)
        self._rebuild_index()

    def _accept_features(self, index, features, matches):
        if index in self.owners:
            self._previous_ids = self.owners[index].copy()
        else:
            self._previous_ids = np.full(len(features.pixels), -1, dtype=np.int64)
            for identity, feature in matches:
                self._previous_ids[feature] = identity

    def estimate(self, source, target, calibration):
        self._source_pose = self._pose.copy()
        result = super().estimate(source, target, calibration)
        result.diagnostics.update(self._stats)
        result.diagnostics["indexed_landmarks"] = len(self._index_ids)
        return result
