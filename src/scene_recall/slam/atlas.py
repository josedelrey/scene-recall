"""Bounded place memory and an explicitly disconnected metric submap forest.

Raw local tracking segments remain immutable identities. Verified cross-segment
camera transforms connect their gauges after separate view-pair confirmation.
The forest is rebuilt after local pose revisions. It performs no pose-graph
averaging and cannot correct drift around a loop. Geometry and future semantic
attachments can follow stable source-frame lineage and atlas revision versions.
"""

from dataclasses import dataclass, field

import numpy as np

from scene_recall.features import RGBDFeatures
from scene_recall.geometry.transforms import is_rigid
from scene_recall.odometry.evaluation import pose_error
from scene_recall.slam.registration import RegistrationConfig, register_anchor


@dataclass(frozen=True)
class AtlasConfig:
    max_entries: int = 64
    recent_entries: int = 16
    candidates: int = 4
    keyframe_stride: int = 5
    min_frame_gap: int = 10
    confirmation_query_frames: int = 0
    confirmation_translation_m: float = 0.05
    confirmation_rotation_deg: float = 3.0
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)

    def __post_init__(self):
        for name in (
            "max_entries",
            "recent_entries",
            "candidates",
            "keyframe_stride",
            "min_frame_gap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.recent_entries >= self.max_entries:
            raise ValueError("atlas needs both historical and recent capacity")
        if (
            isinstance(self.confirmation_query_frames, bool)
            or not isinstance(self.confirmation_query_frames, int)
            or self.confirmation_query_frames < 0
        ):
            raise ValueError("confirmation_query_frames must be a nonnegative integer")
        for name in ("confirmation_translation_m", "confirmation_rotation_deg"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.registration, RegistrationConfig):
            raise TypeError("atlas registration must be a RegistrationConfig")


def appearance_signature(features):
    if features.descriptors is None or not len(features.descriptors):
        return None
    descriptors = np.maximum(features.descriptors, 0)
    roots = np.sqrt(
        descriptors / np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)
    )
    signature = roots.mean(axis=0)
    return signature / max(float(np.linalg.norm(signature)), 1e-12)


class SubmapAtlas:
    """Serial reference-free overlay, separate from the local estimator state."""

    def __init__(self, calibration, config=None):
        self.config = config or AtlasConfig()
        self.calibration = calibration
        self.frames, self.entries, self.forest, self.pending = {}, {}, [], {}
        self.transforms, self.components = {}, {}
        self.version = 0
        self.events = []

    def _relation(self, source, target, T_target_source):
        a, b = self.frames[source], self.frames[target]
        return a[1] @ np.linalg.inv(T_target_source) @ np.linalg.inv(b[1])

    def _rebuild(self):
        segments = sorted({s for s, _ in self.frames.values()})
        adjacency = {s: [] for s in segments}
        for edge in self.forest:
            a, b = self.frames[edge["source"]][0], self.frames[edge["target"]][0]
            relative = self._relation(
                edge["source"], edge["target"], edge["T_target_source"]
            )
            adjacency[a].append((b, relative))
            adjacency[b].append((a, np.linalg.inv(relative)))
        transforms, components = {}, {}
        for root in segments:
            if root in transforms:
                continue
            transforms[root] = np.eye(4)
            components[root] = root
            stack = [root]
            while stack:
                a = stack.pop()
                for b, relative in adjacency[a]:
                    if b not in transforms:
                        transforms[b] = transforms[a] @ relative
                        components[b] = root
                        stack.append(b)
        changed = [
            s
            for s in segments
            if s in self.transforms
            and (
                self.components[s] != components[s]
                or not np.allclose(
                    self.transforms[s], transforms[s], atol=1e-12, rtol=0
                )
            )
        ]
        self.transforms, self.components = transforms, components
        if changed:
            self.version += 1
        return changed

    def revise(self, poses):
        """Update local camera poses without reusing optimized landmark coordinates."""
        for frame, pose in poses.items():
            if frame not in self.frames or not is_rigid(pose):
                raise ValueError("revision requires an existing rigid camera")
        dirty = {
            self.frames[frame][0]
            for frame, pose in poses.items()
            if not np.allclose(self.frames[frame][1], pose, atol=1e-12, rtol=0)
        }
        version = self.version
        for frame, pose in poses.items():
            self.frames[frame] = (self.frames[frame][0], pose.copy())
        affected = self._rebuild()
        if dirty and self.version == version:
            self.version += 1
        return sorted(dirty | set(affected))

    def propose(self, source, target, T_target_source, diagnostic):
        """Confirm a candidate from two distinct source and target observations."""
        if (
            source not in self.frames
            or target not in self.frames
            or source >= target
            or not is_rigid(T_target_source)
            or diagnostic.get("status") != "accepted"
        ):
            raise ValueError("link requires ordered known rigid camera observations")
        a, b = self.frames[source][0], self.frames[target][0]
        event = {
            "source": source,
            "target": target,
            "source_segment": a,
            "target_segment": b,
            "verification": diagnostic,
        }
        if a == b:
            return {**event, "status": "same_local_segment"}
        relation = self._relation(source, target, T_target_source)
        if self.components[a] == self.components[b]:
            expected = np.linalg.inv(self.transforms[a]) @ self.transforms[b]
            error = pose_error(relation, expected)
            return {
                **event,
                "status": "consistent_cycle"
                if error[0] <= self.config.confirmation_translation_m
                and error[1] <= self.config.confirmation_rotation_deg
                else "inconsistent_cycle",
                "cycle_translation_m": error[0],
                "cycle_rotation_deg": error[1],
            }
        key = (a, b)
        previous = self.pending.get(key)
        candidate = {
            "source": source,
            "target": target,
            "T_target_source": T_target_source.copy(),
            "diagnostic": diagnostic,
        }
        if previous is None:
            self.pending[key] = candidate
            return {**event, "status": "tentative"}
        if previous["source"] == source or previous["target"] == target:
            return {**event, "status": "needs_distinct_views"}
        old_relation = self._relation(
            previous["source"], previous["target"], previous["T_target_source"]
        )
        translation, rotation = pose_error(relation, old_relation)
        if (
            translation > self.config.confirmation_translation_m
            or rotation > self.config.confirmation_rotation_deg
        ):
            self.pending[key] = candidate
            return {
                **event,
                "status": "inconsistent_confirmation",
                "confirmation_translation_m": translation,
                "confirmation_rotation_deg": rotation,
            }
        selected = max(
            (previous, candidate), key=lambda c: c["diagnostic"].get("fit_inliers", 0)
        )
        self.forest.append(selected)
        del self.pending[key]
        changed = self._rebuild()
        return {
            **event,
            "status": "connected",
            "support_views": [
                [previous["source"], previous["target"]],
                [source, target],
            ],
            "affected_segments": changed,
            "version": self.version,
        }

    def revoke(self, source, target, reason):
        """Remove one camera constraint and invalidate every affected component.

        Revocation is explicit. New evidence can propose another relationship
        later, while stale pending hypotheses cannot immediately reconnect it.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("revocation requires a nonempty reason")
        indices = [
            i
            for i, edge in enumerate(self.forest)
            if edge["source"] == source and edge["target"] == target
        ]
        if len(indices) != 1:
            raise ValueError("revocation requires one active camera constraint")
        del self.forest[indices[0]]
        self.pending.clear()
        affected = self._rebuild()
        event = {
            "source": source,
            "target": target,
            "status": "revoked",
            "reason": reason,
            "affected_segments": affected,
            "version": self.version,
        }
        self.events.append(event)
        return event

    def observe(self, frame, segment, pose, features, *, boundary=False):
        if (
            frame in self.frames
            or (self.frames and frame <= max(self.frames))
            or not is_rigid(pose)
        ):
            raise ValueError("atlas observations must be increasing rigid cameras")
        self.frames[frame] = (segment, pose.copy())
        self._rebuild()
        signature = appearance_signature(features)
        queries = []
        regular_query = boundary or frame % self.config.keyframe_stride == 0
        confirming = {
            a
            for (a, b), candidate in self.pending.items()
            if b == segment
            and 0 < frame - candidate["target"] <= self.config.confirmation_query_frames
        }
        if signature is not None and (regular_query or confirming):
            candidates = [
                (
                    self.frames[key][0] not in confirming,
                    float(np.linalg.norm(signature - entry[1])),
                    key,
                )
                for key, entry in self.entries.items()
                if frame - key >= self.config.min_frame_gap
                and self.frames[key][0] != segment
            ]
            verified = []
            for _, _, source in sorted(candidates)[: self.config.candidates]:
                registered = register_anchor(
                    self.entries[source][0],
                    features,
                    self.calibration,
                    self.config.registration,
                )
                if registered.T_target_source is None:
                    queries.append(
                        {
                            "source": source,
                            "target": frame,
                            "status": "registration_rejected",
                            "verification": registered.diagnostics,
                        }
                    )
                    continue
                T_depth = (
                    np.linalg.inv(self.calibration.T_RD)
                    @ registered.T_target_source
                    @ self.calibration.T_RD
                )
                verified.append((source, T_depth, registered.diagnostics))
            roots = {
                self.components[self.frames[source][0]] for source, _, _ in verified
            }
            if len(roots) > 1:
                queries.extend(
                    {
                        "source": source,
                        "target": frame,
                        "status": "ambiguous_components",
                        "verification": diagnostic,
                    }
                    for source, _, diagnostic in verified
                )
            else:
                queries.extend(
                    self.propose(source, frame, transform, diagnostic)
                    for source, transform, diagnostic in verified
                )
            if (
                regular_query
                and np.isfinite(features.points_R).all(axis=1).sum()
                >= self.config.registration.min_pairs
            ):
                snapshot = RGBDFeatures(
                    features.pixels.copy(),
                    features.descriptors.copy(),
                    features.points_R.copy(),
                )
                for array in (snapshot.pixels, snapshot.descriptors, snapshot.points_R):
                    array.setflags(write=False)
                self.entries[frame] = (snapshot, signature)
                recent = sorted(self.entries)[-self.config.recent_entries :]
                history = sorted(
                    (k for k in self.entries if k not in recent),
                    key=lambda k: ((k * 2654435761) & 0xFFFFFFFF, k),
                )
                keep = set(recent + history[: self.config.max_entries - len(recent)])
                self.entries = {k: v for k, v in self.entries.items() if k in keep}
        # Pending hypotheses cannot grow with every discarded singleton map.
        live_segments = {self.frames[k][0] for k in self.entries} | {segment}
        self.pending = {
            key: value
            for key, value in self.pending.items()
            if all(s in live_segments for s in key)
        }
        self.events.extend(queries)
        return {
            "queries": queries,
            "entries": len(self.entries),
            "components": len(set(self.components.values())),
            "version": self.version,
        }

    def pose(self, frame):
        segment, local = self.frames[frame]
        return self.transforms[segment] @ local
