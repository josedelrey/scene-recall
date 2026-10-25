"""Experimental observation-level welding of verified RGB-D loop neighborhoods.

The pose graph supplies topology and an initial placement. This module keeps one
copy of each native keyframe factor, builds reversible landmark equivalences from
active loop edges, and jointly revises co-visible camera and landmark variables.
The bounded solve is conditional on cameras outside its selected neighborhood.
"""

from dataclasses import dataclass, field

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.batch_ba import BatchConfig
from scene_recall.multiview.fixed_landmarks import DirectFactors
from scene_recall.multiview.tracking import match_descriptors
from scene_recall.slam.graph import solve_window


@dataclass(frozen=True)
class WeldingConfig:
    max_cameras: int = 8
    max_landmarks: int = 96
    max_observations_per_view: int = 128
    max_welds_per_solve: int = 2
    min_associations: int = 6
    solver: BatchConfig = field(
        default_factory=lambda: BatchConfig(
            max_iterations=8,
            max_translation_m=0.15,
            max_rotation_deg=5,
            max_point_displacement_m=0.15,
        )
    )

    def __post_init__(self):
        for name in (
            "max_cameras",
            "max_landmarks",
            "max_observations_per_view",
            "max_welds_per_solve",
            "min_associations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_cameras < 2:
            raise ValueError("welding requires two cameras")


@dataclass(frozen=True)
class RecordedView:
    factors: DirectFactors
    feature_index: np.ndarray
    native_index: np.ndarray
    segment: int


@dataclass(frozen=True)
class PointState:
    position: np.ndarray
    reference_frame: int
    reference_pose: np.ndarray
    members: tuple[int, ...]


@dataclass(frozen=True)
class LoopValidation:
    source_points: np.ndarray
    target_points: np.ndarray
    target_pixels: np.ndarray


class ObservationWelding:
    """Causal factor ledger with loop-owned, reversible identity equivalences."""

    def __init__(self, calibration, config=None):
        self.calibration = calibration
        self.config = config or WeldingConfig()
        self.views = {}
        self.associations = {}
        self.validation = {}
        self.points = {}
        self.identity_map = {}
        self._next_identity = 0

    def _spatial_order(self, pixels, indices):
        bins = {}
        shape = self.calibration.rgb_shape
        for i in indices:
            cell = tuple(
                np.floor(pixels[i] / [shape[1], shape[0]] * [8, 6]).astype(int)
            )
            bins.setdefault(cell, []).append(i)
        return [
            rows[j]
            for j in range(max(map(len, bins.values()), default=0))
            for _, rows in sorted(bins.items())
            if j < len(rows)
        ]

    def admit(self, frame, segment, measurement):
        if measurement is None:
            return False
        if frame in self.views:
            raise ValueError("keyframe factors already admitted")
        owners = measurement["frontend_identity"]
        native = measurement["native_index"]
        known = [
            i
            for i, owner in enumerate(owners)
            if (segment, int(owner)) in self.identity_map
        ]
        new = [
            i
            for i, owner in enumerate(owners)
            if (segment, int(owner)) not in self.identity_map and native[i] >= 0
        ]
        existing_order = self._spatial_order(measurement["pixels"], known)
        new_order = self._spatial_order(measurement["pixels"], new)
        half = self.config.max_observations_per_view // 2
        order = [
            *existing_order[:half],
            *new_order[:half],
            *existing_order[half:],
            *new_order[half:],
        ]
        selected, used_native = [], set()
        for i in order:
            sample = int(native[i])
            if sample >= 0 and sample in used_native:
                continue
            selected.append(i)
            if sample >= 0:
                used_native.add(sample)
            if len(selected) >= self.config.max_observations_per_view:
                break
        if not selected:
            return False
        identities = self.identity_map.copy()
        next_identity = self._next_identity
        for i in selected:
            owner = segment, int(owners[i])
            if owner not in identities:
                identities[owner] = next_identity
                next_identity += 1
        selected = np.array(selected, dtype=int)
        arrays = {
            name: measurement[name][selected].copy()
            for name in DirectFactors.__dataclass_fields__
        }
        arrays["track_id"] = np.array(
            [identities[segment, int(owners[i])] for i in selected],
            dtype=int,
        )
        packet = DirectFactors(**arrays)
        if not np.all(packet.view == frame):
            raise ValueError("factor packet belongs to another camera")
        feature_index = measurement["feature_index"][selected].copy()
        native_index = native[selected].copy()
        if len(np.unique(feature_index)) != len(feature_index):
            raise ValueError("feature observation has multiple owners")
        native = native_index[native_index >= 0]
        if len(np.unique(native)) != len(native):
            raise ValueError("native depth sample has multiple owners")
        for array in (*arrays.values(), feature_index, native_index):
            array.setflags(write=False)
        self.identity_map = identities
        self._next_identity = next_identity
        self.views[frame] = RecordedView(packet, feature_index, native_index, segment)
        return True

    def propose(self, source, target, first, second, T_target_source, ratio):
        """Record only fitting correspondences from an already verified loop."""
        if source not in self.views or target not in self.views:
            return {"status": "unavailable", "associations": 0}
        pairs, _ = match_descriptors(first.descriptors, second.descriptors, ratio)
        pairs = pairs[np.isfinite(first.points_R[pairs[:, 0]]).all(axis=1)]
        rows_a = {
            int(feature): i
            for i, feature in enumerate(self.views[source].feature_index)
        }
        rows_b = {
            int(feature): i
            for i, feature in enumerate(self.views[target].feature_index)
        }
        matched = []
        for a, b in pairs[::2]:
            a, b = int(a), int(b)
            if a not in rows_a or b not in rows_b:
                continue
            x, y = first.points_R[a], second.points_R[b]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue
            predicted = transform_points(x, T_target_source)
            if predicted[2] <= 0 or np.linalg.norm(predicted - y) > 0.05:
                continue
            if (
                np.linalg.norm(
                    project_points(predicted, self.calibration.K_R) - second.pixels[b]
                )
                > 3
            ):
                continue
            matched.append(
                (
                    int(self.views[source].factors.track_id[rows_a[a]]),
                    int(self.views[target].factors.track_id[rows_b[b]]),
                )
            )
        if len(matched) < self.config.min_associations:
            return {"status": "insufficient", "associations": len(matched)}
        self.associations[source, target] = tuple(matched)
        heldout = pairs[1::2]
        validation = LoopValidation(
            first.points_R[heldout[:, 0]].copy(),
            second.points_R[heldout[:, 1]].copy(),
            second.pixels[heldout[:, 1]].copy(),
        )
        for array in (
            validation.source_points,
            validation.target_points,
            validation.target_pixels,
        ):
            array.setflags(write=False)
        self.validation[source, target] = validation
        return {"status": "admitted", "associations": len(matched)}

    def _validation_score(self, source, target, source_pose, target_pose):
        evidence = self.validation[source, target]
        if not len(evidence.source_points):
            return None
        relative = np.linalg.inv(target_pose) @ source_pose
        prediction = transform_points(evidence.source_points, relative)
        positive = prediction[:, 2] > 0.05
        pixel_error = np.full(len(prediction), np.inf)
        pixel_error[positive] = np.linalg.norm(
            project_points(prediction[positive], self.calibration.K_R)
            - evidence.target_pixels[positive],
            axis=1,
        )
        has_depth = np.isfinite(evidence.target_points).all(axis=1)
        depth_error = np.zeros(len(prediction))
        depth_error[has_depth] = np.linalg.norm(
            prediction[has_depth] - evidence.target_points[has_depth], axis=1
        )
        threshold = 0.03 + 0.01 * np.where(has_depth, evidence.target_points[:, 2], 0)
        valid = positive & (pixel_error <= 3)
        valid &= ~has_depth | (depth_error <= threshold)
        scaled = np.minimum(pixel_error / 3, 10)
        scaled += np.where(has_depth, np.minimum(depth_error / threshold, 10), 0)
        return {
            "count": len(prediction),
            "inlier_fraction": float(valid.mean()),
            "median_scaled_error": float(np.median(scaled)),
        }

    def _identities(self, active):
        parent = {}
        frames = {}
        for key, view in self.views.items():
            for track in view.factors.track_id:
                track = int(track)
                parent.setdefault(track, track)
                frames.setdefault(track, set()).add(key)

        def root(key):
            while parent[key] != key:
                key = parent[key]
            return key

        accepted = {}
        for edge in sorted(active):
            pairs = self.associations.get(edge, ())
            count = 0
            for a, b in pairs:
                if a not in parent or b not in parent:
                    continue
                x, y = root(a), root(b)
                if x == y:
                    count += 1
                    continue
                if frames[x] & frames[y]:
                    continue
                keep, retire = min(x, y), max(x, y)
                parent[retire] = keep
                frames[keep].update(frames[retire])
                count += 1
            accepted[edge] = count
        return {key: root(key) for key in parent}, accepted

    def _window(self, source, target, graph, identities):
        component = graph.components[source]
        available = [
            frame
            for frame in self.views
            if frame in graph.poses and graph.components[frame] == component
        ]
        if source not in available or target not in available:
            return []
        tracks = {
            frame: set(map(int, self.views[frame].factors.track_id))
            for frame in available
        }
        anchors = tracks[source] | tracks[target]
        ranked = sorted(
            (frame for frame in available if frame not in (source, target)),
            key=lambda frame: (
                -len(
                    {identities[i] for i in tracks[frame]}
                    & {identities[i] for i in anchors}
                ),
                min(abs(frame - source), abs(frame - target)),
                frame,
            ),
        )
        return [source, target, *ranked[: self.config.max_cameras - 2]]

    def _factors(self, keys, graph, identities, members):
        counts, native = {}, set()
        for frame in keys:
            packet = self.views[frame].factors
            for track, sample in zip(
                packet.track_id, packet.native_points, strict=True
            ):
                identity = identities[int(track)]
                counts.setdefault(identity, set()).add(frame)
                if np.isfinite(sample).all():
                    native.add(identity)
        anchor_segments = {self.views[keys[0]].segment, self.views[keys[1]].segment}
        eligible = [
            identity
            for identity, cameras in counts.items()
            if len(cameras) >= 2 and identity in native
        ]
        eligible.sort(
            key=lambda identity: (
                -len(
                    {self.views[frame].segment for frame in counts[identity]}
                    & anchor_segments
                ),
                -len(counts[identity]),
                identity,
            )
        )
        selected = set(eligible[: self.config.max_landmarks])
        if not selected:
            return None, ()
        seeds = {}
        for frame in keys:
            packet = self.views[frame].factors
            rgb_pose = graph.poses[frame] @ np.linalg.inv(packet.T_RcD[0])
            for track, sample, transform in zip(
                packet.track_id, packet.native_points, packet.T_RcD, strict=True
            ):
                identity = identities[int(track)]
                if (
                    identity in selected
                    and identity not in seeds
                    and np.isfinite(sample).all()
                ):
                    seeds[identity] = transform_points(sample, rgb_pose @ transform)
        for identity in selected:
            state = self.points.get(identity)
            if (
                state is not None
                and state.members == members[identity]
                and state.reference_frame in graph.poses
            ):
                correction = graph.poses[state.reference_frame] @ np.linalg.inv(
                    state.reference_pose
                )
                seeds[identity] = transform_points(state.position, correction)
        arrays = {name: [] for name in DirectFactors.__dataclass_fields__}
        for camera, frame in enumerate(keys):
            packet = self.views[frame].factors
            mask = np.array([identities[int(i)] in selected for i in packet.track_id])
            if not mask.any():
                continue
            selected_ids = [identities[int(i)] for i in packet.track_id[mask]]
            for name, parts in arrays.items():
                value = getattr(packet, name)[mask]
                if name == "view":
                    value = np.full(len(value), camera, dtype=int)
                elif name == "track_id":
                    value = np.array(selected_ids, dtype=int)
                elif name == "points":
                    value = np.array([seeds[int(i)] for i in selected_ids])
                parts.append(value)
        return DirectFactors(
            **{name: np.concatenate(parts) for name, parts in arrays.items()}
        ), tuple(sorted(selected))

    def refine(self, graph, rejected=()):
        """Apply bounded raw-factor fits after topology has a current placement."""
        active = {
            key
            for key, edge in graph.edges.items()
            if edge.kind == "loop" and key in self.associations and key not in rejected
        }
        identities, accepted = self._identities(active)
        members = {}
        for track, identity in identities.items():
            members.setdefault(identity, []).append(track)
        members = {key: tuple(sorted(value)) for key, value in members.items()}
        reports = []
        for source, target in sorted(active, key=lambda edge: (edge[1], edge[0]))[
            -self.config.max_welds_per_solve :
        ]:
            keys = self._window(source, target, graph, identities)
            if len(keys) < 2 or accepted[source, target] < self.config.min_associations:
                reports.append(
                    {"edge": [source, target], "status": "insufficient support"}
                )
                continue
            factors, tracks = self._factors(keys, graph, identities, members)
            if factors is None:
                reports.append(
                    {"edge": [source, target], "status": "no shared factors"}
                )
                continue
            poses = np.array(
                [
                    graph.poses[frame]
                    @ np.linalg.inv(self.views[frame].factors.T_RcD[0])
                    for frame in keys
                ]
            )
            revised, points, diagnostic = solve_window(
                poses,
                factors,
                self.calibration.K_R,
                keys,
                source,
                config=self.config.solver,
            )
            report = {
                "edge": [source, target],
                "cameras": len(keys),
                "landmarks": len(tracks),
                "observations": len(factors.view),
                "status": diagnostic["status"],
                "reason": diagnostic["reason"],
            }
            before = self._validation_score(
                source, target, poses[keys.index(source)], poses[keys.index(target)]
            )
            after = self._validation_score(
                source, target, revised[keys.index(source)], revised[keys.index(target)]
            )
            report["validation_before"] = before
            report["validation_after"] = after
            if (
                diagnostic["status"] == "refined"
                and before is not None
                and (
                    after["median_scaled_error"] > before["median_scaled_error"] + 0.1
                    or after["inlier_fraction"]
                    < max(0.5, before["inlier_fraction"] - 0.05)
                )
            ):
                report["status"] = "validation_rejected"
            if diagnostic["status"] == "refined":
                if report["status"] == "validation_rejected":
                    reports.append(report)
                    continue
                corrections = {}
                for frame, pose in zip(keys, revised, strict=True):
                    old = graph.poses[frame]
                    new = pose @ self.views[frame].factors.T_RcD[0]
                    corrections[frame] = new @ np.linalg.inv(old)
                updated_poses = {}
                for frame in graph.poses:
                    if graph.components[frame] != graph.components[source]:
                        continue
                    anchors = [
                        key
                        for key in keys
                        if graph.segments[key] == graph.segments[frame]
                    ]
                    if anchors:
                        nearest = min(anchors, key=lambda key: (abs(key - frame), key))
                        updated_poses[frame] = corrections[nearest] @ graph.poses[frame]
                updated_points = {}
                for track, point in zip(tracks, points, strict=True):
                    updated_points[track] = PointState(
                        point.copy(),
                        source,
                        updated_poses[source].copy(),
                        members[track],
                    )
                graph.poses.update(updated_poses)
                self.points.update(updated_points)
                graph.version += 1
            reports.append(report)
        return reports
