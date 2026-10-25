"""Persistent, replayable raw-observation camera and landmark estimation.

The pose-edge container supplies initialization and provenance only. Its
odometry and loop summaries never enter the estimation objective. Active loop
edges own reversible identity equivalences in the inherited native ledger.
"""

from dataclasses import dataclass, field

import numpy as np

from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.batch_ba import BatchConfig
from scene_recall.multiview.fixed_landmarks import DirectFactors
from scene_recall.slam.observation_welding import ObservationWelding
from scene_recall.slam.shared_linearization import (
    ObservationCache,
    RelinearizationConfig,
)
from scene_recall.slam.shared_solver import solve_shared


@dataclass(frozen=True)
class SharedConfig:
    solver: BatchConfig = field(default_factory=lambda: BatchConfig(max_iterations=8))
    rank_check_cameras: int = 24
    require_full_rank: bool = False
    linearization: str = "full"
    relinearization: RelinearizationConfig = field(
        default_factory=RelinearizationConfig
    )
    preconditioner: str = "jacobi"
    admission: str = "balanced"
    retry_linear_failure: bool = False

    def __post_init__(self):
        if not isinstance(self.solver, BatchConfig):
            raise TypeError("invalid shared solver configuration")
        if self.linearization not in ("full", "selective"):
            raise ValueError("unknown shared linearization policy")
        if not isinstance(self.relinearization, RelinearizationConfig):
            raise TypeError("invalid shared relinearization configuration")
        if self.preconditioner not in ("jacobi", "coarse"):
            raise ValueError("unknown shared preconditioner")
        if self.admission not in ("balanced", "covisibility"):
            raise ValueError("unknown shared admission policy")
        if not isinstance(self.retry_linear_failure, bool):
            raise TypeError("shared linear retry policy must be boolean")
        if not isinstance(self.require_full_rank, bool):
            raise TypeError("shared rank policy must be boolean")
        if (
            isinstance(self.rank_check_cameras, bool)
            or not isinstance(self.rank_check_cameras, int)
            or self.rank_check_cameras < 2
        ):
            raise ValueError("rank check camera limit must be an integer >= 2")


class SharedEstimator(ObservationWelding):
    """Retain nonlinear evidence and warm-start states across every update.

    Appending observations triggers one shared solve, including history. Identity
    edits rebuild the variable partition from immutable packets. The complete
    solver remains the reference. Optional endpoint derivative reuse and coarse
    preconditioning retain the same raw objective and full landmark coupling.
    There is no second local objective, imported prior, permanent marginalization
    or incrementally updated Bayes-tree factorization.
    """

    def __init__(self, calibration, config=None, shared=None):
        super().__init__(calibration, config)
        self.shared = shared or SharedConfig()
        self.estimates, self.members, self.raw_components = {}, {}, {}
        self.last_report = None
        self._partition = None
        self.live_identities = {}
        self.track_components = {}
        self.cache = None
        self.admission_report = {}

    def _admission_order(self, segment, measurement, known, new):
        if self.shared.admission == "balanced":
            return super()._admission_order(segment, measurement, known, new)
        # Prefer spatially spread links to the latest measured camera, then
        # other retained evidence. Reserve a quarter for future track support.
        previous = next(
            (
                self.views[k]
                for k in sorted(self.views, reverse=True)
                if self.views[k].segment == segment
            ),
            None,
        )
        tracks = set(previous.factors.track_id) if previous is not None else set()
        recent = [
            i
            for i in known
            if self.identity_map[segment, int(measurement["frontend_identity"][i])]
            in tracks
        ]
        remaining = [i for i in known if i not in set(recent)]
        existing = self._spatial_order(
            measurement["pixels"], recent
        ) + self._spatial_order(measurement["pixels"], remaining)
        fresh = self._spatial_order(measurement["pixels"], new)
        quota = 3 * self.config.max_observations_per_view // 4
        return [
            *existing[:quota],
            *fresh[: self.config.max_observations_per_view - quota],
            *existing[quota:],
            *fresh[self.config.max_observations_per_view - quota :],
        ]

    def admit(self, frame, segment, measurement):
        previous = max(
            (k for k in self.views if self.views[k].segment == segment), default=None
        )
        admitted = super().admit(frame, segment, measurement)
        if admitted:
            packet = self.views[frame].factors
            shared = (
                np.intersect1d(packet.track_id, self.views[previous].factors.track_id)
                if previous is not None
                else np.array([])
            )
            self.admission_report = {
                "frame": frame,
                "previous_camera": previous,
                "candidates": len(measurement["view"]),
                "selected": len(packet.view),
                "previous_shared_tracks": len(shared),
            }
        return admitted

    def revoke(self, source, target):
        super().revoke(source, target)
        # A union split can change every correlated variable. Rebuild its seeds
        # from raw observations instead of retaining a merged point or prior.
        self._partition = None

    def _topology(self, graph, identities):
        parents = {frame: frame for frame in self.views}

        def root(frame):
            while parents[frame] != frame:
                frame = parents[frame]
            return frame

        observers, native = {}, set()
        for frame, view in self.views.items():
            for track, sample in zip(
                view.factors.track_id, view.factors.native_points, strict=True
            ):
                identity = identities[int(track)]
                observers.setdefault(identity, []).append(frame)
                if np.isfinite(sample).all():
                    native.add(identity)
        for identity, frames in observers.items():
            if identity not in native:
                continue
            for frame in frames[1:]:
                a, b = root(frames[0]), root(frame)
                parents[max(a, b)] = min(a, b)
        raw = {frame: root(frame) for frame in parents}
        # A prefix with no usable exposure has no raw pose constraint. Attach it
        # to the first measured gauge in its tracking segment for publication.
        # Its relative placement remains explicitly initialization-dependent.
        labels, fallback = {}, {}
        for segment in set(graph.segments.values()):
            nodes = sorted(k for k in graph.local if graph.segments[k] == segment)
            measured = [k for k in nodes if k in raw]
            if measured:
                labels.setdefault(raw[measured[0]], nodes[0])
            anchor = None
            for frame in nodes:
                if frame in raw:
                    anchor = frame
                fallback[frame] = (
                    anchor
                    if anchor is not None
                    else (measured[0] if measured else None)
                )
        components = {}
        for frame in graph.local:
            anchor = fallback[frame]
            components[frame] = (
                labels.get(raw[anchor], raw[anchor])
                if anchor is not None
                else min(
                    k for k in graph.local if graph.segments[k] == graph.segments[frame]
                )
            )
        return raw, components, observers, native, fallback

    def _initialize_bridges(self, graph, active):
        """Align old gauges through new bridges while preserving solved shape."""
        placement = {key: pose.copy() for key, pose in graph.poses.items()}
        parents = {component: component for component in graph.components.values()}

        def root(component):
            while parents[component] != component:
                component = parents[component]
            return component

        for source, target in sorted(active):
            a, b = root(graph.components[source]), root(graph.components[target])
            if a == b:
                continue
            measurement = graph.edges[source, target].T_target_source
            if a < b:
                correction = (
                    placement[source]
                    @ np.linalg.inv(measurement)
                    @ np.linalg.inv(placement[target])
                )
            else:
                correction = (
                    placement[target] @ measurement @ np.linalg.inv(placement[source])
                )
            retire, keep = max(a, b), min(a, b)
            for key, pose in placement.items():
                if root(graph.components[key]) == retire:
                    placement[key] = correction @ pose
            parents[retire] = keep
        return placement

    def refine(self, graph, rejected=()):
        active = {
            key
            for key, edge in graph.edges.items()
            if edge.kind == "loop" and key in self.associations
        } - set(rejected)
        identities, accepted = self._identities(active)
        active = {
            edge for edge in active if accepted[edge] >= self.config.min_associations
        }
        identities, accepted = self._identities(active)
        raw, components, observers, native, fallback = self._topology(graph, identities)
        members = {}
        for track, identity in identities.items():
            members.setdefault(identity, []).append(track)
        members = {key: tuple(sorted(value)) for key, value in members.items()}
        partition = tuple(sorted((key, value) for key, value in identities.items()))
        # New independent identities do not invalidate existing estimates. A
        # merge, split or revocation requires a complete raw reconstruction.
        edited = (
            self._partition is None
            or any(
                self.members.get(key, value) != value for key, value in members.items()
            )
            or any(key not in members for key in self.members)
        )
        placement = self._initialize_bridges(graph, active)
        initial = {key: pose.copy() for key, pose in placement.items()}
        edited |= raw != {
            key: self.raw_components.get(key, value) for key, value in raw.items()
        }
        keys = sorted(self.views)
        report = {
            "method": "persistent_raw_schur",
            "cameras": len(keys),
            "retained_observations": sum(
                len(v.factors.view) for v in self.views.values()
            ),
            "pose_summary_factors": 0,
            "marginal_prior_factors": 0,
            "raw_rebuild": edited,
            "associations": {f"{a}:{b}": accepted[a, b] for a, b in sorted(active)},
            "components": len(set(raw.values())),
            "unmeasured_cameras": sorted(set(graph.local) - set(keys)),
            "admission": self.admission_report,
        }
        if not keys:
            graph.components = components
            report.update(status="unchanged", reason="no native factor packets")
            self.last_report = report
            return [report]
        selected = {
            identity
            for identity, frames in observers.items()
            if len(frames) >= 2 and identity in native
        }
        poses = np.array(
            [placement[k] @ np.linalg.inv(self.views[k].factors.T_RcD[0]) for k in keys]
        )
        seeds = {}
        for camera, frame in enumerate(keys):
            packet = self.views[frame].factors
            for track, point, exposure in zip(
                packet.track_id, packet.native_points, packet.T_RcD, strict=True
            ):
                identity = identities[int(track)]
                if (
                    identity in selected
                    and identity not in seeds
                    and np.isfinite(point).all()
                ):
                    seeds[identity] = transform_points(point, poses[camera] @ exposure)
        if not edited:
            seeds.update(
                {
                    key: value.copy()
                    for key, value in self.estimates.items()
                    if key in selected and self.members.get(key) == members[key]
                }
            )
        arrays = {name: [] for name in DirectFactors.__dataclass_fields__}
        for camera, frame in enumerate(keys):
            packet = self.views[frame].factors
            ids = np.array([identities[int(track)] for track in packet.track_id])
            mask = np.isin(ids, list(selected))
            if not mask.any():
                continue
            for name, parts in arrays.items():
                value = getattr(packet, name)[mask]
                if name == "view":
                    value = np.full(mask.sum(), camera, dtype=int)
                elif name == "track_id":
                    value = ids[mask]
                elif name == "points":
                    value = np.array([seeds[int(track)] for track in ids[mask]])
                parts.append(value)
        used_cameras = sorted({int(i) for part in arrays["view"] for i in part})
        if used_cameras:
            factors = DirectFactors(
                **{name: np.concatenate(parts) for name, parts in arrays.items()}
            )
            # Isolated singleton-only cameras contain no relative pose evidence.
            # Keep them fixed in their own measured gauge, without damping them
            # into apparent observability.
            fixed = sorted(
                {keys.index(root) for root in raw.values()}
                | (set(range(len(keys))) - set(used_cameras))
            )
            if self.shared.linearization == "selective" and (
                self.cache is None or edited
            ):
                self.cache = ObservationCache(self.shared.relinearization)
            revised, points, fit = solve_shared(
                poses,
                factors,
                self.calibration.K_R,
                fixed,
                self.shared.solver,
                rank_check_cameras=self.shared.rank_check_cameras,
                require_full_rank=self.shared.require_full_rank,
                cache=self.cache,
                preconditioner=self.shared.preconditioner,
                retry_linear_failure=self.shared.retry_linear_failure,
            )
            report.update(
                fit,
                observations=len(factors.view),
                landmarks=len(selected),
                fixed_cameras=[keys[i] for i in fixed],
            )
            report["validation"] = []
            for source, target in sorted(active):
                a, b = keys.index(source), keys.index(target)
                before = self._validation_score(source, target, poses[a], poses[b])
                after = self._validation_score(source, target, revised[a], revised[b])
                report["validation"].append(
                    {"edge": [source, target], "before": before, "after": after}
                )
            if fit["status"] in ("refined", "unchanged"):
                placement.update(
                    {
                        frame: pose @ self.views[frame].factors.T_RcD[0]
                        for frame, pose in zip(keys, revised, strict=True)
                    }
                )
                self.estimates = dict(zip(sorted(selected), points, strict=True))
            elif edited:
                self.estimates = {}
            if fit["status"] == "rejected":
                self.cache = None
        else:
            report.update(
                status="unchanged",
                reason="no shared native tracks",
                observations=0,
                landmarks=0,
            )
            self.estimates = {}
        # Unmeasured cameras are explicit rigid output attachments. Every camera
        # with a packet was jointly optimized or fixed as a component gauge.
        for frame, anchor in fallback.items():
            if frame not in self.views and anchor is not None:
                placement[frame] = (
                    placement[anchor] @ np.linalg.inv(initial[anchor]) @ initial[frame]
                )
        graph.poses, graph.components = placement, components
        graph.version += 1
        self.members, self.raw_components, self._partition = members, raw, partition
        self.live_identities = identities
        track_components = {}
        for frame, view in self.views.items():
            for track in view.factors.track_id:
                track_components.setdefault(int(track), set()).add(components[frame])
        self.track_components = track_components
        self.last_report = report
        return [report]

    def active_point_positions(self, graph, segment):
        anchor = max(
            (k for k in graph.local if graph.segments[k] == segment), default=None
        )
        identities = self.live_identities
        return {
            owner: self.estimates[identities[track]].copy()
            for (owner_segment, owner), track in self.identity_map.items()
            if owner_segment == segment
            and identities.get(track) in self.estimates
            and graph.components.get(anchor) in self.track_components.get(track, ())
        }

    def export_arrays(self, graph):
        """Save native evidence, live identity ownership and optimizer state."""
        arrays = {}
        if self.views:
            for name in DirectFactors.__dataclass_fields__:
                arrays[name] = np.concatenate(
                    [getattr(view.factors, name) for view in self.views.values()]
                )
            for name in ("feature_index", "native_index"):
                arrays[name] = np.concatenate(
                    [getattr(view, name) for view in self.views.values()]
                )
        arrays["identity_owners"] = np.array(
            [
                (segment, owner, track)
                for (segment, owner), track in sorted(self.identity_map.items())
            ],
            dtype=np.int64,
        ).reshape(-1, 3)
        arrays["live_identities"] = np.array(
            sorted(self.live_identities.items()), dtype=np.int64
        ).reshape(-1, 2)
        arrays["point_ids"] = np.array(sorted(self.estimates), dtype=np.int64)
        arrays["point_positions"] = np.array(
            [self.estimates[k] for k in sorted(self.estimates)]
        ).reshape(-1, 3)
        arrays["camera_ids"] = np.array(sorted(graph.poses), dtype=np.int64)
        arrays["depth_poses"] = np.array(
            [graph.poses[k] for k in sorted(graph.poses)]
        ).reshape(-1, 4, 4)
        arrays["raw_components"] = np.array(
            sorted(self.raw_components.items()), dtype=np.int64
        ).reshape(-1, 2)
        arrays["associations"] = np.array(
            [
                (*edge, a, b)
                for edge, pairs in sorted(self.associations.items())
                for a, b in pairs
            ],
            dtype=np.int64,
        ).reshape(-1, 4)
        return arrays
