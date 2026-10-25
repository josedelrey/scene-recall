"""Causal orchestration of local estimation, place memory and reconstruction.

The active estimator owns a stable local gauge. Global placement is a revocable
map-to-local transform, never an extra independent local measurement. Each
process call publishes a coherent trajectory and reconstruction version using
only observations received so far. There is no end-of-sequence optimization.
"""

from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np

from scene_recall.features import RGBDFeatures
from scene_recall.geometry.camera import transform_points
from scene_recall.geometry.transforms import is_rigid
from scene_recall.slam.atlas import AtlasConfig, appearance_signature
from scene_recall.slam.loop_confirmation import LoopConfirmation
from scene_recall.slam.multiview_registration import register_multiview
from scene_recall.slam.observation_welding import ObservationWelding, WeldingConfig
from scene_recall.slam.pose_graph import MetricPoseGraph, PoseGraphConfig
from scene_recall.slam.reconstruction import RevisionReconstruction
from scene_recall.slam.recovery import RecoveryFrontend
from scene_recall.slam.registration import register_anchor
from scene_recall.slam.retrieval import descriptor_votes
from scene_recall.slam.system import IncrementalSLAM, RevisionArchive, SLAMConfig


@dataclass(frozen=True)
class OnlineConfig:
    local: SLAMConfig = field(default_factory=lambda: SLAMConfig(optimize=False))
    atlas: AtlasConfig = field(default_factory=AtlasConfig)
    graph: PoseGraphConfig = field(
        default_factory=lambda: PoseGraphConfig(linear_solver="block_pcg")
    )
    retrieval: str = "shortlist_votes"
    shortlist: int = 12
    same_segment_gap: int = 30
    solve_interval: int = 10
    recovery_horizon: int = 15
    recovery_candidates: int = 3
    recovery: bool = True
    loops: bool = True
    contrast: str = "raw"
    flow_recovery: bool = False
    feature_scale: float = 1.0
    metric_refinement: bool = False
    incremental_mapping: bool = False
    multiview_registration: bool = False
    multiview_scope: str = "all"
    observation_welding: bool = False
    welding_feedback: str = "off"
    welding: WeldingConfig = field(default_factory=WeldingConfig)

    def __post_init__(self):
        if self.multiview_scope not in ("all", "loops"):
            raise ValueError("multiview scope must be all or loops")
        if self.feature_scale not in (1.0, 0.5) or isinstance(self.feature_scale, bool):
            raise ValueError("feature scale must be 1.0 or 0.5")
        if self.contrast not in ("raw", "clahe"):
            raise ValueError("unsupported online contrast mode")
        if self.retrieval not in ("mean", "shortlist_votes", "exhaustive_votes"):
            raise ValueError("unsupported online retriever")
        for name in (
            "shortlist",
            "same_segment_gap",
            "solve_interval",
            "recovery_horizon",
            "recovery_candidates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "recovery",
            "loops",
            "flow_recovery",
            "metric_refinement",
            "incremental_mapping",
            "multiview_registration",
            "observation_welding",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if not isinstance(self.local, SLAMConfig) or not isinstance(
            self.atlas, AtlasConfig
        ):
            raise TypeError("invalid online local or atlas configuration")
        if not isinstance(self.graph, PoseGraphConfig):
            raise TypeError("invalid online graph configuration")
        if not isinstance(self.welding, WeldingConfig):
            raise TypeError("invalid observation welding configuration")
        if self.welding_feedback not in ("off", "camera", "landmark", "joint"):
            raise ValueError("unsupported welding feedback mode")
        if self.observation_welding and self.local.optimize:
            raise ValueError(
                "welding requires raw local factors without a second local prior"
            )
        if self.welding_feedback != "off" and not self.observation_welding:
            raise ValueError("welding feedback requires observation welding")
        if self.shortlist < self.atlas.candidates:
            raise ValueError("shortlist must cover the verification budget")


@dataclass(frozen=True)
class OnlineUpdate:
    frame: int
    segment: int
    status: str
    local_pose: np.ndarray
    pose: np.ndarray
    component: int
    T_component_local: np.ndarray
    version: int
    dirty_frames: tuple
    diagnostics: dict


class OnlineSLAM:
    """Serial streaming SLAM with explicit scientific and capacity boundaries.

    Local failure remains a failure even if historical constraints subsequently
    reconnect its map. The dense reference graph is replaced by optional sparse
    linear solves, with scheduled nonlinear solves and a hard node budget. At
    capacity, tracking and attached placement continue, while new global nodes
    and place queries stop and report degraded operation.
    """

    def __init__(
        self, calibration, config=None, *, mapping=None, feature_extractor=None
    ):
        self.calibration = calibration
        self.config = config or OnlineConfig()
        self.registration = replace(
            self.config.atlas.registration,
            metric_refinement=(
                self.config.metric_refinement
                or self.config.atlas.registration.metric_refinement
            ),
        )
        if self.config.multiview_registration and self.registration.metric_refinement:
            raise ValueError("select one registration refinement")
        self.local = IncrementalSLAM(
            replace(self.config.local, raw_ledger=self.config.observation_welding)
        )
        frontend = RecoveryFrontend(
            self.local.frontend.config,
            horizon=self.config.recovery_horizon,
            candidates=self.config.recovery_candidates,
            contrast=self.config.contrast,
            flow=self.config.flow_recovery,
            feature_scale=self.config.feature_scale,
            registration=self.registration,
            multiview=(
                self.config.multiview_registration
                and self.config.multiview_scope == "all"
            ),
            feature_extractor=feature_extractor,
        )
        frontend.recovery_enabled = self.config.recovery
        self.local.frontend = frontend
        self.archive = RevisionArchive()
        self.graph = MetricPoseGraph(self.config.graph)
        self.confirmation = LoopConfirmation()
        self.reconstruction = (
            RevisionReconstruction(
                calibration, mapping, incremental=self.config.incremental_mapping
            )
            if mapping is not None
            else None
        )
        self.bank, self.segments, self.statuses, self.anchors = {}, [], [], []
        self.segment_nodes, self.segment_roots = {}, {}
        self.poses, self.components = [], []
        self.version = 0
        self.pending_solve, self.last_solve = False, -self.config.solve_interval
        self.events = []
        self.conditioning = {}
        self.welding = (
            ObservationWelding(calibration, self.config.welding)
            if self.config.observation_welding
            else None
        )
        self._feedback_owners = set()

    def _relative_context(self, source, context):
        return (
            self.calibration.T_RD
            @ np.linalg.inv(self.archive.poses[context])
            @ self.archive.poses[source]
            @ np.linalg.inv(self.calibration.T_RD)
        )

    def _invalidate_conditioning(self):
        """Remove derived constraints whose historical geometry has changed.

        Native pair registrations remain immutable. Conditional registrations
        require raw re-verification after a local revision. Retaining a stale
        derived transform would hide its dependence on the revised estimator.
        """
        events, revoked = [], False
        for (source, target), (context, frozen) in list(self.conditioning.items()):
            current = self._relative_context(source, context)
            if np.allclose(current, frozen, atol=1e-12, rtol=0):
                continue
            if (source, target) in self.graph.edges:
                self.graph.revoke(source, target, "historical conditioning revised")
                if self.welding is not None:
                    self.welding.revoke(source, target)
                revoked = True
            del self.conditioning[source, target]
            events.append(
                {
                    "status": "conditioning_invalidated",
                    "source": source,
                    "target": target,
                    "context": context,
                    "verification": {
                        "method": "conditioning_dependency",
                        "status": "invalidated",
                    },
                }
            )
        if events:
            self.confirmation = LoopConfirmation()
        return events, revoked

    def _node(self, frame, segment, pose):
        if len(self.graph.local) >= self.config.graph.max_nodes:
            return False
        previous = self.segment_nodes.get(segment)
        if previous is not None and self.config.welding_feedback != "off":
            measurement = np.linalg.inv(pose) @ self.archive.poses[previous]
            graph_local = self.graph.local[previous] @ np.linalg.inv(measurement)
        else:
            graph_local = pose
            measurement = (
                np.linalg.inv(pose) @ self.graph.local[previous]
                if previous is not None
                else None
            )
        self.graph.add_node(frame, segment, graph_local)
        if previous is not None:
            self.graph.add_edge(previous, frame, measurement)
            correction = self.graph.poses[previous] @ np.linalg.inv(
                self.graph.local[previous]
            )
            self.graph.poses[frame] = correction @ graph_local
            self.graph.components[frame] = self.graph.components[previous]
        else:
            self.graph.poses[frame] = pose.copy()
            self.graph.components[frame] = frame
        self.segment_nodes[segment] = frame
        return True

    def _refresh_confirmation(self):
        # Solved connectivity is authoritative, including rejected/revoked edges.
        self.confirmation.parents = {}
        roots = {}
        for frame, segment in self.graph.segments.items():
            component = self.graph.components[frame]
            roots[component] = min(roots.get(component, segment), segment)
        for frame, segment in self.graph.segments.items():
            self.confirmation.parents[segment] = roots[self.graph.components[frame]]
        for collection in (self.confirmation.pending, self.confirmation.confirmed):
            for group, (candidate, _) in list(collection.items()):
                source, target, transform, _ = candidate
                relation = (
                    self.archive.poses[source]
                    @ np.linalg.inv(transform)
                    @ np.linalg.inv(self.archive.poses[target])
                )
                collection[group] = candidate, relation

    def _recognize(self, frame, segment, features):
        signature = appearance_signature(features)
        if signature is None:
            return []
        eligible = [
            key
            for key in self.bank
            if frame - key
            >= (
                self.config.same_segment_gap
                if self.graph.segments[key] == segment
                else self.config.atlas.min_frame_gap
            )
        ]
        distances = {
            key: float(np.linalg.norm(signature - self.bank[key][1]))
            for key in eligible
        }
        ranked = sorted(eligible, key=lambda key: (distances[key], key))
        if self.config.retrieval != "mean":
            if self.config.retrieval == "shortlist_votes":
                ranked = ranked[: self.config.shortlist]
            votes = {
                key: descriptor_votes(
                    self.bank[key][0], features, self.config.atlas.registration.ratio
                )
                for key in ranked
            }
            ranked.sort(
                key=lambda key: (-votes[key][0], -votes[key][1], distances[key], key)
            )
        candidates, events = [], []
        observations = {}
        for source in ranked[: self.config.atlas.candidates]:
            result = register_anchor(
                self.bank[source][0],
                features,
                self.calibration,
                self.registration,
            )
            if (
                self.config.multiview_registration
                and result.T_target_source is not None
            ):
                contexts = []
                for key in sorted(self.bank, key=lambda key: (abs(key - source), key)):
                    if (
                        key == source
                        or self.graph.segments[key] != self.graph.segments[source]
                    ):
                        continue
                    relative = self._relative_context(source, key)
                    if np.linalg.norm(relative[:3, 3]) < 0.02:
                        continue
                    contexts.append((key, self.bank[key][0], relative))
                    if len(contexts) == 3:
                        break
                result = register_multiview(
                    self.bank[source][0],
                    features,
                    contexts,
                    self.calibration,
                    self.registration,
                    initial=result,
                )
            event = {
                "source": source,
                "target": frame,
                "verification": result.diagnostics,
            }
            if result.T_target_source is None:
                events.append({**event, "status": "registration_rejected"})
                continue
            if result.diagnostics.get("method") == "conditional_three_view_bundle":
                context = result.diagnostics["context"]
                frozen = self._relative_context(source, context)
                self.conditioning[source, frame] = context, frozen.copy()
                result.diagnostics["conditioning_transform"] = frozen.tolist()
            transform = (
                np.linalg.inv(self.calibration.T_RD)
                @ result.T_target_source
                @ self.calibration.T_RD
            )
            candidates.append(
                (
                    source,
                    frame,
                    self.graph.segments[source],
                    segment,
                    self.archive.poses[source],
                    self.archive.poses[frame],
                    transform,
                )
            )
            observations[source, frame] = result.T_target_source
            events.append(event)
        self._refresh_confirmation()
        statuses, added = self.confirmation.propose_batch(candidates)
        status_iter = iter(statuses)
        for event in events:
            if "status" not in event:
                event["status"] = next(status_iter)
        for source, target, measurement, group in added:
            if (source, target) not in self.graph.edges:
                self.graph.add_edge(
                    source, target, measurement, kind="loop", group=group
                )
                if self.welding is not None and (source, target) in observations:
                    welded = self.welding.propose(
                        source,
                        target,
                        self.bank[source][0],
                        features,
                        observations[source, target],
                        self.registration.ratio,
                    )
                    events.append(
                        {
                            "source": source,
                            "target": target,
                            "status": "observation_welding",
                            "welding": welded,
                        }
                    )
                self.pending_solve = True
        return events

    def _remember(self, frame, features):
        signature = appearance_signature(features)
        if (
            signature is None
            or np.isfinite(features.points_R).all(axis=1).sum()
            < self.config.atlas.registration.min_pairs
        ):
            return
        snapshot = RGBDFeatures(
            features.pixels.copy(),
            features.descriptors.copy(),
            features.points_R.copy(),
            features.descriptor_family,
        )
        for array in (snapshot.pixels, snapshot.descriptors, snapshot.points_R):
            array.setflags(write=False)
        self.bank[frame] = snapshot, signature
        recent = sorted(self.bank)[-self.config.atlas.recent_entries :]
        history = sorted(
            (key for key in self.bank if key not in recent),
            key=lambda key: ((key * 2654435761) & 0xFFFFFFFF, key),
        )
        keep = set(recent + history[: self.config.atlas.max_entries - len(recent)])
        self.bank = {key: value for key, value in self.bank.items() if key in keep}
        # Confirmed group state is bounded by graph nodes. Pending hypotheses need
        # a retained source and cannot accumulate across discarded singleton maps.
        self.confirmation.pending = {
            group: value
            for group, value in self.confirmation.pending.items()
            if value[0][0] in self.bank
        }
        retained = set(self.graph.edges)
        for collection in (self.confirmation.pending, self.confirmation.confirmed):
            retained.update(value[0][:2] for value in collection.values())
        self.conditioning = {
            key: value for key, value in self.conditioning.items() if key in retained
        }

    def _solve(self):
        diagnostic = self.graph.solve()
        if self.welding is not None:
            diagnostic["welding"] = self.welding.refine(self.graph)
        if self.config.welding_feedback != "off":
            diagnostic["feedback"] = self._feedback()
        self.pending_solve = False
        if diagnostic["rejected_loops"]:
            # A rejected bridge must not authorize later ambiguous confirmations.
            self.confirmation = LoopConfirmation()
        self._refresh_confirmation()
        return diagnostic

    def _feedback(self):
        """Apply accepted global geometry to the current tracking segment."""
        segment = self.local.segment_id
        anchor = self.segment_nodes.get(segment)
        if anchor is None or anchor not in self.graph.poses:
            return {"status": "unavailable"}
        chart = self.graph.poses[anchor] @ np.linalg.inv(self.archive.poses[anchor])
        inverse = np.linalg.inv(chart)
        cameras = (
            {
                key: inverse @ self.graph.poses[key]
                for key in self.local.frontend.keyframes
                if key in self.graph.poses and self.graph.segments[key] == segment
            }
            if self.config.welding_feedback in ("camera", "joint")
            else {}
        )
        global_points = (
            self.welding.active_point_positions(self.graph, segment)
            if self.config.welding_feedback in ("landmark", "joint")
            else {}
        )
        points = {
            owner: transform_points(point, inverse)
            for owner, point in global_points.items()
            if owner in self.local.frontend.landmarks
        }
        removed = self._feedback_owners - points.keys()
        result = self.local.apply_global_feedback(
            cameras, points, removed, self.calibration
        )
        self._feedback_owners = set(points)
        result["status"] = "applied"
        result["local_prior_replayed"] = False
        self.archive.apply(cameras)
        current = self.local.previous.frame_index
        self.local.synchronize_current(self.archive.poses[current])
        return result

    def _publish(self):
        poses, components, dirty = [], [], []
        for frame, (local, anchor) in enumerate(
            zip(self.archive.poses, self.anchors, strict=True)
        ):
            if anchor is None:
                pose, component = local.copy(), self.segment_roots[self.segments[frame]]
            else:
                pose = (
                    self.graph.poses[anchor]
                    @ np.linalg.inv(
                        self.archive.poses[anchor]
                        if self.config.welding_feedback != "off"
                        else self.graph.local[anchor]
                    )
                    @ local
                )
                component = self.graph.components[anchor]
            if (
                frame >= len(self.poses)
                or self.components[frame] != component
                or not np.allclose(self.poses[frame], pose, atol=1e-12, rtol=0)
            ):
                dirty.append(frame)
            else:
                # The invalidation tolerance also defines publication. Retaining
                # the last bitwise pose prevents unannounced roundoff changes
                # from making an unchanged chunk differ from fresh fusion.
                pose = self.poses[frame]
            poses.append(pose)
            components.append(component)
        if dirty:
            self.version += 1
        self.poses, self.components = poses, components
        return dirty

    def process(self, observation):
        frame = observation.frame_index
        if frame != len(self.segments):
            raise ValueError("online inputs must be consecutive from zero")
        started = perf_counter()
        pose, pair, revisions, diagnostic = self.local.process(
            observation, self.calibration
        )
        segment = self.local.segment_id
        status = (
            "initialized"
            if frame == 0
            else "lost"
            if pair.T_C1C0 is None
            else "tracked"
        )
        boundary = status != "tracked"
        self.segment_roots.setdefault(segment, frame)
        self.archive.append(
            pose, keyframe=frame in self.local.frontend.keyframes, boundary=boundary
        )
        local_dirty = self.archive.apply(revisions)
        self.segments.append(segment)
        self.statuses.append(status)
        revised_nodes = {
            key: self.archive.poses[key]
            for key in local_dirty
            if key in self.graph.local
        }
        if revised_nodes:
            corrections = {
                key: self.graph.poses[key] @ np.linalg.inv(self.graph.local[key])
                for key in revised_nodes
            }
            self.graph.revise(revised_nodes)
            for key, local in revised_nodes.items():
                self.graph.poses[key] = corrections[key] @ local
            self.pending_solve |= any(
                edge.kind == "loop" for edge in self.graph.edges.values()
            )
        invalidations, force_solve = self._invalidate_conditioning()
        node = boundary or frame % self.config.atlas.keyframe_stride == 0
        admitted = node and self._node(frame, segment, self.archive.poses[frame])
        if admitted and self.welding is not None:
            self.welding.admit(frame, segment, self.local.last_raw_measurement)
        self.anchors.append(self.segment_nodes.get(segment))
        features = self.local.frontend._previous[1]
        events = invalidations
        query_start = perf_counter()
        if admitted:
            if self.config.loops:
                events.extend(self._recognize(frame, segment, features))
            self._remember(frame, features)
        diagnostic["place_seconds"] = perf_counter() - query_start
        diagnostic["graph"] = None
        if force_solve or (
            self.pending_solve and frame - self.last_solve >= self.config.solve_interval
        ):
            solve_start = perf_counter()
            diagnostic["graph"] = self._solve()
            diagnostic["graph"]["seconds"] = perf_counter() - solve_start
            self.last_solve = frame
        self.events.extend(events)
        dirty = self._publish()
        diagnostic["mapping"] = None
        if self.reconstruction is not None:
            mapping_start = perf_counter()
            self.reconstruction.admit(observation, segment)
            rebuilt = self.reconstruction.synchronize(
                self.poses, self.components, self.version, dirty
            )
            diagnostic["mapping"] = {
                "rebuilt_chunks": rebuilt,
                "seconds": perf_counter() - mapping_start,
            }
        diagnostic.update(
            queries=events,
            bank_entries=len(self.bank),
            nodes=len(self.graph.local),
            global_capacity_reached=len(self.graph.local)
            >= self.config.graph.max_nodes,
            pending_solve=self.pending_solve,
            total_seconds=perf_counter() - started,
        )
        local = self.archive.poses[frame].copy()
        placed = self.poses[frame].copy()
        return OnlineUpdate(
            frame,
            segment,
            status,
            local,
            placed,
            self.components[frame],
            placed @ np.linalg.inv(local),
            self.version,
            tuple(dirty),
            diagnostic,
        )

    def revoke(self, source, target, reason):
        """Revoke a raw loop and publish disconnection before accepting more data."""
        self.graph.revoke(source, target, reason)
        if self.welding is not None:
            self.welding.revoke(source, target)
        self.conditioning.pop((source, target), None)
        self.confirmation = LoopConfirmation()
        diagnostic = self._solve()
        dirty = self._publish()
        if self.reconstruction is not None:
            self.reconstruction.synchronize(
                self.poses, self.components, self.version, dirty
            )
        self.events.append(
            {"status": "revoked", "source": source, "target": target, "reason": reason}
        )
        return {"version": self.version, "dirty_frames": dirty, "graph": diagnostic}

    def placement(self, frame):
        if not 0 <= frame < len(self.poses) or not is_rigid(self.poses[frame]):
            raise ValueError("placement requires a received frame")
        return self.poses[frame].copy(), self.components[frame], self.version
