"""Serial experimental SLAM frontend, smoother and revision contract.

References never enter this module. The existing Recent frontend initializes
poses and proposes identities. Its fixed landmarks supply no optimizer priors.
Only new keyframe measurements enter the joint graph. Reconstruction consumers
must rebuild affected chunks after a revision, using raw depth and revised poses.
"""

from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np

from scene_recall.geometry.camera import transform_points
from scene_recall.geometry.transforms import is_rigid
from scene_recall.multiview.batch_ba import BatchConfig
from scene_recall.multiview.diagnostics import grid_coverage
from scene_recall.multiview.fixed_landmarks import DirectFactors
from scene_recall.multiview.temporal import (
    PairMotion,
    TimingConfig,
    bind_depth,
    pair_motion,
)
from scene_recall.odometry.pipeline import PairEstimate
from scene_recall.odometry.recent_map import RecentMapConfig, RecentMapRGBDBackend
from scene_recall.slam.graph import marginalize_oldest, solve_window
from scene_recall.slam.marginal import marginalize, prior_supports_camera
from scene_recall.slam.support import SupportConfig, metric_support


@dataclass(frozen=True)
class SLAMConfig:
    window_size: int = 8
    keyframe_interval: int = 5
    max_landmarks: int = 128
    exposure: str = "timed"
    optimize: bool = True
    raw_ledger: bool = False
    raw_only: bool = False
    marginalization: str = "schur"
    stale_keyframes: int = 0
    solver: BatchConfig = field(
        default_factory=lambda: BatchConfig(
            max_iterations=10,
            max_translation_m=0.1,
            max_rotation_deg=5,
            max_point_displacement_m=0.1,
        )
    )
    timing: TimingConfig = field(default_factory=TimingConfig)
    support: SupportConfig = field(default_factory=SupportConfig)

    def __post_init__(self):
        for name, lower in (
            ("window_size", 3),
            ("keyframe_interval", 1),
            ("max_landmarks", 20),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.exposure not in ("timed", "static"):
            raise ValueError("exposure must be timed or static")
        if self.marginalization not in ("schur", "sqrt"):
            raise ValueError("marginalization must be schur or sqrt")
        if (
            isinstance(self.stale_keyframes, bool)
            or not isinstance(self.stale_keyframes, int)
            or self.stale_keyframes < 0
        ):
            raise ValueError("stale_keyframes must be a nonnegative integer")
        if self.stale_keyframes and self.marginalization != "sqrt":
            raise ValueError("point retirement requires square-root marginalization")
        if not isinstance(self.optimize, bool) or not isinstance(self.raw_ledger, bool):
            raise TypeError("optimizer switches must be boolean")
        if not isinstance(self.raw_only, bool):
            raise TypeError("raw-only switch must be boolean")
        if self.raw_only and (self.optimize or not self.raw_ledger):
            raise ValueError(
                "raw-only tracking requires a ledger and shared optimization"
            )
        if not isinstance(self.solver, BatchConfig) or not isinstance(
            self.timing, TimingConfig
        ):
            raise TypeError("invalid SLAM configuration")
        if not isinstance(self.support, SupportConfig):
            raise TypeError("invalid support configuration")


class RevisionFrontend(RecentMapRGBDBackend):
    """Isolate access to legacy frontend internals at this experimental boundary."""

    def __init__(self, config=None, *, support=None):
        super().__init__(config)
        self.support_policy = support or SupportConfig()
        self.retained_support = None
        self.exposure_transforms = {}

    def _bootstrap(self, observation, features, calibration):
        self.retained_support = None
        self.exposure_transforms = {}
        super()._bootstrap(observation, features, calibration)

    def _pixel_pose(self, frame_id, frame, calibration):
        exposure = self.exposure_transforms.get(frame_id)
        if exposure is None or np.array_equal(exposure, calibration.T_RD):
            return frame.T_SR
        # Native tracker points retain their virtual depth-time RGB chart.
        # Only the pixel projection uses the actual retained color exposure.
        return frame.T_SR @ calibration.T_RD @ np.linalg.inv(exposure)

    def retain_support(self, frame, owners):
        """Monitor only native owners actually admitted to the shared ledger."""
        self.retained_support = frame, frozenset(map(int, owners))

    def _needs_keyframe(self, index, features, pose, matches, calibration, diagnostics):
        scheduled = super()._needs_keyframe(
            index, features, pose, matches, calibration, diagnostics
        )
        if not self.support_policy.enabled or self.retained_support is None:
            return scheduled
        frame, owners = self.retained_support
        indices = [feature for owner, feature in matches if owner in owners]
        support = metric_support(features.points_R[indices])
        required = min(
            self.support_policy.minimum,
            max(3, int(np.ceil(len(owners) * self.support_policy.retained_fraction))),
        )
        weak = (
            support["count"] < required
            or support["rotation_rank"] < 3
            or support["spread"] < self.support_policy.minimum_spread
        )
        diagnostics["retained_support"] = {
            **support,
            "source": frame,
            "source_native_owners": len(owners),
            "required": required,
            "early_keyframe": bool(weak and not scheduled),
        }
        return scheduled or weak

    def start(self, observation, calibration):
        features = self._extract(observation, calibration)
        self._bootstrap(observation, features, calibration)
        self._previous = (observation, features)
        self._calibration = calibration

    def revise(self, depth_poses, points, calibration):
        corrections = {}
        for key, pose in depth_poses.items():
            if key in self.keyframes:
                frame = self.keyframes[key]
                changed = pose @ np.linalg.inv(calibration.T_RD)
                corrections[key] = changed @ np.linalg.inv(frame.T_SR)
                self.keyframes[key] = replace(frame, T_SR=changed)
        for identity, landmark in self.landmarks.items():
            if identity in points:
                landmark.point_S = points[identity].copy()
            else:
                # A seed outside the solver follows its latest revised observer.
                key = next(
                    (f for f, _ in reversed(landmark.observations) if f in corrections),
                    None,
                )
                if key is not None:
                    landmark.point_S = transform_points(
                        landmark.point_S, corrections[key]
                    )
        key = self._previous[0].frame_index
        if key in depth_poses:
            self._pose = depth_poses[key].copy()
        self._rebuild_index()


class IncrementalSLAM:
    """Bounded estimator state plus append-only output revisions owned by caller.

    process returns the current depth-camera pose, a pair result and a mapping
    of revised keyframe depth poses. Frame indices increase without reuse. A
    failed frontend transition clears all graph state and starts an independent
    segment. A failed marginalization starts a conditional graph at the incoming
    keyframe and is reported separately from tracking failure.
    """

    def __init__(self, config=None):
        self.config = config or SLAMConfig()
        self.frontend = RevisionFrontend(
            RecentMapConfig(
                max_keyframes=self.config.window_size,
                min_keyframe_gap=self.config.keyframe_interval,
                max_keyframe_gap=self.config.keyframe_interval,
            ),
            support=self.config.support,
        )
        self.previous = None
        self.previous_pose = np.eye(4)
        self.segment_id = 0
        self.revision = 0
        self._next_track = 0
        self.last_measurement = None
        self.last_raw_measurement = None
        self._clear_graph()

    def _clear_graph(self):
        self.poses, self.transforms, self.packets = {}, {}, {}
        self.points, self.identities = {}, {}
        self.last_seen = {}
        self.root_id, self.prior = None, None

    def apply_global_feedback(self, depth_poses, point_positions, removed, calibration):
        """Rebase the active tracker from a global solve without importing a prior.

        The global result is a state revision, not an independent measurement.
        Retained local factors remain single-owned raw observations. A local
        marginal prior is dropped because its old chart cannot be transported
        through a nonrigid historical correction without relinearization.
        """
        if self.config.optimize or not self.config.raw_ledger:
            raise ValueError("global feedback requires the raw-ledger tracking mode")
        if any(
            key not in self.frontend.keyframes or not is_rigid(pose)
            for key, pose in depth_poses.items()
        ):
            raise ValueError("feedback requires active rigid keyframe poses")
        if any(
            owner not in self.frontend.landmarks
            or np.asarray(point).shape != (3,)
            or not np.isfinite(point).all()
            for owner, point in point_positions.items()
        ):
            raise ValueError("feedback requires finite active landmark positions")
        if set(removed) & set(point_positions):
            raise ValueError("a landmark cannot be revised and released together")
        revised = {
            key: pose
            for key, pose in depth_poses.items()
            if not np.allclose(
                pose,
                self.frontend.keyframes[key].T_SR @ calibration.T_RD,
                atol=1e-12,
                rtol=0,
            )
        }
        moved = {
            owner: point
            for owner, point in point_positions.items()
            if not np.allclose(
                point, self.frontend.landmarks[owner].point_S, atol=1e-12, rtol=0
            )
        }
        if not revised and not moved and not removed:
            return {
                "revised_cameras": 0,
                "revised_landmarks": 0,
                "released_landmarks": 0,
            }
        self.frontend.revise(revised, moved, calibration)
        released = 0
        for owner in removed:
            if owner not in self.frontend.landmarks:
                continue
            for frame_id, frame in reversed(list(self.frontend.keyframes.items())):
                matches = np.flatnonzero(self.frontend.owners[frame_id] == owner)
                if not len(matches):
                    continue
                native = frame.features.points_R[matches[0]]
                if np.isfinite(native).all():
                    self.frontend.landmarks[owner].point_S = transform_points(
                        native, frame.T_SR
                    )
                    released += 1
                    break
        for key, pose in revised.items():
            if key in self.poses:
                self.poses[key] = pose @ np.linalg.inv(self.transforms[key])
        for owner, track in self.identities.items():
            if owner in self.frontend.landmarks:
                self.points[track] = self.frontend.landmarks[owner].point_S.copy()
        current = self.previous.frame_index if self.previous is not None else None
        if current in revised:
            self.previous_pose = revised[current].copy()
        if self.prior is not None:
            self.prior = None
            self.root_id = next(iter(self.poses), None)
        self.revision += 1
        return {
            "revised_cameras": len(revised),
            "revised_landmarks": len(moved),
            "released_landmarks": released,
        }

    def synchronize_current(self, depth_pose):
        """Carry an attached nonkeyframe into the revised tracking chart."""
        if not is_rigid(depth_pose):
            raise ValueError("current feedback pose must be rigid")
        self.previous_pose = depth_pose.copy()
        self.frontend._pose = depth_pose.copy()

    def _factors(self):
        packets = list(self.packets.values())
        if not packets:
            return None
        keys = list(self.poses)
        arrays = {
            name: np.concatenate([getattr(p, name) for p in packets])
            for name in DirectFactors.__dataclass_fields__
        }
        arrays["view"] = np.array([keys.index(int(i)) for i in arrays["view"]])
        arrays["points"] = np.array([self.points[int(i)] for i in arrays["track_id"]])
        return DirectFactors(**arrays)

    def _retire(self, K):
        if self.config.marginalization == "sqrt":
            return self._marginalize(K, cameras=(next(iter(self.poses)),))
        factors = self._factors()
        if factors is None:
            raise ValueError("no observations at lag boundary")
        keys, poses = list(self.poses), np.array(list(self.poses.values()))
        prior, keep, retired = marginalize_oldest(
            poses,
            factors,
            K,
            keys,
            self.root_id,
            self.prior,
            self.config.solver,
        )
        # Apply only after successful factorization. Each raw factor is either
        # untouched or consumed in the new prior, never present in both.
        self.prior = prior
        if keys[0] == self.root_id:
            self.root_id = None
        del self.poses[keys[0]], self.transforms[keys[0]]
        consumed = int((~keep).sum())
        self.packets.pop(keys[0], None)
        for key, packet in list(self.packets.items()):
            mask = ~np.isin(packet.track_id, retired)
            if mask.any():
                self.packets[key] = replace(
                    packet,
                    **{
                        name: getattr(packet, name)[mask]
                        for name in packet.__dataclass_fields__
                    },
                )
            else:
                del self.packets[key]
        for track in retired:
            del self.points[int(track)]
            self.last_seen.pop(int(track), None)
        self.identities = {
            identity: track
            for identity, track in self.identities.items()
            if track in self.points
        }
        return {
            "consumed_observations": consumed,
            "retired_landmarks": len(retired),
            "prior_rank": len(prior.b),
        }

    def _marginalize(self, K, *, cameras=(), tracks=()):
        factors = self._factors()
        prior, _, retired, diagnostic = marginalize(
            np.array(list(self.poses.values())),
            factors,
            K,
            list(self.poses),
            self.root_id,
            self.prior,
            self.config.solver,
            cameras=cameras,
            tracks=tracks,
        )
        # Mutate only after successful elimination. Consumed factors cannot
        # reappear in raw packets, even when their frontend identity returns.
        self.prior = prior
        for key in cameras:
            del self.poses[key], self.transforms[key]
            if key == self.root_id:
                self.root_id = None
        for key, packet in list(self.packets.items()):
            mask = ~np.isin(packet.track_id, retired)
            if key in cameras or not mask.any():
                del self.packets[key]
            else:
                self.packets[key] = replace(
                    packet,
                    **{
                        name: getattr(packet, name)[mask]
                        for name in packet.__dataclass_fields__
                    },
                )
        for track in retired:
            del self.points[int(track)]
            self.last_seen.pop(int(track), None)
        self.identities = {
            identity: track
            for identity, track in self.identities.items()
            if track in self.points
        }
        return diagnostic

    def _retire_stale(self, key, K):
        present = set(map(int, self.frontend.owners[key]))
        for identity, track in self.identities.items():
            if identity in present:
                self.last_seen[track] = key
        horizon = self.config.stale_keyframes * self.config.keyframe_interval
        stale = [
            track for track in self.points if key - self.last_seen[track] >= horizon
        ]
        return self._marginalize(K, tracks=stale) if stale else None

    def _ingest(self, observation, calibration, depth_pose, features, motion):
        samples = bind_depth(
            features.pixels, observation.depth, calibration, motion, self.config.timing
        )
        key = observation.frame_index
        pose = depth_pose @ np.linalg.inv(samples.T_RcD)
        owners = self.frontend.owners[key]
        valid = np.flatnonzero(owners >= 0)
        if self.config.raw_ledger and len(valid):
            if len(np.unique(owners[valid])) != len(valid):
                raise ValueError("frontend identity has multiple features in one view")
            seeds = []
            for i in valid:
                if samples.source_index[i] >= 0:
                    seeds.append(transform_points(samples.points_R[i], pose))
                else:
                    seeds.append(self.frontend.landmarks[int(owners[i])].point_S)
            raw = DirectFactors(
                np.full(len(valid), key, dtype=int),
                owners[valid].astype(int).copy(),
                np.array(seeds),
                features.pixels[valid].copy(),
                samples.points_D[valid].copy(),
                np.repeat(samples.T_RcD[None], len(valid), axis=0),
                samples.whitening_D[valid].copy(),
                np.ones(len(valid)),
            )
            self.last_raw_measurement = {
                **{
                    name: getattr(raw, name).copy() for name in raw.__dataclass_fields__
                },
                "feature_index": valid.copy(),
                "native_index": samples.source_index[valid].copy(),
                "frontend_identity": owners[valid].copy(),
            }
        if self.config.raw_only:
            return {"raw_observations": len(valid), "local_prior_factors": 0}
        existing = [int(i) for i in valid if int(owners[i]) in self.identities]
        new = [
            int(i)
            for i in valid
            if int(owners[i]) not in self.identities and samples.source_index[i] >= 0
        ]
        # Spatial round robin, preserving existing track identities first.
        bins = {}
        for i in new:
            cell = tuple(
                np.floor(
                    features.pixels[i]
                    / [calibration.rgb_shape[1], calibration.rgb_shape[0]]
                    * [8, 6]
                ).astype(int)
            )
            bins.setdefault(cell, []).append(i)
        ordered = [
            values[j]
            for j in range(max(map(len, bins.values()), default=0))
            for _, values in sorted(bins.items())
            if j < len(values)
        ]
        selected = (
            existing + ordered[: max(0, self.config.max_landmarks - len(self.points))]
        )
        if not selected:
            return {"status": "rejected", "reason": "no admissible measurements"}
        native_ids = samples.source_index[selected]
        finite_ids = native_ids[native_ids >= 0]
        if len(np.unique(finite_ids)) != len(finite_ids):
            raise ValueError("native measurement has multiple owners")
        if len(np.unique(owners[selected])) != len(selected):
            raise ValueError("landmark has multiple features in one view")
        ids = []
        for i in selected:
            identity = int(owners[i])
            if identity not in self.identities:
                track = self._next_track
                self._next_track += 1
                self.identities[identity] = track
                self.points[track] = transform_points(samples.points_R[i], pose)
                self.last_seen[track] = key
            ids.append(self.identities[identity])
        self.poses[key], self.transforms[key] = pose, samples.T_RcD
        if self.root_id is None and self.prior is None:
            self.root_id = key
        ids = np.array(ids, dtype=int)
        self.packets[key] = DirectFactors(
            np.full(len(ids), key, dtype=int),
            ids,
            np.array([self.points[int(i)] for i in ids]),
            features.pixels[selected].copy(),
            samples.points_D[selected].copy(),
            np.repeat(samples.T_RcD[None], len(ids), axis=0),
            samples.whitening_D[selected].copy(),
            np.ones(len(ids)),
        )
        packet = self.packets[key]
        self.last_measurement = {
            **{
                name: getattr(packet, name).copy()
                for name in packet.__dataclass_fields__
            },
            "feature_index": np.array(selected, dtype=np.int64),
            "native_index": native_ids.copy(),
            "frontend_identity": owners[selected].copy(),
        }
        return {"measurement_count": len(ids), "native_count": len(finite_ids)}

    def process(self, observation, calibration):
        started = perf_counter()
        self.last_measurement = None
        self.last_raw_measurement = None
        if self.previous is None:
            if observation.frame_index != 0:
                raise ValueError("SLAM starts at frame zero")
            self.frontend.start(observation, calibration)
            pair = None
        else:
            pair = self.frontend.estimate(self.previous, observation, calibration)
            if pair.T_C1C0 is None:
                self.segment_id += 1
                self._clear_graph()
        pose = self.frontend._pose.copy()
        key = observation.frame_index
        diagnostic, revisions = {}, {}
        if key in self.frontend.keyframes:
            timing_start = perf_counter()
            if self.config.exposure == "static":
                motion = PairMotion(np.eye(4), "explicit_static_exposure", True)
            elif self.previous is not None and pair.T_C1C0 is not None:
                stamps = [self.previous.timestamp, observation.timestamp]
                proxies = np.array([self.previous_pose, pose]) @ np.linalg.inv(
                    calibration.T_RD
                )
                metadata = observation.provenance or {}
                motion = pair_motion(
                    proxies,
                    stamps,
                    metadata.get("timestamp_color"),
                    observation.timestamp,
                    self.config.timing,
                )
            else:
                motion = PairMotion(np.eye(4), "no_segment_velocity", False)
            diagnostic["exposure_status"] = motion.status
            if self.config.stale_keyframes and self.points:
                try:
                    diagnostic["point_marginalization"] = self._retire_stale(
                        key, calibration.K_R
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    diagnostic["graph_restart"] = str(error)
                    self._clear_graph()
            # A full point budget can outlive every track in the current view.
            # Camera-count eviction alone cannot advance such a stalled graph.
            if (
                self.poses
                and key - next(reversed(self.poses))
                >= self.config.window_size * self.config.keyframe_interval
            ):
                diagnostic["graph_restart"] = "no measurements within lag horizon"
                self._clear_graph()
            if len(self.poses) >= self.config.window_size:
                try:
                    diagnostic["marginalization"] = self._retire(calibration.K_R)
                except (ValueError, np.linalg.LinAlgError) as error:
                    diagnostic["graph_restart"] = str(error)
                    self._clear_graph()
            diagnostic.update(
                self._ingest(
                    observation, calibration, pose, self.frontend._previous[1], motion
                )
            )
            diagnostic["measurement_seconds"] = perf_counter() - timing_start
            factors = self._factors()
            if (
                factors is not None
                and key in self.poses
                and len(self.poses) > 1
                and self.config.optimize
            ):
                keys = list(self.poses)
                counts = {
                    int(i): int(n)
                    for i, n in zip(*np.unique(factors.track_id, return_counts=True))
                }
                supported = np.array([counts[int(i)] >= 2 for i in factors.track_id])
                adequate = all(
                    (
                        self.config.stale_keyframes
                        and k != key
                        and prior_supports_camera(self.prior, k)
                    )
                    or (
                        int((supported & (factors.view == i)).sum())
                        >= self.config.solver.min_observations
                        and grid_coverage(
                            factors.pixels[supported & (factors.view == i)],
                            calibration.rgb_shape,
                        )
                        >= self.config.solver.min_grid_coverage
                    )
                    for i, k in enumerate(keys)
                    if k != self.root_id
                )
                if adequate:
                    solver_start = perf_counter()
                    poses, points, fit = solve_window(
                        np.array(list(self.poses.values())),
                        factors,
                        calibration.K_R,
                        keys,
                        self.root_id,
                        self.prior,
                        self.config.solver,
                    )
                    fit["seconds"] = perf_counter() - solver_start
                    diagnostic["solver"] = fit
                    if fit["status"] == "refined":
                        self.poses.update(zip(keys, poses, strict=True))
                        self.points.update(
                            zip(np.unique(factors.track_id), points, strict=True)
                        )
                        revisions = {
                            k: p @ self.transforms[k] for k, p in self.poses.items()
                        }
                        self.revision += 1
                        pose = revisions[key].copy()
                        self.frontend.revise(
                            revisions,
                            {
                                identity: self.points[track]
                                for identity, track in self.identities.items()
                            },
                            calibration,
                        )
                else:
                    diagnostic["solver"] = {
                        "status": "rejected",
                        "reason": "multiview camera support",
                    }
            diagnostic.update(
                active_cameras=len(self.poses),
                active_points=len(self.points),
                active_observations=sum(len(p.view) for p in self.packets.values()),
            )
        if pair is not None:
            diagnostic["frontend"] = pair.diagnostics
            pair = PairEstimate(
                None
                if pair.T_C1C0 is None
                else np.linalg.inv(pose) @ self.previous_pose,
                pair.reason,
                diagnostics={"slam_revision": self.revision},
            )
        self.previous, self.previous_pose = observation, pose.copy()
        diagnostic["total_seconds"] = perf_counter() - started
        return pose, pair, revisions, diagnostic


class RevisionArchive:
    """Output history with rigid keyframe attachments and map invalidation.

    This archive grows with the recording and is separate from bounded estimator
    state. Every nonkeyframe follows its preceding keyframe exactly. Consumers
    can use dirty frame IDs to rebuild voxel/TSDF chunks from original inputs.
    """

    def __init__(self):
        self.poses, self.anchors, self.relative = [], [], []
        self.current_anchor = None
        self.version = 0

    def append(self, pose, *, keyframe=False, boundary=False):
        index = len(self.poses)
        if boundary or keyframe or self.current_anchor is None:
            self.current_anchor = index
        self.anchors.append(self.current_anchor)
        self.relative.append(
            np.eye(4)
            if self.current_anchor == index
            else np.linalg.inv(self.poses[self.current_anchor]) @ pose
        )
        self.poses.append(pose.copy())

    def apply(self, revisions):
        dirty = []
        for i, anchor in enumerate(self.anchors):
            if anchor in revisions:
                changed = revisions[anchor] @ self.relative[i]
                if not np.allclose(changed, self.poses[i], atol=1e-12, rtol=0):
                    self.poses[i] = changed
                    dirty.append(i)
        if dirty:
            self.version += 1
        return dirty
