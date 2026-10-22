"""Experimental revisable metric keyframe graph with independent gauges.

Odometry edges summarize the supplied local estimator and remain linear.
Verified loop factors use a graduated Geman-McClure objective, with a bounded
information budget per correlated view group. Scales are engineering choices,
not calibrated covariances. This dense reference solver has no real-time claim.
"""

from dataclasses import dataclass

import numpy as np

from scene_recall.geometry.transforms import is_rigid
from scene_recall.multiview.pose_only import retract, skew
from scene_recall.slam.graph import pose_chart
from scene_recall.slam.linear import BlockSystem


@dataclass(frozen=True)
class PoseGraphConfig:
    odometry_translation_m: float = 0.01
    odometry_rotation_deg: float = 1.0
    loop_translation_m: float = 0.03
    loop_rotation_deg: float = 2.0
    robust_scale: float = 3.0
    max_iterations: int = 20
    reject_weight: float = 0.1
    max_nodes: int = 512
    linear_solver: str = "dense"

    def __post_init__(self):
        if self.linear_solver not in ("dense", "block_pcg"):
            raise ValueError("unsupported graph linear solver")
        for name in (
            "odometry_translation_m",
            "odometry_rotation_deg",
            "loop_translation_m",
            "loop_rotation_deg",
            "robust_scale",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_iterations", "max_nodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.reject_weight) or not 0 < self.reject_weight < 1:
            raise ValueError("reject_weight must lie between zero and one")


@dataclass(frozen=True, eq=False)
class PoseEdge:
    source: int
    target: int
    T_target_source: np.ndarray
    kind: str = "odometry"
    group: tuple = ()


def edge_residual(source, target, measurement, *, jacobian=False):
    """Split translation/rotation chart with left world-frame pose increments."""
    left = np.linalg.inv(measurement) @ np.linalg.inv(target)
    error = left @ source
    residual, chart = pose_chart(error, np.eye(4))
    if not jacobian:
        return residual
    adjoint = np.zeros((6, 6))
    R, t = left[:3, :3], left[:3, 3]
    adjoint[:3, :3] = adjoint[3:, 3:] = R
    adjoint[:3, 3:] = skew(t) @ R
    J = chart @ adjoint
    return residual, J, -J


class MetricPoseGraph:
    """Reference-free factor ownership, explicit revocation and map invalidation."""

    def __init__(self, config=None):
        self.config = config or PoseGraphConfig()
        self.local, self.segments, self.edges = {}, {}, {}
        self.poses, self.components = {}, {}
        self.version = 0

    def add_node(self, frame, segment, local_pose):
        if (
            isinstance(frame, bool)
            or not isinstance(frame, int)
            or frame < 0
            or frame in self.local
            or not is_rigid(local_pose)
            or len(self.local) >= self.config.max_nodes
        ):
            raise ValueError(
                "node requires a new nonnegative frame and rigid pose within budget"
            )
        self.local[frame] = local_pose.copy()
        self.segments[frame] = segment

    def add_edge(self, source, target, measurement, *, kind="odometry", group=()):
        if (
            source not in self.local
            or target not in self.local
            or source >= target
            or (source, target) in self.edges
            or not is_rigid(measurement)
            or kind not in ("odometry", "loop")
        ):
            raise ValueError(
                "edge requires unique ordered nodes and a rigid measurement"
            )
        if kind == "odometry" and self.segments[source] != self.segments[target]:
            raise ValueError("odometry cannot bridge a tracking loss")
        owned = measurement.copy()
        owned.setflags(write=False)
        self.edges[source, target] = PoseEdge(source, target, owned, kind, tuple(group))

    def revoke(self, source, target, reason):
        key = source, target
        if key not in self.edges or not isinstance(reason, str) or not reason.strip():
            raise ValueError("revocation requires an existing edge and explicit reason")
        if self.edges[key].kind != "loop":
            raise ValueError("only loop edges can be revoked")
        del self.edges[key]

    def revise(self, poses):
        if any(
            frame not in self.local or not is_rigid(pose)
            for frame, pose in poses.items()
        ):
            raise ValueError("revision requires known rigid cameras")
        for frame, pose in poses.items():
            self.local[frame] = pose.copy()
        # Refresh only local summaries. Raw camera-to-camera loops stay immutable.
        for key, edge in list(self.edges.items()):
            if edge.kind == "odometry":
                owned = np.linalg.inv(self.local[edge.target]) @ self.local[edge.source]
                owned.setflags(write=False)
                self.edges[key] = PoseEdge(*key, owned, edge.kind, edge.group)

    def _initialize(self, edges):
        parents = {k: k for k in self.local}

        def root(k):
            while parents[k] != k:
                k = parents[k]
            return k

        adjacency = {k: [] for k in self.local}
        # Preserve every local chain before using cross-map measurements.
        for edge in sorted(
            edges, key=lambda e: (e.kind != "odometry", e.source, e.target)
        ):
            a, b = root(edge.source), root(edge.target)
            if a == b:
                continue
            parents[max(a, b)] = min(a, b)
            adjacency[edge.source].append(
                (edge.target, np.linalg.inv(edge.T_target_source))
            )
            adjacency[edge.target].append((edge.source, edge.T_target_source))
        poses, components = {}, {}
        for first in sorted(self.local):
            if first in poses:
                continue
            poses[first], components[first] = self.local[first].copy(), first
            stack = [first]
            while stack:
                a = stack.pop()
                for b, relative in adjacency[a]:
                    if b not in poses:
                        poses[b], components[b] = poses[a] @ relative, first
                        stack.append(b)
        return poses, components

    def _scales(self, edges):
        counts = {}
        for edge in edges:
            if edge.kind == "loop":
                group = edge.group or (edge.source, edge.target)
                counts[group] = counts.get(group, 0) + 1
        config = self.config
        result = []
        for edge in edges:
            if edge.kind == "odometry":
                translation, rotation, count = (
                    config.odometry_translation_m,
                    config.odometry_rotation_deg,
                    1,
                )
            else:
                translation, rotation = (
                    config.loop_translation_m,
                    config.loop_rotation_deg,
                )
                count = counts[edge.group or (edge.source, edge.target)]
            result.append(
                np.array([1 / translation] * 3 + [1 / np.radians(rotation)] * 3)
                / np.sqrt(count)
            )
        return result

    def _optimize(self, poses, components, edges, robust):
        free = [k for k in sorted(poses) if components[k] != k]
        layout = {k: slice(i * 6, (i + 1) * 6) for i, k in enumerate(free)}
        scales = self._scales(edges)
        iterations, accepted = 0, 0
        damping = 1e-4

        def system(current, cutoff, linearize=False):
            H = None
            if linearize:
                H = (
                    BlockSystem(6 * len(free))
                    if self.config.linear_solver == "block_pcg"
                    else np.zeros((6 * len(free), 6 * len(free)))
                )
            g = np.zeros(6 * len(free)) if linearize else None
            cost, weights = 0.0, []
            for edge, scale in zip(edges, scales, strict=True):
                residual = edge_residual(
                    current[edge.source],
                    current[edge.target],
                    edge.T_target_source,
                    jacobian=linearize,
                )
                if linearize:
                    residual, Ja, Jb = residual
                r = residual * scale
                squared = float(r @ r)
                # Use the physical normalized innovation for robust decisions,
                # independently of the correlated group's information budget.
                count = 1.0
                if edge.kind == "loop":
                    count = (1 / self.config.loop_translation_m / scale[0]) ** 2
                innovation = squared * count
                weight = (
                    (1 + innovation / cutoff**2) ** -2
                    if robust and edge.kind == "loop"
                    else 1.0
                )
                cost += (
                    cutoff**2 * innovation / (cutoff**2 + innovation) / count
                    if robust and edge.kind == "loop"
                    else squared
                )
                weights.append(weight)
                if not linearize:
                    continue
                blocks = [
                    (layout[k], J * scale[:, None])
                    for k, J in ((edge.source, Ja), (edge.target, Jb))
                    if k in layout
                ]
                for a, A in blocks:
                    g[a] += weight * A.T @ r
                    for b, B in blocks:
                        value = weight * A.T @ B
                        if isinstance(H, BlockSystem):
                            H.add(a, b, value)
                        else:
                            H[a, b] += value
            return cost, weights, H, g

        initial = system(poses, self.config.robust_scale)[0]
        for multiplier in (8, 4, 2, 1) if robust else (1,):
            cutoff = self.config.robust_scale * multiplier
            for _ in range(self.config.max_iterations):
                iterations += 1
                cost, _, H, g = system(poses, cutoff, True)
                if not free or np.linalg.norm(g, ord=np.inf) < 1e-7:
                    break
                try:
                    if isinstance(H, BlockSystem):
                        delta = H.solve(-g, damping)
                    else:
                        diagonal = np.maximum(np.diag(H), 1e-8)
                        delta = np.linalg.solve(H + damping * np.diag(diagonal), -g)
                except np.linalg.LinAlgError:
                    damping *= 10
                    continue
                if not np.isfinite(delta).all():
                    raise ValueError("nonfinite graph update")
                candidate = {
                    k: retract(pose, delta[layout[k]]) if k in layout else pose
                    for k, pose in poses.items()
                }
                next_cost = system(candidate, cutoff)[0]
                if next_cost < cost:
                    poses, accepted = candidate, accepted + 1
                    damping = max(1e-10, damping / 3)
                    if np.linalg.norm(delta, ord=np.inf) < 1e-7:
                        break
                else:
                    damping = min(1e12, damping * 10)
                    if np.linalg.norm(delta, ord=np.inf) < 1e-8:
                        break
        final, weights, _, _ = system(poses, self.config.robust_scale)
        return (
            poses,
            weights,
            {
                "initial_cost": initial,
                "final_cost": final,
                "iterations": iterations,
                "accepted_steps": accepted,
            },
        )

    def solve(self, *, robust=True):
        edges = list(self.edges.values())
        poses, components = self._initialize(edges)
        poses, weights, diagnostic = self._optimize(poses, components, edges, robust)
        rejected = [
            e
            for e, w in zip(edges, weights, strict=True)
            if e.kind == "loop" and w < self.config.reject_weight
        ]
        # Rebuild after rejection so a discarded bridge cannot retain a stale
        # component transform. Keep rejected raw factors for explicit auditing.
        active = [e for e in edges if all(e is not r for r in rejected)]
        rounds = []
        while rejected and active:
            poses, components = self._initialize(active)
            poses, active_weights, refit = self._optimize(
                poses, components, active, robust
            )
            rounds.append(refit)
            additional = [
                e
                for e, w in zip(active, active_weights, strict=True)
                if e.kind == "loop" and w < self.config.reject_weight
            ]
            if not additional:
                break
            rejected.extend(additional)
            active = [e for e in active if all(e is not r for r in additional)]
        if rejected and not active:
            poses, components = self._initialize([])
        diagnostic["pruning_refits"] = rounds
        changed = [
            k
            for k in poses
            if k not in self.poses
            or self.components[k] != components[k]
            or not np.allclose(self.poses[k], poses[k], atol=1e-12, rtol=0)
        ]
        if changed:
            self.version += 1
        self.poses, self.components = poses, components
        return {
            **diagnostic,
            "version": self.version,
            "affected_frames": sorted(changed),
            "gauge_roots": sorted(set(components.values())),
            "edge_weights": weights,
            "rejected_loops": [[e.source, e.target] for e in rejected],
            "active_loops": sum(e.kind == "loop" for e in active),
        }
