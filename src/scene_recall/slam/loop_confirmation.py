"""Causal distinct-view confirmation in bounded temporal evidence groups."""

from dataclasses import dataclass

import numpy as np

from scene_recall.geometry.transforms import is_rigid
from scene_recall.odometry.evaluation import pose_error


@dataclass(frozen=True)
class LoopConfirmationConfig:
    cell_frames: int = 30
    translation_m: float = 0.05
    rotation_deg: float = 3.0

    def __post_init__(self):
        if (
            isinstance(self.cell_frames, bool)
            or not isinstance(self.cell_frames, int)
            or self.cell_frames < 1
        ):
            raise ValueError("cell_frames must be a positive integer")
        if any(
            not np.isfinite(v) or v <= 0
            for v in (self.translation_m, self.rotation_deg)
        ):
            raise ValueError("confirmation tolerances must be finite and positive")


class LoopConfirmation:
    """Require different source and target views before admitting any group.

    Local poses are used to compare hypotheses, never as external correctness
    labels. Exact repeated places and correlated bias can pass this check.
    """

    def __init__(self, config=None):
        self.config = config or LoopConfirmationConfig()
        self.pending, self.confirmed = {}, {}
        self.parents = {}

    def _root(self, segment):
        self.parents.setdefault(segment, segment)
        while self.parents[segment] != segment:
            segment = self.parents[segment]
        return segment

    def propose_batch(self, candidates):
        """Abstain when one query has verified disconnected source explanations."""
        if len({(c[1], c[3]) for c in candidates}) > 1:
            raise ValueError(
                "a confirmation batch must describe one target observation"
            )
        roots = {self._root(c[2]) for c in candidates}
        if len(roots) > 1:
            return ["ambiguous_components"] * len(candidates), []
        statuses, added = [], []
        for candidate in candidates:
            status, measurements = self.propose(*candidate)
            statuses.append(status)
            added.extend(measurements)
        return statuses, added

    def propose(
        self,
        source,
        target,
        source_segment,
        target_segment,
        local_source,
        local_target,
        measurement,
    ):
        if source >= target or not all(
            is_rigid(v) for v in (local_source, local_target, measurement)
        ):
            raise ValueError("confirmation requires ordered rigid observations")
        group = (
            source_segment,
            source // self.config.cell_frames,
            target_segment,
            target // self.config.cell_frames,
        )
        relation = (
            local_source @ np.linalg.inv(measurement) @ np.linalg.inv(local_target)
        )
        candidate = (source, target, measurement.copy(), group)

        def admitted(status, measurements):
            if measurements:
                a, b = self._root(source_segment), self._root(target_segment)
                self.parents[max(a, b)] = min(a, b)
            return status, measurements

        previous = self.confirmed.get(group, self.pending.get(group))
        if previous is None:
            self.pending[group] = candidate, relation
            return "tentative", []
        old, old_relation = previous
        translation, rotation = pose_error(relation, old_relation)
        if (
            translation > self.config.translation_m
            or rotation > self.config.rotation_deg
        ):
            if group not in self.confirmed:
                self.pending[group] = candidate, relation
            return "inconsistent_confirmation", []
        if group in self.confirmed:
            return admitted("supported", [candidate])
        if old[0] == source or old[1] == target:
            return "needs_distinct_views", []
        self.confirmed[group] = previous
        del self.pending[group]
        return admitted("confirmed", [old, candidate])
