"""Streaming RGB-D odometry with explicit breaks in trajectory continuity."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from scene_recall.data import Calibration, Observation


def is_rigid(T: np.ndarray, atol: float = 1e-4) -> bool:
    """Assess a transform without repairing or changing the supplied array."""
    return bool(
        T.shape == (4, 4)
        and np.isfinite(T).all()
        and np.array_equal(T[3], [0, 0, 0, 1])
        and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), rtol=0, atol=atol)
        and np.isclose(np.linalg.det(T[:3, :3]), 1, rtol=0, atol=atol)
    )


@dataclass(frozen=True, eq=False)
class PairEstimate:
    """A forward C_D0 -> C_D1 estimate, independent of reference poses.

    A missing T_C1C0 represents a failed transition. initial_T_C1C0 optionally
    retains the backend initialization before refinement and quality checks.
    For hybrid this is the accepted sparse pose, with RANSAC saved separately.
    candidate_T_C1C0 preserves an ICP candidate even when it is rejected.
    Diagnostics contain JSON-compatible scalar measurements and failure details.
    """

    T_C1C0: np.ndarray | None
    reason: str | None = None
    initial_T_C1C0: np.ndarray | None = None
    diagnostics: dict[str, int | float | str | None] = field(default_factory=dict)
    candidate_T_C1C0: np.ndarray | None = None
    ransac_T_C1C0: np.ndarray | None = None


class PairwiseBackend(Protocol):
    """Minimal boundary for future pairwise estimation backends."""

    def estimate(
        self, source: Observation, target: Observation, calibration: Calibration
    ) -> PairEstimate: ...


@dataclass(frozen=True, eq=False)
class TrajectoryFrame:
    """Camera pose in the local segment frame S, anchored at its first camera.

    T_SC_D maps the current depth optical frame into S. A lost transition
    starts a new segment at identity, with no relationship to the prior segment.
    reference_pose preserves the optional dataset T_WC_D solely for evaluation.
    """

    frame_index: int
    timestamp: float | None
    segment_id: int
    status: str
    T_SC_D: np.ndarray
    reference_pose: np.ndarray | None
    pair: PairEstimate | None = None


def track_observations(
    observations: Iterable[Observation],
    calibration: Calibration,
    backend: PairwiseBackend,
) -> Iterator[TrajectoryFrame]:
    """Estimate consecutive poses while retaining only the previous observation.

    Indices must be zero-based and contiguous. Failures start independent local
    segments. No reference pose initializes or corrects an estimated pose.
    """
    previous = None
    pose = np.eye(4)
    segment = 0
    for index, observation in enumerate(observations):
        if observation.frame_index != index:
            raise ValueError("frame indices must be zero-based and contiguous")
        if observation.rgb.shape[:2] != calibration.rgb_shape:
            raise ValueError("RGB shape must match calibration")
        if observation.depth.shape != calibration.depth_shape:
            raise ValueError("depth shape must match calibration")
        pair = None
        status = "initialized"
        if previous is not None:
            pair = backend.estimate(previous, observation, calibration)
            if pair.T_C1C0 is not None and not is_rigid(pair.T_C1C0, atol=1e-6):
                pair = PairEstimate(
                    None,
                    "backend returned a nonrigid pose",
                    pair.initial_T_C1C0,
                    pair.diagnostics,
                    pair.candidate_T_C1C0,
                    pair.ransac_T_C1C0,
                )
            if pair.T_C1C0 is None:
                segment += 1
                pose = np.eye(4)
                status = "lost"
            else:
                # T_C1C0 maps old camera to new camera, so accumulation inverts it.
                pose = pose @ np.linalg.inv(pair.T_C1C0)
                status = "tracked"
        yield TrajectoryFrame(
            index,
            observation.timestamp,
            segment,
            status,
            pose.copy(),
            observation.T_WC_D,
            pair,
        )
        previous = observation
