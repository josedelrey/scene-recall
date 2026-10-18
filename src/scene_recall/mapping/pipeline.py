"""Select keyframes and build bounded submaps without bridging tracking loss."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from scene_recall.data import Calibration, Observation
from scene_recall.mapping.evaluation import cloud_statistics, compare_clouds
from scene_recall.mapping.voxel import (
    MappingConfig,
    SurfaceSamples,
    VoxelCloud,
    VoxelMap,
    sample_observation,
)
from scene_recall.odometry.pipeline import TrajectoryFrame, is_rigid


def validate_trajectory(frames: list[TrajectoryFrame]) -> None:
    """Reject malformed gauges, resets and statuses before selecting any frames."""
    for index, row in enumerate(frames):
        if row.frame_index != index:
            raise ValueError("trajectory indices must be zero-based and contiguous")
        if not is_rigid(row.T_SC_D):
            raise ValueError("trajectory poses must be rigid")
        if not isinstance(row.segment_id, (int, np.integer)) or row.segment_id < 0:
            raise ValueError("trajectory segment IDs must be nonnegative integers")
        boundary = index == 0 or row.segment_id != frames[index - 1].segment_id
        if index and row.segment_id < frames[index - 1].segment_id:
            raise ValueError("trajectory segments must be ordered and never reused")
        expected = "initialized" if index == 0 else "lost" if boundary else "tracked"
        if row.status != expected:
            raise ValueError("trajectory status must agree with segment boundaries")
        if boundary and not np.allclose(row.T_SC_D, np.eye(4), rtol=0, atol=1e-6):
            raise ValueError("each trajectory segment must start at identity")


def select_keyframes(frames: list[TrajectoryFrame], stride: int) -> list[int]:
    """Choose segment-relative temporal strides plus each segment's endpoints.

    The policy depends only on frame order and estimated segment boundaries.
    It is reproducible across backends sharing those boundaries and never uses
    references. It makes no claim of motion-adaptive or optimal view selection.
    """
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise ValueError("keyframe stride must be a positive integer")
    validate_trajectory(frames)
    selected = []
    start = 0
    for index, row in enumerate(frames):
        if index == 0 or row.segment_id != frames[index - 1].segment_id:
            start = index
        last = (
            index == len(frames) - 1 or frames[index + 1].segment_id != row.segment_id
        )
        if (index - start) % stride == 0 or last:
            selected.append(index)
    return selected


@dataclass(frozen=True)
class Keyframe:
    row: TrajectoryFrame
    samples: SurfaceSamples


@dataclass(frozen=True)
class Submap:
    submap_id: int
    segment_id: int
    keyframes: tuple[Keyframe, ...]
    T_SM: np.ndarray
    T_MW_reference: np.ndarray | None
    estimated: VoxelCloud
    reference: VoxelCloud
    report: dict


def build_submap(
    keyframes: list[Keyframe], submap_id: int, config: MappingConfig
) -> Submap:
    """Keep M anchored at the first estimated camera of this bounded chunk."""
    if not keyframes or len(keyframes) > config.max_keyframes_per_submap:
        raise ValueError("submap must contain a bounded nonempty keyframe population")
    if len({frame.row.segment_id for frame in keyframes}) != 1:
        raise ValueError("a submap cannot span disconnected trajectory segments")
    T_SM = keyframes[0].row.T_SC_D.copy()
    T_MS = np.linalg.inv(T_SM)
    estimated = VoxelMap(config.voxel_size_m, config.max_voxels)
    reference = VoxelMap(config.voxel_size_m, config.max_voxels)
    valid_reference = [
        frame
        for frame in keyframes
        if frame.row.reference_pose is not None and is_rigid(frame.row.reference_pose)
    ]
    anchor = valid_reference[0].row if valid_reference else None
    T_MW = (
        T_MS @ anchor.T_SC_D @ np.linalg.inv(anchor.reference_pose)
        if anchor is not None
        else None
    )
    for frame in keyframes:
        row = frame.row
        estimated.integrate(frame.samples, T_MS @ row.T_SC_D, row.frame_index)
        if row.reference_pose is not None and is_rigid(row.reference_pose):
            reference.integrate(
                frame.samples, T_MW @ row.reference_pose, row.frame_index
            )
    est, ref = estimated.cloud(), reference.cloud()
    paired = len(valid_reference) == len(keyframes)
    report = {
        "submap_id": submap_id,
        "segment_id": keyframes[0].row.segment_id,
        "coordinate_frame": f"segment_{keyframes[0].row.segment_id}_submap_{submap_id}",
        "keyframe_indices": [frame.row.frame_index for frame in keyframes],
        "reference_keyframe_indices": [
            frame.row.frame_index for frame in valid_reference
        ],
        "reference_anchor_frame": None if anchor is None else anchor.frame_index,
        "reference_missing": sum(
            frame.row.reference_pose is None for frame in keyframes
        ),
        "reference_invalid": sum(
            frame.row.reference_pose is not None
            and not is_rigid(frame.row.reference_pose)
            for frame in keyframes
        ),
        "T_SM": T_SM.tolist(),
        "T_MW_reference_diagnostic": None if T_MW is None else T_MW.tolist(),
        "estimated": cloud_statistics(est),
        "reference": cloud_statistics(ref),
        "comparison_status": "paired" if paired else "unavailable_unpaired_references",
        "comparison": compare_clouds(est, ref, config.voxel_size_m) if paired else None,
    }
    return Submap(
        submap_id,
        keyframes[0].row.segment_id,
        tuple(keyframes),
        T_SM,
        T_MW,
        est,
        ref,
        report,
    )


def reconstruct(
    observations: Iterable[Observation],
    frames: list[TrajectoryFrame],
    calibration: Calibration,
    config: MappingConfig,
) -> Iterator[Submap]:
    """Stream observations with bounded keyframe retention and strict pairing.

    Consume to exhaustion to detect trailing count mismatches. Callers publishing
    artifacts must finish input validation before marking an experiment complete.
    """
    selected = set(select_keyframes(frames, config.keyframe_stride))
    pending = []
    submap_id = 0
    count = 0
    for index, observation in enumerate(observations):
        if index >= len(frames):
            raise ValueError("more observations than trajectory frames")
        row = frames[index]
        if observation.frame_index != index or observation.timestamp != row.timestamp:
            raise ValueError("observation index or timestamp disagrees with trajectory")
        count += 1
        if pending and (
            pending[0].row.segment_id != row.segment_id
            or len(pending) == config.max_keyframes_per_submap
        ):
            yield build_submap(pending, submap_id, config)
            submap_id += 1
            pending = []
        if index in selected:
            pending.append(
                Keyframe(row, sample_observation(observation, calibration, config))
            )
    if count != len(frames):
        raise ValueError("fewer observations than trajectory frames")
    if pending:
        yield build_submap(pending, submap_id, config)
