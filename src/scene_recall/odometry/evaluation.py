"""Reference-only trajectory evaluation with fixed metric scale and no gap bridging."""

from collections.abc import Iterable

import numpy as np

from scene_recall.odometry.pipeline import TrajectoryFrame, is_rigid


def error_statistics(values: Iterable[float]) -> dict:
    """Summarize a finite population, using null for empty statistics."""
    array = np.asarray(list(values), dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("error statistics require finite values")
    result = dict.fromkeys(("mean", "median", "rmse", "p95", "max"))
    result["count"] = len(array)
    if len(array):
        result.update(
            mean=float(array.mean()),
            median=float(np.median(array)),
            rmse=float(np.sqrt(np.mean(array**2))),
            p95=float(np.quantile(array, 0.95)),
            max=float(array.max()),
        )
    return result


def pose_error(estimated: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Return SE(3) translation error in meters and rotation error in degrees.

    Rotations must pass the consumer's rigid-pose check. atan2 avoids acos's
    loss of precision near identity. Source reference arrays are never repaired.
    """
    if not is_rigid(estimated) or not is_rigid(reference):
        raise ValueError("pose errors require rigid transforms")
    delta = np.linalg.inv(reference) @ estimated
    R = delta[:3, :3]
    sine = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    cosine = (np.trace(R) - 1) / 2
    return float(np.linalg.norm(delta[:3, 3])), float(
        np.degrees(np.arctan2(sine, cosine))
    )


def _summarize_errors(errors: list[tuple[float, float]]) -> dict:
    return {
        "translation_m": error_statistics(error[0] for error in errors),
        "rotation_deg": error_statistics(error[1] for error in errors),
    }


def _align_positions(estimated: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
    """Fit a proper rigid alignment, rejecting underdetermined position geometry."""
    if len(estimated) < 3:
        return None
    source = estimated - estimated.mean(axis=0)
    target = reference - reference.mean(axis=0)
    if (
        min(
            np.linalg.matrix_rank(source, tol=1e-10),
            np.linalg.matrix_rank(target, tol=1e-10),
        )
        < 2
    ):
        return None
    U, _, Vt = np.linalg.svd(source.T @ target)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(Vt.T @ U.T)
    alignment = np.eye(4)
    alignment[:3, :3] = Vt.T @ correction @ U.T
    alignment[:3, 3] = reference.mean(axis=0) - alignment[:3, :3] @ estimated.mean(
        axis=0
    )
    return alignment


def evaluate_trajectory(
    frames: Iterable[TrajectoryFrame], intervals: Iterable[int] = (1, 5, 10)
) -> dict:
    """Evaluate each segment separately and RPE at exact frame-index intervals.

    Anchored errors align the first valid reference pose of each segment.
    Aligned ATE uses a best-fit SE(3) position alignment, with scale fixed at one.
    Cross-segment RPE is always excluded. Missing and nonrigid references are
    counted and excluded without affecting tracking or repairing source poses.
    """
    rows = list(frames)
    intervals = tuple(intervals)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in intervals
    ):
        raise ValueError("RPE intervals must be positive integers")
    if len(set(intervals)) != len(intervals):
        raise ValueError("RPE intervals must be unique")
    for index, row in enumerate(rows):
        if row.frame_index != index:
            raise ValueError(
                "trajectory frame indices must be zero-based and contiguous"
            )
        if not is_rigid(row.T_SC_D):
            raise ValueError("estimated trajectory contains a nonrigid pose")
        if row.segment_id < 0 or (
            index and row.segment_id < rows[index - 1].segment_id
        ):
            raise ValueError("trajectory segment IDs must be nonnegative and ordered")
    reference_valid = [
        row.reference_pose is not None and is_rigid(row.reference_pose) for row in rows
    ]
    missing = sum(row.reference_pose is None for row in rows)
    invalid = sum(
        row.reference_pose is not None and not valid
        for row, valid in zip(rows, reference_valid)
    )
    segments = []
    for segment_id in sorted({row.segment_id for row in rows}):
        members = [row for row in rows if row.segment_id == segment_id]
        evaluable = [row for row in members if reference_valid[row.frame_index]]
        segment = {
            "segment_id": segment_id,
            "start_frame": members[0].frame_index,
            "end_frame": members[-1].frame_index,
            "frame_count": len(members),
            "reference_count": len(evaluable),
            "anchor_frame": None,
            "anchored_errors": _summarize_errors([]),
            "aligned_ate_translation_m": error_statistics([]),
            "alignment_status": "insufficient_reference_geometry",
            "endpoint_translation_error_m": None,
            "endpoint_rotation_error_deg": None,
        }
        if evaluable:
            anchor = evaluable[0]
            alignment = anchor.reference_pose @ np.linalg.inv(anchor.T_SC_D)
            anchored = [
                pose_error(alignment @ row.T_SC_D, row.reference_pose)
                for row in evaluable
            ]
            segment.update(
                anchor_frame=anchor.frame_index,
                anchored_errors=_summarize_errors(anchored),
            )
            if len(evaluable) > 1:
                segment.update(
                    endpoint_translation_error_m=anchored[-1][0],
                    endpoint_rotation_error_deg=anchored[-1][1],
                )
            positions = np.array([row.T_SC_D[:3, 3] for row in evaluable])
            references = np.array([row.reference_pose[:3, 3] for row in evaluable])
            fitted = _align_positions(positions, references)
            if fitted is not None:
                aligned = positions @ fitted[:3, :3].T + fitted[:3, 3]
                segment.update(
                    aligned_ate_translation_m=error_statistics(
                        np.linalg.norm(aligned - references, axis=1)
                    ),
                    alignment_status="SE3_fixed_scale",
                )
        segments.append(segment)
    rpe = {}
    for interval in intervals:
        errors = []
        crossed = 0
        unavailable = 0
        for index in range(max(0, len(rows) - interval)):
            source, target = rows[index], rows[index + interval]
            if source.segment_id != target.segment_id:
                crossed += 1
                continue
            if not reference_valid[index] or not reference_valid[index + interval]:
                unavailable += 1
                continue
            estimated = np.linalg.inv(target.T_SC_D) @ source.T_SC_D
            reference = np.linalg.inv(target.reference_pose) @ source.reference_pose
            # Relative poses can exceed tolerance even if both endpoint poses pass.
            if not is_rigid(reference):
                unavailable += 1
                continue
            errors.append(pose_error(estimated, reference))
        rpe[str(interval)] = dict(
            _summarize_errors(errors),
            candidate_pairs=max(0, len(rows) - interval),
            excluded_cross_segment=crossed,
            excluded_reference=unavailable,
        )
    pair_errors = []
    initial_errors = []
    final_errors = []
    for index, row in enumerate(rows[1:], start=1):
        entry = {
            "source_frame": index - 1,
            "target_frame": index,
            "initial": None,
            "final": None,
        }
        if (
            reference_valid[index - 1]
            and reference_valid[index]
            and row.pair is not None
        ):
            reference = (
                np.linalg.inv(row.reference_pose) @ rows[index - 1].reference_pose
            )
            if is_rigid(reference):
                if row.pair.initial_T_C1C0 is not None and is_rigid(
                    row.pair.initial_T_C1C0
                ):
                    entry["initial"] = pose_error(row.pair.initial_T_C1C0, reference)
                if row.pair.T_C1C0 is not None:
                    entry["final"] = pose_error(row.pair.T_C1C0, reference)
                # Compare refinement on exactly the same accepted population.
                if entry["initial"] is not None and entry["final"] is not None:
                    initial_errors.append(entry["initial"])
                    final_errors.append(entry["final"])
        pair_errors.append(entry)
    failures = sum(row.status == "lost" for row in rows)
    transitions = max(0, len(rows) - 1)
    return {
        "frame_count": len(rows),
        "transition_count": transitions,
        "tracked_transitions": sum(row.status == "tracked" for row in rows),
        "failed_transitions": failures,
        "tracking_coverage": sum(row.status == "tracked" for row in rows) / transitions
        if transitions
        else None,
        "segment_count": len(segments),
        "largest_segment_frames": max(
            (segment["frame_count"] for segment in segments), default=0
        ),
        "reference_valid_frames": sum(reference_valid),
        "reference_missing_frames": missing,
        "reference_invalid_frames": invalid,
        "segments": segments,
        "continuous_trajectory": segments[0] if len(segments) == 1 else None,
        "rpe_by_frame_interval": rpe,
        "refinement_comparison_accepted_pairs": {
            "initial": _summarize_errors(initial_errors),
            "final": _summarize_errors(final_errors),
        },
        "pair_errors_m_deg": pair_errors,
    }
