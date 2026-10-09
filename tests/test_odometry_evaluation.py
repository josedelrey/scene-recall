import cv2
import numpy as np
import pytest

from scene_recall.odometry.evaluation import evaluate_trajectory, pose_error
from scene_recall.odometry.pipeline import PairEstimate, TrajectoryFrame


def trajectory(scale=1):
    poses = []
    rows = []
    gauge = np.eye(4)
    gauge[:3, :3] = cv2.Rodrigues(np.array([0.2, -0.1, 0.3]))[0]
    gauge[:3, 3] = [4, -2, 1]
    for index, position in enumerate(
        ([0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 1])
    ):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(
            np.array([0.03 * index, 0.02 * index, -0.01 * index])
        )[0]
        pose[:3, 3] = position
        reference = gauge @ pose
        pose[:3, 3] *= scale
        pair = (
            None
            if not poses
            else PairEstimate(
                np.linalg.inv(pose) @ poses[-1],
                initial_T_C1C0=np.linalg.inv(pose) @ poses[-1],
            )
        )
        poses.append(pose)
        rows.append(
            TrajectoryFrame(
                index,
                None,
                0,
                "initialized" if not index else "tracked",
                pose,
                reference,
                pair,
            )
        )
    return rows


def test_exact_trajectory_is_invariant_to_world_gauge():
    rows = trajectory()
    report = evaluate_trajectory(iter(rows), [1, 2, 10])
    assert report["tracking_coverage"] == 1
    assert report["reference_valid_frames"] == 5
    segment = report["continuous_trajectory"]
    assert segment["aligned_ate_translation_m"]["rmse"] < 1e-14
    assert segment["anchored_errors"]["rotation_deg"]["rmse"] < 1e-12
    assert segment["endpoint_translation_error_m"] < 1e-14
    for interval, expected_count in [("1", 4), ("2", 3), ("10", 0)]:
        metrics = report["rpe_by_frame_interval"][interval]
        assert metrics["translation_m"]["count"] == expected_count
        if expected_count:
            assert metrics["translation_m"]["rmse"] < 1e-14
            assert metrics["rotation_deg"]["rmse"] < 1e-12
        else:
            assert metrics["translation_m"]["rmse"] is None


def test_metric_scale_drift_is_not_removed_by_trajectory_alignment():
    report = evaluate_trajectory(trajectory(scale=1.1), [1])
    segment = report["continuous_trajectory"]
    assert segment["alignment_status"] == "SE3_fixed_scale"
    assert segment["aligned_ate_translation_m"]["rmse"] > 0.05
    assert segment["endpoint_translation_error_m"] > 0.2
    assert report["rpe_by_frame_interval"]["1"]["translation_m"][
        "rmse"
    ] == pytest.approx(0.1)


def test_failure_break_excludes_cross_segment_errors():
    rows = trajectory()
    anchor = rows[2].T_SC_D
    for index in range(2, 5):
        previous = rows[index]
        rows[index] = TrajectoryFrame(
            index,
            None,
            1,
            "lost" if index == 2 else "tracked",
            np.linalg.inv(anchor) @ previous.T_SC_D,
            previous.reference_pose,
            PairEstimate(None, "lost") if index == 2 else previous.pair,
        )
    report = evaluate_trajectory(rows, [1, 2])
    assert report["continuous_trajectory"] is None
    assert report["segment_count"] == 2
    assert report["failed_transitions"] == 1
    assert report["tracking_coverage"] == 0.75
    assert report["rpe_by_frame_interval"]["1"]["excluded_cross_segment"] == 1
    assert report["rpe_by_frame_interval"]["1"]["translation_m"]["count"] == 3
    assert report["rpe_by_frame_interval"]["2"]["excluded_cross_segment"] == 2
    for segment in report["segments"]:
        assert segment["anchored_errors"]["translation_m"]["rmse"] < 1e-14


def test_missing_and_nonrigid_references_are_counted_without_repair():
    rows = trajectory()
    reflection = rows[2].reference_pose.copy()
    reflection[:3, 0] *= -1
    original = reflection.copy()
    for index, reference in ((0, None), (2, reflection)):
        previous = rows[index]
        rows[index] = TrajectoryFrame(
            index, None, 0, previous.status, previous.T_SC_D, reference, previous.pair
        )
    report = evaluate_trajectory(rows, [1, 2])
    assert report["reference_missing_frames"] == 1
    assert report["reference_invalid_frames"] == 1
    assert report["reference_valid_frames"] == 3
    assert report["continuous_trajectory"]["anchor_frame"] == 1
    assert report["rpe_by_frame_interval"]["1"]["translation_m"]["count"] == 1
    assert report["rpe_by_frame_interval"]["1"]["excluded_reference"] == 3
    np.testing.assert_array_equal(reflection, original)


def test_collinear_position_alignment_is_explicitly_unavailable():
    rows = []
    for index in range(4):
        pose = np.eye(4)
        pose[0, 3] = index
        rows.append(
            TrajectoryFrame(
                index,
                None,
                0,
                "initialized" if not index else "tracked",
                pose,
                pose.copy(),
            )
        )
    report = evaluate_trajectory(rows)
    segment = report["continuous_trajectory"]
    assert segment["alignment_status"] == "insufficient_reference_geometry"
    assert segment["aligned_ate_translation_m"]["count"] == 0
    assert segment["anchored_errors"]["translation_m"]["rmse"] == 0


def test_refinement_comparison_uses_identical_accepted_population():
    rows = trajectory()
    last = rows[-1]
    initial = last.pair.T_C1C0.copy()
    initial[:3, 3] += [0.1, 0, 0]
    rows[-1] = TrajectoryFrame(
        last.frame_index,
        None,
        0,
        "tracked",
        last.T_SC_D,
        last.reference_pose,
        PairEstimate(last.pair.T_C1C0, initial_T_C1C0=initial),
    )
    report = evaluate_trajectory(rows)
    comparison = report["refinement_comparison_accepted_pairs"]
    assert comparison["initial"]["translation_m"]["count"] == 4
    assert comparison["final"]["translation_m"]["count"] == 4
    assert comparison["initial"]["translation_m"]["rmse"] == pytest.approx(0.05)
    assert comparison["final"]["translation_m"]["rmse"] < 1e-14


def test_rotation_errors_at_identity_and_pi():
    assert pose_error(np.eye(4), np.eye(4)) == (0, 0)
    rotated = np.eye(4)
    rotated[:3, :3] = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    rotated[0, 3] = 2
    assert pose_error(rotated, np.eye(4)) == pytest.approx((2, 180))


@pytest.mark.parametrize("intervals", [[0], [1, 1], [True]])
def test_invalid_intervals_are_rejected(intervals):
    with pytest.raises(ValueError, match="interval"):
        evaluate_trajectory([], intervals)


def test_empty_and_unreferenced_trajectories_produce_unavailable_metrics():
    report = evaluate_trajectory([])
    assert report["continuous_trajectory"] is None
    assert report["tracking_coverage"] is None
    rows = [TrajectoryFrame(0, None, 0, "initialized", np.eye(4), None)]
    report = evaluate_trajectory(rows)
    assert report["reference_missing_frames"] == 1
    assert report["continuous_trajectory"]["anchor_frame"] is None
    assert (
        report["continuous_trajectory"]["anchored_errors"]["translation_m"]["count"]
        == 0
    )
