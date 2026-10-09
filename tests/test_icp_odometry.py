"""Acceptance, frame conventions, and failure policies for depth and hybrid VO."""

from dataclasses import replace

import numpy as np
import pytest
from test_depth_registration import render_room

from scene_recall.data import Calibration, Observation
from scene_recall.odometry import (
    DepthICPBackend,
    HybridConfig,
    HybridRGBDBackend,
    ICPConfig,
    PairEstimate,
    track_observations,
)
from scene_recall.odometry.evaluation import evaluate_trajectory, pose_error
from scene_recall.odometry.icp import increment


def room_pair():
    K = np.array([[360, 0, 159.5], [0, 360, 127.5], [0, 0, 1.0]])
    shape = (256, 320)
    truth = increment(np.array([0.018, -0.012, 0.008, 0.007, -0.004, 0.003]))
    calibration = Calibration(K, K, np.eye(4), shape, shape)
    blank = np.zeros((*shape, 3), np.uint8)
    source = Observation(0, blank, render_room(K, shape, np.eye(4)))
    target = Observation(1, blank, render_room(K, shape, np.linalg.inv(truth)))
    return source, target, calibration, truth


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_depth_only_recovers_motion_with_blank_rgb_and_unrelated_references(direction):
    source, target, calibration, truth = room_pair()
    if direction == "backward":
        source, target = target, source
        truth = np.linalg.inv(truth)
    # Depth-only estimation is independent of RGB calibration and reference motion.
    rig = increment(np.array([0.3, 0.1, 0.2, 0.02, 0.03, 0.04]))
    calibration = replace(calibration, T_RD=rig)
    backend = DepthICPBackend()
    result = backend.estimate(source, target, calibration)
    assert result.reason is None
    translation, rotation = pose_error(result.T_C1C0, truth)
    assert translation < 0.001
    assert rotation < 0.04
    unrelated = increment(np.array([1, 2, 3, 0.1, 0.2, 0.3]))
    repeated = DepthICPBackend().estimate(
        replace(source, rgb=np.full_like(source.rgb, 255), T_WC_D=unrelated),
        replace(target, T_WC_D=np.eye(4)),
        calibration,
    )
    np.testing.assert_array_equal(repeated.T_C1C0, result.T_C1C0)
    assert result.initial_T_C1C0 is None
    assert result.diagnostics["icp_rank"] == 6
    assert result.diagnostics["icp_common_test_support"] > 30


@pytest.mark.parametrize("kind", ["missing", "plane", "iteration_limit"])
def test_depth_failures_never_accumulate_candidates(kind):
    source, target, calibration, _ = room_pair()
    config = ICPConfig()
    if kind != "iteration_limit":
        depth = np.full_like(source.depth, np.nan if kind == "missing" else 2)
        source, target = replace(source, depth=depth), replace(target, depth=depth)
    else:
        config = ICPConfig(max_iterations=1)
    rows = list(
        track_observations([source, target], calibration, DepthICPBackend(config))
    )
    pair = rows[1].pair
    assert pair.T_C1C0 is None
    assert (
        pair.reason
        == {
            "missing": "insufficient_support",
            "plane": "rank_deficient",
            "iteration_limit": "iteration_limit",
        }[kind]
    )
    assert pair.candidate_T_C1C0 is not None
    assert rows[1].status == "lost"
    assert rows[1].segment_id == 1
    np.testing.assert_array_equal(rows[1].T_SC_D, np.eye(4))
    assert (
        evaluate_trajectory(rows)["rpe_by_frame_interval"]["1"][
            "excluded_cross_segment"
        ]
        == 1
    )


class StubSparse:
    def __init__(self, pair):
        self.pair = pair

    def estimate(self, *args):
        return self.pair


class StubDepth:
    def __init__(self, pair):
        self.pair = pair
        self.initial = None

    def refine(self, source, target, calibration, initial):
        self.initial = initial
        return self.pair


@pytest.mark.parametrize("policy", ["sparse", "lost"])
def test_hybrid_refinement_failure_preserves_sparse_stage_and_explicit_policy(policy):
    source, target, calibration, truth = room_pair()
    ransac = np.eye(4)
    candidate = increment(np.array([0.1, 0, 0, 0, 0, 0]))
    backend = HybridRGBDBackend(HybridConfig(refinement_failure=policy))
    backend._sparse = StubSparse(PairEstimate(truth, initial_T_C1C0=ransac))
    backend._depth = StubDepth(
        PairEstimate(None, "rank_deficient", candidate_T_C1C0=candidate)
    )
    result = backend.estimate(source, target, calibration)
    np.testing.assert_array_equal(backend._depth.initial, truth)
    np.testing.assert_array_equal(result.initial_T_C1C0, truth)
    np.testing.assert_array_equal(result.ransac_T_C1C0, ransac)
    np.testing.assert_array_equal(result.candidate_T_C1C0, candidate)
    assert result.diagnostics["geometric_refinement_reason"] == "rank_deficient"
    if policy == "sparse":
        np.testing.assert_array_equal(result.T_C1C0, truth)
        assert result.reason is None
        assert result.diagnostics["geometric_refinement"] == "fallback_sparse"
    else:
        assert result.T_C1C0 is None
        assert result.reason == "ICP refinement failed: rank_deficient"


def test_hybrid_sparse_failure_skips_icp():
    source, target, calibration, _ = room_pair()
    backend = HybridRGBDBackend()
    backend._sparse = StubSparse(PairEstimate(None, "LM refinement failed", np.eye(4)))
    backend._depth = StubDepth(PairEstimate(np.eye(4)))
    result = backend.estimate(source, target, calibration)
    assert result.T_C1C0 is None
    assert result.initial_T_C1C0 is None
    assert result.ransac_T_C1C0 is not None
    assert backend._depth.initial is None
    assert result.diagnostics["geometric_refinement"] == "not_run"


def test_hybrid_refines_complete_sparse_pose_without_using_references():
    source, target, calibration, truth = room_pair()
    sparse = increment(np.array([0.004, -0.002, 0.001, 0.002, -0.001, 0])) @ truth
    backend = HybridRGBDBackend()
    backend._sparse = StubSparse(PairEstimate(sparse, initial_T_C1C0=np.eye(4)))
    source = replace(source, T_WC_D=np.eye(4))
    target = replace(target, T_WC_D=np.linalg.inv(truth))
    rows = list(track_observations([source, target], calibration, backend))
    assert rows[1].pair.diagnostics["geometric_refinement"] == "refined"
    assert pose_error(rows[1].pair.T_C1C0, truth)[0] < 0.001
    evaluation = evaluate_trajectory(rows)
    comparison = evaluation["geometric_refinement_comparison"]
    assert comparison["translation_improved_pairs"] == 1
    assert comparison["initial"]["translation_m"]["count"] == 1
    assert evaluation["pair_errors_m_deg"][0]["ransac"] is not None


def test_hybrid_excessive_correction_is_rejected():
    source, target, calibration, truth = room_pair()
    backend = HybridRGBDBackend()
    backend._sparse = StubSparse(PairEstimate(truth))
    candidate = increment(np.array([0.2, 0, 0, 0, 0, 0])) @ truth
    backend._depth = StubDepth(PairEstimate(candidate, candidate_T_C1C0=candidate))
    result = backend.estimate(source, target, calibration)
    np.testing.assert_array_equal(result.T_C1C0, truth)
    assert result.diagnostics["geometric_refinement_reason"] == "excessive_correction"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"max_iterations": True},
        {"min_support": 5},
        {"min_support_fraction": np.nan},
        {"min_support_fraction": 1.1},
        {"normal_gate_deg": 90},
        {"huber_m": 0},
        {"max_depth_m": 0.2},
        {"translation_tolerance_m": np.inf},
    ],
)
def test_invalid_icp_config(kwargs):
    with pytest.raises(ValueError):
        ICPConfig(**kwargs)


def test_depth_grid_validation():
    source, target, calibration, _ = room_pair()
    with pytest.raises(ValueError, match="depth shape"):
        DepthICPBackend().estimate(
            source, target, replace(calibration, depth_shape=(10, 10))
        )


def test_lower_depth_residual_does_not_imply_reference_accuracy():
    source, target, calibration, truth = room_pair()
    candidate = increment(np.array([0.01, 0, 0, 0, 0, 0])) @ truth
    backend = HybridRGBDBackend()
    backend._sparse = StubSparse(PairEstimate(truth))
    backend._depth = StubDepth(
        PairEstimate(
            candidate,
            diagnostics={
                "icp_common_test_initial_rmse_m": 0.01,
                "icp_common_test_final_rmse_m": 0.005,
            },
            candidate_T_C1C0=candidate,
        )
    )
    rows = list(
        track_observations(
            [
                replace(source, T_WC_D=np.eye(4)),
                replace(target, T_WC_D=np.linalg.inv(truth)),
            ],
            calibration,
            backend,
        )
    )
    comparison = evaluate_trajectory(rows)["geometric_refinement_comparison"]
    assert rows[1].pair.diagnostics["geometric_refinement"] == "refined"
    assert comparison["initial"]["translation_m"]["rmse"] < 1e-12
    assert comparison["final"]["translation_m"]["rmse"] > 0.009
    assert comparison["translation_improved_pairs"] == 0


def test_fallback_is_excluded_from_geometric_improvement_population():
    source, target, calibration, truth = room_pair()
    backend = HybridRGBDBackend()
    backend._sparse = StubSparse(PairEstimate(truth))
    backend._depth = StubDepth(
        PairEstimate(None, "iteration_limit", candidate_T_C1C0=truth)
    )
    rows = list(
        track_observations(
            [
                replace(source, T_WC_D=np.eye(4)),
                replace(target, T_WC_D=np.linalg.inv(truth)),
            ],
            calibration,
            backend,
        )
    )
    evaluation = evaluate_trajectory(rows)
    assert (
        evaluation["refinement_comparison_accepted_pairs"]["initial"]["translation_m"][
            "count"
        ]
        == 1
    )
    assert (
        evaluation["geometric_refinement_comparison"]["initial"]["translation_m"][
            "count"
        ]
        == 0
    )
    assert evaluation["pair_errors_m_deg"][0]["candidate"] is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"refinement_failure": "unknown"},
        {"max_correction_translation_m": 0},
        {"max_correction_rotation_deg": np.nan},
    ],
)
def test_invalid_hybrid_policy(kwargs):
    with pytest.raises(ValueError):
        HybridConfig(**kwargs)


def test_hybrid_requires_component_configs():
    with pytest.raises(TypeError):
        HybridConfig(icp=object())
