"""Known geometry, outlier and gauge contracts for the experimental global graph."""

from dataclasses import replace

import numpy as np
import pytest

from scene_recall.multiview.pose_only import retract
from scene_recall.slam.pose_graph import (
    MetricPoseGraph,
    PoseGraphConfig,
    edge_residual,
)


def pose(x):
    result = np.eye(4)
    result[0, 3] = x
    return result


def chain(count=16):
    graph = MetricPoseGraph()
    for i in range(count):
        graph.add_node(i, 0, pose(i * 0.11))
        if i:
            graph.add_edge(i - 1, i, pose(-0.11))
    return graph


def test_analytic_edge_jacobians_at_nonzero_rotation_and_translation():
    rng = np.random.default_rng(9)
    source, target, measurement = [
        retract(np.eye(4), rng.normal(size=6) * 0.4) for _ in range(3)
    ]
    residual, Ja, Jb = edge_residual(source, target, measurement, jacobian=True)
    for which, analytic in ((0, Ja), (1, Jb)):
        numerical = np.zeros((6, 6))
        for j in range(6):
            delta = np.eye(6)[j] * 1e-6
            positive, negative = (
                [source.copy(), target.copy()],
                [source.copy(), target.copy()],
            )
            positive[which] = retract(positive[which], delta)
            negative[which] = retract(negative[which], -delta)
            numerical[:, j] = (
                edge_residual(*positive, measurement)
                - edge_residual(*negative, measurement)
            ) / 2e-6
        np.testing.assert_allclose(analytic, numerical, atol=1e-7)
    assert np.linalg.norm(residual) > 0.1


def test_loop_reduces_known_drift_without_moving_root():
    graph = chain()
    graph.add_edge(0, 15, pose(-1.575), kind="loop")
    result = graph.solve()
    assert result["final_cost"] < result["initial_cost"]
    assert abs(graph.poses[15][0, 3] - 1.575) < 0.05
    np.testing.assert_array_equal(graph.poses[0], np.eye(4))
    assert graph.local[15][0, 3] == pytest.approx(1.65)


def test_robust_graph_rejects_false_cycle_that_distorts_least_squares():
    robust, linear = chain(), chain()
    for graph in (robust, linear):
        graph.add_edge(0, 15, pose(-1.575), kind="loop")
        graph.add_edge(1, 14, pose(-8), kind="loop")
    diagnostic = robust.solve()
    linear.solve(robust=False)
    assert diagnostic["rejected_loops"] == [[1, 14]]
    assert abs(robust.poses[15][0, 3] - 1.575) < 0.05
    assert abs(linear.poses[15][0, 3] - 1.5) > 1


def test_gauges_remain_disconnected_and_revocation_splits_a_bridge():
    graph = MetricPoseGraph()
    for i, segment, x in ((0, 0, 0), (1, 0, 0.1), (10, 1, 0), (11, 1, 0.1)):
        graph.add_node(i, segment, pose(x))
    graph.add_edge(0, 1, pose(-0.1))
    graph.add_edge(10, 11, pose(-0.1))
    assert graph.solve()["gauge_roots"] == [0, 10]
    with pytest.raises(ValueError, match="tracking loss"):
        graph.add_edge(1, 10, pose(-1))
    graph.add_edge(1, 10, pose(-0.9), kind="loop")
    connected = graph.solve()
    assert connected["gauge_roots"] == [0]
    np.testing.assert_allclose(graph.poses[11], pose(1.1), atol=1e-9)
    graph.revoke(1, 10, "independent evidence rejects place identity")
    split = graph.solve()
    assert split["gauge_roots"] == [0, 10]
    assert split["version"] > connected["version"]
    assert split["affected_frames"] == [10, 11]
    np.testing.assert_allclose(graph.poses[11], pose(0.1), atol=1e-10)


def test_correlated_repetitions_do_not_multiply_information():
    # Multiple identical factors use different node identities with zero-motion
    # odometry. Their total loop information is bounded by one group budget.
    results = []
    for repetitions in (1, 8):
        graph = MetricPoseGraph()
        for i in range(2 * repetitions):
            graph.add_node(i, i // repetitions, pose(0))
            if i % repetitions:
                graph.add_edge(i - 1, i, np.eye(4))
        for i in range(repetitions):
            graph.add_edge(i, repetitions + i, pose(-1), kind="loop", group=(0, 1))
        scales = graph._scales(list(graph.edges.values()))
        information = sum(
            s[0] ** 2
            for e, s in zip(graph.edges.values(), scales, strict=True)
            if e.kind == "loop"
        )
        results.append(information)
    assert results[0] == pytest.approx(results[1])


def test_revisions_refresh_local_summaries_and_preserve_raw_loops():
    graph = chain(4)
    graph.add_edge(0, 3, pose(-0.3), kind="loop")
    original = graph.edges[0, 3].T_target_source.copy()
    graph.solve()
    graph.revise({2: pose(0.21)})
    np.testing.assert_array_equal(graph.edges[0, 3].T_target_source, original)
    np.testing.assert_allclose(graph.edges[1, 2].T_target_source, pose(-0.1))
    assert graph.solve()["version"] == 2


def test_inputs_owned_and_invalid_changes_are_atomic():
    graph = chain(3)
    original = graph.local[0].copy()
    with pytest.raises(ValueError):
        graph.revise({0: pose(1), 99: pose(0)})
    np.testing.assert_array_equal(graph.local[0], original)
    measurement = pose(-0.2)
    graph.add_edge(0, 2, measurement, kind="loop")
    measurement[0, 3] = 5
    assert graph.edges[0, 2].T_target_source[0, 3] == -0.2
    with pytest.raises(ValueError):
        graph.add_edge(0, 2, measurement, kind="loop")
    with pytest.raises(ValueError):
        graph.revoke(0, 2, " ")


def test_coherent_metric_bias_cannot_be_identified_by_internal_residual():
    graph = chain()
    graph.add_edge(0, 15, pose(-1.65), kind="loop")
    result = graph.solve()
    assert result["final_cost"] < 1e-12
    assert abs(graph.poses[15][0, 3] - 1.5) > 0.14


@pytest.mark.parametrize(
    "kwargs",
    [
        {"robust_scale": 0},
        {"max_nodes": True},
        {"max_iterations": 0},
        {"reject_weight": np.nan},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        replace(PoseGraphConfig(), **kwargs)


def test_single_false_bridge_has_no_cycle_evidence_for_rejection():
    graph = MetricPoseGraph()
    graph.add_node(0, 0, pose(0))
    graph.add_node(10, 1, pose(0))
    graph.add_edge(0, 10, pose(-7), kind="loop")
    diagnostic = graph.solve()
    assert diagnostic["rejected_loops"] == []
    assert diagnostic["edge_weights"] == [1]
    assert graph.poses[10][0, 3] == pytest.approx(7)


def test_robust_loss_can_reject_a_correct_loop_when_local_drift_is_large():
    graph = chain()
    graph.add_edge(0, 15, pose(-1), kind="loop")
    diagnostic = graph.solve()
    assert diagnostic["rejected_loops"] == [[0, 15]]
    assert graph.poses[15][0, 3] == pytest.approx(1.65)
