"""Candidate ranking, separate-view evidence and dense revision lineage."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_local_map import scene
from test_slam_recovery import empty_features, translated

from scene_recall.slam.atlas import AtlasConfig
from scene_recall.slam.loop_confirmation import LoopConfirmation, LoopConfirmationConfig
from scene_recall.slam.pose_graph import MetricPoseGraph
from scene_recall.slam.retrieval import descriptor_votes


def propose(bank, a, b, displacement=1):
    return bank.propose(
        a,
        b,
        0,
        1,
        translated(a / 10),
        translated((b - 30) / 10),
        translated(-displacement),
    )


def test_confirmation_requires_both_observations_to_change():
    bank = LoopConfirmation()
    assert propose(bank, 0, 30)[0] == "tentative"
    assert propose(bank, 0, 31, 1.1)[0] == "needs_distinct_views"
    status, measurements = propose(bank, 1, 31)
    assert status == "confirmed"
    assert [m[:2] for m in measurements] == [(0, 30), (1, 31)]
    assert measurements[0][3] == measurements[1][3]
    assert propose(bank, 2, 32)[0] == "supported"
    assert propose(bank, 3, 33, 4)[0] == "inconsistent_confirmation"


def test_inconsistent_pending_can_be_replaced_without_admitting_an_edge():
    bank = LoopConfirmation()
    propose(bank, 0, 30)
    status, measurements = propose(bank, 1, 31, 4)
    assert status == "inconsistent_confirmation"
    assert measurements == []
    assert not bank.confirmed
    assert propose(bank, 2, 32, 4)[0] == "confirmed"


def test_evidence_group_boundary_requires_new_confirmation():
    bank = LoopConfirmation()
    propose(bank, 0, 30)
    propose(bank, 1, 31)
    status, measurements = propose(bank, 2, 60)
    assert status == "tentative"
    assert measurements == []


def test_votes_prioritize_source_metric_support_without_requiring_target_depth():
    _, _, features, _ = scene(2)
    a, b = features[0], features[1]
    count = len(a.points_R)
    assert descriptor_votes(a, b) == (count, count)
    b = replace(b, points_R=np.full_like(b.points_R, np.nan))
    assert descriptor_votes(a, b) == (count, count)
    a = replace(a, points_R=np.full_like(a.points_R, np.nan))
    assert descriptor_votes(a, b) == (0, count)
    assert descriptor_votes(empty_features(), b) == (0, 0)


def test_dense_corrections_do_not_interpolate_across_tracking_loss(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    from slam_loop_graph import dense_poses

    from scene_recall.odometry.pipeline import TrajectoryFrame

    graph = MetricPoseGraph()
    for frame, segment, x in ((0, 0, 0), (2, 0, 0.2), (3, 1, 0), (5, 1, 0.2)):
        graph.add_node(frame, segment, translated(x))
    graph.poses = {
        0: translated(0),
        2: translated(0.1),
        3: translated(10),
        5: translated(10.1),
    }
    graph.components = {0: 0, 2: 0, 3: 3, 5: 3}
    rows = [
        TrajectoryFrame(
            i, None, i // 3, "tracked", translated((i % 3) / 10), None, None
        )
        for i in range(6)
    ]
    poses, components = dense_poses(graph, rows)
    np.testing.assert_allclose(poses[:, 0, 3], [0, 0.05, 0.1, 10, 10.05, 10.1])
    np.testing.assert_array_equal(components, [0, 0, 0, 3, 3, 3])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cell_frames": 0},
        {"cell_frames": True},
        {"translation_m": np.nan},
        {"rotation_deg": 0},
    ],
)
def test_confirmation_config_validates_inputs(kwargs):
    with pytest.raises(ValueError):
        LoopConfirmationConfig(**kwargs)


def test_atlas_rejects_unknown_retrieval_mode():
    with pytest.raises(ValueError):
        AtlasConfig(retrieval="unknown")


def test_disconnected_source_explanations_cause_batch_abstention():
    bank = LoopConfirmation()
    candidates = [
        (a, 60, segment, 2, translated(0), translated(0), translated(-1))
        for a, segment in ((0, 0), (30, 1))
    ]
    statuses, added = bank.propose_batch(candidates)
    assert statuses == ["ambiguous_components", "ambiguous_components"]
    assert added == []
    assert not bank.pending
    assert not bank.confirmed


def test_confirmed_relationships_resolve_source_component_ambiguity():
    bank = LoopConfirmation()
    propose(bank, 0, 30)
    propose(bank, 1, 31)
    candidates = [
        (a, 60, segment, 2, translated(0), translated(0), translated(-1))
        for a, segment in ((0, 0), (30, 1))
    ]
    statuses, added = bank.propose_batch(candidates)
    assert statuses == ["tentative", "tentative"]
    assert added == []
    assert len(bank.pending) == 2
