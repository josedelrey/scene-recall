"""Collective support retains correlations invisible in conditional blocks."""

from dataclasses import replace

import numpy as np
import pytest

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.batch_ba import compact_blocks, compact_linearize
from scene_recall.multiview.fixed_landmarks import DirectFactors
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.slam.observability import rigid_cut_information
from scene_recall.slam.shared_linearization import (
    ObservationCache,
    RelinearizationConfig,
)
from scene_recall.slam.shared_solver import SchurOperator


def bridge(shared=1):
    rng = np.random.default_rng(901)
    points = rng.uniform([-0.6, -0.5, 2], [0.6, 0.5, 3], (24, 3))
    tracks, views = [], []
    for camera in range(4):
        ids = (
            list(range(12)) if camera < 2 else list(range(12, 24)) + list(range(shared))
        )
        tracks.extend(ids)
        views.extend([camera] * len(ids))
    views, tracks = np.array(views), np.array(tracks)
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, 0, 3] = np.arange(4) * 0.04
    native = np.array(
        [
            transform_points(points[track], np.linalg.inv(poses[camera]))
            for camera, track in zip(views, tracks, strict=True)
        ]
    )
    K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]])
    factors = DirectFactors(
        views,
        tracks,
        points[tracks],
        project_points(native, K),
        native,
        np.repeat(np.eye(4)[None], len(views), axis=0),
        np.repeat((np.eye(3) * 40)[None], len(views), axis=0),
        np.ones(len(views)),
    )
    ids, inverse, seeds = landmark_layout(factors)
    return poses, factors, K, ids, inverse, seeds


@pytest.mark.parametrize("shared,rank", [(0, 0), (1, 3), (2, 5), (3, 6)])
def test_collective_cut_detects_missing_rigid_constraints(shared, rank):
    poses, factors, K, ids, inverse, seeds = bridge(shared)
    views = factors.view
    r, JP, JL = compact_linearize(poses, seeds, factors, K, inverse)
    blocks = compact_blocks(
        r, JP, JL, np.ones(r.shape[:2]), inverse, views, len(ids), 4
    )
    inside = np.array([False, False, True, True])
    report = rigid_cut_information(blocks, views, inverse, inside)
    assert report["rank"] == rank
    operator = SchurOperator(blocks, views, inverse, (), 0)
    Z = np.tile(np.eye(6), (4, 1))
    Z[:12] = 0
    S = np.column_stack([operator.multiply(row) for row in np.eye(24)])
    H = Z.T @ S @ Z
    values, vectors = np.linalg.eigh(blocks[0][inside].sum(axis=0))
    W = vectors / np.sqrt(values)
    np.testing.assert_allclose(
        report["normalized_eigenvalues"],
        np.linalg.eigvalsh(W.T @ H @ W),
        atol=1e-12,
    )


def test_centering_a_one_point_cut_does_not_amplify_null_column_roundoff():
    poses, factors, K, _, _, _ = bridge()
    mask = factors.track_id == 0
    factors = replace(
        factors, **{n: getattr(factors, n)[mask] for n in factors.__dataclass_fields__}
    )
    ids, inverse, points = landmark_layout(factors)
    r, JP, JL = compact_linearize(poses, points, factors, K, inverse)
    blocks = compact_blocks(
        r, JP, JL, np.ones(r.shape[:2]), inverse, factors.view, len(ids), 4
    )
    cut = rigid_cut_information(
        blocks, factors.view, inverse, [False, False, True, True], center=points[0]
    )
    assert cut["rank"] == cut["pose_energy_rank"] == 3
    np.testing.assert_allclose(cut["normalized_eigenvalues"][:3], 0, atol=1e-12)


@pytest.mark.parametrize("transport", [False, True])
def test_mixed_cache_anchors_must_not_constrain_an_islands_rotation(transport):
    poses, factors, K, ids, inverse, points = bridge()
    cache = ObservationCache(
        RelinearizationConfig(
            translation_m=0.002, point_m=0.1, gauge_transport=transport
        )
    )
    cache.bind(factors, K, "native")
    cache.linearize(poses, points, inverse)
    points[12:] += np.random.default_rng(71).uniform(-0.005, 0.005, (12, 3))
    poses[2, 0, 3] += 0.003
    r, JP, JL = cache.linearize(poses, points, inverse)
    blocks = compact_blocks(
        r, JP, JL, np.ones(r.shape[:2]), inverse, factors.view, len(ids), 4
    )
    cut = rigid_cut_information(
        blocks, factors.view, inverse, [False, False, True, True]
    )
    assert cut["rank"] == (3 if transport else 6)
