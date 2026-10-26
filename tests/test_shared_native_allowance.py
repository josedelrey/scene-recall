"""Recover frozen sensor allowances without using reference trajectories."""

import runpy
from pathlib import Path

import numpy as np

from scene_recall.multiview.temporal import PairMotion, native_whitening

recover_allowance = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/audit_shared_candidates.py")
)["recover_allowance"]


def test_native_covariance_recovers_shared_translation_and_rotation_allowances():
    points = np.array([[0.2, -0.3, 2], [-0.7, 0.2, 3], [0.1, 0.6, 4]])
    K = np.diag([525.0, 525.0, 1])
    motion = PairMotion(np.eye(4), "controlled", True, 0.003, 0.006)
    whitening = native_whitening(points, K, motion)
    translation, rotation = recover_allowance(points, whitening, K)
    np.testing.assert_allclose([translation, rotation], [0.003, 0.006], atol=1e-12)
    reconstructed = native_whitening(
        points, K, PairMotion(np.eye(4), "recovered", True, translation, rotation)
    )
    np.testing.assert_allclose(reconstructed, whitening, atol=1e-10)
