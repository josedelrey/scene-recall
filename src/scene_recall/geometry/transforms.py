"""Rigid-transform validation with explicit consumer tolerances."""

import numpy as np


def is_rigid(T: np.ndarray, atol: float = 1e-4) -> bool:
    """Assess a transform without repairing or changing the supplied array."""
    return bool(
        T.shape == (4, 4)
        and np.isfinite(T).all()
        and np.array_equal(T[3], [0, 0, 0, 1])
        and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), rtol=0, atol=atol)
        and np.isclose(np.linalg.det(T[:3, :3]), 1, rtol=0, atol=atol)
    )
