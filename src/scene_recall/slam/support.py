"""Metric bridge support without adding a likelihood or claiming covariance."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupportConfig:
    enabled: bool = False
    minimum: int = 24
    retained_fraction: float = 0.5
    minimum_spread: float = 0.02

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("support policy must be boolean")
        if (
            isinstance(self.minimum, bool)
            or not isinstance(self.minimum, int)
            or self.minimum < 3
        ):
            raise ValueError("minimum bridge support must be an integer >= 3")
        for name in ("retained_fraction", "minimum_spread"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")


def metric_support(points):
    """Centered rigid registration information from equal-weight 3D points.

    Eliminating translation gives rotational information trace(C) I - C.
    One point leaves three rotations free and collinear points leave one free.
    The normalized lever arm is scale invariant. This is a support proxy with
    fixed correspondences and metric points, not the rank of the full RGB-D BA.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return {"count": 0, "rotation_rank": 0, "spread": 0.0}
    centered = points - points.mean(axis=0)
    C = centered.T @ centered / len(points)
    values = np.maximum(0, np.linalg.eigvalsh(np.trace(C) * np.eye(3) - C))
    scale = max(float(np.median(np.linalg.norm(points, axis=1))), 1e-12)
    tolerance = max(float(values[-1]) * 1e-9, scale * scale * 1e-12)
    return {
        "count": len(points),
        "rotation_rank": int(np.count_nonzero(values > tolerance)),
        "spread": float(np.sqrt(values[0]) / scale),
    }
