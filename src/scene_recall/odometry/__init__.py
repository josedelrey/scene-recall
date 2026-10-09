"""Dataset-independent selectable RGB-D odometry and trajectory evaluation."""

from scene_recall.odometry.pipeline import (
    PairEstimate,
    PairwiseBackend,
    TrajectoryFrame,
    track_observations,
)

__all__ = ["PairEstimate", "PairwiseBackend", "TrajectoryFrame", "track_observations"]

from scene_recall.odometry.hybrid import HybridConfig, HybridRGBDBackend
from scene_recall.odometry.icp import DepthICPBackend, ICPConfig
from scene_recall.odometry.sparse import SparseConfig, SparseRGBDBackend

__all__ += [
    "DepthICPBackend",
    "HybridConfig",
    "HybridRGBDBackend",
    "ICPConfig",
    "SparseConfig",
    "SparseRGBDBackend",
]
