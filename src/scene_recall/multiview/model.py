"""Explicit identities and fixed-pose measurements for multiview geometry."""

from dataclasses import dataclass

import numpy as np

from scene_recall.data import Calibration
from scene_recall.features import RGBDFeatures
from scene_recall.geometry.transforms import is_rigid
from scene_recall.odometry.pipeline import TrajectoryFrame

FeatureKey = tuple[int, int]


@dataclass(frozen=True)
class TrackConfig:
    """Strict gates, with an audit-only relaxed band for pose sensitivity.

    Thresholds are engineering tolerances, not calibrated noise covariances.
    Relaxed matches never merge identities. No trajectory is optimized.
    """

    ratio: float = 0.7
    max_features: int = 1500
    depth_edge_threshold_m: float | None = 0.05
    reprojection_px: float = 3.0
    depth_absolute_m: float = 0.03
    depth_relative: float = 0.01
    relaxed_factor: float = 2.0
    pair_offsets: tuple[int, ...] = (1, 2, 5, 10, 20)
    min_views: int = 3
    min_depth_views: int = 2

    def __post_init__(self) -> None:
        for name in ("max_features", "min_views", "min_depth_views"):
            value = getattr(self, name)
            lower = 0 if name == "max_features" else 2
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if not np.isfinite(self.ratio) or not 0 < self.ratio < 1:
            raise ValueError("ratio must be strictly between zero and one")
        for name in (
            "reprojection_px",
            "depth_absolute_m",
            "depth_relative",
            "relaxed_factor",
            "depth_edge_threshold_m",
        ):
            value = getattr(self, name)
            if value is None and name == "depth_edge_threshold_m":
                continue
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.relaxed_factor < 1:
            raise ValueError("relaxed_factor must be >= 1")
        if (
            not isinstance(self.pair_offsets, tuple)
            or not self.pair_offsets
            or any(
                isinstance(k, bool) or not isinstance(k, int) or k < 1
                for k in self.pair_offsets
            )
            or tuple(sorted(set(self.pair_offsets))) != self.pair_offsets
        ):
            raise ValueError("pair_offsets must be sorted unique positive integers")


@dataclass(frozen=True, eq=False)
class FeatureFrame:
    """One extracted observation in an estimated segment, without references.

    T_SR maps RGB optical coordinates to segment S. Frame indices can be sparse
    when sampling a trajectory. Feature keys are (frame_index, feature_row).
    """

    frame_index: int
    segment_id: int
    timestamp: float | None
    T_SR: np.ndarray
    features: RGBDFeatures

    def __post_init__(self) -> None:
        for name in ("frame_index", "segment_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.timestamp is not None and not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite or None")
        if not is_rigid(self.T_SR, atol=1e-6):
            raise ValueError("T_SR must be rigid")

    @classmethod
    def from_trajectory(
        cls, row: TrajectoryFrame, calibration: Calibration, features: RGBDFeatures
    ) -> "FeatureFrame":
        """Use only the estimated depth-camera pose and canonical extrinsic."""
        if not is_rigid(calibration.T_RD, atol=1e-6):
            raise ValueError("multiview requires rigid T_RD")
        return cls(
            row.frame_index,
            row.segment_id,
            row.timestamp,
            row.T_SC_D @ np.linalg.inv(calibration.T_RD),
            features,
        )


@dataclass(frozen=True, eq=False)
class Landmark:
    """A metric segment-frame point and its unique per-view measurements.

    supported means >= min_views with >= min_depth_views, all passing strict
    gates under the supplied poses. tentative means less independent support.
    Neither label is a probability of correct identity or static-scene truth.
    """

    track_id: int
    segment_id: int
    observations: tuple[FeatureKey, ...]
    point_S: np.ndarray
    status: str
    depth_views: int
    reprojection_errors_px: np.ndarray
    depth_errors_m: np.ndarray
    parallax_deg: float
    baseline_m: float


@dataclass(frozen=True)
class Association:
    """A direct descriptor proposal and its final identity decision.

    accepted joins components. redundant provides a direct cycle check.
    uncertain, rejected, conflict and incompatible never join identities.
    Residuals describe the endpoint check before track-level revalidation.
    """

    source: FeatureKey
    target: FeatureKey
    ratio: float
    reprojection_px: float | None
    depth_m: float | None
    status: str
    reason: str


@dataclass(frozen=True, eq=False)
class TrackResult:
    frames: tuple[FeatureFrame, ...]
    landmarks: tuple[Landmark, ...]
    associations: tuple[Association, ...]
    pairs: tuple[tuple[int, int], ...]
    config: TrackConfig
    timings: dict[str, float]
