"""Dataset-independent canonical RGB-D data.

Constructors validate inputs without converting or copying them. Field bindings
are frozen, but arrays and provenance are retained by reference. Callers must not
mutate them after construction, including through other references.

Both canonical optical frames use +x right, +y down, and +z forward. Adapters
must establish image-grid calibration, depth units and convention, channel order,
and pose frame conventions before constructing these objects. Structural checks
cannot verify those source-data semantics. Transform validation checks dtype,
shape, finiteness, and the homogeneous last row. Rotation quality is a policy
for consumers that require SO(3). Constructors do not repair transforms.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray


def _validate_array(name: str, array: np.ndarray, dtype: type[np.generic]) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if isinstance(array, np.ma.MaskedArray):
        raise TypeError(f"{name} must not be a masked array")
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}")


def _validate_matrix(name: str, matrix: np.ndarray, shape: tuple[int, int]) -> None:
    _validate_array(name, matrix, np.float64)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_grid_shape(name: str, shape: tuple[int, int]) -> None:
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError(f"{name} must be a (height, width) tuple")
    if any(not isinstance(size, int) or isinstance(size, bool) for size in shape):
        raise TypeError(f"{name} dimensions must be integers")
    if any(size <= 0 for size in shape):
        raise ValueError(f"{name} dimensions must be positive")


def _validate_intrinsics(name: str, K: NDArray[np.float64]) -> None:
    _validate_matrix(name, K, (3, 3))
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"{name} focal lengths must be positive")
    if K[0, 1] != 0 or K[1, 0] != 0 or not np.array_equal(K[2], [0, 0, 1]):
        raise ValueError(f"{name} must have zero-skew pinhole form")


def _validate_transform(name: str, T: NDArray[np.float64]) -> None:
    """Check homogeneous structure without assessing or repairing rotation."""
    _validate_matrix(name, T, (4, 4))
    if not np.array_equal(T[3], [0, 0, 0, 1]):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")


@dataclass(frozen=True, eq=False)
class Calibration:
    """Shared calibration for fixed canonical RGB and depth grids.

    K_R and K_D are float64 (3, 3) zero-skew pinhole matrices for rgb_shape
    and depth_shape respectively. Shapes are (height, width), and intrinsic
    parameters are in pixels. The principal point may lie outside the grid.

    T_RD is a float64 (4, 4) transform from canonical C_D to canonical C_R,
    with translation in meters. Supply identity when these frames coincide,
    even if their grids differ. Rotation quality is not checked or repaired.
    Pinhole structure and the homogeneous last row are checked exactly.
    Canonical frames and calibration need not match native sensor frames.
    """

    K_R: NDArray[np.float64]
    K_D: NDArray[np.float64]
    T_RD: NDArray[np.float64]
    rgb_shape: tuple[int, int]
    depth_shape: tuple[int, int]

    def __post_init__(self) -> None:
        _validate_grid_shape("rgb_shape", self.rgb_shape)
        _validate_grid_shape("depth_shape", self.depth_shape)
        _validate_intrinsics("K_R", self.K_R)
        _validate_intrinsics("K_D", self.K_D)
        _validate_transform("T_RD", self.T_RD)


@dataclass(frozen=True, eq=False)
class Observation:
    """One associated canonical RGB-D pair referenced to optical frame C_D.

    rgb is uint8 (H_R, W_R, 3) in RGB channel order before model preprocessing.
    depth is float32 (H_D, W_D) z-depth in meters in canonical C_D. Each depth
    value must be finite and positive, or NaN. The two views may have different
    grids and optical frames. frame_index is a nonnegative Python integer.

    timestamp is optional finite float seconds referring to the depth view.
    Its clock origin is adapter-defined, and monotonicity is not required.
    T_WC_D is an optional float64 (4, 4) reference pose from canonical C_D to W,
    with translation in meters. Rotation quality is not checked or repaired.
    Adapters must verify a native pose's frame and transform direction before
    supplying T_WC_D. Reference poses are separate from odometry estimates.

    provenance is optional, non-normative metadata. Geometry must not depend
    on it to interpret the canonical fields.
    """

    frame_index: int
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]
    timestamp: float | None = None
    T_WC_D: NDArray[np.float64] | None = None
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        _validate_array("rgb", self.rgb, np.uint8)
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        if any(size <= 0 for size in self.rgb.shape[:2]):
            raise ValueError("rgb image dimensions must be positive")
        _validate_array("depth", self.depth, np.float32)
        if self.depth.ndim != 2 or any(size <= 0 for size in self.depth.shape):
            raise ValueError("depth must have positive shape (height, width)")
        valid = np.isnan(self.depth) | (np.isfinite(self.depth) & (self.depth > 0))
        if not valid.all():
            raise ValueError("depth values must be finite and positive, or NaN")
        if self.timestamp is not None:
            if not isinstance(self.timestamp, float):
                raise TypeError("timestamp must be a float or None")
            if not isfinite(self.timestamp):
                raise ValueError("timestamp must be finite")
        if self.T_WC_D is not None:
            _validate_transform("T_WC_D", self.T_WC_D)
        if self.provenance is not None and not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping or None")


@dataclass(frozen=True, eq=False)
class Sequence:
    """An ordered canonical RGB-D capture with shared calibration.

    sequence_id is an opaque string. observations is a tuple, possibly empty,
    with frame indices exactly 0 through len(observations) - 1 in tuple order.
    All observations use the calibration's fixed RGB and depth grid dimensions.
    """

    sequence_id: str
    calibration: Calibration
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str):
            raise TypeError("sequence_id must be a string")
        if not isinstance(self.calibration, Calibration):
            raise TypeError("calibration must be a Calibration")
        if not isinstance(self.observations, tuple):
            raise TypeError("observations must be a tuple")
        for index, observation in enumerate(self.observations):
            if not isinstance(observation, Observation):
                raise TypeError("observations must contain only Observation instances")
            if observation.frame_index != index:
                raise ValueError(
                    "frame indices must be zero-based and contiguous in order"
                )
            if observation.rgb.shape[:2] != self.calibration.rgb_shape:
                raise ValueError(
                    f"observation {index} rgb shape must match calibration"
                )
            if observation.depth.shape != self.calibration.depth_shape:
                raise ValueError(
                    f"observation {index} depth shape must match calibration"
                )
