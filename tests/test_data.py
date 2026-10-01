from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from scene_recall.data import Calibration, Observation, Sequence
from scene_recall.geometry.camera import (
    backproject_depth,
    project_points,
    transform_points,
)


@pytest.fixture
def calibration() -> Calibration:
    return Calibration(
        K_R=np.array([[8.0, 0.0, 3.0], [0.0, 6.0, 2.0], [0.0, 0.0, 1.0]]),
        K_D=np.array([[2.0, 0.0, 1.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]]),
        T_RD=np.array(
            [
                [0.0, -1.0, 0.0, 0.5],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        rgb_shape=(6, 8),
        depth_shape=(2, 3),
    )


@pytest.fixture
def observation(calibration: Calibration) -> Observation:
    return Observation(
        frame_index=0,
        rgb=np.full((*calibration.rgb_shape, 3), [10, 20, 30], dtype=np.uint8),
        depth=np.array([[np.nan, 1.0, 3.0], [4.0, 5.0, 2.0]], dtype=np.float32),
    )


def test_minimal_sequence_preserves_inputs(
    calibration: Calibration, observation: Observation
) -> None:
    observations = (observation,)
    sequence = Sequence("opaque/capture:001", calibration, observations)

    assert sequence.sequence_id == "opaque/capture:001"
    assert sequence.calibration is calibration
    assert sequence.observations is observations
    assert observation.timestamp is None
    assert observation.T_WC_D is None
    assert observation.provenance is None

    # Reconstructing retains references and leaves input data and flags unchanged.
    for owner, names in [
        (calibration, ("K_R", "K_D", "T_RD")),
        (observation, ("rgb", "depth")),
    ]:
        arrays = {name: getattr(owner, name) for name in names}
        originals = {name: array.copy() for name, array in arrays.items()}
        writeable = {name: array.flags.writeable for name, array in arrays.items()}
        reconstructed = replace(owner)
        for name, array in arrays.items():
            assert getattr(reconstructed, name) is array
            np.testing.assert_array_equal(array, originals[name])
            assert array.flags.writeable == writeable[name]

    with pytest.raises(FrozenInstanceError):
        observation.frame_index = 1


def test_empty_sequence(calibration: Calibration) -> None:
    assert Sequence("empty", calibration, ()).observations == ()


def test_optional_metadata_is_retained_by_reference(observation: Observation) -> None:
    timestamp = 1_234_567_890.1234567
    pose = np.eye(4)
    provenance = {"source_id": "native-42", "original_timestamp": "uninterpreted"}

    result = replace(
        observation, timestamp=timestamp, T_WC_D=pose, provenance=provenance
    )

    assert result.timestamp is timestamp
    assert result.T_WC_D is pose
    assert result.provenance is provenance
    np.testing.assert_array_equal(pose, np.eye(4))
    assert pose.flags.writeable


@pytest.mark.parametrize("timestamp", [np.nan, np.inf, -np.inf])
def test_timestamp_must_be_finite(observation: Observation, timestamp: float) -> None:
    with pytest.raises(ValueError, match="timestamp must be finite"):
        replace(observation, timestamp=timestamp)


@pytest.mark.parametrize("timestamp", [1, True, "1.0"])
def test_timestamp_requires_float(observation: Observation, timestamp: object) -> None:
    with pytest.raises(TypeError, match="timestamp must be a float"):
        replace(observation, timestamp=timestamp)


@pytest.mark.parametrize("timestamp", [0.0, -2.0])
def test_timestamp_has_no_required_origin(
    observation: Observation, timestamp: float
) -> None:
    assert replace(observation, timestamp=timestamp).timestamp == timestamp


@pytest.mark.parametrize("field", ["K_R", "K_D", "T_RD", "T_WC_D"])
def test_matrices_require_float64_numpy_arrays(
    calibration: Calibration, observation: Observation, field: str
) -> None:
    owner = observation if field == "T_WC_D" else calibration
    matrix = np.eye(4 if field.startswith("T_") else 3)

    for invalid in (matrix.astype(np.float32), matrix.tolist()):
        with pytest.raises(TypeError, match=field):
            replace(owner, **{field: invalid})


@pytest.mark.parametrize("field", ["K_R", "K_D", "T_RD", "T_WC_D"])
def test_matrix_shape(
    calibration: Calibration, observation: Observation, field: str
) -> None:
    owner = observation if field == "T_WC_D" else calibration
    with pytest.raises(ValueError, match=f"{field} must have shape"):
        replace(owner, **{field: np.eye(2)})


@pytest.mark.parametrize("field", ["K_R", "K_D", "T_RD", "T_WC_D"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_matrix_values_must_be_finite(
    calibration: Calibration, observation: Observation, field: str, value: float
) -> None:
    owner = observation if field == "T_WC_D" else calibration
    matrix = np.eye(4 if field.startswith("T_") else 3)
    matrix[0, 2] = value
    with pytest.raises(ValueError, match=f"{field} must contain only finite"):
        replace(owner, **{field: matrix})


@pytest.mark.parametrize("field", ["K_R", "K_D"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_positive_focal_lengths(
    calibration: Calibration, field: str, axis: int, value: float
) -> None:
    matrix = getattr(calibration, field).copy()
    matrix[axis, axis] = value
    with pytest.raises(ValueError, match="focal lengths must be positive"):
        replace(calibration, **{field: matrix})


@pytest.mark.parametrize("field", ["K_R", "K_D"])
@pytest.mark.parametrize("position", [(0, 1), (1, 0), (2, 0), (2, 1), (2, 2)])
def test_zero_skew_pinhole_form(
    calibration: Calibration, field: str, position: tuple[int, int]
) -> None:
    matrix = getattr(calibration, field).copy()
    matrix[position] += 1e-8
    with pytest.raises(ValueError, match="zero-skew pinhole form"):
        replace(calibration, **{field: matrix})


def test_principal_point_can_be_outside_grid(calibration: Calibration) -> None:
    matrix = calibration.K_D.copy()
    matrix[0, 2] = -10.0
    matrix[1, 2] = 100.0
    assert replace(calibration, K_D=matrix).K_D is matrix


@pytest.mark.parametrize("field", ["rgb_shape", "depth_shape"])
@pytest.mark.parametrize("shape", [(0, 3), (-1, 3), (2, 0)])
def test_grid_dimensions_must_be_positive(
    calibration: Calibration, field: str, shape: tuple[int, int]
) -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        replace(calibration, **{field: shape})


@pytest.mark.parametrize("field", ["rgb_shape", "depth_shape"])
@pytest.mark.parametrize("shape", [[2, 3], (2,), (2, 3, 4), (2.0, 3), (True, 3)])
def test_grid_shape_requires_two_integer_dimensions(
    calibration: Calibration, field: str, shape: object
) -> None:
    with pytest.raises(TypeError, match=field):
        replace(calibration, **{field: shape})


@pytest.mark.parametrize("field", ["T_RD", "T_WC_D"])
@pytest.mark.parametrize("column", [0, 1, 2, 3])
def test_exact_homogeneous_last_row(
    calibration: Calibration, observation: Observation, field: str, column: int
) -> None:
    owner = observation if field == "T_WC_D" else calibration
    matrix = np.eye(4)
    matrix[3, column] += 1e-8
    with pytest.raises(ValueError, match="homogeneous last row"):
        replace(owner, **{field: matrix})


@pytest.mark.parametrize("field", ["T_RD", "T_WC_D"])
@pytest.mark.parametrize(
    ("rotation", "message"),
    [
        (np.diag([1.0, 1.0, -1.0]), "determinant"),
        (np.diag([1.0, 1.0, 2.0]), "orthonormal"),
        (np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "orthonormal"),
        (np.eye(3) * (1.0 + 2e-6), "orthonormal"),
        (np.eye(3) * (1.0 + 4e-7), "determinant"),
    ],
)
def test_transform_rotation_must_be_proper_and_within_tolerance(
    calibration: Calibration,
    observation: Observation,
    field: str,
    rotation: np.ndarray,
    message: str,
) -> None:
    owner = observation if field == "T_WC_D" else calibration
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    with pytest.raises(ValueError, match=message):
        replace(owner, **{field: matrix})


@pytest.mark.parametrize("field", ["T_RD", "T_WC_D"])
def test_rotation_tolerance_accepts_small_numerical_error_without_modification(
    calibration: Calibration, observation: Observation, field: str
) -> None:
    owner = observation if field == "T_WC_D" else calibration
    matrix = np.eye(4)
    matrix[:3, :3] *= 1.0 + 2e-7
    original = matrix.copy()

    result = replace(owner, **{field: matrix})

    assert getattr(result, field) is matrix
    np.testing.assert_array_equal(matrix, original)


@pytest.mark.parametrize("field", ["rgb", "depth"])
def test_view_dtypes_and_array_types(observation: Observation, field: str) -> None:
    array = getattr(observation, field)
    for invalid in (array.astype(np.float64), array.tolist()):
        with pytest.raises(TypeError, match=field):
            replace(observation, **{field: invalid})


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (0, 3, 3), (2, 0, 3)])
def test_rgb_shape(observation: Observation, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="rgb"):
        replace(observation, rgb=np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize("shape", [(3,), (2, 3, 1), (0, 3), (2, 0)])
def test_depth_shape(observation: Observation, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="depth"):
        replace(observation, depth=np.ones(shape, dtype=np.float32))


@pytest.mark.parametrize("value", [0.0, -0.0, -1.0, np.inf, -np.inf])
def test_depth_rejects_noncanonical_values(
    observation: Observation, value: float
) -> None:
    depth = observation.depth.copy()
    depth[0, 0] = value
    original = depth.copy()

    with pytest.raises(ValueError, match="finite and positive, or NaN"):
        replace(observation, depth=depth)

    np.testing.assert_array_equal(depth, original)


def test_all_nan_depth_is_valid(observation: Observation) -> None:
    depth = np.full(observation.depth.shape, np.nan, dtype=np.float32)
    assert replace(observation, depth=depth).depth is depth


@pytest.mark.parametrize("index", [1.0, True, "0"])
def test_frame_index_requires_integer(observation: Observation, index: object) -> None:
    with pytest.raises(TypeError, match="frame_index must be an integer"):
        replace(observation, frame_index=index)


def test_frame_index_must_be_nonnegative(observation: Observation) -> None:
    with pytest.raises(ValueError, match="frame_index must be nonnegative"):
        replace(observation, frame_index=-1)


@pytest.mark.parametrize("indices", [(1,), (0, 2), (0, 0), (1, 0)])
def test_sequence_requires_contiguous_ordered_indices(
    calibration: Calibration, observation: Observation, indices: tuple[int, ...]
) -> None:
    observations = tuple(replace(observation, frame_index=index) for index in indices)
    with pytest.raises(ValueError, match="zero-based and contiguous"):
        Sequence("capture", calibration, observations)


def test_sequence_accepts_multiple_observations_without_timestamp_ordering(
    calibration: Calibration, observation: Observation
) -> None:
    observations = (
        replace(observation, timestamp=2.0),
        replace(observation, frame_index=1, timestamp=1.0),
    )
    assert Sequence("capture", calibration, observations).observations is observations


@pytest.mark.parametrize("field", ["rgb", "depth"])
def test_sequence_checks_every_observation_grid(
    calibration: Calibration, observation: Observation, field: str
) -> None:
    array = getattr(observation, field)
    mismatched = replace(observation, frame_index=1, **{field: array[:, :-1]})
    with pytest.raises(ValueError, match=f"observation 1 {field} shape"):
        Sequence("capture", calibration, (observation, mismatched))


def test_sequence_and_provenance_container_types(
    calibration: Calibration, observation: Observation
) -> None:
    with pytest.raises(TypeError, match="sequence_id"):
        Sequence(1, calibration, ())
    with pytest.raises(TypeError, match="calibration"):
        Sequence("capture", None, ())
    with pytest.raises(TypeError, match="observations must be a tuple"):
        Sequence("capture", calibration, [observation])
    with pytest.raises(TypeError, match="Observation instances"):
        Sequence("capture", calibration, (None,))
    with pytest.raises(TypeError, match="provenance"):
        replace(observation, provenance="source")


def test_identity_extrinsics_allow_different_grids_and_intrinsics(
    calibration: Calibration, observation: Observation
) -> None:
    aligned = replace(calibration, T_RD=np.eye(4))
    sequence = Sequence("aligned", aligned, (observation,))

    assert sequence.calibration.rgb_shape != sequence.calibration.depth_shape
    assert not np.array_equal(sequence.calibration.K_R, sequence.calibration.K_D)


def test_contract_fields_integrate_with_geometry(
    calibration: Calibration, observation: Observation
) -> None:
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    sequence = Sequence("canonical", calibration, (replace(observation, T_WC_D=pose),))
    view = sequence.observations[0]
    shared = sequence.calibration

    points_D = backproject_depth(view.depth, shared.K_D)
    pixels_R = project_points(transform_points(points_D, shared.T_RD), shared.K_R)
    points_W = transform_points(points_D, view.T_WC_D)

    assert points_D.shape == (*shared.depth_shape, 3)
    np.testing.assert_array_equal(points_D[..., 2], view.depth)
    np.testing.assert_allclose(pixels_R[1, 2], [3.0, 5.0])
    np.testing.assert_allclose(points_W[1, 2], [2.0, 2.5, 5.0])
