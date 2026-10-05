import struct
import zlib
from io import BufferedReader, BytesIO, FileIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scene_recall.data import Sequence
from scene_recall.datasets.scannet import load_scannet_frame
from scene_recall.geometry.camera import backproject_depth, transform_points

K_COLOR = np.array([[8, 0, 3.5], [0, 9, 2.5], [0, 0, 1]], dtype=np.float32)
K_DEPTH = np.array([[4, 0, 1.5], [0, 4.5, 1], [0, 0, 1]], dtype=np.float32)
POSE = np.array(
    [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float32
)
NATIVE_DEPTH = (np.arange(12, dtype=np.uint16) * 500).reshape(3, 4)


def _intrinsic(K: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = K
    return matrix


def _jpeg(color: tuple[int, int, int], format: str = "JPEG") -> bytes:
    stream = BytesIO()
    rgb = np.full((6, 8, 3), color, dtype=np.uint8)
    Image.fromarray(rgb).save(stream, format=format, quality=100, subsampling=0)
    return stream.getvalue()


@pytest.fixture
def sens_factory(tmp_path: Path):
    """Write tiny little-endian version-4 captures without external data."""

    def write(header_changes=None, frame_changes=None):
        header = {
            "version": 4,
            "sensor_name": b"StructureSensor (calibrated)",
            "intrinsic_color": _intrinsic(K_COLOR),
            "extrinsic_color": np.eye(4, dtype=np.float32),
            "intrinsic_depth": _intrinsic(K_DEPTH),
            "extrinsic_depth": np.eye(4, dtype=np.float32),
            "compression": (2, 1),
            "dimensions": (8, 6, 4, 3),
            "depth_shift": 1000.0,
            "frame_count": 2,
        }
        header.update(header_changes or {})
        frames = [
            {
                "pose": POSE,
                "timestamps": (0, 0),
                "color": _jpeg((210, 60, 20)),
                "depth": zlib.compress(NATIVE_DEPTH.astype("<u2").tobytes()),
            },
            {
                "pose": POSE,
                "timestamps": (2_499_000, 2_500_000),
                "color": _jpeg((20, 210, 60)),
                "depth": zlib.compress(np.full((3, 4), 2000, dtype="<u2").tobytes()),
            },
        ]
        for index, changes in (frame_changes or {}).items():
            frames[index].update(changes)
        output = BytesIO()
        output.write(struct.pack("<IQ", header["version"], len(header["sensor_name"])))
        output.write(header["sensor_name"])
        for name in (
            "intrinsic_color",
            "extrinsic_color",
            "intrinsic_depth",
            "extrinsic_depth",
        ):
            output.write(np.asarray(header[name], dtype="<f4").tobytes())
        output.write(struct.pack("<ii", *header["compression"]))
        output.write(struct.pack("<4I", *header["dimensions"]))
        output.write(struct.pack("<fQ", header["depth_shift"], header["frame_count"]))
        for frame in frames:
            output.write(np.asarray(frame["pose"], dtype="<f4").tobytes())
            output.write(
                struct.pack(
                    "<4Q",
                    *frame["timestamps"],
                    len(frame["color"]),
                    len(frame["depth"]),
                )
            )
            output.write(frame["color"])
            output.write(frame["depth"])
        output.write(struct.pack("<Q", 0))
        path = tmp_path / "scene_synthetic.sens"
        path.write_bytes(output.getvalue())
        return path

    return write


def test_canonical_sequence_and_geometry(sens_factory) -> None:
    path = sens_factory()
    sequence = load_scannet_frame(path)
    assert isinstance(sequence, Sequence)
    assert sequence.sequence_id == "scene_synthetic"
    assert len(sequence.observations) == 1
    calibration = sequence.calibration
    observation = sequence.observations[0]
    assert observation.rgb.shape == (6, 8, 3)
    assert observation.depth.shape == (3, 4)
    assert observation.rgb.dtype == np.uint8
    assert observation.depth.dtype == np.float32
    np.testing.assert_allclose(observation.rgb[0, 0], [210, 60, 20], atol=2)
    assert calibration.rgb_shape == (6, 8)
    assert calibration.depth_shape == (3, 4)
    for matrix in (
        calibration.K_R,
        calibration.K_D,
        calibration.T_RD,
        observation.T_WC_D,
    ):
        assert matrix.dtype == np.float64
    np.testing.assert_array_equal(calibration.K_R, K_COLOR)
    np.testing.assert_array_equal(calibration.K_D, K_DEPTH)
    np.testing.assert_array_equal(calibration.T_RD, np.eye(4))
    np.testing.assert_array_equal(observation.T_WC_D, POSE)
    np.testing.assert_array_equal(
        transform_points(np.array([[1.0, 0.0, 2.0]]), observation.T_WC_D), [[1, 3, 5]]
    )
    expected_depth = NATIVE_DEPTH.astype(np.float32) / 1000
    expected_depth[0, 0] = np.nan
    np.testing.assert_array_equal(observation.depth, expected_depth)
    points = backproject_depth(observation.depth, calibration.K_D)
    assert points.shape == (3, 4, 3)
    assert np.isnan(points[0, 0]).all()
    np.testing.assert_array_equal(points[..., 2], observation.depth)
    assert observation.frame_index == 0
    assert observation.timestamp is None
    assert observation.provenance["original_frame_index"] == 0
    assert observation.provenance["timestamp_color"] == 0
    assert observation.provenance["timestamp_depth"] == 0
    assert observation.provenance["source_file"] == str(path)
    assert observation.provenance["timestamp_unavailable_reason"]


def test_selection_skips_undecodable_preceding_payloads(sens_factory) -> None:
    path = sens_factory(frame_changes={0: {"color": b"not JPEG", "depth": b"not zlib"}})
    observation = load_scannet_frame(str(path), 1).observations[0]
    assert observation.frame_index == 0
    assert observation.provenance["original_frame_index"] == 1
    assert observation.provenance["timestamp_color"] == 2_499_000
    assert observation.provenance["timestamp_depth"] == 2_500_000
    assert observation.timestamp == 2.5
    np.testing.assert_array_equal(observation.depth, np.full((3, 4), 2.0))
    np.testing.assert_allclose(observation.rgb[0, 0], [20, 210, 60], atol=2)


def test_preceding_payloads_are_skipped_without_reading(
    sens_factory, monkeypatch
) -> None:
    path = sens_factory(
        frame_changes={0: {"color": b"x" * 100_000, "depth": b"y" * 100_000}}
    )

    class CountingReader(BufferedReader):
        bytes_read = 0

        def read(self, size=-1):
            assert size >= 0, "the reader must never request the entire capture"
            data = super().read(size)
            self.bytes_read += len(data)
            return data

    reader = CountingReader(FileIO(path, "rb"))
    monkeypatch.setattr(Path, "open", lambda self, mode: reader)
    sequence = load_scannet_frame(path, 1)
    assert sequence.observations[0].provenance["original_frame_index"] == 1
    assert reader.bytes_read < 10_000
    assert reader.closed


def test_extraction_does_not_decode_later_payloads(sens_factory) -> None:
    path = sens_factory(frame_changes={1: {"color": b"not JPEG", "depth": b"not zlib"}})
    assert len(load_scannet_frame(path).observations) == 1


def test_stored_depth_shift_is_used(sens_factory) -> None:
    path = sens_factory(header_changes={"depth_shift": 500})
    depth = load_scannet_frame(path).observations[0].depth
    assert np.isnan(depth[0, 0])
    assert depth[0, 1] == 1.0
    assert depth[-1, -1] == 11.0


def test_unavailable_depth_timestamp_with_available_color_timestamp(
    sens_factory,
) -> None:
    path = sens_factory(frame_changes={0: {"timestamps": (123456, 0)}})
    observation = load_scannet_frame(path).observations[0]
    assert observation.timestamp is None
    assert observation.provenance["timestamp_color"] == 123456


def test_invalid_tracking_pose_is_optional(sens_factory) -> None:
    path = sens_factory(frame_changes={0: {"pose": np.full((4, 4), -np.inf)}})
    observation = load_scannet_frame(path).observations[0]
    assert observation.T_WC_D is None
    assert observation.provenance["pose_unavailable_reason"] == (
        "ScanNet lost-tracking sentinel (all -inf)"
    )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_other_nonfinite_poses_are_rejected(sens_factory, bad_value) -> None:
    pose = POSE.copy()
    pose[2, 2] = bad_value
    path = sens_factory(frame_changes={0: {"pose": pose}})
    with pytest.raises(ValueError, match="T_WC_D must contain only finite"):
        load_scannet_frame(path)


@pytest.mark.parametrize("value", [1.000002, 2.0, -1.0, 0.0])
def test_finite_poses_are_preserved_without_rotation_repair(
    sens_factory, value
) -> None:
    pose = POSE.copy()
    pose[2, 2] = value
    path = sens_factory(frame_changes={0: {"pose": pose}})
    observation = load_scannet_frame(path).observations[0]

    assert observation.T_WC_D.dtype == np.float64
    np.testing.assert_array_equal(observation.T_WC_D, pose.astype(np.float64))


@pytest.mark.parametrize("column", [0, 1, 2, 3])
def test_pose_requires_exact_homogeneous_last_row(sens_factory, column) -> None:
    pose = POSE.copy()
    pose[3, column] += 0.01
    path = sens_factory(frame_changes={0: {"pose": pose}})
    with pytest.raises(ValueError, match="T_WC_D must have homogeneous last row"):
        load_scannet_frame(path)


@pytest.mark.parametrize("index", [-1, 2, 100])
def test_out_of_range_indices(sens_factory, index) -> None:
    with pytest.raises(IndexError):
        load_scannet_frame(sens_factory(), index)


@pytest.mark.parametrize("index", [True, 1.0, "0", None, np.int64(0)])
def test_invalid_index_types(sens_factory, index) -> None:
    with pytest.raises(TypeError, match="original_frame_index"):
        load_scannet_frame(sens_factory(), index)


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 3},
        {"version": 5},
        {"compression": (0, 1)},
        {"compression": (1, 1)},
        {"compression": (2, 0)},
        {"compression": (2, 2)},
    ],
)
def test_unsupported_formats(sens_factory, changes) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        load_scannet_frame(sens_factory(header_changes=changes))


@pytest.mark.parametrize("shift", [0, -1000, np.nan, np.inf])
def test_invalid_depth_shift(sens_factory, shift) -> None:
    with pytest.raises(ValueError, match="depth_shift"):
        load_scannet_frame(sens_factory(header_changes={"depth_shift": shift}))


@pytest.mark.parametrize(
    "changes",
    [
        {"sensor_name": b"StructureSensor"},
        {"sensor_name": b"OtherSensor (calibrated)"},
        {"extrinsic_color": POSE},
        {"extrinsic_depth": POSE},
        {"intrinsic_depth": _intrinsic(K_COLOR)},
        {"dimensions": (8, 6, 1, 3)},
        {"dimensions": (0, 6, 4, 3)},
        {"intrinsic_color": np.ones((4, 4))},
    ],
)
def test_unsupported_registered_calibration(sens_factory, changes) -> None:
    with pytest.raises(ValueError, match="unsupported calibration"):
        load_scannet_frame(sens_factory(header_changes=changes))


@pytest.mark.parametrize("size", [0, 3, 11, 30, 100, 300, 400])
def test_truncated_binary_fields(sens_factory, size) -> None:
    path = sens_factory()
    path.write_bytes(path.read_bytes()[:size])
    with pytest.raises(ValueError, match="truncated"):
        load_scannet_frame(path)


def test_truncated_selected_payload(sens_factory) -> None:
    path = sens_factory()
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ValueError, match="truncated"):
        load_scannet_frame(path, 1)


def test_impossible_frame_count(sens_factory) -> None:
    with pytest.raises(ValueError, match="frame headers"):
        load_scannet_frame(sens_factory(header_changes={"frame_count": 1000}))


def test_empty_capture(sens_factory) -> None:
    with pytest.raises(IndexError):
        load_scannet_frame(sens_factory(header_changes={"frame_count": 0}))


def test_invalid_sensor_name_encoding(sens_factory) -> None:
    with pytest.raises(ValueError, match="sensor name encoding"):
        load_scannet_frame(sens_factory(header_changes={"sensor_name": b"\xff"}))


@pytest.mark.parametrize("payload", [b"broken", b"", _jpeg((1, 2, 3), "PNG")])
def test_malformed_color_payload(sens_factory, payload) -> None:
    with pytest.raises(ValueError, match="payload"):
        load_scannet_frame(sens_factory(frame_changes={0: {"color": payload}}))


def test_jpeg_dimensions_must_match_header(sens_factory) -> None:
    stream = BytesIO()
    Image.new("RGB", (2, 2)).save(stream, format="JPEG")
    with pytest.raises(ValueError, match="dimensions"):
        load_scannet_frame(
            sens_factory(frame_changes={0: {"color": stream.getvalue()}})
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"broken",
        b"",
        zlib.compress(b"short"),
        zlib.compress(b"long" * 100),
        zlib.compress(NATIVE_DEPTH.astype("<u2").tobytes())[:-1],
        zlib.compress(NATIVE_DEPTH.astype("<u2").tobytes()) + b"trailing",
    ],
)
def test_malformed_depth_payload(sens_factory, payload) -> None:
    with pytest.raises(ValueError, match="payload"):
        load_scannet_frame(sens_factory(frame_changes={0: {"depth": payload}}))


def test_companion_metadata_is_not_used(sens_factory) -> None:
    path = sens_factory()
    path.with_suffix(".txt").write_text("deliberately invalid companion metadata")
    sequence = load_scannet_frame(path)
    np.testing.assert_array_equal(sequence.calibration.T_RD, np.eye(4))
    np.testing.assert_array_equal(sequence.observations[0].T_WC_D, POSE)
