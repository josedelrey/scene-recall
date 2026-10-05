"""Single-frame extraction for processed ScanNet v2 RGB-D captures.

Binary field order follows the official ScanNet SensReader implementation:
https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
Registered-view calibration follows the official processing pipeline:
https://github.com/ScanNet/ScanNet/blob/master/Calibrate/src/calibration.h

This independently written Python 3 reader supports only version 4, JPEG color,
and zlib uint16 depth. It reads one frame and seeks past preceding payloads.
"""

import struct
import zlib
from dataclasses import dataclass
from io import BytesIO
from os import fstat
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from scene_recall.data import Calibration, Observation, Sequence


@dataclass(frozen=True)
class _SensHeader:
    sensor_name: str
    intrinsic_color: np.ndarray
    extrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    extrinsic_depth: np.ndarray
    rgb_shape: tuple[int, int]
    depth_shape: tuple[int, int]
    depth_shift: float
    frame_count: int


@dataclass(frozen=True)
class _SensFrame:
    camera_to_world: np.ndarray
    timestamp_color: int
    timestamp_depth: int
    color_data: bytes
    depth_data: bytes


def _read_exact(stream: BinaryIO, size: int, file_size: int) -> bytes:
    if size > file_size - stream.tell():
        raise ValueError("truncated .sens file: field exceeds remaining bytes")
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated .sens file: incomplete field")
    return data


def _read_matrix(stream: BinaryIO, file_size: int) -> np.ndarray:
    return np.frombuffer(_read_exact(stream, 64, file_size), dtype="<f4").reshape(4, 4)


def _read_sens_frame(path: Path, source_index: int) -> tuple[_SensHeader, _SensFrame]:
    """Extract native fields without decoding images or interpreting geometry."""
    with path.open("rb") as stream:
        file_size = fstat(stream.fileno()).st_size
        version = struct.unpack("<I", _read_exact(stream, 4, file_size))[0]
        if version != 4:
            raise ValueError(f"unsupported .sens version {version}, expected 4")
        name_size = struct.unpack("<Q", _read_exact(stream, 8, file_size))[0]
        try:
            sensor_name = _read_exact(stream, name_size, file_size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid .sens sensor name encoding") from error
        intrinsic_color = _read_matrix(stream, file_size)
        extrinsic_color = _read_matrix(stream, file_size)
        intrinsic_depth = _read_matrix(stream, file_size)
        extrinsic_depth = _read_matrix(stream, file_size)
        color_compression, depth_compression = struct.unpack(
            "<ii", _read_exact(stream, 8, file_size)
        )
        if (color_compression, depth_compression) != (2, 1):
            raise ValueError(
                "unsupported .sens compression: require JPEG and zlib uint16"
            )
        color_width, color_height, depth_width, depth_height = struct.unpack(
            "<4I", _read_exact(stream, 16, file_size)
        )
        depth_shift = struct.unpack("<f", _read_exact(stream, 4, file_size))[0]
        frame_count = struct.unpack("<Q", _read_exact(stream, 8, file_size))[0]
        if source_index >= frame_count:
            raise IndexError(
                f"source frame index {source_index} outside {frame_count} frames"
            )
        if frame_count > (file_size - stream.tell()) // 96:
            raise ValueError("truncated .sens file: declared frame headers cannot fit")
        header = _SensHeader(
            sensor_name,
            intrinsic_color,
            extrinsic_color,
            intrinsic_depth,
            extrinsic_depth,
            (color_height, color_width),
            (depth_height, depth_width),
            depth_shift,
            frame_count,
        )
        for index in range(source_index + 1):
            camera_to_world = _read_matrix(stream, file_size)
            timestamp_color, timestamp_depth, color_size, depth_size = struct.unpack(
                "<4Q", _read_exact(stream, 32, file_size)
            )
            if color_size == 0 or depth_size == 0:
                raise ValueError(
                    "malformed .sens frame: RGB and depth payloads are required"
                )
            frame_end = stream.tell() + color_size + depth_size
            if frame_end > file_size:
                raise ValueError(
                    "truncated .sens file: frame payload exceeds remaining bytes"
                )
            if index < source_index:
                stream.seek(frame_end)
                continue
            frame = _SensFrame(
                camera_to_world,
                timestamp_color,
                timestamp_depth,
                _read_exact(stream, color_size, file_size),
                _read_exact(stream, depth_size, file_size),
            )
            return header, frame
    raise AssertionError("validated source index must select a frame")


def _registered_calibration(header: _SensHeader) -> Calibration:
    """Require evidence of ScanNet's processed, color-frame depth registration."""
    if header.sensor_name != "StructureSensor (calibrated)":
        raise ValueError(
            "unsupported calibration: require calibrated StructureSensor views"
        )
    if not all(
        np.array_equal(extrinsic, np.eye(4))
        for extrinsic in (header.extrinsic_color, header.extrinsic_depth)
    ):
        raise ValueError(
            "unsupported calibration: registered-view extrinsics must be identity"
        )
    if any(size < 2 for size in (*header.rgb_shape, *header.depth_shape)):
        raise ValueError(
            "unsupported calibration: registered grid dimensions must be at least 2"
        )
    for intrinsic in (header.intrinsic_color, header.intrinsic_depth):
        if not np.array_equal(intrinsic[3], [0, 0, 0, 1]) or np.any(
            intrinsic[:3, 3] != 0
        ):
            raise ValueError(
                "unsupported calibration: require embedded pinhole intrinsics"
            )
    calibration = Calibration(
        K_R=header.intrinsic_color[:3, :3].astype(np.float64),
        K_D=header.intrinsic_depth[:3, :3].astype(np.float64),
        T_RD=np.eye(4, dtype=np.float64),
        rgb_shape=header.rgb_shape,
        depth_shape=header.depth_shape,
    )
    h_r, w_r = header.rgb_shape
    h_d, w_d = header.depth_shape
    expected = calibration.K_R.copy()
    expected[0, 0] *= w_d / w_r
    expected[1, 1] *= h_d / h_r
    expected[0, 2] *= (w_d - 1) / (w_r - 1)
    expected[1, 2] *= (h_d - 1) / (h_r - 1)
    # Allow float32 serialization rounding in the upstream registration rule.
    if not np.allclose(calibration.K_D, expected, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "unsupported calibration: depth intrinsics do not match registration"
        )
    return calibration


def _decode_rgb(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "JPEG":
                raise ValueError("malformed RGB payload: expected JPEG")
            if image.size != (shape[1], shape[0]):
                raise ValueError("RGB payload dimensions do not match .sens header")
            return np.array(image.convert("RGB"), dtype=np.uint8)
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("malformed JPEG RGB payload") from error


def _decode_depth(data: bytes, header: _SensHeader) -> np.ndarray:
    if not np.isfinite(header.depth_shift) or header.depth_shift <= 0:
        raise ValueError("depth_shift must be finite and positive")
    expected_size = 2 * header.depth_shape[0] * header.depth_shape[1]
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(data, expected_size + 1)
    except zlib.error as error:
        raise ValueError("malformed zlib depth payload") from error
    if (
        len(raw) != expected_size
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        raise ValueError(
            "malformed depth payload: require one complete zlib uint16 grid"
        )
    native = np.frombuffer(raw, dtype="<u2").reshape(header.depth_shape)
    depth = native.astype(np.float32)
    depth /= header.depth_shift
    depth[native == 0] = np.nan
    return depth


def load_scannet_frame(
    sens_path: str | Path, original_frame_index: int = 0
) -> Sequence:
    """Return one canonical observation from a processed ScanNet v2 .sens file.

    Only version 4 JPEG/zlib captures labeled ``StructureSensor (calibrated)``
    with identity extrinsics and the pipeline's registered intrinsic relationship
    are accepted. Native grids are preserved. C_D and C_R both refer to the
    registered color optical frame, and W is the stored reconstruction world.
    The companion .txt file is never read.

    ``original_frame_index`` is the zero-based position in the source stream.
    The returned observation has canonical frame_index=0, with the source index
    and raw timestamps retained in provenance. Nonzero depth timestamps are
    interpreted as microseconds on the source clock. Zero means unavailable.
    RGB and depth are associated by their source frame record.

    The all-negative-infinity lost-tracking pose sentinel becomes None with a
    provenance reason. Finite poses with a homogeneous last row are preserved
    as stored, apart from conversion to float64. Rotation quality is not checked,
    and poses are never repaired or projected to SO(3).
    Invalid indices raise TypeError or IndexError. Unsupported or malformed
    source data raises ValueError. Filesystem errors propagate as OSError.

    Only preceding records and the selected record are inspected. Subsequent
    frames and the optional IMU stream are outside this extraction's scope.
    """
    if not isinstance(original_frame_index, int) or isinstance(
        original_frame_index, bool
    ):
        raise TypeError("original_frame_index must be a Python integer")
    if original_frame_index < 0:
        raise IndexError("original_frame_index must be nonnegative")
    path = Path(sens_path).expanduser()
    header, frame = _read_sens_frame(path, original_frame_index)
    calibration = _registered_calibration(header)
    provenance = {
        "dataset": "scannet_v2",
        "source_file": str(path),
        "original_frame_index": original_frame_index,
        "sensor_name": header.sensor_name,
        "sens_version": 4,
        "source_frame_count": header.frame_count,
        "color_compression": "jpeg",
        "depth_compression": "zlib_ushort",
        "depth_shift": header.depth_shift,
        "timestamp_color": frame.timestamp_color,
        "timestamp_depth": frame.timestamp_depth,
        "timestamp_unit": "microseconds",
        "calibration_source": "sens_header",
    }
    pose = frame.camera_to_world.astype(np.float64)
    if np.isneginf(pose).all():
        pose = None
        provenance["pose_unavailable_reason"] = (
            "ScanNet lost-tracking sentinel (all -inf)"
        )
    timestamp = frame.timestamp_depth / 1_000_000 if frame.timestamp_depth else None
    if timestamp is None:
        provenance["timestamp_unavailable_reason"] = "source depth timestamp is zero"
    observation = Observation(
        frame_index=0,
        rgb=_decode_rgb(frame.color_data, header.rgb_shape),
        depth=_decode_depth(frame.depth_data, header),
        timestamp=timestamp,
        T_WC_D=pose,
        provenance=provenance,
    )
    return Sequence(path.stem, calibration, (observation,))
