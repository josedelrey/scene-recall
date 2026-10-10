"""Canonical reading of extracted TUM RGB-D benchmark PNG sequences.

Source conventions and the recommended registered-image pinhole policy:
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from PIL import Image

from scene_recall.data import Calibration, Observation, Sequence


@dataclass(frozen=True)
class TUMConfig:
    """Association limits in seconds, with no clock offset or pose extrapolation."""

    max_rgb_depth_difference_s: float = 0.02
    max_ground_truth_difference_s: float = 0.05

    def __post_init__(self) -> None:
        for value in (
            self.max_rgb_depth_difference_s,
            self.max_ground_truth_difference_s,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("association limits must be finite and positive")


def _table(path: Path, columns: int) -> list[list[str]]:
    rows = []
    previous = -np.inf
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != columns:
            raise ValueError(f"{path.name}:{number}: expected {columns} columns")
        try:
            timestamp = float(fields[0])
        except ValueError as error:
            raise ValueError(f"{path.name}:{number}: invalid timestamp") from error
        if not np.isfinite(timestamp) or timestamp <= previous:
            raise ValueError(f"{path.name}:{number}: timestamps must increase strictly")
        previous = timestamp
        rows.append(fields)
    return rows


def _image_table(root: Path, name: str) -> list[tuple[float, Path]]:
    rows = []
    for timestamp, filename in _table(root / name, 2):
        relative = Path(filename)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"{name}: image path must stay inside the sequence")
        rows.append((float(timestamp), path))
    return rows


def associate_timestamps(
    rgb: list[float], depth: list[float], max_difference_s: float
) -> list[tuple[int, int]]:
    """Greedily accept smallest time differences without reusing either image.

    This follows TUM's association policy, with deterministic ties and inclusive
    tolerance. Return (RGB index, depth index) in depth timestamp order.
    """
    if not np.isfinite(max_difference_s) or max_difference_s <= 0:
        raise ValueError("association tolerance must be finite and positive")
    for stamps in (rgb, depth):
        if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
            raise ValueError("source timestamps must be finite and increase strictly")
    depth_array = np.asarray(depth)
    candidates = []
    for rgb_index, stamp in enumerate(rgb):
        lo, hi = np.searchsorted(
            depth_array,
            [stamp - max_difference_s, stamp + max_difference_s],
            side="left",
        )
        # Include a candidate exactly on the upper boundary.
        hi = np.searchsorted(depth_array, stamp + max_difference_s, side="right")
        for depth_index in range(int(lo), int(hi)):
            difference = abs(stamp - depth[depth_index])
            if difference <= max_difference_s:
                candidates.append((difference, rgb_index, depth_index))
    used_rgb, used_depth = set(), set()
    pairs = []
    for _, rgb_index, depth_index in sorted(candidates):
        if rgb_index not in used_rgb and depth_index not in used_depth:
            used_rgb.add(rgb_index)
            used_depth.add(depth_index)
            pairs.append((rgb_index, depth_index))
    return sorted(pairs, key=lambda pair: pair[1])


def _quaternion_rotation(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    dot = float(first @ second)
    if dot < 0:
        second = -second
        dot = -dot
    dot = np.clip(dot, -1, 1)
    if dot > 0.9995:
        q = (1 - weight) * first + weight * second
        return q / np.linalg.norm(q)
    angle = np.arccos(dot)
    return (
        np.sin((1 - weight) * angle) * first + np.sin(weight * angle) * second
    ) / np.sin(angle)


class TUMReader:
    """Single-pass image reader with lightweight, eagerly validated timestamp tables.

    Accept an extracted official sequence directory containing rgb.txt, depth.txt,
    and optional groundtruth.txt. Native 640x480 PNG grids are preserved with
    TUM's recommended ROS-default K (525, 525, 319.5, 239.5) for both views and
    identity T_RD. This is a documented pinhole approximation, with no additional
    distortion or depth scale correction. Registered C_D equals color C_R.

    Bounds [start, stop) index the complete associated-pair list. Unmatched images
    are omitted, while pairs without ground truth remain valid observations.
    Depth timestamps are epoch seconds. References use linear translation and
    shortest-arc quaternion SLERP at that timestamp, only between nearby samples.
    Quaternion normalization allows source rounding within 1e-3 of unit norm.
    No extrapolation, axis flip, inversion, or mocap-to-camera offset is applied.
    """

    def __init__(
        self,
        sequence_path: str | Path,
        *,
        start: int = 0,
        stop: int | None = None,
        config: TUMConfig | None = None,
    ) -> None:
        self.config = config or TUMConfig()
        self._root = Path(sequence_path).expanduser().resolve()
        self.sequence_id = self._root.name
        self._rgb = _image_table(self._root, "rgb.txt")
        self._depth = _image_table(self._root, "depth.txt")
        self._pairs = associate_timestamps(
            [row[0] for row in self._rgb],
            [row[0] for row in self._depth],
            self.config.max_rgb_depth_difference_s,
        )
        self.frame_count = len(self._pairs)
        for name, value in (("start", start), ("stop", stop)):
            if value is None and name == "stop":
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be a Python integer")
            if value < 0 or value > self.frame_count:
                raise IndexError(f"{name} outside associated frame count")
        self._stop = self.frame_count if stop is None else stop
        if self._stop < start:
            raise IndexError("stop must be greater than or equal to start")
        self._start = self._index = start
        self.closed = False
        K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
        self.calibration = Calibration(K, K.copy(), np.eye(4), (480, 640), (480, 640))
        gt_path = self._root / "groundtruth.txt"
        gt = _table(gt_path, 8) if gt_path.exists() else []
        self._gt = np.array(
            [[float(value) for value in row] for row in gt], dtype=np.float64
        ).reshape(-1, 8)
        if not np.isfinite(self._gt).all():
            raise ValueError("ground truth must contain finite values")
        norms = np.linalg.norm(self._gt[:, 4:], axis=1)
        if np.any(abs(norms - 1) > 1e-3):
            raise ValueError("ground truth quaternions must be unit length within 1e-3")
        self._gt[:, 4:] /= norms[:, None]
        self.metadata = {
            "dataset": "tum_rgbd",
            "rgb_frame_count": len(self._rgb),
            "depth_frame_count": len(self._depth),
            "associated_frame_count": self.frame_count,
            "unmatched_rgb_frames": len(self._rgb) - self.frame_count,
            "unmatched_depth_frames": len(self._depth) - self.frame_count,
            "ground_truth_sample_count": len(self._gt),
            "calibration_policy": "tum_recommended_ros_default_registered_pinhole",
            "depth_units_per_meter": 5000.0,
            "depth_scale_correction": "already_applied_by_tum",
            "reference_policy": "depth_timestamp_linear_translation_shortest_arc_slerp_no_extrapolation",
        }

    def _reference(self, timestamp: float) -> tuple[np.ndarray | None, dict]:
        stamps = self._gt[:, 0]
        index = int(np.searchsorted(stamps, timestamp))
        if index < len(stamps) and stamps[index] == timestamp:
            first = second = self._gt[index]
            weight = 0.0
        elif index == 0 or index == len(stamps):
            return None, {"pose_unavailable_reason": "outside ground truth coverage"}
        else:
            first, second = self._gt[index - 1], self._gt[index]
            if (
                max(timestamp - first[0], second[0] - timestamp)
                > self.config.max_ground_truth_difference_s
            ):
                return None, {
                    "pose_unavailable_reason": "ground truth bracket exceeds tolerance"
                }
            weight = (timestamp - first[0]) / (second[0] - first[0])
        pose = np.eye(4)
        pose[:3, 3] = (1 - weight) * first[1:4] + weight * second[1:4]
        pose[:3, :3] = _quaternion_rotation(_slerp(first[4:], second[4:], weight))
        return pose, {
            "ground_truth_bracket_s": [float(first[0]), float(second[0])],
            "ground_truth_weight": float(weight),
        }

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise ValueError("TUMReader is closed")
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Observation:
        if self.closed or self._index >= self._stop:
            self.close()
            raise StopIteration
        try:
            rgb_index, depth_index = self._pairs[self._index]
            rgb_time, rgb_path = self._rgb[rgb_index]
            depth_time, depth_path = self._depth[depth_index]
            with Image.open(rgb_path) as image:
                if (
                    image.format != "PNG"
                    or image.mode != "RGB"
                    or image.size != (640, 480)
                ):
                    raise ValueError("TUM RGB must be a 640x480 RGB PNG")
                rgb = np.array(image, dtype=np.uint8)
            with Image.open(depth_path) as image:
                if (
                    image.format != "PNG"
                    or image.mode != "I;16"
                    or image.size != (640, 480)
                ):
                    raise ValueError("TUM depth must be a 640x480 uint16 PNG")
                native = np.array(image, dtype=np.uint16)
            depth = native.astype(np.float32) / np.float32(5000)
            depth[native == 0] = np.nan
            pose, reference_metadata = self._reference(depth_time)
            observation = Observation(
                self._index - self._start,
                rgb,
                depth,
                float(depth_time),
                pose,
                {
                    "dataset": "tum_rgbd",
                    "original_frame_index": self._index,
                    "rgb_source_index": rgb_index,
                    "depth_source_index": depth_index,
                    "rgb_file": str(rgb_path.relative_to(self._root)),
                    "depth_file": str(depth_path.relative_to(self._root)),
                    "timestamp_color": rgb_time,
                    "timestamp_depth": depth_time,
                    "timestamp_unit": "seconds_since_unix_epoch",
                    "rgb_depth_difference_s": rgb_time - depth_time,
                    **reference_metadata,
                },
            )
            self._index += 1
            if self._index == self._stop:
                self.close()
            return observation
        except BaseException:
            self.close()
            raise


def load_tum_sequence(
    sequence_path: str | Path,
    *,
    start: int = 0,
    stop: int | None = None,
    config: TUMConfig | None = None,
) -> Sequence:
    """Materialize a bounded range, retaining associated source indices."""
    with TUMReader(sequence_path, start=start, stop=stop, config=config) as reader:
        return Sequence(reader.sequence_id, reader.calibration, tuple(reader))
