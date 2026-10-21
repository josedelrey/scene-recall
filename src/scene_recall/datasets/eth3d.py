"""Processed ETH3D SLAM RGB-D with explicit pinhole calibration.

The authors' RGB-D format extends TUM's timestamp tables with calibration.txt.
Processed registered grids share the RGB optical frame and metric z depth uses
5000 stored units per meter. Raw, stereo and unrectified variants are excluded.
https://github.com/ETH3D/badslam#dataset-format
"""

import numpy as np
from PIL import Image

from scene_recall.data import Calibration, Observation
from scene_recall.datasets.tum import TUMReader


class ETH3DReader(TUMReader):
    """Reuse strict association/reference interpolation with explicit image grids."""

    def __init__(self, sequence_path, *, start=0, stop=None, config=None):
        super().__init__(sequence_path, start=start, stop=stop, config=config)
        parameters = np.loadtxt(self._root / "calibration.txt")
        if parameters.shape != (4,) or not np.isfinite(parameters).all():
            raise ValueError("ETH3D calibration.txt must contain fx fy cx cy")
        fx, fy, cx, cy = parameters
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        if not self._pairs:
            raise ValueError("ETH3D source has no associated observations")
        rgb_index, depth_index = self._pairs[0]
        with (
            Image.open(self._rgb[rgb_index][1]) as rgb,
            Image.open(self._depth[depth_index][1]) as depth,
        ):
            if rgb.size != depth.size:
                raise ValueError(
                    "processed ETH3D RGB-D requires equal registered grids"
                )
            shape = (rgb.height, rgb.width)
        self.calibration = Calibration(K, K.copy(), np.eye(4), shape, shape)
        self.metadata.update(
            dataset="eth3d_slam",
            calibration_policy="eth3d_processed_registered_pinhole_file",
            reference_policy="source_camera_to_world_at_depth_timestamp_no_extrapolation",
            timestamp_unit="source_clock_seconds",
            depth_scale_correction="none",
        )

    def __next__(self):
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
                    or image.mode not in ("L", "RGB")
                    or (image.height, image.width) != self.calibration.rgb_shape
                ):
                    raise ValueError("ETH3D requires calibrated RGB or grayscale PNGs")
                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
            with Image.open(depth_path) as image:
                if (
                    image.format != "PNG"
                    or image.mode != "I;16"
                    or (image.height, image.width) != self.calibration.depth_shape
                ):
                    raise ValueError("ETH3D requires calibrated uint16 depth PNGs")
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
                    "dataset": "eth3d_slam",
                    "original_frame_index": self._index,
                    "rgb_source_index": rgb_index,
                    "depth_source_index": depth_index,
                    "rgb_file": rgb_path.relative_to(self._root).as_posix(),
                    "depth_file": depth_path.relative_to(self._root).as_posix(),
                    "timestamp_color": rgb_time,
                    "timestamp_depth": depth_time,
                    "timestamp_unit": "source_clock_seconds",
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
