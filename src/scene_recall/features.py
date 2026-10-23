"""Native RGB-D feature measurements in the canonical RGB optical frame."""

from dataclasses import dataclass

import cv2
import numpy as np

from scene_recall.data import Calibration, Observation
from scene_recall.geometry.camera import (
    backproject_rgb_pixels,
    register_depth_to_rgb,
    reject_depth_edges,
)
from scene_recall.geometry.transforms import is_rigid


@dataclass(frozen=True, eq=False)
class RGBDFeatures:
    """Row identities are local to one extraction of one observation.

    Pixels are subpixel RGB coordinates. points_R contains metric RGB-frame
    samples or all-NaN rows when conservative depth sampling is unavailable.
    As with canonical data, callers must not mutate retained arrays.
    """

    pixels: np.ndarray
    descriptors: np.ndarray | None
    points_R: np.ndarray
    descriptor_family: str = "sift"

    def __post_init__(self) -> None:
        n = len(self.pixels)
        if self.pixels.shape != (n, 2) or not np.isfinite(self.pixels).all():
            raise ValueError("feature pixels must be finite (N, 2)")
        if self.points_R.shape != (n, 3):
            raise ValueError("feature points must have shape (N, 3)")
        valid = np.isfinite(self.points_R).all(axis=1) & (self.points_R[:, 2] > 0)
        if not (valid | np.isnan(self.points_R).all(axis=1)).all():
            raise ValueError("feature points must be positive-z finite or all NaN")
        dimensions = {"sift": 128, "xfeat": 64}
        if self.descriptor_family not in dimensions:
            raise ValueError("unsupported descriptor family")
        if self.descriptors is not None and (
            self.descriptors.shape != (n, dimensions[self.descriptor_family])
            or self.descriptors.dtype != np.float32
            or not np.isfinite(self.descriptors).all()
        ):
            raise ValueError(
                "descriptors must be finite float32 with the declared family dimension"
            )


def extract_rgbd_features(
    observation: Observation,
    calibration: Calibration,
    sift,
    depth_edge_threshold_m: float | None = 0.05,
    *,
    contrast: str = "raw",
    scale: float = 1.0,
) -> RGBDFeatures:
    """Extract SIFT and sample depth with the Sparse VO visibility policy.

    No reference pose is accessed. The caller owns the SIFT instance so serial
    backends can reuse it without sharing caches or OpenCV state.
    """
    if not is_rigid(calibration.T_RD, atol=1e-6):
        raise ValueError("feature extraction requires rigid T_RD calibration")
    if observation.rgb.shape[:2] != calibration.rgb_shape:
        raise ValueError("RGB shape must match calibration")
    if observation.depth.shape != calibration.depth_shape:
        raise ValueError("depth shape must match calibration")
    if contrast not in ("raw", "clahe"):
        raise ValueError("unsupported feature contrast mode")
    gray = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2GRAY)
    if contrast == "clahe":
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if scale not in (1.0, 0.5):
        raise ValueError("feature scale must be 1.0 or 0.5")
    original_shape = gray.shape
    if scale != 1.0:
        gray = cv2.resize(
            gray,
            (
                max(1, round(gray.shape[1] * scale)),
                max(1, round(gray.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    pixels = np.array(
        [keypoint.pt for keypoint in keypoints], dtype=np.float64
    ).reshape(-1, 2)
    if scale != 1.0:
        # OpenCV resize uses pixel-center coordinates, including odd image sizes.
        ratio = np.array(original_shape[::-1]) / np.array(gray.shape[::-1])
        pixels = (pixels + 0.5) * ratio - 0.5
    points = sample_native_depth(
        observation, calibration, pixels, depth_edge_threshold_m
    )
    return RGBDFeatures(pixels, descriptors, points)


def sample_native_depth(observation, calibration, pixels, depth_edge_threshold_m=0.05):
    """Sample original metric depth at RGB pixels without synthesizing depth."""
    if not is_rigid(calibration.T_RD, atol=1e-6):
        raise ValueError("depth sampling requires rigid T_RD calibration")
    if observation.rgb.shape[:2] != calibration.rgb_shape:
        raise ValueError("RGB shape must match calibration")
    if observation.depth.shape != calibration.depth_shape:
        raise ValueError("depth shape must match calibration")
    depth = observation.depth
    if depth_edge_threshold_m is not None:
        depth = reject_depth_edges(depth, depth_edge_threshold_m)
    if np.array_equal(calibration.T_RD, np.eye(4)):
        points = backproject_rgb_pixels(pixels, depth, calibration.K_R, calibration.K_D)
    else:
        registered = register_depth_to_rgb(
            observation.depth,
            calibration.K_D,
            calibration.K_R,
            calibration.T_RD,
            calibration.rgb_shape,
            valid_mask=np.isfinite(depth),
        )
        points = backproject_rgb_pixels(
            pixels, registered, calibration.K_R, calibration.K_R
        )
    return points
