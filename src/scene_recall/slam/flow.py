"""Native-depth temporal registration from bidirectionally checked optical flow.

Corners are new image measurements, with no invented descriptor identities.
Only the previous and incoming images are used. Disjoint geometric verification
does not make neighboring image patches statistically independent.
"""

from dataclasses import replace

import cv2
import numpy as np

from scene_recall.features import RGBDFeatures, sample_native_depth
from scene_recall.slam.registration import Registration, register_anchor_pairs


def flow_registration(source, target, calibration, *, contrast="raw"):
    """Propose at most 1,000 native corners and verify a frozen PnP estimate."""
    if target.frame_index != source.frame_index + 1:
        raise ValueError("flow requires consecutive observations")
    if contrast not in ("raw", "clahe"):
        raise ValueError("unsupported flow contrast")
    images = [cv2.cvtColor(o.rgb, cv2.COLOR_RGB2GRAY) for o in (source, target)]
    if contrast == "clahe":
        normalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        images = [normalizer.apply(gray) for gray in images]
    diagnostic = {
        "method": "bidirectional_lk_native_pnp",
        "source": source.frame_index,
        "target": target.frame_index,
        "depth_origin": "native_source_observation",
    }

    def rejected(reason):
        return Registration(
            None, {**diagnostic, "status": "rejected", "reason": reason}
        )

    corners = cv2.goodFeaturesToTrack(
        images[0], maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    if corners is None:
        return rejected("no image corners")
    diagnostic["corners"] = len(corners)
    options = {
        "winSize": (21, 21),
        "maxLevel": 4,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    incoming, forward, error = cv2.calcOpticalFlowPyrLK(
        images[0], images[1], corners, None, **options
    )
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(
        images[1], images[0], incoming, None, **options
    )
    a, b = corners.reshape(-1, 2), incoming.reshape(-1, 2)
    cycle = np.linalg.norm(returned.reshape(-1, 2) - a, axis=1)
    height, width = images[1].shape
    valid = (
        forward.ravel().astype(bool)
        & backward.ravel().astype(bool)
        & (cycle <= 1.0)
        & (error.ravel() <= 20.0)
        & np.isfinite(b).all(axis=1)
        & (b[:, 0] >= 0)
        & (b[:, 0] < width)
        & (b[:, 1] >= 0)
        & (b[:, 1] < height)
    )
    a, b = a[valid].astype(np.float64), b[valid].astype(np.float64)
    diagnostic["flow_pairs"] = len(a)
    if not len(a):
        return rejected("no bidirectional flow support")
    first, second = [
        RGBDFeatures(
            pixels, None, sample_native_depth(observation, calibration, pixels)
        )
        for observation, pixels in ((source, a), (target, b))
    ]
    pairs = np.repeat(np.arange(len(a))[:, None], 2, axis=1)
    result = register_anchor_pairs(first, second, pairs, calibration)
    return replace(result, diagnostics={**result.diagnostics, **diagnostic})
