"""Native measurement ownership, temporal recovery and abstention contracts."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from scene_recall.data import Calibration, Observation
from scene_recall.features import (
    RGBDFeatures,
    extract_rgbd_features,
    sample_native_depth,
)
from scene_recall.geometry.camera import reject_depth_edges
from scene_recall.slam.flow import flow_registration
from scene_recall.slam.online import OnlineConfig, OnlineSLAM
from scene_recall.slam.registration import register_anchor_pairs


def translated_pair():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0.8)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, -2]]), (320, 240))
    K = np.array([[200.0, 0, 160], [0, 200, 120], [0, 0, 1]])
    calibration = Calibration(K, K, np.eye(4), image.shape, image.shape)
    observations = [
        Observation(
            i,
            np.repeat(gray[..., None], 3, axis=2),
            np.full(image.shape, 2, np.float32),
        )
        for i, gray in enumerate((image, shifted))
    ]
    return calibration, observations


def test_flow_recovers_native_motion_without_incoming_depth_or_reference():
    calibration, (source, target) = translated_pair()
    target = replace(
        target,
        depth=np.full_like(target.depth, np.nan),
        T_WC_D=np.diag([1.0, 1.0, 1.0, 1.0]),
    )
    result = flow_registration(source, target, calibration)
    assert result.T_target_source is not None
    np.testing.assert_allclose(
        result.T_target_source[:3, 3], [0.04, -0.02, 0], atol=0.002
    )
    assert result.diagnostics["validation_depth_checks"] == 0
    assert result.diagnostics["validation_inliers"] >= 6
    assert result.diagnostics["depth_origin"] == "native_source_observation"


@pytest.mark.parametrize("failure", ["blank", "source_depth", "inconsistent_depth"])
def test_flow_abstains_without_verifiable_geometry(failure):
    calibration, (source, target) = translated_pair()
    if failure == "blank":
        target = replace(target, rgb=np.zeros_like(target.rgb))
    elif failure == "source_depth":
        source = replace(source, depth=np.full_like(source.depth, np.nan))
    else:
        target = replace(target, depth=np.full_like(target.depth, 4))
    assert flow_registration(source, target, calibration).T_target_source is None


def test_flow_requires_temporal_adjacency():
    calibration, (source, target) = translated_pair()
    with pytest.raises(ValueError, match="consecutive"):
        flow_registration(source, replace(target, frame_index=2), calibration)


def test_half_scale_returns_original_camera_pixels_and_depth():
    calibration, (source, _) = translated_pair()

    class Detector:
        def detectAndCompute(self, image, mask):
            assert image.shape == (120, 160)
            return [cv2.KeyPoint(40, 30, 4)], np.zeros((1, 128), np.float32)

    features = extract_rgbd_features(source, calibration, Detector(), scale=0.5)
    np.testing.assert_array_equal(features.pixels, [[80.5, 60.5]])
    np.testing.assert_allclose(features.points_R, [[-0.795, -0.595, 2]])


def test_half_scale_uses_actual_resize_geometry_for_odd_widths():
    calibration, (source, _) = translated_pair()
    source = replace(source, rgb=source.rgb[:, :319], depth=source.depth[:, :319])
    calibration = replace(calibration, rgb_shape=(240, 319), depth_shape=(240, 319))
    expected = cv2.resize(
        cv2.cvtColor(source.rgb, cv2.COLOR_RGB2GRAY),
        (160, 120),
        interpolation=cv2.INTER_AREA,
    )

    class Detector:
        def detectAndCompute(self, image, mask):
            np.testing.assert_array_equal(image, expected)
            return [cv2.KeyPoint(40, 30, 4)], np.zeros((1, 128), np.float32)

    features = extract_rgbd_features(source, calibration, Detector(), scale=0.5)
    np.testing.assert_allclose(features.pixels, [[80.246875, 60.5]])


def test_explicit_associations_cannot_reuse_measurements():
    calibration, _ = translated_pair()
    features = RGBDFeatures(np.zeros((2, 2)), None, np.ones((2, 3)))
    with pytest.raises(ValueError, match="unique"):
        register_anchor_pairs(
            features, features, np.array([[0, 0], [0, 1]]), calibration
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(1, 1), (2, 7), (31, 45)])
def test_depth_filter_preserves_full_neighborhood_contract(dtype, shape):
    rng = np.random.default_rng(8)
    depth = rng.uniform(1, 1.1, shape).astype(dtype)
    depth.ravel()[::19] = np.nan
    depth.ravel()[::37] = 0
    original = depth.copy()
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(depth, 1, constant_values=np.nan), (3, 3)
    )
    valid = np.isfinite(windows).all(axis=(-2, -1)) & (windows > 0).all(axis=(-2, -1))
    for threshold in (0.05, 0.1):
        continuous = (
            windows.max(axis=(-2, -1)) - windows.min(axis=(-2, -1))
        ) <= threshold
        expected = np.where(valid & continuous, depth, np.nan).astype(dtype)
        np.testing.assert_array_equal(reject_depth_edges(depth, threshold), expected)
    np.testing.assert_array_equal(depth, original)


def test_descriptor_family_is_explicit_and_cannot_mislabel_learned_features():
    pixels, points = np.zeros((2, 2)), np.ones((2, 3))
    descriptors = np.ones((2, 64), np.float32)
    features = RGBDFeatures(pixels, descriptors, points, "xfeat")
    assert features.descriptor_family == "xfeat"
    with pytest.raises(ValueError, match="dimension"):
        RGBDFeatures(pixels, descriptors, points)


def test_custom_extractor_cannot_change_family_midstream():
    calibration, observations = translated_pair()

    def extractor(observation, calibration):
        return RGBDFeatures(
            np.empty((0, 2)),
            None,
            np.empty((0, 3)),
            "sift" if observation.frame_index == 0 else "xfeat",
        )

    engine = OnlineSLAM(calibration, feature_extractor=extractor)
    engine.process(observations[0])
    with pytest.raises(ValueError, match="remain fixed"):
        engine.process(observations[1])
    with pytest.raises(ValueError, match="preprocessing"):
        OnlineSLAM(
            calibration, OnlineConfig(contrast="clahe"), feature_extractor=extractor
        )


def test_native_depth_sampling_checks_calibration_for_external_extractors():
    calibration, (source, _) = translated_pair()
    invalid = replace(calibration, T_RD=np.diag([2.0, 1, 1, 1]))
    with pytest.raises(ValueError, match="rigid"):
        sample_native_depth(source, invalid, np.array([[100.0, 100.0]]))
