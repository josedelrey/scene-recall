"""Historical tracker checks use retained color and native depth charts."""

from dataclasses import replace

import numpy as np
from test_local_map import scene
from test_online_slam import engine_scene

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.pose_only import retract
from scene_recall.slam.online import OnlineConfig
from scene_recall.slam.shared_estimation import SharedConfig
from scene_recall.slam.system import RevisionFrontend


def test_historical_pixels_and_native_points_use_their_own_exposures():
    calibration, observations, features, _ = scene(3)
    frontend = RevisionFrontend()
    frontend._bootstrap(observations[0], features[0], calibration)
    frontend._previous = observations[0], features[0]
    frontend._calibration = calibration
    frame = frontend.keyframes[0]
    ids = frontend.owners[0]
    valid = np.flatnonzero(ids >= 0)
    points = np.array([frontend.landmarks[int(i)].point_S for i in ids[valid]])
    exposure = retract(calibration.T_RD, np.array([0.02, 0, 0, 0, 0.04, 0]))
    native = transform_points(points, np.linalg.inv(frame.T_SR @ calibration.T_RD))
    pixels = project_points(transform_points(native, exposure), calibration.K_R)
    changed = frame.features.pixels.copy()
    changed[valid] = pixels
    frontend.keyframes[0] = replace(
        frame, features=replace(frame.features, pixels=changed)
    )
    assert not frontend._valid_points(ids[valid], points, calibration).any()
    frontend.exposure_transforms = {0: exposure}
    assert frontend._valid_points(ids[valid], points, calibration).all()
    # Global chart transport changes both poses and points, retaining the same
    # native-depth and color-time observation predictions.
    correction = retract(np.eye(4), np.array([0.1, 0.2, 0.3, 0.01, -0.02, 0.03]))
    depth_pose = correction @ frame.T_SR @ calibration.T_RD
    frontend.revise({0: depth_pose}, {}, calibration)
    assert frontend._valid_points(
        ids[valid], transform_points(points, correction), calibration
    ).all()
    frontend._bootstrap(observations[1], features[1], calibration)
    assert not frontend.exposure_transforms


def test_static_exposure_pipeline_remains_bitwise_unchanged(monkeypatch):
    config = OnlineConfig(shared_estimation=True)
    config = replace(config, local=replace(config.local, exposure="static"))
    control, observations, _ = engine_scene(monkeypatch, count=17, config=config)
    expected = [control.process(o) for o in observations]
    candidate, _, _ = engine_scene(
        monkeypatch,
        count=17,
        config=replace(config, shared=SharedConfig(tracker_exposure=True)),
    )
    for i, observation in enumerate(observations):
        update = candidate.process(observation)
        np.testing.assert_array_equal(update.pose, expected[i].pose)
        assert update.version == expected[i].version
        assert update.component == expected[i].component
    for key in control.welding.views:
        assert (
            control.welding.views[key].factors.fingerprint()
            == candidate.welding.views[key].factors.fingerprint()
        )
