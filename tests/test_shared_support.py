"""Support selection uses retained evidence and never invents constraints."""

from dataclasses import replace

import numpy as np
import pytest
from test_local_map import scene
from test_online_slam import engine_scene

from scene_recall.features import RGBDFeatures
from scene_recall.mapping.voxel import MappingConfig, VoxelMap
from scene_recall.slam.online import OnlineConfig
from scene_recall.slam.support import SupportConfig, metric_support
from scene_recall.slam.system import RevisionFrontend


@pytest.mark.parametrize(
    "points,rank",
    [
        ([[0, 0, 2]], 0),
        ([[0, 0, 2], [1, 0, 2]], 2),
        ([[0, 0, 2], [1, 0, 2], [0, 1, 2]], 3),
    ],
)
def test_rigid_support_detects_point_and_line_symmetries(points, rank):
    assert metric_support(points)["rotation_rank"] == rank
    assert metric_support(np.array(points) * 5)["spread"] == pytest.approx(
        metric_support(points)["spread"]
    )


def test_support_counts_only_retained_native_owners():
    calibration, observations, features, _ = scene(3)
    frontend = RevisionFrontend(support=SupportConfig(enabled=True))
    frontend._bootstrap(observations[0], features[0], calibration)
    owners = frontend.owners[0]
    valid = np.flatnonzero(owners >= 0)
    matches = [(int(owners[i]), int(i)) for i in valid]
    # All accepted frontend matches cannot conceal a lost retained bridge.
    frontend.retain_support(0, [100000, 100001, 100002])
    diagnostic = {}
    assert frontend._needs_keyframe(
        1, features[1], np.eye(4), matches, calibration, diagnostic
    )
    assert diagnostic["retained_support"]["count"] == 0
    assert diagnostic["retained_support"]["early_keyframe"]
    frontend.support_policy = SupportConfig()
    assert not frontend._needs_keyframe(
        1, features[1], np.eye(4), matches, calibration, {}
    )
    frontend._bootstrap(observations[1], features[1], calibration)
    assert frontend.retained_support is None


def test_missing_native_depth_is_not_metric_support():
    calibration, observations, features, _ = scene(3)
    frontend = RevisionFrontend(support=SupportConfig(enabled=True))
    frontend._bootstrap(observations[0], features[0], calibration)
    owners = frontend.owners[0]
    valid = np.flatnonzero(owners >= 0)
    frontend.retain_support(0, owners[valid])
    missing = RGBDFeatures(
        features[1].pixels,
        features[1].descriptors,
        np.full_like(features[1].points_R, np.nan),
    )
    diagnostic = {}
    assert frontend._needs_keyframe(
        1,
        missing,
        np.eye(4),
        [(int(owners[i]), int(i)) for i in valid],
        calibration,
        diagnostic,
    )
    assert diagnostic["retained_support"]["count"] == 0


def test_early_retention_is_causal_and_reconstruction_exact(monkeypatch):
    config = OnlineConfig(shared_estimation=True)
    config = replace(
        config,
        local=replace(
            config.local,
            exposure="static",
            support=SupportConfig(enabled=True, minimum_spread=1),
        ),
    )
    engine, observations, _ = engine_scene(
        monkeypatch, count=8, config=config, mapping=MappingConfig(pixel_stride=8)
    )
    updates = [engine.process(o) for o in observations]
    assert len(engine.graph.poses) > 2
    for view in engine.welding.views.values():
        native = view.native_index[view.native_index >= 0]
        assert len(native) == len(np.unique(native))
    replay, _, _ = engine_scene(monkeypatch, count=8, config=config)
    reference = np.eye(4)
    reference[:3, 3] = 100
    for i, o in enumerate(observations[:5]):
        update = replay.process(replace(o, T_WC_D=reference))
        np.testing.assert_array_equal(update.pose, updates[i].pose)
        assert update.component == updates[i].component
    reconstruction = engine.reconstruction
    # Each published chunk must equal fresh fusion at its revised camera poses.
    for chunk in reconstruction.chunks:
        grid = VoxelMap(
            reconstruction.config.voxel_size_m, reconstruction.config.max_voxels
        )
        for frame in chunk.frames:
            grid.integrate(reconstruction.samples[frame], engine.poses[frame], frame)
        np.testing.assert_array_equal(chunk.cloud.points, grid.cloud().points)
        np.testing.assert_array_equal(
            chunk.cloud.frame_support, grid.cloud().frame_support
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum": 2},
        {"minimum": True},
        {"retained_fraction": 0},
        {"minimum_spread": np.nan},
    ],
)
def test_invalid_support_policy(kwargs):
    with pytest.raises((ValueError, TypeError)):
        SupportConfig(**kwargs)


def test_support_requires_shared_measurement_owner():
    config = OnlineConfig()
    with pytest.raises(ValueError, match="require shared"):
        replace(
            config, local=replace(config.local, support=SupportConfig(enabled=True))
        )
