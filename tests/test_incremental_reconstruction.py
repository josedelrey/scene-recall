"""Transactional append fusion remains exactly equivalent after pose revisions."""

from dataclasses import fields, replace

import numpy as np
import pytest
from helpers.mapping import plane, pose, samples

from scene_recall.mapping.voxel import VoxelMap
from scene_recall.slam.reconstruction import RevisionReconstruction


def assert_cloud(a, b):
    for f in fields(a):
        np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name))


def test_fork_isolated_for_both_parent_and_child_writes():
    grid = VoxelMap(1)
    grid.integrate(samples([[0.1, 0, 1]], [[10, 20, 30]]), np.eye(4), 0)
    child = grid.fork()
    before = grid.cloud()
    child.integrate(samples([[0.3, 0, 1]], [[30, 40, 50]]), np.eye(4), 1)
    assert_cloud(grid.cloud(), before)
    child_before = child.cloud()
    grid.integrate(samples([[0.7, 0, 1]], [[80, 70, 60]]), np.eye(4), 2)
    assert_cloud(child.cloud(), child_before)


def test_append_revision_chunk_rollover_and_incomplete_dirty_list_are_exact():
    observation, calibration, config = plane()
    config = replace(config, keyframe_stride=1, max_keyframes_per_submap=3)
    baseline = RevisionReconstruction(calibration, config)
    changed = RevisionReconstruction(calibration, config, incremental=True)
    poses, components = [], []
    for frame in range(9):
        poses.append(pose(frame * 0.01))
        components.append(0)
        if frame == 2:
            poses[0] = pose(0.12)
        for engine in (baseline, changed):
            engine.admit(replace(observation, frame_index=frame), 0)
            # Include the revision for the baseline. The append path additionally
            # verifies cached poses rather than trusting an incomplete dirty list.
            engine.synchronize(
                poses,
                components,
                frame,
                [frame, 0] if frame == 2 and engine is baseline else [frame],
            )
        for a, b in zip(baseline.snapshot(), changed.snapshot()):
            assert_cloud(a.cloud, b.cloud)
        assert changed._cache[0] == len(changed.chunks) - 1
    # A revision to an older chunk uses complete raw fusion.
    poses[1] = pose(-0.2)
    for engine in (baseline, changed):
        engine.synchronize(poses, [8] * 9, 10, list(range(9)))
    for a, b in zip(baseline.snapshot(), changed.snapshot()):
        assert_cloud(a.cloud, b.cloud)
        assert a.component == b.component == 8


def test_failed_append_preserves_published_cloud_and_retry_uses_current_poses():
    observation, calibration, config = plane()
    config = replace(config, keyframe_stride=1)
    engine = RevisionReconstruction(calibration, config, incremental=True)
    engine.admit(observation, 0)
    engine.synchronize([pose()], [0], 0, [0])
    before = engine.snapshot()[0].cloud
    engine.admit(replace(observation, frame_index=1), 0)
    with pytest.raises(ValueError, match="rigid"):
        engine.synchronize([pose(), np.zeros((4, 4))], [0, 0], 1, [1])
    with pytest.raises(RuntimeError):
        engine.snapshot()
    assert_cloud(engine.chunks[0].cloud, before)
    engine.synchronize([pose(0.1), pose(0.2)], [0, 0], 2, [])
    fresh = VoxelMap(config.voxel_size_m, config.max_voxels)
    for frame in range(2):
        fresh.integrate(engine.samples[frame], pose(0.1 + frame * 0.1), frame)
    assert_cloud(engine.snapshot()[0].cloud, fresh.cloud())
