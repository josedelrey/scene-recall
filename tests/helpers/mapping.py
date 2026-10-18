"""Analytic planes, surface samples and segment-local trajectory rows."""

import numpy as np

from scene_recall.data import Calibration, Observation
from scene_recall.mapping.voxel import MappingConfig, SurfaceSamples
from scene_recall.odometry.pipeline import TrajectoryFrame


def pose(x=0, z=0, yaw=0):
    T = np.eye(4)
    T[:3, :3] = [
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ]
    T[:3, 3] = [x, 0, z]
    return T


def row(index, segment=0, T=None, reference=None, status=None):
    return TrajectoryFrame(
        index,
        float(index),
        segment,
        status or ("initialized" if index == 0 else "tracked"),
        np.eye(4) if T is None else T,
        reference,
    )


def samples(points, colors=None, colored=None):
    points = np.array(points, dtype=float).reshape(-1, 3)
    return SurfaceSamples(
        points,
        np.full((len(points), 3), 60, dtype=np.uint8)
        if colors is None
        else np.array(colors, dtype=np.uint8),
        np.ones(len(points), dtype=bool)
        if colored is None
        else np.array(colored, dtype=bool),
        np.ones((3, 3), dtype=np.float32),
    )


def plane(T=None):
    K = np.array([[40, 0, 16], [0, 40, 16], [0, 0, 1]], dtype=float)
    calibration = Calibration(K, K, np.eye(4), (33, 33), (33, 33))
    T = np.eye(4) if T is None else T
    v, u = np.indices((33, 33))
    rays = np.stack(((u - 16) / 40, (v - 16) / 40, np.ones((33, 33))), axis=-1)
    depth = ((2 - T[2, 3]) / (rays @ T[:3, :3].T)[..., 2]).astype(np.float32)
    observation = Observation(
        0, np.full((33, 33, 3), [30, 100, 210], dtype=np.uint8), depth, 0.0, T
    )
    config = MappingConfig(pixel_stride=1, depth_edge_threshold_m=1)
    return observation, calibration, config
