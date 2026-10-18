"""Analytic room observations and pairwise backend doubles."""

import numpy as np

from scene_recall.data import Calibration, Observation
from scene_recall.geometry.camera import backproject_depth
from scene_recall.odometry.icp import increment


def render_room(K, shape, camera_to_world):
    """Intersect camera rays with five analytic room planes."""
    rays = backproject_depth(np.ones(shape), K)
    rays_world = rays @ camera_to_world[:3, :3].T
    origin = camera_to_world[:3, 3]
    depth = np.full(shape, np.inf)
    planes = [
        (np.array([0, 0, 1]), 2.5),
        (np.array([1, 0, 0]), 0.65),
        (np.array([-1, 0, 0]), 0.65),
        (np.array([0, 1, 0]), 0.5),
        (np.array([0, -1, 0]), 0.5),
    ]
    for normal, offset in planes:
        denominator = rays_world @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            intersection = (offset - origin @ normal) / denominator
        intersection[intersection <= 0] = np.inf
        depth = np.minimum(depth, intersection)
    depth[~np.isfinite(depth)] = np.nan
    return depth.astype(np.float32)


def room_pair():
    K = np.array([[360, 0, 159.5], [0, 360, 127.5], [0, 0, 1.0]])
    shape = (256, 320)
    truth = increment(np.array([0.018, -0.012, 0.008, 0.007, -0.004, 0.003]))
    calibration = Calibration(K, K, np.eye(4), shape, shape)
    blank = np.zeros((*shape, 3), np.uint8)
    source = Observation(0, blank, render_room(K, shape, np.eye(4)))
    target = Observation(1, blank, render_room(K, shape, np.linalg.inv(truth)))
    return source, target, calibration, truth


class StubSparse:
    def __init__(self, pair):
        self.pair = pair

    def estimate(self, *args):
        return self.pair


class StubDepth:
    def __init__(self, pair):
        self.pair = pair
        self.initial = None

    def refine(self, source, target, calibration, initial):
        self.initial = initial
        return self.pair
