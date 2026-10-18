"""Core geometry and mapping remain usable without odometry backends."""

import subprocess
import sys


def test_geometry_mapping_and_trajectory_types_do_not_load_backends():
    code = """
import importlib
import sys

blocked = (
    'cv2',
    'scene_recall.odometry.sparse',
    'scene_recall.odometry.icp',
    'scene_recall.odometry.hybrid',
)
for name in blocked:
    sys.modules[name] = None
for name in (
    'scene_recall.geometry.transforms',
    'scene_recall.odometry',
    'scene_recall.odometry.pipeline',
    'scene_recall.odometry.evaluation',
    'scene_recall.odometry.io',
    'scene_recall.mapping.voxel',
    'scene_recall.mapping.mesh',
    'scene_recall.mapping.tsdf',
    'scene_recall.mapping.pipeline',
):
    importlib.import_module(name)
from scene_recall.geometry.transforms import is_rigid
from scene_recall.mapping import VoxelMap
from scene_recall.odometry import PairEstimate
import numpy as np
assert is_rigid(np.eye(4))
assert VoxelMap(0.02).cloud().points.shape == (0, 3)
assert PairEstimate(None).T_C1C0 is None
assert all(sys.modules[name] is None for name in blocked)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_package_backend_exports_are_lazy_and_keep_their_public_identity():
    code = """
import importlib
import sys

package = importlib.import_module('scene_recall.odometry')
assert not any('scene_recall.odometry.' + name in sys.modules
               for name in ('sparse', 'icp', 'hybrid'))
for module, names in (
    ('sparse', ('SparseConfig', 'SparseRGBDBackend')),
    ('icp', ('ICPConfig', 'DepthICPBackend')),
    ('hybrid', ('HybridConfig', 'HybridRGBDBackend')),
):
    for name in names:
        assert name in dir(package)
        value = getattr(package, name)
        assert value is getattr(importlib.import_module(package.__name__ + '.' + module), name)
        assert getattr(package, name) is value
    if module == 'sparse':
        assert 'scene_recall.odometry.icp' not in sys.modules
        assert 'scene_recall.odometry.hybrid' not in sys.modules
try:
    getattr(package, 'unknown_backend')
except AttributeError:
    pass
else:
    raise AssertionError('unknown attributes must raise AttributeError')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
