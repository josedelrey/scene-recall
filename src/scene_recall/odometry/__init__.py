"""Dataset-independent selectable RGB-D odometry and trajectory evaluation."""

from importlib import import_module

from scene_recall.odometry.pipeline import (
    PairEstimate,
    PairwiseBackend,
    TrajectoryFrame,
    track_observations,
)

__all__ = [
    "DepthICPBackend",
    "HybridConfig",
    "HybridRGBDBackend",
    "ICPConfig",
    "LocalMapConfig",
    "LocalMapRGBDBackend",
    "PairEstimate",
    "PairwiseBackend",
    "RecentMapConfig",
    "RecentMapRGBDBackend",
    "SparseConfig",
    "SparseRGBDBackend",
    "TrajectoryFrame",
    "track_observations",
]

_BACKEND_MODULES = {
    "DepthICPBackend": "icp",
    "HybridConfig": "hybrid",
    "HybridRGBDBackend": "hybrid",
    "ICPConfig": "icp",
    "LocalMapConfig": "local_map",
    "LocalMapRGBDBackend": "local_map",
    "RecentMapConfig": "recent_map",
    "RecentMapRGBDBackend": "recent_map",
    "SparseConfig": "sparse",
    "SparseRGBDBackend": "sparse",
}


def __getattr__(name: str):
    """Load backend exports on demand while preserving package-level imports."""
    if name not in _BACKEND_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{_BACKEND_MODULES[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
