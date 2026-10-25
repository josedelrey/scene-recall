"""Selective endpoint Jacobians with exact current raw observation residuals.

The cache holds derivatives only. Residuals, robust weights, landmark elimination
and trial acceptance always use the complete current measurement objective.
It is disposable numerical state, never a prior or another measurement owner.
"""

from dataclasses import dataclass

import numpy as np

from scene_recall.geometry.camera import project_points
from scene_recall.multiview.pose_only import projection_jacobian, skew


@dataclass(frozen=True)
class RelinearizationConfig:
    translation_m: float = 0.002
    rotation_deg: float = 0.1
    point_m: float = 0.002
    max_age: int = 8

    def __post_init__(self):
        for name in ("translation_m", "rotation_deg", "point_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            isinstance(self.max_age, bool)
            or not isinstance(self.max_age, int)
            or self.max_age < 1
        ):
            raise ValueError("maximum Jacobian age must be a positive integer")


class ObservationCache:
    """Persistent derivatives keyed by camera and live landmark identity.

    Binding verifies all immutable measurement data before permitting reuse.
    Gauge and identity changes can discard the complete cache explicitly.
    """

    def __init__(self, config=None):
        self.config = config or RelinearizationConfig()
        self.keys = {}
        self.arrays = {}
        self.K = None
        self.model = None

    def bind(self, factors, K, depth_model):
        names = ("pixels", "native_points", "T_RcD", "whitening_D", "weights")
        keys = list(zip(factors.view.tolist(), factors.track_id.tolist(), strict=True))
        old = np.array([self.keys.get(key, -1) for key in keys])
        valid = old >= 0
        if self.K is None or not np.array_equal(K, self.K) or depth_model != self.model:
            valid[:] = False
        safe = np.maximum(old, 0)
        if self.keys:
            for name in names:
                previous, current = self.arrays[name][safe], getattr(factors, name)
                equal = (previous == current) | (np.isnan(previous) & np.isnan(current))
                valid &= equal.reshape(len(keys), -1).all(axis=1)
        shapes = {
            "JP": (2, 3, 6),
            "JL": (2, 3, 3),
            "pose": (4, 4),
            "point": (3,),
            "age": (),
        }
        for name, shape in shapes.items():
            updated = np.zeros((len(keys), *shape))
            if self.keys:
                updated[valid] = self.arrays[name][old[valid]]
            self.arrays[name] = updated
        for name in names:
            self.arrays[name] = getattr(factors, name).copy()
        self.valid = valid
        self.keys = {key: i for i, key in enumerate(keys)}
        self.K, self.model = K.copy(), depth_model
        self.factors = factors
        transform = factors.T_RcD
        self.rotation = transform[:, :3, :3].transpose(0, 2, 1)
        self.offset = -np.einsum("nij,nj->ni", self.rotation, transform[:, :3, 3])
        self.native = np.isfinite(factors.native_points).all(axis=1)
        self.W = factors.whitening_D.copy()
        if depth_model == "axial":
            sigma = np.linalg.norm(np.linalg.inv(self.W[self.native])[:, 2], axis=1)
            self.W[:] = 0
            self.W[self.native, 2, 2] = 1 / sigma
        elif depth_model != "native":
            raise ValueError("unsupported shared depth model")
        self.W[~self.native] = 0
        self.native_derivative = self.W @ self.rotation
        self.refreshed = self.evaluated = self.full_refreshes = 0

    def geometry(self, poses, points, inverse):
        world = points[inverse]
        R = poses[self.factors.view, :3, :3].transpose(0, 2, 1)
        q = np.einsum("nij,nj->ni", R, world - poses[self.factors.view, :3, 3])
        qD = np.einsum("nij,nj->ni", self.rotation, q) + self.offset
        if (
            not np.isfinite(q).all()
            or (q[:, 2] <= 0.05).any()
            or (qD[self.native, 2] <= 0.05).any()
        ):
            return None, R, q
        residual = np.zeros((len(q), 2, 3))
        residual[:, 0, :2] = project_points(q, self.K) - self.factors.pixels
        residual[self.native, 1] = np.einsum(
            "nij,nj->ni",
            self.W[self.native],
            qD[self.native] - self.factors.native_points[self.native],
        )
        return residual, R, q

    def residual(self, poses, points, inverse):
        return self.geometry(poses, points, inverse)[0]

    def linearize(self, poses, points, inverse, *, force=False):
        residual, R, q = self.geometry(poses, points, inverse)
        if residual is None:
            return None, None, None
        world, cameras = points[inverse], poses[self.factors.view]
        old = self.arrays
        relative = cameras[:, :3, :3] @ old["pose"][:, :3, :3].transpose(0, 2, 1)
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
        refresh = (
            ~self.valid
            | (
                np.linalg.norm(cameras[:, :3, 3] - old["pose"][:, :3, 3], axis=1)
                > self.config.translation_m
            )
            | (cosine < np.cos(np.radians(self.config.rotation_deg)))
            | (np.linalg.norm(world - old["point"], axis=1) > self.config.point_m)
            | (old["age"] >= self.config.max_age)
        )
        if force:
            refresh[:] = True
            self.full_refreshes += 1
        D = np.concatenate((-R[refresh], R[refresh] @ skew(world[refresh])), axis=2)
        rgb = projection_jacobian(q[refresh], self.K)
        rgb[:, 2] = 0
        native = self.native_derivative[refresh]
        old["JP"][refresh] = np.stack((rgb @ D, native @ D), axis=1)
        old["JL"][refresh] = np.stack((rgb @ R[refresh], native @ R[refresh]), axis=1)
        old["pose"][refresh], old["point"][refresh] = cameras[refresh], world[refresh]
        old["age"] += 1
        old["age"][refresh] = 0
        self.valid[:] = True
        self.refreshed += int(refresh.sum())
        self.evaluated += len(refresh)
        self.last_fresh = bool(refresh.all())
        return residual, old["JP"], old["JL"]

    def diagnostic(self):
        return {
            "jacobian_observations": self.refreshed,
            "evaluated_observations": self.evaluated,
            "full_refreshes": self.full_refreshes,
            "cache_bytes": sum(value.nbytes for value in self.arrays.values()),
            "reused_fraction": 1 - self.refreshed / max(1, self.evaluated),
        }
