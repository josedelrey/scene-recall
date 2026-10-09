"""Complete sparse RGB-D estimation followed by explicit geometric refinement."""

from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter

import numpy as np

from scene_recall.odometry.icp import DepthICPBackend, ICPConfig
from scene_recall.odometry.pipeline import PairEstimate
from scene_recall.odometry.sparse import SparseConfig, SparseRGBDBackend


@dataclass(frozen=True)
class HybridConfig:
    """Retain accepted sparse motion on ICP failure by default.

    Correction bounds are heuristic safeguards, independent of reference poses.
    Use refinement_failure='lost' for strict sparse-plus-ICP acceptance.
    """

    sparse: SparseConfig = field(default_factory=SparseConfig)
    icp: ICPConfig = field(default_factory=ICPConfig)
    refinement_failure: str = "sparse"
    max_correction_translation_m: float = 0.1
    max_correction_rotation_deg: float = 5.0

    def __post_init__(self):
        if not isinstance(self.sparse, SparseConfig) or not isinstance(
            self.icp, ICPConfig
        ):
            raise TypeError("hybrid requires SparseConfig and ICPConfig")
        if self.refinement_failure not in ("sparse", "lost"):
            raise ValueError("refinement_failure must be sparse or lost")
        for name in ("max_correction_translation_m", "max_correction_rotation_deg"):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


class HybridRGBDBackend:
    """Sparse success is required before attempting depth refinement.

    Fallbacks are accepted sparse estimates with an explicit diagnostic status.
    Successful ICP replaces the sparse estimate without consulting references.
    """

    def __init__(self, config: HybridConfig | None = None):
        self.config = config or HybridConfig()
        self._sparse = SparseRGBDBackend(self.config.sparse)
        self._depth = DepthICPBackend(self.config.icp)

    def estimate(self, source, target, calibration):
        started = perf_counter()
        sparse = self._sparse.estimate(source, target, calibration)
        diagnostics = dict(sparse.diagnostics)
        diagnostics["sparse_seconds"] = perf_counter() - started
        diagnostics["sparse_status"] = "failed" if sparse.T_C1C0 is None else "accepted"
        if sparse.T_C1C0 is None:
            diagnostics["geometric_refinement"] = "not_run"
            return PairEstimate(
                None,
                sparse.reason,
                diagnostics=diagnostics,
                ransac_T_C1C0=sparse.initial_T_C1C0,
            )
        refined = self._depth.refine(source, target, calibration, sparse.T_C1C0)
        diagnostics.update(refined.diagnostics)
        reason = refined.reason
        if refined.T_C1C0 is not None:
            correction = refined.T_C1C0 @ np.linalg.inv(sparse.T_C1C0)
            rotation = correction[:3, :3]
            angle = float(
                np.degrees(
                    np.arctan2(
                        np.linalg.norm(
                            [
                                rotation[2, 1] - rotation[1, 2],
                                rotation[0, 2] - rotation[2, 0],
                                rotation[1, 0] - rotation[0, 1],
                            ]
                        )
                        / 2,
                        (np.trace(rotation) - 1) / 2,
                    )
                )
            )
            translation = float(np.linalg.norm(correction[:3, 3]))
            diagnostics.update(
                icp_correction_translation_m=translation,
                icp_correction_rotation_deg=angle,
            )
            if (
                translation > self.config.max_correction_translation_m
                or angle > self.config.max_correction_rotation_deg
            ):
                reason = "excessive_correction"
        if reason is None:
            pose = refined.T_C1C0
            diagnostics["geometric_refinement"] = "refined"
        else:
            pose = sparse.T_C1C0 if self.config.refinement_failure == "sparse" else None
            diagnostics["geometric_refinement"] = (
                "fallback_sparse" if pose is not None else "failed"
            )
        diagnostics["geometric_refinement_reason"] = reason
        return PairEstimate(
            pose,
            None if pose is not None else f"ICP refinement failed: {reason}",
            sparse.T_C1C0,
            diagnostics,
            refined.candidate_T_C1C0,
            sparse.initial_T_C1C0,
        )
