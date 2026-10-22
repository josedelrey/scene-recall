"""Short-horizon recovery using retained native-depth observations."""

import numpy as np

from scene_recall.features import extract_rgbd_features
from scene_recall.odometry.evaluation import pose_error
from scene_recall.slam.registration import RegistrationConfig, register_anchor
from scene_recall.slam.system import RevisionFrontend


class RecoveryFrontend(RevisionFrontend):
    """Try recent anchors before declaring a new independent local segment.

    All proposals remain in the active local gauge. Withheld correspondences
    validate each frozen fit. Agreement across anchors is an engineering gate,
    without an independence or calibrated probability claim.
    """

    def __init__(self, config, *, horizon=15, candidates=3, contrast="raw"):
        super().__init__(config)
        self.horizon, self.candidates = horizon, candidates
        self.registration = RegistrationConfig()
        self.recovery_enabled = True
        self.contrast = contrast

    def _extract(self, observation, calibration):
        if self.contrast == "raw":
            return super()._extract(observation, calibration)
        return extract_rgbd_features(
            observation,
            calibration,
            self._sift,
            self.config.sparse.depth_edge_threshold_m,
            contrast=self.contrast,
        )

    def _recover_pose(self, first, second, calibration, previous_pose):
        if not self.recovery_enabled:
            return None, {}
        index = self._previous[0].frame_index
        anchors = [(index, first, previous_pose)]
        anchors.extend(
            (key, frame.features, frame.T_SR @ calibration.T_RD)
            for key, frame in reversed(self.keyframes.items())
            if key != index and index + 1 - key <= self.horizon
        )
        accepted, attempts = [], []
        for key, features, pose in anchors[: self.candidates]:
            result = register_anchor(features, second, calibration, self.registration)
            attempts.append({"source": key, **result.diagnostics})
            if result.T_target_source is not None:
                transform = (
                    np.linalg.inv(calibration.T_RD)
                    @ result.T_target_source
                    @ calibration.T_RD
                )
                accepted.append((pose @ np.linalg.inv(transform), key, result))
        if not accepted:
            return None, {"status": "rejected", "attempts": attempts}
        for a, _, _ in accepted[1:]:
            translation, rotation = pose_error(a, accepted[0][0])
            if translation > 0.05 or rotation > 3:
                return None, {"status": "ambiguous_anchors", "attempts": attempts}
        pose, key, _ = max(
            accepted, key=lambda value: value[2].diagnostics["validation_inliers"]
        )
        return pose, {
            "status": "accepted",
            "source": key,
            "attempts": attempts,
            "depth_origin": "native_source_observation",
        }
