"""Short-horizon recovery using retained native-depth observations."""

import numpy as np

from scene_recall.features import RGBDFeatures, extract_rgbd_features
from scene_recall.odometry.evaluation import pose_error
from scene_recall.slam.flow import flow_registration
from scene_recall.slam.registration import RegistrationConfig, register_anchor
from scene_recall.slam.system import RevisionFrontend


class RecoveryFrontend(RevisionFrontend):
    """Try recent anchors before declaring a new independent local segment.

    All proposals remain in the active local gauge. Withheld correspondences
    validate each frozen fit. Agreement across anchors is an engineering gate,
    without an independence or calibrated probability claim.
    """

    def __init__(
        self,
        config,
        *,
        horizon=15,
        candidates=3,
        contrast="raw",
        flow=False,
        feature_scale=1.0,
        feature_extractor=None,
    ):
        super().__init__(config)
        self.horizon, self.candidates = horizon, candidates
        self.registration = RegistrationConfig()
        self.recovery_enabled = True
        self.contrast = contrast
        self.flow = flow
        self.feature_scale = feature_scale
        self._incoming = None
        self.feature_extractor = feature_extractor
        self._descriptor_family = None
        if feature_extractor is not None and (
            contrast != "raw" or feature_scale != 1.0
        ):
            raise ValueError("custom extractors own their image preprocessing")

    def _extract(self, observation, calibration):
        self._incoming = observation
        if self.feature_extractor is not None:
            features = self.feature_extractor(observation, calibration)
            if not isinstance(features, RGBDFeatures):
                raise TypeError("custom extractor must return RGBDFeatures")
            if self._descriptor_family not in (None, features.descriptor_family):
                raise ValueError("descriptor family must remain fixed within a stream")
            self._descriptor_family = features.descriptor_family
            return features
        if self.contrast == "raw" and self.feature_scale == 1.0:
            return super()._extract(observation, calibration)
        return extract_rgbd_features(
            observation,
            calibration,
            self._sift,
            self.config.sparse.depth_edge_threshold_m,
            contrast=self.contrast,
            scale=self.feature_scale,
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
        if self.flow and self._incoming is not None:
            result = flow_registration(
                self._previous[0], self._incoming, calibration, contrast=self.contrast
            )
            attempts.append(result.diagnostics)
            if result.T_target_source is not None:
                transform = (
                    np.linalg.inv(calibration.T_RD)
                    @ result.T_target_source
                    @ calibration.T_RD
                )
                accepted.append(
                    (previous_pose @ np.linalg.inv(transform), index, result)
                )
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
