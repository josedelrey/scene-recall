"""Owned pending image rays and verified metric-anchor identity bridges.

Unknown range stays NaN in immutable raw metadata. Only observable initialized
classes enter the Euclidean shared solver. Registration transforms initialize
placements and never add a second measurement likelihood.
"""

from dataclasses import dataclass

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.fixed_landmarks import DirectFactors
from scene_recall.multiview.tracking import match_descriptors
from scene_recall.slam.observation_welding import LoopValidation, RecordedView
from scene_recall.slam.registration import register_anchor_pairs


@dataclass(frozen=True)
class BearingConfig:
    mode: str = "off"
    extra_per_view: int = 128
    recent_cameras: int = 3
    min_views: int = 3
    min_parallax_deg: float = 1.0
    max_reprojection_px: float = 3.0

    def __post_init__(self):
        if self.mode not in ("off", "retain", "anchor", "multiview"):
            raise ValueError("unknown bearing mode")
        for name, lower, upper in (
            ("extra_per_view", 1, 1024),
            ("recent_cameras", 1, 8),
            ("min_views", 3, 20),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not lower <= value <= upper
            ):
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        for name in ("min_parallax_deg", "max_reprojection_px"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_parallax_deg >= 90:
            raise ValueError("bearing parallax must be below 90 degrees")


class RawObservations(DirectFactors):
    """Raw packet whose seed metadata can explicitly have unknown metric range.

    Solver factors are constructed separately with finite initialized points.
    Pixels, native samples, exposure transforms and weights retain the strict
    DirectFactors validation. A pending point has three NaNs, never partial NaNs.
    """

    def _valid_seed_points(self):
        return (
            np.isfinite(self.points).all(axis=1) | np.isnan(self.points).all(axis=1)
        ).all()


def triangulate_bearings(poses, pixels, K, config=None):
    """Initialize finite range from rays in one already metric-connected gauge."""
    config = config or BearingConfig()
    if len(poses) < config.min_views:
        return None, "insufficient views"
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
    rays = np.einsum("nij,nj->ni", poses[:, :3, :3], rays)
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    cosine = np.clip(rays @ rays.T, -1, 1)
    angle = float(np.degrees(np.arccos(cosine.min())))
    if angle < config.min_parallax_deg:
        return None, "insufficient parallax"
    projection = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]
    information = projection.sum(axis=0)
    values = np.linalg.eigvalsh(information)
    if values[0] <= max(1e-12, values[-1] * 1e-8):
        return None, "unobservable range"
    point = np.linalg.solve(
        information, np.einsum("nij,nj->i", projection, poses[:, :3, 3])
    )
    local = np.einsum("nji,nj->ni", poses[:, :3, :3], point - poses[:, :3, 3])
    if (local[:, 2] <= 0.05).any():
        return None, "failed cheirality"
    error = np.linalg.norm(project_points(local, K) - pixels, axis=1)
    if not np.isfinite(error).all() or error.max() > config.max_reprojection_px:
        return None, "inconsistent image observations"
    return point, "initialized"


def compatible_rays(pixels_a, pixels_b, transform, K, threshold):
    """Sampson pixel gate, with a rotation-only check at zero baseline."""
    a, b = (
        np.c_[pixels_a, np.ones(len(pixels_a))],
        np.c_[pixels_b, np.ones(len(pixels_b))],
    )
    t = transform[:3, 3]
    if np.linalg.norm(t) < 1e-8:
        rays = a @ np.linalg.inv(K).T @ transform[:3, :3].T
        return (rays[:, 2] > 0) & (
            np.linalg.norm(project_points(rays, K) - pixels_b, axis=1) <= threshold
        )
    x, y, z = t / np.linalg.norm(t)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    inverse = np.linalg.inv(K)
    F = inverse.T @ cross @ transform[:3, :3] @ inverse
    line_b, line_a = a @ F.T, b @ F
    denominator = np.sqrt(
        (line_a[:, :2] ** 2).sum(axis=1) + (line_b[:, :2] ** 2).sum(axis=1)
    )
    return np.abs(np.sum(b * line_b, axis=1)) <= threshold * np.maximum(
        denominator, 1e-15
    )


class BearingLedger:
    """Optional methods on the persistent shared raw-observation ledger."""

    def _admit_bearings(self, frame, segment, measurement):
        if measurement is None or not len(measurement["view"]):
            return False
        if frame in self.views:
            raise ValueError("keyframe factors already admitted")
        RawObservations(
            **{name: measurement[name] for name in DirectFactors.__dataclass_fields__}
        )
        if not np.all(measurement["view"] == frame):
            raise ValueError("factor packet belongs to another camera")
        for name in ("feature_index", "native_index", "frontend_identity"):
            values = measurement[name]
            if (
                values.shape != measurement["view"].shape
                or values.dtype.kind not in "iu"
            ):
                raise ValueError(f"invalid {name} provenance")
        if (measurement["feature_index"] < 0).any():
            raise ValueError("feature indices must be nonnegative")
        if len(np.unique(measurement["feature_index"])) != len(measurement["view"]):
            raise ValueError("feature observation has multiple owners")
        owners = measurement["frontend_identity"]
        owned = owners >= 0
        if len(np.unique(owners[owned])) != int(owned.sum()):
            raise ValueError("frontend identity has multiple features in one view")
        if not np.array_equal(
            measurement["native_index"] >= 0,
            np.isfinite(measurement["native_points"]).all(axis=1),
        ):
            raise ValueError("native ownership does not match measured samples")
        baseline = {name: values[owned] for name, values in measurement.items()}
        admitted = super().admit(frame, segment, baseline) if owned.any() else False
        view = self.views.get(frame)
        features = set(view.feature_index) if view is not None else set()
        used_native = (
            set(view.native_index[view.native_index >= 0])
            if view is not None
            else set()
        )
        candidates = [
            i
            for i, feature in enumerate(measurement["feature_index"])
            if feature not in features
        ]
        selected = []
        for i in self._spatial_order(measurement["pixels"], candidates):
            sample = int(measurement["native_index"][i])
            if sample >= 0 and sample in used_native:
                continue
            selected.append(i)
            if sample >= 0:
                used_native.add(sample)
            if len(selected) >= self.shared.bearings.extra_per_view:
                break
        if not selected:
            return admitted
        selected = np.array(selected, dtype=int)
        tracks = []
        for i in selected:
            owner = int(measurement["frontend_identity"][i])
            # Negative keys identify unowned raw features, never frontend points.
            if owner < 0:
                owner = -1 - ((int(frame) << 32) + int(measurement["feature_index"][i]))
            key = segment, owner
            if key not in self.identity_map:
                self.identity_map[key] = self._next_identity
                self._next_identity += 1
            tracks.append(self.identity_map[key])
        arrays = {
            name: measurement[name][selected].copy()
            for name in DirectFactors.__dataclass_fields__
        }
        arrays["track_id"] = np.array(tracks, dtype=int)
        indices = {
            name: measurement[name][selected].copy()
            for name in ("feature_index", "native_index")
        }
        if view is not None:
            arrays = {
                name: np.concatenate((getattr(view.factors, name), value))
                for name, value in arrays.items()
            }
            indices = {
                name: np.r_[getattr(view, name), value]
                for name, value in indices.items()
            }
        packet = RawObservations(**arrays)
        if len(np.unique(indices["feature_index"])) != len(packet.view):
            raise ValueError("bearing observation has multiple owners")
        for array in (*arrays.values(), *indices.values()):
            array.setflags(write=False)
        self.views[frame] = RecordedView(
            packet, indices["feature_index"], indices["native_index"], segment
        )
        return True

    def link_bearings(self, source, target, first, second, registration):
        """Join fitting observations under disjoint native-anchor verification."""
        if source not in self.views or target not in self.views:
            return {"status": "unavailable"}
        if source >= target or self.views[source].segment != self.views[target].segment:
            raise ValueError("bearing links require earlier same-segment cameras")
        if (source, target) in self.associations:
            return {"status": "already_proposed"}
        for frame, features in ((source, first), (target, second)):
            view = self.views[frame]
            if not np.array_equal(
                features.pixels[view.feature_index], view.factors.pixels
            ):
                raise ValueError("bearing pixels differ from their owned observation")
        first, second = (
            self._native_features(source, first),
            self._native_features(target, second),
        )
        attempts = []
        for reverse in (False, True):
            a, b = (second, first) if reverse else (first, second)
            left, right = (target, source) if reverse else (source, target)
            pairs, _ = match_descriptors(
                a.descriptors, b.descriptors, registration.ratio
            )
            pairs = pairs[
                np.isin(pairs[:, 0], self.views[left].feature_index)
                & np.isin(pairs[:, 1], self.views[right].feature_index)
            ]
            result = register_anchor_pairs(a, b, pairs, self.calibration, registration)
            attempts.append(result.diagnostics)
            if result.T_target_source is not None:
                break
        else:
            return {"status": "registration_rejected", "attempts": attempts}
        anchor_pairs = pairs[np.isfinite(a.points_R[pairs[:, 0]]).all(axis=1)]
        q, t = anchor_pairs[::2].T
        predicted = transform_points(a.points_R[q], result.T_target_source)
        good = (predicted[:, 2] > 0.05) & (
            np.linalg.norm(
                project_points(predicted, self.calibration.K_R) - b.pixels[t], axis=1
            )
            <= registration.reprojection_px
        )
        native = np.isfinite(b.points_R[t]).all(axis=1)
        limits = registration.depth_absolute_m + registration.depth_relative * np.where(
            native, b.points_R[t, 2], 0
        )
        good &= ~native | (np.linalg.norm(predicted - b.points_R[t], axis=1) <= limits)
        fitting = list(zip(q[good], t[good], strict=True))
        backward_pairs = pairs[
            ~np.isfinite(a.points_R[pairs[:, 0]]).all(axis=1)
            & np.isfinite(b.points_R[pairs[:, 1]]).all(axis=1)
        ][::2]
        q, t = backward_pairs.T
        predicted = transform_points(
            b.points_R[t], np.linalg.inv(result.T_target_source)
        )
        good = (predicted[:, 2] > 0.05) & (
            np.linalg.norm(
                project_points(predicted, self.calibration.K_R) - a.pixels[q], axis=1
            )
            <= registration.reprojection_px
        )
        fitting.extend(zip(q[good], t[good], strict=True))
        if self.shared.bearings.mode == "multiview":
            # These pairs have no fitting or withheld native anchors. They remain
            # pending until multiview range is independently geometrically usable.
            unbound = pairs[
                ~np.isfinite(a.points_R[pairs[:, 0]]).all(axis=1)
                & ~np.isfinite(b.points_R[pairs[:, 1]]).all(axis=1)
            ][::2]
            good = compatible_rays(
                a.pixels[unbound[:, 0]],
                b.pixels[unbound[:, 1]],
                result.T_target_source,
                self.calibration.K_R,
                registration.reprojection_px,
            )
            fitting.extend(map(tuple, unbound[good]))
        left, right = (target, source) if reverse else (source, target)
        lookup = {
            frame: dict(
                zip(
                    self.views[frame].feature_index,
                    self.views[frame].factors.track_id,
                    strict=True,
                )
            )
            for frame in (source, target)
        }
        links = tuple(
            (int(lookup[left][i]), int(lookup[right][j]))
            for i, j in fitting
            if i in lookup[left] and j in lookup[right]
        )
        if len(links) < self.config.min_associations:
            return {
                "status": "insufficient",
                "associations": len(links),
                "attempts": attempts,
            }
        self.associations[source, target] = links
        heldout = anchor_pairs[1::2]
        validation = LoopValidation(
            a.points_R[heldout[:, 0]].copy(),
            b.points_R[heldout[:, 1]].copy(),
            b.pixels[heldout[:, 1]].copy(),
        )
        for array in (
            validation.source_points,
            validation.target_points,
            validation.target_pixels,
        ):
            array.setflags(write=False)
        self.validation[source, target] = validation
        transform = (
            np.linalg.inv(result.T_target_source) if reverse else result.T_target_source
        )
        depth_transform = (
            np.linalg.inv(self.views[target].factors.T_RcD[0])
            @ transform
            @ self.views[source].factors.T_RcD[0]
        )
        depth_transform.setflags(write=False)
        self.reobservations[source, target] = depth_transform
        self.bearing_links[source, target] = int(left)
        return {
            "status": "admitted",
            "associations": len(links),
            "anchor": int(left),
            "attempts": attempts,
        }

    def _validation_score(self, source, target, source_pose, target_pose):
        if self.bearing_links.get((source, target)) == target:
            source_pose, target_pose = target_pose, source_pose
        return super()._validation_score(source, target, source_pose, target_pose)

    def _triangulated_seeds(self, keys, poses, identities, native, raw, edited):
        seeds, rejected, rows = {}, {}, {}
        if self.shared.bearings.mode != "multiview":
            return seeds, rejected
        for camera, frame in enumerate(keys):
            packet = self.views[frame].factors
            for track, pixel in zip(packet.track_id, packet.pixels, strict=True):
                identity = identities[int(track)]
                if identity not in native:
                    rows.setdefault(identity, []).append((camera, pixel))
        for identity, observations in rows.items():
            cameras = np.array([camera for camera, _ in observations], dtype=int)
            if len(cameras) < self.shared.bearings.min_views:
                reason = "insufficient views"
            elif len({raw[keys[c]] for c in cameras}) > 1:
                reason = "independent metric gauges"
            elif not edited and identity in self.estimates:
                seeds[identity] = self.estimates[identity].copy()
                continue
            else:
                point, reason = triangulate_bearings(
                    poses[cameras],
                    np.array([pixel for _, pixel in observations]),
                    self.calibration.K_R,
                    self.shared.bearings,
                )
                if point is not None:
                    seeds[identity] = point
                    continue
            rejected[reason] = rejected.get(reason, 0) + 1
        return seeds, rejected
