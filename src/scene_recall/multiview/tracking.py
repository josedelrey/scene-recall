"""Deterministic offline track graph with conservative component merges."""

from dataclasses import replace
from itertools import pairwise
from time import perf_counter

import cv2
import numpy as np

from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.geometry import landmark_geometry, pair_geometry
from scene_recall.multiview.model import Association, Landmark, TrackConfig, TrackResult


def match_descriptors(first, second, ratio: float = 0.7):
    """Mutual two-neighbor Lowe checks in both directions, without train reuse."""
    if not np.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("ratio must be strictly between zero and one")
    if first is None or second is None or min(len(first), len(second)) < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def qualified(a, b):
        return {
            row[0].queryIdx: (row[0].trainIdx, row[0].distance / row[1].distance)
            for row in matcher.knnMatch(a, b, k=2)
            if len(row) == 2 and row[0].distance < ratio * row[1].distance
        }

    forward, reverse = qualified(first, second), qualified(second, first)
    matches, ratios = [], []
    for query, (train, forward_ratio) in sorted(forward.items()):
        if train in reverse and reverse[train][0] == query:
            matches.append((query, train))
            ratios.append(max(forward_ratio, reverse[train][1]))
    return np.array(matches, dtype=np.int64).reshape(-1, 2), np.array(ratios)


def build_tracks(frames, calibration, config=None, *, pairs=None) -> TrackResult:
    """Associate a bounded collection of extracted views on fixed estimated poses.

    Default offsets refer to positions in the supplied ordered frame list.
    Explicit pairs use frame indices and permit nonconsecutive revisits. Neither
    mode searches across segment boundaries. No image retrieval or relocalization
    is attempted. Each merge rechecks all members and rejects duplicate views.
    Final IDs are assigned deterministically by each component's smallest key.
    """
    config = config or TrackConfig()
    frames = tuple(frames)
    frame_map = {frame.frame_index: frame for frame in frames}
    if len(frame_map) != len(frames) or list(frame_map) != sorted(frame_map):
        raise ValueError("feature frames must have unique increasing indices")
    if any(a.segment_id > b.segment_id for a, b in pairwise(frames)):
        raise ValueError("feature frame segments must be ordered and never reused")
    height, width = calibration.rgb_shape
    for frame in frames:
        xy = frame.features.pixels
        if (xy < 0).any() or (xy[:, 0] >= width).any() or (xy[:, 1] >= height).any():
            raise ValueError("feature pixels must lie inside the calibrated RGB grid")
    if pairs is None:
        pairs = [
            (frames[i - offset].frame_index, frame.frame_index)
            for i, frame in enumerate(frames)
            for offset in config.pair_offsets
            if i >= offset and frames[i - offset].segment_id == frame.segment_id
        ]
    pairs = tuple(sorted(pairs))
    if len(set(pairs)) != len(pairs):
        raise ValueError("candidate pairs must be unique")
    for source, target in pairs:
        if source not in frame_map or target not in frame_map or source >= target:
            raise ValueError("pairs must reference increasing known frame indices")
        if frame_map[source].segment_id != frame_map[target].segment_id:
            raise ValueError("candidate pairs cannot cross trajectory segments")
    started = perf_counter()
    proposals, associations = [], []
    for source, target in pairs:
        first, second = frame_map[source], frame_map[target]
        matches, ratios = match_descriptors(
            first.features.descriptors, second.features.descriptors, config.ratio
        )
        if not len(matches):
            continue
        scores, pixels, depths, anchored = pair_geometry(
            first, second, matches[:, 0], matches[:, 1], calibration.K_R, config
        )
        for (query, train), ratio, score, pixel, depth, anchor in zip(
            matches, ratios, scores, pixels, depths, anchored
        ):
            status = (
                "proposed"
                if score <= 1
                else "uncertain"
                if score <= config.relaxed_factor
                else "rejected"
            )
            reason = (
                "strict endpoint geometry"
                if score <= 1
                else "relaxed geometry only"
                if status == "uncertain"
                else "endpoint geometry"
                if anchor
                else "no depth anchor"
            )
            association = Association(
                (source, int(query)),
                (target, int(train)),
                float(ratio),
                float(pixel) if anchor and np.isfinite(pixel) else None,
                float(depth) if np.isfinite(depth) else None,
                status,
                reason,
            )
            index = len(associations)
            associations.append(association)
            if status == "proposed":
                # Descriptor confidence first. Geometry and keys break ties.
                proposals.append(
                    (ratio, score, association.source, association.target, index)
                )
    matching_seconds = perf_counter() - started
    # Sparse union graph. Unmatched features do not become singleton landmarks.
    geometry_cache = {
        f.frame_index: (
            np.linalg.inv(f.T_SR),
            transform_points(f.features.points_R, f.T_SR),
        )
        for f in frames
    }
    owner, components = {}, {}
    started = perf_counter()
    for _, _, a, b, index in sorted(proposals):
        root_a, root_b = owner.get(a, a), owner.get(b, b)
        if root_a == root_b:
            associations[index] = replace(
                associations[index], status="redundant", reason="direct cycle check"
            )
            continue
        keys_a, keys_b = components.get(root_a, (a,)), components.get(root_b, (b,))
        keys = tuple(sorted(keys_a + keys_b))
        if len({key[0] for key in keys}) != len(keys):
            associations[index] = replace(
                associations[index],
                status="conflict",
                reason="multiple features in one view",
            )
            continue
        geometry = landmark_geometry(
            keys, frame_map, calibration.K_R, config, geometry_cache
        )
        if geometry is None or geometry["score"] > 1:
            associations[index] = replace(
                associations[index],
                status="incompatible",
                reason="all-view landmark validation",
            )
            continue
        root = keys[0]
        components.pop(root_a, None)
        components.pop(root_b, None)
        components[root] = keys
        owner.update({key: root for key in keys})
        associations[index] = replace(
            associations[index],
            status="accepted",
            reason="all-view landmark validation",
        )
    landmarks = []
    for track_id, (_, keys) in enumerate(sorted(components.items())):
        geometry = landmark_geometry(
            keys, frame_map, calibration.K_R, config, geometry_cache
        )
        centers = geometry.pop("centers")
        directions = geometry.pop("directions")
        baseline = float(np.linalg.norm(centers[:, None] - centers, axis=2).max())
        parallax = float(
            np.degrees(np.arccos(np.clip((directions @ directions.T).min(), -1, 1)))
        )
        geometry.pop("score")
        supported = (
            len(keys) >= config.min_views
            and geometry["depth_views"] >= config.min_depth_views
        )
        landmarks.append(
            Landmark(
                track_id,
                frame_map[keys[0][0]].segment_id,
                keys,
                status="supported" if supported else "tentative",
                parallax_deg=parallax,
                baseline_m=baseline,
                **geometry,
            )
        )
    return TrackResult(
        frames,
        tuple(landmarks),
        tuple(associations),
        pairs,
        config,
        {
            "matching_seconds": matching_seconds,
            "track_construction_seconds": perf_counter() - started,
        },
    )
