"""Track support, covisibility and independent depth-sample checks."""

from collections import Counter, defaultdict
from itertools import combinations, pairwise

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points


def statistics(values):
    """JSON-safe distributions, with explicit empty populations."""
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        "median": float(np.median(values)) if len(values) else None,
        "p90": float(np.percentile(values, 90)) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def grid_coverage(pixels, shape, rows=6, columns=8):
    """Occupied fraction of a fixed RGB image grid, without bounding-box inflation."""
    if not len(pixels):
        return 0.0
    pixels = np.asarray(pixels)
    cells = np.floor(pixels / [shape[1], shape[0]] * [columns, rows]).astype(int)
    cells = np.clip(cells, [0, 0], [columns - 1, rows - 1])
    return len(set(map(tuple, cells))) / (rows * columns)


def summarize_tracks(result, calibration, *, include_covisibility=True):
    """Measure constraints conditional on estimated poses, without using GT.

    Leave-one-depth-out predicts a held view from other depth samples, retaining
    fixed estimated poses. It is independent of that sample, not of VO or the
    original association gates. Covisibility counts shared identities, not an
    information-matrix rank or proof of correct static-scene correspondence.
    """
    frames = {frame.frame_index: frame for frame in result.frames}
    positions = {frame.frame_index: i for i, frame in enumerate(result.frames)}
    per_frame = defaultdict(list)
    shared = defaultdict(lambda: [[], []])
    loo_pixels, loo_depths, loo_pass = [], [], []
    has_gaps = beyond_chain = 0
    edge_set = {
        (edge.source, edge.target)
        for edge in result.associations
        if edge.status in ("accepted", "redundant")
        and positions[edge.target[0]] - positions[edge.source[0]] == 1
    }
    for track in result.landmarks:
        keys = track.observations
        frame_ids = [key[0] for key in keys]
        has_gaps += any(positions[b] - positions[a] > 1 for a, b in pairwise(frame_ids))
        adjacent_edges = sum((a, b) in edge_set for a, b in pairwise(keys))
        beyond_chain += adjacent_edges < len(keys) - 1
        for key in keys:
            per_frame[key[0]].append(key[1])
        for a, b in combinations(frame_ids, 2) if include_covisibility else ():
            shared[a, b][0].append(track.track_id)
            if track.status == "supported":
                shared[a, b][1].append(track.track_id)
        samples = {
            key: transform_points(
                frames[key[0]].features.points_R[key[1]], frames[key[0]].T_SR
            )
            for key in keys
            if np.isfinite(frames[key[0]].features.points_R[key[1]]).all()
        }
        if len(samples) < 2:
            continue
        for key in keys:
            other = [
                sample for sample_key, sample in samples.items() if sample_key != key
            ]
            if not other:
                continue
            frame = frames[key[0]]
            point = transform_points(
                np.median(other, axis=0), np.linalg.inv(frame.T_SR)
            )
            error = float(
                np.linalg.norm(
                    project_points(point, calibration.K_R)
                    - frame.features.pixels[key[1]]
                )
            )
            # Behind-camera predictions count as failures, never disappear as NaNs.
            loo_pixels.append(error if np.isfinite(error) else np.inf)
            valid = np.isfinite(frame.features.points_R[key[1]]).all()
            depth_error = (
                float(np.linalg.norm(point - frame.features.points_R[key[1]]))
                if valid
                else None
            )
            if valid:
                loo_depths.append(depth_error)
            passed = np.isfinite(error) and error <= result.config.reprojection_px
            if valid:
                z = frame.features.points_R[key[1], 2]
                passed &= (
                    depth_error
                    <= result.config.depth_absolute_m + result.config.depth_relative * z
                )
            loo_pass.append(bool(passed))
    pixels_by_track = (
        {
            track.track_id: {
                index: frames[index].features.pixels[feature_id]
                for index, feature_id in track.observations
            }
            for track in result.landmarks
        }
        if include_covisibility
        else {}
    )
    covisibility = []
    for (a, b), (ids, supported) in sorted(shared.items()):
        first, second = frames[a], frames[b]
        coverages = []
        for frame_id in (a, b):
            pixels = [pixels_by_track[track_id][frame_id] for track_id in supported]
            coverages.append(grid_coverage(pixels, calibration.rgb_shape))
        covisibility.append(
            {
                "source": a,
                "target": b,
                "frame_gap": b - a,
                "selected_view_gap": positions[b] - positions[a],
                "shared_tracks": len(ids),
                "shared_supported": len(supported),
                "supported_grid_coverage_min": min(coverages),
                "baseline_m": float(
                    np.linalg.norm(first.T_SR[:3, 3] - second.T_SR[:3, 3])
                ),
            }
        )
    frame_diagnostics = []
    for frame in result.frames:
        points = frame.features.points_R
        assigned = per_frame[frame.frame_index]
        frame_diagnostics.append(
            {
                "frame_index": frame.frame_index,
                "segment_id": frame.segment_id,
                "features": len(points),
                "depth_features": int(np.isfinite(points).all(axis=1).sum()),
                "tracked_features": len(assigned),
                "track_grid_coverage": grid_coverage(
                    frame.features.pixels[assigned], calibration.rgb_shape
                ),
            }
        )
    statuses = Counter(edge.status for edge in result.associations)
    accepted = [
        edge for edge in result.associations if edge.status in ("accepted", "redundant")
    ]
    supported = [track for track in result.landmarks if track.status == "supported"]
    spans_s = [
        frames[t.observations[-1][0]].timestamp - frames[t.observations[0][0]].timestamp
        for t in result.landmarks
        if all(frames[k[0]].timestamp is not None for k in t.observations)
    ]
    by_gap = {}
    for gap in sorted({positions[b] - positions[a] for a, b in result.pairs}):
        edges = [
            edge
            for edge in result.associations
            if positions[edge.target[0]] - positions[edge.source[0]] == gap
        ]
        by_gap[str(gap)] = {
            "tested_pairs": sum(
                positions[b] - positions[a] == gap for a, b in result.pairs
            ),
            "descriptor_proposals": len(edges),
            "statuses": dict(Counter(edge.status for edge in edges)),
        }
    return {
        "frame_count": len(frames),
        "segments": len({f.segment_id for f in result.frames}),
        "tracks": len(result.landmarks),
        "supported_tracks": len(supported),
        "tentative_tracks": len(result.landmarks) - len(supported),
        "track_observations": sum(len(t.observations) for t in result.landmarks),
        "lengths": statistics(len(t.observations) for t in result.landmarks),
        "length_histogram": dict(
            sorted(Counter(str(len(t.observations)) for t in result.landmarks).items())
        ),
        "span_frames": statistics(
            t.observations[-1][0] - t.observations[0][0] for t in result.landmarks
        ),
        "span_seconds": statistics(spans_s),
        "depth_views": statistics(t.depth_views for t in result.landmarks),
        "depth_observation_fraction": sum(t.depth_views for t in result.landmarks)
        / max(1, sum(len(t.observations) for t in result.landmarks)),
        "supported_baseline_m": statistics(t.baseline_m for t in supported),
        "supported_parallax_deg": statistics(t.parallax_deg for t in supported),
        "supported_with_1deg_parallax": sum(t.parallax_deg >= 1 for t in supported),
        "reprojection_px": statistics(
            e for t in result.landmarks for e in t.reprojection_errors_px
        ),
        "depth_error_m": statistics(
            e for t in result.landmarks for e in t.depth_errors_m
        ),
        "leave_one_depth_out": {
            "pixel_errors_px": statistics(loo_pixels),
            "depth_errors_m": statistics(loo_depths),
            "tested_observations": len(loo_pass),
            "strict_pass_fraction": sum(loo_pass) / len(loo_pass) if loo_pass else None,
            "invalid_projections": sum(not np.isfinite(e) for e in loo_pixels),
        },
        "tracks_with_selected_view_gaps": has_gaps,
        "tracks_not_connected_by_adjacent_direct_edges": beyond_chain,
        "association_statuses": dict(statuses),
        "association_reasons": dict(
            Counter(edge.reason for edge in result.associations)
        ),
        "direct_nonconsecutive_edges": sum(
            positions[e.target[0]] - positions[e.source[0]] > 1 for e in accepted
        ),
        "direct_cycle_checks": statuses["redundant"],
        "by_selected_view_gap": by_gap,
        "per_frame": frame_diagnostics,
        "covisibility": covisibility,
    }
