"""Locate causal changes in owned measurements and temporal associations.

Equal source inputs need not produce equal admitted factor populations after
estimator feedback changes tracking. Compare sensor ownership independently of
arbitrary track numbering and report publication deltas only in a common root.
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from scene_recall.geometry.camera import transform_points
from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def compare(first, second):
    loaded = [load_trajectory(path) for path in (first, second)]
    reports = [item[0] for item in loaded]
    if (
        reports[0]["source"]["canonical_input_sha256"]
        != reports[1]["source"]["canonical_input_sha256"]
    ):
        raise ValueError("comparison requires identical received input populations")
    arrays = [dict(np.load(path / "shared_estimation.npz")) for path in (first, second)]
    atlases = [dict(np.load(path / "atlas_emitted.npz")) for path in (first, second)]
    keys = sorted(set(arrays[0]["view"]) | set(arrays[1]["view"]))
    packets = [
        {
            frame: {
                int(feature): i
                for i, feature in enumerate(a["feature_index"])
                if a["view"][i] == frame
            }
            for frame in keys
        }
        for a in arrays
    ]
    measurements = []
    for frame in keys:
        a, b = (p[frame] for p in packets)
        shared = sorted(set(a) & set(b))
        i, j = (
            np.array([a[k] for k in shared], dtype=int),
            np.array([b[k] for k in shared], dtype=int),
        )
        measurements.append(
            {
                "frame": int(frame),
                "first_rows": len(a),
                "second_rows": len(b),
                "same_selected_features": set(a) == set(b),
                "same_pixels_on_common_features": bool(
                    np.array_equal(arrays[0]["pixels"][i], arrays[1]["pixels"][j])
                ),
                "same_native_owners_on_common_features": bool(
                    np.array_equal(
                        arrays[0]["native_index"][i], arrays[1]["native_index"][j]
                    )
                ),
                "maximum_exposure_matrix_difference": float(
                    np.abs(arrays[0]["T_RcD"][i] - arrays[1]["T_RcD"][j]).max(initial=0)
                ),
            }
        )
    links = []
    for previous, frame in pairwise(keys):
        pairs = []
        for a, p in zip(arrays, packets, strict=True):
            by_track = [
                {int(a["track_id"][i]): feature for feature, i in p[key].items()}
                for key in (previous, frame)
            ]
            pairs.append(
                {
                    (by_track[0][track], by_track[1][track])
                    for track in set(by_track[0]) & set(by_track[1])
                }
            )
        links.append(
            {
                "source": int(previous),
                "target": int(frame),
                "first_links": len(pairs[0]),
                "second_links": len(pairs[1]),
                "same_temporal_pairs": pairs[0] == pairs[1],
            }
        )
    common = [
        i
        for i in range(len(reports[0]["frames"]))
        if all(a["component_id"][i] == a["component_id"][0] for a in atlases)
    ]
    poses = [
        np.linalg.inv(a["T_component_depth"][0]) @ a["T_component_depth"]
        for a in atlases
    ]
    changes = [
        (
            i,
            *pose_error(poses[0][i], poses[1][i]),
        )
        for i in common
    ]
    # Score common physical native samples, rather than comparing different
    # optimized landmark populations after causal association changes.
    final = [dict(np.load(path / "atlas.npz")) for path in (first, second)]
    points = [
        dict(zip(a["point_ids"], a["point_positions"], strict=True)) for a in arrays
    ]
    identities = [dict(a["live_identities"]) for a in arrays]
    matched = []
    for frame in keys:
        if not all(a["component_id"][frame] == a["component_id"][0] for a in final):
            continue
        if loaded[0][1][int(frame)].reference_pose is None:
            continue
        for feature in sorted(set(packets[0][frame]) & set(packets[1][frame])):
            i, j = (p[frame][feature] for p in packets)
            rows = (i, j)
            owners = [
                identities[k][arrays[k]["track_id"][row]] for k, row in enumerate(rows)
            ]
            if not all(owner in points[k] for k, owner in enumerate(owners)):
                continue
            if (
                arrays[0]["native_index"][i] < 0
                or arrays[0]["native_index"][i] != arrays[1]["native_index"][j]
            ):
                continue
            if not np.array_equal(arrays[0]["pixels"][i], arrays[1]["pixels"][j]):
                continue
            if not np.isfinite(
                arrays[0]["native_points"][i]
            ).all() or not np.array_equal(
                arrays[0]["native_points"][i], arrays[1]["native_points"][j]
            ):
                continue
            matched.append((int(frame), i, j, *owners))
    matched_scores = []
    anchor = next(
        (
            i
            for i, row in enumerate(loaded[0][1])
            if row.reference_pose is not None
            and all(a["component_id"][i] == a["component_id"][0] for a in final)
        ),
        None,
    )
    if anchor is not None:
        for k in range(2):
            native_errors, reference_errors = [], []
            alignment = final[k]["T_component_depth"][anchor] @ np.linalg.inv(
                loaded[0][1][anchor].reference_pose
            )
            for frame, i, j, a, b in matched:
                point = points[k][(a, b)[k]]
                native = arrays[k]["native_points"][(i, j)[k]]
                prediction = transform_points(
                    point, np.linalg.inv(final[k]["T_component_depth"][frame])
                )
                native_errors.append(float(np.linalg.norm(prediction - native)))
                placed = transform_points(
                    native, alignment @ loaded[0][1][frame].reference_pose
                )
                reference_errors.append(float(np.linalg.norm(point - placed)))
            matched_scores.append(
                {
                    "native_residual_m": error_statistics(native_errors),
                    "reference_placed_native_discrepancy_m": error_statistics(
                        reference_errors
                    ),
                }
            )

    def first_matching(values, predicate):
        return next((value for value in values if predicate(value)), None)

    return {
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "reports_sha256": [sha256(path / "report.json") for path in (first, second)],
        "same_source_sha256": reports[0]["source"]["canonical_input_sha256"],
        "first_feature_selection_change": first_matching(
            measurements, lambda row: not row["same_selected_features"]
        ),
        "first_native_owner_change": first_matching(
            measurements, lambda row: not row["same_native_owners_on_common_features"]
        ),
        "first_pixel_measurement_change": first_matching(
            measurements, lambda row: not row["same_pixels_on_common_features"]
        ),
        "first_temporal_pair_change": first_matching(
            links, lambda row: not row["same_temporal_pairs"]
        ),
        "first_emitted_difference_above_1um": first_matching(
            changes, lambda row: row[1] > 1e-6
        ),
        "first_emitted_difference_above_1mm": first_matching(
            changes, lambda row: row[1] > 1e-3
        ),
        "common_root_emitted_translation_difference_m": error_statistics(
            row[1] for row in changes
        ),
        "matched_native_observations": len(matched),
        "matched_native_anchor": anchor,
        "matched_native_scores": matched_scores,
        "measurements": measurements,
        "temporal_links": links,
        "caveat": "exposure transforms are conditioned on past estimated motion, missing packets differ in population, raw track labels are ignored for temporal feature pairs, matched reference-placed native samples are not independent physical point truth",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh evidence comparison output")
    args.output.write_text(
        json.dumps(compare(args.first, args.second), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
