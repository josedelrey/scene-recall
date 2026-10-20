"""Post-tracking measurement diagnostics with reference-free fitting populations.

References enter only residual evaluation. Sampled packets were selected by
index before scores. Pixel/depth tests diagnose compatibility, not truth labels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from local_map_measurements import refine_measurements

from scene_recall.datasets.exposure import exposure_times
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.tracking import match_descriptors
from scene_recall.odometry.evaluation import error_statistics
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.sparse import SparseConfig, estimate_sparse_pose
from scene_recall.run_metadata import sha256


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        **error_statistics(values),
        "p90": float(np.percentile(values, 90)) if len(values) else None,
    }


def diagnose_case(path):
    report, rows = load_trajectory(path / "legacy")
    T_RD = np.array(report["calibration"]["T_RD"])
    K = np.array(report["calibration"]["K_R"])
    K_D = np.array(report["calibration"]["K_D"])
    duplicate_rows = []
    result = {"run_report_sha256": sha256(path / "legacy/report.json")}
    skews = []
    for f in report["frames"]:
        c, d = exposure_times(report["source"]["dataset"], f["provenance"])
        if c is not None and d is not None:
            skews.append(d - c)
    result["rgb_depth_skew_s"] = stats(skews)
    diagnostics = [f["diagnostics"] for f in report["frames"][1:]]
    result["accepted_pose_diagnostics"] = {
        key: stats(
            [
                d.get(key, np.nan) if d.get(key) is not None else np.nan
                for d in diagnostics
            ]
        )
        for key in (
            "conditional_rgb_chi2_mean",
            "conditional_depth_chi2_mean",
            "signed_depth_median_m",
            "conditional_depth_over_3sigma",
            "map_inliers",
            "map_grid_coverage",
            "map_information_ratio",
        )
    }
    collected = {"recent": {}, "older": {}}
    heldout = {
        n: {"rgb_squared": [], "depth_squared": []}
        for n in ("legacy", "robust_rgb", "rgb", "rgbd")
    }
    trial_rows = []
    with np.load(path / "legacy/measurement_packets.npz") as a:
        numbers = sorted({int(k.split("_")[0]) for k in a.files})
        for i in numbers:
            packet = {
                k: a[f"{i}_{k}"]
                for k in (
                    "points",
                    "pixels",
                    "measured",
                    "covariance",
                    "pose",
                    "K",
                    "anchor_indices",
                    "anchor_points",
                    "target",
                )
            }
            target = int(packet["target"])
            if np.array_equal(T_RD, np.eye(4)):
                anchor_native = np.floor(
                    project_points(packet["anchor_points"], K_D) + 0.5
                ).astype(int)
                anchor_keys = np.column_stack((packet["anchor_indices"], anchor_native))
                current = np.floor(packet["pixels"] + 0.5).astype(int)
                duplicate_rows.append(
                    {
                        "target": target,
                        "support": len(current),
                        "unique_current_rgb_cells": len(np.unique(current, axis=0)),
                        "unique_anchor_native_samples": len(
                            np.unique(anchor_keys, axis=0)
                        ),
                    }
                )
            # Reference relative motion evaluates source depth and matched target
            # measurements, without the estimated map's anchoring error.
            if rows[target].reference_pose is not None:
                ids = packet["anchor_indices"]
                valid = np.array([rows[int(j)].reference_pose is not None for j in ids])
                for label, age in (
                    ("recent", ids == target - 1),
                    ("older", ids < target - 1),
                ):
                    keep = valid & age
                    if not keep.any():
                        continue
                    relatives = np.array(
                        [
                            T_RD
                            @ np.linalg.inv(rows[target].reference_pose)
                            @ rows[int(j)].reference_pose
                            @ np.linalg.inv(T_RD)
                            for j in ids[keep]
                        ]
                    )
                    anchor = packet["anchor_points"][keep]
                    expected = (relatives[:, :3, :3] @ anchor[..., None])[
                        ..., 0
                    ] + relatives[:, :3, 3]
                    residual = project_points(expected, K) - packet["pixels"][keep]
                    depth = packet["measured"][keep, 2]
                    available = np.isfinite(depth)
                    values = {
                        "reference_reprojection_px": np.linalg.norm(residual, axis=1),
                        "reference_depth_m": expected[available, 2] - depth[available],
                    }
                    for key, samples in values.items():
                        collected[label].setdefault(key, []).extend(samples)
            # Spatial holdout prevents reporting training-cost reduction as an
            # independent benefit. All three fits share exactly the same support.
            tile = np.floor(packet["pixels"] / 80).astype(int)
            train = (tile.sum(axis=1) % 2) == 0
            test = ~train
            if train.sum() < 20 or test.sum() < 10:
                continue
            fit = estimate_sparse_pose(
                packet["points"][train], packet["pixels"][train], K, SparseConfig()
            )
            if fit.T_C1C0 is None:
                continue
            poses = {"legacy": fit.T_C1C0}
            for name in ("robust_rgb", "rgb", "rgbd"):
                poses[name], _ = refine_measurements(
                    packet["points"][train],
                    packet["pixels"][train],
                    packet["measured"][train],
                    np.zeros_like(packet["covariance"][train])
                    if name == "robust_rgb"
                    else packet["covariance"][train],
                    fit.T_C1C0,
                    K,
                    name == "rgbd",
                )
            entry = {
                "target": target,
                "training": int(train.sum()),
                "test": int(test.sum()),
                "variants": {},
            }
            for name, pose in poses.items():
                p = transform_points(packet["points"][test], pose)
                rgb = np.sum(
                    (project_points(p, K) - packet["pixels"][test]) ** 2, axis=1
                )
                depth = p[:, 2] - packet["measured"][test, 2]
                depth = depth[np.isfinite(depth)]
                heldout[name]["rgb_squared"].extend(rgb)
                heldout[name]["depth_squared"].extend(depth * depth)
                entry["variants"][name] = {
                    "rgb_rmse_px": float(np.sqrt(np.mean(rgb))),
                    "depth_rmse_m": float(np.sqrt(np.mean(depth * depth)))
                    if len(depth)
                    else None,
                }
            trial_rows.append(entry)
    result["sample_dependence"] = {
        "packets": duplicate_rows,
        "support": sum(r["support"] for r in duplicate_rows),
        "unique_current_rgb_cells": sum(
            r["unique_current_rgb_cells"] for r in duplicate_rows
        ),
        "unique_anchor_native_samples": sum(
            r["unique_anchor_native_samples"] for r in duplicate_rows
        ),
    }
    result["reference_measurement_compatibility"] = {
        label: {key: stats(value) for key, value in values.items()}
        for label, values in collected.items()
    }
    result["spatial_holdout"] = {
        "packets": trial_rows,
        "pooled": {
            name: {
                key.replace("squared", "rmse"): float(np.sqrt(np.mean(value)))
                if value
                else None
                for key, value in values.items()
            }
            for name, values in heldout.items()
        },
    }
    # Cycle disagreement uses descriptors only and does not assume reference
    # trajectories or estimated map coordinates are correct.
    eligible, conflicts, chain_count = 0, 0, 0
    ray_displacement = []
    with np.load(path / "features.npz") as a:
        for t in range(7, len(rows), 7):
            if np.array_equal(T_RD, np.eye(4)):
                points = a[f"{t}_points_R"]
                points = points[np.isfinite(points).all(axis=1)]
                native = np.floor(project_points(points, K_D) + 0.5)
                reconstructed = np.column_stack(
                    (
                        (native - K_D[:2, 2]) / np.diag(K_D)[:2] * points[:, 2:3],
                        points[:, 2],
                    )
                )
                ray_displacement.extend(np.linalg.norm(reconstructed - points, axis=1))
            if any(f"{j}_descriptors" not in a for j in (t - 2, t - 1, t)):
                continue
            descriptors = [a[f"{j}_descriptors"] for j in (t - 2, t - 1, t)]
            first = dict(match_descriptors(descriptors[0], descriptors[1], 0.75)[0])
            second = dict(match_descriptors(descriptors[1], descriptors[2], 0.75)[0])
            direct = dict(match_descriptors(descriptors[0], descriptors[2], 0.75)[0])
            for source, middle in first.items():
                if middle in second:
                    chain_count += 1
                    if source in direct:
                        eligible += 1
                        conflicts += direct[source] != second[middle]
    result["rgb_ray_vs_native_sample_displacement_m"] = stats(ray_displacement)
    result["descriptor_cycles"] = {
        "sampled_targets": list(range(7, len(rows), 7)),
        "two_link_chains": chain_count,
        "also_direct": eligible,
        "conflicts": int(conflicts),
        "conflict_fraction": float(conflicts / eligible) if eligible else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh diagnostic output")
    results = {p.name: diagnose_case(p) for p in args.cases}
    args.output.write_text(
        json.dumps(
            {"driver_sha256": sha256(Path(__file__)), "cases": results},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
