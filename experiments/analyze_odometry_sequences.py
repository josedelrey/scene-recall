"""Analyze saved fixed-configuration comparisons without rerunning estimators.

Keep each sequence's accuracy population separate. --plot requires optional
Matplotlib. References diagnose motion and error only, never modify estimates.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scene_recall.datasets.tum import TUMConfig, TUMReader
from scene_recall.odometry.evaluation import (
    error_statistics,
    pose_error,
    timestamp_pairs,
)
from scene_recall.odometry.pipeline import is_rigid

NAMES = ("sparse", "icp", "hybrid")
COLORS = ("#3167a8", "#d47d20", "#8460a8")


def matched_errors(archives, pairs, names=NAMES):
    """Return errors and indices on identical valid, uninterrupted pairs."""
    references = archives["sparse"]["T_WC_D_reference"]
    indices = []
    errors = {name: [] for name in names}
    for source, target in pairs:
        if not all(
            archives[name]["segment_id"][source] == archives[name]["segment_id"][target]
            for name in names
        ):
            continue
        if not is_rigid(references[source]) or not is_rigid(references[target]):
            continue
        reference = np.linalg.inv(references[target]) @ references[source]
        if not is_rigid(reference):
            continue
        indices.append([int(source), int(target)])
        for name in names:
            estimate = (
                np.linalg.inv(archives[name]["T_SC_D"][target])
                @ archives[name]["T_SC_D"][source]
            )
            errors[name].append(pose_error(estimate, reference))
    return indices, errors


def summarize_errors(errors):
    errors = list(errors)
    return {
        "translation_m": error_statistics(e[0] for e in errors),
        "rotation_deg": error_statistics(e[1] for e in errors),
    }


def summarize_pairs(rows):
    """Preserve failure denominators even when reference motion is unavailable."""
    result = {"attempted": len(rows)}
    for name in NAMES:
        selected = [row[name] for row in rows]
        result[name] = {
            "lost": sum(row["status"] == "lost" for row in selected),
            "accepted_errors": summarize_errors(
                row["error_m_deg"] for row in selected if row["error_m_deg"] is not None
            ),
            "lost_candidate_errors": summarize_errors(
                row["candidate_error_m_deg"]
                for row in selected
                if row["status"] == "lost" and row["candidate_error_m_deg"] is not None
            ),
            "support_fraction": error_statistics(
                row["diagnostics"]["icp_support_fraction"]
                for row in selected
                if row["diagnostics"].get("icp_support_fraction") is not None
            ),
            "condition": error_statistics(
                row["diagnostics"]["icp_condition"]
                for row in selected
                if row["diagnostics"].get("icp_condition") is not None
            ),
        }
    return result


def timestamp_diagnostic(report, archives):
    """Rescore accepted poses at RGB times as a diagnostic sensitivity check.

    Reuse the adapter's exact bounded interpolation policy. RGB-time references
    are exploratory because sparse fitting also uses depth-time source geometry.
    The primary benchmark remains evaluated at the canonical depth timestamps.
    """
    if report["source"].get("dataset", "scannet_v2") != "tum_rgbd":
        return None
    with TUMReader(
        report["source"]["path"],
        config=TUMConfig(**report["source"]["association_config"]),
    ) as reader:
        rgb_refs = [
            reader._reference(f["provenance"]["timestamp_color"])[0]
            for f in report["frames"]
        ]
    depth_refs = archives["sparse"]["T_WC_D_reference"]
    mismatch = []
    scores = {name: {"depth_time": [], "rgb_time": []} for name in NAMES}
    for target in range(1, len(rgb_refs)):
        source = target - 1
        if not all(
            p is not None and is_rigid(p)
            for p in (
                rgb_refs[source],
                rgb_refs[target],
                depth_refs[source],
                depth_refs[target],
            )
        ):
            continue
        rgb_motion = np.linalg.inv(rgb_refs[target]) @ rgb_refs[source]
        depth_motion = np.linalg.inv(depth_refs[target]) @ depth_refs[source]
        if not is_rigid(rgb_motion) or not is_rigid(depth_motion):
            continue
        mismatch.append(pose_error(rgb_motion, depth_motion))
        for name in NAMES:
            estimate = archives[name]["T_C1C0"][target]
            if is_rigid(estimate):
                scores[name]["depth_time"].append(pose_error(estimate, depth_motion))
                scores[name]["rgb_time"].append(pose_error(estimate, rgb_motion))
    endpoints = {}
    for name in NAMES:
        endpoints[name] = []
        for segment in np.unique(archives[name]["segment_id"]):
            indices = [
                int(i)
                for i in np.flatnonzero(archives[name]["segment_id"] == segment)
                if rgb_refs[i] is not None
                and is_rigid(rgb_refs[i])
                and is_rigid(depth_refs[i])
            ]
            if len(indices) < 2:
                continue
            first, last = indices[0], indices[-1]
            estimate = (
                np.linalg.inv(archives[name]["T_SC_D"][first])
                @ archives[name]["T_SC_D"][last]
            )
            endpoints[name].append(
                {
                    "segment_id": int(segment),
                    "first": first,
                    "last": last,
                    "depth_time_error_m_deg": pose_error(
                        depth_refs[first] @ estimate, depth_refs[last]
                    ),
                    "rgb_time_error_m_deg": pose_error(
                        rgb_refs[first] @ estimate, rgb_refs[last]
                    ),
                }
            )
    return {
        "interpretation": "sensitivity only, mixed RGB-time bearings and depth-time source geometry prevent treating RGB-time scores as corrected benchmark accuracy",
        "reference_motion_mismatch": summarize_errors(mismatch),
        "accepted_pose_errors_same_population": {
            name: {clock: summarize_errors(errors) for clock, errors in values.items()}
            for name, values in scores.items()
        },
        "segment_endpoint_sensitivity": endpoints,
    }


def analyze_range(comparison):
    reports = {
        name: json.loads((Path(run["directory"]) / "report.json").read_text())
        for name, run in comparison["runs"].items()
    }
    archives = {}
    for name, run in comparison["runs"].items():
        with np.load(Path(run["directory"]) / "trajectory.npz") as archive:
            archives[name] = {key: archive[key] for key in archive.files}
    base = reports["sparse"]
    stamps = archives["sparse"]["timestamp_s"]
    refs = archives["sparse"]["T_WC_D_reference"]
    clock = bool(np.isfinite(stamps).all())
    pairs = []
    for index in range(1, len(stamps)):
        previous, current = base["frames"][index - 1 : index + 1]
        valid = is_rigid(refs[index - 1]) and is_rigid(refs[index])
        reference = np.linalg.inv(refs[index]) @ refs[index - 1] if valid else None
        valid = valid and is_rigid(reference)
        motion = pose_error(reference, np.eye(4)) if valid else None
        skews = [
            f.get("provenance", {}).get("rgb_depth_difference_s")
            for f in (previous, current)
        ]
        row = {
            "target": current["source_frame_index"],
            "elapsed_s": float(stamps[index] - stamps[index - 1]) if clock else None,
            "reference_motion_m_deg": motion,
            "max_abs_rgb_depth_skew_s": max(abs(s) for s in skews)
            if all(s is not None for s in skews)
            else None,
            "rgb_depth_skew_change_s": skews[1] - skews[0]
            if all(s is not None for s in skews)
            else None,
        }
        for name in NAMES:
            frame = reports[name]["frames"][index]
            error = reports[name]["evaluation"]["pair_errors_m_deg"][index - 1]
            row[name] = {
                "status": frame["status"],
                "reason": frame["reason"],
                "error_m_deg": error["final"],
                "candidate_error_m_deg": error["candidate"],
                "diagnostics": frame["diagnostics"],
            }
        pairs.append(row)
    horizons = {}
    for interval in (1, 5, 10):
        candidates = [(i, i + interval) for i in range(len(stamps) - interval)]
        indices, errors = matched_errors(archives, candidates, ("sparse", "hybrid"))
        horizons[str(interval)] = {
            "pairs": indices,
            "backends": {name: summarize_errors(e) for name, e in errors.items()},
        }
    long_time = {}
    for seconds in (2.0, 5.0):
        candidates, status = timestamp_pairs(stamps, seconds, 0.02)
        indices, errors = matched_errors(archives, candidates)
        sh_indices, sh_errors = matched_errors(
            archives, candidates, ("sparse", "hybrid")
        )
        long_time[str(seconds)] = {
            "status": status,
            "candidate_pairs": len(candidates),
            "common_pairs": indices,
            "common": {name: summarize_errors(e) for name, e in errors.items()},
            "sparse_hybrid_pairs": sh_indices,
            "sparse_hybrid": {
                name: summarize_errors(e) for name, e in sh_errors.items()
            },
        }
    runs = {}
    for name, report in reports.items():
        evaluation = report["evaluation"]
        durations = (
            [
                float(stamps[s["end_frame"]] - stamps[s["start_frame"]])
                for s in evaluation["segments"]
            ]
            if clock
            else []
        )
        accepted = [p for p in pairs if p[name]["status"] == "tracked"]
        losses = [p for p in pairs if p[name]["status"] == "lost"]
        stages = {}
        for stage in ("sparse_seconds", "icp_seconds"):
            values = [
                f["diagnostics"][stage]
                for f in report["frames"]
                if stage in f["diagnostics"]
            ]
            stages[stage] = dict(error_statistics(values), total_s=float(sum(values)))
        runs[name] = {
            "config": report["config"],
            "evaluation": evaluation,
            "timing": report["timing"],
            "environment": report["environment"],
            "report_sha256": hashlib.sha256(
                (
                    Path(comparison["runs"][name]["directory"]) / "report.json"
                ).read_bytes()
            ).hexdigest(),
            "segment_duration_s": error_statistics(durations),
            "trajectory_sha256": hashlib.sha256(
                (
                    Path(comparison["runs"][name]["directory"]) / "trajectory.npz"
                ).read_bytes()
            ).hexdigest(),
            "segment_frames": error_statistics(
                s["frame_count"] for s in evaluation["segments"]
            ),
            "stages": stages,
            "lost_targets": [p["target"] for p in losses],
            "accepted_diagnostics": summarize_pairs(accepted)[name],
            "lost_diagnostics": summarize_pairs(losses)[name],
            "largest_translation_errors": sorted(
                [p for p in pairs if p[name]["error_m_deg"] is not None],
                key=lambda p: p[name]["error_m_deg"][0],
                reverse=True,
            )[:5],
            "largest_rotation_errors": sorted(
                [p for p in pairs if p[name]["error_m_deg"] is not None],
                key=lambda p: p[name]["error_m_deg"][1],
                reverse=True,
            )[:5],
        }
    angular_bins = {}
    for low, high in ((0, 1), (1, 2), (2, None)):
        selected = [
            p
            for p in pairs
            if p["reference_motion_m_deg"] is not None
            and low <= p["reference_motion_m_deg"][1]
            and (high is None or p["reference_motion_m_deg"][1] < high)
        ]
        angular_bins[f"{low}:{high}"] = summarize_pairs(selected)
    motion = [p for p in pairs if p["reference_motion_m_deg"] is not None]
    quality = {
        "duration_s": float(stamps[-1] - stamps[0]) if clock else None,
        "depth_interval_s": error_statistics(p["elapsed_s"] for p in pairs if clock),
        "abs_rgb_depth_skew_s": error_statistics(
            abs(f["provenance"]["rgb_depth_difference_s"])
            for f in base["frames"]
            if "rgb_depth_difference_s" in f.get("provenance", {})
        ),
        "missing_reference_targets": [
            f["source_frame_index"]
            for f in base["frames"]
            if not is_rigid(refs[f["frame_index"]])
        ],
        "motion_pairs": len(motion),
        "sampled_path_length_m": float(
            sum(p["reference_motion_m_deg"][0] for p in motion)
        ),
        "sampled_angular_travel_deg": float(
            sum(p["reference_motion_m_deg"][1] for p in motion)
        ),
        "reference_translation_m": error_statistics(
            p["reference_motion_m_deg"][0] for p in motion
        ),
        "reference_rotation_deg": error_statistics(
            p["reference_motion_m_deg"][1] for p in motion
        ),
        "linear_speed_m_s": error_statistics(
            p["reference_motion_m_deg"][0] / p["elapsed_s"] for p in motion if clock
        ),
        "angular_speed_deg_s": error_statistics(
            p["reference_motion_m_deg"][1] / p["elapsed_s"] for p in motion if clock
        ),
    }
    return {
        "source": {
            **base["source"],
            "dataset": base["source"].get("dataset", "scannet_v2"),
        },
        "calibration": base["calibration"],
        "quality": quality,
        "timestamp_sensitivity": timestamp_diagnostic(base, archives),
        "common_rpe": comparison["common_rpe"],
        "common_time_rpe": comparison.get("common_time_rpe", {}),
        "sparse_hybrid_rpe": horizons,
        "long_time_rpe": long_time,
        "runs": runs,
        "angular_motion_bins_deg": angular_bins,
        "gaps_over_50ms": summarize_pairs(
            [p for p in pairs if clock and p["elapsed_s"] > 0.05]
        ),
        "regular_intervals": summarize_pairs(
            [p for p in pairs if clock and p["elapsed_s"] <= 0.05]
        ),
        "pair_diagnostics": pairs,
        "hybrid_refinement_outcomes": comparison["hybrid_refinement_outcomes"],
        "hybrid_sparse_initialization_max_abs_difference": comparison[
            "hybrid_sparse_initialization_max_abs_difference"
        ],
    }, archives


def plot_sequences(ranges, archives, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    groups = {}
    for row in ranges:
        groups.setdefault(row["source"]["sequence_id"], []).append(row)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    labels = [
        key.replace("rgbd_dataset_freiburg1_", "TUM ")
        if group[0]["source"]["dataset"] == "tum_rgbd"
        else f"ScanNet\n{key}\n{len(group)} clips"
        for key, group in groups.items()
    ]
    x = np.arange(len(groups))
    for panel, ax in enumerate(axes.flat):
        for offset, (name, color) in enumerate(zip(NAMES, COLORS)):
            values = []
            for group in groups.values():
                if panel < 4:
                    interval = "1" if panel < 2 else "10"
                    metric = "translation_m" if panel % 2 == 0 else "rotation_deg"
                    populations = [
                        row["common_rpe"][interval][name][metric] for row in group
                    ]
                    count = sum(p["count"] for p in populations)
                    value = (
                        np.sqrt(
                            sum(
                                p["rmse"] ** 2 * p["count"]
                                for p in populations
                                if p["count"]
                            )
                            / count
                        )
                        if count
                        else np.nan
                    )
                    value *= 1000 if panel % 2 == 0 else 1
                elif panel == 4:
                    value = (
                        100
                        * sum(
                            row["runs"][name]["evaluation"]["failed_transitions"]
                            for row in group
                        )
                        / sum(
                            row["runs"][name]["evaluation"]["transition_count"]
                            for row in group
                        )
                    )
                else:
                    value = sum(
                        row["runs"][name]["evaluation"]["frame_count"] for row in group
                    ) / sum(
                        row["runs"][name]["timing"]["tracking_seconds"] for row in group
                    )
                values.append(value)
            ax.bar(x + (offset - 1) * 0.25, values, 0.23, label=name, color=color)
        ax.set(xticks=x, xticklabels=labels)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    for ax, title, unit in zip(
        axes.flat,
        (
            "One-frame translation",
            "One-frame rotation",
            "Ten-frame translation",
            "Ten-frame rotation",
            "Detected tracking losses",
            "End-to-end throughput",
        ),
        (
            "Common RPE RMSE (mm)",
            "Common RPE RMSE (deg)",
            "Common RPE RMSE (mm)",
            "Common RPE RMSE (deg)",
            "Failed transitions (%)",
            "Frames per second",
        ),
    ):
        ax.set(title=title, ylabel=unit)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Fixed default backends across motion and scenes\nCommon continuous RPE within each sequence · ScanNet clips aggregated by pair count · historical timings"
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"cross_sequence_overview.{suffix}", dpi=160)
    plt.close(fig)
    tum = [
        (r, a) for r, a in zip(ranges, archives) if r["source"]["dataset"] == "tum_rgbd"
    ]
    if not tum:
        return
    fig, axes = plt.subplots(
        3, len(tum), figsize=(5 * len(tum), 10), squeeze=False, constrained_layout=True
    )
    for column, (row, data) in enumerate(tum):
        stamps = data["sparse"]["timestamp_s"]
        time = stamps - stamps[0]
        reference = data["sparse"]["T_WC_D_reference"]
        valid = np.array([is_rigid(r) for r in reference])
        axes[0, column].plot(
            reference[valid, 0, 3],
            reference[valid, 1, 3],
            color="black",
            label="reference",
            lw=1.5,
        )
        for name, color in zip(NAMES, COLORS):
            errors = np.full((len(time), 2), np.nan)
            first = True
            for segment in row["runs"][name]["evaluation"]["segments"]:
                start, end = segment["start_frame"], segment["end_frame"]
                indices = np.arange(start, end + 1)[valid[start : end + 1]]
                if not len(indices):
                    continue
                anchor = indices[0]
                alignment = reference[anchor] @ np.linalg.inv(
                    data[name]["T_SC_D"][anchor]
                )
                poses = alignment @ data[name]["T_SC_D"][indices]
                errors[indices] = [
                    pose_error(pose, reference[i]) for pose, i in zip(poses, indices)
                ]
                axes[0, column].plot(
                    poses[:, 0, 3],
                    poses[:, 1, 3],
                    color=color,
                    label=name if first else None,
                    alpha=0.8,
                    lw=0.8,
                )
                first = False
                # Separate segments explicitly to avoid drawing lines over losses.
                axes[1, column].plot(
                    time[indices], errors[indices, 0] * 1000, color=color, lw=0.9
                )
                axes[2, column].plot(
                    time[indices], errors[indices, 1], color=color, lw=0.9
                )
            lost = np.flatnonzero(data[name]["status"] == "lost")
            axes[2, column].scatter(
                time[lost], np.full(len(lost), -0.5), marker="|", color=color, s=35
            )
        axes[0, column].set(
            title=row["source"]["sequence_id"].replace("rgbd_dataset_", ""),
            xlabel="World x (m)",
            ylabel="World y (m)",
            aspect="equal",
        )
        axes[1, column].set(
            ylabel="Segment-anchored position error (mm)", xlabel="Elapsed time (s)"
        )
        axes[2, column].set(
            ylabel="Segment-anchored rotation error (deg)", xlabel="Elapsed time (s)"
        )
        axes[0, column].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Trajectories and accumulated drift\nEach segment independently anchored to reference\nFragmented traces remain separate · bottom marks show losses",
        fontsize=11,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"trajectories_and_drift.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(
        3, len(tum), figsize=(5 * len(tum), 10), squeeze=False, constrained_layout=True
    )
    for column, (row, data) in enumerate(tum):
        time = data["sparse"]["timestamp_s"] - data["sparse"]["timestamp_s"][0]
        pairs = row["pair_diagnostics"]
        angular = [
            np.nan
            if p["reference_motion_m_deg"] is None
            else p["reference_motion_m_deg"][1]
            for p in pairs
        ]
        axes[0, column].plot(
            time[1:], angular, color="black", lw=0.7, label="reference rotation"
        )
        for name, color in zip(NAMES, COLORS):
            losses = [i for i, p in enumerate(pairs) if p[name]["status"] == "lost"]
            axes[0, column].scatter(
                time[1:][losses],
                np.array(angular)[losses],
                color=color,
                marker="x",
                label=f"{name} losses",
                s=18,
            )
            errors = [
                np.nan if p[name]["error_m_deg"] is None else p[name]["error_m_deg"][1]
                for p in pairs
            ]
            axes[1, column].plot(time[1:], errors, color=color, lw=0.7, label=name)
        outcomes = row["hybrid_refinement_outcomes"]
        axes[2, column].scatter(
            [
                (p["depth_rmse_final_m"] - p["depth_rmse_initial_m"]) * 1000
                for p in outcomes
                if p["depth_rmse_final_m"] is not None
                and p["depth_rmse_initial_m"] is not None
            ],
            [
                p["final_error_m_deg"][1] - p["initial_error_m_deg"][1]
                for p in outcomes
                if p["depth_rmse_final_m"] is not None
                and p["depth_rmse_initial_m"] is not None
            ],
            s=9,
            alpha=0.4,
            color=COLORS[2],
            edgecolors="none",
        )
        axes[2, column].axhline(0, color="gray", lw=0.7)
        axes[2, column].axvline(0, color="gray", lw=0.7)
        axes[0, column].set(
            title=row["source"]["sequence_id"].replace("rgbd_dataset_", ""),
            ylabel="Reference rotation per pair (deg)",
            xlabel="Elapsed time (s)",
        )
        axes[1, column].set(
            ylabel="Accepted one-frame rotation error (deg)", xlabel="Elapsed time (s)"
        )
        axes[2, column].set(
            ylabel="Hybrid change in rotation error (deg)",
            xlabel="Change in held-out depth RMSE (mm)",
        )
        axes[0, column].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Motion, detected losses, and accepted errors\nLower depth residual does not guarantee lower rotation error",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"motion_and_refinement.{suffix}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranges, archives, inputs = [], [], []
    for directory in args.comparison_dirs:
        path = directory / "comparison.json"
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        for comparison in json.loads(path.read_text())["ranges"]:
            # Validate the saved comparison against current report hashes.
            for run in comparison["runs"].values():
                report_path = Path(run["directory"]) / "report.json"
                if (
                    hashlib.sha256(report_path.read_bytes()).hexdigest()
                    != run["report_sha256"]
                ):
                    raise ValueError(f"changed report: {report_path}")
            row, archive = analyze_range(comparison)
            ranges.append(row)
            archives.append(archive)
    for name in NAMES:
        if any(
            row["runs"][name]["config"] != ranges[0]["runs"][name]["config"]
            for row in ranges
        ):
            raise ValueError(f"backend configurations differ across sequences: {name}")
    result = {
        "method": "post-hoc saved trajectories, per-sequence populations, no gap bridging or scale fitting",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": inputs,
        "ranges": ranges,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        columns = [
            "sequence",
            "start",
            "stop",
            "backend",
            "frames",
            "coverage",
            "segments",
            "largest_segment_frames",
            "longest_segment_s",
            "common_rpe1_mm",
            "common_rpe1_deg",
            "common_rpe10_mm",
            "common_rpe10_deg",
            "ate_mm",
            "endpoint_mm",
            "endpoint_deg",
            "fps",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in ranges:
            for name in NAMES:
                run = row["runs"][name]
                evaluation = run["evaluation"]
                continuous = evaluation["continuous_trajectory"]
                values = {
                    "sequence": row["source"]["sequence_id"],
                    "start": row["source"]["start"],
                    "stop": row["source"]["stop"],
                    "backend": name,
                    "frames": evaluation["frame_count"],
                    "coverage": evaluation["tracking_coverage"],
                    "segments": evaluation["segment_count"],
                    "largest_segment_frames": evaluation["largest_segment_frames"],
                    "longest_segment_s": run["segment_duration_s"]["max"],
                    "ate_mm": None
                    if continuous is None
                    or continuous["aligned_ate_translation_m"]["rmse"] is None
                    else continuous["aligned_ate_translation_m"]["rmse"] * 1000,
                    "endpoint_mm": None
                    if continuous is None
                    or continuous["endpoint_translation_error_m"] is None
                    else continuous["endpoint_translation_error_m"] * 1000,
                    "endpoint_deg": None
                    if continuous is None
                    else continuous["endpoint_rotation_error_deg"],
                    "fps": run["timing"]["frames_per_second"],
                }
                for interval in (1, 10):
                    metric = row["common_rpe"][str(interval)][name]
                    values[f"common_rpe{interval}_mm"] = (
                        None
                        if metric["translation_m"]["rmse"] is None
                        else metric["translation_m"]["rmse"] * 1000
                    )
                    values[f"common_rpe{interval}_deg"] = metric["rotation_deg"]["rmse"]
                writer.writerow(values)
    if args.plot:
        plot_sequences(ranges, archives, args.output_dir)
    print(f"Analyzed {len(ranges)} ranges into {args.output_dir}")


if __name__ == "__main__":
    main()
