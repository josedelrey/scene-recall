"""Summarize predeclared SLAM populations, emitted accuracy and reconstruction."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from incremental_slam import with_revised_poses
from summarize_local_map_reliability import geometry_scores

from scene_recall.geometry.transforms import is_rigid
from scene_recall.odometry.evaluation import error_statistics, evaluate_trajectory
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.pipeline import PairEstimate
from scene_recall.run_metadata import sha256


def motion(report, *, emitted=False):
    evaluation = report["emitted_evaluation" if emitted else "evaluation"]
    segment = max(evaluation["segments"], key=lambda s: s["frame_count"])
    return {
        "reference_frames": evaluation["reference_valid_frames"],
        "segments": evaluation["segment_count"],
        "ate_scope": "full sequence"
        if evaluation["segment_count"] == 1
        else "largest segment",
        "ate_m": segment["aligned_ate_translation_m"]["rmse"],
        "root_orientation_rmse_deg": segment["anchored_errors"]["rotation_deg"]["rmse"],
        "endpoint_translation_m": segment["endpoint_translation_error_m"],
        "endpoint_rotation_deg": segment["endpoint_rotation_error_deg"],
        "rpe": evaluation["rpe_by_frame_interval"],
        "time_rpe": evaluation["rpe_by_time_interval_s"],
        "failures": evaluation["failed_transitions"],
    }


def surface(path):
    collected = []
    for p in sorted(path.glob("views_*_radius_0.01.npz")):
        with np.load(p, allow_pickle=False) as a:
            if len(a["evaluated_frame_indices"]):
                collected.append(
                    {
                        k: a[k].copy()
                        for k in (
                            "observed",
                            "sparse_reference_pose",
                            "incremental_slam_reference_pose",
                            "gt",
                            "evaluated_frame_indices",
                        )
                    }
                )
    return (
        {k: np.concatenate([c[k] for c in collected]) for k in collected[0]}
        if collected
        else None
    )


def maps(path, policies, kind):
    gathered = {}
    withheld = {}
    for name in policies:
        if kind == "surface":
            packet = surface(path / ("maps_" + name))
        else:
            directory = path / ("tsdf_" + name)
            if not (directory / "report.json").exists():
                packet = None
            else:
                report = json.loads((directory / "report.json").read_text())
                if report["status"] != "evaluated":
                    packet = None
                    withheld[name] = report.get("reason")
                else:
                    _, rows = load_trajectory(path / name)
                    evaluated = [
                        index
                        for index in report["target_indices"]
                        if rows[index].reference_pose is not None
                        and is_rigid(rows[index].reference_pose)
                    ]
                    with np.load(
                        directory / "geometry_and_predictions.npz", allow_pickle=False
                    ) as a:
                        packet = {
                            "observed": a["observed"].copy(),
                            "sparse_reference_pose": a["budget_reference_pose"].copy(),
                            "incremental_slam_reference_pose": a[
                                name + "_reference_pose"
                            ].copy(),
                            "gt": a["gt"].copy(),
                            "evaluated_frame_indices": np.array(evaluated),
                        }
                    if len(packet["observed"]) != len(evaluated):
                        raise ValueError(
                            "TSDF evaluated target count differs from its reference cohort"
                        )
        if packet is None:
            withheld.setdefault(name, "no matched reference-target population")
        else:
            gathered[name] = packet
    if not gathered:
        return {"status": "withheld", "reasons": withheld}
    first = next(iter(gathered.values()))
    for packet in gathered.values():
        for key in (
            "observed",
            "sparse_reference_pose",
            "gt",
            "evaluated_frame_indices",
        ):
            if not np.array_equal(first[key], packet[key], equal_nan=True):
                raise ValueError("shared reconstruction cohort or controls differ")
    predictions = {
        name: packet["incremental_slam_reference_pose"]
        for name, packet in gathered.items()
    }
    predictions.update(budget=first["sparse_reference_pose"], gt=first["gt"])
    return {
        "status": "evaluated",
        "evaluated_views": len(first["observed"]),
        "target_indices": first["evaluated_frame_indices"].tolist(),
        "scores": geometry_scores(first["observed"], predictions),
        "withheld": withheld,
    }


def summarize(root, replacements_root=None):
    protocol = json.loads((root / "protocol.json").read_text())
    results = {}
    for name, baseline, _ in protocol["cases"]:
        path = root / name
        if (
            replacements_root is not None
            and (replacements_root / name / "comparison.json").exists()
        ):
            path = replacements_root / name
        baseline_report, _ = load_trajectory(Path(baseline))
        comparison = json.loads((path / "comparison.json").read_text())
        result = {
            "case_root": str(path),
            "baseline": motion(baseline_report),
            "variants": {},
            "comparison_sha256": sha256(path / "comparison.json"),
        }
        for policy in protocol["variants"]:
            report, rows = load_trajectory(path / policy)
            if (
                sha256(path / policy / "report.json")
                != comparison["variants"][policy]["report_sha256"]
                or sha256(path / policy / "trajectory.npz")
                != comparison["variants"][policy]["trajectory_sha256"]
            ):
                raise ValueError("saved candidate report or trajectory changed")
            if (
                report["source"] != baseline_report["source"]
                or report["calibration"] != baseline_report["calibration"]
            ):
                raise ValueError("candidate and baseline source or calibration differs")
            # Older raw reports inherited final pair diagnostics in their emitted
            # evaluation. Recompute from the authoritative emitted pose archive.
            with np.load(
                path / policy / "emitted_trajectory.npz", allow_pickle=False
            ) as emitted:
                paired_rows = [
                    replace(row, pair=PairEstimate(np.eye(4)))
                    if row.status == "tracked"
                    else row
                    for row in rows
                ]
                report["emitted_evaluation"] = evaluate_trajectory(
                    with_revised_poses(paired_rows, emitted["T_SC_D"]),
                    (1, 5, 10, 20, 50, 100),
                )
            diagnostics = [r["diagnostics"] for r in report["frames"]]
            result["variants"][policy] = {
                "final": motion(report),
                "emitted": motion(report, emitted=True),
                "loss_indices": [r.frame_index for r in rows if r.status == "lost"],
                "consistency": report["measurement_consistency"],
                "estimator_seconds": error_statistics(
                    d["total_seconds"] for d in diagnostics
                ),
                "keyframe_seconds": error_statistics(
                    d["total_seconds"]
                    for d in diagnostics
                    if "measurement_seconds" in d
                ),
                "nonkeyframe_seconds": error_statistics(
                    d["total_seconds"]
                    for d in diagnostics
                    if "measurement_seconds" not in d
                ),
                "solver_seconds": error_statistics(
                    d["solver"]["seconds"]
                    for d in diagnostics
                    if "solver" in d and "seconds" in d["solver"]
                ),
                **{
                    key: comparison["variants"][policy][key]
                    for key in (
                        "solver_statuses",
                        "solver_reasons",
                        "exposure_statuses",
                        "graph_restarts",
                        "marginalizations",
                        "max_active_cameras",
                        "max_active_points",
                        "max_active_observations",
                        "revisions",
                        "common_with_baseline",
                    )
                },
            }
        for kind in ("surface", "tsdf"):
            result[kind] = maps(path, protocol["variants"], kind)
        results[name] = result
    return {
        "protocol_sha256": sha256(root / "protocol.json"),
        "replacement_protocol_sha256": sha256(replacements_root / "protocol.json")
        if replacements_root is not None
        else None,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--replacements-root", type=Path)
    args = parser.parse_args()
    report = summarize(args.root, args.replacements_root)
    (args.root / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for name, case in report["cases"].items():
        print(name, "Sparse", round(case["baseline"]["ate_m"] * 1000, 3))
        for policy, result in case["variants"].items():
            print(
                " ",
                policy,
                round(result["final"]["ate_m"] * 1000, 3),
                "losses",
                result["final"]["failures"],
                "restarts",
                len(result["graph_restarts"]),
            )


if __name__ == "__main__":
    main()
