"""Post-fit window endpoint and iteration-budget diagnostics without retuning."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from scene_recall.multiview.fixed_landmarks import DirectFactors, solve_fixed
from scene_recall.multiview.joint_ba import solve_joint
from scene_recall.multiview.joint_runner import calibration_from_report
from scene_recall.multiview.pose_only import PoseOnlyConfig
from scene_recall.multiview.temporal import TimingConfig
from scene_recall.multiview.temporal_runner import motion_table
from scene_recall.odometry.evaluation import pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def diagnose(run):
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    identity = report["variants"]["joint"]
    if sha256(run / "joint_factors.npz") != identity["factor_archive_sha256"]:
        raise ValueError("joint archive changed")
    child, _ = load_trajectory(run / "joint")
    config = PoseOnlyConfig(
        **{
            k: v
            for k, v in child["config"].items()
            if k in PoseOnlyConfig.__dataclass_fields__
        }
    )
    calibration = calibration_from_report(report)
    baseline, rows = load_trajectory(Path(report["baseline_run"]))
    motions, _ = motion_table(
        baseline,
        rows,
        report["split"]["train_indices"],
        calibration,
        TimingConfig(**report["timing_config"]),
    )
    results = []
    with np.load(run / "joint_factors.npz", allow_pickle=False) as a:
        for index, window in enumerate(identity["windows"]):
            if "paired_joint" not in window:
                continue
            key = f"window_{index:03d}"
            factors = DirectFactors(
                **{
                    field: a[f"{key}_{field}"]
                    for field in DirectFactors.__dataclass_fields__
                }
            )
            ids = window["frame_indices"]
            free = [ids.index(i) for i in window["free_frame_indices"]]
            initial, fitted = a[f"{key}_initial"], a[f"{key}_fitted_poses"]
            fixed, _ = solve_fixed(initial, factors, calibration.K_R, free, config)
            longer, _, diagnostic = solve_joint(
                initial,
                factors,
                calibration.K_R,
                free,
                replace(config, max_iterations=2 * config.max_iterations),
                max_point_displacement_m=child["config"]["max_point_displacement_m"],
            )
            comparison = {}
            first, last = rows[ids[0]].reference_pose, rows[ids[-1]].reference_pose
            if first is not None and last is not None:
                reference = np.linalg.inv(first) @ last
                for name, poses in (
                    ("initial", initial),
                    ("fixed", fixed),
                    ("joint", fitted),
                    ("joint_double_budget", longer),
                ):
                    depth = [
                        poses[j] @ motions[i].T_RcRd @ calibration.T_RD
                        for j, i in enumerate(ids)
                    ]
                    error = pose_error(np.linalg.inv(depth[0]) @ depth[-1], reference)
                    comparison[name] = {
                        "translation_m": error[0],
                        "rotation_deg": error[1],
                    }
            changes = [pose_error(p, q) for p, q in zip(longer, fitted, strict=True)]
            results.append(
                {
                    "frame_indices": ids,
                    "local_endpoint_errors": comparison,
                    "primary_cost": window.get("final_cost"),
                    "double_budget_solver": diagnostic,
                    "max_double_budget_translation_change_m": max(
                        e[0] for e in changes
                    ),
                    "max_double_budget_rotation_change_deg": max(e[1] for e in changes),
                }
            )
    return {
        "run": str(run.resolve()),
        "windows": results,
        "interpretation": "isolated fits from the same sequential joint starts, endpoint references used after fitting only, doubled budget is diagnostic",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("diagnostic output already exists")
    result = {
        "code_sha256": code_hash(Path(__file__)),
        "cases": [diagnose(run) for run in args.runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
