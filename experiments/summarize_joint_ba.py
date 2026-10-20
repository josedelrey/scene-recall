"""Summarize controlled joint BA poses, paired objectives and held-out map changes."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from summarize_temporal import maps_summary, pose


def paired_maps(run, chunks):
    result = {}
    for method in ("voxel", "tsdf"):
        count, fixed_error, joint_error, gained, lost = 0, 0.0, 0.0, 0, 0
        differences = []
        for chunk in chunks:
            with np.load(
                run / f"maps/targets_{chunk:03d}.npz", allow_pickle=False
            ) as a:
                observed = a["observed"]
                fixed, joint = (
                    a[f"{method}_timed_reference"],
                    a[f"{method}_joint_reference"],
                )
                valid = np.isfinite(observed) & (observed > 0)
                f, j = (
                    np.isfinite(fixed) & (fixed > 0),
                    np.isfinite(joint) & (joint > 0),
                )
                common = valid & f & j
                count += int(common.sum())
                fixed_error += float(np.abs(fixed[common] - observed[common]).sum())
                joint_error += float(np.abs(joint[common] - observed[common]).sum())
                gained += int((valid & ~f & j).sum())
                lost += int((valid & f & ~j).sum())
                for i in range(len(observed)):
                    mask = valid[i]
                    if mask.any():
                        fe = np.where(f[i], np.abs(fixed[i] - observed[i]), np.inf)
                        je = np.where(j[i], np.abs(joint[i] - observed[i]), np.inf)
                        differences.append(
                            float(
                                np.minimum(je[mask], 0.2).mean()
                                - np.minimum(fe[mask], 0.2).mean()
                            )
                        )
        result[method] = {
            "common_samples": count,
            "fixed_common_mae_m": fixed_error / count if count else None,
            "joint_common_mae_m": joint_error / count if count else None,
            "gained_samples": gained,
            "lost_samples": lost,
            "views_better": sum(d < -1e-12 for d in differences),
            "views_worse": sum(d > 1e-12 for d in differences),
            "views_unchanged": sum(abs(d) <= 1e-12 for d in differences),
            "per_view_cap_difference_m": differences,
        }
    return result


def summarize(run):
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    result = {
        "source": report["source"],
        "split": report["split"],
        "acquisition_policy": report["acquisition_policy"],
        "sparse": pose(report["baseline_evaluation"]),
        "variants": {},
        "preparation_seconds": report["cost_seconds"],
        "repeat_exact": report["repeat_exact"],
    }
    for name, variant in report["variants"].items():
        windows = variant["windows"]
        result["variants"][name] = {
            "pose": pose(variant["evaluation"]),
            "statuses": dict(Counter(w["status"] for w in windows)),
            "solver_seconds": variant.get("linear_solver_seconds"),
            "closed_windows": sum("cost_at_close" in w for w in windows),
        }
    windows = [w for w in report["variants"]["joint"]["windows"] if "paired_joint" in w]
    paired = [w for w in windows if "final_cost" in w["paired_joint"]]
    result["joint_behavior"] = {
        "supported_windows": len(windows),
        "common_initial_cost": sum(w["paired_fixed"]["initial_cost"] for w in paired),
        "paired_fixed_cost": sum(w["paired_fixed"]["final_cost"] for w in paired),
        "paired_joint_cost": sum(w["paired_joint"]["final_cost"] for w in paired),
        "maximum_point_shift_m": max(
            (w.get("landmark_displacement_max_m", 0) for w in windows), default=0
        ),
        "maximum_schur_condition": max(
            (w.get("condition", 0) or 0 for w in windows), default=0
        ),
        "rejected_trials": dict(
            Counter(t for w in windows for t in w.get("rejected_trials", []))
        ),
        "termination": dict(Counter(w.get("reason") for w in windows)),
        "closed_costs_unchanged": all(
            np.isclose(
                w["cost_at_close"], w["cost_after_all_windows"], rtol=1e-10, atol=1e-9
            )
            for w in windows
        ),
        "paired_fixed_seconds": report["variants"]["joint"]["paired_fixed_seconds"],
        "paired_joint_seconds": report["variants"]["joint"]["paired_joint_seconds"],
    }
    maps_path = run / "maps/report.json"
    if maps_path.exists():
        maps = json.loads(maps_path.read_text(encoding="utf-8"))
        result["maps"] = maps_summary(maps)
        result["fixed_joint_common_maps"] = paired_maps(
            run, maps["evaluated_chunk_ids"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("summary output already exists")
    report = {str(run): summarize(run) for run in args.runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
