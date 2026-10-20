"""Summarize external trajectory and coverage-aware batch reconstruction scores."""

import argparse
import json
from pathlib import Path

import numpy as np
from summarize_temporal import maps_summary, pose


def paired_maps(run, chunks, control):
    result = {}
    for method in ("voxel", "tsdf"):
        count, before, after, gained, lost = 0, 0.0, 0.0, 0, 0
        differences = []
        for chunk in chunks:
            with np.load(
                run / f"maps/targets_{chunk:03d}.npz", allow_pickle=False
            ) as a:
                observed = a["observed"]
                first, last = (
                    a[f"{method}_{control}_reference"],
                    a[f"{method}_batch_reference"],
                )
                valid = np.isfinite(observed) & (observed > 0)
                f, l = np.isfinite(first) & (first > 0), np.isfinite(last) & (last > 0)
                common = valid & f & l
                count += int(common.sum())
                before += float(np.abs(first[common] - observed[common]).sum())
                after += float(np.abs(last[common] - observed[common]).sum())
                gained += int((valid & ~f & l).sum())
                lost += int((valid & f & ~l).sum())
                for i in range(len(observed)):
                    if valid[i].any():
                        one = np.where(f[i], np.abs(first[i] - observed[i]), np.inf)
                        two = np.where(l[i], np.abs(last[i] - observed[i]), np.inf)
                        differences.append(
                            float(
                                np.minimum(two[valid[i]], 0.2).mean()
                                - np.minimum(one[valid[i]], 0.2).mean()
                            )
                        )
        result[method] = {
            "common_samples": count,
            "control_common_mae_m": before / count if count else None,
            "batch_common_mae_m": after / count if count else None,
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
        "config": report["config"],
        "sparse": pose(report["baseline_evaluation"]),
        "variants": {},
        "batch_behavior": report["variants"]["batch"]["segments"],
        "batch_solver_seconds": report["batch_solver_seconds"],
        "repeat_exact": report["repeat_exact"],
    }
    for name, variant in {
        "sparse": {"evaluation": report["baseline_evaluation"]},
        **report["variants"],
    }.items():
        evaluation = variant["evaluation"]
        segments = evaluation["segments"]
        count = sum(s["anchored_errors"]["rotation_deg"]["count"] for s in segments)
        rotation = (
            np.sqrt(
                sum(
                    s["anchored_errors"]["rotation_deg"]["count"]
                    * (s["anchored_errors"]["rotation_deg"]["rmse"] or 0) ** 2
                    for s in segments
                )
                / count
            )
            if count
            else None
        )
        result["variants"][name] = {
            **pose(evaluation),
            "anchored_rotation_rmse_deg": float(rotation)
            if rotation is not None
            else None,
        }
    if (run / "maps/report.json").exists():
        maps = json.loads((run / "maps/report.json").read_text(encoding="utf-8"))
        result["maps"] = maps_summary(maps)
        result["paired_maps"] = {
            control: paired_maps(
                run,
                maps["evaluated_chunk_ids"]
                if maps["evaluated_chunk_ids"] is not None
                else [c["submap_id"] for c in maps["chunks"]],
                control,
            )
            for control in ("sparse", "clock", "timed", "joint")
            if control == "sparse" or control in report["variants"]
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
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
