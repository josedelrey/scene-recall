"""Summarize fixed pose/map populations without selecting a winning trajectory."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def pose_summary(evaluation):
    segments = evaluation["segments"]
    count = sum(s["aligned_ate_translation_m"]["count"] for s in segments)
    aligned = sum(
        s["aligned_ate_translation_m"]["count"]
        * (s["aligned_ate_translation_m"]["rmse"] or 0) ** 2
        for s in segments
    )
    return {
        "pooled_segment_aligned_ate_rmse_m": float(np.sqrt(aligned / count))
        if count
        else None,
        "segments": [
            {
                k: s[k]
                for k in (
                    "segment_id",
                    "frame_count",
                    "reference_count",
                    "endpoint_translation_error_m",
                    "endpoint_rotation_error_deg",
                    "anchored_errors",
                )
            }
            for s in segments
        ],
        "rpe_frames": {
            gap: {
                "count": v["translation_m"]["count"],
                "translation_rmse_m": v["translation_m"]["rmse"],
                "rotation_rmse_deg": v["rotation_deg"]["rmse"],
            }
            for gap, v in evaluation["rpe_by_frame_interval"].items()
        },
        "rpe_seconds": {
            gap: {
                "count": v["translation_m"]["count"],
                "translation_rmse_m": v["translation_m"]["rmse"],
                "rotation_rmse_deg": v["rotation_deg"]["rmse"],
            }
            for gap, v in evaluation["rpe_by_time_interval_s"].items()
        },
    }


def summarize(run):
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    result = {
        "source": report["source"],
        "training_frames": len(report["split"]["train_indices"]),
        "baseline": pose_summary(report["baseline_evaluation"]),
        "preparation_seconds": report["preparation_seconds"],
        "variants": {},
        "repeat_exact": report["repeat_exact"],
        "code_sha256": report["code_sha256"],
    }
    for name, variant in report["variants"].items():
        windows = variant["windows"]
        fitted = [w for w in windows if "final_cost" in w]
        initial_cost = sum(w["initial_cost"] for w in fitted)
        final_cost = sum(w["final_cost"] for w in fitted)
        result["variants"][name] = {
            "pose": pose_summary(variant["evaluation"]),
            "windows": dict(Counter(w["status"] for w in windows)),
            "skip_reasons": dict(
                Counter(
                    w.get("reason")
                    for w in windows
                    if w["status"] in ("skipped", "rejected")
                )
            ),
            "initial_window_cost_sum": initial_cost,
            "final_window_cost_sum": final_cost,
            "window_cost_reduction_fraction": 1 - final_cost / initial_cost
            if initial_cost
            else None,
            "optimization_seconds": variant["optimization_seconds"],
            "condition_max": max((w["condition"] for w in fitted), default=None),
        }
    map_path = run / "maps/report.json"
    if map_path.exists():
        maps = json.loads(map_path.read_text(encoding="utf-8"))
        scores, pairs = maps["pooled"]["scores"], maps["pooled"]["pairs"]
        result["maps"] = {
            "chunks": maps["evaluated_chunk_ids"],
            "evaluated_views": sum(v["status"] == "evaluated" for v in maps["views"]),
            "seconds": maps["total_seconds"],
            "scores": {},
            "pairs": {},
        }
        for key, value in scores.items():
            metric = value["own"]
            result["maps"]["scores"][key] = {
                "target_samples": metric["target_samples"],
                "coverage": metric["coverage"],
                "mae_m": metric["absolute_depth_error_m"]["mean"],
                "rmse_m": metric["absolute_depth_error_m"]["rmse"],
                "explained_5cm": metric["thresholds"]["0.05"][
                    "explained_fraction_of_target"
                ],
                "capped_missing_20cm_m": metric["missing_penalized_capped_mae_m"][
                    "0.2"
                ],
            }
        for key, value in pairs.items():
            result["maps"]["pairs"][key] = {
                name: metric["absolute_depth_error_m"]["mean"]
                for name, metric in value.items()
            }
            result["maps"]["pairs"][key]["samples"] = value["sparse"]["target_samples"]
        per_view = {}
        for key in pairs:
            method, _, mode = key.split("_")
            delta = [
                v["scores"][key]["own"]["missing_penalized_capped_mae_m"]["0.2"]
                - v["scores"][f"{method}_sparse_{mode}"]["own"][
                    "missing_penalized_capped_mae_m"
                ]["0.2"]
                for v in maps["views"]
                if v["status"] == "evaluated"
            ]
            per_view[key] = {
                "improved": sum(d < 0 for d in delta),
                "worsened": sum(d > 0 for d in delta),
                "median_delta_m": float(np.median(delta)) if delta else None,
            }
        result["maps"]["paired_views"] = per_view
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len({p.name for p in args.runs}) != len(args.runs):
        parser.error("run directory names must be unique")
    result = {p.name: summarize(p) for p in args.runs}
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
