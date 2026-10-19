"""Summarize fixed-population temporal controls and compatible legacy results."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def pose(evaluation):
    segments = evaluation["segments"]
    count = sum(s["aligned_ate_translation_m"]["count"] for s in segments)
    squared = sum(
        s["aligned_ate_translation_m"]["count"]
        * (s["aligned_ate_translation_m"]["rmse"] or 0) ** 2
        for s in segments
    )
    return {
        "ate_rmse_m": float(np.sqrt(squared / count)) if count else None,
        "endpoints": [
            {
                k: s[k]
                for k in (
                    "segment_id",
                    "frame_count",
                    "endpoint_translation_error_m",
                    "endpoint_rotation_error_deg",
                )
            }
            for s in segments
        ],
        "rpe_frames": {
            gap: {
                "count": v["translation_m"]["count"],
                "translation_m": v["translation_m"]["rmse"],
                "rotation_deg": v["rotation_deg"]["rmse"],
            }
            for gap, v in evaluation["rpe_by_frame_interval"].items()
        },
        "rpe_seconds": {
            gap: {
                "count": v["translation_m"]["count"],
                "translation_m": v["translation_m"]["rmse"],
                "rotation_deg": v["rotation_deg"]["rmse"],
            }
            for gap, v in evaluation["rpe_by_time_interval_s"].items()
        },
    }


def clock_pair_samples(run, chunks):
    pairs = {}
    for method in ("voxel", "tsdf"):
        common_count, clock_sum, timed_sum, gained, lost = 0, 0.0, 0.0, 0, 0
        differences = []
        for chunk in chunks:
            with np.load(
                run / f"maps/targets_{chunk:03d}.npz", allow_pickle=False
            ) as archive:
                observed = archive["observed"]
                clock = archive[f"{method}_clock_reference"]
                timed = archive[f"{method}_timed_reference"]
                valid = np.isfinite(observed) & (observed > 0)
                c = np.isfinite(clock) & (clock > 0)
                t = np.isfinite(timed) & (timed > 0)
                common = valid & c & t
                common_count += int(common.sum())
                clock_sum += float(np.abs(clock[common] - observed[common]).sum())
                timed_sum += float(np.abs(timed[common] - observed[common]).sum())
                gained += int((valid & ~c & t).sum())
                lost += int((valid & c & ~t).sum())
                for i in range(len(observed)):
                    mask = valid[i]
                    if not mask.any():
                        continue
                    ce = np.where(c[i], np.abs(clock[i] - observed[i]), np.inf)
                    te = np.where(t[i], np.abs(timed[i] - observed[i]), np.inf)
                    differences.append(
                        float(
                            np.minimum(te[mask], 0.2).mean()
                            - np.minimum(ce[mask], 0.2).mean()
                        )
                    )
        pairs[method] = {
            "common_pixels": common_count,
            "clock_common_mae_m": clock_sum / common_count if common_count else None,
            "timed_common_mae_m": timed_sum / common_count if common_count else None,
            "gained_samples": gained,
            "lost_samples": lost,
            "views_better": sum(v < -1e-12 for v in differences),
            "views_worse": sum(v > 1e-12 for v in differences),
            "views_unchanged": sum(abs(v) <= 1e-12 for v in differences),
            "per_view_cap_difference_m": differences,
        }
    return pairs


def maps_summary(report):
    scores = {}
    for name, values in report["pooled"]["scores"].items():
        if not name.endswith("_reference"):
            continue
        own = values["own"]
        scores[name] = {
            "coverage": own["coverage"],
            "samples": own["target_samples"],
            "capped_20cm_mae_m": own["missing_penalized_capped_mae_m"]["0.2"],
            "mae_m": own["absolute_depth_error_m"]["mean"],
            "explained_5cm": own["thresholds"]["0.05"]["explained_fraction_of_target"],
        }
    return {
        "scores": scores,
        "paired_common_pixels": report["pooled"]["pairs"],
        "chunk_ids": report["evaluated_chunk_ids"],
        "total_seconds": report["total_seconds"],
        "evaluated_views": sum(v["status"] == "evaluated" for v in report["views"]),
    }


def summarize(run, legacy=None):
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    result = {
        "source": report["source"],
        "split": report["split"],
        "baseline": pose(report["baseline_evaluation"]),
        "code_sha256": report["code_sha256"],
        "cost_seconds": report["cost_seconds"],
        "motion_status": report["training_motion_status"],
        "timing_config": report["timing_config"],
        "training_tracks": report["training_track_count"],
        "supported_tracks": report["training_supported_tracks"],
        "native_bindings": report["training_depth_bindings"],
        "repeat_exact": report["repeat_exact"],
        "variants": {},
    }
    for name, variant in report["variants"].items():
        closed = [w for w in variant["windows"] if "cost_at_close" in w]
        result["variants"][name] = {
            "pose": pose(variant["evaluation"]),
            "status": dict(Counter(w["status"] for w in variant["windows"])),
            "reasons": dict(
                Counter(
                    w.get("reason")
                    for w in variant["windows"]
                    if w["status"] in ("rejected", "skipped")
                )
            ),
            "closed_windows": len(closed),
            "changed_closed_costs": sum(
                not np.isclose(
                    w["cost_after_all_windows"],
                    w["cost_at_close"],
                    rtol=1e-10,
                    atol=1e-9,
                )
                for w in closed
            ),
            "increased_closed_costs": sum(
                w["cost_after_all_windows"] > w["cost_at_close"] + 1e-8 for w in closed
            ),
            "total_seconds": variant["optimization_seconds"],
            "factor_preparation_seconds": variant.get("factor_preparation_seconds"),
            "linear_solver_seconds": variant.get("linear_solver_seconds"),
        }
    if (run / "maps/report.json").exists():
        result["maps"] = maps_summary(
            json.loads((run / "maps/report.json").read_text(encoding="utf-8"))
        )
        if {"voxel_clock_reference", "voxel_timed_reference"}.issubset(
            result["maps"]["scores"]
        ):
            result["maps"]["timed_against_clock"] = clock_pair_samples(
                run, result["maps"]["chunk_ids"]
            )
    if legacy is not None:
        previous = json.loads((legacy / "report.json").read_text(encoding="utf-8"))
        if (
            previous["split"] != report["split"]
            or previous["source"] != report["source"]
            or previous["baseline_trajectory_sha256"]
            != report["baseline_trajectory_sha256"]
        ):
            raise ValueError("legacy optimization population or initializer differs")
        result["legacy_anchored"] = pose(previous["variants"]["anchored"]["evaluation"])
        if "maps" in result:
            old_maps = json.loads(
                (legacy / "maps/report.json").read_text(encoding="utf-8")
            )
            if old_maps["evaluated_chunk_ids"] != result["maps"]["chunk_ids"]:
                raise ValueError("legacy map chunks differ")
            for chunk in result["maps"]["chunk_ids"]:
                with (
                    np.load(run / f"maps/targets_{chunk:03d}.npz") as current,
                    np.load(legacy / f"maps/targets_{chunk:03d}.npz") as old,
                ):
                    for key in (
                        "frame_indices",
                        "observed",
                        "voxel_sparse_reference",
                        "tsdf_sparse_reference",
                    ):
                        np.testing.assert_array_equal(
                            current[key], old[key], err_msg=key
                        )
            result["legacy_maps"] = maps_summary(old_maps)
            result["legacy_target_samples_and_sparse_predictions_exact"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("summary output already exists")
    summaries = {}
    for run in args.runs:
        legacy_name = run.name.removesuffix("_assumed").removesuffix("_strict")
        summaries[run.name] = summarize(
            run, args.legacy_root / legacy_name if args.legacy_root else None
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
