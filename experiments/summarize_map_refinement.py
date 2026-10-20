"""Compact paired evidence from completed, frozen local-map comparisons.

Per-scene and per-block populations are descriptive. Overlapping trajectory
pairs and neighboring image pixels are not independent statistical replicates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from incremental_local_map import common_errors

from scene_recall.odometry.evaluation import error_statistics
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256


def metrics(scores):
    own = scores["own"]
    common = scores.get("common_reference_pose", scores["common_all_modes"])
    return {
        "common_mae_m": common["absolute_depth_error_m"]["mean"],
        "common_samples": common["target_samples"],
        "coverage": own["coverage"],
        "explained_5cm": own["thresholds"]["0.05"]["explained_fraction_of_target"],
    }


def capped_scores(directory):
    """Charge missing predictions on every valid target, without dropping views."""
    totals, counts, views = {}, {}, {}
    for path in sorted(directory.glob("views_*_radius_0.01.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "observed" not in data:
                continue
            observed = data["observed"]
            valid = np.isfinite(observed) & (observed > 0)
            for name in data.files:
                if not name.endswith("_reference_pose") or name.startswith("common_"):
                    continue
                prediction = data[name]
                errors = np.minimum(
                    np.where(
                        np.isfinite(prediction), abs(prediction - observed), np.inf
                    ),
                    0.2,
                )
                totals[name] = totals.get(name, 0.0) + float(errors[valid].sum())
                counts[name] = counts.get(name, 0) + int(valid.sum())
                for i, frame in enumerate(data["evaluated_frame_indices"]):
                    views.setdefault(name, []).append(
                        {
                            "frame_index": int(frame),
                            "capped_error_m": float(errors[i][valid[i]].mean())
                            if valid[i].any()
                            else None,
                        }
                    )
    return {
        name: {
            "pooled_m": totals[name] / counts[name] if counts[name] else None,
            "per_view": views[name],
        }
        for name in totals
    }


def summarize_case(directory):
    comparison = json.loads((directory / "comparison.json").read_text())
    result = {
        "directory": str(directory.resolve()),
        "comparison_sha256": sha256(directory / "comparison.json"),
        "source": comparison["source"],
        "variants": {},
        "paired": {},
    }
    loaded = {}
    for name, saved in comparison["variants"].items():
        path = Path(saved["run"])
        for filename, field in (
            ("report.json", "report_sha256"),
            ("trajectory.npz", "trajectory_sha256"),
        ):
            if sha256(path / filename) != saved[field]:
                raise ValueError(f"changed saved input: {path / filename}")
        report, rows = loaded[name] = load_trajectory(path)
        evaluation = report["evaluation"]
        diagnostics = [f["diagnostics"] for f in report["frames"] if f["diagnostics"]]
        result["variants"][name] = {
            "segment_ate_m": [
                {
                    "start": s["start_frame"],
                    "end": s["end_frame"],
                    "references": s["reference_count"],
                    "rmse": s["aligned_ate_translation_m"]["rmse"],
                    "endpoint_translation_m": s["endpoint_translation_error_m"],
                    "endpoint_rotation_deg": s["endpoint_rotation_error_deg"],
                }
                for s in evaluation["segments"]
            ],
            "continuous": evaluation["continuous_trajectory"] is not None,
            "lost_frames": [r.frame_index for r in rows if r.status == "lost"],
            "rpe": evaluation["rpe_by_frame_interval"],
            "tracking_modes": saved["tracking_modes"],
            "policy_diagnostics": {
                key: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "max": float(max(values)),
                    "frames": len(values),
                }
                for key in (
                    "recent_depth_anchors",
                    "carry_matches",
                    "index_queries",
                    "recovery_matches",
                    "reanchored_landmarks",
                    "geometry_rejected_associations",
                    "indexed_landmarks",
                )
                if (values := [d[key] for d in diagnostics if key in d])
            },
        }
    for baseline in ("sparse", "budget", "local"):
        if baseline not in loaded or "recent" not in loaded:
            continue
        paired = {
            str(k): common_errors(loaded[baseline][1], loaded["recent"][1], k)
            for k in (1, 5, 10, 20, 50, 100)
        }
        result["paired"][baseline] = paired
    if "recent" in loaded and "budget" in loaded:
        recent, baseline = loaded["recent"][0], loaded["budget"][0]
        regressions, populations = [], {}
        for index, (a, b) in enumerate(
            zip(
                recent["evaluation"]["pair_errors_m_deg"],
                baseline["evaluation"]["pair_errors_m_deg"],
                strict=True,
            ),
            start=1,
        ):
            if a["final"] is None or b["final"] is None:
                continue
            frame = recent["frames"][index]
            diagnostics = frame["diagnostics"]
            delta = a["final"][0] - b["final"][0]
            event = (
                "insertion"
                if diagnostics.get("keyframe_inserted")
                else "after_insertion"
                if recent["frames"][index - 1]["diagnostics"].get("keyframe_inserted")
                else "other"
            )
            populations.setdefault(event, []).append(delta)
            regressions.append(
                {
                    "target_frame": index,
                    "source_frame_index": frame["source_frame_index"],
                    "translation_delta_m": delta,
                    "recent_error_m_deg": a["final"],
                    "budget_error_m_deg": b["final"],
                    "diagnostics": {
                        k: v
                        for k, v in diagnostics.items()
                        if not k.endswith("_seconds")
                    },
                }
            )
        result["failure_diagnostics"] = {
            "largest_translation_regressions": sorted(
                regressions, key=lambda r: -r["translation_delta_m"]
            )[:10],
            "keyframe_event_translation_delta_m": {
                k: error_statistics(v) for k, v in populations.items()
            },
            "interpretation": "conditional diagnostics, events are selected by each frontend's measurements and are not causal interventions",
        }
    result["fusion"] = {"status": comparison.get("fusion", {}).get("status")}
    if result["fusion"]["status"] == "evaluated":
        fusion = comparison["fusion"]
        if sha256(directory / "maps/report.json") != fusion["report_sha256"]:
            raise ValueError("changed surface-fusion report")
        summary = fusion["summaries"]["0.01"]
        result["fusion"].update(
            evaluated_views=summary["evaluated_views"],
            unavailable=summary["unavailable_status_counts"],
            scores={
                n: metrics(s)
                for n, s in summary["pooled"].items()
                if n.endswith("_reference_pose") or n == "gt"
            },
            capped_missing_penalty_20cm=capped_scores(directory / "maps"),
        )
    else:
        result["fusion"]["reason"] = comparison.get("fusion", {}).get("reason")
    tsdf = comparison.get("tsdf", {})
    result["tsdf"] = {k: v for k, v in tsdf.items() if k != "scores"}
    if tsdf.get("status") == "evaluated":
        result["tsdf"]["scores"] = {
            n: metrics(s)
            for n, s in tsdf["scores"].items()
            if n.endswith("_reference_pose") or n == "gt"
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh evidence output")
    result = {
        "driver_sha256": sha256(Path(__file__)),
        "interpretation": "descriptive paired populations, no independent-pixel or overlapping-pair significance claim",
        "cases": {p.name: summarize_case(p) for p in args.case},
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
