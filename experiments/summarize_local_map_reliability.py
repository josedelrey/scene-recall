"""Paired motion and reconstruction summaries for the frozen reliability study."""

import argparse
import json
from pathlib import Path

import numpy as np
from incremental_local_map import common_errors

from scene_recall.odometry.evaluation import error_statistics
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256


def map_predictions(path):
    parts = []
    for p in sorted(path.glob("views_*_radius_0.01.npz")):
        with np.load(p) as a:
            if len(a["evaluated_frame_indices"]):
                parts.append(
                    {
                        k: a[k].copy()
                        for k in (
                            "observed",
                            "sparse_reference_pose",
                            "local_map_recent_reference_pose",
                            "gt",
                            "evaluated_frame_indices",
                        )
                    }
                )
    return (
        {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None
    )


def geometry_scores(observed, predictions):
    valid = np.isfinite(observed) & (observed > 0)
    common = valid.copy()
    for p in predictions.values():
        common &= np.isfinite(p) & (p > 0)
    result = {}
    for name, p in predictions.items():
        present = valid & np.isfinite(p) & (p > 0)
        residual = np.abs(p - observed)
        result[name] = {
            "valid_target_samples": int(valid.sum()),
            "coverage": float(present.sum() / valid.sum()),
            "explained_fraction_5cm": float(
                (present & (residual <= 0.05)).sum() / valid.sum()
            ),
            "capped_missing_20cm_m": float(
                np.minimum(np.where(present, residual, np.inf)[valid], 0.2).mean()
            ),
            "common_samples": int(common.sum()),
            "common_mae_m": float(residual[common].mean()) if common.any() else None,
        }
    return result


def summarize_case(path):
    paths = {n: path / n for n in ("budget", "legacy", "rgb", "rgbd", "family")}
    loaded = {n: load_trajectory(p) for n, p in paths.items()}
    result = {"comparison_sha256": sha256(path / "comparison.json"), "motion": {}}
    for name, (report, rows) in loaded.items():
        evaluation = report["evaluation"]
        segment = max(evaluation["segments"], key=lambda s: s["frame_count"])
        result["motion"][name] = {
            "segments": evaluation["segment_count"],
            "losses": [r.frame_index for r in rows if r.status == "lost"],
            "reference_count": evaluation["reference_valid_frames"],
            "ate_m": segment["aligned_ate_translation_m"]["rmse"],
            "ate_scope": "full run"
            if evaluation["continuous_trajectory"]
            else "largest segment",
            "endpoint_rotation_deg": segment["endpoint_rotation_error_deg"],
            "rpe": evaluation["rpe_by_frame_interval"],
            "common_with_budget": {
                str(k): common_errors(loaded["budget"][1], rows, k)
                for k in (1, 10, 100)
            },
        }
        if report["source"]["dataset"] == "scannet_v2":
            groups = {"boundary10": [], "other": []}
            for i, entry in enumerate(evaluation["pair_errors_m_deg"], start=1):
                if entry["final"] is not None:
                    label = (
                        "boundary10"
                        if (report["source"]["start"] + i) % 10 == 0
                        else "other"
                    )
                    groups[label].append(entry["final"])
            result["motion"][name]["reference_boundary_sensitivity"] = {
                label: {
                    "translation_m": error_statistics(e[0] for e in errors),
                    "rotation_deg": error_statistics(e[1] for e in errors),
                }
                for label, errors in groups.items()
            }
    gathered = {
        n: map_predictions(path / f"maps_{n}")
        for n in ("legacy", "rgb", "rgbd", "family")
    }
    if all(g is not None for g in gathered.values()):
        first = gathered["legacy"]
        for g in gathered.values():
            for key in (
                "observed",
                "sparse_reference_pose",
                "gt",
                "evaluated_frame_indices",
            ):
                if not np.array_equal(first[key], g[key], equal_nan=True):
                    raise ValueError(
                        "fusion comparison populations or shared controls differ"
                    )
        predictions = {
            n: g["local_map_recent_reference_pose"] for n, g in gathered.items()
        }
        predictions.update(budget=first["sparse_reference_pose"], gt=first["gt"])
        result["surface"] = {
            "evaluated_views": len(first["observed"]),
            "target_indices": first["evaluated_frame_indices"].tolist(),
            "scores": geometry_scores(first["observed"], predictions),
        }
        np.savez_compressed(
            path / "common_surface_predictions.npz",
            observed=first["observed"],
            target_indices=first["evaluated_frame_indices"],
            **predictions,
        )
    else:
        result["surface"] = {
            "status": "withheld",
            "reason": "incomplete matched fusion population",
        }
    tsdf = {
        n: json.loads((path / f"tsdf_{n}/report.json").read_text())
        for n in ("legacy", "rgb", "rgbd", "family")
        if (path / f"tsdf_{n}/report.json").exists()
    }
    if len(tsdf) == 4 and all(r["status"] == "evaluated" for r in tsdf.values()):
        predictions = {}
        reference, indices = None, None
        for name in tsdf:
            with np.load(path / f"tsdf_{name}/geometry_and_predictions.npz") as a:
                if reference is None:
                    reference = a["observed"].copy()
                    predictions.update(
                        budget=a["budget_reference_pose"].copy(), gt=a["gt"].copy()
                    )
                    indices = tsdf[name]["target_indices"]
                else:
                    if (
                        not np.array_equal(reference, a["observed"], equal_nan=True)
                        or indices != tsdf[name]["target_indices"]
                    ):
                        raise ValueError("TSDF target population differs")
                    for control in ("budget", "gt"):
                        key = "budget_reference_pose" if control == "budget" else "gt"
                        if not np.array_equal(
                            predictions[control], a[key], equal_nan=True
                        ):
                            raise ValueError("TSDF shared control changed")
                predictions[name] = a[f"{name}_reference_pose"].copy()
        result["tsdf"] = {
            "evaluated_views": len(reference),
            "scores": geometry_scores(reference, predictions),
        }
    else:
        result["tsdf"] = {
            "status": "withheld",
            "reasons": {n: r.get("reason") for n, r in tsdf.items()},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.root / "frozen_protocol.json").read_text())
    cases = {
        c["name"]: summarize_case(args.root / c["name"]) for c in protocol["cases"]
    }
    result = {
        "driver_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(args.root / "frozen_protocol.json"),
        "cases": cases,
    }
    (args.root / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
