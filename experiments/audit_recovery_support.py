"""Describe image features, metric anchors and temporal match support by tracking status."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scene_recall.multiview.tracking import match_descriptors
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def audit(local_run, features, output):
    if output.exists():
        raise ValueError("choose a fresh output file")
    _, rows = load_trajectory(local_run)
    cv2.setNumThreads(1)
    values, previous = [], None
    with np.load(features, allow_pickle=False) as cache:
        for i, row in enumerate(rows):
            points, descriptors = cache[f"{i}_points_R"], cache.get(f"{i}_descriptors")
            valid = np.isfinite(points).all(axis=1)
            entry = {
                "frame": i,
                "status": row.status,
                "features": len(points),
                "metric_features": int(valid.sum()),
            }
            if previous is not None:
                pairs, _ = match_descriptors(previous[1], descriptors, 0.75)
                entry.update(
                    mutual_matches=len(pairs),
                    source_metric_matches=int(previous[0][pairs[:, 0]].sum())
                    if len(pairs)
                    else 0,
                    both_metric_matches=int(
                        (previous[0][pairs[:, 0]] & valid[pairs[:, 1]]).sum()
                    )
                    if len(pairs)
                    else 0,
                )
            values.append(entry)
            previous = valid, descriptors
    groups = {}
    for status in ("all", "tracked", "lost"):
        selected = [v for v in values[1:] if status == "all" or v["status"] == status]
        if not selected:
            continue
        groups[status] = {
            "transitions": len(selected),
            "quantiles": {
                key: {
                    str(q): float(np.quantile([v[key] for v in selected], q))
                    for q in (0, 0.1, 0.5, 0.9, 1)
                }
                for key in (
                    "features",
                    "metric_features",
                    "mutual_matches",
                    "source_metric_matches",
                    "both_metric_matches",
                )
            },
            "source_matches_below_20": sum(
                v["source_metric_matches"] < 20 for v in selected
            ),
            "both_depth_matches_below_12": sum(
                v["both_metric_matches"] < 12 for v in selected
            ),
        }
    result = {
        "driver_sha256": sha256(Path(__file__)),
        "package_sha256": code_hash(),
        "report_sha256": sha256(local_run / "report.json"),
        "features_sha256": sha256(features),
        "scope": "descriptive support, no reference poses or threshold fitting",
        "groups": groups,
        "frames": values,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    audit(args.local_run, args.features, args.output_file)
