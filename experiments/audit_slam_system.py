"""Read-only full-system error and withheld-prediction agreement audit."""

import argparse
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
from slam_generalization import summary

from scene_recall.odometry.evaluation import pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256


def motion_errors(rows, interval):
    errors = {str(i): [] for i in range(interval)}
    for a, b in pairwise(rows):
        if (
            a.segment_id != b.segment_id
            or a.reference_pose is None
            or b.reference_pose is None
        ):
            continue
        error = pose_error(
            np.linalg.inv(a.T_SC_D) @ b.T_SC_D,
            np.linalg.inv(a.reference_pose) @ b.reference_pose,
        )
        errors[str(b.frame_index % interval)].append(error)
    return {
        key: {
            "translation_m": summary([p[0] for p in values]),
            "rotation_deg": summary([p[1] for p in values]),
        }
        for key, values in errors.items()
    }


def prediction_agreement(directory):
    result = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in ("summary.json", "protocol.json"):
            continue
        case = json.loads(path.read_text())
        if "blocks" not in case:
            continue
        values = {}
        for model in ("native", "axial", "weak_depth", "first_depth"):
            counts = Counter()
            for block in case["blocks"]:
                initial, final = block["initial"], block["models"][model]
                if final["solver"]["status"] != "refined":
                    counts["not_refined"] += 1
                    continue
                counts["refined"] += 1
                if not initial["reference"]["available"]:
                    counts["missing_reference"] += 1
                    continue
                improved_rgb = (
                    final["prediction"]["track_balanced_capped_rgb_px"]["mean"]
                    < initial["prediction"]["track_balanced_capped_rgb_px"]["mean"]
                )
                improved_pose = (
                    final["reference"]["anchored_translation_m"]["rmse"]
                    < initial["reference"]["anchored_translation_m"]["rmse"]
                )
                improved_rotation = (
                    final["reference"]["anchored_rotation_deg"]["rmse"]
                    < initial["reference"]["anchored_rotation_deg"]["rmse"]
                )
                counts["rgb_improves" if improved_rgb else "rgb_worsens"] += 1
                counts[
                    "position_improves" if improved_pose else "position_worsens"
                ] += 1
                counts[
                    "rotation_improves" if improved_rotation else "rotation_worsens"
                ] += 1
                if improved_rgb and not improved_pose:
                    counts["rgb_improves_position_worsens"] += 1
                if improved_rgb and not improved_rotation:
                    counts["rgb_improves_rotation_worsens"] += 1
            values[model] = dict(counts)
        result[path.stem] = {
            "models": values,
            "disconnected_training_blocks": sum(
                len(b["components"]) > 1 for b in case["blocks"]
            ),
            "blocks": len(case["blocks"]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output file")
    cases = json.loads(args.cases.read_text())
    result = {
        "driver_sha256": sha256(Path(__file__)),
        "cases_sha256": sha256(args.cases),
        "trajectories": {},
        "prediction_agreement": prediction_agreement(args.prediction_dir),
    }
    for name, paths in cases.items():
        pair = {}
        for label, key in (("native", "run"), ("control", "initial_run")):
            path = Path(paths[key])
            report, rows = load_trajectory(path)
            interval = report["config"].get("keyframe_interval", 5)
            diagnostics = [f["diagnostics"] for f in report["frames"]]
            pair[label] = {
                "report_sha256": sha256(path / "report.json"),
                "trajectory_sha256": sha256(path / "trajectory.npz"),
                "frame_count": len(rows),
                "solver_statuses": dict(
                    Counter(d["solver"]["status"] for d in diagnostics if "solver" in d)
                ),
                "solver_reasons": dict(
                    Counter(d["solver"]["reason"] for d in diagnostics if "solver" in d)
                ),
                "graph_restarts": sum("graph_restart" in d for d in diagnostics),
                "one_frame_rpe_by_phase": motion_errors(rows, interval),
            }
        result["trajectories"][name] = pair
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
