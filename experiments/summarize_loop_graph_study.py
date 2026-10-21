"""Summarize fixed populations, external geometry and retained limitations."""

import argparse
import json
from pathlib import Path

import numpy as np

from scene_recall.run_metadata import sha256


def summarize(root):
    cases, recordings = {}, {}
    for path in sorted((root / "final").glob("*/*/comparison.json")):
        case = str(path.parent.relative_to(root / "final"))
        report = json.loads(path.read_text())
        source = json.loads((path.parent / "forest_mean" / "report.json").read_text())[
            "source"
        ]
        if case.startswith("natural"):
            recordings.setdefault(source["sequence_id"], set()).update(
                range(source["start"], source["stop"])
            )
        methods = {}
        root_masks = []
        for name, value in report["policies"].items():
            evaluation = value["evaluation"]
            first = evaluation["groups"]["0"]
            loops = value.get("loop_reference_errors", [])
            rejected = set(map(tuple, value["diagnostic"].get("rejected_loops", [])))
            active = [e for e in loops if (e["source"], e["target"]) not in rejected]
            count = sum(g["reference_frames"] for g in evaluation["groups"].values())
            component_rmse = np.sqrt(
                sum(
                    (g["translation_m"]["rmse"] or 0) ** 2 * g["reference_frames"]
                    for g in evaluation["groups"].values()
                )
                / max(count, 1)
            )
            methods[name] = {
                "root_frames": evaluation["root_frames"],
                "root_reference_frames": first["reference_frames"],
                "root_translation_rmse_m": first["translation_m"]["rmse"],
                "root_rotation_rmse_deg": first["rotation_deg"]["rmse"],
                "component_gauged_translation_rmse_m": float(component_rmse),
                "component_reference_frames": count,
                "component_count": evaluation["component_count"],
                "ten_frame_translation_rmse_m": evaluation[
                    "ten_frame_relative_translation_m"
                ]["rmse"],
                "ten_frame_rotation_rmse_deg": evaluation[
                    "ten_frame_relative_rotation_deg"
                ]["rmse"],
                "active_loops": value["diagnostic"].get("active_loops"),
                "scored_active_loops": len(active),
                "unscored_active_loops": (value["diagnostic"].get("active_loops") or 0)
                - len(active),
                "active_loops_outside_10cm_5deg": sum(
                    e["translation_m"] > 0.1 or e["rotation_deg"] > 5 for e in active
                ),
                "active_loops_outside_2cm_1deg": sum(
                    e["translation_m"] > 0.02 or e["rotation_deg"] > 1 for e in active
                ),
                "rejected_loops": len(rejected),
                "solve_seconds_shared_matrix": value["diagnostic"].get("solve_seconds"),
            }
            with np.load(path.parent / name / "atlas.npz") as arrays:
                root_masks.append(arrays["component_id"] == arrays["component_id"][0])
        cases[case] = {
            "source": source,
            "methods": methods,
            "failed_transitions": report["local_failed_transitions"],
            "identical_root_population": all(
                np.array_equal(root_masks[0], mask) for mask in root_masks[1:]
            ),
            "comparison_sha256": sha256(path),
        }
    reconstruction = {}
    for scope in ("maps", "extent", "extent_surface"):
        for path in sorted((root / scope).glob("*/*/report.json")):
            report = json.loads(path.read_text())
            reconstruction[scope + "/" + str(path.parent.relative_to(root / scope))] = (
                report
            )
        for path in sorted((root / scope).glob("*/*/failure.json")):
            reconstruction[scope + "/" + str(path.parent.relative_to(root / scope))] = (
                json.loads(path.read_text())
            )
    natural = [
        r
        for k, r in cases.items()
        if k.startswith("natural/")
        and r["methods"]["forest_mean"]["root_reference_frames"] >= 3
    ]
    outcomes = {"improved": 0, "worsened": 0, "unchanged": 0}
    for case in natural:
        first = case["methods"]["forest_mean"]["root_translation_rmse_m"]
        second = case["methods"]["graph_votes_robust"]["root_translation_rmse_m"]
        outcomes[
            "unchanged"
            if abs(first - second) < 1e-10
            else "improved"
            if second < first
            else "worsened"
        ] += 1
    summary = {
        "natural_populations": sum(k.startswith("natural/") for k in cases),
        "natural_recordings": len(recordings),
        "distinct_natural_inputs": sum(map(len, recordings.values())),
        "natural_inputs_by_recording": {k: len(v) for k, v in recordings.items()},
        "natural_root_translation_outcomes": outcomes,
        "all_cases_preserve_root_population": all(
            c["identical_root_population"] for c in cases.values()
        ),
        "cases": cases,
        "reconstruction": reconstruction,
        "driver_sha256": sha256(Path(__file__)),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.study_root)
