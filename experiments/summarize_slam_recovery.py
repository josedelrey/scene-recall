"""Compact paired localization, connection and reconstruction study tables."""

import argparse
import json
from pathlib import Path

import numpy as np
from incremental_local_map import common_errors

from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256


def summarize(root):
    cases = {}
    for path in sorted(root.rglob("comparison.json")):
        if path.relative_to(root).parts[0].startswith(("serial", "verified_exports")):
            continue
        if "pilot" in str(path) or path.parent.name in (
            "pioneer_full",
            "sfm_lab_room_1",
        ):
            continue
        comparison = json.loads(path.read_text())
        if "policies" not in comparison:
            continue
        policies, loaded = {}, {}
        for name, result in comparison["policies"].items():
            _, rows = load_trajectory(path.parent / name)
            loaded[name] = rows
            evaluation = result["evaluation"]
            largest = max(evaluation["segments"], key=lambda s: s["frame_count"])
            fallback = result["accepted_fallback_reference_errors"]
            current = {
                "frames": len(rows),
                "references": evaluation["reference_valid_frames"],
                "losses": evaluation["failed_transitions"],
                "largest_local_frames": largest["frame_count"],
                "largest_local_ate_m": largest["aligned_ate_translation_m"]["rmse"],
                "fallbacks": len(fallback),
                "fallbacks_outside_10cm_5deg": sum(
                    e["translation_m"] > 0.1 or e["rotation_deg"] > 5 for e in fallback
                ),
                "max_fallback_translation_m": max(
                    (e["translation_m"] for e in fallback), default=None
                ),
                "process_and_atlas_seconds": result["process_and_atlas_seconds"],
            }
            if "atlas" in result:
                atlas = result["atlas"]
                current["atlas"] = {
                    k: atlas[k]
                    for k in (
                        "component_count",
                        "root_component_frames",
                        "largest_component_frames",
                        "causal_root_frames",
                        "links_outside_10cm_5deg",
                        "event_statuses",
                        "peak_entries",
                        "query_seconds",
                    )
                }
                current["atlas"]["links"] = atlas["accepted_link_reference_errors"]
                current["atlas"]["scored_links"] = len(
                    atlas["accepted_link_reference_errors"]
                )
                current["atlas"]["unscored_links"] = atlas["event_statuses"].get(
                    "connected", 0
                ) - len(atlas["accepted_link_reference_errors"])
                current["atlas"]["root_translation_m"] = atlas["groups"]["0"][
                    "anchored_translation_m"
                ]
                current["atlas"]["root_rotation_deg"] = atlas["groups"]["0"][
                    "anchored_rotation_deg"
                ]
            policies[name] = current
        if "control" in loaded:
            for name, rows in loaded.items():
                policies[name]["common_with_control"] = {
                    str(k): common_errors(loaded["control"], rows, k)
                    for k in (1, 10, 100)
                }
        cases[str(path.parent.relative_to(root))] = {
            "comparison_sha256": sha256(path),
            "policies": policies,
        }
    maps = {}
    for path in sorted(root.glob("maps_*/report.json")):
        result = json.loads(path.read_text())
        maps[path.parent.name] = {"sha256": sha256(path), **result}
    agreements = {}
    prior = root.parent / "slam_system_study/cases.json"
    if prior.exists():
        for name, original in json.loads(prior.read_text()).items():
            current = root / "regression" / name / "control"
            if (current / "trajectory.npz").exists():
                _, a = load_trajectory(Path(original["initial_run"]))
                _, b = load_trajectory(current)
                same = len(a) == len(b) and all(
                    x.segment_id == y.segment_id
                    and x.status == y.status
                    and np.array_equal(x.T_SC_D, y.T_SC_D)
                    for x, y in zip(a, b, strict=True)
                )
                agreements[name] = same
    no_atlas = root / "xyz80_no_atlas"
    independent = {}
    if (no_atlas / "comparison.json").exists():
        for name in ("control", "low_pnp", "rgbd"):
            _, a = load_trajectory(root / "outage_xyz80" / name)
            _, b = load_trajectory(no_atlas / name)
            independent[name] = all(
                x.segment_id == y.segment_id
                and x.status == y.status
                and np.array_equal(x.T_SC_D, y.T_SC_D)
                for x, y in zip(a, b, strict=True)
            )
    result = {
        "cases": cases,
        "reconstruction": maps,
        "default_pose_segment_status_agreement": agreements,
        "atlas_does_not_change_local_estimator": independent,
    }
    (root / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    assert all(agreements.values()) and all(independent.values())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root)
