"""Summarize pending evidence and causal outcomes without hiding populations."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.evaluation import score_components


def summarize(run, reference):
    report, rows = load_trajectory(run)
    control, _original = load_trajectory(reference)
    if (
        report["source"]["canonical_input_sha256"]
        != control["source"]["canonical_input_sha256"]
    ):
        raise ValueError("paired sources differ")
    a, b = [dict(np.load(p / "atlas.npz")) for p in (reference, run)]
    common = set(
        map(
            int,
            np.flatnonzero(
                (a["component_id"] == a["component_id"][0])
                & (b["component_id"] == b["component_id"][0])
            ),
        )
    )
    paired = {
        name: score_components(
            state["T_component_depth"],
            state["component_id"],
            [r.reference_pose for r in rows],
            common,
        )
        for name, state in [("reference", a), ("candidate", b)]
    }
    result = {
        "report_sha256": sha256(run / "report.json"),
        "reference_report_sha256": sha256(reference / "report.json"),
        "received": len(rows),
        "connected": report["revised_evaluation"]["root_frames"],
        "whole": report["revised_evaluation"]["groups"]["0"],
        "paired": paired,
        "paired_received_indices": sorted(common),
        "failed_transitions": report["evaluation"]["failed_transitions"],
        "timing": report["timing"],
    }
    if (run / "shared_estimation.npz").exists():
        raw = dict(np.load(run / "shared_estimation.npz"))
        identities = dict(raw["live_identities"])
        live = np.array([identities[t] for t in raw["track_id"]])
        native = np.isfinite(raw["native_points"]).all(axis=1)
        natives = set(live[native])
        counts = Counter(live)
        selected = {k for k, v in counts.items() if v >= 2 and k in natives} | set(
            raw.get("bearing_seed_ids", ())
        )
        active = np.isin(live, list(selected))
        pending_tracks = {
            int(track) for _, owner, track in raw["identity_owners"] if owner < 0
        }
        raw_pending = np.isin(raw["track_id"], list(pending_tracks))
        proposals = []
        for f in report["frames"]:
            for query in f["diagnostics"].get("queries", []):
                if "bearing" in query:
                    proposals.append(query["bearing"])
        fits = [
            fit
            for f in report["frames"]
            if f["diagnostics"].get("graph")
            for fit in f["diagnostics"]["graph"].get("shared", [])
        ]
        last = fits[-1] if fits else {}
        row_keys = np.stack((raw["view"], raw["feature_index"]), axis=1)
        native_keys = np.stack(
            (raw["view"][native], raw["native_index"][native]), axis=1
        )
        result["evidence"] = {
            "retained_cameras": len(np.unique(raw["view"])),
            "retained_rows": len(live),
            "native_rows": int(native.sum()),
            "image_only_rows": int((~native).sum()),
            "frontend_unowned_rows": int(raw_pending.sum()),
            "frontend_unowned_rows_eligible": int((raw_pending & active).sum()),
            "eligible_rows": int(active.sum()),
            "eligible_image_only_rows": int((active & ~native).sum()),
            "pure_initialized_classes": len(raw.get("bearing_seed_ids", ())),
            "pure_initialized_rows": int(
                np.isin(live, raw.get("bearing_seed_ids", ())).sum()
            ),
            "cached_points": len(raw["point_ids"]),
            "unique_features": len(np.unique(row_keys, axis=0)) == len(row_keys),
            "unique_native": len(np.unique(native_keys, axis=0)) == len(native_keys),
            "proposal_outcomes": dict(Counter(p["status"] for p in proposals)),
            "proposal_edges": len(raw.get("bearing_edges", ())),
            "active_bearing_edges": sum(
                f"{x}:{y}" in last.get("associations", {})
                for x, y, _ in raw.get("bearing_edges", ())
            ),
            "last_solve_status": last.get("status"),
            "last_bearings": last.get("bearings"),
            "solver_statuses": dict(Counter(f["status"] for f in fits)),
            "scope": "eligible counts describe structurally initialized rows at final attempt, optimized point cache and rejection status reported separately",
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("choose a fresh evidence summary")
    manifest = json.loads(args.manifest.read_text())
    results = {}
    for case, row in manifest.items():
        reference = args.root / case / "control"
        results[case] = {
            name: summarize(args.root / case / name, reference)
            for name in row["policies"]
            if (args.root / case / name / "report.json").exists()
        }
    args.output.write_text(
        json.dumps(
            {
                "package_sha256": code_hash(),
                "driver_sha256": sha256(Path(__file__)),
                "manifest_sha256": sha256(args.manifest),
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
