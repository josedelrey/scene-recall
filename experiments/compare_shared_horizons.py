"""Compare an earlier trajectory under later causal publication horizons."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256
from scene_recall.slam.evaluation import score_components


def compare(runs, prefix):
    loaded = {label: load_trajectory(path) for label, path in runs.items()}
    if any(len(rows) < prefix for _, rows in loaded.values()):
        raise ValueError("every run must cover the requested prefix")
    signatures = {
        json.dumps(
            {
                "dataset": report["source"]["dataset"],
                "sequence": report["source"]["sequence_id"],
                "start": report["source"]["start"],
                "calibration": report["calibration"],
                "provenance": [
                    frame["provenance"] for frame in report["frames"][:prefix]
                ],
            },
            sort_keys=True,
        )
        for report, _ in loaded.values()
    }
    if len(signatures) != 1:
        raise ValueError("horizon comparisons require the same received prefix")
    atlas = {label: dict(np.load(path / "atlas.npz")) for label, path in runs.items()}
    emitted = {
        label: dict(np.load(path / "atlas_emitted.npz")) for label, path in runs.items()
    }
    references = [row.reference_pose for row in next(iter(loaded.values()))[1]][:prefix]
    common = {
        i
        for i in range(prefix)
        if all(
            state["component_id"][i] == state["component_id"][0]
            for state in atlas.values()
        )
    }
    results = {}
    for label, (report, rows) in loaded.items():
        results[label] = {
            "inputs_received": len(rows),
            "report_sha256": sha256(runs[label] / "report.json"),
            "revised_prefix": score_components(
                atlas[label]["T_component_depth"][:prefix],
                atlas[label]["component_id"][:prefix],
                references,
                common,
            ),
        }
    changes, reproducibility = {}, {}
    for first, second in combinations(runs, 2):
        # Normalize the gauge before measuring physical trajectory revisions.
        a = (
            np.linalg.inv(atlas[first]["T_component_depth"][0])
            @ atlas[first]["T_component_depth"][:prefix]
        )
        b = (
            np.linalg.inv(atlas[second]["T_component_depth"][0])
            @ atlas[second]["T_component_depth"][:prefix]
        )
        errors = [pose_error(a[i], b[i]) for i in sorted(common)]
        changes[f"{first}:{second}"] = {
            "translation_m": error_statistics(e[0] for e in errors),
            "rotation_deg": error_statistics(e[1] for e in errors),
        }
        if loaded[first][0]["online_config"] == loaded[second][0]["online_config"]:
            reproducibility[f"{first}:{second}"] = {
                name: bool(
                    np.array_equal(
                        emitted[first][name][:prefix],
                        emitted[second][name][:prefix],
                        equal_nan=True,
                    )
                )
                for name in emitted[first]
                if emitted[first][name].ndim > 0
            }
    return {
        "prefix_frames": prefix,
        "common_root_indices": sorted(common),
        "runs": results,
        "gauge_normalized_revisions": changes,
        "identical_policy_emitted_prefix": reproducibility,
        "scope": "fixed received prefix, final publication at each horizon, references used only for scoring",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="label=/path/to/run")
    parser.add_argument("--prefix", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh horizon report")
    runs = {
        label: Path(path)
        for label, path in (value.split("=", 1) for value in args.runs)
    }
    args.output.write_text(
        json.dumps(compare(runs, args.prefix), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
