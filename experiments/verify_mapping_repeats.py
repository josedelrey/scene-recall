"""Verify repeated mapping geometry and diagnostics, excluding runtime fields."""

import argparse
import json
from pathlib import Path

import numpy as np


def verify(first: Path, second: Path) -> dict:
    a, b = [json.loads((path / "report.json").read_text()) for path in (first, second)]
    heldout = a.get("experiment") == "held_out_fusion_views"
    consensus = a.get("experiment") == "single_frame_training_free_space_consensus"
    motion = a.get("experiment") == "motion_budget_keyframes"
    volumetric = a.get("experiment") == "projective_tsdf_fusion"
    keys = (
        (
            "experiment",
            "format_version",
            "mapping_config",
            "tsdf_config",
            "render",
            "split",
            "saved_report_sha256",
            "saved_arrays_sha256",
            "odometry",
            "verified_canonical_input_sha256",
            "environment",
            "chunks",
            "pooled",
            "views",
            "paired_blocks",
        )
        if volumetric
        else (
            "experiment",
            "format_version",
            "rotation_radius_m",
            "mapping_config",
            "baseline_split",
            "selection",
            "selection_diagnostics",
            "saved_report_sha256",
            "saved_arrays_sha256",
            "odometry",
            "verified_canonical_input_sha256",
            "environment",
            "chunks",
            "pooled",
            "views",
            "paired_blocks",
        )
        if motion
        else (
            "experiment",
            "rule",
            "saved_report_sha256",
            "saved_arrays_sha256",
            "odometry",
            "mapping_config",
            "split",
            "unavailable_views",
            "verified_input_sha256",
            "script_sha256",
            "package_code_sha256",
            "environment",
            "pooled",
            "views",
        )
        if consensus
        else (
            "format_version",
            "experiment",
            "mapping_config",
            "offset",
            "render",
            "split",
            "source",
            "calibration",
            "verified_canonical_input_sha256",
            "submaps",
            "summaries",
            "views",
        )
        if heldout
        else (
            "config",
            "verified_canonical_input_sha256",
            "calibration",
            "frame_count",
            "keyframe_count",
            "segment_count",
            "submaps",
            "consistency",
        )
    )
    for key in keys:
        if a[key] != b[key]:
            raise ValueError(f"mapping reports differ: {key}")
    if consensus:
        clouds = lambda report: [
            {key: value for key, value in row.items() if key != "consensus_seconds"}
            for row in report["clouds"]
        ]
        if clouds(a) != clouds(b):
            raise ValueError("consensus cloud reports differ")
    elif a["environment"]["code_sha256"] != b["environment"]["code_sha256"]:
        raise ValueError("mapping package code fingerprints differ")
    if heldout:
        if a["odometry"] != b["odometry"]:
            raise ValueError("odometry identity differs")
    elif not consensus and not motion and not volumetric:
        for key in ("backend", "report_sha256", "trajectory_sha256"):
            if a["odometry"][key] != b["odometry"][key]:
                raise ValueError(f"odometry identity differs: {key}")
    provenance = (
        "inputs.json"
        if heldout or consensus or motion or volumetric
        else "keyframes.json"
    )
    if (first / provenance).read_bytes() != (second / provenance).read_bytes():
        raise ValueError("input provenance differs")
    archives = sorted(path.name for path in first.glob("*.npz"))
    if archives != sorted(path.name for path in second.glob("*.npz")):
        raise ValueError("map archive sets differ")
    arrays = 0
    for name in archives:
        with (
            np.load(first / name, allow_pickle=False) as left,
            np.load(second / name, allow_pickle=False) as right,
        ):
            if left.files != right.files:
                raise ValueError(f"map array names differ: {name}")
            for key in left.files:
                if not np.array_equal(
                    left[key], right[key], equal_nan=left[key].dtype.kind in "fc"
                ):
                    raise ValueError(f"map arrays differ: {name}/{key}")
                arrays += 1
    return {
        "first": str(first),
        "second": str(second),
        "archives": len(archives),
        "identical_arrays": arrays,
        "status": "exact_match",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.first, args.second), indent=2))


if __name__ == "__main__":
    main()
