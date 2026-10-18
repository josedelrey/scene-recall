"""Verify repeated mapping geometry and diagnostics, excluding runtime fields."""

import argparse
import json
from pathlib import Path

import numpy as np


def verify(first: Path, second: Path) -> dict:
    a, b = [json.loads((path / "report.json").read_text()) for path in (first, second)]
    for key in (
        "config",
        "verified_canonical_input_sha256",
        "calibration",
        "frame_count",
        "keyframe_count",
        "segment_count",
        "submaps",
        "consistency",
    ):
        if a[key] != b[key]:
            raise ValueError(f"mapping reports differ: {key}")
    if a["environment"]["code_sha256"] != b["environment"]["code_sha256"]:
        raise ValueError("mapping package code fingerprints differ")
    for key in ("backend", "report_sha256", "trajectory_sha256"):
        if a["odometry"][key] != b["odometry"][key]:
            raise ValueError(f"odometry identity differs: {key}")
    if (first / "keyframes.json").read_bytes() != (
        second / "keyframes.json"
    ).read_bytes():
        raise ValueError("keyframe provenance differs")
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
                if not np.array_equal(left[key], right[key]):
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
