"""Create an unchanged budget-Sparse baseline and immutable feature replay cache."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scene_recall.features import extract_rgbd_features
from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig
from scene_recall.run_metadata import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum-path", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    args.output_dir.mkdir(parents=True)
    cache = {}

    def extract(observation, calibration, sift, threshold):
        key = observation.frame_index
        if key not in cache:
            cache[key] = extract_rgbd_features(
                observation, calibration, sift, threshold
            )
            for array in (
                cache[key].pixels,
                cache[key].points_R,
                cache[key].descriptors,
            ):
                if array is not None:
                    array.setflags(write=False)
        return cache[key]

    with patch("scene_recall.odometry.sparse.extract_rgbd_features", extract):
        run_odometry(
            args.tum_path,
            args.output_dir / "budget",
            dataset="tum_rgbd",
            backend_name="sparse",
            config=SparseConfig(max_features=1500),
            start=args.start,
            stop=args.stop,
            rpe_intervals=(1, 5, 10, 20, 50, 100),
        )
    features = args.output_dir / "features.npz"
    np.savez_compressed(
        features,
        **{
            f"{i}_{key}": getattr(packet, key)
            for i, packet in cache.items()
            for key in packet.__dataclass_fields__
            if getattr(packet, key) is not None
        },
    )
    manifest = {
        "driver_sha256": sha256(Path(__file__)),
        "baseline_report_sha256": sha256(args.output_dir / "budget/report.json"),
        "baseline_trajectory_sha256": sha256(args.output_dir / "budget/trajectory.npz"),
        "features_sha256": sha256(features),
        "feature_frames": len(cache),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
