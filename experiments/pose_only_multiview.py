"""Refine a Sparse trajectory using training-only multiview RGB-D constraints."""

import argparse
from pathlib import Path

from scene_recall.multiview.pose_evaluation import evaluate_pose_maps
from scene_recall.multiview.pose_only import PoseOnlyConfig
from scene_recall.multiview.pose_runner import run_pose_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--window-step", type=int, default=4)
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--evaluate-maps", action="store_true")
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument("--map-chunks", type=int, nargs="+")
    args = parser.parse_args()
    if not args.maps_only:
        run_pose_experiment(
            args.odometry_run,
            args.output_dir,
            config=PoseOnlyConfig(
                window_size=args.window_size, window_step=args.window_step
            ),
            controls=not args.no_controls,
        )
    if args.evaluate_maps or args.maps_only:
        evaluate_pose_maps(
            args.output_dir,
            args.output_dir / "maps",
            variants=("anchored",)
            if args.no_controls
            else ("anchored", "frozen", "linear"),
            chunk_ids=args.map_chunks,
        )


if __name__ == "__main__":
    main()
