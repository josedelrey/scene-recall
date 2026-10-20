"""Offline persistent-track batch RGB-D adjustment and frozen-cohort map checks."""

import argparse
from pathlib import Path

from scene_recall.multiview.batch_ba import BatchConfig
from scene_recall.multiview.batch_runner import run_batch_experiment
from scene_recall.multiview.pose_evaluation import evaluate_pose_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-model", choices=("native", "axial"), default="native")
    parser.add_argument("--min-track-views", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--evaluate-maps", action="store_true")
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument("--map-chunks", type=int, nargs="+")
    parser.add_argument("--map-output-dir", type=Path)
    args = parser.parse_args()
    if not args.maps_only:
        run_batch_experiment(
            args.fixed_run,
            args.output_dir,
            config=BatchConfig(
                depth_model=args.depth_model,
                min_track_views=args.min_track_views,
                max_iterations=args.max_iterations,
            ),
        )
    if args.evaluate_maps or args.maps_only:
        evaluate_pose_maps(
            args.output_dir,
            args.map_output_dir or args.output_dir / "maps",
            variants=("clock", "timed", "batch"),
            chunk_ids=args.map_chunks,
        )


if __name__ == "__main__":
    main()
