"""Offline temporal RGB-D measurement preparation and fixed-landmark controls."""

import argparse
from pathlib import Path

from scene_recall.multiview.pose_evaluation import evaluate_pose_maps
from scene_recall.multiview.pose_only import PoseOnlyConfig
from scene_recall.multiview.temporal import TimingConfig
from scene_recall.multiview.temporal_runner import run_temporal_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-unknown-static", action="store_true")
    parser.add_argument("--motion-margin", type=float, default=2.0)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--window-step", type=int, default=4)
    parser.add_argument("--evaluate-maps", action="store_true")
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument("--map-chunks", type=int, nargs="+")
    parser.add_argument("--map-output-dir", type=Path)
    parser.add_argument(
        "--map-variants",
        nargs="+",
        choices=("clock", "static", "timed", "open"),
        default=("clock", "timed"),
    )
    args = parser.parse_args()
    if not args.maps_only:
        run_temporal_experiment(
            args.odometry_run,
            args.output_dir,
            config=PoseOnlyConfig(
                window_size=args.window_size, window_step=args.window_step
            ),
            timing_config=TimingConfig(
                unknown_static=args.allow_unknown_static,
                motion_margin_factor=args.motion_margin,
            ),
        )
    if args.evaluate_maps or args.maps_only:
        evaluate_pose_maps(
            args.output_dir,
            args.map_output_dir or args.output_dir / "maps",
            variants=tuple(args.map_variants),
            chunk_ids=args.map_chunks,
        )


if __name__ == "__main__":
    main()
