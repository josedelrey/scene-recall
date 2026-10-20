"""Controlled offline joint RGB-D BA from saved temporal fixed-landmark packets."""

import argparse
from pathlib import Path

from scene_recall.multiview.joint_runner import run_joint_experiment
from scene_recall.multiview.pose_evaluation import evaluate_pose_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-unknown-static", action="store_true")
    parser.add_argument("--max-point-displacement", type=float, default=0.10)
    parser.add_argument("--evaluate-maps", action="store_true")
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument("--map-chunks", type=int, nargs="+")
    parser.add_argument("--map-output-dir", type=Path)
    args = parser.parse_args()
    if not args.maps_only:
        run_joint_experiment(
            args.fixed_run,
            args.output_dir,
            allow_unknown_static=args.allow_unknown_static,
            max_point_displacement_m=args.max_point_displacement,
        )
    if args.evaluate_maps or args.maps_only:
        evaluate_pose_maps(
            args.output_dir,
            args.map_output_dir or args.output_dir / "maps",
            variants=("clock", "timed", "joint"),
            chunk_ids=args.map_chunks,
        )


if __name__ == "__main__":
    main()
