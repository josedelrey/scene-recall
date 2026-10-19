"""Create a self-contained offline Rerun recording from saved experiments."""

import argparse
from pathlib import Path

from scene_recall.visualization.recording import export_recording
from scene_recall.visualization.rerun import DisplayConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--mapping-run", type=Path)
    inputs.add_argument(
        "--experiment-run", type=Path, help="saved held-out or fusion experiment"
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        help="original held-out baseline for a comparison experiment",
    )
    parser.add_argument(
        "--odometry-runs",
        type=Path,
        nargs="+",
        help="verified odometry cohort for an experiment",
    )
    parser.add_argument(
        "--odometry-run",
        type=Path,
        help="saved trajectory, also relocates the mapping input",
    )
    parser.add_argument(
        "--source-path", type=Path, help="relocate unchanged source images"
    )
    parser.add_argument("--output", type=Path, required=True, help="fresh .rrd file")
    parser.add_argument(
        "--images",
        choices=("keyframes", "evaluated", "all", "none"),
        help="default: mapping keyframes, all odometry frames, no raw images for experiments",
    )
    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="read only saved maps, without odometry or dataset",
    )
    parser.add_argument(
        "--submap",
        type=int,
        nargs="+",
        dest="submap_ids",
        help="include these submap IDs",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="display cap per cloud, default 200000",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="minimum map frame support for display",
    )
    args = parser.parse_args()
    try:
        if args.experiment_run:
            from scene_recall.visualization.experiments import export_experiment

            if args.mapping_run or args.odometry_run or args.maps_only:
                raise ValueError(
                    "experiment runs use --odometry-runs and exclude --maps-only"
                )
            export_experiment(
                args.output.expanduser(),
                experiment_run=args.experiment_run.expanduser(),
                baseline_run=args.baseline_run.expanduser()
                if args.baseline_run
                else None,
                odometry_runs=[p.expanduser() for p in args.odometry_runs or []],
                source_path=args.source_path.expanduser() if args.source_path else None,
                images=args.images,
                submap_ids=args.submap_ids,
                display=DisplayConfig(args.max_points, args.min_support),
            )
        else:
            if args.baseline_run or args.odometry_runs or args.images == "evaluated":
                raise ValueError(
                    "--baseline-run, --odometry-runs and --images evaluated require --experiment-run"
                )
            export_recording(
                args.output.expanduser(),
                mapping_run=args.mapping_run.expanduser() if args.mapping_run else None,
                odometry_run=args.odometry_run.expanduser()
                if args.odometry_run
                else None,
                source_path=args.source_path.expanduser() if args.source_path else None,
                images=args.images,
                maps_only=args.maps_only,
                submap_ids=args.submap_ids,
                display=DisplayConfig(args.max_points, args.min_support),
            )
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Recording saved to {args.output}")
    print(f"Open with: uv run --locked --extra visualization rerun {args.output}")


if __name__ == "__main__":
    main()
