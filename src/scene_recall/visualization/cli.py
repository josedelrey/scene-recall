"""Create a self-contained offline Rerun recording from saved experiments."""

import argparse
from pathlib import Path

from scene_recall.visualization.recording import export_recording
from scene_recall.visualization.rerun import DisplayConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-run", type=Path)
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
        choices=("keyframes", "all", "none"),
        help="default: mapping keyframes, or all odometry frames",
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
        export_recording(
            args.output.expanduser(),
            mapping_run=args.mapping_run.expanduser() if args.mapping_run else None,
            odometry_run=args.odometry_run.expanduser() if args.odometry_run else None,
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
