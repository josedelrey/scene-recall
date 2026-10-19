"""Construct conservative multiview tracks on a saved estimated trajectory."""

import argparse
import sys
from pathlib import Path

from scene_recall.multiview import TrackConfig
from scene_recall.multiview.runner import run_multiview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=1500)
    parser.add_argument(
        "--pair-offsets", type=int, nargs="+", default=[1, 2, 5, 10, 20]
    )
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    report = run_multiview(
        args.odometry_run,
        args.output_dir,
        config=TrackConfig(
            max_features=args.max_features, pair_offsets=tuple(args.pair_offsets)
        ),
        stride=args.stride,
        source_override=args.source,
        controls=args.controls,
        invocation=sys.argv,
    )
    summary = report["summary"]
    print(
        f"Tracks: {summary['tracks']}, supported: {summary['supported_tracks']}, observations: {summary['track_observations']}"
    )


if __name__ == "__main__":
    main()
