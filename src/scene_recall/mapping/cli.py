"""Reconstruct and diagnose RGB-D submaps from a saved odometry run."""

import argparse
from dataclasses import fields
from pathlib import Path

from scene_recall.mapping.runner import run_mapping
from scene_recall.mapping.voxel import MappingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-path", type=Path, help="relocate the unchanged source dataset"
    )
    parser.add_argument(
        "--pair-keyframe-offsets", type=int, nargs="+", default=[1, 3, 9]
    )
    for field in fields(MappingConfig):
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(field.default),
            default=field.default,
        )
    args = parser.parse_args()
    try:
        config = MappingConfig(
            **{field.name: getattr(args, field.name) for field in fields(MappingConfig)}
        )
        run_mapping(
            args.odometry_run.expanduser(),
            args.output_dir.expanduser(),
            config,
            args.source_path,
            tuple(args.pair_keyframe_offsets),
        )
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
