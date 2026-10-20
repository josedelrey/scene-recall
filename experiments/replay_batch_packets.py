"""Rebuild batch factors and poses in a new process without reading RGB-D media."""

import argparse
import json
from pathlib import Path

from scene_recall.multiview.batch_runner import replay_batch_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("replay output already exists")
    report = {str(run): replay_batch_experiment(run) for run in args.runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
