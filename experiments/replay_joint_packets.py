"""Independently re-solve saved joint BA packets without opening source media."""

import argparse
import json
from pathlib import Path

from scene_recall.multiview.joint_runner import replay_joint_experiment
from scene_recall.run_metadata import code_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("replay report already exists")
    report = {
        "code_sha256": code_hash(Path(__file__)),
        "cases": [replay_joint_experiment(run) for run in args.runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
