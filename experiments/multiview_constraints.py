"""Reproducible fixed-setting ScanNet/TUM multiview constraint investigation.

Use one explicit source per invocation. Each source gets fresh Sparse odometry,
multiview tracks, adjacent-only and tighter-gate controls, and an exact repeat.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--sens-path", type=Path)
    sources.add_argument("--tum-path", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    source_flag = "--sens-path" if args.sens_path else "--tum-path"
    source = args.sens_path or args.tum_path
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/rgbd_odometry.py"),
            source_flag,
            str(source),
            "--start",
            str(args.start),
            *([] if args.stop is None else ["--stop", str(args.stop)]),
            "--backend",
            "sparse",
            "--output-dir",
            str(args.output_dir / "odometry"),
        ],
        check=True,
        env=env,
        cwd=root,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/rgbd_tracks.py"),
            "--odometry-run",
            str(args.output_dir / "odometry"),
            "--output-dir",
            str(args.output_dir / "tracks"),
            "--stride",
            str(args.stride),
            "--controls",
        ],
        check=True,
        env=env,
        cwd=root,
    )
    report = json.loads((args.output_dir / "tracks/report.json").read_text())
    print(
        json.dumps(
            {
                k: report["summary"][k]
                for k in (
                    "tracks",
                    "supported_tracks",
                    "lengths",
                    "span_frames",
                    "association_statuses",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
