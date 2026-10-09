"""Controlled RGB and depth outages on one ScanNet clip for all three backends.

These artificial outages test failure policy and recovery on the next adjacent
pair. They do not measure natural failure detection or sensor robustness.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

import scene_recall
from scene_recall.datasets.scannet import ScanNetReader
from scene_recall.odometry import (
    DepthICPBackend,
    HybridRGBDBackend,
    SparseRGBDBackend,
    track_observations,
)
from scene_recall.odometry.evaluation import evaluate_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens-path", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.start < 0:
        parser.error("start must be nonnegative")
    if args.output_dir.exists():
        parser.error("output directory exists, choose a fresh directory")
    source = args.sens_path.expanduser().resolve()
    cv2.setNumThreads(1)
    with ScanNetReader(source, start=args.start, stop=args.start + 20) as reader:
        calibration = reader.calibration
        observations = list(reader)
    if len(observations) != 20:
        parser.error("probe requires 20 input frames")
    # Black RGB at 5..7 preserves all depth. Missing depth at 12..14 preserves RGB.
    altered = [
        replace(
            obs,
            rgb=np.zeros_like(obs.rgb) if 5 <= obs.frame_index <= 7 else obs.rgb,
            depth=np.full_like(obs.depth, np.nan)
            if 12 <= obs.frame_index <= 14
            else obs.depth,
        )
        for obs in observations
    ]
    results = {}
    for name, factory in (
        ("sparse", SparseRGBDBackend),
        ("icp", DepthICPBackend),
        ("hybrid", HybridRGBDBackend),
    ):
        backend = factory()
        started = perf_counter()
        rows = list(track_observations(altered, calibration, backend))
        results[name] = {
            "config": asdict(backend.config),
            "tracking_seconds": perf_counter() - started,
            "evaluation": evaluate_trajectory(rows),
            "lost_target_indices": [
                row.frame_index for row in rows if row.status == "lost"
            ],
            "failure_reasons": dict(
                Counter(row.pair.reason for row in rows if row.status == "lost")
            ),
            "frames": [
                {
                    "frame_index": row.frame_index,
                    "segment_id": row.segment_id,
                    "status": row.status,
                    "reason": None if row.pair is None else row.pair.reason,
                    "diagnostics": {} if row.pair is None else row.pair.diagnostics,
                }
                for row in rows
            ],
        }
        print(
            f"{name}: lost targets {results[name]['lost_target_indices']}", flush=True
        )
    stat = source.stat()
    code_files = sorted(Path(scene_recall.__file__).parent.rglob("*.py")) + [
        Path(__file__)
    ]
    report = {
        "invocation": sys.argv,
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code_files
        },
        "source": {
            "path": str(source),
            "start": args.start,
            "stop": args.start + 20,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "perturbation": {
            "black_rgb_local_frames": [5, 6, 7],
            "missing_depth_local_frames": [12, 13, 14],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "runs": results,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
