"""Controlled RGB/depth outages on real inputs, without reference-based recovery."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.datasets.tum import TUMReader
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.local_map import LocalMapConfig, LocalMapRGBDBackend
from scene_recall.odometry.pipeline import track_observations
from scene_recall.odometry.recent_map import RecentMapConfig, RecentMapRGBDBackend
from scene_recall.odometry.sparse import SparseConfig, SparseRGBDBackend
from scene_recall.run_metadata import code_hash


def corrupt(observations, mode, start, stop):
    """Change only the chosen image measurement, preserving source references."""
    result = []
    for observation in observations:
        if start <= observation.frame_index < stop:
            if mode == "depth_gap":
                observation = replace(
                    observation, depth=np.full_like(observation.depth, np.nan)
                )
            elif mode == "rgb_gap":
                observation = replace(observation, rgb=np.zeros_like(observation.rgb))
            elif mode == "depth_bias":
                observation = replace(
                    observation, depth=observation.depth * np.float32(1.5)
                )
        result.append(observation)
    return result


def run_probe(source, output, stop=60, gap=(20, 30), include_recent=False):
    if output.exists() or output.is_symlink():
        raise ValueError("choose a fresh probe output")
    if not 0 < gap[0] < gap[1] < stop:
        raise ValueError("outage must lie strictly inside the sequence")
    cv2.setNumThreads(1)
    with TUMReader(source, stop=stop) as reader:
        observations = list(reader)
        calibration = reader.calibration
    if len(observations) != stop:
        raise ValueError("source is shorter than requested probe")
    digest = hashlib.sha256()
    for observation in observations:
        for array in (observation.rgb, observation.depth):
            digest.update(array.tobytes())
    sparse = SparseConfig(max_features=1500)
    local = LocalMapConfig(sparse=sparse)
    result = {
        "source": str(source.resolve()),
        "source_rgbd_sha256": digest.hexdigest(),
        "code_sha256": code_hash(),
        "stop": stop,
        "outage_range": list(gap),
        "sparse_config": asdict(sparse),
        "local_config": asdict(local),
        "cases": {},
    }
    arrays = {}
    backends = [
        ("sparse", SparseRGBDBackend, sparse),
        ("local", LocalMapRGBDBackend, local),
    ]
    if include_recent:
        recent = RecentMapConfig(sparse=sparse)
        result["recent_config"] = asdict(recent)
        backends.append(("recent", RecentMapRGBDBackend, recent))
    for mode in ("clean", "depth_gap", "rgb_gap", "depth_bias"):
        mutated = corrupt(observations, mode, *gap)
        result["cases"][mode] = {}
        for name, factory, config in backends:
            tick = perf_counter()
            rows = list(track_observations(mutated, calibration, factory(config)))
            elapsed = perf_counter() - tick
            arrays[f"{mode}_{name}_poses"] = np.array([r.T_SC_D for r in rows])
            arrays[f"{mode}_{name}_segments"] = np.array([r.segment_id for r in rows])
            result["cases"][mode][name] = {
                "tracking_seconds": elapsed,
                "evaluation": evaluate_trajectory(rows, (1, 5, 10, 20)),
                "lost_frames": [r.frame_index for r in rows if r.status == "lost"],
                "frames": [
                    {
                        "frame_index": r.frame_index,
                        "status": r.status,
                        "diagnostics": {} if r.pair is None else r.pair.diagnostics,
                    }
                    for r in rows
                ],
            }
            print(
                f"{mode} {name}: {result['cases'][mode][name]['lost_frames']}",
                flush=True,
            )
    output.mkdir(parents=True)
    np.savez_compressed(output / "trajectories.npz", **arrays)
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop", type=int, default=60)
    parser.add_argument("--gap", type=int, nargs=2, default=(20, 30))
    parser.add_argument("--include-recent", action="store_true")
    args = parser.parse_args()
    run_probe(
        args.tum_path, args.output_dir, args.stop, tuple(args.gap), args.include_recent
    )


if __name__ == "__main__":
    main()
