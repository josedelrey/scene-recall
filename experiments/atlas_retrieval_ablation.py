"""Replay identical local poses with fixed retrieval budgets, without reference access."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from slam_recovery import atlas_metrics

from scene_recall.data import Calibration
from scene_recall.features import RGBDFeatures
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.atlas import AtlasConfig, SubmapAtlas


def run(local_run, features, output):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    report, rows = load_trajectory(local_run)
    if report["config"].get("optimize", False):
        raise ValueError("this ablation requires unrevised causal local poses")
    calibration = Calibration(
        **{
            k: np.array(v) if k not in ("rgb_shape", "depth_shape") else tuple(v)
            for k, v in report["calibration"].items()
        }
    )
    configs = {str(k): AtlasConfig(candidates=k) for k in (4, 64)}
    protocol = {
        "phase": "frozen before retrieval ablation",
        "source": {
            "report": sha256(local_run / "report.json"),
            "trajectory": sha256(local_run / "trajectory.npz"),
            "features": sha256(features),
        },
        "budgets": {k: asdict(v) for k, v in configs.items()},
        "driver_sha256": sha256(Path(__file__)),
        "package_sha256": code_hash(),
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    cv2.setNumThreads(1)
    result = {}
    with np.load(features, allow_pickle=False) as cache:
        for name, config in configs.items():
            atlas = SubmapAtlas(calibration, config)
            started = perf_counter()
            for i, row in enumerate(rows):
                packet = RGBDFeatures(
                    cache[f"{i}_pixels"],
                    cache.get(f"{i}_descriptors"),
                    cache[f"{i}_points_R"],
                )
                atlas.observe(
                    i,
                    row.segment_id,
                    row.T_SC_D,
                    packet,
                    boundary=row.status != "tracked",
                )
            result[name] = {
                "wall_seconds": perf_counter() - started,
                **atlas_metrics(atlas, [r.reference_pose for r in rows]),
                "event_statuses": dict(Counter(e["status"] for e in atlas.events)),
                "rejection_reasons": dict(
                    Counter(
                        e["verification"].get("reason")
                        for e in atlas.events
                        if e["status"] == "registration_rejected"
                    )
                ),
            }
            (output / f"events_{name}.json").write_text(
                json.dumps(atlas.events, indent=2, allow_nan=False) + "\n"
            )
            print("Completed retrieval budget", name, flush=True)
    (output / "comparison.json").write_text(
        json.dumps({"retrieval_budgets": result}, indent=2, allow_nan=False) + "\n"
    )
    if protocol["package_sha256"] != code_hash() or protocol["driver_sha256"] != sha256(
        Path(__file__)
    ):
        raise RuntimeError("source changed during ablation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.local_run, args.features, args.output_dir)
