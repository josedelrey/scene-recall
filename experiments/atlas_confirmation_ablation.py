"""Isolate hypothesis-directed confirmation on identical local poses and bank entries."""

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


def run(local_run, features, output, budgets):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    report, rows = load_trajectory(local_run)
    if report["config"].get("optimize", False):
        raise ValueError("confirmation replay requires unrevised causal local poses")
    calibration = Calibration(
        **{
            k: tuple(v) if k.endswith("shape") else np.array(v)
            for k, v in report["calibration"].items()
        }
    )
    configs = {
        "regular4": AtlasConfig(),
        **{
            f"confirm{k}": AtlasConfig(candidates=k, confirmation_query_frames=10)
            for k in budgets
        },
    }
    protocol = {
        "phase": "frozen before confirmation ablation",
        "local_report_sha256": sha256(local_run / "report.json"),
        "local_trajectory_sha256": sha256(local_run / "trajectory.npz"),
        "baseline_atlas_sha256": sha256(local_run / "atlas.npz"),
        "features_sha256": sha256(features),
        "configs": {k: asdict(v) for k, v in configs.items()},
        "outage": report["experiment"].get("outage"),
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    cv2.setNumThreads(1)
    result, retained = {}, None
    with np.load(features, allow_pickle=False) as cache:
        for name, config in configs.items():
            atlas = SubmapAtlas(calibration, config)
            started = perf_counter()
            banks, causal = [], []
            for i, row in enumerate(rows):
                outage = protocol["outage"]
                if outage and outage[0] <= i < outage[1]:
                    packet = RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))
                else:
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
                banks.append(sorted(atlas.entries))
                causal.append(atlas.components[row.segment_id] == 0)
            poses = np.array([atlas.pose(i) for i in range(len(rows))])
            components = np.array([atlas.components[r.segment_id] for r in rows])
            if name == "regular4":
                retained = banks
                with np.load(local_run / "atlas.npz") as baseline:
                    if not np.array_equal(
                        poses, baseline["T_component_depth"]
                    ) or not np.array_equal(components, baseline["component_id"]):
                        raise ValueError(
                            "unchanged atlas does not reproduce the original scientific arrays"
                        )
            elif retained != banks:
                raise ValueError(
                    "confirmation changed the keyframe retention population"
                )
            np.savez_compressed(
                output / f"{name}.npz",
                T_component_depth=poses,
                component_id=components,
                source=np.array([e["source"] for e in atlas.forest], dtype=np.int64),
                target=np.array([e["target"] for e in atlas.forest], dtype=np.int64),
                T_target_source=np.array(
                    [e["T_target_source"] for e in atlas.forest]
                ).reshape(-1, 4, 4),
            )
            result[name] = {
                "wall_seconds": perf_counter() - started,
                **atlas_metrics(atlas, [r.reference_pose for r in rows]),
                "event_statuses": dict(Counter(e["status"] for e in atlas.events)),
                "causal_root_frames": sum(causal),
                "unchanged_local_poses": True,
                "unchanged_retained_keyframes": True,
            }
            (output / f"events_{name}.json").write_text(
                json.dumps(atlas.events, indent=2, allow_nan=False) + "\n"
            )
            print("Completed", name, flush=True)
    (output / "comparison.json").write_text(
        json.dumps({"confirmation_policies": result}, indent=2, allow_nan=False) + "\n"
    )
    if protocol["package_sha256"] != code_hash() or protocol["driver_sha256"] != sha256(
        Path(__file__)
    ):
        raise RuntimeError("source changed during confirmation ablation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 64])
    args = parser.parse_args()
    run(args.local_run, args.features, args.output_dir, args.budgets)
