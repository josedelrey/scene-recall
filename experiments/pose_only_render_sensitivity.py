"""Post-hoc surface-disc radius sensitivity with frozen maps and target samples."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from scene_recall.mapping.evaluation import render_depth
from scene_recall.mapping.voxel import VoxelCloud
from scene_recall.multiview.pose_evaluation import score_maps
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256


def run(experiment, output):
    if output.exists():
        raise ValueError("choose a fresh diagnostic output")
    report = json.loads((experiment / "report.json").read_text(encoding="utf-8"))
    maps = json.loads((experiment / "maps/report.json").read_text(encoding="utf-8"))
    _, rows = load_trajectory(Path(report["baseline_run"]))
    K = np.array(report["calibration"]["K_D"], dtype=float)
    K[:2] /= report["mapping_config"]["pixel_stride"]
    hashes = {str(p): sha256(p) for p in (experiment / "maps").glob("*.npz")}
    pooled = {str(radius): {} for radius in (0.005, 0.01, 0.02)}
    for chunk in maps["chunks"]:
        index = chunk["submap_id"]
        clouds = {}
        for name in ("sparse", "anchored"):
            with np.load(experiment / f"maps/{name}_{index:03d}_voxel.npz") as archive:
                clouds[name] = VoxelCloud(
                    **{f.name: archive[f.name] for f in fields(VoxelCloud)}
                )
        with np.load(experiment / f"maps/targets_{index:03d}.npz") as targets:
            for i, frame in enumerate(targets["frame_indices"]):
                observed = targets["observed"][i]
                pose = (
                    np.array(chunk["maps"]["sparse"]["T_MW_reference_diagnostic"])
                    @ rows[frame].reference_pose
                )
                for radius in (0.005, 0.01, 0.02):
                    arrays = {
                        "observed": observed,
                        **{
                            f"voxel_{name}_reference": render_depth(
                                cloud, pose, K, observed.shape, radius
                            )
                            for name, cloud in clouds.items()
                        },
                    }
                    for key, array in arrays.items():
                        pooled[str(radius)].setdefault(key, []).append(array.ravel())
    summaries = {}
    for radius, arrays in pooled.items():
        values = {k: np.concatenate(v) for k, v in arrays.items()}
        observed = values.pop("observed")
        summaries[radius] = score_maps(observed, values)
    assert all(sha256(Path(path)) == digest for path, digest in hashes.items())
    output.mkdir(parents=True)
    result = {
        "experiment": "posthoc_radius_sensitivity",
        "source_pose_experiment_sha256": sha256(experiment / "report.json"),
        "fixed_maps_verified_unchanged": True,
        "summaries": summaries,
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for radius, summary in summaries.items():
        print(
            radius,
            {
                k: v["own"]["missing_penalized_capped_mae_m"]["0.2"]
                for k, v in summary["scores"].items()
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment, args.output_dir)


if __name__ == "__main__":
    main()
