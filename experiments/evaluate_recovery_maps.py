"""Fixed-view root-map evaluation without inventing poses for disconnected maps.

The first 24 cadence-10 training candidates and interleaved targets are chosen
before localization masks or reference availability are inspected. Atlas and
local maps admit only candidates connected to frame zero. All predictions use
the same reference target poses, keeping rendering pose error out of fusion.
This measures accessible root-map reconstruction after recovery. It differs
from the existing per-segment reconstruction cohort and never relabels it.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from summarize_local_map_reliability import geometry_scores

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.mapping.evaluation import render_depth
from scene_recall.mapping.mesh import render_mesh_depth
from scene_recall.mapping.tsdf import TSDFConfig, TSDFView, fuse_tsdf
from scene_recall.mapping.voxel import MappingConfig, VoxelMap, sample_observation
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, peak_rss, sha256


def evaluate(run, output, policies):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    config = MappingConfig(pixel_stride=8)
    tsdf = TSDFConfig()
    loaded = {name: load_trajectory(run / name) for name in policies}
    first, rows = loaded[policies[0]]
    count = min(len(rows), config.keyframe_stride * config.max_keyframes_per_submap)
    train = list(range(0, count, config.keyframe_stride))
    targets = list(range(config.keyframe_stride // 2, count, config.keyframe_stride))
    output.mkdir(parents=True)
    protocol = {
        "phase": "frozen before reconstruction",
        "scope": "first root component, final atlas versions, reference target poses",
        "mapping": asdict(config),
        "tsdf": asdict(tsdf),
        "train": train,
        "targets": targets,
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "inputs": {
            n: {
                p: sha256(run / n / p)
                for p in ("report.json", "trajectory.npz", "atlas.npz")
            }
            for n in policies
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if any(rows[i].reference_pose is None for i in train):
        result = {"status": "withheld", "reason": "missing fixed training references"}
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    reader, _ = open_source(first)
    digest, metadata, samples = hashlib.sha256(), {}, {}
    started = perf_counter()
    with reader:
        validate_source(reader, first)
        for observation in verified_observations(reader, first, rows, digest, metadata):
            if observation.frame_index in set(train + targets):
                samples[observation.frame_index] = sample_observation(
                    observation, reader.calibration, config
                )
        K = reader.calibration.K_D.copy()
    if digest.hexdigest() != first["source"]["canonical_input_sha256"]:
        raise ValueError("canonical replay identity changed")
    K[:2] /= config.pixel_stride
    reference_to_root = np.linalg.inv(rows[0].reference_pose)
    poses = {"gt": {i: reference_to_root @ rows[i].reference_pose for i in train}}
    for name, (report, current) in loaded.items():
        if (
            report["source"]["canonical_input_sha256"]
            != first["source"]["canonical_input_sha256"]
        ):
            raise ValueError("reconstruction input populations differ")
        poses[name + "_local"] = {
            i: current[i].T_SC_D
            for i in train
            if current[i].segment_id == current[0].segment_id
        }
        with np.load(run / name / "atlas.npz") as atlas:
            poses[name + "_atlas"] = {
                i: atlas["T_component_depth"][i].copy()
                for i in train
                if atlas["component_id"][i] == atlas["component_id"][0]
            }
    clouds, meshes, map_stats, mesh_arrays = {}, {}, {}, {}
    for name, cameras in poses.items():
        current = VoxelMap(config.voxel_size_m, config.max_voxels)
        views = []
        for i, pose in cameras.items():
            current.integrate(samples[i], pose, i)
            views.append(
                TSDFView(
                    i,
                    pose,
                    samples[i].depth[:: config.pixel_stride, :: config.pixel_stride],
                )
            )
        clouds[name] = current.cloud()
        try:
            mesh = fuse_tsdf(views, K, tsdf).mesh()
        except ValueError as error:
            if "capacity" not in str(error) and "budget" not in str(error):
                raise
            raise ValueError(
                "fixed TSDF reconstruction exceeded resource limits"
            ) from error
        meshes[name] = mesh
        mesh_arrays[name + "_vertices"] = mesh.vertices
        mesh_arrays[name + "_triangles"] = mesh.triangles
        map_stats[name] = {
            "training_indices": list(cameras),
            "voxels": len(clouds[name].points),
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
        }
    arrays = {"observed": [], "evaluated_frame_indices": []}
    for i in targets:
        if rows[i].reference_pose is None:
            continue
        arrays["evaluated_frame_indices"].append(i)
        observed = samples[i].depth[:: config.pixel_stride, :: config.pixel_stride]
        arrays["observed"].append(observed)
        pose = reference_to_root @ rows[i].reference_pose
        for name in poses:
            arrays.setdefault(name + "_surface", []).append(
                render_depth(clouds[name], pose, K, observed.shape, 0.01)
            )
            arrays.setdefault(name + "_mesh", []).append(
                render_mesh_depth(meshes[name], pose, K, observed.shape)
            )
    arrays = {k: np.array(v) for k, v in arrays.items()}
    np.savez_compressed(output / "predictions.npz", **arrays)
    np.savez_compressed(output / "meshes.npz", **mesh_arrays)
    result = {
        "status": "evaluated",
        "protocol_sha256": sha256(output / "protocol.json"),
        "maps": map_stats,
        "scores": geometry_scores(
            arrays["observed"],
            {
                k: v
                for k, v in arrays.items()
                if k not in ("observed", "evaluated_frame_indices")
            },
        ),
        "wall_seconds": perf_counter() - started,
        "peak_rss_mib": peak_rss(),
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if protocol["package_sha256"] != code_hash() or protocol["driver_sha256"] != sha256(
        Path(__file__)
    ):
        raise RuntimeError("source changed during reconstruction")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["control", "low_pnp", "rgbd"])
    args = parser.parse_args()
    evaluate(args.run, args.output_dir, args.policies)


if __name__ == "__main__":
    main()
