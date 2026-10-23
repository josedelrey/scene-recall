"""Fixed-cohort map scoring and exact reconstruction consistency audits."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from summarize_local_map_reliability import geometry_scores

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.mapping.evaluation import render_depth
from scene_recall.mapping.voxel import MappingConfig, VoxelMap, sample_observation
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def evaluate(run, output, policies, *, tolerance_m=0.0, reference_cohort="strict"):
    if reference_cohort not in ("strict", "available"):
        raise ValueError("unsupported reference cohort")
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("audit tolerance must be finite and nonnegative")
    if output.exists():
        raise ValueError("choose a fresh map evaluation directory")
    loaded = {policy: load_trajectory(run / policy) for policy in policies}
    report, rows = loaded[policies[0]]
    if len({r["source"]["canonical_input_sha256"] for r, _ in loaded.values()}) != 1:
        raise ValueError("map evaluation policies have different source inputs")
    config = MappingConfig(pixel_stride=8)
    count = min(len(rows), 240)
    train = list(range(0, count, 10))
    targets = list(range(5, count, 10))
    missing_training = [i for i in train if rows[i].reference_pose is None]
    if reference_cohort == "available":
        train = [i for i in train if i not in missing_training]
    output.mkdir(parents=True)
    protocol = {
        "phase": "frozen before reconstruction scoring",
        "scope": "end-of-stream online map version, fixed first 24 training views and interleaved reference targets",
        "train": train,
        "reference_cohort": reference_cohort,
        "missing_fixed_training_reference_indices": missing_training,
        "targets": targets,
        "mapping": asdict(config),
        "audit_position_tolerance_m": tolerance_m,
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "inputs": {
            policy: {
                name: sha256(run / policy / name)
                for name in ("report.json", "atlas.npz", "reconstruction.npz")
            }
            for policy in policies
        },
        "caveat": "fusion targets are withheld from fusion, their inputs may enter tracking and loop estimation, native reference poses are diagnostic cameras rather than surface truth",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    archives = {
        policy: dict(np.load(run / policy / "atlas.npz")) for policy in policies
    }
    reconstructions = {
        policy: dict(np.load(run / policy / "reconstruction.npz"))
        for policy in policies
    }
    needed = set(train + targets)
    for reconstruction in reconstructions.values():
        for key, values in reconstruction.items():
            if key.endswith("_frames"):
                needed.update(map(int, values))
    digest, metadata, samples, target_samples = hashlib.sha256(), {}, {}, {}
    reader, _ = open_source(report)
    with reader:
        validate_source(reader, report)
        for original in verified_observations(reader, report, rows, digest, metadata):
            frame = original.frame_index
            if frame not in needed:
                continue
            observation = original
            intervention = report["experiment"]["intervention"]
            if intervention and intervention[1] <= frame < intervention[2]:
                observation = (
                    replace(original, rgb=np.zeros_like(original.rgb))
                    if intervention[0] == "rgb"
                    else replace(original, depth=np.full_like(original.depth, np.nan))
                )
            samples[frame] = sample_observation(observation, reader.calibration, config)
            if frame in targets:
                target_samples[frame] = sample_observation(
                    original, reader.calibration, config
                )
        K = reader.calibration.K_D.copy()
    K[:2] /= config.pixel_stride
    consistency = {}
    for policy, reconstruction in reconstructions.items():
        indices = sorted(
            int(key.split("_")[0]) for key in reconstruction if key.endswith("_frames")
        )
        verified, exact, maximum = 0, 0, 0.0
        for index in indices:
            grid = VoxelMap(config.voxel_size_m, config.max_voxels)
            frames = reconstruction[f"{index}_frames"]
            for frame in frames:
                grid.integrate(
                    samples[int(frame)],
                    archives[policy]["T_component_depth"][frame],
                    int(frame),
                )
            chunk_exact = True
            for key, value in asdict(grid.cloud()).items():
                if not np.array_equal(value, reconstruction[f"{index}_{key}"]):
                    chunk_exact = False
                    saved = reconstruction[f"{index}_{key}"]
                    difference = (
                        float(np.max(np.abs(value - saved)))
                        if value.shape == saved.shape
                        else float("inf")
                    )
                    if (
                        key not in ("points", "position_std_m")
                        or not np.isfinite(difference)
                        or difference > tolerance_m
                    ):
                        raise ValueError(
                            f"{policy} chunk {index} differs from fresh raw fusion: {key}, maximum {difference}"
                        )
                    maximum = max(maximum, difference)
            if not all(
                archives[policy]["component_id"][frame]
                == reconstruction[f"{index}_component"]
                for frame in frames
            ):
                raise ValueError("reconstruction component metadata is stale")
            if reconstruction[f"{index}_version"] != archives[policy]["version"]:
                raise ValueError("reconstruction version metadata is stale")
            verified += 1
            exact += chunk_exact
        consistency[policy] = {
            "verified_chunks": verified,
            "exact_chunks": exact,
            "maximum_position_or_scatter_difference_m": maximum,
            "admitted_frames": sum(len(reconstruction[f"{i}_frames"]) for i in indices),
        }
    if (
        not train
        or rows[0].reference_pose is None
        or any(rows[i].reference_pose is None for i in train)
    ):
        result = {
            "status": "withheld",
            "reason": "missing fixed training reference",
            "consistency": consistency,
        }
    else:
        root = np.linalg.inv(rows[0].reference_pose)
        poses = {
            "reference": {
                i: root @ rows[i].reference_pose
                for i in train
                if len(samples[i].points)
            }
        }
        for policy, archive in archives.items():
            poses[policy] = {
                i: archive["T_component_depth"][i]
                for i in train
                if archive["component_id"][i] == archive["component_id"][0]
                and len(samples[i].points)
            }
        common_training = set.intersection(*(set(poses[policy]) for policy in policies))
        if any(set(poses[policy]) != common_training for policy in policies):
            for policy in policies:
                poses[policy + "_common_training"] = {
                    frame: poses[policy][frame] for frame in sorted(common_training)
                }
        clouds, map_stats = {}, {}
        for policy, cameras in poses.items():
            grid = VoxelMap(config.voxel_size_m, config.max_voxels)
            for frame, pose in cameras.items():
                grid.integrate(samples[frame], pose, frame)
            clouds[policy] = grid.cloud()
            map_stats[policy] = {
                "training_frames": list(cameras),
                "voxels": len(clouds[policy].points),
            }
        arrays = {"observed": [], "target_indices": []}
        for frame in targets:
            if rows[frame].reference_pose is None:
                continue
            observed = target_samples[frame].depth[
                :: config.pixel_stride, :: config.pixel_stride
            ]
            arrays["observed"].append(observed)
            arrays["target_indices"].append(frame)
            camera = root @ rows[frame].reference_pose
            for policy, cloud in clouds.items():
                arrays.setdefault(policy, []).append(
                    render_depth(cloud, camera, K, observed.shape, 0.01)
                )
        arrays = {key: np.array(value) for key, value in arrays.items()}
        np.savez_compressed(output / "predictions.npz", **arrays)
        result = {
            "status": "evaluated",
            "consistency": consistency,
            "maps": map_stats,
            "common_training_indices": sorted(common_training),
            "scores": geometry_scores(
                arrays["observed"],
                {
                    key: value
                    for key, value in arrays.items()
                    if key not in ("observed", "target_indices")
                },
            ),
        }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["local", "online"])
    parser.add_argument("--audit-tolerance-m", type=float, default=0.0)
    parser.add_argument(
        "--reference-cohort", choices=("strict", "available"), default="strict"
    )
    args = parser.parse_args()
    evaluate(
        args.run,
        args.output_dir,
        args.policies,
        tolerance_m=args.audit_tolerance_m,
        reference_cohort=args.reference_cohort,
    )


if __name__ == "__main__":
    main()
