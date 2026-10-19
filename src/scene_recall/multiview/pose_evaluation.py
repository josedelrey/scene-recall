"""Frozen-population reconstruction checks for experimental pose trajectories."""

import hashlib
import json
import tempfile
from dataclasses import fields
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.geometry.transforms import is_rigid
from scene_recall.mapping.evaluation import heldout_metrics, render_depth
from scene_recall.mapping.heldout import check_cohort
from scene_recall.mapping.mesh import render_mesh_depth
from scene_recall.mapping.pipeline import Keyframe, build_submap
from scene_recall.mapping.tsdf import TSDFConfig, TSDFView, fuse_tsdf
from scene_recall.mapping.voxel import MappingConfig, sample_observation
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def score_maps(observed, predictions):
    """Keep all valid targets, baseline populations and exact paired intersections."""
    valid = np.isfinite(observed) & (observed > 0)

    def metric(predicted, mask):
        result = heldout_metrics(observed, predicted, mask)
        residual = np.where(
            np.isfinite(predicted) & (predicted > 0),
            np.abs(predicted - observed),
            np.inf,
        )
        result["missing_penalized_capped_mae_m"] = {
            str(cap): float(np.minimum(residual[mask], cap).mean())
            if mask.any()
            else None
            for cap in (0.05, 0.2, 1.0)
        }
        return result

    scores, pairs = {}, {}
    for name, predicted in predictions.items():
        method, variant, mode = name.split("_")
        baseline = predictions[f"{method}_sparse_{mode}"]
        base_mask = valid & np.isfinite(baseline) & (baseline > 0)
        scores[name] = {
            "own": metric(predicted, valid),
            "baseline_population": metric(predicted, base_mask),
        }
        if variant != "sparse":
            common = base_mask & np.isfinite(predicted) & (predicted > 0)
            pairs[name] = {
                "sparse": metric(baseline, common),
                "refined": metric(predicted, common),
            }
    return {"scores": scores, "pairs": pairs}


def evaluate_pose_maps(
    experiment: Path,
    output: Path,
    *,
    variants=("anchored", "frozen", "linear"),
    chunk_ids=None,
):
    """Fuse identical training RGB-D samples with surface voxels and TSDF meshes.

    Ground truth supplies target render poses only in this post-optimization
    module. It neither adjusts trajectories nor chooses training observations.
    Native target samples never enter fusion. Estimated-target renders are a
    separate diagnostic and cannot replace the reference-target result.
    """
    if output.exists() or output.is_symlink():
        raise ValueError("map evaluation output already exists")
    experiment_report = json.loads(
        (experiment / "report.json").read_text(encoding="utf-8")
    )
    for filename, key in (
        ("training_measurements.npz", "training_measurements_sha256"),
        ("native_measurements.npz", "native_measurements_sha256"),
    ):
        if (
            key in experiment_report
            and sha256(experiment / filename) != experiment_report[key]
        ):
            raise ValueError("optimization measurement archive changed")
    baseline_run = Path(experiment_report["baseline_run"])
    if (
        sha256(baseline_run / "report.json")
        != experiment_report["baseline_report_sha256"]
        or sha256(baseline_run / "trajectory.npz")
        != experiment_report["baseline_trajectory_sha256"]
    ):
        raise ValueError("baseline changed after optimization")
    for name in variants:
        identity = experiment_report["variants"][name]
        if (
            "factor_archive_sha256" in identity
            and sha256(experiment / f"{name}_factors.npz")
            != identity["factor_archive_sha256"]
        ):
            raise ValueError("optimization factor archive changed")
        if (
            sha256(experiment / name / "trajectory.npz")
            != identity["trajectory_sha256"]
            or sha256(experiment / name / "report.json") != identity["report_sha256"]
        ):
            raise ValueError("refined trajectory changed after optimization")
    config = MappingConfig(**experiment_report["mapping_config"])
    tsdf_config = TSDFConfig(
        voxel_size_m=config.voxel_size_m, truncation_m=3 * config.voxel_size_m
    )
    loaded = {"sparse": load_trajectory(Path(experiment_report["baseline_run"]))}
    loaded.update({name: load_trajectory(experiment / name) for name in variants})
    split = check_cohort(loaded, config, experiment_report["offset"])
    if split != experiment_report["split"]:
        raise ValueError("training or held-out populations differ from optimization")
    if chunk_ids is not None:
        if (
            not chunk_ids
            or len(set(chunk_ids)) != len(chunk_ids)
            or any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or not 0 <= i < len(split["chunks"])
                for i in chunk_ids
            )
        ):
            raise ValueError("map chunks must be unique valid chunk IDs")
        chunk_ids = sorted(chunk_ids)
    training = set(split["train_indices"])
    with np.load(
        experiment / "training_measurements.npz", allow_pickle=False
    ) as measurements:
        if set(map(int, measurements["frame_index"])) != training:
            raise ValueError(
                "optimizer measurements differ from map training population"
            )
        if any(int(f) not in training for f in measurements["observation_keys"][:, 0]):
            raise ValueError("held-out feature entered optimization")
    baseline, baseline_rows = loaded["sparse"]
    reader, _ = open_source(baseline)
    started = perf_counter()
    digest, metadata = hashlib.sha256(), {}
    cv2.setNumThreads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        reader,
        tempfile.TemporaryDirectory(prefix=".pose-maps-", dir=output.parent) as temp,
    ):
        staging = Path(temp) / "result"
        staging.mkdir()
        validate_source(reader, baseline)
        K = reader.calibration.K_D.copy()
        K[:2] /= config.pixel_stride
        iterator = iter(
            verified_observations(reader, baseline, baseline_rows, digest, metadata)
        )
        chunks, view_rows, pooled = [], [], {}
        for chunk_id, (train, targets) in enumerate(
            zip(split["chunks"], split["target_indices_by_chunk"], strict=True)
        ):
            if chunk_ids is not None and chunk_id not in chunk_ids:
                continue
            selected = set(train + targets)
            samples = {}
            for observation in iterator:
                if observation.frame_index in selected:
                    samples[observation.frame_index] = sample_observation(
                        observation, reader.calibration, config
                    )
                if observation.frame_index == train[-1]:
                    break
            submaps, meshes, cost, voxel_arrays = {}, {}, {}, {}
            for name, (_, rows) in loaded.items():
                tick = perf_counter()
                submap = build_submap(
                    [Keyframe(rows[i], samples[i]) for i in train], chunk_id, config
                )
                submaps[name] = submap
                voxel_arrays[name] = submap.estimated
                cost[name] = {"surface_voxel_seconds": perf_counter() - tick}
                np.savez_compressed(
                    staging / f"{name}_{chunk_id:03d}_voxel.npz",
                    **{
                        f.name: getattr(submap.estimated, f.name)
                        for f in fields(submap.estimated)
                    },
                )
                views = [
                    TSDFView(
                        i,
                        np.eye(4)
                        if i == train[0]
                        else np.linalg.inv(submap.T_SM) @ rows[i].T_SC_D,
                        samples[i].depth[
                            :: config.pixel_stride, :: config.pixel_stride
                        ],
                    )
                    for i in train
                ]
                tick = perf_counter()
                volume = fuse_tsdf(views, K, tsdf_config)
                mesh = volume.mesh()
                meshes[name] = mesh
                cost[name]["tsdf_fuse_mesh_seconds"] = perf_counter() - tick
                np.savez_compressed(
                    staging / f"{name}_{chunk_id:03d}_tsdf.npz",
                    keys=volume.keys,
                    signed_distance_m=volume.signed_distance_m,
                    frame_support=volume.frame_support,
                    vertices=mesh.vertices,
                    triangles=mesh.triangles,
                    training_frame_indices=np.array(train),
                )
                del volume
            paired = all(
                m.report["comparison_status"] == "paired" for m in submaps.values()
            )
            buffers = {}
            for index in targets:
                row = baseline_rows[index]
                if (
                    not paired
                    or row.reference_pose is None
                    or not is_rigid(row.reference_pose)
                ):
                    view_rows.append(
                        {
                            "frame_index": index,
                            "submap_id": chunk_id,
                            "status": "unavailable_reference",
                        }
                    )
                    continue
                observed = samples[index].depth[
                    :: config.pixel_stride, :: config.pixel_stride
                ]
                predictions = {}
                tick = perf_counter()
                for name, submap in submaps.items():
                    target_row = loaded[name][1][index]
                    poses = {
                        "reference": submap.T_MW_reference @ row.reference_pose,
                        "estimated": np.linalg.inv(submap.T_SM) @ target_row.T_SC_D,
                    }
                    for mode, pose in poses.items():
                        predictions[f"voxel_{name}_{mode}"] = render_depth(
                            voxel_arrays[name], pose, K, observed.shape, 0.01
                        )
                        predictions[f"tsdf_{name}_{mode}"] = render_mesh_depth(
                            meshes[name], pose, K, observed.shape
                        )
                cost.setdefault("render_seconds", 0.0)
                cost["render_seconds"] += perf_counter() - tick
                metrics = score_maps(observed, predictions)
                view_rows.append(
                    {
                        "frame_index": index,
                        "submap_id": chunk_id,
                        "segment_id": row.segment_id,
                        "status": "evaluated",
                        **metrics,
                    }
                )
                for key, array in {"observed": observed, **predictions}.items():
                    buffers.setdefault(key, []).append(array)
                    pooled.setdefault(key, []).append(array.ravel())
            np.savez_compressed(
                staging / f"targets_{chunk_id:03d}.npz",
                frame_indices=np.array(
                    [
                        r["frame_index"]
                        for r in view_rows
                        if r["submap_id"] == chunk_id and r["status"] == "evaluated"
                    ],
                    dtype=int,
                ),
                **{k: np.stack(v) for k, v in buffers.items()},
            )
            chunks.append(
                {
                    "submap_id": chunk_id,
                    "training_indices": train,
                    "target_indices": targets,
                    "maps": {name: m.report for name, m in submaps.items()},
                    "cost": cost,
                }
            )
            print(
                f"Evaluated map chunk {chunk_id}: {len(train)} training, {len(targets)} targets",
                flush=True,
            )
        for _ in iterator:
            pass
        summary = {}
        if pooled:
            combined = {k: np.concatenate(v) for k, v in pooled.items()}
            observed = combined.pop("observed")
            summary = score_maps(observed, combined)
        report = {
            "format_version": 1,
            "experiment": "pose_only_frozen_population_maps",
            "pose_experiment_sha256": sha256(experiment / "report.json"),
            "mapping_config": experiment_report["mapping_config"],
            "tsdf_config": {
                f.name: getattr(tsdf_config, f.name) for f in fields(tsdf_config)
            },
            "split": split,
            "evaluated_chunk_ids": chunk_ids,
            "chunks": chunks,
            "views": view_rows,
            "pooled": summary,
            "canonical_inputs_verified": digest.hexdigest(),
            "total_seconds": perf_counter() - started,
            "code_sha256": code_hash(),
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if output.exists() or output.is_symlink():
            raise ValueError("output appeared during map evaluation")
        staging.rename(output)
    return report
