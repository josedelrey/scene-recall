"""Test training-view free-space consensus without changing baseline fusion.

Only single-keyframe voxels are candidates. Remove one when at least two
distinct training views contradict it and contradictions outnumber agreements.
Use the existing consistency tolerance, native filtered depth and nearest pixel.
Occluded, invalid, off-image and behind-camera samples provide no evidence.
"""

import argparse
import hashlib
import json
import platform
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
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.geometry.transforms import is_rigid
from scene_recall.mapping.evaluation import heldout_metrics, render_depth
from scene_recall.mapping.heldout import check_cohort, prediction_poses
from scene_recall.mapping.pipeline import Keyframe, build_submap
from scene_recall.mapping.voxel import MappingConfig, VoxelCloud, sample_observation
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def free_space_consensus(cloud, training_views, K, tolerance_m):
    """Return a subset plus auditable per-voxel evidence, leaving inputs intact.

    Views are (distinct frame ID, camera-to-map pose, filtered native depth).
    No target observation or target pose enters this function. Frame support
    supplies a conservative lower bound on agreement with the fused centroid.
    This bounded offline pass costs O(single-frame voxels * training views).
    """
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("tolerance must be finite and positive")
    training_views = list(training_views)
    if len({index for index, _, _ in training_views}) != len(training_views):
        raise ValueError("training frame IDs must be distinct")
    candidate = np.flatnonzero(cloud.frame_support == 1)
    agreements = np.zeros(len(cloud.points), dtype=np.int64)
    contradictions = np.zeros(len(cloud.points), dtype=np.int64)
    for _, pose, depth in training_views:
        if not is_rigid(pose) or depth.ndim != 2:
            raise ValueError("training views require rigid poses and 2D depth")
        points = transform_points(cloud.points[candidate], np.linalg.inv(pose))
        pixels = project_points(points, K)
        h, w = depth.shape
        inside = (
            (points[:, 2] > 0)
            & np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= w - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= h - 1)
        )
        ids = candidate[inside]
        uv = np.floor(pixels[inside] + 0.5).astype(np.intp)
        measured = depth[uv[:, 1], uv[:, 0]]
        valid = np.isfinite(measured) & (measured > 0)
        residual = points[inside, 2] - measured
        agreements[ids] += valid & (np.abs(residual) <= tolerance_m)
        contradictions[ids] += valid & (residual < -tolerance_m)
    keep = ~(
        (cloud.frame_support == 1)
        & (contradictions >= 2)
        & (contradictions > np.maximum(agreements, cloud.frame_support))
    )
    subset = VoxelCloud(
        **{f.name: getattr(cloud, f.name)[keep] for f in fields(VoxelCloud)}
    )
    return subset, {
        "keep": keep,
        "agreements": agreements,
        "contradictions": contradictions,
    }


def coverage_score(observed, prediction, population):
    """Add bounded all-target loss with missing predictions charged the full cap."""
    metrics = heldout_metrics(observed, prediction, population)
    valid = np.isfinite(prediction) & (prediction > 0)
    error = np.where(valid, np.abs(prediction - observed), np.inf)
    metrics["missing_penalized_capped_mae_m"] = {
        str(cap): float(np.minimum(error[population], cap).mean())
        if population.any()
        else None
        for cap in (0.05, 0.2, 1.0)
    }
    return metrics


def compare_predictions(observed, baseline, proposal, anchor):
    """Expose exact common errors, fixed populations and lost good predictions."""
    valid = np.isfinite(observed) & (observed > 0)
    before = np.isfinite(baseline) & (baseline > 0)
    after = np.isfinite(proposal) & (proposal > 0)
    paired = anchor & before & after
    result = {
        "all_valid": {},
        "baseline_common": {},
        "paired_baseline_common": {},
    }
    for name, prediction in (("baseline", baseline), ("consensus", proposal)):
        for key, mask in (
            ("all_valid", valid),
            ("baseline_common", anchor),
            ("paired_baseline_common", paired),
        ):
            result[key][name] = coverage_score(observed, prediction, mask)
    lost = valid & before & ~after
    gained = valid & ~before & after
    result["transitions"] = {
        "lost_predictions": int(lost.sum()),
        "gained_predictions": int(gained.sum()),
        "changed_predictions": int(
            (valid & before & after & (np.abs(proposal - baseline) > 1e-12)).sum()
        ),
        "lost_previously_within_5cm": int(
            (lost & (np.abs(baseline - observed) <= 0.05)).sum()
        ),
        "baseline_error_on_lost": heldout_metrics(observed, baseline, lost),
    }
    return result


def run(saved, runs, output):
    """Verify saved inputs/maps/renders, then publish one fixed-rule experiment."""
    if output.exists():
        raise ValueError("choose a fresh output directory")
    report = json.loads((saved / "report.json").read_text())
    config = MappingConfig(**report["mapping_config"])
    if 0.01 not in report["render"]["radii_m"]:
        raise ValueError("experiment requires the baseline 10 mm render radius")
    loaded = {}
    for path in runs:
        meta, frames = load_trajectory(path)
        name = meta["backend"]
        if name in loaded or report["odometry"][name] != {
            "report_sha256": sha256(path / "report.json"),
            "trajectory_sha256": sha256(path / "trajectory.npz"),
        }:
            raise ValueError("duplicate or changed odometry input")
        loaded[name] = (meta, frames)
    if set(loaded) != set(report["odometry"]):
        raise ValueError("provide every backend from the saved cohort")
    split = check_cohort(loaded, config, report["offset"])
    if split != report["split"]:
        raise ValueError("training/target partition differs")
    meta, frames = next(iter(loaded.values()))
    digest, metadata = hashlib.sha256(), {}
    reader, _ = open_source(meta)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    cv2.setNumThreads(1)
    with (
        reader,
        tempfile.TemporaryDirectory(prefix=".consensus-", dir=output.parent) as temp,
    ):
        staging = Path(temp) / "result"
        staging.mkdir()
        validate_source(reader, meta)
        calibration = reader.calibration
        K = calibration.K_D.copy()
        K[:2] /= config.pixel_stride
        iterator = iter(verified_observations(reader, meta, frames, digest, metadata))
        cloud_rows, view_rows, buffers, input_hashes = [], [], {}, {}
        for chunk, (train, targets) in enumerate(
            zip(split["chunks"], split["target_indices_by_chunk"], strict=True)
        ):
            selected, samples = set(train + targets), {}
            for observation in iterator:
                if observation.frame_index in selected:
                    samples[observation.frame_index] = sample_observation(
                        observation, calibration, config
                    )
                if observation.frame_index == train[-1]:
                    break
            submaps = {
                name: build_submap(
                    [Keyframe(rows[i], samples[i]) for i in train], chunk, config
                )
                for name, (_, rows) in loaded.items()
            }
            filtered = {}
            for name, submap in submaps.items():
                for mode in ("estimated", "reference"):
                    filename = f"{name}_{chunk:03d}_{mode}.npz"
                    input_hashes[filename] = sha256(saved / filename)
                    with np.load(saved / filename, allow_pickle=False) as archive:
                        for f in fields(VoxelCloud):
                            np.testing.assert_array_equal(
                                getattr(getattr(submap, mode), f.name), archive[f.name]
                            )
                poses = {
                    i: np.eye(4)
                    if i == train[0]
                    else np.linalg.inv(submap.T_SM) @ loaded[name][1][i].T_SC_D
                    for i in train
                }
                sources = [(name, submap.estimated, poses)]
                if name == next(iter(submaps)) and submap.T_MW_reference is not None:
                    sources.append(
                        (
                            "gt",
                            submap.reference,
                            {
                                i: submap.T_MW_reference @ frames[i].reference_pose
                                for i in submap.report["reference_keyframe_indices"]
                            },
                        )
                    )
                for label, cloud, source_poses in sources:
                    tick = perf_counter()
                    subset, evidence = free_space_consensus(
                        cloud,
                        [(i, T, samples[i].depth) for i, T in source_poses.items()],
                        calibration.K_D,
                        config.consistency_tolerance_m,
                    )
                    seconds = perf_counter() - tick
                    filtered[label] = subset
                    np.savez_compressed(
                        staging / f"{label}_{chunk:03d}_evidence.npz",
                        keys=cloud.keys,
                        frame_support=cloud.frame_support,
                        training_frame_indices=np.array(list(source_poses)),
                        **evidence,
                    )
                    np.savez_compressed(
                        staging / f"{label}_{chunk:03d}_consensus.npz",
                        **{f.name: getattr(subset, f.name) for f in fields(VoxelCloud)},
                    )
                    cloud_rows.append(
                        {
                            "map": label,
                            "submap_id": chunk,
                            "segment_id": submap.segment_id,
                            "baseline_voxels": len(cloud.points),
                            "single_frame_voxels": int(
                                (cloud.frame_support == 1).sum()
                            ),
                            "removed_voxels": int((~evidence["keep"]).sum()),
                            "consensus_seconds": seconds,
                        }
                    )
            filename = f"views_{chunk:03d}_radius_0.01.npz"
            input_hashes[filename] = sha256(saved / filename)
            with np.load(saved / filename, allow_pickle=False) as archive:
                baseline = {key: archive[key] for key in archive.files}
            evaluated = baseline["evaluated_frame_indices"]
            expected = [
                row["frame_index"]
                for row in report["views"]["0.01"]
                if row["submap_id"] == chunk and row["status"] == "evaluated"
            ]
            np.testing.assert_array_equal(evaluated, expected)
            archive_arrays = {"evaluated_frame_indices": evaluated}
            for j, index in enumerate(evaluated):
                observed = samples[index].depth[
                    :: config.pixel_stride, :: config.pixel_stride
                ]
                np.testing.assert_array_equal(observed, baseline["observed"][j])
                poses = prediction_poses(
                    submaps, {name: rows[index] for name, (_, rows) in loaded.items()}
                )
                predictions = {}
                for name, (cloud, pose) in poses.items():
                    reproduced = render_depth(cloud, pose, K, observed.shape, 0.01)
                    np.testing.assert_allclose(
                        reproduced, baseline[name][j], rtol=0, atol=1e-12
                    )
                    label = name.split("_")[0]
                    predictions[name] = render_depth(
                        filtered[label], pose, K, observed.shape, 0.01
                    )
                    archive_arrays.setdefault(name, []).append(predictions[name])
                anchor = baseline["common_all_modes"][j]
                scores = {
                    name: compare_predictions(
                        observed, baseline[name][j], depth, anchor
                    )
                    for name, depth in predictions.items()
                }
                view_rows.append(
                    {
                        "frame_index": int(index),
                        "submap_id": chunk,
                        "segment_id": frames[index].segment_id,
                        "scores": scores,
                    }
                )
                for name, arr in {
                    "observed": observed,
                    "anchor": anchor,
                    **{"baseline_" + k: baseline[k][j] for k in predictions},
                    **predictions,
                }.items():
                    buffers.setdefault(name, []).append(arr.reshape(-1))
                if (j + 1) % 10 == 0:
                    print(
                        f"Chunk {chunk}: verified/rendered {j + 1}/{len(evaluated)}",
                        flush=True,
                    )
            np.savez_compressed(
                staging / filename,
                **{k: np.asarray(v) for k, v in archive_arrays.items()},
            )
            print(
                f"Completed chunk {chunk}: {len(evaluated)} evaluated views", flush=True
            )
        for _ in iterator:
            pass
        if digest.hexdigest() != report["verified_canonical_input_sha256"]:
            raise ValueError("canonical input differs from saved run")
        combined = {key: np.concatenate(values) for key, values in buffers.items()}
        scores = {
            name: compare_predictions(
                combined["observed"],
                combined["baseline_" + name],
                combined[name],
                combined["anchor"],
            )
            for name in combined
            if name not in ("observed", "anchor") and not name.startswith("baseline_")
        }
        result = {
            "experiment": "single_frame_training_free_space_consensus",
            "rule": {
                "minimum_contradictions": 2,
                "tolerance_m": config.consistency_tolerance_m,
            },
            "saved_report_sha256": sha256(saved / "report.json"),
            "saved_arrays_sha256": input_hashes,
            "odometry": report["odometry"],
            "mapping_config": report["mapping_config"],
            "split": split,
            "unavailable_views": [
                r for r in report["views"]["0.01"] if r["status"] != "evaluated"
            ],
            "verified_input_sha256": digest.hexdigest(),
            "script_sha256": sha256(Path(__file__)),
            "package_code_sha256": code_hash(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
            },
            "clouds": cloud_rows,
            "pooled": scores,
            "views": view_rows,
            "total_seconds": perf_counter() - started,
        }
        (staging / "report.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        (staging / "inputs.json").write_text(
            json.dumps(
                [metadata[i] for i in sorted(metadata)], indent=2, allow_nan=False
            )
            + "\n"
        )
        if output.exists():
            raise ValueError("output appeared during evaluation")
        staging.rename(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--odometry-run", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.saved, args.odometry_run, args.output_dir)
