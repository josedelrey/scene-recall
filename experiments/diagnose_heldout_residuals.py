"""Controlled residual probes for one saved, paired held-out submap.

Keep production fusion, calibration and rendering unchanged. The saved common
mask is an anchor population, with missing predictions reported explicitly.
Temporal subsets and unvoxelized samples are diagnostic counterfactuals only.
"""

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.geometry.camera import (
    backproject_depth,
    reject_depth_edges,
    transform_points,
)
from scene_recall.mapping.evaluation import heldout_metrics, render_depth
from scene_recall.mapping.heldout import check_cohort
from scene_recall.mapping.voxel import (
    MappingConfig,
    SurfaceSamples,
    VoxelCloud,
    VoxelMap,
    sample_observation,
)
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def point_cloud(points):
    """Wrap raw samples for the shared renderer without voxel averaging."""
    n = len(points)
    return VoxelCloud(
        np.zeros((n, 3), dtype=np.int64),
        points,
        np.zeros((n, 3), dtype=np.uint8),
        np.ones(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        np.zeros(n),
    )


def plane_depth(pose, K, shape, plane_z=2.0):
    """Intersect camera rays with the exact map-frame plane z = plane_z."""
    rays = backproject_depth(np.ones(shape), K) @ pose[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (plane_z - pose[2, 3]) / rays[..., 2]
    return np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)


def synthetic_probe(saved, runs, output):
    """Hold real camera motion fixed, replacing the scene with one perfect plane."""
    report = json.loads((saved / "report.json").read_text())
    config = MappingConfig(**report["mapping_config"])
    _, frames = load_trajectory(runs[0])
    train = report["split"]["train_indices"]
    with np.load(saved / "views_000_radius_0.01.npz") as archive:
        targets = archive["evaluated_frame_indices"]
    align = np.linalg.inv(frames[train[0]].reference_pose)
    poses = {i: align @ frames[i].reference_pose for i in set(train) | set(targets)}
    K = np.array(report["calibration"]["K_D"])
    depths = {}
    for i, pose in poses.items():
        depth = plane_depth(pose, K, tuple(report["calibration"]["depth_shape"]))
        depth = reject_depth_edges(depth, config.depth_edge_threshold_m)
        depths[i] = np.where(
            (depth >= config.min_depth_m) & (depth <= config.max_depth_m), depth, np.nan
        )[:: config.pixel_stride, :: config.pixel_stride]
    K[:2] /= config.pixel_stride
    observed = np.stack([depths[i] for i in targets])
    mapping, raw = VoxelMap(config.voxel_size_m, config.max_voxels), []
    for i in train:
        points = backproject_depth(depths[i], K).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        samples = SurfaceSamples(
            points,
            np.zeros((len(points), 3), dtype=np.uint8),
            np.zeros(len(points), dtype=bool),
            depths[i],
        )
        mapping.integrate(samples, poses[i], i)
        raw.append(transform_points(points, poses[i]))
    predictions = {}
    for name, cloud in (
        ("raw", point_cloud(np.concatenate(raw))),
        ("voxel", mapping.cloud()),
    ):
        predictions[name] = np.stack(
            [
                render_depth(cloud, poses[i], K, observed.shape[1:], 0.01)
                for i in targets
            ]
        )
    summary = {
        "scene": "noiseless infinite map-frame plane z=2m, real GT camera motion",
        "own": {
            name: heldout_metrics(observed, array, np.isfinite(observed))
            for name, array in predictions.items()
        },
        "paired": paired_summary(observed, predictions, np.isfinite(observed)),
    }
    (output / "synthetic.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


def minimum_layers(layers):
    """Nearest positive prediction, with unobserved pixels kept unavailable."""
    return np.fmin.reduce(layers, axis=0)


def summarize(observed, predicted, anchor):
    return {
        "all_valid": heldout_metrics(observed, predicted, np.isfinite(observed)),
        "saved_common": heldout_metrics(observed, predicted, anchor),
    }


def paired_summary(observed, predictions, population):
    """Compare finite predictions on one exact intersection without error gating."""
    common = population.copy()
    for array in predictions.values():
        common &= np.isfinite(array) & (array > 0)
    return {
        name: heldout_metrics(observed, array, common)
        for name, array in predictions.items()
    }


def analyze(saved, runs, output):
    """Summarize retained controls, including target-independent strata."""
    provenance = json.loads((output / "provenance.json").read_text())
    if provenance["saved_report_sha256"] != sha256(saved / "report.json"):
        raise ValueError("saved evaluation changed since rendering")
    saved_report = json.loads((saved / "report.json").read_text())
    for run_path in runs:
        run_report, _ = load_trajectory(run_path)
        if saved_report["odometry"][run_report["backend"]] != {
            "report_sha256": sha256(run_path / "report.json"),
            "trajectory_sha256": sha256(run_path / "trajectory.npz"),
        }:
            raise ValueError("analysis trajectory differs from saved evaluation")
    with np.load(output / "predictions.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    observed, anchor = arrays.pop("observed"), arrays.pop("anchor")
    arrays.pop("common_every_probe", None)
    with np.load(saved / "views_000_radius_0.01.npz") as baseline:
        np.testing.assert_array_equal(observed, baseline["observed"])
        np.testing.assert_array_equal(anchor, baseline["common_all_modes"])
        targets = baseline["evaluated_frame_indices"]
        for name in ("gt", "sparse", "hybrid"):
            original = "gt" if name == "gt" else f"{name}_reference_pose"
            np.testing.assert_allclose(
                arrays[f"{name}_v0.02_r0.01"], baseline[original], atol=1e-12, rtol=0
            )
    scores = {
        name: summarize(observed, array, anchor) for name, array in arrays.items()
    }
    groups = {
        "voxel_1cm_2cm": [
            f"{n}_v{v}_r0.01"
            for n in ("gt", "sparse", "hybrid")
            for v in ("0.01", "0.02")
        ],
        "voxel_1cm_2cm_4cm": [
            f"{n}_v{v}_r0.01"
            for n in ("gt", "sparse", "hybrid")
            for v in ("0.01", "0.02", "0.04")
        ],
        "raw_radius": [
            "raw_gt_all_r2.5",
            "raw_gt_all_r5",
            "raw_gt_all_r10",
            "saved_gt",
        ],
        "temporal": [
            "raw_gt_nearest1",
            "raw_gt_nearest2",
            "raw_gt_nearest6",
            "raw_gt_within1s",
            "raw_gt_all_r10",
            "saved_gt",
        ],
        "raw_vs_voxel": [
            "raw_gt_all_r10",
            "raw_sparse_all_r10",
            "raw_hybrid_all_r10",
            "saved_gt",
            "saved_sparse_reference_pose",
            "saved_hybrid_reference_pose",
        ],
        "voxel_radius_5mm_10mm": [
            f"{n}_v0.02_r{v}"
            for n in ("gt", "sparse", "hybrid")
            for v in ("0.005", "0.01")
        ],
        "self": ["self_v0", "self_v0.02", "saved_gt"],
    }
    for radius in (0.015, 0.02):
        with np.load(saved / f"views_000_radius_{radius:g}.npz") as archive:
            for name in ("gt", "sparse_reference_pose", "hybrid_reference_pose"):
                arrays[f"saved_{name}_r{radius:g}"] = archive[name]
    groups["original_radii"] = [
        "saved_gt",
        "saved_sparse_reference_pose",
        "saved_hybrid_reference_pose",
    ] + [
        f"saved_{n}_r{r:g}"
        for r in (0.015, 0.02)
        for n in ("gt", "sparse_reference_pose", "hybrid_reference_pose")
    ]
    paired = {
        name: paired_summary(observed, {k: arrays[k] for k in names}, anchor)
        for name, names in groups.items()
    }
    # Strata depend only on the measured target or on explicitly labelled residuals.
    error = arrays["saved_gt"] - observed
    total = float(np.abs(error[anchor]).sum())
    strata = {}
    distance = np.stack(
        [
            cv2.distanceTransform(
                v.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            for v in np.isfinite(observed)
        ]
    )
    masks = {
        f"depth_{low:g}_{high:g}m": anchor & (observed >= low) & (observed < high)
        for low, high in ((0.3, 1), (1, 2), (2, 4.001))
    }
    masks.update(
        {
            f"invalid_distance_{low}_{high}_lattice_px": anchor
            & (distance >= low)
            & (distance < high)
            for low, high in ((0, 2), (2, 5), (5, 10), (10, 1000))
        }
    )
    masks.update(
        {
            f"absolute_error_{low:g}_{high:g}m": anchor
            & (abs(error) > low)
            & (abs(error) <= high)
            for low, high in (
                (0, 0.02),
                (0.02, 0.05),
                (0.05, 0.1),
                (0.1, 0.2),
                (0.2, 100),
            )
        }
    )
    for name, mask in masks.items():
        strata[name] = {
            "metric": heldout_metrics(observed, arrays["saved_gt"], mask),
            "absolute_error_share": float(np.abs(error[mask]).sum()) / total,
        }
    _, frames = load_trajectory(runs[0])
    with np.load(output / "source_layers.npz") as archive:
        layers, train = archive["depth"], archive["train_indices"]
        np.testing.assert_array_equal(archive["target_indices"], targets)
    ages = np.abs(
        np.array([frames[i].timestamp for i in targets])[:, None]
        - np.array([frames[i].timestamp for i in train])[None, :]
    )
    winner = np.argmin(np.where(np.isfinite(layers), layers, np.inf), axis=1)
    winner_age = np.take_along_axis(ages[:, :, None, None], winner[:, None], axis=1)[
        :, 0
    ]
    raw = arrays["raw_gt_all_r10"]
    raw_valid = anchor & np.isfinite(raw)
    raw_total = float(np.abs((raw - observed)[raw_valid]).sum())
    age_rows = []
    for low, high in ((0, 0.25), (0.25, 1), (1, 2), (2, 100)):
        mask = raw_valid & (winner_age >= low) & (winner_age < high)
        age_rows.append(
            {
                "age_range_s": [low, high],
                "metric": heldout_metrics(observed, raw, mask),
                "absolute_error_share": float(np.abs((raw - observed)[mask]).sum())
                / raw_total,
            }
        )
    summary = {
        **provenance,
        "analysis_script_sha256": sha256(Path(__file__)),
        "predictions_sha256": sha256(output / "predictions.npz"),
        "scores": scores,
        "paired": paired,
        "gt_strata": strata,
        "raw_winner_age": age_rows,
        "notes": [
            "No residual gating or calibration/pose fitting",
            "Saved-common counterfactual coverage is reported explicitly",
            "Every paired group uses its own exact finite intersection",
            "Temporal age is confounded with viewpoint and source count",
            "Self controls reuse target depth and are not held-out scores",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    for group, metrics in paired.items():
        print(
            group,
            {
                name: [score["target_samples"], score["absolute_depth_error_m"]["mean"]]
                for name, score in metrics.items()
            },
            flush=True,
        )
    return summary


def support_probe(saved, runs, output):
    """Attribute original GT winners to single-frame or multiframe voxels."""
    report = json.loads((saved / "report.json").read_text())
    with np.load(saved / "views_000_radius_0.01.npz") as archive:
        observed, anchor, gt = (
            archive["observed"],
            archive["common_all_modes"],
            archive["gt"],
        )
        targets = archive["evaluated_frame_indices"]
    _, frames = load_trajectory(runs[0])
    align = np.linalg.inv(frames[report["split"]["train_indices"][0]].reference_pose)
    K = np.array(report["calibration"]["K_D"])
    K[:2] /= report["mapping_config"]["pixel_stride"]
    with np.load(saved / "sparse_000_reference.npz") as archive:
        cloud = VoxelCloud(**{f.name: archive[f.name] for f in fields(VoxelCloud)})
    predictions = {}
    for name, mask in (
        ("single", cloud.frame_support == 1),
        ("multiple", cloud.frame_support >= 2),
    ):
        subset = VoxelCloud(
            **{f.name: getattr(cloud, f.name)[mask] for f in fields(VoxelCloud)}
        )
        predictions[name] = np.stack(
            [
                render_depth(
                    subset,
                    align @ frames[i].reference_pose,
                    K,
                    observed.shape[1:],
                    0.01,
                )
                for i in targets
            ]
        )
    np.testing.assert_allclose(
        np.fmin(predictions["single"], predictions["multiple"]), gt, atol=1e-12, rtol=0
    )
    one, multi = predictions["single"], predictions["multiple"]
    single_wins = np.isfinite(one) & (~np.isfinite(multi) | (one < multi))
    total = float(np.abs((gt - observed)[anchor]).sum())
    summary = {}
    for name, mask in (
        ("single_wins", single_wins),
        ("multi_or_tie_wins", ~single_wins),
    ):
        mask = anchor & mask
        error = (gt - observed)[mask]
        summary[name] = {
            "metric": heldout_metrics(observed, gt, mask),
            "absolute_error_share": float(abs(error).sum()) / total,
            "tail_20cm_count": int((abs(error) > 0.2).sum()),
        }
    summary["paired"] = paired_summary(observed, {"all": gt, "multiple": multi}, anchor)
    summary["all_valid_multiple"] = heldout_metrics(
        observed, multi, np.isfinite(observed)
    )
    np.savez_compressed(output / "support_predictions.npz", **predictions)
    (output / "support.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


def run(saved, runs, output):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    report = json.loads((saved / "report.json").read_text())
    config = MappingConfig(**report["mapping_config"])
    loaded = {}
    for path in runs:
        meta, frames = load_trajectory(path)
        if meta["backend"] in loaded:
            raise ValueError("duplicate backend")
        if report["odometry"][meta["backend"]] != {
            "report_sha256": sha256(path / "report.json"),
            "trajectory_sha256": sha256(path / "trajectory.npz"),
        }:
            raise ValueError("trajectory differs from saved held-out run")
        loaded[meta["backend"]] = (meta, frames)
    split = check_cohort(loaded, config, report["offset"])
    if split != report["split"] or len(split["chunks"]) != 1:
        raise ValueError("probe requires one matching paired submap")
    if set(loaded) != {"sparse", "hybrid"} or split["train_indices"][0] != 0:
        raise ValueError("probe requires Sparse and Hybrid with the initial segment")
    train, targets = split["chunks"][0], split["target_indices_by_chunk"][0]
    meta, frames = next(iter(loaded.values()))
    digest, metadata, samples = hashlib.sha256(), {}, {}
    reader, _ = open_source(meta)
    with reader:
        validate_source(reader, meta)
        calibration = reader.calibration
        selected = set(train + targets)
        for observation in verified_observations(
            reader, meta, frames, digest, metadata
        ):
            if observation.frame_index in selected:
                samples[observation.frame_index] = sample_observation(
                    observation, calibration, config
                )
    if digest.hexdigest() != report["verified_canonical_input_sha256"]:
        raise ValueError("input differs from held-out run")
    with np.load(saved / "views_000_radius_0.01.npz") as archive:
        baseline = {k: archive[k] for k in archive.files}
    np.testing.assert_array_equal(baseline["evaluated_frame_indices"], targets)
    observed = np.stack([samples[i].depth[::4, ::4] for i in targets])
    if config.pixel_stride != 4:
        raise ValueError("this probe requires the original stride-4 protocol")
    np.testing.assert_array_equal(observed, baseline["observed"])
    anchor = baseline["common_all_modes"]
    K = calibration.K_D.copy()
    K[:2] /= config.pixel_stride
    shape = observed.shape[1:]
    reference_poses = {
        i: np.linalg.inv(frames[train[0]].reference_pose) @ frames[i].reference_pose
        for i in selected
    }
    predictions = {
        "saved_" + k: baseline[k]
        for k in (
            "gt",
            "sparse_reference_pose",
            "hybrid_reference_pose",
            "sparse_estimated_pose",
            "hybrid_estimated_pose",
        )
    }
    output.mkdir(parents=True)
    # Render each source separately so temporal selection never sees residuals.
    raw_layers = []
    for j, target in enumerate(targets):
        layers = [
            render_depth(
                point_cloud(samples[i].points),
                np.linalg.inv(reference_poses[i]) @ reference_poses[target],
                K,
                shape,
                0.01,
            )
            for i in train
        ]
        raw_layers.append(np.asarray(layers, dtype=np.float32))
        if j % 10 == 0:
            print(
                f"Raw GT sources rendered for target {j + 1}/{len(targets)}", flush=True
            )
    raw_layers = np.stack(raw_layers)
    np.savez_compressed(
        output / "source_layers.npz",
        depth=raw_layers,
        train_indices=train,
        target_indices=targets,
    )
    times = np.array([frames[i].timestamp for i in train])
    predictions["raw_gt_all_r10"] = minimum_layers(raw_layers.swapaxes(0, 1))
    for label in ("nearest1", "nearest2", "nearest6", "within1s", "beyond2s"):
        result = []
        for j, target in enumerate(targets):
            age = np.abs(times - frames[target].timestamp)
            if label.startswith("nearest"):
                indices = np.argsort(age)[: int(label[7:])]
            else:
                indices = np.flatnonzero(age <= 1 if label == "within1s" else age > 2)
            result.append(minimum_layers(raw_layers[j, indices]))
        predictions["raw_gt_" + label] = np.stack(result)
    for backend, (_, rows) in loaded.items():
        poses = {i: rows[i].T_SC_D for i in selected}
        for size in (0.01, 0.02, 0.04):
            mapping = VoxelMap(size, max_voxels=2_000_000)
            for i in train:
                mapping.integrate(samples[i], poses[i], i)
            cloud = mapping.cloud()
            # Original map reproduction and footprint controls for every backend.
            for radius in (0.005, 0.01) if size == 0.02 else (0.01,):
                predictions[f"{backend}_v{size:g}_r{radius:g}"] = np.stack(
                    [
                        render_depth(cloud, reference_poses[i], K, shape, radius)
                        for i in targets
                    ]
                )
        print(f"Voxel and footprint controls completed for {backend}", flush=True)
    for size in (0.01, 0.02, 0.04):
        mapping = VoxelMap(size, max_voxels=2_000_000)
        for i in train:
            mapping.integrate(samples[i], reference_poses[i], i)
        cloud = mapping.cloud()
        for radius in (0.0025, 0.005, 0.01) if size == 0.02 else (0.01,):
            predictions[f"gt_v{size:g}_r{radius:g}"] = np.stack(
                [
                    render_depth(cloud, reference_poses[i], K, shape, radius)
                    for i in targets
                ]
            )
    for name in ("gt", "sparse", "hybrid"):
        poses = (
            reference_poses
            if name == "gt"
            else {i: loaded[name][1][i].T_SC_D for i in selected}
        )
        cloud = point_cloud(
            np.concatenate(
                [transform_points(samples[i].points, poses[i]) for i in train]
            )
        )
        for radius in (0.0025, 0.005) if name == "gt" else (0.01,):
            predictions[f"raw_{name}_all_r{radius * 1000:g}"] = np.stack(
                [
                    render_depth(cloud, reference_poses[i], K, shape, radius)
                    for i in targets
                ]
            )
    # Self-view controls intentionally include target depth and are never held-out scores.
    for size in (0, 0.02):
        result = []
        for target in targets:
            if size:
                mapping = VoxelMap(size)
                mapping.integrate(samples[target], np.eye(4), target)
                cloud = mapping.cloud()
            else:
                cloud = point_cloud(samples[target].points)
            result.append(render_depth(cloud, np.eye(4), K, shape, 0.01))
        predictions[f"self_v{size:g}"] = np.stack(result)
    np.testing.assert_allclose(
        predictions["gt_v0.02_r0.01"], baseline["gt"], atol=1e-12
    )
    for name in loaded:
        np.testing.assert_allclose(
            predictions[f"{name}_v0.02_r0.01"],
            baseline[f"{name}_reference_pose"],
            atol=1e-12,
        )
    np.savez_compressed(
        output / "predictions.npz", observed=observed, anchor=anchor, **predictions
    )
    provenance = {
        "saved_report_sha256": sha256(saved / "report.json"),
        "generator_script_sha256": sha256(Path(__file__)),
        "package_code_sha256": code_hash(),
        "verified_input_sha256": digest.hexdigest(),
        "train_indices": train,
        "target_indices": targets,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    return analyze(saved, runs, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--odometry-run", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="reuse rendered diagnostic arrays and their verified input provenance",
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.analyze_only:
        analyze(args.saved, args.odometry_run, args.output_dir)
    else:
        run(args.saved, args.odometry_run, args.output_dir)
    support_probe(args.saved, args.odometry_run, args.output_dir)
    synthetic_probe(args.saved, args.odometry_run, args.output_dir)
