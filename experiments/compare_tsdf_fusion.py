"""Frozen-frame comparison of surface voxels and projective TSDF fusion."""

import argparse
import hashlib
import json
import platform
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.mapping.cli import (
    _code_hash,
    _peak_rss,
    load_trajectory,
    open_source,
    sha256,
    validate_source,
    verified_observations,
)
from scene_recall.mapping.evaluation import (
    compare_clouds,
    heldout_metrics,
    render_depth,
)
from scene_recall.mapping.heldout import (
    check_cohort,
    diagnostic,
    paired_blocks,
    prediction_poses,
)
from scene_recall.mapping.mesh import render_mesh_depth, save_mesh
from scene_recall.mapping.pipeline import Keyframe, build_submap
from scene_recall.mapping.tsdf import TSDFConfig, TSDFView, fuse_tsdf
from scene_recall.mapping.voxel import MappingConfig, VoxelCloud, sample_observation


def training_view(index, pose, samples, stride):
    """Retain precisely the baseline's sampled depth and corresponding color."""
    depth = samples.depth[::stride, ::stride]
    valid = np.isfinite(depth) & (depth > 0)
    colors, colored = (
        np.zeros(depth.shape + (3,), dtype=np.uint8),
        np.zeros(depth.shape, dtype=bool),
    )
    if int(valid.sum()) != len(samples.points):
        raise ValueError("sample points disagree with depth lattice")
    colors[valid], colored[valid] = samples.colors, samples.colored
    return TSDFView(index, pose, depth, colors, colored)


def score_cohort(observed, predictions, anchor):
    valid = np.isfinite(observed) & (observed > 0)
    common = valid.copy()
    for name, depth in predictions.items():
        # The incomplete point-disc control is diagnostic, outside the primary comparison.
        if not name.startswith("tsdf_"):
            common &= np.isfinite(depth) & (depth > 0)
    scores = {}
    for name, depth in predictions.items():
        scores[name] = {}
        for label, mask in (
            ("own", valid),
            ("common_all_modes", common),
            ("baseline_common", anchor),
        ):
            metric = heldout_metrics(observed, depth, mask)
            residual = np.where(
                np.isfinite(depth) & (depth > 0), np.abs(depth - observed), np.inf
            )
            metric["missing_penalized_capped_mae_m"] = {
                str(cap): float(np.minimum(residual[mask], cap).mean())
                if mask.any()
                else None
                for cap in (0.05, 0.2, 1.0)
            }
            scores[name][label] = metric
    return scores, common


def run(saved, runs, output, tsdf_config=None):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    report = json.loads((saved / "report.json").read_text())
    config = MappingConfig(**report["mapping_config"])
    tsdf_config = tsdf_config or TSDFConfig(
        voxel_size_m=config.voxel_size_m, truncation_m=3 * config.voxel_size_m
    )
    if tsdf_config.voxel_size_m != config.voxel_size_m:
        raise ValueError("keep baseline voxel size fixed")
    if 0.01 not in report["render"]["radii_m"]:
        raise ValueError("a 10 mm baseline render is required")
    loaded = {}
    for path in runs:
        meta, rows = load_trajectory(path)
        name = meta["backend"]
        if name in loaded or report["odometry"][name] != {
            "report_sha256": sha256(path / "report.json"),
            "trajectory_sha256": sha256(path / "trajectory.npz"),
        }:
            raise ValueError("duplicate or changed odometry input")
        loaded[name] = (meta, rows)
    if set(loaded) != set(report["odometry"]):
        raise ValueError("provide all saved backends")
    split = check_cohort(loaded, config, report["offset"])
    if split != report["split"]:
        raise ValueError("baseline partition differs")
    meta, frames = next(iter(loaded.values()))
    reader, _ = open_source(meta)
    digest, metadata = hashlib.sha256(), {}
    started = perf_counter()
    cv2.setNumThreads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        reader,
        tempfile.TemporaryDirectory(prefix=".tsdf-", dir=output.parent) as temp,
    ):
        staging = Path(temp) / "result"
        staging.mkdir()
        validate_source(reader, meta)
        K = reader.calibration.K_D.copy()
        K[:2] /= config.pixel_stride
        iterator = iter(verified_observations(reader, meta, frames, digest, metadata))
        chunk_reports, view_rows, buffers, input_hashes, cost = [], [], {}, {}, []
        for chunk, (train, targets) in enumerate(
            zip(split["chunks"], split["target_indices_by_chunk"], strict=True)
        ):
            tick = perf_counter()
            selected, samples = set(train + targets), {}
            for observation in iterator:
                if observation.frame_index in selected:
                    samples[observation.frame_index] = sample_observation(
                        observation, reader.calibration, config
                    )
                if observation.frame_index == train[-1]:
                    break
            chunk_cost = {
                "submap_id": chunk,
                "read_verify_sample_seconds": perf_counter() - tick,
                "maps": {},
            }
            submaps = {}
            for name, (_, rows) in loaded.items():
                tick = perf_counter()
                submaps[name] = build_submap(
                    [Keyframe(rows[i], samples[i]) for i in train], chunk, config
                )
                chunk_cost["maps"][name] = {
                    "baseline_build_with_reference_diagnostics_seconds": perf_counter()
                    - tick
                }
                for mode in ("estimated", "reference"):
                    filename = f"{name}_{chunk:03d}_{mode}.npz"
                    input_hashes[filename] = sha256(saved / filename)
                    with np.load(saved / filename, allow_pickle=False) as archive:
                        for field in fields(VoxelCloud):
                            np.testing.assert_array_equal(
                                getattr(getattr(submaps[name], mode), field.name),
                                archive[field.name],
                            )
            sources = {}
            for name, submap in submaps.items():
                sources[name] = [
                    training_view(
                        i,
                        np.eye(4)
                        if i == train[0]
                        else np.linalg.inv(submap.T_SM) @ loaded[name][1][i].T_SC_D,
                        samples[i],
                        config.pixel_stride,
                    )
                    for i in train
                ]
            first = next(iter(submaps.values()))
            if first.T_MW_reference is not None:
                sources["gt"] = [
                    training_view(
                        i,
                        first.T_MW_reference @ frames[i].reference_pose,
                        samples[i],
                        config.pixel_stride,
                    )
                    for i in first.report["reference_keyframe_indices"]
                ]
            clouds, meshes, map_reports = {}, {}, {}
            for name, views in sources.items():
                tick = perf_counter()
                volume = fuse_tsdf(views, K, tsdf_config)
                fuse_seconds = perf_counter() - tick
                tick = perf_counter()
                cloud = volume.surface(views, K)
                surface_seconds = perf_counter() - tick
                clouds[name] = cloud
                tick = perf_counter()
                meshes[name] = mesh = volume.mesh()
                mesh_seconds = perf_counter() - tick
                chunk_cost["maps"].setdefault(name, {}).update(
                    tsdf_fuse_seconds=fuse_seconds,
                    tsdf_surface_seconds=surface_seconds,
                    tsdf_mesh_seconds=mesh_seconds,
                )
                color_lookup = {
                    tuple(key): color
                    for key, color in zip(cloud.keys, cloud.colors, strict=True)
                }
                colors = np.array(
                    [
                        color_lookup[tuple(key)]
                        for key in np.floor(
                            mesh.vertices / tsdf_config.voxel_size_m
                        ).astype(np.int64)
                    ],
                    dtype=np.uint8,
                ).reshape(-1, 3)
                np.savez_compressed(
                    staging / f"{name}_{chunk:03d}_mesh.npz",
                    vertices=mesh.vertices,
                    triangles=mesh.triangles,
                    colors=colors,
                )
                save_mesh(staging / f"{name}_{chunk:03d}_mesh.ply", mesh, colors)
                np.savez_compressed(
                    staging / f"{name}_{chunk:03d}_volume.npz",
                    keys=volume.keys,
                    signed_distance_m=volume.signed_distance_m,
                    frame_support=volume.frame_support,
                    origin_key=volume.origin_key,
                    grid_shape=volume.grid_shape,
                    training_frame_indices=[v.frame_index for v in views],
                )
                np.savez_compressed(
                    staging / f"{name}_{chunk:03d}_surface.npz",
                    **{f.name: getattr(cloud, f.name) for f in fields(cloud)},
                )
                map_reports[name] = {
                    "grid_shape": volume.grid_shape,
                    "active_voxels": len(volume.keys),
                    "observed_voxels": int((volume.frame_support > 0).sum()),
                    "surface_points": len(cloud.points),
                    "mesh_vertices": len(mesh.vertices),
                    "mesh_triangles": len(mesh.triangles),
                    "zero_near_view_support_points": int(
                        (cloud.view_support == 0).sum()
                    ),
                    "training_frame_indices": [v.frame_index for v in views],
                }
                print(
                    f"Chunk {chunk} {name}: {len(volume.keys)} active voxels, {len(cloud.points)} surface points, {fuse_seconds + surface_seconds:.2f} s",
                    flush=True,
                )
                del volume
            if all(s.report["comparison_status"] == "paired" for s in submaps.values()):
                for name, submap in submaps.items():
                    map_reports[name]["fixed_baseline_gt_comparison"] = compare_clouds(
                        clouds[name], submap.reference, config.voxel_size_m
                    )
            filename = f"views_{chunk:03d}_radius_0.01.npz"
            input_hashes[filename] = sha256(saved / filename)
            with np.load(saved / filename, allow_pickle=False) as archive:
                baseline = {k: archive[k] for k in archive.files}
            original_rows = [
                r for r in report["views"]["0.01"] if r["submap_id"] == chunk
            ]
            expected = [
                r["frame_index"] for r in original_rows if r["status"] == "evaluated"
            ]
            np.testing.assert_array_equal(baseline["evaluated_frame_indices"], expected)
            arrays = {}
            tick = perf_counter()
            for row in original_rows:
                index = row["frame_index"]
                new_row = {
                    k: row[k]
                    for k in ("frame_index", "segment_id", "submap_id", "status")
                }
                if row["status"] != "evaluated":
                    view_rows.append(new_row)
                    continue
                j = expected.index(index)
                observed = samples[index].depth[
                    :: config.pixel_stride, :: config.pixel_stride
                ]
                np.testing.assert_array_equal(observed, baseline["observed"][j])
                predictions = {}
                poses = prediction_poses(
                    submaps, {name: rows[index] for name, (_, rows) in loaded.items()}
                )
                for name, (cloud, pose) in poses.items():
                    prediction = render_depth(cloud, pose, K, observed.shape, 0.01)
                    np.testing.assert_allclose(
                        prediction, baseline[name][j], rtol=0, atol=1e-12
                    )
                    predictions["baseline_" + name] = prediction
                    predictions["tsdf_" + name] = render_depth(
                        clouds[name.split("_")[0]], pose, K, observed.shape, 0.01
                    )
                    predictions["mesh_" + name] = render_mesh_depth(
                        meshes[name.split("_")[0]], pose, K, observed.shape
                    )
                new_row["scores"], common = score_cohort(
                    observed, predictions, baseline["common_all_modes"][j]
                )
                view_rows.append(new_row)
                for name, array in {
                    "observed": observed,
                    "baseline_common": baseline["common_all_modes"][j],
                    "common_all_versions": common,
                    **predictions,
                }.items():
                    arrays.setdefault(name, []).append(array)
                    buffers.setdefault(name, []).append(array.reshape(-1))
                if index in {expected[0], expected[len(expected) // 2], expected[-1]}:
                    diagnostic(
                        staging / f"view_{index:04d}.png",
                        observed,
                        predictions,
                        common,
                        f"Surface voxels versus TSDF, held-out frame {index}",
                    )
            np.savez_compressed(
                staging / filename,
                evaluated_frame_indices=np.array(expected, dtype=np.int64),
                **{k: np.stack(v) for k, v in arrays.items()},
            )
            chunk_cost["render_and_score_seconds"] = perf_counter() - tick
            cost.append(chunk_cost)
            chunk_reports.append(
                {
                    "submap_id": chunk,
                    "baseline": {n: s.report for n, s in submaps.items()},
                    "tsdf": map_reports,
                }
            )
            print(
                f"Completed chunk {chunk}: {len(expected)} evaluated views", flush=True
            )
        for _ in iterator:
            pass
        if digest.hexdigest() != report["verified_canonical_input_sha256"]:
            raise ValueError("canonical input differs from baseline")
        combined = {k: np.concatenate(v) for k, v in buffers.items()}
        pooled = {}
        if combined:
            pooled, _ = score_cohort(
                combined["observed"],
                {
                    k: v
                    for k, v in combined.items()
                    if k not in ("observed", "baseline_common", "common_all_versions")
                },
                combined["baseline_common"],
            )
        evaluated = [r for r in view_rows if r["status"] == "evaluated"]
        result = {
            "experiment": "projective_tsdf_fusion",
            "format_version": 1,
            "mapping_config": report["mapping_config"],
            "render": {
                "primary": "baseline_discs_vs_connected_tsdf_mesh",
                "disc_control": "revoxelized_tsdf_points",
                "disc_radius_m": 0.01,
                "near_m": 0.05,
            },
            "tsdf_config": asdict(tsdf_config),
            "split": split,
            "saved_report_sha256": sha256(saved / "report.json"),
            "saved_arrays_sha256": input_hashes,
            "odometry": report["odometry"],
            "verified_canonical_input_sha256": digest.hexdigest(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "code_sha256": _code_hash(),
                "script_sha256": sha256(Path(__file__)),
            },
            "chunks": chunk_reports,
            "pooled": pooled,
            "views": view_rows,
            "paired_blocks": {
                name: paired_blocks(evaluated, "baseline_" + name, "mesh_" + name)
                for name in ["gt"]
                + [
                    f"{n}_{mode}"
                    for n in loaded
                    for mode in ("reference_pose", "estimated_pose")
                ]
            },
            "timing": {"chunks": cost, "total_seconds": perf_counter() - started},
            "peak_process_rss_mib_linux": _peak_rss(),
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
    parser.add_argument("--truncation-m", type=float, default=0.06)
    args = parser.parse_args()
    baseline_config = json.loads((args.saved / "report.json").read_text())[
        "mapping_config"
    ]
    run(
        args.saved,
        args.odometry_run,
        args.output_dir,
        TSDFConfig(
            voxel_size_m=baseline_config["voxel_size_m"], truncation_m=args.truncation_m
        ),
    )
