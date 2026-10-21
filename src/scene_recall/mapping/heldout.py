"""Controlled held-out RGB-D validation using the existing surface-map pipeline."""

import argparse
import csv
import hashlib
import json
import platform
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.geometry.transforms import is_rigid
from scene_recall.mapping.evaluation import heldout_metrics, render_depth
from scene_recall.mapping.export import save_cloud
from scene_recall.mapping.pipeline import Keyframe, build_submap, select_keyframes
from scene_recall.mapping.voxel import MappingConfig, sample_observation
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, peak_rss, sha256


def heldout_split(frames, config: MappingConfig, offset: int) -> dict:
    """Interleave targets with the unchanged segment-relative training policy.

    Targets must lie inside a single training chunk's temporal extent. Gaps
    between bounded chunks are excluded instead of choosing a map by its score.
    A tracking boundary restarts the schedule and never supplies a map transform.
    """
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or not 0 < offset < config.keyframe_stride
    ):
        raise ValueError("held-out offset must lie strictly inside keyframe stride")
    train = select_keyframes(frames, config.keyframe_stride)
    chunks = []
    for index in train:
        if (
            not chunks
            or frames[chunks[-1][0]].segment_id != frames[index].segment_id
            or len(chunks[-1]) == config.max_keyframes_per_submap
        ):
            chunks.append([])
        chunks[-1].append(index)
    candidates = []
    start = 0
    for i, row in enumerate(frames):
        if i == 0 or row.segment_id != frames[i - 1].segment_id:
            start = i
        if (i - start) % config.keyframe_stride == offset and i not in train:
            candidates.append(i)
    targets = [[i for i in candidates if chunk[0] < i < chunk[-1]] for chunk in chunks]
    assigned = {i for group in targets for i in group}
    return {
        "train_indices": train,
        "chunks": chunks,
        "target_indices_by_chunk": targets,
        "excluded_outside_chunk_extent": [i for i in candidates if i not in assigned],
    }


def check_cohort(loaded, config, offset):
    """Refuse backend comparisons unless input identity and partitions coincide."""
    first_report, first_frames = next(iter(loaded.values()))
    split = heldout_split(first_frames, config, offset)
    for report, frames in loaded.values():
        if (
            report["source"] != first_report["source"]
            or report["calibration"] != first_report["calibration"]
        ):
            raise ValueError("backend source identity or calibration differs")
        identity = lambda report: [
            {
                key: row[key]
                for key in (
                    "frame_index",
                    "source_frame_index",
                    "provenance",
                    "segment_id",
                    "status",
                )
            }
            for row in report["frames"]
        ]
        if identity(report) != identity(first_report):
            raise ValueError(
                "backend provenance or segmentation differs, evaluate separate cohorts"
            )
        if heldout_split(frames, config, offset) != split:
            raise ValueError("backend training or held-out partition differs")
        for a, b in zip(first_frames, frames, strict=True):
            if a.timestamp != b.timestamp or a.segment_id != b.segment_id:
                raise ValueError("backend timestamps or disconnected segments differ")
            if (a.reference_pose is None) != (b.reference_pose is None) or (
                a.reference_pose is not None
                and not np.array_equal(
                    a.reference_pose, b.reference_pose, equal_nan=True
                )
            ):
                raise ValueError("backend reference poses differ")
    return split


def prediction_poses(submaps, rows):
    """Reference-pose geometry and estimated-pose end-to-end gauges stay explicit."""
    result = {}
    for name, submap in submaps.items():
        row = rows[name]
        if row.segment_id != submap.segment_id:
            raise ValueError("cannot render across disconnected segments")
        if (
            row.reference_pose is None
            or not is_rigid(row.reference_pose)
            or submap.T_MW_reference is None
        ):
            raise ValueError(
                "reference-pose validation requires valid target and anchor"
            )
        result[f"{name}_reference_pose"] = (
            submap.estimated,
            submap.T_MW_reference @ row.reference_pose,
        )
        result[f"{name}_estimated_pose"] = (
            submap.estimated,
            np.linalg.inv(submap.T_SM) @ row.T_SC_D,
        )
    first = next(iter(submaps))
    submap = submaps[first]
    result["gt"] = (
        submap.reference,
        submap.T_MW_reference @ rows[first].reference_pose,
    )
    return result


def score_predictions(observed, predictions):
    """Use one target mask, plus exact per-mode and across-mode intersections."""
    valid = np.isfinite(observed) & (observed > 0)
    common = {"all_modes": valid.copy()}
    for mode in ("reference_pose", "estimated_pose"):
        common[mode] = valid.copy()
        for name, depth in predictions.items():
            if name == "gt" or name.endswith(mode):
                common[mode] &= np.isfinite(depth) & (depth > 0)
    for depth in predictions.values():
        common["all_modes"] &= np.isfinite(depth) & (depth > 0)
    scores = {}
    for name, depth in predictions.items():
        scores[name] = {
            "own": heldout_metrics(observed, depth, valid),
            "common_all_modes": heldout_metrics(observed, depth, common["all_modes"]),
        }
        for mode in ("reference_pose", "estimated_pose"):
            if name == "gt" or name.endswith(mode):
                scores[name][f"common_{mode}"] = heldout_metrics(
                    observed, depth, common[mode]
                )
    return scores, common


def paired_blocks(rows, first, second, block_views=10):
    """Describe paired view changes with a seeded temporal-block bootstrap.

    Block means receive equal weight. Intervals describe sensitivity to these
    blocks on one sequence and do not establish cross-sequence significance.
    Missing common-population errors remain unavailable, never zero-filled.
    """
    result = {}
    for metric in ("common_mae_m", "explained_5cm"):
        blocks, view_deltas = [], []
        for segment in sorted({r["segment_id"] for r in rows}):
            group = [r for r in rows if r["segment_id"] == segment]
            for start in range(0, len(group), block_views):
                deltas = []
                for row in group[start : start + block_views]:

                    def value(name, row=row, metric=metric):
                        score = row["scores"][name]
                        if metric == "common_mae_m":
                            return score["common_all_modes"]["absolute_depth_error_m"][
                                "mean"
                            ]
                        return score["own"]["thresholds"]["0.05"][
                            "explained_fraction_of_target"
                        ]

                    a, b = value(first), value(second)
                    if a is not None and b is not None:
                        deltas.append(b - a)
                if deltas:
                    blocks.append(float(np.mean(deltas)))
                    view_deltas.extend(deltas)
        rng = np.random.default_rng(0)
        ci = None
        if len(blocks) >= 2:
            means = np.asarray(blocks)[
                rng.integers(len(blocks), size=(10000, len(blocks)))
            ].mean(axis=1)
            ci = np.quantile(means, [0.025, 0.975]).tolist()
        result[metric] = {
            "second_minus_first": [first, second],
            "paired_views": len(view_deltas),
            "negative_delta_views": int((np.asarray(view_deltas) < 0).sum()),
            "block_views": block_views,
            "block_count": len(blocks),
            "mean_block_delta": float(np.mean(blocks)) if blocks else None,
            "block_bootstrap_95_interval": ci,
        }
    return result


def diagnostic(path, observed, predictions, common, title):
    """Show identical pixel coordinates and residual scales for every map."""
    names = ["observed"] + list(predictions) + ["common_all_modes"]
    tile_w, tile_h = 480, 395
    image = Image.new(
        "RGB", (tile_w * 3, 80 + tile_h * ((len(names) + 2) // 3)), "#14212c"
    )
    draw = ImageDraw.Draw(image)
    draw.text((15, 10), title, fill="white", font_size=20)
    draw.text(
        (15, 38),
        "Residual: red = map in front, blue = behind, white = zero, gray = missing, black = invalid target",
        fill="white",
        font_size=15,
    )
    valid = np.isfinite(observed)
    for i, name in enumerate(names):
        rgb = np.zeros(observed.shape + (3,), dtype=np.uint8)
        if name == "observed":
            intensity = np.nan_to_num(np.clip((observed - 0.3) / 3.7, 0, 1)) * 255
            rgb[:] = intensity[..., None].astype(np.uint8)
        elif name == "common_all_modes":
            rgb[valid] = [85, 85, 85]
            rgb[common] = [60, 210, 160]
        else:
            prediction = predictions[name]
            visible = valid & np.isfinite(prediction)
            residual = np.nan_to_num((prediction - observed) / 0.15)
            amount = np.clip(np.abs(residual), 0, 1)
            rgb[valid] = [85, 85, 85]
            rgb[visible] = (255 * (1 - amount[visible, None])).astype(np.uint8)
            rgb[visible & (residual < 0), 0] = 255
            rgb[visible & (residual >= 0), 2] = 255
        x, y = (i % 3) * tile_w, 80 + (i // 3) * tile_h
        draw.text(
            (x + 10, y + 5),
            name + (" (+/-150 mm)" if name in predictions else ""),
            fill="white",
            font_size=17,
        )
        image.paste(
            Image.fromarray(rgb).resize((480, 360), Image.Resampling.NEAREST),
            (x, y + 30),
        )
    image.save(path)


def run_heldout(
    runs,
    output,
    config=None,
    offset=5,
    radii=(0.01,),
    source_override=None,
):
    """Build maps only from training frames, validate inputs, then publish atomically."""
    if config is None:
        config = MappingConfig(max_keyframes_per_submap=64)
    if output.exists():
        raise ValueError("output already exists, choose a new output directory")
    if (
        not radii
        or len(set(radii)) != len(radii)
        or any(not np.isfinite(r) or r <= 0 for r in radii)
    ):
        raise ValueError("render radii must be unique, finite and positive")
    loaded = {}
    for run in runs:
        report, frames = load_trajectory(run)
        name = report["backend"]
        if (
            name
            not in (
                "sparse",
                "hybrid",
                "icp",
                "local_map",
                "local_map_recent",
                "incremental_slam",
            )
            or name in loaded
        ):
            raise ValueError("provide distinct supported odometry backends")
        loaded[name] = (report, frames)
    if not loaded:
        raise ValueError("at least one odometry run is required")
    split = check_cohort(loaded, config, offset)
    first_report, first_frames = next(iter(loaded.values()))
    reader, source_path = open_source(first_report, source_override)
    started = perf_counter()
    cv2.setNumThreads(1)
    metadata, submap_reports, view_rows = {}, [], {str(r): [] for r in radii}
    digest = hashlib.sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        reader,
        tempfile.TemporaryDirectory(prefix=".heldout-", dir=output.parent) as temporary,
    ):
        staging = Path(temporary) / "result"
        staging.mkdir()
        validate_source(reader, first_report)
        calibration = reader.calibration
        K = calibration.K_D.copy()
        K[:2] /= config.pixel_stride
        iterator = iter(
            verified_observations(reader, first_report, first_frames, digest, metadata)
        )
        for chunk_id, (train, targets) in enumerate(
            zip(split["chunks"], split["target_indices_by_chunk"], strict=True)
        ):
            selected = set(train + targets)
            samples = {}
            for observation in iterator:
                index = observation.frame_index
                if index in selected:
                    samples[index] = sample_observation(
                        observation, calibration, config
                    )
                if index == train[-1]:
                    break
            submaps = {
                name: build_submap(
                    [Keyframe(frames[i], samples[i]) for i in train], chunk_id, config
                )
                for name, (_, frames) in loaded.items()
            }
            reference = None
            for name, submap in submaps.items():
                submap_reports.append({"backend": name, **submap.report})
                for mode in ("estimated", "reference"):
                    save_cloud(
                        staging / f"{name}_{chunk_id:03d}_{mode}",
                        getattr(submap, mode),
                        submap.report,
                    )
                if submap.report["comparison_status"] != "paired":
                    continue
                if reference is not None:
                    for field in fields(reference):
                        if not np.array_equal(
                            getattr(reference, field.name),
                            getattr(submap.reference, field.name),
                        ):
                            raise ValueError(
                                "controlled reference map arrays differ across backends"
                            )
                reference = submap.reference
            paired = all(
                s.report["comparison_status"] == "paired" for s in submaps.values()
            )
            archive = {"frame_indices": np.array(targets, dtype=np.int64)}
            for radius in radii:
                buffers = {}
                for index in targets:
                    status = "evaluated"
                    if not paired:
                        status = "unavailable_unpaired_training_references"
                    elif first_frames[index].reference_pose is None or not is_rigid(
                        first_frames[index].reference_pose
                    ):
                        status = "unavailable_target_reference"
                    if status != "evaluated":
                        view_rows[str(radius)].append(
                            {
                                "frame_index": index,
                                "segment_id": first_frames[index].segment_id,
                                "submap_id": chunk_id,
                                "status": status,
                            }
                        )
                        continue
                    observed = samples[index].depth[
                        :: config.pixel_stride, :: config.pixel_stride
                    ]
                    poses = prediction_poses(
                        submaps,
                        {name: frames[index] for name, (_, frames) in loaded.items()},
                    )
                    predictions = {
                        name: render_depth(cloud, pose, K, observed.shape, radius)
                        for name, (cloud, pose) in poses.items()
                    }
                    scores, common = score_predictions(observed, predictions)
                    row = {
                        "frame_index": index,
                        "segment_id": first_frames[index].segment_id,
                        "submap_id": chunk_id,
                        "status": status,
                        "lattice_samples": observed.size,
                        "valid_target_samples": int(np.isfinite(observed).sum()),
                        "scores": scores,
                    }
                    view_rows[str(radius)].append(row)
                    for name, array in {
                        "observed": observed,
                        **predictions,
                        **{f"common_{k}": v for k, v in common.items()},
                    }.items():
                        buffers.setdefault(name, []).append(array)
                    if index in {targets[0], targets[len(targets) // 2], targets[-1]}:
                        diagnostic(
                            staging / f"view_{index:04d}_radius_{radius:g}.png",
                            observed,
                            predictions,
                            common["all_modes"],
                            f"Held-out frame {index}, segment {first_frames[index].segment_id}, disc radius {radius:g} m",
                        )
                np.savez_compressed(
                    staging / f"views_{chunk_id:03d}_radius_{radius:g}.npz",
                    **archive,
                    **{name: np.stack(arrays) for name, arrays in buffers.items()},
                    evaluated_frame_indices=np.array(
                        [
                            r["frame_index"]
                            for r in view_rows[str(radius)]
                            if r["submap_id"] == chunk_id and r["status"] == "evaluated"
                        ],
                        dtype=np.int64,
                    ),
                )
            print(
                f"Validated submap {chunk_id}: {len(train)} training, {len(targets)} held-out views",
                flush=True,
            )
        # Consume trailing nonselected inputs too, triggering the fingerprint check.
        for _ in iterator:
            pass
        summaries = {}
        for radius in radii:
            arrays = {}
            for path in sorted(staging.glob(f"views_*_radius_{radius:g}.npz")):
                with np.load(path, allow_pickle=False) as data:
                    for name in data.files:
                        if name not in ("frame_indices", "evaluated_frame_indices"):
                            arrays.setdefault(name, []).append(data[name].reshape(-1))
            pooled = {}
            if arrays:
                combined = {
                    name: np.concatenate(items) for name, items in arrays.items()
                }
                observed = combined.pop("observed")
                pooled, _ = score_predictions(
                    observed,
                    {
                        name: arr
                        for name, arr in combined.items()
                        if not name.startswith("common_")
                    },
                )
            evaluated = [
                row for row in view_rows[str(radius)] if row["status"] == "evaluated"
            ]
            summaries[str(radius)] = {
                "pooled": pooled,
                "evaluated_views": len(evaluated),
                "target_view_candidates": len(view_rows[str(radius)]),
                "unavailable_status_counts": {
                    status: sum(
                        row["status"] == status for row in view_rows[str(radius)]
                    )
                    for status in sorted(
                        {
                            row["status"]
                            for row in view_rows[str(radius)]
                            if row["status"] != "evaluated"
                        }
                    )
                },
                "lattice_samples": sum(row["lattice_samples"] for row in evaluated),
                "valid_target_samples": sum(
                    row["valid_target_samples"] for row in evaluated
                ),
                "paired_blocks": {},
            }
            if "sparse" in loaded and "hybrid" in loaded:
                for mode in ("reference_pose", "estimated_pose"):
                    summaries[str(radius)]["paired_blocks"][mode] = paired_blocks(
                        evaluated, f"sparse_{mode}", f"hybrid_{mode}"
                    )
        report = {
            "format_version": 1,
            "experiment": "held_out_fusion_views",
            "mapping_config": asdict(config),
            "offset": offset,
            "render": {
                "radii_m": list(radii),
                "near_m": 0.05,
                "surface": "camera_facing_metric_discs_nearest_z",
                "evaluation_lattice": "native pixels [::pixel_stride, ::pixel_stride]",
            },
            "split": split,
            "source": {**first_report["source"], "resolved_path": str(source_path)},
            "calibration": first_report["calibration"],
            "verified_canonical_input_sha256": digest.hexdigest(),
            "odometry": {
                report["backend"]: {
                    "report_sha256": sha256(run / "report.json"),
                    "trajectory_sha256": sha256(run / "trajectory.npz"),
                }
                for run, (report, _) in zip(runs, loaded.values(), strict=True)
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "code_sha256": code_hash(),
            },
            "submaps": submap_reports,
            "summaries": summaries,
            "views": view_rows,
            "timing": {"total_seconds": perf_counter() - started},
            "peak_process_rss_mib_linux": peak_rss(),
        }
        csv_rows = []
        for radius, summary in summaries.items():
            for name, metric in summary["pooled"].items():
                own, common = metric["own"], metric["common_all_modes"]
                csv_rows.append(
                    {
                        "radius_m": radius,
                        "prediction": name,
                        "target_samples": own["target_samples"],
                        "predicted_samples": own["predicted_samples"],
                        "coverage": own["coverage"],
                        "common_samples": common["target_samples"],
                        "common_mae_m": common["absolute_depth_error_m"]["mean"],
                        "common_rmse_m": common["absolute_depth_error_m"]["rmse"],
                        "common_p95_m": common["absolute_depth_error_m"]["p95"],
                        "explained_5cm": own["thresholds"]["0.05"][
                            "explained_fraction_of_target"
                        ],
                    }
                )
        if csv_rows:
            with (staging / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
        selected = set(
            split["train_indices"]
            + [i for group in split["target_indices_by_chunk"] for i in group]
        )
        (staging / "inputs.json").write_text(
            json.dumps(
                [metadata[i] for i in sorted(selected)], indent=2, allow_nan=False
            )
            + "\n"
        )
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise ValueError("output appeared during evaluation, refusing overwrite")
        staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry-run", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--heldout-offset", type=int, default=5)
    parser.add_argument("--render-radii-m", type=float, nargs="+", default=[0.01])
    for field in fields(MappingConfig):
        default = 64 if field.name == "max_keyframes_per_submap" else field.default
        parser.add_argument(
            "--" + field.name.replace("_", "-"), type=type(default), default=default
        )
    args = parser.parse_args()
    try:
        config = MappingConfig(
            **{f.name: getattr(args, f.name) for f in fields(MappingConfig)}
        )
        run_heldout(
            args.odometry_run,
            args.output_dir,
            config,
            args.heldout_offset,
            tuple(args.render_radii_m),
            args.source_path,
        )
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
