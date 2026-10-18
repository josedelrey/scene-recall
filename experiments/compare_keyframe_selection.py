"""Compare temporal and exact-budget motion keyframes on frozen fusion holdouts."""

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
    depth_consistency,
    heldout_metrics,
    render_depth,
    summarize_consistency,
)
from scene_recall.mapping.heldout import check_cohort, paired_blocks
from scene_recall.mapping.keyframes import select_motion_keyframes
from scene_recall.mapping.pipeline import Keyframe, build_submap
from scene_recall.mapping.voxel import MappingConfig, VoxelCloud, sample_observation
from scene_recall.odometry.pipeline import is_rigid


def selection_plan(frames, split, rotation_radius_m):
    """Freeze all target identities before passing eligible poses to selection."""
    reserved = set(split["excluded_outside_chunk_extent"])
    reserved.update(i for group in split["target_indices_by_chunk"] for i in group)
    selections, diagnostics = [], []
    for train in split["chunks"]:
        candidates = [
            (i, frames[i].T_SC_D)
            for i in range(train[0], train[-1] + 1)
            if i not in reserved
        ]
        if len({frames[i].segment_id for i, _ in candidates}) != 1:
            raise ValueError("candidate chunk crosses a tracking reset")
        selected, evidence = select_motion_keyframes(
            candidates, len(train), rotation_radius_m
        )
        selections.append({"temporal": train, "motion": selected})
        diagnostics.append(evidence)
    return selections, diagnostics


def score_cohort(observed, predictions, anchor):
    """Use one exact intersection across all maps, retaining all-target costs."""
    valid = np.isfinite(observed) & (observed > 0)
    common = valid.copy()
    for depth in predictions.values():
        common &= np.isfinite(depth) & (depth > 0)
    scores = {}
    for name, depth in predictions.items():
        scores[name] = {}
        for population, mask in (
            ("all_valid", valid),
            ("baseline_common", anchor),
            ("common_all_versions", common),
        ):
            metric = heldout_metrics(observed, depth, mask)
            error = np.where(
                np.isfinite(depth) & (depth > 0), np.abs(depth - observed), np.inf
            )
            metric["missing_penalized_capped_mae_m"] = {
                str(cap): float(np.minimum(error[mask], cap).mean())
                if mask.any()
                else None
                for cap in (0.05, 0.2, 1.0)
            }
            scores[name][population] = metric
    return scores, common


def training_consistency(indices, frames, samples, K, tolerance):
    """Describe selected-view pairs without treating differing overlap as paired."""
    rows = []
    for offset in (1, 3, 9):
        for a, b in zip(indices, indices[offset:]):
            for source, target in ((a, b), (b, a)):
                for mode in ("estimated", "reference"):
                    A, B = (
                        (frames[source].T_SC_D, frames[target].T_SC_D)
                        if mode == "estimated"
                        else (
                            frames[source].reference_pose,
                            frames[target].reference_pose,
                        )
                    )
                    if A is None or B is None or not is_rigid(A) or not is_rigid(B):
                        continue
                    rows.append(
                        {
                            "source": source,
                            "target": target,
                            "offset": offset,
                            "mode": mode,
                            **depth_consistency(
                                samples[source],
                                samples[target],
                                np.linalg.inv(B) @ A,
                                K,
                                tolerance,
                            ),
                        }
                    )
    return {
        "pairs": rows,
        "summaries": {
            f"{mode}_offset{offset}": summarize_consistency(
                [r for r in rows if r["mode"] == mode and r["offset"] == offset]
            )
            for mode in ("estimated", "reference")
            for offset in (1, 3, 9)
        },
    }


def block_comparisons(view_rows, backends):
    rows = [
        {
            "segment_id": row["segment_id"],
            "scores": {
                name: {
                    "own": s["all_valid"],
                    "common_all_modes": s["common_all_versions"],
                }
                for name, s in row["scores"].items()
            },
        }
        for row in view_rows
        if row["status"] == "evaluated"
    ]
    return {
        f"{backend}_{mode}": paired_blocks(
            rows, f"{backend}_temporal_{mode}", f"{backend}_motion_{mode}"
        )
        for backend in backends
        for mode in ("reference_pose", "estimated_pose", "gt")
    }


def run(saved, runs, output, rotation_radius_m=1.0):
    """Reuse verified sampling, fusion and scoring with fixed baseline boundaries."""
    if output.exists():
        raise ValueError("choose a fresh output directory")
    baseline_report = json.loads((saved / "report.json").read_text())
    config = MappingConfig(**baseline_report["mapping_config"])
    if 0.01 not in baseline_report["render"]["radii_m"]:
        raise ValueError("a saved 10 mm baseline is required")
    loaded = {}
    for path in runs:
        report, frames = load_trajectory(path)
        name = report["backend"]
        if name in loaded or baseline_report["odometry"][name] != {
            "report_sha256": sha256(path / "report.json"),
            "trajectory_sha256": sha256(path / "trajectory.npz"),
        }:
            raise ValueError("duplicate or changed odometry input")
        loaded[name] = (report, frames)
    if set(loaded) != set(baseline_report["odometry"]):
        raise ValueError("provide all saved backends")
    split = check_cohort(loaded, config, baseline_report["offset"])
    if split != baseline_report["split"]:
        raise ValueError("baseline partition differs")
    started = perf_counter()
    cv2.setNumThreads(1)
    plans, selection_diagnostics, selection_seconds = {}, {}, {}
    for name, (_, frames) in loaded.items():
        tick = perf_counter()
        plans[name], selection_diagnostics[name] = selection_plan(
            frames, split, rotation_radius_m
        )
        selection_seconds[name] = perf_counter() - tick
    report, first_frames = next(iter(loaded.values()))
    reader, _ = open_source(report)
    digest, metadata = hashlib.sha256(), {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        reader,
        tempfile.TemporaryDirectory(prefix=".keyframes-", dir=output.parent) as temp,
    ):
        staging = Path(temp) / "result"
        staging.mkdir()
        validate_source(reader, report)
        calibration = reader.calibration
        K = calibration.K_D.copy()
        K[:2] /= config.pixel_stride
        iterator = iter(
            verified_observations(reader, report, first_frames, digest, metadata)
        )
        chunks, view_rows, buffers, saved_hashes = [], [], {}, {}
        costs = {"selection_seconds": selection_seconds, "chunks": []}
        for chunk, (train, targets) in enumerate(
            zip(split["chunks"], split["target_indices_by_chunk"], strict=True)
        ):
            selected = set(targets)
            for plan in plans.values():
                selected.update(i for ids in plan[chunk].values() for i in ids)
            samples = {}
            tick = perf_counter()
            for observation in iterator:
                if observation.frame_index in selected:
                    samples[observation.frame_index] = sample_observation(
                        observation, calibration, config
                    )
                if observation.frame_index == train[-1]:
                    break
            chunk_cost = {
                "submap_id": chunk,
                "read_verify_sample_seconds": perf_counter() - tick,
                "maps": {},
            }
            submaps, chunk_report = {}, {"submap_id": chunk, "maps": {}}
            for name, (_, frames) in loaded.items():
                for strategy, indices in plans[name][chunk].items():
                    label = f"{name}_{strategy}"
                    tick = perf_counter()
                    submap = build_submap(
                        [Keyframe(frames[i], samples[i]) for i in indices],
                        chunk,
                        config,
                    )
                    chunk_cost["maps"][label] = {
                        "build_with_reference_diagnostics_seconds": perf_counter()
                        - tick
                    }
                    submaps[label] = submap
                    tick = perf_counter()
                    consistency = training_consistency(
                        indices,
                        frames,
                        samples,
                        calibration.K_D,
                        config.consistency_tolerance_m,
                    )
                    chunk_cost["maps"][label]["training_consistency_seconds"] = (
                        perf_counter() - tick
                    )
                    chunk_report["maps"][label] = {
                        **submap.report,
                        "sampled_surface_points": sum(
                            len(samples[i].points) for i in indices
                        ),
                        "selected_frame_gaps": np.diff(indices).tolist(),
                        "training_consistency": consistency,
                    }
                    for mode in ("estimated", "reference"):
                        filename = f"{label}_{chunk:03d}_{mode}.npz"
                        cloud = getattr(submap, mode)
                        if strategy == "temporal":
                            original = f"{name}_{chunk:03d}_{mode}.npz"
                            saved_hashes[original] = sha256(saved / original)
                            with np.load(
                                saved / original, allow_pickle=False
                            ) as archive:
                                for f in fields(VoxelCloud):
                                    np.testing.assert_array_equal(
                                        getattr(cloud, f.name), archive[f.name]
                                    )
                        np.savez_compressed(
                            staging / filename,
                            **{
                                f.name: getattr(cloud, f.name)
                                for f in fields(VoxelCloud)
                            },
                        )
                # A common fixed reference map makes both surface-distance directions comparable.
                if submaps[f"{name}_temporal"].report["comparison_status"] == "paired":
                    for strategy in ("temporal", "motion"):
                        label = f"{name}_{strategy}"
                        tick = perf_counter()
                        chunk_report["maps"][label][
                            "fixed_temporal_reference_comparison"
                        ] = compare_clouds(
                            submaps[label].estimated,
                            submaps[f"{name}_temporal"].reference,
                            config.voxel_size_m,
                        )
                        chunk_cost["maps"][label][
                            "fixed_reference_comparison_seconds"
                        ] = perf_counter() - tick
            filename = f"views_{chunk:03d}_radius_0.01.npz"
            saved_hashes[filename] = sha256(saved / filename)
            with np.load(saved / filename, allow_pickle=False) as archive:
                original = {key: archive[key] for key in archive.files}
            original_rows = [
                r for r in baseline_report["views"]["0.01"] if r["submap_id"] == chunk
            ]
            expected = [
                r["frame_index"] for r in original_rows if r["status"] == "evaluated"
            ]
            np.testing.assert_array_equal(original["evaluated_frame_indices"], expected)
            paired = all(
                s.report["comparison_status"] == "paired" for s in submaps.values()
            )
            arrays, evaluated = {}, []
            tick = perf_counter()
            for row in original_rows:
                index = row["frame_index"]
                status = row["status"]
                if status == "evaluated" and not paired:
                    status = "unavailable_selected_training_references"
                new_row = {
                    "frame_index": index,
                    "segment_id": row["segment_id"],
                    "submap_id": chunk,
                    "status": status,
                }
                if status != "evaluated":
                    view_rows.append(new_row)
                    continue
                j = expected.index(index)
                observed = samples[index].depth[
                    :: config.pixel_stride, :: config.pixel_stride
                ]
                np.testing.assert_array_equal(observed, original["observed"][j])
                predictions = {}
                for label, submap in submaps.items():
                    name, strategy = label.split("_")
                    target = loaded[name][1][index]
                    for mode, cloud, pose in (
                        (
                            "reference_pose",
                            submap.estimated,
                            submap.T_MW_reference @ target.reference_pose,
                        ),
                        (
                            "estimated_pose",
                            submap.estimated,
                            np.linalg.inv(submap.T_SM) @ target.T_SC_D,
                        ),
                        (
                            "gt",
                            submap.reference,
                            submap.T_MW_reference @ target.reference_pose,
                        ),
                    ):
                        depth = render_depth(cloud, pose, K, observed.shape, 0.01)
                        if strategy == "temporal":
                            old_name = "gt" if mode == "gt" else f"{name}_{mode}"
                            np.testing.assert_allclose(
                                depth, original[old_name][j], rtol=0, atol=1e-12
                            )
                        predictions[f"{label}_{mode}"] = depth
                scores, common = score_cohort(
                    observed, predictions, original["common_all_modes"][j]
                )
                new_row["scores"] = scores
                view_rows.append(new_row)
                evaluated.append(index)
                for key, value in {
                    "observed": observed,
                    "baseline_common": original["common_all_modes"][j],
                    "common_all_versions": common,
                    **predictions,
                }.items():
                    arrays.setdefault(key, []).append(value)
                    buffers.setdefault(key, []).append(value.reshape(-1))
                if len(evaluated) % 10 == 0:
                    print(
                        f"Chunk {chunk}: {len(evaluated)}/{len(expected)} held-out views",
                        flush=True,
                    )
            chunk_cost["render_and_score_seconds"] = perf_counter() - tick
            costs["chunks"].append(chunk_cost)
            chunks.append(chunk_report)
            np.savez_compressed(
                staging / filename,
                evaluated_frame_indices=np.array(evaluated, dtype=np.int64),
                **{key: np.stack(values) for key, values in arrays.items()},
            )
            print(
                f"Completed chunk {chunk}: {len(train)} keyframes per strategy, {len(evaluated)} evaluated views",
                flush=True,
            )
        for _ in iterator:
            pass
        if digest.hexdigest() != baseline_report["verified_canonical_input_sha256"]:
            raise ValueError("canonical input differs from baseline")
        combined = {key: np.concatenate(value) for key, value in buffers.items()}
        pooled = {}
        if combined:
            pooled, _ = score_cohort(
                combined["observed"],
                {
                    key: value
                    for key, value in combined.items()
                    if key not in ("observed", "baseline_common", "common_all_versions")
                },
                combined["baseline_common"],
            )
        result = {
            "experiment": "motion_budget_keyframes",
            "format_version": 1,
            "rotation_radius_m": rotation_radius_m,
            "mapping_config": baseline_report["mapping_config"],
            "baseline_split": split,
            "selection": plans,
            "selection_diagnostics": selection_diagnostics,
            "saved_report_sha256": sha256(saved / "report.json"),
            "saved_arrays_sha256": saved_hashes,
            "odometry": baseline_report["odometry"],
            "verified_canonical_input_sha256": digest.hexdigest(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "code_sha256": _code_hash(),
                "script_sha256": sha256(Path(__file__)),
            },
            "chunks": chunks,
            "pooled": pooled,
            "views": view_rows,
            "paired_blocks": block_comparisons(view_rows, loaded),
            "timing": {**costs, "total_seconds": perf_counter() - started},
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
    parser.add_argument("--rotation-radius-m", type=float, default=1.0)
    args = parser.parse_args()
    run(args.saved, args.odometry_run, args.output_dir, args.rotation_radius_m)
