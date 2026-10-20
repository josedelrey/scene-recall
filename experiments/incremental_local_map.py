"""Matched incremental frontend comparisons with common-population errors.

The default matrix includes original Sparse, a feature-budget Sparse control,
Hybrid with that same budget, and local-map tracking. Optional variants isolate
landmark updates, map lifetime and fallback. References enter evaluation only.
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from scene_recall.geometry.transforms import is_rigid
from scene_recall.mapping.heldout import check_cohort, run_heldout
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.odometry.hybrid import HybridConfig
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.local_map import LocalMapConfig
from scene_recall.odometry.recent_map import RecentMapConfig
from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig
from scene_recall.run_metadata import sha256


def variants():
    sparse = SparseConfig(max_features=1500)
    local = LocalMapConfig(sparse=sparse)
    return {
        "sparse": ("sparse", SparseConfig()),
        "budget": ("sparse", sparse),
        "hybrid": ("hybrid", HybridConfig(sparse=sparse)),
        "local": ("local_map", local),
        "median": ("local_map", replace(local, landmark_update="median")),
        "no_fallback": ("local_map", replace(local, pairwise_fallback=False)),
        "short_map": ("local_map", replace(local, max_keyframes=2)),
        "recent": ("local_map_recent", RecentMapConfig(sparse=sparse)),
        "anchors": (
            "local_map_recent",
            RecentMapConfig(sparse=sparse, association_policy="exhaustive"),
        ),
        "index": (
            "local_map_recent",
            RecentMapConfig(sparse=sparse, anchor_policy="frozen"),
        ),
    }


def common_errors(first, second, interval):
    """Score exactly the same valid endpoint pairs continuous in both runs."""
    errors = {"baseline": [], "candidate": []}
    indices = []
    for i in range(len(first) - interval):
        j = i + interval
        if any(rows[i].segment_id != rows[j].segment_id for rows in (first, second)):
            continue
        if any(
            r.reference_pose is None or not is_rigid(r.reference_pose)
            for r in (first[i], first[j])
        ):
            continue
        reference = np.linalg.inv(first[j].reference_pose) @ first[i].reference_pose
        if not is_rigid(reference):
            continue
        indices.append([i, j])
        for name, rows in (("baseline", first), ("candidate", second)):
            errors[name].append(
                pose_error(np.linalg.inv(rows[j].T_SC_D) @ rows[i].T_SC_D, reference)
            )
    return {
        "pairs": indices,
        **{
            name: {
                "translation_m": error_statistics(e[0] for e in values),
                "rotation_deg": error_statistics(e[1] for e in values),
            }
            for name, values in errors.items()
        },
    }


def summarize(paths):
    loaded = {name: load_trajectory(path) for name, path in paths.items()}
    baseline, first = loaded["sparse"]
    result = {}
    for name, (report, rows) in loaded.items():
        if (
            report["source"] != baseline["source"]
            or report["calibration"] != baseline["calibration"]
        ):
            raise ValueError("comparison source or calibration differs")
        if len(rows) != len(first):
            raise ValueError("comparison lengths differ")
        expected = variants().get(name)
        if expected is None or report["config"] != asdict(expected[1]):
            raise ValueError("comparison configuration differs from frozen variant")
        for field in (
            "python",
            "numpy",
            "opencv",
            "opencv_threads",
            "code_sha256",
            "openblas_num_threads_env",
            "omp_num_threads_env",
        ):
            if report["environment"][field] != baseline["environment"][field]:
                raise ValueError("comparison implementation or environment differs")
        if [r.get("provenance") for r in report["frames"]] != [
            r.get("provenance") for r in baseline["frames"]
        ]:
            raise ValueError("comparison input provenance differs")
        for a, b in zip(first, rows, strict=True):
            if a.timestamp != b.timestamp or (a.reference_pose is None) != (
                b.reference_pose is None
            ):
                raise ValueError("comparison clocks or references differ")
            if a.reference_pose is not None and not np.array_equal(
                a.reference_pose, b.reference_pose, equal_nan=True
            ):
                raise ValueError("comparison references differ")
        diagnostics = [r["diagnostics"] for r in report["frames"] if r["diagnostics"]]
        mode_errors, regressions = {}, []
        base_errors = baseline["evaluation"]["pair_errors_m_deg"]
        for i, entry in enumerate(report["evaluation"]["pair_errors_m_deg"], start=1):
            error = entry["final"]
            if error is None:
                continue
            diagnostic = report["frames"][i]["diagnostics"]
            mode = diagnostic.get("tracking_mode", report["backend"])
            mode_errors.setdefault(mode, []).append(error)
            base_error = base_errors[i - 1]["final"]
            if base_error is not None:
                regressions.append(
                    {
                        "target_frame": i,
                        "mode": mode,
                        "translation_delta_m": error[0] - base_error[0],
                        "rotation_delta_deg": error[1] - base_error[1],
                        "map_inliers": diagnostic.get("map_inliers"),
                        "map_reason": diagnostic.get("map_reason"),
                    }
                )
        result[name] = {
            "run": str(paths[name].resolve()),
            "report_sha256": sha256(paths[name] / "report.json"),
            "trajectory_sha256": sha256(paths[name] / "trajectory.npz"),
            "config": report["config"],
            "code_sha256": report["environment"]["code_sha256"],
            "evaluation": report["evaluation"],
            "timing": report["timing"],
            "tracking_modes": dict(
                Counter(d.get("tracking_mode", report["backend"]) for d in diagnostics)
            ),
            "tracking_mode_pair_errors": {
                mode: {
                    "translation_m": error_statistics(e[0] for e in errors),
                    "rotation_deg": error_statistics(e[1] for e in errors),
                }
                for mode, errors in mode_errors.items()
            },
            "largest_translation_regressions": sorted(
                regressions,
                key=lambda r: (-r["translation_delta_m"], r["target_frame"]),
            )[:10],
            "map_support": {
                key: error_statistics(d[key] for d in diagnostics if key in d)
                for key in (
                    "map_inliers",
                    "map_confirmed_inliers",
                    "map_latest_keyframe_inliers",
                    "map_information_ratio",
                    "map_grid_coverage",
                )
            },
            "map_rejections": dict(
                Counter(d["map_reason"] for d in diagnostics if d.get("map_reason"))
            ),
            "keyframe_insertions": sum(
                d.get("keyframe_inserted", False) for d in diagnostics
            ),
            "recovered_sparse_losses": [
                i
                for i, (a, b) in enumerate(zip(first, rows, strict=True))
                if a.status == "lost" and b.status == "tracked"
            ],
            "additional_losses": [
                i
                for i, (a, b) in enumerate(zip(first, rows, strict=True))
                if a.status == "tracked" and b.status == "lost"
            ],
            "frontend_stage_seconds": {
                key: error_statistics(d[key] for d in diagnostics if key in d)
                for key in (
                    "feature_seconds",
                    "map_tracking_seconds",
                    "map_update_seconds",
                    "frontend_seconds",
                )
            },
            "max_active_landmarks": max(
                (d.get("active_landmarks", 0) for d in diagnostics), default=0
            ),
            "max_active_keyframes": max(
                (d.get("active_keyframes", 0) for d in diagnostics), default=0
            ),
            "common_with_sparse": {
                str(k): common_errors(first, rows, k) for k in (1, 5, 10, 20, 50, 100)
            },
        }
    return {
        "source": baseline["source"],
        "variants": result,
        "comparison_script_sha256": sha256(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--tum-path", type=Path)
    sources.add_argument("--sens-path", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(variants()),
        default=["sparse", "budget", "hybrid", "local"],
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--evaluate-maps", action="store_true")
    args = parser.parse_args()
    if "sparse" not in args.variants or len(set(args.variants)) != len(args.variants):
        parser.error("variants must be unique and include sparse")
    paths = {name: args.output_dir / name for name in args.variants}
    if not args.summarize_only:
        if args.output_dir.exists():
            parser.error("choose a fresh comparison output directory")
        for name, path in paths.items():
            backend, config = variants()[name]
            print(f"Running {name}", flush=True)
            run_odometry(
                args.tum_path or args.sens_path,
                path,
                dataset="tum_rgbd" if args.tum_path else "scannet_v2",
                backend_name=backend,
                config=config,
                start=args.start,
                stop=args.stop,
                rpe_intervals=(1, 5, 10, 20, 50, 100),
            )
    result = summarize(paths)
    if args.evaluate_maps:
        result["fusion_validation"] = {}
        config = MappingConfig(pixel_stride=8, max_keyframes_per_submap=24)
        for name in args.variants:
            if variants()[name][0] not in ("local_map", "local_map_recent"):
                continue
            cohort = {k: load_trajectory(paths[k]) for k in ("sparse", name)}
            try:
                check_cohort(cohort, config, 5)
            except ValueError as error:
                result["fusion_validation"][name] = {
                    "status": "withheld",
                    "reason": str(error),
                }
                continue
            output = args.output_dir / f"maps_{name}"
            if args.summarize_only and output.exists():
                report = json.loads(
                    (output / "report.json").read_text(encoding="utf-8")
                )
                expected = {
                    cohort[k][0]["backend"]: {
                        "report_sha256": sha256(paths[k] / "report.json"),
                        "trajectory_sha256": sha256(paths[k] / "trajectory.npz"),
                    }
                    for k in ("sparse", name)
                }
                if (
                    report["odometry"] != expected
                    or report["mapping_config"] != asdict(config)
                    or report["offset"] != 5
                    or report["render"]["radii_m"] != [0.01]
                ):
                    raise ValueError(
                        "saved fusion evaluation inputs or settings changed"
                    )
            else:
                report = run_heldout(
                    [paths["sparse"], paths[name]], output, config=config
                )
            result["fusion_validation"][name] = {
                "status": "evaluated",
                "report_sha256": sha256(output / "report.json"),
                "summaries": report["summaries"],
            }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
