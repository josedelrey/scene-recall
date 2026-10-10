"""Run selectable RGB-D odometry and reference evaluation on ScanNet or TUM."""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

import scene_recall
from scene_recall.datasets.scannet import ScanNetReader
from scene_recall.datasets.tum import TUMConfig, TUMReader
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.hybrid import HybridConfig, HybridRGBDBackend
from scene_recall.odometry.icp import DepthICPBackend, ICPConfig
from scene_recall.odometry.pipeline import TrajectoryFrame, track_observations
from scene_recall.odometry.sparse import SparseConfig, SparseRGBDBackend


def _pose_array(poses) -> np.ndarray:
    return np.array(
        [np.full((4, 4), np.nan) if pose is None else pose for pose in poses],
        dtype=np.float64,
    ).reshape(-1, 4, 4)


def save_run(
    output: Path, rows: list[TrajectoryFrame], report: dict, start: int
) -> None:
    """Write poses, per-frame diagnostics, and a strict JSON evaluation report."""
    # Serialize first so invalid report values cannot leave partial output files.
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "trajectory.npz",
        frame_index=np.array([row.frame_index for row in rows], dtype=np.int64),
        source_frame_index=np.array(
            [start + row.frame_index for row in rows], dtype=np.int64
        ),
        timestamp_s=np.array(
            [np.nan if row.timestamp is None else row.timestamp for row in rows]
        ),
        segment_id=np.array([row.segment_id for row in rows], dtype=np.int64),
        status=np.array([row.status for row in rows], dtype=str),
        T_SC_D=_pose_array(row.T_SC_D for row in rows),
        T_WC_D_reference=_pose_array(row.reference_pose for row in rows),
        reference_present=np.array(
            [row.reference_pose is not None for row in rows], dtype=bool
        ),
        T_C1C0=_pose_array(
            None if row.pair is None else row.pair.T_C1C0 for row in rows
        ),
        candidate_T_C1C0=_pose_array(
            None if row.pair is None else row.pair.candidate_T_C1C0 for row in rows
        ),
        ransac_T_C1C0=_pose_array(
            None if row.pair is None else row.pair.ransac_T_C1C0 for row in rows
        ),
        initial_T_C1C0=_pose_array(
            None if row.pair is None else row.pair.initial_T_C1C0 for row in rows
        ),
    )
    columns = [
        "frame_index",
        "source_frame_index",
        "timestamp_s",
        "segment_id",
        "status",
        "reason",
        "segment_x_m",
        "segment_y_m",
        "segment_z_m",
        "source_keypoints",
        "target_keypoints",
        "ratio_matches",
        "unique_matches",
        "depth_valid_matches",
        "ransac_inliers",
        "inliers",
        "inlier_ratio",
        "source_coverage",
        "target_coverage",
        "refinement",
        "inlier_rmse_px",
    ]
    columns += sorted(
        {key for row in rows if row.pair is not None for key in row.pair.diagnostics}
        - set(columns)
    )
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **({} if row.pair is None else row.pair.diagnostics),
                    "frame_index": row.frame_index,
                    "source_frame_index": start + row.frame_index,
                    "timestamp_s": row.timestamp,
                    "segment_id": row.segment_id,
                    "status": row.status,
                    "reason": None if row.pair is None else row.pair.reason,
                    **dict(
                        zip(
                            ("segment_x_m", "segment_y_m", "segment_z_m"),
                            row.T_SC_D[:3, 3],
                        )
                    ),
                }
            )
    (output / "report.json").write_text(serialized, encoding="utf-8")


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(scene_recall.__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--sens-path", type=Path)
    sources.add_argument("--tum-path", type=Path)
    parser.add_argument("--tum-max-rgb-depth-difference-s", type=float, default=0.02)
    parser.add_argument("--tum-max-ground-truth-difference-s", type=float, default=0.05)
    parser.add_argument(
        "--rpe-time-intervals-s", type=float, nargs="+", default=[0.1, 0.5, 1.0]
    )
    parser.add_argument("--rpe-time-tolerance-s", type=float, default=0.02)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("sparse", "icp", "hybrid"), default="sparse"
    )
    parser.add_argument("--icp-max-iterations", type=int, default=35)
    parser.add_argument("--icp-min-support", type=int, default=30)
    parser.add_argument("--icp-min-support-fraction", type=float, default=0.2)
    parser.add_argument(
        "--hybrid-refinement-failure", choices=("sparse", "lost"), default="sparse"
    )
    parser.add_argument("--hybrid-max-correction-m", type=float, default=0.1)
    parser.add_argument("--hybrid-max-correction-deg", type=float, default=5.0)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--threshold-px", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=0)
    parser.add_argument("--min-correspondences", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--depth-edge-threshold-m", type=float, default=0.05)
    parser.add_argument("--no-depth-edge-filter", action="store_true")
    parser.add_argument("--no-mutual", action="store_true")
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--rpe-intervals", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()
    if args.start < 0 or (args.stop is not None and args.stop < args.start):
        parser.error("require 0 <= start <= stop")
    if any(value <= 0 for value in args.rpe_intervals) or len(
        set(args.rpe_intervals)
    ) != len(args.rpe_intervals):
        parser.error("RPE intervals must be unique positive integers")
    if (
        any(not np.isfinite(value) or value <= 0 for value in args.rpe_time_intervals_s)
        or len(set(args.rpe_time_intervals_s)) != len(args.rpe_time_intervals_s)
        or not np.isfinite(args.rpe_time_tolerance_s)
        or args.rpe_time_tolerance_s < 0
    ):
        parser.error(
            "time RPE requires unique positive finite intervals and nonnegative finite tolerance"
        )
    try:
        tum_config = TUMConfig(
            args.tum_max_rgb_depth_difference_s, args.tum_max_ground_truth_difference_s
        )
        config = SparseConfig(
            ratio=args.ratio,
            mutual_matching=not args.no_mutual,
            max_features=args.max_features,
            iterations=args.iterations,
            confidence=args.confidence,
            reprojection_threshold_px=args.threshold_px,
            seed=args.seed,
            refine=not args.no_refine,
            min_correspondences=args.min_correspondences,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            min_coverage=args.min_coverage,
            depth_edge_threshold_m=None
            if args.no_depth_edge_filter
            else args.depth_edge_threshold_m,
        )
        icp_config = ICPConfig(
            max_iterations=args.icp_max_iterations,
            min_support=args.icp_min_support,
            min_support_fraction=args.icp_min_support_fraction,
        )
        if args.backend == "sparse":
            backend = SparseRGBDBackend(config)
        elif args.backend == "icp":
            config = icp_config
            backend = DepthICPBackend(config)
        else:
            config = HybridConfig(
                config,
                icp_config,
                args.hybrid_refinement_failure,
                args.hybrid_max_correction_m,
                args.hybrid_max_correction_deg,
            )
            backend = HybridRGBDBackend(config)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    output = args.output_dir.expanduser()
    if any(
        (output / name).exists()
        for name in ("trajectory.npz", "frames.csv", "report.json")
    ):
        parser.error("output already contains a run, choose a new output directory")
    if output.exists() and not output.is_dir():
        parser.error("output must be a directory")
    source = (args.sens_path or args.tum_path).expanduser().resolve()
    source_stat = source.stat()
    code_fingerprint = _code_fingerprint()
    cv2.setNumThreads(1)
    started = perf_counter()
    reader_factory = (
        ScanNetReader(source, start=args.start, stop=args.stop)
        if args.sens_path
        else TUMReader(source, start=args.start, stop=args.stop, config=tum_config)
    )
    input_digest = hashlib.sha256()
    provenance = []

    def record_inputs(observations):
        for observation in observations:
            metadata = dict(observation.provenance or {})
            provenance.append(metadata)
            input_digest.update(
                json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
            )
            for array in (observation.rgb, observation.depth, observation.T_WC_D):
                input_digest.update(b"none" if array is None else array.tobytes())
            yield observation

    with reader_factory as reader:
        calibration = reader.calibration
        sequence_id, source_count = reader.sequence_id, reader.frame_count
        source_metadata = dict(getattr(reader, "metadata", {}))
        rows = []
        failures = 0
        for row in track_observations(record_inputs(reader), calibration, backend):
            rows.append(row)
            failures += row.status == "lost"
            if len(rows) % 100 == 0:
                print(
                    f"Processed {len(rows)} frames, tracking failures: {failures}",
                    flush=True,
                )
    elapsed = perf_counter() - started
    evaluation = evaluate_trajectory(
        rows,
        args.rpe_intervals,
        time_intervals_s=args.rpe_time_intervals_s,
        time_tolerance_s=args.rpe_time_tolerance_s,
    )
    report = {
        "format_version": 2,
        "backend": args.backend,
        "initial_pose_stage": {
            "sparse": "ransac",
            "icp": None,
            "hybrid": "accepted_sparse",
        }[args.backend],
        "config": asdict(config),
        "source": {
            "dataset": "scannet_v2" if args.sens_path else "tum_rgbd",
            "sequence_id": sequence_id,
            "adapter": source_metadata,
            "association_config": None if args.sens_path else asdict(tum_config),
            "canonical_input_sha256": input_digest.hexdigest(),
            "path": str(source),
            "size_bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "source_frame_count": source_count,
            "start": args.start,
            "stop": source_count if args.stop is None else args.stop,
        },
        "calibration": {
            "K_R": calibration.K_R.tolist(),
            "K_D": calibration.K_D.tolist(),
            "T_RD": calibration.T_RD.tolist(),
            "rgb_shape": calibration.rgb_shape,
            "depth_shape": calibration.depth_shape,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "code_sha256": code_fingerprint,
            "invocation": sys.argv,
        },
        "timing": {
            "tracking_seconds": elapsed,
            "frames_per_second": len(rows) / elapsed if elapsed else None,
        },
        "timestamp_available_frames": sum(row.timestamp is not None for row in rows),
        "evaluation": evaluation,
        "frames": [
            {
                "frame_index": row.frame_index,
                "source_frame_index": args.start + row.frame_index,
                "provenance": provenance[row.frame_index],
                "segment_id": row.segment_id,
                "status": row.status,
                "reason": None if row.pair is None else row.pair.reason,
                "diagnostics": {} if row.pair is None else row.pair.diagnostics,
            }
            for row in rows
        ],
    }
    save_run(output, rows, report, args.start)
    print(
        f"Frames: {len(rows)}, tracked transitions: {evaluation['tracked_transitions']}/{evaluation['transition_count']}"
    )
    print(
        f"Failed transitions: {evaluation['failed_transitions']}, independent segments: {evaluation['segment_count']}"
    )
    for interval, metrics in evaluation["rpe_by_frame_interval"].items():
        translation, rotation = metrics["translation_m"], metrics["rotation_deg"]
        if translation["count"]:
            print(
                f"RPE {interval} frames ({translation['count']} pairs): RMSE {1000 * translation['rmse']:.3f} mm, {rotation['rmse']:.3f} deg"
            )
        else:
            print(f"RPE {interval} frames: unavailable")
    print(f"Run saved to {output}")


if __name__ == "__main__":
    main()
