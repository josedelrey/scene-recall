"""Run selectable RGB-D odometry and reference evaluation on ScanNet or TUM."""

import argparse
import sys
from pathlib import Path

import numpy as np

from scene_recall.datasets.tum import TUMConfig
from scene_recall.odometry.hybrid import HybridConfig
from scene_recall.odometry.icp import ICPConfig
from scene_recall.odometry.local_map import LocalMapConfig
from scene_recall.odometry.recent_map import RecentMapConfig
from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig


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
        "--backend",
        choices=("sparse", "icp", "hybrid", "local_map", "local_map_recent"),
        default="sparse",
    )
    parser.add_argument("--local-map-keyframes", type=int, default=6)
    parser.add_argument("--local-map-landmarks", type=int, default=4000)
    parser.add_argument(
        "--local-map-update", choices=("frozen", "median"), default="frozen"
    )
    parser.add_argument("--local-map-no-fallback", action="store_true")
    parser.add_argument(
        "--recent-map-anchor", choices=("recent", "frozen"), default="recent"
    )
    parser.add_argument(
        "--recent-map-association", choices=("indexed", "exhaustive"), default="indexed"
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
            max_features=(args.max_features or 1500)
            if args.backend in ("local_map", "local_map_recent")
            else args.max_features,
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
        if args.backend == "icp":
            config = icp_config
        elif args.backend == "hybrid":
            config = HybridConfig(
                config,
                icp_config,
                args.hybrid_refinement_failure,
                args.hybrid_max_correction_m,
                args.hybrid_max_correction_deg,
            )
        elif args.backend in ("local_map", "local_map_recent"):
            config_type = (
                RecentMapConfig
                if args.backend == "local_map_recent"
                else LocalMapConfig
            )
            options = (
                {
                    "anchor_policy": args.recent_map_anchor,
                    "association_policy": args.recent_map_association,
                }
                if args.backend == "local_map_recent"
                else {}
            )
            config = config_type(
                sparse=config,
                max_keyframes=args.local_map_keyframes,
                max_landmarks=args.local_map_landmarks,
                landmark_update=args.local_map_update,
                pairwise_fallback=not args.local_map_no_fallback,
                **options,
            )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    try:
        run_odometry(
            args.sens_path or args.tum_path,
            args.output_dir,
            dataset="scannet_v2" if args.sens_path else "tum_rgbd",
            backend_name=args.backend,
            config=config,
            start=args.start,
            stop=args.stop,
            tum_config=tum_config,
            rpe_intervals=tuple(args.rpe_intervals),
            rpe_time_intervals_s=tuple(args.rpe_time_intervals_s),
            rpe_time_tolerance_s=args.rpe_time_tolerance_s,
            invocation=sys.argv,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
