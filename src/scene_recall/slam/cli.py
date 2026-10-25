"""Run the experimental causal RGB-D SLAM system on raw recordings."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from scene_recall.datasets.tum import TUMConfig
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.slam.online import OnlineConfig
from scene_recall.slam.runner import run_online


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--tum-path", type=Path)
    sources.add_argument("--eth3d-path", type=Path)
    sources.add_argument("--sens-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument(
        "--retrieval",
        choices=("mean", "shortlist_votes", "exhaustive_votes"),
        default="shortlist_votes",
    )
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--no-loops", action="store_true")
    parser.add_argument("--optimize-local", action="store_true")
    parser.add_argument(
        "--local-exposure", choices=("timed", "static"), default="timed"
    )
    parser.add_argument("--map", action="store_true")
    parser.add_argument("--contrast", choices=("raw", "clahe"), default="raw")
    parser.add_argument("--flow-recovery", action="store_true")
    parser.add_argument("--metric-refinement", action="store_true")
    parser.add_argument("--multiview-registration", action="store_true")
    parser.add_argument("--observation-welding", action="store_true")
    parser.add_argument("--shared-estimation", action="store_true")
    parser.add_argument(
        "--welding-feedback",
        choices=("camera", "landmark", "joint"),
        nargs="?",
        const="landmark",
        default="off",
    )
    parser.add_argument("--multiview-scope", choices=("all", "loops"), default="all")
    parser.add_argument("--incremental-mapping", action="store_true")
    parser.add_argument("--feature-scale", type=float, choices=(1.0, 0.5), default=1.0)
    parser.add_argument("--intervention", choices=("rgb", "depth"))
    parser.add_argument("--outage", type=int, nargs=2)
    parser.add_argument("--max-reference-gap-s", type=float, default=0.05)
    args = parser.parse_args()
    if bool(args.intervention) != bool(args.outage):
        parser.error("intervention and outage must be supplied together")
    config = OnlineConfig()
    config = replace(
        config,
        local=replace(
            config.local, optimize=args.optimize_local, exposure=args.local_exposure
        ),
        recovery=not args.no_recovery,
        loops=not args.no_loops,
        retrieval=args.retrieval,
        contrast=args.contrast,
        flow_recovery=args.flow_recovery,
        metric_refinement=args.metric_refinement,
        multiview_registration=args.multiview_registration,
        observation_welding=args.observation_welding,
        shared_estimation=args.shared_estimation,
        welding_feedback=args.welding_feedback,
        multiview_scope=args.multiview_scope,
        incremental_mapping=args.incremental_mapping,
        feature_scale=args.feature_scale,
    )
    source, dataset = (
        (args.tum_path, "tum_rgbd")
        if args.tum_path
        else (args.eth3d_path, "eth3d_slam")
        if args.eth3d_path
        else (args.sens_path, "scannet_v2")
    )
    run_online(
        source,
        args.output_dir,
        dataset=dataset,
        start=args.start,
        stop=args.stop,
        config=config,
        mapping=MappingConfig(pixel_stride=8) if args.map else None,
        association=TUMConfig(max_ground_truth_difference_s=args.max_reference_gap_s),
        intervention=(args.intervention, *args.outage) if args.outage else None,
        invocation=sys.argv,
    )


if __name__ == "__main__":
    main()
