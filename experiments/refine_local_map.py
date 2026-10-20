"""Frozen-policy comparisons of recent-depth tracking and the original frontend.

Use the same settings for historical regression clips and independent scenes.
References enter only post-tracking evaluation. Backend execution is serial
within a case. Concurrent cases must not be used for runtime ranking.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from incremental_local_map import summarize, variants

from scene_recall.mapping.heldout import check_cohort, run_heldout
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.runner import run_odometry
from scene_recall.run_metadata import code_hash, sha256


def run_case(
    source,
    output,
    *,
    dataset,
    start=0,
    stop=None,
    names=("sparse", "budget", "hybrid", "local", "recent"),
    resume=False,
    evaluate_maps=False,
    tsdf=False,
):
    if (output.exists() or output.is_symlink()) and not resume:
        raise ValueError("choose a fresh case directory or explicitly resume")
    implementation = code_hash(
        Path(__file__).parents[1] / "src/scene_recall/odometry/cli.py"
    )
    choices = variants()
    if (
        "sparse" not in names
        or len(set(names)) != len(names)
        or any(n not in choices for n in names)
    ):
        raise ValueError("variants must be unique, known and include sparse")
    paths = {name: output / name for name in names}
    for name, path in paths.items():
        backend, config = choices[name]
        if path.exists():
            report, _ = load_trajectory(path)
            if (
                report["environment"]["code_sha256"] != implementation
                or report["config"] != asdict(config)
                or report["backend"] != backend
                or report["source"]["path"] != str(source.resolve())
                or report["source"]["dataset"] != dataset
                or report["source"]["start"] != start
                or (stop is not None and report["source"]["stop"] != stop)
            ):
                raise ValueError("resume inputs, settings or implementation differ")
        else:
            print(f"Running {name}", flush=True)
            run_odometry(
                source,
                path,
                dataset=dataset,
                backend_name=backend,
                config=config,
                start=start,
                stop=stop,
                rpe_intervals=(1, 5, 10, 20, 50, 100),
            )
    result = summarize(paths)
    result["refinement_driver_sha256"] = sha256(Path(__file__))
    if evaluate_maps:
        group = [n for n in ("budget", "hybrid", "local", "recent") if n in names]
        if not {"budget", "local", "recent"}.issubset(group):
            raise ValueError("mapping requires budget, local and recent variants")
        loaded = {n: load_trajectory(paths[n]) for n in group}
        config = MappingConfig(pixel_stride=8, max_keyframes_per_submap=24)
        try:
            check_cohort(loaded, config, 5)
        except ValueError as error:
            result["fusion"] = {"status": "withheld", "reason": str(error)}
        else:
            target = output / "maps"
            if target.exists():
                report = json.loads(
                    (target / "report.json").read_text(encoding="utf-8")
                )
                expected = {
                    loaded[n][0]["backend"]: {
                        "report_sha256": sha256(paths[n] / "report.json"),
                        "trajectory_sha256": sha256(paths[n] / "trajectory.npz"),
                    }
                    for n in group
                }
                if report["odometry"] != expected or report["mapping_config"] != asdict(
                    config
                ):
                    raise ValueError("saved fusion inputs differ")
            else:
                report = run_heldout([paths[n] for n in group], target, config=config)
            result["fusion"] = {
                "status": "evaluated",
                "report_sha256": sha256(target / "report.json"),
                "summaries": report["summaries"],
            }
            if tsdf:
                from refinement_tsdf import evaluate_tsdf

                result["tsdf"] = evaluate_tsdf(paths, group, output / "tsdf", report)
    if (
        code_hash(Path(__file__).parents[1] / "src/scene_recall/odometry/cli.py")
        != implementation
    ):
        raise ValueError("implementation changed during comparison")
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--tum-path", type=Path)
    sources.add_argument("--sens-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(variants()),
        default=["sparse", "budget", "hybrid", "local", "recent"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-maps", action="store_true")
    parser.add_argument("--tsdf-first-chunk", action="store_true")
    args = parser.parse_args()
    run_case(
        args.tum_path or args.sens_path,
        args.output_dir,
        dataset="tum_rgbd" if args.tum_path else "scannet_v2",
        start=args.start,
        stop=args.stop,
        names=tuple(args.variants),
        resume=args.resume,
        evaluate_maps=args.evaluate_maps,
        tsdf=args.tsdf_first_chunk,
    )


if __name__ == "__main__":
    main()
