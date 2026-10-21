"""Matched cadence-four reconstruction for unchanged and interpolated poses.

The original cadence-ten maps sample SLAM keyframes. This separate diagnostic
uses intermediate training poses and preserves common native RGB-D and targets.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from refinement_tsdf import evaluate_tsdf
from summarize_local_map_reliability import geometry_scores

from scene_recall.mapping.heldout import run_heldout
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.run_metadata import code_hash, sha256


def evaluate(paths, output):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    output.mkdir(parents=True)
    paths = {key: Path(value) for key, value in paths.items()}
    config = MappingConfig(
        pixel_stride=8, keyframe_stride=4, max_keyframes_per_submap=60
    )
    protocol = {
        "phase": "frozen before map scoring",
        "paths": {k: str(v) for k, v in paths.items()},
        "config": asdict(config),
        "offset": 2,
        "driver_sha256": sha256(Path(__file__)),
        "package_sha256": code_hash(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    fusions, gathered = {}, {}
    for name, run in paths.items():
        fusions[name] = run_heldout(
            [run], output / ("surface_" + name), config=config, offset=2
        )
        blocks = []
        for path in sorted(
            (output / ("surface_" + name)).glob("views_*_radius_0.01.npz")
        ):
            with np.load(path, allow_pickle=False) as archive:
                if len(archive["evaluated_frame_indices"]):
                    blocks.append(
                        {
                            key: archive[key].copy()
                            for key in (
                                "observed",
                                "incremental_slam_reference_pose",
                                "gt",
                                "evaluated_frame_indices",
                            )
                        }
                    )
        if blocks:
            gathered[name] = {
                key: np.concatenate([b[key] for b in blocks]) for key in blocks[0]
            }
    fusion = fusions["native"]
    if any(
        f["split"] != fusion["split"]
        or f["verified_canonical_input_sha256"]
        != fusion["verified_canonical_input_sha256"]
        for f in fusions.values()
    ):
        raise ValueError("interpolation reconstruction populations differ")
    surface = {"status": "withheld", "reason": "no paired reference-target population"}
    if len(gathered) == len(paths):
        arrays = gathered["native"]
        for packet in gathered.values():
            for key in ("observed", "gt", "evaluated_frame_indices"):
                if not np.array_equal(arrays[key], packet[key], equal_nan=True):
                    raise ValueError("interpolation controls or targets differ")
        surface = {
            "status": "evaluated",
            "evaluated_views": len(arrays["observed"]),
            "scores": geometry_scores(
                arrays["observed"],
                {
                    **{
                        name: packet["incremental_slam_reference_pose"]
                        for name, packet in gathered.items()
                    },
                    "gt": arrays["gt"],
                },
            ),
        }
    tsdf = evaluate_tsdf(paths, list(paths), output / "tsdf", fusion)
    result = {"surface": surface, "tsdf": tsdf}
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if (
        protocol["driver_sha256"] != sha256(Path(__file__))
        or protocol["package_sha256"] != code_hash()
    ):
        raise RuntimeError("source changed during evaluation")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        type=Path,
        help="JSON case mapping to native and interpolated run paths",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    args.output_dir.mkdir(parents=True)
    cases = json.loads(args.cases.read_text())
    results = {}
    for name, paths in cases.items():
        results[name] = evaluate(paths, args.output_dir / name)
        print(
            name,
            results[name]["surface"]["status"],
            results[name]["tsdf"]["status"],
            flush=True,
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
