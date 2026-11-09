"""Matched native surface placement across distant visits as a memory proxy.

Reference depth poses define fixed spatial bins for the same samples under all
policies. No semantic object labels or independent surface truth are assumed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.geometry.camera import transform_points
from scene_recall.mapping.voxel import MappingConfig, sample_observation
from scene_recall.odometry.evaluation import error_statistics
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def visit_bins(keys, frames, errors, min_views, min_samples):
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique))
    seen = np.unique(np.c_[inverse, frames], axis=0)
    views = np.bincount(seen[:, 0], minlength=len(unique))
    sums = np.zeros((len(unique), *errors.shape[1:]))
    np.add.at(sums, inverse, errors)
    ready = (views >= min_views) & (counts >= min_samples)
    return {
        tuple(k): x / n
        for k, x, n in zip(unique[ready], sums[ready], counts[ready], strict=True)
    }, len(unique)


def evaluate(reference, candidate, output, *, stride=5, size=0.1, tolerance=0.05):
    if output.exists():
        raise ValueError("choose a fresh revisit report")
    if stride < 1 or size <= 0 or tolerance <= 0:
        raise ValueError("revisit sampling and tolerances must be positive")
    paths = (reference, candidate)
    loaded = [load_trajectory(path) for path in paths]
    source, rows = loaded[0]
    if (
        source["source"]["canonical_input_sha256"]
        != loaded[1][0]["source"]["canonical_input_sha256"]
    ):
        raise ValueError("revisit comparison requires identical inputs")
    atlases = [dict(np.load(path / "atlas.npz")) for path in paths]
    common = [
        i
        for i, row in enumerate(rows)
        if row.reference_pose is not None
        and all(a["component_id"][i] == a["component_id"][0] for a in atlases)
    ]
    early = {i for i in common if i % stride == 0 and i < len(rows) // 3}
    late = {i for i in common if i % stride == 0 and i >= 2 * len(rows) // 3}
    min_views, min_samples = 3, 20
    result = {
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "report_sha256": [sha256(p / "report.json") for p in paths],
        "scope": "same native surface samples in reference spatial bins in first and last thirds, no object ground truth, post-hoc geometry diagnostic",
        "early_frames": sorted(early),
        "late_frames": sorted(late),
        "grid_size_m": size,
        "tolerance_m": tolerance,
        "min_views_per_visit": min_views,
        "min_samples_per_visit": min_samples,
    }
    if not early or not late:
        result.update(status="withheld", reason="no matched root views in one visit")
    else:
        reader, _ = open_source(source)
        anchor = common[0]
        ref_root = np.linalg.inv(rows[anchor].reference_pose)
        roots = [np.linalg.inv(a["T_component_depth"][anchor]) for a in atlases]
        visits = [([], [], []), ([], [], [])]
        with reader:
            validate_source(reader, source)
            for observation in verified_observations(
                reader, source, rows, hashlib.sha256(), {}
            ):
                frame = observation.frame_index
                if frame not in early and frame not in late:
                    continue
                measured = sample_observation(
                    observation, reader.calibration, MappingConfig(pixel_stride=8)
                ).points
                points = transform_points(
                    measured, ref_root @ rows[frame].reference_pose
                )
                differences = np.stack(
                    [
                        transform_points(measured, root @ a["T_component_depth"][frame])
                        - points
                        for root, a in zip(roots, atlases, strict=True)
                    ],
                    axis=1,
                )
                keys, frames, errors = visits[int(frame in late)]
                keys.append(np.floor(points / size).astype(np.int64))
                frames.append(np.full(len(points), frame, dtype=np.int64))
                errors.append(differences)
        data = [
            visit_bins(
                np.concatenate(k),
                np.concatenate(f),
                np.concatenate(e),
                min_views,
                min_samples,
            )
            for k, f, e in visits
        ]
        first, last = (v[0] for v in data)
        selected = sorted(first.keys() & last.keys())
        result.update(
            status="evaluated" if selected else "withheld",
            shared_patches=len(selected),
            bins_seen=[v[1] for v in data],
            supported_patches=[len(first), len(last)],
            anchor=anchor,
        )
        if selected:
            drift = np.linalg.norm(
                np.array([last[k] - first[k] for k in selected]), axis=2
            )
            result["scores"] = {
                name: {
                    "revisit_placement_shift_m": error_statistics(drift[:, i]),
                    "within_tolerance_fraction": float(
                        np.mean(drift[:, i] <= tolerance)
                    ),
                }
                for i, name in enumerate(("reference", "candidate"))
            }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.reference, args.candidate, args.output, stride=args.stride)


if __name__ == "__main__":
    main()
