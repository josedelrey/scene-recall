"""Complete evaluation of saved reliability trajectories after a baseline-link error.

No trajectory or saved report is rewritten. Raw inputs and frozen settings are
verified before feature-cache reconstruction and the usual fusion evaluation.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from incremental_local_map import common_errors
from investigate_local_map import POLICIES, exact_replay, fingerprints
from refinement_tsdf import evaluate_tsdf

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.features import extract_rgbd_features
from scene_recall.mapping.heldout import run_heldout
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.sparse import SparseConfig
from scene_recall.run_metadata import sha256


def complete_case(path, baseline):
    if (path / "comparison.json").exists() or (path / "features.npz").exists():
        raise ValueError("case already completed or has a feature cache")
    names = ("budget", "legacy", "rgb", "rgbd", "family")
    loaded = {n: load_trajectory(path / n) for n in names}
    identity = fingerprints()
    first, rows = loaded["legacy"]
    for name, (report, _) in loaded.items():
        expected = (
            SparseConfig(max_features=1500) if name == "budget" else POLICIES[name]
        )
        if (
            report["experiment"]["implementation"] != identity
            or report["config"] != asdict(expected)
            or report["source"] != first["source"]
            or report["calibration"] != first["calibration"]
        ):
            raise ValueError("saved tracking settings or implementation differ")
    replay = exact_replay(baseline, path / "legacy")
    if (
        not all(replay["arrays"].values())
        or not replay["shared_diagnostics_exact"]
        or not replay["source_exact"]
        or not replay["calibration_exact"]
    ):
        raise ValueError("corrected baseline is not an exact match")
    reader, _ = open_source(first)
    digest, metadata, arrays = hashlib.sha256(), {}, {}
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=1500)
    with reader:
        validate_source(reader, first)
        for observation in verified_observations(reader, first, rows, digest, metadata):
            f = extract_rgbd_features(observation, reader.calibration, sift)
            for key in ("pixels", "points_R", "descriptors"):
                value = getattr(f, key)
                if value is not None:
                    arrays[f"{observation.frame_index}_{key}"] = value
    if digest.hexdigest() != first["source"]["canonical_input_sha256"]:
        raise ValueError("canonical source changed")
    np.savez_compressed(path / "features.npz", **arrays)
    result = {
        "protocol_sha256": sha256(path / "protocol.json"),
        "legacy_replay": replay,
        "completion": {
            "driver_sha256": sha256(Path(__file__)),
            "corrected_baseline": str(baseline.resolve()),
            "reason": "corrected baseline-directory link using saved source range metadata, tracking artifacts unchanged",
        },
        "variants": {},
    }
    paths = {n: path / n for n in names}
    for name, (report, current) in loaded.items():
        result["variants"][name] = {
            "evaluation": report["evaluation"],
            "common_with_legacy": {
                str(k): common_errors(rows, current, k) for k in (1, 5, 10, 20, 50, 100)
            },
            "report_sha256": sha256(path / name / "report.json"),
            "trajectory_sha256": sha256(path / name / "trajectory.npz"),
        }
        if name != "budget":
            fusion = run_heldout(
                [paths["budget"], paths[name]],
                path / f"maps_{name}",
                config=MappingConfig(pixel_stride=8, max_keyframes_per_submap=24),
            )
            result["variants"][name]["fusion"] = {
                "status": "evaluated",
                "report_sha256": sha256(path / f"maps_{name}/report.json"),
            }
            result["variants"][name]["tsdf"] = evaluate_tsdf(
                paths, ["budget", name], path / f"tsdf_{name}", fusion
            )
    if fingerprints() != identity:
        raise ValueError("implementation changed during completion")
    (path / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    complete_case(args.case, args.baseline)
