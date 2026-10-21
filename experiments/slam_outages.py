"""Controlled missing-depth, black-RGB and biased-depth SLAM interventions.

Inputs are transformed explicitly after canonical reading. These diagnostic
outputs are not version-2 mapping runs. Interventions test mechanisms and do not
estimate natural tracking failure probabilities.
"""

import argparse
import hashlib
import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from incremental_slam import with_revised_poses

from scene_recall.datasets.tum import TUMReader
from scene_recall.features import RGBDFeatures, extract_rgbd_features
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.pipeline import TrajectoryFrame
from scene_recall.odometry.sparse import SparseConfig, SparseRGBDBackend
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.system import IncrementalSLAM, RevisionArchive, SLAMConfig


def run(source, output, *, configurations=None, engine_factory=IncrementalSLAM):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    output.mkdir(parents=True)
    cv2.setNumThreads(1)
    configurations = configurations or {
        "control": SLAMConfig(optimize=False),
        "native": SLAMConfig(),
    }
    scenarios = ("clean", "missing_depth", "black_rgb", "biased_depth")
    systems = {
        (scenario, name): engine_factory(config)
        for scenario in scenarios
        for name, config in configurations.items()
    }
    archives = {key: RevisionArchive() for key in systems}
    sparse = {
        scenario: SparseRGBDBackend(SparseConfig(max_features=1500))
        for scenario in scenarios
    }
    sparse_poses = {s: np.eye(4) for s in scenarios}
    segments, previous = {s: 0 for s in scenarios}, {}
    rows, diagnostics = {}, {key: [] for key in systems}
    digest = hashlib.sha256()
    protocol = {
        "source": str(source.resolve()),
        "range": [0, 60],
        "interval": [20, 30],
        "depth_scale": 1.5,
        "policies": {n: asdict(c) for n, c in configurations.items()},
        "phase": "frozen before evaluation",
        "implementation": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "engine_factory": f"{engine_factory.__module__}.{engine_factory.__qualname__}",
        "engine_factory_sha256": sha256(Path(inspect.getfile(engine_factory))),
        "feature_policy": "same original SIFT measurements, explicitly invalidated or rescaled by intervention",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    feature_packets = {}

    def extract(observation, *_):
        return feature_packets[
            (observation.provenance["slam_intervention"], observation.frame_index)
        ]

    sift = cv2.SIFT_create(nfeatures=1500)
    with (
        TUMReader(source, stop=60) as reader,
        patch("scene_recall.odometry.local_map.extract_rgbd_features", extract),
        patch("scene_recall.odometry.sparse.extract_rgbd_features", extract),
    ):
        for original in reader:
            i = original.frame_index
            digest.update(
                json.dumps(dict(original.provenance), sort_keys=True).encode()
            )
            digest.update(original.rgb.tobytes())
            digest.update(original.depth.tobytes())
            base = extract_rgbd_features(original, reader.calibration, sift, 0.05)
            for key in list(feature_packets):
                if key[1] < i - 1:
                    del feature_packets[key]
            for scenario in scenarios:
                changed = 20 <= i < 30
                depth, rgb, features = original.depth, original.rgb, base
                if changed and scenario == "missing_depth":
                    depth = np.full_like(depth, np.nan)
                    features = replace(
                        base, points_R=np.full_like(base.points_R, np.nan)
                    )
                elif changed and scenario == "black_rgb":
                    rgb = np.zeros_like(rgb)
                    features = RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))
                elif changed and scenario == "biased_depth":
                    depth = depth * 1.5
                    features = replace(base, points_R=base.points_R * 1.5)
                feature_packets[scenario, i] = features
                clean = replace(
                    original,
                    rgb=rgb,
                    depth=depth,
                    T_WC_D=None,
                    provenance={**original.provenance, "slam_intervention": scenario},
                )
                pair = (
                    None
                    if i == 0
                    else sparse[scenario].estimate(
                        previous[scenario], clean, reader.calibration
                    )
                )
                if pair is not None:
                    if pair.T_C1C0 is None:
                        segments[scenario] += 1
                        sparse_poses[scenario] = np.eye(4)
                    else:
                        sparse_poses[scenario] = sparse_poses[scenario] @ np.linalg.inv(
                            pair.T_C1C0
                        )
                status = (
                    "initialized"
                    if i == 0
                    else "lost"
                    if pair.T_C1C0 is None
                    else "tracked"
                )
                rows.setdefault((scenario, "sparse"), []).append(
                    TrajectoryFrame(
                        i,
                        original.timestamp,
                        segments[scenario],
                        status,
                        sparse_poses[scenario].copy(),
                        original.T_WC_D,
                        pair,
                    )
                )
                previous[scenario] = clean
                for name in configurations:
                    key = scenario, name
                    engine, archive = systems[key], archives[key]
                    pose, pair, revised, diagnostic = engine.process(
                        clean, reader.calibration
                    )
                    archive.append(
                        pose,
                        keyframe=i in engine.frontend.keyframes,
                        boundary=pair is None or pair.T_C1C0 is None,
                    )
                    archive.apply(revised)
                    status = (
                        "initialized"
                        if i == 0
                        else "lost"
                        if pair.T_C1C0 is None
                        else "tracked"
                    )
                    rows.setdefault(key, []).append(
                        TrajectoryFrame(
                            i,
                            original.timestamp,
                            engine.segment_id,
                            status,
                            pose,
                            original.T_WC_D,
                            pair,
                        )
                    )
                    diagnostics[key].append(diagnostic)
            if (i + 1) % 20 == 0:
                print(i + 1, flush=True)
    result = {"canonical_rgbd_sha256": digest.hexdigest(), "scenarios": {}}
    arrays = {}
    for key, history in rows.items():
        scenario, name = key
        if key in archives:
            history = with_revised_poses(history, archives[key].poses)
        result["scenarios"].setdefault(scenario, {})[name] = {
            "evaluation": evaluate_trajectory(history, (1, 10)),
            "loss_indices": [r.frame_index for r in history if r.status == "lost"],
            "graph_restarts": [
                i
                for i, d in enumerate(diagnostics.get(key, []))
                if "graph_restart" in d
            ],
        }
        arrays[scenario + "_" + name] = np.array([r.T_SC_D for r in history])
    np.savez_compressed(output / "trajectories.npz", **arrays)
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.tum_path, args.output_dir)


if __name__ == "__main__":
    main()
