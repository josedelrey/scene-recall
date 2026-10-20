"""Audit preserved baselines and replay saved features without reference poses."""

import argparse
import json
from pathlib import Path

import numpy as np
from local_map_measurements import MeasurementConfig, MeasurementMapBackend

from scene_recall.data import Calibration, Observation
from scene_recall.features import RGBDFeatures
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.pipeline import track_observations
from scene_recall.odometry.sparse import SparseConfig, SparseRGBDBackend
from scene_recall.run_metadata import sha256


def replay_prefix(path, name, count):
    report, saved = load_trajectory(path / name)
    count = min(count, len(saved))
    c = report["calibration"]
    calibration = Calibration(
        np.array(c["K_R"]),
        np.array(c["K_D"]),
        np.array(c["T_RD"]),
        tuple(c["rgb_shape"]),
        tuple(c["depth_shape"]),
    )
    features = {}
    with np.load(path / "features.npz") as a:
        for i in range(count):
            features[i] = RGBDFeatures(
                a[f"{i}_pixels"],
                a.get(f"{i}_descriptors", None),
                a[f"{i}_points_R"],
            )
    config = dict(report["config"])
    if name == "budget":
        backend = SparseRGBDBackend(SparseConfig(**config))
        backend._features = lambda o, c: features[o.frame_index]
    else:
        config["sparse"] = SparseConfig(**config["sparse"])
        backend = MeasurementMapBackend(MeasurementConfig(**config))
        backend._extract = lambda o, c: features[o.frame_index]
    rgb = np.zeros((*calibration.rgb_shape, 3), np.uint8)
    depth = np.ones(calibration.depth_shape, np.float32)
    # Deliberately omit every evaluation reference and use synthetic images.
    observations = [
        Observation(i, rgb, depth, timestamp=saved[i].timestamp) for i in range(count)
    ]
    replay = list(track_observations(observations, calibration, backend))
    poses = np.array_equal(
        [r.T_SC_D for r in saved[:count]], [r.T_SC_D for r in replay]
    )
    statuses = [r.status for r in saved[:count]] == [r.status for r in replay]
    numeric = True
    for i, r in enumerate(replay[1:], start=1):
        expected = report["frames"][i]["diagnostics"]
        numeric &= all(
            v == r.pair.diagnostics.get(k)
            for k, v in expected.items()
            if not k.endswith("_seconds")
        )
    return {
        "frames": count,
        "poses_exact": poses,
        "statuses_exact": statuses,
        "numerical_diagnostics_exact": numeric,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replay-prefix", type=int, default=40)
    args = parser.parse_args()
    protocol = json.loads((args.root / "frozen_protocol.json").read_text())
    root = Path(__file__).parents[1]
    manifest = json.loads((args.root / "protected_baselines.json").read_text())
    preserved = {
        p: (root / p).is_file() and sha256(root / p) == value
        for p, value in manifest.items()
    }
    replay = {}
    for case in protocol["cases"]:
        path = args.root / case["name"]
        replay[case["name"]] = {
            n: replay_prefix(path, n, args.replay_prefix) for n in protocol["policies"]
        }
        print(case["name"], replay[case["name"]], flush=True)
    result = {
        "driver_sha256": sha256(Path(__file__)),
        "protected_files": len(preserved),
        "protected_files_unchanged": all(preserved.values()),
        "changed_protected_files": [p for p, passed in preserved.items() if not passed],
        "replay": replay,
    }
    passed = result["protected_files_unchanged"] and all(
        v["poses_exact"] and v["statuses_exact"] and v["numerical_diagnostics_exact"]
        for case in replay.values()
        for v in case.values()
    )
    result["passed"] = bool(passed)
    (args.root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise ValueError("reliability study verification failed")


if __name__ == "__main__":
    main()
