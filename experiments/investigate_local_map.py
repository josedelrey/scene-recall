"""Frozen, experiment-only reliability controls with exact legacy replay checks.

Run one fresh output directory per source window. No production registration or
threshold change is made. Each alternative receives its own online map history.
Cached feature arrays are immutable measurements shared across alternatives.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from incremental_local_map import common_errors
from local_map_measurements import MeasurementConfig, MeasurementMapBackend
from refinement_tsdf import evaluate_tsdf

from scene_recall.features import extract_rgbd_features
from scene_recall.mapping.heldout import run_heldout
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig
from scene_recall.run_metadata import sha256

POLICIES = {
    "legacy": MeasurementConfig(),
    "rgb": MeasurementConfig(objective="rgb"),
    "rgbd": MeasurementConfig(objective="rgbd"),
    "family": MeasurementConfig(appearance="family"),
}


def fingerprints():
    root = Path(__file__).parent
    return {
        p.name: sha256(p)
        for p in (
            Path(__file__),
            root / "local_map_measurements.py",
            root / "refinement_tsdf.py",
            root / "incremental_local_map.py",
        )
    }


def exact_replay(old, new):
    """Compare every archive array and shared non-timing diagnostic exactly."""
    with np.load(old / "trajectory.npz") as a, np.load(new / "trajectory.npz") as b:
        arrays = {
            k: bool(np.array_equal(a[k], b[k], equal_nan=True))
            if a[k].dtype.kind not in "US"
            else bool(np.array_equal(a[k], b[k]))
            for k in a.files
        }
    first = json.loads((old / "report.json").read_text())
    second = json.loads((new / "report.json").read_text())
    numeric = []
    for a, b in zip(first["frames"], second["frames"], strict=True):
        numeric.append(
            all(
                value == b["diagnostics"].get(key)
                for key, value in a["diagnostics"].items()
                if not key.endswith("_seconds")
            )
        )
    return {
        "arrays": arrays,
        "shared_diagnostics_exact": all(numeric),
        "source_exact": first["source"] == second["source"],
        "calibration_exact": first["calibration"] == second["calibration"],
        "old_report_sha256": sha256(old / "report.json"),
        "old_archive_sha256": sha256(old / "trajectory.npz"),
    }


def run_case(
    source, output, *, dataset, start, stop, policies, baseline=None, maps=False
):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    output.mkdir(parents=True)
    identity = fingerprints()
    protocol = {
        "source": str(source.resolve()),
        "dataset": dataset,
        "start": start,
        "stop": stop,
        "policies": {
            k: asdict(SparseConfig(max_features=1500) if k == "budget" else POLICIES[k])
            for k in policies
        },
        "implementation": identity,
        "assumptions": [
            "fixed camera poses, no shared pose covariance",
            "one-pixel RGB noise and axial sigma .005 + .003 z^2",
            "static RGB-depth exposure and canonical RGB-ray sampling",
            "Cauchy 2.5, ten iterations, 10cm and 5deg bounds from prior pose-only experiment",
        ],
        "fusion": asdict(MappingConfig(pixel_stride=8, max_keyframes_per_submap=24)),
        "tsdf_chunk": 0,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    cache = {}
    active_backend = []

    def factory(config):
        backend = MeasurementMapBackend(config)
        active_backend[:] = [backend]
        return backend

    def extract(observation, calibration, sift, threshold):
        i = observation.frame_index
        if i not in cache:
            f = extract_rgbd_features(observation, calibration, sift, threshold)
            for a in (f.pixels, f.points_R, f.descriptors):
                if a is not None:
                    a.setflags(write=False)
            cache[i] = f
        return cache[i]

    def publish(path, rows, report, offset):
        report["experiment"] = {
            "implementation": identity,
            "feature_cache": "immutable per-frame measurements shared across policies",
        }
        save_run(path, rows, report, offset)
        if report["backend"] == "local_map_recent":
            np.savez_compressed(
                path / "measurement_packets.npz",
                **{
                    f"{i}_{k}": v
                    for i, packet in enumerate(active_backend[0].packets)
                    for k, v in packet.items()
                },
            )

    with (
        patch("scene_recall.odometry.local_map.extract_rgbd_features", extract),
        patch("scene_recall.odometry.sparse.extract_rgbd_features", extract),
        patch("scene_recall.odometry.runner.save_run", publish),
    ):
        for name in policies:
            print(f"Policy {name}", flush=True)
            if name == "budget":
                run_odometry(
                    source,
                    output / name,
                    dataset=dataset,
                    backend_name="sparse",
                    config=SparseConfig(max_features=1500),
                    start=start,
                    stop=stop,
                    rpe_intervals=(1, 5, 10, 20, 50, 100),
                )
            else:
                with patch(
                    "scene_recall.odometry.runner.RecentMapRGBDBackend",
                    factory,
                ):
                    run_odometry(
                        source,
                        output / name,
                        dataset=dataset,
                        backend_name="local_map_recent",
                        config=POLICIES[name],
                        start=start,
                        stop=stop,
                        rpe_intervals=(1, 5, 10, 20, 50, 100),
                    )
    result = {"protocol_sha256": sha256(output / "protocol.json"), "variants": {}}
    if baseline:
        result["legacy_replay"] = exact_replay(baseline, output / "legacy")
        if (
            not all(result["legacy_replay"]["arrays"].values())
            or not result["legacy_replay"]["shared_diagnostics_exact"]
        ):
            raise ValueError("legacy baseline changed")
    loaded = {n: load_trajectory(output / n) for n in policies}
    for name, (report, rows) in loaded.items():
        reference = loaded["legacy"][1]
        result["variants"][name] = {
            "evaluation": report["evaluation"],
            "common_with_legacy": {
                str(k): common_errors(reference, rows, k)
                for k in (1, 5, 10, 20, 50, 100)
            },
            "report_sha256": sha256(output / name / "report.json"),
            "trajectory_sha256": sha256(output / name / "trajectory.npz"),
        }
        if maps and name != "budget":
            try:
                fusion = run_heldout(
                    [output / "budget", output / name],
                    output / f"maps_{name}",
                    config=MappingConfig(pixel_stride=8, max_keyframes_per_submap=24),
                )
            except ValueError as error:
                if "segmentation" not in str(error):
                    raise
                result["variants"][name]["fusion"] = {
                    "status": "withheld",
                    "reason": str(error),
                }
            else:
                result["variants"][name]["fusion"] = {
                    "status": "evaluated",
                    "report_sha256": sha256(output / f"maps_{name}" / "report.json"),
                }
                result["variants"][name]["tsdf"] = evaluate_tsdf(
                    {"budget": output / "budget", name: output / name},
                    ["budget", name],
                    output / f"tsdf_{name}",
                    fusion,
                )
    if fingerprints() != identity:
        raise ValueError("implementation changed during experiment")
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    # Cache is an audit artifact, not a reusable source with independent identity.
    np.savez_compressed(
        output / "features.npz",
        **{
            f"{i}_{k}": getattr(f, k)
            for i, f in cache.items()
            for k in ("pixels", "points_R", "descriptors")
            if getattr(f, k) is not None
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--tum-path", type=Path)
    sources.add_argument("--sens-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("budget", *POLICIES),
        default=["budget", *POLICIES],
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--evaluate-maps", action="store_true")
    args = parser.parse_args()
    if "legacy" not in args.policies or (
        args.evaluate_maps and "budget" not in args.policies
    ):
        parser.error("legacy is required, mapping also requires budget")
    run_case(
        args.tum_path or args.sens_path,
        args.output_dir,
        dataset="tum_rgbd" if args.tum_path else "scannet_v2",
        start=args.start,
        stop=args.stop,
        policies=args.policies,
        baseline=args.baseline,
        maps=args.evaluate_maps,
    )


if __name__ == "__main__":
    main()
