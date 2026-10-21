"""Isolated warm-feature atlas and frozen-factor graph timings in opposite orders."""

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from slam_loop_graph import build_graph, dense_poses, packet, verify_feature_cache

from scene_recall.data import Calibration
from scene_recall.odometry.evaluation import error_statistics
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, peak_rss, sha256
from scene_recall.slam.atlas import AtlasConfig, SubmapAtlas


def run(local_run, features, graph_run, output, count=200):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    verify_feature_cache(local_run, features)
    report, rows = load_trajectory(local_run)
    graph_report, graph_rows = load_trajectory(graph_run / "forest_mean")
    if count < 1 or count > len(rows):
        raise ValueError("prefix length outside recording")
    calibration = Calibration(
        **{
            k: tuple(v) if k.endswith("shape") else np.array(v)
            for k, v in report["calibration"].items()
        }
    )
    output.mkdir(parents=True)
    protocol = {
        "phase": "frozen before isolated timing",
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "replay_helper_sha256": sha256(Path(__file__).with_name("slam_loop_graph.py")),
        "local_report_sha256": sha256(local_run / "report.json"),
        "features_sha256": sha256(features),
        "graph_constraints_sha256": sha256(graph_run / "constraints_votes.npz"),
        "graph_local_archive_sha256": sha256(
            graph_run / "forest_mean" / "trajectory.npz"
        ),
        "count": count,
        "config": asdict(AtlasConfig()),
        "scope": "atlas excludes frontend and feature extraction, graph excludes retrieval, all inputs resident, opposite serial orders",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "machine": platform.machine(),
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    cv2.setNumThreads(1)
    with np.load(features, allow_pickle=False) as cache:
        packets = {i: packet(cache, i) for i in range(count)}
    atlas_results, expected = [], {}
    for order in (("mean", "descriptor_votes"), ("descriptor_votes", "mean")):
        for mode in order:
            atlas = SubmapAtlas(calibration, AtlasConfig(retrieval=mode))
            started, times = perf_counter(), []
            for i, row in enumerate(rows[:count]):
                stamp = perf_counter()
                atlas.observe(
                    i,
                    row.segment_id,
                    row.T_SC_D,
                    packets[i],
                    boundary=row.status != "tracked",
                )
                times.append(perf_counter() - stamp)
            seconds = perf_counter() - started
            current = np.array([atlas.pose(i) for i in range(count)])
            if mode in expected:
                np.testing.assert_array_equal(current, expected[mode])
            expected[mode] = current
            atlas_results.append(
                {
                    "mode": mode,
                    "order": list(order),
                    "wall_seconds": seconds,
                    "frames_per_second_atlas_only": count / seconds,
                    "observe_seconds": error_statistics(times),
                }
            )
    with np.load(graph_run / "constraints_votes.npz", allow_pickle=False) as cached:
        measurements = [
            (int(a), int(b), T, tuple(map(int, group)))
            for a, b, T, group in zip(
                cached["source"],
                cached["target"],
                cached["T_target_source"],
                cached["group"],
                strict=True,
            )
        ]
    graph_results = []
    for order in ((False, True), (True, False)):
        for robust in order:
            started = perf_counter()
            graph, diagnostic = build_graph(graph_rows, measurements, robust)
            poses, components = dense_poses(graph, graph_rows)
            with np.load(
                graph_run
                / ("graph_votes_robust" if robust else "graph_votes_linear")
                / "atlas.npz"
            ) as baseline:
                np.testing.assert_array_equal(poses, baseline["T_component_depth"])
                np.testing.assert_array_equal(components, baseline["component_id"])
            graph_results.append(
                {
                    "robust": robust,
                    "order": list(order),
                    "wall_seconds_with_build_and_dense_export": perf_counter()
                    - started,
                    "solve_seconds": diagnostic["solve_seconds"],
                    "nodes": diagnostic["nodes"],
                    "edges": diagnostic["edges"],
                }
            )
    result = {
        "atlas": atlas_results,
        "graph": graph_results,
        "graph_sequence": graph_report["source"]["sequence_id"],
        "exact_repeat": True,
        "peak_rss_mib": peak_rss(),
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if protocol["package_sha256"] != code_hash() or protocol["driver_sha256"] != sha256(
        Path(__file__)
    ):
        raise RuntimeError("source changed during timing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--graph-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    run(args.local_run, args.features, args.graph_run, args.output_dir, args.count)
