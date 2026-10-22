"""Isolated dense/block graph timing on identical controlled metric chains."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.run_metadata import code_hash, peak_rss
from scene_recall.slam.pose_graph import MetricPoseGraph, PoseGraphConfig


def run(count, solver):
    graph = MetricPoseGraph(
        PoseGraphConfig(linear_solver=solver, max_nodes=max(512, count))
    )
    for i in range(count):
        local = np.eye(4)
        local[:3, :3] = cv2.Rodrigues(np.array([0, 0, 0.0001 * i]))[0]
        local[:3, 3] = [0.011 * i, 0, 0]
        graph.add_node(i, 0, local)
        if i:
            graph.add_edge(i - 1, i, np.linalg.inv(local) @ graph.local[i - 1])
    for target in range(30, count, 30):
        measurement = np.eye(4)
        measurement[0, 3] = -target * 0.01
        graph.add_edge(0, target, measurement, kind="loop", group=(0, target))
    started = perf_counter()
    diagnostic = graph.solve()
    elapsed = perf_counter() - started
    return graph, {
        "nodes": count,
        "edges": len(graph.edges),
        "solver": solver,
        "config": asdict(graph.config),
        "seconds": elapsed,
        "peak_rss_mib": peak_rss(),
        "diagnostic": diagnostic,
        "package_sha256": code_hash(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--solver", choices=("dense", "block_pcg"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("choose a fresh timing directory")
    args.output_dir.mkdir(parents=True)
    graph, report = run(args.nodes, args.solver)
    np.savez_compressed(
        args.output_dir / "poses.npz", poses=np.array(list(graph.poses.values()))
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
