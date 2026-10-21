"""Reproducible backend stress controls with known spatial relationships.

Truth generates controlled measurements and scores results. It is absent from
all graph solver inputs. These are backend controls, not end-to-end SLAM scores.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.pose_graph import MetricPoseGraph, PoseGraphConfig


def pose(x, angle=0):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.array([0.0, angle / 3, angle]))[0]
    result[:3, 3] = [x, np.sin(x) / 4, np.cos(x) / 10 - 0.1]
    return result


def run(output):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    output.mkdir(parents=True)
    protocol = {
        "seed": 0,
        "config": asdict(PoseGraphConfig()),
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "scope": "synthetic backend controls, no image matching or dataset tuning",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    truth = np.array([pose(i / 10, 0.03 * i) for i in range(30)])
    local = np.array([pose(i * 0.103, 0.031 * i) for i in range(30)])
    base_loops = [
        (i, i + 20, np.linalg.inv(truth[i + 20]) @ truth[i], (0, 1)) for i in range(8)
    ]
    scenarios = {
        "clean": (truth, local, np.zeros(30, dtype=int), base_loops),
        "false_cycles": (
            truth,
            local,
            np.zeros(30, dtype=int),
            base_loops + [(2, 25, pose(-8, 1), (2, 3)), (4, 27, pose(6, -1), (4, 5))],
        ),
        "coherent_scale_bias": (truth, truth.copy(), np.zeros(30, dtype=int), []),
    }
    biased = scenarios["coherent_scale_bias"][1]
    biased[:, :3, 3] *= 1.1
    scenarios["coherent_scale_bias"][3].extend(
        (i, i + 20, np.linalg.inv(biased[i + 20]) @ biased[i], (0, 1)) for i in range(8)
    )
    disconnected = truth.copy()
    disconnected[15:] = np.linalg.inv(truth[15]) @ truth[15:]
    scenarios["disconnected"] = truth, disconnected, np.repeat([0, 1], 15), []
    scenarios["correct_bridge"] = (
        truth,
        disconnected,
        np.repeat([0, 1], 15),
        [
            (0, 15, np.linalg.inv(truth[15]), (0, 1)),
            (1, 16, np.linalg.inv(truth[16]) @ truth[1], (0, 1)),
        ],
    )
    scenarios["false_bridge_without_cycle_support"] = (
        truth,
        disconnected,
        np.repeat([0, 1], 15),
        [(0, 15, pose(-7), (0, 1))],
    )
    reports, packets = {}, {}
    for name, (expected, initial, segments, loops) in scenarios.items():
        reports[name] = {}
        for robust in (False, True):
            graph = MetricPoseGraph()
            for i, current in enumerate(initial):
                graph.add_node(i, int(segments[i]), current)
                if i and segments[i] == segments[i - 1]:
                    graph.add_edge(i - 1, i, np.linalg.inv(current) @ initial[i - 1])
            for a, b, transform, group in loops:
                graph.add_edge(a, b, transform, kind="loop", group=group)
            diagnostic = graph.solve(robust=robust)
            errors = []
            for i in range(len(initial)):
                root = graph.components[i]
                alignment = graph.poses[root] @ np.linalg.inv(expected[root])
                errors.append(pose_error(graph.poses[i], alignment @ expected[i]))
            reports[name]["robust" if robust else "linear"] = {
                "translation_m": error_statistics(e[0] for e in errors),
                "rotation_deg": error_statistics(e[1] for e in errors),
                "root_frames": sum(c == 0 for c in graph.components.values()),
                "diagnostic": diagnostic,
            }
            packets[name + ("_robust" if robust else "_linear")] = np.array(
                list(graph.poses.values())
            )
        packets[name + "_truth"] = expected
        packets[name + "_local"] = initial
        packets[name + "_segments"] = segments
        packets[name + "_source"] = np.array([v[0] for v in loops], dtype=np.int64)
        packets[name + "_target"] = np.array([v[1] for v in loops], dtype=np.int64)
        packets[name + "_measurements"] = np.array([v[2] for v in loops]).reshape(
            -1, 4, 4
        )
    np.savez_compressed(output / "packets.npz", **packets)
    (output / "report.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False) + "\n"
    )
    if protocol["package_sha256"] != code_hash() or protocol["driver_sha256"] != sha256(
        Path(__file__)
    ):
        raise RuntimeError("source changed during controls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir)
