"""Controlled shared-Schur memory and linear-solve scaling, without datasets."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.batch_ba import (
    BatchConfig,
    compact_blocks,
    compact_linearize,
)
from scene_recall.multiview.fixed_landmarks import DirectFactors, direct_cost
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.run_metadata import code_hash
from scene_recall.slam.shared_solver import SchurOperator


def benchmark(cameras, topology):
    rng = np.random.default_rng(310)
    K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]])
    count = 96 if topology == "long_tracks" else 16 * (cameras + 5)
    points = rng.uniform([-0.8, -0.55, 1.6], [0.8, 0.55, 3.2], (count, 3))
    poses = np.repeat(np.eye(4)[None], cameras, axis=0)
    poses[:, 0, 3] = 0.04 * np.sin(np.arange(cameras) / 5)
    views = np.repeat(np.arange(cameras), 96)
    tracks = np.concatenate(
        [
            np.arange(96)
            if topology == "long_tracks"
            else np.arange(16 * i, 16 * i + 96)
            for i in range(cameras)
        ]
    )
    native = np.concatenate(
        [
            transform_points(points[tracks[views == i]], np.linalg.inv(poses[i]))
            for i in range(cameras)
        ]
    )
    pixels = project_points(native, K) + rng.normal(0, 0.2, (len(views), 2))
    factors = DirectFactors(
        views,
        tracks,
        points[tracks],
        pixels,
        native,
        np.repeat(np.eye(4)[None], len(views), axis=0),
        np.repeat((np.eye(3) * 40)[None], len(views), axis=0),
        np.ones(len(views)),
    )
    config = BatchConfig()
    ids, inverse, seeds = landmark_layout(factors)
    started = perf_counter()
    residual, JP, JL = compact_linearize(poses, seeds, factors, K, inverse)
    _, weights = direct_cost(residual, factors, config)
    blocks = compact_blocks(
        residual, JP, JL, weights, inverse, views, len(ids), cameras
    )
    linearized = perf_counter()
    operator = SchurOperator(blocks, views, inverse, (0,), 0.001)
    dp, diagnostic = operator.solve()
    finished = perf_counter()
    return {
        "cameras": cameras,
        "observations": len(views),
        "landmarks": len(ids),
        "topology": topology,
        "linearization_seconds": linearized - started,
        "solve_seconds": finished - linearized,
        "compact_block_bytes": sum(array.nbytes for array in blocks),
        "operator_bytes": sum(
            getattr(operator, name).nbytes
            for name in ("B", "Ci", "b", "rhs", "diagonal", "free", "fixed")
        ),
        "dense_camera_schur_bytes_if_materialized": 8 * (6 * (cameras - 1)) ** 2,
        "max_translation_step_m": float(np.linalg.norm(dp[:, :3], axis=1).max()),
        **diagnostic,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", type=int, nargs="+", default=[8, 32, 128, 512])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh benchmark output")
    protocol = {
        "package_sha256": code_hash(),
        "solver": asdict(BatchConfig()),
        "scope": "synthetic single linearization and checked damped step, not end-to-end SLAM or observability certification",
        "seed": 310,
        "cameras": args.cameras,
    }
    args.output.with_suffix(".protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    results = [
        benchmark(count, topology)
        for topology in ("long_tracks", "chain")
        for count in args.cameras
    ]
    args.output.write_text(
        json.dumps(
            {"protocol": protocol, "results": results}, indent=2, allow_nan=False
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
