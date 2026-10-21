"""Known-model support-budget control, separate from real-data policy tuning.

More accurate initialization can contain information missing from a reduced
bundle objective. Paired initializations and sensor seeds test this effect
without marginalization, wrong identities, calibration error or correlated noise.
"""

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from slam_generalization import reference_errors, seed_from_first_depth, split_tracks
from slam_model_controls import generate

from scene_recall.multiview.batch_ba import BatchConfig, solve_batch
from scene_recall.run_metadata import code_hash, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    args.output_dir.mkdir(parents=True)
    config = BatchConfig(
        loss="linear",
        max_iterations=40,
        max_point_displacement_m=1,
        max_translation_m=1,
        max_rotation_deg=30,
    )
    sources = {
        str(Path(__file__)): sha256(Path(__file__)),
        **{
            name: sha256(Path(__file__).with_name(name))
            for name in ("slam_model_controls.py", "slam_generalization.py")
        },
    }
    protocol = {
        "phase": "frozen before results",
        "config": asdict(config),
        "sources": sources,
        "package_sha256": code_hash(),
        "budgets": [96, 384, 1536],
        "initial_sigmas_m": [0.004, 0.02],
        "seeds": list(range(5)),
        "geometries": ["near", "far", "plane"],
        "motion": "mixed",
        "scenario": "independent",
        "training_fraction": 0.75,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    results = []
    for geometry, sigma, count, seed in product(
        protocol["geometries"],
        protocol["initial_sigmas_m"],
        protocol["budgets"],
        protocol["seeds"],
    ):
        initial, factors, K, truth = generate(
            seed, geometry, "mixed", "independent", tracks=count, initial_sigma=sigma
        )
        train, _ = split_tracks(seed_from_first_depth(factors, initial))
        started = perf_counter()
        fitted, _, diagnostic = solve_batch(initial, train, K, config)
        results.append(
            {
                "geometry": geometry,
                "initial_sigma_m": sigma,
                "tracks": count,
                "seed": seed,
                "train_tracks": len(np.unique(train.track_id)),
                "initial": reference_errors(initial, list(truth)),
                "final": reference_errors(fitted, list(truth)),
                "solver": diagnostic,
                "seconds": perf_counter() - started,
            }
        )
        if seed == protocol["seeds"][-1]:
            print(geometry, sigma, count, flush=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    if (
        sources
        != {
            name: sha256(
                Path(name)
                if Path(name).is_absolute()
                else Path(__file__).with_name(name)
            )
            for name in sources
        }
        or protocol["package_sha256"] != code_hash()
    ):
        raise RuntimeError("source changed during evaluation")


if __name__ == "__main__":
    main()
