"""Causal incremental policies against the complete shared observation solver.

The control is the complete shared solver. Each case can override observation
budget, exposure, loop activation or relinearization thresholds explicitly.
Runs never overwrite existing evidence and workers use one numerical thread.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from shared_estimation_study import summarize

from scene_recall.datasets.tum import TUMConfig
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.online import OnlineConfig
from scene_recall.slam.runner import run_online

POLICIES = {
    "control": {},
    "selective": {"linearization": "selective"},
    "coarse": {"preconditioner": "coarse"},
    "connected": {"admission": "covisibility"},
    "incremental": {
        "linearization": "selective",
        "preconditioner": "coarse",
        "retry_linear_failure": True,
    },
    "combined": {
        "linearization": "selective",
        "preconditioner": "coarse",
        "admission": "covisibility",
        "retry_linear_failure": True,
    },
}


def configuration(case, policy):
    config = OnlineConfig(
        contrast="clahe", feature_scale=0.5, flow_recovery=True, shared_estimation=True
    )
    shared = replace(config.shared, **POLICIES[policy])
    if "thresholds" in case:
        shared = replace(
            shared,
            relinearization=replace(shared.relinearization, **case["thresholds"]),
        )
    if "depth_model" in case:
        shared = replace(
            shared, solver=replace(shared.solver, depth_model=case["depth_model"])
        )
    return replace(
        config,
        shared=shared,
        loops=case.get("loops", True),
        local=replace(config.local, exposure=case.get("exposure", "timed")),
        welding=replace(
            config.welding,
            max_observations_per_view=case.get("observation_budget", 128),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.worker:
        name, policy = args.worker
        case = manifest[name]
        run_online(
            case["source"],
            args.root / name / policy,
            dataset=case["dataset"],
            start=case.get("start", 0),
            stop=case.get("stop"),
            config=configuration(case, policy),
            association=TUMConfig(
                max_ground_truth_difference_s=case.get("reference_gap_s", 0.05)
            ),
            mapping=MappingConfig(pixel_stride=8),
            invocation=sys.argv,
        )
        return
    args.root.mkdir(parents=True, exist_ok=True)
    protocol = args.root / "execution_protocol.json"
    if not protocol.exists():
        protocol.write_text(
            json.dumps(
                {
                    "package_sha256": code_hash(),
                    "driver_sha256": sha256(Path(__file__)),
                    "summary_driver_sha256": sha256(
                        Path(__file__).with_name("shared_estimation_study.py")
                    ),
                    "manifest_sha256": sha256(args.manifest),
                    "manifest": manifest,
                    "policies": {
                        name: {
                            p: asdict(configuration(case, p)) for p in case["policies"]
                        }
                        for name, case in manifest.items()
                    },
                    "threads": 1,
                    "control": "complete shared raw observation solver, not the older separate local/global estimator",
                },
                indent=2,
            )
            + "\n"
        )
    elif json.loads(protocol.read_text())["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("manifest changed after freezing the protocol")
    if not args.summarize_only:
        environment = {
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        for name, case in manifest.items():
            for policy in case["policies"]:
                destination = args.root / name / policy
                if (destination / "report.json").exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                print(f"Running {name}/{policy}", flush=True)
                with destination.with_suffix(".log").open("x") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--manifest",
                            str(args.manifest.resolve()),
                            "--root",
                            str(args.root.resolve()),
                            "--worker",
                            name,
                            policy,
                        ],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
    (args.root / "summary.json").write_text(
        json.dumps(summarize(args.root, manifest), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
