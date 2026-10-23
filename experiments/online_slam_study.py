"""Run isolated raw-stream policies from a frozen manifest and compare cohorts."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.evaluation import score_components

POLICIES = {
    "fast_joint": [
        "--contrast",
        "clahe",
        "--flow-recovery",
        "--feature-scale",
        "0.5",
        "--optimize-local",
    ],
    "multiview_joint": [
        "--contrast",
        "clahe",
        "--flow-recovery",
        "--feature-scale",
        "0.5",
        "--optimize-local",
        "--multiview-registration",
    ],
    "multiview_loops": [
        "--contrast",
        "clahe",
        "--flow-recovery",
        "--feature-scale",
        "0.5",
        "--multiview-registration",
        "--multiview-scope",
        "loops",
    ],
    "multiview": [
        "--contrast",
        "clahe",
        "--flow-recovery",
        "--feature-scale",
        "0.5",
        "--multiview-registration",
    ],
    "multiview_local": [
        "--contrast",
        "clahe",
        "--flow-recovery",
        "--feature-scale",
        "0.5",
        "--multiview-registration",
        "--no-loops",
    ],
    "fast_mapping": [
        "--contrast",
        "clahe",
        "--flow-recovery",
        "--feature-scale",
        "0.5",
        "--incremental-mapping",
    ],
    "fast_local": [
        "--contrast",
        "clahe",
        "--flow-recovery",
        "--feature-scale",
        "0.5",
        "--no-loops",
    ],
    "metric_local": [
        "--contrast",
        "clahe",
        "--flow-recovery",
        "--feature-scale",
        "0.5",
        "--metric-refinement",
        "--no-loops",
    ],
    "metric": [
        "--contrast",
        "clahe",
        "--flow-recovery",
        "--feature-scale",
        "0.5",
        "--metric-refinement",
    ],
    "local": ["--no-recovery", "--no-loops"],
    "recovery": ["--no-loops"],
    "online": [],
    "mean": ["--retrieval", "mean"],
    "native": ["--optimize-local"],
    "exhaustive": ["--retrieval", "exhaustive_votes"],
    "clahe": ["--contrast", "clahe"],
    "flow": ["--contrast", "clahe", "--flow-recovery"],
    "fast": ["--contrast", "clahe", "--flow-recovery", "--feature-scale", "0.5"],
    "half": ["--contrast", "clahe", "--feature-scale", "0.5"],
    "xfeat": [],
    "xfeat_flow": ["--flow-recovery"],
}


def summarize(root, manifest):
    results = {}
    for name, case in manifest.items():
        loaded, arrays, emitted = {}, {}, {}
        for policy in case.get("policies", ["local", "online"]):
            destination = root / name / policy
            if not (destination / "report.json").exists():
                continue
            report, rows = load_trajectory(destination)
            loaded[policy] = report, rows
            with np.load(destination / "atlas.npz") as archive:
                arrays[policy] = archive["T_component_depth"], archive["component_id"]
            with np.load(destination / "atlas_emitted.npz") as archive:
                emitted[policy] = archive["T_component_depth"], archive["component_id"]
        if not loaded:
            continue
        digests = {r["source"]["canonical_input_sha256"] for r, _ in loaded.values()}
        if len(digests) != 1:
            raise ValueError("policy input populations differ")
        references = [row.reference_pose for row in next(iter(loaded.values()))[1]]
        common = {
            i
            for i in range(len(references))
            if all(c[i] == c[0] for _, c in arrays.values())
        }
        common_emitted = {
            i
            for i in range(len(references))
            if all(c[i] == c[0] for _, c in emitted.values())
        }
        results[name] = {}
        for policy, (report, rows) in loaded.items():
            poses, components = arrays[policy]
            scored = [
                edge["reference_error"]
                for edge in report["loop_reference_errors"]
                if edge["reference_error"] is not None
            ]
            solves = [
                frame["diagnostics"]["graph"]
                for frame in report["frames"]
                if frame["diagnostics"]["graph"] is not None
            ]
            solved_count = len(solves[-1]["edge_weights"]) if solves else 0
            rejected = (
                {tuple(pair) for pair in solves[-1]["rejected_loops"]}
                if solves
                else set()
            )
            with np.load(root / name / policy / "constraints.npz") as constraints:
                active = {
                    tuple(pair)
                    for i, pair in enumerate(constraints["endpoints"])
                    if i < solved_count
                    and constraints["kind"][i] == "loop"
                    and tuple(pair) not in rejected
                }
                pending = sum(
                    i >= solved_count and kind == "loop"
                    for i, kind in enumerate(constraints["kind"])
                )
            active_errors = [
                edge["reference_error"]
                for edge in report["loop_reference_errors"]
                if (edge["source"], edge["target"]) in active
                and edge["reference_error"] is not None
            ]
            rescued, lost, rescue_errors = [], [], []
            control_policy = (
                "local"
                if "local" in loaded
                else "online"
                if "online" in loaded
                else None
            )
            control_policy = case.get("control", control_policy)
            if control_policy is not None and control_policy not in loaded:
                raise ValueError("comparison control is missing")
            if control_policy is not None:
                control = loaded[control_policy][1]
                for i in range(1, len(rows)):
                    if control[i].status == "lost" and rows[i].status == "tracked":
                        rescued.append(i)
                        if references[i] is not None and references[i - 1] is not None:
                            rescue_errors.append(
                                pose_error(
                                    np.linalg.inv(rows[i].T_SC_D) @ rows[i - 1].T_SC_D,
                                    np.linalg.inv(references[i]) @ references[i - 1],
                                )
                            )
                    if control[i].status == "tracked" and rows[i].status == "lost":
                        lost.append(i)
            results[name][policy] = {
                "frames": len(rows),
                "failures": report["evaluation"]["failed_transitions"],
                "transition_control_policy": control_policy,
                "largest_emitted_component": max(
                    report["causal_evaluation"]["groups"].items(),
                    key=lambda item: (item[1]["frames"], -int(item[0])),
                ),
                "largest_revised_component": max(
                    report["revised_evaluation"]["groups"].items(),
                    key=lambda item: (item[1]["frames"], -int(item[0])),
                ),
                "rescued_transition_indices": rescued,
                "new_failed_transition_indices": lost,
                "rescued_transition_reference_translation_m": error_statistics(
                    e[0] for e in rescue_errors
                ),
                "rescued_transition_reference_rotation_deg": error_statistics(
                    e[1] for e in rescue_errors
                ),
                "rescued_transitions_outside_10cm_5deg": sum(
                    a > 0.1 or b > 5 for a, b in rescue_errors
                ),
                "rescue_scoring_scope": "final local archive relative poses, identical to causal local poses when optimize is false",
                "causal": report["causal_evaluation"],
                "revised": report["revised_evaluation"],
                "common_revised": score_components(
                    poses, components, references, common
                ),
                "common_root_population": len(common),
                "common_emitted": score_components(
                    *emitted[policy], references, common_emitted
                ),
                "common_emitted_root_population": len(common_emitted),
                "timing": report["timing"],
                "loops": len(report["loop_reference_errors"]),
                "reference_scored_loops": len(scored),
                "loops_outside_10cm_5deg": sum(a > 0.1 or b > 5 for a, b in scored),
                "active_loops": len(active),
                "pending_loops": int(pending),
                "rejected_loops": len(rejected),
                "reference_scored_active_loops": len(active_errors),
                "active_loops_outside_10cm_5deg": sum(
                    a > 0.1 or b > 5 for a, b in active_errors
                ),
                "events": report["event_statuses"],
                "capacity_reached": report["capacity_reached"],
                "report_sha256": sha256(root / name / policy / "report.json"),
            }
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--xfeat-checkout", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.summarize_only:
        summarize(args.output_dir, manifest)
        return
    if args.output_dir.exists():
        raise ValueError("choose a fresh study directory")
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must lie between one and four")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "phase": "frozen before any raw policy run",
                "cases": manifest,
                "package_sha256": code_hash(),
                "driver_sha256": sha256(Path(__file__)),
                "workers": args.workers,
                "timing_scope": "concurrent study profiles, use separate serial processes for latency claims",
            },
            indent=2,
        )
        + "\n"
    )

    def run(item):
        name, case, policy = item
        destination = args.output_dir / name / policy
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_flag = {
            "tum_rgbd": "--tum-path",
            "eth3d_slam": "--eth3d-path",
            "scannet_v2": "--sens-path",
        }[case["dataset"]]
        command = [
            sys.executable,
            "-m",
            "scene_recall.slam.cli",
            source_flag,
            case["source"],
            "--start",
            str(case.get("start", 0)),
            "--output-dir",
            str(destination),
            "--map",
            *POLICIES[policy],
        ]
        if policy.startswith("xfeat"):
            if args.xfeat_checkout is None:
                raise ValueError("XFeat policies require --xfeat-checkout")
            command = [
                sys.executable,
                str(Path(__file__).with_name("xfeat_online.py")),
                "--checkout",
                str(args.xfeat_checkout),
                "--source",
                case["source"],
                "--dataset",
                case["dataset"],
                "--start",
                str(case.get("start", 0)),
                "--output-dir",
                str(destination),
                *POLICIES[policy],
            ]
        if case.get("stop") is not None:
            command.extend(["--stop", str(case["stop"])])
        if case.get("max_reference_gap_s") is not None:
            command.extend(["--max-reference-gap-s", str(case["max_reference_gap_s"])])
        if case.get("intervention"):
            kind, begin, end = case["intervention"]
            command.extend(["--intervention", kind, "--outage", str(begin), str(end)])
        with (destination.parent / (policy + ".log")).open("w") as log:
            result = subprocess.run(
                command,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            )
        print(name, policy, result.returncode, flush=True)
        return name, policy, result.returncode

    jobs = [
        (name, case, policy)
        for name, case in manifest.items()
        for policy in case.get("policies", ["local", "online"])
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(run, jobs))
    (args.output_dir / "outcomes.json").write_text(
        json.dumps(outcomes, indent=2) + "\n"
    )
    summarize(args.output_dir, manifest)
    if any(code for _, _, code in outcomes):
        raise RuntimeError("one or more raw runs failed, inspect retained logs")


if __name__ == "__main__":
    main()
