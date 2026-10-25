"""Serial raw shared-estimation studies and common-population diagnostics.

Supply a JSON manifest mapping case names to source, dataset, start, stop,
exposure and policies. Each worker is isolated for meaningful peak RSS. Saved
controls can be linked into the case directory before summary or map scoring.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from scene_recall.datasets.tum import TUMConfig
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.evaluation import score_components
from scene_recall.slam.online import OnlineConfig
from scene_recall.slam.runner import run_online

POLICIES = ("control", "shared", "shared_no_loops", "strict", "welding", "landmark")


def worker(case, policy, output):
    config = OnlineConfig(contrast="clahe", feature_scale=0.5, flow_recovery=True)
    config = replace(
        config,
        local=replace(config.local, exposure=case.get("exposure", "timed")),
        shared_estimation=policy in ("shared", "shared_no_loops", "strict"),
        shared=replace(config.shared, require_full_rank=policy == "strict"),
        loops=policy != "shared_no_loops",
        observation_welding=policy in ("welding", "landmark"),
        welding_feedback="landmark" if policy == "landmark" else "off",
    )
    run_online(
        case["source"],
        output,
        dataset=case["dataset"],
        start=case.get("start", 0),
        stop=case.get("stop"),
        config=config,
        association=TUMConfig(
            max_ground_truth_difference_s=case.get("reference_gap_s", 0.05)
        ),
        mapping=MappingConfig(pixel_stride=8),
        invocation=sys.argv,
    )


def summarize(root, manifest):
    results = {}
    for name, case in manifest.items():
        loaded, states = {}, {}
        for policy in case["policies"]:
            run = root / name / policy
            if not (run / "report.json").exists():
                continue
            loaded[policy] = load_trajectory(run)
            states[policy] = {
                stage: dict(np.load(run / f"{stage}.npz"))
                for stage in ("atlas", "atlas_emitted")
            }
        if not loaded:
            continue
        if (
            len(
                {
                    report["source"]["canonical_input_sha256"]
                    for report, _ in loaded.values()
                }
            )
            != 1
        ):
            raise ValueError("comparison source populations differ")
        references = [row.reference_pose for row in next(iter(loaded.values()))[1]]
        common = {
            stage: {
                i
                for i in range(len(references))
                if all(
                    state[stage]["component_id"][i] == state[stage]["component_id"][0]
                    for state in states.values()
                )
            }
            for stage in ("atlas", "atlas_emitted")
        }
        results[name] = {}
        for policy, (report, rows) in loaded.items():
            state = states[policy]
            fits = [
                frame["diagnostics"]["graph"]["shared"][0]
                for frame in report["frames"]
                if frame["diagnostics"].get("graph")
                and "shared" in frame["diagnostics"]["graph"]
            ]
            revision = [
                pose_error(a, b)
                for a, b in zip(
                    state["atlas"]["T_component_depth"],
                    state["atlas_emitted"]["T_component_depth"],
                    strict=True,
                )
            ]
            entry = {
                "source": report["source"],
                "failures": [i for i, row in enumerate(rows) if row.status == "lost"],
                "causal": report["causal_evaluation"],
                "revised": report["revised_evaluation"],
                "common": {
                    stage: {
                        "population": len(common[stage]),
                        "scores": score_components(
                            state[stage]["T_component_depth"],
                            state[stage]["component_id"],
                            references,
                            common[stage],
                        ),
                    }
                    for stage in common
                },
                "emitted_to_final_translation_m": error_statistics(
                    error[0] for error in revision
                ),
                "emitted_to_final_rotation_deg": error_statistics(
                    error[1] for error in revision
                ),
                "shared_statuses": dict(Counter(fit["status"] for fit in fits)),
                "shared_reasons": dict(Counter(fit.get("reason") for fit in fits)),
                "final_shared_solve": fits[-1] if fits else None,
                "timing": report["timing"],
                "capacity_reached": report["capacity_reached"],
                "report_sha256": sha256(root / name / policy / "report.json"),
            }
            if "control" in states:
                entry["paired_control"] = {}
                for stage in common:
                    candidate, control = state[stage], states["control"][stage]
                    indices = {
                        i
                        for i in range(len(references))
                        if candidate["component_id"][i] == candidate["component_id"][0]
                        and control["component_id"][i] == control["component_id"][0]
                    }
                    entry["paired_control"][stage] = {
                        "indices": sorted(indices),
                        "received_frames": len(indices),
                        "candidate": score_components(
                            candidate["T_component_depth"],
                            candidate["component_id"],
                            references,
                            indices,
                        ),
                        "control": score_components(
                            control["T_component_depth"],
                            control["component_id"],
                            references,
                            indices,
                        ),
                    }
            inliers = [
                frame["diagnostics"].get("frontend", {}).get("map_inliers")
                for frame in report["frames"]
            ]
            entry["map_inliers"] = error_statistics(
                value for value in inliers if value is not None
            )
            artifact = root / name / policy / "shared_estimation.npz"
            if artifact.exists():
                entry["landmarks"] = landmark_diagnostics(artifact, report, references)
            results[name][policy] = entry
    return results


def landmark_diagnostics(path, report, references):
    """Measure native consistency without equating it to external accuracy."""
    from scene_recall.geometry.camera import transform_points

    arrays = dict(np.load(path))
    if "view" not in arrays:
        return {"retained_observations": 0}
    identities = dict(arrays["live_identities"])
    points = dict(zip(arrays["point_ids"], arrays["point_positions"], strict=True))
    poses = dict(zip(arrays["camera_ids"], arrays["depth_poses"], strict=True))
    native_errors, rgb_errors = [], []
    reference_errors, reference_samples = [], {}
    atlas = dict(np.load(path.parent / "atlas.npz"))
    alignments = {}
    for frame, reference in enumerate(references):
        if reference is not None:
            component = int(atlas["component_id"][frame])
            alignments.setdefault(
                component, atlas["T_component_depth"][frame] @ np.linalg.inv(reference)
            )
    K = np.array(report["calibration"]["K_R"])
    for frame, track, pixel, native, exposure in zip(
        arrays["view"],
        arrays["track_id"],
        arrays["pixels"],
        arrays["native_points"],
        arrays["T_RcD"],
        strict=True,
    ):
        identity = identities.get(track)
        if identity not in points:
            continue
        rgb_pose = poses[frame] @ np.linalg.inv(exposure)
        rgb = transform_points(points[identity], np.linalg.inv(rgb_pose))
        if rgb[2] > 0:
            projection = K @ rgb
            rgb_errors.append(np.linalg.norm(projection[:2] / projection[2] - pixel))
        if np.isfinite(native).all():
            prediction = transform_points(points[identity], np.linalg.inv(poses[frame]))
            native_errors.append(np.linalg.norm(prediction - native))
            component = int(atlas["component_id"][frame])
            if references[frame] is not None and component in alignments:
                reference_point = transform_points(
                    native, alignments[component] @ references[frame]
                )
                reference_errors.append(
                    np.linalg.norm(points[identity] - reference_point)
                )
                reference_samples.setdefault((identity, component), []).append(
                    reference_point
                )
    reference_radii = [
        float(
            np.sqrt(np.mean(np.sum((samples - np.mean(samples, axis=0)) ** 2, axis=1)))
        )
        for values in reference_samples.values()
        if len(values) >= 2
        for samples in [np.array(values)]
    ]
    return {
        "retained_observations": len(arrays["view"]),
        "optimized_points": len(points),
        "live_identity_classes": len(set(identities.values())),
        "raw_depth_residual_m": error_statistics(native_errors),
        "raw_rgb_residual_px": error_statistics(rgb_errors),
        "reference_placed_native_discrepancy_m": error_statistics(reference_errors),
        "reference_identity_rms_radius_m": error_statistics(reference_radii),
        "unique_view_feature_owners": len(
            set(zip(arrays["view"], arrays["feature_index"], strict=True))
        )
        == len(arrays["view"]),
        "unique_native_owners": len(
            {
                (int(frame), int(native))
                for frame, native in zip(
                    arrays["view"], arrays["native_index"], strict=True
                )
                if native >= 0
            }
        )
        == int((arrays["native_index"] >= 0).sum()),
        "caveat": "in-sample geometric residuals under fixed calibration and exposure, not external point accuracy",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("CASE", "POLICY"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.worker:
        name, policy = args.worker
        worker(manifest[name], policy, args.root / name / policy)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        protocol = args.root / "execution_protocol.json"
        if not protocol.exists():
            protocol.write_text(
                json.dumps(
                    {
                        "manifest": manifest,
                        "manifest_sha256": sha256(args.manifest),
                        "driver_sha256": sha256(Path(__file__)),
                        "package_sha256": code_hash(),
                        "execution": "serial isolated raw workers, one OpenCV/BLAS/OMP thread",
                    },
                    indent=2,
                )
                + "\n"
            )
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        for name, case in manifest.items():
            if args.cases and name not in args.cases:
                continue
            for policy in case["policies"]:
                if policy not in POLICIES:
                    raise ValueError("unsupported study policy")
                destination = args.root / name / policy
                if (destination / "report.json").exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                print(f"Running {name}/{policy}", flush=True)
                with destination.with_suffix(".log").open("w") as log:
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
