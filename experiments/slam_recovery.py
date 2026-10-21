"""Verified local fallback and a reference-free submap atlas on canonical replay.

Original local trajectories remain standard archives. The separate atlas export
retains raw segment identities, component gauges and source-frame lineage.
Joining an atlas component never relabels a failed tracking transition as tracked.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
from incremental_slam import with_revised_poses

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.features import RGBDFeatures, extract_rgbd_features
from scene_recall.odometry.evaluation import (
    error_statistics,
    evaluate_trajectory,
    pose_error,
)
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.odometry.pipeline import TrajectoryFrame
from scene_recall.odometry.sparse import estimate_feature_pair
from scene_recall.run_metadata import code_hash, peak_rss, sha256
from scene_recall.slam.atlas import AtlasConfig, SubmapAtlas
from scene_recall.slam.registration import RegistrationConfig, register_features
from scene_recall.slam.system import (
    IncrementalSLAM,
    RevisionArchive,
    RevisionFrontend,
    SLAMConfig,
)


class RecoveryFrontend(RevisionFrontend):
    def __init__(self, config, policy):
        super().__init__(config)
        self.policy = policy

    def _recover_pose(self, first, second, calibration, previous_pose):
        started = perf_counter()
        if self.policy == "low_pnp":
            fit = estimate_feature_pair(
                first,
                second,
                calibration,
                replace(self.config.sparse, min_correspondences=12, min_inliers=8),
            )
            transform = fit.T_C1C0
            diagnostic = {
                "status": "accepted" if transform is not None else "rejected",
                "reason": fit.reason,
                "fit": fit.diagnostics,
            }
        else:
            fit = register_features(first, second, calibration)
            transform = (
                None
                if fit.T_target_source is None
                else np.linalg.inv(calibration.T_RD)
                @ fit.T_target_source
                @ calibration.T_RD
            )
            diagnostic = fit.diagnostics
        diagnostic = {
            **diagnostic,
            "seconds": perf_counter() - started,
            "policy": self.policy,
        }
        if transform is not None:
            diagnostic["T_target_source"] = transform.tolist()
        return (
            None if transform is None else previous_pose @ np.linalg.inv(transform)
        ), diagnostic


def make_system(policy):
    config = SLAMConfig(optimize=policy in ("native", "rgbd_native"))
    engine = IncrementalSLAM(config)
    if policy in ("rgbd", "rgbd_native", "low_pnp"):
        engine.frontend = RecoveryFrontend(
            engine.frontend.config, "low_pnp" if policy == "low_pnp" else "rgbd"
        )
    return engine


def execution_header(report, protocol, timing):
    """Separate this replay's measurements from inherited input-run summaries."""
    current = {**report, "initial_pose_stage": None}
    inherited = {
        key: current.pop(key)
        for key in ("environment", "timing", "measurement_consistency")
        if key in current
    }
    current["input_report_header_context"] = inherited
    current["environment"] = {
        "package_sha256": protocol["package_sha256"],
        "experiment_driver_sha256": protocol["driver_sha256"],
    }
    current["timing"] = {
        "process_and_atlas_seconds": timing,
        "scope": "replay profile, shared extraction when multiple policies, use isolated timing for speed comparisons",
    }
    return current


def atlas_metrics(atlas, references):
    sizes = Counter(atlas.components[segment] for segment, _ in atlas.frames.values())
    groups = {}
    for component, count in sorted(sizes.items()):
        indices = [
            i
            for i, (segment, _) in atlas.frames.items()
            if atlas.components[segment] == component
        ]
        valid = [i for i in indices if references[i] is not None]
        errors = []
        if valid:
            alignment = atlas.pose(valid[0]) @ np.linalg.inv(references[valid[0]])
            errors = [
                pose_error(atlas.pose(i), alignment @ references[i]) for i in valid
            ]
        groups[str(component)] = {
            "frames": count,
            "reference_frames": len(valid),
            "first_frame": min(indices),
            "last_frame": max(indices),
            "anchored_translation_m": error_statistics(v[0] for v in errors),
            "anchored_rotation_deg": error_statistics(v[1] for v in errors),
        }
    verified = []
    for edge in atlas.forest:
        a, b = edge["source"], edge["target"]
        if references[a] is not None and references[b] is not None:
            truth = np.linalg.inv(references[b]) @ references[a]
            error = pose_error(edge["T_target_source"], truth)
            verified.append(
                {
                    "source": a,
                    "target": b,
                    "translation_m": error[0],
                    "rotation_deg": error[1],
                }
            )
    return {
        "component_count": len(sizes),
        "largest_component_frames": max(sizes.values(), default=0),
        "root_component_frames": sizes.get(0, 0),
        "groups": groups,
        "accepted_link_reference_errors": verified,
        "active_links": len(atlas.forest),
        "scored_links": len(verified),
        "unscored_links": len(atlas.forest) - len(verified),
        "links_outside_10cm_5deg": sum(
            e["translation_m"] > 0.1 or e["rotation_deg"] > 5 for e in verified
        ),
    }


def run(baseline, output, policies, features=None, *, outage=None, atlas_enabled=True):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    report, reference_rows = load_trajectory(baseline)
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("policies must be nonempty and unique")
    if outage is not None and not 0 <= outage[0] < outage[1] <= len(reference_rows):
        raise ValueError("outage must be a nonempty half-open input interval")
    cv2.setNumThreads(1)
    engines = {name: make_system(name) for name in policies}
    archives = {name: RevisionArchive() for name in policies}
    atlases, histories = {}, {name: [] for name in policies}
    emitted = {name: [] for name in policies}
    causal = {name: [] for name in policies}
    timings = {name: [] for name in policies}
    cache = np.load(features, allow_pickle=False) if features else None
    output.mkdir(parents=True)
    protocol = {
        "baseline_report_sha256": sha256(baseline / "report.json"),
        "baseline_trajectory_sha256": sha256(baseline / "trajectory.npz"),
        "features_sha256": sha256(features) if features else None,
        "policies": policies,
        "atlas_enabled": atlas_enabled,
        "outage": outage,
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "registration": asdict(RegistrationConfig()),
        "atlas": asdict(AtlasConfig()),
        "local_configs": {
            name: asdict(engine.config) for name, engine in engines.items()
        },
        "low_pnp_override": {"min_correspondences": 12, "min_inliers": 8},
        "phase": "frozen before replay",
        "scope": "local losses retained, atlas gauge connections evaluated separately",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    reader, _ = open_source(report)
    latest = {}

    def extract(observation, calibration, sift, threshold):
        key = observation.frame_index
        if key not in latest:
            latest.clear()
            if outage and outage[0] <= key < outage[1]:
                packet = RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))
            elif cache is None:
                packet = extract_rgbd_features(
                    observation, calibration, sift, threshold
                )
            else:
                packet = RGBDFeatures(
                    cache[f"{key}_pixels"],
                    cache.get(f"{key}_descriptors"),
                    cache[f"{key}_points_R"],
                )
            latest[key] = packet
        return latest[key]

    digest, metadata, references = hashlib.sha256(), {}, []
    started = perf_counter()
    with (
        reader,
        patch("scene_recall.odometry.local_map.extract_rgbd_features", extract),
    ):
        validate_source(reader, report)
        for observation in verified_observations(
            reader, report, reference_rows, digest, metadata
        ):
            references.append(observation.T_WC_D)
            clean = replace(observation, T_WC_D=None)
            if outage and outage[0] <= clean.frame_index < outage[1]:
                clean = replace(clean, rgb=np.zeros_like(clean.rgb))
            for name, engine in engines.items():
                local_started = perf_counter()
                pose, pair, changes, diagnostic = engine.process(
                    clean, reader.calibration
                )
                archive = archives[name]
                boundary = pair is None or pair.T_C1C0 is None
                archive.append(
                    pose,
                    keyframe=clean.frame_index in engine.frontend.keyframes,
                    boundary=boundary,
                )
                dirty = archive.apply(changes)
                atlas_record = None
                if atlas_enabled:
                    if name not in atlases:
                        atlases[name] = SubmapAtlas(reader.calibration)
                    atlas = atlases[name]
                    atlas.revise(
                        {i: archive.poses[i] for i in dirty if i in atlas.frames}
                    )
                    atlas_started = perf_counter()
                    atlas_record = atlas.observe(
                        clean.frame_index,
                        engine.segment_id,
                        archive.poses[-1],
                        engine.frontend._previous[1],
                        boundary=boundary,
                    )
                    atlas_record["seconds"] = perf_counter() - atlas_started
                    causal[name].append(
                        (
                            atlas.pose(clean.frame_index).copy(),
                            atlas.components[engine.segment_id],
                            atlas.version,
                        )
                    )
                timings[name].append(perf_counter() - local_started)
                emitted[name].append(pose.copy())
                histories[name].append(
                    (engine.segment_id, pair, diagnostic, atlas_record)
                )
            if len(references) % 50 == 0:
                print("Processed", len(references), "inputs", flush=True)
    if cache is not None:
        cache.close()
    results = {
        "wall_seconds": perf_counter() - started,
        "peak_rss_mib": peak_rss(),
        "policies": {},
    }
    for name, engine in engines.items():
        archive = archives[name]
        rows = [
            TrajectoryFrame(
                i,
                reference_rows[i].timestamp,
                segment,
                "initialized"
                if i == 0
                else "lost"
                if pair.T_C1C0 is None
                else "tracked",
                archive.poses[i],
                references[i],
                pair,
            )
            for i, (segment, pair, _, _) in enumerate(histories[name])
        ]
        rows = with_revised_poses(rows, archive.poses)
        frames = [
            {
                **report["frames"][i],
                "segment_id": row.segment_id,
                "status": row.status,
                "reason": None if row.pair is None else row.pair.reason,
                "diagnostics": diagnostic,
            }
            for i, (row, (_, _, diagnostic, _)) in enumerate(
                zip(rows, histories[name], strict=True)
            )
        ]
        evaluation = evaluate_trajectory(rows, (1, 5, 10, 20, 50, 100))
        current = {
            **report,
            "backend": "incremental_slam",
            "config": asdict(engine.config),
            "frames": frames,
            "evaluation": evaluation,
            "emitted_evaluation": evaluate_trajectory(
                with_revised_poses(rows, emitted[name]), (1, 5, 10, 20, 50, 100)
            ),
            "experiment": {
                "type": "verified recovery and explicit submap atlas",
                "policy": name,
                "protocol_sha256": sha256(output / "protocol.json"),
                "atlas_changes_tracking": False,
                "outage": outage,
            },
        }
        current = execution_header(current, protocol, error_statistics(timings[name]))
        save_run(output / name, rows, current, report["source"]["start"])
        result = {
            "evaluation": evaluation,
            "process_and_atlas_seconds": error_statistics(timings[name]),
            "recovery_statuses": dict(
                Counter(
                    d.get("frontend", {})
                    .get("recovery", {})
                    .get("status", "not_attempted")
                    for _, _, d, _ in histories[name]
                )
            ),
        }
        fallback_errors = []
        for i, (_, _, diagnostic, _) in enumerate(histories[name]):
            recovered = diagnostic.get("frontend", {}).get("recovery", {})
            if (
                recovered.get("status") == "accepted"
                and i
                and references[i] is not None
                and references[i - 1] is not None
            ):
                truth = np.linalg.inv(references[i]) @ references[i - 1]
                error = pose_error(np.array(recovered["T_target_source"]), truth)
                fallback_errors.append(
                    {"frame": i, "translation_m": error[0], "rotation_deg": error[1]}
                )
        result["accepted_fallback_reference_errors"] = fallback_errors
        if atlas_enabled:
            atlas = atlases[name]
            result["atlas"] = atlas_metrics(atlas, references)
            result["atlas"]["event_statuses"] = dict(
                Counter(e["status"] for e in atlas.events)
            )
            arrays = {
                "T_component_depth": np.array(
                    [atlas.pose(i) for i in range(len(rows))]
                ),
                "component_id": np.array(
                    [atlas.components[row.segment_id] for row in rows]
                ),
                "local_segment_id": np.array([row.segment_id for row in rows]),
                "source_frame_index": np.array(
                    [report["source"]["start"] + i for i in range(len(rows))]
                ),
                "version": np.array(atlas.version),
            }
            np.savez_compressed(output / name / "atlas.npz", **arrays)
            np.savez_compressed(
                output / name / "atlas_emitted.npz",
                T_component_depth=np.array([v[0] for v in causal[name]]),
                component_id=np.array([v[1] for v in causal[name]]),
                version=np.array([v[2] for v in causal[name]]),
            )
            np.savez_compressed(
                output / name / "atlas_links.npz",
                source=np.array([e["source"] for e in atlas.forest], dtype=np.int64),
                target=np.array([e["target"] for e in atlas.forest], dtype=np.int64),
                T_target_source=np.array(
                    [e["T_target_source"] for e in atlas.forest]
                ).reshape(-1, 4, 4),
            )
            result["atlas"]["query_seconds"] = error_statistics(
                a["seconds"] for _, _, _, a in histories[name]
            )
            result["atlas"]["peak_entries"] = max(
                a["entries"] for _, _, _, a in histories[name]
            )
            result["atlas"]["retained_frames"] = sorted(atlas.entries)
            result["atlas"]["causal_root_frames"] = sum(v[1] == 0 for v in causal[name])
            (output / name / "atlas_events.json").write_text(
                json.dumps(atlas.events, indent=2, allow_nan=False) + "\n"
            )
        results["policies"][name] = result
    (output / "comparison.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    if protocol["package_sha256"] != code_hash() or protocol["driver_sha256"] != sha256(
        Path(__file__)
    ):
        raise RuntimeError("source changed during replay")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("control", "native", "low_pnp", "rgbd", "rgbd_native"),
        default=["control", "low_pnp", "rgbd"],
    )
    parser.add_argument("--outage", type=int, nargs=2)
    parser.add_argument("--no-atlas", action="store_true")
    args = parser.parse_args()
    run(
        args.baseline_run,
        args.output_dir,
        args.policies,
        args.features,
        outage=args.outage,
        atlas_enabled=not args.no_atlas,
    )


if __name__ == "__main__":
    main()
