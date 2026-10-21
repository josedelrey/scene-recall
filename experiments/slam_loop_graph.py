"""Frozen-local replay of retrieval and confirmed metric loop-graph experiments.

References are scored only after all hypotheses and graph results are frozen.
Raw local arrays and losses remain unchanged. New atlas arrays carry component
gauges, source lineage and revisions for downstream reconstruction rebuilding.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
from slam_recovery import execution_header, make_system

from scene_recall.data import Calibration
from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.features import RGBDFeatures, extract_rgbd_features
from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, peak_rss, sha256
from scene_recall.slam.atlas import AtlasConfig, SubmapAtlas, appearance_signature
from scene_recall.slam.loop_confirmation import LoopConfirmation, LoopConfirmationConfig
from scene_recall.slam.pose_graph import MetricPoseGraph, PoseGraphConfig
from scene_recall.slam.registration import RegistrationConfig, register_anchor
from scene_recall.slam.retrieval import descriptor_votes


def packet(cache, i):
    return RGBDFeatures(
        cache[f"{i}_pixels"], cache.get(f"{i}_descriptors"), cache[f"{i}_points_R"]
    )


def verify_feature_cache(local_run, features):
    """Bind a supplied cache to a verified extraction and the same input cohort."""
    parent = features.parent
    manifest = parent / "manifest.json"
    if manifest.exists():
        provenance = json.loads(manifest.read_text())
        source = json.loads((parent / "budget" / "report.json").read_text())["source"]
        current = json.loads((local_run / "report.json").read_text())["source"]
        if any(
            source[k] != current[k] for k in ("canonical_input_sha256", "start", "stop")
        ):
            raise ValueError("feature cache input cohort differs from local trajectory")
    else:
        provenance = json.loads((parent / "protocol.json").read_text())
        if provenance["local_report_sha256"] != sha256(local_run / "report.json"):
            raise ValueError("query feature cache belongs to another local report")
    if provenance["features_sha256"] != sha256(features):
        raise ValueError("feature cache does not match its extraction provenance")


def prepare_features(local_run, output):
    """Verify raw source identity and extract only query/keyframe observations."""
    report, rows = load_trajectory(local_run)
    reader, _ = open_source(report)
    sparse = make_system("control").frontend.config.sparse
    sift = cv2.SIFT_create(nfeatures=sparse.max_features)
    digest, metadata, arrays = hashlib.sha256(), {}, {}
    with reader:
        validate_source(reader, report)
        for observation in verified_observations(
            reader, report, rows, digest, metadata
        ):
            i = observation.frame_index
            if i % 5 and rows[i].status == "tracked":
                continue
            features = extract_rgbd_features(
                observation, reader.calibration, sift, sparse.depth_edge_threshold_m
            )
            arrays[f"{i}_pixels"] = features.pixels
            arrays[f"{i}_points_R"] = features.points_R
            if features.descriptors is not None:
                arrays[f"{i}_descriptors"] = features.descriptors
    if digest.hexdigest() != report["source"]["canonical_input_sha256"]:
        raise ValueError("canonical replay identity changed")
    np.savez_compressed(output, **arrays)


def score(poses, components, references, indices=None):
    groups = {}
    for root in sorted(set(components)):
        selected = [
            i
            for i, c in enumerate(components)
            if c == root
            and references[i] is not None
            and (indices is None or i in indices)
        ]
        errors = []
        if selected:
            first = selected[0]
            alignment = poses[first] @ np.linalg.inv(references[first])
            errors = [pose_error(poses[i], alignment @ references[i]) for i in selected]
        groups[str(int(root))] = {
            "frames": int(np.sum(components == root)),
            "reference_frames": len(selected),
            "translation_m": error_statistics(e[0] for e in errors),
            "rotation_deg": error_statistics(e[1] for e in errors),
        }
    pairs = []
    for i in range(len(poses) - 10):
        j = i + 10
        if (
            components[i] != components[j]
            or references[i] is None
            or references[j] is None
        ):
            continue
        if indices is not None and (i not in indices or j not in indices):
            continue
        pairs.append(
            pose_error(
                np.linalg.inv(poses[j]) @ poses[i],
                np.linalg.inv(references[j]) @ references[i],
            )
        )
    return {
        "component_count": len(groups),
        "root_frames": int(np.sum(components == components[0])),
        "largest_component_frames": max(
            (g["frames"] for g in groups.values()), default=0
        ),
        "groups": groups,
        "ten_frame_relative_translation_m": error_statistics(e[0] for e in pairs),
        "ten_frame_relative_rotation_deg": error_statistics(e[1] for e in pairs),
    }


def dense_poses(graph, rows):
    """Interpolate left corrections within each local segment, preserving scale."""
    poses, components = [], []
    nodes = {}
    for frame in graph.poses:
        nodes.setdefault(graph.segments[frame], []).append(frame)
    for i, row in enumerate(rows):
        keys = sorted(nodes[row.segment_id])
        before = max((k for k in keys if k <= i), default=keys[0])
        after = min((k for k in keys if k >= i), default=keys[-1])
        first = graph.poses[before] @ np.linalg.inv(graph.local[before])
        if before == after:
            correction = first
        else:
            second = graph.poses[after] @ np.linalg.inv(graph.local[after])
            alpha = (i - before) / (after - before)
            correction = np.eye(4)
            rotation = cv2.Rodrigues(second[:3, :3] @ first[:3, :3].T)[0].ravel()
            correction[:3, :3] = cv2.Rodrigues(alpha * rotation)[0] @ first[:3, :3]
            correction[:3, 3] = (1 - alpha) * first[:3, 3] + alpha * second[:3, 3]
        poses.append(correction @ row.T_SC_D)
        components.append(graph.components[before])
    return np.array(poses), np.array(components, dtype=np.int64)


def build_graph(rows, measurements, robust):
    graph = MetricPoseGraph()
    selected = {
        i for i, row in enumerate(rows) if i % 5 == 0 or row.status != "tracked"
    }
    selected.update(
        i
        for i, row in enumerate(rows)
        if i == len(rows) - 1 or rows[i + 1].segment_id != row.segment_id
    )
    selected.update(i for e in measurements for i in e[:2])
    last = {}
    for i in sorted(selected):
        row = rows[i]
        graph.add_node(i, row.segment_id, row.T_SC_D)
        if row.segment_id in last:
            previous = last[row.segment_id]
            graph.add_edge(
                previous, i, np.linalg.inv(row.T_SC_D) @ rows[previous].T_SC_D
            )
        last[row.segment_id] = i
    for source, target, transform, group in measurements:
        if (source, target) not in graph.edges:
            graph.add_edge(source, target, transform, kind="loop", group=group)
    started = perf_counter()
    diagnostic = graph.solve(robust=robust)
    diagnostic["solve_seconds"] = perf_counter() - started
    diagnostic["nodes"] = len(graph.local)
    diagnostic["edges"] = len(graph.edges)
    return graph, diagnostic


def run(local_run, output, features=None, *, policies=None):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    report, rows = load_trajectory(local_run)
    if report["config"].get("optimize", False):
        raise ValueError("this experiment requires unrevised causal local poses")
    policies = policies or [
        "forest_mean",
        "forest_votes",
        "graph_mean_linear",
        "graph_mean_robust",
        "graph_votes_linear",
        "graph_votes_robust",
    ]
    cv2.setNumThreads(1)
    if features is not None:
        verify_feature_cache(local_run, features)
    output.mkdir(parents=True)
    protocol = {
        "phase": "frozen before replay",
        "local_run": str(local_run.resolve()),
        "local_report_sha256": sha256(local_run / "report.json"),
        "local_trajectory_sha256": sha256(local_run / "trajectory.npz"),
        "features_sha256": sha256(features) if features else None,
        "policies": policies,
        "atlas": asdict(AtlasConfig()),
        "registration": asdict(RegistrationConfig()),
        "confirmation": asdict(LoopConfirmationConfig()),
        "graph": asdict(PoseGraphConfig()),
        "same_segment_min_frame_gap": 30,
        "timing_scope": "shared deterministic registration and vote computations, use isolated timing for speed comparisons",
        "outage": report["experiment"].get("outage"),
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "scope": "offline fixed local trajectories, causal retrieval and confirmation, final global graph, references score only",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if features is None:
        features = output / "query_features.npz"
        prepare_features(local_run, features)
        protocol["features_sha256"] = sha256(features)
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    calibration = Calibration(
        **{
            k: tuple(v) if k.endswith("shape") else np.array(v)
            for k, v in report["calibration"].items()
        }
    )
    atlases = {
        mode: SubmapAtlas(calibration, AtlasConfig(retrieval=mode))
        for mode in ("mean", "descriptor_votes")
    }
    confirmations = {mode: LoopConfirmation() for mode in ("mean", "votes")}
    events, measurements, times = (
        {m: [] for m in confirmations},
        {m: {} for m in confirmations},
        {m: [] for m in confirmations},
    )
    forest_times, causal = {m: [] for m in atlases}, {m: [] for m in atlases}
    accepted_registrations = {}
    with np.load(features, allow_pickle=False) as cache:
        for i, row in enumerate(rows):
            regular = i % 5 == 0 or row.status != "tracked"
            outage = protocol["outage"]
            f = (
                packet(cache, i)
                if regular and not (outage and outage[0] <= i < outage[1])
                else RGBDFeatures(np.empty((0, 2)), None, np.empty((0, 3)))
            )
            bank = dict(atlases["mean"].entries)
            identities = {
                id(entry[0]): key
                for atlas in atlases.values()
                for key, entry in atlas.entries.items()
            }
            registered, vote_cache = {}, {}

            def cached_registration(
                first,
                second,
                current_calibration,
                config,
                *,
                identities=identities,
                cache=registered,
            ):
                key = identities[id(first)]
                if key not in cache:
                    cache[key] = register_anchor(
                        first, second, current_calibration, config
                    )
                return cache[key]

            def cached_votes(
                first, second, ratio=0.75, *, identities=identities, cache=vote_cache
            ):
                key = identities[id(first)]
                if key not in cache:
                    cache[key] = descriptor_votes(first, second, ratio)
                return cache[key]

            # Both forests use identical local poses and bank admission.
            with (
                patch("scene_recall.slam.atlas.register_anchor", cached_registration),
                patch("scene_recall.slam.atlas.descriptor_votes", cached_votes),
            ):
                for mode, atlas in atlases.items():
                    started = perf_counter()
                    atlas.observe(
                        i,
                        row.segment_id,
                        row.T_SC_D,
                        f,
                        boundary=row.status != "tracked",
                    )
                    forest_times[mode].append(perf_counter() - started)
                    causal[mode].append(atlas.components[row.segment_id] == 0)
            if list(atlases["mean"].entries) != list(
                atlases["descriptor_votes"].entries
            ):
                raise ValueError("retrieval altered the retained bank")
            if not regular or f.descriptors is None:
                continue
            signature = appearance_signature(f)
            eligible = [
                k
                for k in bank
                if i - k >= (30 if rows[k].segment_id == row.segment_id else 10)
            ]
            distances = {
                k: float(np.linalg.norm(signature - bank[k][1])) for k in eligible
            }
            ranked = {"mean": sorted(eligible, key=lambda k: (distances[k], k))}
            started = perf_counter()
            votes = {k: cached_votes(bank[k][0], f) for k in eligible}
            ranked["votes"] = sorted(
                eligible, key=lambda k: (-votes[k][0], -votes[k][1], distances[k], k)
            )
            rank_seconds = perf_counter() - started
            for mode, confirmation in confirmations.items():
                started = perf_counter()
                candidates, candidate_events = [], []
                for source in ranked[mode][:4]:
                    if source not in registered:
                        registered[source] = register_anchor(
                            bank[source][0], f, calibration
                        )
                    result = registered[source]
                    event = {
                        "source": source,
                        "target": i,
                        "verification": result.diagnostics,
                        "rank": ranked[mode].index(source) + 1,
                    }
                    if result.T_target_source is None:
                        events[mode].append(
                            {**event, "status": "registration_rejected"}
                        )
                        continue
                    transform = (
                        np.linalg.inv(calibration.T_RD)
                        @ result.T_target_source
                        @ calibration.T_RD
                    )
                    candidates.append(
                        (
                            source,
                            i,
                            rows[source].segment_id,
                            row.segment_id,
                            rows[source].T_SC_D,
                            row.T_SC_D,
                            transform,
                        )
                    )
                    candidate_events.append(event)
                    accepted_registrations[source, i] = transform
                statuses, added = confirmation.propose_batch(candidates)
                events[mode].extend(
                    {**event, "status": status}
                    for event, status in zip(candidate_events, statuses, strict=True)
                )
                for measurement in added:
                    measurements[mode][measurement[:2]] = measurement
                times[mode].append(
                    perf_counter() - started + (rank_seconds if mode == "votes" else 0)
                )
            if i % 100 == 0:
                print(
                    "replayed",
                    i,
                    "confirmed",
                    {m: len(v) for m, v in measurements.items()},
                    flush=True,
                )
    results, arrays, graph_objects = {}, {}, {}
    for name in policies:
        if name.startswith("forest"):
            mode = "mean" if name.endswith("mean") else "descriptor_votes"
            atlas = atlases[mode]
            poses = np.array([atlas.pose(i) for i in range(len(rows))])
            components = np.array([atlas.components[r.segment_id] for r in rows])
            diagnostic = {
                "event_statuses": dict(Counter(e["status"] for e in atlas.events)),
                "causal_root_frames": sum(causal[mode]),
                "observe_seconds": error_statistics(forest_times[mode]),
            }
            version = atlas.version
            if name == "forest_mean":
                with np.load(local_run / "atlas.npz") as baseline:
                    if not np.array_equal(
                        poses, baseline["T_component_depth"]
                    ) or not np.array_equal(components, baseline["component_id"]):
                        raise ValueError(
                            "unchanged forest failed exact baseline reproduction"
                        )
                diagnostic["exact_baseline_reproduction"] = True
        else:
            mode = "votes" if "votes" in name else "mean"
            graph, diagnostic = build_graph(
                rows, list(measurements[mode].values()), name.endswith("robust")
            )
            poses, components = dense_poses(graph, rows)
            diagnostic["event_statuses"] = dict(
                Counter(e["status"] for e in events[mode])
            )
            diagnostic["query_seconds"] = error_statistics(times[mode])
            diagnostic["confirmed_groups"] = len(confirmations[mode].confirmed)
            version = graph.version
            graph_objects[name] = graph
        arrays[name] = poses, components
        destination = output / name
        destination.mkdir()
        derived = execution_header(
            report,
            protocol,
            diagnostic.get("observe_seconds", diagnostic.get("query_seconds")),
        )
        derived["experiment"] = {
            **report["experiment"],
            "global_policy": name,
            "local_poses_unchanged": True,
            "offline_global_revision": name.startswith("graph"),
        }
        (destination / "report.json").write_text(
            json.dumps(derived, indent=2, allow_nan=False) + "\n"
        )
        shutil.copyfile(local_run / "trajectory.npz", destination / "trajectory.npz")
        np.savez_compressed(
            destination / "atlas.npz",
            T_component_depth=poses,
            component_id=components,
            local_segment_id=np.array([r.segment_id for r in rows]),
            source_frame_index=np.array(
                [report["source"]["start"] + i for i in range(len(rows))]
            ),
            version=np.array(version),
        )
        results[name] = {
            "diagnostic": diagnostic,
            "local_archive_sha256": sha256(destination / "trajectory.npz"),
        }
        print("solved", name, flush=True)
    # Reference-only scoring starts after all selection and optimization.
    references = [r.reference_pose for r in rows]
    common = {
        i for i in range(len(rows)) if all(c[i] == c[0] for _, c in arrays.values())
    }
    for name, (poses, components) in arrays.items():
        results[name]["evaluation"] = score(poses, components, references)
        results[name]["common_root_evaluation"] = score(
            poses, components, references, common
        )
        if name in graph_objects:
            graph = graph_objects[name]
            scored = []
            for edge, weight in zip(
                graph.edges.values(),
                results[name]["diagnostic"]["edge_weights"],
                strict=True,
            ):
                if (
                    edge.kind != "loop"
                    or references[edge.source] is None
                    or references[edge.target] is None
                ):
                    continue
                error = pose_error(
                    edge.T_target_source,
                    np.linalg.inv(references[edge.target]) @ references[edge.source],
                )
                scored.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "translation_m": error[0],
                        "rotation_deg": error[1],
                        "weight_before_pruning": weight,
                    }
                )
            results[name]["loop_reference_errors"] = scored
    for mode in confirmations:
        (output / f"events_{mode}.json").write_text(
            json.dumps(events[mode], indent=2, allow_nan=False) + "\n"
        )
        values = list(measurements[mode].values())
        np.savez_compressed(
            output / f"constraints_{mode}.npz",
            source=np.array([v[0] for v in values], dtype=np.int64),
            target=np.array([v[1] for v in values], dtype=np.int64),
            T_target_source=np.array([v[2] for v in values]).reshape(-1, 4, 4),
            group=np.array([v[3] for v in values], dtype=np.int64).reshape(-1, 4),
        )
    proposal_errors = []
    for (a, b), transform in accepted_registrations.items():
        if references[a] is not None and references[b] is not None:
            error = pose_error(transform, np.linalg.inv(references[b]) @ references[a])
            proposal_errors.append(
                {
                    "source": a,
                    "target": b,
                    "translation_m": error[0],
                    "rotation_deg": error[1],
                }
            )
    np.savez_compressed(
        output / "accepted_registrations.npz",
        source=np.array([k[0] for k in accepted_registrations], dtype=np.int64),
        target=np.array([k[1] for k in accepted_registrations], dtype=np.int64),
        T_target_source=np.array(list(accepted_registrations.values())).reshape(
            -1, 4, 4
        ),
    )
    summary = {
        "policies": results,
        "common_root_frames": len(common),
        "local_failed_transitions": sum(r.status == "lost" for r in rows),
        "peak_rss_mib": peak_rss(),
        "proposal_reference_errors": proposal_errors,
    }
    (output / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    if protocol["package_sha256"] != code_hash() or protocol["driver_sha256"] != sha256(
        Path(__file__)
    ):
        raise RuntimeError("source changed during replay")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    args = parser.parse_args()
    run(args.local_run, args.output_dir, args.features)
