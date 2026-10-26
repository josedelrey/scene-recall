"""Retrospective native identity and warm-start controls on one frozen ledger.

This does not replay tracking or establish causal pipeline accuracy. It extracts
image descriptors again, verifies exact retained pixel indexing, then changes
only reversible associations on the same immutable raw measurement population.
"""

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np

from scene_recall.datasets.tum import TUMConfig, TUMReader
from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.fixed_landmarks import DirectFactors
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.evaluation import score_components
from scene_recall.slam.observation_welding import LoopValidation, RecordedView
from scene_recall.slam.online import OnlineConfig, OnlineSLAM
from scene_recall.slam.pose_graph import MetricPoseGraph
from scene_recall.slam.registration import RegistrationConfig
from scene_recall.slam.shared_estimation import SharedConfig, SharedEstimator
from scene_recall.slam.shared_solver import solve_shared


def study(run, output):
    if output.exists():
        raise ValueError("choose a fresh diagnostic output directory")
    report, rows = load_trajectory(run)
    if report["source"]["dataset"] != "tum_rgbd":
        raise ValueError("this frozen descriptor diagnostic currently supports TUM")
    arrays = dict(np.load(run / "shared_estimation.npz"))
    output.mkdir(parents=True)
    protocol = {
        "scope": "retrospective fixed-measurement identity and initialization diagnostic, no causal tracking feedback",
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "source_sha256": sha256(run / "shared_estimation.npz"),
        "source_report_sha256": sha256(run / "report.json"),
        "unavailable_original_loop_validation": "original validation rows were not exported, empty diagnostics do not alter the raw objective",
        "conditions": ["control", "reobserve", "reobserve_warm"],
        "iterations": 32,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    source = report["source"]
    reader = TUMReader(
        source["path"],
        start=source["start"],
        stop=source["stop"],
        config=TUMConfig(**source["association_config"]),
    )
    config = OnlineConfig(
        contrast=report["online_config"]["contrast"],
        feature_scale=report["online_config"]["feature_scale"],
        same_segment_gap=report["online_config"]["same_segment_gap"],
        metric_refinement=report["online_config"]["metric_refinement"],
    )
    config = replace(
        config,
        atlas=replace(
            config.atlas,
            registration=RegistrationConfig(
                **report["online_config"]["atlas"]["registration"]
            ),
        ),
    )
    extractor = OnlineSLAM(reader.calibration, config).local.frontend
    features = {}
    keys = set(map(int, np.unique(arrays["view"])))
    cv2.setNumThreads(1)
    with reader:
        for observation in reader:
            key = observation.frame_index
            if key not in keys:
                continue
            f = extractor._extract(
                replace(observation, T_WC_D=None), reader.calibration
            )
            mask = arrays["view"] == key
            np.testing.assert_array_equal(
                f.pixels[arrays["feature_index"][mask]], arrays["pixels"][mask]
            )
            features[key] = f
    shared = SharedEstimator(
        reader.calibration,
        shared=SharedConfig(
            preconditioner="coarse",
            retry_linear_failure=True,
            solver=replace(SharedConfig().solver, max_iterations=32),
        ),
    )
    for frame in sorted(keys):
        mask = arrays["view"] == frame
        factors = DirectFactors(
            **{
                name: arrays[name][mask].copy()
                for name in DirectFactors.__dataclass_fields__
            }
        )
        shared.views[frame] = RecordedView(
            factors,
            arrays["feature_index"][mask],
            arrays["native_index"][mask],
            rows[frame].segment_id,
        )
    shared.identity_map = {
        (int(s), int(owner)): int(track)
        for s, owner, track in arrays["identity_owners"]
    }
    for source, target, a, b in arrays["associations"]:
        shared.associations.setdefault((int(source), int(target)), []).append(
            (int(a), int(b))
        )
    shared.associations = {k: tuple(v) for k, v in shared.associations.items()}
    for edge in shared.associations:
        shared.validation[edge] = LoopValidation(
            np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2))
        )
    graph = MetricPoseGraph()
    final = np.load(run / "atlas.npz")
    for key, pose in zip(arrays["camera_ids"], arrays["depth_poses"], strict=True):
        key = int(key)
        graph.add_node(key, rows[key].segment_id, pose)
        graph.poses[key] = pose.copy()
        graph.components[key] = int(final["component_id"][key])
    constraints = np.load(run / "constraints.npz")
    for edge, pose, kind in zip(
        constraints["endpoints"],
        constraints["T_target_source"],
        constraints["kind"],
        strict=True,
    ):
        graph.add_edge(*map(int, edge), pose, kind=str(kind))
    old_identities = dict(arrays["live_identities"])
    old_members = defaultdict(set)
    for raw, live in old_identities.items():
        old_members[int(live)].add(int(raw))
    old_points = dict(zip(arrays["point_ids"], arrays["point_positions"], strict=True))
    conditions, graphs, events = {}, {}, []
    for condition in protocol["conditions"]:
        state, placement = deepcopy(shared), deepcopy(graph)
        if condition != "control":
            for target in sorted(keys):
                recent = [
                    k
                    for k in sorted(keys, reverse=True)
                    if k < target
                    and target - k < config.same_segment_gap
                    and state.views[k].segment == state.views[target].segment
                ][:3]
                for source in recent:
                    result = state.reobserve(
                        source,
                        target,
                        features[source],
                        features[target],
                        extractor.registration,
                    )
                    if condition == "reobserve":
                        events.append({"source": source, "target": target, **result})
        initialized = {}

        def optimize(
            poses,
            factors,
            K,
            fixed,
            solver,
            *,
            condition=condition,
            placement=placement,
            state=state,
            initialized=initialized,
            **kwargs,
        ):
            if condition == "reobserve_warm":
                active = {
                    e
                    for e, v in placement.edges.items()
                    if v.kind == "loop" and e in state.associations
                } | state.reobservations.keys()
                identities, accepted = state._identities(active)
                active = {
                    e for e in active if accepted[e] >= state.config.min_associations
                }
                identities, _ = state._identities(active)
                members = defaultdict(set)
                for raw, live in identities.items():
                    members[live].add(raw)
                packet = factors.points.copy()
                camera_keys = sorted(state.views)
                for live in np.unique(factors.track_id):
                    if (
                        members[live] != old_members.get(int(live))
                        or live not in old_points
                    ):
                        continue
                    mask = factors.track_id == live
                    camera = int(factors.view[np.flatnonzero(mask)[0]])
                    frame = camera_keys[camera]
                    old_pose = graph.poses[frame] @ np.linalg.inv(
                        state.views[frame].factors.T_RcD[0]
                    )
                    packet[mask] = transform_points(
                        old_points[live], poses[camera] @ np.linalg.inv(old_pose)
                    )
                    initialized[int(live)] = True
                factors = replace(factors, points=packet)
            return solve_shared(poses, factors, K, fixed, solver, **kwargs)

        started = perf_counter()
        with patch("scene_recall.slam.shared_estimation.solve_shared", optimize):
            fit = state.refine(placement)[0]
        conditions[condition] = {
            "solver": fit,
            "seconds": perf_counter() - started,
            "warm_points": len(initialized),
        }
        graphs[condition] = placement
        np.savez_compressed(
            output / (condition + ".npz"), **state.export_arrays(placement)
        )
    camera_keys = sorted(graph.local)
    common = {
        i
        for i, key in enumerate(camera_keys)
        if all(
            g.components[key] == g.components[camera_keys[0]] for g in graphs.values()
        )
    }
    for condition, placement in graphs.items():
        conditions[condition]["matched_keyframes"] = score_components(
            np.array([placement.poses[k] for k in camera_keys]),
            [placement.components[k] for k in camera_keys],
            [rows[k].reference_pose for k in camera_keys],
            common,
        )
    result = {
        "protocol": protocol,
        "conditions": conditions,
        "events": events,
        "matched_camera_ids": [camera_keys[i] for i in sorted(common)],
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.run, args.output)


if __name__ == "__main__":
    main()
