"""Reconstruct accepted cycle measurements and separate registration error from drift.

References score frozen registrations only. They never select correspondences,
fit transforms, change the atlas, or enter the emitted camera-constraint packet.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from slam_recovery import make_system

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.features import RGBDFeatures, extract_rgbd_features
from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.registration import register_anchor


def run(local_run, output, features=None):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    report, rows = load_trajectory(local_run)
    if report["config"].get("optimize", False):
        raise ValueError("this audit requires unrevised causal local poses")
    events = [
        e
        for e in json.loads((local_run / "atlas_events.json").read_text())
        if e["status"] in ("consistent_cycle", "inconsistent_cycle")
    ]
    selected = {e[key] for e in events for key in ("source", "target")}
    output.mkdir(parents=True)
    protocol = {
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "report_sha256": sha256(local_run / "report.json"),
        "atlas_sha256": sha256(local_run / "atlas.npz"),
        "events_sha256": sha256(local_run / "atlas_events.json"),
        "features_sha256": sha256(features) if features else None,
        "scope": "post-hoc cycle triage, fixed registration gates, references score only",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    cv2.setNumThreads(1)
    reader, _ = open_source(report)
    digest, metadata, packets = hashlib.sha256(), {}, {}
    sparse = make_system(report["experiment"]["policy"]).frontend.config.sparse
    sift = cv2.SIFT_create(nfeatures=sparse.max_features)
    with reader:
        validate_source(reader, report)
        for observation in verified_observations(
            reader, report, rows, digest, metadata
        ):
            if features is None and observation.frame_index in selected:
                packets[observation.frame_index] = extract_rgbd_features(
                    observation,
                    reader.calibration,
                    sift,
                    sparse.depth_edge_threshold_m,
                )
        calibration = reader.calibration
    if features:
        with np.load(features, allow_pickle=False) as cache:
            packets = {
                i: RGBDFeatures(
                    cache[f"{i}_pixels"],
                    cache.get(f"{i}_descriptors"),
                    cache[f"{i}_points_R"],
                )
                for i in selected
            }
    with np.load(local_run / "atlas.npz") as atlas:
        poses = atlas["T_component_depth"].copy()
    records, constraints, groups = [], [], defaultdict(list)
    for event in events:
        a, b = event["source"], event["target"]
        result = register_anchor(packets[a], packets[b], calibration)
        if (
            result.T_target_source is None
            or result.diagnostics != event["verification"]
        ):
            raise ValueError("accepted cycle measurement did not replay exactly")
        transform = (
            np.linalg.inv(calibration.T_RD) @ result.T_target_source @ calibration.T_RD
        )
        constraints.append(transform)
        expected = np.linalg.inv(poses[b]) @ poses[a]
        record = {
            "source": a,
            "target": b,
            "status": event["status"],
            "atlas_error_m_deg": pose_error(transform, expected),
        }
        if rows[a].reference_pose is not None and rows[b].reference_pose is not None:
            truth = np.linalg.inv(rows[b].reference_pose) @ rows[a].reference_pose
            record["reference_error_m_deg"] = pose_error(transform, truth)
            groups[event["status"]].append(record)
        records.append(record)
    np.savez_compressed(
        output / "cycle_constraints.npz",
        source=np.array([e["source"] for e in records], dtype=np.int64),
        target=np.array([e["target"] for e in records], dtype=np.int64),
        T_target_source=np.array(constraints).reshape(-1, 4, 4),
    )
    summary = {
        "cycle_queries": len(events),
        "reference_scored": sum(len(g) for g in groups.values()),
        "groups": {
            name: {
                "count": len(group),
                "registration_translation_m": error_statistics(
                    r["reference_error_m_deg"][0] for r in group
                ),
                "registration_rotation_deg": error_statistics(
                    r["reference_error_m_deg"][1] for r in group
                ),
                "atlas_translation_m": error_statistics(
                    r["atlas_error_m_deg"][0] for r in group
                ),
                "atlas_rotation_deg": error_statistics(
                    r["atlas_error_m_deg"][1] for r in group
                ),
                "registrations_outside_10cm_5deg": sum(
                    r["reference_error_m_deg"][0] > 0.1
                    or r["reference_error_m_deg"][1] > 5
                    for r in group
                ),
                "registrations_outside_2cm_1deg": sum(
                    r["reference_error_m_deg"][0] > 0.02
                    or r["reference_error_m_deg"][1] > 1
                    for r in group
                ),
            }
            for name, group in groups.items()
        },
        "records": records,
    }
    (output / "report.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    if protocol["package_sha256"] != code_hash() or protocol["driver_sha256"] != sha256(
        Path(__file__)
    ):
        raise RuntimeError("source changed during cycle audit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    args = parser.parse_args()
    run(args.local_run, args.output_dir, args.features)
