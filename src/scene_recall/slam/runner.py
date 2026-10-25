"""Raw streaming SLAM runs with separate causal and revised scientific exports."""

import hashlib
import json
import platform
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.datasets.eth3d import ETH3DReader
from scene_recall.datasets.scannet import ScanNetReader
from scene_recall.datasets.tum import TUMConfig, TUMReader
from scene_recall.odometry.evaluation import evaluate_trajectory, pose_error
from scene_recall.odometry.io import save_run
from scene_recall.odometry.pipeline import (
    PairEstimate,
    TrajectoryFrame,
    validate_trajectory,
)
from scene_recall.run_metadata import code_hash, peak_rss
from scene_recall.slam.evaluation import latency_statistics, score_components
from scene_recall.slam.online import OnlineConfig, OnlineSLAM


def run_online(
    source,
    output,
    *,
    dataset,
    start=0,
    stop=None,
    config=None,
    mapping=None,
    association=None,
    intervention=None,
    invocation=None,
    feature_extractor=None,
):
    """Process original images, then score frozen estimates using references.

    intervention is an optional (kind, start, stop) tuple in replay-local input
    indices. Source and delivered image digests distinguish controlled inputs.
    No cached local poses, feature packets or future cameras enter the engine.
    """
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError("choose a fresh output directory")
    if intervention is not None:
        kind, begin, end = intervention
        if kind not in ("rgb", "depth") or not 0 <= begin < end:
            raise ValueError("invalid controlled intervention")
    association, config = association or TUMConfig(), config or OnlineConfig()
    if dataset not in ("tum_rgbd", "eth3d_slam", "scannet_v2"):
        raise ValueError("unsupported SLAM dataset")
    reader = (
        ScanNetReader(source, start=start, stop=stop)
        if dataset == "scannet_v2"
        else (ETH3DReader if dataset == "eth3d_slam" else TUMReader)(
            source, start=start, stop=stop, config=association
        )
    )
    cv2.setNumThreads(1)
    engine = OnlineSLAM(
        reader.calibration, config, mapping=mapping, feature_extractor=feature_extractor
    )
    source_hash, delivered_hash = hashlib.sha256(), hashlib.sha256()
    rows, emitted, emitted_components, versions, frames = [], [], [], [], []
    protocol = {
        "phase": "frozen before streaming",
        "scope": "raw causal processing, no final retrospective solve",
        "config": asdict(config),
        "mapping": asdict(mapping) if mapping else None,
        "feature_extractor": None
        if feature_extractor is None
        else deepcopy(feature_extractor.provenance),
        "intervention": intervention,
        "package_sha256": code_hash(),
        "source": str(source),
        "start": start,
        "stop": stop,
    }
    # Keep an incomplete-run protocol after failure without publishing a run.
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = output.parent / (output.name + ".protocol.json")
    if protocol_path.exists() or protocol_path.is_symlink():
        raise ValueError("run protocol already exists")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    started = perf_counter()
    with reader:
        for original in reader:
            payload = json.dumps(
                dict(original.provenance or {}), sort_keys=True, allow_nan=False
            ).encode()
            source_hash.update(payload)
            for array in (original.rgb, original.depth, original.T_WC_D):
                source_hash.update(b"none" if array is None else array.tobytes())
            observation = replace(original, T_WC_D=None)
            if intervention and begin <= original.frame_index < end:
                observation = (
                    replace(observation, rgb=np.zeros_like(observation.rgb))
                    if kind == "rgb"
                    else replace(
                        observation, depth=np.full_like(observation.depth, np.nan)
                    )
                )
            delivered_hash.update(payload)
            for array in (observation.rgb, observation.depth):
                delivered_hash.update(array.tobytes())
            update = engine.process(observation)
            pair = (
                None
                if update.frame == 0
                else PairEstimate(
                    None
                    if update.status == "lost"
                    else np.linalg.inv(update.local_pose) @ rows[-1].T_SC_D,
                    "local tracking failure" if update.status == "lost" else None,
                )
            )
            rows.append(
                TrajectoryFrame(
                    update.frame,
                    original.timestamp,
                    update.segment,
                    update.status,
                    update.local_pose,
                    original.T_WC_D,
                    pair,
                )
            )
            emitted.append(update.pose)
            emitted_components.append(update.component)
            versions.append(update.version)
            frames.append(
                {
                    "frame_index": update.frame,
                    "source_frame_index": start + update.frame,
                    "segment_id": update.segment,
                    "status": update.status,
                    "provenance": dict(original.provenance or {}),
                    "diagnostics": update.diagnostics,
                    "dirty_frames": list(update.dirty_frames),
                }
            )
            if len(rows) % 100 == 0:
                print(
                    f"Processed {len(rows)} frames, local losses {engine.local.segment_id}, graph nodes {len(engine.graph.local)}",
                    flush=True,
                )
    elapsed = perf_counter() - started
    if not rows:
        raise ValueError("online run requires at least one observation")
    references = [row.reference_pose for row in rows]
    # Local historical revisions have the same attachment rule as the online
    # engine. Recompute adjacent pairs from that one final local archive.
    rows = [
        replace(
            row,
            T_SC_D=engine.archive.poses[i],
            pair=None
            if i == 0
            else replace(
                row.pair,
                T_C1C0=None
                if row.status == "lost"
                else np.linalg.inv(engine.archive.poses[i])
                @ engine.archive.poses[i - 1],
            ),
        )
        for i, row in enumerate(rows)
    ]
    validate_trajectory(rows)
    loops = []
    # Evaluate only after all estimator state and placement decisions are frozen.
    for edge in engine.graph.edges.values():
        if edge.kind != "loop":
            continue
        reference_error = None
        if references[edge.source] is not None and references[edge.target] is not None:
            truth = np.linalg.inv(references[edge.target]) @ references[edge.source]
            reference_error = pose_error(edge.T_target_source, truth)
        loops.append(
            {
                "source": edge.source,
                "target": edge.target,
                "group": edge.group,
                "reference_error": reference_error,
            }
        )
    report = {
        "format_version": 2,
        "backend": "online_slam",
        "initial_pose_stage": None,
        "config": asdict(config.local),
        "online_config": asdict(config),
        "feature_extractor": protocol["feature_extractor"],
        "source": {
            "dataset": dataset,
            "sequence_id": reader.sequence_id,
            "path": str(source),
            "source_frame_count": reader.frame_count,
            "start": start,
            "stop": start + len(rows),
            "association_config": None
            if dataset == "scannet_v2"
            else asdict(association),
            "adapter": dict(getattr(reader, "metadata", {})),
            "canonical_input_sha256": source_hash.hexdigest(),
            "delivered_rgbd_sha256": delivered_hash.hexdigest(),
        },
        "calibration": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(reader.calibration).items()
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "package_sha256": protocol["package_sha256"],
            "invocation": invocation or [],
        },
        "experiment": {
            "causal_online": True,
            "offline_global_revision": False,
            "intervention": intervention,
            "pending_global_solve": engine.pending_solve,
        },
        "timing": {
            "raw_pipeline_seconds": elapsed,
            "raw_frames_per_second": len(rows) / elapsed,
            "process_latency": latency_statistics(
                [f["diagnostics"]["total_seconds"] for f in frames]
            ),
            "peak_rss_mib": peak_rss(),
            "scope": "serial raw reading, extraction, local estimation, place recognition, scheduled graph solves and optional mapping, excludes export and scoring",
        },
        "evaluation": evaluate_trajectory(rows),
        "causal_evaluation": score_components(emitted, emitted_components, references),
        "revised_evaluation": score_components(
            engine.poses, engine.components, references
        ),
        "loop_reference_errors": loops,
        "event_statuses": dict(Counter(event["status"] for event in engine.events)),
        "capacity_reached": len(engine.graph.local) >= config.graph.max_nodes,
        "frames": frames,
    }
    with tempfile.TemporaryDirectory(
        prefix=".online-slam-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "run"
        save_run(staging, rows, report, start)
        _export_online(staging, engine, emitted, emitted_components, versions, start)
        (staging / "protocol.json").write_text(protocol_path.read_text())
        if output.exists() or output.is_symlink():
            raise ValueError("output appeared during SLAM export")
        staging.rename(output)
    protocol_path.unlink()
    return report


def _export_online(output, engine, emitted, emitted_components, versions, start):
    for name, poses, components, version in (
        ("atlas", engine.poses, engine.components, engine.version),
        ("atlas_emitted", emitted, emitted_components, versions),
    ):
        np.savez_compressed(
            output / (name + ".npz"),
            T_component_depth=np.array(poses),
            component_id=np.array(components),
            local_segment_id=np.array(engine.segments),
            source_frame_index=np.arange(start, start + len(engine.poses)),
            version=np.array(version),
        )
    (output / "events.json").write_text(json.dumps(engine.events, indent=2) + "\n")
    edge_arrays = {
        "endpoints": np.array(list(engine.graph.edges), dtype=np.int64).reshape(-1, 2),
        "T_target_source": np.array(
            [edge.T_target_source for edge in engine.graph.edges.values()]
        ).reshape(-1, 4, 4),
        "kind": np.array([edge.kind for edge in engine.graph.edges.values()]),
    }
    np.savez_compressed(output / "constraints.npz", **edge_arrays)
    if engine.config.shared_estimation:
        np.savez_compressed(
            output / "shared_estimation.npz",
            **engine.welding.export_arrays(engine.graph),
        )
    if engine.reconstruction is not None:
        arrays = {}
        for index, chunk in enumerate(engine.reconstruction.snapshot()):
            for key, value in asdict(chunk.cloud).items():
                arrays[f"{index}_{key}"] = value
            arrays[f"{index}_frames"] = np.array(chunk.frames)
            arrays[f"{index}_component"] = np.array(chunk.component)
            arrays[f"{index}_version"] = np.array(chunk.version)
        np.savez_compressed(output / "reconstruction.npz", **arrays)
