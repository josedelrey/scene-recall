"""Run RGB-D tracking, reference evaluation and scientific export."""

import hashlib
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.datasets.scannet import ScanNetReader
from scene_recall.datasets.tum import TUMConfig, TUMReader
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.hybrid import HybridConfig, HybridRGBDBackend
from scene_recall.odometry.icp import DepthICPBackend, ICPConfig
from scene_recall.odometry.io import save_run
from scene_recall.odometry.pipeline import track_observations
from scene_recall.odometry.sparse import SparseConfig, SparseRGBDBackend
from scene_recall.run_metadata import code_hash


def run_odometry(
    source: Path,
    output: Path,
    *,
    dataset: str,
    backend_name: str,
    config: SparseConfig | ICPConfig | HybridConfig,
    start: int = 0,
    stop: int | None = None,
    tum_config: TUMConfig | None = None,
    rpe_intervals: tuple[int, ...] = (1, 5, 10),
    rpe_time_intervals_s: tuple[float, ...] = (0.1, 0.5, 1.0),
    rpe_time_tolerance_s: float = 0.02,
    invocation: list[str] | None = None,
) -> dict:
    """Track a dataset source, evaluate the trajectory and save format-2 exports.

    The invocation is supplied by the caller so programmatic runs do not inherit
    an unrelated process command line. Dataset and backend names use the saved
    report identifiers. The backend configuration must match the selected name.
    """
    if dataset not in ("scannet_v2", "tum_rgbd"):
        raise ValueError("unsupported odometry dataset")
    backends = {
        "sparse": (SparseConfig, SparseRGBDBackend),
        "icp": (ICPConfig, DepthICPBackend),
        "hybrid": (HybridConfig, HybridRGBDBackend),
    }
    if backend_name not in backends:
        raise ValueError("unsupported odometry backend")
    config_type, backend_factory = backends[backend_name]
    if not isinstance(config, config_type):
        raise TypeError("configuration must match the selected odometry backend")
    backend = backend_factory(config)
    tum_config = tum_config or TUMConfig()
    output = output.expanduser()
    if any(
        (output / name).exists()
        for name in ("trajectory.npz", "frames.csv", "report.json")
    ):
        raise ValueError("output already contains a run, choose a new output directory")
    if output.exists() and not output.is_dir():
        raise ValueError("output must be a directory")
    source = source.expanduser().resolve()
    source_stat = source.stat()
    # Retain the historical extra contribution of the odometry CLI source.
    code_fingerprint = code_hash(Path(__file__).with_name("cli.py"))
    cv2.setNumThreads(1)
    started = perf_counter()
    reader_factory = (
        ScanNetReader(source, start=start, stop=stop)
        if dataset == "scannet_v2"
        else TUMReader(source, start=start, stop=stop, config=tum_config)
    )
    input_digest = hashlib.sha256()
    provenance = []

    def record_inputs(observations):
        for observation in observations:
            metadata = dict(observation.provenance or {})
            provenance.append(metadata)
            input_digest.update(
                json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
            )
            for array in (observation.rgb, observation.depth, observation.T_WC_D):
                input_digest.update(b"none" if array is None else array.tobytes())
            yield observation

    with reader_factory as reader:
        calibration = reader.calibration
        sequence_id, source_count = reader.sequence_id, reader.frame_count
        source_metadata = dict(getattr(reader, "metadata", {}))
        rows = []
        failures = 0
        for row in track_observations(record_inputs(reader), calibration, backend):
            rows.append(row)
            failures += row.status == "lost"
            if len(rows) % 100 == 0:
                print(
                    f"Processed {len(rows)} frames, tracking failures: {failures}",
                    flush=True,
                )
    elapsed = perf_counter() - started
    evaluation = evaluate_trajectory(
        rows,
        rpe_intervals,
        time_intervals_s=rpe_time_intervals_s,
        time_tolerance_s=rpe_time_tolerance_s,
    )
    report = {
        "format_version": 2,
        "backend": backend_name,
        "initial_pose_stage": {
            "sparse": "ransac",
            "icp": None,
            "hybrid": "accepted_sparse",
        }[backend_name],
        "config": asdict(config),
        "source": {
            "dataset": dataset,
            "sequence_id": sequence_id,
            "adapter": source_metadata,
            "association_config": None
            if dataset == "scannet_v2"
            else asdict(tum_config),
            "canonical_input_sha256": input_digest.hexdigest(),
            "path": str(source),
            "size_bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "source_frame_count": source_count,
            "start": start,
            "stop": source_count if stop is None else stop,
        },
        "calibration": {
            "K_R": calibration.K_R.tolist(),
            "K_D": calibration.K_D.tolist(),
            "T_RD": calibration.T_RD.tolist(),
            "rgb_shape": calibration.rgb_shape,
            "depth_shape": calibration.depth_shape,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "code_sha256": code_fingerprint,
            "invocation": list(invocation) if invocation is not None else [],
        },
        "timing": {
            "tracking_seconds": elapsed,
            "frames_per_second": len(rows) / elapsed if elapsed else None,
        },
        "timestamp_available_frames": sum(row.timestamp is not None for row in rows),
        "evaluation": evaluation,
        "frames": [
            {
                "frame_index": row.frame_index,
                "source_frame_index": start + row.frame_index,
                "provenance": provenance[row.frame_index],
                "segment_id": row.segment_id,
                "status": row.status,
                "reason": None if row.pair is None else row.pair.reason,
                "diagnostics": {} if row.pair is None else row.pair.diagnostics,
            }
            for row in rows
        ],
    }
    save_run(output, rows, report, start)
    print(
        f"Frames: {len(rows)}, tracked transitions: {evaluation['tracked_transitions']}/{evaluation['transition_count']}"
    )
    print(
        f"Failed transitions: {evaluation['failed_transitions']}, independent segments: {evaluation['segment_count']}"
    )
    for interval, metrics in evaluation["rpe_by_frame_interval"].items():
        translation, rotation = metrics["translation_m"], metrics["rotation_deg"]
        if translation["count"]:
            print(
                f"RPE {interval} frames ({translation['count']} pairs): RMSE {1000 * translation['rmse']:.3f} mm, {rotation['rmse']:.3f} deg"
            )
        else:
            print(f"RPE {interval} frames: unavailable")
    print(f"Run saved to {output}")
    return report
