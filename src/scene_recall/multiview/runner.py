"""Build multiview measurements from verified saved odometry and RGB-D inputs."""

import hashlib
import os
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.features import extract_rgbd_features
from scene_recall.multiview.diagnostics import summarize_tracks
from scene_recall.multiview.io import measurement_arrays, save_tracks
from scene_recall.multiview.model import FeatureFrame, TrackConfig
from scene_recall.multiview.tracking import build_tracks
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, peak_rss, sha256


def run_multiview(
    odometry_run: Path,
    output: Path,
    *,
    config: TrackConfig | None = None,
    stride: int = 1,
    source_override: Path | None = None,
    controls: bool = False,
    invocation: list[str] | None = None,
) -> dict:
    """Extract a bounded offline feature graph, with no trajectory corrections.

    Full source verification precedes publication, including unselected frames.
    Optional controls compare adjacent-only association, tighter gates and an
    exact scientific-array repeat using the same extracted measurements.
    """
    if output.exists() or output.is_symlink():
        raise ValueError("multiview output already exists")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    config = config or TrackConfig()
    report, trajectory = load_trajectory(odometry_run)
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=config.max_features)
    reader, source = open_source(report, source_override)
    digest, metadata_by_frame = hashlib.sha256(), {}
    selected = []
    extraction_seconds = 0.0
    started = perf_counter()
    with reader:
        validate_source(reader, report)
        calibration = reader.calibration
        observations = verified_observations(
            reader, report, trajectory, digest, metadata_by_frame
        )
        for observation in observations:
            row = trajectory[observation.frame_index]
            # Keep every segment's root even when temporal sampling would skip it.
            if row.frame_index % stride and row.status == "tracked":
                continue
            before = perf_counter()
            features = extract_rgbd_features(
                observation, calibration, sift, config.depth_edge_threshold_m
            )
            extraction_seconds += perf_counter() - before
            selected.append(FeatureFrame.from_trajectory(row, calibration, features))
    replay_seconds = perf_counter() - started
    print(f"Extracted {len(selected)} views in {replay_seconds:.1f}s", flush=True)
    result = build_tracks(selected, calibration, config)
    print(f"Constructed {len(result.landmarks)} tracks", flush=True)
    result.timings.update(
        extraction_seconds=extraction_seconds, verified_replay_seconds=replay_seconds
    )
    metadata = {
        "odometry_run": str(odometry_run.resolve()),
        "odometry_report_sha256": sha256(odometry_run / "report.json"),
        "odometry_trajectory_sha256": sha256(odometry_run / "trajectory.npz"),
        "source": report["source"],
        "reopened_source": str(source),
        "canonical_inputs_verified": True,
        "stride": stride,
        "selected_input_sha256": {
            str(f.frame_index): metadata_by_frame[f.frame_index]["canonical_sha256"]
            for f in selected
        },
        "vo_failures": [
            {"frame_index": f["frame_index"], "reason": f.get("reason")}
            for f in report["frames"]
            if f["status"] == "lost"
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "code_sha256": code_hash(),
            "invocation": invocation or [],
        },
    }
    if controls:
        control_results = {}
        for name, settings in (
            ("adjacent_only", replace(config, pair_offsets=(1,))),
            (
                "tighter_gates",
                replace(
                    config,
                    reprojection_px=config.reprojection_px / 2,
                    depth_absolute_m=config.depth_absolute_m / 2,
                    depth_relative=config.depth_relative / 2,
                ),
            ),
        ):
            control = build_tracks(selected, calibration, settings)
            summary = summarize_tracks(control, calibration, include_covisibility=False)
            control_results[name] = {
                key: value
                for key, value in summary.items()
                if key not in ("per_frame", "covisibility")
            }
            control_results[name]["timings"] = control.timings
            print(f"Completed {name} control", flush=True)
        repeated = build_tracks(selected, calibration, config)
        original_arrays, repeat_arrays = (
            measurement_arrays(result),
            measurement_arrays(repeated),
        )
        mismatches = []
        for key, original in original_arrays.items():
            kwargs = {} if original.dtype.kind in "US" else {"equal_nan": True}
            if not np.array_equal(original, repeat_arrays[key], **kwargs):
                mismatches.append(key)
        if mismatches:
            raise RuntimeError(f"multiview scientific repeat differs: {mismatches}")
        metadata["controls"] = control_results
        metadata["repeat"] = {
            "matched_arrays": len(original_arrays),
            "mismatches": mismatches,
        }
    metadata["peak_rss_mib"] = peak_rss()
    result.timings["total_before_export_seconds"] = perf_counter() - started
    return save_tracks(output, result, calibration, metadata)
