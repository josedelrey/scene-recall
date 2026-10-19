"""Versioned, pickle-free multiview measurements for subsequent optimization."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from scene_recall.data import Calibration
from scene_recall.features import RGBDFeatures
from scene_recall.multiview.diagnostics import summarize_tracks
from scene_recall.multiview.model import FeatureFrame, TrackConfig


def load_measurements(run: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Load optimization-ready numeric tables and validate identity references.

    This does not resume descriptor association. Images and descriptors are not
    needed to consume the saved per-view RGB pixels and metric depth samples.
    """
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    with np.load(run / "measurements.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if (
        report["format_version"] != 1
        or arrays["format_version"].shape != ()
        or arrays["format_version"].item() != 1
    ):
        raise ValueError("unsupported multiview archive format")
    frames, tracks = arrays["frame_index"], arrays["track_id"]
    if (
        frames.ndim != 1
        or frames.dtype.kind not in "iu"
        or (frames < 0).any()
        or (np.diff(frames) <= 0).any()
    ):
        raise ValueError("invalid multiview frame indices")
    if not np.array_equal(tracks, np.arange(len(tracks))):
        raise ValueError("invalid multiview track IDs")
    settings = dict(report["config"])
    settings["pair_offsets"] = tuple(settings["pair_offsets"])
    TrackConfig(**settings)
    values = dict(report["calibration"])
    for name in ("K_R", "K_D", "T_RD"):
        values[name] = np.array(values[name], dtype=float)
    for name in ("rgb_shape", "depth_shape"):
        values[name] = tuple(values[name])
    Calibration(**values)
    for name, count in (
        ("segment_id", len(frames)),
        ("timestamp_s", len(frames)),
        ("track_segment_id", len(tracks)),
        ("track_status", len(tracks)),
        ("depth_views", len(tracks)),
    ):
        if arrays[name].shape != (count,):
            raise ValueError(f"invalid {name} length")
    if arrays["T_SR"].shape != (len(frames), 4, 4):
        raise ValueError("invalid pose table shape")
    for offsets, count, total in (
        ("feature_offsets", len(frames), len(arrays["pixels"])),
        ("track_offsets", len(tracks), len(arrays["observation_keys"])),
    ):
        values = arrays[offsets]
        if (
            values.dtype.kind not in "iu"
            or values.shape != (count + 1,)
            or values[0] != 0
            or values[-1] != total
            or (np.diff(values) < 0).any()
        ):
            raise ValueError(f"invalid {offsets}")
    sizes = dict(zip(map(int, frames), np.diff(arrays["feature_offsets"])))
    for i, frame_index in enumerate(frames):
        lo, hi = arrays["feature_offsets"][i : i + 2]
        features = RGBDFeatures(
            arrays["pixels"][lo:hi], None, arrays["points_R"][lo:hi]
        )
        stamp = arrays["timestamp_s"][i]
        FeatureFrame(
            int(frame_index),
            int(arrays["segment_id"][i]),
            None if np.isnan(stamp) else float(stamp),
            arrays["T_SR"][i],
            features,
        )
    segments = dict(zip(map(int, frames), arrays["segment_id"]))
    for name in ("observation_keys", "edge_source", "edge_target"):
        keys = arrays[name]
        if keys.dtype.kind not in "iu" or keys.ndim != 2 or keys.shape[1] != 2:
            raise ValueError(f"invalid {name}")
        if any(f not in sizes or k < 0 or k >= sizes[f] for f, k in keys):
            raise ValueError(f"unknown feature identity in {name}")
    keys = arrays["observation_keys"]
    if len(set(map(tuple, keys))) != len(keys):
        raise ValueError("a feature belongs to multiple tracks")
    for index in range(len(tracks)):
        members = keys[
            arrays["track_offsets"][index] : arrays["track_offsets"][index + 1]
        ]
        if (
            len(members) < 2
            or len(set(members[:, 0])) != len(members)
            or any(
                segments[f] != arrays["track_segment_id"][index] for f in members[:, 0]
            )
        ):
            raise ValueError("invalid track membership or segment")
    if (
        arrays["point_S"].shape != (len(tracks), 3)
        or not np.isfinite(arrays["point_S"]).all()
    ):
        raise ValueError("invalid landmark coordinates")
    if (
        arrays["pixels"].shape != (len(arrays["points_R"]), 2)
        or not np.isfinite(arrays["pixels"]).all()
    ):
        raise ValueError("invalid measurement pixels")
    if arrays["reprojection_errors_px"].shape != (len(keys),) or arrays[
        "depth_errors_m"
    ].shape != (len(keys),):
        raise ValueError("invalid track residual lengths")
    if report["summary"]["tracks"] != len(tracks) or report["summary"][
        "frame_count"
    ] != len(frames):
        raise ValueError("multiview report and archive disagree")
    return report, arrays


def measurement_arrays(result):
    """Return flat measurement tables with offsets, including unused features.

    A track observation indexes the frame's local feature row. Its depth sample
    remains in R and the landmark in S. Descriptors are deliberately omitted:
    reopening association requires re-extraction under the saved environment.
    """
    frames, tracks, edges = result.frames, result.landmarks, result.associations
    keys = [key for t in tracks for key in t.observations]
    return {
        "format_version": np.array(1, dtype=np.int64),
        "frame_index": np.array([f.frame_index for f in frames], dtype=np.int64),
        "segment_id": np.array([f.segment_id for f in frames], dtype=np.int64),
        "timestamp_s": np.array(
            [np.nan if f.timestamp is None else f.timestamp for f in frames],
            dtype=float,
        ),
        "T_SR": np.array([f.T_SR for f in frames]).reshape(-1, 4, 4),
        "feature_offsets": np.cumsum([0] + [len(f.features.pixels) for f in frames]),
        "pixels": np.concatenate([f.features.pixels for f in frames])
        if frames
        else np.empty((0, 2)),
        "points_R": np.concatenate([f.features.points_R for f in frames])
        if frames
        else np.empty((0, 3)),
        "track_id": np.array([t.track_id for t in tracks], dtype=np.int64),
        "track_segment_id": np.array([t.segment_id for t in tracks], dtype=np.int64),
        "track_status": np.array([t.status for t in tracks], dtype="U16"),
        "point_S": np.array([t.point_S for t in tracks]).reshape(-1, 3),
        "track_offsets": np.cumsum([0] + [len(t.observations) for t in tracks]),
        "observation_keys": np.array(keys, dtype=np.int64).reshape(-1, 2),
        "reprojection_errors_px": np.concatenate(
            [t.reprojection_errors_px for t in tracks]
        )
        if tracks
        else np.empty(0),
        "depth_errors_m": np.concatenate([t.depth_errors_m for t in tracks])
        if tracks
        else np.empty(0),
        "depth_views": np.array([t.depth_views for t in tracks], dtype=np.int64),
        "baseline_m": np.array([t.baseline_m for t in tracks]),
        "parallax_deg": np.array([t.parallax_deg for t in tracks]),
        "candidate_pairs": np.array(result.pairs, dtype=np.int64).reshape(-1, 2),
        "edge_source": np.array([e.source for e in edges], dtype=np.int64).reshape(
            -1, 2
        ),
        "edge_target": np.array([e.target for e in edges], dtype=np.int64).reshape(
            -1, 2
        ),
        "edge_ratio": np.array([e.ratio for e in edges]),
        "edge_reprojection_px": np.array(
            [np.nan if e.reprojection_px is None else e.reprojection_px for e in edges]
        ),
        "edge_depth_m": np.array(
            [np.nan if e.depth_m is None else e.depth_m for e in edges]
        ),
        "edge_status": np.array([e.status for e in edges], dtype="U16"),
        "edge_reason": np.array([e.reason for e in edges], dtype="U64"),
    }


def save_tracks(output: Path, result, calibration, metadata: dict):
    """Publish a complete fresh result, preserving any existing destination."""
    if output.exists() or output.is_symlink():
        raise ValueError("multiview output already exists")
    started = perf_counter()
    summary = summarize_tracks(result, calibration)
    result.timings["diagnostics_seconds"] = perf_counter() - started
    report = {
        "format_version": 1,
        "config": asdict(result.config),
        "calibration": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(calibration).items()
        },
        "metadata": metadata,
        "timings": result.timings,
        "summary": summary,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".multiview-", dir=output.parent) as temp:
        staging = Path(temp) / "result"
        staging.mkdir()
        np.savez_compressed(staging / "measurements.npz", **measurement_arrays(result))
        (staging / "report.json").write_text(serialized, encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise ValueError("output appeared during multiview export")
        staging.rename(output)
    return report
