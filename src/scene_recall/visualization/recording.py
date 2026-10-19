"""Verify saved scientific runs and export self-contained Rerun recordings."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path

import numpy as np

from scene_recall.data import Calibration
from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.geometry.transforms import is_rigid
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256
from scene_recall.visualization.rerun import (
    COMPARISON_GUIDE,
    DisplayConfig,
    SceneLogger,
    blueprint,
    require_rerun,
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _calibration(metadata):
    return Calibration(
        **{
            key: tuple(value) if key.endswith("shape") else np.array(value, dtype=float)
            for key, value in metadata.items()
        }
    )


def _cloud(path, submap, mode, *, require_metadata=True, pose_source=True):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name]
            for name in (
                "keys",
                "points",
                "colors",
                "frame_support",
                "color_support",
                "position_std_m",
            )
        }
        metadata = (
            json.loads(str(archive["metadata_json"]))
            if "metadata_json" in archive
            else None
        )
    if require_metadata and metadata is None:
        raise ValueError(f"map metadata is missing: {path.name}")
    if metadata is not None and any(
        metadata.get(key) != value for key, value in submap.items()
    ):
        raise ValueError(f"map metadata differs from mapping report: {path.name}")
    if pose_source and metadata is not None and metadata.get("pose_source") != mode:
        raise ValueError(f"map pose source differs: {path.name}")
    count = len(arrays["points"])
    if (
        any(
            arrays[name].shape
            != ((count, 3) if name in ("keys", "points", "colors") else (count,))
            for name in arrays
        )
        or count != submap[mode]["voxel_count"]
    ):
        raise ValueError(f"invalid map array shape or voxel count: {path.name}")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError(f"nonfinite map values: {path.name}")
    if arrays["colors"].dtype != np.uint8 or not np.issubdtype(
        arrays["keys"].dtype, np.integer
    ):
        raise ValueError(f"invalid map colors or voxel keys: {path.name}")
    if any(
        (arrays[name] < 0).any()
        for name in ("frame_support", "color_support", "position_std_m")
    ):
        raise ValueError(f"negative map support or scatter: {path.name}")
    return arrays


def _validate_submaps(report, frames):
    ids = set()
    for submap in report["submaps"]:
        index, segment = submap["submap_id"], submap["segment_id"]
        if (
            any(type(value) is not int or value < 0 for value in (index, segment))
            or index in ids
        ):
            raise ValueError("invalid or duplicate submap identity")
        ids.add(index)
        indices = submap["keyframe_indices"]
        if (
            not indices
            or any(type(i) is not int or i < 0 for i in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("invalid submap keyframe indices")
        if not is_rigid(np.array(submap["T_SM"])):
            raise ValueError("invalid submap-to-segment transform")
        alignment = submap["T_MW_reference_diagnostic"]
        if alignment is not None and not is_rigid(np.array(alignment)):
            raise ValueError("invalid submap reference alignment")
        if frames and (
            indices[-1] >= len(frames)
            or any(
                frames[i].segment_id != segment
                for i in range(indices[0], indices[-1] + 1)
            )
            or not np.allclose(
                submap["T_SM"], frames[indices[0]].T_SC_D, rtol=0, atol=1e-6
            )
        ):
            raise ValueError("submap gauge or frame range differs from trajectory")


@contextmanager
def recording_stream(output):
    """Stage one self-contained recording and publish only after a clean close."""
    rr = require_rerun()
    if output.suffix != ".rrd":
        raise ValueError("output must have the .rrd extension")
    if output.exists():
        raise ValueError("output already exists, choose a new .rrd path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recording-", dir=output.parent) as folder:
        staged = Path(folder) / "scene.rrd"
        recording = rr.RecordingStream("scene_recall")
        try:
            recording.save(staged)
            recording.send_recording_name(output.stem)
            yield recording
            recording.flush()
        finally:
            recording.disconnect()
        os.link(staged, output)


def export_recording(
    output: Path,
    *,
    mapping_run: Path | None = None,
    odometry_run: Path | None = None,
    source_path: Path | None = None,
    images: str | None = None,
    maps_only: bool = False,
    submap_ids: list[int] | None = None,
    display: DisplayConfig | None = None,
) -> Path:
    """Verify saved inputs and publish a complete recording without overwriting.

    Source images are verified across the entire saved range before publication.
    Map-only and image-free modes need no original dataset. No mapper, tracker
    or evaluation routine runs here.
    """
    rr = require_rerun()
    display = display or DisplayConfig()
    if output.suffix != ".rrd":
        raise ValueError("output must have the .rrd extension")
    if output.exists():
        raise ValueError("output already exists, choose a new .rrd path")
    if mapping_run is None and odometry_run is None:
        raise ValueError("provide --mapping-run or --odometry-run")
    if maps_only and (
        mapping_run is None or odometry_run is not None or images not in (None, "none")
    ):
        raise ValueError(
            "--maps-only requires a mapping run and excludes odometry and images"
        )
    if images not in (None, "all", "keyframes", "none"):
        raise ValueError("images must be all, keyframes or none")
    if submap_ids is not None and (mapping_run is None or not submap_ids):
        raise ValueError("submap selection requires a mapping run and at least one ID")
    mapping = _read_json(mapping_run / "report.json") if mapping_run else None
    if mapping is not None and (
        mapping.get("format_version") != 1
        or mapping.get("representation")
        != "equal_keyframe_weighted_colored_surface_voxels"
    ):
        raise ValueError("expected a standard format-1 mapping run")
    if mapping and odometry_run is None and not maps_only:
        odometry_run = Path(mapping["odometry"]["run"]).expanduser()
        if not odometry_run.is_dir():
            raise ValueError(
                "saved odometry run is unavailable. Relocate it with --odometry-run "
                "or use --maps-only"
            )
    odometry, frames = load_trajectory(odometry_run) if odometry_run else (None, [])
    if mapping and odometry:
        for name in ("report.json", "trajectory.npz"):
            key = "report_sha256" if name == "report.json" else "trajectory_sha256"
            if sha256(odometry_run / name) != mapping["odometry"][key]:
                raise ValueError("odometry input differs from saved mapping run")
        if (
            mapping["calibration"] != odometry["calibration"]
            or mapping["verified_canonical_input_sha256"]
            != odometry["source"]["canonical_input_sha256"]
        ):
            raise ValueError("mapping calibration or source differs from odometry")
    image_mode = "none" if maps_only else images or ("keyframes" if mapping else "all")
    if image_mode == "keyframes" and mapping is None:
        raise ValueError("--images keyframes requires --mapping-run")
    if source_path is not None and image_mode == "none":
        raise ValueError("--source-path requires image logging")
    submaps = mapping["submaps"] if mapping else []
    if mapping:
        _validate_submaps(mapping, frames)
    if submap_ids is not None:
        if set(submap_ids) - {s["submap_id"] for s in submaps}:
            raise ValueError("unknown submap ID")
        submaps = [s for s in submaps if s["submap_id"] in submap_ids]
    selected_images = {i for s in submaps for i in s["keyframe_indices"]}
    segments = sorted({row.segment_id for row in frames})
    source_start = (odometry or mapping)["source"]["start"]
    calibration = _calibration((odometry or mapping)["calibration"])
    with recording_stream(output) as recording:
        recording.send_blueprint(
            blueprint(
                segments,
                submaps,
                mapping["consistency"]["pair_keyframe_offsets"] if mapping else (),
                show_images=image_mode != "none",
            )
        )
        logger = SceneLogger(recording, calibration, display)
        guide = (
            "# SceneRecall\n\n"
            "Use the frame timeline to scrub synchronized cameras, images and diagnostics. "
            "Segment tabs contain independent coordinate frames. No transform connects them.\n\n"
            "Maps and complete paths are final, static results, not incremental replay. "
            "Submap tabs compare estimated and reference-pose maps in the recorded local gauge. "
            "Reference alignment is diagnostic and never corrects estimates. Missing or invalid "
            "reference poses break the reference path.\n\n"
            f"{COMPARISON_GUIDE}\n\n"
            "RGB and native z-depth use separate calibrated optical cameras. Depth is in meters. "
            "Images are shown only at recorded samples, so sparse images are never projected "
            "using a different frame's pose. Select an image to inspect its sample time.\n\n"
            "Frame support colors run blue (1) through cyan to yellow (the full cloud maximum), "
            "on a log2 scale per cloud. Select a point to inspect frame/color support and positional "
            "scatter. Display limits never alter scientific arrays or scores.\n\n"
            f"Rerun SDK: {rr.__version__}\n\n"
            f"Display: {display}\n\nImage mode: {image_mode}\n\n"
            "NPZ, PLY and JSON remain the scientific outputs. The RRD embeds its data and blueprint "
            "and can be opened without source files or network access.\n\n"
        )
        recording.log(
            "recording/guide",
            rr.TextDocument(guide, media_type="text/markdown"),
            static=True,
        )
        for name, report in (("mapping", mapping), ("odometry", odometry)):
            if report is not None:
                recording.log(
                    f"recording/{name}_report",
                    rr.TextDocument(
                        json.dumps(report, indent=2), media_type="application/json"
                    ),
                    static=True,
                )
        for _, group in groupby(frames, key=lambda row: row.segment_id):
            logger.segment(list(group), source_start)
        for row, metadata in zip(
            frames, odometry["frames"] if odometry else [], strict=True
        ):
            logger.set_frame(row, source_start)
            logger.scalars("diagnostics/tracking", metadata.get("diagnostics", {}))
            recording.log(
                "diagnostics/events",
                rr.TextLog(json.dumps(metadata, sort_keys=True)),
            )
        for submap in submaps:
            clouds = {
                mode: _cloud(
                    mapping_run / f"submap_{submap['submap_id']:03d}_{mode}.npz",
                    submap,
                    mode,
                )
                for mode in ("estimated", "reference")
            }
            rows = frames[
                submap["keyframe_indices"][0] : submap["keyframe_indices"][-1] + 1
            ]
            logger.submap(submap, clouds, rows, source_start)
        if mapping:
            for pair in mapping["consistency"]["pairs"]:
                recording.reset_time()
                recording.set_time(
                    "frame",
                    sequence=source_start
                    + max(pair["source_frame"], pair["target_frame"]),
                )
                direction = (
                    "forward"
                    if pair["source_frame"] < pair["target_frame"]
                    else "reverse"
                )
                root = f"diagnostics/consistency/offset_{pair['keyframe_offset']}/{direction}"
                for mode in ("estimated", "reference"):
                    logger.scalars(f"{root}/{mode}", pair[mode])
        if image_mode != "none":
            reader, _ = open_source(odometry, source_path)
            with reader:
                validate_source(reader, odometry)
                iterator = verified_observations(
                    reader, odometry, frames, hashlib.sha256(), {}
                )
                for observation in iterator:
                    row = frames[observation.frame_index]
                    logger.set_frame(row, source_start)
                    if (
                        image_mode == "all"
                        or observation.frame_index in selected_images
                    ):
                        logger.observation(observation, row)
                    else:
                        root = f"segments/{row.segment_id}/estimated/camera"
                        for sensor in ("rgb", "depth"):
                            recording.log(
                                root + f"/{sensor}/image", rr.Clear(recursive=True)
                            )
    return output
