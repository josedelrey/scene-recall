"""Create a self-contained offline Rerun recording from saved experiments."""

import argparse
import hashlib
import json
import os
import tempfile
from itertools import groupby
from pathlib import Path

import numpy as np

from scene_recall.data import Calibration
from scene_recall.mapping.cli import (
    load_trajectory,
    open_source,
    sha256,
    validate_source,
    verified_observations,
)
from scene_recall.odometry.pipeline import is_rigid
from scene_recall.visualization.rerun import (
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


def _cloud(path, submap, mode):
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
        metadata = json.loads(str(archive["metadata_json"]))
    if any(metadata.get(key) != value for key, value in submap.items()):
        raise ValueError(f"map metadata differs from mapping report: {path.name}")
    if metadata.get("pose_source") != mode:
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".recording-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "scene.rrd"
        # No default/global recording, server, browser, or viewer process is created.
        recording = rr.RecordingStream("scene_recall")
        try:
            recording.save(staged)
            recording.send_blueprint(
                blueprint(
                    segments,
                    submaps,
                    mapping["consistency"]["pair_keyframe_offsets"] if mapping else (),
                    show_images=image_mode != "none",
                )
            )
            recording.send_recording_name(output.stem)
            logger = SceneLogger(recording, calibration, display)
            guide = (
                "# SceneRecall\n\n"
                "Use the frame timeline to scrub synchronized cameras, images and diagnostics. "
                "Segment tabs contain independent coordinate frames. No transform connects them.\n\n"
                "Maps and complete paths are final, static results, not incremental replay. "
                "Submap tabs compare estimated and reference-pose maps in the recorded local gauge. "
                "Reference alignment is diagnostic and never corrects estimates. Missing or invalid "
                "reference poses break the reference path.\n\n"
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
            recording.flush()
        finally:
            recording.disconnect()
        # Hard-link publication is atomic and refuses a concurrent overwrite.
        os.link(staged, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-run", type=Path)
    parser.add_argument(
        "--odometry-run",
        type=Path,
        help="saved trajectory, also relocates the mapping input",
    )
    parser.add_argument(
        "--source-path", type=Path, help="relocate unchanged source images"
    )
    parser.add_argument("--output", type=Path, required=True, help="fresh .rrd file")
    parser.add_argument(
        "--images",
        choices=("keyframes", "all", "none"),
        help="default: mapping keyframes, or all odometry frames",
    )
    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="read only saved maps, without odometry or dataset",
    )
    parser.add_argument(
        "--submap",
        type=int,
        nargs="+",
        dest="submap_ids",
        help="include these submap IDs",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="display cap per cloud, default 200000",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="minimum map frame support for display",
    )
    args = parser.parse_args()
    try:
        export_recording(
            args.output.expanduser(),
            mapping_run=args.mapping_run.expanduser() if args.mapping_run else None,
            odometry_run=args.odometry_run.expanduser() if args.odometry_run else None,
            source_path=args.source_path.expanduser() if args.source_path else None,
            images=args.images,
            maps_only=args.maps_only,
            submap_ids=args.submap_ids,
            display=DisplayConfig(args.max_points, args.min_support),
        )
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Recording saved to {args.output}")
    print(f"Open with: uv run --locked --extra visualization rerun {args.output}")


if __name__ == "__main__":
    main()
