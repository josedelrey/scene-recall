"""Reconstruct and diagnose RGB-D submaps from a verified saved run."""

import hashlib
import json
import platform
import tempfile
from collections import deque
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import __version__ as pillow_version

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.mapping.evaluation import depth_consistency, summarize_consistency
from scene_recall.mapping.export import export_submap
from scene_recall.mapping.pipeline import reconstruct
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.pipeline import is_rigid
from scene_recall.run_metadata import code_hash, peak_rss, sha256


def run_mapping(
    run: Path,
    output: Path,
    config: MappingConfig,
    source_override: Path | None = None,
    pair_offsets: tuple[int, ...] = (1, 3, 9),
) -> dict:
    """Verify canonical input identity before atomically publishing a new run."""
    if output.exists():
        raise ValueError("output already exists, choose a new output directory")
    if (
        not pair_offsets
        or len(set(pair_offsets)) != len(pair_offsets)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in pair_offsets
        )
    ):
        raise ValueError("pair offsets must be unique positive keyframe offsets")
    odometry, frames = load_trajectory(run)
    source = odometry["source"]
    reader, source_path = open_source(odometry, source_override)
    cv2.setNumThreads(1)
    started = perf_counter()
    digest = hashlib.sha256()
    keyframe_metadata = {}
    submaps, pairs = [], []
    tail = deque(maxlen=max(pair_offsets))
    excluded_cross_segment = excluded_reference = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        reader,
        tempfile.TemporaryDirectory(prefix=".mapping-", dir=output.parent) as temporary,
    ):
        staging = Path(temporary) / "result"
        staging.mkdir()
        calibration = reader.calibration
        validate_source(reader, odometry)

        for submap in reconstruct(
            verified_observations(reader, odometry, frames, digest, keyframe_metadata),
            frames,
            calibration,
            config,
        ):
            for current in submap.keyframes:
                for offset in pair_offsets:
                    if len(tail) < offset:
                        continue
                    previous = tail[-offset]
                    if previous.row.segment_id != current.row.segment_id:
                        excluded_cross_segment += 1
                        continue
                    if any(
                        frame.row.reference_pose is None
                        or not is_rigid(frame.row.reference_pose)
                        for frame in (previous, current)
                    ):
                        excluded_reference += 1
                        continue
                    for first, second in ((previous, current), (current, previous)):
                        entry = {
                            "source_frame": first.row.frame_index,
                            "target_frame": second.row.frame_index,
                            "segment_id": current.row.segment_id,
                            "keyframe_offset": offset,
                        }
                        for name, a, b in (
                            ("estimated", first.row.T_SC_D, second.row.T_SC_D),
                            (
                                "reference",
                                first.row.reference_pose,
                                second.row.reference_pose,
                            ),
                        ):
                            entry[name] = depth_consistency(
                                first.samples,
                                second.samples,
                                np.linalg.inv(b) @ a,
                                calibration.K_D,
                                config.consistency_tolerance_m,
                            )
                        pairs.append(entry)
                tail.append(current)
            submaps.append(submap.report)
            export_submap(staging, submap, source["start"])
            print(
                f"Built submap {submap.submap_id}, segment {submap.segment_id}: "
                f"{len(submap.keyframes)} keyframes, {len(submap.estimated.points)} voxels",
                flush=True,
            )
        if digest.hexdigest() != source["canonical_input_sha256"]:
            raise ValueError(
                "canonical RGB-D input fingerprint differs from odometry run"
            )
        selected = [index for submap in submaps for index in submap["keyframe_indices"]]
        keyframes = [keyframe_metadata[index] for index in selected]
        (staging / "keyframes.json").write_text(
            json.dumps(keyframes, indent=2, allow_nan=False) + "\n"
        )
        report = {
            "format_version": 1,
            "representation": "equal_keyframe_weighted_colored_surface_voxels",
            "config": asdict(config),
            "odometry": {
                "run": str(run.resolve()),
                "backend": odometry["backend"],
                "report_sha256": sha256(run / "report.json"),
                "trajectory_sha256": sha256(run / "trajectory.npz"),
            },
            "source": {**source, "resolved_path": str(source_path)},
            "verified_canonical_input_sha256": digest.hexdigest(),
            "calibration": odometry["calibration"],
            "coordinate_policy": "independent segments, bounded chunks retain T_SM, reference first-pose diagnostic alignment only",
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "pillow": pillow_version,
                "code_sha256": code_hash(),
                "platform": platform.platform(),
            },
            "timing": {"total_seconds": perf_counter() - started},
            "peak_process_rss_mib_linux": peak_rss(),
            "frame_count": len(frames),
            "keyframe_count": len(selected),
            "segment_count": len({frame.segment_id for frame in frames}),
            "submaps": submaps,
            "consistency": {
                "pair_keyframe_offsets": pair_offsets,
                "tolerance_m": config.consistency_tolerance_m,
                "excluded_cross_segment_pairs": excluded_cross_segment,
                "excluded_reference_pairs": excluded_reference,
                "by_offset": {
                    str(offset): {
                        name: summarize_consistency(
                            [
                                entry[name]
                                for entry in pairs
                                if entry["keyframe_offset"] == offset
                            ]
                        )
                        for name in ("estimated", "reference")
                    }
                    for offset in pair_offsets
                },
                "pairs": pairs,
            },
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        # Renaming the staged directory publishes only a completely validated run.
        if output.exists():
            raise ValueError(
                "output appeared during reconstruction, refusing overwrite"
            )
        staging.rename(output)
    print(f"Mapping run saved to {output}")
    return report
