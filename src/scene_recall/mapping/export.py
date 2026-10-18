"""Scientific map archives and colored PLY exports."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from scene_recall.mapping.pipeline import Submap
from scene_recall.mapping.voxel import VoxelCloud


def save_cloud(path: Path, cloud: VoxelCloud, metadata: dict) -> None:
    """Keep full precision statistics in NPZ and portable RGB in PLY."""
    np.savez_compressed(
        path.with_suffix(".npz"),
        **{field.name: getattr(cloud, field.name) for field in fields(cloud)},
        metadata_json=np.array(json.dumps(metadata, allow_nan=False)),
    )
    records = np.empty(
        len(cloud.points),
        dtype=[
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("frame_support", "<u4"),
        ],
    )
    for axis, name in enumerate(("x", "y", "z")):
        records[name] = cloud.points[:, axis]
    for channel, name in enumerate(("red", "green", "blue")):
        records[name] = cloud.colors[:, channel]
    records["frame_support"] = cloud.frame_support
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment coordinate_frame {metadata['coordinate_frame']}\n"
        "comment units meters\n"
        f"element vertex {len(records)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uint frame_support\nend_header\n"
    )
    with path.with_suffix(".ply").open("wb") as stream:
        stream.write(header.encode("ascii"))
        records.tofile(stream)


def export_submap(output: Path, submap: Submap, start: int) -> None:
    """Persist scientific arrays and provenance independently of visualization."""
    metadata = {
        **submap.report,
        "source_keyframe_indices": [
            start + frame.row.frame_index for frame in submap.keyframes
        ],
        "keyframe_provenance_file": "keyframes.json",
    }
    for name in ("estimated", "reference"):
        save_cloud(
            output / f"submap_{submap.submap_id:03d}_{name}",
            getattr(submap, name),
            {**metadata, "pose_source": name},
        )
