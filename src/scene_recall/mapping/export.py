"""Portable map archives, colored PLY and a self-contained WebGL viewer."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scene_recall.geometry.camera import transform_points
from scene_recall.mapping.pipeline import Submap
from scene_recall.mapping.voxel import VoxelCloud


def save_cloud(path: Path, cloud: VoxelCloud, metadata: dict) -> None:
    """Keep full precision statistics in NPZ and viewer-compatible RGB in PLY."""
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


def _preview(path: Path, submap: Submap) -> None:
    image = Image.new("RGB", (1400, 760), "#101a24")
    draw = ImageDraw.Draw(image)
    angle, pitch = 0.55, -0.4
    R_y = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    clouds = (submap.estimated, submap.reference)
    populated = [cloud.points for cloud in clouds if len(cloud.points)]
    center = np.concatenate(populated).mean(axis=0) if populated else np.zeros(3)
    projected = [(cloud.points - center) @ (R_x @ R_y).T for cloud in clouds]
    extent = max(
        (np.abs(points[:, :2]).max() for points in projected if len(points)), default=1
    )
    scale = 290 / max(extent, 0.01)
    pixels = np.asarray(image).copy()
    for panel, (cloud, points) in enumerate(zip(clouds, projected)):
        xy = np.rint(points[:, :2] * scale + [350 + panel * 700, 395]).astype(int)
        valid = (xy[:, 0] >= panel * 700) & (xy[:, 0] < (panel + 1) * 700)
        valid &= (xy[:, 1] >= 90) & (xy[:, 1] < 735)
        order = np.argsort(points[:, 2])[::-1]
        order = order[valid[order]]
        pixels[xy[order, 1], xy[order, 0]] = cloud.colors[order]
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.text(
        (25, 18),
        f"Submap {submap.submap_id} / independent segment {submap.segment_id}",
        fill="white",
        font_size=23,
    )
    draw.text((25, 57), "Estimated poses", fill="#ffc55c", font_size=21)
    draw.text(
        (725, 57),
        "Reference poses / first-pose diagnostic alignment",
        fill="#66d9c3",
        font_size=21,
    )
    image.save(path)


def export_submap(output: Path, submap: Submap, start: int) -> dict:
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
    _preview(output / f"submap_{submap.submap_id:03d}.png", submap)
    T_MS = np.linalg.inv(submap.T_SM)
    trajectories = {
        "estimated": [
            transform_points(frame.row.T_SC_D[None, :3, 3], T_MS)[0].tolist()
            for frame in submap.keyframes
        ],
        "reference": [
            transform_points(
                frame.row.reference_pose[None, :3, 3], submap.T_MW_reference
            )[0].tolist()
            for frame in submap.keyframes
            if frame.row.reference_pose is not None
            and submap.T_MW_reference is not None
            and frame.row.frame_index in submap.report["reference_keyframe_indices"]
        ],
    }
    view = {
        "id": submap.submap_id,
        "segment": submap.segment_id,
        "trajectories": trajectories,
    }
    for name in ("estimated", "reference"):
        cloud = getattr(submap, name)
        # Display reduction is deterministic and never changes exported maps or metrics.
        indices = np.linspace(
            0, len(cloud.points) - 1, min(60_000, len(cloud.points)), dtype=int
        )
        view[name] = {
            "points": cloud.points[indices].round(5).ravel().tolist(),
            "colors": cloud.colors[indices].ravel().tolist(),
            "support": cloud.frame_support[indices].tolist(),
            "total": len(cloud.points),
        }
    return view


def save_viewer(output: Path, views: list[dict]) -> None:
    payload = json.dumps(views, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    (output / "viewer.html").write_text(
        template.replace("__MAP_DATA__", payload), encoding="utf-8"
    )
