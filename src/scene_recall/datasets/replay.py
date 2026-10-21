"""Reopen saved dataset sources and verify their canonical RGB-D inputs."""

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from scene_recall.datasets.eth3d import ETH3DReader
from scene_recall.datasets.scannet import ScanNetReader
from scene_recall.datasets.tum import TUMConfig, TUMReader


def open_source(odometry: dict, source_override: Path | None = None):
    """Open the original adapter with the saved range and association policy."""
    source = odometry["source"]
    source_path = (source_override or Path(source["path"])).expanduser().resolve()
    reader_args = {"start": source["start"], "stop": source["stop"]}
    if source["dataset"] == "tum_rgbd":
        reader = TUMReader(
            source_path, config=TUMConfig(**source["association_config"]), **reader_args
        )
    elif source["dataset"] == "scannet_v2":
        reader = ScanNetReader(source_path, **reader_args)
    elif source["dataset"] == "eth3d_slam":
        reader = ETH3DReader(
            source_path, config=TUMConfig(**source["association_config"]), **reader_args
        )
    else:
        raise ValueError("unsupported odometry dataset")
    return reader, source_path


def validate_source(reader, odometry: dict) -> None:
    """Check adapter identity and calibration against the saved experiment."""
    source = odometry["source"]
    calibration = reader.calibration
    for field in fields(calibration):
        if not np.array_equal(
            getattr(calibration, field.name), odometry["calibration"][field.name]
        ):
            raise ValueError("dataset calibration differs from odometry calibration")
    if (
        reader.sequence_id != source["sequence_id"]
        or reader.frame_count != source["source_frame_count"]
    ):
        raise ValueError("dataset identity or frame count differs from odometry source")


def verified_observations(reader, odometry, frames, digest, metadata_by_frame):
    """Stream canonical inputs and verify the full fingerprint on exhaustion."""
    count = 0
    source = odometry["source"]
    for index, observation in enumerate(reader):
        count += 1
        if index >= len(frames):
            raise ValueError("dataset contains extra observations")
        if (
            observation.frame_index != index
            or observation.timestamp != frames[index].timestamp
        ):
            raise ValueError("observation index or timestamp disagrees with trajectory")
        metadata = dict(observation.provenance or {})
        if metadata != odometry["frames"][index]["provenance"]:
            raise ValueError("dataset provenance differs from odometry source")
        ref = frames[index].reference_pose
        if (ref is None) != (observation.T_WC_D is None) or (
            ref is not None
            and not np.array_equal(ref, observation.T_WC_D, equal_nan=True)
        ):
            raise ValueError("dataset reference poses differ from saved trajectory")
        payload = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
        digest.update(payload)
        frame_digest = hashlib.sha256(payload)
        for array in (observation.rgb, observation.depth, observation.T_WC_D):
            payload = b"none" if array is None else array.tobytes()
            digest.update(payload)
            frame_digest.update(payload)
        metadata_by_frame[index] = {
            "frame_index": index,
            "source_frame_index": source["start"] + index,
            "timestamp_s": observation.timestamp,
            "provenance": metadata,
            "canonical_sha256": frame_digest.hexdigest(),
        }
        yield observation
    if count != len(frames):
        raise ValueError("fewer observations than trajectory frames")
    if digest.hexdigest() != odometry["source"]["canonical_input_sha256"]:
        raise ValueError("canonical RGB-D input fingerprint differs from odometry run")
