"""Small TUM sources and immutable saved odometry fixtures."""

import hashlib
import json

import numpy as np
from PIL import Image

from scene_recall.datasets.tum import TUMConfig, TUMReader
from scene_recall.odometry.io import save_run
from scene_recall.odometry.pipeline import TrajectoryFrame


def _make_source(tmp_path):
    source = tmp_path / "tum"
    source.mkdir()
    Image.fromarray(np.full((480, 640, 3), [100, 180, 50], dtype=np.uint8)).save(
        source / "rgb.png"
    )
    Image.fromarray(np.full((480, 640), 10000, dtype=np.uint16)).save(
        source / "depth.png"
    )
    (source / "rgb.txt").write_text("1 rgb.png\n1.03 rgb.png\n1.06 rgb.png\n")
    (source / "depth.txt").write_text("1 depth.png\n1.03 depth.png\n1.06 depth.png\n")
    (source / "groundtruth.txt").write_text(
        "1 0 0 0 0 0 0 1\n1.03 0 0 0 0 0 0 1\n1.06 0 0 0 0 0 0 1\n"
    )
    return source


def _save_source_run(source, run, config, *, continuous):
    with TUMReader(source, config=config) as reader:
        observations = list(reader)
        calibration = reader.calibration
        metadata = reader.metadata
    rows = [
        TrajectoryFrame(
            i,
            o.timestamp,
            0 if continuous else int(i >= 1),
            ("initialized" if i == 0 else "tracked")
            if continuous
            else ("initialized", "lost", "tracked")[i],
            np.eye(4),
            o.T_WC_D,
        )
        for i, o in enumerate(observations)
    ]
    digest = hashlib.sha256()
    for o in observations:
        digest.update(
            json.dumps(dict(o.provenance), sort_keys=True, allow_nan=False).encode()
        )
        for array in (o.rgb, o.depth, o.T_WC_D):
            digest.update(b"none" if array is None else array.tobytes())
    report = {
        "format_version": 2,
        "backend": "sparse",
        "evaluation": {"frame_count": 3},
        "source": {
            "dataset": "tum_rgbd",
            "sequence_id": source.name,
            "path": str(source),
            "start": 0,
            "stop": 3,
            "source_frame_count": 3,
            "association_config": {
                "max_rgb_depth_difference_s": config.max_rgb_depth_difference_s,
                "max_ground_truth_difference_s": config.max_ground_truth_difference_s,
            },
            "adapter": metadata,
            "canonical_input_sha256": digest.hexdigest(),
        },
        "calibration": {
            "K_R": calibration.K_R.tolist(),
            "K_D": calibration.K_D.tolist(),
            "T_RD": calibration.T_RD.tolist(),
            "rgb_shape": calibration.rgb_shape,
            "depth_shape": calibration.depth_shape,
        },
        "frames": [
            {
                "frame_index": i,
                "source_frame_index": i,
                "segment_id": row.segment_id,
                "status": row.status,
                "provenance": dict(observations[i].provenance),
            }
            for i, row in enumerate(rows)
        ],
    }
    save_run(run, rows, report, 0)
    return source, run


def make_run(tmp_path, *, continuous=False):
    """Save one three-frame source with an optional continuous trajectory."""
    source = _make_source(tmp_path)
    return _save_source_run(
        source, tmp_path / "odometry", TUMConfig(), continuous=continuous
    )


def continuous_run(tmp_path, heldout_depth=2.0, missing_reference=None):
    """Construct held-out depth and reference omissions before publishing a run."""
    source = _make_source(tmp_path)
    Image.fromarray(
        np.full((480, 640), round(heldout_depth * 5000), dtype=np.uint16)
    ).save(source / "heldout.png")
    (source / "depth.txt").write_text("1 depth.png\n1.03 heldout.png\n1.06 depth.png\n")
    if missing_reference is not None:
        stamps = ["1", "1.03", "1.06"]
        (source / "groundtruth.txt").write_text(
            "".join(
                f"{stamp} 0 0 0 0 0 0 1\n"
                for i, stamp in enumerate(stamps)
                if i != missing_reference
            )
        )
    return _save_source_run(
        source,
        tmp_path / "odometry",
        TUMConfig(max_ground_truth_difference_s=0.01),
        continuous=True,
    )
