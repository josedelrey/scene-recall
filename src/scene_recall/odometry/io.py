"""Read and write versioned odometry archives and per-frame diagnostics."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from scene_recall.odometry.pipeline import TrajectoryFrame, validate_trajectory


def _pose_array(poses) -> np.ndarray:
    return np.array(
        [np.full((4, 4), np.nan) if pose is None else pose for pose in poses],
        dtype=np.float64,
    ).reshape(-1, 4, 4)


def save_run(
    output: Path, rows: list[TrajectoryFrame], report: dict, start: int
) -> None:
    """Stage a complete run and atomically publish it to a fresh directory.

    Failures discard staging files. Existing destinations, including empty
    directories and dangling symlinks, are refused without modifying them.
    """
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists, choose a new output directory")
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".odometry-", dir=output.parent) as temp:
        staging = Path(temp) / "result"
        staging.mkdir()
        _write_run(staging, rows, serialized, start)
        if output.exists() or output.is_symlink():
            raise ValueError("output appeared during export, refusing overwrite")
        staging.rename(output)


def _write_run(
    output: Path, rows: list[TrajectoryFrame], serialized: str, start: int
) -> None:
    """Write scientific files inside a caller-owned staging directory."""
    np.savez_compressed(
        output / "trajectory.npz",
        frame_index=np.array([row.frame_index for row in rows], dtype=np.int64),
        source_frame_index=np.array(
            [start + row.frame_index for row in rows], dtype=np.int64
        ),
        timestamp_s=np.array(
            [np.nan if row.timestamp is None else row.timestamp for row in rows]
        ),
        segment_id=np.array([row.segment_id for row in rows], dtype=np.int64),
        status=np.array([row.status for row in rows], dtype=str),
        T_SC_D=_pose_array(row.T_SC_D for row in rows),
        T_WC_D_reference=_pose_array(row.reference_pose for row in rows),
        reference_present=np.array(
            [row.reference_pose is not None for row in rows], dtype=bool
        ),
        T_C1C0=_pose_array(
            None if row.pair is None else row.pair.T_C1C0 for row in rows
        ),
        candidate_T_C1C0=_pose_array(
            None if row.pair is None else row.pair.candidate_T_C1C0 for row in rows
        ),
        ransac_T_C1C0=_pose_array(
            None if row.pair is None else row.pair.ransac_T_C1C0 for row in rows
        ),
        initial_T_C1C0=_pose_array(
            None if row.pair is None else row.pair.initial_T_C1C0 for row in rows
        ),
    )
    columns = [
        "frame_index",
        "source_frame_index",
        "timestamp_s",
        "segment_id",
        "status",
        "reason",
        "segment_x_m",
        "segment_y_m",
        "segment_z_m",
        "source_keypoints",
        "target_keypoints",
        "ratio_matches",
        "unique_matches",
        "depth_valid_matches",
        "ransac_inliers",
        "inliers",
        "inlier_ratio",
        "source_coverage",
        "target_coverage",
        "refinement",
        "inlier_rmse_px",
    ]
    columns += sorted(
        {key for row in rows if row.pair is not None for key in row.pair.diagnostics}
        - set(columns)
    )
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **({} if row.pair is None else row.pair.diagnostics),
                    "frame_index": row.frame_index,
                    "source_frame_index": start + row.frame_index,
                    "timestamp_s": row.timestamp,
                    "segment_id": row.segment_id,
                    "status": row.status,
                    "reason": None if row.pair is None else row.pair.reason,
                    **dict(
                        zip(
                            ("segment_x_m", "segment_y_m", "segment_z_m"),
                            row.T_SC_D[:3, 3],
                        )
                    ),
                }
            )
    (output / "report.json").write_text(serialized, encoding="utf-8")


def load_trajectory(run: Path) -> tuple[dict, list[TrajectoryFrame]]:
    """Read format-2 exports and reject mismatched or malformed archive metadata."""
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report["format_version"] != 2:
        raise ValueError("mapping requires an odometry format-2 export")
    source = report["source"]
    count = report["evaluation"]["frame_count"]
    if count <= 0 or source["stop"] - source["start"] != count:
        raise ValueError("odometry run must contain a consistent nonempty frame range")
    rows = []
    with np.load(run / "trajectory.npz", allow_pickle=False) as archive:
        expected = {
            "frame_index": (count,),
            "source_frame_index": (count,),
            "timestamp_s": (count,),
            "segment_id": (count,),
            "status": (count,),
            "T_SC_D": (count, 4, 4),
            "T_WC_D_reference": (count, 4, 4),
            "reference_present": (count,),
        }
        for name, shape in expected.items():
            if archive[name].shape != shape:
                raise ValueError(f"invalid trajectory archive shape: {name}")
        for name in ("frame_index", "source_frame_index", "segment_id"):
            if archive[name].dtype.kind not in "iu":
                raise ValueError(f"trajectory {name} must contain integers")
        if archive["reference_present"].dtype.kind != "b":
            raise ValueError("reference_present must be boolean")
        if not np.array_equal(
            archive["frame_index"], np.arange(count)
        ) or not np.array_equal(
            archive["source_frame_index"], np.arange(source["start"], source["stop"])
        ):
            raise ValueError("trajectory source range disagrees with report")
        if len(report["frames"]) != count:
            raise ValueError("odometry frame metadata count disagrees with archive")
        for index in range(count):
            stamp = archive["timestamp_s"][index]
            if not np.isnan(stamp) and not np.isfinite(stamp):
                raise ValueError("trajectory timestamps must be finite or NaN")
            ref = archive["T_WC_D_reference"][index]
            present = archive["reference_present"][index]
            if not present and not np.isnan(ref).all():
                raise ValueError("missing reference pose must be all NaN")
            row = TrajectoryFrame(
                index,
                None if np.isnan(stamp) else float(stamp),
                int(archive["segment_id"][index]),
                str(archive["status"][index]),
                archive["T_SC_D"][index].copy(),
                ref.copy() if present else None,
            )
            meta = report["frames"][index]
            if any(
                meta[name] != value
                for name, value in (
                    ("frame_index", index),
                    ("source_frame_index", source["start"] + index),
                    ("segment_id", row.segment_id),
                    ("status", row.status),
                )
            ):
                raise ValueError("trajectory frame metadata disagrees with archive")
            rows.append(row)
    validate_trajectory(rows)
    return report, rows
