"""Reconstruct and diagnose RGB-D submaps from a saved odometry run."""

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from collections import deque
from dataclasses import asdict, fields
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import __version__ as pillow_version

import scene_recall
from scene_recall.datasets.scannet import ScanNetReader
from scene_recall.datasets.tum import TUMConfig, TUMReader
from scene_recall.mapping.evaluation import depth_consistency, summarize_consistency
from scene_recall.mapping.export import export_submap, save_viewer
from scene_recall.mapping.pipeline import reconstruct, validate_trajectory
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.pipeline import TrajectoryFrame, is_rigid


def _peak_rss() -> float | None:
    if not sys.platform.startswith("linux"):
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _code_hash() -> str:
    digest = hashlib.sha256()
    root = Path(scene_recall.__file__).parent
    for path in sorted(root.rglob("*")):
        if path.suffix in (".py", ".html"):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


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
    source_path = (source_override or Path(source["path"])).expanduser().resolve()
    reader_args = {"start": source["start"], "stop": source["stop"]}
    if source["dataset"] == "tum_rgbd":
        reader = TUMReader(
            source_path, config=TUMConfig(**source["association_config"]), **reader_args
        )
    elif source["dataset"] == "scannet_v2":
        reader = ScanNetReader(source_path, **reader_args)
    else:
        raise ValueError("unsupported odometry dataset")
    cv2.setNumThreads(1)
    started = perf_counter()
    digest = hashlib.sha256()
    keyframe_metadata = {}
    submaps, views, pairs = [], [], []
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
        for field in fields(calibration):
            if not np.array_equal(
                getattr(calibration, field.name), odometry["calibration"][field.name]
            ):
                raise ValueError(
                    "dataset calibration differs from odometry calibration"
                )
        if (
            reader.sequence_id != source["sequence_id"]
            or reader.frame_count != source["source_frame_count"]
        ):
            raise ValueError(
                "dataset identity or frame count differs from odometry source"
            )

        def verified_observations():
            for index, observation in enumerate(reader):
                if index >= len(frames):
                    raise ValueError("dataset contains extra observations")
                metadata = dict(observation.provenance or {})
                if metadata != odometry["frames"][index]["provenance"]:
                    raise ValueError("dataset provenance differs from odometry source")
                ref = frames[index].reference_pose
                if (ref is None) != (observation.T_WC_D is None) or (
                    ref is not None
                    and not np.array_equal(ref, observation.T_WC_D, equal_nan=True)
                ):
                    raise ValueError(
                        "dataset reference poses differ from saved trajectory"
                    )
                payload = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
                digest.update(payload)
                frame_digest = hashlib.sha256(payload)
                for array in (observation.rgb, observation.depth, observation.T_WC_D):
                    payload = b"none" if array is None else array.tobytes()
                    digest.update(payload)
                    frame_digest.update(payload)
                keyframe_metadata[index] = {
                    "frame_index": index,
                    "source_frame_index": source["start"] + index,
                    "timestamp_s": observation.timestamp,
                    "provenance": metadata,
                    "canonical_sha256": frame_digest.hexdigest(),
                }
                yield observation

        for submap in reconstruct(verified_observations(), frames, calibration, config):
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
            views.append(export_submap(staging, submap, source["start"]))
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
        save_viewer(staging, views)
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
                "code_sha256": _code_hash(),
                "platform": platform.platform(),
            },
            "timing": {"total_seconds": perf_counter() - started},
            "peak_process_rss_mib_linux": _peak_rss(),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-path", type=Path, help="relocate the unchanged source dataset"
    )
    parser.add_argument(
        "--pair-keyframe-offsets", type=int, nargs="+", default=[1, 3, 9]
    )
    for field in fields(MappingConfig):
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(field.default),
            default=field.default,
        )
    args = parser.parse_args()
    try:
        config = MappingConfig(
            **{field.name: getattr(args, field.name) for field in fields(MappingConfig)}
        )
        run_mapping(
            args.odometry_run.expanduser(),
            args.output_dir.expanduser(),
            config,
            args.source_path,
            tuple(args.pair_keyframe_offsets),
        )
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
