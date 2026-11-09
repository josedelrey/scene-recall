"""Independent OpenCV dense RGB-D odometry control in an isolated environment.

This has no loop closure or historical bundle adjustment. Native depth is
registered into RGB using static calibration, with no reference-pose input.
Core SceneRecall dependencies do not need opencv-contrib-python-headless.
"""

import argparse
import hashlib
import json
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.datasets.replay import open_source, validate_source
from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.io import save_run
from scene_recall.odometry.pipeline import PairEstimate, TrajectoryFrame
from scene_recall.run_metadata import code_hash, peak_rss, sha256
from scene_recall.slam.evaluation import score_components
from scene_recall.slam.reconstruction import RevisionReconstruction


class OpenCVReference:
    def __init__(self, calibration):
        if not hasattr(cv2, "rgbd"):
            raise RuntimeError("use an isolated OpenCV 4 contrib environment with rgbd")
        self.calibration = calibration
        self.odometry = cv2.rgbd.RgbdOdometry_create(
            calibration.K_R, minDepth=0.3, maxDepth=4.0
        )
        h, w = calibration.depth_shape
        pixels = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).reshape(
            -1, 2
        )
        self.rays = (
            np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(calibration.K_D).T
        )
        self.previous = None

    def prepare(self, observation):
        depth = observation.depth.ravel()
        valid = np.isfinite(depth) & (depth >= 0.3) & (depth <= 4)
        points = transform_points(
            self.rays[valid] * depth[valid, None], self.calibration.T_RD
        )
        positive = points[:, 2] > 0
        points = points[positive]
        pixels = np.rint(project_points(points, self.calibration.K_R)).astype(int)
        h, w = self.calibration.rgb_shape
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < h)
        )
        registered = np.full(h * w, np.inf)
        np.minimum.at(
            registered, pixels[valid, 1] * w + pixels[valid, 0], points[valid, 2]
        )
        registered[~np.isfinite(registered)] = np.nan
        registered = registered.reshape(h, w).astype(np.float32)
        gray = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return gray, registered, (np.isfinite(registered) * 255).astype(np.uint8)

    def estimate(self, source, target, calibration):
        first = self.previous or self.prepare(source)
        second = self.prepare(target)
        self.previous = second
        started = perf_counter()
        cv2.setRNGSeed(0)
        ok, transform = self.odometry.compute(*first, *second)
        if ok:
            transform = np.linalg.inv(calibration.T_RD) @ transform @ calibration.T_RD
        return PairEstimate(
            transform if ok else None,
            None if ok else "OpenCV dense odometry rejected",
            diagnostics={"seconds": perf_counter() - started},
        )


def synthetic_control():
    rng = np.random.default_rng(4)
    image = cv2.GaussianBlur(
        rng.integers(0, 256, (240, 320), dtype=np.uint8), (3, 3), 0
    )
    shifted = cv2.warpAffine(image, np.array([[1.0, 0, -4], [0, 1, 0]]), (320, 240))
    depth = np.full(image.shape, 2, np.float32)
    mask = np.full(image.shape, 255, np.uint8)
    K = np.array([[200.0, 0, 160], [0, 200, 120], [0, 0, 1]])
    odometry = cv2.rgbd.RgbdOdometry_create(K, minDepth=0.3, maxDepth=4.0)
    ok, transform = odometry.compute(image, depth, mask, shifted, depth, mask)
    expected = np.eye(4)
    expected[0, 3] = -0.04
    if not ok or np.linalg.norm(transform[:3, 3] - expected[:3, 3]) > 0.003:
        raise ValueError("OpenCV source-to-target convention control failed")
    return {
        "seed": 4,
        "depth_m": 2,
        "fx_px": 200,
        "image_shift_px": -4,
        "expected_T_target_source": expected.tolist(),
        "estimated_T_target_source": transform.tolist(),
    }


def run(reference, output):
    protocol_file = output.with_suffix(".protocol.json")
    if any(p.exists() or p.is_symlink() for p in (output, protocol_file)):
        raise ValueError("choose a fresh external reference output")
    reference_report = json.loads((reference / "report.json").read_text())
    protocol = {
        "scope": "causal external dense RGB-D odometry, no historical or loop optimization, static native-to-RGB calibration, fixed CLAHE and depth range .3 to4m",
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "driver_sha256": sha256(Path(__file__)),
        "package_sha256": code_hash(),
        "control_report_sha256": sha256(reference / "report.json"),
        "synthetic_control": synthetic_control(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol_file.write_text(json.dumps(protocol, indent=2) + "\n")
    reader, _ = open_source(reference_report)
    digest = hashlib.sha256()
    metadata = []

    def recorded():
        for observation in reader:
            provenance = dict(observation.provenance or {})
            digest.update(
                json.dumps(provenance, sort_keys=True, allow_nan=False).encode()
            )
            for array in (observation.rgb, observation.depth, observation.T_WC_D):
                digest.update(b"none" if array is None else array.tobytes())
            metadata.append({"provenance": provenance})
            yield observation

    cv2.setNumThreads(1)
    started = perf_counter()
    latencies = []
    with reader:
        validate_source(reader, reference_report)
        backend = OpenCVReference(reader.calibration)
        reconstruction = RevisionReconstruction(
            reader.calibration, MappingConfig(pixel_stride=8)
        )
        poses, components = {}, {}
        rows = []
        # Reconstruction follows exactly the same native sample/fusion policy.
        previous = None
        pose = np.eye(4)
        segment = 0
        for observation in recorded():
            step_start = perf_counter()
            pair = (
                None
                if previous is None
                else backend.estimate(previous, observation, reader.calibration)
            )
            if pair is None:
                status = "initialized"
            elif pair.T_C1C0 is None:
                segment += 1
                pose = np.eye(4)
                status = "lost"
            else:
                pose = pose @ np.linalg.inv(pair.T_C1C0)
                status = "tracked"
            frame = observation.frame_index
            rows.append(
                TrajectoryFrame(
                    frame,
                    observation.timestamp,
                    segment,
                    status,
                    pose.copy(),
                    observation.T_WC_D,
                    pair,
                )
            )
            poses[frame], components[frame] = pose.copy(), segment
            reconstruction.admit(observation, segment)
            reconstruction.synchronize(poses, components, len(rows), [frame])
            previous = observation
            latencies.append(perf_counter() - step_start)
            if len(rows) % 100 == 0:
                print(
                    f"External reference {len(rows)} frames, {segment} losses",
                    flush=True,
                )
    elapsed = perf_counter() - started
    if digest.hexdigest() != reference_report["source"]["canonical_input_sha256"]:
        raise ValueError("external control received different canonical inputs")
    component_scores = score_components(
        np.array(list(poses.values())),
        np.array(list(components.values())),
        [r.reference_pose for r in rows],
    )
    report = {
        "format_version": 2,
        "backend": "opencv_rgbd_odometry",
        "initial_pose_stage": None,
        "config": protocol,
        "source": reference_report["source"],
        "calibration": reference_report["calibration"],
        "environment": protocol,
        "experiment": {
            "causal_online": True,
            "intervention": None,
            "offline_global_revision": False,
        },
        "evaluation": evaluate_trajectory(rows),
        "causal_evaluation": component_scores,
        "revised_evaluation": component_scores,
        "frames": [
            {
                **entry,
                "frame_index": row.frame_index,
                "source_frame_index": reference_report["source"]["start"]
                + row.frame_index,
                "segment_id": row.segment_id,
                "status": row.status,
            }
            for entry, row in zip(metadata, rows, strict=True)
        ],
        "timing": {
            "raw_pipeline_seconds": elapsed,
            "peak_rss_mib": peak_rss(),
            "process_latency": {
                "p50_s": float(np.median(latencies)),
                "p95_s": float(np.percentile(latencies, 95)),
                "max_s": max(latencies),
            },
        },
    }
    destination = output
    with tempfile.TemporaryDirectory(
        prefix=".external-", dir=output.parent
    ) as temporary:
        output = Path(temporary) / "result"
        save_run(output, rows, report, reference_report["source"]["start"])
        archive = {
            "T_component_depth": np.array(list(poses.values())),
            "component_id": np.array(list(components.values())),
            "local_segment_id": np.array(list(components.values())),
            "source_frame_index": np.arange(len(rows))
            + reference_report["source"]["start"],
        }
        np.savez_compressed(
            output / "atlas.npz", **archive, version=np.array(len(rows))
        )
        np.savez_compressed(
            output / "atlas_emitted.npz", **archive, version=np.arange(1, len(rows) + 1)
        )
        arrays = {}
        for i, chunk in enumerate(reconstruction.snapshot()):
            for key, value in asdict(chunk.cloud).items():
                arrays[f"{i}_{key}"] = value
            arrays.update(
                {
                    f"{i}_frames": np.array(chunk.frames),
                    f"{i}_component": np.array(chunk.component),
                    f"{i}_version": np.array(chunk.version),
                }
            )
        np.savez_compressed(output / "reconstruction.npz", **arrays)
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        if destination.exists() or destination.is_symlink():
            raise ValueError("external output appeared during export")
        output.rename(destination)
    protocol_file.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.output)


if __name__ == "__main__":
    main()
