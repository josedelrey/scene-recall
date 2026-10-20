"""First frozen chunk TSDF control using the surface-fusion comparison cohort."""

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.geometry.transforms import is_rigid
from scene_recall.mapping.heldout import score_predictions
from scene_recall.mapping.mesh import render_mesh_depth
from scene_recall.mapping.tsdf import TSDFConfig, TSDFView, fuse_tsdf
from scene_recall.mapping.voxel import MappingConfig, sample_observation
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256


def evaluate_tsdf(paths, group, output, fusion):
    """Never choose the chunk by its score, support or reference availability."""
    identity = {
        n: {
            "report_sha256": sha256(paths[n] / "report.json"),
            "trajectory_sha256": sha256(paths[n] / "trajectory.npz"),
        }
        for n in group
    }
    if output.exists():
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        if report["inputs"] != identity or report["driver_sha256"] != sha256(
            Path(__file__)
        ):
            raise ValueError("saved TSDF inputs or implementation differ")
        return report
    config = MappingConfig(**fusion["mapping_config"])
    tsdf = TSDFConfig(
        voxel_size_m=config.voxel_size_m, truncation_m=3 * config.voxel_size_m
    )
    loaded = {n: load_trajectory(paths[n]) for n in group}
    first, rows = loaded[group[0]]
    train = fusion["split"]["chunks"][0]
    targets = fusion["split"]["target_indices_by_chunk"][0]
    report = {
        "inputs": identity,
        "driver_sha256": sha256(Path(__file__)),
        "training_indices": train,
        "target_indices": targets,
        "mapping_config": asdict(config),
        "tsdf_config": asdict(tsdf),
        "chunk_id": 0,
    }
    if any(
        rows[i].reference_pose is None or not is_rigid(rows[i].reference_pose)
        for i in train
    ):
        report.update(status="withheld", reason="missing training references")
    else:
        report["status"] = "evaluated"
        reader, _ = open_source(first)
        digest, metadata, samples = hashlib.sha256(), {}, {}
        selected = set(train + targets)
        with reader:
            validate_source(reader, first)
            for observation in verified_observations(
                reader, first, rows, digest, metadata
            ):
                if observation.frame_index in selected:
                    samples[observation.frame_index] = sample_observation(
                        observation, reader.calibration, config
                    )
            K = reader.calibration.K_D.copy()
        if digest.hexdigest() != first["source"]["canonical_input_sha256"]:
            raise ValueError("TSDF canonical input identity changed")
        K[:2] /= config.pixel_stride
        mesh_arrays, meshes, transforms = {}, {}, {}
        for name in [*group, "gt"]:
            current = rows if name == "gt" else loaded[name][1]
            poses = {
                i: (current[i].reference_pose if name == "gt" else current[i].T_SC_D)
                for i in train
            }
            transforms[name] = np.linalg.inv(poses[train[0]])
            views = [
                TSDFView(
                    i,
                    transforms[name] @ poses[i],
                    samples[i].depth[:: config.pixel_stride, :: config.pixel_stride],
                )
                for i in train
            ]
            try:
                volume = fuse_tsdf(views, K, tsdf)
                mesh = volume.mesh()
            except ValueError as error:
                if "capacity" not in str(error) and "budget" not in str(error):
                    raise
                report.update(
                    status="withheld", reason="TSDF resource limit: " + str(error)
                )
                break
            meshes[name] = mesh
            mesh_arrays.update(
                {f"{name}_vertices": mesh.vertices, f"{name}_triangles": mesh.triangles}
            )
        if report["status"] == "evaluated":
            arrays = {}
            for index in targets:
                if rows[index].reference_pose is None or not is_rigid(
                    rows[index].reference_pose
                ):
                    continue
                observed = samples[index].depth[
                    :: config.pixel_stride, :: config.pixel_stride
                ]
                arrays.setdefault("observed", []).append(observed)
                for name in group:
                    current = loaded[name][1]
                    T_MW = np.linalg.inv(rows[train[0]].reference_pose)
                    for mode, pose in (
                        ("reference_pose", T_MW @ rows[index].reference_pose),
                        ("estimated_pose", transforms[name] @ current[index].T_SC_D),
                    ):
                        arrays.setdefault(f"{name}_{mode}", []).append(
                            render_mesh_depth(meshes[name], pose, K, observed.shape)
                        )
                arrays.setdefault("gt", []).append(
                    render_mesh_depth(
                        meshes["gt"],
                        transforms["gt"] @ rows[index].reference_pose,
                        K,
                        observed.shape,
                    )
                )
            if arrays:
                combined = {k: np.stack(v) for k, v in arrays.items()}
                observed = combined["observed"]
                predictions = {k: v for k, v in combined.items() if k != "observed"}
                report["scores"], _ = score_predictions(observed, predictions)
                report["evaluated_views"] = len(observed)
                valid = np.isfinite(observed) & (observed > 0)
                report["capped_missing_penalty_20cm_m"] = {
                    k: float(
                        np.minimum(
                            np.where(np.isfinite(v), np.abs(v - observed), np.inf)[
                                valid
                            ],
                            0.2,
                        ).mean()
                    )
                    if valid.any()
                    else None
                    for k, v in predictions.items()
                }
                mesh_arrays.update(combined)
            else:
                report.update(status="withheld", reason="no target references")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".refinement-tsdf-", dir=output.parent
        ) as temporary:
            stage = Path(temporary) / "result"
            stage.mkdir()
            np.savez_compressed(stage / "geometry_and_predictions.npz", **mesh_arrays)
            (stage / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            if output.exists():
                raise ValueError("TSDF output appeared during evaluation")
            stage.rename(output)
        return report
    output.mkdir(parents=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
