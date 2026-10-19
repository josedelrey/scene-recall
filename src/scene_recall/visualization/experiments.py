"""Explicit adapters for saved fusion experiments, without fusion or scoring."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.geometry.transforms import is_rigid
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256
from scene_recall.visualization.recording import (
    _calibration,
    _cloud,
    _read_json,
    _validate_submaps,
    recording_stream,
)
from scene_recall.visualization.rerun import SceneLogger, blueprint, require_rerun

HELDOUT = "held_out_fusion_views"
TSDF = "projective_tsdf_fusion"
SUPPORT = "single_frame_training_free_space_consensus"
MOTION = "motion_budget_keyframes"


@dataclass(frozen=True)
class Reconstruction:
    """One scientific representation in an explicitly owned local gauge."""

    label: str
    backend: str
    mode: str
    report: dict
    path: Path
    kind: str = "cloud"
    counts: tuple[int, ...] = ()

    @property
    def root(self):
        return (
            f"experiments/{self.label}/{self.backend}/segments/"
            f"{self.report['segment_id']}/submaps/{self.report['submap_id']}"
        )


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def _geometry(spec):
    if spec.kind == "cloud":
        return _cloud(
            spec.path,
            {k: v for k, v in spec.report.items() if k != "backend"},
            spec.mode,
            require_metadata=False,
            pose_source=False,
        )
    arrays = _arrays(spec.path)
    names = (
        ("vertices", "colors", "triangles")
        if spec.kind == "mesh"
        else ("points", "colors", "keys")
    )
    count = len(arrays[names[0]])
    if spec.counts and count != spec.counts[0]:
        raise ValueError(f"geometry count differs from report: {spec.path.name}")
    for name in names:
        a = arrays[name]
        if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
            raise ValueError(f"invalid geometry: {spec.path.name}/{name}")
        if name != "triangles" and len(a) != count:
            raise ValueError(f"geometry lengths differ: {spec.path.name}")
    if arrays["colors"].dtype != np.uint8:
        raise ValueError("expected uint8 exported colors")
    if spec.kind == "mesh":
        triangles = arrays["triangles"]
        if spec.counts and len(triangles) != spec.counts[1]:
            raise ValueError(f"triangle count differs from report: {spec.path.name}")
        if not np.issubdtype(triangles.dtype, np.integer) or (
            triangles.size and (triangles.min() < 0 or triangles.max() >= count)
        ):
            raise ValueError(f"invalid triangle indices: {spec.path.name}")
    else:
        if not np.issubdtype(arrays["keys"].dtype, np.integer):
            raise ValueError(f"invalid voxel keys: {spec.path.name}")
        attributes = ("view_support", "color_support", "signed_residual_std_m")
        for name in attributes:
            a = arrays[name]
            if a.shape != (count,) or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError(f"invalid geometry attribute: {spec.path.name}/{name}")
    return arrays


def _load_cohort(report, baseline, runs):
    loaded = {}
    for run in runs:
        metadata, rows = load_trajectory(run)
        backend = metadata["backend"]
        expected = report["odometry"].get(backend)
        if backend in loaded or expected != {
            "report_sha256": sha256(run / "report.json"),
            "trajectory_sha256": sha256(run / "trajectory.npz"),
        }:
            raise ValueError("odometry input differs from saved experiment")
        if (
            metadata["calibration"] != baseline["calibration"]
            or any(
                baseline["source"].get(k) != v for k, v in metadata["source"].items()
            )
            or metadata["source"]["canonical_input_sha256"]
            != baseline["verified_canonical_input_sha256"]
        ):
            raise ValueError(
                "experiment source, calibration or canonical input differs"
            )
        loaded[backend] = (metadata, rows)
    if loaded.keys() != report["odometry"].keys():
        raise ValueError(
            "provide exactly the experiment's odometry cohort via --odometry-runs"
        )
    # Report order defines the GT baseline owner in the experiment writers.
    loaded = {name: loaded[name] for name in report["odometry"]}
    first, frames = next(iter(loaded.values()))
    for metadata, rows in loaded.values():
        if metadata["source"] != first["source"] or len(rows) != len(frames):
            raise ValueError("cohort source selections differ")
        for a, b in zip(frames, rows, strict=True):
            if (
                (a.frame_index, a.segment_id, a.timestamp)
                != (b.frame_index, b.segment_id, b.timestamp)
                or (a.reference_pose is None) != (b.reference_pose is None)
                or a.reference_pose is not None
                and not np.array_equal(
                    a.reference_pose, b.reference_pose, equal_nan=True
                )
            ):
                raise ValueError("cohort frame identities or references differ")
    return loaded


def _specifications(run, report, baseline_run, baseline):
    """Map every saved prediction key to its representation and pose convention."""
    experiment = report["experiment"]
    specs, predictions = [], {}
    first = next(iter(report["odometry"]))
    base_maps = {(s["backend"], s["submap_id"]): s for s in baseline["submaps"]}
    if len(base_maps) != len(baseline["submaps"]):
        raise ValueError("duplicate baseline submap identity")

    def add(label, backend, mode, submap, path, kind="cloud", prefix="", counts=()):
        spec = Reconstruction(label, backend, mode, submap, path, kind, counts)
        specs.append(spec)
        chunk = submap["submap_id"]
        if mode == "reference":
            key = prefix + (f"{backend}_gt" if experiment == MOTION else "gt")
            predictions[chunk, key] = (spec, "reference_pose")
        else:
            for pose_mode in ("reference_pose", "estimated_pose"):
                predictions[chunk, prefix + f"{backend}_{pose_mode}"] = (
                    spec,
                    pose_mode,
                )
        return spec

    if experiment in (HELDOUT, TSDF, SUPPORT):
        for (backend, chunk), submap in base_maps.items():
            for mode in ("estimated", "reference"):
                # The controlled GT map is identical across backends.
                if mode == "reference" and backend != first:
                    continue
                add(
                    "baseline",
                    backend,
                    mode,
                    submap,
                    baseline_run / f"{backend}_{chunk:03d}_{mode}.npz",
                    prefix="baseline_" if experiment != HELDOUT else "",
                )
    if experiment == TSDF:
        for chunk_report in report["chunks"]:
            chunk = chunk_report["submap_id"]
            for name, stats in chunk_report["tsdf"].items():
                backend, mode = (
                    (first, "reference") if name == "gt" else (name, "estimated")
                )
                submap = chunk_report["baseline"][backend]
                if submap != {
                    k: v for k, v in base_maps[backend, chunk].items() if k != "backend"
                }:
                    raise ValueError("TSDF baseline gauge differs from linked baseline")
                for label, kind, suffix in (
                    ("tsdf", "surface", "surface"),
                    ("mesh", "mesh", "mesh"),
                ):
                    path = run / f"{name}_{chunk:03d}_{suffix}.npz"
                    # Original point-control runs predate connected mesh export.
                    if kind == "mesh" and "mesh_vertices" not in stats:
                        continue
                    counts = (
                        (stats["mesh_vertices"], stats["mesh_triangles"])
                        if kind == "mesh"
                        else (stats["surface_points"],)
                    )
                    add(
                        label,
                        backend,
                        mode,
                        submap,
                        path,
                        kind,
                        prefix=label + "_",
                        counts=counts,
                    )
    elif experiment == SUPPORT:
        for cloud in report["clouds"]:
            name, chunk = cloud["map"], cloud["submap_id"]
            backend, mode = (
                (first, "reference") if name == "gt" else (name, "estimated")
            )
            submap = dict(base_maps[backend, chunk])
            submap[mode] = {
                **submap[mode],
                "voxel_count": cloud["baseline_voxels"] - cloud["removed_voxels"],
            }
            add(
                "consensus",
                backend,
                mode,
                submap,
                run / f"{name}_{chunk:03d}_consensus.npz",
            )
    elif experiment == MOTION:
        for chunk_report in report["chunks"]:
            chunk = chunk_report["submap_id"]
            for label, submap in chunk_report["maps"].items():
                backend, strategy = label.rsplit("_", 1)
                for mode in ("estimated", "reference"):
                    # Use the full saved label for prediction identity, then retain
                    # the actual odometry backend on the geometry descriptor.
                    spec = Reconstruction(
                        strategy,
                        backend,
                        mode,
                        submap,
                        run / f"{label}_{chunk:03d}_{mode}.npz",
                    )
                    specs.append(spec)
                    keys = (
                        [(f"{label}_gt", "reference_pose")]
                        if mode == "reference"
                        else [
                            (f"{label}_{p}", p)
                            for p in ("reference_pose", "estimated_pose")
                        ]
                    )
                    for key, pose_mode in keys:
                        predictions[chunk, key] = (spec, pose_mode)
    return specs, predictions


def _view_archive(run, baseline_run, report, chunk, radius, rows, shape):
    filename = f"views_{chunk:03d}_radius_{float(radius):g}.npz"
    arrays = _arrays(run / filename)
    expected = [
        r["frame_index"] for r in rows if r.get("status", "evaluated") == "evaluated"
    ]
    if not np.array_equal(arrays["evaluated_frame_indices"], expected):
        raise ValueError("evaluated frame indices differ from experiment report")
    if report["experiment"] == SUPPORT and expected:
        original = _arrays(baseline_run / filename)
        if not np.array_equal(original["evaluated_frame_indices"], expected):
            raise ValueError("consensus target frames differ from baseline")
        arrays.update(
            observed=original["observed"],
            baseline_common=original["common_all_modes"],
            **{
                "baseline_" + name: original[name]
                for name in next(
                    r for r in rows if r.get("status", "evaluated") == "evaluated"
                )["scores"]
            },
        )
    expected_shape = (len(expected), *shape)
    for name, a in arrays.items():
        if name.endswith("frame_indices"):
            if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
                raise ValueError("invalid archived frame indices")
            continue
        if a.shape != expected_shape:
            raise ValueError(f"invalid evaluation array shape: {filename}/{name}")
        if a.dtype == np.bool_:
            continue
        if (
            not np.issubdtype(a.dtype, np.floating)
            or np.isinf(a).any()
            or (a <= 0).any()
        ):
            raise ValueError(f"invalid saved metric depth: {filename}/{name}")
    if expected:
        valid = np.isfinite(arrays["observed"])
        expected_keys = set()
        for row in rows:
            if row.get("status", "evaluated") == "evaluated":
                keys = set(row["scores"])
                if report["experiment"] == SUPPORT:
                    keys |= {"baseline_" + k for k in keys}
                if expected_keys and keys != expected_keys:
                    raise ValueError("prediction cohort changes within a chunk")
                expected_keys = keys
        actual_keys = {
            k
            for k, a in arrays.items()
            if a.dtype != np.bool_
            and k not in ("observed", "frame_indices", "evaluated_frame_indices")
        }
        if actual_keys != expected_keys:
            raise ValueError("saved prediction keys differ from report")
        for name, a in arrays.items():
            if a.dtype == np.bool_ and (a & ~valid).any():
                raise ValueError("saved common mask includes invalid targets")
        common_name = (
            "common_all_modes"
            if report["experiment"] == HELDOUT
            else "common_all_versions"
        )
        if common_name in arrays:
            common = valid.copy()
            for key in expected_keys:
                # Connected-mesh comparisons deliberately exclude the incomplete
                # TSDF point-disc control from their primary common population.
                if (
                    report["experiment"] == TSDF
                    and report.get("render", {}).get("primary")
                    == "baseline_discs_vs_connected_tsdf_mesh"
                    and key.startswith("tsdf_")
                ):
                    continue
                common &= np.isfinite(arrays[key])
            if not np.array_equal(arrays[common_name], common):
                raise ValueError("saved common mask differs from prediction visibility")
    return arrays


def _evaluation_tabs(spec, evaluations):
    import rerun.blueprint as rrb

    geometry = spec.root + "/" + spec.mode + "/map"
    tabs = []
    for root, name, masks in evaluations:
        scene = rrb.Spatial3DView(
            name="Reconstruction and observed residuals",
            origin=spec.root,
            contents=[
                f"+ {geometry}/mesh"
                if spec.kind == "mesh"
                else f"+ {geometry}/rgb"
                if spec.kind == "cloud"
                else f"+ {geometry}",
                f"+ {root}/camera/residual_points",
                f"+ {root}/camera/lattice",
            ],
        )
        images = rrb.Tabs(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    name="Observed z-depth (m)",
                    origin=root + "/camera/lattice",
                    contents=[root + "/camera/lattice/observed"],
                ),
                rrb.Spatial2DView(
                    name="Predicted z-depth (m)",
                    origin=root + "/camera/lattice",
                    contents=[root + "/camera/lattice/predicted"],
                ),
            ),
            rrb.Spatial2DView(
                name="Signed error: red front / blue behind", origin=root + "/residual"
            ),
            rrb.TensorView(
                name="Signed error values (m)", origin=root + "/signed_error_m"
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(name="Valid target", origin=root + "/valid_target"),
                rrb.Spatial2DView(
                    name="Covered valid target", origin=root + "/coverage"
                ),
                *[
                    rrb.Spatial2DView(name=mask, origin=root + "/masks/" + mask)
                    for mask in masks
                ],
                name="Evaluation populations",
            ),
            rrb.TextDocumentView(name="Saved view scores", origin=root + "/report"),
        )
        tabs.append(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Tabs(
                        scene,
                        rrb.Spatial3DView(
                            name="Observed residual surface",
                            origin=spec.root,
                            contents=[
                                f"+ {root}/camera/residual_points",
                                f"+ {root}/camera/lattice",
                            ],
                        ),
                    ),
                    images,
                    column_shares=[3, 2],
                ),
                rrb.Tabs(
                    *[
                        rrb.TimeSeriesView(
                            name=title,
                            origin=root + "/scores",
                            contents=[
                                f"+ {root}/scores/{population}/{metric}"
                                for population in (
                                    "own",
                                    "all_valid",
                                    "common_all_modes",
                                    "common_all_versions",
                                    "paired_baseline_common",
                                )
                                for metric in metrics
                            ],
                        )
                        for title, metrics in (
                            (
                                "Saved depth error (m)",
                                (
                                    "absolute_depth_error_m/mean",
                                    "absolute_depth_error_m/rmse",
                                ),
                            ),
                            (
                                "Saved coverage and explained fraction",
                                (
                                    "coverage",
                                    "thresholds/0.05/explained_fraction_of_target",
                                ),
                            ),
                        )
                    ]
                ),
                row_shares=[4, 1],
                name=name,
            )
        )
    return tabs


def export_experiment(
    output,
    *,
    experiment_run,
    odometry_runs,
    baseline_run=None,
    source_path=None,
    images=None,
    submap_ids=None,
    display=None,
):
    """Export known experiment schemas into the shared offline recording sink."""
    rr = require_rerun()
    report = _read_json(experiment_run / "report.json")
    experiment = report.get("experiment")
    if experiment not in (HELDOUT, TSDF, SUPPORT, MOTION):
        raise ValueError("unsupported visualization experiment schema")
    if report.get("format_version", 1) != 1:
        raise ValueError("unsupported experiment format version")
    if experiment == HELDOUT:
        if baseline_run is not None:
            raise ValueError("a standard held-out run does not need --baseline-run")
        baseline_run, baseline = experiment_run, report
    else:
        if baseline_run is None:
            raise ValueError(
                "this experiment requires its saved held-out --baseline-run"
            )
        if sha256(baseline_run / "report.json") != report["saved_report_sha256"]:
            raise ValueError("linked baseline report hash differs")
        baseline = _read_json(baseline_run / "report.json")
        if baseline.get("experiment") != HELDOUT:
            raise ValueError("expected a held-out baseline")
        for name, digest in report["saved_arrays_sha256"].items():
            if Path(name).name != name or sha256(baseline_run / name) != digest:
                raise ValueError("linked baseline archive hash differs")
        fingerprint = report.get(
            "verified_canonical_input_sha256", report.get("verified_input_sha256")
        )
        if (
            fingerprint != baseline["verified_canonical_input_sha256"]
            or report["odometry"] != baseline["odometry"]
            or report["mapping_config"] != baseline["mapping_config"]
        ):
            raise ValueError("experiment provenance differs from baseline")
    loaded = _load_cohort(report, baseline, odometry_runs)
    first_metadata, first_frames = next(iter(loaded.values()))
    calibration = _calibration(baseline["calibration"])
    start = first_metadata["source"]["start"]
    image_mode = images or "none"
    if image_mode not in ("none", "evaluated", "all", "keyframes"):
        raise ValueError("invalid experiment image mode")
    if source_path is not None and image_mode == "none":
        raise ValueError("--source-path requires image logging")
    specs, predictions = _specifications(experiment_run, report, baseline_run, baseline)
    for spec in specs:
        _validate_submaps({"submaps": [spec.report]}, loaded[spec.backend][1])
    ids = {s.report["submap_id"] for s in specs}
    if submap_ids is not None:
        if not submap_ids or set(submap_ids) - ids:
            raise ValueError("unknown or empty submap selection")
        specs = [s for s in specs if s.report["submap_id"] in submap_ids]
    selected_ids = {s.report["submap_id"] for s in specs}
    stride = report["mapping_config"]["pixel_stride"]
    shape = tuple((n + stride - 1) // stride for n in calibration.depth_shape)
    views = (
        report["views"]
        if experiment == HELDOUT
        else {"0.01": report["views"] + report.get("unavailable_views", [])}
    )
    tabs, evaluation_views, selected_images = [], {}, set()
    with recording_stream(output) as recording:
        logger = SceneLogger(recording, calibration, display)
        for name, value in (("experiment", report), ("mapping", baseline)):
            recording.log(
                "recording/" + name + "_report",
                rr.TextDocument(
                    json.dumps(value, indent=2), media_type="application/json"
                ),
                static=True,
            )
        recording.log(
            "recording/odometry_report",
            rr.TextDocument(
                json.dumps({name: m for name, (m, _) in loaded.items()}, indent=2),
                media_type="application/json",
            ),
            static=True,
        )
        logged_roots = set()
        for spec in specs:
            root, submap = spec.root, spec.report
            segment_root = root.rsplit("/submaps/", 1)[0]
            recording.log(
                segment_root,
                rr.Transform3D(scale=0.0),
                rr.ViewCoordinates.RDF,
                static=True,
            )
            logger.transform(root, np.asarray(submap["T_SM"]), static=True)
            arrays = _geometry(spec)
            geometry_root = root + "/" + spec.mode + "/map"
            getattr(logger, spec.kind)(
                geometry_root + ("/mesh" if spec.kind == "mesh" else ""), arrays
            )
            recording.log(
                root + "/" + spec.mode + "/provenance",
                rr.TextDocument(
                    json.dumps(submap, indent=2), media_type="application/json"
                ),
                static=True,
            )
            if root in logged_roots:
                continue
            logged_roots.add(root)
            rows = loaded[spec.backend][1][
                submap["keyframe_indices"][0] : submap["keyframe_indices"][-1] + 1
            ]
            alignment = submap["T_MW_reference_diagnostic"]
            references = [
                np.asarray(alignment) @ r.reference_pose
                if alignment is not None
                and r.reference_pose is not None
                and is_rigid(r.reference_pose)
                else None
                for r in rows
            ]
            logger.trajectory(
                root,
                rows,
                [np.linalg.inv(submap["T_SM"]) @ r.T_SC_D for r in rows],
                references,
                start,
            )
            if image_mode != "none":
                logger.transform(
                    root + "/estimated/camera/rgb",
                    np.linalg.inv(calibration.T_RD),
                    static=True,
                )
            for frame in loaded[spec.backend][1]:
                if not rows[0].frame_index <= frame.frame_index <= rows[-1].frame_index:
                    logger.set_frame(frame, start)
                    for mode in ("estimated", "reference"):
                        recording.log(
                            root + f"/{mode}/camera", rr.Clear(recursive=True)
                        )
        for radius, all_rows in views.items():
            for chunk in sorted(selected_ids):
                rows = sorted(
                    (r for r in all_rows if r["submap_id"] == chunk),
                    key=lambda r: r["frame_index"],
                )
                indices = [r["frame_index"] for r in rows]
                if len(set(indices)) != len(indices) or any(
                    type(i) is not int or not 0 <= i < len(first_frames)
                    for i in indices
                ):
                    raise ValueError("invalid or duplicate evaluation frame identity")
                arrays = _view_archive(
                    experiment_run, baseline_run, report, chunk, radius, rows, shape
                )
                expected = arrays["evaluated_frame_indices"].tolist()
                masks = {k: a for k, a in arrays.items() if a.dtype == np.bool_}
                keys = [
                    k
                    for k in arrays
                    if k not in masks
                    and k
                    not in ("observed", "frame_indices", "evaluated_frame_indices")
                ]
                for key in keys:
                    if (chunk, key) not in predictions:
                        raise ValueError(f"unrecognized prediction convention: {key}")
                    spec, pose_mode = predictions[chunk, key]
                    if spec not in specs:
                        continue
                    root = (
                        spec.root
                        + f"/{spec.mode}/evaluation/radius_{radius}/{pose_mode}"
                    )
                    evaluation_views.setdefault(spec.root + "/" + spec.mode, []).append(
                        (root, f"{pose_mode} / radius {radius} m", list(masks))
                    )
                    # Clear every omitted source frame, including unavailable targets.
                    for frame in loaded[spec.backend][1]:
                        if frame.frame_index not in expected:
                            logger.set_frame(frame, start)
                            recording.log(root, rr.Clear(recursive=True))
                    evaluated_rows = {r["frame_index"]: r for r in rows}
                    for j, index in enumerate(expected):
                        row = evaluated_rows[index]
                        frame = loaded[spec.backend][1][index]
                        if (
                            frame.segment_id != spec.report["segment_id"]
                            or row["segment_id"] != frame.segment_id
                        ):
                            raise ValueError(
                                "evaluation target crosses a segment boundary"
                            )
                        training = spec.report["keyframe_indices"]
                        if index in training or not training[0] < index < training[-1]:
                            raise ValueError(
                                "evaluated target is not a held-out interior frame"
                            )
                        if (
                            frame.reference_pose is None
                            or not is_rigid(frame.reference_pose)
                            or spec.report["T_MW_reference_diagnostic"] is None
                        ):
                            raise ValueError(
                                "evaluated target lacks a valid reference alignment"
                            )
                        pose = (
                            np.asarray(spec.report["T_MW_reference_diagnostic"])
                            @ frame.reference_pose
                            if pose_mode == "reference_pose"
                            else np.linalg.inv(spec.report["T_SM"]) @ frame.T_SC_D
                        )
                        logger.set_frame(frame, start)
                        logger.evaluation(
                            root,
                            pose,
                            arrays["observed"][j],
                            arrays[key][j],
                            {k: a[j] for k, a in masks.items()},
                            stride,
                        )
                        if experiment == SUPPORT:
                            base_key = key.removeprefix("baseline_")
                            mode = (
                                "baseline"
                                if key.startswith("baseline_")
                                else "consensus"
                            )
                            scores = {
                                population: values[mode]
                                for population, values in row["scores"][
                                    base_key
                                ].items()
                                if population != "transitions"
                            }
                        else:
                            scores = row["scores"][key]
                        logger.scalars(root + "/scores", scores, missing_nan=True)
                        recording.log(
                            root + "/report",
                            rr.TextDocument(
                                json.dumps({**row, "scores": scores}, indent=2),
                                media_type="application/json",
                            ),
                        )
                        if image_mode in ("all", "evaluated"):
                            selected_images.add(index)
                for row in rows:
                    logger.set_frame(first_frames[row["frame_index"]], start)
                    recording.log(
                        f"diagnostics/evaluation/radius_{radius}/submap_{chunk}",
                        rr.TextLog(
                            json.dumps({k: v for k, v in row.items() if k != "scores"})
                        ),
                    )
        if image_mode != "none":
            reader, _ = open_source(first_metadata, source_path)
            with reader:
                validate_source(reader, first_metadata)
                for observation in verified_observations(
                    reader, first_metadata, first_frames, hashlib.sha256(), {}
                ):
                    index = observation.frame_index
                    image_specs = {s.root: s for s in specs}.values()
                    for spec in image_specs:
                        if (
                            not spec.report["keyframe_indices"][0]
                            <= index
                            <= spec.report["keyframe_indices"][-1]
                        ):
                            continue
                        frame = loaded[spec.backend][1][index]
                        logger.set_frame(frame, start)
                        camera = spec.root + "/estimated/camera"
                        if (
                            image_mode == "all"
                            or (
                                image_mode == "keyframes"
                                and index in spec.report["keyframe_indices"]
                            )
                            or (image_mode == "evaluated" and index in selected_images)
                        ):
                            logger.observation(observation, frame, camera_root=camera)
                        else:
                            for sensor in ("rgb", "depth"):
                                recording.log(
                                    camera + f"/{sensor}/image",
                                    rr.Clear(recursive=True),
                                )
        import rerun.blueprint as rrb

        grouped = {}
        for spec in sorted(specs, key=lambda s: (s.kind != "mesh", s.label)):
            root = spec.root
            geometry = root + "/" + spec.mode + "/map"
            # Multiple pose modes share a reconstruction tab, with no gauge mixing.
            context = rrb.Spatial3DView(
                name="Final reconstruction and trajectories",
                origin=root,
                contents=[
                    f"+ {geometry}/mesh"
                    if spec.kind == "mesh"
                    else f"+ {geometry}/rgb"
                    if spec.kind == "cloud"
                    else f"+ {geometry}",
                    f"+ {root}/estimated/camera/**",
                    f"+ {root}/reference/camera/**",
                    f"+ {root}/estimated/trajectory",
                    f"+ {root}/reference/trajectory",
                ],
            )
            panels = [context]
            if image_mode != "none":
                panels.append(
                    rrb.Vertical(
                        rrb.Spatial2DView(
                            name="RGB at estimated pose",
                            origin=root + "/estimated/camera/rgb",
                        ),
                        rrb.Spatial2DView(
                            name="Native depth", origin=root + "/estimated/camera/depth"
                        ),
                    )
                )
            grouped.setdefault(spec.report["submap_id"], {}).setdefault(
                spec.backend, {}
            ).setdefault(spec.mode, []).append(
                rrb.Tabs(
                    rrb.Horizontal(*panels, name="Reconstruction"),
                    *_evaluation_tabs(
                        spec, evaluation_views.get(root + "/" + spec.mode, [])
                    ),
                    rrb.TextDocumentView(
                        name="Map provenance",
                        origin=root + "/" + spec.mode + "/provenance",
                    ),
                    name=spec.label,
                )
            )
        tabs.extend(
            rrb.Tabs(
                *[
                    rrb.Tabs(
                        *[
                            rrb.Tabs(*variants, name=mode + " map")
                            for mode, variants in modes.items()
                        ],
                        name=backend,
                    )
                    for backend, modes in backends.items()
                ],
                name=f"Submap {chunk}",
            )
            for chunk, backends in grouped.items()
        )
        tabs.append(
            rrb.TextLogView(
                name="Evaluation availability", origin="diagnostics/evaluation"
            )
        )
        recording.log(
            "recording/guide",
            rr.TextDocument(
                "# SceneRecall fusion evaluation\n\nScrub the frame timeline. Each representation and target-pose mode has a separate scene. "
                "Maps are final static reconstructions. Segment gauges are disconnected. Amber trajectories use estimates, teal use fixed training reference alignment. "
                "Reference poses never correct estimates. Reference maps are sensor-consistency diagnostics, not independent surface truth.\n\n"
                "Observed and predicted images contain saved metric z-depth on the native [::stride, ::stride] lattice. "
                "Predictions retain their saved nearest-z visibility, including geometry behind measured surfaces. "
                "Residual = predicted minus observed. Red is in front, blue behind, gray missing prediction, black invalid target. "
                "Colors saturate at +/-0.15 m. Signed tensors and point attributes retain unsaturated errors. "
                "Residual points lie on the observed surface in the evaluated target pose. Missing predictions have no invented error. "
                "White masks show membership in the named saved population. Coverage uses every valid target. "
                "Conditional and common-pixel errors use their original report populations. Scores are copied, never recalculated or clipped.\n\n"
                "Unlogged and unavailable frames clear evaluated payloads. Raw RGB uses the separately calibrated camera at the estimated pose in Reconstruction. "
                "Meshes preserve all exported vertices, connectivity and colors. Point caps affect display only. "
                "TSDF view_support is near-surface diagnostic support, distinct from voxel frame_support.\n\n"
                f"Experiment: {experiment}\n\nRerun SDK: {rr.__version__}\n\n"
                "The recording embeds all shown data, reports and its workspace. It works offline without source files. NPZ, PLY and JSON remain authoritative.",
                media_type="text/markdown",
            ),
            static=True,
        )
        recording.send_blueprint(
            blueprint(
                [], [], show_images=False, extra_tabs=tabs, show_diagnostics=False
            )
        )
    return output
