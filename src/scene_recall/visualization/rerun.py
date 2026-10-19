"""Rerun scene logging without dataset access or algorithm execution.

Entity paths encode independent segment gauges and submap-local reference
alignments. Canonical camera frames use right/down/forward optical axes.
"""

from dataclasses import dataclass
from importlib import import_module

import numpy as np

from scene_recall.data import Calibration, Observation
from scene_recall.geometry.transforms import is_rigid
from scene_recall.odometry.pipeline import TrajectoryFrame

ESTIMATED = [255, 190, 70]
REFERENCE = [70, 210, 190]
COMPARISON_GUIDE = (
    "Compare VO / GT uses one 3D camera for both final reconstructions. "
    "In the Blueprint panel, expand this view and use the eye icons on its "
    "estimated and reference branches to show VO, GT or both without switching views. "
    "VO is amber and GT is teal. These view-local colors preserve recorded RGB and support. "
    "Both maps use the saved submap frame and fixed reference alignment. "
    "GT means reference-pose fusion of sensor depth, not independent surface truth. "
    "Unpaired keyframes remain labeled and have no scientific map-distance comparison."
)


def require_rerun():
    """Load the optional SDK only when a recording is requested."""
    try:
        return import_module("rerun")
    except ModuleNotFoundError as error:
        if error.name != "rerun":
            raise
        raise RuntimeError(
            "Rerun visualization requires the optional dependency. "
            "Run: uv sync --locked --extra visualization"
        ) from error


@dataclass(frozen=True)
class DisplayConfig:
    """Deterministic display limits, never used by scientific computation."""

    max_points: int = 200_000
    min_support: int = 1

    def __post_init__(self):
        for name in ("max_points", "min_support"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def display_indices(support: np.ndarray, config: DisplayConfig) -> np.ndarray:
    indices = np.flatnonzero(support >= config.min_support)
    if len(indices) > config.max_points:
        indices = indices[
            np.linspace(0, len(indices) - 1, config.max_points, dtype=np.intp)
        ]
    return indices


def support_colors(support: np.ndarray) -> np.ndarray:
    """Log-scaled blue/cyan/yellow support coloring with an embedded legend."""
    level = np.log2(np.maximum(support, 1)) / max(
        float(np.log2(max(int(support.max()), 2))) if len(support) else 1, 1
    )
    stops = np.array([[40, 70, 180], [40, 200, 190], [255, 220, 70]])
    return np.column_stack(
        [np.interp(level, [0, 0.5, 1], stops[:, c]) for c in range(3)]
    ).astype(np.uint8)


def _strips(poses):
    """Split paths at unavailable references instead of bridging gaps."""
    strips, current = [], []
    for pose in poses:
        if pose is None:
            if len(current) > 1:
                strips.append(current)
            current = []
        else:
            current.append(pose[:3, 3])
    if len(current) > 1:
        strips.append(current)
    return strips


class SceneLogger:
    """Log canonical observations, accepted poses and final scientific maps.

    The caller owns the RecordingStream and its sink. These methods can also
    serve a future ROS2 adapter without coupling it to experiment directories.
    """

    def __init__(self, recording, calibration: Calibration, display=None):
        self.rr = require_rerun()
        self.recording = recording
        self.calibration = calibration
        self.display = display or DisplayConfig()

    def set_frame(self, row: TrajectoryFrame, source_start: int):
        self.recording.reset_time()
        self.recording.set_time("frame", sequence=source_start + row.frame_index)
        if row.timestamp is not None:
            self.recording.set_time("sensor_time", duration=row.timestamp)

    def segment_space(self, segment_id):
        # Rerun's zero-scale sentinel disconnects this gauge from its parent.
        # Merely omitting a transform would imply an identity relationship.
        self.recording.log(
            f"segments/{segment_id}",
            self.rr.Transform3D(scale=0.0),
            self.rr.ViewCoordinates.RDF,
            static=True,
        )

    def transform(self, path, T, *, static=False):
        self.recording.log(
            path,
            self.rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
            static=static,
        )

    def camera_model(self, root, *, color=ESTIMATED, rgb=False):
        rr, calibration = self.rr, self.calibration
        K, shape = (
            (calibration.K_R, calibration.rgb_shape)
            if rgb
            else (calibration.K_D, calibration.depth_shape)
        )
        self.recording.log(
            root,
            rr.Pinhole(
                image_from_camera=K,
                resolution=[shape[1], shape[0]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.12,
                color=color,
            ),
            static=False,
        )

    def observation(
        self, observation: Observation, row: TrajectoryFrame, *, camera_root=None
    ):
        """Embed lossless RGB and native metric depth in their calibrated frames."""
        root = camera_root or f"segments/{row.segment_id}/estimated/camera"
        self.camera_model(root + "/rgb", rgb=True)
        self.recording.log(root + "/rgb/image", self.rr.Image(observation.rgb))
        # NaN is canonical missing depth. Rerun uses zero for invalid samples.
        self.recording.log(
            root + "/depth/image",
            self.rr.DepthImage(np.nan_to_num(observation.depth, nan=0), meter=1.0),
        )

    def trajectory(self, root, rows, estimated, reference, source_start):
        """Log complete paths and time-varying camera poses in one local gauge."""
        rr, recording = self.rr, self.recording
        recording.log(root, rr.ViewCoordinates.RDF, static=True)
        for mode, poses, color in (
            ("estimated", estimated, ESTIMATED),
            ("reference", reference, REFERENCE),
        ):
            path = f"{root}/{mode}"
            strips = _strips(poses)
            if strips:
                recording.log(
                    path + "/trajectory",
                    rr.LineStrips3D(strips, colors=color, radii=0.003),
                    static=True,
                )
            for row, pose in zip(rows, poses, strict=True):
                self.set_frame(row, source_start)
                if pose is None:
                    recording.log(path + "/camera", rr.Clear(recursive=True))
                else:
                    self.camera_model(path + "/camera/depth", color=color)
                    self.transform(path + "/camera", pose)
                    recording.log(
                        path + "/camera", rr.TransformAxes3D(axis_length=0.08)
                    )

    def segment(self, rows, source_start):
        self.segment_space(rows[0].segment_id)
        root = f"segments/{rows[0].segment_id}"
        anchor = next(
            (
                r
                for r in rows
                if r.reference_pose is not None and is_rigid(r.reference_pose)
            ),
            None,
        )
        alignment = (
            None
            if anchor is None
            else anchor.T_SC_D @ np.linalg.inv(anchor.reference_pose)
        )
        references = [
            alignment @ row.reference_pose
            if alignment is not None
            and row.reference_pose is not None
            and is_rigid(row.reference_pose)
            else None
            for row in rows
        ]
        self.trajectory(root, rows, [r.T_SC_D for r in rows], references, source_start)
        camera = root + "/estimated/camera"
        self.transform(
            camera + "/rgb", np.linalg.inv(self.calibration.T_RD), static=True
        )
        for row, reference in zip(rows, references, strict=True):
            self.set_frame(row, source_start)
            if reference is not None:
                error = np.linalg.inv(reference) @ row.T_SC_D
                self.scalars(
                    "diagnostics/pose_error",
                    {
                        "translation_m": float(np.linalg.norm(error[:3, 3])),
                        "rotation_deg": float(
                            np.degrees(
                                np.arccos(
                                    np.clip((np.trace(error[:3, :3]) - 1) / 2, -1, 1)
                                )
                            )
                        ),
                    },
                )
            else:
                for metric in ("translation_m", "rotation_deg"):
                    self.recording.log(
                        "diagnostics/pose_error/" + metric,
                        self.rr.Scalars(float("nan")),
                    )

    def scalars(self, root, values, *, missing_nan=False):
        """Expose finite report measurements while preserving raw JSON separately."""
        for key, value in values.items():
            path = f"{root}/{key}"
            if isinstance(value, dict):
                self.scalars(path, value, missing_nan=missing_nan)
            elif value is None and missing_nan:
                self.recording.log(path, self.rr.Scalars(float("nan")))
            elif (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(value)
            ):
                if root.startswith("diagnostics/maps/"):
                    self.recording.log(
                        path,
                        self.rr.SeriesPoints(
                            colors=ESTIMATED if "/estimated" in root else REFERENCE,
                            marker_sizes=5,
                        ),
                        static=True,
                    )
                self.recording.log(path, self.rr.Scalars(value))

    def cloud(self, root, arrays):
        rr, recording = self.rr, self.recording
        indices = display_indices(arrays["frame_support"], self.display)
        points = arrays["points"][indices]
        attributes = {
            name: arrays[name][indices]
            for name in ("frame_support", "color_support", "position_std_m")
        }
        attributes.update(
            {
                f"voxel_key_{axis}": arrays["keys"][indices, i]
                for i, axis in enumerate("xyz")
            }
        )
        for mode, colors in (
            ("rgb", arrays["colors"][indices]),
            ("support", support_colors(arrays["frame_support"])[indices]),
        ):
            recording.log(
                f"{root}/{mode}",
                rr.Points3D(points, colors=colors, radii=0.004),
                rr.AnyValues(**attributes),
                static=True,
            )
        recording.log(
            root,
            rr.AnyValues(
                total_voxels=len(arrays["points"]),
                displayed_voxels=len(indices),
                min_support=self.display.min_support,
                support_color_max=int(arrays["frame_support"].max())
                if len(arrays["frame_support"])
                else 0,
            ),
            static=True,
        )

    def mesh(self, root, arrays):
        """Preserve the exported connectivity and vertex colors without decimation."""
        self.recording.log(
            root,
            self.rr.Mesh3D(
                vertex_positions=arrays["vertices"],
                triangle_indices=arrays["triangles"],
                vertex_colors=arrays["colors"],
            ),
            static=True,
        )

    def surface(self, root, arrays):
        """Display TSDF point controls using their actual near-view support."""
        support = arrays["view_support"]
        # Zero near-view support is meaningful and remains visible by default.
        indices = display_indices(
            np.ones(len(support)), DisplayConfig(self.display.max_points)
        )
        self.recording.log(
            root,
            self.rr.Points3D(
                arrays["points"][indices], colors=arrays["colors"][indices], radii=0.004
            ),
            self.rr.AnyValues(
                view_support=support[indices],
                color_support=arrays["color_support"][indices],
                signed_residual_std_m=arrays["signed_residual_std_m"][indices],
            ),
            static=True,
        )

    def evaluation(self, root, pose, observed, predicted, masks, pixel_stride):
        """Log saved z-buffer predictions and signed errors on observed surfaces.

        No visibility or scoring is recomputed. Missing predictions remain gray
        and invalid targets black. Only display colors saturate at 150 mm.
        """
        rr, recording = self.rr, self.recording
        self.transform(root + "/camera", pose)
        K = self.calibration.K_D.copy()
        K[:2] /= pixel_stride
        recording.log(
            root + "/camera/lattice",
            rr.Pinhole(
                image_from_camera=K,
                resolution=[observed.shape[1], observed.shape[0]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.12,
            ),
        )
        for name, depth in (("observed", observed), ("predicted", predicted)):
            recording.log(
                root + "/camera/lattice/" + name,
                rr.DepthImage(
                    np.nan_to_num(depth, nan=0), meter=1.0, depth_range=[0, 4]
                ),
            )
        valid = np.isfinite(observed) & (observed > 0)
        covered = valid & np.isfinite(predicted) & (predicted > 0)
        residual = np.where(covered, predicted - observed, np.nan)
        colors = np.zeros((*observed.shape, 3), dtype=np.uint8)
        colors[valid & ~covered] = [110, 110, 110]
        level = np.clip(np.abs(residual[covered]) / 0.15, 0, 1)
        white = np.full((len(level), 3), 245.0)
        ends = np.where((residual[covered] < 0)[:, None], [220, 45, 45], [45, 90, 220])
        colors[covered] = (white * (1 - level[:, None]) + ends * level[:, None]).astype(
            np.uint8
        )
        recording.log(root + "/residual", rr.Image(colors))
        recording.log(
            root + "/signed_error_m",
            rr.Tensor(residual, dim_names=["v", "u"], value_range=[-0.15, 0.15]),
        )
        recording.log(root + "/valid_target", rr.Image(valid.astype(np.uint8) * 255))
        recording.log(root + "/coverage", rr.Image(covered.astype(np.uint8) * 255))
        for name, mask in masks.items():
            recording.log(
                root + "/masks/" + name, rr.Image(mask.astype(np.uint8) * 255)
            )
        v, u = np.nonzero(valid)
        indices = np.arange(len(v))
        if len(indices) > self.display.max_points:
            indices = np.linspace(0, len(v) - 1, self.display.max_points, dtype=np.intp)
        v, u = v[indices], u[indices]
        rays = np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(K).T
        points = rays * observed[v, u, None]
        recording.log(
            root + "/camera/residual_points",
            rr.Points3D(points, colors=colors[v, u], radii=0.006),
            rr.AnyValues(
                native_u=u * pixel_stride,
                native_v=v * pixel_stride,
                observed_m=observed[v, u],
                predicted_m=predicted[v, u],
                signed_error_m=residual[v, u],
                prediction_valid=covered[v, u],
                **{name: mask[v, u] for name, mask in masks.items()},
            ),
        )

    def submap(self, report, clouds, rows, source_start):
        self.segment_space(report["segment_id"])
        root = f"segments/{report['segment_id']}/submaps/{report['submap_id']}"
        self.transform(root, np.array(report["T_SM"]), static=True)
        self.recording.log(
            root,
            self.rr.ViewCoordinates.RDF,
            self.rr.AnyValues(
                comparison_status=report["comparison_status"],
                keyframe_count=len(report["keyframe_indices"]),
                reference_keyframe_count=len(report["reference_keyframe_indices"]),
            ),
            static=True,
        )
        for name, arrays in clouds.items():
            self.cloud(f"{root}/{name}/map", arrays)
        if rows:
            T_MS = np.linalg.inv(np.array(report["T_SM"]))
            alignment = report["T_MW_reference_diagnostic"]
            references = [
                np.array(alignment) @ row.reference_pose
                if alignment is not None
                and row.reference_pose is not None
                and is_rigid(row.reference_pose)
                else None
                for row in rows
            ]
            self.trajectory(
                root,
                rows,
                [T_MS @ row.T_SC_D for row in rows],
                references,
                source_start,
            )
        self.recording.reset_time()
        self.recording.set_time(
            "frame", sequence=source_start + report["keyframe_indices"][-1]
        )
        for mode in ("estimated", "reference"):
            self.scalars(f"diagnostics/maps/{mode}", report[mode])


def comparison_view(root, geometries, *, paired, name="Compare VO / GT (amber / teal)"):
    """Compare geometry already expressed in the same saved submap frame.

    Only the view's colors change. Geometry, transforms and attributes continue
    to come from the recording. Native visibility controls retain this view's eye.
    """
    import rerun.blueprint as rrb

    rr = require_rerun()
    overrides = {}
    for mode, color in (("estimated", ESTIMATED), ("reference", REFERENCE)):
        path, kind = geometries[mode]
        overrides[path] = (
            # Mesh vertex colors require one entry per vertex. A uniform material
            # with an empty color override avoids copying a full color array.
            rr.Mesh3D.from_fields(vertex_colors=[], albedo_factor=color)
            if kind == "mesh"
            else rr.Points3D.from_fields(colors=color)
        ).visualizer()
    return rrb.Spatial3DView(
        name=name + ("" if paired else " / unpaired keyframes"),
        origin=root,
        contents=list(overrides),
        overrides=overrides,
    )


def blueprint(
    segment_ids,
    submaps,
    pair_offsets=(),
    *,
    show_images=True,
    extra_tabs=(),
    show_diagnostics=True,
):
    """Embed a curated native workspace, with no shared view of unrelated gauges."""
    import rerun.blueprint as rrb

    tabs = list(extra_tabs)
    for segment in segment_ids:
        root = f"segments/{segment}"
        scene = rrb.Spatial3DView(
            name="Estimated segment and aligned reference trajectory",
            origin=root,
            contents=[
                f"+ {root}/estimated/**",
                f"+ {root}/reference/**",
                *[
                    f"+ {root}/submaps/{s['submap_id']}/estimated/map/rgb"
                    for s in submaps
                    if s["segment_id"] == segment
                ],
            ],
        )
        panels = [scene]
        if show_images:
            panels.append(
                rrb.Vertical(
                    rrb.Spatial2DView(
                        name="RGB", origin=root + "/estimated/camera/rgb"
                    ),
                    rrb.Spatial2DView(
                        name="Depth (m)", origin=root + "/estimated/camera/depth"
                    ),
                )
            )
        tabs.append(
            rrb.Horizontal(
                *panels,
                column_shares=[3, 1] if show_images else [1],
                name=f"Segment {segment}",
            )
        )
    for submap in submaps:
        root = f"segments/{submap['segment_id']}/submaps/{submap['submap_id']}"
        modes = [
            comparison_view(
                root,
                {
                    mode: (f"{root}/{mode}/map/rgb", "cloud")
                    for mode in ("estimated", "reference")
                },
                paired=submap["comparison_status"] == "paired",
            )
        ]
        for coloring in ("rgb", "support"):
            modes.append(
                rrb.Horizontal(
                    *[
                        rrb.Spatial3DView(
                            name=label,
                            origin=root,
                            contents=[
                                f"+ {root}/{mode}/map/{coloring}",
                                f"+ {root}/{mode}/camera/**",
                                f"+ {root}/{mode}/trajectory",
                            ],
                        )
                        for mode, label in (
                            ("estimated", "Estimated poses"),
                            (
                                "reference",
                                "Reference poses / submap first-pose alignment"
                                if submap["comparison_status"] == "paired"
                                else "Reference poses / unpaired keyframes",
                            ),
                        )
                    ],
                    name="RGB" if coloring == "rgb" else "Frame support",
                )
            )
        tabs.append(
            rrb.Tabs(
                *modes,
                name=f"Submap {submap['submap_id']} / segment {submap['segment_id']}",
            )
        )
    tabs.append(
        rrb.Tabs(
            rrb.TextDocumentView(name="Guide", origin="recording/guide"),
            rrb.TextDocumentView(
                name="Mapping report", origin="recording/mapping_report"
            ),
            rrb.TextDocumentView(
                name="Odometry report", origin="recording/odometry_report"
            ),
            rrb.TextDocumentView(
                name="Experiment report", origin="recording/experiment_report"
            ),
            name="Guide and provenance",
        )
    )

    def plot(name, root, suffixes):
        return rrb.TimeSeriesView(
            name=name,
            origin=root,
            contents=[f"+ {root}/{suffix}" for suffix in suffixes],
        )

    tracking = "diagnostics/tracking"
    map_root = "diagnostics/maps"
    pair_roots = [
        f"offset_{offset}/{direction}/{mode}"
        for offset in pair_offsets
        for direction in ("forward", "reverse")
        for mode in ("estimated", "reference")
    ]
    diagnostics = rrb.Tabs(
        rrb.Horizontal(
            plot(
                "Match and inlier counts",
                tracking,
                ["correspondences", "inliers", "depth_valid_matches"],
            ),
            plot(
                "Inlier and coverage fractions",
                tracking,
                ["inlier_ratio", "source_coverage", "target_coverage"],
            ),
            name="Tracking",
        ),
        rrb.Horizontal(
            plot(
                "Reprojection residual (px)",
                tracking,
                ["inlier_rmse_px", "final_original_consensus_rmse_px"],
            ),
            plot(
                "ICP held-out residual (m)",
                tracking,
                ["icp_common_test_initial_rmse_m", "icp_common_test_final_rmse_m"],
            ),
            name="Odometry residuals",
        ),
        rrb.Horizontal(
            rrb.TimeSeriesView(
                name="Position difference (m)",
                origin="diagnostics/pose_error/translation_m",
            ),
            rrb.TimeSeriesView(
                name="Rotation difference (deg)",
                origin="diagnostics/pose_error/rotation_deg",
            ),
            name="Pose diagnostics",
        ),
        rrb.Horizontal(
            plot(
                "Voxel counts",
                map_root,
                [f"{mode}/voxel_count" for mode in ("estimated", "reference")],
            ),
            plot(
                "Multiframe fraction",
                map_root,
                [f"{mode}/multiframe_fraction" for mode in ("estimated", "reference")],
            ),
            name="Map diagnostics",
        ),
        rrb.Horizontal(
            plot(
                "Depth RMSE (m)",
                "diagnostics/consistency",
                [f"{root}/absolute_depth_error_m/rmse" for root in pair_roots],
            ),
            plot(
                "Valid target overlap",
                "diagnostics/consistency",
                [f"{root}/valid_target_fraction" for root in pair_roots],
            ),
            name="Cross-view consistency",
        ),
        rrb.TextLogView(
            name="Frame status and diagnostics", origin="diagnostics/events"
        ),
        active_tab=0 if segment_ids else 3,
    )
    return rrb.Blueprint(
        rrb.Vertical(rrb.Tabs(*tabs), diagnostics, row_shares=[4, 1])
        if show_diagnostics
        else rrb.Tabs(*tabs),
        rrb.TimePanel(timeline="frame", play_state="paused"),
        rrb.BlueprintPanel(state="expanded"),
        auto_layout=False,
        auto_views=False,
    )
