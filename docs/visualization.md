# Offline Rerun visualization

Rerun is SceneRecall's visualization workflow. Generate a portable `.rrd`
recording from saved results, then open it in the native Rerun viewer. A
recording embeds logged arrays, images, diagnostics, provenance reports and a
curated blueprint. Once generated, it needs neither the original dataset nor
network access. NPZ, PLY and JSON remain authoritative scientific formats.

## Install, generate and open

Install the optional SDK and native viewer once in an environment with access
to the package index:

```bash
uv sync --locked --extra visualization
uv run --locked --extra visualization python scripts/rgbd_recording.py \
  --mapping-run outputs/mapping/desk_sparse_200 \
  --output outputs/visualization/desk_sparse_200.rrd
uv run --locked --extra visualization rerun outputs/visualization/desk_sparse_200.rrd
```

Use the same locked viewer version as the exporter. The supported SDK minor
series is bounded in `pyproject.toml`, with an exact version in `uv.lock`.
The SDK bundles the native viewer. No viewer is launched during export, so
recordings can be generated on a machine without a display. The native viewer
needs a working graphics backend.

After installation, export and viewing work offline. With a populated local
package cache, add `--offline` to `uv run` to prevent package-index access.
Rerun's viewer has its own analytics preference, which can be disabled with
`uv run --locked --extra visualization rerun analytics disable`.

The `--mapping-run` input is a standard format-1 surface-voxel run from
`scripts/rgbd_mapping.py`, including existing runs created before this workflow.
The exporter finds the original format-2 odometry run through its saved report.
It reads saved maps and accepted poses, without executing tracking, fusion or
scoring. Existing HTML files in old ignored experiment folders are unused.
New mapping runs produce only scientific exports until a recording is requested.

Use `--experiment-run` for the held-out and volumetric workflows below. All
workflows use the same scene logger, offline recording sink and native viewer.

To inspect an odometry experiment before reconstructing a map:

```bash
uv run --locked --extra visualization python scripts/rgbd_recording.py \
  --odometry-run outputs/odometry/desk_sparse_200 \
  --output outputs/visualization/desk_odometry.rrd
```

## What the workspace shows

- **Segment tabs** show estimated maps placed by their saved `T_SM`, the complete
  accepted trajectory, a moving calibrated camera, RGB and metric z-depth.
  Estimated paths are amber. Reference paths and frustums are teal, aligned once
  at the first valid reference in that segment with fixed metric scale.
- **Submap tabs** open a **Compare VO / GT** view with a shared 3D camera.
  The existing side-by-side RGB and frame-support views remain available.
  Both maps use the submap frame and its saved first-pose diagnostic alignment.
  Reference submaps have separate alignments and are never treated as a globally
  corrected map.
  Unpaired reference populations are labeled in the view title, and scientific
  map-distance comparisons remain withheld.
- **Diagnostics** show tracking measurements, position and rotation differences
  from aligned references, map statistics, directed cross-view consistency and
  per-frame status logs. Map-statistic dots are sampled at each submap's final
  keyframe and describe the complete map. Raw saved reports are available under
  Guide and provenance. A reference-pose reconstruction is a diagnostic, not independent
  surface truth.

### Compare geometry from one viewpoint

Open a submap's **Compare VO / GT** view and orbit, pan or zoom to the desired
viewpoint. In the left **Blueprint** panel, expand that view and use the eye
icons beside its `estimated` and `reference` branches. The icons appear when
hovering over a branch:

- Show `estimated` only for VO geometry in amber.
- Show `reference` only for GT geometry in teal.
- Show both for a colored overlay of geometric misalignments.

Keep the same view open while toggling visibility to retain its 3D camera.
These are native, per-view Rerun controls. Independent RGB, support and
evaluation views keep their own cameras. The comparison includes only final
geometry, avoiding occlusion from projected depth or moving cameras. Its color
overrides leave the recorded RGB, support, point attributes and mesh connectivity
unchanged. Opaque overlapping surfaces can occlude each other, so toggle each
map individually when inspecting close misalignments.

Here **GT** means reconstruction from reference poses and measured depth, which
is a diagnostic rather than independent surface truth. Coordinates are already
in the saved submap frame: estimated fusion uses `inv(T_SM) @ T_SC_D`, and
reference fusion uses `T_MW_reference_diagnostic @ T_WC_D`. The comparison
applies no additional alignment, scale adjustment or pose correction. Unpaired
reference populations remain labeled, with scientific comparisons withheld.
Each comparison stays within one submap, segment and backend gauge.

Regenerate a recording to get this workspace. Existing `.rrd` files retain
their embedded blueprint.

Scrub the **frame** timeline, whose indices are the original source-frame
indices, or play it to move through the capture. `sensor_time` retains supplied
sensor seconds as a duration when available, without assuming a Unix epoch.
Use `frame` for captures with missing timestamps. Complete paths and final maps
are static context, not incremental reconstruction history. Current cameras
are temporal and can be inspected in the entity tree.

Disconnected segments have independent scene origins and separate views.
There is no invented placement between them. Reference paths break at missing
or nonrigid poses. Estimated results remain available in their absence.

RGB and depth retain separate native grids, intrinsics and extrinsics.
`T_RD` maps depth optical coordinates into RGB optical coordinates, so RGB's
camera-to-depth transform is its inverse. Optical axes are right/down/forward.
Depth is logged in meters, with invalid NaNs converted to Rerun's zero sentinel
only in the recording. The original depth arrays remain unchanged.

Select a map point to inspect its integer voxel key, frame support, color
support and positional scatter. Submap ID, pose mode and voxel key identify
the corresponding scientific voxel. Support color runs from blue at one frame through cyan to yellow at
the full cloud maximum, using log2 scaling separately for each cloud. Compare
numeric support across maps because their color maxima can differ.

## Image and display controls

| Option | Behavior |
| --- | --- |
| `--images keyframes` | Mapping default. Embed selected mapping keyframes |
| `--images all` | Embed every observation. Default for odometry-only runs |
| `--images none` | Keep maps, poses and diagnostics without opening the dataset |
| `--maps-only` | Read only mapping results, without odometry or dataset files |
| `--submap 0 2` | Include selected submaps and their keyframe images |
| `--min-support 2` | Display only voxels observed in at least two keyframes |
| `--max-points 500000` | Change the deterministic per-cloud display cap |

The default cap is 200,000 points per cloud after support filtering. A larger
cap can show every voxel. Arrays and evaluation always retain their original
population. Point rendering uses float32 visualization coordinates, so use the
float64 NPZ coordinates for numerical work.

Images are displayed only at logged frames. Sparse keyframe mode clears RGB
and depth at unlogged frames so an old depth image cannot be projected from a
new camera pose. Choose `--images all` for continuous RGB-D playback. Submap
selection limits map and keyframe-image payloads, while segment trajectories
and diagnostic context retain the complete run.

Full-resolution image recordings can be large. Use keyframes, image-free
export or selected submaps when examining long experiments. The exporter
streams observations and loads one pair of map archives at a time. Recording
size and viewer memory still grow with retained data.

## Relocation and verification

If odometry moved, supply its new directory with `--odometry-run`. Both its
report and trajectory hashes must match the saved mapping input. With images,
`--source-path` can relocate an unchanged source under the existing adapter
provenance rules. TUM permits relocation while preserving sequence identity.
ScanNet binds absolute file provenance, so moving that source still fails
strict validation. Use image-free export when the source is unavailable.

Source image export verifies calibration, frame identities, provenance,
reference arrays and the entire canonical input fingerprint, including frames
whose images are omitted. Map archive metadata, shape and voxel counts are
checked against the mapping report. These checks detect input disagreement,
without claiming to authenticate map values against an independent checksum.

## Held-out and volumetric experiments

The experiment adapter reads these existing output schemas:

| Saved run | Visualization |
| --- | --- |
| `scripts/rgbd_heldout.py` | Surface-voxel maps, every saved radius, evaluated depths, common masks and scores |
| `experiments/compare_tsdf_fusion.py` | Colored connected triangle meshes, TSDF point controls and linked baseline maps |
| `experiments/support_aware_fusion.py` | Consensus-filtered maps and their linked baselines, with the original paired populations |
| `experiments/compare_keyframe_selection.py` | Temporal and motion-selected maps, each strategy's reference map, predictions and diagnostics |

Supply the exact odometry cohort with `--odometry-runs`, including every backend
listed in the report. Directory relocation is allowed when report and trajectory
hashes still match. The standard held-out report already contains calibration
and map context:

```bash
uv run --locked --extra visualization python scripts/rgbd_recording.py \
  --experiment-run outputs/heldout/desk \
  --odometry-runs outputs/odometry/desk_sparse outputs/odometry/desk_hybrid \
  --output outputs/visualization/desk_heldout.rrd
```

Comparison experiments also require the original held-out `--baseline-run`.
Its report and retained input archives must match the experiment's saved hashes.
It supplies baseline geometry, calibration and, for consensus runs, observed
depth and baseline predictions that were not duplicated in the proposal archive.
For example, after the [volumetric experiment](volumetric_mapping.md#reproduction):

```bash
uv run --locked --extra visualization python scripts/rgbd_recording.py \
  --experiment-run outputs/heldout/tsdf/connected/desk \
  --baseline-run outputs/heldout/final/desk \
  --odometry-runs \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_sparse \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_hybrid \
  --output outputs/visualization/desk_volumetric.rrd
```

Substitute a support-consensus or motion-keyframe comparison directory for
`--experiment-run` to inspect those experiments. Earlier TSDF point-only runs
are also supported, with their original common population and no invented mesh.

Experiment export defaults to `--images none`. It embeds saved evaluated depth,
predictions and diagnostics without opening the original dataset. Add
`--images evaluated` for native RGB-D at evaluated observations, `keyframes`
for training observations or `all` for the complete source selection. Raw RGB-D
is shown in Reconstruction at the **estimated** pose, with separate calibrated
RGB and depth cameras. The evaluation views use their named target-pose mode.
Source verification and relocation follow the same rules as standard recordings.
`--submap` limits reconstruction and evaluation payloads. `--maps-only` belongs
to standard mapping export, while experiment replay requires saved odometry.

### Native workspace and scientific conventions

Navigate through submap, backend, map pose source and representation tabs.
Within each backend, **Compare VO / GT** offers one shared-camera view per
representation with both map modes logged in the same submap frame, including
meshes and TSDF point controls. Controlled experiments log the shared reference
map only under the first backend, so other backends retain their existing
reconstruction and evaluation views without a comparison tab. Motion-keyframe
runs retain a comparison for each backend and strategy's own reference map.
Reconstruction shows the final colored map or mesh alongside accepted and
aligned reference trajectories and moving cameras. Mesh connectivity and vertex
colors are preserved without decimation. Surface-voxel RGB and support entities
remain selectable in the entity tree. TSDF points retain their own near-surface
`view_support`, including zero-support points. This diagnostic differs from
voxel `frame_support`. `--min-support` filters surface-voxel displays only.
`--max-points` caps each point cloud and each spatial residual sample. Meshes,
depth images, masks and numerical scores retain their complete populations.

Each target-pose/radius tab combines a 3D reconstruction with observed-surface
residual points, synchronized observed/predicted metric depth, signed-error
images and tensors, validity/coverage/common masks, saved per-view scores and
temporal plots. A separate observed residual surface view avoids mesh occlusion
when inspecting spatial errors. Error plots use meters, with coverage and
explained fractions in a separate plot tab. Select a spatial point for its native
pixel coordinates, measured/predicted depth, unsaturated error and saved masks.

Use the **frame** timeline and scrub to evaluated observations. Evaluation
availability logs retain withheld statuses. Unlogged and unavailable frames
clear evaluation payloads, preventing old residuals from appearing at another
camera pose. Final maps remain visible throughout. Raw reports, including pooled
scores, paired blocks, training selection and resource diagnostics, are embedded
under Guide and provenance. No tracking, fusion, rendering or scoring algorithm
runs during export.

The evaluation camera uses `K_D` with its first two rows divided by the saved
pixel stride. Pixel `(u, v)` corresponds exactly to native pixel
`(stride*u, stride*v)`, with no half-pixel shift. Observed depth is the saved
filtered target population, which can differ from raw sensor depth. Predictions
retain the saved renderer's nearest-z visibility and missing pixels. Residual
points lie on the **observed** surface, transformed through the selected target
pose into that reconstruction's submap frame.

Residual is **predicted minus observed**. Red places predicted geometry in front
of the measurement, blue behind, gray marks a missing prediction and black an
invalid target. Colors saturate at 150 mm. Tensors, point attributes and saved
scores keep unsaturated values. Missing predictions have no fabricated error.
All-target coverage and explained fractions remain separate from conditional
accuracy and common-pixel populations. Native depth color ranges are fixed to
0–4 m for comparison, while depth values outside that display range are retained.

Estimated targets use `inv(T_SM) @ T_SC_D`. Reference targets use the saved
training alignment `T_MW_reference_diagnostic @ T_WC_D`. Backend and segment
gauges stay disconnected, and each reference submap retains its own alignment.
Reference-target and estimated-target consistency are labeled separately.
Missing or invalid references never trigger a fallback to estimates.

The connected TSDF experiment's `common_all_versions` mask excludes TSDF point
controls from the primary baseline/mesh intersection. Its report calls that
population `common_all_modes`. The adapter preserves both original names and
the saved mask. The point controls can still be missing predictions within that
population. Consensus reports retain baseline-common and paired-baseline-common
populations, and motion reports retain their cross-strategy intersection. None
of these masks is reconstructed from a display-filtered cloud.

### Limits and verification

Export validates archive shapes, frame identities, map counts, triangle indices,
calibration, transforms, baseline hashes, cohort provenance and common-mask
visibility conventions. Experiment geometry itself has no independent content
checksum in the original reports. Structural checks cannot authenticate every
saved coordinate or numerical score. Unknown schemas fail explicitly.

Recordings can be large because every prediction, mask and diagnostic is
embedded. Export loads one evaluation archive at a time, and viewer memory grows
with retained content. Allow the native viewer to finish loading before judging
an empty view. Use submap selection and point caps for long runs. Dense TSDF
interior slicing is not implemented. Analysis-only residual probes and synthetic
pose-bias reports do not retain the same reconstruction/trajectory replay schema
and are not accepted as experiment inputs.

Representative validation on 2026-10-19 covered full Desk held-out playback at
three radii with evaluated RGB-D, Desk TSDF meshes and point controls, Desk
consensus and motion runs, and RPY TSDF with unavailable reference comparisons.
Scientific input files were hash-checked before and after export. The generated
recordings passed native `rerun rrd verify`. Automated fixtures additionally cover
pose-mode separation, sampled intrinsics, residual signs, missing geometry,
unavailable references, malformed archives and atomic publication failure.
Shared-camera validation checked orbiting followed by VO-only, GT-only and
overlay visibility in Rerun 0.38.1. Native rendering also checked the point-cloud,
mesh and TSDF point-control color overrides. A regenerated Desk TSDF experiment
passed `rrd verify`, with all 47 scientific input files hash-checked unchanged.

Output requires a fresh `.rrd` path. Export stages the recording beside the
output and publishes it atomically after validation and flushing. A failed
export discards temporary files. Existing scientific files are never rewritten.
For a structural check with the bundled viewer:

```bash
uv run --locked --extra visualization rerun rrd verify \
  outputs/visualization/desk_sparse_200.rrd
```

## Architecture and extension boundary

`scene_recall.visualization.rerun.SceneLogger` accepts canonical `Observation`,
`Calibration`, accepted trajectory rows and saved map arrays. It owns no reader,
tracker or mapper and logs to a caller-owned `RecordingStream`.
`scene_recall.visualization.recording.export_recording` handles experiment
loading, validation and recording publication. The CLI parses display and
source options and invokes that exporter. Core mapping and
odometry never import the Rerun SDK. A normal `uv sync --locked` keeps the core
usable without the visualization extra.

Entity paths separate `segments/<id>/estimated`, `segments/<id>/reference` and
`segments/<id>/submaps/<id>/{estimated,reference}`. Camera models and metric
transforms are explicit. Segment roots use Rerun's disconnected-transform
sentinel so a new root view cannot silently overlay unrelated gauges. This boundary can accept future ROS2 observations
without introducing ROS2 dependencies into the scientific code. Future semantic
entities can be added under the appropriate segment or submap, preserving their
IDs and coordinate ownership. No ROS2 transport or semantic inference is
implemented here.

`scene_recall.visualization.experiments` adapts the explicit scientific
experiment schemas into this same logger and recording sink. It owns input
verification and representation/pose descriptors, keeping dataset access and
report conventions outside the scene logger. Scientific NPZ/PLY/JSON outputs
and evaluation workflows remain unchanged.

See the upstream [Rerun quick start](https://rerun.io/docs/getting-started) and
[transform conventions](https://rerun.io/docs/concepts/logging-and-ingestion/transforms)
for native viewer and logging concepts. Rerun's
[viewer navigation](https://rerun.io/docs/getting-started/configure-the-viewer/navigating-the-viewer)
describes entity visibility controls, and
[view customization](https://rerun.io/docs/concepts/visualization/customize-views)
explains per-view overrides.
