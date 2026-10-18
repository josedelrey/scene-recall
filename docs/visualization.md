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

The mapping input must be a standard format-1 surface-voxel run from
`scripts/rgbd_mapping.py`, including existing runs created before this workflow.
The exporter finds the original format-2 odometry run through its saved report.
It reads saved maps and accepted poses, without executing tracking, fusion or
scoring. Existing HTML files in old ignored experiment folders are unused.
New mapping runs produce only scientific exports until a recording is requested.

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
- **Submap tabs** compare estimated and reference-pose reconstructions side by
  side in the submap frame. Each uses its saved first-pose diagnostic alignment.
  Switch between RGB and frame-support coloring. Reference submaps have
  separate alignments and are never treated as a globally corrected map.
  Unpaired reference populations are labeled in the view title, and scientific
  map-distance comparisons remain withheld.
- **Diagnostics** show tracking measurements, position and rotation differences
  from aligned references, map statistics, directed cross-view consistency and
  per-frame status logs. Map-statistic dots are sampled at each submap's final
  keyframe and describe the complete map. Raw saved reports are available under
  Guide and provenance. A reference-pose reconstruction is a diagnostic, not independent
  surface truth.

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
tracker or mapper and logs to a caller-owned `RecordingStream`. The CLI handles
experiment loading, validation and recording publication. Core mapping and
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

The initial exporter supports standard mapping and odometry runs. Specialized
held-out and TSDF experiment report schemas require dedicated adapters rather
than guessing coordinate frames from archive filenames. Their NPZ/PLY/JSON
outputs and evaluation workflows remain unchanged.

See the upstream [Rerun quick start](https://rerun.io/docs/getting-started) and
[transform conventions](https://rerun.io/docs/concepts/logging-and-ingestion/transforms)
for native viewer and logging concepts.
