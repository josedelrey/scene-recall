# Multiframe RGB-D mapping

SceneRecall reconstructs colored surface-voxel maps from canonical RGB-D
observations and format-2 [odometry exports](odometry.md#outputs-and-reproducibility).
Mapping consumes accepted poses without correcting them. Reference poses create
a separate reconstruction diagnostic. See [held-out validation](heldout_mapping_validation.md)
for scoring withheld fusion views.

## Representation and boundaries

The implementation lives in `src/scene_recall/mapping/`:

- `voxel.py`: filtering, native-depth sampling, RGB visibility and voxel fusion.
- `pipeline.py`: keyframe selection, continuity validation and bounded submaps.
- `evaluation.py`: cross-view depth consistency and paired map distances.
- `export.py` and `viewer.html`: archives, colored PLY, previews and offline 3D.
- `cli.py`: dataset/export pairing, provenance checks and atomic publication.

The representation averages surface samples within metric voxel cells.
It provides inspectable geometry and support without adding a TSDF, mesh,
optimizer, pose graph or loop closure. Each frame contributes one geometric
centroid per occupied voxel and one mean RGB sample per colored voxel. A frame
with many pixels in a voxel therefore has the same fusion weight as another
frame observing that voxel. Colors are averaged in stored RGB space. Geometry
survives missing color, with color support counted separately.

Voxel support counts distinct integrated keyframes. Positional scatter measures
the population spread of per-frame centroids inside a voxel. It is bounded by
cell size and changes with viewing density. It cannot establish surface
accuracy or detect all doubled surfaces. NPZ retains the keys, float64 XYZ,
uint8 RGB, frame support, color support, scatter and JSON metadata. PLY retains
XYZ, RGB and frame support for external 3D tools.

Defaults use 2 cm voxels, every fourth native depth pixel, depth from 0.3 to
4 m inclusive, and the existing conservative 5 cm depth-edge filter. Borders,
holes and discontinuities are rejected before sampling. Colors use `T_RD` and
`K_R`, with a native-depth nearest-pixel z-buffer in the RGB camera rejecting
occluded background samples. No depth interpolation, distortion correction or
hole filling is introduced. The TUM adapter's existing registered pinhole
approximation and metric conversion remain authoritative.

Keyframes occur every ten frames relative to the start of each estimated
segment, with both segment endpoints included. This deliberately simple policy
keeps comparisons reproducible. It uses no reference pose and makes no
motion-adaptive view-selection claim. Default submaps hold at most 24
keyframes and 500,000 voxels per pose mode. Voxel-cap overflow raises before
modifying the map. It does not silently discard geometry. Increase
`--max-keyframes-per-submap` explicitly when a comparison requires a larger map.

Each tracking failure starts an independent segment. A `lost` frame is the
first camera of the new segment and is eligible for integration. Segment
boundaries must agree with status, ordered IDs and identity reset poses.
A submap never crosses a segment boundary. Its first estimated camera defines
map frame M. With the existing camera-to-segment convention:

```text
T_SM = T_SC_D[first keyframe]
T_MC_D[i] = inverse(T_SM) @ T_SC_D[i]
```

Submaps in the same segment retain `T_SM`, which can place their estimated
geometry together without inventing new constraints. Transforms between
disconnected segments remain unknown. The viewer selects independent submaps
and never overlays unrelated segment origins as a global scene.

For a reference diagnostic, let `a` be the first valid reference keyframe in a
submap. The separate reference alignment is:

```text
T_MW_reference = inverse(T_SM) @ T_SC_D[a] @ inverse(T_WC_D[a])
T_MC_D_reference[i] = T_MW_reference @ T_WC_D[i]
```

This is first-pose alignment with metric scale fixed. No best-fit geometry or
trajectory alignment hides subsequent drift. Reference alignment never enters
estimated integration, keyframe selection, segment placement or tracking.
Reference submaps have their own diagnostic alignments, so combining them
requires respecting their recorded `T_MW_reference` values. Missing and
nonrigid references are counted and skipped in reference integration. Estimated
geometry remains available. A map-distance comparison is withheld unless both
maps use exactly the same keyframes. Depth-pair diagnostics require valid
references at both endpoints and evaluate both pose modes on that paired cohort.

## Running and inspecting a reconstruction

Create an odometry export with the existing `scripts/rgbd_odometry.py` if needed.
Then map it into a fresh directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  scripts/rgbd_mapping.py \
  --odometry-run outputs/odometry/desk_sparse_200 \
  --pair-keyframe-offsets 1 3 9 \
  --output-dir outputs/mapping/desk_example
```

`--source-path` can relocate an unchanged TUM directory while preserving its
name. ScanNet exports currently bind absolute source-file provenance, so moving
a ScanNet source causes strict provenance validation to fail. Dataset identity, range,
calibration, timestamps, reference arrays and per-frame provenance must match
the saved run. The runner rereads all canonical inputs, including nonkeyframes,
and verifies the original RGB/depth/reference SHA-256 fingerprint before
publishing output. The report also hashes the odometry report, trajectory
archive and current package sources, including the HTML template. Source
indices and individual keyframe input hashes remain in `keyframes.json`.
Malformed or mismatched inputs discard the staging directory. Existing outputs
are refused. Original odometry runs are preserved.

Each completed run contains:

- `report.json`: configuration, provenance, segment/submap geometry, diagnostic
  populations, per-pair results, total runtime and Linux peak process RSS.
- `keyframes.json`: selected local/source indices, timestamps, input hashes and
  original adapter provenance.
- `submap_NNN_estimated.npz/.ply` and `submap_NNN_reference.npz/.ply`.
- `submap_NNN.png`: paired colored previews at a shared scale and orientation.
- `viewer.html`: a self-contained WebGL viewer with synchronized orbit/zoom,
  camera paths, submap selection, minimum frame support and support coloring.

Open `viewer.html` in a WebGL-capable desktop browser. It needs no network,
server, CDN or extra Python dependency. Display sampling is deterministic and
limited to 60,000 points per panel. Exported maps and evaluation use all voxels.
Raw RGB-D retention is bounded by the submap population and diagnostic history.
Reports, provenance and viewer payloads grow with the number of submaps, so
this runner targets bounded offline experiments.

## Geometric diagnostics

Cross-view consistency projects sampled source depth into the target native
depth grid using the appropriate relative pose. Both directions are evaluated
at configurable keyframe offsets. Reports include projected in-bounds counts,
valid-target overlap, untrimmed absolute z-depth errors, agreement within 5 cm,
and samples in front of or behind the observed surface. There is no residual
rejection that could hide pose errors. True occlusion, disocclusion, dynamics,
nearest-pixel projection and sensor/calibration effects also create residuals.
Pooled RMSE alone is consequently insufficient. Fractions use valid projected
samples, with the separate overlap denominator always reported.

Map comparison measures each voxel centroid against the other cloud in both
directions. It computes exact nearest distances inside a radius of one voxel
width by searching neighboring cells. Distances beyond that radius are capped
at the configured voxel width, 2 cm by default. Reports include the fraction
strictly within the cap, capped-distance statistics and matched-only statistics. These are descriptive surface agreement
and coverage proxies. A capped RMSE is not an unbounded reconstruction error,
and matched-only error omits unmatched geometry. Voxel counts and support
fractions also depend on coverage and viewpoint, so extra voxels alone do not
prove duplicate surfaces.

Both reconstructions share the same depth and calibration. The reference map
is a pose diagnostic, not an independently measured perfect scene. In
particular, Hybrid directly optimizes depth consistency, which can favor its
short-range scores without establishing globally correct motion.

## Comparisons and repeat verification

Pass saved trajectories through identical mapping settings:

```bash
uv run --locked python experiments/compare_rgbd_maps.py \
  --odometry-run outputs/odometry/desk_sparse outputs/odometry/desk_hybrid \
  --output-dir outputs/mapping/comparison
```

Create both trajectories with the same source and range using the
[odometry runner](odometry.md#running-a-capture). The comparison uses a 64-keyframe
submap bound by default, while the standalone mapping runner defaults to 24.
It launches each run in a separate process and writes `summary.csv` and
`comparison.json`, including report hashes and source/keyframe/submap identity
checks. Different sequence results remain separate. Backend comparisons require
matching keyframes and segment membership.

Repeat a mapping run with the same options into a fresh directory, then verify:

```bash
uv run --locked python experiments/verify_mapping_repeats.py \
  outputs/mapping/desk_example outputs/mapping/desk_repeat
```

The verifier compares map archives, provenance and numerical diagnostics,
excluding volatile runtime and resource fields.

## Limits

Surface voxels do not model free space, signed distance, normals or connectivity.
They cannot remove moving objects, reconcile inconsistent surfaces or infer
calibrated uncertainty. Fixed temporal keyframes can undersample fast motion
and oversample stationary views. Per-voxel source-frame IDs and semantic
associations are not retained. Keyframe hashes and provenance permit later input
verification and reintegration.

Cross-view consistency and agreement with a reference-pose map share depth and
calibration. Neither establishes external surface accuracy. Evaluate reference
and estimated target poses separately with the
[held-out workflow](heldout_mapping_validation.md).

Total runtime includes rereading and hashing all inputs, sampling, fusion,
diagnostics and exports. It excludes the prior odometry run and does not isolate
fusion latency. Reports and viewer payloads grow with submap count. The runner
is intended for offline processing. Interactive WebGL behavior still needs
manual browser verification beyond static previews and JavaScript syntax checks.
