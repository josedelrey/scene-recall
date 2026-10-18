# Multiframe RGB-D mapping

SceneRecall now reconstructs colored surface voxel maps from canonical RGB-D
observations and existing format-2 odometry exports. Sparse remains the default
odometry backend. Mapping consumes accepted poses without correcting them.
Ground-truth poses create a separate reconstruction diagnostic. The first real
evaluation uses all 573 associated frames of TUM `freiburg1_desk`, with Sparse
and Hybrid trajectories already produced by the odometry milestone.

## Representation and boundaries

The implementation lives in `src/scene_recall/mapping/`:

- `voxel.py`: filtering, native-depth sampling, RGB visibility and voxel fusion.
- `pipeline.py`: keyframe selection, continuity validation and bounded submaps.
- `evaluation.py`: cross-view depth consistency and paired map distances.
- `export.py` and `viewer.html`: archives, colored PLY, previews and offline 3D.
- `cli.py`: dataset/export pairing, provenance checks and atomic publication.

This first representation averages surface samples within metric voxel cells.
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
modifying the map. It does not silently discard geometry. The experiment uses
64 keyframes per submap so that the 59 selected Desk frames fit in one map.

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
  --odometry-run outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_sparse \
  --max-keyframes-per-submap 64 --pair-keyframe-offsets 1 3 9 27 \
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
at 2 cm. Reports include the fraction strictly within the cap, capped-distance
statistics and matched-only statistics. These are descriptive surface agreement
and coverage proxies. A capped RMSE is not an unbounded reconstruction error,
and matched-only error omits unmatched geometry. Voxel counts and support
fractions also depend on coverage and viewpoint, so extra voxels alone do not
prove duplicate surfaces.

Both reconstructions share the same depth and calibration. The reference map
is a pose diagnostic, not an independently measured perfect scene. In
particular, Hybrid directly optimizes depth consistency, which can favor its
short-range scores without establishing globally correct motion.

## Reproducible Desk and disconnected RPY experiments

Run the saved trajectories through identical geometry settings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/compare_rgbd_maps.py \
  --odometry-run \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_sparse \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_hybrid \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_rpy_0_all_sparse \
  --output-dir outputs/mapping/reproduction
```

The comparison starts each run in a separate process. It writes `summary.csv`
and `comparison.json`, including report hashes and explicit checks of source
identity, keyframe input identity and segment/submap membership before labeling
backend cohorts comparable. Different sequence results remain separate.

The recorded artifacts are under `outputs/mapping/final/`. Preliminary bounded
24-keyframe Desk reconstructions are in `outputs/mapping/desk_sparse/` and
`outputs/mapping/desk_hybrid/`. Generated artifacts remain ignored by Git.
No odometry backend, dependency requirement or lockfile was changed.

Desk selects frames 0, 10, ..., 570, 572, for 59 keyframes in one segment. All
selected references are valid. Sparse and Hybrid use identical keyframe hashes,
segmentation and calibration. Their reference map arrays are identical.

| Desk pose mode | Voxels | Voxels supported by multiple frames | Estimated → reference within 2 cm | Reference → estimated within 2 cm |
| --- | ---: | ---: | ---: | ---: |
| Sparse | 124,493 | 41.21% | 45.15% | 52.74% |
| Hybrid | 104,802 | 45.91% | 55.63% | 53.02% |
| Reference | 106,815 | 41.15% | — | — |

Hybrid's estimated-to-reference coverage improves by 10.49 percentage points,
while reverse coverage changes by only 0.28 points. Sparse produces 16.55%
more voxels than the reference map. Hybrid produces 1.89% fewer. This supports
examining excess or dispersed estimated geometry, with coverage confounds still
present. Capped-distance RMSE is 16.56 / 15.92 mm for Sparse's forward/reverse
directions and 15.71 / 15.90 mm for Hybrid. The cap limits their interpretation.

The depth consistency results below pool both directions. Frame offsets are
usually ten times keyframe offsets, with a shorter final endpoint interval.
The long-offset cohort has 64 directed pairs. Each cell gives agreement within
5 cm / valid-target overlap as percentages of their respective denominators.

| Keyframe offset | Approximate frame gap | Sparse | Hybrid | Reference |
| --- | ---: | ---: | ---: | ---: |
| 1 | 10 | 91.65 / 73.94 | 95.60 / 74.72 | 91.37 / 73.98 |
| 3 | 30 | 83.14 / 55.08 | 86.49 / 55.47 | 85.80 / 55.26 |
| 9 | 90 | 56.27 / 45.92 | 66.86 / 47.07 | 68.58 / 46.39 |
| 27 | 270 | 20.42 / 33.39 | 33.30 / 32.41 | 50.47 / 31.35 |

Sparse's spatial consistency deteriorates with separation. Hybrid improves
these agreement fractions on Desk, while both estimates remain substantially
below reference agreement over the longest interval. Raw long-offset RMSE is
492.1 mm for Sparse, 476.1 mm for Hybrid and 467.9 mm for reference poses.
Large occlusion contributions explain why those values alone obscure the
agreement differences. These measurements justify keeping Hybrid available
for mapping experiments. One sequence and correlated depth diagnostics do not
justify replacing Sparse as the general default.

The RPY run uses 72 keyframes from all 694 observations. The natural loss at
frame 115 yields independent submaps for 0–114 and 115–693, containing 13 and
59 keyframes. It excludes 26 cross-segment candidate pairs and seven pairs
with unavailable reference endpoints. The second map's last selected frame
has no reference, so its map-distance comparison is explicitly unavailable.
Estimated integration still includes that frame. Its reference reconstruction
contains the other 58 selected observations. This exercises the intended
missing-reference behavior without silently changing the estimated population.

The final runs took 27.99 s for Desk Sparse, 32.44 s for Desk Hybrid and
35.92 s for RPY Sparse. Peak process RSS was 389.1, 375.0 and 500.0 MiB,
respectively. These are single-run measurements on the current host and include
input validation, diagnostics and exports. Some verification work overlapped
the experiment, so these resource observations include possible contention.

A separate complete Desk Sparse repeat matched both map archives exactly,
including all 14 stored arrays/metadata entries, keyframe provenance and
numerical report diagnostics. Runtime and peak RSS were excluded from equality
checks. `outputs/mapping/final/reproducibility.json` records the result. Repeat
a run into a fresh directory, then verify it with:

```bash
uv run --locked python experiments/verify_mapping_repeats.py \
  outputs/mapping/final/rgbd_dataset_freiburg1_desk_0_all_sparse \
  outputs/mapping/desk_sparse_repeat_final
```

## Verification and interpretation limits

All 509 tests pass, including 30 mapping tests. Ruff lint, formatting and
whitespace checks pass.

Controlled tests cover an analytic rotated plane, camera-to-map direction,
negative voxel cells, equal per-frame weights, color support and visibility,
capacity atomicity, duplicate integration, metric filtering, exact capped
nearest-distance comparison against brute force, pose-induced double surfaces,
segment endpoints and resets, reference independence, missing references,
observation/export mismatches, output preservation and repeated map equality.
The real TUM adapter is exercised in end-to-end export tests. The package wheel
was built offline using locally cached build tools and checked for the viewer
template. Generated viewer JavaScript passed a syntax check. PNG previews were
inspected visually. Interactive WebGL rendering was not browser-tested because
the browser tool blocks local-file URLs and the sandbox disallows a local HTTP
listener. Manual browser interaction remains a verification gap.

The standard locked UV command attempted an editable rebuild after the
package-data change, but its temporary cache lacked setuptools and network DNS
was unavailable. Tests and experiments used the existing `.venv` through
`uv run --no-sync` with a writable temporary UV cache. No dependencies were
installed or changed. Normal locked commands above apply when build tools are
available. Offline wheel verification used the existing cached setuptools and
wheel packages.

The representation retains surface samples without modeling signed distance
or free space. It cannot remove moving objects, resolve mutually
inconsistent surfaces, or infer calibrated confidence. Fixed temporal sampling
can undersample fast motion and oversample stationary views. Per-voxel frame
IDs, normals, exposure compensation and semantic associations remain deferred.
Input hashes and retained keyframe provenance support later reintegration.
Timing includes rereading and hashing every input, sampling, fusion,
diagnostics and exports. It excludes the earlier odometry run. It does not
isolate integration latency or establish real-time operation.

Held-out view validation is now implemented and reported in
[Held-out RGB-D map validation](heldout_mapping_validation.md). It qualifies the
apparent Hybrid advantage above by separating reference-pose geometry from
estimated-pose self-consistency. The next useful steps are motion/overlap-aware
keyframe selection and voxel-size and submap-length sensitivity. Then add surface
normals or a TSDF if reconstruction requirements justify them. Introduce
multiframe pose constraints, relocalization or loop closure when map evidence
makes their benefit measurable. Persistent semantic memory can build on stable
submap identity, explicit coordinate transforms and retained frame provenance.
