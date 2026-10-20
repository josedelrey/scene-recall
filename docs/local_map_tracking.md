# Experimental incremental local-map tracking

`local_map` estimates incoming camera poses against retained metric landmarks.
Sparse remains the default. The frontend has bounded state, explicit fallback
and independent trajectory segments. Loop closure, global relocalization,
pose-graph optimization and joint bundle adjustment are outside its scope.

## Run and reuse

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  scripts/rgbd_odometry.py \
  --tum-path data/tum/rgbd_dataset_freiburg1_xyz \
  --backend local_map --output-dir outputs/odometry/xyz_local_map
```

`--sens-path` selects ScanNet instead. The usual source range and Sparse solver
flags apply. Local-map CLI controls are `--local-map-keyframes`,
`--local-map-landmarks`, `--local-map-update frozen|median`, and
`--local-map-no-fallback`. The local-map feature budget defaults to 1500.
`LocalMapConfig` exposes the remaining engineering thresholds programmatically.

```python
from scene_recall.odometry import LocalMapConfig, LocalMapRGBDBackend
from scene_recall.odometry.pipeline import track_observations

backend = LocalMapRGBDBackend(LocalMapConfig())
rows = list(track_observations(observations, calibration, backend))
```

Use one backend per serial stream and pass the same calibration object. The
backend retains feature measurements and estimated poses, with no reference-pose
access. Its keyframes, ownership arrays and landmarks are inspectable in memory.
Saved version-2 trajectories and per-frame diagnostics work with existing
mapping and held-out surface fusion. The active landmark map is currently an
in-memory frontend state, without restart/resume serialization.

Exact repeats require the same starting history. The existing
`verify_clip` helper in `experiments/verify_odometry_repeats.py` supports
local-map prefixes. Its CLI retains the older backend-comparison report schema.
A later clip starts with a different map and cannot be expected to reproduce
the full run through a coordinate change alone.

## Estimation and map policy

1. Extract SIFT once per input using the existing conservative RGB-D sampling.
   Missing depth removes the depth factor, while RGB pixels remain usable.
2. Match retained keyframe descriptors directly to incoming descriptors with
   mutual Lowe checks in both directions. Proposals share stable landmark IDs.
   Conflicting feature-to-landmark assignments are excluded. Each landmark and
   incoming feature contribute at most one correspondence to pose fitting.
3. Estimate map-to-RGB pose with the existing seeded EPnP RANSAC and LM solver.
   Filter consensus with available metric depth, refit and recheck final support.
   A pixel must pass the 3 px reprojection threshold. Its measured RGB-frame
   point, when available, must agree within `0.03 m + 0.01 * measured_z`.
4. Require at least 15 final inliers, 25% consensus, 15% occupied cells in a
   fixed 6 by 8 image grid, and normalized bearing-information eigenvalue ratio
   at least `1e-5`. Translation columns use median scene depth. These are
   safeguards, without calibrated uncertainty or correctness guarantees.
5. Invert the accepted map-to-RGB pose and apply `T_RD` to obtain the absolute
   depth-camera pose in the current segment. Return its relative change through
   the existing `PairEstimate` interface. A map rejection tries the complete
   Sparse estimator on already extracted consecutive measurements. Failure of
   both paths resets the map at the incoming depth camera's identity pose.

The map holds at most six keyframes and 4000 landmarks. Insert keyframes after
at least three inputs when translation exceeds 8 cm, rotation exceeds 8 degrees,
or current map support divided by the latest keyframe's owned features falls
below 0.65. This support heuristic can exceed one because other keyframes also
contribute. Force insertion after 20 inputs. FIFO eviction removes observations
and deletes orphan landmarks. New depth anchors are distributed across image
cells within the remaining landmark budget. Rejected or ambiguous descriptor
associations cannot become fresh anchors under different identities.

Default `frozen` points retain their original metric anchor, including after its
source keyframe expires while another observation retains the identity. This
avoids repeatedly averaging a point with poses estimated from that same point.
The optional `median` variant reuses existing multiview landmark construction
and validates a proposed coordinate median against every active observation.
Keyframe poses stay fixed in both variants. Counts of landmarks seen in multiple
active keyframes describe support and do not establish independent information.

Map tracking always uses mutual two-direction ratio checks. Sparse's
`mutual_matching` and bounding-box `min_coverage` govern fallback, while map
tracking uses its explicit grid and information gates. Depth is a consensus
gate, rather than a jointly optimized residual. There is no timing compensation
for RGB/depth exposure differences.

## Reproduce the comparison

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/incremental_local_map.py \
  --tum-path data/tum/rgbd_dataset_freiburg1_xyz \
  --output-dir outputs/local_map/xyz --evaluate-maps
```

Default variants are original unlimited-feature `sparse`, 1500-feature `budget`,
1500-feature `hybrid`, and `local`. Optional `median`, `no_fallback` and
`short_map` variants isolate point updating, fallback and a two-keyframe budget.
Pass `--variants sparse budget local median no_fallback short_map` to select
them. `--summarize-only` rebuilds comparisons from saved runs.

`comparison.json` preserves own-population evaluation, exact common RPE pairs
against Sparse, losses, recoveries, map support, stage timing, configurations and
report/archive hashes. It verifies canonical input identity, calibration,
provenance, reference arrays, software fingerprints and numerical environment.
No reference is used to select poses or keyframes. A fragmented trajectory has
no full-sequence ATE, and common-pair accuracy is conditional on continuity.

Map evaluation uses the existing surface-voxel and withheld-fusion-view workflow,
with identical temporal training frames and reference target poses. It refuses
to compare differing segmentation. The fixed protocol uses 2 cm voxels, stride
8 depth sampling, every tenth training frame, offset-five target views, at most
24 training views per submap and 1 cm render discs. Report both common-mask
errors and coverage/explained target fractions. Withheld views are excluded
from fusion only. The online frontend has already observed their images.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/local_map_outages.py \
  --tum-path data/tum/rgbd_dataset_freiburg1_xyz \
  --output-dir outputs/local_map/outages
```

This controlled probe compares the same-budget Sparse and local-map paths on
clean inputs, a ten-frame missing-depth interval, black RGB, and 50% depth-scale
bias. It retains trajectories and all loss indices. These interventions test
specific failure mechanisms and do not estimate natural failure probabilities.

## Evaluation on 2026-10-20

Fixed settings were evaluated on all associated XYZ, Desk and RPY inputs and
two 200-frame ScanNet clips. Position ATE uses fixed-scale SE(3) alignment and
is in millimeters. Budget Sparse and Hybrid use 1500 features, matching local
map tracking. Original Sparse remains unlimited.

| Case | Sparse | Budget Sparse | Budget Hybrid | Local map |
| --- | ---: | ---: | ---: | ---: |
| XYZ, 792 inputs | 55.99 | 55.11 | 25.47 | 37.26 |
| Desk, 573 inputs | 76.53 | 75.84 | 61.44 | 39.24 |
| RPY, largest segment | 50.26 | 50.64 | 62.91 | 51.74 |
| ScanNet 1000:1200 | 38.39 | 38.39 | 83.36 | 60.86 |
| ScanNet 3500:3700 | 49.44 | 49.80 | 209.28 | 57.00 |

RPY has 694 inputs and the same loss at frame 115 in every variant. Its table
entry covers the 579-frame segment 115:694, without joining gauges. Its first
115-frame segment also worsens, from 57.05 to 67.86 mm. There is no full-run
RPY ATE. Other cases remain continuous. Local-map transition modes are
791 map fits for XYZ, 557 map fits plus 15 fallbacks for Desk, 680 map fits plus
12 fallbacks and one loss for RPY, and 190/198 map fits plus 9/1 fallbacks for
the two ScanNet clips. The natural RPY loss has weak input support, with only
75 source keypoints and 12 final Sparse inliers. Map matching supplies 36
proposals but still fails the final-inlier gate. Thresholds stay unchanged.

Single-frame translation RPE worsens in every case. For XYZ it rises from
5.27 to 6.21 mm, while 100-frame RPE improves from 37.59 to 27.36 mm. Desk's
corresponding values are 9.49 to 10.26 mm and 86.26 to 47.73 mm. Both ScanNet
clips worsen in position and orientation. On ScanNet 3500:3700 the largest
per-frame regressions coincide with lower map support, despite passing grid
and information checks. Median point updating and a two-keyframe window do
not resolve that regression. This supports retaining the experimental status.

Fusion scores use identical samples and reference target poses. The paired
depth MAE below uses each case's exact common reference-target pixel mask.
Coverage and explained fractions use all valid withheld target depths. Arrows
show Sparse to local-map tracking, and percentages retain missing predictions.

| Case | Common MAE, mm | Coverage, % | Explained within 5 cm, % |
| --- | ---: | ---: | ---: |
| XYZ | 36.77 → 36.52 | 94.68 → 94.93 | 82.77 → 83.52 |
| Desk | 55.84 → 57.36 | 91.41 → 91.88 | 69.53 → 65.45 |
| RPY | 126.90 → 106.84 | 87.03 → 86.03 | 50.90 → 50.52 |
| ScanNet 1000:1200 | 46.21 → 57.78 | 86.13 → 87.08 | 58.77 → 51.08 |
| ScanNet 3500:3700 | 71.15 → 68.00 | 83.54 → 83.55 | 44.84 → 46.79 |

Evaluated view counts are 75, 55, 44, 20 and 20 respectively. One XYZ target
lacks a reference. RPY withholds 23 targets because their training chunk lacks
reference frame 445. Better ATE does not establish better fusion, as Desk shows.
Reference-pose fusion shares sensor and timing errors and is not independent
surface ground truth. These related sequences and one ScanNet capture do not
establish generalization or statistical significance.

Two serial 40-frame XYZ replays in opposite backend orders measured 2.44–2.57
frames/s for local map, 3.97–4.12 for Sparse, 4.36–4.60 for budget Sparse and
1.42–1.43 for budget Hybrid. Opening, decoding and tracking are included.
In the latter half of the local-map prefixes, median extraction and map tracking
times are about 167 and 269 ms. This short, cached-input check includes map
initialization and is indicative rather than a real-time guarantee. Concurrent
full-sequence timings are not used for backend ranking.

All 12 trajectory arrays and numerical diagnostics, excluding timing, match
exactly across both replay orders for all four variants. The separate 200-frame
local-map run also exactly matches the full XYZ prefix. Original Sparse's three
full TUM runs exactly reproduce saved baseline archives and diagnostics.
Verification passed 818 tests, Ruff lint/format and whitespace checks. Outputs
are retained under ignored `outputs/local_map/`, including `comparison.json`
per case, `evidence_summary.json`, `prefix_reproducibility.json` and
`baseline_equivalence.json`. No new dependency is required.

## Controlled findings and remaining limits

On the first 200 XYZ inputs, position ATE is 17.25 mm for Sparse, 11.98 mm for
frozen local-map tracking, 12.96 mm with median point updates and 16.13 mm with
two keyframes. Disabling fallback reproduces every frozen-map trajectory array
exactly. The gain therefore comes from map tracking, with a benefit from the
larger retained view set on this clip. Median updates do not improve the frozen
result on either this clip or ScanNet 3500:3700.

The analytic control below supplies exact feature pixels and identities with
known camera motion. A 1.5% bias affects only the first view's depth. Endpoint
error is 0.19 mm for Sparse and 3.42 mm for frozen map tracking, without losses.
The bias passes the depth gate and persists through frozen anchors. Depth
consensus alone cannot correct the anchor scale. This demonstrates a possible
failure mechanism, without attributing real ScanNet errors to that mechanism.

```bash
uv run --locked python experiments/local_map_anchor_bias.py \
  --output-dir outputs/local_map/anchor_bias
```

The frontend remains experimental. SIFT extraction and brute-force matching
across retained views are substantial costs. Single-depth anchors carry noise
and calibration bias, and repeated observations share pose-estimation errors.
No model accounts for RGB/depth exposure motion or correlated depth noise.
Long occlusion, absent RGB features, moving geometry and prolonged corrupt depth
remain limitations. The bounded window supplies local continuity without
global recovery. Earlier offline BA experiments use future measurements and
different optimization populations, so they are not matched online baselines.

Before promotion, evaluate independent scenes, stronger rotations and wider
frame gaps. Develop visibility/covisibility search to reduce matching cost,
then test uncertainty-aware landmarks and exposure-aware pose factors with
fixed controls. Joint local optimization should preserve committed boundaries
and use the existing observability checks. More threshold searches on these
same clips would provide weaker evidence.
