# Experimental incremental local-map tracking

`local_map` estimates incoming camera poses against retained metric landmarks.
The opt-in `local_map_recent` revision refreshes depth measurements and searches
a bounded landmark descriptor index. Both remain experimental. Sparse remains
the default. Each frontend has bounded state, explicit fallback
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

Select `--backend local_map_recent` to use the revision. Its feature budget and
geometric gates match `local_map`. `--recent-map-anchor recent|frozen` and
`--recent-map-association indexed|exhaustive` expose independent policy controls.
`RecentMapConfig` and `RecentMapRGBDBackend` provide the equivalent Python API.
This backend requires `--local-map-update frozen`, the default, because it owns
the point-refresh policy itself.

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

The refinement driver compares original Sparse, budget Sparse, budget Hybrid,
the original local map and the recent-depth revision. Use the same settings for
development and independent scenes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/refine_local_map.py \
  --tum-path /path/to/independent_tum_sequence --start 0 --stop 300 \
  --output-dir outputs/local_map_refinement/independent \
  --evaluate-maps --tsdf-first-chunk
```

`--resume` accepts only matching source, configuration and implementation hashes.
`--variants sparse budget local recent anchors index` runs the development
ablations. `anchors` refreshes depth with exhaustive association, while `index`
uses indexed association with frozen depth anchors. Default settings are fixed
before independent evaluation, with no scene-specific threshold selection.

The canonical reader rejects duplicated reference timestamps. For a source with
ambiguous references, `experiments/prepare_tum_references.py --source ORIGINAL
--output-dir PREPARED` creates a separate copy omitting every row at duplicated
timestamps. It preserves image bytes and all other reference rows, records both
table hashes, and leaves the original untouched. Decreasing timestamps still
require a separate audit. This is an explicit dataset preparation step.

Surface fusion uses one shared training/target cohort and exact common
reference-target masks across budget Sparse, Hybrid and both map backends.
Differing segment boundaries withhold the fusion comparison. Optional TSDF
evaluation uses chunk zero, chosen before examining scores, with the same
training and withheld target frames. It includes reference-pose fusion and a
20 cm capped absolute-error score that charges missing predictions 20 cm.
Reference targets never enter fusion. They have already been observed by the
online tracker, so this evaluates reconstruction rather than novel-view tracking.

The original experiment remains available:

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
Pass `--include-recent` to either the outage probe or the analytic anchor-bias
control to include the revision while retaining the original controls.

## Recent-depth and indexed association policy

Persistent identity and retained-view geometry remain the map's constraints.
On each incoming frame, mutual descriptor matches first carry identities from
the previous accepted image. Insufficient count or image-grid support triggers
search against one latest retained descriptor per landmark. A rejected map fit
also triggers that search before Sparse fallback. Search always runs before
keyframe insertion to prevent reappearing known features from being seeded
under fresh identities. Ambiguous associations reserve their pixels and cannot
create new landmarks. The index contains at most the map's 4000 identities.

For an identity observed in the preceding image, valid source depth can replace
the nominal map point during PnP. The point is transformed by that source's
accepted pose and must pass pixel and available-depth checks in every retained
keyframe that observes it. Other identities retain their nominal metric points.
This gives local motion a recent measurement while preserving map localization
through missing depth, occlusion or reappearing landmarks.

At keyframe insertion, a new depth point can refresh the nominal landmark only
if it agrees with every retained observation and the incoming observation. If
refresh fails, the old point must still agree with the incoming observation
before that observation joins the landmark. Rejected observations remain
reserved. This maintains consistency inductively through insertion and eviction.
Committed keyframe poses never change. There is no pose smoothing or trajectory
revision, and reference poses do not enter association, estimation or insertion.

This policy does not estimate independent landmark uncertainty. Recent points
and retained poses share estimation errors. Depth remains a consensus gate,
and the pose solver minimizes pixel residuals. All-view agreement bounds the
stored geometry without proving correctness or supplying calibrated information.
`recent_depth_anchors` counts valid candidates in the last point-selection
check, which can occur during insertion. It is not a count of independent pose
inliers and should not be divided by `map_inliers` as an information fraction.

## Refinement evaluation on 2026-10-20

The revision keeps the original numeric gates and budgets. Policies were frozen
before inspecting four predeclared 300-frame windows from two independent
[TUM sequences](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download):
Freiburg 2 Desk and Freiburg 3 Long Office Household. The windows start at 0 and
1200 associated inputs. These are four windows from two scenes, without four
independent scene replicates. All five backends remain continuous on them.
The canonical registered-image pinhole approximation stays unchanged.

The table reports fixed-scale position ATE in millimeters. Budget Sparse and
Hybrid use the same 1500-feature cap as both map frontends. Reference counts
are 281, 182, 297 and 300 respectively. Missing references remain missing.

| Independent window | Budget Sparse | Hybrid | Original map | Recent map |
| --- | ---: | ---: | ---: | ---: |
| Freiburg 2 Desk, 0:300 | 17.35 | 90.36 | 36.33 | 27.88 |
| Freiburg 2 Desk, 1200:1500 | 13.05 | 105.76 | 7.85 | 15.24 |
| Freiburg 3 Office, 0:300 | 37.31 | 97.12 | 22.67 | 8.75 |
| Freiburg 3 Office, 1200:1500 | 21.50 | 59.62 | 18.09 | 7.51 |

Single-frame translation RPE improves relative to the original map on all four
windows, by approximately 10–30%, while remaining approximately 0.5–5.2% worse
than budget Sparse. Office 100-frame RPE improves from 66.00 to 17.35 mm and
25.90 to 16.84 mm versus budget Sparse. Both Desk windows remain worse at that
horizon. In particular, the original map's late-Desk 16.01 mm grows to 32.91 mm.
The recent measurement policy trades away a useful long-lived constraint there.

Historical full-run controls show the same motion/anchoring trade-off. Each cell
below is position ATE / one-frame translation RPE, both in millimeters.

| Case | Budget Sparse | Original map | Recent map |
| --- | ---: | ---: | ---: |
| XYZ, 792 inputs | 55.11 / 5.29 | 37.26 / 6.21 | 47.43 / 5.46 |
| Desk, 573 inputs | 75.84 / 9.48 | 39.24 / 10.26 | 46.69 / 9.90 |
| RPY, largest segment ATE | 50.64 / 8.18 | 51.74 / 11.04 | 58.53 / 9.08 |

All retain RPY's loss at frame 115. Its ATE covers only the 579-frame segment
115:694, while RPE pools eligible pairs within both segments. There is no
full-sequence RPY ATE. Lower jitter does not preserve every longer-horizon gain.

Downstream surface fusion uses the shared reference-target cohort. The next
table reports capped absolute error in millimeters over every valid target
sample, charging a missing prediction 200 mm. This supplements common-mask
depth errors, coverage and explained fractions retained in the reports.

| Independent window | Views | Budget Sparse | Original map | Recent map |
| --- | ---: | ---: | ---: | ---: |
| Freiburg 2 Desk, 0:300 | 23 | 52.84 | 47.83 | 53.38 |
| Freiburg 2 Desk, 1200:1500 | 6 | 68.11 | 67.89 | 68.85 |
| Freiburg 3 Office, 0:300 | 6 | 37.66 | 37.46 | 35.97 |
| Freiburg 3 Office, 1200:1500 | 29 | 59.59 | 46.58 | 43.47 |

Missing training references withhold entire chunks. The early-Desk final chunk,
late-Desk first chunk and early-Office first chunk are unavailable. Lower late-Desk
common-mask MAE, 37.66 to 33.33 mm, accompanies worse coverage and a worse capped
score. On late Office, explained target fraction improves from 76.65% to 80.14%
and coverage from 91.78% to 92.44%. The gain transfers to TSDF: its capped score
improves from 27.90 to 25.38 mm. Early-Desk TSDF instead worsens slightly, from
34.45 to 34.99 mm. The predetermined first-chunk TSDF comparisons for late Desk
and early Office are withheld. No better-scoring chunk replaces them.
Early-Desk position ATE improves while its capped surface error worsens, so ATE
alone would have selected a worse reconstruction policy on that window.

ScanNet regressions persist. On 1000:1200, ATE changes 60.86 to 62.03 mm and
one-frame RPE 13.71 to 12.02 mm, versus budget Sparse's 38.39 and 9.08 mm.
TSDF capped error worsens from 46.25 to 48.73 mm. On 3500:3700, ATE improves
57.00 to 54.35 mm, one-frame RPE worsens 10.39 to 10.51 mm, and TSDF capped error
improves 70.99 to 68.57 mm. Neither clip establishes a general ScanNet solution.

Full XYZ and Desk first-chunk TSDF capped error also worsens against the original
map, 25.98 to 30.91 mm and 50.45 to 51.62 mm. RPY's predetermined first chunk
improves 92.40 to 76.01 mm, but its surface score over all eligible chunks worsens
84.00 to 87.79 mm. A favorable early-chunk mesh is not a full-run geometry gain.

Development ablations separate the policies. On XYZ200, original/recent-depth
with exhaustive matching/indexed frozen/full revision ATE is
11.98/12.62/13.04/14.47 mm, while one-frame RPE is
6.04/5.89/5.81/5.58 mm. On ScanNet 3500:3700 the corresponding one-frame RPE is
10.39/9.60/12.32/10.51 mm. Indexed association can lose accurate local motion.
The largest full-revision regressions there have only 18–29 temporal identity
matches, yet pass coverage and information gates after map search. These
diagnostics indicate insufficient protection against uncertain retained anchors.
They do not identify sensor bias as the physical cause.

A supplementary Freiburg 2 Desk window, 1500:1700, was selected after the primary
runs using reference availability alone. It is the earliest nonoverlapping
200-frame block on a fixed grid with all training references available. The
selection record retains every rejected candidate and all primary results.
Settings stay unchanged. Budget/original/recent ATE is 25.52/12.67/7.43 mm and
TSDF capped error is 35.79/35.37/34.23 mm. Surface coverage still decreases from
91.27% to 90.63% and explained fraction from 82.66% to 81.85% relative to the
original map. This availability-selected control does not replace the primary
generalization test or establish uniformly better reconstruction.

The analytic initial-depth-bias control improves endpoint error from 3.425 to
0.190 mm, matching Sparse's 0.190 mm. Both map policies bridge the ten-frame
missing-depth interval without losses, while Sparse loses ten transitions.
All three lose eleven transitions with black RGB. With 50% corrupt depth, all
remain continuous but produce inaccurate motion. The revision reduces that
probe's position ATE from 13.52 to 9.83 mm. Continuity is not proof of correctness.

Freiburg 2's original archive has two conflicting reference rows at one repeated
timestamp. A separate prepared copy omits both, with table hashes and original
archives retained. The ambiguous timestamp lies outside the primary windows.
No dataset reader, calibration, reference tolerance or benchmark alignment was
changed to obtain these results. Reports, frozen protocols and paired evidence
are retained under ignored `outputs/local_map_refinement/`.

After concurrent evaluations finished, two serial 40-frame XYZ prefixes in
opposite backend orders measured 3.69–3.79 frames/s for the revision,
2.53–2.56 for the original map, 4.32–4.35 for budget Sparse, 3.92–3.97 for
unlimited Sparse and 1.42 for Hybrid. Opening, decoding and tracking are included.
The last 20 frames have median extraction near 170 ms and map tracking near
48 ms for the revision, versus 262 ms for the original map. Keyframe search
and updates still cause latency variation. These short cached-input prefixes
are indicative, without a real-time or machine-isolation guarantee.

All 14 replay checks match every trajectory array and non-timing diagnostic
exactly, including 200-frame prefixes against full runs. All 19 retained legacy
replays match all 12 trajectory arrays, complete evaluations and non-timing
frame diagnostics from the original study. Verification passes 837 tests,
Ruff lint/format and whitespace checks. No dependency change is required.
`evidence_summary.json`, `baseline_equivalence.json`,
`prefix_reproducibility.json`, `serial_timing.json` and frozen protocol records
retain the detailed paired populations and fingerprints in the output directory.

## Original fixed-anchor controls on 2026-10-20

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

Both map policies remain experimental. Before promotion, expand beyond these two
new rooms and the one ScanNet capture, with stronger rotations and wider frame
gaps. Test uncertainty-aware depth and landmark factors, including RGB/depth
exposure motion, against the current recent/frozen controls. Visibility and
covisibility search should preserve older appearances without exhaustive matching
or weak temporal support. Joint local optimization should preserve committed
boundaries and use the existing observability checks. Promotion requires
consistent local-motion, continuity and coverage-aware reconstruction gains,
plus longer controlled latency measurements. More threshold searches on these
same clips would provide weaker evidence.

## Reliability controls

The reliability investigation is isolated in `experiments/`. It leaves the
production frontends, defaults, original comparison drivers and saved baselines
unchanged. The candidate module is not registered as a backend or CLI option.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/investigate_local_map.py \
  --tum-path /path/to/independent_sequence --start 0 --stop 300 \
  --output-dir outputs/local_map_reliability/independent --evaluate-maps
```

Each case runs budget Sparse, the instrumented recent-map baseline, weighted
RGB, weighted RGB-D and descriptor-family association. All use the same immutable
SIFT measurements and the original consensus, geometry and insertion thresholds.
Every candidate maintains its own online map history. The instrumented baseline
can verify every trajectory array and shared numerical diagnostic against an
existing recent-map run with `--baseline PATH`. Fresh output directories are
required. Experiment reports preserve script fingerprints separately from the
unchanged package fingerprint.

Weighted fitting propagates a source RGB-ray point covariance to the target.
The source covariance uses one-pixel feature uncertainty and axial depth sigma
`0.005 + 0.003 z²` meters, inherited from the earlier measurement experiment.
The RGB residual covariance includes source point uncertainty plus one-pixel
incoming RGB uncertainty. The RGB-D alternative adds one axial target-depth row
and its covariance, retaining the pixel/depth cross terms induced by the source
point. Missing depth contributes only RGB rows. It does not count a target 3D
point on the RGB ray as an independent lateral measurement. The solver uses
bounded Cauchy IRLS, with per-step frozen whitening and line-search checks.
It rechecks the original consensus and support gates before accepting a change.
These are conditional engineering covariances, without camera-pose uncertainty,
inter-landmark correlations, depth bias or exposure-motion compensation.
The method is an approximate weighted fit, without a calibrated likelihood or
posterior-covariance claim.

Descriptor-family matching retains all active keyframe appearances. It takes the
minimum descriptor distance within an identity, then applies mutual ratio tests
over distinct landmark identities and target features. This prevents two views
of one landmark competing in the ratio denominator. Temporal propagation,
ambiguous-association reservation and insertion geometry remain unchanged.
Appearance matching is tested separately from the measurement alternatives.

`experiments/local_map_reliability_controls.py` generates known-correspondence
trials with independent noise, coherent depth bias and hidden depth-exposure
motion. `experiments/diagnose_local_map_reliability.py` evaluates every seventh
input's saved packets, descriptor-only three-view cycles, shared native samples
and RGB/depth compatibility. A spatial checkerboard holdout compares ordinary
PnP, robust RGB, weighted RGB and weighted RGB-D on identical support. Reference
poses enter only the diagnostics, never fitting or association. Residuals on
accepted support are selection-conditioned and do not calibrate a noise model.

The frozen evaluation retains four prior 300-frame windows from Freiburg 2 Desk
and Freiburg 3 Office, plus new 600:800 windows from both scenes. These are six
windows from two independent scenes relative to development, not six independent
scene replicates or two newly acquired scenes. ScanNet uses two 200-frame clips
from the same existing capture. There is no second independent ScanNet capture.
No scene-specific thresholds or score-selected windows are introduced.

Reconstruction uses the existing shared temporal training/target split, 2 cm
surface voxels, stride-eight depth and first-chunk TSDF. Missing training
references withhold the chunk. All alternatives are also scored on one exact
common reference-target mask. Coverage and the 20 cm capped missing-prediction
score retain the full valid-target denominator. Targets are withheld from
fusion, while their images have already been seen by the online frontend.
Concurrent cases and shared extraction caches preclude a controlled runtime
ranking.

The complete fixed matrix can be reproduced with portable source paths:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/run_local_map_reliability.py \
  --desk-path /path/to/prepared_freiburg2_desk \
  --office-path /path/to/freiburg3_long_office_household \
  --sens-path /path/to/scene0000_00.sens \
  --output-dir outputs/local_map_reliability/repeat
```

Optional `--baseline-root` supplies the prior refinement matrix for exact
legacy comparisons. `--protect` accepts scientific output roots to fingerprint
before execution. The default is serial execution. `--workers` permits concurrent
cases, whose timing must not be interpreted as a latency comparison.
`experiments/summarize_local_map_reliability.py --root ROOT` checks the shared
reconstruction controls and computes common-mask scores.
`experiments/verify_local_map_reliability.py --root ROOT` verifies protected
files and replays every policy's first 40 inputs from saved features with all
reference poses removed. Saved feature packets support numerical replay, while
canonical source hashes remain the raw-input identity authority.

Propagating measurement covariance is supported by prior RGB-D work, including
[Proença and Gao](https://arxiv.org/abs/1706.04034) and
[Belter, Nowicki and Skrzypczyński](https://link.springer.com/article/10.1007/s00138-018-0936-9).
Their results motivate testing sensor and processing uncertainty. They do not
validate the noise assumptions, fixed-camera approximation or experimental
results reported here.

## Reliability findings on 2026-10-20

All 40 online runs remain continuous across 2,000 distinct inputs. Weighted
RGB-D lowers one-frame translation RPE in all eight windows and position ATE in
all six TUM windows. Both ScanNet clips worsen in ATE and endpoint orientation.
This is a useful local-motion improvement, without meeting the frontend's
reliability criteria. Sparse stays default and all candidates stay experimental.

The table reports millimeters. ATE is fixed-scale SE(3)-aligned position error.
Recent denotes the unchanged recent-depth baseline. RGB-D denotes the conditional
weighted RGB-D experiment, which also changes the robust fitting objective.
Budget Sparse uses the same 1500-feature cap as every map alternative.

| Window | Budget ATE | Recent ATE | RGB-D ATE | Recent one-frame RPE | RGB-D one-frame RPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Desk 0:300 | 17.35 | 27.88 | 26.23 | 3.41 | 3.27 |
| Desk 1200:1500 | 13.05 | 15.24 | 12.27 | 2.18 | 1.89 |
| Desk 600:800 | 4.78 | 11.16 | 10.60 | 2.15 | 2.09 |
| Office 0:300 | 37.31 | 8.75 | 8.56 | 5.14 | 4.47 |
| Office 1200:1500 | 21.50 | 7.51 | 5.33 | 3.99 | 3.20 |
| Office 600:800 | 16.31 | 20.14 | 13.11 | 3.78 | 3.53 |
| ScanNet 1000:1200 | 38.39 | 62.03 | 65.29 | 12.02 | 11.48 |
| ScanNet 3500:3700 | 49.80 | 54.35 | 59.85 | 10.51 | 9.52 |

Desk 600:800 has only 105 valid reference poses and no controlled reconstruction
chunk because 11 training references are unavailable. The other primary-window
reference gaps remain as documented above. These populations are descriptive
windows from two TUM scenes and one ScanNet capture, without scene-level
significance estimates or a new independent ScanNet capture.

Downstream capped errors below use millimeters, charge missing predictions
200 mm and preserve each valid-target denominator. Surface fusion covers 124
target views and first-chunk TSDF covers 106. Every alternative shares exactly
the same evaluated targets, reference-pose targets and across-alternative common
masks. Withheld cells remain unavailable.

| Window | Recent surface | RGB-D surface | Recent TSDF | RGB-D TSDF |
| --- | ---: | ---: | ---: | ---: |
| Desk 0:300 | 53.38 | 52.11 | 34.99 | 34.70 |
| Desk 1200:1500 | 68.85 | 67.69 | withheld | withheld |
| Desk 600:800 | withheld | withheld | withheld | withheld |
| Office 0:300 | 35.97 | 36.59 | withheld | withheld |
| Office 1200:1500 | 43.47 | 44.05 | 25.38 | 25.58 |
| Office 600:800 | 61.03 | 54.17 | 53.22 | 45.96 |
| ScanNet 1000:1200 | 75.19 | 72.81 | 48.73 | 43.44 |
| ScanNet 3500:3700 | 80.56 | 82.83 | 68.57 | 71.58 |

Weighted RGB-D improves four of seven available surface comparisons and three
of five TSDF comparisons. Office's two primary surface scores worsen despite
better ATE, while the new middle window improves both representations. ScanNet
1000 improves TSDF but still trails budget Sparse's 36.04 mm. ScanNet 3500
worsens both representations. No method wins consistently across motion,
orientation, coverage and reconstruction.

Appearance history has a real but scene-dependent effect. Descriptor-family
matching improves middle-Office surface/TSDF scores from 61.03/53.22 to
53.69/45.20 mm. On ScanNet 1000 it improves surface error from 75.19 to 67.36 mm,
yet worsens TSDF from 48.73 to 66.97 mm and ATE from 62.03 to 76.91 mm.
Keeping more descriptors does not establish correct association or a better map.
The combined family-plus-weighted alternative was not tested, so additivity
is unknown.

### Interpretation and development decision

- Landmark information remains conditional on frozen camera poses. Counts of
  repeated observations do not remove shared pose error. Between 6.3% and 8.4%
  of sampled support duplicates an anchor's native depth sample. Unique samples
  can also share sensor and pose errors. The experiment does not estimate these
  correlations or calibrated posterior uncertainty.
- RGB-D consistency is a modeling limitation. Desk's reported color/depth skew
  has RMS 3.35–14.07 ms across these windows, compared with 0.21–0.42 ms for
  Office. ScanNet's stored zero clocks remain unknown. Office regressions with
  much smaller recorded skew rule out attributing every failure to this timing
  proxy alone. Calibration, acquisition latency, viewpoint-dependent sampling,
  correlated depth error and reference error remain unresolved physical causes.
- The spatial holdout changes little under covariance weighting alone. On
  ScanNet 1000, robust RGB, weighted RGB and weighted RGB-D give 1.438, 1.434 and
  1.426 px held-out RGB RMSE, versus 1.446 px for PnP. Weighted RGB-D changes
  held-out depth RMSE from 16.36 to 16.28 mm. Such conditional packet gains do
  not predict the subsequent online map history or reconstruction reliably.
- Short descriptor cycles disagree in about 0.27–0.55% of eligible cycles. This
  measures consistency only on the mutually matchable subset. Cycle-consistent
  wrong associations and missed associations remain possible. More appearance
  history helps one window and harms others, without a reliable association
  confidence model.
- The known-correspondence controls reduce translation RMSE from 2.919 to
  2.426 mm under independent noise, but hidden 6 ms depth motion increases the
  RGB-D result to 3.059 mm, above PnP's unchanged 2.919 mm. Coherent depth bias
  is not estimated. Real ten-frame depth gaps retain zero map losses versus ten
  for Sparse, black RGB gives eleven losses for every policy, and 50% biased
  depth still passes without losses. Weighted RGB-D ATE then rises from 7.11
  to 10.90 mm. Continuity does not certify measurement correctness.

The ScanNet ranking is not explained solely by the previously observed
reference jumps at multiples of ten. On 1000:1200, non-boundary one-frame
translation RPE is 8.58 mm for budget Sparse, 11.52 mm for Recent and 11.30 mm
for weighted RGB-D. These reference-relative results still do not establish
physical pose truth or generalize to another ScanNet capture.

Keep the recent-map frontend and these alternatives as reproducible controls.
Stop tuning the frozen-keyframe policy against the retained windows. Move toward
a broader SLAM architecture that preserves raw measurements and supports
revisable local poses and landmarks, explicit shared uncertainty, observation
identities for reused native samples and reconstruction reintegration. A bounded
local graph is the next useful architectural step. A global pose graph alone
cannot repair inconsistent local factors or wrong visual identities.

Before promotion, evaluate that architecture on a second ScanNet capture,
stronger rotations and wider input cadence, with independent checks of camera
geometry and exposure timing. Continue to report continuity, orientation,
local and longer-horizon motion, fixed-cohort coverage-aware reconstruction
and controlled runtime. Do not use per-scene method selection or threshold
searches on this matrix. Loop closure and relocalization should consume an
explicitly qualified frontend, rather than assume these limitations are solved.

### Preservation and reproducibility

The evidence is retained under ignored `outputs/local_map_reliability/`.
`summary.json` contains every variant and exact paired populations.
`measurement_diagnostics.json`, `controls_verified/`, `outages_verified/` and
`verification.json` retain the measurement and failure controls. The six prior
Recent runs reproduce all 12 trajectory arrays and shared numerical diagnostics
exactly. All 40 forty-input feature replays reproduce exact poses, statuses and
numerical diagnostics with reference poses removed. Five additional raw-input
pilot repeats are exact. All 806 protected prior scientific files are unchanged.
A recorded baseline-link correction resolves swapped ScanNet directory labels
using their source metadata, without rewriting trajectories or changing ranges.
The suite launcher now checks these links before execution.

The complete suite passes 849 tests. Ruff lint, formatting and whitespace checks
pass. Package source, defaults, dependencies and lockfile are unchanged. No
commit, push or context synchronization was performed. Cached and concurrent
execution supports no controlled runtime or real-time claim. Independent
ScanNet generalization, calibrated uncertainty, full-sequence evaluation of the
new candidates and native Windows replay remain unverified.
