# Pose-only multiview investigation

Measured on 2026-10-19 on native Windows with Python 3.12.13, NumPy 2.5.3 and
OpenCV 5.0.0. The implementation and uncertainty assumptions are described in
[the pose-only guide](pose_only_multiview.md).

Joint local pose fitting improves aligned position ATE in all five evaluated
cases. Orientation, long-horizon drift and reference-target reconstruction are
mixed. Keep Sparse as the default. The evidence supports further bounded
experiments, with explicit depth/timing and window-stitching checks before a
joint RGB-D BA comparison. It does not establish a general reconstruction gain.

## Frozen protocol

Fresh default Sparse baselines cover complete TUM Freiburg 1 XYZ, Desk and RPY
sequences. Existing Sparse runs cover ScanNet `scene0000_00` ranges `[0,120)` and
`[3500,3620)`. Together these contain 2,299 source observations. The new stage
uses 467 segment-relative training views at stride 5 plus segment endpoints.
Targets use offset 2 and never enter track construction, residuals or fusion.
The inherited full-rate Sparse initializer already consumed the original stream.

The primary formulation uses depth-anchored shared tracks, propagated pixel/depth
uncertainty, Cauchy loss, six-camera windows, step 4, at most 256 tracks, fixed
window roots, support/coverage checks and a normalized condition-number guard.
Parameters are unchanged across cases. Controls use frozen initial window
landmarks or the same anchored factors with linear loss. GT is used only for
source verification and subsequent evaluation.

Optimization uses the full training trajectory. Mapping evaluates the first and
last temporally populated submap, selected by the split rather than by scores.
Each submap contains at most 24 training views. All variants use the same fused
observations, filtering, 2 cm voxels and stride-4 depth lattice. Surface voxels use
1 cm metric discs. TSDF uses 6 cm truncation and connected mesh rendering.

| Case | Source frames | Training views | Refined windows / attempted | Evaluated map chunks | Held-out views |
| --- | ---: | ---: | --- | --- | ---: |
| XYZ | 792 | 160 | 37/40 | 0, 6 | 37 |
| Desk | 573 | 116 | 22/29 | 0, 4 | 41 |
| RPY | 694 | 141 | 22/35 | 0, 5 | 43 |
| ScanNet start | 120 | 25 | 6/6 | 0 | 23 |
| ScanNet later motion | 120 | 25 | 4/6 | 0 | 23 |

There are 167 evaluated target views. Comparisons retain identical target samples
inside each case. First/last submaps are a bounded reconstruction cohort, without
a claim that all intermediate reconstruction has been evaluated. RPY retains
its loss at source frame 115 and its independent 115/579-frame segments.

## Position accuracy and controls

ATE uses proper SE(3) position alignment with metric scale fixed. RPY pools
squared errors after independent segment alignments. No alignment is used by
the optimizer or to join segments.

| Case | Sparse ATE, mm | Anchored Cauchy, mm | Frozen landmarks, mm | Anchored linear, mm |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 55.93 | 49.35 | 55.77 | 50.79 |
| Desk | 76.51 | 62.44 | 76.30 | 62.81 |
| RPY | 51.46 | 50.05 | 50.92 | 49.95 |
| ScanNet start | 23.08 | 19.38 | 22.88 | 19.26 |
| ScanNet later motion | 33.59 | 28.92 | 28.85 | 28.60 |

The primary ATE reductions range from 2.7% to 18.4%. Frozen landmarks retain much
of the initial trajectory bias on XYZ, Desk and ScanNet-start. Robust and linear
fits are often close on the already conservative tracks. Cauchy is useful for
corrupted synthetic measurements, without being uniformly superior on real data.

## Local error, longer horizons and orientation

All entries below compare Sparse to anchored Cauchy on identical reference-valid,
within-segment pairs. Units are millimeters and degrees. Full populations and
additional intervals are retained in the reports.

| Case | RPE 10 frames translation | RPE 10 frames rotation | RPE 50 frames translation | RPE 50 frames rotation |
| --- | --- | --- | --- | --- |
| XYZ | 15.85 -> 14.53 | 0.793 -> 0.730 | 41.72 -> 35.02 | 1.995 -> 1.727 |
| Desk | 24.81 -> 23.23 | 1.389 -> 1.323 | 56.22 -> 45.24 | 2.815 -> 2.366 |
| RPY | 26.93 -> 26.63 | 1.515 -> 1.510 | 57.06 -> 57.98 | 3.250 -> 3.259 |
| ScanNet start | 22.76 -> 21.31 | 0.631 -> 0.633 | 43.52 -> 33.88 | 1.138 -> 1.056 |
| ScanNet later motion | 24.29 -> 23.07 | 0.495 -> 0.476 | 59.54 -> 53.20 | 0.878 -> 0.809 |

At 200 frames, translation/rotation RMSE is XYZ 63.35/2.103 -> 54.62/2.018,
Desk 141.70/4.597 -> 111.92/4.133, and RPY 85.47/7.436 -> 86.98/7.247.
There are 589/373/371 pairs respectively. ScanNet clips are shorter than this
interval and lack usable timestamps, so no time-based result is invented.

| TUM case | 10 s translation RMSE, mm | 10 s rotation RMSE, degrees | Pairs |
| --- | --- | --- | ---: |
| XYZ | 84.37 -> 74.02 | 2.341 -> 2.527 | 489 |
| Desk | 163.72 -> 125.74 | 4.650 -> 4.397 | 273 |
| RPY | 96.35 -> 97.07 | 9.136 -> 8.870 | 277 |

Endpoint drift exposes trade-offs hidden by aligned position ATE:

| Case or segment | Endpoint translation, mm | Endpoint rotation, degrees |
| --- | --- | --- |
| XYZ | 140.08 -> 129.10 | 2.492 -> 4.105 |
| Desk | 189.79 -> 152.84 | 4.906 -> 4.404 |
| RPY first segment | 154.74 -> 154.74 | 5.480 -> 5.480 |
| RPY second segment | 240.59 -> 246.18 | 20.745 -> 20.507 |
| ScanNet start | 94.98 -> 72.07 | 2.460 -> 2.091 |
| ScanNet later motion | 109.91 -> 101.44 | 1.165 -> 1.226 |

XYZ improves local and intermediate rotation RPE while worsening 10 s and
endpoint orientation. RPY improves aligned position ATE while worsening some
long-horizon translation and endpoint position. Position ATE alone therefore
cannot establish a better trajectory.

## Reference-target reconstruction

Primary rendering uses GT target camera poses in each map's first-training-camera
gauge. It does not correct the refined map. Common submap placement error cancels
in this gauge, so these local geometry scores complement global drift metrics.

The 20 cm capped loss below averages `min(abs(depth_error), 0.2)` over every valid
target sample, charging 0.2 m for a missing prediction. Values are millimeters.
Lower values alone still need coverage, explained fraction and paired errors.

| Case | Surface-voxel capped loss | TSDF capped loss | Surface-voxel coverage, % | TSDF coverage, % |
| --- | --- | --- | --- | --- |
| XYZ | 35.99 -> 31.50 | 31.00 -> 30.77 | 97.88 -> 97.76 | 92.03 -> 92.05 |
| Desk | 31.63 -> 28.71 | 34.03 -> 34.02 | 97.20 -> 97.04 | 91.98 -> 92.24 |
| RPY | 74.02 -> 71.23 | 61.97 -> 61.24 | 97.94 -> 97.90 | 91.38 -> 91.31 |
| ScanNet start | 33.39 -> 26.12 | 24.50 -> 21.59 | 98.61 -> 98.38 | 95.52 -> 95.67 |
| ScanNet later motion | 39.00 -> 41.07 | 46.46 -> 44.75 | 97.34 -> 97.27 | 95.26 -> 95.41 |

| Case | Surface-voxel explained within 5 cm, % | TSDF explained within 5 cm, % |
| --- | --- | --- |
| XYZ | 82.19 -> 83.77 | 86.62 -> 86.82 |
| Desk | 84.52 -> 86.85 | 84.34 -> 84.73 |
| RPY | 53.02 -> 58.25 | 63.45 -> 63.82 |
| ScanNet start | 80.18 -> 90.31 | 91.12 -> 94.07 |
| ScanNet later motion | 75.89 -> 73.69 | 66.92 -> 69.58 |

ScanNet-start provides the clearest cross-representation gain: all 23 voxel
target views improve capped loss, as do 18/23 TSDF targets. XYZ and Desk improve
voxel capped loss on 33/37 and 31/41 views. Their TSDF result is much weaker:
XYZ's pooled capped gain is only 0.24 mm and its median paired view gets slightly
worse. Desk's pooled capped change is effectively zero.

Desk TSDF illustrates a coverage/accuracy trade-off. Its untrimmed depth MAE
worsens from 25.93 to 26.85 mm, and exact common-pixel MAE worsens from 25.18 to
26.13 mm, while coverage and explained fraction rise. This is not a supported
general geometry improvement.

Later ScanNet motion improves aligned trajectory ATE but worsens surface voxels:
18/23 target views have worse capped loss. Exact common-pixel MAE rises from
40.11 to 42.37 mm on 342,964 shared samples. The regression therefore survives
coverage control. TSDF improves common-pixel MAE from 43.21 to 41.80 mm on
337,048 samples using the same poses and training observations. Fusion behavior
interacts with pose changes, without a universal representation-independent gain.

RPY's first segment is unchanged because every primary window there fails the
support/coverage gate. Its 23 evaluated first-chunk targets consequently remain
identical. Gains occur in the 20 late-chunk targets. The aggregate result must
not be interpreted as repairing the difficult first segment or the tracking loss.

## Negative-result investigations

The post-hoc radius check keeps later-ScanNet voxel maps, target depths and
reference cameras fixed. At disc radii 5/10/20 mm, Sparse/refined 20 cm capped
losses are 108.20/109.79, 39.00/41.07 and 42.19/45.87 mm. The regression persists
across all three radii. This rules out a single chosen radius or coverage-only
explanation. Voxel assignment, visibility, pose orientation, scene changes and
reference/calibration errors remain possible contributors whose effects have
not been causally separated.

The controlled static-scene probes isolate two different mechanisms:

- With a common 4 cm initial pose bias, anchored fitting reduces translation
  RMSE from 36.37 mm to effectively zero under exact measurements. Frozen
  landmarks reduce their own objective from 11.81 to 11.23 while retaining
  35.92 mm pose error. Initial landmark consistency can reinforce pose bias.
- With a 4 cm bias in the central anchor's measured depth, anchored fitting
  reduces its objective from 60.29 to 13.00 while translation RMSE worsens from
  1.66 to 12.79 mm. Frozen fitting also worsens poses. Robust loss cannot remove
  a coherent measurement bias that camera motion can explain.

These probes establish possible failure mechanisms, without assigning those
causes to the real sequences. Similar real-data behavior under linear and
Cauchy losses suggests that robust-loss choice alone does not explain the
orientation/fusion regressions. Calibration/timing approximations and sequential
window stitching remain hypotheses requiring separate controls.

### Exploratory shorter-window RPY schedule

After observing the skipped RPY windows, a separately labeled diagnostic uses
three-camera windows and step 2, with all other settings unchanged. All 27
training-measurement arrays match the primary run exactly. The same first/last
mapping chunks and target populations are retained. This diagnostic is not used
to retune the primary method.

It refines 53/69 windows, including four in the first segment, compared with
22/35 and none there under the primary schedule. Pooled aligned position ATE
is almost unchanged: 50.03 versus 50.05 mm. The long-segment endpoint improves
to 223.45 mm/19.44 degrees, compared with the primary 246.18 mm/20.51 degrees.
However, 200-frame translation RPE remains worse than Sparse: 86.28 versus
85.47 mm.

Reference-target voxel capped loss is 71.96 mm, between Sparse's 74.02 and the
primary 71.23. TSDF capped loss worsens to 64.73 mm from Sparse's 61.97 and the
primary 61.24, with explained-within-5-cm falling to 61.21% from 63.45% Sparse.
The first chunk's TSDF loss rises from 76.87 to 81.37 mm. The last chunk also
worsens, from 44.93 to 45.69 mm. More refined windows and better endpoint drift
therefore do not establish better reconstruction. Window schedule, anchor choice
and repeated-factor effects remain important prerequisites for joint BA.

Reproduce this exploratory control with `--window-size 3 --window-step 2
--no-controls --evaluate-maps --map-chunks 0 5` on the full RPY baseline.

## Cost, robustness and repeatability

| Case | Verified replay and training tracks, s | Primary window fitting, s | Four-trajectory map evaluation, s |
| --- | ---: | ---: | ---: |
| XYZ | 68.5 | 1.37 | 63.8 |
| Desk | 38.8 | 0.57 | 61.1 |
| RPY | 40.2 | 0.64 | 116.2 |
| ScanNet start | 8.3 | 0.10 | 51.4 |
| ScanNet later motion | 9.8 | 0.10 | 87.1 |

These single-run timings use one OpenCV/OpenBLAS/OMP thread. Host load was not
isolated. Preparation includes decoding, full source verification, training-only
SIFT and association. Map cost includes Sparse and all three refinements, both
representations, both reference/estimated target modes, diagnostics and exports.
It is not the cost of a single map. Sparse initialization is additional and
retained separately in each trajectory report.

All 25 skipped windows fail support or coverage. No real-data window triggers
the numerical rank/condition rejection. The worst accepted normalized condition
number is approximately 40,690. One primary Desk window reaches the 20-iteration limit.
Accepted refinement indicates a cost decrease, without certifying convergence.
RPY's original loss and independent segment gauges are preserved.

Cached pose repeats are exact for every case. Independent ScanNet-motion source
replay reproduces all 129 saved scientific arrays across training measurements,
three refined trajectories, voxel maps, TSDF values/meshes and target predictions.
Baseline report and trajectory hashes remain unchanged. All within-case target
populations match. Package source hash is
`5fe64cd5ad4d37f88cb0f2ecccd0309715a65742778ebc50a7bed1c356e6285c`.

The 24 new tests pass, including Jacobians, gauge transformations, depth holes,
degenerate geometry, outliers, bias counterexamples, lost segments, interpolation,
held-out data poisoning, source integrity, fixed populations and failed publication.
The full-suite attempt reports 713 passed, two symbolic-link privilege skips and
one previously observed Windows `WinError 5` in an existing odometry-publication
test. That case passes its targeted retry. The host issue remains unresolved.
The additional new cases are covered by the final focused run. Linux and remote
CI were not run. Lint and formatting are checked separately.

## Reproduction and retained evidence

Primary runs live under `outputs/pose_only/final/{xyz,desk,rpy,scannet_start,scannet_motion}/`.
Their `report.json` files retain configurations, splits, pose errors and window
diagnostics. `maps/report.json` and NPZ files retain reconstruction populations
and predictions. `outputs/pose_only/summary.json` and `verification.json` provide
derived summaries and repeat evidence. These ignored artifacts do not travel
with the repository.

Create each full TUM baseline with `scripts/rgbd_odometry.py --tum-path ...`
and no stop bound. Then use the pose-only command in the implementation guide.
Pass `--map-chunks 0 6` for XYZ, `0 4` for Desk, `0 5` for RPY and `0` for each
ScanNet clip. ScanNet source ranges are selected during baseline odometry.

```bash
uv run --locked python experiments/summarize_pose_only.py \
  --runs outputs/pose_only/final/xyz outputs/pose_only/final/desk \
         outputs/pose_only/final/rpy outputs/pose_only/final/scannet_start \
         outputs/pose_only/final/scannet_motion \
  --output outputs/pose_only/new_summary.json

uv run --locked python experiments/pose_only_render_sensitivity.py \
  --experiment outputs/pose_only/final/scannet_motion \
  --output-dir outputs/pose_only/new_radius_check
```

## Decision about joint RGB-D BA

The tracks contain usable multiview information. Bounded pose-only fitting
improves translation and some independently scored local reconstruction. The
evidence supports a small controlled joint-BA comparison after these prerequisites:

1. Establish a defensible depth/timing/calibration uncertainty policy. Hard depth
   anchors can cause coherent pose bias, and adding landmark variables alone
   does not establish identifiability or eliminate bad depth measurements.
2. Check sequential window stitching and orientation drift explicitly. Retain
   rooted segment gauges, conditioning checks and correction bounds. Never
   select an optimizer using aligned position ATE alone.
3. Preserve training-only associations and fixed map cohorts. Evaluate both
   surface voxels and TSDF at reference target poses, including coverage and
   missing predictions. Keep estimated-target consistency separate.
4. Compare joint BA against both Sparse and this fixed pose-only method using
   the same tracks. Treat reassociation, new priors and changed keyframe policies
   as separately measured interventions.

Sparse remains the default. No full joint landmark BA, loop closure, global pose
graph, relocalization or SLAM frontend was introduced in this milestone.
