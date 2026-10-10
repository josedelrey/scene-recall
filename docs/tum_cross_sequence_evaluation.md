# Cross-sequence RGB-D odometry evaluation

October 10, 2026. Full `freiburg1_rpy` and `freiburg1_desk` runs use the
existing Sparse, Depth ICP, and Hybrid defaults. The comparison reuses the
serial runner, canonical TUM adapter, reference evaluator, and version 2
exports. Backend source files, settings, convergence gates, and failure
policies remain unchanged. No commit was created.

## Evaluation design

Evaluate every associated observation in both new sequences. Reuse the saved
full `freiburg1_xyz` experiment and the five 200-frame `scene0000_00` ScanNet
clips as historical context. Keep their individual reference populations and
reference provenance. ScanNet clip aggregates weight squared errors by pair
count. Do not pool ScanNet reconstruction references with TUM motion capture.

The official [TUM sequence descriptions](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
describe `rpy` as rotations around the three principal axes at an approximately
fixed position, and `desk` as sweeps over four office desks. Measured motion
from associated ground-truth poses characterizes the actual evaluated samples.
These sequences change motion and viewing conditions together, so they do not
isolate a single causal factor.

Association, calibration, and scoring follow [TUM support](tum.md). RGB/depth
matching is unique within 20 ms. Ground truth uses bounded translation
interpolation and quaternion SLERP at depth timestamps with a 50 ms bracket
tolerance. Missing references exclude accuracy samples without removing
observations or influencing estimation. Depth uses PNG values divided by 5000
and the registered-image ROS-default pinhole calibration.

All estimator runs execute serially with one OpenCV, OpenBLAS, and OMP thread.
The original runner orders Sparse, ICP, then Hybrid for each full sequence.
There is one full-run timing sample per backend and sequence. Stage timing,
latency distributions, and repeat clips provide descriptive checks. They do
not establish real-time performance or confidence intervals for runtime.
Report inspection and analysis ran intermittently during the full benchmark.
No estimator runs overlapped. Cache, system load, and this background work
limit interpretation of small timing differences.

The primary local metrics are translation and rotation RPE at 1, 5, and 10
frames, and 0.1, 0.5, and 1.0 seconds. Time endpoints use a 20 ms tolerance.
Matched three-backend metrics require valid references and continuity in every
backend. Supplementary Sparse/Hybrid populations reveal how ICP fragmentation
changes the comparison population. Additional 2 s and 5 s horizons assess
drift, with unavailable or very small populations reported explicitly.

Every loss starts an independent local segment. Segment duration, size,
coverage, and lost indices accompany accuracy. Position ATE fits a proper
rigid alignment with fixed metric scale. Anchored drift and endpoint errors
use the first valid reference pose once, without further fitting. Fragmented
runs have no full-sequence ATE or endpoint. Segment metrics and plots remain
available, and reference anchoring of plotted segments does not stitch them.

Hybrid diagnostics compare accepted Sparse initialization against successful
refinements on exactly the same reference-valid pairs. Fallbacks remain in
tracking statistics and are excluded from refinement improvements. Common
held-out depth samples diagnose the optimized objective. They share sensor
data with fitting and do not independently validate motion.

An exploratory timing diagnostic evaluates the same accepted poses against
references interpolated at RGB timestamps, retaining both valid clocks on the
same population. The primary scores remain depth-timed. Sparse combines RGB
bearings with depth-time source geometry, so changing evaluation timestamps
alone cannot correct the underlying measurement model. Neither these scores
nor reference-based diagnostics alter any estimate or configuration.

## Associated input populations

These statistics describe the evaluated associated observations, whose time
span can differ from the download page's nominal recording duration. Motion
uses adjacent reference-valid poses, without smoothing or crossing missing
references. Path length is the sampled polyline length over those valid pairs.

| Sequence | Raw RGB / depth | Associated frames | Valid references | Duration s | Median / max absolute skew ms | Mean translation m/s | Mean rotation °/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xyz | 798 / 798 | 792 | 789 | 26.594 | 7.320 / 17.230 | 0.302 | 17.130 |
| rpy | 723 / 722 | 694 | 690 | 24.061 | 11.698 / 19.930 | 0.092 | 64.761 |
| desk | 613 / 595 | 573 | 573 | 19.817 | 11.648 / 19.742 | 0.468 | 33.927 |

Association leaves 29 RGB and 28 depth images unmatched on `rpy`, and 40 RGB
and 22 depth images on `desk`. Maximum depth intervals are 70.350 ms and
67.933 ms, respectively. Gaps and skew remain in the input. Missing `rpy`
references at associated indices 441, 442, 443, and 445 remain tracked and
exclude only reference-dependent metrics. Both new sequences' final frames
have valid references. Sampled path lengths are 2.205 m on `rpy`, 9.304 m on
`desk`, and 8.003 m on `xyz`.

The unchanged Python package fingerprint is identical between new TUM runs
and the historical XYZ runs. Backend settings are checked for equality across
all three TUM sequences and the historical ScanNet clips. ScanNet's saved
runner fingerprint predates TUM integration. Its exact estimate reproduction
through the shared runner was established in the
[previous evaluation](tum_odometry_evaluation.md). Historical timing samples
are reused and retain their cache and system-load limitations.

## Matched accuracy and tracking results

All values below use identical reference-valid, uninterrupted pairs within
each sequence or ScanNet clip. RPE values are translation RMSE in millimeters
and rotation RMSE in degrees. Coverage counts all attempted transitions,
including those without references. Historical ScanNet aggregates cover five
200-frame clips, rather than a continuous 1000-frame trajectory.

| Sample | Backend | Coverage | Segments | RPE 1 mm / ° | RPE 10 mm / ° | Frames/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TUM xyz | sparse | 100.00% | 1 | 5.299 / 0.301 | 17.168 / 0.864 | 4.24 |
| TUM xyz | icp | 96.21% | 31 | 2.778 / 0.447 | 13.069 / 1.283 | 1.76 |
| TUM xyz | hybrid | 100.00% | 1 | 2.785 / 0.445 | 12.977 / 1.288 | 1.31 |
| TUM rpy | sparse | 99.86% | 2 | 8.135 / 0.753 | 23.338 / 1.470 | 4.98 |
| TUM rpy | icp | 93.22% | 48 | 5.223 / 0.638 | 27.328 / 2.273 | 1.69 |
| TUM rpy | hybrid | 99.86% | 2 | 5.261 / 0.635 | 27.275 / 2.261 | 1.16 |
| TUM desk | sparse | 100.00% | 1 | 9.586 / 0.609 | 26.364 / 1.537 | 3.99 |
| TUM desk | icp | 94.06% | 35 | 6.044 / 0.737 | 30.559 / 2.288 | 1.54 |
| TUM desk | hybrid | 100.00% | 1 | 6.080 / 0.750 | 28.257 / 2.052 | 1.08 |
| ScanNet 5 clips | sparse | 100.00% | 5 | 8.621 / 0.222 | 23.721 / 0.580 | 3.43 |
| ScanNet 5 clips | icp | 99.20% | 13 | 9.841 / 0.253 | 52.799 / 1.574 | 2.09 |
| ScanNet 5 clips | hybrid | 100.00% | 5 | 9.298 / 0.253 | 51.163 / 1.568 | 1.33 |

Common populations at intervals 1, 5, and 10 contain 757/644/527 pairs for
XYZ, 639/499/371 for RPY, 538/430/319 for Desk, and 987/937/886 across ScanNet
clips. The common populations retain 54.2% of possible RPY and 56.7% of Desk
ten-frame intervals. ICP fragmentation accounts for most exclusions, alongside
reference gaps and the RPY Sparse loss. These conditional errors must accompany
continuity.

![Cross-sequence local accuracy, tracking failures, and throughput](/home/rlyeh/repos/scene-recall/outputs/odometry/tum_cross_sequence_analysis/cross_sequence_overview.png)

Hybrid lowers one-frame translation RMSE on all three TUM sequences. Its
ten-frame translation advantage on XYZ does not persist on RPY or Desk, where
it is respectively 16.9% and 7.2% worse than Sparse on common pairs. ScanNet's
earlier 115.7% increase remains a stronger version of that adverse long-horizon
behavior. Rotation is generally worse with depth refinement. The lower RPY
one-frame rotation RMSE is sensitive to outliers and image timing, as detailed
below. Sequence, motion, horizon, and metric all affect the observed ranking.

## Time horizons and comparison populations

The following time RPE table uses the three-backend common population. Values
are translation millimeters / rotation degrees, both RMSE.

| Sequence | Horizon s | Common pairs | Sparse | ICP | Hybrid |
| --- | ---: | ---: | ---: | ---: | ---: |
| rpy | 0.1 | 552 | 12.228 / 0.886 | 11.621 / 1.077 | 11.644 / 1.071 |
| rpy | 0.5 | 289 | 28.109 / 1.736 | 34.758 / 2.797 | 34.483 / 2.774 |
| rpy | 1.0 | 109 | 36.715 / 2.232 | 50.170 / 4.105 | 49.145 / 4.054 |
| desk | 0.1 | 473 | 14.337 / 0.863 | 11.486 / 1.115 | 11.055 / 1.153 |
| desk | 0.5 | 251 | 30.534 / 1.747 | 38.008 / 2.695 | 35.273 / 2.331 |
| desk | 1.0 | 137 | 46.794 / 2.338 | 49.877 / 3.942 | 43.036 / 3.021 |

Sparse/Hybrid-only ten-frame comparisons retain 666 RPY pairs and 563 Desk
pairs, rather than 371 and 319. RPY translation errors are 26.934 vs 36.241 mm,
and rotation 1.515 vs 2.426°. Desk translation errors are 24.810 vs 26.593 mm,
and rotation 1.389 vs 1.928°. The broader population preserves the adverse
ten-frame trade-off, while changing its magnitude. Three-backend common errors
alone would understate the RPY Hybrid disadvantage on the broader population.

At 2 s, only nine RPY pairs remain continuous in all three backends. At 5 s,
none remain. Sparse/Hybrid retain 547 and 393 reference-valid pairs, respectively.
Their 2 s errors are 57.547/3.331 vs 77.647/4.960, and 5 s errors are
70.163/5.817 vs 89.316/7.096. These are useful longer-horizon diagnostics for
the same two methods, with no supported long-horizon ICP ranking.

Desk has 59 three-backend common pairs at 2 s and none at 5 s. Sparse/Hybrid
retain 493 and 413 pairs. Their 2 s errors are 60.219/2.957 vs 76.709/3.995,
and 5 s errors are 113.278/4.179 vs 141.442/5.983. Hybrid has larger relative
errors at these longer horizons despite its better full-trajectory position
ATE and anchored endpoint translation. Absolute and relative measures describe
different aspects of the trajectory, and their ranking need not agree.

## Full-trajectory translation and rotation drift

Only XYZ and Desk have continuous Sparse and Hybrid trajectories. The following
table uses all valid references in those runs. Translation values are millimeters.

| Sequence | Backend | Position ATE | Anchored translation RMSE | Endpoint translation / rotation |
| --- | --- | ---: | ---: | ---: |
| xyz | sparse | 55.994 | 101.797 | 140.110 / 2.499° |
| xyz | hybrid | 25.514 | 43.161 | 64.746 / 3.403° |
| desk | sparse | 76.525 | 140.942 | 189.811 / 4.908° |
| desk | hybrid | 61.071 | 75.902 | 104.457 / 8.522° |

On Desk, Hybrid reduces position ATE by 20.2%, anchored translation RMSE by
46.1%, and endpoint translation by 45.0%. Endpoint rotation increases by 73.6%.
This resembles XYZ's accumulated translation improvement with a rotational
cost, despite different intermediate-horizon behavior. Position ATE has no
rotation score. Better fitted positions cannot establish orientation quality
for integrating 3D surfaces.

![Estimated trajectories and segment-anchored drift](/home/rlyeh/repos/scene-recall/outputs/odometry/tum_cross_sequence_analysis/trajectories_and_drift.png)

## Rotation-dominated motion and drift

Sparse and Hybrid lose only target 115 on `rpy`, producing identical segment
boundaries. The rejected Sparse estimate has 23 correspondences and 12 final
inliers, below the fixed minimum of 15. Source and target RGB have only 75 and
129 SIFT keypoints. The images are visibly blurred, with 5.043 degrees of
reference rotation in 36.100 ms. The recorded failure is insufficient final
inliers. Blur and fast rotation are plausible contributors, rather than
separately established causes. Hybrid requires Sparse success, so depth
refinement does not run on this transition.
Independent ICP accepts that same pair with 3.142 mm translation and 0.292°
rotation error. The present Hybrid policy therefore does not exploit every
case of complementary sensor robustness.

These two runs have no full-sequence ATE or endpoint. Their largest segment
contains 579 frames and lasts 20.221 s. Identical segment boundaries support
the following direct comparisons. All translation values are millimeters.

| Segment | Backend | Anchored translation RMSE | Position ATE | Endpoint translation / rotation |
| --- | --- | ---: | ---: | ---: |
| 0–114, 115 frames | sparse | 86.007 | 57.055 | 154.742 / 5.480° |
| 0–114, 115 frames | hybrid | 104.641 | 67.902 | 119.501 / 9.766° |
| 115–693, 579 frames | sparse | 189.343 | 50.262 | 240.585 / 20.745° |
| 115–693, 579 frames | hybrid | 91.097 | 62.923 | 94.054 / 13.611° |

Hybrid improves the long segment's anchored translation and endpoint errors,
while increasing its best-fit position ATE. Its short-segment endpoint rotation
is worse. Sparse's median common one-frame rotation error is 0.314°, compared
with Hybrid's 0.446°, even though Hybrid has lower one-frame rotation RMSE.
The RMSE ranking reflects suppression of large outliers and does not mean
rotation improves on most pairs. Only 219/644 successful reference-valid
refinements improve rotation, while 430/644 improve translation.

The RGB-time sensitivity check is especially relevant on `rpy`. Over the same
685 accepted Sparse pairs, rotation RMSE changes from 0.754° at depth times to
0.437° at RGB times. ICP changes from 0.638° to 0.923°, and Hybrid from 0.650°
to 0.897° on their respective identical-clock populations. Relative reference
motion at the two clocks differs by 0.666° rotation RMSE, with a 3.353° maximum.
At target 477, the depth interval is 65.504 ms and reference rotation is 7.215°.
RGB/depth skew switches from approximately +14.9 ms to −14.9 ms, and Sparse
reports a 3.664° rotation error despite 398 inliers. This is evidence that
association timing materially affects local scoring during fast rotation.

Accumulated rotation error remains substantial under either clock. Sparse's
long-segment endpoint changes from 20.745° to 20.165° in the timing diagnostic.
Hybrid changes from 13.611° to 13.336°. Timing alone therefore does not account
for that drift. Pinhole calibration approximation, rolling shutter, depth bias,
scene geometry, and objective bias remain possible contributors. This
experiment does not separate their causal roles.

## ICP convergence and tracking losses

On `rpy`, ICP loses 47/693 transitions and creates 48 segments. Its longest
segment has 83 frames over 2.736 s, and median segment duration is 0.232 s.
Every recorded loss is a finest-scale iteration-limit failure after 35
iterations. All final candidates have rank six, with at least 34.556% support.
Losses occur throughout the sequence and in every angular-motion bin, including
11/111 reference-valid pairs below 1° per transition and 24/375 at or above 2°.
The data do not support explaining these losses solely by fast camera motion.

Retained candidates are useful for diagnosis, and remain rejected. Many have
small reference errors, but the candidate at 340→341 is wrong by 143.053 mm and
1.834°. Its held-out depth RMSE still falls from 19.450 to 13.803 mm over 5663
common points, with rank six and 34.556% support. Numerical rank, support, and
residual reduction do not establish correct physical motion. These outcomes
do not justify accepting every iteration-limit candidate.

Hybrid refines 650 `rpy` transitions, falls back to Sparse on 42 ICP iteration
limits, and does not run ICP at the one Sparse loss. All 644 reference-valid
successful refinements reduce common held-out depth RMSE. Rotation worsens on
425 of those pairs and translation on 214. The complete accepted Sparse
initialization matches standalone Sparse exactly. Fallback preserves continuity
when depth refinement fails, while successful optimization can still worsen
reference accuracy and accumulate drift.

Initialization also changes the convergence population. Of the 47 independent
ICP losses on `rpy`, Hybrid successfully refines 15 and falls back on 32.
Hybrid additionally falls back on 10 transitions accepted by independent ICP.
The different initial poses affect projective associations and the path to
convergence. Sparse initialization provides no universal convergence guarantee.

Desk ICP loses 34/572 transitions and creates 35 segments, with a longest
segment of 119 frames over 3.941 s. All losses are finest-scale iteration
limits with rank-six final candidates. Median support is 90.412% on losses and
91.593% on accepted pairs. Median fitting residuals are 3.533 and 3.244 mm,
respectively. The worst rejected Desk candidate has 12.679 mm translation and
1.481° rotation error. Support and residual alone do not separate the observed
convergence losses. Per-iteration traces are unavailable, so slow convergence
and association oscillation cannot be distinguished here.

## Desk refinement successes and large errors

Hybrid refines 538 Desk transitions and retains Sparse on 34 refinement
failures, consisting of 33 iteration limits and one excessive correction.
Translation improves on 356/538 successful refinements and rotation on 143/538.
All 538 reduce common held-out depth RMSE, including 182 with worse translation
and 395 with worse rotation. On this same successfully refined population,
translation RMSE falls from 9.527 to 5.986 mm, while rotation rises from 0.585°
to 0.731°.

At 142→143, Sparse has 430 inliers and a 0.969 px reprojection RMSE. Hybrid
reduces held-out depth RMSE from 13.686 to 6.823 mm over 11,198 common points,
but increases reference translation error from 1.705 to 23.744 mm and rotation
from 0.298° to 2.145°. The accepted ICP solution has rank six, 75.0% support,
and condition 6.76. This illustrates accepted objective disagreement, beyond
tracking failures and low-support estimates.

At 167→168, the 5° Hybrid correction bound rejects a 5.601° correction and
keeps Sparse's 22.614 mm / 4.011° error. The converged candidate is better
against the reference at 16.444 mm / 2.093°. The bound is a reference-independent
heuristic and sometimes discards a helpful correction. RGB/depth skew changes
from −13.7 to +17.3 ms across this pair, complicating interpretation of the
large Sparse rotation error. These two examples do not justify removing or
tuning the bound on these evaluation sequences.

On Desk's identical accepted populations, the RGB-time sensitivity check
changes Sparse rotation RMSE from 0.602° to 0.501°, ICP from 0.737° to 0.906°,
and Hybrid from 0.737° to 0.852°. Clock choice affects the numbers but preserves
Sparse's lower rotation error on this sequence. Frame gaps above 50 ms occur
22 times on Desk without a tracking loss in any backend, and 28 times on RPY
with two independent ICP losses. Gaps alone do not explain most losses.

![Camera motion, detected tracking losses, and refinement outcomes](/home/rlyeh/repos/scene-recall/outputs/odometry/tum_cross_sequence_analysis/motion_and_refinement.png)

## Computational performance

The new full-run host is an Intel Core i7-13700HX, using Python 3.12.3,
NumPy 2.5.3, and OpenCV 5.0.0. Each estimator uses the existing single-thread
configuration. Historical runs record software settings but lack a saved CPU
model, so small cross-date timing differences remain uncertain.

Sparse is fastest on every sequence. On RPY, Hybrid takes 599.810 s compared
with Sparse's 139.471 s and independent ICP's 410.099 s. On Desk, the times are
528.326, 143.724, and 371.837 s. Hybrid takes 4.30 times Sparse's wall time on
RPY and 3.68 times on Desk. The effective associated input rate is approximately
29 frames/s. Processing at 1.08–4.98 frames/s does not meet that input rate.
The benchmark supports offline evaluation, with no demonstrated real-time
operation.

Stage latency is measured per invocation, including unsuccessful invocations.
Hybrid's ICP stage runs only after Sparse acceptance. The following values are
median / 95th percentile milliseconds. Standalone Sparse has no exported
per-pair stage timer, so its end-to-end throughput is the supported measurement.

| Sequence | Independent ICP | Hybrid Sparse stage | Hybrid ICP stage |
| --- | ---: | ---: | ---: |
| xyz | 526.6 / 745.6 | 223.1 / 293.0 | 514.4 / 684.5 |
| rpy | 561.2 / 792.3 | 213.9 / 307.0 | 622.8 / 866.5 |
| desk | 616.9 / 841.3 | 245.2 / 371.2 | 647.8 / 869.8 |

Depth refinement accounts for roughly three quarters of Hybrid's measured
stage time on the new sequences. Better Sparse initialization does not make
Hybrid's ICP stage uniformly faster than independent ICP. These single-run
latencies describe the current implementation and workload, with the timing
limitations stated above. Memory use and accelerator performance were not measured.

## Development decision and remaining uncertainty

This completes the initial odometry evaluation milestone. The measurements
support retaining Sparse as the default: it preserves all Desk and XYZ
transitions, loses only one RPY transition, is consistently faster, and has
stronger multi-step rotational accuracy than the depth-based alternatives.
Keep Hybrid selectable for its measured translation gains on XYZ and Desk.
Its rotational cost, occasional harmful accepted refinements, and larger
long-horizon relative errors warrant explicit map-level validation.
Independent ICP remains useful as an experimental estimator and geometric
diagnostic, with segment fragmentation visible in downstream products.

The next development step should be **multiframe 3D mapping with keyframes and
local submaps**, using the unchanged Sparse trajectory as the baseline and
Hybrid as an alternative in the same mapping pipeline:

1. Fuse canonical metric depth into bounded voxel or surfel submaps using
   estimated camera-to-submap poses. Preserve frame provenance and segment IDs.
2. Start a new submap at every tracking loss. Preserve the discontinuity and
   add explicit relocalization or submap alignment later. Reference poses must
   remain diagnostic and never stitch estimated segments.
3. Validate fusion on XYZ and Desk, then use RPY to expose orientation-driven
   double surfaces and segment handling. Compare maps made from Sparse and
   Hybrid with a reference-pose fusion diagnostic on identical depth inputs.
   Such a diagnostic isolates pose effects while sharing depth and calibration.
4. Measure surface consistency, duplicate geometry, coverage, integration cost,
   and behavior at failure boundaries. Add multiframe constraints when map
   quality demonstrates their value. Defer backend tuning and broader sequence
   collection until a concrete mapping failure motivates them.

The motion/scene changes are informative but confounded. Three Freiburg 1
sequences share one sensor family and office environment, while ScanNet covers
only one capture. ScanNet references come from reconstruction and are coupled
to scene geometry. TUM references are independent motion capture. This matrix
does not isolate calibration, sensor depth bias, scene structure, rolling
shutter, synchronization, or reference differences as the cause of a
cross-dataset ranking change. It provides descriptive sequence results, with
no broad generalization claim or independent-sample statistical significance.

The registered-image pinhole approximation remains unchanged. RGB/depth
association does not make exposures simultaneous. Finest-scale iteration
traces and calibrated pose uncertainty are unavailable. A low residual, full
rank, or high support remains insufficient evidence of correct motion.
Mapping should expose these limits through provenance, conservative fusion,
and submap boundaries. Further odometry optimization is not required to begin
that next architectural milestone.

## Verification

All 479 tests pass, including four new diagnostic tests for shared continuity
populations, failure denominators, candidate rotation statistics, clock
sensitivity, transform direction, reference preservation, and legacy ScanNet
report compatibility. Lint, formatting, and whitespace checks pass.

All 12 repeated runs match the corresponding full-run numerical diagnostics
after excluding stage runtimes. Six prefix runs match all 12 archive arrays
exactly. The six later clips reproduce exact pair estimates, references,
provenance, and failure boundaries. Rebased estimated trajectories differ by
at most 6.67 × 10⁻¹⁶. No reference alignment enters this repeat verification.
The source manifest hashes 2659 files, including all 2653 source images and
six timestamp/reference tables. Backend configurations and package fingerprints
match between full runs and repeats. TUM package identity also matches the
historical XYZ evaluation.

Figures were rendered with Matplotlib 3.11.2 and inspected visually. The
analysis verifies saved report hashes, fingerprints trajectory archives,
records script identity, and checks backend configuration equality across
datasets. No backend, project dependency, or lockfile was changed. No commit
was created.

## Reproduction and artifacts

Run the full benchmark into a fresh directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/scene-recall-uv-cache \
uv run --offline python experiments/compare_odometry_backends.py \
  --tum-path /home/rlyeh/datasets/tum/rgbd_dataset_freiburg1_rpy \
             /home/rlyeh/datasets/tum/rgbd_dataset_freiburg1_desk \
  --ranges all --output-dir outputs/odometry/tum_cross_sequence_repeat
```

The measured new full runs are in `outputs/odometry/tum_cross_sequence/`.
Each backend exports `report.json`, `trajectory.npz`, and `frames.csv`.
`comparison.json` preserves input fingerprints, settings, common metrics,
refinement outcomes, failures, timing, and report hashes. Original XYZ and
ScanNet outputs are preserved.

Analyze saved results without rerunning estimators:

```bash
UV_CACHE_DIR=/tmp/scene-recall-uv-cache uv run --offline python \
  experiments/analyze_odometry_sequences.py \
  outputs/odometry/tum_freiburg1_xyz \
  outputs/odometry/tum_cross_sequence \
  outputs/odometry/backend_comparison \
  --output-dir outputs/odometry/tum_cross_sequence_analysis
```

Add `--plot` using optional Matplotlib 3.11.2, as in the previous experiment:

```bash
MPLCONFIGDIR=/tmp/scene-recall-mpl uv run --with matplotlib==3.11.2 python \
  experiments/analyze_odometry_sequences.py \
  outputs/odometry/tum_freiburg1_xyz \
  outputs/odometry/tum_cross_sequence \
  outputs/odometry/backend_comparison \
  --output-dir outputs/odometry/tum_cross_sequence_analysis --plot
```

`analysis.json` records input report hashes, script identity, per-pair
diagnostics, matched Sparse/Hybrid cohorts, extended horizons, motion bins,
tracking gaps, runtime distributions, and timing sensitivity. `summary.csv`
provides sequence-level comparison rows. PNG and SVG figures show matched
errors, tracking failures, throughput, trajectories, anchored drift, and
refinement trade-offs. Generated outputs remain ignored by Git. The scripts
and this report are versionable repository files.

Repeat prefixes and later clips, including the natural Sparse failure at `rpy`
target 115 and ICP failure at `desk` target 117:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/scene-recall-uv-cache \
uv run --offline python experiments/compare_odometry_backends.py \
  --tum-path /home/rlyeh/datasets/tum/rgbd_dataset_freiburg1_rpy \
             /home/rlyeh/datasets/tum/rgbd_dataset_freiburg1_desk \
  --ranges 0:20 100:125 \
  --output-dir outputs/odometry/tum_cross_sequence_repeats

UV_CACHE_DIR=/tmp/scene-recall-uv-cache uv run --offline python \
  experiments/verify_odometry_repeats.py \
  --full-dir outputs/odometry/tum_cross_sequence \
  --repeat-dir outputs/odometry/tum_cross_sequence_repeats --hash-sources
```

The verifier checks configurations, calibration, software fingerprints,
per-frame provenance, stage diagnostics excluding runtime, segment boundaries,
and all prefix archive arrays. Later clips compare exact pair estimates and
trajectory coordinates after rebasing from estimated poses only. Reference
poses do not enter that coordinate change. `reproducibility.json` records
each check, and `source_manifest.json` hashes the original tables and every
RGB/depth PNG named in them.
