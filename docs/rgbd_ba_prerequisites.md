# RGB-D BA prerequisite audit

The 2026-10-19 investigation stops before implementing joint Bundle Adjustment.
The current single-pose RGB-D model omits exposure motion that is material in
the representative TUM sequences. Joint landmark freedom could absorb that
model error and produce a misleading objective improvement. Sparse stays the
default. The pose-only implementation and scientific artifacts are preserved.

The subsequent [temporal measurement and fixed-landmark control](temporal_fixed_landmarks.md)
addresses this audit with explicit assumptions and an independent experiment.
Its [evaluation](temporal_fixed_landmarks_evaluation.md) records remaining gates.
The audit results below retain their original measurement model and meaning.

## Reproduce the diagnostics

Use the completed training-only experiments in
[the pose-only evaluation](pose_only_evaluation.md).

```bash
uv run --locked python experiments/rgbd_ba_prerequisites.py \
  outputs/pose_only/final/xyz outputs/pose_only/final/desk \
  outputs/pose_only/final/rpy outputs/pose_only/final/scannet_start \
  outputs/pose_only/final/scannet_motion \
  --verify-source --repeat-cached --output-dir outputs/ba_prerequisites/new_audit
```

The destination must be fresh. `--verify-source` replays the canonical source
fingerprint, rebuilds associations from training images only, and compares all
27 measurement arrays exactly. Without it, the audit uses cached measurements
and checks saved window factor fingerprints. This weaker verification mode is
identified in the report. `--repeat-cached` repeats all diagnostics and isolated
pose-only fits, requiring exact scientific results after excluding elapsed times
and the raw-replay verification flag.

The report includes input and implementation hashes, environment, exposure
sensitivity, same-population reference evaluations, information spectra, window
objective changes, isolated fits and synthetic controls. Preparation, raw replay,
existing pose-only solves and other diagnostic costs are reported separately.
It exports no trajectory backend or map and adds no dependency.

`scene_recall.multiview.audit` contains numeric exposure and information checks.
The experiment handles archives, source replay and external evaluation.

## Formulation and coordinate checks

`T_AB` maps B to A. `T_SR = T_SC_D inv(T_RD)` maps RGB optical coordinates into
the estimated segment. Each diagnostic window is expressed relative to its RGB
root, fixed exactly. Independent trajectory segments stay separate. External
relative depth-camera references are conjugated by `T_RD` for comparison to RGB
motions. Tests cover common-world gauge changes and finite-difference
information derivatives.

The current anchored model predicts a track from one fixed measured anchor:

\[
q_{aj}=T_j^{-1}T_a b(u_a,z_a),\qquad
r_{aj}=[\pi(q_{aj})-u_j,\ q_{aj,z}-z_j].
\]

Propagated anchor/target covariance is frozen at Sparse initialization. Depth
sigma is the engineering model `0.005 + 0.003 z²` meters and pixel sigma is one.
Anchor noise is correlated across star factors. Existing per-factor whitening
and track weighting omit these cross-factor correlations.

For conditioning only, the audit inspects a possible independent-observation
RGB-D model with camera poses and shared points:

\[
q_{ij}=T_i^{-1}P_j,\qquad
r_{ij}=[\pi(q_{ij})-u_{ij},\ q_{ij,z}-z_{ij}].
\]

Points use coordinate medians of measured, Sparse-transformed depth samples
inside each window. Missing depth has zero depth-row information. Rows use the
sigmas above, Cauchy IRLS weights at initialization, and total track weight one.
The audit eliminates each point information block analytically:

\[
S=H_{pp}-\sum_j H_{pL_j}H_{L_jL_j}^{-1}H_{L_jp}.
\]

It reports diagonal-normalized spectra before damping, point-block conditions
and explicit exclusions. No pose or landmark update is computed from this joint
model. These are local working-curvature diagnostics, without calibrated
covariance or convergence claims.

This direct model includes the anchor observation. The anchored model omits its
zero self-factor and propagates anchor uncertainty. The existing frozen-point
control also retains propagated covariance. Future BA needs an additional
fixed-landmark control with exactly the joint objective's observations,
whitening, weighting, loss and window schedule. Comparing only existing anchored
fitting with direct-observation BA would confound landmark freedom with a
changed objective. The report records both block counts.

## Exposure timing is a substantial unresolved prerequisite

TUM RGB and registered depth have separate acquisition times. SceneRecall keeps
both in provenance, while canonical observations and references use depth time.
Feature pixels come from color and their depth is sampled on the registered
depth grid. One pose cannot generally describe both exposures during motion.

References are used only to predict static-point displacement between the two
times. No association, point estimate or pose is corrected. This measures a
motion budget without proving that a depth sampled at a color keypoint belongs
to the same physical surface at both exposures. Samples below include all
finite depth-supported SIFT rows in training views after the existing depth-edge
filter, including unused rows, where both reference epochs are available.

| Training case | Views | Median absolute skew, ms | Median predicted displacement, px | Samples above 1 px |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 160 | 5.90 | 1.06 | 51.2% |
| Desk | 116 | 11.67 | 2.26 | 77.2% |
| RPY | 141 | 11.28 | 3.34 | 85.9% |
| ScanNet start | 25 | unavailable | unavailable | unavailable |
| ScanNet later motion | 25 | unavailable | unavailable | unavailable |

Only 48/160 XYZ, 8/116 Desk and 12/141 RPY training views have absolute skew at
most 3 ms. Clock thresholds alone do not bound motion error. ScanNet's two clips
lack usable exposure clocks, preventing a synchronization claim.

On identical reference-valid source populations, changing only the diagnostic
reference epoch changes Sparse one-frame rotation RMSE from 0.300 to 0.332
degrees on XYZ, 0.602 to 0.501 on Desk, and 0.754 to 0.437 on RPY. RPY retains
the same 685 pairs. Anchored fitting gives 0.754 versus 0.436 degrees on that
RPY population. The epoch effect exceeds the local difference between these
optimizers. This sensitivity test leaves official depth-epoch evaluation intact.

Reference-pose transfers on the selected factors still have several-pixel
errors at the color epoch. Timing is consequential, without being the sole
established cause. Calibration approximation, depth association, reference
uncertainty, localization and scene motion remain possible contributors.

[TUM's format documentation](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
describes spatial pre-registration and recommends ROS-default pinhole parameters
for these images. Spatial registration does not remove exposure motion. The
provided depth-scale correction is already applied. This audit introduces no
additional correction or calibration change.

## Synthetic controls and depth uncertainty

The new synchronized/asynchronous control uses 96 exact feature identities on a
static slanted plane, exact calibration and initially exact color-epoch poses.
Depth comes from depth-time ray-plane intersections. There is no sensor noise,
association failure or reference input to the solver.

The synchronized control stays at numerical zero. Alternating ±12 ms skew lowers
the unchanged anchored objective from 0.6992 to 0.5177 while position RMSE rises
from zero to 3.04 mm and rotation RMSE to 0.0653 degrees. A color timestamp does
not correct the depth sample. This isolates a mechanism that could also bias a
joint fit, without quantifying its contribution to real sequence errors.

The prior coherent anchor-depth-bias control remains reproducible:

```bash
uv run --locked python experiments/pose_only_bias_probe.py \
  --output-dir outputs/ba_prerequisites/new_depth_bias_control
```

A 4 cm anchor-depth bias lowers cost from 60.294 to 13.000 while position RMSE
worsens from 1.66 to 12.79 mm. Cauchy loss does not remove coherent bias. Uniform
depth-scale bias is also not identifiable from these measurements alone:
scaling points and camera translations preserves projections and fits scaled
depths. Fixing the root pose supplies no independent metric reference.

The report includes signed depth transfers, their ratios to frozen propagated
depth sigma, and absolute per-view median depth errors. These are conditional
on VO-selected tracks and external references. Factors repeat across overlaps
and share anchors. Gating truncates their distribution. These dependent samples
cannot calibrate sensor noise or establish independent error bars.

On identical reference-valid transfer populations, the following depth-epoch
results summarize the fixed selected factors. Z-scores use the frozen propagated
depth variance, without estimating or updating that variance from references.

| Case | Median pixel transfer, px | Depth transfer RMSE, mm | Depth z-score RMSE | Absolute z-score above 3 |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 2.28 | 12.35 | 0.899 | 0.82% |
| Desk | 2.52 | 12.31 | 0.959 | 0.86% |
| RPY | 4.04 | 19.36 | 1.240 | 2.96% |
| ScanNet start | 2.04 | 14.45 | 0.549 | 0.05% |
| ScanNet later motion | 3.11 | 18.00 | 0.534 | 0.00% |

These depth statistics do not establish a universal depth-weight failure.
Color-epoch transfers on RPY reduce median pixel error to 3.15 px, while depth
RMSE remains 19.35 mm. Replacing the reference timestamp alone leaves substantial
geometric disagreement.

## Geometry, initialization and stitching

All 91 supported windows have full-rank RGB-D pose Schur information after fixing
the root. Maximum normalized conditions are 21,616 for XYZ, 29,341 for Desk,
13,874 for RPY, 12,469 for ScanNet start and 9,868 for later motion. Multiplying
depth sigma by four preserves rank with overall maximum 44,127. Removing depth
leaves one null direction in every window, consistent with the monocular scale
gauge. This supports bounded numerical experiments once the measurement issue
is addressed, without validating uncertainty assumptions.

Information checks use measured window geometry and Sparse alone, without using
the saved full-track point or external reference points. The report retains
supported track length, baseline and parallax distributions. Coordinate medians
depend on axis choice and VO-based association selects geometry compatible with
the initializer. Future joint experiments need initialization sensitivity checks
and must avoid treating that median as an independent prior.

| Training graph | Supported tracks | Median length | Median baseline, m | Median parallax, degrees |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 10,585 | 5 | 0.151 | 4.84 |
| Desk | 5,157 | 4 | 0.194 | 9.13 |
| RPY | 5,020 | 4 | 0.051 | 1.70 |
| ScanNet start | 520 | 4 | 0.163 | 3.49 |
| ScanNet later motion | 811 | 4 | 0.325 | 6.92 |

These are complete training-graph distributions, rather than only selected
window points. RPY has substantially less translational parallax than Desk.

The audit evaluates each unchanged objective on the final stitched trajectory
and independently re-runs the pose-only solver with each window rooted at its
original Sparse pose.

| Case | Earlier window costs raised by later overlap | Increase in summed closed-window cost |
| --- | ---: | ---: |
| XYZ | 34/37 | 0.96% |
| Desk | 17/22 | 1.10% |
| RPY | 17/22 | 1.71% |
| ScanNet start | 5/6 | 1.10% |
| ScanNet later motion | 3/4 | 1.05% |

The worst individual RPY increase is 18.16%. These sums reuse observations and
are not a global likelihood. Overlap damage alone does not establish worse
final poses than Sparse. On RPY's 22 supported windows, depth-epoch endpoint
position RMSE is 36.08 mm for Sparse, 33.22 mm for isolated fits and 35.24 mm for
stitched relative motions. Rotation is 2.059, 2.078 and 2.091 degrees. Report
local fitting and sequential transport separately. Any changed stitching policy
must also apply to the matched fixed-landmark control.
Isolated fits also reset window initialization, so this comparison does not
isolate transport alone.

## Measured cost and verification

Native Windows, Python 3.12.13, NumPy 2.5.3 and OpenCV 5.0.0 were used. OpenCV
uses one thread. The final audited artifact is
`outputs/ba_prerequisites/final/report.json`. All five cases replayed exactly,
covering 2,299 raw source observations and 467 training views. Every cached
scientific repeat was exact. All 127 protected reports, trajectories, training
archives, voxel/TSDF maps and target prediction archives retained their hashes.
The implementation hash includes package Python sources and the audit entry
point. Baseline source identities remain those of the original experiments.

| Case | Cached preparation, s | Raw replay and track preparation, s | Isolated pose-only solves, s | Other diagnostics, s |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 4.26 | 92.36 | 0.522 | 12.43 |
| Desk | 3.20 | 52.26 | 0.238 | 6.40 |
| RPY | 3.48 | 45.00 | 0.245 | 6.51 |
| ScanNet start | 0.42 | 9.57 | 0.053 | 0.42 |
| ScanNet later motion | 0.36 | 9.87 | 0.042 | 0.52 |

Cached preparation includes integrity hashing of scientific inputs. Other
diagnostics include information construction, source timestamp/reference-table
reads, external evaluation and final integrity checks. These wall-clock costs
are observational. A regression suite ran concurrently during part of replay,
so the values are not isolated performance benchmarks or comparable speedups.
They exclude original Sparse initialization and existing map-generation costs.
The repeat uses another cached audit pass, without repeating raw preparation.

All 15 new prerequisite tests pass. They cover exposure/gauge geometry,
finite-difference Schur information, scale ambiguity, missing depth, coherent
timing bias, nonidentity calibration, factor identity, reference poisoning,
source replay and artifact tampering. The earlier focused run had 70 passes
across this audit, multiview tracking and pose-only suites, before the final two
tests were added. The full run had 728 passes, two skips and two existing
Windows publication failures with `WinError 5`. Both failing tests passed on
focused retry. This is not a clean full-suite result. Publication reliability
remains an operational limitation. Ruff lint, formatting and diff whitespace
checks pass. Remote CI was not run.

## External evaluation and decision

No new trajectory is promoted or fused in this stopped milestone. Sparse,
anchored, frozen and linear retain their full trajectory evaluation, independent
segments and identical mapping cohorts. The first/last submap cohort contains
167 held-out targets across five cases. Voxel and TSDF artifacts are protected
by input hashes. Existing longer-horizon, endpoint, orientation, coverage-aware
and common-pixel results remain in [the pose-only evaluation](pose_only_evaluation.md).

ScanNet later-motion voxel capped error worsens from 39.00 to 41.07 mm while
TSDF improves from 46.46 to 44.75 mm. XYZ endpoint orientation worsens from 2.492
to 4.105 degrees despite better aligned position ATE. These regressions remain
unresolved. The present diagnostics do not establish timing as their sole cause,
especially on ScanNet without usable clocks.

Before joint fitting:

1. Define explicit color/depth epochs and a reference-free temporal policy.
   Preserve native depth support and visibility when transporting it to color
   time. Account for estimated-motion uncertainty, with no GT correction.
   Alternatively establish a sufficiently synchronized, low-motion cohort using
   clocks and estimated motion alone.
2. Validate uncertainty and association on synchronized, asynchronous,
   missing-depth and coherent-bias controls. Do not calibrate noise from the
   already gated residual population alone.
3. Freeze observations, objective, loss, weights and windows for matched fixed
   and free landmark runs. Retain current Sparse and anchored alternatives.
   Report isolated and stitched results.
4. Repeat external trajectory and held-out voxel/TSDF evaluation with the same
   target IDs, native samples, missing-coverage penalties and common pixels.
   Report regressions by case and view.

Joint BA remains deferred. The tracks appear numerically usable, but the current
evidence cannot support attributing a gain specifically to refinement of uncertain
landmarks under this unresolved observation model.
