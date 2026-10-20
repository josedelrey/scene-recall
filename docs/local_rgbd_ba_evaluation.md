# Controlled local RGB-D BA evaluation

Measured on Ubuntu on 2026-10-20. Joint landmark refinement produces small,
mixed external gains beyond matched fixed landmarks. It does **not** provide
sufficient evidence for backend promotion. Retain the opt-in experiment for
focused work on measurement error and observability. Sparse remains the default.

The [method and reproduction guide](local_rgbd_ba.md) gives the mathematical
objective, Schur solver, gauge policy, safeguards and archive contracts.
No existing backend, baseline artifact or mapping policy was replaced.

For shared landmarks across a full segment, see the
[offline batch refinement guide](batch_multiview.md).

## Protocol and available evidence

The Windows temporal packets were unavailable on this machine. Fresh packets
were generated from existing Ubuntu full Sparse trajectories and locally
available datasets. These are fresh matched Ubuntu comparisons. They are not
bitwise replays of the Windows experiments or their slightly different Sparse
initializations. All comparisons below use the same Ubuntu initializers.

The full XYZ, Desk and RPY streams contain 792, 573 and 694 observations, with
160, 116 and 141 training views. RPY's loss at frame 115 preserves two independent
segments. The existing every-fifth training schedule, offset-two target policy,
six-camera windows, four-camera advance, 256-track cap, frozen measurement
factors and uncertainty assumptions are unchanged. Only landmark freedom is
added. All committed overlap cameras stay fixed.

There are 30 supported XYZ windows, nine Desk windows and one RPY window.
Unsupported windows remain skipped for both methods. Every supported packet has
paired fixed and joint fits from exactly the same saved camera starts. Full-rate
sequential comparisons inherit each method's earlier window corrections.
Shared point initializations remain identical across methods and windows reuse
their original measured initializers. No cross-window point carry-over is added.

Primary map chunks remain XYZ `0 6`, Desk `0 4` and RPY `0 5`, covering 37, 41
and 43 evaluated target views. This retains the documented TUM populations from
the temporal experiment. Each comparison uses identical target samples, training
depth and mapping settings. The older Ubuntu fusion outputs use different
keyframe/submap policies and are not substituted for these matched populations.

Only supported known-clock TUM data supplies the primary evidence. ScanNet was
not rerun for a joint ranking. Its missing acquisition clocks remain a separate
limitation, and the runner rejects them by default. A test verifies that running
unknown clocks requires both explicit assumed-static packets and an explicit
joint-runner opt-in. TUM acquisition stamps remain proxies, with unresolved
latency, exposure midpoints and rolling shutter.

## Pose accuracy and drift

Aligned position ATE RMSE in mm, pooled by reference count across independently
aligned metric-scale segments:

| Population | Sparse | Clock only | Temporal fixed | Joint BA |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 55.994 | 55.860 | 48.142 | 47.282 |
| Desk | 76.525 | 77.065 | 75.378 | 75.513 |
| RPY | 51.456 | 51.403 | 50.499 | 49.162 |

Joint fitting improves XYZ by 0.860 mm and RPY by 1.337 mm relative to fixed
landmarks. Desk worsens by 0.135 mm. These three related Freiburg 1 sequences
and overlapping windows do not establish statistical significance or broad
scene generalization.

Root-anchored orientation RMSE in degrees, pooled by reference count:

| Population | Sparse | Clock only | Temporal fixed | Joint BA |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 1.955 | 1.967 | 2.198 | 2.298 |
| Desk | 4.793 | 5.024 | 5.111 | 5.235 |
| RPY | 13.377 | 12.997 | 12.797 | 12.580 |

Endpoint translation in mm / rotation in degrees, without global alignment:

| Segment | Sparse | Temporal fixed | Joint BA |
| --- | ---: | ---: | ---: |
| XYZ | 140.110 / 2.499 | 127.519 / 3.149 | 130.564 / 4.062 |
| Desk | 189.811 / 4.908 | 186.507 / 5.393 | 184.478 / 5.603 |
| RPY segment 0 | 154.742 / 5.480 | 156.459 / 5.552 | 156.459 / 5.552 |
| RPY segment 1 | 240.585 / 20.745 | 234.965 / 19.993 | 229.327 / 19.740 |

XYZ and Desk orientation regress despite reduced joint objectives. RPY's first
segment has no supported fit. Its clock-only change must not be credited to BA.

Longer-horizon RPE RMSE in mm / degrees, on identical eligible pairs:

| Population and interval | Sparse | Temporal fixed | Joint BA |
| --- | ---: | ---: | ---: |
| XYZ, 50 frames | 41.755 / 1.996 | 38.788 / 1.992 | 37.668 / 1.963 |
| Desk, 50 frames | 56.221 / 2.815 | 55.506 / 2.823 | 55.723 / 2.882 |
| RPY, 50 frames | 57.063 / 3.250 | 56.690 / 3.070 | 56.308 / 3.038 |
| XYZ, 10 seconds | 84.479 / 2.352 | 71.726 / 2.364 | 70.132 / 2.429 |
| Desk, 10 seconds | 163.742 / 4.650 | 162.968 / 4.684 | 163.657 / 4.972 |
| RPY, 10 seconds | 96.351 / 9.136 | 93.573 / 8.823 | 90.568 / 8.650 |

Reports retain one through 200-frame and 0.1 through 10-second RPE. Segment
losses and missing references never get bridged. These short sequences do not
establish long-run SLAM drift behavior.

## Held-out voxel and TSDF reconstruction

Primary scores are absolute depth error capped at 20 cm, in mm, with missing
predictions charged the cap over every valid target sample. Rendering uses
reference target poses after fitting. Optimized feature points do not replace
original native training depth in fusion.

| Population | Representation | Sparse | Clock only | Temporal fixed | Joint BA |
| --- | --- | ---: | ---: | ---: | ---: |
| XYZ | Voxel | 35.990 | 26.763 | 26.769 | 26.233 |
| XYZ | TSDF | 31.004 | 30.186 | 30.009 | 30.710 |
| Desk | Voxel | 31.630 | 28.220 | 28.336 | 28.165 |
| Desk | TSDF | 34.023 | 32.288 | 32.605 | 32.316 |
| RPY | Voxel | 74.018 | 70.693 | 70.693 | 70.693 |
| RPY | TSDF | 61.972 | 62.016 | 62.016 | 62.016 |

XYZ voxel improves by 0.536 mm while TSDF worsens by 0.701 mm relative to fixed.
TSDF worsens in 33 of 37 target views. On exact fixed/joint common pixels its
uncapped MAE also worsens, from 19.938 to 20.080 mm. Coverage falls from 92.080%
to 91.879%, with 5,016 lost and 4,011 gained samples. Thus coverage loss explains
part of the regression, while common-pixel geometry also worsens.

Desk's pooled scores improve by 0.171 mm for voxels and 0.289 mm for TSDF.
Common-pixel MAE improves from 30.074 to 29.914 mm for voxels and 24.831 to
24.307 mm for TSDF. Voxel coverage changes from 97.431% to 97.415%, and TSDF
from 92.583% to 92.757%. View-level gains are mixed: voxel improves in 18/41
views, TSDF in 20/41. The pooled gain is not a uniform per-view improvement.

RPY's only fit spans frames 275 through 300, outside the primary map chunks.
Constant corrections cancel in the later submap gauge. Identical primary maps
therefore supply no evidence for RPY landmark refinement. A separately labelled
post hoc chunk-2 diagnostic covers that fitting window, adding 23 target views.
Its fixed-to-joint capped scores improve from 78.804 to 76.707 mm for voxels
and 57.728 to 57.465 mm for TSDF. Voxel coverage changes from 96.188% to 96.174%,
TSDF from 90.376% to 90.417%. This diagnostic is not pooled into primary results
and cannot establish robustness beyond the single supported window.

## Optimization behavior and regression checks

Summed isolated packet costs, with exactly matched starts and factors:

| Population | Initial | Fixed final | Joint final | Maximum sequential point movement |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 4359.662 | 3850.522 | 2766.429 | 23.56 mm |
| Desk | 816.477 | 719.105 | 513.911 | 19.96 mm |
| RPY | 72.018 | 69.659 | 56.433 | 11.26 mm |

The 19–29% objective gains beyond fixed landmarks do not consistently translate
into better external geometry. Costs reuse observations across windows and are
not independent likelihood evidence. No point-displacement, camera-update or
cheirality bound rejected a real-data step. Maximum sequential Schur conditions
were approximately 3,593, 16,043 and 3,942. All 40 closed-window objectives survive
subsequent stitching unchanged. There were no supported-window rank rejections.

Eleven joint windows reach the 20-iteration cap. Separate diagnostic fits double
the budget from the same sequential starts. Across all cases, the largest pose
change is 0.0018 mm / 0.00014 degrees, and the largest relative objective decrease
is below one part per million. Incomplete numerical convergence does not explain
the observed orientation or TSDF regressions at their measured scale.

From the same sequential starts, local window endpoint translation worsens
relative to fixed fitting in 11/30 XYZ and 3/9 Desk windows. Rotation worsens in
12/30 XYZ and 4/9 Desk windows. RPY improves both in its one fit. These external
checks locate mixed effects within windows, rather than attributing everything
to tail completion. Neither these counts nor the synthetic controls identify a
unique physical cause in real data. Association errors, correlated depth,
calibration, frozen motion error and reference uncertainty remain plausible.

## Synthetic controls

Six views share 96 known identities. Factors are built from their own generated
measurements and pose initialization. Evaluation truth is withheld from fitting.
The synchronized model uses identity within-pair motion. The asynchronous model
uses training-pose interpolation and a deliberately wrong reported epoch, without
feeding exact generative motion into the deployable solver.

Independent noise uses 0.5 px RGB noise and half the nominal native covariance
standard deviation. The correlated case additionally shares a 25 mm standard
deviation range offset per landmark across all views. Stochastic rows below
average pose RMSE across five fixed noise seeds. Scene and initial pose geometry
stay fixed across those seeds. Deterministic controls repeat identically.

| Control | Fixed pose mm / degrees | Joint pose mm / degrees | Landmark consequence |
| --- | ---: | ---: | --- |
| Exact measurements and poses | 0 / 0 | 0 / 0 | No artificial movement |
| Exact measurements, perturbed poses | 2.290 / 0.0842 | <0.000001 / <0.000001 | Joint geometry recovers to numerical precision |
| Independent measurement noise | 5.001 / 0.1425 | 7.378 / 0.1580 | Landmark RMSE improves 8.935 to 6.932 mm while poses worsen |
| Correlated depth noise | 4.901 / 0.1406 | 7.327 / 0.1573 | Landmark RMSE remains 26.672 / 25.872 mm |
| Coherent +4 cm depth bias | 0.512 / 0.00538 | 1.123 / 0.01179 | Both retain about 40.5 mm point error |
| Coherent +2% depth scale | 0.330 / 0 | 0.728 / 0 | Joint cost approaches zero with about 59.8 mm point error |
| Hidden 6 ms timing error | 0.475 / 0.0218 | 0.082 / 0.00385 | Joint is better than fixed, but worsens initially exact geometry |
| Collinear geometry | Rejected | Rejected | Common undamped observability gate |
| No depth, free metric scale | Rejected | Rejected | Root fixing leaves a scale null direction |

Independent-noise regression checks preserve the same draws while changing one
source. A noise-free root reduces fixed/joint position RMSE to 4.850/4.998 mm
and rotation to 0.1209/0.1020 degrees. Pixel-only noise gives 3.311/4.297 mm,
and native-depth-only noise 3.856/4.615 mm. Root observation noise therefore
contributes materially to this counterexample, without being its sole source.
This is a finite controlled scene, not evidence that fixed geometry universally
outperforms BA under random noise. Correlated errors remain poorly represented
by the unchanged independent-block uncertainty model. Robust loss cannot repair
coherent metric bias or make an incorrect clock observable.

## Computational cost and verification

Observed seconds on an Intel i7-13700HX, Python 3.12, NumPy 2.5.3 and OpenCV 5.0.0.
OpenBLAS/OMP and OpenCV use one thread. Some tests and independent diagnostics
ran concurrently, so these are observational costs, not isolated benchmarks.

| Population | Shared preparation | Fixed numerical solver | Joint numerical solver | Four-method voxel/TSDF evaluation |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 98.74 | 0.503 | 5.253 | 81.65 |
| Desk | 67.87 | 0.129 | 1.071 | 75.16 |
| RPY | 65.36 | 0.0099 | 0.0970 | 157.69 |

Preparation includes full-stream integrity decoding, training SIFT, binding and
association. Solver columns exclude preparation, factor construction, external
pose scoring, mapping, repeats and isolated paired diagnostics. Joint solving
costs about 8–10 times the small fixed solve here. It uses a dense camera system
of at most 30 free coordinates plus independent 3×3 point blocks. There is no
new dependency, GPU requirement or production dense joint Hessian.

The retained reports under `outputs/joint_ba/` include primary runs, unchanged
fixed controls, `summary.json`, `synthetic_verified.json`, `diagnostics.json`,
`replay.json`, `raw_rpy_replay.json` and `verification.json`. The original
synthetic drafts and partial summaries remain exploratory outputs.

Tests cover analytic point derivatives, exact Schur/full-system equivalence,
shared identities, fixed cameras, missing depth, nonidentity calibration,
rigid-coordinate changes, disconnected components, scale and collinear gauges,
singular individual point blocks, update bounds, independent segments, overlap
objectives, source-free replay, tamper rejection and explicit unknown-clock
policy. Source preparation and mapping retain the existing training/held-out
boundary tests. Full-rate Sparse initialization still inherited the original
stream, as in the baseline. No new target observations enter optimization.

All real optimizers repeat poses, points and diagnostics exactly. Independent
packet replay verifies every joint array and completed full-rate trajectory.
Raw RPY replay reconstructs all graph/native arrays exactly and verifies source
identity. The full suite passed 765 tests in 119.87 seconds. After the final
singular-landmark contract test was added, all 14 focused joint tests passed.
Ruff lint, format and whitespace checks pass. A complete XYZ mapping repeat
reproduces all 132 arrays in 18 archives, pooled scores and per-view scores
exactly. All 471 protected prior scientific artifacts retain their hashes.
Trajectory timestamps, source indices, segment boundaries, statuses and
reference arrays remain exactly equal to Sparse. `verification.json` records
these checks. Remote CI and
Windows execution of the new implementation were not run. No commit was made.

## Development decision

There is evidence that shared landmark freedom can recover controlled geometry
and provide small real local improvements. There is insufficient evidence for a
reliable external improvement over the matched fixed-landmark optimizer.
Orientation and TSDF regressions, correlated/bias counterexamples and single-window
RPY support prevent promotion. Keep the modular offline implementation available.

Further work is justified only as targeted measurement-model and observability
research, with independent scenes, better timing/calibration evidence and
correlation-aware controls. Do not broaden this result into online SLAM, global
optimization or loop closure. Lower costs alone do not justify those steps.
