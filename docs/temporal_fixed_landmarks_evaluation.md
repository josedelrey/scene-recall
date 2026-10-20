# Temporal RGB-D and matched fixed-landmark evaluation

Measured locally on 2026-10-19/20. This experiment implements a bounded,
reference-free two-exposure model and an independent fixed-landmark pose backend.
Joint landmark optimization was outside this milestone. The subsequent
[controlled joint experiment](local_rgbd_ba_evaluation.md) reuses this measurement
model. Sparse and the earlier anchor-star pose-only experiment remain unchanged.

The [method and reproduction guide](temporal_fixed_landmarks.md) specifies the
residuals, calibration, uncertainty assumptions, window policy and archive
contracts. Position improvements alone do not validate the timing model.

Summarize the seven saved primary controls without modifying them:

```bash
uv run --locked python experiments/summarize_temporal.py \
  outputs/temporal_fixed/conservative/xyz \
  outputs/temporal_fixed/conservative/desk \
  outputs/temporal_fixed/conservative/rpy \
  outputs/temporal_fixed/conservative/scannet_start_assumed \
  outputs/temporal_fixed/conservative/scannet_motion_assumed \
  outputs/temporal_fixed/conservative/scannet_start_strict \
  outputs/temporal_fixed/conservative/scannet_motion_strict \
  --legacy-root outputs/pose_only/final \
  --output outputs/temporal_fixed/new_summary.json
```

## Protocol and retained artifacts

Primary outputs are under `outputs/temporal_fixed/conservative/`:

- Full TUM XYZ, Desk and RPY streams contain 792, 573 and 694 observations.
- ScanNet `scene0000_00` uses `[0,120)` and `[3500,3620)`.
- There are 2,299 distinct source observations and 467 training views across
  these five populations. RPY retains its loss at frame 115 and two independent
  segments. No window crosses that loss.
- Every fifth observation trains, with the existing offset-two targets.
  Primary map chunks are XYZ `0 6`, Desk `0 4`, RPY `0 5`, and ScanNet `0`.
  The same 167 target views and exact target samples are retained from the
  previous experiment. Sparse voxel and TSDF predictions reproduce exactly.
- TUM acquisition clocks are available. Both ScanNet clips have unavailable
  source clocks. Their `*_strict` runs reject depth initialization and preserve
  Sparse exactly. `*_assumed` runs explicitly assume coincident acquisition.

Four new alternatives share the selected training graph and native bindings:

| Alternative | Interpretation |
| --- | --- |
| `clock` | Frozen estimated RGB-to-depth epoch conversion, without optimization |
| `static` | Direct RGB/native-depth objective with static extrinsic and static point initialization |
| `timed` | Direct objective with frozen estimated exposure motion and committed overlap cameras fixed |
| `open` | Timed objective with only the window's first camera fixed |

`static` is conditional on the timed graph and bindings. Its geometry initializer
uses the static transform. It does not independently rebuild a static graph.
The old anchored method has a different objective and supplies historical
context rather than isolating landmark freedom.

All new fitting uses training images, native depth and estimated training motion.
Full-stream decoding checks input integrity. Reference poses and decoded target
measurements do not construct factors or determine acceptance. The inherited
Sparse initializer previously consumed the original full-rate stream.

## Synthetic measurement controls

`experiments/temporal_controls.py` generates known feature identities, varying
translation and rotation, alternating 12 ms exposure offsets, nonidentity
calibration, missing/weak support, coherent depth bias and hidden clock error.
Each deployable variant rebuilds its own point initializer using its assumed
motion model and supplied initial poses. Evaluation truth never initializes
those points. A separately labelled `known_motion_control` uses generative motion
only to check the ideal mathematical model.

Representative pose RMSE after fitting, in mm / degrees:

| Control | Static motion | Estimated interpolated motion | Interpretation |
| --- | ---: | ---: | --- |
| Synchronized, exact initialization | 0 / 0 | 0 / 0 | No artificial update |
| Asynchronous, exact initialization | 0.389 / 0.0460 | 0.0176 / 0.00189 | Timing bias strongly reduced, interpolation remains approximate |
| Asynchronous, noisy initialization | 4.549 / 0.0950 | 1.754 / 0.0778 | Estimated bridge helps this controlled case |
| Nonidentity depth-to-RGB calibration, exact initialization | 0.460 / 0.0453 | 0.0179 / 0.00189 | Calibration composition is consistent |
| Coherent +4 cm axial depth bias | 0.678 / 0.0455 | 0.502 / 0.00450 | Low pose error can coexist with biased shared geometry |

With asynchronous exact initialization, the ideal known-motion control retains
zero error. Injecting an unmodelled 6 ms clock error into the estimated bridge
raises pose error to 0.475 mm / 0.0218 degrees and point initialization error to
3.25 mm. The coherent depth-bias case retains about 40.5 mm of landmark error
despite small pose errors. Timing compensation does not calibrate depth bias.
The weak collinear configuration is rejected before a pose update.

These controls demonstrate consistency and expose failure mechanisms. They
are small, ideal feature-identity experiments without real visibility changes.
They do not calibrate physical sensor noise or prove real exposure midpoint
accuracy. `controls_verified.json` repeats the corrected reference-free controls
exactly. Earlier exploratory control files are superseded.

## Support, conditioning and overlap

The primary native binding uses calibrated RGB/depth sampling footprints,
actual native ray identities, unique samples per view and an extra motion
uncertainty boundary margin. Merely inflating depth covariance cannot repair a
sample associated with the wrong surface.

| Population | Supported tracks | Native bindings | Timed refined / attempted windows |
| --- | ---: | ---: | ---: |
| XYZ | 6,144 | 78,748 | 30 / 40 |
| Desk | 2,023 | 24,524 | 9 / 29 |
| RPY | 754 | 12,550 | 1 / 35 |
| ScanNet start, assumed static | 426 | 8,828 | 6 / 6 |
| ScanNet motion, assumed static | 684 | 12,357 | 3 / 6 |

Remaining primary windows fail existing track support or spatial-coverage gates.
The admitted common joint-model Schur systems have no detected null directions,
with maximum normalized condition about 17,401, below the fixed `1e8` guard.
This is a local algebraic check at Sparse initialization, without a calibrated
uncertainty interpretation. Tests demonstrate that an unanchored component can
pass a fixed-point pose check and fail the common free-point model check.

Desk rejects depth in four training views because a motion interval is unsupported
and in two because estimated within-pair motion exceeds the bound. RPY rejects
one training pair for excessive motion. Output completion uses twelve explicit
identity fallbacks on Desk, while retaining known estimated transforms for pairs
rejected only by the depth quality gate. Those fallbacks remain a limitation.

With first-camera-only boundaries, 35 of 49 closed objectives increase after
later windows. With committed overlap cameras fixed, all 49 remain unchanged.
Open-policy ATE differs only slightly here, but earlier objective improvement
cannot be claimed to survive stitching under that policy. Locking boundaries
is conservative and forgoes corrections to previously committed poses. It is
not marginalization or a globally consistent solution.

The RPY `--motion-margin 0` diagnostic admits 21 windows. Its timed ATE is
52.679 mm, worse than clock-only 51.404 mm and the conservative 50.499 mm.
More constraints and accepted residual decreases do not establish better poses.
The margin was retained as a geometric safety policy, without fitting its value
to held-out scores. This ablation changes association and support as intended,
so it does not isolate only covariance weighting.

## Trajectory accuracy

ATE is the existing per-segment rigidly aligned position RMSE, pooled by sample
count. Values are mm. ScanNet rows require the explicit static timing assumption.

| Population | Sparse | Old anchored | Clock only | Static direct | Timed locked | Timed open |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XYZ | 55.931 | 49.346 | 55.797 | 48.114 | 47.993 | 47.880 |
| Desk | 76.513 | 62.439 | 77.053 | 80.852 | 75.289 | 75.314 |
| RPY | 51.456 | 50.052 | 51.404 | 45.079 | 50.499 | 50.499 |
| ScanNet start | 23.076 | 19.384 | 23.076 | 20.772 | 20.772 | 21.038 |
| ScanNet motion | 33.594 | 28.918 | 33.594 | 33.992 | 33.992 | 33.944 |

The static direct RPY result has lower aligned ATE than the temporal control.
It does not overturn the synthetic inconsistency of the static measurement
model. Initial landmark bias, fixed geometry, epoch conversion and trajectory
error can cancel. This experiment cannot physically attribute that cancellation.
No backend or timing parameter was selected using this ranking.

Endpoint translation / rotation, mm / degrees, in each segment's rooted gauge:

| Population | Sparse | Clock only | Timed locked |
| --- | ---: | ---: | ---: |
| XYZ | 140.075 / 2.492 | 144.320 / 2.544 | 126.776 / 3.086 |
| Desk | 189.787 / 4.906 | 188.736 / 5.156 | 186.133 / 5.360 |
| RPY segment 0 | 154.742 / 5.480 | 156.459 / 5.552 | 156.459 / 5.552 |
| RPY segment 1 | 240.586 / 20.745 | 239.766 / 20.257 | 234.966 / 19.993 |
| ScanNet start | 94.980 / 2.460 | 94.980 / 2.460 | 95.839 / 2.570 |
| ScanNet motion | 109.914 / 1.166 | 109.914 / 1.166 | 112.827 / 1.244 |

XYZ and Desk retain endpoint orientation regressions. ScanNet start improves
aligned position while worsening its endpoint. The later-motion clip regresses.

Longer-horizon RPE, mm / degrees:

| Population and interval | Sparse | Clock only | Timed locked |
| --- | ---: | ---: | ---: |
| XYZ, 50 frames | 41.724 / 1.995 | 41.038 / 2.042 | 38.858 / 1.995 |
| Desk, 50 frames | 56.219 / 2.815 | 56.192 / 2.832 | 55.491 / 2.822 |
| RPY, 50 frames | 57.063 / 3.250 | 57.024 / 3.099 | 56.690 / 3.071 |
| ScanNet start, 50 frames | 43.516 / 1.138 | 43.516 / 1.138 | 39.367 / 1.074 |
| ScanNet motion, 50 frames | 59.540 / 0.878 | 59.540 / 0.878 | 59.990 / 0.893 |
| XYZ, 10 seconds | 84.371 / 2.341 | 84.110 / 2.362 | 71.512 / 2.355 |
| Desk, 10 seconds | 163.723 / 4.650 | 163.757 / 4.561 | 162.887 / 4.680 |
| RPY, 10 seconds | 96.352 / 9.136 | 95.914 / 8.983 | 93.573 / 8.823 |

At one frame, clock conversion changes RPY rotation RMSE from 0.754 to 0.491
degrees, with essentially no further optimizer change. XYZ translation changes
5.267 to 5.101 to 5.113 mm. Desk changes 9.487 to 8.663 to 8.674 mm.
Time-based ScanNet RPE is unavailable with these clocks. The clips are only
120 frames, so they do not establish long-run drift behavior.

## Held-out reconstruction

The primary score is absolute depth error capped at 20 cm, with missing
predictions charged the cap over all valid observed target samples. Values are
mm. Rendering uses reference target poses after fitting. Voxel/TSDF parameters,
training observations and target populations are unchanged.

| Population | Representation | Sparse | Clock only | Timed locked |
| --- | --- | ---: | ---: | ---: |
| XYZ | Voxel | 35.991 | 26.763 | 26.779 |
| XYZ | TSDF | 31.004 | 30.186 | 30.042 |
| Desk | Voxel | 31.635 | 28.219 | 28.366 |
| Desk | TSDF | 34.028 | 32.283 | 32.593 |
| RPY | Voxel | 74.018 | 70.692 | 70.692 |
| RPY | TSDF | 61.972 | 62.016 | 62.016 |
| ScanNet start | Voxel | 33.387 | 33.387 | 30.582 |
| ScanNet start | TSDF | 24.499 | 24.499 | 23.598 |
| ScanNet motion | Voxel | 39.001 | 39.001 | 39.554 |
| ScanNet motion | TSDF | 46.458 | 46.458 | 47.267 |

Most apparent TUM mapping gains relative to Sparse come from epoch completion.
The optimizer slightly harms XYZ voxel and both Desk scores relative to clock
only. Exact common-pixel Desk MAE also worsens: voxel 29.930 to 30.155 mm,
TSDF 24.491 to 24.892 mm. Desk TSDF worsens in 38 of 41 target views.
ScanNet motion common-pixel MAE worsens 40.259 to 40.826 mm for voxels and
43.288 to 44.365 mm for TSDF. Coverage changes alone cannot explain these cases.
The observations do not identify a unique physical cause.

Timed coverage ranges from 97.24% to 98.47% for voxels and 90.82% to 95.51%
for TSDF across these cases. Largest clock-versus-Sparse coverage change is
RPY TSDF, 91.38% to 90.82%. Reports also retain explained fractions, uncapped
errors, gained/lost samples and exact common-pixel comparisons.

The conservative RPY fit is only frames 275 through 300. Primary chunks `0 5`
do not cover that window. A constant correction in the later local submap
cancels under its own map gauge, explaining identical timed/clock primary maps.
A separately labelled, post hoc training-support diagnostic evaluates chunk `2`,
which covers the fitted window. It adds 23 views without changing the primary
167-view cohort. Voxel cap score is Sparse 79.560, clock 79.968 and timed
78.804 mm. TSDF is 62.183, 57.834 and 57.728 mm. Clock/timed coverage is
96.220/96.188% voxel and 90.402/90.376% TSDF. This small local effect does not
establish broader RPY robustness and must not be pooled into the primary ranking.

## Costs and verification

Seconds on the local Windows environment. Preparation includes full-source
integrity decoding, training SIFT, native binding and association. These wall
times are observational, with some concurrent validation, without isolated
hardware benchmarking or speedup claims.

| Population | Preparation | SIFT | Native binding | Association | Timed factor preparation | Timed information checks | Timed linear solver |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XYZ | 119.17 | 15.92 | 36.08 | 44.23 | 4.72 | 0.341 | 0.862 |
| Desk | 68.18 | 9.41 | 23.63 | 20.00 | 1.05 | 0.072 | 0.175 |
| RPY | 71.94 | 10.31 | 29.23 | 14.71 | 0.17 | 0.005 | 0.015 |
| ScanNet start | 15.40 | 5.69 | 4.99 | 1.52 | 0.38 | 0.026 | 0.093 |
| ScanNet motion | 15.86 | 5.67 | 4.68 | 2.42 | 0.45 | 0.032 | 0.082 |

Complete timed fitting, including factor preparation and checks, costs
5.99/1.31/0.19/0.50/0.57 seconds respectively. Those costs exclude original
Sparse initialization, pose evaluation and mapping. Primary mapping of Sparse,
clock and timed with both representations costs about 76/77/138/60/91 seconds.
No GPU or new dependency is required.

Validation at this revision:

- Full test suite: **750 passed, 2 skipped**, 180.92 seconds.
- Eighteen new tests cover clocks, calibration, native identities, foreground
  visibility, motion uncertainty, camera and Schur derivatives, weak geometry,
  overlap invariants, independent segments, target/reference isolation,
  deterministic archive replay and tamper rejection.
- All seven primary runs repeat optimizer outputs, factors and numerical
  diagnostics exactly. Independent packet replays reconstruct every full-rate
  trajectory exactly. A fresh RPY source replay reproduces all 27 graph arrays
  and every native measurement field exactly.
- All 127 protected Sparse and earlier pose-only scientific artifacts retain
  their audit hashes. New target samples and Sparse map predictions match the
  earlier five map cohorts exactly.
- Ruff and whitespace checks pass locally. Remote CI was not run.

The final package includes a post-experiment timestamp edge-case correction
allowing a known TUM timestamp of zero. All evaluated TUM values are positive,
and ScanNet zeros remain unavailable. Saved execution hashes are retained.
Current-code packet and raw-preparation replays verify numerical compatibility.

Retained summaries are `outputs/temporal_fixed/summary.json`,
`controls_verified.json`, `packet_replay.json`, `raw_rpy_replay.json`,
`verification.json`, `rpy_margin0/` and `rpy_local_maps/`. Earlier
`outputs/temporal_fixed/final/` runs use superseded sampling/margin checks and
are exploratory artifacts, excluded from the primary evidence.

For the separate RPY local map diagnostic:

```bash
uv run --locked python experiments/temporal_fixed_landmarks.py \
  --odometry-run outputs/pose_only/full_baselines/rpy \
  --output-dir outputs/temporal_fixed/conservative/rpy --maps-only \
  --map-chunks 2 --map-output-dir outputs/temporal_fixed/new_rpy_local_maps
```

## Decision for subsequent joint BA

The direct objective, frozen numeric factors, common observability gate and
committed-camera boundaries now provide a controlled fixed-landmark baseline.
A future bounded joint experiment can reuse those exact packets and change only
landmark freedom. Both isolated packet comparisons and sequential trajectories
are needed because later window initialization depends on earlier results.

This is sufficient to investigate joint BA conditionally on the frozen motion
model in supported known-clock windows. It does not establish fully corrected
real acquisition timing or justify backend promotion. Important limitations are:

- Frame stamps remain acquisition proxies, without measured exposure midpoint,
  fixed latency or rolling-shutter calibration.
- Sparse translation already uses mixed-time depth. Its color-epoch
  interpretation, interpolation and frozen within-pair motion are approximate.
  Hidden clock-error controls and explicit unsupported-motion fallbacks remain
  necessary in future comparisons.
- Native sampling, motion and depth errors are correlated. The frozen covariance
  and boundary margins are conservative engineering models, without calibrated
  probabilistic coverage. Coherent depth bias remains uncorrected.
- ScanNet timing is unknown. Its static controls cannot validate temporal
  compensation. Conservative RPY support is very limited.
- Full-rank local information does not validate point identities or eliminate
  fixed-landmark bias. Locked boundaries sacrifice later correction of earlier
  cameras and do not create a global objective.

Future BA should preserve point identities, masks, whiteners, initial points,
motion transforms, camera boundaries, common gates and external cohorts. Report
orientation and coverage regressions, retain clock-only and legacy alternatives,
and test biased-depth and hidden-latency controls before interpreting lower
joint residuals as better geometry. Broader scenes and timing evidence are
required before a general backend recommendation.
