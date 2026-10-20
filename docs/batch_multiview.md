# Persistent-track batch multiview investigation

Measured on Ubuntu on 2026-10-20. Keeping shared landmarks across an entire
supported training graph substantially improves the full TUM XYZ trajectory.
It does not establish a robust replacement for Sparse. Desk and RPY lack a
fully constrained batch graph under the retained measurement policy, and local
reconstruction gains remain mixed. A separate full-sequence XYZ reconstruction
diagnostic shows a large TSDF improvement. Sparse stays the default.

This investigation follows the [local RGB-D BA experiment](local_rgbd_ba.md).
All earlier backends, baseline artifacts, measurement archives and primary map
populations are preserved. No online tracking, place recognition, loop closure,
new dependency or automatic commit is introduced.

## Direction and implementation

The previous joint solver converged but discarded point history at six-camera
window boundaries. Committed overlap cameras prevented later evidence from
correcting earlier drift. The saved XYZ graph already contained 62,411 training
observations and tracks spanning up to 65 views. Using that evidence offered a
more concrete hypothesis than further loss, iteration or uncertainty tuning.

`multiview.batch_ba` implements a bounded offline camera/point fit with one shared
3D point per admitted training identity throughout its original segment. Camera
zero is the original segment root and remains fixed exactly. Native mode uses
the existing RGB 2D and native-depth 3D objective, frozen two-exposure transforms,
whitening, separate block Cauchy losses at scale 2.5 and inverse-track-length
weights. Points initialize from geometric medians of measured depth samples
transformed by Sparse. Saved world points and reference poses never initialize
them. There is no pose prior or inter-segment constraint.

Each track needs three remaining views and two native depth samples. Cameras
need 12 observations and 15% occupancy on the existing 6 by 8 grid. Removing a
camera can invalidate tracks, so pruning repeats to a fixed point. The original
segment root always enters the system, including when it has no observations.
Disconnected components retain their unobservable gauges and cause rejection.
They are never made observable through damping or additional fixed cameras.

Compact observation Jacobians and independent 3 by 3 point elimination retain a
dense camera Schur system, limited to 200 cameras. Every iteration checks the
undamped, diagonally normalized spectrum against the existing `1e8` condition
limit. Failure restores the complete initialization. Bounds are 0.5 m camera
center movement, 30 degrees orientation change and 0.5 m point movement relative
to the initial root gauge. They accommodate a longer trajectory than local BA.
No primary accepted step hits a bound. The iteration budget is 40.

Corrections interpolate across unsupported training cameras and carry beyond
the last supported camera. Existing full-rate completion and frozen conversion
back to depth exposure times preserve metadata, independent segment roots and
losses. A rejected segment equals the existing clock-only control, rather than
receiving a landmark update. Clock-only changes must not be credited to BA.
Unknown acquisition clocks are rejected explicitly.

This is a combined architecture and data-use intervention. It changes point
persistence, camera boundaries, support selection and the old 256-track cap.
Its gain cannot be attributed exclusively to one of those changes. The native
and axial ablation below do use identical selected observations and point seeds.

## Frozen evaluation protocol

The current Ubuntu Sparse runs and temporal packets supply 792 XYZ, 573 Desk
and 694 RPY observations, with 160, 116 and 141 training views. Features and
native bindings were constructed from every-fifth training images only, before
this investigation. Existing full-rate Sparse initialization previously used
the full stream. Target images and depth are held out of new refinement and
fusion, rather than wholly unseen by baseline VO. References enter scoring only.

Primary maps retain XYZ chunks `0 6`, Desk `0 4` and RPY `0 5`, covering exactly
37, 41 and 43 evaluated target views. Training depth, voxel size, native sampling,
TSDF truncation and rendering settings are unchanged. Native depth, rather than
optimized feature points, supplies both reconstruction methods. Scores use
reference target poses, valid target samples and a 20 cm cap charged to missing
predictions. Exact common-pixel errors and coverage are retained separately.

These related Freiburg 1 sequences are development evidence. Neither their
correlated frames nor overlapping tracks constitute independent trials. No
significance or independent-scene generalization claim follows. ScanNet is
excluded from the new batch ranking because its tested acquisition clocks are
unavailable. Reported TUM timestamps remain acquisition proxies, with unresolved
latency, motion interpolation, calibration and correlated depth error.

## Trajectory evidence

Aligned metric-scale position ATE RMSE in mm, pooled by reference count across
independent segments:

| Population | Sparse | Clock only | Temporal fixed | Local joint BA | Batch native |
| --- | ---: | ---: | ---: | ---: | ---: |
| XYZ | 55.994 | 55.860 | 48.142 | 47.282 | 24.382 |
| Desk | 76.525 | 77.065 | 75.378 | 75.513 | 77.065 |
| RPY | 51.456 | 51.403 | 50.499 | 49.162 | 51.403 |

XYZ improves 56.5% relative to Sparse and 48.4% relative to local joint BA.
This improvement also appears in orientation, endpoint and longer-horizon RPE:

| XYZ metric | Sparse | Clock only | Temporal fixed | Local joint BA | Batch native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Root-anchored orientation RMSE, degrees | 1.955 | 1.967 | 2.198 | 2.298 | 1.079 |
| Endpoint position, mm | 140.110 | 144.358 | 127.519 | 130.564 | 59.632 |
| Endpoint rotation, degrees | 2.499 | 2.548 | 3.149 | 4.062 | 1.101 |
| 50-frame translation RPE RMSE, mm | 41.755 | 41.072 | 38.788 | 37.668 | 30.740 |
| 50-frame rotation RPE RMSE, degrees | 1.996 | 2.044 | 1.992 | 1.963 | 1.627 |
| 10-second translation RPE RMSE, mm | 84.479 | 84.219 | 71.726 | 70.132 | 37.949 |
| 10-second rotation RPE RMSE, degrees | 2.352 | 2.373 | 2.364 | 2.429 | 1.465 |

The XYZ system admits 157 cameras, 6,120 landmarks and 46,556 observations.
Frames 75, 80 and 85 fail support or spatial coverage and receive interpolated
corrections. The accepted fit has no Schur nullity and a final normalized
condition of approximately 1.63 million. It stops after 23 iterations. Cost falls
from 5,214.832 to 3,037.625, maximum camera-center change is 91.74 mm and maximum
point change is 140.59 mm. These are diagnostics, not the accuracy evidence.

Desk rejects with 18 undamped null directions. RPY's two original segments
reject with 12 and 24 null directions. Unsupported segment roots and disconnected
camera groups prevent one well-constrained fit. The clock-only equality in their
batch columns is verified numerically. The experiment adds no robustness claim
for either sequence.

## Primary reconstruction evidence

Coverage-penalized capped depth error in mm on the original target populations:

| Population | Representation | Sparse | Clock only | Temporal fixed | Local joint BA | Batch native |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| XYZ | Voxel | 35.990 | 26.763 | 26.769 | 26.233 | 27.152 |
| XYZ | TSDF | 31.004 | 30.186 | 30.009 | 30.710 | 29.791 |
| Desk | Voxel | 31.630 | 28.220 | 28.336 | 28.165 | 28.220 |
| Desk | TSDF | 34.023 | 32.288 | 32.605 | 32.316 | 32.288 |
| RPY | Voxel | 74.018 | 70.693 | 70.693 | 70.693 | 70.693 |
| RPY | TSDF | 61.972 | 62.016 | 62.016 | 62.016 | 62.016 |

XYZ's large trajectory gain gives only a 0.396 mm TSDF gain versus clock-only
and a 0.389 mm voxel regression. On exact clock/batch common pixels, voxel
uncapped MAE worsens from 31.358 to 31.756 mm, while TSDF improves from 20.069 to
19.154 mm. TSDF improves in 20/37 views, voxel in 13/37. This is mixed local
reconstruction evidence. Desk and RPY map changes versus Sparse are clock-only.

## Full-sequence reconstruction diagnostic

The original 24-view submaps cancel common placement error and cover short
intervals. A separately labelled, post hoc extent diagnostic fuses all 160 XYZ
training views into one map. Optimizer inputs, fitted trajectories, depth
filtering, voxel/TSDF settings and target offset remain unchanged. All methods
use the same 158 target candidates, of which 157 have valid reference poses.
This population is never pooled with the primary 37-target comparison.

Training frame 195 lacks a reference pose. The original all-paired map policy
therefore withholds the entire 160-view chunk. That initial diagnostic is retained
as an unavailable result. An explicit `--allow-partial-training-reference` option
enables reference-target rendering only when the first training anchor and each
scored target have valid references. Native training depth and estimated poses
still supply every map sample, including frame 195. Missing references are never
imputed, and the incomplete reference training cloud stays labelled unpaired.
The default evaluation policy remains unchanged. Target 197 remains unscored.

| XYZ method | Voxel capped loss, mm | TSDF capped loss, mm | Voxel coverage, % | TSDF coverage, % |
| --- | ---: | ---: | ---: | ---: |
| Sparse | 44.259 | 90.347 | 98.956 | 97.033 |
| Clock only | 34.397 | 96.188 | 98.907 | 96.976 |
| Temporal fixed | 33.652 | 77.344 | 98.946 | 97.071 |
| Local joint BA | 34.673 | 63.731 | 99.026 | 97.078 |
| Batch native | 32.061 | 38.551 | 99.276 | 98.269 |

Batch lowers long-map TSDF capped error by 39.5% versus local joint BA and 57.3%
versus Sparse. Against local joint, TSDF improves in 156/157 target views, and
uncapped MAE on their exact common pixels falls from 88.485 to 49.652 mm. Against
clock-only it improves in all 157 views, with common-pixel MAE falling from
139.651 to 49.728 mm. Thus the result includes better common-pixel geometry as
well as coverage. Voxel gains are smaller. Against clock-only, voxel improves in
68/157 views despite a better pooled score. This remains one sequence and a
post hoc map-extent diagnostic, rather than independent generalization evidence.

## Alternative hypotheses and limits

An axial-only depth ablation keeps exactly the native batch graph, point seeds,
RGB residuals and frozen transforms, replacing each native 3D block with its
z residual divided by the marginal native z standard deviation. XYZ ATE is
23.990 mm and orientation RMSE 1.062 degrees. Its primary voxel/TSDF losses are
27.171/29.888 mm. Slightly better trajectory scores coexist with slightly worse
map scores than native batch. This supplies no basis for changing the default
measurement model or claiming that lateral depth constraints explain the issue.

Earlier exploratory probes admitted two-view tracks, split the graph using
12-shared-track camera links, fixed additional component roots and propagated
endpoint corrections. RPY ATE worsened to about 65 mm even though the long
segment endpoint improved markedly. Connectivity counts alone do not establish
full pose information, and a better endpoint does not establish a better path.
Those local ignored probes are retained separately from the strict batch method.
Repeating the component experiment with consistent interpolation across all
unsupported views gives RPY ATE 64.784 mm and Desk 72.635 mm. This does not repair
the cross-sequence regression. No component result is substituted for the primary
batch method or used to claim a map gain.

The earlier [noise, depth bias, scale and timing counterexamples](local_rgbd_ba_evaluation.md#synthetic-controls)
remain relevant. Batch history does not calibrate timestamps, resolve coherent
metric bias or make correlated RGB-D observations independent. Synthetic tests
verify exact recovery from perturbed poses and rejection of disconnected,
depthless, collinear and missing-root geometry. Algebraic tests compare the
compact damped and undamped steps with the existing joint solver exactly to
numerical tolerance, including point back-substitution.

## Reproduction and retained artifacts

Generate temporal packets as described in the
[temporal guide](temporal_fixed_landmarks.md), then use fresh output directories:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/batch_multiview.py \
  --fixed-run outputs/joint_ba/controls/xyz \
  --output-dir outputs/final_multiview/new_xyz \
  --evaluate-maps --map-chunks 0 6

uv run --locked python experiments/replay_batch_packets.py \
  outputs/final_multiview/new_xyz \
  --output outputs/final_multiview/new_replay.json

uv run --locked python experiments/summarize_batch.py \
  outputs/final_multiview/new_xyz \
  --output outputs/final_multiview/new_summary.json

uv run --locked python experiments/evaluate_batch_horizon.py \
  --batch-run outputs/final_multiview/new_xyz \
  --joint-run outputs/joint_ba/primary/xyz \
  --output-dir outputs/final_multiview/new_long_xyz --map-budget 160 \
  --allow-partial-training-reference
```

Use `--depth-model axial` in a separate run for the matched measurement ablation.
Desk and RPY use map chunks `0 4` and `0 5`. The CLI never changes Sparse defaults.
The batch runner reads verified numeric packets without opening image media.
Preparation hashes, factor fingerprints, initialization, fitted poses and points,
solver diagnostics, full-rate trajectories and source identities are retained.

Primary artifacts are under `outputs/final_multiview/primary/{xyz,desk,rpy}`.
The matched axial run is under `axial/xyz`. `replay.json` records a separate-process
reconstruction of every factor, point, diagnostic and full-rate trajectory with
exact array equality. The earlier `probe_*`, `exploratory_xyz` and rejected strict
full-horizon diagnostic remain exploratory outputs. They are excluded from the
primary tables. Outputs and datasets are ignored and require separate transfer
between machines.

The scored 160-view diagnostic is under `horizon160_partial/xyz`, with the strict
unavailable result under `horizon160/xyz`. `summary_primary.json` and
`summary_diagnostics.json` retain external scores and exact common-pixel/view
comparisons. The native fit uses about 2.48 s for cached factor preparation and
25.02 s for fitting/completion. These are observational CPU timings, excluding
the earlier raw feature preparation and map work. Full long-map evaluation of
Sparse and four alternatives takes about 335 s. The final axial diagnostic stops
at iteration 40 on relative decrease, and an 80-iteration allowance reproduces
all its trajectory arrays exactly.

The full suite passes 782 tests. Focused tests cover analytic derivatives,
exact Schur equivalence, missing depth, root and segment preservation, update
bounds, unobservable graphs, source-free operation, archive tampering and the
partial-reference opt-in. Ruff lint, formatting and whitespace checks pass.
Separate-process packet replay reproduces all factors, fitted points, diagnostics
and full-rate poses exactly. An independent full-sequence Sparse/batch map repeat
matches 34 arrays in five archives, including voxel clouds, TSDF meshes and all
target predictions. Pooled and per-view scores also match exactly. All 181
protected earlier scientific artifacts are unchanged, and trajectory metadata,
references and segment roots remain exact.
No remote CI or Windows run was performed. No commit was created.

## Recommendation

Retain batch adjustment as an opt-in offline trajectory and reconstruction
reference. XYZ shows
that the previous bounded architecture left recoverable trajectory accuracy
unused. The benefit requires supported persistent geometry and does not justify
promoting a backend across these sequences. Sparse remains the operational
baseline.

Proceed to local-map tracking with this evidence rather than extending loss and
threshold searches on these same clips. Preserve longer-lived landmarks and
permit consistent revision of earlier camera states, while treating weak
connectivity, measurement timing and depth bias as explicit limitations. Broader
claims require independent scenes and acquisition evidence. Further iteration
on the present objective has diminishing value without those measurements.
