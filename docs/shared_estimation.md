# Persistent shared observation estimation

SceneRecall has an opt-in `--shared-estimation` research backend. Tracking
keyframes, historical keyframes and loop-owned landmark identities enter one
relinearizable camera–landmark objective. Its raw ledger survives local frontend
eviction. The previous sparse, online, local-joint and welding defaults remain
unchanged. This backend is a coherent shared estimation experiment, with explicit
limits on observability, measurement modeling and computation.

## Why replace the welding overlay

The earlier [welding experiment](observation_level_welding.md) combines a pose
graph with small conditional bundles. Its active feedback revisions work, but
historical local information disappears when a marginal prior is invalidated.
Cameras outside each bundle receive nearest-camera corrections. Different
neighborhoods can consequently satisfy different conditional objectives, even
when state transport and reconstruction are internally correct.

The shared backend removes those two approximations from the retained
keyframe estimation problem. Every admitted historical observation is replayable
and every measured camera participates in the same solve. A loop changes point
identity, rather than introducing another likelihood for pixels already used
by odometry or registration. There is no local marginal prior to invalidate and
no second optimizer whose result is imported as an independent measurement.

## Objective and provenance

Let `C_i` map RGB camera `i` into its component frame and let `X_j` be a point in
that frame. A recorded observation contains RGB pixel `u_ij`, optional native
depth-camera point `d_ij`, exposure transform `E_ij` from depth to RGB, and frozen
native whitening `W_ij`. With fixed calibration, the objective is

```text
F(C, X, a) = sum_(i,k) rho(||project(C_i^-1 X_a(i,k)) - u_ik||^2)
           + sum_(i,k with depth)
             rho(||W_ik (E_ik^-1 C_i^-1 X_a(i,k) - d_ik)||^2)
```

`a(i,k)` is the current landmark class of frontend track `k`. The existing
Cauchy loss uses scale 2.5 and the existing RGB and native-depth residuals.
Each admitted RGB observation and each admitted native depth sample has one
owner within its view. Both residual blocks belong to that recorded observation.
They are never copied into a local prior or a pose-summary likelihood.

Selection remains bounded to 128 observations per view, preserving existing
tracks and spatially distributed new native observations. This retains all
*admitted* evidence, rather than every frontend feature or every input frame.
Packets preserve feature indices, segment/frontend identities, native indices,
pixels, depth points, exposure transforms and whitening. The packets are
immutable. Their original initialization points are provenance, not additional
constraints on the live point variables.

Registration and distinct-view confirmation still verify loops. Fitting rows
propose reversible, edge-owned track equivalences. A union cannot give one
landmark two observations in the same camera. Removing a loop rebuilds the live
identity partition and point seeds from raw evidence. Unchanged cameras retain
their solved shape during identity edits. New bridges initialize a rigid
alignment between old gauges using the verified registration transform. The
transform supplies no residual in the shared optimization.

Withheld registration rows remain diagnostic contradiction measurements. They
are reported before and after shared solves. They are not independent external
truth, may share frontend tracks with fitting factors, and are not a new
global acceptance oracle. The shared solver accepts finite decreasing raw
objective steps. An incorrect verified identity can still bias the result.

Raw observation blocks are independent under the optimization model. Their
induced camera and point correlations are retained exactly at each
linearization. This does not model correlations from calibration, clock error,
coherent depth bias, adaptive track selection, or common sensor processing.
Fixed whitening is an engineering noise model, not a calibrated posterior.

## Temporary elimination and relinearization

`slam/shared_solver.py` uses the existing analytic observation Jacobians and
robust block weights. Its normal system is

```text
[ B  E ] [delta_C] = -[g_C]
[ E' D ] [delta_X]    [g_X]

S = B - E D^-1 E'
b = -g_C + E D^-1 g_X
S delta_C = b
delta_X = -D^-1 (g_X + E' delta_C)
```

`B` contains camera blocks, `D` contains independent 3×3 point blocks and `E`
contains one 6×3 block per observation. The matrix-free Schur product performs
both accumulations over all observations of a point. It preserves the complete
cross-camera coupling induced by that point, including coupling across the
frontend window boundary. A block-Jacobi preconditioned conjugate-gradient solve
checks the true linear residual. Point back substitution uses exactly the same
damped system as the camera solve.

Levenberg–Marquardt damping applies to both camera and point diagonals before
elimination. A candidate must reduce the original robust nonlinear objective
and obey the configured displacement bounds. Damping and eliminated blocks are
discarded after the step. The next nonlinear iteration recomputes raw
Jacobians and robust weights. A new keyframe, association merge or revocation
can therefore relinearize the entire retained history. There is no stale
Gaussian approximation that needs approximate nonrigid transport.

The selected per-solve bounds are 0.5 m camera translation, 30 degrees rotation
and 0.5 m point movement, with eight nonlinear iterations. Earlier bounded
welding used tighter 0.15 m / 5 degree / 0.15 m limits and a smaller variable
population. The complete system comparison therefore does not isolate the
effect of factor sharing from correction bounds or solve scheduling. Complete
RPY reaches a 0.399 m camera update, and Desk reaches the 0.5 m limit. Weak
directions can still move substantially despite a decreasing objective.

One measured camera fixes each raw connected component's rigid gauge.
Singleton-only tracks remain in the ledger but do not enter a relative-camera
solve. They contain no relative camera information after their free point is
eliminated. Tracks require a native observation somewhere in their history.
There is no depth-free scale inference in this experiment.

Small systems, up to 24 cameras by default, receive a complete undamped Schur
rank diagnostic. Larger systems inspect undamped point and conditional camera
blocks and report that the rank scope is only `blocks`. This is not a global
observability certificate. The selected shared policy permits updates in a
partially observable system, with an initialization-dependent choice of weak
directions. It does not report finite covariance in those directions. An
optional `SharedConfig(require_full_rank=True)` ablation rejects weak systems.
Singular point blocks and failed linear residual checks reject the numerical
solve and restore its complete input state.

This implementation uses persistent nonlinear state and incremental ingestion
with full scheduled relinearization. It does not reuse a Bayes tree or claim
iSAM2 update complexity. The [Ceres iterative Schur formulation](https://ceres-solver.readthedocs.io/latest/nnls_solving.html#iterative-schur)
describes the linear operator. [iSAM2](https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf)
provides the stronger incremental factorization model for future work.
Keeping the raw nonlinear graph first makes later selective relinearization
testable against this complete reference objective. No dependency was added.

The shared solver reuses `BatchConfig` for robust loss, iteration, displacement
and conditioning settings. Its batch-only selection fields do not control this
backend. Shared admission uses two-view native support and the ledger budget,
while the pose container's `max_nodes` sets retained camera capacity.

## Tracking, publication and reconstruction

`slam/shared_estimation.py` reuses the native ledger and reversible identity
construction. The existing pose-edge container stores initialization and
measurement provenance. Its pose-only solver is bypassed. Odometry and
registration pose summaries have zero objective weight.

`OnlineSLAM` selects a raw-only local frontend. It emits new packets but never
creates a second local smoother, local marginal prior, or duplicated raw local
factor population. Every admitted keyframe triggers the shared solve, including
when loops are disabled. The local frontend keeps its bounded appearance map
and performs between-keyframe PnP and recovery as before.

The active tracker receives shared camera and point states in its segment chart.
For tracking-segment root `r`, the chart is `A = G_r`, with global depth pose
`G_r` and an exactly identity local root pose. Active camera states become
`A^-1 G_i` and point states become `A^-1 X_j`. They are state updates, not
probabilistic factors. All received keyframe poses in the current segment's
archive use the same chart, including evicted frontend keyframes. Unadmitted frontend landmark hypotheses remain native
seeds, and can become shared variables when future packets admit them.

Raw connectivity, rather than a summary odometry edge, determines shared
components. Some initial cameras have no usable depth exposure. They receive
explicit rigid output attachments to the first measured camera in their
tracking segment. Their relative placement is initialization-dependent.
Other unmeasured cameras attach to their preceding measured camera. A measured
camera is jointly optimized or fixed as a component gauge. No nearest-camera
deformation is applied to measured historical states.

Nonkeyframes retain their recorded rigid attachment to the preceding keyframe.
Their internal motion is still a frontend estimate. They are not extra bundle
variables. Tracking losses remain explicit, even if later raw identities join
components. New tracker-relative motions extend the graph initialization chart
without rewriting old odometry summaries after a shared revision.

Publication compares poses and component IDs before incrementing the version
and reporting dirty frames. The existing publication tolerance retains bitwise
unchanged poses. Reconstruction refuses mixed gauges within a chunk. The shared
path can split a chunk when raw connectivity exposes multiple gauges, then
reintegrates its original camera-local samples under exactly the published
poses. A revoked bridge can therefore disconnect both poses and reconstruction
without retaining a merged landmark prior.

## Running and scientific exports

```bash
uv run python scripts/rgbd_slam.py \
  --tum-path /path/to/rgbd_dataset_freiburg1_xyz \
  --output-dir outputs/shared_xyz --stop 120 \
  --contrast clahe --feature-scale 0.5 --flow-recovery \
  --shared-estimation --map
```

Use a fresh output directory. Shared estimation requires matched local and
ledger keyframe cadences, and cannot be combined with `--optimize-local`,
`--observation-welding`, `--welding-feedback`, or conditional multiview
registration. ScanNet requires an explicit `--local-exposure static` assumption
in these comparisons. No usable timing model for that capture was introduced.

The usual emitted and revised atlas, trajectory, event, protocol and
reconstruction files remain available. `shared_estimation.npz` additionally
exports original raw packets, native and feature ownership, frontend-to-ledger
identities, live classes, loop-owned equivalences, current points, camera poses,
and raw component membership. Per-keyframe diagnostics record raw objective
costs, accepted steps, linear residuals and iteration counts, rank scope,
history rebuilds, support counts, and registration validation.

`experiments/shared_estimation_study.py` runs serial isolated policies from a
JSON manifest, using one OpenCV/BLAS/OMP thread. It reports common root-connected
populations separately for emitted and revised poses. Its native point and RGB
residual summaries are in-sample consistency diagnostics. Use
`experiments/evaluate_online_maps.py` for fixed-camera map scoring and exact
fresh-fusion checks. `experiments/benchmark_shared_solver.py` measures controlled
linear-system scaling on chain and long-track graphs.

Map scoring defaults to the existing first-240-frame cohort. Add `--extent full`
to the map evaluator for fixed stride-10 training and interleaved reference
targets over the complete recording. Its reference availability and common
training reports remain explicit. The voxel budget still applies.

## Evaluation

The ignored `outputs/shared_estimation/` root retains the frozen selection,
isolated-run protocols, diagnostic prototypes, final manifest, complete raw
runs, point states, map scores, prefix comparisons and preservation checks.
Authoritative raw results are under `verified/`, selected by
`verified_manifest.json`. Earlier prototypes and the shared runs under `final/`
are excluded. An export audit found that the prototype's latest-camera chart
could move the local root away from identity. The corrected shared path uses a
segment-root chart and updates all archived keyframes in that chart. Its raw
runner validates the exported trajectory. A controlled loss/bridge/revocation
test also exposed a keyframe phase mismatch after tracking loss. Shared
admission now follows actual frontend keyframes, including shifted segment
starts. Existing online admission defaults are unchanged.

### Protocol and populations

The comparison uses raw RGB-D, CLAHE, half-resolution SIFT, flow recovery and
stride-8 mapping samples. References are stripped before estimation. Complete
XYZ, RPY and Desk recordings contain 792, 694 and 573 inputs. ScanNet supplies
160 inputs under the same explicit static-exposure assumption in both policies.
ETH3D `sfm_lab_room_1` supplies another 263 inputs. There are 2,482 distinct
inputs across five recordings. Shorter XYZ/RPY horizons, policy ablations,
timing repeats and reference-policy repeats reuse those inputs.

The intended Room 900:1200 source retained metadata but lacked images after
historical cleanup. The complete available ETH3D recording replaced it based
on availability before its policy runs. This is an additional sensor and
recording, previously studied in this repository, rather than a blind new
capture. No second ScanNet capture was acquired.

The ETH3D clock runs near 13.6 Hz with small pose/depth timestamp offsets. The
unchanged 50 ms reference bracket tolerance yields only one usable reference,
so its near-zero aligned error is uninformative. Both raw policies were
repeated with the existing 100 ms diagnostic reference option. Every atlas and
reconstruction array remained bitwise identical. Only diagnostic reference
availability changed. No reference pose entered estimation. For map evaluation,
the explicit available-reference cohort anchors both maps at common referenced
training frame 10 when frame 0 lacks a reference.

The table reports paired control/shared errors on each pair's common
root-connected population. Translation is RMSE in mm and rotation is RMSE in
degrees. `N` counts received frames, with unavailable references excluded by the
scorer. Emitted and revised populations are separate. A no-loop ablation does
not reduce the population of the principal control/shared comparison.

| Recording | N, emitted / revised | Emitted translation | Revised translation | Revised rotation |
| --- | ---: | ---: | ---: | ---: |
| XYZ 120 | 120 / 120 | 14.54 / 20.25 | 15.04 / 13.92 | 0.621 / 0.682 |
| XYZ complete | 792 / 792 | 20.90 / 19.91 | 21.89 / 18.49 | 0.972 / 0.748 |
| RPY 240 | 195 / 235 | 119.60 / 79.59 | 109.14 / 69.79 | 5.205 / 3.939 |
| RPY 400 | 350 / 390 | 122.84 / 77.58 | 113.38 / 82.97 | 5.157 / 4.075 |
| RPY complete | 634 / 679 | 117.67 / 94.96 | 112.76 / 77.99 | 5.167 / 4.097 |
| Desk complete | 398 / 403 | 54.51 / 61.82 | 59.84 / 62.78 | 2.901 / 3.820 |
| ScanNet 160, static | 160 / 160 | 91.41 / 60.42 | 91.22 / 60.52 | 2.501 / 1.770 |
| ETH3D complete, 100 ms references | 140 / 140 | 14.79 / 7.64 | 14.32 / 7.73 | 0.444 / 0.299 |

All TUM and ScanNet policies retain zero tracking losses. Both ETH3D policies
lose tracking at the same transition. Shared geometry can nevertheless expose
additional raw disconnections. Control root populations are 694 and 573 for
complete RPY and Desk, while shared emitted/revised populations are 634/679 and
398/403. ETH3D has 157 control root frames versus 140 shared root frames.
Continuous frontend tracking is therefore not evidence that the selected raw
ledger constrains one connected trajectory.

Median frontend map inliers decline from 148 to 135 on complete XYZ, 71.5 to
58 on RPY and 96 to 87 on Desk. ScanNet stays near 67 and ETH3D stays at 51.
Zero added losses do not establish unchanged tracking margin or landmark
quality. Desk remains an external regression despite decreasing internal costs.

### Reconstruction

The surface score caps missing or inaccurate depth at 20 cm in fixed reference
cameras. Lower is better. Coverage is the fraction of valid target samples with
a prediction. The 400-frame RPY prefix score still uses the first 240-frame
training/target cohort, while complete-recording rows cover the full extent.

| Recording and cohort | Surface error, control / shared, mm | Coverage, control / shared |
| --- | ---: | ---: |
| XYZ 120 | 52.76 / 40.85 | 0.9284 / 0.9076 |
| XYZ complete, full extent | 42.47 / 31.66 | 0.9738 / 0.9653 |
| RPY 240 | 88.85 / 76.29 | 0.8564 / 0.8689 |
| RPY 400, first 240 | 88.59 / 76.31 | 0.8576 / 0.8666 |
| RPY complete, full extent | 84.40 / 87.34 | 0.9428 / 0.9476 |
| Desk complete, full extent | 57.29 / 58.16 | 0.9508 / 0.9098 |
| ScanNet 160, static | 63.38 / 54.14 | 0.8687 / 0.8643 |
| ETH3D complete, full extent | 114.29 / 117.30 | 0.4635 / 0.4422 |

Different root populations reduce fusion availability on Desk and ETH3D. Under
identical common training views, Desk scores 58.92 / 58.16 mm with coverage
0.9077 / 0.9098. ETH3D scores 117.56 / 117.30 mm with coverage 0.4434 / 0.4422.
Those small matched-cohort gains do not remove the coverage loss in the complete
system. Complete RPY uses the same training cohort and worsens surface error
despite improved trajectory RMSE. XYZ and ScanNet also lose some coverage while
improving capped error. Reference-pose fusion is not a lower bound or independent
surface truth.

Every audited saved chunk matches fresh native-sample fusion exactly, including
cloud positions, colors, support and scatter, component membership and version.
This verifies reconstruction publication consistency. It does not establish
geometric accuracy.

### Historical revisions and the XYZ/RPY diagnosis

The corrected XYZ 120 and RPY 240/400 runs match the complete runs' emitted
prefix arrays exactly, including versions and components. There is no final
retrospective solve in these raw runs.

On 235 common root-connected frames of RPY's first 240 inputs, extending the
publication horizon from 240 to 400 changes poses by 0.90 mm median, 4.78 mm
RMSE and 7.52 mm p95. Extending to 694 changes them by 0.85 mm median, 4.97 mm
RMSE and 8.09 mm p95. Revised external RMSE on that fixed prefix changes from
69.79 to 71.14 to 71.27 mm. Later constraints still move earlier geometry and
slightly worsen its external score, but the historical deformation is bounded
in this experiment. XYZ's 120-to-792 change is 0.56 mm median and 2.42 mm p95,
with prefix RMSE improving from 13.92 to 13.78 mm.

RPY's fixed first-240-frame surface score changes only from 76.29 to 76.31 to
76.37 mm across those horizons. Its poorer 87.34 mm full-extent result therefore
cannot be attributed solely to later deformation of that prefix. Later views
and their spatial relationships introduce additional external error.

The earlier feedback interpretation needs a correction. Its first-240-frame
emitted-to-final shift is already 109.66 mm median at the 240-input horizon,
versus 108.84 mm at 400. Its 240-to-400 pose change is 8.64 mm RMSE and 18.71 mm
p95. Two additional late welding events do not by themselves establish the
cause of the much larger emitted-to-final discrepancy.

On the same 195 frames that stay root-connected across emission and final
publication in all compared policies, old welding at 400 has 33.97 mm median
emitted-to-final displacement, old landmark feedback has 117.91 mm, and the new
shared graph at 694 has 7.99 mm. Their p95 values are 140.68, 169.29 and 42.49
mm. The new all-received-prefix p95 is much larger because some frames start in
different raw gauges and later rejoin. Such coordinate changes must not be
presented as measured physical deformation.

XYZ's previous lower loop-validation error did not imply external accuracy.
The shared model improves its final trajectory and surface score while its
120-frame emitted trajectory still regresses. This is consistent with stronger
joint historical correction, without proving one unique cause of the earlier
regression. Different feedback changes subsequent identities and initialization,
so the complete raw comparison does not isolate a single optimizer mechanism.

### Landmarks, ablations and numerical limits

Complete shared XYZ retains 19,999 observations and 3,022 optimized points.
RPY retains 12,381 observations and 1,945 optimized points. Median native depth
residuals are 4.11 and 3.44 mm, respectively, while median discrepancies from
reference-placed native samples are 13.43 and 22.75 mm. Reference-placed
within-identity RMS radii have medians 5.82 and 6.25 mm. These are different
diagnostics. Small fitting residuals do not imply externally correct point
placement, and reference-placed depth is still the same sensor's measurement.
All exports retain unique feature and native-depth owners.

Without loops, complete XYZ reaches 34.08 mm revised error versus 18.49 mm with
loops on the full population. RPY without loops retains only 55 first-component
frames and rejects 54 shared solves at the linear residual limit. Its revised
error on those 55 frames is 97.71 mm versus control's 72.80 mm. It cannot be
compared to a full-trajectory score or used as evidence that disconnected
fragments are accurate. Temporal raw support and solver conditioning remain
material limitations.

The corrected strict-rank RPY 240 policy abstains when weak historical cameras
block the graph. On the five-policy common revised cohort it scores 79.55 mm
versus 69.79 mm for the selected partial-observability policy. Their surface
scores are 79.69 and 76.29 mm. Damped updates can therefore help this case, but
do not certify its weak directions or provide a calibrated covariance.

Retrospective frozen-ledger controls separate fitting from causal tracking.
On XYZ 120 with unjoined native identities, 32 iterations reduce Cauchy cost
from 1404.23 to 694.34 while keyframe translation RMSE worsens from 14.36 to
14.48 mm. On RPY 400, axial-depth fitting reduces its own Cauchy cost from
2184.32 to 2167.97 while keyframe error worsens from 75.90 to 76.35 mm. The
changes are small externally and are not evidence of a significant physical
effect. They directly demonstrate that an internal decrease is insufficient
for promotion. Native/axial costs have different dimensions and are not compared
to one another. RPY's native-identity replay fails its PCG residual check and
rolls back completely. An additional live-native fit changes the already small
objective very little and gives no substantial external gain.

### Computation and reproducibility

Four isolated XYZ 120 pairs, in opposite policy orders after the accuracy work
finished, average 7.04 control inputs/s and 4.81 shared inputs/s. Ranges are
6.10–8.18 and 4.52–5.29 inputs/s. Mean process p95 is 0.638 / 1.011 seconds.
Peak RSS ranges are 161.5–163.5 / 158.2–158.8 MiB. Raw reading, extraction,
tracking, recognition, solving and mapping are included, with export/scoring
excluded. All 248 compared trajectory, atlas and reconstruction arrays across
the eight repeats match their saved controls exactly, including NaNs.

The complete accuracy matrix ran independent recordings concurrently. Its
1.66 inputs/s on XYZ and 1.70 on RPY, p95 latencies above 3.6 seconds, and
roughly 313 / 295 MiB peak RSS are descriptive measurements, not isolated
throughput comparisons. They expose a practical update cost without providing
a real-time guarantee. The final RPY solve needs up to 495 PCG iterations in a
single nonlinear step. Removing the local marginal prior saves some short-run
memory, but retained historical storage still grows.

The isolated synthetic benchmark measures one linearization and one checked
damped linear step, rather than complete SLAM. At 512 cameras and 49,152
observations, long tracks take 10 PCG iterations and 0.127 seconds to solve.
A chain takes 189 iterations and 1.709 seconds. Linearization takes about
0.357 seconds in both. Compact blocks plus operator arrays occupy 7.24 and
8.74 MiB, compared with 71.72 MiB for a materialized dense camera Schur matrix.
Those byte counts exclude packets, Jacobians and temporary workspaces. The
chain's iterations grow from 22 at 8 cameras to 189 at 512, demonstrating that
linear storage is not constant update cost.

The final Linux full suite passed 1,053 tests, including the additional
available-reference anchor test. Ruff lint,
formatting and whitespace checks pass. The suite checks dense normal-equation
agreement, dense nonlinear agreement, rank diagnostics, atomic rollback,
identity revocation, bridge shape preservation, loss/capacity handling, local
root identity, causal prefix equivalence, raw runner round trips and exact
fusion. Native Windows and hosted CI were not run.

All 354 protected files from the welding and feedback studies retain their
SHA256 values. `source_snapshots/` retains package variants matching the
corrected raw runs' recorded package digests, plus research drivers and the
unchanged dependency lock. The variants differ in export validation and
post-loss admission. The earlier full TUM runs have no losses, and the final
ETH3D run exercises the repaired phase. Protocols retain source hashes,
parameters, explicit exclusions and the Room/ETH3D substitution. No commit,
push, dataset modification or context synchronization was performed.

## Remaining limitations

The following limitations apply independently of any accuracy result:

- Raw storage grows linearly with admitted history until the existing 512-node
  budget. At capacity, new global admission stops and tracking continues with
  attachments. There is no disk-backed eviction or recoverable compact prior.
- Each keyframe solve revisits all retained observations. Linear storage does
  not establish constant update time or real-time operation. Long chains can
  require many PCG iterations, and the global rank is uncertified above the
  diagnostic limit.
- Keyframe downselection, frontend identity proposals, fixed intrinsics,
  color/depth timing, sensor depth bias and incorrect loops remain sources of
  external geometric error. A consistent objective can prefer an inaccurate
  trajectory when these assumptions are wrong.
- Every nonlinear update is bounded to eight iterations by default. An earlier
  trajectory can change substantially as later observations arrive. Internal
  cost reductions, exact elimination and exact refusion do not prove physical
  accuracy or convergence to a unique optimum.
- Reference-target depth scores are independent of fusion, but their frames may
  enter tracking and estimation. Reference-pose fusion is not independent surface
  truth. TUM recordings share a sensor, and ScanNet static exposure is an explicit
  assumption. These populations do not establish broad unseen-sensor transfer.

The next scalability step is selective relinearization and cached sparse
factorization against this same raw objective, with explicit recovery on
identity edits. The next accuracy step requires independent identity, exposure,
depth-bias and calibration evidence. Reintroducing overlapping pose-summary
likelihoods would defeat the ownership and correlation contract.
