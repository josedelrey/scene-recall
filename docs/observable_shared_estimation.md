# Geometric support in persistent shared estimation

The shared RGB-D backend has two new opt-in ways to retain geometric support:
early keyframes based on the actual retained landmark cohort, and verified
short-range reobservation that can reconnect fragmented landmark identities.
Historical tracker image checks can also use the shared exposure chart.
A collective Schur-cut diagnostic tests camera groups rather than individual
camera blocks. Optional transport of reused Jacobians preserves rigid gauge
null directions in the current chart.

These are experimental options. The complete shared solver remains the
reference. Sparse, online, shared estimation, sampling and numerical defaults
are unchanged. No new dependency is required.

## Why the previous graph was insufficient

A connected camera-landmark graph need not determine relative camera poses.
In the complete RPY recording, cameras 325, 330, 335 and 340 share only one
landmark with the rest of their component. All four conditional camera blocks
look healthy. Rotating the entire group and its internal landmarks about the
boundary landmark leaves the nonlinear objective unchanged. There are three
collective rotational freedoms.

More iterations, damping and preconditioning cannot recover those missing
measurements. Increasing the per-view budget cannot recover observations that
were never admitted. Many affected RPY packets were already below the existing
128-observation budget. See [the preceding investigation](incremental_shared_estimation.md)
for the original full-graph diagnosis and sensor controls.

## Collective information with landmark correlations

Let the fresh undamped joint normal system have camera blocks B_i, point blocks
D_j and observation cross blocks E_ij. After temporary landmark elimination,
S = B - E D^-1 E^T. For a camera set A, restrict every camera in A to the same
six-dimensional left motion and hold the other cameras stationary. The resulting
collective information is

```math
H_A = \sum_{i\in A} B_i -
      \sum_j \left(\sum_{i\in A} E_{ij}\right) D_j^{-1}
             \left(\sum_{i\in A} E_{ij}\right)^T.
```

The cross blocks must be summed before elimination. Summing conditional camera
information would discard the correlations responsible for the island's null
modes. The implementation constructs only a 6 by 6 matrix, with linear work in
observation count and linear point workspace. It can center the rotational chart
on a boundary landmark. Column whitening uses the positive subspace of uneliminated camera energy.
Directions below 1e-12 of its largest eigenvalue are excluded from whitening,
so cancellation noise in an exactly null energy direction is not amplified.
The cut uses a 1e-8 absolute and relative normalized rank threshold. These are
numerical diagnostics, with no calibrated uncertainty claim.

A deficient cut identifies a collective direction with no detected information
at that numerical tolerance in the ungauged system.
If a fixed component root lies inside A, that motion may include a removed gauge
freedom. The diagnostic reports fixed cameras inside the cut. A full-rank cut
does not establish rigidity within either group. Neither cut rank nor a complete
local Hessian proves correct associations, unbiased sensors or accurate geometry.

Controlled metric examples give cut ranks zero, three, five and six for zero,
one, two and three noncollinear shared points. The original RPY 325:345 cut has
rank three. Its first three normalized eigenvalues are within 3e-13 of zero.

## Retaining support before it disappears

`--support-keyframes` monitors frontend identities with native measurements
actually admitted to the latest shared packet. Tracking a large number of other
frontend landmarks cannot conceal loss of that retained cohort. For the current
accepted matches, it counts valid current metric points and evaluates their
centered rotational information

```math
C = \frac1n\sum_j (p_j-\bar p)(p_j-\bar p)^T,
\qquad H_\omega = \operatorname{tr}(C)I-C.
```

A single point leaves three rotations free. Collinear points leave one free.
The support spread is sqrt(lambda_min(H_omega)) divided by median point range.
It is invariant to a uniform change of metric scale. The default experimental
trigger requests an early keyframe when support is below
min(24, max(3, ceil(0.5 times the previous native cohort))), rotational rank is
below three, or normalized spread is below 0.02. The usual maximum sampling gap
still applies. A tracking reset clears the cohort. A view without an admitted
packet cannot supply one.

This is a reference-free engineering trigger. It uses frontend metric samples as
a visibility proxy before the incoming native exposure binding and final ledger
selection. It is not a BA observability certificate. Sudden visibility loss may
already be too late, and repeated early insertions can retain weak observations.
The bounded active frontend still keeps eight keyframes by default, so denser
sampling also shortens its time span. The complete shared historical ledger
keeps every admitted packet. Retiring an active frontend view does not retire its
historical likelihood.

## Recovering fragmented identities from native evidence

`--shared-reobserve-recent 3` tries up to three recent retained cameras in the
same tracking segment and inside the existing historical exclusion interval.
It cannot bypass cross-segment loop confirmation. Historical loop retrieval and
confirmation retain their existing policy.

Reobservation rebuilds each feature's metric location from its retained native
sample and frozen RGB/depth exposure transform. Unretained features and features
without an owned native sample cannot enter registration. Frontend map-point
seeds and estimated global poses do not select the rigid registration hypothesis.
Mutual descriptor correspondences are split deterministically. Even rows fit a
symmetric metric transform. Odd rows validate its frozen result using metric
and bidirectional image errors, spatial coverage and geometric support gates.
Only fitting correspondences passing the welding gates propose identity unions.

The proposal owns reversible equivalences in the shared ledger, with explicit
source and target camera IDs and an immutable depth-camera transform used only
for initialization. It introduces no pose-summary factor and does not overwrite
an odometry edge. Existing raw RGB and depth factors remain their single owners.
Conflicting classes with two observations from one camera cannot be merged.
Redundant proposals remain available as alternative association provenance.

Revocation removes that proposal, rebuilds the partition and point state from
raw evidence, updates active tracking and republishes component-consistent
reconstruction. The export includes `reobservation_edges` and
`reobservation_transforms` when present, in addition to raw factors, all proposed
associations and the live identity partition. Per-frame events preserve the
registration verification and admission outcome.

The validation rows are withheld from registration fitting, not from the final
shared observation objective. They are not independent sensor truth or an extra
likelihood. Association selection and correlated native depth error can still
bias estimation. This method adds no new raw samples at a fixed cadence. It can
recover information by assigning already retained measurements to a shared
landmark when the frontend had fragmented its identity.

## Preserving gauge structure during selective optimization

The earlier selective cache stores derivatives evaluated at different old camera
and point states. Such mixed anchors can introduce artificial information along
current rigid gauge directions. With `--shared-linearization selective
--shared-gauge-transport`, the local RGB projection derivative can be reused,
while its camera rotation is transported to the current chart. Native derivatives
are recomputed directly. For every reused observation, enforce

```math
J_P = J_L [-I\quad [X_{\mathrm{current}}]_\times].
```

Then common rigid camera and point motions cancel exactly. This includes island
rotations fixing a boundary point. Tests show a mixed-anchor cache falsely raising
a one-point island cut from rank three to six. The transported cache preserves
rank three. Fresh rows remain exactly those of the complete linearization.
Residuals, robust weights, landmark elimination, trial costs, recovery and full
refresh rules retain their existing contracts.

This is a constrained approximate derivative, not a fully current Jacobian or a
covariance guarantee. It preserves the specified rigid symmetries but does not
prove every other geometric nullspace is preserved. It uses additional temporary
arrays and algebra. The exact objective acceptance rule cannot supply missing
observability or guarantee a good future tracking branch.

## Historical tracker exposure charts

A frozen projection audit found that the shared solver and the historical
tracker were using different image-camera charts. A shared packet stores the
color/depth exposure transform E_k. The frontend retains native depth anchors
in a virtual RGB chart using the static calibration T_RD. These two poses are

```math
T_{WR}^{virtual} = T_{WD} T_{RD}^{-1},
\qquad T_{WR}^{color} = T_{WD} E_k^{-1}.
```

The historical native check correctly uses the virtual chart for its static
native anchor coordinates. Its image check must instead use the color chart.
Previously both used the virtual chart, including after shared physical points
were fed back to tracking. A point accepted by the shared RGB likelihood could
then fail the historical tracker's three-pixel gate purely from exposure motion.

`--shared-tracker-exposure` supplies retained packet exposure transforms to the
active frontend. Historical image checks use
T_WR_virtual T_RD E_k^-1, while native checks retain their existing chart.
The transforms come only from already admitted raw packets. A segment reset
clears them. Missing packet exposures keep the legacy check. Exact static
exposures take the original code path, with bitwise equality tested.

This corrects a specific historical projection inconsistency. Incoming pose
estimation, native association and past-motion exposure prediction still use
existing approximations. E_k remains frozen rather than jointly calibrated.
The frozen audit is not a reconstruction of actual historical tracker decisions.
Complete causal controls are needed to establish its external effects.

## Research basis

[ORB-SLAM](https://arxiv.org/abs/1502.00956) motivates maintaining covisibility and
admitting keyframes generously when tracking support changes. SceneRecall's
shared raw objective and direct identity proposals differ from its essential
pose graph. [Good Features to Track for Visual SLAM](https://www.cv-foundation.org/openaccess/content_cvpr_2015/ext/1B_029_ext.pdf)
motivates evaluating feature utility through observability rather than count
alone. The centered metric support trigger here is a simpler sensor-specific
proxy, with explicit limitations.

[Observability-based Rules for Designing Consistent EKF SLAM Estimators](https://people.csail.mit.edu/ghuang/paper/Huang2009IJRR.pdf)
and [An Observability-Constrained Sliding Window Filter for SLAM](https://intra.ece.ucr.edu/~mourikis/papers/Huang2011-IROS.pdf)
show why linearization must preserve the model's unobservable directions.
The derivative transport here is derived for this shared left-pose/raw-point
objective. It is not an implementation of FEJ filtering.

[iSAM2](https://people.csail.mit.edu/kaess/pub/Kaess12ijrr.pdf) remains relevant to
future incremental factorization and selective relinearization. A Bayes tree can
reuse numerical work but cannot create missing geometric constraints. The
current implementation still rebuilds the full Schur blocks at each solve.

## Reproduction

Run the experimental shared backend on a recording, for example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/rgbd_slam.py \
  --tum-path local_inputs/recording --output-dir outputs/support_run \
  --shared-estimation --contrast clahe --feature-scale 0.5 --flow-recovery \
  --support-keyframes --map
```

The options can be evaluated separately. Do not treat their combination as
validated by separate ablations. The study driver has complete, support,
uniform interval-two, native-reobservation and incremental transport policies.
A manifest freezes source paths, input intervals, exposure, reference tolerance,
active window size and policy selection:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  experiments/observable_shared_study.py \
  --manifest outputs/observable_shared/long_manifest.json \
  --root outputs/new_observable_study
```

Each worker snapshots package sources, the dependency lock and experiment driver.
Its `protocol.json` records the package actually loaded before streaming. Use a
fresh output root. A top-level dispatch protocol may predate a worker source
variant. Per-worker source hashes and snapshots are authoritative for replay.
Matched trajectory and landmark summaries use the existing shared-study scorer.
`diagnose_shared_graph.py --global-rank --region 325 345` adds collective cut
information. `evaluate_online_maps.py --extent full --reference-cohort available`
audits exact refusion and scores complete and common-training maps. Frozen-factor
`replay_shared_factors.py --selective --gauge-transport` is retrospective and must
not be presented as causal pipeline accuracy.

## Evaluation protocol

The causal study uses complete TUM XYZ792, RPY694 and Desk573, ScanNet
scene0000_00 frames 0:160, complete ETH3D Room263 and complete ETH3D Sofa533.
These are 3,015 distinct received inputs across six recordings. Sofa is an
independent recording relative to the preceding shared five-recording matrix,
although earlier nonshared studies used it. This is not blind validation on
newly captured scenes. ScanNet uses the explicit static exposure control.
ETH reference association allows 0.1 seconds for bracket coverage. Reference
poses never enter estimation, support selection, registration or publication.

The reference is the complete shared observation solver, with the previous
validated runs linked read-only into a fresh `outputs/observable_shared` root.
Support retention, native reobservation and historical exposure policies each
use the complete solver. Full-current-Jacobian coarse/retry controls and their
support/reobservation combination are separate policies. Gauge transport uses
selective derivatives, coarse preconditioning and numerical retries, with both
the complete solver and the preceding untransported incremental policy retained
as controls. All use CLAHE, half-scale features and the existing recovery policy.
No policy receives cached future tracking states or an end-of-stream solve.

Uniform interval-two keyframes are a cadence control on Desk and ETH Room.
ETH Room also has an eight-versus-sixteen active-keyframe control. These do not
match every policy's physical active time span or exact camera budget. A better
result cannot establish that the support score alone is responsible. Source
hashes, raw measurement populations, native ownership and temporal association
pairs are compared separately from arbitrary frontend track labels.

Trajectory comparisons use the intersection of each policy's final root-connected
frames and the reference's final root-connected frames. Tables report that
population explicitly, along with total connected coverage. Missing reference
poses reduce the scored count without changing received coverage. Emitted and
revised trajectories are scored separately, with unit-scale anchor alignment.
Disconnected camera gauges are never compared as if their relative pose were
observed. Landmark residuals and reference-placed native discrepancies are paired
on the same owned observations. Native depth is not independent point truth.

Maps use fixed stride-ten fusion views and interleaved reference-pose depth
targets across the complete recording. A stride-five sensitivity control also
includes the odd five-frame camera cohort. Every saved chunk is independently
reintegrated from native depth at its exact published poses. Map scoring includes
coverage and a 20 cm capped missing-surface error. Common-training maps compare
identical received fusion views, so connection changes cannot masquerade as
geometric gains. Targets are withheld from fusion, but may have entered tracking.
A post-hoc reverse-support diagnostic additionally reports unique voxel support,
observed free-space contradictions, occlusion-only and unassessed voxels. It
uses the same held-out native depth and reference poses, rather than independent
surface truth. All voxel denominators remain explicit.

Each worker uses one BLAS/OpenMP thread. Multiple independent experiments were
run concurrently. Their pipeline durations and peak RSS describe these runs and
are not isolated hardware speed comparisons. Frozen same-ledger solves separate
numerical agreement from changed causal evidence. The shared graph's existing
512-camera admission cap remains. Suffix attachments after that cap would need
separate reporting rather than a claim of unlimited estimation scalability.


## Causal trajectory results

These results are exploratory deterministic runs dated 2026-10-26. No option is
promoted. Translation errors below are millimeters, using unit-scale anchor
alignment rather than fitting a scale. Each matched score uses that policy's
intersection with the complete reference. Different rows can have different
intersections. Whole-root errors include every scored frame in the policy's
final root component. Connected counts include frames without a reference pose.

| Recording | Policy | Connected / received | Matched scored frames | Reference matched RMSE | Policy matched RMSE | Policy whole-root RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Desk | control | 403 / 573 | 403 | 62.78 | 62.78 | 62.78 |
| Desk | support | 568 / 573 | 399 | 61.67 | 43.95 | 73.05 |
| Desk | reobserve | 548 / 573 | 383 | 61.29 | 52.61 | 61.51 |
| Desk | support_coarse | 568 / 573 | 399 | 61.67 | 45.23 | 61.89 |
| RPY | control | 679 / 694 | 675 | 77.99 | 77.99 | 77.99 |
| RPY | support | 694 / 694 | 675 | 77.99 | 139.89 | 138.97 |
| RPY | reobserve | 669 / 694 | 660 | 74.35 | 227.89 | 227.07 |
| RPY | support_coarse | 693 / 694 | 674 | 77.99 | 99.96 | 101.64 |
| XYZ | control | 792 / 792 | 789 | 18.49 | 18.49 | 18.49 |
| XYZ | support | 792 / 792 | 789 | 18.49 | 19.04 | 19.04 |
| XYZ | reobserve | 792 / 792 | 789 | 18.49 | 19.29 | 19.29 |
| XYZ | support_coarse | 792 / 792 | 789 | 18.49 | 18.43 | 18.43 |
| ScanNet | control | 160 / 160 | 160 | 60.52 | 60.52 | 60.52 |
| ScanNet | support | 160 / 160 | 160 | 60.52 | 61.14 | 61.14 |
| ScanNet | reobserve | 160 / 160 | 160 | 60.52 | 60.00 | 60.00 |
| ETH Room | control | 140 / 263 | 139 | 7.73 | 7.73 | 7.73 |
| ETH Room | support | 157 / 263 | 139 | 7.73 | 13.68 | 18.72 |
| ETH Room | reobserve | 140 / 263 | 139 | 7.73 | 8.30 | 8.30 |
| ETH Sofa | control | 357 / 533 | 325 | 35.06 | 35.06 | 35.06 |
| ETH Sofa | support | 361 / 533 | 325 | 35.06 | 36.63 | 38.02 |
| ETH Sofa | reobserve | 357 / 533 | 325 | 35.06 | 33.64 | 33.64 |

Desk support retention improves translation from 61.67 to 43.95 mm on the same
399 scored frames and increases connected coverage from 403 to 568 of 573.
The whole-root 73.05 mm result is worse than the reference's 62.78 mm on its
smaller 403-frame population. Support adds 169 frames and loses four reference
frames. The added-only support population has 115.66 mm RMSE. This is evidence
of a useful matched improvement and a remaining difficult-region problem.
Matched rotation improves from 3.69 to 2.83 degrees on the same 399 frames. Native reobservation also improves Desk's
matched translation and removes the detected extra global null directions.

A uniform interval-two control performs much worse on Desk, with 561 connected
frames, 113.78 mm matched RMSE and 123.22 mm whole-root RMSE. More keyframes alone
are insufficient. Doubling ETH Room's active window from eight to sixteen
reduces support's matched error from 13.68 to 12.84 mm, while its matched control
remains about 7.62 mm. That time-span change does not resolve the regression.
The separate Room interval-two whole-root error is 21.20 mm.

RPY support reaches every received frame but worsens matched error from 77.99 to
139.89 mm. Full-Jacobian coarse preconditioning and retry recovery reduce this
to 99.97 mm on 674 matched frames, still worse than their matched reference.
RPY native reobservation reaches fewer frames and worsens 74.35 to 227.89 mm
on 660 matched frames. These are substantive external failures despite stronger
support in the originally identified group. XYZ differences remain small.
The 0.07 mm XYZ support/coarse advantage is one deterministic run, with no
statistical or cross-scene promotion claim.

The support-plus-reobservation combination regresses matched translation on all
four completed recordings: Desk 85.26, ScanNet 62.81, Room 12.83 and Sofa
36.84 mm. The unstarted full RPY and XYZ combination trials were stopped
prospectively after that negative transfer screen. Their original manifests
remain unchanged, with hashed `policy_stops.json` amendments and explicit
`.not_run.json` records. All six-recording single-policy matrices completed.
No difficult input or scored observation was removed from a completed run.

Gauge transport has a mathematical consistency benefit but fails to improve
causal accuracy reliably. Desk reaches 548 frames with 171.28 mm matched error,
compared with 61.29 mm for the same reference frames. RPY is 91.27 versus
68.19 mm on 655 common scored frames. XYZ is 18.78 versus 18.49 mm on 789.
The preceding untransported incremental policy and complete solver are retained
as controls. No incremental policy is promoted.

Historical exposure-chart correction alone yields Desk 72.71 versus 62.35 mm
on 398 paired frames, RPY 203.47 versus 77.99 mm on 675 and XYZ 19.08 versus
18.49 mm on 789. It fixes the stated chart contract but does not solve causal
sensor-model consistency. The three static or simultaneous recordings produce
bitwise identical estimation archives to their complete controls, across all
291 checked arrays. No exposure default changes.

## Observability, landmarks and long-horizon revisions

Fresh, undamped full-graph audits fix each raw component's gauge and unused
camera variables before examining the reduced system. Counts are numerical
null directions at the recorded threshold, not calibrated posterior uncertainty.
They concern the retained optimized graph, not every output attachment.

| Recording / policy | Reduced dimension | Extra null directions | RPY 325:345 rigid-cut rank |
| --- | ---: | ---: | ---: |
| Desk / complete | 666 | 11 | n/a |
| Desk / support | 1308 | 2 | n/a |
| Desk / reobserve | 660 | 0 | n/a |
| Desk / support_coarse | 1314 | 4 | n/a |
| RPY / complete | 792 | 19 | 3 |
| RPY / support | 2358 | 21 | 6 |
| RPY / reobserve | 792 | 20 | 5 |
| RPY / support_coarse | 2412 | 25 | 6 |
| RPY / exposure | 810 | 20 | 6 |

Support replaces the original RPY one-landmark cut with geometrically sufficient
boundary support, while other groups remain weak. Its leading null-mode camera
participation includes frames 348, 456 and 415. Native reobservation supplies
two boundary points to the original group, leaving one collective rotational
freedom. The transported RPY graph has a disconnected component root inside
this interval, so a zero cut there includes a gauge freedom. It cannot be
reported as an observed connection. Desk reobservation has zero detected extra
null directions after gauge fixing, but multiple components and output
attachments still limit any full-recording observability claim.

The native candidate audit reconstructs the original motion allowance from each
packet's whitening covariance without reference poses. It rebinds all extracted
features and exactly reproduces every retained native index and whitening
matrix. At RPY frames 320 and 340, only 14 of 362 and five of 460 extracted
features bind to native depth, respectively. Their retained packets contain
17 and 17 image observations, with 14 and three native bindings. The native
rotation allowance at frame 340 is about 0.015 radians. Increasing the packet
cap cannot recover native samples rejected by uncertainty and boundary gates.
Every consecutive symmetric native registration in the audited 315:350 region
fails even with all bound features. A one-sided native-anchor/RGB-bearing
registration can verify 335-to-330 with 38 and 335-to-340 with 26 anchor
correspondences. This is a retrospective feasibility diagnostic, not a new
causal rescue implementation or an external accuracy result.

Active RPY reobservation associations have a reference-placed native discrepancy
p95 of 26.43 mm and maximum 68.09 mm. The median live-class reference-placed
native RMS radius is 6.65 mm, with one class above 5 cm and none above 10 cm.
These diagnostics do not establish association correctness. Depth calibration,
exposure, binding and reference uncertainty remain confounded. Unique raw
feature and native owners are retained in all completed study runs.

At fixed raw evidence, 32-iteration replays of RPY complete and transported
solvers agree in objective to reported precision and in pose entries within
6e-16. Desk objectives differ by about 8.2e-5 and pose entries by at most
9.0e-7. Their external camera-only errors are effectively unchanged. Rebuilding
additional native aliases on the frozen RPY ledger gives about 77.53 mm camera
error versus 77.46 mm without them. Reusing 1,914 unchanged point seeds gives
77.52 mm. This does not reproduce the 227 mm causal failure. Feedback changes
future evidence and initialization, so retrospective association tests cannot
predict the complete causal branch. Objectives with different identity
partitions are different models and must not be compared as identical costs.

Matched emitted-prefix replays are exact for Desk support/reobservation at 120
frames and the new complete-default XYZ control at 120. All ten immutable raw
packet fields also match their full-run prefixes. RPY support's first 240 frames
have 0.92 mm RMS historical translation change between 240 and 694 inputs,
with a 4.11 mm maximum. Their external error changes from 59.39 to 59.07 mm.
Reobservation's 235 common prefix frames change by 4.04 mm RMS and 19.81 mm
maximum, while external prefix error remains 55.62 mm. The severe full-run
regressions are not a deformation of the entire early prefix. Later weak
regions, changed connectivity and causal tracking feedback remain implicated.

## Reconstruction results and their limits

Every audited saved native reconstruction chunk reintegrates exactly at its
published poses, including component and version metadata. This establishes
reconstruction consistency, independently of physical map accuracy.

The following use stride-five fusion and interleaved held-out targets. The
20 cm capped missing-surface score includes missing projections. Common-training
scores fuse identical received views within each comparison. Desk uses the
intersection across all five listed policies. RPY support/coarse and native
reobservation are separate comparisons, so their common-training cohorts differ.

| Recording / policy | Full capped error, mm | Common-training error, mm | Full target coverage |
| --- | ---: | ---: | ---: |
| Desk / control | 55.78 | 56.71 | 0.9461 |
| Desk / support | 45.03 | 47.09 | 0.9734 |
| Desk / reobserve | 43.92 | 46.32 | 0.9696 |
| Desk / support_coarse | 49.17 | 47.85 | 0.9733 |
| Desk / support_reobserve | 46.69 | 48.88 | 0.9704 |
| RPY support/coarse / control | 92.88 | 92.88 | 0.9714 |
| RPY support/coarse / support | 96.02 | 95.26 | 0.9749 |
| RPY support/coarse / support_coarse | 91.14 | 84.96 | 0.9729 |
| RPY native / control | 92.88 | 92.92 | 0.9714 |
| RPY native / reobserve | 81.25 | 80.98 | 0.9709 |

Desk's common-training support map also increases reverse supported-voxel
fraction from 0.8972 to 0.9432 and reduces unsupported free-space fraction from
0.0499 to 0.0252. Its full map contains newly connected difficult regions and
has a lower support fraction, 0.8717. This distinction agrees with the trajectory
population decomposition.

RPY support worsens both forward and reverse diagnostics. Coarse support's
common-training forward improvement coexists with reverse support falling from
0.7817 to 0.7285 and unsupported free space rising from 0.1577 to 0.1729.
Native reobservation reduces the paired capped forward score from 92.92 to
80.98 mm, but reverse support falls from 0.7952 to 0.7675, while unsupported
free space improves from 0.1547 to 0.1306 and occlusion-only fraction increases.
Those mixed results cannot support a general reconstruction improvement claim.
The reference-pose map itself has nonzero errors and free-space contradictions.
These are sensor-dependent diagnostics with occlusion and calibration limits.

Stride-ten XYZ capped scores are 31.66 mm for complete, 31.64 for support,
31.18 for reobservation, 31.37 for coarse support and 31.16 for exposure.
These small map differences coexist with mostly worse trajectory errors.
On transfer recordings, support's stride-ten scores are ScanNet 54.40 versus
54.14 mm, Room 116.37 versus 117.30 mm and Sofa 25.83 versus 26.88 mm.
Room's common-training support score is 119.89 mm, worse than its control.
Native reobservation scores are 54.62, 116.92 and 26.22 mm respectively.
Exposure regresses Desk's common-training score from 58.16 to 68.70 mm and
RPY's full stride-ten score from 87.34 to 104.61 mm. Static-recording exposure
maps equal their controls because the estimation and reconstruction arrays are
bitwise identical.

## Cost, continuity and scalability

All three TUM and ScanNet support runs retain zero failed tracking transitions.
ETH Room retains one and Sofa 26, equal to their complete controls. Better
root-connected coverage does not imply better immediate tracking continuity.
Support uses more retained cameras, observations and repeated historical work.
The following end-to-end durations include concurrent experiment contention.

| Recording / policy | Pipeline seconds | Frame latency p95, seconds | Peak RSS, MiB | Retained raw observations |
| --- | ---: | ---: | ---: | ---: |
| Desk / control | 246.5 | 2.45 | 260.1 | 12615 |
| Desk / support | 954.4 | 9.83 | 273.4 | 21363 |
| Desk / support_coarse | 792.0 | 6.35 | 275.4 | 21425 |
| RPY / control | 408.8 | 3.84 | 294.6 | 12381 |
| RPY / support | 3127.6 | 15.29 | 332.6 | 30565 |
| RPY / support_coarse | 1783.7 | 7.69 | 334.5 | 30872 |
| XYZ / control | 475.8 | 3.67 | 313.5 | 19999 |
| XYZ / support | 584.6 | 3.95 | 323.8 | 24854 |
| XYZ / support_coarse | 426.1 | 2.85 | 324.7 | 25104 |

The isolated frozen Desk support system has 220 cameras, 12,251 active
observations and 4,355 landmarks. Four alternating-order repeats reduce PCG
iterations from 199 to 76 with coarse preconditioning. Median checked step time
is 0.451 seconds for Jacobi and 0.232 seconds for coarse. Maximum step entry
difference is 2.59e-9. No own pipeline or scoring worker ran during this timing
control. This measures one damped linear system, without a claim about nonlinear
or causal accuracy, hard real-time latency or other system activity.

Desk support/Jacobi has nine recorded numerical rollback events and ends with a
rejected solve. Its final exported optimized-point cache is empty, so missing
point-cache residuals are unavailable rather than evidence of perfect landmark
accuracy. The coarse/retry control avoids those rollback events, but its causal
branch and final graph differ. RPY support/coarse still has 25 extra numerical
null directions. Recovery and preconditioning do not supply missing evidence.

Retained likelihood storage remains linear in admitted observation count.
Matrix-free Schur products avoid materializing the dense camera matrix, while
coarse construction and the existing full historical block rebuild still cost
work as the graph grows. Peak RSS observations do not establish bounded memory.
The 512-camera cap remains, and no unbounded or multihour scalability claim is
supported. The global-rank audit materializes a dense matrix offline and is not
an online cost measurement. Support RPY has 2,358 reduced camera dimensions,
requiring an explicit audit rank limit above the default 2,048.

## Decision and remaining work

Keep the complete solver and validated defaults. Retain these options as
independently selectable research components. The strongest positive result is
Desk's matched trajectory and common-training reconstruction improvement with
better connectivity. The central negative result is that repairing one rigid
cut does not repair long-sequence estimation. Full RPY, transfer failures and
combination regressions rule out promotion.

The most promising next architecture retains RGB-bearing observations before
native depth binding or landmark initialization succeeds, with explicit pending
ownership and later multiview initialization. Covisibility selection should
preserve collective boundary information across groups, rather than only count
features or add cadence. Such a ledger must keep raw ownership distinct from a
landmark's initialized state and support reversible association changes without
adding a second likelihood for an existing sample.

The exposure mismatch also motivates a shared continuous or interpolated motion
state for actual RGB and depth exposure times. [SplineFusion](https://www.bmva-archive.org.uk/bmvc/2013/Papers/paper0093/paper0093.pdf)
is a relevant RGB-D continuous-time reference. That larger sensor model needs
joint timing/calibration sensitivity controls and independent geometric tests.
The one historical projection correction implemented here is insufficient.
Incremental factorization becomes more valuable once these evidence and state
contracts are stable. It must preserve correlations and gauge structure through
relinearization, association splits and component changes.

Evidence is under ignored `outputs/observable_shared`. `final_comparison.json`
contains exact paired trajectory, rotation, latency and memory data. The original
manifests, worker protocols, source/driver snapshots, graph audits, frozen
controls, prefix parity reports and map reports preserve population and source
provenance. `final_integrity.json` verifies all 468 protected baseline artifacts,
ten source snapshots, eight driver snapshots and source availability for all
38 new worker runs. All 38 workers retain unique feature and native owners.
Fixed-interval source hashes agree across policies. Native refusion is exact for 118 chunks across
25 distinct saved reconstruction artifacts. No protected artifact changed.
Final verification passed 1,116 tests in 117.85 seconds, Ruff and whitespace
checks. No dependency changed
and no commit was created.
